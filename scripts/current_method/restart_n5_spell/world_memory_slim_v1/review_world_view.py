"""Compact, provenance-preserving world view for the round-wide reviewer."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


REVIEW_WORLD_VIEW_SCHEMA_VERSION = "review_world_view_action_continuation_v3"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def build_review_world_view(
    *, complete_dialogue: Mapping[str, Any], planning_state: Mapping[str, Any],
    compiled_round_plan: Mapping[str, Any], realized_scenes: Sequence[Mapping[str, Any]],
    scene_contexts: Mapping[str, Mapping[str, Any]],
    scene_dependency_dag: Mapping[str, Any],
    narrative_order_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the reviewer view without mirroring authoritative utterance records."""
    scenes = list(compiled_round_plan.get("scenes") or [])
    anchor_by_id: dict[str, dict[str, Any]] = {}
    related_refs: set[str] = set()
    narrative_actions: list[dict[str, Any]] = []
    constraints: list[dict[str, Any]] = []
    for scene in scenes:
        anchor = scene.get("source_anchor") or {}
        anchor_id = str(scene.get("source_anchor_id") or
                        anchor.get("source_anchor_id") or "")
        if anchor_id and anchor_id not in anchor_by_id:
            anchor_by_id[anchor_id] = {
                "source_anchor_id": anchor_id,
                "root_id": anchor.get("root_id"),
                "source_kind": anchor.get("source_kind"),
                "session_ref": anchor.get("session_ref"),
                "scene_ref": anchor.get("scene_ref"),
                "turn_refs": list(anchor.get("source_turn_refs") or []),
                "exact_excerpt": deepcopy(list(anchor.get("exact_excerpt") or [])),
                "source_refs": list(anchor.get("source_turn_refs") or []),
            }
        linked = scene.get("linked_existing_refs") or {}
        for values in linked.values():
            if isinstance(values, list):
                related_refs.update(str(value) for value in values)
        narrative_actions.append({
            "scene_ref": scene.get("scene_ref"),
            "scene_alias": scene.get("scene_alias"),
            "primary_action": deepcopy(scene.get("primary_action")),
            "secondary_action": deepcopy(scene.get("secondary_action")),
            "action_contracts": deepcopy(scene.get("narrative_action_contracts")),
            "durable_update_compilation_stage": "post_round_validation",
        })
        context = scene_contexts.get(str(scene.get("scene_ref"))) or {}
        constraints.append({
            "scene_ref": scene.get("scene_ref"),
            "source_anchor_id": anchor_id,
            "placement_choice_id": scene.get("placement_choice_id"),
            "placement": scene.get("placement"),
            "preceding_known_context_refs": list(
                context.get("preceding_known_context_refs") or []),
            "following_constraint_only_refs": list(
                context.get("following_context_constraint_refs") or []),
            "following_context_role": "constraint_only",
            "same_round_predecessor_refs": list(
                context.get("same_round_predecessor_refs") or []),
            "predecessor_dialogue_sha256": deepcopy(
                context.get("predecessor_dialogue_sha256") or {}),
            "context_snapshot_sha256": context.get("frozen_context_sha256"),
        })

    def relevant(rows: Any, keys: Sequence[str]) -> list[dict[str, Any]]:
        result = []
        for row in rows if isinstance(rows, list) else []:
            refs = {str(row.get(key) or "") for key in keys}
            refs.update(str(value) for value in row.get("subject_refs") or [])
            refs.update(str(value) for key, values in row.items()
                        if key.endswith("_refs") and isinstance(values, list)
                        for value in values)
            if refs & related_refs:
                result.append(deepcopy(dict(row)))
        return result

    related_state = {
        "root_id": planning_state.get("root_id"),
        "entities": relevant(planning_state.get("entities"), ("entity_ref",)),
        "active_events": relevant(planning_state.get("active_events"),
                                  ("memory_ref", "event_ref")),
        "open_threads": relevant(planning_state.get("open_threads"),
                                 ("memory_ref", "thread_ref")),
        "unfinished_tasks": relevant(planning_state.get("tasks"),
                                    ("memory_ref", "task_ref")),
        "key_states": relevant(planning_state.get("key_states"), ("memory_ref",)),
        "recent_scene_end_state": deepcopy(
            planning_state.get("recent_round_end") or {}),
        "scene_lineage": deepcopy(planning_state.get("scene_lineage") or []),
    }
    value = {
        "review_world_view_schema_version": REVIEW_WORLD_VIEW_SCHEMA_VERSION,
        "root_id": complete_dialogue.get("root_id"),
        "complete_dialogue_snapshot_sha256": complete_dialogue.get(
            "complete_dialogue_sha256"),
        "historical_anchors": list(anchor_by_id.values()),
        "related_planning_state": related_state,
        "interaction_links": relevant(planning_state.get("interaction_links"),
                                      ("interaction_link_ref",)),
        "explicit_relations": relevant(planning_state.get("explicit_relations"),
                                       ("memory_ref", "relation_ref")),
        "realized_scenes": deepcopy(list(realized_scenes)),
        "dependency_graph": deepcopy(dict(scene_dependency_dag)),
        "narrative_constraints": constraints,
        "narrative_order": deepcopy(dict(narrative_order_manifest)),
        "narrative_actions": narrative_actions,
        "excluded_record_types": ["parent_utterance"],
    }
    value["review_world_view_sha256"] = _sha(value)
    return value


def validate_review_world_view(value: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if value.get("review_world_view_schema_version") != REVIEW_WORLD_VIEW_SCHEMA_VERSION:
        errors.append("review_world_view_version")
    if not value.get("root_id"):
        errors.append("review_world_view_root")
    anchors = value.get("historical_anchors")
    if not isinstance(anchors, list) or any(
            not row.get("source_refs") or not row.get("exact_excerpt")
            for row in anchors if isinstance(row, Mapping)):
        errors.append("review_world_view_anchor_provenance")
    if any(row.get("record_type") == "parent_utterance"
           for section in (value.get("related_planning_state") or {}).values()
           if isinstance(section, list) for row in section if isinstance(row, Mapping)):
        errors.append("review_world_view_parent_utterance_mirror")
    if any(row.get("following_context_role") != "constraint_only"
           for row in value.get("narrative_constraints") or []):
        errors.append("review_world_view_constraint_label_lost")
    if value.get("excluded_record_types") != ["parent_utterance"]:
        errors.append("review_world_view_exclusions")
    actions = value.get("narrative_actions")
    if not isinstance(actions, list) or any(
            not isinstance(row.get("primary_action"), Mapping) or
            row.get("durable_update_compilation_stage") != "post_round_validation"
            for row in actions if isinstance(row, Mapping)):
        errors.append("review_world_view_narrative_actions")
    if "candidate_update_hints" in value:
        errors.append("review_world_view_legacy_action_hints_forbidden")
    expected = dict(value)
    actual_sha = expected.pop("review_world_view_sha256", None)
    if actual_sha != _sha(expected):
        errors.append("review_world_view_sha")
    return {"valid": not errors, "errors": errors,
            "review_world_view_schema_version": REVIEW_WORLD_VIEW_SCHEMA_VERSION}
