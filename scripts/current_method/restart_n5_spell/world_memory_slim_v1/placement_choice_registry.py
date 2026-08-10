"""Independent source-anchor and narrative-placement registries for v11.

Source anchors answer *what historical material does this scene develop?*.
Placement choices answer *where is the scene situated on the narrative
timeline, and what dialogue is knowable there?*.  The Planner selects one ID
from each frozen registry; only local code expands boundaries and knowledge
partitions.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


SOURCE_ANCHOR_REGISTRY_VERSION = "source_anchor_registry_v1"
PLACEMENT_CHOICE_REGISTRY_VERSION = "placement_choice_registry_v1"
PLACEMENT_CHOICE_COMPILER_VERSION = "placement_choice_compiler_v1"
PLACEMENT_CHOICE_SCHEMA_VERSION = "placement_choice_v1"

SOURCE_LINK_TYPES = frozenset({
    "direct_continuation", "causal_followup", "consequence_followup",
    "backstory_expansion", "missing_event_fill", "time_skip",
    "location_shift", "related_character_shift", "parallel_thread",
    "connected_new_entity", "connected_new_event", "external_constraint",
    "information_transfer", "task_followup", "relationship_followup",
    "state_followup",
})

_CHOICE_FIELDS = frozenset({
    "placement_choice_schema_version", "placement_choice_id", "root_id",
    "placement", "after_turn_ref", "before_turn_ref",
    "preceding_known_turn_refs", "following_constraint_turn_refs",
    "allowed_time_semantics", "knowledge_boundary", "narrative_order_key",
})


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def compile_source_anchor_registry(
    anchor_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Project content linkage without carrying any placement permission."""
    rows: list[dict[str, Any]] = []
    for anchor in anchor_registry.get("anchors") or []:
        source_anchor_id = str(anchor.get("anchor_id") or "")
        if not source_anchor_id:
            raise ValueError("source_anchor_id_missing")
        rows.append({
            "source_anchor_id": source_anchor_id,
            "root_id": str(anchor_registry.get("root_id") or ""),
            "anchor_type": anchor.get("anchor_type"),
            "source_kind": anchor.get("source_kind"),
            "session_ref": anchor.get("session_ref"),
            "scene_ref": anchor.get("scene_ref"),
            "source_turn_refs": list(anchor.get("turn_refs") or []),
            "exact_excerpt": deepcopy(anchor.get("exact_excerpt") or []),
            "linked_entity_refs": list(anchor.get("linked_entity_refs") or []),
            "linked_interaction_refs": list(anchor.get("linked_interaction_refs") or []),
            "linked_relation_refs": list(anchor.get("linked_relation_refs") or []),
            "linked_event_refs": list(anchor.get("linked_event_refs") or []),
            "linked_task_refs": list(anchor.get("linked_task_refs") or []),
            "linked_thread_refs": list(anchor.get("linked_thread_refs") or []),
            "linked_state_refs": list(anchor.get("linked_state_refs") or []),
            "linked_information_refs": list(anchor.get("linked_information_refs") or []),
            "linked_causal_refs": list(anchor.get("linked_causal_refs") or []),
            "allowed_link_types": sorted(SOURCE_LINK_TYPES),
        })
    result = {
        "source_anchor_registry_version": SOURCE_ANCHOR_REGISTRY_VERSION,
        "root_id": str(anchor_registry.get("root_id") or ""),
        "source_anchor_count": len(rows),
        "source_anchors": rows,
        "placement_fields_excluded": [
            "placement", "available_placements", "boundary",
            "after_turn_ref", "before_turn_ref"],
    }
    result["source_anchor_registry_sha256"] = _sha(result)
    return result


def validate_source_anchor_registry(
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if registry.get("source_anchor_registry_version") != SOURCE_ANCHOR_REGISTRY_VERSION:
        errors.append("source_anchor_registry_version")
    rows = list(registry.get("source_anchors") or [])
    ids = [str(row.get("source_anchor_id") or "") for row in rows]
    if not rows or any(not value for value in ids) or len(ids) != len(set(ids)):
        errors.append("source_anchor_ids")
    forbidden = {"placement", "available_placements", "boundary",
                 "after_turn_ref", "before_turn_ref"}
    for row in rows:
        source_anchor_id = str(row.get("source_anchor_id") or "")
        if forbidden & set(row):
            errors.append(f"source_anchor_contains_placement:{source_anchor_id}")
        if row.get("root_id") != registry.get("root_id"):
            errors.append(f"source_anchor_root:{source_anchor_id}")
        if not row.get("source_turn_refs"):
            errors.append(f"source_anchor_turn_refs:{source_anchor_id}")
        if not set(row.get("allowed_link_types") or []).issubset(SOURCE_LINK_TYPES):
            errors.append(f"source_anchor_link_types:{source_anchor_id}")
    return {"valid": not errors, "errors": list(dict.fromkeys(errors)),
            "source_anchor_count": len(rows)}


def resolve_source_anchor(
    registry: Mapping[str, Any], source_anchor_id: str,
) -> dict[str, Any]:
    matches = [row for row in registry.get("source_anchors") or []
               if row.get("source_anchor_id") == source_anchor_id]
    if len(matches) != 1:
        raise ValueError(f"unknown_or_duplicate_source_anchor_id:{source_anchor_id}")
    return deepcopy(dict(matches[0]))


def _choice(
    *, choice_id: str, root_id: str, placement: str,
    all_turn_refs: Sequence[str], after_turn_ref: str | None,
    before_turn_ref: str | None, allowed_time_semantics: Sequence[str],
    narrative_position: float,
) -> dict[str, Any]:
    positions = {ref: index for index, ref in enumerate(all_turn_refs)}
    if after_turn_ref is None:
        known: list[str] = []
    else:
        known = list(all_turn_refs[:positions[after_turn_ref] + 1])
    if before_turn_ref is None:
        following: list[str] = []
    else:
        following = list(all_turn_refs[positions[before_turn_ref]:])
    return {
        "placement_choice_schema_version": PLACEMENT_CHOICE_SCHEMA_VERSION,
        "placement_choice_id": choice_id,
        "root_id": root_id,
        "placement": placement,
        "after_turn_ref": after_turn_ref,
        "before_turn_ref": before_turn_ref,
        "preceding_known_turn_refs": known,
        "following_constraint_turn_refs": following,
        "allowed_time_semantics": list(allowed_time_semantics),
        "knowledge_boundary": {
            "known_through_turn_ref": after_turn_ref,
            "constraint_from_turn_ref": before_turn_ref,
            "following_context_role": "following_context_constraint_only",
            "following_context_is_character_knowledge": False,
        },
        "narrative_order_key": f"{narrative_position:012.3f}",
    }


def compile_placement_choice_registry(
    *, complete_dialogue: Mapping[str, Any], anchor_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile a finite, deterministic menu of meaningful round placements."""
    root_id = str(complete_dialogue.get("root_id") or "")
    if not root_id or root_id != anchor_registry.get("root_id"):
        raise ValueError("placement_choice_root_mismatch")
    all_turn_refs = [str(row.get("turn_ref") or "")
                     for row in complete_dialogue.get("turns") or []]
    if not all_turn_refs or any(not ref for ref in all_turn_refs):
        raise ValueError("placement_choice_dialogue_turns_invalid")
    positions = {ref: index for index, ref in enumerate(all_turn_refs)}
    choices: list[dict[str, Any]] = []
    choices.append(_choice(
        choice_id="PC_TAIL_AFTER", root_id=root_id, placement="after",
        all_turn_refs=all_turn_refs, after_turn_ref=all_turn_refs[-1],
        before_turn_ref=None,
        allowed_time_semantics=("current_tail_continuation", "future_consequence",
                                "time_skip_after_current_dialogue"),
        narrative_position=float(len(all_turn_refs) - 1) + 0.25))

    tail_anchors = [row for row in anchor_registry.get("anchors") or []
                    if row.get("anchor_type") == "tail_anchor"]
    if tail_anchors:
        tail_refs = [str(ref) for ref in tail_anchors[0].get("turn_refs") or []
                     if str(ref) in positions]
        if tail_refs:
            before_ref = tail_refs[0]
            before_index = positions[before_ref]
            after_ref = all_turn_refs[before_index - 1] if before_index else None
            choices.append(_choice(
                choice_id="PC_TAIL_BEFORE", root_id=root_id, placement="before",
                all_turn_refs=all_turn_refs, after_turn_ref=after_ref,
                before_turn_ref=before_ref,
                allowed_time_semantics=("immediate_pre_tail_context",
                                        "missing_precondition"),
                narrative_position=float(before_index) - 0.25))
            choices.append(_choice(
                choice_id="PC_PARALLEL_TAIL", root_id=root_id, placement="parallel",
                all_turn_refs=all_turn_refs, after_turn_ref=after_ref,
                before_turn_ref=before_ref,
                allowed_time_semantics=("parallel_to_tail_interval",),
                narrative_position=float(before_index) - 0.125))

    gaps = []
    for anchor in anchor_registry.get("anchors") or []:
        if anchor.get("anchor_type") != "historical_gap_anchor":
            continue
        boundary = anchor.get("boundary") or {}
        after_ref = str(boundary.get("after_turn_ref") or "")
        before_ref = str(boundary.get("before_turn_ref") or "")
        if after_ref in positions and before_ref in positions:
            gaps.append((positions[after_ref], after_ref, before_ref))
    gaps.sort()
    for index, (after_index, after_ref, before_ref) in enumerate(gaps, 1):
        between_position = (float(positions[after_ref]) +
                            float(positions[before_ref])) / 2.0
        choices.append(_choice(
            choice_id=f"PC_GAP_{index:04d}", root_id=root_id,
            placement="between", all_turn_refs=all_turn_refs,
            after_turn_ref=after_ref, before_turn_ref=before_ref,
            allowed_time_semantics=("historical_gap_fill", "missing_event_fill",
                                    "backstory_expansion"),
            narrative_position=between_position))
        choices.append(_choice(
            choice_id=f"PC_PARALLEL_{index:04d}", root_id=root_id,
            placement="parallel", all_turn_refs=all_turn_refs,
            after_turn_ref=after_ref, before_turn_ref=before_ref,
            allowed_time_semantics=("parallel_to_historical_boundary",),
            narrative_position=float(after_index) + 0.125))

    result = {
        "placement_choice_registry_version": PLACEMENT_CHOICE_REGISTRY_VERSION,
        "placement_choice_compiler_version": PLACEMENT_CHOICE_COMPILER_VERSION,
        "root_id": root_id,
        "complete_dialogue_sha256": complete_dialogue.get("complete_dialogue_sha256"),
        "choice_count": len(choices),
        "choices": choices,
        "finite_menu": True,
        "planner_may_override_boundaries": False,
    }
    result["placement_choice_registry_sha256"] = _sha(result)
    validation = validate_placement_choice_registry(
        result, complete_dialogue=complete_dialogue)
    if not validation["valid"]:
        raise ValueError("placement_choice_registry_invalid:" +
                         ",".join(validation["errors"]))
    return result


def validate_placement_choice(
    choice: Mapping[str, Any], *, complete_dialogue: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    choice_id = str(choice.get("placement_choice_id") or "")
    if set(choice) != _CHOICE_FIELDS:
        errors.append(f"placement_choice_schema:{choice_id}")
    if choice.get("placement_choice_schema_version") != PLACEMENT_CHOICE_SCHEMA_VERSION:
        errors.append(f"placement_choice_version:{choice_id}")
    if choice.get("root_id") != complete_dialogue.get("root_id"):
        errors.append(f"placement_choice_root:{choice_id}")
    placement = str(choice.get("placement") or "")
    if placement not in {"after", "before", "between", "parallel"}:
        errors.append(f"placement_choice_enum:{choice_id}")
    turn_refs = [str(row.get("turn_ref") or "")
                 for row in complete_dialogue.get("turns") or []]
    positions = {ref: index for index, ref in enumerate(turn_refs)}
    after_ref = choice.get("after_turn_ref")
    before_ref = choice.get("before_turn_ref")
    if after_ref is not None and str(after_ref) not in positions:
        errors.append(f"placement_choice_after_ref:{choice_id}")
    if before_ref is not None and str(before_ref) not in positions:
        errors.append(f"placement_choice_before_ref:{choice_id}")
    if placement == "after" and (after_ref != turn_refs[-1] or before_ref is not None):
        errors.append(f"placement_choice_tail_after_boundary:{choice_id}")
    if placement in {"before", "between", "parallel"} and before_ref is None:
        errors.append(f"placement_choice_before_required:{choice_id}")
    if (after_ref is not None and before_ref is not None and
            str(after_ref) in positions and str(before_ref) in positions and
            positions[str(after_ref)] >= positions[str(before_ref)]):
        errors.append(f"placement_choice_boundary_order:{choice_id}")
    known = list(choice.get("preceding_known_turn_refs") or [])
    following = list(choice.get("following_constraint_turn_refs") or [])
    if len(known) != len(set(known)) or len(following) != len(set(following)):
        errors.append(f"placement_choice_partition_duplicates:{choice_id}")
    if set(known) & set(following) or known + following != turn_refs:
        errors.append(f"placement_choice_knowledge_partition:{choice_id}")
    boundary = choice.get("knowledge_boundary") or {}
    expected_boundary = {
        "known_through_turn_ref": after_ref,
        "constraint_from_turn_ref": before_ref,
        "following_context_role": "following_context_constraint_only",
        "following_context_is_character_knowledge": False,
    }
    if boundary != expected_boundary:
        errors.append(f"placement_choice_knowledge_boundary:{choice_id}")
    if not choice.get("allowed_time_semantics"):
        errors.append(f"placement_choice_time_semantics:{choice_id}")
    if not str(choice.get("narrative_order_key") or ""):
        errors.append(f"placement_choice_narrative_order:{choice_id}")
    return {"valid": not errors, "errors": errors,
            "placement_choice_id": choice_id}


def validate_placement_choice_registry(
    registry: Mapping[str, Any], *, complete_dialogue: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if registry.get("placement_choice_registry_version") != PLACEMENT_CHOICE_REGISTRY_VERSION:
        errors.append("placement_choice_registry_version")
    if registry.get("placement_choice_compiler_version") != PLACEMENT_CHOICE_COMPILER_VERSION:
        errors.append("placement_choice_compiler_version")
    if registry.get("root_id") != complete_dialogue.get("root_id"):
        errors.append("placement_choice_registry_root")
    choices = list(registry.get("choices") or [])
    ids = [str(row.get("placement_choice_id") or "") for row in choices]
    if not choices or any(not value for value in ids) or len(ids) != len(set(ids)):
        errors.append("placement_choice_ids")
    for choice in choices:
        errors.extend(validate_placement_choice(
            choice, complete_dialogue=complete_dialogue)["errors"])
    if "PC_TAIL_AFTER" not in ids:
        errors.append("placement_choice_tail_after_missing")
    if not any(value.startswith("PC_GAP_") for value in ids):
        errors.append("placement_choice_gap_missing")
    if not any(value.startswith("PC_PARALLEL_") for value in ids):
        errors.append("placement_choice_parallel_missing")
    return {"valid": not errors, "errors": list(dict.fromkeys(errors)),
            "choice_count": len(choices)}


def resolve_placement_choice(
    registry: Mapping[str, Any], placement_choice_id: str,
) -> dict[str, Any]:
    matches = [row for row in registry.get("choices") or []
               if row.get("placement_choice_id") == placement_choice_id]
    if len(matches) != 1:
        raise ValueError(
            f"unknown_or_duplicate_placement_choice_id:{placement_choice_id}")
    return deepcopy(dict(matches[0]))


def compact_source_anchor_view(
    source_anchor_registry: Mapping[str, Any], *, excerpt_limit: int = 7,
) -> dict[str, Any]:
    rows = []
    for anchor in source_anchor_registry.get("source_anchors") or []:
        excerpt = list(anchor.get("exact_excerpt") or [])
        source_turn_refs = list(anchor.get("source_turn_refs") or [])
        rows.append({
            "source_anchor_id": anchor.get("source_anchor_id"),
            "anchor_type": anchor.get("anchor_type"),
            "source_kind": anchor.get("source_kind"),
            # Full turn membership remains frozen in the local authoritative
            # registry.  The Planner needs a bounded reference span, not a
            # repeated copy of every historical turn ref.
            "source_turn_ref_span": {
                "first": source_turn_refs[0] if source_turn_refs else None,
                "last": source_turn_refs[-1] if source_turn_refs else None,
                "count": len(source_turn_refs),
            },
            "linked_entity_refs": list(anchor.get("linked_entity_refs") or []),
            "linked_interaction_refs": list(anchor.get("linked_interaction_refs") or []),
            "linked_relation_refs": list(anchor.get("linked_relation_refs") or []),
            "linked_event_refs": list(anchor.get("linked_event_refs") or []),
            "linked_task_refs": list(anchor.get("linked_task_refs") or []),
            "linked_thread_refs": list(anchor.get("linked_thread_refs") or []),
            "linked_state_refs": list(anchor.get("linked_state_refs") or []),
            "allowed_link_types": list(anchor.get("allowed_link_types") or []),
            "exact_excerpt": excerpt[-excerpt_limit:],
            "excerpt_is_tail_of_source_anchor": len(excerpt) > excerpt_limit,
        })
    return {"root_id": source_anchor_registry.get("root_id"),
            "source_anchors": rows}


def compact_placement_choice_view(
    placement_choice_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Planner selection view omits repeated full turn partitions."""
    return {
        "root_id": placement_choice_registry.get("root_id"),
        "registry_sha256": placement_choice_registry.get(
            "placement_choice_registry_sha256"),
        "choices": [{
            "placement_choice_id": row.get("placement_choice_id"),
            "placement": row.get("placement"),
            "after_turn_ref": row.get("after_turn_ref"),
            "before_turn_ref": row.get("before_turn_ref"),
            "allowed_time_semantics": list(row.get("allowed_time_semantics") or []),
            "knowledge_boundary": deepcopy(row.get("knowledge_boundary") or {}),
            "narrative_order_key": row.get("narrative_order_key"),
            "preceding_known_turn_count": len(row.get("preceding_known_turn_refs") or []),
            "following_constraint_turn_count": len(
                row.get("following_constraint_turn_refs") or []),
        } for row in placement_choice_registry.get("choices") or []],
    }


def migrate_legacy_scene_anchor_placement(
    scene: Mapping[str, Any], *, placement_choice_id: str,
) -> dict[str, Any]:
    """Explicit shadow/fixture migration; never called by the live validator."""
    if "historical_anchor_id" not in scene or "placement" not in scene:
        raise ValueError("legacy_scene_anchor_placement_fields_missing")
    result = deepcopy(dict(scene))
    result["source_anchor_id"] = str(result.pop("historical_anchor_id"))
    result.pop("placement")
    result["placement_choice_id"] = str(placement_choice_id)
    return result


def migrate_legacy_round_plan_source_placement(
    plan: Mapping[str, Any], *, placement_choice_id: str,
) -> dict[str, Any]:
    """Explicit whole-fixture migration for offline tests and frozen mocks."""
    result = deepcopy(dict(plan))
    result["round_plan_version"] = "continuous_round_plan_v2"
    result["scenes"] = [migrate_legacy_scene_anchor_placement(
        scene, placement_choice_id=placement_choice_id)
        for scene in result.get("scenes") or []]
    return result


__all__ = [
    "PLACEMENT_CHOICE_COMPILER_VERSION", "PLACEMENT_CHOICE_REGISTRY_VERSION",
    "PLACEMENT_CHOICE_SCHEMA_VERSION", "SOURCE_ANCHOR_REGISTRY_VERSION",
    "compact_placement_choice_view", "compact_source_anchor_view",
    "compile_placement_choice_registry", "compile_source_anchor_registry",
    "migrate_legacy_round_plan_source_placement",
    "migrate_legacy_scene_anchor_placement", "resolve_placement_choice",
    "resolve_source_anchor", "validate_placement_choice",
    "validate_placement_choice_registry", "validate_source_anchor_registry",
]
