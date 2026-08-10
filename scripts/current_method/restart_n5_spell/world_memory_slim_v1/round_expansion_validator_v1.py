"""Prompt, normalizer, local validation, and retry for a round-wide reviewer."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

from .durable_update_compiler import durable_candidate_action_eligible
from .round_validator_contract import (
    ACTION_REALIZATION_FIELDS, ACTION_REALIZATION_STATUSES,
    CONNECTED_PERSON_DURABLE_UPDATE_FIELDS,
    CONNECTION_STATUSES, CONSISTENCY_STATUSES, CONTINUITY_STATUSES,
    DEVELOPMENT_STATUSES, DUPLICATE_STATUSES, DURABLE_UPDATE_FIELDS,
    MODEL_CONTINUATION_ACTION_REALIZATION_FIELDS,
    MODEL_NEW_ACTION_REALIZATION_FIELDS, MODEL_SCENE_REVIEW_FIELDS,
    MODEL_OPTIONAL_SCENE_REVIEW_FIELDS,
    MODEL_TOP_LEVEL_FIELDS,
    OBSERVATION_FIELDS, ROUND_DECISIONS, ROUND_REASON_CODES,
    ROUND_VALIDATOR_ABSOLUTE_MAX_ATTEMPTS, ROUND_VALIDATOR_MAX_ATTEMPTS,
    ROUND_VALIDATOR_PROMPT_VERSION,
    ROUND_VALIDATOR_RETRY_PROMPT_VERSION, ROUND_VALIDATOR_SCHEMA_VERSION,
    ROUND_VALIDATOR_TERMINAL_INSTRUCTION,
    SCENE_DECISIONS, SCENE_REASON_CODES, SCENE_REVIEW_FIELDS,
    TOP_LEVEL_FIELDS, VERIFICATION_STATUSES, VERIFIED_CONNECTION_FIELDS,
    VERIFIED_CONNECTION_KINDS, round_validator_output_schema,
)
from .relation_registry import RELATION_TYPE_REGISTRY
from .round_validator_contract_repair_v11 import (
    ROUND_VALIDATOR_CONTRACT_REPAIR_PROMPT_VERSION_V11,
    apply_round_validator_contract_repair,
    build_round_validator_contract_repair_plan,
    build_round_validator_contract_repair_prompt,
)


ROUND_VALIDATOR_INPUT_VERSION = (
    "round_expansion_validator_compact_input_v8_reference_scope_index"
)
ROUND_VALIDATOR_INPUT_PROJECTION_VERSION = (
    "round_validator_realized_scene_projection_v4_compact_reference_scopes"
)
ROUND_VALIDATOR_RETRY_POLICY_VERSION = (
    "round_validator_retry_policy_v5_dialogue_only_top_level_defaults"
)
ROUND_VALIDATOR_CONTEXT_LIMIT = 40_960
ROUND_VALIDATOR_MAX_OUTPUT_TOKENS = 12_288
ROUND_VALIDATOR_PROMPT_SAFETY_MARGIN = 1_024
DURABLE_UPDATE_POLICIES = frozenset({"strict", "dialogue_only"})
DIALOGUE_ONLY_DEGRADABLE_EVIDENCE_SUFFIXES = frozenset({
    "progression_evidence_not_action_evidence",
    "dependency_context_source_not_declared",
})
_FINAL_ID_FIELDS = frozenset({
    "memory_id", "memory_ref", "final_memory_id", "entity_id", "final_entity_id",
    "event_id", "final_event_id", "relation_id", "final_relation_id",
    "thread_id", "handoff_id", "logical_key", "session_ref", "turn_id",
})


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
    separators=(",", ":")).encode("utf-8")).hexdigest()


def _compact_action(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    fields = (
        "action_ref", "action_mode", "continuation_of", "root_action_ref",
        "resolved_action_type", "resolved_arguments", "action_type", "arguments",
        "progression",
    )
    return {key: deepcopy(value.get(key)) for key in fields if key in value}


def _compact_participants(value: Any) -> list[dict[str, Any]]:
    result = []
    for row in value if isinstance(value, list) else []:
        if not isinstance(row, Mapping):
            continue
        result.append({key: deepcopy(row.get(key)) for key in (
            "participant_ref", "display_name", "scene_role") if key in row})
    return result


def _compact_realized_scene(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep one semantic plan/dialogue copy and remove compiler bookkeeping."""
    return {
        "scene_ref": value.get("scene_ref"),
        "scene_alias": value.get("scene_alias"),
        "source_anchor_id": value.get("source_anchor_id"),
        "placement_choice_id": value.get("placement_choice_id"),
        "placement": value.get("placement"),
        "link_type": value.get("link_type"),
        "bridge_reason": value.get("bridge_reason"),
        "new_development": value.get("new_development"),
        "goal": value.get("goal"),
        "linked_existing_refs": deepcopy(value.get("linked_existing_refs") or {}),
        "speaker_participants": _compact_participants(
            value.get("speaker_participants")),
        "relationship_dynamic": value.get("relationship_dynamic"),
        "interaction_beats": deepcopy(value.get("interaction_beats") or []),
        "predecessor_scene_refs": deepcopy(
            value.get("predecessor_scene_refs") or []),
        "dependency_edges": deepcopy(value.get("dependency_edges") or []),
        "branch_role": value.get("branch_role"),
        "merge_target_scene_ref": value.get("merge_target_scene_ref"),
        "primary_action": _compact_action(value.get("primary_action")),
        "secondary_action": _compact_action(value.get("secondary_action")),
        "desired_outcome": value.get("desired_outcome"),
        "dialogue_turns": deepcopy(value.get("dialogue_turns") or []),
    }


def _compact_historical_context(
    review_world_view: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anchor_index: list[dict[str, Any]] = []
    turns_by_ref: dict[str, dict[str, Any]] = {}
    for anchor in review_world_view.get("historical_anchors") or []:
        if not isinstance(anchor, Mapping):
            continue
        source_refs = [str(ref) for ref in anchor.get("source_refs") or []]
        anchor_index.append({
            "source_anchor_id": anchor.get("source_anchor_id"),
            "source_kind": anchor.get("source_kind"),
            "source_refs": source_refs,
        })
        for turn in anchor.get("exact_excerpt") or []:
            if not isinstance(turn, Mapping):
                continue
            turn_ref = str(turn.get("turn_ref") or "")
            compact = {key: deepcopy(turn.get(key)) for key in (
                "turn_ref", "speaker_ref", "speaker_label", "text") if key in turn}
            previous = turns_by_ref.get(turn_ref)
            if previous is not None and previous != compact:
                raise ValueError(f"round_validator_historical_turn_conflict:{turn_ref}")
            turns_by_ref.setdefault(turn_ref, compact)
    return anchor_index, list(turns_by_ref.values())


def _used_action_contracts(
    realized_scenes: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for scene in realized_scenes:
        for action in (scene.get("primary_action"), scene.get("secondary_action")):
            if not isinstance(action, Mapping):
                continue
            action_type = str(action.get("resolved_action_type") or
                              action.get("action_type") or "")
            contract = action.get("action_contract") or {}
            if action_type and isinstance(contract, Mapping):
                result[action_type] = {
                    "validator_checks": deepcopy(contract.get("validator_checks"))
                }
    return dict(sorted(result.items()))


def _compact_local_gate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"status": None, "valid": False, "errors": ["gate_missing"]}
    return {
        "status": value.get("status"),
        "valid": value.get("valid"),
        "errors": deepcopy(value.get("errors") or []),
    }


def _compact_related_world_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result = deepcopy(dict(value))
    result["entities"] = [{
        key: deepcopy(row.get(key)) for key in (
            "entity_ref", "display_name", "entity_type", "origin",
            "verification_status") if key in row
    } for row in value.get("entities") or [] if isinstance(row, Mapping)]
    return result


def _compact_interaction_links(value: Any) -> list[dict[str, Any]]:
    return [{
        key: deepcopy(row.get(key)) for key in (
            "interaction_link_ref", "participant_refs", "link_type", "origin",
            "verification_status") if key in row
    } for row in value or [] if isinstance(row, Mapping)]


def _project_narrative_order(
    value: Any, *, realized_scene_refs: set[str],
) -> dict[str, Any]:
    """Expose only scenes that actually reached round review.

    The full narrative-order manifest remains a run artifact and is still used by local
    closure.  This is a prompt-only projection; retaining unavailable planned scenes here
    invites the model to review dialogue that does not exist.
    """
    source = deepcopy(dict(value)) if isinstance(value, Mapping) else {}
    result = {key: deepcopy(source.get(key)) for key in (
        "narrative_order_manifest_version", "root_id", "round_index", "storage_mode",
    ) if key in source}
    result["generation_order_scene_refs"] = [
        ref for ref in source.get("generation_order_scene_refs") or []
        if str(ref) in realized_scene_refs
    ]
    result["ordered_scene_refs"] = [
        ref for ref in source.get("ordered_scene_refs") or []
        if str(ref) in realized_scene_refs
    ]
    scenes = []
    for row in source.get("scenes") or []:
        if (not isinstance(row, Mapping) or
                str(row.get("scene_ref") or "") not in realized_scene_refs):
            continue
        projected = {key: deepcopy(row.get(key)) for key in (
            "scene_ref", "generation_order_index", "narrative_order_key", "placement",
            "source_anchor_id", "placement_choice_id", "after_turn_ref",
            "before_turn_ref", "dependency_scene_refs", "branch_role",
            "merge_target_scene_ref",
        ) if key in row}
        projected["dependency_scene_refs"] = [
            ref for ref in projected.get("dependency_scene_refs") or []
            if str(ref) in realized_scene_refs
        ]
        if (projected.get("merge_target_scene_ref") is not None and
                str(projected.get("merge_target_scene_ref")) not in realized_scene_refs):
            projected["merge_target_scene_ref"] = None
        scenes.append(projected)
    result["scenes"] = scenes
    result["narrative_order_projection_sha256"] = _sha(result)
    return result


def _project_narrative_constraints(
    values: Any, *, realized_scene_refs: set[str],
) -> list[dict[str, Any]]:
    """Keep knowledge boundaries without repeating every already-known turn ref."""
    fields = (
        "scene_ref", "source_anchor_id", "placement_choice_id", "placement",
        "following_constraint_only_refs", "following_context_role",
        "same_round_predecessor_refs", "predecessor_dialogue_sha256",
        "context_snapshot_sha256",
    )
    return [
        {key: deepcopy(row.get(key)) for key in fields if key in row}
        for row in values or []
        if (isinstance(row, Mapping) and
            str(row.get("scene_ref") or "") in realized_scene_refs)
    ]


def _project_scene_dependency_dag(
    value: Mapping[str, Any], *, realized_scene_refs: set[str],
) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["nodes"] = [
        ref for ref in result.get("nodes") or [] if str(ref) in realized_scene_refs
    ]
    result["edges"] = [
        deepcopy(dict(edge)) for edge in result.get("edges") or []
        if (isinstance(edge, Mapping) and
            str(edge.get("from_scene_ref") or "") in realized_scene_refs and
            str(edge.get("to_scene_ref") or "") in realized_scene_refs)
    ]
    result.pop("scene_dependency_dag_sha256", None)
    result["scene_dependency_projection_sha256"] = _sha(result)
    return result


def _project_realized_scene(
    value: Mapping[str, Any], *, realized_scene_refs: set[str],
) -> dict[str, Any]:
    result = _compact_realized_scene(value)
    result["predecessor_scene_refs"] = [
        ref for ref in result.get("predecessor_scene_refs") or []
        if str(ref) in realized_scene_refs
    ]
    result["dependency_edges"] = [
        deepcopy(dict(edge)) for edge in result.get("dependency_edges") or []
        if (isinstance(edge, Mapping) and
            str(edge.get("predecessor_scene_ref") or "") in realized_scene_refs)
    ]
    if (result.get("merge_target_scene_ref") is not None and
            str(result.get("merge_target_scene_ref")) not in realized_scene_refs):
        result["merge_target_scene_ref"] = None
    return result


def build_round_validator_input(
    *, round_blueprint: Mapping[str, Any], review_world_view: Mapping[str, Any],
    realized_scenes: Sequence[Mapping[str, Any]], scene_dependency_dag: Mapping[str, Any],
    structural_gate_results: Mapping[str, Any],
) -> dict[str, Any]:
    anchor_index, historical_turns = _compact_historical_context(review_world_view)
    blueprint = round_blueprint.get("round_blueprint") or {}
    realized_scene_refs = {
        str(row.get("scene_ref") or "") for row in realized_scenes
        if isinstance(row, Mapping) and row.get("scene_ref")
    }
    projected_scenes = [_project_realized_scene(
        row, realized_scene_refs=realized_scene_refs) for row in realized_scenes]
    scenes_by_ref = {str(row.get("scene_ref") or ""): row
                     for row in projected_scenes}

    def dependency_ancestry(scene_ref: str) -> list[str]:
        seen: set[str] = set()
        stack = list((scenes_by_ref.get(scene_ref) or {}).get(
            "predecessor_scene_refs") or [])
        while stack:
            ref = str(stack.pop())
            if ref in seen or ref not in scenes_by_ref:
                continue
            seen.add(ref)
            stack.extend(scenes_by_ref[ref].get("predecessor_scene_refs") or [])
        return sorted(seen)

    for scene in projected_scenes:
        scene_ref = str(scene.get("scene_ref") or "")
        scene["review_reference_scope"] = {
            "historical_source_anchor_id": str(
                scene.get("source_anchor_id") or ""),
            "current_scene_ref": scene_ref,
            "dependency_scene_refs": dependency_ancestry(scene_ref),
        }
    value = {
        "round_validator_input_version": ROUND_VALIDATOR_INPUT_VERSION,
        "input_projection_version": ROUND_VALIDATOR_INPUT_PROJECTION_VERSION,
        "root_id": (review_world_view.get("root_id") or
                    round_blueprint.get("root_id")),
        "round_summary": {
            "round_index": round_blueprint.get("round_index"),
            "scene_budget": deepcopy(round_blueprint.get("scene_budget") or {}),
            "target_added_turns": deepcopy(
                round_blueprint.get("target_added_turns") or {}),
            "round_goal": blueprint.get("round_goal"),
            "starting_situation": blueprint.get("starting_situation"),
            "central_development": blueprint.get("central_development"),
            "ending_direction": blueprint.get("ending_direction"),
        },
        "scenes": projected_scenes,
        "historical_anchor_index": anchor_index,
        "historical_turns": historical_turns,
        "related_world_state": _compact_related_world_state(
            review_world_view.get("related_planning_state") or {}),
        "interaction_links": _compact_interaction_links(
            review_world_view.get("interaction_links") or []),
        "explicit_relations": deepcopy(
            review_world_view.get("explicit_relations") or []),
        "narrative_constraints": _project_narrative_constraints(
            review_world_view.get("narrative_constraints") or [],
            realized_scene_refs=realized_scene_refs),
        "narrative_order": _project_narrative_order(
            review_world_view.get("narrative_order") or {},
            realized_scene_refs=realized_scene_refs),
        "scene_dependency_dag": _project_scene_dependency_dag(
            scene_dependency_dag, realized_scene_refs=realized_scene_refs),
        "action_contracts_by_type": _used_action_contracts(realized_scenes),
        "structural_gate_results": {
            str(scene_ref): _compact_local_gate(row)
            for scene_ref, row in structural_gate_results.items()
            if str(scene_ref) in realized_scene_refs
        },
        "excluded_domains": ["qa_gold", "solver_reward", "questioner", "distractor",
                             "parent_utterance_memory_mirror"],
    }
    value["frozen_round_validator_input_sha256"] = _sha(value)
    return value


def audit_round_validator_prompt_budget(
    value: Mapping[str, Any], *, token_counter: Callable[[str], int],
    context_limit: int = ROUND_VALIDATOR_CONTEXT_LIMIT,
    max_output_tokens: int = ROUND_VALIDATOR_MAX_OUTPUT_TOKENS,
    safety_margin_tokens: int = ROUND_VALIDATOR_PROMPT_SAFETY_MARGIN,
    durable_update_policy: str = "strict",
) -> dict[str, Any]:
    prompt = build_round_validator_prompt(
        value, durable_update_policy=durable_update_policy)
    prompt_tokens = int(token_counter(prompt))
    total_reserved = prompt_tokens + int(max_output_tokens) + int(safety_margin_tokens)
    return {
        "budget_audit_version": "round_validator_prompt_budget_audit_v1",
        "round_validator_input_version": value.get("round_validator_input_version"),
        "durable_update_policy": durable_update_policy,
        "prompt_tokens": prompt_tokens,
        "max_output_tokens": int(max_output_tokens),
        "safety_margin_tokens": int(safety_margin_tokens),
        "context_limit": int(context_limit),
        "total_reserved_tokens": total_reserved,
        "remaining_tokens": int(context_limit) - total_reserved,
        "valid": total_reserved <= int(context_limit),
    }


def build_round_validator_prompt(
    value: Mapping[str, Any], *, durable_update_policy: str = "strict",
) -> str:
    if durable_update_policy not in DURABLE_UPDATE_POLICIES:
        raise ValueError(
            f"round_validator_durable_update_policy_invalid:{durable_update_policy}")
    selection_marker = "!"

    def allowed_connection_kinds(action: Mapping[str, Any]) -> list[str]:
        arguments = action.get("resolved_arguments") or action.get("arguments") or {}
        legacy = arguments.get("connection_to_existing_world") or {}
        target_type = str(arguments.get("connection_type") or
                          legacy.get("connection_type") or "")
        return sorted({
            "existing_person": {"interaction_link", "explicit_relation"},
            "person": {"interaction_link", "explicit_relation"},
            "relation": {"interaction_link", "explicit_relation"},
            "interaction_link": {"interaction_link", "explicit_relation"},
            "existing_event": {"event_participant"},
            "event": {"event_participant"},
            "existing_task": {"task_participant"},
            "task": {"task_participant"},
            "information_source": {"information_source"},
            "open_thread": {"thread_participant"},
        }.get(target_type, set()))

    def action_template(action: Any) -> dict[str, Any] | None:
        if not isinstance(action, Mapping):
            return None
        result = {
            "status": None,
            "evidence_turn_refs": [selection_marker],
            "progression_evidence_turn_refs": [],
            "reason": "",
        }
        if action.get("action_mode") == "continuation":
            result["progression_evidence_turn_refs"] = [selection_marker]
        return result

    scene_templates = []
    continuation_slots = []
    durable_candidate_slots = []
    for scene in value.get("scenes") or []:
        scene_alias = str(scene.get("scene_alias") or "")
        scene_templates.append({
            "scene_alias": scene.get("scene_alias"), "decision": "accept",
            "historical_connection_status": "valid", "continuity_status": "valid",
            "future_knowledge_leak": False,
            "historical_consistency_status": "valid",
            "semantic_duplicate_status": "not_duplicate",
            "new_development_status": "valid", "reason_codes": ["supported"],
            "historical_supporting_turn_refs": [selection_marker],
            "current_scene_supporting_turn_refs": [selection_marker],
            "dependency_context_turn_refs": [], "uses_scene_refs": [],
            "scene_requires_update_commit": False,
            "primary_action_realization": action_template(
                scene.get("primary_action")),
            "secondary_action_realization": action_template(
                scene.get("secondary_action")),
        })
        for role in ("primary", "secondary"):
            action = scene.get(f"{role}_action")
            if not isinstance(action, Mapping):
                continue
            action_type = str(action.get("resolved_action_type") or
                              action.get("action_type") or "")
            action_mode = str(action.get("action_mode") or "")
            slot = {
                "scene_alias": scene_alias,
                "scene_ref": scene.get("scene_ref"),
                "source_action_role": role,
                "action_ref": action.get("action_ref"),
                "action_mode": action_mode,
                "action_type": action_type,
            }
            if action_mode == "continuation":
                continuation_slots.append({
                    "action_ref": action.get("action_ref"),
                    "required_output_fields": sorted(
                        MODEL_CONTINUATION_ACTION_REALIZATION_FIELDS),
                    "progression": deepcopy(action.get("progression")),
                })
            if (durable_update_policy == "strict" and
                    durable_candidate_action_eligible(action)):
                durable_candidate_slots.append({
                    "scene_ref": scene.get("scene_ref"),
                    "source_action_role": role,
                    "action_ref": action.get("action_ref"),
                    "action_type": action_type,
                    "allowed_verified_connection_kinds": (
                        allowed_connection_kinds(action)
                        if action_type == "introduce_connected_person" else []),
                })
    verified_connection_shape = {
        "verification_status": None,
        "connection_kind": None,
        "relation_type": None,
        "evidence_turn_refs": [selection_marker],
        "reason": "",
    }
    exact_shapes = {
        "durable_update": {field: ({
            "candidate_update_key": "DU001", "source_scene_ref": "R01:S01",
            "source_action_role": "primary",
            "evidence_turn_refs": [selection_marker],
            "verification_status": None,
            "atomic_group_key": None, "depends_on_candidate_keys": [],
        }[field]) for field in sorted(DURABLE_UPDATE_FIELDS)},
        "observation": {field: ({
            "observation_key": "OBS001", "update_type": "add_relation",
            "source_scene_refs": [], "evidence_turn_refs": [],
            "description": "audit-only unhinted observation",
            "disposition": "audit_only_unhinted_update",
        }[field]) for field in sorted(OBSERVATION_FIELDS)},
    }
    exact_shapes["connected_person_durable_update"] = {
        **deepcopy(exact_shapes["durable_update"]),
        "verified_connection": verified_connection_shape,
    }
    durable_templates = []
    for index, slot in enumerate(durable_candidate_slots, 1):
        template = {
            "atomic_group_key": None,
            "candidate_update_key": f"DU{index:03d}",
            "depends_on_candidate_keys": [],
            "evidence_turn_refs": [selection_marker],
            "source_action_role": slot["source_action_role"],
            "source_scene_ref": slot["scene_ref"],
            "verification_status": None,
        }
        if slot["action_type"] == "introduce_connected_person":
            template["verified_connection"] = deepcopy(verified_connection_shape)
        durable_templates.append(template)
    durable_scope_rule = (
        "DIALOGUE-ONLY MODE: review every scene in order. Return durable_updates=[] "
        "exactly. Do not propose, infer, or describe Memory updates in durable_updates; "
        "dialogue acceptance is independent of Memory growth."
        if durable_update_policy == "dialogue_only" else
        "Durable items are flat. verified_connection is only for a NEW person; "
        "relation_type is null except explicit_relation."
    )
    final_scope_rule = (
        "durable_updates must be the exact empty array []. Focus the available output "
        "budget on one complete review for every listed scene."
        if durable_update_policy == "dialogue_only" else
        "durable_updates cite only eligible slots; Do not create one candidate per "
        "scene/action. NEW person candidates need connection evidence. An event naming "
        "a round-local person depends on its person candidate."
    )
    return "\n\n".join([
        f"ROUND EXPANSION VALIDATOR — {ROUND_VALIDATOR_SCHEMA_VERSION}. Return JSON only.",
        "Only review aliases in scenes. [] updates is valid. Keep ref scopes. Primary unrealized "
        "normally rejects; secondary may be partial. Continuations require an accepted realized "
        "ancestor plus action/progression evidence. No operations, final IDs, or injected "
        "identities. Candidate keys are unique. " + durable_scope_rule + " Each scene carries "
        "review_reference_scope is an index, not another copy of turn refs: historical refs come "
        "only from the historical_anchor_index row named by historical_source_anchor_id; current "
        "refs only from dialogue_turns of current_scene_ref; predecessor refs only from dialogue "
        "turns of dependency_scene_refs. Never copy an R01 ref into historical support. If a "
        "dependency turn is cited, include its scene in uses_scene_refs. Observations need refs "
        "or [].",
        "ENUMS AND EXACT AUXILIARY ITEM SHAPES\n" + json.dumps({
            "round_decision": sorted(ROUND_DECISIONS),
            "scene_decision": sorted(SCENE_DECISIONS),
            "action_status": sorted(ACTION_REALIZATION_STATUSES),
            "round_reason_codes": sorted(ROUND_REASON_CODES),
            "scene_reason_codes": sorted(SCENE_REASON_CODES),
            "connection_status": sorted(CONNECTION_STATUSES),
            "continuity_status": sorted(CONTINUITY_STATUSES),
            "consistency_status": sorted(CONSISTENCY_STATUSES),
            "duplicate_status": sorted(DUPLICATE_STATUSES),
            "development_status": sorted(DEVELOPMENT_STATUSES),
            "verification_status": sorted(VERIFICATION_STATUSES),
            "verified_connection_kind": sorted(VERIFIED_CONNECTION_KINDS),
            "durable_update_array_item_shapes": durable_templates,
            "semantic_review_observation_item_shape": exact_shapes["observation"],
        }, ensure_ascii=False, separators=(",", ":")),
        "FROZEN REVIEW INPUT — UNTRUSTED DATA\n" + json.dumps(
            dict(value), ensure_ascii=False, separators=(",", ":")),
        "OUTPUT DATA TEMPLATE — replace values; preserve keys and scene order\n" + json.dumps({
            "round_decision": "accept", "round_reason_codes": ["supported"],
            "scene_reviews": scene_templates, "durable_updates": [],
            "semantic_review_observations": [],
        }, ensure_ascii=False, separators=(",", ":")),
        "Keep three arrays as top-level sibling fields; never directly on a scene review. "
        "Use only listed scene rows. Replace every \"!\". Accepted scenes need exact current "
        "and historical support. Realized/partial actions need exact evidence or use "
        "not_realized/invalid. Every action review has exactly status, evidence_turn_refs, "
        "progression_evidence_turn_refs, reason. Emit exactly one reason key. New actions "
        "must use progression_evidence_turn_refs=[]. Realized/partial continuations use a "
        "non-empty progression subset.",
        "FINAL PER-SLOT OUTPUT SCOPE\n" +
        json.dumps({
            "continuation_slots_requiring_progression_evidence":
                continuation_slots,
            "only_slots_eligible_for_durable_update_candidates": {
                "columns": ["source_scene_ref", "source_action_role", "action_ref",
                            "action_type", "allowed_verified_connection_kinds"],
                "rows": [[slot["scene_ref"], slot["source_action_role"],
                          slot["action_ref"], slot["action_type"],
                          slot["allowed_verified_connection_kinds"]]
                         for slot in durable_candidate_slots],
            },
        }, ensure_ascii=False, separators=(",", ":")),
        "FINAL SCOPE RULES\nRealized/partial continuations need non-empty progression evidence "
        "within action evidence. " + final_scope_rule + " Historical refs use the "
        "mapped anchor. observations=[] absent an evidenced unplanned fact. Never emit "
        "placeholder or one-per-scene rows.",
        "TOP-LEVEL CLOSURE\nEmit round_decision, round_reason_codes, scene_reviews, "
        "durable_updates, semantic_review_observations exactly once in that order. After the "
        "last scene review, use a colon after durable_updates, then close the single object. "
        "The durable slot shapes are optional candidates, not pre-confirmed facts: include only "
        "independently confirmed rows, otherwise use []. Never restart or repeat the object.",
        "Begin exactly {\"round_decision\":.",
        ROUND_VALIDATOR_TERMINAL_INSTRUCTION,
    ])


def build_round_validator_retry_prompt(
    *, frozen_input_sha256: str, contract_errors: Sequence[str],
    frozen_base_prompt: str,
) -> str:
    return "\n\n".join([
        f"ROUND VALIDATOR CONTRACT RETRY — {ROUND_VALIDATOR_RETRY_PROMPT_VERSION}.",
        f"FROZEN INPUT SHA256\n{frozen_input_sha256}",
        "Repair JSON/schema/reference errors only. Do not change review standards and do "
        "not seek confirmation. Previous raw output is intentionally omitted. Return the "
        "review DATA object requested by the frozen prompt, never its schema definition.",
        "ERRORS\n" + json.dumps(list(contract_errors), separators=(",", ":")),
        frozen_base_prompt,
    ])


def normalize_round_validator_output(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return {"parsed": deepcopy(dict(raw)), "parse_valid": True, "errors": []}
    if not isinstance(raw, str):
        return {"parsed": None, "parse_valid": False,
                "errors": ["validator_parse_invalid_type"]}
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate_json_object_key:{key}")
            result[key] = item
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        return {"parsed": None, "parse_valid": False,
                "errors": [f"validator_parse_invalid_json:{exc}"]}
    if not isinstance(value, Mapping):
        return {"parsed": None, "parse_valid": False,
                "errors": ["validator_parse_not_object"]}
    return {"parsed": dict(value), "parse_valid": True, "errors": []}


def materialize_round_validator_optional_model_fields(output: Any) -> Any:
    """Fill contract defaults without changing any semantic review decision."""
    result = deepcopy(output)
    if not isinstance(result, dict):
        return result
    reviews = result.get("scene_reviews")
    if not isinstance(reviews, list):
        return result
    for review in reviews:
        if not isinstance(review, dict):
            continue
        for key in MODEL_OPTIONAL_SCENE_REVIEW_FIELDS:
            review.setdefault(key, [])
    return result


def materialize_dialogue_only_top_level_defaults(output: Any) -> dict[str, Any]:
    """Fill only model-omitted arrays that are deterministic in dialogue-only mode.

    The dialogue-only prompt requires ``durable_updates=[]`` and treats absent
    unplanned observations as ``semantic_review_observations=[]``.  Omitting either
    trailing array does not change a completed scene review, so local materialization
    avoids spending a full-model retry on a deterministic no-op.  Unknown keys and
    missing review/decision fields remain hard schema failures.
    """
    result = deepcopy(output)
    materialized_fields: list[str] = []
    required_review_fields = {
        "round_decision", "round_reason_codes", "scene_reviews",
    }
    defaultable_fields = {
        "durable_updates", "semantic_review_observations",
    }
    if not isinstance(result, dict):
        return {"output": result, "materialized_fields": materialized_fields}
    keys = set(result)
    if (not required_review_fields.issubset(keys) or
            not keys.issubset(required_review_fields | defaultable_fields)):
        return {"output": result, "materialized_fields": materialized_fields}
    for field in ("durable_updates", "semantic_review_observations"):
        if field not in result:
            result[field] = []
            materialized_fields.append(field)
    return {"output": result, "materialized_fields": materialized_fields}


def rehome_dialogue_only_trailing_empty_arrays(
    output: Any, *, expected_scene_count: int,
) -> dict[str, Any]:
    """Move one exact dialogue-only closure row out of ``scene_reviews``.

    Some model responses close the scene array one object too late and emit the two
    required empty arrays as a final pseudo-review.  Rehoming is deterministic only
    when the pseudo-review is exact, both values are empty, and all expected scene
    reviews precede it.  Every other malformed row remains a contract failure.
    """
    result = deepcopy(output)
    rehomed = False
    if not isinstance(result, dict):
        return {"output": result, "rehomed": rehomed}
    reviews = result.get("scene_reviews")
    if (not isinstance(reviews, list) or
            len(reviews) != int(expected_scene_count) + 1):
        return {"output": result, "rehomed": rehomed}
    trailing = reviews[-1]
    exact = {
        "durable_updates": [],
        "semantic_review_observations": [],
    }
    if trailing != exact:
        return {"output": result, "rehomed": rehomed}
    if any(field in result and result[field] != [] for field in exact):
        return {"output": result, "rehomed": rehomed}
    result["scene_reviews"] = reviews[:-1]
    result.update(exact)
    return {"output": result, "rehomed": True}


def _optional_model_field_materialization_eligible(output: Any) -> bool:
    if not isinstance(output, Mapping) or set(output) != MODEL_TOP_LEVEL_FIELDS:
        return False
    reviews = output.get("scene_reviews")
    if not isinstance(reviews, list):
        return False
    allowed_shapes = {
        MODEL_SCENE_REVIEW_FIELDS,
        MODEL_SCENE_REVIEW_FIELDS - MODEL_OPTIONAL_SCENE_REVIEW_FIELDS,
    }
    return all(isinstance(review, Mapping) and set(review) in allowed_shapes
               for review in reviews)


def migrate_historical_connected_person_validator_fixture_v5(
    output: Any, *, realized_scenes: Sequence[Mapping[str, Any]],
) -> Any:
    """Explicitly migrate read-only pre-v5 validator fixtures for shadow tests.

    The live path never invokes this helper.  It adds only the new typed
    connection review envelope; it does not confirm a new fact, change a
    decision, or write Memory.
    """
    result = deepcopy(output)
    actions: dict[tuple[str, str], Mapping[str, Any]] = {}
    for scene in realized_scenes:
        scene_ref = str(scene.get("scene_ref") or "")
        for role in ("primary", "secondary"):
            action = scene.get(f"{role}_action")
            if isinstance(action, Mapping):
                actions[(scene_ref, role)] = action

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            updates = value.get("durable_updates")
            if isinstance(updates, list):
                for update in updates:
                    if not isinstance(update, dict) or "verified_connection" in update:
                        continue
                    action = actions.get((str(update.get("source_scene_ref") or ""),
                                          str(update.get("source_action_role") or "")))
                    action_type = str((action or {}).get("resolved_action_type") or
                                      (action or {}).get("action_type") or "")
                    if action_type != "introduce_connected_person":
                        continue
                    args = (action or {}).get("resolved_arguments") or \
                        (action or {}).get("arguments") or {}
                    old = args.get("connection_to_existing_world") or {}
                    target_type = str(args.get("connection_type") or
                                      old.get("connection_type") or "person")
                    kind = {
                        "person": "interaction_link", "relation": "interaction_link",
                        "interaction_link": "interaction_link",
                        "existing_person": "interaction_link",
                        "event": "event_participant", "existing_event": "event_participant",
                        "task": "task_participant", "existing_task": "task_participant",
                        "information_source": "information_source",
                        "open_thread": "thread_participant",
                    }.get(target_type, "interaction_link")
                    update["verified_connection"] = {
                        "verification_status": deepcopy(
                            update.get("verification_status")),
                        "connection_kind": kind, "relation_type": None,
                        "evidence_turn_refs": deepcopy(
                            update.get("evidence_turn_refs") or []),
                        "reason": "explicit historical shadow migration; no Memory promotion",
                    }
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result)
    return result


def normalize_exact_duplicate_scene_reviews(output: Any) -> dict[str, Any]:
    """Apply lossless scene-review compatibility normalization.

    ``duplicate_status`` was the model-facing name before the compiled contract
    standardized on ``semantic_duplicate_status``. Renaming the sole legacy
    field changes no decision or evidence. If both names are present, leave the
    object untouched so the strict schema rejects the ambiguity.

    Byte-equivalent repeated reviews are also removed; conflicting repeats
    remain invalid.
    """
    result = deepcopy(output)
    removed: list[dict[str, Any]] = []
    renamed: list[dict[str, str]] = []
    if not isinstance(result, dict) or not isinstance(result.get("scene_reviews"), list):
        return {"output": result, "removed_exact_duplicates": removed,
                "renamed_scene_fields": renamed}
    for review in result["scene_reviews"]:
        if (isinstance(review, dict) and "duplicate_status" in review and
                "semantic_duplicate_status" not in review):
            review["semantic_duplicate_status"] = review.pop("duplicate_status")
            renamed.append({"scene_alias": str(review.get("scene_alias") or ""),
                            "from": "duplicate_status",
                            "to": "semantic_duplicate_status"})
    unique: list[Any] = []
    first_by_alias: dict[str, Any] = {}
    for index, review in enumerate(result["scene_reviews"]):
        alias = str(review.get("scene_alias") or "") if isinstance(
            review, Mapping) else ""
        first = first_by_alias.get(alias) if alias else None
        if first is not None and review == first:
            removed.append({"scene_alias": alias, "duplicate_index": index})
            continue
        if alias and first is None:
            first_by_alias[alias] = review
        unique.append(review)
    result["scene_reviews"] = unique
    return {"output": result, "removed_exact_duplicates": removed,
            "renamed_scene_fields": renamed}


def normalize_unambiguous_action_review_fields(
    output: Any, *, realized_scenes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rehome a misplaced progression list only when its owner is unique.

    This compatibility step moves submitted data without selecting, deleting, or
    changing evidence. Ambiguous ownership remains a schema error.
    """
    result = deepcopy(output)
    moves: list[dict[str, str]] = []
    if not isinstance(result, dict) or not isinstance(
            result.get("scene_reviews"), list):
        return {"output": result, "moves": moves}
    scenes_by_alias = {
        str(scene.get("scene_alias") or ""): scene
        for scene in realized_scenes if isinstance(scene, Mapping)
    }
    for review in result["scene_reviews"]:
        if not isinstance(review, dict) or \
                "progression_evidence_turn_refs" not in review:
            continue
        alias = str(review.get("scene_alias") or "")
        scene = scenes_by_alias.get(alias)
        if not isinstance(scene, Mapping):
            continue
        candidates: list[tuple[str, dict[str, Any]]] = []
        for role in ("primary", "secondary"):
            planned = scene.get(f"{role}_action")
            realization = review.get(f"{role}_action_realization")
            if (isinstance(planned, Mapping) and
                    planned.get("action_mode") == "continuation" and
                    isinstance(realization, dict) and
                    "progression_evidence_turn_refs" not in realization):
                candidates.append((role, realization))
        if len(candidates) != 1:
            continue
        role, realization = candidates[0]
        realization["progression_evidence_turn_refs"] = deepcopy(
            review["progression_evidence_turn_refs"])
        review.pop("progression_evidence_turn_refs")
        moves.append({
            "scene_alias": alias,
            "field": "progression_evidence_turn_refs",
            "source": "scene_review",
            "target": f"{role}_action_realization",
        })
    return {"output": result, "moves": moves}


def compile_round_validator_model_output(
    output: Any, *, realized_scenes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Inject deterministic action identity into a semantic-only model response."""
    normalization = normalize_exact_duplicate_scene_reviews(output)
    output = normalization["output"]
    action_normalization = normalize_unambiguous_action_review_fields(
        output, realized_scenes=realized_scenes)
    output = action_normalization["output"]
    if not isinstance(output, Mapping):
        return {"valid": False, "classification": "schema_invalid",
                "errors": ["validator_model_output_not_object"],
                "compiled_output": None}
    if set(output) != MODEL_TOP_LEVEL_FIELDS:
        schema_definition = any(key in output for key in (
            "type", "properties", "required", "additionalProperties"))
        return {"valid": False, "classification": "schema_invalid",
                "errors": ["validator_returned_schema_definition" if schema_definition
                           else "validator_model_top_level_schema"],
                "compiled_output": None}
    scenes_by_alias = {str(row.get("scene_alias")): row for row in realized_scenes}
    reviews = output.get("scene_reviews")
    if not isinstance(reviews, list):
        return {"valid": False, "classification": "schema_invalid",
                "errors": ["validator_model_scene_reviews_not_list"],
                "compiled_output": None}
    errors: list[str] = []
    compiled_reviews = []
    ignored_new_progression_fields: list[str] = []

    def compile_action(value: Any, planned: Any, prefix: str) -> Any:
        if planned is None:
            if value is not None:
                errors.append(f"{prefix}:unexpected_without_planned_action")
            return None
        if not isinstance(value, Mapping):
            errors.append(f"{prefix}:planned_action_review_required")
            return None
        mode = str(planned.get("action_mode") or "")
        expected = (MODEL_CONTINUATION_ACTION_REALIZATION_FIELDS
                    if mode == "continuation"
                    else MODEL_NEW_ACTION_REALIZATION_FIELDS)
        legacy_new_fields = expected - {"progression_evidence_turn_refs"}
        if set(value) != expected and not (
                mode == "new" and set(value) == legacy_new_fields):
            errors.append(f"{prefix}:{mode}_model_schema")
            return None
        if (mode == "new" and "progression_evidence_turn_refs" in value and
                value.get("progression_evidence_turn_refs")):
            ignored_new_progression_fields.append(prefix)
        return {
            "action_ref": planned.get("action_ref"),
            "action_mode": mode,
            "continuation_of": planned.get("continuation_of"),
            "action_type": (planned.get("resolved_action_type") or
                            planned.get("action_type")),
            "status": deepcopy(value.get("status")),
            "evidence_turn_refs": deepcopy(value.get("evidence_turn_refs")),
            "progression_evidence_turn_refs": (
                [] if mode == "new" else deepcopy(
                    value.get("progression_evidence_turn_refs") or [])),
            "reason": deepcopy(value.get("reason")),
        }

    for index, review in enumerate(reviews):
        prefix = f"scene_review:{index}"
        if (not isinstance(review, Mapping) or
                set(review) not in {
                    MODEL_SCENE_REVIEW_FIELDS,
                    MODEL_SCENE_REVIEW_FIELDS - MODEL_OPTIONAL_SCENE_REVIEW_FIELDS,
                }):
            errors.append(f"{prefix}:model_schema")
            continue
        alias = str(review.get("scene_alias") or "")
        scene = scenes_by_alias.get(alias)
        if scene is None:
            errors.append(f"{prefix}:unknown_scene_alias:{alias}")
            continue
        compiled_review = {key: deepcopy(review.get(key)) for key in
                           MODEL_SCENE_REVIEW_FIELDS}
        for key in MODEL_OPTIONAL_SCENE_REVIEW_FIELDS:
            if key not in review:
                compiled_review[key] = []
        compiled_review["primary_action_realization"] = compile_action(
            review.get("primary_action_realization"), scene.get("primary_action"),
            f"{prefix}:primary_action_realization")
        compiled_review["secondary_action_realization"] = compile_action(
            review.get("secondary_action_realization"), scene.get("secondary_action"),
            f"{prefix}:secondary_action_realization")
        compiled_reviews.append(compiled_review)
    if errors:
        classification = ("reference_invalid" if any(
            "unknown_scene_alias" in error for error in errors)
            else "schema_invalid")
        return {"valid": False, "classification": classification,
                "errors": list(dict.fromkeys(errors)), "compiled_output": None}
    compiled = {
        "validator_schema_version": ROUND_VALIDATOR_SCHEMA_VERSION,
        "round_decision": deepcopy(output.get("round_decision")),
        "round_reason_codes": deepcopy(output.get("round_reason_codes")),
        "scene_reviews": compiled_reviews,
        "durable_updates": deepcopy(output.get("durable_updates")),
        "semantic_review_observations": deepcopy(
            output.get("semantic_review_observations")),
    }
    return {"valid": True, "classification": "valid", "errors": [],
            "compiled_output": compiled,
            "compatibility_normalization": {
                "removed_exact_duplicate_scene_reviews": normalization[
                    "removed_exact_duplicates"],
                "unambiguous_action_field_rehomes": action_normalization[
                    "moves"],
                "ignored_new_action_progression_fields":
                    ignored_new_progression_fields,
                "semantic_decision_changed": False,
            }}


def _find_final_fields(value: Any, path: str = "$validator") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _FINAL_ID_FIELDS:
                errors.append(f"validator_final_or_program_id_forbidden:{path}.{key}")
            errors.extend(_find_final_fields(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_find_final_fields(child, f"{path}[{index}]"))
    return errors


def validate_round_validator_output(
    output: Any, *, realized_scenes: Sequence[Mapping[str, Any]],
    review_world_view: Mapping[str, Any], local_gate_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(output, Mapping) or set(output) != TOP_LEVEL_FIELDS:
        return {"valid": False, "classification": "schema_invalid",
                "errors": ["validator_top_level_schema"]}
    if output.get("validator_schema_version") != ROUND_VALIDATOR_SCHEMA_VERSION:
        errors.append("validator_schema_version")
    if output.get("round_decision") not in ROUND_DECISIONS:
        errors.append("validator_round_decision")
    codes = output.get("round_reason_codes")
    if not isinstance(codes, list) or any(value not in ROUND_REASON_CODES for value in codes):
        errors.append("validator_round_reason_codes")
    errors.extend(_find_final_fields(output))
    scenes = {str(row.get("scene_ref")): row for row in realized_scenes}
    alias_to_ref = {str(row.get("scene_alias")): str(row.get("scene_ref"))
                    for row in realized_scenes}
    all_turn_refs = {str(turn.get("turn_ref"))
                     for scene in realized_scenes for turn in scene.get("dialogue_turns") or []}
    scene_turn_refs = {str(scene.get("scene_ref")): {
        str(turn.get("turn_ref")) for turn in scene.get("dialogue_turns") or []}
        for scene in realized_scenes}
    turn_ref_to_scene = {turn_ref: scene_ref for scene_ref, refs in
                         scene_turn_refs.items() for turn_ref in refs}
    direct_predecessors = {str(scene.get("scene_ref")): {
        str(ref) for ref in scene.get("predecessor_scene_refs") or []}
        for scene in realized_scenes}

    def dependency_ancestry(scene_ref: str) -> set[str]:
        result: set[str] = set()
        pending = list(direct_predecessors.get(scene_ref, set()))
        while pending:
            candidate = pending.pop()
            if candidate in result:
                continue
            result.add(candidate)
            pending.extend(direct_predecessors.get(candidate, set()))
        return result
    historical_turn_refs = {
        str(turn.get("turn_ref"))
        for anchor in review_world_view.get("historical_anchors") or []
        for turn in anchor.get("exact_excerpt") or []}
    reviews = output.get("scene_reviews")
    if not isinstance(reviews, list):
        errors.append("validator_scene_reviews_not_list")
        reviews = []
    seen_aliases: set[str] = set()
    rejected_refs: set[str] = set()
    action_evidence: dict[tuple[str, str], set[str]] = {}
    action_status: dict[tuple[str, str], str] = {}
    review_decision: dict[str, str] = {}
    action_ref_location: dict[str, tuple[str, str]] = {}
    for scene_ref, scene in scenes.items():
        for role, action in (("primary", scene.get("primary_action")),
                             ("secondary", scene.get("secondary_action"))):
            if isinstance(action, Mapping) and action.get("action_ref"):
                action_ref_location[str(action["action_ref"])] = (scene_ref, role)

    def validate_realization(value: Any, *, prefix: str, scene_ref: str,
                             role: str, planned: Any) -> None:
        if planned is None:
            if value is not None:
                errors.append(f"{prefix}:unexpected_without_planned_action")
            return
        if not isinstance(value, Mapping) or set(value) != ACTION_REALIZATION_FIELDS:
            errors.append(f"{prefix}:schema")
            return
        if value.get("action_type") != planned.get("action_type"):
            if value.get("action_type") != planned.get("resolved_action_type"):
                errors.append(f"{prefix}:action_type_mismatch")
        if value.get("action_ref") != planned.get("action_ref"):
            errors.append(f"{prefix}:action_ref_mismatch")
        if value.get("action_mode") != planned.get("action_mode"):
            errors.append(f"{prefix}:action_mode_mismatch")
        if value.get("continuation_of") != planned.get("continuation_of"):
            errors.append(f"{prefix}:continuation_of_mismatch")
        status = str(value.get("status") or "")
        if status not in ACTION_REALIZATION_STATUSES:
            errors.append(f"{prefix}:status")
        evidence = value.get("evidence_turn_refs")
        if not isinstance(evidence, list) or any(
                str(ref) not in scene_turn_refs.get(scene_ref, set()) for ref in evidence):
            errors.append(f"{prefix}:evidence_turn_refs")
            evidence = []
        if status in {"realized", "partially_realized"} and not evidence:
            errors.append(f"{prefix}:evidence_required")
        if status in {"not_realized", "invalid"} and evidence:
            errors.append(f"{prefix}:evidence_for_nonrealized")
        progression_evidence = value.get("progression_evidence_turn_refs")
        if not isinstance(progression_evidence, list) or any(
                str(ref) not in scene_turn_refs.get(scene_ref, set())
                for ref in progression_evidence):
            errors.append(f"{prefix}:progression_evidence_turn_refs")
            progression_evidence = []
        if planned.get("action_mode") == "continuation":
            if status in {"realized", "partially_realized"} and not progression_evidence:
                errors.append(f"{prefix}:continuation_progression_evidence_required")
            if any(str(ref) not in {str(item) for item in evidence}
                   for ref in progression_evidence):
                errors.append(f"{prefix}:progression_evidence_not_action_evidence")
        elif progression_evidence:
            errors.append(f"{prefix}:new_action_progression_evidence_forbidden")
        if not isinstance(value.get("reason"), str) or not value.get("reason", "").strip():
            errors.append(f"{prefix}:reason")
        action_evidence[(scene_ref, role)] = {str(ref) for ref in evidence}
        action_status[(scene_ref, role)] = status
    for index, review in enumerate(reviews):
        prefix = f"scene_review:{index}"
        if not isinstance(review, Mapping) or set(review) != SCENE_REVIEW_FIELDS:
            errors.append(f"{prefix}:schema")
            continue
        alias = str(review.get("scene_alias") or "")
        if alias not in alias_to_ref:
            errors.append(f"{prefix}:unknown_scene_alias:{alias}")
        if alias in seen_aliases:
            errors.append(f"{prefix}:duplicate_scene_alias:{alias}")
        seen_aliases.add(alias)
        if review.get("decision") not in SCENE_DECISIONS:
            errors.append(f"{prefix}:decision")
        if review.get("historical_connection_status") not in CONNECTION_STATUSES:
            errors.append(f"{prefix}:connection")
        if review.get("continuity_status") not in CONTINUITY_STATUSES:
            errors.append(f"{prefix}:continuity")
        if not isinstance(review.get("future_knowledge_leak"), bool):
            errors.append(f"{prefix}:future_knowledge")
        if review.get("historical_consistency_status") not in CONSISTENCY_STATUSES:
            errors.append(f"{prefix}:consistency")
        if review.get("semantic_duplicate_status") not in DUPLICATE_STATUSES:
            errors.append(f"{prefix}:duplicate")
        if review.get("new_development_status") not in DEVELOPMENT_STATUSES:
            errors.append(f"{prefix}:development")
        reason_codes = review.get("reason_codes")
        if not isinstance(reason_codes, list) or any(
                value not in SCENE_REASON_CODES for value in reason_codes):
            errors.append(f"{prefix}:reason_codes")
        scene_ref = alias_to_ref.get(alias)
        historical_support = review.get("historical_supporting_turn_refs")
        if not isinstance(historical_support, list) or any(
                str(ref) not in historical_turn_refs for ref in historical_support):
            errors.append(f"{prefix}:historical_supporting_turn_refs")
        current_support = review.get("current_scene_supporting_turn_refs")
        if not isinstance(current_support, list) or any(
                str(ref) not in scene_turn_refs.get(str(scene_ref), set())
                for ref in current_support):
            errors.append(f"{prefix}:current_scene_supporting_turn_refs")
        dependency_support = review.get("dependency_context_turn_refs")
        ancestry = dependency_ancestry(str(scene_ref))
        if not isinstance(dependency_support, list) or any(
                turn_ref_to_scene.get(str(ref)) not in ancestry
                for ref in dependency_support):
            errors.append(f"{prefix}:dependency_context_turn_refs")
        uses = review.get("uses_scene_refs")
        if not isinstance(uses, list) or any(str(ref) not in ancestry for ref in uses):
            errors.append(f"{prefix}:uses_scene_refs")
            uses = []
        used_refs = {str(ref) for ref in uses}
        if isinstance(dependency_support, list) and any(
                turn_ref_to_scene.get(str(ref)) not in used_refs
                for ref in dependency_support):
            errors.append(f"{prefix}:dependency_context_source_not_declared")
        if not isinstance(review.get("scene_requires_update_commit"), bool):
            errors.append(f"{prefix}:scene_requires_update_commit")
        scene = scenes.get(str(scene_ref)) or {}
        if scene_ref:
            review_decision[str(scene_ref)] = str(review.get("decision") or "")
        validate_realization(
            review.get("primary_action_realization"),
            prefix=f"{prefix}:primary_action_realization",
            scene_ref=str(scene_ref), role="primary", planned=scene.get("primary_action"))
        validate_realization(
            review.get("secondary_action_realization"),
            prefix=f"{prefix}:secondary_action_realization",
            scene_ref=str(scene_ref), role="secondary", planned=scene.get("secondary_action"))
        if scene_ref and review.get("decision") in {"reject", "uncertain"}:
            rejected_refs.add(scene_ref)
    required_aliases = {str(row.get("scene_alias")) for row in realized_scenes
                        if (local_gate_results.get(str(row.get("scene_ref"))) or {}).get(
                            "status") == "local_gate_passed"}
    if seen_aliases != required_aliases:
        errors.append("validator_scene_review_coverage")

    # Cross-review continuation consistency is a promotion dependency, not a
    # malformed reviewer response. Preserve the submitted statuses and expose
    # deterministic blockers for scene-decision closure.
    continuation_consistency_blockers: list[dict[str, Any]] = []
    for scene_ref, scene in scenes.items():
        for role, action in (("primary", scene.get("primary_action")),
                             ("secondary", scene.get("secondary_action"))):
            if not isinstance(action, Mapping) or action.get("action_mode") != "continuation":
                continue
            current_status = action_status.get((scene_ref, role))
            if current_status not in {"realized", "partially_realized"}:
                continue
            source_ref = str(action.get("continuation_of") or "")
            source_location = action_ref_location.get(source_ref)
            if source_location is None:
                continuation_consistency_blockers.append({
                    "scene_ref": scene_ref,
                    "action_ref": str(action.get("action_ref") or ""),
                    "source_action_ref": source_ref,
                    "reason": "source_action_missing",
                })
                continue
            source_scene_ref, source_role = source_location
            if action_status.get((source_scene_ref, source_role)) != "realized":
                continuation_consistency_blockers.append({
                    "scene_ref": scene_ref,
                    "action_ref": str(action.get("action_ref") or ""),
                    "source_action_ref": source_ref,
                    "source_scene_ref": source_scene_ref,
                    "reason": "source_action_not_realized",
                })
            if review_decision.get(source_scene_ref) != "accept":
                continuation_consistency_blockers.append({
                    "scene_ref": scene_ref,
                    "action_ref": str(action.get("action_ref") or ""),
                    "source_action_ref": source_ref,
                    "source_scene_ref": source_scene_ref,
                    "reason": "source_scene_not_accepted",
                })

    updates = output.get("durable_updates")
    if not isinstance(updates, list):
        errors.append("validator_durable_updates_not_list")
        updates = []
    update_keys: set[str] = set()
    dependencies_by_key: dict[str, set[str]] = {}
    for index, update in enumerate(updates):
        prefix = f"durable_update:{index}"
        if not isinstance(update, Mapping) or set(update) not in {
                DURABLE_UPDATE_FIELDS, CONNECTED_PERSON_DURABLE_UPDATE_FIELDS}:
            errors.append(f"{prefix}:schema")
            continue
        key = str(update.get("candidate_update_key") or "")
        if not key or key in update_keys:
            errors.append(f"{prefix}:candidate_key")
        update_keys.add(key)
        dependencies_by_key[key] = {str(value) for value in
                                    update.get("depends_on_candidate_keys") or []}
        for field in ("evidence_turn_refs", "depends_on_candidate_keys"):
            if not isinstance(update.get(field), list):
                errors.append(f"{prefix}:{field}_not_list")
        source_ref = str(update.get("source_scene_ref") or "")
        role = str(update.get("source_action_role") or "")
        evidence = {str(value) for value in update.get("evidence_turn_refs") or []}
        if source_ref not in scenes:
            errors.append(f"{prefix}:source_scene_ref")
        if source_ref in rejected_refs:
            errors.append(f"{prefix}:source_scene_rejected")
        if role not in {"primary", "secondary"}:
            errors.append(f"{prefix}:source_action_role")
        if action_status.get((source_ref, role)) != "realized":
            errors.append(f"{prefix}:source_action_not_realized")
        allowed_evidence = action_evidence.get((source_ref, role), set())
        if not evidence or any(ref not in allowed_evidence for ref in evidence):
            errors.append(f"{prefix}:evidence_turn_refs")
        if update.get("verification_status") not in VERIFICATION_STATUSES:
            errors.append(f"{prefix}:verification_status")
        if update.get("atomic_group_key") is not None and not isinstance(
                update.get("atomic_group_key"), str):
            errors.append(f"{prefix}:atomic_group_key")
        scene = scenes.get(source_ref) or {}
        planned_action = scene.get(f"{role}_action") if role in {"primary", "secondary"} else None
        action_type = str((planned_action or {}).get("resolved_action_type") or
                          (planned_action or {}).get("action_type") or "")
        connection = update.get("verified_connection")
        if action_type != "introduce_connected_person":
            if "verified_connection" in update:
                errors.append(f"{prefix}:verified_connection_for_non_person_action")
        elif (set(update) != CONNECTED_PERSON_DURABLE_UPDATE_FIELDS or
              not isinstance(connection, Mapping) or
              set(connection) != VERIFIED_CONNECTION_FIELDS):
            errors.append(f"{prefix}:verified_connection_schema")
        else:
            connection_evidence = {str(value) for value in
                                   connection.get("evidence_turn_refs") or []}
            if connection.get("verification_status") not in VERIFICATION_STATUSES:
                errors.append(f"{prefix}:verified_connection_status")
            if connection.get("connection_kind") not in VERIFIED_CONNECTION_KINDS:
                errors.append(f"{prefix}:verified_connection_kind")
            if (not connection_evidence or any(
                    ref not in allowed_evidence for ref in connection_evidence)):
                errors.append(f"{prefix}:verified_connection_evidence_turn_refs")
            if not isinstance(connection.get("reason"), str) or not str(
                    connection.get("reason") or "").strip():
                errors.append(f"{prefix}:verified_connection_reason")
            arguments = (planned_action or {}).get("resolved_arguments") or \
                (planned_action or {}).get("arguments") or {}
            legacy_connection = arguments.get("connection_to_existing_world") or {}
            target_type = str(
                arguments.get("connection_type")
                or legacy_connection.get("connection_type")
                or ""
            )
            compatible = {
                "existing_person": {"interaction_link", "explicit_relation"},
                "person": {"interaction_link", "explicit_relation"},
                "relation": {"interaction_link", "explicit_relation"},
                "interaction_link": {"interaction_link", "explicit_relation"},
                "existing_event": {"event_participant"},
                "event": {"event_participant"},
                "existing_task": {"task_participant"},
                "task": {"task_participant"},
                "information_source": {"information_source"},
                "open_thread": {"thread_participant"},
            }
            kind = str(connection.get("connection_kind") or "")
            if kind not in compatible.get(target_type, set()):
                errors.append(f"{prefix}:verified_connection_target_kind_mismatch")
            relation_type = connection.get("relation_type")
            if kind == "explicit_relation":
                if relation_type not in RELATION_TYPE_REGISTRY:
                    errors.append(f"{prefix}:verified_connection_relation_type")
            elif relation_type is not None:
                errors.append(f"{prefix}:verified_connection_relation_type_forbidden")
    for key, dependencies in dependencies_by_key.items():
        if any(value not in update_keys or value == key for value in dependencies):
            errors.append(f"durable_update:{key}:depends_on_candidate_keys")
    observations = output.get("semantic_review_observations")
    if not isinstance(observations, list):
        errors.append("validator_observations_not_list")
        observations = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping) or set(observation) != OBSERVATION_FIELDS:
            errors.append(f"observation:{index}:schema")
        elif observation.get("disposition") != "audit_only_unhinted_update":
            errors.append(f"observation:{index}:disposition")
        else:
            source_refs = {str(value) for value in observation.get("source_scene_refs") or []}
            evidence = {str(value) for value in observation.get("evidence_turn_refs") or []}
            if not source_refs or any(ref not in scenes for ref in source_refs):
                errors.append(f"observation:{index}:source_scene_refs")
            if not evidence or any(ref not in all_turn_refs for ref in evidence):
                errors.append(f"observation:{index}:evidence_turn_refs")
    classification = "reference_invalid" if any(
        any(word in error for word in ("unknown", "refs", "evidence", "coverage"))
        for error in errors) else "schema_invalid"
    return {"valid": not errors, "classification": "valid" if not errors else classification,
            "errors": list(dict.fromkeys(errors)),
            "continuation_consistency_blockers":
                continuation_consistency_blockers,
            "validator_schema_version": ROUND_VALIDATOR_SCHEMA_VERSION}


def validator_retry_disposition(validation: Mapping[str, Any]) -> str:
    if validation.get("valid"):
        return "accept_final_output"
    if validation.get("classification") in {"parse_invalid", "schema_invalid",
                                             "reference_invalid"}:
        return "contract_retry"
    return "stop_without_retry"


def classify_dialogue_only_evidence_alignment(
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Allow only non-empty, in-scope evidence-set declaration mismatches.

    The main validator has already checked that action/progression turn refs
    exist in the corresponding scene and that dependency refs are ancestors.
    Empty evidence, unknown refs, wrong-scene refs, coverage failures, and
    closure errors therefore remain hard failures.
    """
    errors = [str(value) for value in validation.get("errors") or []]
    suffixes = [value.rsplit(":", 1)[-1] for value in errors]
    degradable = bool(errors) and all(
        suffix in DIALOGUE_ONLY_DEGRADABLE_EVIDENCE_SUFFIXES
        for suffix in suffixes)
    return {
        "alignment_contract_version": (
            "dialogue_only_degraded_evidence_alignment_v1"),
        "admissible": degradable,
        "dialogue_validation_status": (
            "degraded_evidence_alignment" if degradable else "rejected"),
        "world_update_status": (
            "skipped_evidence_mismatch" if degradable else "not_evaluated"),
        "canonical_validation_errors": errors,
        "degradable_error_suffixes": sorted(
            DIALOGUE_ONLY_DEGRADABLE_EVIDENCE_SUFFIXES),
    }


def run_round_validator_with_retry(
    *, backend_call: Callable[[str, int], Any], validator_input: Mapping[str, Any],
    realized_scenes: Sequence[Mapping[str, Any]], review_world_view: Mapping[str, Any],
    local_gate_results: Mapping[str, Mapping[str, Any]],
    maximum_attempts: int = ROUND_VALIDATOR_MAX_ATTEMPTS,
    durable_update_policy: str = "strict",
    allow_terminal_contract_repair: bool = True,
) -> dict[str, Any]:
    if maximum_attempts != ROUND_VALIDATOR_MAX_ATTEMPTS:
        raise ValueError("round_validator_attempt_budget_not_frozen")
    if durable_update_policy not in DURABLE_UPDATE_POLICIES:
        raise ValueError(
            f"round_validator_durable_update_policy_invalid:{durable_update_policy}")
    frozen_sha = str(validator_input.get("frozen_round_validator_input_sha256") or "")
    base_prompt = build_round_validator_prompt(
        validator_input, durable_update_policy=durable_update_policy)
    prompt = base_prompt
    attempts = []
    final = None
    durable_update_projection_audit: dict[str, Any] | None = None
    evidence_alignment_audit: dict[str, Any] | None = None
    request_mode = "full_review"
    repair_plan: dict[str, Any] | None = None
    repair_base_output: dict[str, Any] | None = None
    absolute_attempt_limit = (
        ROUND_VALIDATOR_ABSOLUTE_MAX_ATTEMPTS
        if allow_terminal_contract_repair else maximum_attempts)
    for attempt in range(1, absolute_attempt_limit + 1):
        if str(validator_input.get("frozen_round_validator_input_sha256") or "") != frozen_sha:
            raise RuntimeError("round_validator_frozen_input_drift")
        raw = backend_call(prompt, attempt)
        normalized = normalize_round_validator_output(raw)
        dialogue_only_default_materialization = {
            "output": normalized.get("parsed"), "materialized_fields": []}
        dialogue_only_top_level_rehome = {
            "output": normalized.get("parsed"), "rehomed": False}
        if normalized["parse_valid"] and request_mode != "contract_only_repair":
            compatibility = normalize_exact_duplicate_scene_reviews(
                normalized["parsed"])
            normalized["parsed"] = compatibility["output"]
            if durable_update_policy == "dialogue_only":
                dialogue_only_top_level_rehome = (
                    rehome_dialogue_only_trailing_empty_arrays(
                        normalized["parsed"],
                        expected_scene_count=len(realized_scenes)))
                normalized["parsed"] = dialogue_only_top_level_rehome["output"]
                dialogue_only_default_materialization = (
                    materialize_dialogue_only_top_level_defaults(
                        normalized["parsed"]))
                normalized["parsed"] = (
                    dialogue_only_default_materialization["output"])
        repair_application: dict[str, Any] | None = None
        if request_mode == "contract_only_repair" and normalized["parse_valid"]:
            if repair_plan is None or repair_base_output is None:
                raise RuntimeError("round_validator_contract_repair_state_missing")
            repair_application = apply_round_validator_contract_repair(
                base_model_output=repair_base_output,
                repair_output=normalized["parsed"], repair_plan=repair_plan)
            if repair_application["valid"]:
                normalized = {
                    "parsed": repair_application["reconstructed_output"],
                    "parse_valid": True, "errors": [],
                }
            else:
                validation = {
                    "valid": False,
                    "classification": repair_application["classification"],
                    "errors": repair_application["errors"],
                }
                compiled_output = None
        if not normalized["parse_valid"]:
            validation = {"valid": False, "classification": "parse_invalid",
                          "errors": normalized["errors"]}
            compiled_output = None
        elif not (request_mode == "contract_only_repair" and
                  repair_application is not None and
                  not repair_application["valid"]):
            compilation = compile_round_validator_model_output(
                normalized["parsed"], realized_scenes=realized_scenes)
            compiled_output = compilation["compiled_output"]
            if not compilation["valid"]:
                validation = {key: deepcopy(compilation[key]) for key in (
                    "valid", "classification", "errors")}
            else:
                if durable_update_policy == "dialogue_only":
                    original_updates = deepcopy(
                        compiled_output.get("durable_updates") or [])
                    compiled_output["durable_updates"] = []
                    durable_update_projection_audit = {
                        "projection_version": (
                            "round_validator_dialogue_only_projection_v1"),
                        "policy": durable_update_policy,
                        "source_attempt": attempt,
                        "candidate_count": len(original_updates),
                        "candidate_sha256": _sha(original_updates),
                        "candidates": original_updates,
                        "semantic_status_upgrade_performed": False,
                        "authoritative_memory_write_allowed": False,
                        "reason": (
                            "Batch QA admission is based on reviewed dialogue. "
                            "Durable update candidates remain audit-only in this mode."
                        ),
                    }
                validation = validate_round_validator_output(
                    compiled_output, realized_scenes=realized_scenes,
                    review_world_view=review_world_view,
                    local_gate_results=local_gate_results)
                if (durable_update_policy == "dialogue_only" and
                        not validation["valid"]):
                    evidence_alignment_audit = (
                        classify_dialogue_only_evidence_alignment(validation))
                    if evidence_alignment_audit["admissible"]:
                        validation = {
                            "valid": True,
                            "classification": "degraded_evidence_alignment",
                            "errors": [],
                            "canonical_validation_errors": deepcopy(
                                evidence_alignment_audit[
                                    "canonical_validation_errors"]),
                            "dialogue_validation_status": (
                                "degraded_evidence_alignment"),
                            "world_update_status": "skipped_evidence_mismatch",
                        }
        disposition = validator_retry_disposition(validation)
        attempts.append({"attempt": attempt, "frozen_input_sha256": frozen_sha,
                         "request_mode": request_mode,
                         "prompt_version": (
                             ROUND_VALIDATOR_CONTRACT_REPAIR_PROMPT_VERSION_V11
                             if request_mode == "contract_only_repair"
                             else (ROUND_VALIDATOR_RETRY_PROMPT_VERSION
                                   if request_mode == "full_contract_retry"
                                   else ROUND_VALIDATOR_PROMPT_VERSION)),
                         "validation": validation, "disposition": disposition,
                         "repair_application": deepcopy(repair_application),
                         "dialogue_only_materialized_fields": deepcopy(
                             dialogue_only_default_materialization[
                                 "materialized_fields"]),
                         "dialogue_only_top_level_rehomed": bool(
                             dialogue_only_top_level_rehome["rehomed"])})
        if validation["valid"]:
            final = compiled_output
            break
        if (disposition != "contract_retry" or
                attempt == absolute_attempt_limit):
            break
        next_repair_plan = None
        if normalized["parse_valid"]:
            repair_model_output = deepcopy(normalized["parsed"])
            if _optional_model_field_materialization_eligible(
                    repair_model_output):
                repair_model_output = materialize_round_validator_optional_model_fields(
                    repair_model_output)
            if (durable_update_policy == "dialogue_only" and
                    isinstance(repair_model_output, dict)):
                repair_model_output["durable_updates"] = []
            historical_anchor_refs = {
                str(row.get("source_anchor_id") or ""): [
                    str(ref) for ref in row.get("source_refs") or []]
                for row in validator_input.get("historical_anchor_index") or []
                if isinstance(row, Mapping)
            }
            eligible_durable_action_refs = None
            if validator_input.get("round_validator_input_version") in {
                    ROUND_VALIDATOR_INPUT_VERSION,
                    "round_expansion_validator_compact_input_v7_explicit_reference_scopes",
                    "round_expansion_validator_compact_input_v6_knowledge_order_projection",
                    "round_expansion_validator_compact_input_v5_validator_checks_only",
            }:
                eligible_durable_action_refs = []
                for scene in validator_input.get("scenes") or []:
                    for role in ("primary", "secondary"):
                        action = scene.get(f"{role}_action")
                        if not isinstance(action, Mapping):
                            continue
                        if durable_candidate_action_eligible(action):
                            eligible_durable_action_refs.append(str(
                                action.get("action_ref") or ""))
            next_repair_plan = build_round_validator_contract_repair_plan(
                model_output=repair_model_output,
                realized_scenes=realized_scenes,
                compilation_errors=validation["errors"],
                historical_anchor_refs=historical_anchor_refs,
                historical_turns=validator_input.get("historical_turns") or [],
                eligible_durable_action_refs=eligible_durable_action_refs,
            )
        terminal_contract_repair = bool(
            allow_terminal_contract_repair and
            attempt == maximum_attempts and
            request_mode in {"full_review", "full_contract_retry"} and
            next_repair_plan is not None and
            next_repair_plan.get("valid"))
        if attempt >= maximum_attempts and not terminal_contract_repair:
            break
        if next_repair_plan is not None and next_repair_plan.get("valid"):
            request_mode = "contract_only_repair"
            repair_plan = next_repair_plan
            repair_base_output = repair_model_output
            prompt = build_round_validator_contract_repair_prompt(
                repair_plan=repair_plan, frozen_input_sha256=frozen_sha)
        else:
            request_mode = "full_contract_retry"
            prompt = build_round_validator_retry_prompt(
                frozen_input_sha256=frozen_sha,
                contract_errors=validation["errors"],
                frozen_base_prompt=base_prompt)
    return {
        "retry_policy_version": ROUND_VALIDATOR_RETRY_POLICY_VERSION,
        "prompt_version": ROUND_VALIDATOR_PROMPT_VERSION,
        "frozen_input_sha256": frozen_sha, "attempts": attempts,
        "physical_calls": len(attempts), "valid_output": final,
        "completed": final is not None,
        "durable_update_policy": durable_update_policy,
        "durable_update_projection_audit": durable_update_projection_audit,
        "evidence_alignment_audit": evidence_alignment_audit,
        "dialogue_validation_status": (
            "degraded_evidence_alignment"
            if evidence_alignment_audit and evidence_alignment_audit.get(
                "admissible") else "canonical" if final is not None else "rejected"),
        "world_update_status": (
            "skipped_evidence_mismatch"
            if evidence_alignment_audit and evidence_alignment_audit.get(
                "admissible") else "skipped_dialogue_only"
            if durable_update_policy == "dialogue_only" else "eligible_for_commit"),
        "contract_only_repair_used": any(
            row.get("request_mode") == "contract_only_repair" for row in attempts),
        "terminal_contract_repair_allowed": allow_terminal_contract_repair,
    }
