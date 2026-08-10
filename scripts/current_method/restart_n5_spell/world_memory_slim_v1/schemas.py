"""Strict, versioned schemas for the lightweight world-memory pipeline."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..core import canonical_json, sha256_bytes


WORLD_MEMORY_SCHEMA_VERSION = "world_memory_v1"
PLANNER_SCHEMA_VERSION = "slim_world_planner_v2"
REALIZER_SCHEMA_VERSION = "natural_realizer_v2"
EVIDENCE_COMMITTER_SCHEMA_VERSION = "evidence_committer_v1"
QA_REPLAY_SCHEMA_VERSION = "qa_replay_pool_v1"
DIFFICULTY_STRATEGY_SCHEMA_VERSION = "difficulty_strategy_memory_v1"
PLANNER_MUTATION_SCHEMA_VERSION = "planner_mutation_request_v1"

ALLOWED_OPERATIONS = frozenset({
    "add_person", "add_shared_task", "add_role_handoff",
    "add_information_asymmetry", "add_third_party_report", "change_trust",
    "change_authority", "create_conflict", "resolve_conflict",
    "update_state", "open_thread", "close_thread",
})
MEMORY_STATUSES = frozenset({"active", "superseded", "completed", "disputed"})
SEMANTIC_STATUSES = frozenset({
    "source_grounded", "model_claimed", "semantically_verified", "rejected",
})

# These names are rejected as data. They are retained here only so the new
# schemas fail closed when an old Planner/Realizer payload is supplied.
LEGACY_OR_MODEL_ASSIGNED_FIELDS = frozenset({
    "completion_evidence_phrases", "canonical_marker", "exact_marker",
    "evidence_turn_index", "final_turn_id", "final_session_id", "handoff_id",
    "state_instance_id", "completion_status", "temporality", "conditionality",
    "fixed_4x4_stage_contract",
})

WORLD_MEMORY_SCHEMA: dict[str, Any] = {
    "schema_version": WORLD_MEMORY_SCHEMA_VERSION,
    "required": [
        "world_memory_schema_version", "root_id", "memory_version", "entities",
        "relations", "event_history", "current_states", "knowledge_visibility",
        "open_threads", "completed_threads", "memory_update_log",
        "parent_child_lineage", "pending_memory_updates",
    ],
    "record_required": [
        "memory_id", "event_identity_key", "record_type", "subject_ids", "value",
        "source_turn_ids", "status", "created_round", "last_updated_round",
        "provenance", "semantic_status", "handoff_id",
    ],
    "statuses": sorted(MEMORY_STATUSES),
    "semantic_statuses": sorted(SEMANTIC_STATUSES),
}

SLIM_PLANNER_SCHEMA: dict[str, Any] = {
    "schema_version": PLANNER_SCHEMA_VERSION,
    "required": [
        "planner_schema_version", "selected_memory_threads", "scene_budget",
        "scene_goals",
    ],
    "scene_budget_required": ["target", "minimum", "maximum"],
    "scene_goal_required": ["goal", "participants", "selected_action_ids"],
    "forbidden_fields": sorted(LEGACY_OR_MODEL_ASSIGNED_FIELDS),
}

PLANNER_PROGRAM_OWNED_FIELDS = frozenset({
    "scene_key", "delta_id", "proposed_delta_id", "operation", "operation_class",
    "event_identity_key", "subject_ids", "pre_state", "current_state",
    "post_state", "intended_post_state", "memory_id", "target_memory_id",
    "predecessor_memory_id",
    "related_memory_ids", "proposed_graph_deltas", "evidence_turn_index",
    "evidence_turn_keys", "final_turn_id", "final_session_id", "handoff_id",
})

NATURAL_REALIZER_SCHEMA: dict[str, Any] = {
    "schema_version": REALIZER_SCHEMA_VERSION,
    "required": [
        "realizer_schema_version", "scene_key", "dialogue_turns",
        "claimed_graph_updates",
    ],
    "optional_with_defaults": {"new_open_thread_claims": []},
    "dialogue_turn_required": ["turn_key", "speaker", "text"],
    "claimed_update_required": [
        "proposed_delta_id", "claimed_status", "candidate_evidence_turn_keys",
    ],
    "claimed_statuses": ["realized", "not_realized", "uncertain"],
    "open_thread_claim_required": [
        "description", "candidate_evidence_turn_keys", "related_proposed_delta_id",
    ],
    "forbidden_fields": sorted(LEGACY_OR_MODEL_ASSIGNED_FIELDS),
}

REALIZER_CLAIM_ONLY_RETRY_SCHEMA: dict[str, Any] = {
    "schema_version": "realizer_claim_only_retry_v1",
    "required": ["claimed_graph_updates", "new_open_thread_claims"],
    "claimed_update_required": list(
        NATURAL_REALIZER_SCHEMA["claimed_update_required"]),
    "claimed_statuses": list(NATURAL_REALIZER_SCHEMA["claimed_statuses"]),
    "open_thread_claim_required": list(
        NATURAL_REALIZER_SCHEMA["open_thread_claim_required"]),
}

EVIDENCE_COMMITTER_SCHEMA: dict[str, Any] = {
    "schema_version": EVIDENCE_COMMITTER_SCHEMA_VERSION,
    "required": [
        "evidence_committer_schema_version", "proposed_delta_id", "decision",
        "confirmed_post_state", "evidence_turn_keys", "reason", "semantic_status",
    ],
    "decisions": ["confirmed", "rejected", "uncertain"],
}

QA_REPLAY_SCHEMA: dict[str, Any] = {
    "schema_version": QA_REPLAY_SCHEMA_VERSION,
    "required": [
        "qa_replay_schema_version", "qa_id", "parent_qa_id", "generation_mode",
        "question", "answer", "options", "evidence_turn_ids",
        "difficulty_vector", "solver_results", "difficulty_bucket",
        "mutation_operation",
    ],
    "generation_modes": ["fresh", "successor"],
    "difficulty_buckets": ["grounded_easy", "frontier", "too_hard"],
}

DIFFICULTY_STRATEGY_SCHEMA: dict[str, Any] = {
    "schema_version": DIFFICULTY_STRATEGY_SCHEMA_VERSION,
    "required": [
        "difficulty_strategy_schema_version", "operator", "attempts",
        "grounded_rate", "frontier_rate", "too_hard_rate", "bad_ground_rate",
        "recommended_weight",
    ],
}

PLANNER_MUTATION_SCHEMA: dict[str, Any] = {
    "schema_version": PLANNER_MUTATION_SCHEMA_VERSION,
    "required": [
        "planner_mutation_schema_version", "source_qa_id", "missing_dependency",
        "requested_graph_mutation", "status",
    ],
    "statuses": ["pending", "accepted", "rejected", "completed"],
}


def _exact_keys(value: Any, expected: Sequence[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == set(expected)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: Any, *, nonempty: bool = False) -> bool:
    return (isinstance(value, list) and (bool(value) or not nonempty) and
            all(_text(item) for item in value))


def _find_forbidden(value: Any, path: str = "$model") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in LEGACY_OR_MODEL_ASSIGNED_FIELDS:
                errors.append(f"forbidden_field:{path}.{key}")
            errors.extend(_find_forbidden(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_find_forbidden(child, f"{path}[{index}]"))
    return errors


def validate_world_memory(memory: Any) -> dict[str, Any]:
    errors: list[str] = []
    required = WORLD_MEMORY_SCHEMA["required"]
    if not _exact_keys(memory, required):
        return {"valid": False, "errors": ["world_memory_top_level_schema"]}
    if memory["world_memory_schema_version"] != WORLD_MEMORY_SCHEMA_VERSION:
        errors.append("world_memory_schema_version")
    if not _text(memory["root_id"]):
        errors.append("world_memory_root_id")
    if not isinstance(memory["memory_version"], int) or memory["memory_version"] < 1:
        errors.append("world_memory_version")
    list_fields = required[3:]
    for field in list_fields:
        if not isinstance(memory[field], list):
            errors.append(f"world_memory_list:{field}")
    if errors:
        return {"valid": False, "errors": errors}
    entity_fields = {
        "entity_id", "display_name", "source_turn_ids", "status",
        "created_round", "last_updated_round", "provenance", "semantic_status",
    }
    for index, entity in enumerate(memory["entities"]):
        if not _exact_keys(entity, entity_fields):
            errors.append(f"entity_schema:{index}")
            continue
        if (not _text(entity["entity_id"]) or not _text(entity["display_name"]) or
                not _string_list(entity["source_turn_ids"], nonempty=True) or
                entity["status"] not in MEMORY_STATUSES or
                entity["semantic_status"] not in SEMANTIC_STATUSES or
                not isinstance(entity["provenance"], Mapping)):
            errors.append(f"entity_value:{index}")
    record_fields = WORLD_MEMORY_SCHEMA["record_required"]
    for field in ("event_history", "current_states", "open_threads",
                  "completed_threads"):
        for index, record in enumerate(memory[field]):
            if not _exact_keys(record, record_fields):
                errors.append(f"memory_record_schema:{field}:{index}")
                continue
            if (not _text(record["memory_id"]) or
                    not _text(record["event_identity_key"]) or
                    not _text(record["record_type"]) or
                    not _string_list(record["subject_ids"]) or
                    not isinstance(record["value"], str) or
                    not _string_list(record["source_turn_ids"], nonempty=True) or
                    record["status"] not in MEMORY_STATUSES or
                    record["semantic_status"] not in SEMANTIC_STATUSES or
                    not isinstance(record["created_round"], int) or
                    not isinstance(record["last_updated_round"], int) or
                    not isinstance(record["provenance"], Mapping) or
                    not (record["handoff_id"] is None or _text(record["handoff_id"]))):
                errors.append(f"memory_record_value:{field}:{index}")
    return {"valid": not errors, "errors": errors}


def _find_named_fields(
    value: Any, forbidden: frozenset[str], path: str = "$model",
) -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in forbidden:
                errors.append(f"planner_program_field:{path}.{key}")
            errors.extend(_find_named_fields(child, forbidden, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_find_named_fields(child, forbidden, f"{path}[{index}]"))
    return errors


def validate_slim_planner_output(
    plan: Any, *, action_menu: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors = _find_forbidden(plan)
    errors.extend(_find_named_fields(plan, PLANNER_PROGRAM_OWNED_FIELDS))
    if not _exact_keys(plan, SLIM_PLANNER_SCHEMA["required"]):
        errors.append("planner_top_level_schema")
        return {"valid": False, "errors": errors}
    if plan["planner_schema_version"] != PLANNER_SCHEMA_VERSION:
        errors.append("planner_schema_version")
    if not _string_list(plan["selected_memory_threads"], nonempty=True):
        errors.append("selected_memory_threads")
    budget = plan["scene_budget"]
    if not _exact_keys(budget, SLIM_PLANNER_SCHEMA["scene_budget_required"]):
        errors.append("scene_budget_schema")
    elif (not all(isinstance(budget[key], int) and budget[key] > 0 for key in budget)
          or not budget["minimum"] <= budget["target"] <= budget["maximum"]):
        errors.append("scene_budget_values")
    goals = plan["scene_goals"]
    if not isinstance(goals, list) or not goals:
        errors.append("scene_goals")
        return {"valid": False, "errors": errors}
    if isinstance(budget, Mapping) and not (
            budget.get("minimum", 0) <= len(goals) <= budget.get("maximum", -1)):
        errors.append("scene_goal_count_outside_budget")
    action_by_id: dict[str, Mapping[str, Any]] = {}
    valid_threads: set[str] = set()
    expected_budget: Mapping[str, Any] | None = None
    if action_menu is not None:
        action_by_id = {
            str(row.get("action_id")): row for row in action_menu.get("actions") or []
            if isinstance(row, Mapping)
        }
        valid_threads = {
            str(row.get("thread_id")) for row in action_menu.get("memory_threads") or []
            if isinstance(row, Mapping)
        }
        expected_budget = action_menu.get("scene_budget")
        if (expected_budget is not None and
                (not isinstance(budget, Mapping) or
                 dict(budget) != dict(expected_budget))):
            errors.append("scene_budget_not_frozen")
        for thread_id in plan["selected_memory_threads"]:
            if thread_id not in valid_threads:
                errors.append(f"selected_thread_unknown:{thread_id}")
    selected_threads = set(plan["selected_memory_threads"])
    seen_action_ids: set[str] = set()
    seen_mutex_groups: dict[str, int] = {}
    for scene_index, scene in enumerate(goals):
        prefix = f"scene:{scene_index}"
        if not _exact_keys(scene, SLIM_PLANNER_SCHEMA["scene_goal_required"]):
            errors.append(f"{prefix}:schema")
            continue
        if (not _text(scene["goal"]) or
                not _string_list(scene["participants"], nonempty=True) or
                not _string_list(scene["selected_action_ids"], nonempty=True)):
            errors.append(f"{prefix}:values")
        for action_id in scene.get("selected_action_ids") or []:
            if action_id in seen_action_ids:
                errors.append(f"action_selected_more_than_once:{action_id}")
            seen_action_ids.add(str(action_id))
            if action_menu is None:
                continue
            action = action_by_id.get(str(action_id))
            if action is None:
                errors.append(f"selected_action_unknown:{action_id}")
                continue
            eligible = set(str(value) for value in
                           action.get("eligible_participants") or [])
            participants = set(scene.get("participants") or [])
            if not participants <= eligible or not participants:
                errors.append(f"{prefix}:participant_not_eligible:{action_id}")
            if not participants & eligible:
                errors.append(f"{prefix}:action_has_no_participant:{action_id}")
            thread_id = str(action.get("thread_id") or "")
            if thread_id not in selected_threads:
                errors.append(f"{prefix}:action_thread_not_selected:{action_id}")
            group = str(action.get("mutually_exclusive_group") or "")
            if group and group in seen_mutex_groups and seen_mutex_groups[group] != scene_index:
                errors.append(f"mutually_exclusive_actions_cross_scenes:{group}")
            seen_mutex_groups[group] = scene_index
    return {"valid": not errors, "errors": errors}


def assign_proposed_delta_ids(plan: Mapping[str, Any], root_id: str) -> dict[str, Any]:
    """Fail closed: v2 IDs require a frozen action menu and compiler expansion."""
    del plan, root_id
    raise RuntimeError(
        "planner_v1_delta_authoring_removed; use compile_selected_actions")


def _validate_realizer_dialogue_fields(
    *, scene_key: Any, turns: Any,
) -> tuple[list[str], set[str]]:
    """Validate dialogue-only fields and return exact response-local keys."""
    errors: list[str] = []
    if not _text(scene_key):
        errors.append("realizer_scene_key")
    if not isinstance(turns, list) or not turns:
        return errors + ["realizer_dialogue_turns"], set()
    keys: list[str] = []
    for index, turn in enumerate(turns):
        if not _exact_keys(turn, NATURAL_REALIZER_SCHEMA["dialogue_turn_required"]):
            errors.append(f"realizer_turn_schema:{index}")
            continue
        if not all(_text(turn[field]) for field in ("turn_key", "speaker", "text")):
            errors.append(f"realizer_turn_value:{index}")
            continue
        keys.append(str(turn["turn_key"]))
    if len(keys) != len(set(keys)):
        errors.append("realizer_turn_keys_unique")
    return errors, set(keys)


def _validate_realizer_claim_fields(
    *, claims: Any, thread_claims: Any, valid_turn_keys: set[str],
) -> list[str]:
    """Validate claims without interpreting dialogue text or repairing references."""
    errors: list[str] = []
    if not isinstance(claims, list):
        errors.append("claimed_graph_updates")
        claims = []
    for index, claim in enumerate(claims):
        if not _exact_keys(claim, NATURAL_REALIZER_SCHEMA["claimed_update_required"]):
            errors.append(f"claimed_update_schema:{index}")
            continue
        status = claim["claimed_status"]
        evidence = claim["candidate_evidence_turn_keys"]
        value_invalid = False
        if not _text(claim["proposed_delta_id"]):
            errors.append(f"realizer_claim_delta_id_invalid:{index}")
            value_invalid = True
        if status not in set(NATURAL_REALIZER_SCHEMA["claimed_statuses"]):
            errors.append(f"realizer_claim_status_invalid:{index}")
            value_invalid = True
        if not _string_list(evidence):
            errors.append(f"realizer_claim_evidence_schema:{index}")
            value_invalid = True
            evidence = []
        if status == "realized" and not evidence:
            errors.append(f"realizer_claim_realized_evidence_empty:{index}")
            value_invalid = True
        if status == "not_realized" and evidence:
            errors.append(f"realizer_claim_not_realized_evidence_nonempty:{index}")
            value_invalid = True
        missing = [str(key) for key in evidence if key not in valid_turn_keys]
        for key in missing:
            errors.append(f"realizer_claim_evidence_not_in_current_scene:{index}:{key}")
        if missing:
            value_invalid = True
        if value_invalid:
            errors.append(f"claimed_update_value:{index}")

    if not isinstance(thread_claims, list):
        errors.append("new_open_thread_claims")
        thread_claims = []
    for index, claim in enumerate(thread_claims):
        if not _exact_keys(claim, NATURAL_REALIZER_SCHEMA["open_thread_claim_required"]):
            errors.append(f"open_thread_claim_schema:{index}")
            continue
        evidence = claim["candidate_evidence_turn_keys"]
        if (not _text(claim["description"]) or
                not _string_list(evidence, nonempty=True) or
                not set(evidence) <= valid_turn_keys or
                claim["related_proposed_delta_id"] is not None):
            errors.append(f"open_thread_claim_value:{index}")
    return list(dict.fromkeys(errors))


def validate_natural_realizer_output(output: Any) -> dict[str, Any]:
    """Validate a full scene while reporting dialogue and claim gates separately."""
    general_errors = _find_forbidden(output)
    required = set(NATURAL_REALIZER_SCHEMA["required"])
    optional = set(NATURAL_REALIZER_SCHEMA["optional_with_defaults"])
    if not isinstance(output, Mapping):
        return {
            "valid": False, "errors": [*general_errors, "realizer_top_level_schema"],
            "dialogue_valid": False, "claims_valid": False,
            "dialogue_errors": ["realizer_top_level_schema"],
            "claim_errors": ["realizer_top_level_schema"],
            "retry_mode": "full_scene_retry",
        }
    top_level_valid = required <= set(output) and set(output) <= required | optional
    if not top_level_valid:
        general_errors.append("realizer_top_level_schema")
    dialogue_errors: list[str] = []
    claim_errors: list[str] = []
    if output.get("realizer_schema_version") != REALIZER_SCHEMA_VERSION:
        general_errors.append("realizer_schema_version")
    dialogue_errors, valid_keys = _validate_realizer_dialogue_fields(
        scene_key=output.get("scene_key"), turns=output.get("dialogue_turns"))
    claim_errors = _validate_realizer_claim_fields(
        claims=output.get("claimed_graph_updates"),
        thread_claims=output.get("new_open_thread_claims", []),
        valid_turn_keys=valid_keys)
    errors = list(dict.fromkeys([*general_errors, *dialogue_errors, *claim_errors]))
    dialogue_valid = not dialogue_errors
    claims_valid = not claim_errors
    return {
        "valid": not errors,
        "errors": errors,
        "dialogue_valid": dialogue_valid,
        "claims_valid": claims_valid,
        "dialogue_errors": dialogue_errors,
        "claim_errors": claim_errors,
        "retry_mode": (None if not errors else "full_scene_retry"),
    }


def validate_realizer_claim_only_output(
    output: Any, *, valid_turn_keys: set[str], expected_delta_ids: set[str],
) -> dict[str, Any]:
    """Validate the strict two-field response used to repair claims only."""
    errors = _find_forbidden(output)
    required = set(REALIZER_CLAIM_ONLY_RETRY_SCHEMA["required"])
    if not isinstance(output, Mapping) or set(output) != required:
        errors.append("realizer_claim_only_top_level_schema")
        return {"valid": False, "errors": list(dict.fromkeys(errors))}
    errors.extend(_validate_realizer_claim_fields(
        claims=output["claimed_graph_updates"],
        thread_claims=output["new_open_thread_claims"],
        valid_turn_keys=valid_turn_keys))
    claims = output["claimed_graph_updates"]
    claimed_ids: list[str] = []
    if isinstance(claims, list):
        for claim in claims:
            if isinstance(claim, Mapping):
                claimed_ids.append(str(claim.get("proposed_delta_id") or ""))
    for delta_id in claimed_ids:
        if delta_id not in expected_delta_ids:
            errors.append(f"realizer_claim_not_current_scene_delta:{delta_id}")
    for delta_id in expected_delta_ids:
        if delta_id not in claimed_ids:
            errors.append(f"realizer_claim_missing_delta:{delta_id}")
    seen: set[str] = set()
    for delta_id in claimed_ids:
        if delta_id in seen:
            errors.append(f"realizer_claim_duplicate_delta:{delta_id}")
        seen.add(delta_id)
    return {"valid": not errors, "errors": list(dict.fromkeys(errors))}


def validate_evidence_committer_output(output: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(output, Mapping):
        return {"valid": False, "errors": ["evidence_committer_schema"]}
    required = set(EVIDENCE_COMMITTER_SCHEMA["required"])
    actual = set(output)
    for field in sorted(required - actual):
        errors.append(f"evidence_committer_missing_field:{field}")
    for field in sorted(actual - required):
        errors.append(f"evidence_committer_unexpected_field:{field}")
    if required - actual:
        errors.append("evidence_committer_schema")
        return {"valid": False, "errors": errors}
    if output["evidence_committer_schema_version"] != EVIDENCE_COMMITTER_SCHEMA_VERSION:
        errors.append("evidence_committer_schema_version")
    decision = output["decision"]
    if (not isinstance(decision, str) or
            decision not in EVIDENCE_COMMITTER_SCHEMA["decisions"]):
        errors.append("evidence_committer_decision")
    if (not _text(output["proposed_delta_id"]) or
            not isinstance(output["confirmed_post_state"], str) or
            not _string_list(output["evidence_turn_keys"]) or
            not _text(output["reason"]) or
            not isinstance(output["semantic_status"], str) or
            output["semantic_status"] not in SEMANTIC_STATUSES):
        errors.append("evidence_committer_values")
    if decision == "confirmed" and (
            output["semantic_status"] != "semantically_verified" or
            not _text(output["confirmed_post_state"]) or
            not output["evidence_turn_keys"]):
        errors.append("confirmed_requires_verified_evidence")
    if decision == "rejected" and output["semantic_status"] != "rejected":
        errors.append("rejected_requires_rejected_status")
    if decision == "uncertain" and output["semantic_status"] == "semantically_verified":
        errors.append("uncertain_cannot_be_verified")
    return {"valid": not errors, "errors": errors}


def validate_qa_replay(record: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not _exact_keys(record, QA_REPLAY_SCHEMA["required"]):
        return {"valid": False, "errors": ["qa_replay_schema"]}
    if record["qa_replay_schema_version"] != QA_REPLAY_SCHEMA_VERSION:
        errors.append("qa_replay_schema_version")
    if (not _text(record["qa_id"]) or
            not (record["parent_qa_id"] is None or _text(record["parent_qa_id"])) or
            record["generation_mode"] not in QA_REPLAY_SCHEMA["generation_modes"] or
            not _text(record["question"]) or not _text(record["answer"]) or
            not isinstance(record["options"], list) or
            not _string_list(record["evidence_turn_ids"]) or
            not isinstance(record["difficulty_vector"], Mapping) or
            not isinstance(record["solver_results"], list) or
            record["difficulty_bucket"] not in QA_REPLAY_SCHEMA["difficulty_buckets"] or
            not (record["mutation_operation"] is None or
                 record["mutation_operation"] in ALLOWED_OPERATIONS)):
        errors.append("qa_replay_values")
    return {"valid": not errors, "errors": errors}


def validate_difficulty_strategy(record: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not _exact_keys(record, DIFFICULTY_STRATEGY_SCHEMA["required"]):
        return {"valid": False, "errors": ["difficulty_strategy_schema"]}
    if record["difficulty_strategy_schema_version"] != DIFFICULTY_STRATEGY_SCHEMA_VERSION:
        errors.append("difficulty_strategy_schema_version")
    rates = ("grounded_rate", "frontier_rate", "too_hard_rate", "bad_ground_rate")
    if (record["operator"] not in ALLOWED_OPERATIONS or
            not isinstance(record["attempts"], int) or record["attempts"] < 0 or
            not all(isinstance(record[key], (int, float)) and 0 <= record[key] <= 1
                    for key in rates) or
            not isinstance(record["recommended_weight"], (int, float)) or
            record["recommended_weight"] < 0):
        errors.append("difficulty_strategy_values")
    return {"valid": not errors, "errors": errors}


def validate_planner_mutation(record: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not _exact_keys(record, PLANNER_MUTATION_SCHEMA["required"]):
        return {"valid": False, "errors": ["planner_mutation_schema"]}
    if record["planner_mutation_schema_version"] != PLANNER_MUTATION_SCHEMA_VERSION:
        errors.append("planner_mutation_schema_version")
    if (not _text(record["source_qa_id"]) or not _text(record["missing_dependency"]) or
            not _text(record["requested_graph_mutation"]) or
            record["status"] not in PLANNER_MUTATION_SCHEMA["statuses"]):
        errors.append("planner_mutation_values")
    return {"valid": not errors, "errors": errors}


def schema_sha256s() -> dict[str, str]:
    from .affordance_compiler import (
        AFFORDANCE_SCHEMA_VERSION,
        COMPILED_DELTA_SCHEMA,
        COMPILED_DELTA_SCHEMA_VERSION,
        DELTA_AFFORDANCE_SCHEMA,
    )
    schemas = {
        WORLD_MEMORY_SCHEMA_VERSION: WORLD_MEMORY_SCHEMA,
        PLANNER_SCHEMA_VERSION: SLIM_PLANNER_SCHEMA,
        AFFORDANCE_SCHEMA_VERSION: DELTA_AFFORDANCE_SCHEMA,
        COMPILED_DELTA_SCHEMA_VERSION: COMPILED_DELTA_SCHEMA,
        REALIZER_SCHEMA_VERSION: NATURAL_REALIZER_SCHEMA,
        REALIZER_CLAIM_ONLY_RETRY_SCHEMA["schema_version"]:
            REALIZER_CLAIM_ONLY_RETRY_SCHEMA,
        EVIDENCE_COMMITTER_SCHEMA_VERSION: EVIDENCE_COMMITTER_SCHEMA,
        QA_REPLAY_SCHEMA_VERSION: QA_REPLAY_SCHEMA,
        DIFFICULTY_STRATEGY_SCHEMA_VERSION: DIFFICULTY_STRATEGY_SCHEMA,
        PLANNER_MUTATION_SCHEMA_VERSION: PLANNER_MUTATION_SCHEMA,
    }
    return {name: sha256_bytes(canonical_json(schema).encode())
            for name, schema in schemas.items()}
