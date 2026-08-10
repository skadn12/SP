"""Deterministic affordance/delta checks with no dialogue keyword semantics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from ..core import canonical_json, sha256_bytes
from .affordance_compiler import (
    AUTHORITATIVE_SEMANTIC_STATUSES,
    CREATE_OPERATIONS,
    OPERATION_CATEGORIES,
    TRANSITION_OPERATIONS,
    UPDATE_OPERATIONS,
)


STRUCTURAL_DELTA_VALIDATOR_VERSION = "structural_delta_validator_v2"

COMPILED_DELTA_FIELDS = frozenset({
    "delta_id", "proposed_delta_id", "scene_key", "action_id", "root_id",
    "operation", "operation_class", "target_memory_id",
    "event_identity_key", "subject_ids", "pre_state", "intended_post_state",
    "predecessor_memory_id", "related_memory_ids", "transition_participants",
    "rationale", "provenance",
})


def _memory_records(memory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [record for field in (
        "event_history", "current_states", "open_threads", "completed_threads")
        for record in (memory.get(field) or []) if isinstance(record, Mapping)]


def _authoritative(record: Mapping[str, Any]) -> bool:
    return (
        record.get("semantic_status") in AUTHORITATIVE_SEMANTIC_STATUSES
        and record.get("status") not in {"superseded", "disputed"}
    )


def validate_planner_memory_references(
    memory: Mapping[str, Any], plan: Mapping[str, Any], *,
    action_menu: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check compiler-expanded references; Planner thread IDs are not memory IDs."""
    memory_ids = {str(row.get("memory_id")) for row in _memory_records(memory)
                  if row.get("memory_id")}
    references: list[str] = []
    for scene in plan.get("scene_goals") or []:
        references.extend(str(value) for value in
                          scene.get("related_memory_ids") or [])
        for delta in scene.get("proposed_graph_deltas") or []:
            references.extend(str(value) for value in
                              delta.get("related_memory_ids") or [])
    missing = sorted({value for value in references if value not in memory_ids})
    errors = [f"memory_reference_missing:{value}" for value in missing]
    if action_menu is not None:
        valid_threads = {str(row.get("thread_id"))
                         for row in action_menu.get("memory_threads") or []}
        for value in plan.get("selected_memory_threads") or []:
            if str(value) not in valid_threads:
                errors.append(f"planner_thread_reference_missing:{value}")
    return {
        "valid": not errors,
        "errors": errors,
        "referenced_memory_ids": list(dict.fromkeys(references)),
    }


def _action_delta_matches(
    delta: Mapping[str, Any], action: Mapping[str, Any], *, menu_sha: str,
) -> bool:
    category = str(action.get("operation_class") or "")
    expected_pre_state = (str(action.get("current_state") or "")
                          if category in {"update", "transition"} else None)
    related = list(action.get("related_memory_ids") or [])
    expected = {
        "root_id": action.get("root_id"),
        "operation": action.get("operation"),
        "operation_class": category,
        "target_memory_id": action.get("target_memory_id"),
        "event_identity_key": action.get("event_identity_key"),
        "subject_ids": list(action.get("subject_ids") or []),
        "pre_state": expected_pre_state,
        "intended_post_state": action.get("default_post_state"),
        "predecessor_memory_id": related[0] if related else None,
        "related_memory_ids": related,
        "transition_participants": action.get("transition_participants"),
        "rationale": action.get("reason"),
    }
    if any(delta.get(key) != value for key, value in expected.items()):
        return False
    provenance = delta.get("provenance")
    return (
        isinstance(provenance, Mapping)
        and provenance.get("generated_by") == "delta_affordance_compiler_v1"
        and provenance.get("affordance_schema_version") == "delta_affordance_v1"
        and provenance.get("action_menu_sha256") == menu_sha
    )


def validate_structural_deltas(
    memory: Mapping[str, Any], deltas: Sequence[Mapping[str, Any]], *,
    expected_root_id: str, action_menu: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate identities, references and states against the frozen action menu."""
    errors: list[str] = []
    accepted: list[str] = []
    rejected: list[str] = []
    root_id = str(memory.get("root_id") or "")
    root_mismatch = root_id != expected_root_id or action_menu.get("root_id") != root_id
    if root_mismatch:
        errors.append("root_identity_mismatch")

    entity_ids = {str(row.get("entity_id")) for row in memory.get("entities") or []
                  if isinstance(row, Mapping) and _authoritative(row)}
    all_records = _memory_records(memory)
    authoritative_records = [row for row in all_records if _authoritative(row)]
    authoritative_by_id = {str(row.get("memory_id")): row
                           for row in authoritative_records if row.get("memory_id")}
    all_event_keys = {str(row.get("event_identity_key")) for row in all_records
                      if row.get("event_identity_key")}
    all_event_keys.update(
        str(row.get("event_identity_key"))
        for row in memory.get("pending_memory_updates") or []
        if isinstance(row, Mapping) and row.get("event_identity_key"))
    current_by_id = {str(row.get("memory_id")): row
                     for row in memory.get("current_states") or []
                     if isinstance(row, Mapping) and _authoritative(row)}
    open_by_id = {str(row.get("memory_id")): row
                  for row in memory.get("open_threads") or []
                  if isinstance(row, Mapping) and _authoritative(row)}
    completed_events = {
        str(row.get("event_identity_key"))
        for row in authoritative_records if row.get("status") == "completed"
    }
    action_by_id = {str(row.get("action_id")): row
                    for row in action_menu.get("actions") or []
                    if isinstance(row, Mapping)}
    menu_sha = sha256_bytes(canonical_json(action_menu).encode())
    event_counts = Counter(str(delta.get("event_identity_key") or "")
                           for delta in deltas if isinstance(delta, Mapping))
    action_counts = Counter(str(delta.get("action_id") or "")
                            for delta in deltas if isinstance(delta, Mapping))
    mutex_counts = Counter(
        str(action_by_id.get(str(delta.get("action_id") or ""), {}).get(
            "mutually_exclusive_group") or "")
        for delta in deltas if isinstance(delta, Mapping))

    seen_delta_ids: set[str] = set()
    seen_proposed_ids: set[str] = set()
    for index, delta in enumerate(deltas):
        if not isinstance(delta, Mapping):
            delta_id = f"delta_at_{index}"
            rejected.append(delta_id)
            errors.append(f"{delta_id}:delta_schema")
            continue
        delta_id = str(delta.get("proposed_delta_id") or f"delta_at_{index}")
        local: list[str] = []
        short_id = str(delta.get("delta_id") or "")
        if set(delta) != COMPILED_DELTA_FIELDS:
            local.append("delta_schema")
        if short_id in seen_delta_ids or delta_id in seen_proposed_ids:
            local.append("duplicate_delta_id")
        seen_delta_ids.add(short_id)
        seen_proposed_ids.add(delta_id)
        if not short_id or delta_id != f"{root_id}:{short_id}":
            local.append("program_delta_identity_invalid")

        action_id = str(delta.get("action_id") or "")
        action = action_by_id.get(action_id)
        if action is None:
            local.append("action_id_missing")
        elif not _action_delta_matches(delta, action, menu_sha=menu_sha):
            local.append("action_delta_content_mismatch")
        if action_counts[action_id] > 1:
            local.append("action_selected_more_than_once")
        group = str((action or {}).get("mutually_exclusive_group") or "")
        if group and mutex_counts[group] > 1:
            local.append("mutually_exclusive_action_conflict")

        operation = str(delta.get("operation") or "")
        category = str(delta.get("operation_class") or "")
        event_key = str(delta.get("event_identity_key") or "")
        subject_ids = delta.get("subject_ids")
        related_ids = delta.get("related_memory_ids")
        predecessor = delta.get("predecessor_memory_id")
        target_id = delta.get("target_memory_id")
        if OPERATION_CATEGORIES.get(operation) != category:
            local.append("operation_class_mismatch")
        if not event_key or not event_key.startswith(root_id + ":"):
            local.append("mixed_root")
        if delta.get("root_id") != root_id or root_mismatch:
            local.append("mixed_root")
        if event_counts[event_key] > 1:
            local.append("same_round_event_conflict")
        if not isinstance(subject_ids, list) or not subject_ids or not all(
                isinstance(value, str) and value for value in subject_ids):
            local.append("subject_ids_invalid")
            subject_ids = []
        if not isinstance(related_ids, list) or not related_ids:
            local.append("related_memory_ids_missing")
            related_ids = []
        missing_related = [value for value in related_ids
                           if value not in authoritative_by_id]
        if missing_related:
            local.append("related_memory_reference_missing")
        if predecessor is not None and predecessor not in authoritative_by_id:
            local.append("predecessor_memory_id_missing")

        existing_subjects = [value for value in subject_ids if value in entity_ids]
        missing_subjects = [value for value in subject_ids if value not in entity_ids]

        if operation in CREATE_OPERATIONS:
            # Create operations deliberately do not compare a pre_state.
            if target_id is not None:
                local.append("create_target_must_be_new")
            if event_key in all_event_keys:
                local.append("create_identity_already_exists")
            if operation == "add_person":
                if not missing_subjects:
                    local.append("add_person_requires_new_entity")
                if not existing_subjects and not related_ids:
                    local.append("new_person_isolated")
            elif missing_subjects:
                local.append("subject_reference_missing")
            if not existing_subjects and operation != "add_person":
                local.append("create_not_connected")
            if event_key in completed_events:
                local.append("duplicate_base_event_completion")

        elif operation in UPDATE_OPERATIONS:
            if missing_subjects:
                local.append("subject_reference_missing")
            if operation == "close_thread":
                target = open_by_id.get(str(target_id))
            else:
                target = current_by_id.get(str(target_id))
            if target is None:
                local.append("update_target_missing")
            else:
                if delta.get("pre_state") != target.get("value"):
                    local.append("pre_state_mismatch")
                if event_key != target.get("event_identity_key"):
                    local.append("update_event_identity_mismatch")
            if (action is not None and delta.get("intended_post_state") not in
                    action.get("allowed_post_state_directions", [])):
                local.append("post_state_direction_not_allowed")
            if event_key in completed_events and operation in {
                    "resolve_conflict", "close_thread"}:
                local.append("duplicate_base_event_completion")

        elif operation in TRANSITION_OPERATIONS:
            if missing_subjects:
                local.append("subject_reference_missing")
            target = current_by_id.get(str(target_id))
            if target is None:
                local.append("transition_target_missing")
            elif delta.get("pre_state") != target.get("value"):
                local.append("pre_state_mismatch")
            transition = delta.get("transition_participants")
            if (not isinstance(transition, Mapping)
                    or set(transition) != {"source_entity_id", "target_entity_id"}
                    or any(value not in entity_ids for value in transition.values())
                    or transition["source_entity_id"] == transition["target_entity_id"]):
                local.append("transition_participants_invalid")
        else:
            local.append("operation_not_allowed")

        local = list(dict.fromkeys(local))
        if local:
            rejected.append(delta_id)
            errors.extend(f"{delta_id}:{error}" for error in local)
        else:
            accepted.append(delta_id)
    return {
        "validator_version": STRUCTURAL_DELTA_VALIDATOR_VERSION,
        "structurally_valid": not errors,
        "errors": errors,
        "accepted_delta_ids": accepted,
        "rejected_delta_ids": rejected,
        "operation_class_counts": dict(Counter(
            str(delta.get("operation_class") or "") for delta in deltas
            if isinstance(delta, Mapping))),
    }
