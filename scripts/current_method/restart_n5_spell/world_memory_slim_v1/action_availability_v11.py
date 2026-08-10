"""Authoritative typed-reference and narrative-action availability view for v11.

This is deliberately not the legacy concrete affordance/action menu.  It does
not allocate action IDs, bundles, operations, transactions, or Memory writes.
It only tells the Planner which high-level narrative actions have the
authoritative references required by their shared contracts.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .narrative_action_contracts import NARRATIVE_ACTION_CONTRACTS


TYPED_REFERENCE_REGISTRY_VERSION = "typed_reference_registry_v11_v2_person_event_growth"
ACTION_AVAILABILITY_VIEW_VERSION = (
    "narrative_action_availability_v11_v6_authoritative_connection_refs"
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _record_ref(row: Mapping[str, Any]) -> str:
    return str(row.get("memory_ref") or row.get("memory_id") or "")


def _unique(values: Sequence[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def compile_typed_reference_registry(
    planning_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile disjoint, authoritative reference sets from Planning State."""
    participants = _unique([
        row.get("entity_ref") for row in planning_state.get("entities") or []
        if row.get("entity_ref")
    ])
    existing_people = _unique([
        row.get("entity_ref") for row in planning_state.get("entities") or []
        if row.get("entity_ref") and row.get("entity_type", "person") == "person"
    ])
    tasks = _unique([_record_ref(row) for row in planning_state.get("tasks") or []])
    threads = _unique([_record_ref(row) for row in planning_state.get("open_threads") or []])
    events = _unique([_record_ref(row) for row in planning_state.get("active_events") or []])
    state_rows = list(planning_state.get("key_states") or [])
    states = _unique([_record_ref(row) for row in state_rows
                      if row.get("record_type") != "conflict"])
    conflicts = _unique([_record_ref(row) for row in state_rows
                         if row.get("record_type") == "conflict"])
    relations = _unique([_record_ref(row)
                         for row in planning_state.get("explicit_relations") or []])
    interactions = _unique([
        row.get("interaction_link_ref")
        for row in planning_state.get("interaction_links") or []
    ])
    information = _unique([_record_ref(row)
                           for row in planning_state.get("knowledge_asymmetries") or []])
    state_targets = _unique([*participants, *tasks, *states])
    result = {
        "typed_reference_registry_version": TYPED_REFERENCE_REGISTRY_VERSION,
        "root_id": str(planning_state.get("root_id") or ""),
        "participant_refs": participants,
        "existing_person_refs": existing_people,
        "task_refs": tasks,
        "thread_refs": threads,
        "event_refs": events,
        "state_refs": states,
        "state_target_refs": state_targets,
        "conflict_refs": conflicts,
        "relation_refs": relations,
        "interaction_link_refs": interactions,
        "information_refs": information,
    }
    result["all_typed_refs"] = _unique([
        *participants, *tasks, *threads, *events, *states, *conflicts,
        *relations, *interactions, *information,
    ])
    result["typed_reference_registry_sha256"] = _sha(result)
    return result


def compile_action_availability_view(
    *, planning_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile high-level action preconditions without planning any scene."""
    refs = compile_typed_reference_registry(planning_state)
    participants = refs["participant_refs"]
    connections = {
        "existing_person": refs["existing_person_refs"],
        "existing_task": refs["task_refs"],
        "existing_event": refs["event_refs"],
        "information_source": refs["information_refs"],
        "open_thread": refs["thread_refs"],
    }

    def row(available: bool, reason: str, **allowed: Any) -> dict[str, Any]:
        return {"available": bool(available), **deepcopy(allowed), "reason": reason}

    actions = {
        "continue_open_thread": row(
            bool(refs["thread_refs"]),
            "Requires an authoritative open thread." if refs["thread_refs"] else
            "No authoritative open thread exists in the current Planning State.",
            allowed_thread_refs=refs["thread_refs"]),
        "advance_shared_task": row(
            bool(refs["task_refs"]),
            "Requires an authoritative task." if refs["task_refs"] else
            "No authoritative task exists in the current Planning State.",
            allowed_task_refs=refs["task_refs"],
            allowed_participant_refs=participants),
        "introduce_external_constraint": row(
            bool(participants), "Requires at least one affected participant.",
            allowed_participant_refs=participants),
        "add_information_asymmetry": row(
            len(participants) >= 2, "Requires two distinct authoritative participants.",
            allowed_participant_refs=participants),
        "reveal_or_transfer_information": row(
            len(participants) >= 2, "Requires authoritative sender and receiver participants.",
            allowed_participant_refs=participants),
        "create_misunderstanding": row(
            bool(participants), "Requires an authoritative mistaken participant.",
            allowed_participant_refs=participants),
        "create_or_escalate_conflict": row(
            len(participants) >= 2, "Requires at least two distinct authoritative participants.",
            allowed_participant_refs=participants),
        "resolve_or_deescalate_conflict": row(
            len(participants) >= 2 and bool(refs["conflict_refs"]),
            "Requires an authoritative conflict and at least two distinct participants."
            if refs["conflict_refs"] else
            "No authoritative conflict exists in the current Planning State.",
            allowed_conflict_refs=refs["conflict_refs"],
            allowed_participant_refs=participants),
        "assign_or_handoff_role": row(
            bool(participants), "A new assignment may use null from_participant; all named participants must exist.",
            allowed_participant_refs=participants,
            allowed_task_refs=refs["task_refs"]),
        "introduce_connected_person": row(
            any(connections.values()), "Requires one explicit authoritative existing-world connection.",
            narrative_alias_pattern="^[a-z][a-z0-9_]{0,31}$",
            allowed_connections=connections),
        "introduce_connected_event": row(
            bool(participants) and any((participants, refs["task_refs"],
                                        refs["thread_refs"], refs["event_refs"])),
            "Requires authoritative participants and one typed existing-world connection.",
            narrative_alias_pattern="^[a-z][a-z0-9_]{0,31}$",
            allowed_participant_refs=participants,
            allowed_connected_existing={
                "person": participants, "task": refs["task_refs"],
                "open_thread": refs["thread_refs"], "event": refs["event_refs"],
            }),
        "develop_character_or_task_state": row(
            bool(refs["state_target_refs"]), "Requires an authoritative participant, task, or state target.",
            allowed_target_refs=refs["state_target_refs"]),
        "supersede_or_correct_information": row(
            bool(participants), "Affected participants must be authoritative; prior information may be described.",
            allowed_participant_refs=participants,
            allowed_information_refs=refs["information_refs"]),
        "close_or_defer_thread": row(
            bool(refs["thread_refs"]),
            "Requires an authoritative open thread." if refs["thread_refs"] else
            "No authoritative open thread exists in the current Planning State.",
            allowed_thread_refs=refs["thread_refs"]),
    }
    if set(actions) != set(NARRATIVE_ACTION_CONTRACTS):
        raise RuntimeError("action_availability_registry_drift")
    result = {
        "action_availability_view_version": ACTION_AVAILABILITY_VIEW_VERSION,
        "root_id": str(planning_state.get("root_id") or ""),
        "typed_reference_registry": refs,
        "actions": actions,
        "contract_scope": "high_level_narrative_action_preconditions_only",
    }
    result["action_availability_view_sha256"] = _sha(result)
    return result


def validate_action_availability_view(value: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if value.get("action_availability_view_version") != ACTION_AVAILABILITY_VIEW_VERSION:
        errors.append("action_availability_view_version")
    actions = value.get("actions")
    if not isinstance(actions, Mapping) or set(actions) != set(NARRATIVE_ACTION_CONTRACTS):
        errors.append("action_availability_action_set")
    refs = value.get("typed_reference_registry")
    if not isinstance(refs, Mapping) or refs.get(
            "typed_reference_registry_version") != TYPED_REFERENCE_REGISTRY_VERSION:
        errors.append("typed_reference_registry_version")
    if "round_local_creation_slots" in value:
        errors.append("planner_creation_slots_forbidden")
    for action_type, row in (actions or {}).items():
        if not isinstance(row, Mapping) or not isinstance(row.get("available"), bool):
            errors.append(f"action_availability_shape:{action_type}")
    return {"valid": not errors, "errors": errors,
            "action_availability_view_version": ACTION_AVAILABILITY_VIEW_VERSION,
            "typed_reference_registry_version": TYPED_REFERENCE_REGISTRY_VERSION}


def validate_action_against_availability(
    action: Mapping[str, Any], *, path: str,
    availability_view: Mapping[str, Any],
    additional_participant_refs: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    """Apply action-type availability and exact typed-reference membership."""
    if action.get("action_mode") != "new":
        return []
    action_type = str(action.get("action_type") or "")
    action_row = (availability_view.get("actions") or {}).get(action_type)
    if not isinstance(action_row, Mapping):
        return []  # unknown action type is owned by the shared registry validator
    errors: list[str] = []
    if action_row.get("available") is not True:
        errors.append(f"{path}:action_unavailable:{action_type}")
    arguments = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
    refs = availability_view.get("typed_reference_registry") or {}

    def allowed_refs(registry_field: str) -> set[str]:
        allowed = set(str(row) for row in refs.get(registry_field) or [])
        if registry_field in {"participant_refs", "state_target_refs"}:
            allowed.update(str(value) for value in additional_participant_refs)
        return allowed

    def authoritative_refs(registry_field: str) -> set[str]:
        """Return the frozen world registry without round-local aliases.

        Round-local people may participate in a scene or event, but the
        explicit existing-world edge of a connected-person/event action must
        terminate at a record that was authoritative before this round.
        """
        return set(str(row) for row in refs.get(registry_field) or [])

    def require(field: str, registry_field: str, *, nullable: bool = False) -> None:
        if field not in arguments:
            return
        value = arguments.get(field)
        if nullable and value is None:
            return
        allowed = allowed_refs(registry_field)
        if str(value) not in allowed:
            errors.append(f"{path}.arguments.{field}:unknown_typed_ref:{registry_field}:{value}")

    def require_many(field: str, registry_field: str) -> None:
        values = arguments.get(field)
        if not isinstance(values, list):
            return
        allowed = allowed_refs(registry_field)
        for value in values:
            if str(value) not in allowed:
                errors.append(f"{path}.arguments.{field}:unknown_typed_ref:{registry_field}:{value}")

    for field in ("affected_participants", "responsible_participants", "participants"):
        require_many(field, "participant_refs")
    for field in ("informed_participant", "uninformed_participant", "sender", "receiver",
                  "mistaken_participant", "to_participant"):
        require(field, "participant_refs")
    require("from_participant", "participant_refs", nullable=True)
    if action_type == "continue_open_thread" or action_type == "close_or_defer_thread":
        require("thread_ref", "thread_refs")
    if action_type == "advance_shared_task":
        require("task_ref", "task_refs")
    if action_type == "develop_character_or_task_state":
        require("target", "state_target_refs")
    if action_type == "resolve_or_deescalate_conflict":
        require("conflict_ref_or_subject", "conflict_refs")
    if action_type == "introduce_connected_person":
        alias = str(arguments.get("participant_alias") or "")
        if re.fullmatch(r"[a-z][a-z0-9_]{0,31}", alias) is None:
            errors.append(f"{path}.arguments.participant_alias:narrative_alias_invalid:{alias}")
        registry_by_type = {
            "existing_person": "existing_person_refs", "existing_task": "task_refs",
            "existing_event": "event_refs", "information_source": "information_refs",
            "open_thread": "thread_refs",
        }
        registry_field = registry_by_type.get(str(arguments.get("connection_type") or ""))
        value = str(arguments.get("connection_target_ref") or "")
        allowed = set(str(row) for row in refs.get(registry_field or "") or [])
        if not registry_field or value not in allowed:
            errors.append(
                f"{path}.arguments.connection_target_ref:"
                f"unknown_typed_ref:{registry_field or 'connection_type'}:{value}")
    if action_type == "introduce_connected_event":
        alias = str(arguments.get("event_alias") or "")
        if re.fullmatch(r"[a-z][a-z0-9_]{0,31}", alias) is None:
            errors.append(f"{path}.arguments.event_alias:narrative_alias_invalid:{alias}")
        require_many("participant_refs", "participant_refs")
        connected = arguments.get("connected_existing")
        if isinstance(connected, Mapping):
            registry_by_type = {
                "person": "participant_refs", "task": "task_refs",
                "open_thread": "thread_refs", "event": "event_refs",
            }
            registry_field = registry_by_type.get(str(connected.get("ref_type") or ""))
            value = str(connected.get("ref") or "")
            allowed = authoritative_refs(registry_field or "")
            if not registry_field or value not in allowed:
                errors.append(
                    f"{path}.arguments.connected_existing.ref:"
                    f"unknown_typed_ref:{registry_field or 'ref_type'}:{value}")
    return list(dict.fromkeys(errors))


def compact_action_availability_for_planner(
    availability_view: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose only selectable actions while retaining exact safe references."""
    actions = {
        str(action_type): deepcopy(dict(row))
        for action_type, row in (availability_view.get("actions") or {}).items()
        if isinstance(row, Mapping) and row.get("available") is True
    }
    return {
        "action_availability_view_version": availability_view.get(
            "action_availability_view_version"),
        "root_id": availability_view.get("root_id"),
        "typed_reference_registry": deepcopy(
            availability_view.get("typed_reference_registry") or {}),
        "available_actions": actions,
        "unavailable_actions_omitted": True,
        "contract_scope": availability_view.get("contract_scope"),
    }
