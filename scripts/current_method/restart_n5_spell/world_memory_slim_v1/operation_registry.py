"""Central operation contracts for controlled world expansion v1."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


OPERATION_REGISTRY_VERSION = "operation_registry_v2_verified_entity_connections"


def _entry(
    operation: str,
    operation_class: str,
    target_types: tuple[str, ...],
    payload_schema: Mapping[str, Any],
    *,
    requires_pre_state: bool = False,
    temporary: bool = False,
    repeatability: str = "non_repeatable",
    dependencies: tuple[str, ...] = (),
    atomic: tuple[str, ...] = (),
    scene_presence: tuple[str, ...] = ("mentioned_only", "reported_only"),
    bridge: bool = False,
    weight: int = 1,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "operation_class": operation_class,
        "allowed_target_types": list(target_types),
        "payload_schema": dict(payload_schema),
        "requires_pre_state": requires_pre_state,
        "allows_temporary_refs": temporary,
        "repeatability": repeatability,
        "dependency_rules": list(dependencies),
        "atomic_group_rules": list(atomic),
        "allowed_scene_presence": list(scene_presence),
        "allows_connectivity_bridge": bridge,
        "complexity_weight": weight,
    }


_CREATE = "create"
_UPDATE = "update"
_TRANSITION = "transition"

OPERATION_REGISTRY: dict[str, dict[str, Any]] = {
    "add_person": _entry(
        "add_person", _CREATE, ("entity",),
        {"display_role": "string",
         "scene_presence": ["speaking", "mentioned_only", "reported_only"]},
        temporary=True, repeatability="non_repeatable_per_identity_payload",
        atomic=("identity_bearing_add_relation",),
        scene_presence=("speaking", "mentioned_only", "reported_only"), weight=2),
    "add_relation": _entry(
        "add_relation", _CREATE, ("relation",),
        {"source_ref": "entity_ref", "target_ref": "entity_ref", "relation_type": "registered_relation_type"},
        temporary=True, repeatability="non_repeatable_per_entity_pair_relation_type",
        dependencies=("entity_refs_exist_or_same_scene_add_person",),
        atomic=("temporary_entity_identity_relation",), bridge=True, weight=2),
    "add_interaction_link": _entry(
        "add_interaction_link", _CREATE, ("interaction_link",),
        {"source_ref": "entity_ref", "target_ref": "entity_ref",
         "link_type": "string", "description": "string"},
        temporary=True, repeatability="non_repeatable_per_entity_pair_link_type",
        dependencies=("entity_refs_exist_or_same_scene_add_person",), bridge=True),
    "add_event_participant": _entry(
        "add_event_participant", _CREATE, ("event_participation",),
        {"event_ref": "memory_ref", "participant_ref": "entity_ref",
         "participation_description": "string"},
        temporary=True, repeatability="non_repeatable_per_event_participant",
        dependencies=("event_exists", "entity_exists_or_same_scene_add_person"), bridge=True),
    "add_task_participant": _entry(
        "add_task_participant", _CREATE, ("task_participation",),
        {"task_ref": "memory_ref", "participant_ref": "entity_ref",
         "participation_description": "string"},
        temporary=True, repeatability="non_repeatable_per_task_participant",
        dependencies=("task_exists", "entity_exists_or_same_scene_add_person"), bridge=True),
    "add_information_source": _entry(
        "add_information_source", _CREATE, ("information_source",),
        {"information_ref": "memory_ref", "source_ref": "entity_ref",
         "source_description": "string"},
        temporary=True, repeatability="non_repeatable_per_information_source",
        dependencies=("information_exists", "entity_exists_or_same_scene_add_person"), bridge=True),
    "add_thread_participant": _entry(
        "add_thread_participant", _CREATE, ("thread_participation",),
        {"thread_ref": "memory_ref", "participant_ref": "entity_ref",
         "participation_description": "string"},
        temporary=True, repeatability="non_repeatable_per_thread_participant",
        dependencies=("thread_exists", "entity_exists_or_same_scene_add_person"), bridge=True),
    "add_event": _entry(
        "add_event", _CREATE, ("event",),
        {"participant_refs": "entity_ref_list", "description": "string", "narrative_time_expression": "optional_string"},
        repeatability="repeatable", bridge=True),
    "create_state": _entry(
        "create_state", _CREATE, ("state",),
        {"subject_ref": "entity_ref", "state_category": "state_category", "state_dimension": "string", "initial_value": "string", "custom_description": "optional_string"},
        repeatability="non_repeatable_per_subject_category_dimension"),
    "add_shared_task": _entry(
        "add_shared_task", _CREATE, ("task",),
        {"participant_refs": "entity_ref_list", "task_description": "string"},
        repeatability="conditionally_repeatable", bridge=True),
    "open_thread": _entry(
        "open_thread", _CREATE, ("thread",),
        {"subject_refs": "entity_ref_list", "description": "string"},
        repeatability="non_repeatable_per_subject"),
    "add_third_party_report": _entry(
        "add_third_party_report", _CREATE, ("knowledge",),
        {"reporter_ref": "entity_ref", "about_refs": "entity_ref_list", "reported_content": "string"}),
    "add_information_asymmetry": _entry(
        "add_information_asymmetry", _CREATE, ("knowledge",),
        {"knower_refs": "entity_ref_list", "non_knower_refs": "entity_ref_list", "information": "string"}),
    "create_conflict": _entry(
        "create_conflict", _CREATE, ("conflict",),
        {"participant_refs": "entity_ref_list", "conflict_description": "string"}, bridge=True, weight=2),
    "create_misunderstanding": _entry(
        "create_misunderstanding", _CREATE, ("conflict",),
        {"participant_refs": "entity_ref_list", "misunderstanding": "string"}),
    "introduce_external_constraint": _entry(
        "introduce_external_constraint", _CREATE, ("constraint",),
        {"affected_refs": "entity_ref_list", "constraint": "string"}),
    "add_role_handoff": _entry(
        "add_role_handoff", _TRANSITION, ("task", "responsibility", "role"),
        {"target_memory_ref": "memory_ref", "source_ref": "entity_ref", "target_ref": "entity_ref", "handoff_description": "string"},
        requires_pre_state=True, repeatability="once_per_active_responsibility", weight=2),
    "update_state": _entry(
        "update_state", _UPDATE, ("state",),
        {"target_memory_ref": "memory_ref", "new_value": "string"}, requires_pre_state=True),
    "change_trust": _entry(
        "change_trust", _UPDATE, ("relationship_state",),
        {"target_memory_ref": "memory_ref", "new_value": "string"}, requires_pre_state=True),
    "change_authority": _entry(
        "change_authority", _UPDATE, ("relationship_state", "responsibility"),
        {"target_memory_ref": "memory_ref", "new_value": "string"}, requires_pre_state=True),
    "resolve_conflict": _entry(
        "resolve_conflict", _UPDATE, ("conflict",),
        {"target_memory_ref": "memory_ref", "resolution": "string"}, requires_pre_state=True, weight=2),
    "close_thread": _entry(
        "close_thread", _UPDATE, ("thread",),
        {"target_memory_ref": "memory_ref", "closure": "string"}, requires_pre_state=True),
    "supersede": _entry(
        "supersede", _UPDATE, ("state", "relation", "event", "thread", "knowledge", "task", "responsibility"),
        {"target_type": "registered_target_type", "target_memory_ref": "memory_ref", "authoritative_version": "integer", "replacement_payload": "object", "supersede_reason": "string"},
        requires_pre_state=True, weight=2),
}


def operation_contract(operation: str) -> dict[str, Any]:
    """Return a defensive copy; callers cannot mutate the registry."""
    if operation not in OPERATION_REGISTRY:
        raise KeyError(f"unknown_operation:{operation}")
    return deepcopy(OPERATION_REGISTRY[operation])


def validate_operation_registry() -> dict[str, Any]:
    required = {
        "operation", "operation_class", "allowed_target_types", "payload_schema",
        "requires_pre_state", "allows_temporary_refs", "repeatability",
        "dependency_rules", "atomic_group_rules", "allowed_scene_presence",
        "allows_connectivity_bridge", "complexity_weight",
    }
    errors: list[str] = []
    for name, row in OPERATION_REGISTRY.items():
        if set(row) != required or row.get("operation") != name:
            errors.append(f"registry_contract:{name}")
        if row.get("operation_class") not in {_CREATE, _UPDATE, _TRANSITION}:
            errors.append(f"registry_class:{name}")
    return {"valid": not errors, "errors": errors, "operation_count": len(OPERATION_REGISTRY)}
