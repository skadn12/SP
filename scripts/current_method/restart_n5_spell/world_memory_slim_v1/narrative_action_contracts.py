"""Shared high-level narrative action contracts for continuous expansion v11.

The registry is deliberately independent from ``operation_registry``.  These
contracts describe what a scene should *do*; they are not one-shot World
Memory transactions and therefore have no action IDs, bundle IDs, pre-state,
or allocation metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence


NARRATIVE_ACTION_CONTRACT_REGISTRY_VERSION = (
    "narrative_action_contract_registry_v5_compiler_derived_scene_presence"
)
ACTION_REALIZATION_STATUSES = frozenset({
    "realized", "partially_realized", "not_realized", "invalid",
})
ACTION_MODES = frozenset({"new", "continuation"})
PROGRESSION_STAGES = frozenset({
    "development", "escalation", "consolidation", "resolution",
})
CANONICAL_ACTION_SLOT_PATTERN = re.compile(
    r"^S(0[1-7])\.(primary_action|secondary_action)$")


def _string(*, nullable: bool = False, enum: Sequence[str] = ()) -> dict[str, Any]:
    value: dict[str, Any] = {"type": ["string", "null"] if nullable else "string"}
    if not nullable:
        value["minLength"] = 1
    if enum:
        value["enum"] = list(enum)
    return value


def _strings(*, minimum: int = 1) -> dict[str, Any]:
    return {"type": "array", "minItems": minimum, "uniqueItems": True,
            "items": {"type": "string", "minLength": 1}}


def _object(properties: Mapping[str, Any], *, required: Sequence[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
            "required": list(required or properties), "properties": deepcopy(dict(properties))}


def _entry(action_type: str, description: str, arguments: Mapping[str, Any],
           planner: Sequence[str], realizer: Sequence[str], validator: Sequence[str]) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "description": description,
        "planner_requirements": list(planner),
        "realizer_requirements": list(realizer),
        "validator_checks": list(validator),
        "argument_schema": _object(arguments),
    }


NARRATIVE_ACTION_CONTRACTS: dict[str, dict[str, Any]] = {
    "continue_open_thread": _entry(
        "continue_open_thread", "Make concrete progress on an existing open thread.",
        {"thread_ref": _string(), "continuation_mode": _string(), "new_progress": _string()},
        ["identify an existing open thread", "state concrete new progress"],
        ["show the thread affecting the interaction", "add progress rather than restating it"],
        ["thread reference exists", "dialogue contains the claimed new progress"]),
    "advance_shared_task": _entry(
        "advance_shared_task", "Advance an existing shared task with a step or obstacle.",
        {"task_ref": _string(), "progress_type": _string(),
         "responsible_participants": _strings(), "new_step_or_obstacle": _string()},
        ["identify the task", "assign responsible participants", "specify a new step or obstacle"],
        ["make the task progress observable", "respect responsibility and dependencies"],
        ["task reference exists", "participants and progress are supported by dialogue"]),
    "introduce_external_constraint": _entry(
        "introduce_external_constraint", "Introduce a concrete external constraint on an existing plan.",
        {"constraint": _string(), "affected_participants": _strings(),
         "affected_plan_or_task": _string()},
        ["state the constraint", "identify affected participants and plan or task"],
        ["make the constraint materially affect choices or action"],
        ["constraint is concrete", "dialogue shows its effect"]),
    "add_information_asymmetry": _entry(
        "add_information_asymmetry", "Create unequal knowledge that affects the interaction.",
        {"informed_participant": _string(), "uninformed_participant": _string(),
         "hidden_information": _string(), "information_source": _string()},
        ["identify informed and uninformed participants", "specify concrete hidden information"],
        ["preserve the knowledge boundary", "make unequal knowledge affect the interaction"],
        ["information is concrete", "uninformed participant does not use unavailable knowledge"]),
    "reveal_or_transfer_information": _entry(
        "reveal_or_transfer_information", "Reveal or transfer concrete information between participants.",
        {"sender": _string(), "receiver": _string(), "information": _string(),
         "transfer_mode": _string()},
        ["identify sender, receiver, information, and transfer mode"],
        ["show the transfer in dialogue", "show its immediate interaction effect"],
        ["sender and receiver exist", "the stated information is actually transferred"]),
    "create_misunderstanding": _entry(
        "create_misunderstanding", "Create a concrete mistaken belief with an actual contrary situation.",
        {"mistaken_participant": _string(), "mistaken_belief": _string(),
         "misunderstanding_source": _string(), "actual_situation": _string()},
        ["state the mistaken belief and actual situation", "identify its source"],
        ["let behavior reflect the mistake without leaking the actual situation"],
        ["mistake and actual situation differ", "dialogue realizes the misunderstanding"]),
    "create_or_escalate_conflict": _entry(
        "create_or_escalate_conflict", "Create or intensify a concrete conflict.",
        {"participants": _strings(minimum=2), "conflict_subject": _string(),
         "conflict_source": _string(),
         "escalation_or_creation": _string(enum=("create", "escalate"))},
        ["identify participants, subject, source, and whether this creates or escalates"],
        ["show incompatible positions or pressures in dialogue"],
        ["conflict is concrete", "dialogue supports creation or escalation"]),
    "resolve_or_deescalate_conflict": _entry(
        "resolve_or_deescalate_conflict", "Resolve or reduce an existing conflict.",
        {"conflict_ref_or_subject": _string(), "resolution_mode": _string(),
         "participants": _strings(minimum=2), "new_working_arrangement": _string()},
        ["identify the conflict and participants", "specify the new working arrangement"],
        ["show the arrangement being reached or enacted"],
        ["conflict exists or is concretely identified", "dialogue supports de-escalation"]),
    "assign_or_handoff_role": _entry(
        "assign_or_handoff_role", "Assign or transfer a task, role, or responsibility.",
        {"task_or_responsibility": _string(), "from_participant": _string(nullable=True),
         "to_participant": _string(), "handoff_reason": _string()},
        ["identify the responsibility, recipient, and reason", "use null source only for a new assignment"],
        ["show acceptance, assignment, or transfer in dialogue"],
        ["participants exist", "dialogue supports the responsibility change"]),
    "introduce_connected_person": _entry(
        "introduce_connected_person", "Introduce a person through an explicit path into the existing world.",
        {"participant_alias": _string(), "display_name": _string(),
         "entity_role": _string(),
         "connection_type": _string(enum=(
             "existing_person", "existing_event", "existing_task",
             "information_source", "open_thread")),
         "connection_target_ref": _string(),
         "connection_description": _string()},
        ["use a narrative participant alias", "name and describe the person",
         "identify one typed existing-world connection without asserting a social relation",
         "do not declare scene presence; local code derives it from actual speakers"],
        ["make the named identity and connection observable"],
        ["identity is concrete", "connection target exists",
         "speaking does not itself authorize Memory persistence",
         "an explicit relation requires separate dialogue evidence"]),
    "introduce_connected_event": _entry(
        "introduce_connected_event",
        "Introduce a concrete event connected to the authoritative world.",
        {"event_alias": _string(), "description": _string(),
         "participant_refs": _strings(),
         "connected_existing": _object({
             "ref_type": _string(enum=("person", "task", "open_thread", "event")),
             "ref": _string()}),
         "narrative_time_expression": _string(nullable=True),
         "narrative_time_relation": _string(enum=(
             "past", "present_completed", "future", "unspecified")),
         "occurrence_status": _string(enum=("occurred", "planned", "scheduled", "proposed")),
         "event_consequence": _string()},
        ["identify participants and a typed existing-world connection",
         "distinguish an occurred event from a plan, schedule, or proposal"],
        ["make the occurrence status and consequence observable in dialogue",
         "do not narrate a planned event as already completed"],
        ["participant and connected refs exist", "the event is concrete",
         "only occurred events are eligible for authoritative event-history compilation"]),
    "develop_character_or_task_state": _entry(
        "develop_character_or_task_state", "Develop a character or task dimension without asserting a transaction.",
        {"target": _string(), "dimension": _string(), "direction": _string(), "trigger": _string()},
        ["identify target, dimension, direction, and trigger"],
        ["show the development through choices, information, or consequences"],
        ["target exists", "dialogue supports the direction and trigger"]),
    "supersede_or_correct_information": _entry(
        "supersede_or_correct_information", "Correct or supersede prior information.",
        {"old_information_ref_or_description": _string(), "new_information": _string(),
         "correction_source": _string(), "affected_participants": _strings()},
        ["identify old information, replacement, source, and affected participants"],
        ["make the correction explicit and show who learns it"],
        ["old information is identified", "dialogue supports the correction"]),
    "close_or_defer_thread": _entry(
        "close_or_defer_thread", "Resolve, cancel, defer, wait on, or transfer an open thread.",
        {"thread_ref": _string(),
         "closure_mode": _string(enum=("resolved", "cancelled", "deferred", "waiting_for_external_condition", "transferred")),
         "reason": _string(), "next_condition": _string()},
        ["identify the thread, closure mode, reason, and next condition"],
        ["show the disposition in dialogue", "do not confuse deferral with completion"],
        ["thread reference exists", "dialogue supports the selected disposition"]),
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def action_contract_registry() -> dict[str, Any]:
    value = {"registry_version": NARRATIVE_ACTION_CONTRACT_REGISTRY_VERSION,
             "actions": [deepcopy(NARRATIVE_ACTION_CONTRACTS[key])
                         for key in NARRATIVE_ACTION_CONTRACTS]}
    value["registry_sha256"] = hashlib.sha256(_canonical(value).encode()).hexdigest()
    return value


def action_contract(action_type: str) -> dict[str, Any]:
    if action_type not in NARRATIVE_ACTION_CONTRACTS:
        raise KeyError(f"unknown_narrative_action_type:{action_type}")
    return deepcopy(NARRATIVE_ACTION_CONTRACTS[action_type])


def _validate_schema(value: Any, schema: Mapping[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, Mapping):
            return [f"{path}:not_object"]
        required = set(schema.get("required") or [])
        properties = schema.get("properties") or {}
        missing = sorted(required - set(value))
        extra = sorted(set(value) - set(properties))
        errors.extend(f"{path}:missing:{field}" for field in missing)
        if schema.get("additionalProperties") is False:
            errors.extend(f"{path}:unexpected:{field}" for field in extra)
        for key in set(value) & set(properties):
            errors.extend(_validate_schema(value[key], properties[key], f"{path}.{key}"))
    elif kind == "array":
        if not isinstance(value, list):
            return [f"{path}:not_array"]
        if len(value) < int(schema.get("minItems") or 0):
            errors.append(f"{path}:too_short")
        if schema.get("uniqueItems") and len({_canonical(row) for row in value}) != len(value):
            errors.append(f"{path}:duplicate_items")
        for index, row in enumerate(value):
            errors.extend(_validate_schema(row, schema.get("items") or {}, f"{path}[{index}]"))
    else:
        allowed = set(kind) if isinstance(kind, list) else {kind}
        actual = "null" if value is None else "string" if isinstance(value, str) else "other"
        if actual not in allowed:
            errors.append(f"{path}:type")
        elif actual == "string" and len(value.strip()) < int(schema.get("minLength") or 0):
            errors.append(f"{path}:empty")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}:enum")
    return errors


def validate_action_instance(
    value: Any, *, path: str = "action", known_world_refs: set[str] | None = None,
    authoritative_entity_refs: set[str] | None = None,
    action_availability_view: Mapping[str, Any] | None = None,
    round_participant_aliases: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return {"valid": False, "errors": [f"{path}:shape"]}
    mode = str(value.get("action_mode") or "")
    if mode not in ACTION_MODES:
        return {"valid": False, "errors": [f"{path}:action_mode"]}
    if mode == "continuation":
        expected = {"action_mode", "continuation_of", "progression"}
        if set(value) != expected:
            return {"valid": False, "errors": [f"{path}:continuation_shape"]}
        continuation_of = value.get("continuation_of")
        progression = value.get("progression")
        if not isinstance(continuation_of, str) or not continuation_of.strip():
            errors.append(f"{path}:continuation_of")
        elif CANONICAL_ACTION_SLOT_PATTERN.fullmatch(continuation_of) is None:
            errors.append(f"{path}:continuation_slot_format:{continuation_of}")
        if (not isinstance(progression, Mapping) or
                set(progression) != {"stage", "new_step"}):
            errors.append(f"{path}:progression_shape")
        else:
            if progression.get("stage") not in PROGRESSION_STAGES:
                errors.append(f"{path}:progression_stage")
            if not isinstance(progression.get("new_step"), str) or not str(
                    progression.get("new_step") or "").strip():
                errors.append(f"{path}:progression_new_step")
        return {"valid": not errors, "errors": errors,
                "action_mode": mode, "action_type": None,
                "contract_version": NARRATIVE_ACTION_CONTRACT_REGISTRY_VERSION}
    if set(value) != {"action_mode", "action_type", "arguments"}:
        return {"valid": False, "errors": [f"{path}:new_shape"]}
    action_type = str(value.get("action_type") or "")
    contract = NARRATIVE_ACTION_CONTRACTS.get(action_type)
    if contract is None:
        return {"valid": False, "errors": [f"{path}:unknown_action_type:{action_type}"]}
    errors.extend(_validate_schema(value.get("arguments"), contract["argument_schema"],
                                   f"{path}.arguments"))
    arguments = value.get("arguments") if isinstance(value.get("arguments"), Mapping) else {}
    if action_type == "introduce_connected_event":
        occurrence_status = arguments.get("occurrence_status")
        time_relation = arguments.get("narrative_time_relation")
        if (occurrence_status == "occurred" and
                time_relation not in {"past", "present_completed"}):
            errors.append(
                f"{path}.arguments:event_temporal_status_mismatch:"
                f"{occurrence_status}:{time_relation}")
        elif (occurrence_status in {"planned", "scheduled", "proposed"} and
              time_relation not in {"future", "unspecified"}):
            errors.append(
                f"{path}.arguments:event_temporal_status_mismatch:"
                f"{occurrence_status}:{time_relation}")
    if action_type in {"introduce_connected_person", "introduce_connected_event"}:
        alias_field = ("participant_alias" if action_type ==
                       "introduce_connected_person" else "event_alias")
        alias = str(arguments.get(alias_field) or "")
        if re.fullmatch(r"[a-z][a-z0-9_]{0,31}", alias) is None:
            errors.append(f"{path}.arguments.{alias_field}:narrative_alias_invalid:{alias}")
    entity_fields = {
        "affected_participants", "responsible_participants", "participants",
        "affected_participants",
    }
    single_entity_fields = {
        "informed_participant", "uninformed_participant", "sender", "receiver",
        "mistaken_participant", "from_participant", "to_participant",
    }
    if authoritative_entity_refs is not None:
        allowed_entities = set(authoritative_entity_refs) | set(round_participant_aliases)
        for field in entity_fields:
            for ref in arguments.get(field) or []:
                if str(ref) not in allowed_entities:
                    errors.append(f"{path}.arguments.{field}:unknown_entity:{ref}")
        for field in single_entity_fields:
            ref = arguments.get(field)
            if ref is not None and field in arguments and str(ref) not in allowed_entities:
                errors.append(f"{path}.arguments.{field}:unknown_entity:{ref}")
    if known_world_refs is not None:
        known_or_round_participant = set(known_world_refs) | set(round_participant_aliases)
        for field in ("thread_ref", "task_ref"):
            ref = arguments.get(field)
            if ref and str(ref) not in known_world_refs:
                errors.append(f"{path}.arguments.{field}:unknown_world_ref:{ref}")
        if action_type == "introduce_connected_person":
            ref = arguments.get("connection_target_ref")
            if not ref or str(ref) not in known_world_refs:
                errors.append(f"{path}.arguments.connection_target_ref:unknown_world_ref:{ref}")
        if action_type == "introduce_connected_event":
            connected = arguments.get("connected_existing") or {}
            ref = connected.get("ref") if isinstance(connected, Mapping) else None
            if not ref or str(ref) not in known_world_refs:
                errors.append(f"{path}.arguments.connected_existing.ref:unknown_world_ref:{ref}")
        if action_type == "develop_character_or_task_state":
            ref = arguments.get("target")
            if not ref or str(ref) not in known_or_round_participant:
                errors.append(f"{path}.arguments.target:unknown_world_ref:{ref}")
    if action_availability_view is not None:
        # Local import keeps the registry definition independent from the
        # availability projection while letting the instance gate enforce it.
        from .action_availability_v11 import validate_action_against_availability
        errors.extend(validate_action_against_availability(
            value, path=path, availability_view=action_availability_view,
            additional_participant_refs=round_participant_aliases))
    return {"valid": not errors, "errors": list(dict.fromkeys(errors)),
            "action_mode": mode, "action_type": action_type,
            "contract_version": NARRATIVE_ACTION_CONTRACT_REGISTRY_VERSION}


def action_instance_fingerprint(value: Mapping[str, Any]) -> str:
    if value.get("action_mode") == "continuation":
        material = {"action_mode": "continuation",
                    "continuation_of": value.get("continuation_of"),
                    "progression": value.get("progression")}
    else:
        material = {"action_type": value.get("action_type"),
                    "arguments": value.get("arguments")}
    return hashlib.sha256(_canonical(material).encode()).hexdigest()


def contract_subset_for_actions(actions: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    types = sorted({str(row.get("resolved_action_type") or row.get("action_type"))
                    for row in actions if isinstance(row, Mapping) and
                    (row.get("resolved_action_type") or row.get("action_type"))})
    return {"registry_version": NARRATIVE_ACTION_CONTRACT_REGISTRY_VERSION,
            "actions": [action_contract(value) for value in types]}


def validate_action_contract_registry() -> dict[str, Any]:
    errors: list[str] = []
    expected = {
        "action_type", "description", "planner_requirements",
        "realizer_requirements", "validator_checks", "argument_schema",
    }
    for key, row in NARRATIVE_ACTION_CONTRACTS.items():
        if set(row) != expected or row.get("action_type") != key:
            errors.append(f"contract_shape:{key}")
    return {"valid": not errors, "errors": errors,
            "action_type_count": len(NARRATIVE_ACTION_CONTRACTS),
            "registry_version": NARRATIVE_ACTION_CONTRACT_REGISTRY_VERSION}


def migrate_legacy_action_hints_to_narrative_actions(
    plan: Mapping[str, Any], *, action_menu: Mapping[str, Any],
) -> dict[str, Any]:
    """Explicit offline/historical migration; never called by live validation."""
    result = deepcopy(dict(plan))
    result["round_plan_version"] = "continuous_round_plan_v4"
    action_by_id = {str(row.get("action_id")): row
                    for row in action_menu.get("actions") or []}
    for scene_index, scene in enumerate(result.get("scenes") or [], 1):
        hints = list(scene.pop("candidate_update_hints", []) or [])
        selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        if hints:
            hint = hints[0]
            if isinstance(hint, str):
                action = action_by_id.get(hint)
                if action:
                    selected.append((action, {}))
            elif hint.get("hint_mode") == "single_update":
                action = action_by_id.get(str(hint.get("action_id")))
                if action:
                    selected.append((action, dict(hint.get("proposed_content") or {})))
            elif hint.get("hint_mode") == "atomic_bundle":
                for action_id, content in (hint.get("proposal_contents") or {}).items():
                    action = action_by_id.get(str(action_id))
                    if action:
                        selected.append((action, dict(content or {})))
        linked = scene.get("linked_existing_refs") or {}
        linked_entities = [str(value) for value in linked.get("entity_refs") or []]
        speakers = [str(row.get("participant_ref"))
                    for row in scene.get("speaker_participants") or []]
        fallback = (linked_entities or speakers or ["legacy:unknown"])[0]

        def convert(action: Mapping[str, Any], content: Mapping[str, Any]) -> dict[str, Any] | None:
            operation = str(action.get("operation") or "")
            payload = {**dict(action.get("frozen_payload") or {}), **dict(content)}
            if operation == "create_state":
                return {"action_mode": "new", "action_type": "develop_character_or_task_state", "arguments": {
                    "target": str(payload.get("subject_ref") or fallback),
                    "dimension": str(payload.get("state_dimension") or "character_development"),
                    "direction": str(payload.get("initial_value") or "develop"),
                    "trigger": str(payload.get("custom_description") or scene.get("new_development") or "scene development")}}
            if operation == "create_conflict":
                return {"action_mode": "new", "action_type": "create_or_escalate_conflict", "arguments": {
                    "participants": list(payload.get("participant_refs") or speakers[:2]),
                    "conflict_subject": str(payload.get("conflict_description") or "scene conflict"),
                    "conflict_source": str(scene.get("new_development") or "new pressure"),
                    "escalation_or_creation": "create"}}
            if operation == "create_misunderstanding":
                return {"action_mode": "new", "action_type": "create_misunderstanding", "arguments": {
                    "mistaken_participant": str((payload.get("participant_refs") or speakers or [fallback])[0]),
                    "mistaken_belief": str(payload.get("misunderstanding") or "a mistaken interpretation"),
                    "misunderstanding_source": str(scene.get("bridge_reason") or "incomplete context"),
                    "actual_situation": str(scene.get("desired_outcome") or "the actual situation differs")}}
            if operation == "add_information_asymmetry":
                return {"action_mode": "new", "action_type": "add_information_asymmetry", "arguments": {
                    "informed_participant": str((payload.get("knower_refs") or [fallback])[0]),
                    "uninformed_participant": str((payload.get("non_knower_refs") or speakers or [fallback])[-1]),
                    "hidden_information": str(payload.get("information") or scene.get("new_development")),
                    "information_source": str(scene.get("source_anchor_id") or "historical anchor")}}
            if operation == "introduce_external_constraint":
                return {"action_mode": "new", "action_type": "introduce_external_constraint", "arguments": {
                    "constraint": str(payload.get("constraint") or scene.get("new_development")),
                    "affected_participants": list(payload.get("affected_refs") or speakers),
                    "affected_plan_or_task": str(scene.get("goal") or "current plan")}}
            if operation == "add_person":
                return {"action_mode": "new", "action_type": "introduce_connected_person", "arguments": {
                    "participant_alias": f"legacy_person_{scene_index}",
                    "display_name": str(payload.get("display_role") or
                                        f"Legacy scene {scene_index} person"),
                    "entity_role": "historical shadow migration fixture",
                    "connection_type": "existing_person",
                    "connection_target_ref": fallback,
                    "connection_description": str(
                        scene.get("bridge_reason") or
                        "connected through an existing person")}}
            return None

        converted = [value for action, content in selected
                     if (value := convert(action, content)) is not None]
        if not converted:
            converted = [{"action_mode": "new", "action_type": "develop_character_or_task_state", "arguments": {
                "target": fallback, "dimension": f"scene_{scene_index}_development",
                "direction": "advance", "trigger": str(scene.get("new_development") or "new development")}}]
        scene["primary_action"] = converted[0]
        scene["secondary_action"] = converted[1] if len(converted) > 1 else None
    return result


def migrate_legacy_connected_person_contract_v2(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Explicit shadow-only migration for pre-v2 person/event actions.

    Live validation never calls this helper.  Historical fixtures did not carry
    a separate display name or typed target, so the migration uses a visibly
    synthetic display name rather than extracting identity from free text.
    """
    result = deepcopy(dict(plan))
    for scene_index, scene in enumerate(result.get("scenes") or [], 1):
        for slot in ("primary_action", "secondary_action"):
            action = scene.get(slot)
            if not isinstance(action, Mapping):
                continue
            arguments = action.get("arguments")
            if not isinstance(arguments, Mapping):
                continue
            action_type = action.get("action_type")
            if (action_type == "introduce_connected_person" and
                    "connection_to_existing_world" in arguments):
                old_connection = arguments.get("connection_to_existing_world") or {}
                old_type = str(old_connection.get("connection_type") or "person")
                type_map = {
                    "person": "existing_person", "relation": "existing_person",
                    "interaction_link": "existing_person", "task": "existing_task",
                    "event": "existing_event", "information_source": "information_source",
                    "open_thread": "open_thread",
                }
                target_ref = str(old_connection.get("existing_ref") or "")
                target_root = target_ref.split(":", 1)[0]
                action["arguments"] = {
                    "participant_alias": f"legacy_person_{scene_index}",
                    "display_name": f"Historical shadow person {scene_index}",
                    "entity_role": str(arguments.get("introduction_mode") or
                                       "historical shadow participant"),
                    "connection_type": type_map.get(old_type, "existing_person"),
                    "connection_target_ref": target_ref,
                    "connection_description": str(old_connection.get("description") or
                                                  "historical shadow connection"),
                }
            elif action_type == "introduce_connected_event" and (
                    "event_alias" not in arguments or
                    "temporary_event_key" in arguments):
                participants = [str(value) for value in
                                scene.get("speaker_participants") or [] if str(value)]
                target = participants[0] if participants else ""
                action["arguments"] = {
                    "event_alias": f"legacy_event_{scene_index}",
                    "description": str(arguments.get("new_progress") or
                                       "historical shadow connected event"),
                    "participant_refs": participants,
                    "connected_existing": {"ref_type": "person", "ref": target},
                    "narrative_time_expression": None,
                    "narrative_time_relation": "future",
                    "occurrence_status": "proposed",
                    "event_consequence": str(arguments.get("new_progress") or
                                             "historical shadow development"),
                }
            elif (action_type == "introduce_connected_person" and
                  "temporary_entity_key" in arguments):
                migrated = deepcopy(dict(arguments))
                migrated.pop("temporary_entity_key", None)
                migrated.pop("scene_presence", None)
                migrated["participant_alias"] = f"legacy_person_{scene_index}"
                action["arguments"] = migrated
            if (action_type == "introduce_connected_person" and
                    isinstance(action.get("arguments"), Mapping)):
                migrated = deepcopy(dict(action["arguments"]))
                migrated.pop("scene_presence", None)
                action["arguments"] = migrated
    return result
