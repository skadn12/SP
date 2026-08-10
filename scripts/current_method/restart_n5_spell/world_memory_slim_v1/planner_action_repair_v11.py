"""Planner action-only repair contract for continuous-expansion v11.

The model must explicitly resubmit only action slots that failed the local
contract.  Local code freezes and reconstructs the accepted non-action outline;
it never guesses, deletes, or semantically rewrites an action.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .narrative_action_contracts import validate_action_instance


PLANNER_ACTION_REPAIR_VERSION_V11 = (
    "planner_action_only_repair_v11_v12_connected_reference_templates"
)
PLANNER_ACTION_REPAIR_PROMPT_VERSION_V11 = (
    "anchor_first_slim_outline_action_only_repair_v11_prompt_v13_connected_reference_templates"
)
_ACTION_REF = re.compile(r"^S(0[1-7])\.(primary_action|secondary_action)$")
_ACTION_ERROR = re.compile(
    r"^action:(S0[1-7]\.(?:primary_action|secondary_action))"
    r"(?:\.arguments(?:\.([^:]+))?)?:(.+)$"
)
_PARTICIPANT_ACTION_ERROR = re.compile(
    r"^scene:([1-7]):action:(primary_action|secondary_action):"
    r"participant_alias_action_redeclared:(.+)$"
)

COMPLETE_ACTION_REPLACEMENT_INSTRUCTIONS_V11 = (
    "COMPLETE REPLACEMENT ENVELOPE\n"
    "Every replacement must start with action_mode. For a new action, the only keys "
    "directly inside replacement are action_mode, action_type, and arguments:\n"
    "{\n"
    "  \"action_mode\": \"new\",\n"
    "  \"action_type\": \"<EXACT_AVAILABLE_ACTION_TYPE>\",\n"
    "  \"arguments\": {\"<REQUIRED_ARGUMENT_NAME>\": \"<VALID_VALUE>\"}\n"
    "}\n"
    "For ordinary actions, all contract argument fields, including target, dimension, "
    "direction, trigger, sender, "
    "receiver, participants, connection fields, and information fields, belong inside the "
    "arguments object. Never place them beside action_type. For a continuation, the only "
    "keys directly inside replacement are action_mode, continuation_of, and progression:\n"
    "{\n"
    "  \"action_mode\": \"continuation\",\n"
    "  \"continuation_of\": \"<EXACT_ALLOWED_ACTION_SLOT>\",\n"
    "  \"progression\": {\"stage\": \"development\", "
    "\"new_step\": \"<CONCRETE_NEW_STEP>\"}\n"
    "}\n"
    "Angle-bracket values are shape metavariables and must be replaced. A round participant "
    "alias may be used only in a field whose frozen action availability explicitly permits "
    "round participants. Display labels and shortened names such as A, B, or Friend C are "
    "not participant refs. Connected-person and connected-event repairs are the exception: "
    "use the dedicated connection_choice_id shapes supplied in the prompt; local code adds "
    "the authoritative connection and narrative alias."
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _non_action_view(outline: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(outline))
    for scene in result.get("scenes") or []:
        if isinstance(scene, dict):
            scene.pop("primary_action", None)
            scene.pop("secondary_action", None)
    return result


def _action_slots(outline: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, scene in enumerate(outline.get("scenes") or [], 1):
        if not isinstance(scene, Mapping):
            continue
        for slot in ("primary_action", "secondary_action"):
            action = scene.get(slot)
            if action is not None:
                result[f"S{index:02d}.{slot}"] = deepcopy(action)
    return result


def _introduced_narrative_alias(action: Any) -> tuple[str, str] | None:
    """Return the identity alias owned by a new introduction action."""
    if not isinstance(action, Mapping) or action.get("action_mode") != "new":
        return None
    arguments = action.get("arguments")
    if not isinstance(arguments, Mapping):
        return None
    action_type = str(action.get("action_type") or "")
    if action_type == "introduce_connected_person":
        alias = arguments.get("participant_alias")
        return ("person", str(alias)) if isinstance(alias, str) and alias else None
    if action_type == "introduce_connected_event":
        alias = arguments.get("event_alias")
        return ("event", str(alias)) if isinstance(alias, str) and alias else None
    return None


def _repair_slot_alias(action_ref: str, kind: str) -> str:
    """Allocate a stable non-authoritative alias for one repair slot."""
    scene, slot = action_ref.lower().split(".", 1)
    role = "primary" if slot == "primary_action" else "secondary"
    return f"{kind}_{scene}_{role}"


def _error_rows(errors: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for value in errors:
        match = _ACTION_ERROR.fullmatch(str(value))
        participant_match = _PARTICIPANT_ACTION_ERROR.fullmatch(str(value))
        if match is None and participant_match is None:
            continue
        if participant_match is not None:
            rows.append({
                "action_ref": (
                    f"S{int(participant_match.group(1)):02d}."
                    f"{participant_match.group(2)}"
                ),
                "error_code": "participant_alias_action_redeclared",
                "invalid_value": participant_match.group(3),
            })
            continue
        assert match is not None
        detail = match.group(3)
        category = ("unknown_typed_ref" if "unknown_typed_ref:" in detail
                    else detail.split(":", 1)[0])
        row = {"action_ref": match.group(1), "error_code": category}
        if match.group(2):
            row["argument_path"] = match.group(2)
        if category == "redundant_independent_action" and ":" in detail:
            row["first_action_ref"] = detail.split(":", 1)[1]
        elif "unknown_typed_ref:" in detail:
            parts = detail.split(":")
            if len(parts) >= 2:
                row["required_reference_registry"] = parts[-2]
        elif category.startswith("continuation_source_") and ":" in detail:
            row["continuation_source_ref"] = detail.rsplit(":", 1)[-1]
        rows.append(row)
    return rows


def planner_action_only_retry_eligible(
    *, outline: Any, errors: Sequence[str], parse_errors: Sequence[str] = (),
) -> bool:
    """Return true only when a parsed outline failed exclusively in action slots."""
    if parse_errors or not isinstance(outline, Mapping) or not errors:
        return False
    scenes = outline.get("scenes")
    if not isinstance(scenes, list) or not 5 <= len(scenes) <= 7:
        return False
    return all(
        _ACTION_ERROR.fullmatch(str(error)) is not None or
        _PARTICIPANT_ACTION_ERROR.fullmatch(str(error)) is not None
        for error in errors
    )


def _repair_targets(errors: Sequence[str]) -> list[str]:
    rows = _error_rows(errors)
    direct = [row["action_ref"] for row in rows
              if row["error_code"] != "continuation_source_unresolved"]
    # An invalid root action can make an otherwise well-formed continuation
    # chain unresolved. Repair the root first and revalidate the chain locally.
    values = direct or [row["action_ref"] for row in rows]
    return list(dict.fromkeys(values))


def _available_contracts(
    action_contracts: Mapping[str, Any],
    action_availability_view: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_type = {str(row.get("action_type")): row
               for row in action_contracts.get("actions") or []
               if isinstance(row, Mapping)}
    result = []
    for action_type, availability in (
            action_availability_view.get("actions") or {}).items():
        if not isinstance(availability, Mapping) or availability.get("available") is not True:
            continue
        contract = by_type.get(str(action_type))
        if contract is None:
            continue
        result.append({
            "action_type": action_type,
            "description": contract.get("description"),
            "argument_schema": deepcopy(contract.get("argument_schema") or {}),
            "availability": deepcopy(dict(availability)),
        })
    return result


def _authoritative_connection_options(
    action_availability_view: Mapping[str, Any],
) -> dict[str, list[dict[str, str]]]:
    """Flatten frozen connection maps into exact copyable typed pairs."""
    actions = action_availability_view.get("actions") or {}
    person = (actions.get("introduce_connected_person") or {}).get(
        "allowed_connections") or {}
    event = (actions.get("introduce_connected_event") or {}).get(
        "allowed_connected_existing") or {}
    return {
        "introduce_connected_person": [
            {"connection_type": str(kind), "connection_target_ref": str(ref)}
            for kind, refs in person.items() for ref in refs
        ],
        "introduce_connected_event": [
            {"ref_type": str(kind), "ref": str(ref)}
            for kind, refs in event.items() for ref in refs
        ],
    }


def _authoritative_connection_choices(
    action_availability_view: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Assign compact IDs to deterministic existing-world connection payloads."""
    options = _authoritative_connection_options(action_availability_view)
    rows: list[dict[str, Any]] = []
    for action_type, prefix in (
            ("introduce_connected_person", "CCP"),
            ("introduce_connected_event", "CCE")):
        for index, payload in enumerate(options[action_type], 1):
            rows.append({
                "connection_choice_id": f"{prefix}{index:04d}",
                "action_type": action_type,
                "program_connection_payload": deepcopy(payload),
            })
    return rows


def _materialize_program_owned_connected_action(
    replacement: Mapping[str, Any], *, action_ref: str, spec: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Expand a selected choice into fields the model is not allowed to copy."""
    action_type = str(replacement.get("action_type") or "")
    if replacement.get("action_mode") != "new" or action_type not in {
            "introduce_connected_person", "introduce_connected_event"}:
        return deepcopy(dict(replacement)), []
    arguments = replacement.get("arguments")
    if not isinstance(arguments, Mapping):
        return None, ["program_connection_arguments_not_object"]
    choices = {
        str(row.get("connection_choice_id") or ""): row
        for row in spec.get("authoritative_connection_choices") or []
        if isinstance(row, Mapping)
    }
    choice_id = str(arguments.get("connection_choice_id") or "")
    choice = choices.get(choice_id)
    if choice is None:
        return None, [f"program_connection_choice_unknown:{choice_id}"]
    if choice.get("action_type") != action_type:
        return None, [f"program_connection_choice_action_mismatch:{choice_id}"]
    canonical_fields = {
        "introduce_connected_person": {
            "participant_alias", "display_name", "entity_role", "connection_type",
            "connection_target_ref", "connection_description",
        },
        "introduce_connected_event": {
            "event_alias", "description", "participant_refs", "connected_existing",
            "narrative_time_expression", "narrative_time_relation",
            "occurrence_status", "event_consequence",
        },
    }[action_type]
    program_fields = ({"participant_alias", "connection_type", "connection_target_ref"}
                      if action_type == "introduce_connected_person"
                      else {"event_alias", "connected_existing"})
    expected_model_fields = (canonical_fields - program_fields) | {
        "connection_choice_id"}
    if set(arguments) != expected_model_fields:
        return None, ["program_connection_argument_schema"]
    alias_key = ("introduce_connected_person" if
                 action_type == "introduce_connected_person" else
                 "introduce_connected_event")
    alias = str(next((row.get(alias_key) for row in
                      spec.get("replacement_alias_choices") or []
                      if row.get("action_ref") == action_ref), "") or "")
    if not alias:
        return None, ["program_connection_alias_missing"]
    payload = deepcopy(dict(choice.get("program_connection_payload") or {}))
    materialized_arguments = {
        key: deepcopy(value) for key, value in arguments.items()
        if key != "connection_choice_id"
    }
    if action_type == "introduce_connected_person":
        materialized_arguments.update({
            "participant_alias": alias,
            "connection_type": payload.get("connection_type"),
            "connection_target_ref": payload.get("connection_target_ref"),
        })
    else:
        materialized_arguments.update({
            "event_alias": alias,
            "connected_existing": {
                "ref_type": payload.get("ref_type"), "ref": payload.get("ref")},
        })
    return {
        "action_mode": "new", "action_type": action_type,
        "arguments": materialized_arguments,
    }, []


def _reference_only_repair_viable(
    action: Mapping[str, Any], *, mutable_paths: Sequence[str],
    action_availability_view: Mapping[str, Any],
    round_aliases: Sequence[str],
) -> bool:
    """Return true only when the same action contract has a legal replacement ref."""
    action_type = str(action.get("action_type") or "")
    availability = (action_availability_view.get("actions") or {}).get(
        action_type) or {}
    if action_type == "develop_character_or_task_state" and set(mutable_paths) == {
            "target"}:
        return bool(availability.get("allowed_target_refs") or round_aliases)
    if action_type == "introduce_connected_person":
        return (
            bool(mutable_paths) and
            all(path == "connection_target_ref" or
                path.startswith("connection_target_ref.")
                for path in mutable_paths) and
            bool(availability.get("allowed_connections") or {}))
    if action_type == "introduce_connected_event":
        return (
            bool(mutable_paths) and
            all(path == "connected_existing" or
                path.startswith("connected_existing.")
                for path in mutable_paths) and
            bool(availability.get("allowed_connected_existing") or {}))
    return False


def compile_planner_action_repair_spec_v11(
    *, outline: Mapping[str, Any], errors: Sequence[str],
    validation: Mapping[str, Any], action_contracts: Mapping[str, Any],
    action_availability_view: Mapping[str, Any],
    authoritative_role_table: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Freeze an outline and expose only context needed to repair bad actions."""
    if not planner_action_only_retry_eligible(outline=outline, errors=errors):
        raise ValueError("planner_action_only_retry_not_eligible")
    targets = _repair_targets(errors)
    slots = _action_slots(outline)
    if not targets or any(ref not in slots for ref in targets):
        raise ValueError("planner_action_repair_target_missing")
    target_set = set(targets)
    continuation_view = (((validation.get("action_continuation_validation") or {}).get(
        "allowed_continuation_slot_view") or {}).get("by_scene_ref") or {})
    aliases_seen: set[str] = set()
    aliases_by_scene: dict[str, list[str]] = {}
    for scene in outline.get("scenes") or []:
        for speaker in scene.get("speaker_participants") or []:
            if isinstance(speaker, Mapping) and speaker.get("participant_alias"):
                aliases_seen.add(str(speaker["participant_alias"]))
        aliases_by_scene[str(scene.get("scene_alias") or "")] = sorted(aliases_seen)
    diagnostics = _error_rows(errors)
    diagnostics_by_ref: dict[str, list[dict[str, str]]] = {}
    for row in diagnostics:
        diagnostics_by_ref.setdefault(row["action_ref"], []).append(row)
    role_choices = [{
        "participant_ref": str(row.get("entity_ref") or row.get("participant_ref") or ""),
        "display_name": str(row.get("display_name") or ""),
    } for row in authoritative_role_table
        if row.get("entity_ref") or row.get("participant_ref")]
    contexts = []
    reference_only_constraints: list[dict[str, Any]] = []
    for ref in targets:
        match = _ACTION_REF.fullmatch(ref)
        assert match is not None
        scene = (outline.get("scenes") or [])[int(match.group(1)) - 1]
        scene_ref = ref.split(".", 1)[0]
        original_action = deepcopy(slots[ref])
        context = {
            "action_ref": ref,
            "scene_alias": scene.get("scene_alias"),
            "goal": scene.get("goal"),
            "new_development": scene.get("new_development"),
            "desired_outcome": scene.get("desired_outcome"),
            "dependencies": deepcopy(scene.get("dependencies") or []),
            "speaker_participant_refs": [
                row.get("participant_ref") or row.get("participant_alias")
                for row in scene.get("speaker_participants") or []
                if isinstance(row, Mapping)
            ],
            "available_round_participant_aliases": aliases_by_scene.get(
                str(scene.get("scene_alias") or ""), []),
            "allowed_continuation_of": list(continuation_view.get(scene_ref) or []),
        }
        target_diagnostics = diagnostics_by_ref.get(ref, [])
        mutable_paths = sorted({
            str(row.get("argument_path") or "")
            for row in target_diagnostics
            if row.get("error_code") in {"unknown_world_ref", "unknown_typed_ref"}
            and row.get("argument_path")
        })
        reference_only = bool(
            mutable_paths and isinstance(original_action, Mapping) and
            original_action.get("action_mode") == "new" and
            all(row.get("error_code") in {"unknown_world_ref", "unknown_typed_ref"}
                for row in target_diagnostics) and
            _reference_only_repair_viable(
                original_action, mutable_paths=mutable_paths,
                action_availability_view=action_availability_view,
                round_aliases=context["available_round_participant_aliases"])
        )
        if reference_only:
            arguments = deepcopy(original_action.get("arguments") or {})
            program_owned_fields = set()
            if original_action.get("action_type") == "introduce_connected_person":
                program_owned_fields = {
                    "participant_alias", "connection_type", "connection_target_ref",
                }
            elif original_action.get("action_type") == "introduce_connected_event":
                program_owned_fields = {"event_alias", "connected_existing"}
            else:
                # Ordinary reference repairs need the complete frozen action as
                # their response template. Connected actions use the dedicated
                # choice contract below so an invalid local reference is never
                # echoed back into the retry prompt.
                context["original_action"] = original_action
            frozen_arguments = {
                key: value for key, value in arguments.items()
                if key not in ({path.split(".", 1)[0] for path in mutable_paths} |
                               program_owned_fields)
            }
            reference_only_constraints.append({
                "action_ref": ref,
                "required_action_mode": "new",
                "required_action_type": original_action.get("action_type"),
                "frozen_narrative_alias": (
                    arguments.get("participant_alias")
                    if original_action.get("action_type") ==
                    "introduce_connected_person"
                    else arguments.get("event_alias")),
                "mutable_argument_paths": mutable_paths,
                "frozen_arguments": frozen_arguments,
                "authoritative_reference_choices": role_choices,
                "round_participant_alias_choices": list(
                    context["available_round_participant_aliases"]),
                "connection_choice_ids": [
                    str(row.get("connection_choice_id") or "")
                    for row in _authoritative_connection_choices(
                        action_availability_view)
                    if row.get("action_type") == original_action.get("action_type")
                ],
            })
        contexts.append(context)
    untouched = {ref: value for ref, value in slots.items() if ref not in target_set}
    signatures = [{"action_ref": ref, "action": deepcopy(value)}
                  for ref, value in untouched.items()
                  if isinstance(value, Mapping) and value.get("action_mode") == "new"]
    reserved_aliases = []
    for ref, action in untouched.items():
        owned = _introduced_narrative_alias(action)
        if owned is not None:
            reserved_aliases.append({
                "alias_kind": owned[0], "alias": owned[1],
                "owner_action_ref": ref,
            })
    constraint_by_ref = {
        str(row.get("action_ref") or ""): row
        for row in reference_only_constraints
    }
    replacement_alias_choices = []
    for ref in targets:
        frozen_alias = str((constraint_by_ref.get(ref) or {}).get(
            "frozen_narrative_alias") or "")
        required_type = str((constraint_by_ref.get(ref) or {}).get(
            "required_action_type") or "")
        replacement_alias_choices.append({
            "action_ref": ref,
            "introduce_connected_person": (
                frozen_alias if required_type == "introduce_connected_person"
                and frozen_alias else _repair_slot_alias(ref, "person")),
            "introduce_connected_event": (
                frozen_alias if required_type == "introduce_connected_event"
                and frozen_alias else _repair_slot_alias(ref, "event")),
        })
    result = {
        "repair_version": PLANNER_ACTION_REPAIR_VERSION_V11,
        "target_action_refs": targets,
        "diagnostics": diagnostics,
        "target_scene_contexts": contexts,
        "available_action_contracts": _available_contracts(
            action_contracts, action_availability_view),
        "typed_reference_registry": deepcopy(
            action_availability_view.get("typed_reference_registry") or {}),
        "authoritative_connection_options": _authoritative_connection_options(
            action_availability_view),
        "authoritative_connection_choices": _authoritative_connection_choices(
            action_availability_view),
        "authoritative_reference_choices": role_choices,
        "reference_only_constraints": reference_only_constraints,
        "untouched_new_action_signatures": signatures,
        "reserved_narrative_aliases": reserved_aliases,
        "replacement_alias_choices": replacement_alias_choices,
        "frozen_outline_sha256": _sha(outline),
        "frozen_non_action_outline_sha256": _sha(_non_action_view(outline)),
        "frozen_untouched_action_slots_sha256": _sha(untouched),
    }
    result["repair_spec_sha256"] = _sha(result)
    return result


def build_planner_action_only_retry_prompt_v11(
    spec: Mapping[str, Any], *, previous_contract_errors: Sequence[str] = (),
) -> str:
    """Build a compact prompt that cannot rewrite dialogue-outline fields."""
    targets = list(spec.get("target_action_refs") or [])
    template = {"action_repairs": [
        build_planner_action_repair_replacement_template_v11(
            action_ref=ref, spec=spec)
        for ref in targets
    ]}
    sections = [
        "PLANNER ACTION-ONLY REPAIR v11. Return one JSON object only.",
        "The complete outline is locally frozen and valid outside the named action slots. "
        "Do not return or rewrite round fields, scenes, anchors, placements, speakers, beats, "
        "dependencies, goals, or desired outcomes. Explicitly resubmit one replacement for "
        "each target action slot; local code will reconstruct and fully revalidate the outline.",
        "TARGET ACTION DIAGNOSTICS\n" + _canonical(spec.get("diagnostics") or []),
        "FROZEN TARGET SCENE CONTEXT\n" + _canonical(
            spec.get("target_scene_contexts") or []),
        "AVAILABLE ACTION CONTRACTS — only available=true actions\n" + _canonical(
            spec.get("available_action_contracts") or []),
        "AUTHORITATIVE TYPED REFERENCES — copy exact values only\n" + _canonical(
            spec.get("typed_reference_registry") or {}),
        "AUTHORITATIVE REFERENCE CHOICES — labels explain identity; output the exact ref, "
        "never the label\n" + _canonical(
            spec.get("authoritative_reference_choices") or []),
        "REFERENCE-ONLY FREEZE — for these targets preserve action_type and every frozen "
        "argument; change only the listed invalid reference path\n" + _canonical(
            spec.get("reference_only_constraints") or []),
        "PROGRAM CONNECTION CHOICES — for introduce_connected_person or "
        "introduce_connected_event select exactly one connection_choice_id. Do not return "
        "the choice payload, connection type/ref, participant_alias, or event_alias; local "
        "code expands those deterministic fields.\n" + _canonical(
            spec.get("authoritative_connection_choices") or []),
        "CONNECTED ACTION REPAIR SHAPES — use exactly these model-owned argument keys\n" +
        _canonical({
            "introduce_connected_person": {
                "action_mode": "new",
                "action_type": "introduce_connected_person",
                "arguments": {
                    "connection_choice_id": "<ONE_CCP_CHOICE_ID>",
                    "display_name": "<NAME>",
                    "entity_role": "<ROLE>",
                    "connection_description": "<CONCRETE_CONNECTION>",
                },
            },
            "introduce_connected_event": {
                "action_mode": "new",
                "action_type": "introduce_connected_event",
                "arguments": {
                    "connection_choice_id": "<ONE_CCE_CHOICE_ID>",
                    "description": "<EVENT>",
                    "participant_refs": ["<VALID_PARTICIPANT_REF>"],
                    "narrative_time_expression": None,
                    "narrative_time_relation": "unspecified",
                    "occurrence_status": "proposed",
                    "event_consequence": "<CONSEQUENCE>",
                },
            },
        }),
        "UNTOUCHED NEW ACTIONS — avoid an identical new signature\n" + _canonical(
            spec.get("untouched_new_action_signatures") or []),
        "RESERVED NARRATIVE ALIASES — these identities already belong to untouched action "
        "slots; never introduce them again\n" + _canonical(
            spec.get("reserved_narrative_aliases") or []),
        "PROGRAM-ASSIGNED REPLACEMENT ALIASES — shown for audit only. Never return these "
        "aliases; local code inserts the alias for the repaired action slot.\n" + _canonical(
            spec.get("replacement_alias_choices") or []),
        COMPLETE_ACTION_REPLACEMENT_INSTRUCTIONS_V11,
        "RULES\n"
        "- Top level has exactly action_repairs.\n"
        "- Repair exactly the listed action_ref values, once each.\n"
        "- replacement is one complete, non-empty action object. Never delete an action.\n"
        "- new has exactly action_mode, action_type, arguments.\n"
        "- continuation has exactly action_mode, continuation_of, progression.\n"
        "- A target listed in REFERENCE-ONLY FREEZE is not a request to redesign the "
        "action. Keep required_action_type and frozen_arguments byte-for-byte equivalent; "
        "replace only mutable_argument_paths with an exact permitted ref or listed round "
        "alias.\n"
        "- continuation_of must be listed for that target scene and must name a dependency "
        "ancestor slot.\n"
        "- A connected-person repair omits participant_alias, connection_type, and "
        "connection_target_ref and supplies connection_choice_id instead. A connected-event "
        "repair omits event_alias and connected_existing and supplies connection_choice_id "
        "instead. Local code expands these fields and the program-owned alias. Continue an existing "
        "introduction with continuation_of rather than introducing its alias again. Never output "
        "internal allocation or final IDs. Narrative aliases "
        "are not authoritative typed references, but an alias listed in the target scene's "
        "available_round_participant_aliases may be used only in participant-valued fields "
        "whose action availability permits it; local compilation resolves it.\n"
        "- Do not copy an identical untouched new action. Do not output action_id, bundle_id, "
        "Memory operation, schema, previous errors, or allowed-reference metadata.\n"
        "- The angle-bracket replacement text in the output skeleton is an instruction, not "
        "a valid value. Replace every occurrence with a complete action object. A literal "
        "placeholder, null, or empty object is invalid.",
    ]
    if previous_contract_errors:
        sections.append("PREVIOUS PATCH CONTRACT ERRORS\n" + _canonical(
            list(previous_contract_errors)))
    sections.extend([
        "FROZEN OUTLINE SHA256\n" + str(spec.get("frozen_outline_sha256") or ""),
        "OUTPUT ONLY THIS EXACT SHAPE; replace every angle-bracket instruction with one "
        "valid action branch\n" +
        json.dumps(template, ensure_ascii=False, indent=2),
    ])
    return "\n\n".join(sections)


def build_planner_action_repair_replacement_template_v11(
    *, action_ref: str, spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a target-specific skeleton for the model-facing repair prompt."""
    constraint = next((row for row in spec.get("reference_only_constraints") or []
                       if isinstance(row, Mapping) and
                       row.get("action_ref") == action_ref), None)
    if isinstance(constraint, Mapping) and constraint.get(
            "required_action_type") in {
                "introduce_connected_person", "introduce_connected_event"}:
        action_type = str(constraint["required_action_type"])
        choice_placeholder = (
            "<ONE_EXACT_" + ("CCP" if action_type == "introduce_connected_person"
                              else "CCE") + "_CHOICE_ID>")
        arguments = {"connection_choice_id": choice_placeholder}
        arguments.update(deepcopy(constraint.get("frozen_arguments") or {}))
        return {
            "action_ref": action_ref,
            "replacement": {
                "action_mode": "new", "action_type": action_type,
                "arguments": arguments,
            },
        }
    return {
        "action_ref": action_ref,
        "replacement": "<COMPLETE_NEW_OR_CONTINUATION_ACTION_OBJECT_REQUIRED>",
    }


def validate_planner_action_repair_response_v11(
    response: Any, *, spec: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    targets = list(spec.get("target_action_refs") or [])
    if not isinstance(response, Mapping):
        return {"valid": False, "errors": ["planner_action_repair:not_object"]}
    if set(response) != {"action_repairs"}:
        errors.append("planner_action_repair:top_level_exact_keys")
    repairs = response.get("action_repairs")
    repairs = repairs if isinstance(repairs, list) else []
    if not isinstance(response.get("action_repairs"), list):
        errors.append("planner_action_repair:repairs_not_array")
    refs = []
    contexts = {
        str(row.get("action_ref") or ""): row
        for row in spec.get("target_scene_contexts") or []
        if isinstance(row, Mapping)
    }
    available_rows = {
        str(row.get("action_type") or ""): row
        for row in spec.get("available_action_contracts") or []
        if isinstance(row, Mapping) and row.get("action_type")
    }
    typed_refs = spec.get("typed_reference_registry")
    typed_refs = dict(typed_refs) if isinstance(typed_refs, Mapping) else {}
    frozen_availability = {
        "actions": {
            action_type: deepcopy(dict(row.get("availability") or {}))
            for action_type, row in available_rows.items()
        },
        "typed_reference_registry": deepcopy(typed_refs),
    }
    reference_constraints = {
        str(row.get("action_ref") or ""): row
        for row in spec.get("reference_only_constraints") or []
        if isinstance(row, Mapping)
    }
    reserved_aliases = {
        str(row.get("alias") or ""): str(row.get("owner_action_ref") or "")
        for row in spec.get("reserved_narrative_aliases") or []
        if isinstance(row, Mapping) and row.get("alias")
    }
    alias_choices = {
        str(row.get("action_ref") or ""): row
        for row in spec.get("replacement_alias_choices") or []
        if isinstance(row, Mapping) and row.get("action_ref")
    }
    submitted_aliases: dict[str, str] = {}
    materialized_replacements: dict[str, dict[str, Any]] = {}
    replacement_validations: list[dict[str, Any]] = []
    for index, repair in enumerate(repairs, 1):
        path = f"planner_action_repair:item:{index}"
        if not isinstance(repair, Mapping) or set(repair) != {"action_ref", "replacement"}:
            errors.append(f"{path}:exact_keys")
            continue
        ref = str(repair.get("action_ref") or "")
        refs.append(ref)
        if _ACTION_REF.fullmatch(ref) is None:
            errors.append(f"{path}:action_ref_format")
        replacement = repair.get("replacement")
        if not isinstance(replacement, Mapping):
            errors.append(f"{path}:replacement_not_object")
            continue
        if not replacement:
            errors.append(f"{path}:replacement_empty")
            continue
        context = contexts.get(ref)
        if context is None:
            errors.append(f"{path}:target_context_missing")
            continue
        materialized, materialize_errors = _materialize_program_owned_connected_action(
            replacement, action_ref=ref, spec=spec)
        errors.extend(f"{path}:{value}" for value in materialize_errors)
        if materialized is None:
            continue
        materialized_replacements[ref] = materialized
        mode = materialized.get("action_mode")
        if mode == "new":
            action_type = str(materialized.get("action_type") or "")
            if action_type not in available_rows:
                errors.append(f"{path}:replacement_action_not_available:{action_type}")
        elif mode == "continuation":
            continuation_of = str(replacement.get("continuation_of") or "")
            if continuation_of not in set(context.get("allowed_continuation_of") or []):
                errors.append(
                    f"{path}:replacement_continuation_not_allowed:{continuation_of}")
        validation = validate_action_instance(
            materialized,
            path=f"{path}:replacement",
            known_world_refs=set(str(value) for value in
                                 typed_refs.get("all_typed_refs") or []),
            authoritative_entity_refs=set(str(value) for value in
                                          typed_refs.get("participant_refs") or []),
            action_availability_view=frozen_availability,
            round_participant_aliases=set(str(value) for value in
                                          context.get(
                                              "available_round_participant_aliases") or []),
        )
        replacement_validations.append({
            "action_ref": ref,
            "valid": validation["valid"],
            "errors": validation["errors"],
        })
        errors.extend(validation["errors"])
        owned_alias = _introduced_narrative_alias(materialized)
        if owned_alias is not None:
            kind, alias = owned_alias
            expected_key = (
                "introduce_connected_person" if kind == "person"
                else "introduce_connected_event"
            )
            expected_alias = str(
                (alias_choices.get(ref) or {}).get(expected_key) or "")
            if alias != expected_alias:
                errors.append(
                    f"{path}:replacement_alias_not_program_assigned:{alias}")
            if alias in reserved_aliases:
                errors.append(
                    f"{path}:replacement_alias_reserved:{alias}:"
                    f"{reserved_aliases[alias]}")
            first_ref = submitted_aliases.get(alias)
            if first_ref is not None and first_ref != ref:
                errors.append(
                    f"{path}:replacement_alias_collision:{alias}:{first_ref}")
            submitted_aliases.setdefault(alias, ref)
        constraint = reference_constraints.get(ref)
        if constraint is not None:
            if materialized.get("action_mode") != constraint.get(
                    "required_action_mode"):
                errors.append(f"{path}:reference_only_action_mode_changed")
            if materialized.get("action_type") != constraint.get(
                    "required_action_type"):
                errors.append(f"{path}:reference_only_action_type_changed")
            replacement_arguments = materialized.get("arguments")
            if not isinstance(replacement_arguments, Mapping):
                errors.append(f"{path}:reference_only_arguments_missing")
            else:
                for key, frozen_value in (
                        constraint.get("frozen_arguments") or {}).items():
                    if replacement_arguments.get(key) != frozen_value:
                        errors.append(
                            f"{path}:reference_only_frozen_argument_changed:{key}")
    if len(refs) != len(set(refs)):
        errors.append("planner_action_repair:duplicate_action_ref")
    if set(refs) != set(targets) or len(refs) != len(targets):
        errors.append("planner_action_repair:target_set_mismatch")
    return {"valid": not errors, "errors": list(dict.fromkeys(errors)),
            "target_action_refs": targets,
            "replacement_validations": replacement_validations,
            "materialized_replacements": materialized_replacements}


def apply_planner_action_repairs_v11(
    *, frozen_outline: Mapping[str, Any], response: Any, spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply explicit replacements and prove every non-target value stayed frozen."""
    validation = validate_planner_action_repair_response_v11(response, spec=spec)
    if not validation["valid"]:
        return {"valid": False, "errors": validation["errors"],
                "reconstructed_outline": None}
    candidate = deepcopy(dict(frozen_outline))
    replacements = deepcopy(validation["materialized_replacements"])
    for ref, replacement in replacements.items():
        match = _ACTION_REF.fullmatch(ref)
        assert match is not None
        candidate["scenes"][int(match.group(1)) - 1][match.group(2)] = replacement
    targets = set(spec.get("target_action_refs") or [])
    original_slots = _action_slots(frozen_outline)
    candidate_slots = _action_slots(candidate)
    untouched_original = {ref: value for ref, value in original_slots.items()
                          if ref not in targets}
    untouched_candidate = {ref: value for ref, value in candidate_slots.items()
                           if ref not in targets}
    audits = {
        "frozen_outline_sha256_matches": _sha(frozen_outline) == spec.get(
            "frozen_outline_sha256"),
        "non_action_outline_sha256_unchanged": (
            _sha(_non_action_view(candidate)) == spec.get(
                "frozen_non_action_outline_sha256")),
        "untouched_action_slots_sha256_unchanged": (
            _sha(untouched_candidate) == spec.get(
                "frozen_untouched_action_slots_sha256") == _sha(untouched_original)),
        "target_action_refs_exact": set(replacements) == targets,
    }
    errors = [f"planner_action_repair:{key}" for key, value in audits.items() if not value]
    return {"valid": not errors, "errors": errors,
            "reconstructed_outline": candidate if not errors else None,
            "freeze_audit": audits, "repair_response_validation": validation}


__all__ = [
    "COMPLETE_ACTION_REPLACEMENT_INSTRUCTIONS_V11",
    "PLANNER_ACTION_REPAIR_PROMPT_VERSION_V11",
    "PLANNER_ACTION_REPAIR_VERSION_V11",
    "apply_planner_action_repairs_v11",
    "build_planner_action_only_retry_prompt_v11",
    "build_planner_action_repair_replacement_template_v11",
    "compile_planner_action_repair_spec_v11",
    "planner_action_only_retry_eligible",
    "validate_planner_action_repair_response_v11",
]
