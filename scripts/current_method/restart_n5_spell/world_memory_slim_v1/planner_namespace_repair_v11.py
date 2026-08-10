"""Frozen Planner repair for action-type and link-type namespace errors."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .continuous_round_plan_v1 import PLACEMENT_LINK_TYPES
from .planner_action_repair_v11 import (
    COMPLETE_ACTION_REPLACEMENT_INSTRUCTIONS_V11,
    apply_planner_action_repairs_v11,
    build_planner_action_repair_replacement_template_v11,
    compile_planner_action_repair_spec_v11,
    validate_planner_action_repair_response_v11,
)


PLANNER_NAMESPACE_REPAIR_VERSION_V11 = (
    "planner_namespace_only_repair_v11_v5_connected_reference_templates"
)
PLANNER_NAMESPACE_REPAIR_PROMPT_VERSION_V11 = (
    "anchor_first_slim_outline_namespace_only_repair_v11_prompt_v6_connected_reference_templates"
)
_ACTION_ERROR = re.compile(
    r"^action:(S0[1-7]\.(?:primary_action|secondary_action))"
    r"(?:\.arguments(?:\.[^:]+)?)?:(.+)$"
)
_LINK_TYPE_ERROR = re.compile(r"^scene:([1-7]):link_type:(.+)$")
_SCENE_REF = re.compile(r"^S(0[1-7])$")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _without_targets(
    outline: Mapping[str, Any], *, action_refs: Sequence[str],
    link_scene_refs: Sequence[str],
) -> dict[str, Any]:
    value = deepcopy(dict(outline))
    action_set = set(action_refs)
    link_set = set(link_scene_refs)
    for index, scene in enumerate(value.get("scenes") or [], 1):
        if not isinstance(scene, dict):
            continue
        scene_ref = f"S{index:02d}"
        if scene_ref in link_set:
            scene.pop("link_type", None)
        for slot in ("primary_action", "secondary_action"):
            if f"{scene_ref}.{slot}" in action_set:
                scene.pop(slot, None)
    return value


def planner_namespace_only_retry_eligible(
    *, outline: Any, errors: Sequence[str], parse_errors: Sequence[str] = (),
) -> bool:
    """Accept only parsed 5-7 scene plans with action/link namespace errors."""
    if parse_errors or not isinstance(outline, Mapping) or not errors:
        return False
    scenes = outline.get("scenes")
    if not isinstance(scenes, list) or not 5 <= len(scenes) <= 7:
        return False
    action_or_link = [
        _ACTION_ERROR.fullmatch(str(error)) is not None or
        _LINK_TYPE_ERROR.fullmatch(str(error)) is not None
        for error in errors
    ]
    return all(action_or_link) and any(
        _LINK_TYPE_ERROR.fullmatch(str(error)) is not None for error in errors)


def compile_planner_namespace_repair_spec_v11(
    *, outline: Mapping[str, Any], errors: Sequence[str],
    validation: Mapping[str, Any], action_contracts: Mapping[str, Any],
    action_availability_view: Mapping[str, Any],
    authoritative_role_table: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if not planner_namespace_only_retry_eligible(outline=outline, errors=errors):
        raise ValueError("planner_namespace_only_retry_not_eligible")
    action_errors = [str(error) for error in errors
                     if _ACTION_ERROR.fullmatch(str(error)) is not None]
    link_matches = [_LINK_TYPE_ERROR.fullmatch(str(error)) for error in errors]
    link_scene_refs = list(dict.fromkeys(
        f"S{int(match.group(1)):02d}" for match in link_matches if match is not None))
    action_spec = None
    if action_errors:
        action_spec = compile_planner_action_repair_spec_v11(
            outline=outline, errors=action_errors, validation=validation,
            action_contracts=action_contracts,
            action_availability_view=action_availability_view,
            authoritative_role_table=authoritative_role_table)
    action_refs = list((action_spec or {}).get("target_action_refs") or [])
    link_contexts = []
    for scene_ref in link_scene_refs:
        match = _SCENE_REF.fullmatch(scene_ref)
        assert match is not None
        scene = (outline.get("scenes") or [])[int(match.group(1)) - 1]
        link_contexts.append({
            "scene_ref": scene_ref,
            "source_anchor_id": scene.get("source_anchor_id"),
            "placement_choice_id": scene.get("placement_choice_id"),
            "goal": scene.get("goal"),
            "bridge_reason": scene.get("bridge_reason"),
            "new_development": scene.get("new_development"),
        })
    link_type_choices = [
        {"choice_id": f"LT{index:02d}", "link_type": link_type}
        for index, link_type in enumerate(sorted(PLACEMENT_LINK_TYPES), 1)
    ]
    result = {
        "repair_version": PLANNER_NAMESPACE_REPAIR_VERSION_V11,
        "target_action_refs": action_refs,
        "target_link_scene_refs": link_scene_refs,
        "action_repair_spec": action_spec,
        "link_type_diagnostics": [{
            "scene_ref": f"S{int(match.group(1)):02d}",
            "error_code": "invalid_link_type",
        } for match in link_matches if match is not None],
        "link_target_scene_contexts": link_contexts,
        "allowed_link_types": sorted(PLACEMENT_LINK_TYPES),
        "link_type_choices": link_type_choices,
        "frozen_outline_sha256": _sha(outline),
        "frozen_non_target_fields_sha256": _sha(_without_targets(
            outline, action_refs=action_refs, link_scene_refs=link_scene_refs)),
    }
    result["repair_spec_sha256"] = _sha(result)
    return result


def build_planner_namespace_only_retry_prompt_v11(
    spec: Mapping[str, Any], *, previous_contract_errors: Sequence[str] = (),
) -> str:
    action_spec = spec.get("action_repair_spec") or {}
    template = {
        "action_repairs": [
            build_planner_action_repair_replacement_template_v11(
                action_ref=ref, spec=action_spec)
            for ref in spec.get("target_action_refs") or []],
        "link_type_repairs": [{
            "scene_ref": ref,
            "replacement_choice_id": "<ONE_EXACT_LINK_CHOICE_ID_REQUIRED>",
        }
                              for ref in spec.get("target_link_scene_refs") or []],
    }
    sections = [
        "PLANNER NAMESPACE-ONLY REPAIR v11. Return one JSON object only.",
        "The full outline is valid outside the named action slots and link_type fields. "
        "Those other fields are frozen. Do not return scenes or rewrite goals, speakers, "
        "beats, dependencies, anchors, placements, or desired outcomes.",
        "FIELD NAMESPACE\n"
        "link_type describes how a scene connects to history. Select one opaque choice_id "
        "from LINK TYPE CHOICES; local code expands it to the frozen link_type. Do not return "
        "a natural-language category.",
        "LINK TYPE TARGETS\n" + _canonical(spec.get("link_type_diagnostics") or []),
        "FROZEN LINK TARGET CONTEXT\n" + _canonical(
            spec.get("link_target_scene_contexts") or []),
        "LINK TYPE CHOICES\n" + _canonical(spec.get("link_type_choices") or []),
        "RULES\n"
        "- Top level has exactly action_repairs and link_type_repairs.\n"
        "- Return each requested action_ref and scene_ref exactly once.\n"
        "- replacement for an action is one complete new or continuation branch.\n"
        "- replacement_choice_id is one exact ID from LINK TYPE CHOICES.\n"
        "- Do not return link_type or the invalid old value.\n"
        "- Do not output schema, diagnostics, allowed values, previous errors, or metadata.\n"
        "- Angle-bracket text in the output skeleton is an instruction, not a valid value. "
        "Replace it completely. Empty action objects and literal placeholders are invalid.",
    ]
    if action_spec:
        sections[3:3] = [
            "ACTION FIELD NAMESPACES\n"
            "- action_type is one exact AVAILABLE ACTION CONTRACT value.\n"
            "- progression.stage is development, escalation, consolidation, or resolution.\n"
            "Never synthesize or rename action_type.",
            "ACTION TARGET DIAGNOSTICS\n" + _canonical(
                action_spec.get("diagnostics") or []),
            "FROZEN TARGET SCENE CONTEXT\n" + _canonical(
                action_spec.get("target_scene_contexts") or []),
            "AVAILABLE ACTION CONTRACTS — exact registered action_type values only\n" +
            _canonical(action_spec.get("available_action_contracts") or []),
            "AUTHORITATIVE TYPED REFERENCES — copy exact values only\n" + _canonical(
                action_spec.get("typed_reference_registry") or {}),
            "AUTHORITATIVE REFERENCE CHOICES — labels are explanatory only; output the "
            "exact participant_ref\n" + _canonical(
                action_spec.get("authoritative_reference_choices") or []),
            "REFERENCE-ONLY FREEZE — preserve valid action semantics and replace only the "
            "listed invalid reference paths\n" + _canonical(
                action_spec.get("reference_only_constraints") or []),
            "PROGRAM CONNECTION CHOICES — connected actions select one choice ID; local "
            "code inserts the exact typed connection and narrative alias\n" + _canonical(
                action_spec.get("authoritative_connection_choices") or []),
            "CONNECTED ACTION REPAIR RULE — introduce_connected_person omits "
            "participant_alias/connection_type/connection_target_ref and adds "
            "connection_choice_id. introduce_connected_event omits "
            "event_alias/connected_existing and adds connection_choice_id.",
            "UNTOUCHED NEW ACTIONS — avoid identical independent signatures\n" +
            _canonical(action_spec.get("untouched_new_action_signatures") or []),
            COMPLETE_ACTION_REPLACEMENT_INSTRUCTIONS_V11,
        ]
    if previous_contract_errors:
        sections.append("PREVIOUS PATCH CONTRACT ERRORS\n" + _canonical(
            list(previous_contract_errors)))
    sections.extend([
        "FROZEN OUTLINE SHA256\n" + str(spec.get("frozen_outline_sha256") or ""),
        "OUTPUT ONLY THIS EXACT SHAPE; replace every angle-bracket instruction\n" +
        json.dumps(template, ensure_ascii=False, indent=2),
    ])
    return "\n\n".join(sections)


def validate_planner_namespace_repair_response_v11(
    response: Any, *, spec: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(response, Mapping):
        return {"valid": False, "errors": ["planner_namespace_repair:not_object"]}
    if set(response) != {"action_repairs", "link_type_repairs"}:
        errors.append("planner_namespace_repair:top_level_exact_keys")
    action_spec = spec.get("action_repair_spec")
    action_response = {"action_repairs": response.get("action_repairs")}
    if action_spec is not None:
        errors.extend(validate_planner_action_repair_response_v11(
            action_response, spec=action_spec)["errors"])
    elif response.get("action_repairs") != []:
        errors.append("planner_namespace_repair:unexpected_action_repairs")
    rows = response.get("link_type_repairs")
    if not isinstance(rows, list):
        errors.append("planner_namespace_repair:link_repairs_not_array")
        rows = []
    refs: list[str] = []
    choices = {str(row.get("choice_id")): str(row.get("link_type"))
               for row in spec.get("link_type_choices") or []
               if isinstance(row, Mapping)}
    choice_contract = bool(choices)
    allowed = set(spec.get("allowed_link_types") or [])
    for index, row in enumerate(rows, 1):
        prefix = f"planner_namespace_repair:link_item:{index}"
        expected_fields = ({"scene_ref", "replacement_choice_id"}
                           if choice_contract else {"scene_ref", "replacement"})
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            errors.append(f"{prefix}:exact_keys")
            continue
        ref = str(row.get("scene_ref") or "")
        refs.append(ref)
        if _SCENE_REF.fullmatch(ref) is None:
            errors.append(f"{prefix}:scene_ref_format")
        if choice_contract:
            if row.get("replacement_choice_id") not in choices:
                errors.append(f"{prefix}:replacement_choice_id_not_allowed")
        elif row.get("replacement") not in allowed:
            errors.append(f"{prefix}:replacement_not_allowed")
    targets = list(spec.get("target_link_scene_refs") or [])
    if len(refs) != len(set(refs)):
        errors.append("planner_namespace_repair:duplicate_scene_ref")
    if set(refs) != set(targets) or len(refs) != len(targets):
        errors.append("planner_namespace_repair:link_target_set_mismatch")
    return {"valid": not errors, "errors": list(dict.fromkeys(errors)),
            "target_action_refs": list(spec.get("target_action_refs") or []),
            "target_link_scene_refs": targets}


def apply_planner_namespace_repairs_v11(
    *, frozen_outline: Mapping[str, Any], response: Any,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_planner_namespace_repair_response_v11(response, spec=spec)
    if not validation["valid"]:
        return {"valid": False, "errors": validation["errors"],
                "reconstructed_outline": None}
    candidate = deepcopy(dict(frozen_outline))
    action_audit = None
    action_spec = spec.get("action_repair_spec")
    if action_spec is not None:
        action_result = apply_planner_action_repairs_v11(
            frozen_outline=frozen_outline,
            response={"action_repairs": response["action_repairs"]},
            spec=action_spec)
        if not action_result["valid"]:
            return {"valid": False, "errors": action_result["errors"],
                    "reconstructed_outline": None,
                    "action_repair_audit": action_result}
        candidate = action_result["reconstructed_outline"]
        action_audit = {key: value for key, value in action_result.items()
                        if key != "reconstructed_outline"}
    for row in response["link_type_repairs"]:
        match = _SCENE_REF.fullmatch(str(row["scene_ref"]))
        assert match is not None
        choices = {str(value.get("choice_id")): str(value.get("link_type"))
                   for value in spec.get("link_type_choices") or []
                   if isinstance(value, Mapping)}
        replacement = (choices[str(row["replacement_choice_id"])]
                       if choices else row["replacement"])
        candidate["scenes"][int(match.group(1)) - 1]["link_type"] = replacement
    audits = {
        "frozen_outline_sha256_matches": (
            _sha(frozen_outline) == spec.get("frozen_outline_sha256")),
        "non_target_fields_sha256_unchanged": (
            _sha(_without_targets(
                candidate,
                action_refs=spec.get("target_action_refs") or [],
                link_scene_refs=spec.get("target_link_scene_refs") or [],
            )) == spec.get("frozen_non_target_fields_sha256")),
    }
    errors = [f"planner_namespace_repair:{key}"
              for key, value in audits.items() if not value]
    return {"valid": not errors, "errors": errors,
            "reconstructed_outline": candidate if not errors else None,
            "freeze_audit": audits, "action_repair_audit": action_audit,
            "repair_response_validation": validation}


__all__ = [
    "PLANNER_NAMESPACE_REPAIR_PROMPT_VERSION_V11",
    "PLANNER_NAMESPACE_REPAIR_VERSION_V11",
    "apply_planner_namespace_repairs_v11",
    "build_planner_namespace_only_retry_prompt_v11",
    "compile_planner_namespace_repair_spec_v11",
    "planner_namespace_only_retry_eligible",
    "validate_planner_namespace_repair_response_v11",
]
