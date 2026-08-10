"""Frozen speaker-only Planner retry for continuous expansion v11.

The first parsed outline remains authoritative outside speaker_participants in
the named scenes.  The model must explicitly resubmit valid authoritative
speakers; local code never substitutes or deletes a temporary entity.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence


PLANNER_SPEAKER_REPAIR_VERSION_V11 = "planner_speaker_only_repair_v11_v2_alias_guard"
PLANNER_SPEAKER_REPAIR_PROMPT_VERSION_V11 = (
    "anchor_first_slim_outline_speaker_only_repair_v11_prompt_v2_alias_guard"
)
_SPEAKER_ERROR = re.compile(
    r"^scene:(?P<scene>[1-7]):(?:speaker:(?:[1-4]):"
    r"(?:non_authoritative|role|intents|schema)(?::.*)?|"
    r"speaker_count|speaker_duplicate)$"
)
_SCENE_REF = re.compile(r"^S(0[1-7])$")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _without_target_speakers(
    outline: Mapping[str, Any], target_scene_refs: set[str],
) -> dict[str, Any]:
    result = deepcopy(dict(outline))
    for index, scene in enumerate(result.get("scenes") or [], 1):
        if isinstance(scene, dict) and f"S{index:02d}" in target_scene_refs:
            scene.pop("speaker_participants", None)
    return result


def _untouched_speakers(
    outline: Mapping[str, Any], target_scene_refs: set[str],
) -> dict[str, Any]:
    return {
        f"S{index:02d}": deepcopy(scene.get("speaker_participants"))
        for index, scene in enumerate(outline.get("scenes") or [], 1)
        if isinstance(scene, Mapping) and f"S{index:02d}" not in target_scene_refs
    }


def planner_speaker_only_retry_eligible(
    *, outline: Any, errors: Sequence[str], parse_errors: Sequence[str] = (),
) -> bool:
    if parse_errors or not isinstance(outline, Mapping) or not errors:
        return False
    scenes = outline.get("scenes")
    if not isinstance(scenes, list) or not 5 <= len(scenes) <= 7:
        return False
    matches = [_SPEAKER_ERROR.fullmatch(str(error)) for error in errors]
    if any(match is None for match in matches):
        return False
    affected = {int(match.group("scene")) for match in matches if match is not None}
    # An alias declaration/reference error needs the full participant contract;
    # authoritative-only repair would silently erase a legitimate new speaker.
    return not any(
        isinstance(speaker, Mapping) and "participant_alias" in speaker
        for scene_number in affected
        for scene in [scenes[scene_number - 1]]
        if isinstance(scene, Mapping)
        for speaker in (scene.get("speaker_participants") or [])
    )


def compile_planner_speaker_repair_spec_v11(
    *, outline: Mapping[str, Any], errors: Sequence[str],
    authoritative_role_table: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not planner_speaker_only_retry_eligible(outline=outline, errors=errors):
        raise ValueError("planner_speaker_only_retry_not_eligible")
    scene_numbers = sorted({
        int(match.group("scene"))
        for error in errors
        for match in [_SPEAKER_ERROR.fullmatch(str(error))]
        if match is not None
    })
    targets = [f"S{value:02d}" for value in scene_numbers]
    target_set = set(targets)
    allowed_roles = [{
        "participant_ref": str(row.get("entity_ref") or ""),
        "display_name": str(row.get("display_name") or ""),
        "entity_type": str(row.get("entity_type") or "person"),
    } for row in authoritative_role_table if row.get("entity_ref")]
    if len(allowed_roles) < 2:
        raise ValueError("planner_speaker_repair_requires_two_authoritative_roles")
    contexts = []
    for scene_number, scene_ref in zip(scene_numbers, targets):
        scene = (outline.get("scenes") or [])[scene_number - 1]
        contexts.append({
            "scene_ref": scene_ref,
            "scene_alias": scene.get("scene_alias"),
            "goal": scene.get("goal"),
            "new_development": scene.get("new_development"),
            "desired_outcome": scene.get("desired_outcome"),
            "source_anchor_id": scene.get("source_anchor_id"),
            "dependencies": deepcopy(scene.get("dependencies") or []),
        })
    result = {
        "repair_version": PLANNER_SPEAKER_REPAIR_VERSION_V11,
        "target_scene_refs": targets,
        "diagnostics": [
            {"scene_ref": f"S{value:02d}",
             "error_code": "speaker_contract_invalid"}
            for value in scene_numbers
        ],
        "target_scene_contexts": contexts,
        "authoritative_role_table": allowed_roles,
        "temporary_creation_slots_are_speakers": False,
        "frozen_outline_sha256": _sha(outline),
        "frozen_non_target_fields_sha256": _sha(
            _without_target_speakers(outline, target_set)),
        "frozen_untouched_speakers_sha256": _sha(
            _untouched_speakers(outline, target_set)),
    }
    result["repair_spec_sha256"] = _sha(result)
    return result


def build_planner_speaker_only_retry_prompt_v11(
    spec: Mapping[str, Any], *, previous_contract_errors: Sequence[str] = (),
) -> str:
    template = {"speaker_repairs": [{
        "scene_ref": scene_ref,
        "speaker_participants": [{
            "participant_ref": "COPY_ONE_EXACT_AUTHORITATIVE_REF",
            "scene_role": "primary",
            "intents": ["one to three scene-specific intents"],
        }, {
            "participant_ref": "COPY_A_DIFFERENT_EXACT_AUTHORITATIVE_REF",
            "scene_role": "supporting",
            "intents": ["one to three contrasting scene-specific intents"],
        }],
    } for scene_ref in spec.get("target_scene_refs") or []]}
    sections = [
        "PLANNER SPEAKER-ONLY REPAIR v11. Return one JSON object only.",
        "The outline, actions, anchors, placements, goals, beats, dependencies, and "
        "outcomes are valid and frozen. Replace speaker_participants only for the named "
        "scenes. Do not return or rewrite any other field.",
        "TARGET DIAGNOSTICS\n" + _canonical(spec.get("diagnostics") or []),
        "FROZEN TARGET SCENE CONTEXT\n" + _canonical(
            spec.get("target_scene_contexts") or []),
        "AUTHORITATIVE SPEAKERS — copy participant_ref exactly; use only this list\n" +
        _canonical(spec.get("authoritative_role_table") or []),
        "RULES\n- Top level has exactly speaker_repairs.\n"
        "- Repair every listed scene_ref exactly once.\n"
        "- Each repaired scene has 2-4 distinct authoritative speakers.\n"
        "- Each speaker has exactly participant_ref, scene_role, intents.\n"
        "- scene_role is primary or supporting.\n"
        "- intents contains 1-3 non-empty scene-specific strings.\n"
        "- Program-owned internal creation identifiers are never speakers.\n"
        "- Do not invent, normalize, or case-change an ID.\n"
        "- Do not return scenes, actions, errors, schema, or allowed-reference metadata.",
    ]
    if previous_contract_errors:
        sections.append("PREVIOUS PATCH CONTRACT ERRORS\n" + _canonical(
            list(previous_contract_errors)))
    sections.extend([
        "FROZEN OUTLINE SHA256\n" + str(spec.get("frozen_outline_sha256") or ""),
        "OUTPUT ONLY THIS EXACT SHAPE; replace placeholders with exact allowed values\n" +
        json.dumps(template, ensure_ascii=False, indent=2),
    ])
    return "\n\n".join(sections)


def validate_planner_speaker_repair_response_v11(
    response: Any, *, spec: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(response, Mapping):
        return {"valid": False, "errors": ["planner_speaker_repair:not_object"]}
    if set(response) != {"speaker_repairs"}:
        errors.append("planner_speaker_repair:top_level_exact_keys")
    repairs = response.get("speaker_repairs")
    repairs = repairs if isinstance(repairs, list) else []
    if not isinstance(response.get("speaker_repairs"), list):
        errors.append("planner_speaker_repair:repairs_not_array")
    allowed = {str(row.get("participant_ref") or "")
               for row in spec.get("authoritative_role_table") or []}
    seen_refs: list[str] = []
    for index, repair in enumerate(repairs, 1):
        path = f"planner_speaker_repair:item:{index}"
        if not isinstance(repair, Mapping) or set(repair) != {
                "scene_ref", "speaker_participants"}:
            errors.append(f"{path}:exact_keys")
            continue
        scene_ref = str(repair.get("scene_ref") or "")
        seen_refs.append(scene_ref)
        if _SCENE_REF.fullmatch(scene_ref) is None:
            errors.append(f"{path}:scene_ref_format")
        speakers = repair.get("speaker_participants")
        if not isinstance(speakers, list) or not 2 <= len(speakers) <= 4:
            errors.append(f"{path}:speaker_count")
            continue
        participant_refs: list[str] = []
        for speaker_index, speaker in enumerate(speakers, 1):
            speaker_path = f"{path}:speaker:{speaker_index}"
            if not isinstance(speaker, Mapping) or set(speaker) != {
                    "participant_ref", "scene_role", "intents"}:
                errors.append(f"{speaker_path}:exact_keys")
                continue
            participant_ref = str(speaker.get("participant_ref") or "")
            participant_refs.append(participant_ref)
            if participant_ref not in allowed:
                errors.append(f"{speaker_path}:non_authoritative:{participant_ref}")
            if speaker.get("scene_role") not in {"primary", "supporting"}:
                errors.append(f"{speaker_path}:scene_role")
            intents = speaker.get("intents")
            if (not isinstance(intents, list) or not 1 <= len(intents) <= 3 or
                    not all(isinstance(value, str) and value.strip()
                            for value in intents)):
                errors.append(f"{speaker_path}:intents")
        if len(participant_refs) != len(set(participant_refs)):
            errors.append(f"{path}:duplicate_participant_ref")
    targets = list(spec.get("target_scene_refs") or [])
    if len(seen_refs) != len(set(seen_refs)):
        errors.append("planner_speaker_repair:duplicate_scene_ref")
    if set(seen_refs) != set(targets) or len(seen_refs) != len(targets):
        errors.append("planner_speaker_repair:target_set_mismatch")
    return {"valid": not errors, "errors": list(dict.fromkeys(errors)),
            "target_scene_refs": targets}


def apply_planner_speaker_repairs_v11(
    *, frozen_outline: Mapping[str, Any], response: Any,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_planner_speaker_repair_response_v11(
        response, spec=spec)
    if not validation["valid"]:
        return {"valid": False, "errors": validation["errors"],
                "reconstructed_outline": None}
    candidate = deepcopy(dict(frozen_outline))
    targets = set(spec.get("target_scene_refs") or [])
    replacements = {str(row["scene_ref"]): deepcopy(
        row["speaker_participants"]) for row in response["speaker_repairs"]}
    for scene_ref, speakers in replacements.items():
        match = _SCENE_REF.fullmatch(scene_ref)
        assert match is not None
        candidate["scenes"][int(match.group(1)) - 1][
            "speaker_participants"] = speakers
    audits = {
        "frozen_outline_sha256_matches": (
            _sha(frozen_outline) == spec.get("frozen_outline_sha256")),
        "non_target_fields_sha256_unchanged": (
            _sha(_without_target_speakers(candidate, targets)) ==
            spec.get("frozen_non_target_fields_sha256")),
        "untouched_speakers_sha256_unchanged": (
            _sha(_untouched_speakers(candidate, targets)) ==
            spec.get("frozen_untouched_speakers_sha256")),
        "target_scene_refs_exact": set(replacements) == targets,
    }
    errors = [f"planner_speaker_repair:{key}"
              for key, valid in audits.items() if not valid]
    return {"valid": not errors, "errors": errors,
            "reconstructed_outline": candidate if not errors else None,
            "freeze_audit": audits, "repair_response_validation": validation}


__all__ = [
    "PLANNER_SPEAKER_REPAIR_PROMPT_VERSION_V11",
    "PLANNER_SPEAKER_REPAIR_VERSION_V11",
    "apply_planner_speaker_repairs_v11",
    "build_planner_speaker_only_retry_prompt_v11",
    "compile_planner_speaker_repair_spec_v11",
    "planner_speaker_only_retry_eligible",
    "validate_planner_speaker_repair_response_v11",
]
