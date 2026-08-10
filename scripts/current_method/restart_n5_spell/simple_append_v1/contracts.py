"""Small Planner and one-request Round Realizer contracts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


PLANNER_SCHEMA_VERSION = "simple_append_planner_v1"
COMPILED_PLAN_VERSION = "simple_append_compiled_plan_v2"
REALIZER_SCHEMA_VERSION = "simple_append_round_realizer_v2"
NORMALIZED_CHILD_VERSION = "simple_append_child_v1"
SPEAKER_ROSTER_CONTRACT_VERSION = "simple_append_speaker_roster_v1"

# The round-level cap remains the primary context/safety guard.  A scene may
# naturally carry more than the old 24-turn per-scene limit when the total
# round remains bounded; this is intentionally a simple-append-only policy.
SCENE_HARD_MINIMUM = 9
SCENE_HARD_MAXIMUM = 48

_PLANNER_TOP_KEYS = {"planner_schema_version", "round_goal", "scenes"}
_PLANNER_SCENE_KEYS = {
    "goal", "participants", "new_development", "interaction_beats",
    "transition_from_previous", "desired_outcome",
}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_planner_output(
    value: Any, *, parent_speakers: Sequence[str],
) -> dict[str, Any]:
    """Validate a semantic outline; all IDs and ordering are program-owned."""
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != _PLANNER_TOP_KEYS:
        return {"valid": False, "errors": ["planner_top_level_schema"]}
    if value.get("planner_schema_version") != PLANNER_SCHEMA_VERSION:
        errors.append("planner_schema_version")
    if not _text(value.get("round_goal")):
        errors.append("planner_round_goal")
    scenes = value.get("scenes")
    if not isinstance(scenes, list) or not 2 <= len(scenes) <= 3:
        errors.append("planner_scene_count_2_to_3")
        scenes = []
    known = set(str(value) for value in parent_speakers)
    developments: set[str] = set()
    for index, scene in enumerate(scenes, 1):
        prefix = f"scene_{index:02d}"
        if not isinstance(scene, Mapping) or set(scene) != _PLANNER_SCENE_KEYS:
            errors.append(f"{prefix}:schema")
            continue
        for key in ("goal", "new_development", "transition_from_previous",
                    "desired_outcome"):
            if not _text(scene.get(key)):
                errors.append(f"{prefix}:{key}")
        participants = scene.get("participants")
        if (not isinstance(participants, list) or
                not 2 <= len(participants) <= 5 or
                any(not _text(row) for row in participants) or
                len(participants) != len(set(participants))):
            errors.append(f"{prefix}:participants")
        elif not known.intersection(str(row) for row in participants):
            errors.append(f"{prefix}:not_connected_to_parent_speaker")
        beats = scene.get("interaction_beats")
        if (not isinstance(beats, list) or not 3 <= len(beats) <= 7 or
                any(not _text(row) for row in beats)):
            errors.append(f"{prefix}:interaction_beats")
        development = " ".join(str(scene.get("new_development") or "").split()).casefold()
        if development in developments:
            errors.append(f"{prefix}:duplicate_new_development")
        developments.add(development)
    return {"valid": not errors, "errors": errors,
            "scene_count": len(scenes)}


def compile_round_plan(
    planner_output: Mapping[str, Any], *, tail_context: Mapping[str, Any],
    round_index: int = 1,
) -> dict[str, Any]:
    validation = validate_planner_output(
        planner_output, parent_speakers=tail_context["parent_speakers"])
    if not validation["valid"]:
        raise ValueError("simple_append_planner_invalid:" +
                         ",".join(validation["errors"]))
    scene_count = len(planner_output["scenes"])
    target_per_scene = 18 if scene_count == 2 else 16
    scenes: list[dict[str, Any]] = []
    for index, raw in enumerate(planner_output["scenes"], 1):
        scene_ref = f"R{round_index:02d}:S{index:02d}"
        scenes.append({
            "scene_ref": scene_ref,
            "sequence_index": index,
            "predecessor_scene_ref": scenes[-1]["scene_ref"] if scenes else None,
            **deepcopy(dict(raw)),
            "turn_guidance": {
                "soft_target": target_per_scene,
                "soft_range": [14, 18] if scene_count == 3 else [16, 22],
                "hard_minimum": SCENE_HARD_MINIMUM,
                "hard_maximum": SCENE_HARD_MAXIMUM,
            },
        })
    round_speakers = list(dict.fromkeys(
        str(participant)
        for scene in scenes
        for participant in scene["participants"]
    ))
    speaker_roster_contract = {
        "speaker_roster_contract_version": SPEAKER_ROSTER_CONTRACT_VERSION,
        "round_speakers": round_speakers,
        "scene_core_speakers": {
            str(scene["scene_ref"]): list(scene["participants"])
            for scene in scenes
        },
    }
    result = {
        "compiled_plan_version": COMPILED_PLAN_VERSION,
        "root_id": str(tail_context["root_id"]),
        "round_index": round_index,
        "append_mode": "parent_tail_only",
        "source_parent_tail_turn_id": str(tail_context["parent_tail_turn_id"]),
        "source_session_ids": list(tail_context["selected_session_ids"]),
        "round_goal": str(planner_output["round_goal"]),
        "speaker_roster_contract": speaker_roster_contract,
        "round_turn_guidance": {
            "soft_target": 48 if scene_count == 3 else 36,
            "soft_range": [42, 54] if scene_count == 3 else [32, 44],
            "hard_minimum": 9 * scene_count,
            "hard_maximum": min(66, 24 * scene_count),
        },
        "scenes": scenes,
    }
    result["compiled_plan_sha256"] = _sha(result)
    return result


def normalize_round_realizer_output(
    value: Any, *, compiled_plan: Mapping[str, Any],
    parent_utterances: Sequence[str],
    parent_turn_refs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate one round response and assign stable child turn references."""
    errors: list[str] = []
    warnings: list[str] = []
    if (not isinstance(value, Mapping) or
            set(value) != {"realizer_schema_version", "scene_dialogues"}):
        return {"valid": False, "errors": ["realizer_top_level_schema"],
                "warnings": []}
    if value.get("realizer_schema_version") != REALIZER_SCHEMA_VERSION:
        errors.append("realizer_schema_version")
    expected = list(compiled_plan.get("scenes") or [])
    expected_refs = [str(row["scene_ref"]) for row in expected]
    # Per-scene participants are the intended focus. Their program-frozen
    # round union is the hard speaking boundary, so a planned character can
    # enter an adjacent scene without admitting an undeclared speaker.
    derived_scene_rosters = {
        str(scene["scene_ref"]): [str(row) for row in scene.get("participants") or []]
        for scene in expected
    }
    derived_round_roster = list(dict.fromkeys(
        str(participant)
        for scene in expected
        for participant in scene.get("participants") or []
    ))
    roster_contract = compiled_plan.get("speaker_roster_contract")
    if roster_contract is None:
        # Explicit compatibility for historical compiled plans created before
        # the frozen roster was serialized.
        round_speaker_roster = set(derived_round_roster)
    elif (not isinstance(roster_contract, Mapping) or
          roster_contract.get("speaker_roster_contract_version") !=
          SPEAKER_ROSTER_CONTRACT_VERSION or
          roster_contract.get("round_speakers") != derived_round_roster or
          roster_contract.get("scene_core_speakers") != derived_scene_rosters):
        errors.append("compiled_speaker_roster_contract")
        round_speaker_roster = set(derived_round_roster)
    else:
        round_speaker_roster = set(str(row) for row in roster_contract["round_speakers"])
    raw_dialogues = value.get("scene_dialogues")
    if not isinstance(raw_dialogues, Mapping):
        errors.append("realizer_scene_dialogues")
        raw_dialogues = {}
    actual_refs = {str(ref) for ref in raw_dialogues}
    if actual_refs != set(expected_refs):
        errors.append("realizer_scene_ref_set")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for scene_ref in expected_refs:
        raw_turns = raw_dialogues.get(scene_ref)
        if not isinstance(raw_turns, list):
            errors.append(f"{scene_ref}:dialogue_not_array")
            raw_turns = []
        grouped[scene_ref] = raw_turns
    if parent_turn_refs is not None and len(parent_turn_refs) != len(parent_utterances):
        raise ValueError("simple_append_parent_turn_ref_alignment")
    parent_exact: dict[str, list[str]] = {}
    for index, text in enumerate(parent_utterances):
        if not _text(text):
            continue
        canonical = " ".join(str(text).split()).casefold()
        source_ref = (str(parent_turn_refs[index]) if parent_turn_refs is not None
                      else f"parent_utterance_{index + 1:06d}")
        parent_exact.setdefault(canonical, []).append(source_ref)
    seen_text: dict[str, str] = {}
    cross_scene_exact_duplicates: list[dict[str, str]] = []
    parent_exact_copies: list[dict[str, Any]] = []
    normalized_scenes: list[dict[str, Any]] = []
    total_turns = 0
    long_scene_count = 0
    for index, planned in enumerate(expected, 1):
        scene_ref = str(planned["scene_ref"])
        prefix = f"scene_{index:02d}"
        turns = grouped[scene_ref]
        if not turns:
            errors.append(f"{prefix}:dialogue_turns")
            continue
        hard_minimum = int(planned["turn_guidance"]["hard_minimum"])
        hard_maximum = int(planned["turn_guidance"]["hard_maximum"])
        if len(turns) < hard_minimum or len(turns) > hard_maximum:
            errors.append(f"{prefix}:hard_turn_guard")
        if len(turns) >= 12:
            long_scene_count += 1
        planned_speakers = set(str(row) for row in planned["participants"])
        scene_speakers: set[str] = set()
        scene_seen_text: set[str] = set()
        normalized_turns: list[dict[str, Any]] = []
        for turn_index, turn in enumerate(turns, 1):
            if (not isinstance(turn, Mapping) or
                    set(turn) != {"speaker", "text"}):
                errors.append(f"{prefix}:turn_{turn_index:03d}:schema")
                continue
            speaker, text = turn.get("speaker"), turn.get("text")
            speaker_name = str(speaker) if _text(speaker) else ""
            if not speaker_name or speaker_name not in round_speaker_roster:
                errors.append(f"{prefix}:turn_{turn_index:03d}:speaker")
            if not _text(text):
                errors.append(f"{prefix}:turn_{turn_index:03d}:text")
                continue
            canonical = " ".join(str(text).split()).casefold()
            turn_ref = f"{scene_ref}:T{turn_index:03d}"
            if canonical in parent_exact:
                parent_exact_copies.append({
                    "duplicate_turn_ref": turn_ref,
                    "source_turn_refs": list(parent_exact[canonical]),
                })
            if canonical in scene_seen_text:
                errors.append(f"{prefix}:turn_{turn_index:03d}:child_exact_duplicate")
            elif canonical in seen_text:
                first_turn_ref = seen_text[canonical]
                warnings.append(
                    f"{prefix}:turn_{turn_index:03d}:cross_scene_exact_duplicate:"
                    f"{first_turn_ref}")
                cross_scene_exact_duplicates.append({
                    "first_turn_ref": first_turn_ref,
                    "duplicate_turn_ref": turn_ref,
                })
            else:
                seen_text[canonical] = turn_ref
            scene_seen_text.add(canonical)
            scene_speakers.add(speaker_name)
            normalized_turns.append({
                "turn_ref": turn_ref,
                "speaker": speaker_name,
                "text": str(text).strip(),
            })
        if len(scene_speakers) < 2:
            errors.append(f"{prefix}:speaker_coverage")
        if not planned_speakers.intersection(scene_speakers):
            errors.append(f"{prefix}:core_speaker_coverage")
        for borrowed in sorted(scene_speakers - planned_speakers):
            warnings.append(
                f"{prefix}:round_roster_speaker_outside_core:{borrowed}")
        total_turns += len(normalized_turns)
        normalized_scenes.append({
            "scene_ref": scene_ref,
            "session_ref": f"{scene_ref}:SESSION",
            "sequence_index": index,
            "goal": str(planned["goal"]),
            "dialogue_turns": normalized_turns,
        })
    hard_round = (compiled_plan.get("round_turn_guidance") or {})
    if total_turns < int(hard_round.get("hard_minimum") or 0):
        errors.append("realizer_round_too_short")
    if total_turns > int(hard_round.get("hard_maximum") or 10**9):
        errors.append("realizer_round_too_long")
    parent_copy_count = len(parent_exact_copies)
    parent_copy_ratio = parent_copy_count / total_turns if total_turns else 0.0
    parent_copy_excessive = (
        parent_copy_count >= 3 or
        (parent_copy_count >= 2 and parent_copy_ratio >= 0.10)
    )
    if parent_copy_excessive:
        errors.append("realizer_parent_exact_copy_excessive")
    else:
        for row in parent_exact_copies:
            warnings.append(
                "parent_exact_copy_warning:"
                f"{row['duplicate_turn_ref']}:"
                f"{','.join(row['source_turn_refs'])}")
    required_long = min(2, len(expected))
    if long_scene_count < required_long:
        warnings.append("long_scene_target_not_met")
    soft_range = list(hard_round.get("soft_range") or [])
    if (len(soft_range) != 2 or
            not int(soft_range[0]) <= total_turns <= int(soft_range[1])):
        warnings.append("round_soft_turn_target_not_met")
    child = {
        "child_schema_version": NORMALIZED_CHILD_VERSION,
        "root_id": str(compiled_plan["root_id"]),
        "round_index": int(compiled_plan["round_index"]),
        "append_mode": "parent_tail_only",
        "source_parent_tail_turn_id": str(
            compiled_plan.get("source_parent_tail_turn_id") or ""),
        "scenes": normalized_scenes,
        "scene_count": len(normalized_scenes),
        "turn_count": total_turns,
        "long_scene_count": long_scene_count,
    }
    child["child_sha256"] = _sha(child)
    return {"valid": not errors, "errors": errors, "warnings": warnings,
            "child": child, "turn_count": total_turns,
            "long_scene_count": long_scene_count,
            "duplicate_policy_version": "simple_append_exact_duplicate_v2",
            "parent_exact_copy_count": parent_copy_count,
            "parent_exact_copy_ratio": parent_copy_ratio,
            "parent_exact_copy_excessive": parent_copy_excessive,
            "parent_exact_copies": parent_exact_copies,
            "cross_scene_exact_duplicate_count": len(
                cross_scene_exact_duplicates),
            "cross_scene_exact_duplicates": cross_scene_exact_duplicates}


def shadow_project_flat_realizer_v1(value: Any) -> dict[str, Any]:
    """Group historical v1 flat turns for audit only, never live admission."""
    if (not isinstance(value, Mapping) or
            value.get("realizer_schema_version") !=
            "simple_append_round_realizer_v1" or
            not isinstance(value.get("dialogue_turns"), list)):
        raise ValueError("simple_append_v1_shadow_input_invalid")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, turn in enumerate(value["dialogue_turns"], 1):
        if not isinstance(turn, Mapping):
            raise ValueError(f"simple_append_v1_shadow_turn_invalid:{index}")
        scene_ref = str(turn.get("scene_ref") or "")
        speaker = turn.get("speaker")
        text = turn.get("text")
        if not scene_ref or not _text(speaker) or not _text(text):
            raise ValueError(f"simple_append_v1_shadow_turn_contract:{index}")
        grouped.setdefault(scene_ref, []).append({
            "speaker": str(speaker), "text": str(text),
        })
    projection = {
        "realizer_schema_version": REALIZER_SCHEMA_VERSION,
        "scene_dialogues": grouped,
    }
    return {
        "projection": projection,
        "shadow_projection_only": True,
        "authoritative_admission_allowed": False,
    }
