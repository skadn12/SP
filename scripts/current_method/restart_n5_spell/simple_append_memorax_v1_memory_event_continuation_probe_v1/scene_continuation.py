"""Mechanical continuation packaging for partially realized Child scenes.

This module deliberately does not decide whether a continuation is good.  The
existing Adherence Auditor supplies the semantic statuses; this layer selects
eligible scenes, gives a future Realizer only those scenes, and materializes
new turn references deterministically.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping

from .contracts import is_transport_speaker_label


REALIZATION_STATUSES = frozenset({"realized", "partial", "not_realized"})
PARTICIPANT_STATUSES = frozenset({"pass", "fail"})


def _scene_ref(scene: Mapping[str, Any], index: int, *, round_index: int) -> str:
    return str(scene.get("scene_ref") or
               f"R{int(round_index):02d}:S{index:02d}")


def _turn_view(turn: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only source turn content; the model never owns canonical refs."""
    return {
        "speaker": str(turn.get("speaker") or ""),
        "text": str(turn.get("text") or ""),
    }


def build_scene_continuation_request(
    *, plan: Mapping[str, Any], child: Mapping[str, Any],
    adherence_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a request containing only scenes eligible for appended completion.

    A participant failure is a hard semantic failure and is excluded.  A scene
    whose participants are correct but whose development or durable outcome is
    partial/not-realized is eligible for continuation.  Fully realized scenes
    are frozen and omitted.
    """
    plan_scenes = plan.get("scenes")
    child_scenes = child.get("scenes")
    audit_scenes = adherence_audit.get("scenes")
    if not isinstance(plan_scenes, list) or not isinstance(child_scenes, list):
        raise ValueError("scene_continuation_scene_lists_required")
    if not isinstance(audit_scenes, list):
        raise ValueError("scene_continuation_audit_required")
    if not (len(plan_scenes) == len(child_scenes) == len(audit_scenes)):
        raise ValueError("scene_continuation_scene_count_mismatch")
    try:
        round_index = int(child.get("round_index") or 1)
    except (TypeError, ValueError) as exc:
        raise ValueError("scene_continuation_round_index_invalid") from exc
    if round_index < 1:
        raise ValueError("scene_continuation_round_index_invalid")

    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for index, (plan_scene, child_scene, audit_scene) in enumerate(
            zip(plan_scenes, child_scenes, audit_scenes), 1):
        if not all(isinstance(value, Mapping)
                   for value in (plan_scene, child_scene, audit_scene)):
            raise ValueError(f"scene_continuation_scene_{index}_object")
        ref = _scene_ref(child_scene, index, round_index=round_index)
        participant = str(audit_scene.get("participant_adherence") or "")
        development = str(audit_scene.get("development_status") or "")
        outcome = str(audit_scene.get("durable_outcome_status") or "")
        if participant not in PARTICIPANT_STATUSES:
            raise ValueError(f"scene_continuation_scene_{index}_participant_status")
        if development not in REALIZATION_STATUSES:
            raise ValueError(f"scene_continuation_scene_{index}_development_status")
        if outcome not in REALIZATION_STATUSES:
            raise ValueError(f"scene_continuation_scene_{index}_outcome_status")

        row = {
            "scene_ref": ref,
            "memory_event_id": str(child_scene.get("memory_event_id") or
                                   plan_scene.get("memory_event_id") or ""),
            "required_participants": list(
                plan_scene.get("required_participants") or
                plan_scene.get("participants") or []),
            "selected_plan": {
                key: deepcopy(plan_scene.get(key))
                for key in (
                    "memory_event_id", "goal", "before_state",
                    "required_transition", "after_state", "new_development",
                    "durable_outcome", "actor_bindings", "interaction_beats",
                )
            },
            "actual_scene_turns": [
                _turn_view(turn) for turn in child_scene.get("dialogue_turns") or []
                if isinstance(turn, Mapping)
            ],
            "adherence": {
                "participant_adherence": participant,
                "development_status": development,
                "durable_outcome_status": outcome,
                "development_evidence_turn_refs": list(
                    audit_scene.get("development_evidence_turn_refs") or []),
                "durable_outcome_evidence_turn_refs": list(
                    audit_scene.get("durable_outcome_evidence_turn_refs") or []),
                "reason": str(audit_scene.get("reason") or ""),
            },
            "completion_instruction": (
                "Continue only from the actual last state in this scene. "
                "Do not replay completed beats. Realize the remaining "
                "required transition and durable outcome concretely."
            ),
        }
        if participant == "pass" and (
                development != "realized" or outcome != "realized"):
            eligible.append(row)
        else:
            excluded.append({
                "scene_ref": ref,
                "reason": (
                    "participant_failure" if participant == "fail" else
                    "already_fully_realized"),
                "participant_adherence": participant,
                "development_status": development,
                "durable_outcome_status": outcome,
            })
    return {
        "version": "scene_continuation_request_v1",
        "round_index": round_index,
        "eligible_scene_count": len(eligible),
        "eligible_scenes": eligible,
        "excluded_scenes": excluded,
    }


def materialize_scene_continuation_output(
    raw: Any, *, request: Mapping[str, Any], child: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a continuation response to eligible scenes and create turn refs."""
    errors: list[str] = []
    if not isinstance(raw, Mapping) or set(raw) != {"continuations"}:
        return {"valid": False, "errors": ["scene_continuation_top_level"],
                "continuations": []}
    rows = raw.get("continuations")
    eligible = request.get("eligible_scenes")
    if not isinstance(rows, list) or not isinstance(eligible, list):
        return {"valid": False, "errors": ["scene_continuation_rows_shape"],
                "continuations": []}
    normalizations: list[str] = []
    expected_count = len(eligible)
    if (expected_count > 0 and len(rows) > expected_count
            and len(rows) % expected_count == 0
            and all(row == rows[index % expected_count]
                    for index, row in enumerate(rows))):
        rows = rows[:expected_count]
        normalizations.append("collapsed_exact_repeated_continuation_block")
    if len(rows) != len(eligible):
        errors.append("scene_continuation_count_mismatch")
    materialized: list[dict[str, Any]] = []
    for index, (raw_row, source) in enumerate(zip(rows, eligible), 1):
        prefix = f"continuation_{index}"
        if not isinstance(raw_row, Mapping) or set(raw_row) != {"dialogue_turns"}:
            errors.append(f"{prefix}_shape")
            continue
        raw_turns = raw_row.get("dialogue_turns")
        if not isinstance(raw_turns, list) or not raw_turns:
            errors.append(f"{prefix}_turns")
            continue
        source_turns = list(source.get("actual_scene_turns") or [])
        scene_ref = str(source.get("scene_ref") or "")
        source_child_scene = next((
            scene for scene in child.get("scenes") or []
            if isinstance(scene, Mapping) and
            str(scene.get("scene_ref") or "") == scene_ref
        ), None)
        existing_count = (len(source_child_scene.get("dialogue_turns") or [])
                          if isinstance(source_child_scene, Mapping)
                          else len(source_turns))
        actual_speakers = {
            str(turn.get("speaker") or "").strip()
            for turn in source_turns
            if isinstance(turn, Mapping) and
            str(turn.get("speaker") or "").strip()
        }
        turns: list[dict[str, Any]] = []
        for turn_index, raw_turn in enumerate(raw_turns, existing_count + 1):
            if not isinstance(raw_turn, Mapping) or set(raw_turn) != {"speaker", "text"}:
                errors.append(f"{prefix}_turn_{turn_index}_shape")
                continue
            speaker = str(raw_turn.get("speaker") or "").strip()
            text = str(raw_turn.get("text") or "").strip()
            if not speaker or is_transport_speaker_label(speaker):
                errors.append(f"{prefix}_turn_{turn_index}_speaker")
            else:
                actual_speakers.add(speaker)
            if not text:
                errors.append(f"{prefix}_turn_{turn_index}_text")
            turns.append({
                "turn_ref": f"{scene_ref}:T{turn_index:03d}",
                "speaker": speaker,
                "text": text,
            })
        required_participants = {
            str(value).strip()
            for value in (
                source.get("required_participants") or
                source.get("participants") or [])
            if str(value).strip()
        }
        for participant in sorted(required_participants - actual_speakers):
            errors.append(
                f"{prefix}_required_participant_missing:{participant}")
        materialized.append({
            "scene_ref": scene_ref,
            "memory_event_id": source.get("memory_event_id"),
            "dialogue_turns": turns,
        })
    return {"valid": not errors, "errors": errors,
            "normalizations": normalizations,
            "continuations": materialized}


def build_scene_continuation_prompt(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a compact prompt for one bounded failed-scene continuation call."""
    scenes = list(request.get("eligible_scenes") or [])
    payload = [{
        "required_participants": list(
            scene.get("required_participants") or
            scene.get("participants") or []),
        "selected_plan": deepcopy(scene.get("selected_plan") or {}),
        "actual_scene_turns": deepcopy(scene.get("actual_scene_turns") or []),
        "adherence": deepcopy(scene.get("adherence") or {}),
        "completion_instruction": scene.get("completion_instruction"),
    } for scene in scenes]
    text = f"""You are the Failed-Scene Continuation Realizer. Continue only
the incomplete scenes below. Each scene already happened exactly as shown;
never rewrite, summarize, restart, or replay those turns. Begin immediately
after its final actual turn and add only the interaction needed to complete the
remaining required_transition and after_state.

    Required participants must remain covered across the existing scene.
    The list is not a speaker allowlist: auxiliary existing people or a
    scene-local passerby may speak, but must not replace a bound actor.
    Preserve actor bindings. A planned call, promise, intention, preparation, or discussion is not a completed
action or durable result. Make the missing transition and outcome explicitly
true in the appended dialogue. Do not create a new topic or alter a scene that
is not present here. Prefer a concise 4-10 turn continuation; semantic
completion matters more than reaching a turn quota.

There are exactly {len(payload)} input scenes. Return exactly {len(payload)}
continuation objects in the same order. Do not output scene IDs, turn IDs,
Memory IDs, status labels, explanations, or any fields other than
dialogue_turns. Every dialogue turn has exactly speaker and text.

INCOMPLETE SCENES
{json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)}

OUTPUT ONLY JSON
{{"continuations":[
  {{"dialogue_turns":[{{"speaker":"...","text":"..."}}]}}
]}}
The single row above illustrates shape only. The actual continuations array
must contain exactly {len(payload)} objects."""
    return {
        "prompt_version": "failed_scene_continuation_realizer_prompt_v1",
        "prompt": text,
        "eligible_scene_count": len(payload),
        "full_parent_included": False,
        "memory_included": False,
    }


def merge_scene_continuations(
    child: Mapping[str, Any], materialized: Mapping[str, Any],
) -> dict[str, Any]:
    """Append continuation turns to their original scenes without rewriting."""
    if not materialized.get("valid"):
        raise ValueError("scene_continuation_materialization_invalid")
    result = deepcopy(dict(child))
    scenes = result.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("scene_continuation_child_scenes_required")
    by_ref = {
        str(row.get("scene_ref")): row
        for row in materialized.get("continuations") or []
        if isinstance(row, Mapping)
    }
    for scene in scenes:
        ref = str(scene.get("scene_ref") or "")
        addition = by_ref.get(ref)
        if addition is None:
            continue
        if str(scene.get("memory_event_id") or "") != str(
                addition.get("memory_event_id") or ""):
            raise ValueError(f"scene_continuation_event_mismatch:{ref}")
        scene.setdefault("dialogue_turns", []).extend(
            deepcopy(addition.get("dialogue_turns") or []))
    return result
