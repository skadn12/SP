"""Mechanical contract for semantic Realizer-adherence judgments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


PARTICIPANT_STATUSES = {"pass", "fail"}
REALIZATION_STATUSES = {"realized", "partial", "not_realized"}
MAX_EVIDENCE_POSITIONS = 4

_JUDGMENT_KEYS = {
    "participant_adherence",
    "participant_evidence_positions",
    "development_status",
    "development_evidence_positions",
    "durable_outcome_status",
    "durable_outcome_evidence_positions",
    "reason",
}


def build_adherence_auditor_input(
    *, plan: Mapping[str, Any], child: Mapping[str, Any],
) -> dict[str, Any]:
    """Align frozen plans and Child scenes without asking a model for IDs."""
    plan_scenes = plan.get("scenes")
    child_scenes = child.get("scenes")
    if not isinstance(plan_scenes, list) or not isinstance(child_scenes, list):
        raise ValueError("adherence_input_scene_lists_required")
    if len(plan_scenes) != len(child_scenes):
        raise ValueError("adherence_input_scene_count_mismatch")

    scenes: list[dict[str, Any]] = []
    for index, (plan_scene, child_scene) in enumerate(
            zip(plan_scenes, child_scenes), 1):
        if not isinstance(plan_scene, Mapping) or not isinstance(
                child_scene, Mapping):
            raise ValueError(f"adherence_input_scene_{index}_object")
        plan_event = str(plan_scene.get("memory_event_id") or "")
        child_event = str(child_scene.get("memory_event_id") or "")
        if not plan_event or plan_event != child_event:
            raise ValueError(f"adherence_input_scene_{index}_event_mismatch")
        turns = child_scene.get("dialogue_turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"adherence_input_scene_{index}_turns")
        dialogue = []
        for position, turn in enumerate(turns, 1):
            if not isinstance(turn, Mapping):
                raise ValueError(
                    f"adherence_input_scene_{index}_turn_{position}_object")
            dialogue.append({
                "position": position,
                "speaker": str(turn.get("speaker") or ""),
                "text": str(turn.get("text") or ""),
            })
        required_participants = list(
            plan_scene.get("required_participants") or
            plan_scene.get("participants") or [])
        scenes.append({
            "scene_index": index,
            "required_participants": required_participants,
            "actual_speakers": list(dict.fromkeys(
                str(turn.get("speaker") or "")
                for turn in turns if isinstance(turn, Mapping))),
            "selected_actor_bindings": list(
                plan_scene.get("actor_bindings") or []),
            "verified_current_state": str(
                plan_scene.get("verified_current_state") or ""),
            "selected_new_development": str(
                plan_scene.get("new_development") or ""),
            "selected_durable_outcome": str(
                plan_scene.get("durable_outcome") or ""),
            "selected_interaction_beats": list(
                plan_scene.get("interaction_beats") or []),
            "child_dialogue": dialogue,
        })
    result = {
        "adherence_auditor_input_version": (
            "realizer_adherence_input_v3_required_participants"),
        "scenes": scenes,
    }
    if plan.get("development_policy") == "development_gain_v2":
        result["development_policy"] = "development_gain_v2"
        for row, scene in zip(scenes, plan_scenes):
            row["expected_information_not_realized_evidence"] = scene.get("expected_information")
            row["exact_target_after_state"] = scene.get("after_state")
            row["depends_on_scene_refs"] = list(scene.get("depends_on_scene_refs") or [])
    return result


def build_adherence_auditor_prompt(
    *, auditor_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the future semantic-only Auditor prompt; this function calls no API."""
    scene_count = len(auditor_input.get("scenes") or [])
    payload = json.dumps(
        auditor_input, ensure_ascii=False, sort_keys=True, indent=2)
    prompt = f"""You are a Realizer Adherence Auditor. Judge whether each Child
scene faithfully realizes its frozen selected plan. Judge scenes in input order.

For participant_adherence, return pass only when the participant who performs
the selected action remains the same. Use selected_actor_bindings as the
authoritative assignment of action ownership. Exchanging the actor or subject
is fail. required_participants must appear, but it is not a speaker allowlist.
An additional speaker is allowed when auxiliary and does not take over a bound
core action. Do not call it actor drift merely because another person receives,
accepts, assists, or responds to the assigned actor's action.

For an introduction, connection, or handoff, a formal sentence such as
"A, this is B" is not required. Treat the action as realized when the assigned
actor demonstrably brokers the contact or transfers the relevant information,
and the intended parties then directly interact in the planned context. Mere
co-presence, an unexplained speaker appearance, or unrelated interaction is not
enough.

For development_status and durable_outcome_status, use only realized, partial,
or not_realized. Judge the exact selected state, not a later state that was
never requested. If the selected outcome is an agreement, decision, awareness,
assignment, or scheduled arrangement, explicit dialogue that reaches that
coordination state is realized; do not require the future meeting, event, or
implementation to have happened. For example, explicitly agreeing to integrate
an idea realizes "A agrees to integrate the idea", and agreeing on Thursday at
5 PM realizes "the meeting is scheduled". If the selected outcome is an actual
completed action or result, mere discussion, preparation, intention, or the
beginning of that action is not completion. An explicit resultative statement
or completed action is sufficient evidence and needs no redundant confirmation.
Use partial only when the dialogue clearly leaves the selected outcome
unfinished (for example, "I will write it later", "I started but stopped", or
"we may meet"). Scan the entire scene before deciding. Cite the strongest
decisive 1-based positions from the corresponding child_dialogue, including
later positions when they establish the selected state. Do not default to the
first turns or reuse the same positions across scenes without textual support.
When a selected development asks a participant to describe, explain, recall, or
provide details, the development is realized once the dialogue contains those
concrete details. Do not require an extra reaction, agreement, or later event
unless that is part of the selected development or durable outcome. In
particular, do not interpret wording such as "to continue the narrative" as a
requirement for another scene after the requested description has occurred.
Do not output scene IDs, turn IDs, final verdicts, or repairs.

There are exactly {scene_count} input scenes. Return exactly {scene_count}
objects in `scene_audits`, in the same order as the input scenes. Do not omit,
merge, or add a scene. Even when a scene is uncertain or fails, return its
object with the allowed status and a reason.

Every object must contain exactly these fields:
- participant_adherence: pass or fail
- participant_evidence_positions: 1-4 decisive scene-local integers
- development_status: realized, partial, or not_realized
- development_evidence_positions: 1-4 decisive scene-local integers; it may
  be empty only for not_realized
- durable_outcome_status: realized, partial, or not_realized
- durable_outcome_evidence_positions: 1-4 decisive scene-local integers; it
  may be empty only for not_realized
- reason: concise semantic explanation

No semantic verdict or evidence position is prefilled for you. Derive every
value independently from the corresponding selected plan and full scene.

AUDITOR INPUT
{payload}

OUTPUT ONLY VALID JSON. The top-level object must have exactly one key,
`scene_audits`, whose value is an array of exactly {scene_count} objects using
the fields above."""
    gain_policy = auditor_input.get("development_policy") == "development_gain_v2"
    if gain_policy:
        prompt += """\nDEVELOPMENT GAIN v2: The expected information is a planning target only.
Determine actual realization from Child dialogue evidence. Preserve the exact
target_after_state: 'clearer understanding' does not require full understanding,
an agreed appointment does not require the appointment to have occurred, and
completed preparation does not prove submission. A new arrangement or specific
new fact can be realized without a later action. Mere reassurances or discussion
cannot substitute for a selected concrete result. A target dependent on prior
work must not be treated as realized merely because the plan assumes that work.
Cite exact scene positions establishing the target, not the plan or Memory.
"""
    return {
        "prompt_version": (
            "realizer_adherence_auditor_prompt_v7_concise_description" +
            (":development_gain_v2" if gain_policy else "")),
        "prompt": prompt,
        "scene_count": scene_count,
    }


def _positions_to_refs(
    *, positions: Any, turns: list[Any], prefix: str, errors: list[str],
    warnings: list[str], allow_empty: bool = False,
) -> list[str]:
    if not isinstance(positions, list):
        errors.append(f"{prefix}_shape")
        return []
    if not positions:
        if not allow_empty:
            errors.append(f"{prefix}_shape")
        return []
    if len(positions) > MAX_EVIDENCE_POSITIONS:
        # Evidence is provenance metadata, not the semantic verdict. Keep the
        # contract bounded for downstream artifacts while preserving an audit
        # warning instead of rejecting an otherwise well-formed judgment.
        warnings.append(
            f"{prefix}_too_many:{len(positions)}>{MAX_EVIDENCE_POSITIONS}")
    if any(not isinstance(value, int) or isinstance(value, bool)
           for value in positions):
        errors.append(f"{prefix}_integer")
        return []
    if len(positions) != len(set(positions)):
        errors.append(f"{prefix}_duplicate")
    refs: list[str] = []
    for position in positions:
        if not 1 <= position <= len(turns):
            errors.append(f"{prefix}_out_of_range:{position}")
            continue
        turn = turns[position - 1]
        ref = str(turn.get("turn_ref") or "") if isinstance(
            turn, Mapping) else ""
        if not ref:
            errors.append(f"{prefix}_missing_canonical_ref:{position}")
            continue
        refs.append(ref)
    return refs[:MAX_EVIDENCE_POSITIONS]


def validate_and_materialize_adherence_audit(
    value: Any, *, plan: Mapping[str, Any], child: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate semantic judgments and deterministically build the final audit."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(value, Mapping) or set(value) != {"scene_audits"}:
        return {
            "valid": False,
            "errors": ["adherence_audit_top_level_shape"],
            "audit": None,
        }
    raw_audits = value.get("scene_audits")
    plan_scenes = plan.get("scenes")
    child_scenes = child.get("scenes")
    if not isinstance(plan_scenes, list) or not isinstance(child_scenes, list):
        return {
            "valid": False,
            "errors": ["adherence_audit_source_scene_shape"],
            "audit": None,
        }
    if len(plan_scenes) != len(child_scenes):
        errors.append("adherence_audit_source_scene_count")
    expected_count = min(len(plan_scenes), len(child_scenes))
    if not isinstance(raw_audits, list) or len(raw_audits) != expected_count:
        errors.append("adherence_audit_scene_count")
        raw_audits = raw_audits if isinstance(raw_audits, list) else []

    materialized: list[dict[str, Any]] = []
    for index, (plan_scene, child_scene, judgment) in enumerate(
            zip(plan_scenes, child_scenes, raw_audits), 1):
        prefix = f"adherence_scene_{index}"
        if not isinstance(plan_scene, Mapping) or not isinstance(
                child_scene, Mapping):
            errors.append(f"{prefix}_source_object")
            continue
        if str(plan_scene.get("memory_event_id") or "") != str(
                child_scene.get("memory_event_id") or ""):
            errors.append(f"{prefix}_event_mismatch")
        if not isinstance(judgment, Mapping):
            errors.append(f"{prefix}_judgment_object")
            continue
        if set(judgment) != _JUDGMENT_KEYS:
            errors.append(f"{prefix}_judgment_shape")
        participant = str(judgment.get("participant_adherence") or "")
        development = str(judgment.get("development_status") or "")
        outcome = str(judgment.get("durable_outcome_status") or "")
        reason = str(judgment.get("reason") or "").strip()
        if participant not in PARTICIPANT_STATUSES:
            errors.append(f"{prefix}_participant_status")
        if development not in REALIZATION_STATUSES:
            errors.append(f"{prefix}_development_status")
        if outcome not in REALIZATION_STATUSES:
            errors.append(f"{prefix}_durable_outcome_status")
        if not reason:
            errors.append(f"{prefix}_reason")
        turns = child_scene.get("dialogue_turns")
        turns = turns if isinstance(turns, list) else []
        participant_refs = _positions_to_refs(
            positions=judgment.get("participant_evidence_positions"),
            turns=turns,
            prefix=f"{prefix}_participant_evidence_positions",
            errors=errors,
            warnings=warnings,
        )
        development_refs = _positions_to_refs(
            positions=judgment.get("development_evidence_positions"),
            turns=turns,
            prefix=f"{prefix}_development_evidence_positions",
            errors=errors,
            warnings=warnings,
            allow_empty=development == "not_realized",
        )
        outcome_refs = _positions_to_refs(
            positions=judgment.get("durable_outcome_evidence_positions"),
            turns=turns,
            prefix=f"{prefix}_durable_outcome_evidence_positions",
            errors=errors,
            warnings=warnings,
            allow_empty=outcome == "not_realized",
        )
        scene_warnings = [warning for warning in warnings
                          if warning.startswith(f"{prefix}_")]
        scene_pass = (
            participant == "pass"
            and development == "realized"
            and outcome == "realized"
        )
        materialized.append({
            "scene_ref": str(child_scene.get("scene_ref") or ""),
            "memory_event_id": str(
                child_scene.get("memory_event_id") or ""),
            "selected_new_development": str(
                plan_scene.get("new_development") or ""),
            "selected_durable_outcome": str(
                plan_scene.get("durable_outcome") or ""),
            "participant_adherence": participant,
            "participant_evidence_turn_refs": participant_refs,
            "development_status": development,
            "development_evidence_turn_refs": development_refs,
            "durable_outcome_status": outcome,
            "durable_outcome_evidence_turn_refs": outcome_refs,
            "reason": reason,
            "evidence_warnings": scene_warnings,
            "scene_verdict": "PASS" if scene_pass else "FAIL",
        })

    audit = {
        "audit_version": "realizer_adherence_audit_v1",
        "scenes": materialized,
        "summary": {
            "scene_count": len(materialized),
            "scene_pass_count": sum(
                scene["scene_verdict"] == "PASS" for scene in materialized),
            "scene_fail_count": sum(
                scene["scene_verdict"] == "FAIL" for scene in materialized),
            "participant_drift_count": sum(
                scene["participant_adherence"] == "fail"
                for scene in materialized),
            "development_partial_or_missing_count": sum(
                scene["development_status"] != "realized"
                for scene in materialized),
            "durable_outcome_partial_or_missing_count": sum(
                scene["durable_outcome_status"] != "realized"
                for scene in materialized),
            "evidence_overflow_count": sum(
                bool(scene.get("evidence_warnings"))
                for scene in materialized),
        },
    }
    return {"valid": not errors, "errors": errors,
            "warnings": warnings, "audit": audit}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
