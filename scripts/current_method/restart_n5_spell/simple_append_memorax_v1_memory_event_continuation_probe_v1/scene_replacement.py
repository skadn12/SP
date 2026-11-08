"""Bounded replacement packaging for a failed realized scene.

Replacement is deliberately downstream of the existing Planner and Gate.  It
can only consume a candidate that already passed the Gate and was not selected
for the original round.  This module does no semantic approval; the final
Adherence Auditor remains authoritative.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping

from .contracts import is_transport_speaker_label


def _scene_ref(scene: Mapping[str, Any], index: int, *, round_index: int) -> str:
    return str(scene.get("scene_ref") or
               f"R{int(round_index):02d}:S{index:02d}")


def _candidate_ref(event_id: str, index: int) -> str:
    return f"{event_id}:C{int(index):02d}"


def _failed_refs(*, child: Mapping[str, Any], adherence_audit: Mapping[str, Any]) -> list[str]:
    scenes = child.get("scenes")
    audits = adherence_audit.get("scenes")
    if not isinstance(scenes, list) or not isinstance(audits, list):
        return []
    try:
        round_index = int(child.get("round_index") or 1)
    except (TypeError, ValueError):
        return []
    result: list[str] = []
    for index, (scene, audit) in enumerate(zip(scenes, audits), 1):
        if not isinstance(scene, Mapping) or not isinstance(audit, Mapping):
            continue
        if str(audit.get("scene_verdict") or "") != "PASS":
            result.append(_scene_ref(scene, index, round_index=round_index))
    return result


def _selected_refs(plan: Mapping[str, Any]) -> set[str]:
    return {
        str(scene.get("selected_candidate_ref") or "")
        for scene in plan.get("scenes") or []
        if isinstance(scene, Mapping) and scene.get("selected_candidate_ref")
    }


def build_scene_replacement_request(
    *, raw_plan: Mapping[str, Any], materialized_plan: Mapping[str, Any],
    child: Mapping[str, Any], adherence_audit: Mapping[str, Any],
    passed_candidate_refs: list[str] | tuple[str, ...],
    outline_conflict_edges: list[list[str]] | tuple[tuple[str, str], ...] = (),
) -> dict[str, Any]:
    """Choose at most one unused Gate-approved candidate for a failed scene."""
    failed = _failed_refs(child=child, adherence_audit=adherence_audit)
    selected = _selected_refs(materialized_plan)
    failed_selected_refs = {
        str(scene.get("selected_candidate_ref") or "")
        for index, scene in enumerate(materialized_plan.get("scenes") or [], 1)
        if isinstance(scene, Mapping) and
        _scene_ref(scene, index, round_index=int(child.get("round_index") or 1))
        in set(failed)
    }
    surviving_selected_refs = selected - failed_selected_refs
    passed = {str(ref) for ref in passed_candidate_refs}
    conflict_edges = {
        tuple(sorted((str(edge[0]), str(edge[1]))))
        for edge in outline_conflict_edges
        if isinstance(edge, (list, tuple)) and len(edge) == 2
    }
    raw_scenes = raw_plan.get("candidate_pool") or raw_plan.get("scenes")
    if not isinstance(raw_scenes, list):
        raw_scenes = []
    used_event_ids = {
        str(scene.get("memory_event_id") or "")
        for scene in materialized_plan.get("scenes") or []
        if isinstance(scene, Mapping)
    }
    candidates: list[dict[str, Any]] = []
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, Mapping):
            continue
        event_id = str(raw_scene.get("memory_event_id") or "")
        rows = raw_scene.get("next_step_candidates")
        if not isinstance(rows, list):
            continue
        for index, candidate in enumerate(rows, 1):
            ref = _candidate_ref(event_id, index)
            conflicts_with_survivor = any(
                tuple(sorted((ref, survivor))) in conflict_edges
                for survivor in surviving_selected_refs)
            if (ref not in passed or ref in selected or event_id in used_event_ids
                    or conflicts_with_survivor):
                continue
            if not isinstance(candidate, Mapping):
                continue
            scene = deepcopy(dict(raw_scene))
            scene.pop("next_step_candidates", None)
            scene["selected_candidate_ref"] = ref
            scene["before_state"] = str(candidate.get("before_state") or "")
            scene["required_transition"] = str(
                candidate.get("required_transition") or "")
            scene["after_state"] = str(candidate.get("after_state") or "")
            scene["actor_bindings"] = deepcopy(candidate.get("actor_bindings") or [])
            scene["interaction_beats"] = deepcopy(
                candidate.get("interaction_beats") or [])
            scene["new_development"] = scene["required_transition"]
            scene["durable_outcome"] = scene["after_state"]
            candidates.append({
                "candidate_ref": ref,
                "memory_event_id": event_id,
                "scene": scene,
            })

    # Only one replacement is allowed in this bounded pass.  A later failed
    # replacement is quarantined; it never causes another Planner/Realizer call.
    selected = candidates[0] if failed and candidates else None
    if selected is not None:
        selected["scene"]["scene_ref"] = failed[0]
    return {
        "version": "scene_replacement_request_v1",
        "replacement_limit": 1,
        "failed_scene_refs": failed,
        "selected_replacement": selected,
        "available_replacement_count": len(candidates),
        "unused_gate_passed_candidate_refs": [
            row["candidate_ref"] for row in candidates],
    }


def build_scene_replacement_prompt(request: Mapping[str, Any]) -> dict[str, Any]:
    selected = request.get("selected_replacement")
    scene = selected.get("scene") if isinstance(selected, Mapping) else None
    if not isinstance(scene, Mapping):
        return {
            "prompt_version": "scene_replacement_realizer_prompt_v1",
            "prompt": "",
            "replacement_count": 0,
            "full_parent_included": False,
            "memory_included": False,
        }
    payload = {
        "required_participants": list(
            scene.get("required_participants") or
            scene.get("participants") or []),
        "goal": scene.get("goal"),
        "before_state": scene.get("before_state"),
        "required_transition": scene.get("required_transition"),
        "after_state": scene.get("after_state"),
        "durable_outcome": scene.get("durable_outcome"),
        "actor_bindings": deepcopy(scene.get("actor_bindings") or []),
        "interaction_beats": deepcopy(scene.get("interaction_beats") or []),
    }
    prompt = f"""You are the bounded Replacement Realizer. One earlier scene was
quarantined after its single allowed continuation attempt. Write exactly one
replacement scene for the approved plan below. Do not mention the rejected
scene or replay Parent dialogue. Required participants must appear, but they
are not a speaker allowlist. An auxiliary established person or scene-local
passerby may speak, but must not replace a bound actor or take over the core
development. Make the required transition and durable outcome explicit
in natural dialogue. Produce 9-48 turns to preserve the existing scene
contract. Return only JSON with exactly one scenes item; each turn must contain
only speaker and text.

APPROVED REPLACEMENT PLAN
{json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)}

OUTPUT ONLY JSON
{{"scenes":[{{"dialogue_turns":[{{"speaker":"...","text":"..."}}]}}]}}"""
    return {
        "prompt_version": "scene_replacement_realizer_prompt_v1",
        "prompt": prompt,
        "replacement_count": 1,
        "full_parent_included": False,
        "memory_included": False,
    }


def materialize_scene_replacement_output(
    raw: Any, *, request: Mapping[str, Any],
) -> dict[str, Any]:
    selected = request.get("selected_replacement")
    if not isinstance(selected, Mapping):
        return {"valid": True, "errors": [], "replacements": [],
                "skipped": "no_replacement_candidate"}
    if not isinstance(raw, Mapping) or set(raw) != {"scenes"}:
        return {"valid": False, "errors": ["scene_replacement_top_level"],
                "replacements": []}
    rows = raw.get("scenes")
    if not isinstance(rows, list) or len(rows) != 1:
        return {"valid": False, "errors": ["scene_replacement_count"],
                "replacements": []}
    row = rows[0]
    turns = row.get("dialogue_turns") if isinstance(row, Mapping) else None
    if not isinstance(turns, list) or not 9 <= len(turns) <= 48:
        return {"valid": False, "errors": ["scene_replacement_turns"],
                "replacements": []}
    materialized: list[dict[str, Any]] = []
    errors: list[str] = []
    actual_speakers: set[str] = set()
    for index, turn in enumerate(turns, 1):
        if not isinstance(turn, Mapping) or set(turn) != {"speaker", "text"}:
            errors.append(f"replacement_turn_{index}_shape")
            continue
        speaker = str(turn.get("speaker") or "").strip()
        text = str(turn.get("text") or "").strip()
        if not speaker or is_transport_speaker_label(speaker):
            errors.append(f"replacement_turn_{index}_speaker")
        elif speaker:
            actual_speakers.add(speaker)
        if not text:
            errors.append(f"replacement_turn_{index}_text")
        materialized.append({"speaker": speaker, "text": text})
    source_scene = selected.get("scene")
    required_participants = {
        str(value) for value in (
            (source_scene or {}).get("required_participants") or
            (source_scene or {}).get("participants") or [])
        if str(value)
    }
    for participant in sorted(required_participants - actual_speakers):
        errors.append(f"replacement_required_participant_missing:{participant}")
    return {
        "valid": not errors,
        "errors": errors,
        "replacements": [{
            "candidate_ref": selected.get("candidate_ref"),
            "memory_event_id": selected.get("memory_event_id"),
            "scene_ref": str((selected.get("scene") or {}).get("scene_ref") or ""),
            "memory_event_ref": (source_scene or {}).get("memory_event_ref"),
            "memory_event_key": (source_scene or {}).get("memory_event_key"),
            "memory_event_display": (source_scene or {}).get("memory_event_display"),
            "development_mode": (source_scene or {}).get("development_mode"),
            "new_people": deepcopy((source_scene or {}).get("new_people") or []),
            "dialogue_turns": materialized,
        }],
    }


def merge_scene_replacement(
    child: Mapping[str, Any], materialized: Mapping[str, Any],
) -> dict[str, Any]:
    if not materialized.get("valid"):
        raise ValueError("scene_replacement_materialization_invalid")
    rows = materialized.get("replacements") or []
    if not rows:
        return deepcopy(dict(child))
    row = rows[0]
    target_ref = str(row.get("scene_ref") or "")
    result = deepcopy(dict(child))
    scenes = result.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("scene_replacement_child_scenes_required")
    for scene in scenes:
        if not isinstance(scene, dict) or str(scene.get("scene_ref") or "") != target_ref:
            continue
        scene["memory_event_id"] = row.get("memory_event_id")
        for key in ("memory_event_ref", "memory_event_key",
                    "memory_event_display", "development_mode", "new_people"):
            if key in row:
                scene[key] = deepcopy(row.get(key))
        scene["dialogue_turns"] = [
            {"turn_ref": f"{target_ref}:T{index + 1:03d}", **dict(turn)}
            for index, turn in enumerate(row.get("dialogue_turns") or [])]
        return result
    raise ValueError("scene_replacement_target_scene_missing")


def materialize_replacement_plan(
    plan: Mapping[str, Any], request: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace the failed plan slot with the selected approved candidate."""
    selected = request.get("selected_replacement")
    if not isinstance(selected, Mapping):
        return deepcopy(dict(plan))
    source = selected.get("scene")
    if not isinstance(source, Mapping):
        raise ValueError("scene_replacement_plan_source_missing")
    target_ref = str(source.get("scene_ref") or "")
    result = deepcopy(dict(plan))
    # Planner materialization historically omits ``round_index``.  Derive the
    # target round from the already authoritative scene ref so R02 (and later)
    # replacement slots are matched against R02:Sxx rather than the R01
    # fallback generated by ``_scene_ref``.
    target_round_index = int(result.get("round_index") or 0)
    if target_round_index < 1 and target_ref.startswith("R"):
        round_token = target_ref.split(":", 1)[0][1:]
        if round_token.isdigit():
            target_round_index = int(round_token)
    if target_round_index < 1:
        target_round_index = 1
    scenes = result.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("scene_replacement_plan_scenes_required")
    for index, scene in enumerate(scenes, 1):
        if not isinstance(scene, Mapping):
            continue
        ref = _scene_ref(scene, index, round_index=target_round_index)
        if ref == target_ref:
            replacement = deepcopy(dict(source))
            replacement["scene_ref"] = target_ref
            scenes[index - 1] = replacement
            return result
    raise ValueError("scene_replacement_plan_target_missing")
