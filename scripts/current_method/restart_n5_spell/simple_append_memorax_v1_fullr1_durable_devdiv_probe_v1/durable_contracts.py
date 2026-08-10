"""Mechanical local-span Planner contract for durable development diversity."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

from ..simple_append_v1.contracts import (
    PLANNER_SCHEMA_VERSION as BASE_PLANNER_SCHEMA_VERSION,
    compile_round_plan as compile_base_round_plan,
    normalize_round_realizer_output,
    validate_planner_output as validate_base_planner_output,
)


PLANNER_SCHEMA_VERSION = "simple_append_durable_devdiv_local_span_planner_v1"
COMPILED_PLAN_VERSION = "simple_append_durable_devdiv_local_span_compiled_plan_v1"
PARENT_ANCHOR_REGISTRY_VERSION = "simple_append_parent_anchor_position_registry_v1"

_TOP_KEYS = {"planner_schema_version", "round_goal", "scenes"}
_SCENE_KEYS = {
    "goal", "participants", "primary_anchor_span", "new_development",
    "interaction_beats", "transition_from_previous", "desired_outcome",
    "durable_outcome",
}
_BASE_SCENE_KEYS = {
    "goal", "participants", "new_development", "interaction_beats",
    "transition_from_previous", "desired_outcome",
}
_SPAN_KEYS = {"start_position", "end_position"}


def _sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _base_view(value: Mapping[str, Any]) -> dict[str, Any]:
    scenes = []
    for scene in value.get("scenes") or []:
        if isinstance(scene, Mapping):
            scenes.append({key: deepcopy(scene.get(key)) for key in _BASE_SCENE_KEYS})
        else:
            scenes.append(scene)
    return {
        "planner_schema_version": BASE_PLANNER_SCHEMA_VERSION,
        "round_goal": value.get("round_goal"),
        "scenes": scenes,
    }


def build_parent_anchor_registry(
    parent_turns: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assign global 1-based positions to authoritative Parent turns."""
    entries: list[dict[str, Any]] = []
    for position, turn in enumerate(parent_turns, 1):
        turn_ref = str(turn.get("turn_id") or "")
        session_ref = str(turn.get("session_id") or "")
        text = str(turn.get("text") or "")
        if not turn_ref or not session_ref or not text.strip():
            raise ValueError(f"parent_anchor_registry_turn_invalid:{position}")
        entries.append({
            "position": position,
            "turn_ref": turn_ref,
            "session_ref": session_ref,
            "speaker": str(turn.get("speaker") or ""),
            "text": text,
            "partition": "parent",
        })
    result = {
        "parent_anchor_registry_version": PARENT_ANCHOR_REGISTRY_VERSION,
        "position_base": 1,
        "parent_turn_count": len(entries),
        "entries": entries,
    }
    result["registry_sha256"] = _sha(result)
    return result


def _registry_by_position(
    registry: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    if registry.get("parent_anchor_registry_version") != \
            PARENT_ANCHOR_REGISTRY_VERSION:
        raise ValueError("parent_anchor_registry_version_invalid")
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ValueError("parent_anchor_registry_entries_invalid")
    result: dict[int, Mapping[str, Any]] = {}
    for row in entries:
        if not isinstance(row, Mapping):
            raise ValueError("parent_anchor_registry_entry_invalid")
        position = row.get("position")
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError("parent_anchor_registry_position_invalid")
        if position in result or row.get("partition") != "parent":
            raise ValueError("parent_anchor_registry_entry_invalid")
        result[position] = row
    return result


def validate_planner_output(
    value: Any, *, parent_speakers: Sequence[str],
    parent_anchor_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one local Parent span and unique derived anchor sessions."""
    if not isinstance(value, Mapping) or set(value) != _TOP_KEYS:
        return {
            "valid": False,
            "errors": ["planner_top_level_schema"],
            "scene_count": 0,
            "derived_anchor_session_refs": [],
            "materialized_anchors": [],
        }
    registry = _registry_by_position(parent_anchor_registry)
    errors: list[str] = []
    if value.get("planner_schema_version") != PLANNER_SCHEMA_VERSION:
        errors.append("planner_schema_version")
    scenes = value.get("scenes")
    if not isinstance(scenes, list):
        scenes = []
    derived_sessions: list[str] = []
    materialized: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, 1):
        prefix = f"scene_{index:02d}"
        if not isinstance(scene, Mapping) or set(scene) != _SCENE_KEYS:
            errors.append(f"{prefix}:schema")
            continue
        if not _text(scene.get("durable_outcome")):
            errors.append(f"{prefix}:durable_outcome")
        span = scene.get("primary_anchor_span")
        if not isinstance(span, Mapping) or set(span) != _SPAN_KEYS:
            errors.append(f"{prefix}:primary_anchor_span_schema")
            continue
        start = span.get("start_position")
        end = span.get("end_position")
        if (isinstance(start, bool) or not isinstance(start, int) or
                isinstance(end, bool) or not isinstance(end, int)):
            errors.append(f"{prefix}:primary_anchor_span_positions")
            continue
        if start <= 0 or end <= 0 or start > end:
            errors.append(f"{prefix}:primary_anchor_span_order")
            continue
        positions = list(range(start, end + 1))
        if len(positions) > 6:
            errors.append(f"{prefix}:primary_anchor_span_too_long")
            continue
        missing = [position for position in positions if position not in registry]
        if missing:
            errors.append(f"{prefix}:primary_anchor_span_not_parent")
            continue
        rows = [registry[position] for position in positions]
        sessions = {str(row["session_ref"]) for row in rows}
        if len(sessions) != 1:
            errors.append(f"{prefix}:primary_anchor_span_cross_session")
            continue
        session_ref = next(iter(sessions))
        derived_sessions.append(session_ref)
        materialized.append({
            "scene_index": index,
            "primary_anchor_span": {
                "start_position": start,
                "end_position": end,
            },
            "anchor_positions": list(positions),
            "anchor_turn_refs": [str(row["turn_ref"]) for row in rows],
            "anchor_speakers": [str(row["speaker"]) for row in rows],
            "anchor_turn_texts": [str(row["text"]) for row in rows],
            "derived_anchor_session_ref": session_ref,
            "anchor_turns": [deepcopy(dict(row)) for row in rows],
        })

    if len(derived_sessions) != len(set(derived_sessions)):
        errors.append("planner_derived_anchor_sessions_not_unique")

    base_validation = validate_base_planner_output(
        _base_view(value), parent_speakers=parent_speakers)
    errors.extend(
        error for error in base_validation["errors"] if error not in errors)
    return {
        "valid": not errors,
        "errors": errors,
        "scene_count": len(scenes),
        "derived_anchor_session_refs": derived_sessions,
        "unique_derived_anchor_session_count": len(set(derived_sessions)),
        "materialized_anchors": materialized,
    }


def compile_round_plan(
    planner_output: Mapping[str, Any], *, tail_context: Mapping[str, Any],
    parent_anchor_registry: Mapping[str, Any], round_index: int = 1,
) -> dict[str, Any]:
    validation = validate_planner_output(
        planner_output,
        parent_speakers=tail_context["parent_speakers"],
        parent_anchor_registry=parent_anchor_registry,
    )
    if not validation["valid"]:
        raise ValueError(
            "durable_devdiv_planner_invalid:" + ",".join(validation["errors"]))
    compiled = compile_base_round_plan(
        _base_view(planner_output), tail_context=tail_context,
        round_index=round_index)
    for compiled_scene, source_scene, anchor in zip(
            compiled["scenes"], planner_output["scenes"],
            validation["materialized_anchors"]):
        compiled_scene["primary_anchor_span"] = deepcopy(
            anchor["primary_anchor_span"])
        compiled_scene["anchor_positions"] = list(anchor["anchor_positions"])
        compiled_scene["anchor_turn_refs"] = list(anchor["anchor_turn_refs"])
        compiled_scene["anchor_speakers"] = list(anchor["anchor_speakers"])
        compiled_scene["anchor_turn_texts"] = list(anchor["anchor_turn_texts"])
        compiled_scene["derived_anchor_session_ref"] = str(
            anchor["derived_anchor_session_ref"])
        compiled_scene["primary_anchor_materialization"] = [{
            "position": row["position"],
            "turn_ref": row["turn_ref"],
            "speaker": row["speaker"],
            "text": row["text"],
        } for row in anchor["anchor_turns"]]
        compiled_scene["durable_outcome"] = str(
            source_scene["durable_outcome"]).strip()
    compiled["compiled_plan_version"] = COMPILED_PLAN_VERSION
    compiled["durable_development_contract"] = {
        "derived_anchor_session_refs": list(
            validation["derived_anchor_session_refs"]),
        "derived_anchor_sessions_unique": True,
        "durable_outcome_required_per_scene": True,
        "model_supplied_session_ref": False,
        "model_supplied_discrete_anchor_positions": False,
    }
    compiled.pop("compiled_plan_sha256", None)
    compiled["compiled_plan_sha256"] = _sha(compiled)
    return compiled


__all__ = [
    "COMPILED_PLAN_VERSION",
    "PARENT_ANCHOR_REGISTRY_VERSION",
    "PLANNER_SCHEMA_VERSION",
    "build_parent_anchor_registry",
    "compile_round_plan",
    "normalize_round_realizer_output",
    "validate_planner_output",
]
