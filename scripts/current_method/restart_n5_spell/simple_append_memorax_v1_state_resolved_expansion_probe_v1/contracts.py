"""Mechanical contracts for Resolver -> Planner -> one-call Realizer."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from ..simple_append_v1.contracts import (
    PLANNER_SCHEMA_VERSION as BASE_PLANNER_SCHEMA_VERSION,
    compile_round_plan as compile_base_round_plan,
    normalize_round_realizer_output,
    validate_planner_output as validate_base_planner_output,
)


RESOLVER_SCHEMA_VERSION = "simple_append_state_resolver_v1"
RESOLVED_SET_VERSION = "simple_append_resolved_development_set_v1"
PLANNER_SCHEMA_VERSION = "simple_append_state_resolved_planner_v1"
COMPILED_PLAN_VERSION = "simple_append_state_resolved_compiled_plan_v1"

_RESOLVER_TOP_KEYS = {"resolver_schema_version", "developments"}
_DEVELOPMENT_KEYS = {
    "event_summary", "participants", "latest_state", "evidence_turn_refs",
}
_PLANNER_TOP_KEYS = {"planner_schema_version", "round_goal", "scenes"}
_PLANNER_SCENE_KEYS = {
    "development_id", "goal", "participants", "new_development",
    "interaction_beats", "transition_from_previous", "desired_outcome",
    "durable_outcome",
}
_BASE_SCENE_KEYS = {
    "goal", "participants", "new_development", "interaction_beats",
    "transition_from_previous", "desired_outcome",
}
_STOP_WORDS = {
    "a", "an", "the", "to", "and", "or", "of", "in", "on", "at", "for",
    "from", "with", "has", "have", "had", "is", "are", "was", "were", "be",
    "been", "being", "already", "still", "then", "later", "now", "will",
    "would", "should", "could", "their", "his", "her", "they", "them", "it",
    "this", "that",
}
_WORD_NORMALIZATION = {
    "decides": "decide", "decided": "decide", "deciding": "decide",
    "resumes": "resume", "resumed": "resume", "resuming": "resume",
    "starts": "start", "started": "start", "starting": "start",
    "begins": "begin", "began": "begin", "beginning": "begin",
    "plans": "plan", "planned": "plan", "planning": "plan",
    "calls": "call", "called": "call", "calling": "call",
    "agrees": "agree", "agreed": "agree", "agreeing": "agree",
}


def _sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _content_tokens(value: Any) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", _normalized(value))
    return {
        _WORD_NORMALIZATION.get(token, token)
        for token in tokens
        if token not in _STOP_WORDS
    }


def _base_planner_view(value: Mapping[str, Any]) -> dict[str, Any]:
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


def validate_resolver_output(
    value: Any, *, parent_turns: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != _RESOLVER_TOP_KEYS:
        return {"valid": False, "errors": ["resolver_top_level_schema"],
                "development_count": 0}
    if value.get("resolver_schema_version") != RESOLVER_SCHEMA_VERSION:
        errors.append("resolver_schema_version")
    developments = value.get("developments")
    if not isinstance(developments, list) or not 3 <= len(developments) <= 6:
        errors.append("resolver_development_count_3_to_6")
        developments = []
    parent_by_ref = {
        str(row.get("turn_id") or ""): row for row in parent_turns
        if str(row.get("turn_id") or "")
    }
    parent_speakers = {
        str(row.get("speaker") or "") for row in parent_turns
        if str(row.get("speaker") or "")
    }
    summaries: set[str] = set()
    for index, development in enumerate(developments, 1):
        prefix = f"development_{index:02d}"
        if not isinstance(development, Mapping) or set(development) != _DEVELOPMENT_KEYS:
            errors.append(f"{prefix}:schema")
            continue
        for key in ("event_summary", "latest_state"):
            if not _text(development.get(key)):
                errors.append(f"{prefix}:{key}")
        summary = _normalized(development.get("event_summary"))
        if summary in summaries:
            errors.append(f"{prefix}:duplicate_event_summary")
        summaries.add(summary)
        participants = development.get("participants")
        if (not isinstance(participants, list) or
                not 2 <= len(participants) <= 5 or
                any(not _text(row) for row in participants) or
                len(participants) != len(set(participants))):
            errors.append(f"{prefix}:participants")
        elif any(str(row) not in parent_speakers for row in participants):
            errors.append(f"{prefix}:participant_not_parent_speaker")
        refs = development.get("evidence_turn_refs")
        if (not isinstance(refs, list) or not 1 <= len(refs) <= 8 or
                any(not _text(row) for row in refs) or
                len(refs) != len(set(refs))):
            errors.append(f"{prefix}:evidence_turn_refs")
        elif any(str(ref) not in parent_by_ref for ref in refs):
            errors.append(f"{prefix}:evidence_ref_not_parent")
    return {"valid": not errors, "errors": errors,
            "development_count": len(developments)}


def materialize_resolved_developments(
    resolver_output: Mapping[str, Any], *, parent_turns: Sequence[Mapping[str, Any]],
    root_id: str, parent_dialogue_sha256: str,
) -> dict[str, Any]:
    validation = validate_resolver_output(
        resolver_output, parent_turns=parent_turns)
    if not validation["valid"]:
        raise ValueError("state_resolver_invalid:" + ",".join(validation["errors"]))
    parent_by_ref = {str(row["turn_id"]): row for row in parent_turns}
    developments = []
    for index, raw in enumerate(resolver_output["developments"], 1):
        evidence = []
        for ref in raw["evidence_turn_refs"]:
            row = parent_by_ref[str(ref)]
            evidence.append({
                "turn_ref": str(row["turn_id"]),
                "session_ref": str(row["session_id"]),
                "speaker": str(row["speaker"]),
                "text": str(row["text"]),
            })
        developments.append({
            "development_id": f"D{index:02d}",
            "event_summary": str(raw["event_summary"]).strip(),
            "participants": [str(row) for row in raw["participants"]],
            "latest_state": str(raw["latest_state"]).strip(),
            "evidence_turn_refs": [str(row) for row in raw["evidence_turn_refs"]],
            "evidence_source_turns": evidence,
        })
    result = {
        "resolved_set_version": RESOLVED_SET_VERSION,
        "root_id": str(root_id),
        "parent_dialogue_sha256": str(parent_dialogue_sha256),
        "developments": developments,
    }
    result["resolved_set_sha256"] = _sha(result)
    return result


def validate_planner_output(
    value: Any, *, resolved_set: Mapping[str, Any],
    parent_speakers: Sequence[str],
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != _PLANNER_TOP_KEYS:
        return {"valid": False, "errors": ["planner_top_level_schema"],
                "scene_count": 0, "development_ids": []}
    if value.get("planner_schema_version") != PLANNER_SCHEMA_VERSION:
        errors.append("planner_schema_version")
    scenes = value.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 3:
        errors.append("planner_scene_count_exactly_3")
        scenes = []
    resolved = {
        str(row["development_id"]): row
        for row in resolved_set.get("developments") or []
        if isinstance(row, Mapping) and row.get("development_id")
    }
    selected_ids: list[str] = []
    for index, scene in enumerate(scenes, 1):
        prefix = f"scene_{index:02d}"
        if not isinstance(scene, Mapping) or set(scene) != _PLANNER_SCENE_KEYS:
            errors.append(f"{prefix}:schema")
            continue
        development_id = str(scene.get("development_id") or "")
        selected_ids.append(development_id)
        if development_id not in resolved:
            errors.append(f"{prefix}:development_id_unknown")
            continue
        if not _text(scene.get("durable_outcome")):
            errors.append(f"{prefix}:durable_outcome")
        participants = scene.get("participants")
        allowed = set(str(row) for row in resolved[development_id]["participants"])
        if isinstance(participants, list) and any(
                str(row) not in allowed for row in participants):
            errors.append(f"{prefix}:participant_outside_development")
    if len(selected_ids) != len(set(selected_ids)):
        errors.append("planner_development_ids_not_unique")
    base = validate_base_planner_output(
        _base_planner_view(value), parent_speakers=parent_speakers)
    errors.extend(error for error in base["errors"] if error not in errors)
    return {"valid": not errors, "errors": errors,
            "scene_count": len(scenes), "development_ids": selected_ids}


def _obvious_repetition(latest_state: str, proposed: str) -> bool:
    latest = _normalized(latest_state)
    candidate = _normalized(proposed)
    if not latest or not candidate:
        return False
    if latest == candidate or candidate in latest:
        return True
    latest_tokens = _content_tokens(latest_state)
    candidate_tokens = _content_tokens(proposed)
    if len(candidate_tokens) < 4:
        return False
    overlap = len(latest_tokens & candidate_tokens) / len(candidate_tokens)
    return overlap >= 0.85


def validate_planner_continuation(
    value: Mapping[str, Any], *, resolved_set: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = {
        str(row["development_id"]): row
        for row in resolved_set.get("developments") or []
    }
    errors: list[str] = []
    checks = []
    for index, scene in enumerate(value.get("scenes") or [], 1):
        if not isinstance(scene, Mapping):
            continue
        development_id = str(scene.get("development_id") or "")
        development = resolved.get(development_id)
        if development is None:
            continue
        repeated_fields = [
            field for field in ("new_development", "durable_outcome")
            if _obvious_repetition(
                str(development["latest_state"]), str(scene.get(field) or ""))
        ]
        checks.append({
            "scene_index": index,
            "development_id": development_id,
            "latest_state": development["latest_state"],
            "new_development": scene.get("new_development"),
            "durable_outcome": scene.get("durable_outcome"),
            "obvious_repetition_fields": repeated_fields,
        })
        errors.extend(
            f"scene_{index:02d}:{field}:obvious_latest_state_repetition"
            for field in repeated_fields
        )
    return {"valid": not errors, "errors": errors, "checks": checks,
            "gate_scope": "normalized_exact_and_high_containment_only"}


def compile_round_plan(
    planner_output: Mapping[str, Any], *, resolved_set: Mapping[str, Any],
    tail_context: Mapping[str, Any], round_index: int = 1,
) -> dict[str, Any]:
    validation = validate_planner_output(
        planner_output, resolved_set=resolved_set,
        parent_speakers=tail_context["parent_speakers"])
    if not validation["valid"]:
        raise ValueError("state_resolved_planner_invalid:" +
                         ",".join(validation["errors"]))
    continuation = validate_planner_continuation(
        planner_output, resolved_set=resolved_set)
    if not continuation["valid"]:
        raise ValueError("state_resolved_planner_repetition:" +
                         ",".join(continuation["errors"]))
    compiled = compile_base_round_plan(
        _base_planner_view(planner_output), tail_context=tail_context,
        round_index=round_index)
    resolved = {
        str(row["development_id"]): row
        for row in resolved_set["developments"]
    }
    for target, source in zip(compiled["scenes"], planner_output["scenes"]):
        development_id = str(source["development_id"])
        target["development_id"] = development_id
        target["resolved_development"] = deepcopy(resolved[development_id])
        target["durable_outcome"] = str(source["durable_outcome"]).strip()
    compiled["compiled_plan_version"] = COMPILED_PLAN_VERSION
    compiled["state_resolution_contract"] = {
        "resolved_set_sha256": str(resolved_set["resolved_set_sha256"]),
        "development_ids_unique": True,
        "latest_states_frozen": True,
        "model_generated_development_ids": False,
        "anchor_contract_used": False,
    }
    compiled.pop("compiled_plan_sha256", None)
    compiled["compiled_plan_sha256"] = _sha(compiled)
    return compiled


__all__ = [
    "COMPILED_PLAN_VERSION",
    "PLANNER_SCHEMA_VERSION",
    "RESOLVED_SET_VERSION",
    "RESOLVER_SCHEMA_VERSION",
    "compile_round_plan",
    "materialize_resolved_developments",
    "normalize_round_realizer_output",
    "validate_planner_continuation",
    "validate_planner_output",
    "validate_resolver_output",
]
