"""Append-only Parent + Child materialization for QA fact sourcing."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping


EXPANDED_DIALOGUE_VERSION = "simple_append_expanded_dialogue_v1"
MULTI_ROUND_EXPANDED_DIALOGUE_VERSION = (
    "simple_append_expanded_dialogue_multi_round_v1")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def assemble_expanded_dialogue(
    *, parent: Mapping[str, Any], child: Mapping[str, Any],
) -> dict[str, Any]:
    root_id = str(parent.get("root_id") or "")
    if not root_id or child.get("root_id") != root_id:
        raise ValueError("simple_append_root_mismatch")
    parent_turns = list((parent.get("dialogue") or {}).get("turns") or [])
    if not parent_turns:
        raise ValueError("simple_append_parent_turns_missing")
    turns: list[dict[str, Any]] = []
    for row in parent_turns:
        turns.append({
            "turn_ref": str(row["turn_id"]),
            "session_ref": str(row["session_id"]),
            "scene_ref": None,
            "speaker": str(row["speaker"]),
            "text": str(row["text"]),
            "origin": "source_grounded",
        })
    first_child_index = len(turns) + 1
    child_scene_refs: list[str] = []
    for scene in child.get("scenes") or []:
        scene_ref = str(scene["scene_ref"])
        child_scene_refs.append(scene_ref)
        for row in scene.get("dialogue_turns") or []:
            turns.append({
                "turn_ref": str(row["turn_ref"]),
                "session_ref": str(scene["session_ref"]),
                "scene_ref": scene_ref,
                "speaker": str(row["speaker"]),
                "text": str(row["text"]),
                "origin": "generated_extension",
            })
    result = {
        "expanded_dialogue_version": EXPANDED_DIALOGUE_VERSION,
        "root_id": root_id,
        "append_mode": "parent_tail_only",
        "parent_dialogue_sha256": str(parent["dialogue"]["dialogue_hash"]),
        "parent_turn_count": len(parent_turns),
        "child_turn_count": len(turns) - len(parent_turns),
        "turn_count": len(turns),
        "child_scene_refs": child_scene_refs,
        "child_boundary": {
            "first_child_turn_index_1_based": first_child_index,
            "last_parent_turn_ref": str(parent_turns[-1]["turn_id"]),
            "first_child_turn_ref": (turns[first_child_index - 1]["turn_ref"]
                                     if len(turns) >= first_child_index else None),
        },
        "turns": turns,
        "qa_fact_source": "complete_parent_plus_child",
        "memory_is_qa_fact_source": False,
    }
    result["complete_dialogue_sha256"] = _sha(result)
    return result


def validate_expanded_dialogue(
    value: Mapping[str, Any], *, parent: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if value.get("expanded_dialogue_version") != EXPANDED_DIALOGUE_VERSION:
        errors.append("expanded_dialogue_version")
    parent_turns = list(parent["dialogue"]["turns"])
    turns = list(value.get("turns") or [])
    if len(turns) <= len(parent_turns):
        errors.append("expanded_dialogue_child_missing")
    expected_parent = [{
        "turn_ref": str(row["turn_id"]),
        "session_ref": str(row["session_id"]),
        "scene_ref": None,
        "speaker": str(row["speaker"]),
        "text": str(row["text"]),
        "origin": "source_grounded",
    } for row in parent_turns]
    if turns[:len(parent_turns)] != expected_parent:
        errors.append("expanded_dialogue_parent_changed_or_reordered")
    child = turns[len(parent_turns):]
    if any(row.get("origin") != "generated_extension" for row in child):
        errors.append("expanded_dialogue_child_origin")
    refs = [str(row.get("turn_ref") or "") for row in turns]
    if any(not ref for ref in refs) or len(refs) != len(set(refs)):
        errors.append("expanded_dialogue_turn_refs")
    boundary = value.get("child_boundary") or {}
    if boundary.get("first_child_turn_index_1_based") != len(parent_turns) + 1:
        errors.append("expanded_dialogue_boundary_index")
    if child and boundary.get("first_child_turn_ref") != child[0].get("turn_ref"):
        errors.append("expanded_dialogue_boundary_ref")
    return {"valid": not errors, "errors": errors,
            "parent_turn_count": len(parent_turns),
            "child_turn_count": len(child), "turn_count": len(turns)}


def append_expansion_round(
    *, prior_expanded: Mapping[str, Any], child: Mapping[str, Any],
) -> dict[str, Any]:
    """Append a later round without rewriting any prior dialogue record."""
    root_id = str(prior_expanded.get("root_id") or "")
    if not root_id or child.get("root_id") != root_id:
        raise ValueError("simple_append_multi_round_root_mismatch")
    prior_turns = deepcopy(list(prior_expanded.get("turns") or []))
    if not prior_turns:
        raise ValueError("simple_append_multi_round_prior_turns_missing")
    round_index = int(child.get("round_index") or 0)
    expected_prefix = f"R{round_index:02d}:"
    if round_index < 2:
        raise ValueError("simple_append_multi_round_index_invalid")
    new_turns: list[dict[str, Any]] = []
    new_scene_refs: list[str] = []
    for scene in child.get("scenes") or []:
        scene_ref = str(scene.get("scene_ref") or "")
        if not scene_ref.startswith(expected_prefix):
            raise ValueError("simple_append_multi_round_scene_ref_invalid")
        new_scene_refs.append(scene_ref)
        for row in scene.get("dialogue_turns") or []:
            new_turns.append({
                "turn_ref": str(row["turn_ref"]),
                "session_ref": str(scene["session_ref"]),
                "scene_ref": scene_ref,
                "speaker": str(row["speaker"]),
                "text": str(row["text"]),
                "origin": "generated_extension",
            })
    if not new_turns:
        raise ValueError("simple_append_multi_round_child_turns_missing")
    prior_refs = {str(row.get("turn_ref") or "") for row in prior_turns}
    new_refs = [row["turn_ref"] for row in new_turns]
    if (len(prior_refs) != len(prior_turns) or
            len(new_refs) != len(set(new_refs)) or
            prior_refs.intersection(new_refs)):
        raise ValueError("simple_append_multi_round_turn_ref_collision")
    boundaries = deepcopy(list(prior_expanded.get("round_boundaries") or []))
    if not boundaries:
        first_child = int((prior_expanded.get("child_boundary") or {}).get(
            "first_child_turn_index_1_based") or 0)
        if first_child:
            boundaries.append({
                "round_index": 1,
                "first_turn_index_1_based": first_child,
                "last_turn_index_1_based": len(prior_turns),
                "first_turn_ref": prior_turns[first_child - 1]["turn_ref"],
                "last_turn_ref": prior_turns[-1]["turn_ref"],
                "scene_refs": list(prior_expanded.get("child_scene_refs") or []),
            })
    boundary = {
        "round_index": round_index,
        "first_turn_index_1_based": len(prior_turns) + 1,
        "last_turn_index_1_based": len(prior_turns) + len(new_turns),
        "first_turn_ref": new_turns[0]["turn_ref"],
        "last_turn_ref": new_turns[-1]["turn_ref"],
        "scene_refs": new_scene_refs,
    }
    boundaries.append(boundary)
    turns = prior_turns + new_turns
    original_parent_count = int(prior_expanded.get(
        "original_parent_turn_count") or
        prior_expanded.get("parent_turn_count") or 0)
    result = {
        "expanded_dialogue_version": MULTI_ROUND_EXPANDED_DIALOGUE_VERSION,
        "root_id": root_id,
        "append_mode": "multi_round_parent_tail_only",
        "parent_dialogue_sha256": str(
            prior_expanded.get("parent_dialogue_sha256") or ""),
        "original_parent_turn_count": original_parent_count,
        "parent_turn_count": original_parent_count,
        "child_turn_count": len(turns) - original_parent_count,
        "latest_round_child_turn_count": len(new_turns),
        "turn_count": len(turns),
        "round_count": round_index,
        "child_scene_refs": [
            ref for row in boundaries for ref in row.get("scene_refs") or []],
        "round_boundaries": boundaries,
        "latest_round_boundary": boundary,
        "child_boundary": deepcopy(prior_expanded.get("child_boundary") or {}),
        "turns": turns,
        "qa_fact_source": "complete_parent_plus_child",
        "memory_is_qa_fact_source": False,
        "prior_complete_dialogue_sha256": str(
            prior_expanded.get("complete_dialogue_sha256") or ""),
    }
    result["complete_dialogue_sha256"] = _sha(result)
    return result


def validate_appended_round(
    value: Mapping[str, Any], *, prior_expanded: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if value.get("expanded_dialogue_version") != \
            MULTI_ROUND_EXPANDED_DIALOGUE_VERSION:
        errors.append("multi_round_expanded_dialogue_version")
    prior_turns = list(prior_expanded.get("turns") or [])
    turns = list(value.get("turns") or [])
    if turns[:len(prior_turns)] != prior_turns:
        errors.append("multi_round_prior_dialogue_changed")
    new_turns = turns[len(prior_turns):]
    if not new_turns:
        errors.append("multi_round_new_turns_missing")
    if any(row.get("origin") != "generated_extension" for row in new_turns):
        errors.append("multi_round_new_turn_origin")
    refs = [str(row.get("turn_ref") or "") for row in turns]
    if any(not ref for ref in refs) or len(refs) != len(set(refs)):
        errors.append("multi_round_turn_ref_collision")
    return {"valid": not errors, "errors": errors,
            "prior_turn_count": len(prior_turns),
            "new_turn_count": len(new_turns), "turn_count": len(turns)}


def expanded_dialogue_markdown(value: Mapping[str, Any]) -> str:
    lines = [
        "# Simple Append Expanded Dialogue",
        "",
        f"Root: `{value.get('root_id')}`",
        "",
    ]
    child_started = False
    current_session: str | None = None
    for row in value.get("turns") or []:
        if row.get("origin") == "generated_extension" and not child_started:
            lines.extend(["---", "", "# CHILD START - ROUND 1", ""])
            child_started = True
            current_session = None
        session = str(row.get("scene_ref") or row.get("session_ref") or "")
        if session != current_session:
            label = "Child scene" if child_started else "Parent session"
            lines.extend([f"## {label} `{session}`", ""])
            current_session = session
        lines.append(
            f"- `{row.get('turn_ref')}` | **{row.get('speaker')}**: {row.get('text')}")
    lines.append("")
    return "\n".join(lines)
