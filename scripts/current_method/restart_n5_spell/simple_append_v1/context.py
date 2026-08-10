"""Deterministic tail-session context selection for simple append expansion."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from ..unified_dialogue_root import validate_unified_dialogue_root


TAIL_CONTEXT_VERSION = "simple_append_tail_context_v1"
MULTI_ROUND_TAIL_CONTEXT_VERSION = "simple_append_multi_round_tail_context_v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def build_tail_context(
    parent: Mapping[str, Any], *, session_count: int = 2,
) -> dict[str, Any]:
    """Select the final one or two complete source sessions without mutation."""
    if session_count not in {1, 2}:
        raise ValueError("simple_append_session_count_must_be_one_or_two")
    validation = validate_unified_dialogue_root(parent)
    if not validation["valid"]:
        raise ValueError("simple_append_parent_invalid:" +
                         ",".join(validation["errors"]))

    sessions = list(parent["session_metadata"])
    selected_sessions = sessions[-min(session_count, len(sessions)):]
    selected_ids = {str(row["session_id"]) for row in selected_sessions}
    selected_turns = [deepcopy(row) for row in parent["dialogue"]["turns"]
                      if str(row["session_id"]) in selected_ids]
    speakers = list(dict.fromkeys(
        str(row["speaker"]) for row in parent["dialogue"]["turns"]))
    result = {
        "tail_context_version": TAIL_CONTEXT_VERSION,
        "root_id": str(parent["root_id"]),
        "parent_dialogue_sha256": str(parent["dialogue"]["dialogue_hash"]),
        "parent_session_count": int(parent["dialogue"]["session_count"]),
        "parent_turn_count": int(parent["dialogue"]["turn_count"]),
        "selected_session_count": len(selected_sessions),
        "selected_session_ids": [str(row["session_id"])
                                 for row in selected_sessions],
        "selected_session_summaries": [{
            "session_id": str(row["session_id"]),
            "source_session_index": int(row["source_session_index"]),
            "summary": str(row["summary"]),
        } for row in selected_sessions],
        "selected_turns": selected_turns,
        "selected_turn_count": len(selected_turns),
        "parent_speakers": speakers,
        "first_selected_turn_id": str(selected_turns[0]["turn_id"]),
        "parent_tail_turn_id": str(selected_turns[-1]["turn_id"]),
        "selection_rule": "last_complete_parent_sessions",
        "append_only": True,
    }
    result["tail_context_sha256"] = _sha(result)
    return result


def build_tail_context_from_expanded_dialogue(
    expanded: Mapping[str, Any], *, session_count: int = 2,
) -> dict[str, Any]:
    """Select final generated/source sessions from a frozen complete dialogue."""
    if session_count not in {1, 2}:
        raise ValueError("simple_append_session_count_must_be_one_or_two")
    turns = expanded.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("simple_append_expanded_tail_turns_required")
    session_order: list[str] = []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in turns:
        if not isinstance(row, Mapping):
            raise ValueError("simple_append_expanded_tail_turn_invalid")
        session_ref = str(row.get("session_ref") or "")
        turn_ref = str(row.get("turn_ref") or "")
        if not session_ref or not turn_ref:
            raise ValueError("simple_append_expanded_tail_turn_invalid")
        if session_ref not in grouped:
            session_order.append(session_ref)
            grouped[session_ref] = []
        grouped[session_ref].append(row)
    selected_ids = session_order[-min(session_count, len(session_order)):]
    selected_turns = [{
        "turn_id": str(row["turn_ref"]),
        "session_id": str(row["session_ref"]),
        "speaker": str(row["speaker"]),
        "text": str(row["text"]),
    } for session_ref in selected_ids for row in grouped[session_ref]]
    speakers = list(dict.fromkeys(str(row.get("speaker") or "")
                                  for row in turns))
    if any(not value for value in speakers):
        raise ValueError("simple_append_expanded_tail_speaker_invalid")
    result = {
        "tail_context_version": MULTI_ROUND_TAIL_CONTEXT_VERSION,
        "root_id": str(expanded.get("root_id") or ""),
        "parent_dialogue_sha256": str(
            expanded.get("complete_dialogue_sha256") or ""),
        "parent_session_count": len(session_order),
        "parent_turn_count": len(turns),
        "selected_session_count": len(selected_ids),
        "selected_session_ids": selected_ids,
        "selected_session_summaries": [{
            "session_id": session_ref,
            "source_session_index": session_order.index(session_ref),
            "summary": "Frozen preceding dialogue session.",
        } for session_ref in selected_ids],
        "selected_turns": selected_turns,
        "selected_turn_count": len(selected_turns),
        "parent_speakers": speakers,
        "first_selected_turn_id": str(selected_turns[0]["turn_id"]),
        "parent_tail_turn_id": str(selected_turns[-1]["turn_id"]),
        "selection_rule": "last_complete_sessions_from_accumulated_dialogue",
        "append_only": True,
    }
    if not result["root_id"] or not result["parent_dialogue_sha256"]:
        raise ValueError("simple_append_expanded_tail_identity_invalid")
    result["tail_context_sha256"] = _sha(result)
    return result
