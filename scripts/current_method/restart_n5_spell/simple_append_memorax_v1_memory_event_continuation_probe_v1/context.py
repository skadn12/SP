"""Bounded complete-session context views for the isolated Fresh R1 probe."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

from ..unified_dialogue_root import validate_unified_dialogue_root


FULL_R1_CONTEXT_VERSION = "simple_append_full_r1_context_v1"
QA_CONTEXT_VIEW_VERSION = "simple_append_full_r1_qa_context_v1"
RAW_DIALOGUE_CONTEXT_BUDGET_TOKENS = 4096

TokenCounter = Callable[[str], int]


def _sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def raw_dialogue_text(turns: Sequence[Mapping[str, Any]]) -> str:
    """Serialize only utterance text, matching the probe's raw-token metric."""
    return "\n".join(str(row.get("text") or "") for row in turns)


def _count(turns: Sequence[Mapping[str, Any]], token_counter: TokenCounter) -> int:
    return int(token_counter(raw_dialogue_text(turns)))


def _select_complete_sessions(
    *, session_order: Sequence[str], grouped: Mapping[str, list[Mapping[str, Any]]],
    token_counter: TokenCounter, budget_tokens: int,
) -> tuple[list[str], list[Mapping[str, Any]]]:
    selected_reversed: list[str] = []
    selected_turns: list[Mapping[str, Any]] = []
    for session_id in reversed(list(session_order)):
        candidate_ids = list(reversed(selected_reversed + [session_id]))
        candidate_turns = [
            row for current in candidate_ids for row in grouped[current]
        ]
        if selected_reversed and _count(candidate_turns, token_counter) > budget_tokens:
            break
        selected_reversed.append(session_id)
        selected_turns = candidate_turns
        if _count(candidate_turns, token_counter) > budget_tokens:
            # A single oversized session remains intact by contract.
            break
    selected_ids = list(reversed(selected_reversed))
    return selected_ids, selected_turns


def build_fresh_r1_context(
    parent: Mapping[str, Any], *, token_counter: TokenCounter,
    budget_tokens: int = RAW_DIALOGUE_CONTEXT_BUDGET_TOKENS,
) -> dict[str, Any]:
    """Use the full Parent when it fits, otherwise retain complete tail sessions."""
    if int(budget_tokens) <= 0:
        raise ValueError("full_r1_context_budget_invalid")
    validation = validate_unified_dialogue_root(parent)
    if not validation["valid"]:
        raise ValueError(
            "full_r1_parent_invalid:" + ",".join(validation["errors"]))

    turns = list(parent["dialogue"]["turns"])
    sessions = list(parent["session_metadata"])
    session_order = [str(row["session_id"]) for row in sessions]
    grouped = {
        session_id: [row for row in turns if str(row["session_id"]) == session_id]
        for session_id in session_order
    }
    full_tokens = _count(turns, token_counter)
    if full_tokens <= int(budget_tokens):
        selected_ids = session_order
        selected_turns = turns
        selection_rule = "full_parent_within_raw_dialogue_budget"
    else:
        selected_ids, selected_turns = _select_complete_sessions(
            session_order=session_order, grouped=grouped,
            token_counter=token_counter, budget_tokens=int(budget_tokens))
        selection_rule = "complete_tail_sessions_within_raw_dialogue_budget"
    metadata_by_id = {str(row["session_id"]): row for row in sessions}
    selected_tokens = _count(selected_turns, token_counter)
    speakers = list(dict.fromkeys(str(row["speaker"]) for row in turns))
    result = {
        "tail_context_version": FULL_R1_CONTEXT_VERSION,
        "root_id": str(parent["root_id"]),
        "parent_dialogue_sha256": str(parent["dialogue"]["dialogue_hash"]),
        "parent_session_count": int(parent["dialogue"]["session_count"]),
        "parent_turn_count": int(parent["dialogue"]["turn_count"]),
        "selected_session_count": len(selected_ids),
        "selected_session_ids": selected_ids,
        "selected_session_summaries": [{
            "session_id": session_id,
            "source_session_index": int(
                metadata_by_id[session_id]["source_session_index"]),
            "summary": str(metadata_by_id[session_id]["summary"]),
        } for session_id in selected_ids],
        "selected_turns": deepcopy(selected_turns),
        "selected_turn_count": len(selected_turns),
        "parent_speakers": speakers,
        "first_selected_turn_id": str(selected_turns[0]["turn_id"]),
        "parent_tail_turn_id": str(selected_turns[-1]["turn_id"]),
        "selection_rule": selection_rule,
        "raw_dialogue_context_budget_tokens": int(budget_tokens),
        "full_parent_raw_dialogue_tokens": full_tokens,
        "selected_raw_dialogue_tokens": selected_tokens,
        "full_parent_included": len(selected_turns) == len(turns),
        "complete_sessions_only": True,
        "append_only": True,
    }
    result["tail_context_sha256"] = _sha(result)
    return result


def select_fresh_r1_qa_context(
    expanded: Mapping[str, Any], *, token_counter: TokenCounter,
    budget_tokens: int = RAW_DIALOGUE_CONTEXT_BUDGET_TOKENS,
) -> dict[str, Any]:
    """Return a QA dialogue view without splitting any source/generated session."""
    turns = expanded.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("full_r1_qa_turns_required")
    session_order: list[str] = []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in turns:
        if not isinstance(row, Mapping):
            raise ValueError("full_r1_qa_turn_invalid")
        session_id = str(row.get("session_ref") or row.get("session_id") or "")
        if not session_id:
            raise ValueError("full_r1_qa_session_ref_required")
        if session_id not in grouped:
            session_order.append(session_id)
            grouped[session_id] = []
        grouped[session_id].append(row)
    full_tokens = _count(turns, token_counter)
    if full_tokens <= int(budget_tokens):
        selected_ids = session_order
        selected_turns = list(turns)
        selection_rule = "full_parent_child_within_raw_dialogue_budget"
    else:
        selected_ids, selected_turns = _select_complete_sessions(
            session_order=session_order, grouped=grouped,
            token_counter=token_counter, budget_tokens=int(budget_tokens))
        selection_rule = "complete_tail_sessions_within_raw_dialogue_budget"
    view = deepcopy(dict(expanded))
    view["turns"] = deepcopy(selected_turns)
    view["turn_count"] = len(selected_turns)
    view["parent_turn_count"] = sum(
        str(row.get("origin") or "") == "source_grounded"
        for row in selected_turns)
    view["child_turn_count"] = sum(
        str(row.get("origin") or "") == "generated_extension"
        for row in selected_turns)
    view["qa_context_selection"] = {
        "qa_context_view_version": QA_CONTEXT_VIEW_VERSION,
        "selection_rule": selection_rule,
        "raw_dialogue_context_budget_tokens": int(budget_tokens),
        "full_parent_child_raw_dialogue_tokens": full_tokens,
        "selected_raw_dialogue_tokens": _count(selected_turns, token_counter),
        "full_parent_child_included": len(selected_turns) == len(turns),
        "selected_session_ids": selected_ids,
        "selected_turn_count": len(selected_turns),
        "complete_sessions_only": True,
    }
    return view
