"""Read a promoted complete dialogue as authoritative history for R2+."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping


INCREMENTAL_ROUND_INPUT_VERSION = "memory_event_incremental_round_input_v1"
_ROUND_REF = re.compile(r"^R(\d+):S\d+(?::T\d+)?$")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_prior_expanded_dialogue(
    value: Mapping[str, Any], *, round_index: int,
) -> dict[str, Any]:
    errors: list[str] = []
    turns = value.get("turns")
    if not isinstance(turns, list) or not turns:
        return {"valid": False, "errors": ["prior_turns_required"]}
    if not str(value.get("root_id") or ""):
        errors.append("prior_root_id_required")
    if int(value.get("turn_count") or 0) != len(turns):
        errors.append("prior_turn_count_mismatch")
    refs: list[str] = []
    session_refs: list[str] = []
    round_indices: set[int] = set()
    for index, row in enumerate(turns, 1):
        if not isinstance(row, Mapping):
            errors.append(f"prior_turn_{index}_object")
            continue
        ref = str(row.get("turn_ref") or "")
        session_ref = str(row.get("session_ref") or "")
        if not ref or not session_ref or not str(row.get("speaker") or "").strip() \
                or not str(row.get("text") or "").strip():
            errors.append(f"prior_turn_{index}_shape")
        refs.append(ref)
        session_refs.append(session_ref)
        match = _ROUND_REF.match(ref)
        if match:
            round_indices.add(int(match.group(1)))
    if len(refs) != len(set(refs)):
        errors.append("prior_turn_refs_duplicate")
    expected_prior = int(round_index) - 1
    if int(round_index) < 2:
        errors.append("incremental_round_index_invalid")
    if not round_indices or max(round_indices) != expected_prior:
        errors.append("prior_latest_round_mismatch")
    if any(index >= int(round_index) for index in round_indices):
        errors.append("prior_contains_target_or_future_round")
    frozen_sha = str(value.get("complete_dialogue_sha256") or "")
    if len(frozen_sha) != 64:
        errors.append("prior_complete_dialogue_sha256")
    else:
        body = deepcopy(dict(value))
        body.pop("complete_dialogue_sha256", None)
        if _sha(body) != frozen_sha:
            errors.append("prior_complete_dialogue_sha256_mismatch")
    return {
        "valid": not errors,
        "errors": errors,
        "turn_count": len(turns),
        "session_count": len(dict.fromkeys(session_refs)),
        "prior_round_indices": sorted(round_indices),
        "expected_round_index": int(round_index),
    }


def build_round_history_parent(
    prior_expanded: Mapping[str, Any], *, round_index: int,
) -> dict[str, Any]:
    validation = validate_prior_expanded_dialogue(
        prior_expanded, round_index=round_index)
    if not validation["valid"]:
        raise ValueError("incremental_round_input_invalid:" + ",".join(
            validation["errors"]))
    root_id = str(prior_expanded["root_id"])
    source_turns = list(prior_expanded["turns"])
    session_order = list(dict.fromkeys(
        str(row["session_ref"]) for row in source_turns))
    session_index = {session_ref: index for index, session_ref in
                     enumerate(session_order, 1)}
    turns = [{
        "turn_id": str(row["turn_ref"]),
        "session_id": str(row["session_ref"]),
        "speaker": str(row["speaker"]),
        "text": str(row["text"]),
        "partition": "history",
        "source_session_index": session_index[str(row["session_ref"])],
        "source_turn_index": index,
        "source_origin": str(row.get("origin") or ""),
    } for index, row in enumerate(source_turns, 1)]
    metadata = [{
        "session_id": session_ref,
        "source_session_index": index,
        "summary": "Frozen prior dialogue session.",
        "turn_ids": [str(row["turn_ref"]) for row in source_turns
                     if str(row["session_ref"]) == session_ref],
    } for index, session_ref in enumerate(session_order, 1)]
    return {
        "round_history_view_version": INCREMENTAL_ROUND_INPUT_VERSION,
        "root_id": root_id,
        "dialogue": {
            "dialogue_hash": str(
                prior_expanded["complete_dialogue_sha256"]),
            "session_count": len(metadata),
            "turn_count": len(turns),
            "turns": turns,
        },
        "session_metadata": metadata,
        "target_round_index": int(round_index),
        "prior_expanded_dialogue": deepcopy(dict(prior_expanded)),
    }


def build_incremental_context(
    parent: Mapping[str, Any], *, token_counter: Callable[[str], int],
    budget_tokens: int = 10_000,
) -> dict[str, Any]:
    turns = list(parent["dialogue"]["turns"])
    text = "\n".join(str(row["text"]) for row in turns)
    raw_tokens = int(token_counter(text))
    if raw_tokens > int(budget_tokens):
        raise ValueError("incremental_round_history_exceeds_budget")
    sessions = list(parent["session_metadata"])
    return {
        "tail_context_version": "incremental_round_full_history_context_v1",
        "root_id": str(parent["root_id"]),
        "parent_dialogue_sha256": str(parent["dialogue"]["dialogue_hash"]),
        "parent_session_count": len(sessions),
        "parent_turn_count": len(turns),
        "selected_session_count": len(sessions),
        "selected_session_ids": [str(row["session_id"]) for row in sessions],
        "selected_turns": deepcopy(turns),
        "selected_turn_count": len(turns),
        "parent_speakers": list(dict.fromkeys(
            str(row["speaker"]) for row in turns)),
        "first_selected_turn_id": str(turns[0]["turn_id"]),
        "parent_tail_turn_id": str(turns[-1]["turn_id"]),
        "selection_rule": "full_incremental_history_within_budget",
        "raw_dialogue_context_budget_tokens": int(budget_tokens),
        "full_parent_raw_dialogue_tokens": raw_tokens,
        "selected_raw_dialogue_tokens": raw_tokens,
        "full_parent_included": True,
        "complete_sessions_only": True,
        "append_only": True,
    }


def load_incremental_round_input(
    path: Path, *, round_index: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prior = json.loads(path.read_text(encoding="utf-8"))
    validation = validate_prior_expanded_dialogue(
        prior, round_index=round_index)
    parent = build_round_history_parent(prior, round_index=round_index)
    return parent, prior, validation
