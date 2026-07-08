#!/usr/bin/env python3
"""Compatibility subset for the historical graph-expansion challenger.

Only the APIs used by the restored r2 LLM graph-expansion generator are
implemented here.  Full v1/v1.2 generation and evaluation logic was not present
in the cleaned workspace.
"""

from __future__ import annotations

import hashlib
from typing import Any


GENERATED_DIALOGUE_LEAK_PATTERNS = (
    "gold answer",
    "correct answer",
    "answer key",
    "option letter",
)
QUESTION_LEAK_PATTERNS = (
    "the dialogue says the answer",
    "choose the answer",
    "which option is correct",
)


def md5_text(text: str) -> str:
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()


def flatten_source_records(payload: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        for key in ("results", "records", "samples", "items", "questions", "data"):
            nested = value.get(key)
            if isinstance(nested, list):
                visit(nested)
                return
        if any(
            key in value
            for key in (
                "sample_id",
                "sample_name",
                "sample_index",
                "qa_index",
                "timeline_events",
                "state_updates",
                "turns",
                "new_turns",
                "dialogue",
                "conversation",
                "qa",
            )
        ):
            records.append(value)

    visit(payload)
    return records


def get_dialogue_turns(dialogue_record: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("new_turns", "turns", "dialogue", "conversation", "conversation_turns", "messages"):
        value = dialogue_record.get(key)
        if isinstance(value, list):
            return [turn for turn in value if isinstance(turn, dict)]
    return []


def dialogue_turn_id(turn: dict[str, Any]) -> str:
    return str(
        turn.get("dia_id")
        or turn.get("turn_id")
        or turn.get("id")
        or turn.get("turn_key")
        or turn.get("timestamp")
        or ""
    )


def dialogue_turn_text_line(turn: dict[str, Any]) -> str:
    turn_id = dialogue_turn_id(turn) or f"D1:{turn.get('turn_index') or 0}"
    speaker = str(turn.get("speaker") or turn.get("role") or turn.get("name") or "Speaker")
    text = str(turn.get("text") or turn.get("content") or turn.get("utterance") or "")
    return f"[{turn_id}] {speaker}: {text}"


def evaluate_samples(_samples: list[dict[str, Any]], _args: Any) -> dict[str, Any]:
    raise RuntimeError(
        "Historical run_graph_expansion_challenger_v1.evaluate_samples is not restored. "
        "Use scripts/eval_v13_pilot.py for solver evaluation, or pass --skip-solver-eval "
        "to generate_r2_llm_graph_expansion_challenger.py."
    )
