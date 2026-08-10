"""Compact prompt for local transcript-grounded atomic fact extraction."""

from __future__ import annotations

from typing import Any, Mapping

from ..core import sha256_bytes
from .fact_contract import PER_WINDOW_FACT_HARD_MAX, SOURCE_POSITION_MAX


FACT_PROMPT_VERSION = "local_atomic_fact_prompt_v1"


def build_fact_prompt(window: Mapping[str, Any]) -> dict[str, Any]:
    lines = [
        f"{turn.get('global_position')} | {turn.get('speaker')} | {turn.get('text')}"
        for turn in window.get("turns") or []
    ]
    prompt = "\n".join([
        "ROLE: transcript-grounded atomic Fact Extractor.",
        "You see exactly one natural Parent session or Child scene.",
        "Extract a small number of concrete facts stated directly in these turns.",
        "Do not summarize the window and do not infer motives, values, lessons,",
        "causes, dependencies, revisions, temporal relations, or question types.",
        "Do not compare this window with any other window.",
        "Prefer an exact action, plan, decision, event, explicit state, update,",
        "result, or explicitly stated information. Keep one fact atomic.",
        "Do not split repeated paraphrases of the same fact into separate facts.",
        "Use the exact global position numbers shown in the transcript.",
        "Use the concrete subject/entity label from the transcript where possible.",
        "A fact must be directly supported by its listed source positions.",
        "Return only one JSON object.",
        "",
        "OUTPUT CONTRACT",
        "Top-level keys: exactly facts.",
        f"facts contains 0-{PER_WINDOW_FACT_HARD_MAX} items; no minimum quota.",
        "Each item has exactly fact_text, source_positions, subject.",
        f"source_positions has 1-{SOURCE_POSITION_MAX} unique positions from this window.",
        "Do not output IDs, canonical turn IDs, scopes, relation/reasoning types,",
        "state-transition labels, questions, answers, options, or distractors.",
        "Empty facts is valid when no precise atomic fact can be extracted.",
        "",
        f"WINDOW {window.get('window_id')} ({window.get('window_kind')})",
        *lines,
        "",
        "Return only the JSON object with facts.",
    ])
    return {
        "prompt_version": FACT_PROMPT_VERSION,
        "window_id": window.get("window_id"),
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "character_count": len(prompt),
        "token_estimate": max(1, (len(prompt) + 3) // 4),
        "window_turn_count": len(window.get("turns") or []),
        "allowed_global_positions": list(window.get("global_positions") or []),
        "output_fields": ["fact_text", "source_positions", "subject"],
    }

