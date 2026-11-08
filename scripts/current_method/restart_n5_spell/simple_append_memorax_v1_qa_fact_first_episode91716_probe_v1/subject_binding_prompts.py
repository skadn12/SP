"""Slim prompt for deterministic subject binding selection."""

from __future__ import annotations

from typing import Any, Mapping

from ..core import sha256_bytes
from .subject_binding_contract import PER_WINDOW_FACT_HARD_MAX, SOURCE_POSITION_MAX


SUBJECT_BINDING_PROMPT_VERSION = "local_atomic_fact_subject_binding_prompt_v1"


def build_subject_binding_prompt(
    window: Mapping[str, Any], *, entity_refs: list[str],
) -> dict[str, Any]:
    lines = [
        f"{turn.get('global_position')} | {turn.get('speaker')} | {turn.get('text')}"
        for turn in window.get("turns") or []
    ]
    prompt = "\n".join([
        "ROLE: transcript-grounded atomic Fact Extractor.",
        "You see exactly one natural Parent session or Child scene.",
        "Extract only concrete facts directly stated in these turns.",
        "Do not infer motives, values, lessons, causes, dependencies, or reasoning types.",
        "Do not compare this window with another window and do not write questions.",
        "Keep each fact atomic and use only the smallest supporting positions.",
        "The program will resolve canonical turn IDs and all mechanical metadata.",
        "For each fact, bind its subject using exactly one allowed mode:",
        "- speaker_at_position: use when the fact is about what that speaker says or does.",
        "  The position must be one of the fact's source_positions.",
        "- known_entity: use when a speaker states a fact about another known entity.",
        "  entity_ref must exactly match one canonical label in KNOWN_ENTITY_REFS.",
        "Do not output a free subject field. If the subject cannot be bound safely, omit the fact.",
        "Example: if Neighbors B says 'I apologized', bind speaker_at_position to that turn.",
        "Example: if Neighbors B says 'Neighbors A resumed practice', bind known_entity to Neighbors A.",
        "Return only one JSON object.",
        "",
        "OUTPUT CONTRACT",
        "Top-level keys: exactly facts.",
        f"facts contains 0-{PER_WINDOW_FACT_HARD_MAX} items; no minimum quota.",
        "Each item has exactly fact_text, source_positions, subject_binding.",
        f"source_positions has 1-{SOURCE_POSITION_MAX} unique global positions from this window.",
        "subject_binding is exactly {mode, position} or {mode, entity_ref}.",
        "Do not output IDs, canonical turn IDs, scope, subject, questions, answers, options, or distractors.",
        "Empty facts is valid when no precise atomic fact can be safely extracted.",
        "",
        f"KNOWN_ENTITY_REFS: {', '.join(entity_refs)}",
        f"WINDOW {window.get('window_id')} ({window.get('window_kind')})",
        *lines,
        "",
        "Return only the JSON object with facts.",
    ])
    return {
        "prompt_version": SUBJECT_BINDING_PROMPT_VERSION,
        "window_id": window.get("window_id"),
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "character_count": len(prompt),
        "token_estimate": max(1, (len(prompt) + 3) // 4),
        "window_turn_count": len(window.get("turns") or []),
        "allowed_global_positions": list(window.get("global_positions") or []),
        "output_fields": ["fact_text", "source_positions", "subject_binding"],
    }
