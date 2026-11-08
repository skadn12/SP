"""Auditable exact-dialogue duplication checks, without model calls.

This detects copied speech, not semantic equivalence or whether a new event was
realized. Short speech remains visible in diagnostics but cannot alone reject a
scene. References may contain frozen history and accepted additions from either
route; reference identifiers never affect text matching.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


POLICY_VERSION = "dialogue_exact_duplicate_v1"


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _prepare(turns: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    if not isinstance(turns, list):
        raise ValueError(f"{name}: turns must be a list")
    prepared = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"{name} turn {index + 1}: expected an object")
        speaker, text = turn.get("speaker"), turn.get("text")
        if not isinstance(speaker, str) or not _normalize(speaker):
            raise ValueError(f"{name} turn {index + 1}: missing speaker")
        if not isinstance(text, str) or not _normalize(text):
            raise ValueError(f"{name} turn {index + 1}: missing text")
        prepared.append({
            "key": (_normalize(speaker), _normalize(text)),
            "speaker": speaker,
            "text": text,
            "turn_ref": turn.get("turn_ref") or f"{name}:T{index + 1:03d}",
            "content_chars": len("".join(text.split())),
        })
    return prepared


def inspect_duplicates(
    new_turns: list[dict[str, Any]],
    references: dict[str, list[dict[str, Any]]],
    *,
    min_content_chars: int = 40,
    min_consecutive: int = 3,
    max_duplicate_fraction: float = 0.5,
    min_repeated_turns: int = 4,
) -> dict[str, Any]:
    """Return PASS/REJECT, exact matches, copied spans, ratios and thresholds.

    ``turn_ref`` is optional; one-based source-local positions are always given.
    Match keys consist of speaker and text with whitespace normalized, retaining
    case and punctuation. ``min_content_chars`` excludes whitespace. A maximal
    contiguous copied span triggers rejection when it contains at least
    ``min_consecutive`` substantive turns (short intervening copied turns are
    included in its reported length). Noncontiguous copying triggers rejection
    at the configured substantive fraction AND minimum repeated-turn count.
    A new turn is counted only once even if it matches multiple source turns.
    Malformed inputs/thresholds raise ValueError rather than silently passing.
    """
    for name, value in (
        ("min_content_chars", min_content_chars),
        ("min_consecutive", min_consecutive),
        ("min_repeated_turns", min_repeated_turns),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (isinstance(max_duplicate_fraction, bool)
            or not isinstance(max_duplicate_fraction, (int, float))
            or not 0 < max_duplicate_fraction <= 1):
        raise ValueError("max_duplicate_fraction must be in (0, 1]")
    if not isinstance(references, dict):
        raise ValueError("references must map source names to turn lists")
    if any(not isinstance(name, str) or not name for name in references):
        raise ValueError("reference source names must be nonempty strings")

    new = _prepare(new_turns, "new")
    sources = {name: _prepare(turns, name) for name, turns in references.items()}
    index: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for source, turns in sources.items():
        for position, turn in enumerate(turns):
            index[turn["key"]].append((source, position))

    matches = []
    consecutive_spans = []
    for position, turn in enumerate(new):
        source_matches = index.get(turn["key"], [])
        if not source_matches:
            continue
        matches.append({
            "new_position": position + 1,
            "new_turn_ref": turn["turn_ref"],
            "speaker": turn["speaker"],
            "text": turn["text"],
            "content_chars": turn["content_chars"],
            "substantive": turn["content_chars"] >= min_content_chars,
            "sources": [
                {"source": source, "position": old_position + 1,
                 "turn_ref": sources[source][old_position]["turn_ref"]}
                for source, old_position in source_matches
            ],
        })
        for source, old_position in source_matches:
            old = sources[source]
            if (position > 0 and old_position > 0
                    and new[position - 1]["key"] == old[old_position - 1]["key"]):
                continue  # This pair belongs to an already-started maximal span.
            length = 0
            while (position + length < len(new) and old_position + length < len(old)
                   and new[position + length]["key"] == old[old_position + length]["key"]):
                length += 1
            substantive_count = sum(
                new[j]["content_chars"] >= min_content_chars
                for j in range(position, position + length)
            )
            consecutive_spans.append({
                "source": source,
                "new_start_position": position + 1,
                "new_end_position": position + length,
                "new_start_ref": turn["turn_ref"],
                "new_end_ref": new[position + length - 1]["turn_ref"],
                "source_start_position": old_position + 1,
                "source_end_position": old_position + length,
                "source_start_ref": old[old_position]["turn_ref"],
                "source_end_ref": old[old_position + length - 1]["turn_ref"],
                "length": length,
                "substantive_turns": substantive_count,
                "rejecting": substantive_count >= min_consecutive,
            })

    substantive_total = sum(t["content_chars"] >= min_content_chars for t in new)
    repeated_substantive = sum(match["substantive"] for match in matches)
    fraction = repeated_substantive / substantive_total if substantive_total else 0.0
    reasons = []
    if any(span["rejecting"] for span in consecutive_spans):
        reasons.append("CONSECUTIVE_SUBSTANTIVE_TURNS_COPIED")
    if repeated_substantive >= min_repeated_turns and fraction >= max_duplicate_fraction:
        reasons.append("SUBSTANTIVE_DUPLICATE_FRACTION_EXCEEDED")
    return {
        "status": "REJECT" if reasons else "PASS",
        "reasons": reasons,
        "policy": {
            "version": POLICY_VERSION,
            "matching": "same speaker and text; whitespace normalized; case/punctuation retained",
            "min_content_chars": min_content_chars,
            "min_consecutive": min_consecutive,
            "max_duplicate_fraction": max_duplicate_fraction,
            "min_repeated_turns": min_repeated_turns,
            "fraction_denominator": "substantive new turns",
        },
        "new_turn_count": len(new),
        "duplicate_turn_count": len(matches),
        "duplicate_fraction": len(matches) / len(new) if new else 0.0,
        "substantive_turn_count": substantive_total,
        "repeated_substantive_turn_count": repeated_substantive,
        "substantive_duplicate_fraction": fraction,
        "max_consecutive_span_length": max((s["length"] for s in consecutive_spans), default=0),
        "max_consecutive_substantive_count": max(
            (s["substantive_turns"] for s in consecutive_spans), default=0),
        "reference_turn_counts": {name: len(turns) for name, turns in sources.items()},
        "matches": matches,
        "consecutive_spans": consecutive_spans,
        "limitations": "Exact copying only; paraphrase and plan fulfillment require separate checks.",
    }
