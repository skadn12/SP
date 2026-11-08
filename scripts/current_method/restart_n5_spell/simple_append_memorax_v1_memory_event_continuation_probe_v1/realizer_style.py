"""Pure text-free continuity and style context for Realizer prompts."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


_REALIZER_CONTINUITY_TAIL_MAX_TURNS = 4
_REALIZER_CONTINUITY_TAIL_MAX_CHARS = 800


def _median_int(values: Sequence[int]) -> int:
    """Return a deterministic integer median without adding a statistics dependency."""
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return 0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def _build_realizer_style_context(
        turns: Sequence[Mapping[str, Any]], *,
        max_turns: int = _REALIZER_CONTINUITY_TAIL_MAX_TURNS,
        max_chars: int = _REALIZER_CONTINUITY_TAIL_MAX_CHARS,
) -> dict[str, Any]:
    """Represent the recent tail as text-free continuity and style statistics.

    Recent text is inspected locally but never exposed to Realizer. This keeps
    speaker/turn rhythm without creating a second source for copied dialogue.
    """
    if int(max_turns) < 1 or int(max_chars) < 1:
        raise ValueError("realizer_style_context_limit_invalid")
    selected: list[Mapping[str, Any]] = []
    used_chars = 0
    for row in reversed(list(turns)):
        text = str(row.get("text") or "")
        candidate_chars = len(text)
        if selected and used_chars + candidate_chars > int(max_chars):
            break
        selected.insert(0, row)
        used_chars += candidate_chars
        if len(selected) >= int(max_turns):
            break

    recent_turns = list(turns[-24:])
    text_lengths = [len(str(row.get("text") or "")) for row in recent_turns]
    sentence_counts = [
        max(1, len([part for part in re.split(r"[.!?]+", str(
            row.get("text") or "")) if part.strip()]))
        for row in recent_turns
        if str(row.get("text") or "").strip()
    ]
    speakers = [str(row.get("speaker") or "") for row in recent_turns]
    adjacent_repeats = sum(
        left == right for left, right in zip(speakers, speakers[1:])
        if left and right)
    tail_shapes = []
    for row in selected:
        text = str(row.get("text") or "")
        sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
        tail_shapes.append({
            "turn_ref": str(row.get("turn_id") or ""),
            "speaker": str(row.get("speaker") or ""),
            "character_count": len(text),
            "word_count": len(text.split()),
            "sentence_count": max(1, len(sentences)) if text.strip() else 0,
            "ending": ("question" if text.rstrip().endswith("?") else
                       "exclamation" if text.rstrip().endswith("!") else
                       "statement"),
        })
    return {
        "style_context_version": "realizer_continuity_shape_profile_v1",
        "selection_rule": "last_complete_turn_shapes_max_4_max_800_characters",
        "continuity_tail": {
            "turn_count": len(tail_shapes),
            "char_count": used_chars,
            "turn_shapes": tail_shapes,
            "last_speaker": tail_shapes[-1]["speaker"] if tail_shapes else "",
        },
        # These objects deliberately contain no dialogue text or story facts.
        "style_profile": {
            "source_turn_count": len(recent_turns),
            "register": "natural_informal_dialogue",
            "speaker_turn_pattern": (
                "alternating" if len(speakers) > 1 and not adjacent_repeats
                else "mixed"),
            "turn_length_characters": {
                "median": _median_int(text_lengths),
                "minimum": min(text_lengths, default=0),
                "maximum": max(text_lengths, default=0),
            },
            "sentences_per_turn_median": _median_int(sentence_counts),
        },
    }


build_realizer_style_context = _build_realizer_style_context
