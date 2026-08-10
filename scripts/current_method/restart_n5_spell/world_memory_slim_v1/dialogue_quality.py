"""Dialogue Quality Gate v1; deliberately ignorant of proposal claims."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Mapping, Sequence


DIALOGUE_QUALITY_VERSION = "dialogue_quality_gate_v1"
_TURN_PATTERN = re.compile(r"^R\d{2}:S\d{2}:T\d{3}$")


def normalize_exact_utterance(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).strip()
    value = re.sub(r"\s+", " ", value)
    pairs = (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’"))
    for left, right in pairs:
        if len(value) >= 2 and value.startswith(left) and value.endswith(right):
            value = value[len(left):-len(right)].strip()
            break
    return value


def validate_dialogue_quality(
    *, scene_ref: str, dialogue_turns: Any,
    speaker_participants: Sequence[Mapping[str, Any]], turn_budget: Mapping[str, Any],
    parent_utterances: Sequence[str] = (), previous_child_utterance_blacklist: Sequence[str] = (),
    roles_applicable: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(dialogue_turns, list) or not dialogue_turns:
        return {"valid": False, "errors": ["dialogue_turns_missing"],
                "quality_gate_version": DIALOGUE_QUALITY_VERSION}
    if not turn_budget["minimum"] <= len(dialogue_turns) <= turn_budget["maximum"]:
        errors.append("turn_count_outside_frozen_budget")
    planned = {str(row.get("participant_ref")): str(row.get("scene_role"))
               for row in speaker_participants}
    refs: list[str] = []
    speakers: list[str] = []
    normalized: list[str] = []
    for index, turn in enumerate(dialogue_turns):
        if not isinstance(turn, Mapping) or set(turn) != {"turn_ref", "speaker_ref", "text"}:
            errors.append(f"dialogue_turn_schema:{index}")
            continue
        ref = str(turn.get("turn_ref") or "")
        speaker = str(turn.get("speaker_ref") or "")
        text = turn.get("text")
        refs.append(ref)
        speakers.append(speaker)
        if not _TURN_PATTERN.fullmatch(ref) or not ref.startswith(scene_ref + ":T"):
            errors.append(f"dialogue_turn_ref:{index}")
        if speaker not in planned:
            errors.append(f"dialogue_speaker_unplanned:{index}")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"dialogue_text_empty:{index}")
        else:
            normalized.append(normalize_exact_utterance(text))
    if len(refs) != len(set(refs)):
        errors.append("dialogue_turn_ref_duplicate")
    counts = Counter(speakers)
    for speaker in planned:
        if counts[speaker] == 0:
            errors.append(f"planned_speaker_missing:{speaker}")
        if roles_applicable and planned[speaker] == "primary" and counts[speaker] < 2:
            errors.append(f"primary_speaker_undercovered:{speaker}")
        if roles_applicable and planned[speaker] == "supporting" and counts[speaker] < 1:
            errors.append(f"supporting_speaker_undercovered:{speaker}")
    longest_run = 0
    current_run = 0
    previous = None
    for speaker in speakers:
        current_run = current_run + 1 if speaker == previous else 1
        longest_run = max(longest_run, current_run)
        previous = speaker
    if longest_run > 2:
        errors.append("same_speaker_run_exceeds_two")
    parent = {normalize_exact_utterance(value) for value in parent_utterances}
    prior = {normalize_exact_utterance(value) for value in previous_child_utterance_blacklist}
    parent_copies = [value for value in normalized if value in parent]
    prior_copies = [value for value in normalized if value in prior]
    duplicates = [value for value, count in Counter(normalized).items() if count > 1]
    if parent_copies:
        errors.append("parent_exact_copy")
    if prior_copies:
        errors.append("previous_child_exact_copy")
    if duplicates:
        errors.append("scene_internal_exact_duplicate")
    return {
        "valid": not errors, "errors": errors,
        "quality_gate_version": DIALOGUE_QUALITY_VERSION,
        "actual_turn_count": len(dialogue_turns), "planned_speakers": sorted(planned),
        "actual_speakers": sorted(set(speakers)), "speaker_turn_counts": dict(sorted(counts.items())),
        "longest_same_speaker_run": longest_run,
        "parent_exact_copies": parent_copies,
        "previous_child_exact_copies": prior_copies,
        "internal_exact_duplicates": duplicates,
        "primary_supporting_coverage": "evaluated" if roles_applicable else "not_applicable",
        "role_inferred_for_shadow_audit": False,
    }

