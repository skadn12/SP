"""Mechanical local atomic-fact schema and deterministic materialization."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


FACT_SCHEMA_VERSION = "local_atomic_fact_v1"
TOP_KEYS = {"facts"}
FACT_KEYS = {"fact_text", "source_positions", "subject"}
PER_WINDOW_FACT_HARD_MAX = 8
SOURCE_POSITION_MIN = 1
SOURCE_POSITION_MAX = 3


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_fact_extractor_output(
    value: Any, *, expanded_dialogue: Mapping[str, Any], window: Mapping[str, Any],
) -> dict[str, Any]:
    envelope_errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != TOP_KEYS:
        envelope_errors.append("fact_top_level_schema")
        value = value if isinstance(value, Mapping) else {}
    facts = value.get("facts")
    if not isinstance(facts, list):
        envelope_errors.append("facts_schema")
        facts = []
    if len(facts) > PER_WINDOW_FACT_HARD_MAX:
        envelope_errors.append("fact_capacity_exceeded")

    full_turns = {
        position: turn
        for position, turn in enumerate(expanded_dialogue.get("turns") or [], 1)
        if isinstance(turn, Mapping)
    }
    allowed = set(window.get("global_positions") or [])
    items: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for index, raw in enumerate(facts, 1):
        errors = list(envelope_errors)
        row = raw if isinstance(raw, Mapping) else {}
        if not isinstance(raw, Mapping) or set(row) != FACT_KEYS:
            errors.append("fact_candidate_schema")
        text = row.get("fact_text")
        if not _nonempty_text(text):
            errors.append("fact_text_empty")
        subject = row.get("subject")
        if not _nonempty_text(subject):
            errors.append("subject_empty")
        positions = row.get("source_positions")
        position_schema_valid = (
            isinstance(positions, list) and
            all(isinstance(pos, int) and not isinstance(pos, bool) for pos in positions)
        )
        if not position_schema_valid:
            errors.append("source_positions_schema")
            positions = []
        if len(positions) < SOURCE_POSITION_MIN:
            errors.append("source_positions_below_minimum")
        if len(positions) > SOURCE_POSITION_MAX:
            errors.append("source_positions_above_maximum")
        if len(positions) != len(set(positions)):
            errors.append("source_positions_duplicate")
        if any(position not in full_turns for position in positions):
            errors.append("source_position_out_of_range")
        if any(position not in allowed for position in positions):
            errors.append("source_position_outside_window")
        source_turn_ids = [
            str(full_turns[position].get("turn_ref") or "")
            for position in positions if position in full_turns and position in allowed
        ]
        item = {
            "local_fact_index": index,
            "window_id": window.get("window_id"),
            "valid": not errors,
            "mechanical_errors": sorted(set(errors)),
            "fact_text": str(text).strip() if isinstance(text, str) else text,
            "source_positions": list(positions),
            "source_turn_ids": source_turn_ids,
            "subject": str(subject).strip() if isinstance(subject, str) else subject,
        }
        items.append(item)
        if item["valid"]:
            valid.append(dict(item))
    return {
        "schema_version": FACT_SCHEMA_VERSION,
        "window_id": window.get("window_id"),
        "window_kind": window.get("window_kind"),
        "boundary_ref": window.get("boundary_ref"),
        "envelope_valid": not envelope_errors,
        "envelope_errors": sorted(set(envelope_errors)),
        "raw_fact_count": len(facts),
        "mechanically_valid_count": len(valid),
        "quarantine_count": len(facts) - len(valid),
        "items": items,
        "valid_facts": valid,
        "empty_facts_valid": not facts and not envelope_errors,
    }


def materialize_facts(
    validations: Sequence[Mapping[str, Any]],
    *, windows_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for validation in validations:
        window_id = str(validation.get("window_id") or "")
        window = windows_by_id.get(window_id)
        if window is None:
            raise ValueError(f"materialization_window_missing:{window_id}")
        for fact in validation.get("valid_facts") or []:
            fact_id = f"F{len(materialized) + 1:04d}"
            materialized.append({
                "fact_id": fact_id,
                "fact_text": fact["fact_text"],
                "source_positions": list(fact["source_positions"]),
                "source_turn_ids": list(fact["source_turn_ids"]),
                "window_id": window_id,
                "window_kind": window["window_kind"],
                "window_order": window["window_order"],
                "boundary_ref": window["boundary_ref"],
                "origin": window["origin"],
                "subject": fact["subject"],
            })
    return materialized
