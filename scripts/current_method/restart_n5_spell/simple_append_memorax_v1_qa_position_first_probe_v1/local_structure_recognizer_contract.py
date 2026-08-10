"""Mechanical per-window contract for local structure recognition."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from .structure_miner_contract import (
    CANDIDATE_KEYS, EVIDENCE_MAX, EVIDENCE_MIN, STRUCTURE_TYPES, TOP_KEYS,
)


LOCAL_RECOGNIZER_SCHEMA_VERSION = "local_structure_recognizer_v1"
PER_WINDOW_HARD_MAX = 5


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_local_recognizer_output(
    value: Any, *, expanded_dialogue: Mapping[str, Any], window: Mapping[str, Any],
) -> dict[str, Any]:
    envelope_errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != TOP_KEYS:
        envelope_errors.append("recognizer_top_level_schema")
        value = value if isinstance(value, Mapping) else {}
    structures = value.get("structures")
    if not isinstance(structures, list):
        envelope_errors.append("structures_schema")
        structures = []
    if len(structures) > PER_WINDOW_HARD_MAX:
        envelope_errors.append("per_window_capacity_exceeded")

    full_turns = {
        position: turn
        for position, turn in enumerate(expanded_dialogue.get("turns") or [], 1)
        if isinstance(turn, Mapping)
    }
    allowed_positions = set(window.get("global_positions") or [])
    items: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for index, raw in enumerate(structures, 1):
        errors = list(envelope_errors)
        row = raw if isinstance(raw, Mapping) else {}
        if not isinstance(raw, Mapping) or set(row) != CANDIDATE_KEYS:
            errors.append("structure_candidate_schema")
        structure_type = str(row.get("structure_type") or "")
        if structure_type not in STRUCTURE_TYPES:
            errors.append("structure_type_not_allowed")
        positions = row.get("evidence_positions")
        position_schema_valid = (
            isinstance(positions, list) and
            all(isinstance(pos, int) and not isinstance(pos, bool) for pos in positions)
        )
        if not position_schema_valid:
            errors.append("evidence_positions_schema")
            positions = []
        if len(positions) < EVIDENCE_MIN:
            errors.append("evidence_below_minimum")
        if len(positions) > EVIDENCE_MAX:
            errors.append("evidence_above_maximum")
        if len(positions) != len(set(positions)):
            errors.append("evidence_duplicate_positions")
        if any(pos not in full_turns for pos in positions):
            errors.append("evidence_position_out_of_range")
        if any(pos not in allowed_positions for pos in positions):
            errors.append("evidence_position_outside_window")
        variable = row.get("discriminative_variable")
        if not _text(variable):
            errors.append("discriminative_variable_empty")
        canonical = [
            str(full_turns[pos].get("turn_ref"))
            for pos in positions if pos in full_turns and pos in allowed_positions
        ]
        item = {
            "candidate_id": f"{window.get('window_id')}:SM{index:02d}",
            "window_id": window.get("window_id"),
            "valid": not errors,
            "mechanical_errors": sorted(set(errors)),
            "structure_type": structure_type,
            "evidence_positions": list(positions),
            "evidence_count": len(positions),
            "evidence_span": (
                max(positions) - min(positions) if positions else None
            ),
            "canonical_turn_ids": canonical,
            "discriminative_variable": (
                str(variable).strip() if isinstance(variable, str) else variable
            ),
        }
        items.append(item)
        if item["valid"]:
            valid.append(dict(item))
    return {
        "schema_version": LOCAL_RECOGNIZER_SCHEMA_VERSION,
        "window_id": window.get("window_id"),
        "window_kind": window.get("window_kind"),
        "boundary_ref": window.get("boundary_ref"),
        "envelope_valid": not envelope_errors,
        "envelope_errors": sorted(set(envelope_errors)),
        "raw_structure_count": len(structures),
        "mechanically_valid_count": len(valid),
        "quarantine_count": len(structures) - len(valid),
        "items": items,
        "valid_structures": valid,
        "structure_type_distribution": dict(Counter(
            row["structure_type"] for row in valid
        )),
        "empty_structures_valid": not structures and not envelope_errors,
    }
