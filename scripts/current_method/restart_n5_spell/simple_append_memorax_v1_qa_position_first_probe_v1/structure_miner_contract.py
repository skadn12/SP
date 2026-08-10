"""Minimal mechanical contract for dialogue reasoning-structure mining."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


STRUCTURE_MINER_SCHEMA_VERSION = "structure_miner_minimal_v1"
STRUCTURE_HARD_MAX = 10
EVIDENCE_MIN = 2
EVIDENCE_MAX = 6
TOP_KEYS = {"structures"}
CANDIDATE_KEYS = {
    "structure_type", "evidence_positions", "discriminative_variable",
}
STRUCTURE_TYPES = {
    "temporal_competition",
    "revision",
    "considered_vs_adopted",
    "intended_vs_executed",
    "information_visibility",
    "dependency",
    "competing_causes_results",
    "same_dimension_comparison",
}


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _position_map(dialogue: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {
        index: turn
        for index, turn in enumerate(dialogue.get("turns") or [], 1)
        if isinstance(turn, Mapping)
    }


def validate_structure_miner_output(
    value: Any, *, expanded_dialogue: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate only schema and references; semantic truth remains an audit task."""
    envelope_errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != TOP_KEYS:
        envelope_errors.append("miner_top_level_schema")
        value = value if isinstance(value, Mapping) else {}
    structures = value.get("structures")
    if not isinstance(structures, list):
        envelope_errors.append("structures_schema")
        structures = []
    if len(structures) > STRUCTURE_HARD_MAX:
        envelope_errors.append("structure_capacity_exceeded")

    position_map = _position_map(expanded_dialogue)
    items: list[dict[str, Any]] = []
    valid_structures: list[dict[str, Any]] = []
    for index, raw in enumerate(structures, 1):
        errors = list(envelope_errors)
        row = raw if isinstance(raw, Mapping) else {}
        if not isinstance(raw, Mapping) or set(row) != CANDIDATE_KEYS:
            errors.append("structure_candidate_schema")
        structure_type = str(row.get("structure_type") or "")
        if structure_type not in STRUCTURE_TYPES:
            errors.append("structure_type_not_allowed")
        positions = row.get("evidence_positions")
        valid_position_schema = (
            isinstance(positions, list) and
            all(isinstance(pos, int) and not isinstance(pos, bool) for pos in positions)
        )
        if not valid_position_schema:
            errors.append("evidence_positions_schema")
            positions = []
        if len(positions) < EVIDENCE_MIN:
            errors.append("evidence_below_minimum")
        if len(positions) > EVIDENCE_MAX:
            errors.append("evidence_above_maximum")
        if len(positions) != len(set(positions)):
            errors.append("evidence_duplicate_positions")
        if any(pos not in position_map for pos in positions):
            errors.append("evidence_position_out_of_range")
        variable = row.get("discriminative_variable")
        if not _text(variable):
            errors.append("discriminative_variable_empty")

        turns = [position_map[pos] for pos in positions if pos in position_map]
        parent_sessions = sorted({
            str(turn.get("session_ref")) for turn in turns
            if turn.get("origin") == "source_grounded" and turn.get("session_ref")
        })
        child_scenes = sorted({
            str(turn.get("scene_ref")) for turn in turns
            if turn.get("origin") == "generated_extension" and turn.get("scene_ref")
        })
        origins = sorted({str(turn.get("origin")) for turn in turns})
        containers = sorted(set(parent_sessions + child_scenes))
        warnings: list[str] = []
        if len(parent_sessions) > 1:
            warnings.append("cross_parent_sessions")
        if len(child_scenes) > 1:
            warnings.append("cross_child_scenes")
        if len(origins) > 1:
            warnings.append("cross_parent_child")
        if len(containers) > 1:
            warnings.append("cross_story_containers")
        span = max(positions) - min(positions) if positions else None
        item = {
            "candidate_id": f"SM{index:02d}",
            "valid": not errors,
            "mechanical_errors": sorted(set(errors)),
            "audit_warnings": sorted(set(warnings)),
            "structure_type": structure_type,
            "evidence_positions": list(positions),
            "evidence_count": len(positions),
            "evidence_span": span,
            "canonical_turn_ids": [
                str(turn.get("turn_ref")) for turn in turns
            ],
            "parent_session_refs": parent_sessions,
            "child_scene_refs": child_scenes,
            "origins": origins,
            "discriminative_variable": (
                str(variable).strip() if isinstance(variable, str) else variable
            ),
        }
        items.append(item)
        if item["valid"]:
            valid_structures.append({
                "candidate_id": item["candidate_id"],
                "structure_type": structure_type,
                "evidence_positions": list(positions),
                "canonical_turn_ids": list(item["canonical_turn_ids"]),
                "discriminative_variable": str(variable).strip(),
                "evidence_count": len(positions),
                "evidence_span": span,
                "audit_warnings": list(item["audit_warnings"]),
            })
    return {
        "schema_version": STRUCTURE_MINER_SCHEMA_VERSION,
        "envelope_valid": not envelope_errors,
        "envelope_errors": sorted(set(envelope_errors)),
        "raw_structure_count": len(structures),
        "mechanically_valid_count": len(valid_structures),
        "quarantine_count": len(structures) - len(valid_structures),
        "items": items,
        "valid_structures": valid_structures,
        "structure_type_distribution": dict(Counter(
            row["structure_type"] for row in valid_structures
        )),
        "empty_structures_valid": not structures and not envelope_errors,
    }
