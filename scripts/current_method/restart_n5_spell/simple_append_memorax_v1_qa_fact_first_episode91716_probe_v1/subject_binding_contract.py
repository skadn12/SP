"""Constrained subject binding for the episode-91716 fact extractor.

The model chooses a binding mode, while the program resolves the resulting
subject from frozen transcript metadata.  This removes the free-form subject
field that caused the legacy C01 attribution error.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


SUBJECT_BINDING_SCHEMA_VERSION = "local_atomic_fact_subject_binding_v1"
TOP_KEYS = {"facts"}
FACT_KEYS = {"fact_text", "source_positions", "subject_binding"}
BINDING_KEYS = {"mode", "position"}
ENTITY_BINDING_KEYS = {"mode", "entity_ref"}
VALID_MODES = {"speaker_at_position", "known_entity"}
PER_WINDOW_FACT_HARD_MAX = 8
SOURCE_POSITION_MIN = 1
SOURCE_POSITION_MAX = 3


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def known_entity_refs(expanded_dialogue: Mapping[str, Any]) -> list[str]:
    """Return the frozen canonical speaker labels available to the model."""

    return sorted({
        str(turn.get("speaker")).strip()
        for turn in expanded_dialogue.get("turns") or []
        if isinstance(turn, Mapping) and _text(turn.get("speaker"))
    })


def _turns_by_position(expanded_dialogue: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {
        position: turn
        for position, turn in enumerate(expanded_dialogue.get("turns") or [], 1)
        if isinstance(turn, Mapping)
    }


def _binding_error(
    binding: Any,
    *,
    fact_text: str | None,
    positions: Sequence[int],
    allowed_positions: set[int],
    turns_by_position: Mapping[int, Mapping[str, Any]],
    entity_refs: set[str],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    resolved: dict[str, Any] = {
        "mode": None,
        "subject": None,
        "asserting_speaker": None,
        "binding_position": None,
        "entity_ref": None,
    }
    if not isinstance(binding, Mapping):
        return ["subject_binding_schema"], resolved
    mode = binding.get("mode")
    if mode not in VALID_MODES:
        errors.append("subject_binding_mode_invalid")
        return errors, resolved
    resolved["mode"] = mode
    if mode == "speaker_at_position":
        if set(binding) != BINDING_KEYS:
            errors.append("speaker_binding_fields_invalid")
        position = binding.get("position")
        if not isinstance(position, int) or isinstance(position, bool):
            errors.append("speaker_binding_position_invalid")
        elif position not in allowed_positions:
            errors.append("speaker_binding_position_outside_window")
        elif position not in positions:
            errors.append("speaker_binding_position_not_in_evidence")
        else:
            speaker = turns_by_position[position].get("speaker")
            if not _text(speaker):
                errors.append("speaker_binding_speaker_missing")
            else:
                resolved["binding_position"] = position
                resolved["subject"] = str(speaker).strip()
                resolved["asserting_speaker"] = str(speaker).strip()
    else:
        if set(binding) != ENTITY_BINDING_KEYS:
            errors.append("entity_binding_fields_invalid")
        entity_ref = binding.get("entity_ref")
        if not _text(entity_ref):
            errors.append("entity_binding_entity_ref_invalid")
        elif str(entity_ref).strip() not in entity_refs:
            errors.append("entity_binding_entity_ref_unknown")
        else:
            resolved["entity_ref"] = str(entity_ref).strip()
            resolved["subject"] = str(entity_ref).strip()
            source_speakers = sorted({
                str(turns_by_position[pos].get("speaker")).strip()
                for pos in positions
                if pos in turns_by_position and _text(turns_by_position[pos].get("speaker"))
            })
            resolved["asserting_speaker"] = source_speakers[0] if len(source_speakers) == 1 else None
    # Lexical guard only: an explicitly named entity must agree with the
    # selected binding. Ambiguous cases remain quarantined for review.
    if not errors and isinstance(fact_text, str):
        mentioned = {
            entity for entity in entity_refs
            if entity and entity.casefold() in fact_text.casefold()
        }
        if mode == "speaker_at_position" and any(
            entity != resolved["subject"] for entity in mentioned
        ):
            errors.append("named_entity_requires_known_entity_binding")
        if mode == "known_entity" and any(
            entity != resolved["subject"] for entity in mentioned
        ):
            errors.append("named_entity_binding_conflict")
    return errors, resolved


def validate_subject_binding_output(
    value: Any,
    *,
    expanded_dialogue: Mapping[str, Any],
    window: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate v2 output and resolve subject metadata deterministically."""

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
    turns = _turns_by_position(expanded_dialogue)
    allowed = set(window.get("global_positions") or [])
    entity_refs = set(known_entity_refs(expanded_dialogue))
    items: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for index, raw in enumerate(facts, 1):
        errors = list(envelope_errors)
        row = raw if isinstance(raw, Mapping) else {}
        if not isinstance(raw, Mapping) or set(row) != FACT_KEYS:
            errors.append("fact_candidate_schema")
        if "subject" in row:
            errors.append("free_subject_forbidden")
        text = row.get("fact_text")
        if not _text(text):
            errors.append("fact_text_empty")
        positions = row.get("source_positions")
        position_schema_valid = (
            isinstance(positions, list)
            and all(isinstance(pos, int) and not isinstance(pos, bool) for pos in positions)
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
        if any(pos not in turns for pos in positions):
            errors.append("source_position_out_of_range")
        if any(pos not in allowed for pos in positions):
            errors.append("source_position_outside_window")
        binding_errors, resolved = _binding_error(
            row.get("subject_binding"),
            fact_text=text if isinstance(text, str) else None,
            positions=positions,
            allowed_positions=allowed, turns_by_position=turns,
            entity_refs=entity_refs,
        )
        errors.extend(binding_errors)
        source_turn_ids = [
            str(turns[pos].get("turn_ref") or "")
            for pos in positions if pos in turns and pos in allowed
        ]
        item = {
            "local_fact_index": index,
            "window_id": window.get("window_id"),
            "valid": not errors,
            "mechanical_errors": sorted(set(errors)),
            "fact_text": str(text).strip() if isinstance(text, str) else text,
            "source_positions": list(positions),
            "source_turn_ids": source_turn_ids,
            "subject_binding": row.get("subject_binding"),
            "subject": resolved["subject"],
            "asserting_speaker": resolved["asserting_speaker"],
            "binding_mode": resolved["mode"],
            "binding_position": resolved["binding_position"],
        }
        items.append(item)
        if item["valid"]:
            valid.append(dict(item))
    return {
        "schema_version": SUBJECT_BINDING_SCHEMA_VERSION,
        "window_id": window.get("window_id"),
        "window_kind": window.get("window_kind"),
        "boundary_ref": window.get("boundary_ref"),
        "entity_refs": sorted(entity_refs),
        "envelope_valid": not envelope_errors,
        "envelope_errors": sorted(set(envelope_errors)),
        "raw_fact_count": len(facts),
        "mechanically_valid_count": len(valid),
        "quarantine_count": len(facts) - len(valid),
        "items": items,
        "valid_facts": valid,
        "empty_facts_valid": not facts and not envelope_errors,
    }


def materialize_subject_bound_facts(
    validations: Sequence[Mapping[str, Any]],
    *,
    windows_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Assign stable fact IDs after validation; no model-provided IDs survive."""

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
                "asserting_speaker": fact["asserting_speaker"],
                "binding_mode": fact["binding_mode"],
                "binding_position": fact["binding_position"],
            })
    return materialized
