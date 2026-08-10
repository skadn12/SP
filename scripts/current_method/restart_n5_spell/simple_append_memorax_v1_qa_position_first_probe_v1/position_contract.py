"""Position-only semantic Writer contract and deterministic MCQ materializer."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ..core import normalize_option


POSITION_SCHEMA_VERSION = "position_distractor_provenance_writer_v2"
QUESTION_CAPACITY_MAX = 12
PREFERRED_MIN_EVIDENCE = 2
PREFERRED_MAX_EVIDENCE = 6
HARD_MAX_EVIDENCE = 8

TOP_KEYS = {"position_schema_version", "abstention_summary", "questions"}
QUESTION_KEYS = {
    "structure_type", "evidence_positions", "question", "options",
    "gold_text", "distractor_sources",
}
OPTION_KEYS = {"A", "B", "C", "D"}
SOURCE_SCOPES = {"parent_only", "child_only", "parent_child"}
QUESTION_TYPE_BY_STRUCTURE = {
    "temporal_sequence": "temporal_sequence",
    "state_change": "state_change",
    "information_revision": "information_revision",
    "information_visibility": "information_visibility",
    "dependency": "task_dependency_or_constraint",
    "task_dependency_or_constraint": "task_dependency_or_constraint",
    "causal": "causal",
    "competing_result": "causal",
    "comparison": "cross_turn_comparison",
    "cross_turn_comparison": "cross_turn_comparison",
    "intended_vs_executed": "cross_turn_comparison",
}


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _scope(turns: list[Mapping[str, Any]]) -> str | None:
    origins = {str(row.get("origin")) for row in turns}
    if origins == {"source_grounded"}:
        return "parent_only"
    if origins == {"generated_extension"}:
        return "child_only"
    if origins == {"source_grounded", "generated_extension"}:
        return "parent_child"
    return None


def _position_map(expanded_dialogue: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {
        index: row for index, row in enumerate(expanded_dialogue.get("turns") or [], 1)
        if isinstance(row, Mapping)
    }


def _container_key(turn: Mapping[str, Any]) -> str:
    """Return a deterministic local story container for provenance auditing."""
    return str(turn.get("scene_ref") or turn.get("session_ref") or "")


def _distractor_provenance_audit(
    source_map: Mapping[str, Any], *, positions: list[int],
    position_map: Mapping[int, Mapping[str, Any]], gold_key: str | None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Check mechanical provenance and report, without judging semantic near-miss quality."""
    errors: list[str] = []
    warnings: list[str] = []
    expected = OPTION_KEYS - ({gold_key} if gold_key else set())
    if not isinstance(source_map, Mapping):
        return ["distractor_sources_schema"], warnings, {}
    if set(source_map) != expected:
        errors.append("distractor_sources_keys")
    audit: dict[str, Any] = {}
    evidence_containers = {
        _container_key(position_map[position]) for position in positions
        if position in position_map
    }
    for letter in sorted(expected):
        raw_sources = source_map.get(letter)
        integer_sources = (
            isinstance(raw_sources, list) and
            all(isinstance(pos, int) and not isinstance(pos, bool) for pos in raw_sources)
        )
        valid_list = (
            integer_sources and bool(raw_sources) and
            len(raw_sources) == len(set(raw_sources))
        )
        if not valid_list:
            errors.append(f"distractor_source_schema:{letter}")
            raw_sources = []
        out_of_range = [pos for pos in raw_sources if pos not in position_map]
        if out_of_range:
            errors.append(f"distractor_source_position_out_of_range:{letter}")
        containers = sorted({
            _container_key(position_map[pos]) for pos in raw_sources if pos in position_map
        })
        if containers and evidence_containers and not (set(containers) & evidence_containers):
            warnings.append(f"cross_story_arc_provenance:{letter}")
        audit[letter] = {
            "positions": list(raw_sources),
            "canonical_turn_ids": [
                str(position_map[pos].get("turn_ref"))
                for pos in raw_sources if pos in position_map
            ],
            "containers": containers,
            "shares_evidence_container": bool(set(containers) & evidence_containers),
        }
    return errors, warnings, audit


def materialize_position_writer_output(
    value: Any, *, expanded_dialogue: Mapping[str, Any],
) -> dict[str, Any]:
    """Turn semantic positions into the pre-existing deterministic MCQ shape."""
    envelope_errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != TOP_KEYS:
        envelope_errors.append("position_top_level_schema")
        value = value if isinstance(value, Mapping) else {}
    if value.get("position_schema_version") != POSITION_SCHEMA_VERSION:
        envelope_errors.append("position_schema_version")
    questions = value.get("questions")
    if not isinstance(questions, list):
        envelope_errors.append("questions_schema")
        questions = []
    if len(questions) > QUESTION_CAPACITY_MAX:
        envelope_errors.append("question_capacity")
    abstention = value.get("abstention_summary")
    if (not isinstance(abstention, Mapping) or
            set(abstention) != {"estimated_skipped_count", "reason_codes"} or
            not isinstance(abstention.get("estimated_skipped_count"), int) or
            abstention.get("estimated_skipped_count", -1) < 0 or
            not isinstance(abstention.get("reason_codes"), list)):
        envelope_errors.append("abstention_summary")
    elif not questions and (abstention.get("estimated_skipped_count", 0) < 1 or
                            not abstention.get("reason_codes")):
        envelope_errors.append("empty_batch_without_abstention_evidence")

    position_map = _position_map(expanded_dialogue)
    normalized: list[dict[str, Any]] = []
    materialized: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    seen_signatures: set[tuple[Any, ...]] = set()
    for index, raw in enumerate(questions, 1):
        errors = list(envelope_errors)
        row = raw if isinstance(raw, Mapping) else {}
        if not isinstance(raw, Mapping) or set(row) != QUESTION_KEYS:
            errors.append("question_schema")
        structure_type = str(row.get("structure_type") or "")
        question_type = QUESTION_TYPE_BY_STRUCTURE.get(structure_type)
        if question_type is None:
            errors.append("structure_type")
        if not _text(row.get("question")):
            errors.append("question")
        positions = row.get("evidence_positions")
        valid_positions = isinstance(positions, list) and all(
            isinstance(position, int) and not isinstance(position, bool)
            for position in positions)
        if not valid_positions:
            errors.append("evidence_positions_schema")
            positions = []
        if len(positions) < PREFERRED_MIN_EVIDENCE:
            errors.append("evidence_below_minimum")
        if len(positions) > HARD_MAX_EVIDENCE:
            errors.append("evidence_over_hard_maximum")
        if len(positions) != len(set(positions)):
            errors.append("evidence_duplicate_positions")
        if any(position not in position_map for position in positions):
            errors.append("evidence_position_out_of_range")
        options = row.get("options")
        if not isinstance(options, Mapping) or set(options) != OPTION_KEYS or any(not _text(options.get(key)) for key in OPTION_KEYS):
            errors.append("options_schema")
            options = {}
        else:
            if len({normalize_option(options[key]) for key in OPTION_KEYS}) != len(OPTION_KEYS):
                errors.append("duplicate_options")
        gold = row.get("gold_text")
        matching = [key for key in OPTION_KEYS if options.get(key) == gold]
        if not _text(gold) or len(matching) != 1:
            errors.append("gold_not_exactly_one_option")
        gold_key = matching[0] if len(matching) == 1 else None
        provenance_errors, provenance_warnings, provenance_audit = _distractor_provenance_audit(
            row.get("distractor_sources"), positions=positions,
            position_map=position_map, gold_key=gold_key)
        errors.extend(provenance_errors)
        turns = [position_map[position] for position in positions if position in position_map]
        scope = _scope(turns) if len(turns) == len(positions) else None
        if scope is None:
            errors.append("source_scope_unresolved")
        signature = (str(row.get("question") or "").casefold(),
                     tuple(sorted(normalize_option(value) for value in options.values())))
        if signature in seen_signatures:
            errors.append("exact_duplicate")
        seen_signatures.add(signature)
        canonical_turn_ids = [str(turns[index].get("turn_ref")) for index in range(len(turns))]
        evidence_aliases = [f"D{position:03d}" for position in positions]
        valid = not errors
        item = {
            "candidate_index": index,
            "candidate_id": f"Q{index:02d}",
            "valid": valid,
            "structural_errors": sorted(set(errors)),
            "warnings": sorted(set(
                provenance_warnings +
                (["evidence_above_preferred_range"] if len(positions) > PREFERRED_MAX_EVIDENCE else [])
            )),
            "structure_type": structure_type,
            "question_type": question_type,
            "evidence_positions": list(positions),
            "evidence_count": len(positions),
            "canonical_turn_ids": canonical_turn_ids,
            "evidence_turn_ids": evidence_aliases,
            "source_scope": scope,
            "distractor_sources": dict(row.get("distractor_sources") or {}),
            "distractor_provenance_audit": provenance_audit,
        }
        items.append(item)
        if valid:
            materialized.append({
                "candidate_id": f"Q{index:02d}",
                "question_type": question_type,
                "source_scope": scope,
                "question": str(row["question"]).strip(),
                "options": dict(options),
                "gold_text": str(gold),
                "correct_option": matching[0],
                "evidence_positions": list(positions),
                "canonical_turn_ids": canonical_turn_ids,
                "evidence_turn_ids": evidence_aliases,
                "supporting_turn_ids": evidence_aliases,
                "distractor_sources": {
                    letter: list(row["distractor_sources"][letter])
                    for letter in sorted(OPTION_KEYS - {matching[0]})
                },
                "distractor_source_turn_ids": {
                    letter: [
                        str(position_map[position].get("turn_ref"))
                        for position in row["distractor_sources"][letter]
                    ]
                    for letter in sorted(OPTION_KEYS - {matching[0]})
                },
            })
            normalized.append({
                "question_type": question_type,
                "question": str(row["question"]).strip(),
                "options": dict(options),
                "correct_option": matching[0],
                "evidence_turn_refs": evidence_aliases,
            })
    return {
        "envelope_valid": not envelope_errors,
        "envelope_errors": sorted(set(envelope_errors)),
        "question_count": len(questions),
        "structural_valid_count": len(normalized),
        "structural_invalid_count": len(questions) - len(normalized),
        "validity_rate": len(normalized) / len(questions) if questions else None,
        "items": items,
        "materialized_candidates": materialized,
        "normalized_batch": {
            "mcq_batch_schema_version": "simple_append_mcq_batch_v5",
            "questions": normalized,
        },
        "zero_question_batch_valid": not questions and not envelope_errors,
        "evidence_statistics": {
            "min": min((row["evidence_count"] for row in items), default=None),
            "max": max((row["evidence_count"] for row in items), default=None),
            "over_8_count": sum(row["evidence_count"] > HARD_MAX_EVIDENCE for row in items),
        },
        "question_type_distribution": dict(Counter(
            row["question_type"] for row in items if row["valid"])),
        "program_materialization": True,
    }
