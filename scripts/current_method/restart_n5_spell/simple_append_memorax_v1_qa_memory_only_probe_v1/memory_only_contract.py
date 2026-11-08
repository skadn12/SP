"""Small, deterministic contract for a Memory-only QA Writer experiment."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..core import normalize_option

QUESTION_TYPES = frozenset({
    "causal", "temporal_sequence", "state_change", "cross_turn_comparison",
    "relationship_or_decision", "cross_context_synthesis", "information_visibility",
    "information_revision", "task_dependency_or_constraint",
})
SOURCE_TYPES = frozenset({"people", "events", "locations", "items", "themes"})
LETTERS = ("A", "B", "C", "D")
MAX_QUESTIONS = 12


def memory_source_refs(overview: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Create stable local labels without changing the frozen Memory data."""
    refs: dict[str, dict[str, str]] = {}
    for source_type in sorted(SOURCE_TYPES):
        rows = overview.get(source_type) or []
        for index, row in enumerate(rows, 1):
            if not isinstance(row, Mapping):
                continue
            key = str(row.get("canonical_key") or row.get("display_key") or "").strip()
            if not key:
                continue
            ref = f"M{source_type[:1].upper()}{index:03d}"
            refs[ref] = {"source_type": source_type, "memory_key": key,
                         "display_key": str(row.get("display_key") or key)}
    return refs


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_writer_output(
    value: Any, *, source_refs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != {"questions"}:
        return {"valid": False, "errors": ["top_level_schema"],
                "questions": [], "raw_question_count": 0}
    questions = value.get("questions")
    if not isinstance(questions, list) or len(questions) > MAX_QUESTIONS:
        return {"valid": False, "errors": ["question_count_0_to_12"],
                "questions": [], "raw_question_count": 0}
    valid: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for index, row in enumerate(questions, 1):
        row_errors: list[str] = []
        if not isinstance(row, Mapping):
            row_errors.append("question_object")
            row = {}
        expected = {"question_type", "question", "options", "gold_text",
                    "memory_source_refs"}
        if set(row) != expected:
            row_errors.append("question_schema")
        if row.get("question_type") not in QUESTION_TYPES:
            row_errors.append("question_type")
        if not _text(row.get("question")):
            row_errors.append("question")
        options = row.get("options")
        if (not isinstance(options, Mapping) or set(options) != set(LETTERS) or
                any(not _text(options.get(letter)) for letter in LETTERS)):
            row_errors.append("options")
            options = {}
        gold = row.get("gold_text")
        matches = [letter for letter in LETTERS
                   if normalize_option(options.get(letter)) == normalize_option(gold)]
        if len(matches) != 1:
            row_errors.append("gold_text_not_unique")
        refs = row.get("memory_source_refs")
        if (not isinstance(refs, list) or not refs or
                any(not isinstance(ref, str) or ref not in source_refs for ref in refs)):
            row_errors.append("memory_source_refs")
        if row_errors:
            quarantine.append({"question_index": index, "errors": sorted(set(row_errors)),
                               "question": row})
            continue
        materialized = dict(row)
        materialized["correct_option"] = matches[0]
        materialized["memory_grounded"] = True
        materialized["transcript_verified"] = False
        valid.append(materialized)
    return {"valid": not errors, "errors": errors, "questions": valid,
            "quarantine": quarantine, "raw_question_count": len(questions),
            "valid_question_count": len(valid),
            "quarantine_count": len(quarantine)}


def materialize_solver_rows(
    normalized: Mapping[str, Any], *, root_id: str,
) -> list[dict[str, Any]]:
    """Emit the existing Solver-compatible shape, without evidence IDs."""
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(normalized.get("questions") or [], 1):
        rows.append({
            "qa_id": f"{root_id}-memory-only-mcq-{index:02d}",
            "question": row["question"],
            "question_type": row["question_type"],
            "options": dict(row["options"]),
            "correct_option": row["correct_option"],
            "canonical_answer_text": row["options"][row["correct_option"]],
            "memory_source_refs": list(row["memory_source_refs"]),
            "memory_grounded": True,
            "transcript_verified": False,
            "evidence_turn_refs": [],
            "evidence_status": "deferred_to_transcript_validation",
        })
    return rows


def source_ref_rows(overview: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Serialize the exact Memory records shown to Writer, with local refs."""
    refs = memory_source_refs(overview)
    rows: list[dict[str, Any]] = []
    for ref, meta in refs.items():
        source_rows = overview.get(meta["source_type"]) or []
        row = next((x for x in source_rows if isinstance(x, Mapping) and
                    str(x.get("canonical_key") or x.get("display_key") or "").strip() ==
                    meta["memory_key"]), None)
        rows.append({"memory_ref": ref, **meta, "content": row.get("content", []) if row else []})
    return rows
