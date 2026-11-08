"""Mechanical contract for reverse MCQ materialization from fixed facts.

The future Writer selects fact IDs only. It never writes distractor text or
provenance. The program uses the frozen fact text and source refs verbatim.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "fact_bundle_writer_contract_v1"
TOP_KEYS = {"fact_bundle_writer_schema_version", "questions"}
QUESTION_KEYS = {
    "bundle_id", "question_type", "question", "answer_dimension",
    "gold_fact_id", "distractor_fact_ids",
}
OPTION_LETTERS = ("A", "B", "C", "D")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _index_facts(facts: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("fact_id")): row
        for row in facts
        if isinstance(row, Mapping) and _text(row.get("fact_id"))
    }


def _index_bundles(
    bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("bundle_id")): row
        for row in bundles
        if isinstance(row, Mapping) and _text(row.get("bundle_id"))
    }


def validate_writer_output(
    value: Any,
    *,
    facts: Sequence[Mapping[str, Any]],
    bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate only references and semantic fields; no option text is accepted."""
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != TOP_KEYS:
        errors.append("top_level_schema")
        value = value if isinstance(value, Mapping) else {}
    if value.get("fact_bundle_writer_schema_version") != SCHEMA_VERSION:
        errors.append("schema_version")
    rows = value.get("questions")
    if not isinstance(rows, list):
        errors.append("questions_schema")
        rows = []
    fact_index = _index_facts(facts)
    bundle_index = _index_bundles(bundles)
    items: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for index, raw in enumerate(rows, 1):
        row_errors: list[str] = []
        row = raw if isinstance(raw, Mapping) else {}
        if not isinstance(raw, Mapping) or set(row) != QUESTION_KEYS:
            row_errors.append("question_schema")
        bundle_id = row.get("bundle_id")
        bundle = bundle_index.get(str(bundle_id))
        if bundle is None:
            row_errors.append("bundle_unknown")
            bundle_fact_ids: set[str] = set()
        else:
            bundle_fact_ids = {str(item) for item in bundle.get("fact_ids") or []}
        if not _text(row.get("question")):
            row_errors.append("question_empty")
        if not _text(row.get("question_type")):
            row_errors.append("question_type_empty")
        if not _text(row.get("answer_dimension")):
            row_errors.append("answer_dimension_empty")
        gold = str(row.get("gold_fact_id") or "")
        if gold not in fact_index or gold not in bundle_fact_ids:
            row_errors.append("gold_fact_not_in_bundle")
        distractors = row.get("distractor_fact_ids")
        if (not isinstance(distractors, list) or len(distractors) != 3 or
                len(set(map(str, distractors))) != len(distractors)):
            row_errors.append("distractor_fact_ids_shape")
            distractors = []
        distractor_ids = [str(item) for item in distractors]
        if gold and gold in distractor_ids:
            row_errors.append("gold_repeated_as_distractor")
        if any(item not in fact_index or item not in bundle_fact_ids
               for item in distractor_ids):
            row_errors.append("distractor_fact_not_in_bundle")
        item = {
            "writer_index": index,
            "valid": not row_errors,
            "errors": sorted(set(row_errors)),
            "bundle_id": bundle_id,
            "gold_fact_id": gold,
            "distractor_fact_ids": distractor_ids,
        }
        items.append(item)
        if item["valid"]:
            valid.append(dict(row))
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not errors and not any(not row["valid"] for row in items),
        "errors": sorted(set(errors)),
        "raw_question_count": len(rows),
        "valid_question_count": len(valid),
        "quarantine_count": len(rows) - len(valid),
        "items": items,
        "valid_questions": valid,
    }


def _gold_letter(bundle_id: str, question: str) -> str:
    digest = hashlib.sha256(f"{bundle_id}\0{question}".encode()).digest()
    return OPTION_LETTERS[digest[0] % len(OPTION_LETTERS)]


def materialize_mcq(
    row: Mapping[str, Any],
    *,
    facts: Sequence[Mapping[str, Any]],
    bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the existing MCQ item entirely from fixed fact records."""
    validation = validate_writer_output({
        "fact_bundle_writer_schema_version": SCHEMA_VERSION,
        "questions": [dict(row)],
    }, facts=facts, bundles=bundles)
    if not validation["valid"]:
        raise ValueError("fact_bundle_writer_row_invalid:" + ",".join(
            validation["items"][0]["errors"] if validation["items"] else
            validation["errors"]))
    fact_index = _index_facts(facts)
    bundle = _index_bundles(bundles)[str(row["bundle_id"])]
    selected = [str(row["gold_fact_id"])] + [
        str(item) for item in row["distractor_fact_ids"]
    ]
    gold = _gold_letter(str(row["bundle_id"]), str(row["question"]))
    letters = [letter for letter in OPTION_LETTERS if letter != gold]
    option_fact_ids = {gold: selected[0]}
    option_fact_ids.update({letter: fact_id for letter, fact_id in zip(letters, selected[1:])})
    options = {
        letter: str(fact_index[fact_id].get("fact_text") or "").strip()
        for letter, fact_id in option_fact_ids.items()
    }
    if any(not text for text in options.values()):
        raise ValueError("fact_text_empty_during_materialization")
    evidence = list(fact_index[selected[0]].get("source_turn_ids") or [])
    distractor_sources = {
        letter: list(fact_index[fact_id].get("source_turn_ids") or [])
        for letter, fact_id in option_fact_ids.items() if letter != gold
    }
    return {
        "coverage_role": (
            "parent_child" if str(bundle.get("bundle_class") or "") == "parent_child"
            else "general"
        ),
        "question_type": row["question_type"],
        "question": row["question"],
        "options": options,
        "correct_option": gold,
        "evidence_turn_refs": evidence,
        "answer_dimension": row["answer_dimension"],
        "distractor_sources": distractor_sources,
    }


def materialize_batch(
    value: Mapping[str, Any],
    *,
    facts: Sequence[Mapping[str, Any]],
    bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validation = validate_writer_output(value, facts=facts, bundles=bundles)
    materialized: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(validation["valid_questions"], 1):
        try:
            materialized.append(materialize_mcq(row, facts=facts, bundles=bundles))
        except Exception as error:
            errors.append({"writer_index": index, "error": str(error)})
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": validation["valid"] and not errors,
        "validation": validation,
        "materialized_questions": materialized,
        "materialization_errors": errors,
    }
