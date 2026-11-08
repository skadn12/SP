"""Read frozen materialized atomic QA and normalize it without repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import AtomicQA, OPTION_KEYS, extract_entities, extract_topic_terms


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_atomic_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load materialized rows or a batch envelope; never call a Writer."""
    source = Path(path)
    if source.is_dir():
        rows: list[dict[str, Any]] = []
        for child in sorted(source.glob("*.json")):
            value = _read_json(child)
            if isinstance(value, Mapping) and value.get("qa_id"):
                rows.append(dict(value))
        return rows
    value = _read_json(source)
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    if not isinstance(value, Mapping):
        raise ValueError(f"atomic_qa_json_object_required:{source}")
    for key in ("items", "questions", "qas"):
        rows = value.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, Mapping)]
    if value.get("qa_id"):
        return [dict(value)]
    raise ValueError(f"atomic_qa_rows_not_found:{source}")


def load_evaluations(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if source.is_file():
        values = [_read_json(source)]
    elif source.is_dir():
        values = [_read_json(child) for child in sorted(
            source.glob("*_evaluation.json"))]
    else:
        raise FileNotFoundError(source)
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if isinstance(value, Mapping) and value.get("qa_id"):
            result[str(value["qa_id"])] = dict(value)
    return result


def _root_id(row: Mapping[str, Any], transcript: Mapping[str, Any]) -> str:
    if transcript.get("root_id"):
        return str(transcript["root_id"])
    qa_id = str(row.get("qa_id") or "")
    return qa_id.split("-simple-append-mcq-", 1)[0] or qa_id


def _canonical_refs(row: Mapping[str, Any]) -> list[str]:
    refs = row.get("evidence_turn_ids")
    if refs is None:
        refs = row.get("evidence_turn_refs")
    return [str(ref) for ref in refs] if isinstance(refs, list) else []


def _grounding_decision(evaluation: Mapping[str, Any] | None) -> str:
    if not evaluation:
        return "missing_grounding_record"
    grounding = evaluation.get("grounding")
    if isinstance(grounding, Mapping):
        return str(grounding.get("decision") or "missing_grounding_decision")
    return str(evaluation.get("grounding_decision") or "missing_grounding_decision")


def _normalize_one(
    row: Mapping[str, Any], *, transcript: Mapping[str, Any],
    evaluation: Mapping[str, Any] | None, source_path: str,
    require_grounding: bool, require_full_context_stable: bool,
) -> tuple[AtomicQA | None, dict[str, Any]]:
    qa_id = str(row.get("qa_id") or row.get("candidate_id") or "")
    reasons: list[str] = []
    question = str(row.get("question") or "").strip()
    options = row.get("options")
    options_dict = dict(options) if isinstance(options, Mapping) else {}
    letter = str(row.get("reference_answer_letter") or
                 row.get("correct_option") or "").strip().upper()
    gold = str(row.get("canonical_answer_text") or
                row.get("reference_answer_text") or
                row.get("gold_text") or "").strip()
    refs = _canonical_refs(row)
    turns = transcript.get("turns")
    turn_map = {
        str(turn.get("turn_ref")): turn
        for turn in turns if isinstance(turn, Mapping) and turn.get("turn_ref")
    } if isinstance(turns, list) else {}
    evidence_texts = [str(turn_map[ref].get("text") or "") for ref in refs
                      if ref in turn_map]
    turn_indices = [index for index, turn in enumerate(turns, 1)
                    if isinstance(turn, Mapping) and
                    str(turn.get("turn_ref")) in refs] \
        if isinstance(turns, list) else []
    grounding = _grounding_decision(evaluation)
    full_context_correct = int((evaluation or {}).get("full_context_correct") or 0)
    full_context_total = int((evaluation or {}).get("full_context_total") or 0)
    if not qa_id:
        reasons.append("missing_qa_id")
    if not question:
        reasons.append("missing_question")
    if set(options_dict) != set(OPTION_KEYS) or any(
            not str(options_dict.get(key) or "").strip() for key in OPTION_KEYS):
        reasons.append("missing_or_invalid_options")
    if letter not in OPTION_KEYS:
        reasons.append("missing_correct_option")
    if not gold:
        reasons.append("missing_gold")
    if letter in OPTION_KEYS and gold and str(options_dict.get(letter)) != gold:
        reasons.append("gold_option_binding_mismatch")
    if not refs or len(refs) != len(set(refs)):
        reasons.append("missing_or_duplicate_evidence")
    if refs and len(evidence_texts) != len(refs):
        reasons.append("evidence_ref_not_in_transcript")
    if row.get("invalid_evidence_turn_ids"):
        reasons.append("invalid_evidence_turn_ids")
    if require_grounding and grounding != "pass_grounding":
        reasons.append("grounding_not_pass")
    if (require_full_context_stable and grounding == "pass_grounding" and
            (full_context_total != 5 or full_context_correct != 5)):
        reasons.append("full_context_not_stably_correct")
    if reasons:
        return None, {
            "record_type": "ineligible",
            "qa_id": qa_id,
            "source_path": source_path,
            "grounding_decision": grounding,
            "full_context_correct": full_context_correct,
            "full_context_total": full_context_total,
            "ineligibility_reasons": reasons,
        }
    entities = extract_entities(question, gold, *evidence_texts)
    topics = extract_topic_terms(question, gold, *evidence_texts)
    atomic = AtomicQA(
        qa_id=qa_id,
        root_id=_root_id(row, transcript),
        question=question,
        gold_text=gold,
        correct_option=letter,
        options={key: str(options_dict[key]) for key in OPTION_KEYS},
        evidence_turn_refs=refs,
        evidence_texts=evidence_texts,
        source_scope=str(row.get("source_scope") or ""),
        entities=entities,
        event_terms=topics,
        turn_indices=turn_indices,
        question_type=str(row.get("question_type") or ""),
        grounding_decision=grounding,
        source_path=source_path,
    )
    return atomic, {"record_type": "eligible", **atomic.to_dict()}


def build_atomic_ir(
    rows: Iterable[Mapping[str, Any]], *, transcript: Mapping[str, Any],
    evaluations: Mapping[str, Mapping[str, Any]] | None = None,
    source_path: str = "", require_grounding: bool = True,
    require_full_context_stable: bool = True,
) -> tuple[list[AtomicQA], list[dict[str, Any]]]:
    eligible: list[AtomicQA] = []
    records: list[dict[str, Any]] = []
    evaluations = evaluations or {}
    for row in rows:
        qa_id = str(row.get("qa_id") or row.get("candidate_id") or "")
        atomic, record = _normalize_one(
            row, transcript=transcript, evaluation=evaluations.get(qa_id),
            source_path=source_path, require_grounding=require_grounding,
            require_full_context_stable=require_full_context_stable,
        )
        records.append(record)
        if atomic is not None:
            eligible.append(atomic)
    return eligible, records
