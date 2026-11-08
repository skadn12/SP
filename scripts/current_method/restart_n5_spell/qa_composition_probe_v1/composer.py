"""Future Composer request construction; this module performs no API call."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import AtomicQA, COMPOSER_TYPES


COMPOSER_REQUEST_SCHEMA_VERSION = "qa_composition_composer_request_v1"


def _atomic_payload(qa: AtomicQA) -> dict[str, Any]:
    return {
        "qa_id": qa.qa_id,
        "question": qa.question,
        "gold_text": qa.gold_text,
        "correct_option": qa.correct_option,
        "options": dict(qa.options),
        "evidence_turn_refs": list(qa.evidence_turn_refs),
        "evidence_texts": list(qa.evidence_texts),
        "source_scope": qa.source_scope,
    }


def build_composer_request(
    pair: Mapping[str, Any], *, qa1: AtomicQA, qa2: AtomicQA,
) -> dict[str, Any]:
    composition_type = str(pair.get("candidate_type") or "")
    if composition_type not in COMPOSER_TYPES:
        raise ValueError("uncertain_pair_not_composer_ready")
    required_groups = [
        {"source_qa_id": qa1.qa_id, "turn_refs": list(qa1.evidence_turn_refs)},
        {"source_qa_id": qa2.qa_id, "turn_refs": list(qa2.evidence_turn_refs)},
    ]
    payload = {
        "pair_id": str(pair["pair_id"]),
        "composition_type": composition_type,
        "atomic_qa_1": _atomic_payload(qa1),
        "atomic_qa_2": _atomic_payload(qa2),
        "required_qa_ids": [qa1.qa_id, qa2.qa_id],
        "required_evidence_groups": required_groups,
        "pair_metadata": dict(pair),
    }
    instructions = """You are a Composite Question Composer. Return exactly one JSON object.
The new question must test a relation between the two atomic answers.
Do NOT simply concatenate the two original questions.
Do NOT ask for two independent answers joined by 'and'.
The composite answer must require comparing, linking, or interpreting both atomic answers.
The question must not be answerable by copying one sentence from either atomic evidence alone.
Use only the supplied atomic gold answers and evidence. Do not select, add, or guess evidence.
Use exactly four unique options in one answer dimension, with one correct_option letter.
Required output keys: pair_id, composition_type, question, answer_dimension,
gold_text, options, correct_option, required_qa_ids,
required_evidence_turn_refs, required_evidence_groups, relation_summary,
is_mechanical_conjunction.
""".strip()
    return {
        "request_schema_version": COMPOSER_REQUEST_SCHEMA_VERSION,
        "request_id": f"composer-{pair['pair_id']}",
        "pair_id": str(pair["pair_id"]),
        "composition_type": composition_type,
        "payload": payload,
        "prompt": instructions + "\n\nINPUT_JSON:\n" + _json(payload),
    }


def _json(value: Any) -> str:
    import json
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)

