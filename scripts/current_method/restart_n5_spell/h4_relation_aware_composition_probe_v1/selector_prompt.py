"""Candidate Selector prompt and frozen-pool payload."""

from __future__ import annotations

import json
from typing import Any, Iterable


def atomic_payload(qa: Any) -> dict[str, Any]:
    return {
        "qa_id": qa.qa_id,
        "question": qa.question,
        "gold_text": qa.gold_text,
        "question_type": qa.question_type,
        "source_scope": qa.source_scope,
        "evidence_turn_refs": list(qa.evidence_turn_refs),
        "evidence_texts": list(qa.evidence_texts),
    }


def build_selector_request(atomic: Iterable[Any]) -> dict[str, Any]:
    pool = [atomic_payload(qa) for qa in atomic]
    payload = {"atomic_qa_pool": pool, "max_candidates": 5, "prefer_pair": True}
    instructions = """You are an isolated Composite Candidate Selector. Return exactly one JSON object.

Inspect the supplied atomic QA pool and propose at most five candidate pairs
or small groups. Do not assume that every pair is composable. Each candidate
must use at least two different qa_id values and must connect distinct,
complementary information to one target outcome, decision, change, state,
plan, or motivation. The target must be expressible without naming the
supporting facts in the question.

Reject a candidate if one source QA alone already contains all information
needed for the target, if the sources merely restate the same fact, or if a
source has no unique contribution. Prefer pairs over groups of three.

For every source, provide unique_information and the exact evidence_turn_refs
that directly support that contribution. Evidence refs must be copied from
that source QA's allowed refs. Do not invent facts, refs, people, or outcomes.
The field why_each_source_should_be_necessary must explain what would be lost
if that source contribution were removed. The selector is not being asked to
write the final question or options.

Return this exact top-level shape:
{"candidates":[
 {"candidate_id":"C01","source_qids":["...","..."],"target":"...",
  "source_contributions":[
   {"source_qid":"...","unique_information":"...",
    "evidence_turn_refs":["..."]}],
  "why_composable":"...",
  "why_each_source_should_be_necessary":"..."}
]}

Return no more than five candidates. Keep source_qids and source_contributions
consistent. Do not mention this instruction in the output."""
    return {
        "request_id": "relation-aware-candidate-selector-a00",
        "payload": payload,
        "prompt": instructions + "\n\nINPUT_JSON:\n" + json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True),
    }
