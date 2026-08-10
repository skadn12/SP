from __future__ import annotations

import inspect
import json
import re
from typing import Any, Mapping, Sequence

from .balanced_factorial_distractor_canary import (
    BALANCED_FACTORIAL_SCHEMA,
    validate_and_render_balanced_factorial,
)
from .core import canonical_json, sha256_bytes


SEMANTIC_PARITY_RETRY_VERSION = "semantic_parity_distractor_retry_v1"
SEMANTIC_PARITY_PROMPT_VERSION = "semantic_parity_balanced_2x2_writer_v1"
SEMANTIC_PARITY_SCHEMA_VERSION = "semantic_parity_balanced_2x2_schema_v1"
MIN_SHARED_FRAME_WORD_COVERAGE = 0.50
MIN_OPTION_LENGTH_RATIO = 0.80
MAX_OPTION_LENGTH_RATIO = 1.20


def semantic_parity_schema_sha256() -> str:
    return sha256_bytes(canonical_json(BALANCED_FACTORIAL_SCHEMA).encode())


def semantic_parity_prompt_sha256() -> str:
    material = canonical_json({
        "version": SEMANTIC_PARITY_PROMPT_VERSION,
        "schema_version": SEMANTIC_PARITY_SCHEMA_VERSION,
        "schema": BALANCED_FACTORIAL_SCHEMA,
        "source": inspect.getsource(build_semantic_parity_prompt),
    })
    return sha256_bytes(material.encode())


def _words(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(value))


def validate_semantic_parity_bundle(
    raw: Any, *, correct_answer_text: str,
) -> dict[str, Any]:
    result = validate_and_render_balanced_factorial(
        raw, correct_answer_text=correct_answer_text)
    if result["decision"] != "balanced_factorial_accept":
        return {**result, "decision": "semantic_parity_schema_invalid"}

    reasons: list[str] = []
    if result["literal_frame_word_coverage"] < MIN_SHARED_FRAME_WORD_COVERAGE:
        reasons.append("shared_frame_coverage_below_half")
    gold_count = max(1, len(_words(correct_answer_text)))
    ratios = []
    for row in result["rendered_combinations"]:
        ratio = len(_words(row["rendered_text"])) / gold_count
        ratios.append(ratio)
        if ratio < MIN_OPTION_LENGTH_RATIO or ratio > MAX_OPTION_LENGTH_RATIO:
            reasons.append("option_length_not_semantically_parallel")
    if reasons:
        return {
            **result,
            "decision": "semantic_parity_schema_invalid",
            "reasons": sorted(set(reasons)),
            "strict_option_length_ratios": ratios,
        }
    return {
        **result,
        "decision": "semantic_parity_accept",
        "strict_option_length_ratios": ratios,
        "semantic_parity_validation_scope":
            "structural_slot_balance_not_semantic_entailment",
    }


def build_semantic_parity_prompt(
    *,
    base_qa: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, str]],
    previous_distractors: Sequence[str],
    previous_grounding_diagnosis: str,
) -> str:
    if len(previous_distractors) != 3:
        raise ValueError("semantic_parity_requires_three_previous_distractors")
    if not str(previous_grounding_diagnosis).startswith("bad_ground"):
        raise ValueError("semantic_parity_requires_bad_ground_source")
    payload = {
        "frozen_question_answer": {
            "qa_id": str(base_qa["qa_id"]),
            "question": str(base_qa["question"]),
            "correct_answer_text": str(base_qa["correct_answer_text"]),
            "answer_support_span_ids": list(
                base_qa["answer_support_span_ids"]),
        },
        "minimal_answer_evidence": [dict(row) for row in evidence_rows],
        "previous_failed_distractors": list(previous_distractors),
        "previous_final_decision": "bad_ground",
        "previous_grounding_diagnosis": previous_grounding_diagnosis,
    }
    rules = [
        "Return one shared answer frame and exactly two binary semantic slots S1 and S2; do not output options directly.",
        "The frame rendered with both gold values must reproduce correct_answer_text exactly, character for character.",
        "The shared frame must retain at least half of the correct answer's words and each placeholder must occur exactly once.",
        "Each contrast value must have the same narrow semantic type, grammatical role, scope, specificity, and number of semantic facets as its gold value.",
        "All four 2x2 combinations must be equally complete, natural, and plausible answers to the frozen question without context.",
        "No combination may be a subset, broader summary, narrower detail, missing-purpose answer, or uniquely comprehensive answer relative to another combination.",
        "For an action question, every combination must describe the same class of direct action addressing the same stated problem; do not use unrelated wellness, hobby, awareness, or generic discussion activities.",
        "For a purpose question, every combination must preserve the same number of object, setting, beneficiary, and intended-use facets; change values but never omit a facet.",
        "Each contrast may be counterfactual but must be false under the frozen evidence and must not create a second correct answer.",
        "Do not reuse any previous failed distractor verbatim and do not use negation, absurdity, contradiction, or an unrelated topic.",
        "The program will render the balanced 2x2 options so each gold/contrast slot value occurs exactly twice.",
        "Return strict JSON only with no Markdown, commentary, options, or additional fields.",
    ]
    example = {
        "frame": "To document the {S1} and the {S2} for the public report.",
        "slots": [
            {"slot_id": "S1", "gold_value": "group interactions",
             "contrast_value": "individual activities"},
            {"slot_id": "S2", "gold_value": "surrounding environment",
             "contrast_value": "event schedule"},
        ],
    }
    return "\n\n".join([
        f"PROMPT_VERSION: {SEMANTIC_PARITY_PROMPT_VERSION}",
        "SEMANTIC-PARITY 2x2 RULES\n- " + "\n- ".join(rules),
        "FORMAT EXAMPLE (unrelated to the frozen item)\n" + json.dumps(
            example, ensure_ascii=False, indent=2),
        "FROZEN INPUT\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2),
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            BALANCED_FACTORIAL_SCHEMA, ensure_ascii=False,
            sort_keys=True, indent=2),
    ])

