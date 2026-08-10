from __future__ import annotations

import inspect
import json
import re
from typing import Any, Mapping, Sequence

from .core import LETTERS, canonical_json, sha256_bytes
from .lineage_distractor_writer import (
    validate_attempt_feedback,
    validate_distractor_object,
)
from .minimal_admission import (
    SPAN_IDS_ONLY_SCHEMA_PROFILE,
    SPELL_STYLE_MINIMAL_MODE,
    materialize_spell_style_minimal,
)
from .questioner_contract import default_reasoning_contracts, span_catalog_index


CONTRASTIVE_DISTRACTOR_CANARY_VERSION = "contrastive_distractor_canary_v1"
CONTRASTIVE_DISTRACTOR_PROMPT_VERSION = (
    "spell_contrastive_distractor_writer_single_v1"
)
CONTRASTIVE_DISTRACTOR_SCHEMA_VERSION = "contrastive_distractor_schema_v1"
ANSWER_PLACEMENT_VERSION = "contrastive_distractor_answer_placement_v1"

CONTRASTIVE_DISTRACTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["distractors"],
    "properties": {
        "distractors": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        }
    },
}

TARGET_SOURCE_ITEMS = (
    ("item_01", "johns_motivation_for_politics"),
    ("item_02", "marias_volunteering_inspiration"),
)

_PLACEMENT_SCHEDULE = {
    "johns_motivation_for_politics": ("B", "D"),
    "marias_volunteering_inspiration": ("C", "A"),
}


def contrastive_distractor_schema_sha256() -> str:
    return sha256_bytes(canonical_json(CONTRASTIVE_DISTRACTOR_SCHEMA).encode())


def contrastive_distractor_prompt_sha256() -> str:
    material = (
        CONTRASTIVE_DISTRACTOR_PROMPT_VERSION
        + CONTRASTIVE_DISTRACTOR_SCHEMA_VERSION
        + inspect.getsource(build_contrastive_distractor_prompt)
        + canonical_json(CONTRASTIVE_DISTRACTOR_SCHEMA)
    )
    return sha256_bytes(material.encode())


def deterministic_answer_placement(qa_id: str, attempt_index: int) -> dict[str, Any]:
    if qa_id not in _PLACEMENT_SCHEDULE or attempt_index not in (0, 1):
        raise ValueError("contrastive_answer_placement_input_invalid")
    target_letter = _PLACEMENT_SCHEDULE[qa_id][attempt_index]
    seed_material = canonical_json({
        "version": ANSWER_PLACEMENT_VERSION,
        "qa_id": qa_id,
        "attempt_index": attempt_index,
    })
    return {
        "answer_placement_version": ANSWER_PLACEMENT_VERSION,
        "placement_seed": int(sha256_bytes(seed_material.encode())[:8], 16),
        "target_letter": target_letter,
        "content_dependent": False,
    }


def _evidence_rows(*, base_qa: Mapping[str, Any],
                   span_catalog: Mapping[str, Any]) -> list[dict[str, str]]:
    catalog = span_catalog_index(dict(span_catalog))
    rows = []
    for span_id in base_qa.get("answer_support_span_ids") or []:
        span = catalog.get(str(span_id))
        if span is None:
            raise ValueError(f"source_answer_span_missing:{span_id}")
        rows.append({
            "span_id": str(span["span_id"]),
            "turn_id": str(span["turn_id"]),
            "session_id": str(span["session_id"]),
            "speaker": str(span["speaker"]),
            "exact_text": str(span["exact_text"]),
        })
    if not rows:
        raise ValueError("source_answer_support_empty")
    return rows


def build_contrastive_distractor_prompt(
    *,
    frozen_dialogue: Sequence[Mapping[str, Any]],
    span_catalog: Mapping[str, Any],
    base_qa: Mapping[str, Any],
    attempt_index: int,
    structured_feedback: Mapping[str, Any] | None,
) -> str:
    feedback = validate_attempt_feedback(attempt_index, structured_feedback)
    evidence = _evidence_rows(base_qa=base_qa, span_catalog=span_catalog)
    payload = {
        "attempt_index": attempt_index,
        "frozen_question_answer": {
            "qa_id": base_qa["qa_id"],
            "question": base_qa["question"],
            "correct_answer_text": base_qa["correct_answer_text"],
            "answer_support_span_ids": list(base_qa["answer_support_span_ids"]),
        },
        "frozen_answer_evidence": evidence,
        "full_frozen_dialogue": [dict(turn) for turn in frozen_dialogue],
        "previous_attempt_structured_feedback": feedback,
    }
    rules = [
        "Return exactly one JSON object containing exactly three distinct distractors.",
        "Do not change, rewrite, explain, or output the frozen question, correct answer, or evidence IDs.",
        "Each distractor must directly answer the same question and have the same answer type as the correct answer.",
        "Every distractor must be grammatically substitutable for the correct answer in the question.",
        "Use closely parallel syntax and comparable length, specificity, detail, sentiment, and plausibility; exact wording or exact token length is not required.",
        "Make each distractor a near-neighbor error by changing one key relation such as actor, experience, motivation, trigger, time, state, consequence, or ordering.",
        "Prefer dialogue-internal people, experiences, motivations, events, states, and outcomes of the same type when they can form a plausible but incorrect answer.",
        "Without the dialogue, all four choices must look credible; the correct answer must not be uniquely natural, detailed, specific, causally coherent, or long.",
        "Do not use absurd, unrelated, generic, cross-type, or obviously irrelevant distractors.",
        "Do not copy or paraphrase the correct answer closely enough to become another correct answer, and do not repeat a distractor.",
        "The frozen dialogue is the only factual source. Do not invent a new correct explanation or change the gold answer.",
        "If attempt 1 feedback exists, use only the three previous strings and fixed diagnostic labels. Do not infer or add free-text Solver feedback.",
        "Return JSON only, with no Markdown, notes, evidence, explanations, or additional fields.",
    ]
    return "\n\n".join([
        f"PROMPT_VERSION: {CONTRASTIVE_DISTRACTOR_PROMPT_VERSION}",
        "CONTRASTIVE DISTRACTOR RULES\n- " + "\n- ".join(rules),
        "FROZEN INPUT\n" + json.dumps(payload, ensure_ascii=False,
                                        sort_keys=True, indent=2),
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            CONTRASTIVE_DISTRACTOR_SCHEMA, ensure_ascii=False,
            sort_keys=True, indent=2),
    ])


def audit_frozen_source_qa(*, raw_qa: Mapping[str, Any],
                           materialized_qa: Mapping[str, Any],
                           span_catalog: Mapping[str, Any]) -> dict[str, Any]:
    evidence = _evidence_rows(base_qa=raw_qa, span_catalog=span_catalog)
    raw_support = list(raw_qa.get("answer_support_span_ids") or [])
    checks = {
        "qa_id_match": raw_qa.get("qa_id") == materialized_qa.get("qa_id"),
        "question_match": raw_qa.get("question") == materialized_qa.get("question"),
        "correct_answer_match": raw_qa.get("correct_answer_text") ==
            materialized_qa.get("canonical_answer_text"),
        "support_span_set_match": set(raw_support) == set(
            materialized_qa.get("evidence_span_ids") or []),
        "support_spans_exist": len(evidence) == len(raw_support) > 0,
        "source_minimal_admission": materialized_qa.get("admission_mode") ==
            SPELL_STYLE_MINIMAL_MODE,
        "local_option_binding": materialized_qa.get("local_option_binding") is True,
    }
    return {
        "qa_id": raw_qa.get("qa_id"),
        "decision": "source_qa_accept" if all(checks.values()) else
            "source_qa_reject",
        "checks": checks,
        "frozen_question": raw_qa.get("question"),
        "frozen_correct_answer_text": raw_qa.get("correct_answer_text"),
        "frozen_answer_support_span_ids": raw_support,
        "support_evidence": evidence,
    }


def materialize_contrastive_candidate(
    *,
    base_qa: Mapping[str, Any],
    distractors: Sequence[str],
    target_letter: str,
    span_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    if target_letter not in LETTERS:
        raise ValueError("target_letter_invalid")
    validation = validate_distractor_object(
        {"distractors": list(distractors)},
        correct_answer_text=str(base_qa["correct_answer_text"]),
    )
    if validation["decision"] != "distractor_accept":
        raise ValueError(f"distractor_not_accepted:{validation['reasons']}")
    raw = {
        "qa_id": str(base_qa["qa_id"]),
        "question": str(base_qa["question"]),
        "correct_answer_text": str(base_qa["correct_answer_text"]),
        "distractors": list(validation["distractors"]),
        "answer_support_span_ids": list(base_qa["answer_support_span_ids"]),
        "reasoning_steps": list(base_qa.get("reasoning_steps") or []),
    }
    reasoning_contract = default_reasoning_contracts(
        (target_letter, target_letter, target_letter))[0]
    qa, errors, diagnostics = materialize_spell_style_minimal(
        raw,
        target_letter,
        span_catalog=dict(span_catalog),
        strict_v7_reasons=("strict_v7_not_rerun_for_contrastive_canary",),
        reasoning_contract=reasoning_contract,
        schema_profile=SPAN_IDS_ONLY_SCHEMA_PROFILE,
    )
    minimal_decision = "accept" if qa is not None and not errors else "reject"
    admission = {
        "item_index": 0,
        "qa_id": raw["qa_id"],
        "target_letter": target_letter,
        "strict_v7": {
            "decision": "diagnostic_not_rerun",
            "reasons": ["strict_v7_not_rerun_for_contrastive_canary"],
            "diagnostics": {},
        },
        SPELL_STYLE_MINIMAL_MODE: {
            "decision": minimal_decision,
            "reasons": sorted(set(errors)),
            "diagnostics": diagnostics,
        },
        "solver_eligible": minimal_decision == "accept",
    }
    return {
        "raw_v8_qa": raw,
        "materialized_qa": qa,
        "admission": admission,
        "reasoning_contracts": [reasoning_contract],
    }


def should_run_attempt_one(*, attempt_index: int, final_decision: str | None,
                           grounding_diagnosis: str | None) -> bool:
    return (
        attempt_index == 0
        and final_decision == "bad_ground"
        and grounding_diagnosis == "bad_ground_semantic_leakage"
    )


def option_length_diagnostics(materialized_qa: Mapping[str, Any]) -> dict[str, Any]:
    options = materialized_qa.get("options") or {}
    answer_letter = materialized_qa.get("reference_answer_letter")
    counts = {
        letter: len(re.findall(r"\w+", str(options.get(letter) or "")))
        for letter in LETTERS
    }
    correct = counts.get(str(answer_letter), 0)
    distractor_counts = [count for letter, count in counts.items()
                         if letter != answer_letter]
    longest = max(counts.values(), default=0)
    shortest = min((count for count in counts.values() if count > 0), default=0)
    return {
        "word_counts": counts,
        "correct_word_count": correct,
        "correct_is_unique_longest": correct == longest and
            list(counts.values()).count(longest) == 1,
        "distractor_word_counts": distractor_counts,
        "max_min_word_count_ratio": (longest / shortest) if shortest else None,
        "diagnostic_only": True,
    }


def route_conclusion(attempts: Sequence[Mapping[str, Any]]) -> str:
    decisions = [str(row.get("final_decision") or "") for row in attempts]
    if "retain_frontier" in decisions:
        return "contrastive_distractor_route_supported"
    if any(decision in {"reject_easy", "reject_all_wrong"}
           for decision in decisions):
        return "grounding_improved_question_or_ambiguity_bottleneck"
    if decisions and all(decision == "bad_ground" for decision in decisions):
        return "contrastive_distractor_semantic_leakage_persists"
    return "contrastive_distractor_canary_inconclusive"
