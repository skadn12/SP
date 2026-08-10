from __future__ import annotations

import inspect
import json
import re
from typing import Any, Mapping, Sequence

from .core import canonical_json, sha256_bytes


ADVERSARIAL_DISTRACTOR_SEARCH_VERSION = "adversarial_distractor_search_v1"
ADVERSARIAL_DISTRACTOR_PROMPT_VERSION = (
    "adversarial_four_bundle_distractor_writer_v1"
)
ADVERSARIAL_DISTRACTOR_SCHEMA_VERSION = (
    "adversarial_four_bundle_distractor_schema_v1"
)
BUNDLE_IDS = ("B1", "B2", "B3", "B4")
MIN_LENGTH_RATIO = 0.80
MAX_LENGTH_RATIO = 1.20
MIN_GOLD_TOKEN_OVERLAP = 0.45


ADVERSARIAL_DISTRACTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["bundles"],
    "properties": {
        "bundles": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["bundle_id", "distractors"],
                "properties": {
                    "bundle_id": {
                        "type": "string",
                        "enum": list(BUNDLE_IDS),
                    },
                    "distractors": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        }
    },
}


def adversarial_distractor_schema_sha256() -> str:
    return sha256_bytes(canonical_json(ADVERSARIAL_DISTRACTOR_SCHEMA).encode())


def adversarial_distractor_prompt_sha256() -> str:
    material = canonical_json({
        "version": ADVERSARIAL_DISTRACTOR_PROMPT_VERSION,
        "schema_version": ADVERSARIAL_DISTRACTOR_SCHEMA_VERSION,
        "schema": ADVERSARIAL_DISTRACTOR_SCHEMA,
        "source": inspect.getsource(build_adversarial_distractor_prompt),
    })
    return sha256_bytes(material.encode())


def _tokens(value: str) -> list[str]:
    return [token.casefold() for token in re.findall(
        r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(value))]


def validate_adversarial_bundles(
    raw: Any, *, correct_answer_text: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(raw, dict):
        return {"decision": "adversarial_bundles_schema_invalid",
                "reasons": ["object_required"]}
    if set(raw) != {"bundles"}:
        reasons.append("top_level_fields_invalid")
    bundles_raw = raw.get("bundles")
    if not isinstance(bundles_raw, list) or len(bundles_raw) != 4:
        return {"decision": "adversarial_bundles_schema_invalid",
                "reasons": sorted(set([*reasons, "bundle_count_invalid"]))}

    gold_tokens = _tokens(correct_answer_text)
    gold_count = max(1, len(gold_tokens))
    gold_set = set(gold_tokens)
    bundles: list[dict[str, Any]] = []
    seen_ids: list[str] = []
    seen_sets: set[tuple[str, ...]] = set()
    for row in bundles_raw:
        if not isinstance(row, dict) or set(row) != {
                "bundle_id", "distractors"}:
            reasons.append("bundle_fields_invalid")
            continue
        bundle_id = row.get("bundle_id")
        if bundle_id not in BUNDLE_IDS:
            reasons.append("bundle_id_invalid")
        else:
            seen_ids.append(str(bundle_id))
        values = row.get("distractors")
        if not isinstance(values, list) or len(values) != 3:
            reasons.append("distractor_count_invalid")
            continue
        if any(not isinstance(value, str) or not value.strip()
               for value in values):
            reasons.append("distractor_string_invalid")
            continue
        distractors = [value.strip() for value in values]
        folded = [value.casefold() for value in distractors]
        if len(set(folded)) != 3:
            reasons.append("duplicate_distractor_within_bundle")
        if correct_answer_text.casefold() in folded:
            reasons.append("distractor_equals_correct_answer")
        canonical_set = tuple(sorted(folded))
        if canonical_set in seen_sets:
            reasons.append("duplicate_bundle")
        seen_sets.add(canonical_set)

        diagnostics = []
        for value in distractors:
            tokens = _tokens(value)
            ratio = len(tokens) / gold_count
            overlap = len(gold_set.intersection(tokens)) / max(1, len(gold_set))
            if ratio < MIN_LENGTH_RATIO or ratio > MAX_LENGTH_RATIO:
                reasons.append("distractor_length_not_parallel")
            if overlap < MIN_GOLD_TOKEN_OVERLAP:
                reasons.append("distractor_lexical_frame_not_parallel")
            diagnostics.append({
                "text": value,
                "word_count": len(tokens),
                "gold_length_ratio": ratio,
                "gold_token_overlap": overlap,
            })
        bundles.append({
            "bundle_id": bundle_id,
            "distractors": distractors,
            "diagnostics": diagnostics,
        })

    if sorted(seen_ids) != sorted(BUNDLE_IDS):
        reasons.append("bundle_ids_not_exact")
    if reasons:
        return {
            "decision": "adversarial_bundles_schema_invalid",
            "reasons": sorted(set(reasons)),
            "bundles": bundles,
        }
    bundles.sort(key=lambda row: BUNDLE_IDS.index(row["bundle_id"]))
    return {
        "decision": "adversarial_bundles_accept",
        "reasons": [],
        "bundles": bundles,
        "validation_scope": (
            "structure_length_and_lexical_parallelism_not_semantic_falsity"
        ),
    }


def deterministic_bundle_target_letter(
    qa_id: str, bundle_index: int,
) -> dict[str, Any]:
    if bundle_index not in range(4):
        raise ValueError("bundle_index_invalid")
    offset = int(sha256_bytes(str(qa_id).encode())[:8], 16) % 4
    letter = "ABCD"[(offset + bundle_index) % 4]
    seed_material = canonical_json({
        "version": ADVERSARIAL_DISTRACTOR_SEARCH_VERSION,
        "qa_id": qa_id,
        "bundle_index": bundle_index,
    })
    return {
        "target_letter": letter,
        "placement_seed": int(sha256_bytes(seed_material.encode())[:8], 16),
        "content_dependent": False,
    }


def is_grounding_search_success(final_decision: str) -> bool:
    return final_decision in {"reject_easy", "retain_frontier"}


def build_adversarial_distractor_prompt(
    *,
    base_qa: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, str]],
    prior_failed_sets: Sequence[Sequence[str]],
) -> str:
    payload = {
        "frozen_question_answer": {
            "qa_id": str(base_qa["qa_id"]),
            "question": str(base_qa["question"]),
            "correct_answer_text": str(base_qa["correct_answer_text"]),
            "answer_support_span_ids": list(
                base_qa["answer_support_span_ids"]),
        },
        "minimal_answer_evidence": [dict(row) for row in evidence_rows],
        "prior_bad_ground_option_sets": [
            list(values) for values in prior_failed_sets],
    }
    rules = [
        "Return exactly four distinct candidate bundles B1, B2, B3, and B4; each bundle contains exactly three distractors.",
        "The question, correct answer, and evidence are frozen. Output distractors only and never rewrite or restate the correct answer as a distractor.",
        "Every distractor must directly answer the exact question, preserve the same answer type, actor, time scope, causal role, grammatical form, specificity, and number of semantic facets as the correct answer.",
        "Keep every distractor between 80% and 120% of the correct answer length and retain the same lexical/syntactic frame wherever possible.",
        "Change only one decisive value or relation per distractor. Do not omit a purpose, beneficiary, setting, object, trigger, or outcome that the correct answer contains.",
        "For an action question, every option must be the same class of direct action by the same actor addressing the same stated problem. Do not switch to hobbies, awareness, discussion, education, military service, or unrelated charity domains.",
        "For a purpose question, every option must preserve the same photographer/action, event setting, beneficiary, campaign use, and level of detail; vary only the specific photographic focus or relationship being documented.",
        "Make the false values locally plausible and mutually confusable. The gold must not be uniquely conventional, coherent, comprehensive, topical, or natural without the evidence.",
        "The three distractors in a bundle must be false under the evidence, distinct from one another, and must not form synonymous second correct answers.",
        "The four bundles must explore genuinely different near-neighbor false values, not mere paraphrases or reordered copies of a prior failed set.",
        "Do not reuse any prior failed distractor verbatim. Do not use negation, absurdity, category mismatch, missing-detail answers, or unrelated events.",
        "Return strict JSON only with no Markdown, explanations, labels beyond bundle_id, or additional fields.",
    ]
    return "\n\n".join([
        f"PROMPT_VERSION: {ADVERSARIAL_DISTRACTOR_PROMPT_VERSION}",
        "ADVERSARIAL OPTION-SET RULES\n- " + "\n- ".join(rules),
        "FROZEN INPUT\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2),
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            ADVERSARIAL_DISTRACTOR_SCHEMA, ensure_ascii=False,
            sort_keys=True, indent=2),
    ])

