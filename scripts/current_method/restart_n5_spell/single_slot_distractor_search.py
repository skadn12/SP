from __future__ import annotations

import inspect
import json
import re
from typing import Any, Mapping, Sequence

from .adversarial_distractor_search import BUNDLE_IDS
from .core import canonical_json, sha256_bytes


SINGLE_SLOT_SEARCH_VERSION = "single_slot_distractor_search_v1"
SINGLE_SLOT_PROMPT_VERSION = "single_slot_four_bundle_writer_v1"
SINGLE_SLOT_SCHEMA_VERSION = "single_slot_four_bundle_schema_v1"
MIN_SHARED_FRAME_COVERAGE = 0.50
MIN_LENGTH_RATIO = 0.80
MAX_LENGTH_RATIO = 1.20


SINGLE_SLOT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["frame", "gold_value", "bundles"],
    "properties": {
        "frame": {"type": "string", "minLength": 1},
        "gold_value": {"type": "string", "minLength": 1},
        "bundles": {
            "type": "array", "minItems": 4, "maxItems": 4,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["bundle_id", "contrast_values"],
                "properties": {
                    "bundle_id": {
                        "type": "string", "enum": list(BUNDLE_IDS)},
                    "contrast_values": {
                        "type": "array", "minItems": 3, "maxItems": 3,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    },
}


def single_slot_schema_sha256() -> str:
    return sha256_bytes(canonical_json(SINGLE_SLOT_SCHEMA).encode())


def single_slot_prompt_sha256() -> str:
    material = canonical_json({
        "version": SINGLE_SLOT_PROMPT_VERSION,
        "schema_version": SINGLE_SLOT_SCHEMA_VERSION,
        "schema": SINGLE_SLOT_SCHEMA,
        "source": inspect.getsource(build_single_slot_prompt),
    })
    return sha256_bytes(material.encode())


def _words(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(value))


def _render(frame: str, value: str) -> str:
    return frame.replace("{S1}", value)


def validate_single_slot_bundles(
    raw: Any, *, correct_answer_text: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(raw, dict):
        return {"decision": "single_slot_schema_invalid",
                "reasons": ["object_required"]}
    if set(raw) != {"frame", "gold_value", "bundles"}:
        reasons.append("top_level_fields_invalid")
    frame = raw.get("frame")
    gold_value = raw.get("gold_value")
    bundles_raw = raw.get("bundles")
    if not isinstance(frame, str) or frame.count("{S1}") != 1:
        reasons.append("single_slot_frame_invalid")
        frame = ""
    if re.search(r"\{(?!S1\})[^{}]+\}", frame):
        reasons.append("unknown_placeholder")
    if not isinstance(gold_value, str) or not gold_value.strip():
        reasons.append("gold_value_invalid")
        gold_value = ""
    gold_value = gold_value.strip()
    rendered_gold = _render(frame, gold_value)
    if rendered_gold != correct_answer_text:
        reasons.append("rendered_gold_mismatch")
    literal = frame.replace("{S1}", "")
    coverage = len(_words(literal)) / max(1, len(_words(correct_answer_text)))
    if coverage < MIN_SHARED_FRAME_COVERAGE:
        reasons.append("shared_frame_coverage_below_half")
    if not isinstance(bundles_raw, list) or len(bundles_raw) != 4:
        return {
            "decision": "single_slot_schema_invalid",
            "reasons": sorted(set([*reasons, "bundle_count_invalid"])),
            "rendered_gold": rendered_gold,
        }

    seen_ids: list[str] = []
    seen_sets: set[tuple[str, ...]] = set()
    rendered_bundles: list[dict[str, Any]] = []
    gold_count = max(1, len(_words(correct_answer_text)))
    for row in bundles_raw:
        if not isinstance(row, dict) or set(row) != {
                "bundle_id", "contrast_values"}:
            reasons.append("bundle_fields_invalid")
            continue
        bundle_id = row.get("bundle_id")
        if bundle_id not in BUNDLE_IDS:
            reasons.append("bundle_id_invalid")
        else:
            seen_ids.append(str(bundle_id))
        values = row.get("contrast_values")
        if not isinstance(values, list) or len(values) != 3:
            reasons.append("contrast_value_count_invalid")
            continue
        if any(not isinstance(value, str) or not value.strip()
               for value in values):
            reasons.append("contrast_value_invalid")
            continue
        values = [value.strip() for value in values]
        if len({value.casefold() for value in values}) != 3:
            reasons.append("duplicate_contrast_value")
        if any(value.casefold() == gold_value.casefold() for value in values):
            reasons.append("contrast_equals_gold")
        canonical_set = tuple(sorted(value.casefold() for value in values))
        if canonical_set in seen_sets:
            reasons.append("duplicate_bundle")
        seen_sets.add(canonical_set)
        rendered = [_render(frame, value) for value in values]
        diagnostics = []
        for value, text in zip(values, rendered):
            ratio = len(_words(text)) / gold_count
            if ratio < MIN_LENGTH_RATIO or ratio > MAX_LENGTH_RATIO:
                reasons.append("rendered_length_not_parallel")
            diagnostics.append({
                "contrast_value": value,
                "rendered_text": text,
                "gold_length_ratio": ratio,
            })
        rendered_bundles.append({
            "bundle_id": bundle_id,
            "contrast_values": values,
            "distractors": rendered,
            "diagnostics": diagnostics,
        })
    if sorted(seen_ids) != sorted(BUNDLE_IDS):
        reasons.append("bundle_ids_not_exact")
    if reasons:
        return {
            "decision": "single_slot_schema_invalid",
            "reasons": sorted(set(reasons)),
            "frame": frame, "gold_value": gold_value,
            "rendered_gold": rendered_gold,
            "shared_frame_word_coverage": coverage,
            "bundles": rendered_bundles,
        }
    rendered_bundles.sort(
        key=lambda row: BUNDLE_IDS.index(row["bundle_id"]))
    return {
        "decision": "single_slot_accept", "reasons": [],
        "frame": frame, "gold_value": gold_value,
        "rendered_gold": rendered_gold,
        "shared_frame_word_coverage": coverage,
        "bundles": rendered_bundles,
        "validation_scope": (
            "exact_shared_frame_and_length_not_semantic_falsity"),
    }


def build_single_slot_prompt(
    *,
    base_qa: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, str]],
    prior_failed_distractors: Sequence[str],
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
        "prior_bad_ground_distractors": list(prior_failed_distractors),
    }
    rules = [
        "Decompose the frozen correct answer into one shared frame containing exactly one {S1} placeholder and one gold_value; rendering them must reproduce the correct answer exactly, character for character.",
        "Keep at least half of the correct answer words literal in the shared frame. The program, not you, will render every complete option.",
        "Return four bundles B1-B4, each with exactly three distinct contrast values for S1. Never output full option strings.",
        "Every contrast value must have the same narrow grammatical and semantic type as gold_value and produce a natural direct answer to the exact question.",
        "For an action addressing unemployment, all contrast values must be plausible concrete unemployment interventions by the same actor, such as employment, hiring, training, placement, or income-support actions. Do not use unrelated charity drives, hobbies, awareness, cleanup, gardening, military, education, health, or generic volunteering.",
        "Make every rendered option 80%-120% of the correct answer length. Prefer the same number of words in gold_value and contrast values.",
        "Each contrast must be false under the frozen evidence, but should look at least as causally relevant to the question as the gold without context.",
        "Do not reuse prior failed distractors or repeat a contrast value across bundles. Do not use negation, absurdity, category mismatch, or a synonym of the gold.",
        "Return strict JSON only with no Markdown, explanations, full options, or additional fields.",
    ]
    example = {
        "frame": "She opened a community {S1}.",
        "gold_value": "food pantry",
        "bundles": [{
            "bundle_id": f"B{index + 1}",
            "contrast_values": [
                f"job center {index + 1}",
                f"skills clinic {index + 1}",
                f"hiring office {index + 1}",
            ],
        } for index in range(4)],
    }
    return "\n\n".join([
        f"PROMPT_VERSION: {SINGLE_SLOT_PROMPT_VERSION}",
        "SINGLE-SLOT OPTION RULES\n- " + "\n- ".join(rules),
        "FORMAT EXAMPLE (unrelated to the frozen item)\n" + json.dumps(
            example, ensure_ascii=False, indent=2),
        "FROZEN INPUT\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2),
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            SINGLE_SLOT_SCHEMA, ensure_ascii=False, sort_keys=True, indent=2),
    ])

