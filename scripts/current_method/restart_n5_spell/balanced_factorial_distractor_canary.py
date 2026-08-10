from __future__ import annotations

import inspect
import json
import re
from typing import Any, Mapping, Sequence

from .core import LETTERS, canonical_json, sha256_bytes


BALANCED_FACTORIAL_CANARY_VERSION = "balanced_factorial_distractor_canary_v1"
BALANCED_FACTORIAL_PROMPT_VERSION = "balanced_factorial_distractor_writer_single_v1"
BALANCED_FACTORIAL_SCHEMA_VERSION = "balanced_factorial_distractor_schema_v1"
BALANCED_FACTORIAL_PLACEMENT_VERSION = "balanced_factorial_answer_placement_v1"

BALANCED_FACTORIAL_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["frame", "slots"],
    "properties": {
        "frame": {"type": "string", "minLength": 1},
        "slots": {
            "type": "array", "minItems": 2, "maxItems": 2,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["slot_id", "gold_value", "contrast_value"],
                "properties": {
                    "slot_id": {"type": "string", "enum": ["S1", "S2"]},
                    "gold_value": {"type": "string", "minLength": 1},
                    "contrast_value": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}


def balanced_factorial_schema_sha256() -> str:
    return sha256_bytes(canonical_json(BALANCED_FACTORIAL_SCHEMA).encode())


def balanced_factorial_prompt_sha256() -> str:
    material = (
        BALANCED_FACTORIAL_PROMPT_VERSION + BALANCED_FACTORIAL_SCHEMA_VERSION
        + inspect.getsource(build_balanced_factorial_prompt)
        + canonical_json(BALANCED_FACTORIAL_SCHEMA)
    )
    return sha256_bytes(material.encode())


def deterministic_balanced_factorial_placement(qa_id: str) -> dict[str, Any]:
    """Place gold deterministically for any QA without inspecting option text."""
    if not isinstance(qa_id, str) or not qa_id.strip():
        raise ValueError("balanced_factorial_placement_input_invalid")
    seed_material = canonical_json({
        "version": BALANCED_FACTORIAL_PLACEMENT_VERSION,
        "qa_id": qa_id,
    })
    placement_seed = int(sha256_bytes(seed_material.encode())[:8], 16)
    return {
        "answer_placement_version": BALANCED_FACTORIAL_PLACEMENT_VERSION,
        "placement_seed": placement_seed,
        "target_letter": LETTERS[placement_seed % len(LETTERS)],
        "content_dependent": False,
    }


def _words(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", value)


def _render(frame: str, s1: str, s2: str) -> str:
    return frame.replace("{S1}", s1).replace("{S2}", s2)


def validate_and_render_balanced_factorial(
    raw: Any, *, correct_answer_text: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(raw, dict):
        return {"decision": "balanced_factorial_schema_invalid",
                "reasons": ["object_required"]}
    if set(raw) != {"frame", "slots"}:
        reasons.append("top_level_fields_invalid")
    frame = raw.get("frame")
    slots_raw = raw.get("slots")
    if not isinstance(frame, str) or not frame.strip():
        reasons.append("frame_invalid")
        frame = ""
    if not isinstance(slots_raw, list) or len(slots_raw) != 2:
        reasons.append("slots_count_invalid")
        slots_raw = []
    slots: dict[str, dict[str, str]] = {}
    for row in slots_raw:
        if not isinstance(row, dict) or set(row) != {
                "slot_id", "gold_value", "contrast_value"}:
            reasons.append("slot_fields_invalid")
            continue
        slot_id = row.get("slot_id")
        gold, contrast = row.get("gold_value"), row.get("contrast_value")
        if slot_id not in {"S1", "S2"}:
            reasons.append("slot_id_invalid")
            continue
        if slot_id in slots:
            reasons.append("duplicate_slot_id")
        if not isinstance(gold, str) or not gold.strip():
            reasons.append("gold_value_invalid")
            continue
        if not isinstance(contrast, str) or not contrast.strip():
            reasons.append("contrast_value_invalid")
            continue
        if gold.strip().casefold() == contrast.strip().casefold():
            reasons.append("contrast_equals_gold")
        slots[str(slot_id)] = {
            "gold_value": gold.strip(), "contrast_value": contrast.strip()}
    if set(slots) != {"S1", "S2"}:
        reasons.append("required_slots_missing")
    if frame.count("{S1}") != 1 or frame.count("{S2}") != 1:
        reasons.append("frame_slot_binding_invalid")
    if re.search(r"\{[^{}]+\}", frame.replace("{S1}", "").replace("{S2}", "")):
        reasons.append("unknown_frame_placeholder")
    s1 = slots.get("S1", {"gold_value": "", "contrast_value": ""})
    s2 = slots.get("S2", {"gold_value": "", "contrast_value": ""})
    gold = _render(frame, s1["gold_value"], s2["gold_value"])
    if gold != correct_answer_text:
        reasons.append("rendered_gold_mismatch")
    literal_coverage = len(_words(frame.replace("{S1}", "").replace(
        "{S2}", ""))) / max(1, len(_words(correct_answer_text)))
    if literal_coverage < 0.25:
        reasons.append("shared_frame_too_small")
    combinations = [
        ("contrast_gold", s1["contrast_value"], s2["gold_value"]),
        ("gold_contrast", s1["gold_value"], s2["contrast_value"]),
        ("contrast_contrast", s1["contrast_value"], s2["contrast_value"]),
    ]
    distractors = []
    rendered_combinations = []
    correct_words = _words(correct_answer_text)
    for combination_id, value1, value2 in combinations:
        rendered = _render(frame, value1, value2)
        ratio = len(_words(rendered)) / max(1, len(correct_words))
        if ratio < 0.70 or ratio > 1.30:
            reasons.append("rendered_length_not_parallel")
        distractors.append(rendered)
        rendered_combinations.append({
            "combination_id": combination_id,
            "S1": value1, "S2": value2,
            "rendered_text": rendered, "word_count_ratio": ratio,
        })
    all_options = [gold, *distractors]
    if len({value.casefold() for value in all_options}) != 4:
        reasons.append("rendered_options_not_unique")
    value_frequencies = {
        "S1": {s1["gold_value"]: 2, s1["contrast_value"]: 2},
        "S2": {s2["gold_value"]: 2, s2["contrast_value"]: 2},
    }
    if reasons:
        return {"decision": "balanced_factorial_schema_invalid",
                "reasons": sorted(set(reasons)), "rendered_gold": gold,
                "literal_frame_word_coverage": literal_coverage}
    return {
        "decision": "balanced_factorial_accept", "reasons": [],
        "frame": frame, "slots": slots, "rendered_gold": gold,
        "distractors": distractors,
        "rendered_combinations": rendered_combinations,
        "value_frequencies_across_four_options": value_frequencies,
        "gold_is_unique_slotwise_mode": False,
        "literal_frame_word_coverage": literal_coverage,
    }


def build_balanced_factorial_prompt(
    *, base_qa: Mapping[str, Any], evidence_rows: Sequence[Mapping[str, str]],
) -> str:
    payload = {
        "qa_id": str(base_qa["qa_id"]),
        "question": str(base_qa["question"]),
        "correct_answer_text": str(base_qa["correct_answer_text"]),
        "minimal_answer_evidence": [dict(row) for row in evidence_rows],
    }
    rules = [
        "Return one shared frame and exactly two binary semantic slots named S1 and S2.",
        "The frame with both gold values must reproduce correct_answer_text exactly, character for character.",
        "Each placeholder {S1} and {S2} must appear exactly once.",
        "For each slot, contrast_value must be the same narrow semantic type as gold_value and comparable in specificity and length.",
        "Every gold/contrast combination must be a natural, internally coherent, independently plausible answer to the question without context.",
        "Contrast values may be counterfactual but must be false according to the minimal evidence.",
        "Do not use negation, anti-social motives, internal contradictions, absurdity, or unrelated domains as contrast values.",
        "For civic motivation questions, both values remain plausible civic experiences or public problems. For volunteering inspiration questions, both values remain plausible people or prosocial motivations for volunteering.",
        "Keep at least one quarter of answer words literal in the shared frame and keep every combination within roughly 70%-130% of gold length.",
        "The program will construct a balanced 2x2 set, so do not output options or distractors.",
        "Return JSON only with no notes or extra fields.",
    ]
    example = {
        "frame": "Her {S1}'s belief in {S2}.",
        "slots": [
            {"slot_id": "S1", "gold_value": "mentor",
             "contrast_value": "teacher"},
            {"slot_id": "S2", "gold_value": "helping neighbors in need",
             "contrast_value": "organizing food deliveries for vulnerable families"},
        ],
    }
    return "\n\n".join([
        f"PROMPT_VERSION: {BALANCED_FACTORIAL_PROMPT_VERSION}",
        "BALANCED 2x2 RULES\n- " + "\n- ".join(rules),
        "FORMAT EXAMPLE (unrelated to frozen item)\n" + json.dumps(
            example, ensure_ascii=False, indent=2),
        "FROZEN INPUT\n" + json.dumps(payload, ensure_ascii=False,
                                        sort_keys=True, indent=2),
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            BALANCED_FACTORIAL_SCHEMA, ensure_ascii=False,
            sort_keys=True, indent=2),
    ])
