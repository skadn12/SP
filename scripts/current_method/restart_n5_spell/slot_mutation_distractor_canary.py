from __future__ import annotations

import inspect
import json
import re
from typing import Any, Mapping, Sequence

from .core import canonical_json, sha256_bytes


SLOT_MUTATION_CANARY_VERSION = "slot_mutation_distractor_canary_v1"
SLOT_MUTATION_PROMPT_VERSION = "contrastive_slot_mutation_writer_single_v1"
SLOT_MUTATION_SCHEMA_VERSION = "contrastive_slot_mutation_schema_v1"

SLOT_MUTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["frame", "slots", "mutations"],
    "properties": {
        "frame": {"type": "string", "minLength": 1},
        "slots": {
            "type": "array", "minItems": 2, "maxItems": 4,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["slot_id", "gold_value"],
                "properties": {
                    "slot_id": {"type": "string", "pattern": "^S[1-4]$"},
                    "gold_value": {"type": "string", "minLength": 1},
                },
            },
        },
        "mutations": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["changed_slot", "replacement_value"],
                "properties": {
                    "changed_slot": {"type": "string", "pattern": "^S[1-4]$"},
                    "replacement_value": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}


def slot_mutation_schema_sha256() -> str:
    return sha256_bytes(canonical_json(SLOT_MUTATION_SCHEMA).encode())


def slot_mutation_prompt_sha256() -> str:
    material = (
        SLOT_MUTATION_PROMPT_VERSION + SLOT_MUTATION_SCHEMA_VERSION
        + inspect.getsource(build_slot_mutation_prompt)
        + canonical_json(SLOT_MUTATION_SCHEMA)
    )
    return sha256_bytes(material.encode())


def _words(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", value)


def _render(frame: str, slots: Mapping[str, str]) -> str:
    rendered = frame
    for slot_id, value in slots.items():
        rendered = rendered.replace("{" + slot_id + "}", value)
    return rendered


def validate_and_render_slot_mutations(
    raw: Any, *, correct_answer_text: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(raw, dict):
        return {"decision": "slot_mutation_schema_invalid",
                "reasons": ["object_required"]}
    if set(raw) != {"frame", "slots", "mutations"}:
        reasons.append("top_level_fields_invalid")
    frame = raw.get("frame")
    slots_raw = raw.get("slots")
    mutations_raw = raw.get("mutations")
    if not isinstance(frame, str) or not frame.strip():
        reasons.append("frame_invalid")
        frame = ""
    if not isinstance(slots_raw, list) or not 2 <= len(slots_raw) <= 4:
        reasons.append("slots_count_invalid")
        slots_raw = []
    if not isinstance(mutations_raw, list) or len(mutations_raw) != 3:
        reasons.append("mutations_count_invalid")
        mutations_raw = []

    slots: dict[str, str] = {}
    for row in slots_raw:
        if not isinstance(row, dict) or set(row) != {"slot_id", "gold_value"}:
            reasons.append("slot_fields_invalid")
            continue
        slot_id, value = row.get("slot_id"), row.get("gold_value")
        if not isinstance(slot_id, str) or not re.fullmatch(r"S[1-4]", slot_id):
            reasons.append("slot_id_invalid")
            continue
        if slot_id in slots:
            reasons.append("duplicate_slot_id")
        if not isinstance(value, str) or not value.strip():
            reasons.append("gold_slot_value_invalid")
            continue
        slots[slot_id] = value.strip()
    placeholders = re.findall(r"\{(S[1-4])\}", frame)
    if sorted(placeholders) != sorted(slots) or any(
            frame.count("{" + slot_id + "}") != 1 for slot_id in slots):
        reasons.append("frame_slot_binding_invalid")
    if re.search(r"\{[^{}]+\}", re.sub(r"\{S[1-4]\}", "", frame)):
        reasons.append("unknown_frame_placeholder")
    rendered_gold = _render(frame, slots)
    if rendered_gold != correct_answer_text:
        reasons.append("rendered_gold_mismatch")
    literal_frame = re.sub(r"\{S[1-4]\}", "", frame)
    correct_words = _words(correct_answer_text)
    literal_coverage = len(_words(literal_frame)) / max(1, len(correct_words))
    if literal_coverage < 0.25:
        reasons.append("shared_frame_too_small")

    distractors: list[str] = []
    mutation_records: list[dict[str, Any]] = []
    for row in mutations_raw:
        if not isinstance(row, dict) or set(row) != {
                "changed_slot", "replacement_value"}:
            reasons.append("mutation_fields_invalid")
            continue
        slot_id, replacement = row.get("changed_slot"), row.get("replacement_value")
        if slot_id not in slots:
            reasons.append("mutation_slot_unknown")
            continue
        if not isinstance(replacement, str) or not replacement.strip():
            reasons.append("replacement_value_invalid")
            continue
        replacement = replacement.strip()
        if replacement.casefold() == slots[slot_id].casefold():
            reasons.append("replacement_equals_gold")
        mutated = dict(slots)
        mutated[slot_id] = replacement
        rendered = _render(frame, mutated)
        ratio = len(_words(rendered)) / max(1, len(correct_words))
        if ratio < 0.70 or ratio > 1.30:
            reasons.append("rendered_length_not_parallel")
        distractors.append(rendered)
        mutation_records.append({
            "changed_slot": slot_id,
            "gold_value": slots[slot_id],
            "replacement_value": replacement,
            "rendered_distractor": rendered,
            "rendered_word_count_ratio": ratio,
        })
    if len(set(value.casefold() for value in distractors)) != len(distractors):
        reasons.append("duplicate_rendered_distractor")
    if any(value.casefold() == correct_answer_text.casefold()
           for value in distractors):
        reasons.append("rendered_distractor_equals_gold")
    if reasons:
        return {"decision": "slot_mutation_schema_invalid",
                "reasons": sorted(set(reasons)), "rendered_gold": rendered_gold,
                "literal_frame_word_coverage": literal_coverage}
    return {
        "decision": "slot_mutation_accept", "reasons": [],
        "frame": frame, "slots": slots, "mutations": mutation_records,
        "rendered_gold": rendered_gold,
        "distractors": distractors,
        "literal_frame_word_coverage": literal_coverage,
    }


def build_slot_mutation_prompt(
    *, base_qa: Mapping[str, Any], evidence_rows: Sequence[Mapping[str, str]],
) -> str:
    payload = {
        "qa_id": str(base_qa["qa_id"]),
        "question": str(base_qa["question"]),
        "correct_answer_text": str(base_qa["correct_answer_text"]),
        "minimal_answer_evidence": [dict(row) for row in evidence_rows],
    }
    rules = [
        "Return a reusable answer frame, 2-4 gold slots, and exactly three single-slot mutations.",
        "The frame plus all gold slot values must reproduce correct_answer_text exactly, character for character.",
        "Write placeholders exactly as {S1}, {S2}, {S3}, or {S4}; each declared placeholder appears exactly once.",
        "Keep at least one quarter of the correct answer words as literal shared frame text outside placeholders.",
        "Each mutation changes exactly one declared slot; every other word and slot remains frozen by construction.",
        "Each replacement must stay in the same narrow semantic category as its gold slot and remain plausible for the question without context.",
        "Mutations may be counterfactual and need not appear elsewhere in the dialogue. They must be false according to the minimal evidence.",
        "Do not introduce unrelated domains. For a civic motivation, all alternatives remain civic motivations; for a volunteering inspiration, all alternatives remain volunteering-related inspirations.",
        "Keep every rendered option within roughly 70%-130% of the correct answer length.",
        "Do not output complete distractor strings; the program will render them from the frozen frame and one replacement.",
        "Return JSON only with no notes or additional fields.",
    ]
    example = {
        "frame": "Her {S1}'s belief in {S2}.",
        "slots": [
            {"slot_id": "S1", "gold_value": "mentor"},
            {"slot_id": "S2", "gold_value": "helping neighbors in need"},
        ],
        "mutations": [
            {"changed_slot": "S1", "replacement_value": "teacher"},
            {"changed_slot": "S2", "replacement_value":
             "organizing food deliveries for vulnerable families"},
            {"changed_slot": "S2", "replacement_value":
             "supporting unhoused residents through local outreach"},
        ],
    }
    return "\n\n".join([
        f"PROMPT_VERSION: {SLOT_MUTATION_PROMPT_VERSION}",
        "SLOT-MUTATION RULES\n- " + "\n- ".join(rules),
        "FORMAT EXAMPLE (unrelated to the frozen item)\n" + json.dumps(
            example, ensure_ascii=False, indent=2),
        "FROZEN INPUT\n" + json.dumps(payload, ensure_ascii=False,
                                        sort_keys=True, indent=2),
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            SLOT_MUTATION_SCHEMA, ensure_ascii=False, sort_keys=True, indent=2),
    ])
