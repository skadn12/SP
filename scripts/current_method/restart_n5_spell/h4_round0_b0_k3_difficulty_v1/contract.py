"""Frozen B0 prompt, permutation, parsing, and canonical-scoring contract."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..core import normalize_option, parse_boxed_mcq_answer
from ..prompts import (
    SOLVER_POSITION_NEUTRAL_RESPONDER_PROMPT,
    SOLVER_TRAILING_FORMAT_INSTRUCTION,
)
from ..structured_qa_box_protocol import (
    classify_structured_result,
    parse_structured_box,
    structured_box_choices,
)
from . import question_keys


STRUCTURED_SYSTEM_PROMPT = (
    "You are a strict structured dialogue QA solver. Use only the supplied "
    "dialogue when it is present. Return the final answer in exactly one "
    "<box>...</box> pair. Do not provide explanations, reasoning, or any "
    "text outside the box."
)
PROMPT_CONTRACT_VERSION = "h4_round0_b0_prompt_v1"
PERMUTATION_CONTRACT_VERSION = "h4_round0_b0_cyclic_rebinding_v1"
CANONICAL_SCORING_VERSION = "h4_round0_b0_canonical_scoring_v1"


def dialogue_rows(expanded: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "turn_id": str(row.get("turn_ref") or ""),
            "session_id": str(row.get("session_ref") or ""),
            "speaker": str(row.get("speaker") or ""),
            "text": str(row.get("text") or ""),
        }
        for row in expanded.get("turns") or []
    ]


def _normalized_unique(values: Iterable[Any]) -> bool:
    normalized = [normalize_option(str(value)) for value in values]
    return bool(normalized) and all(normalized) and len(normalized) == len(set(normalized))


def canonical_gold(candidate: Mapping[str, Any]) -> list[str]:
    qtype = str(candidate.get("question_type") or "")
    if qtype == "single_choice":
        return [str(candidate.get("correct_option") or "")]
    return [str(value) for value in candidate.get("gold") or []]


def position_variants(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create cyclic B0 displays with an explicit display/canonical bijection."""
    qtype = str(candidate.get("question_type") or "")
    field = "items" if qtype == "ordering" else "options"
    raw = candidate.get(field)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field}_missing")
    keys = list(question_keys(qtype, len(raw)))
    if not keys or tuple(raw) != tuple(keys):
        raise ValueError(f"{qtype}_keys_or_count_invalid")
    original = {key: str(raw[key]) for key in keys}
    if not _normalized_unique(original.values()):
        raise ValueError(f"{field}_must_be_nonempty_and_unique")
    gold = canonical_gold(candidate)
    variants: list[dict[str, Any]] = []
    for shift in range(len(keys)):
        if qtype == "ordering":
            display_order = keys[shift:] + keys[:shift]
            displayed = {key: original[key] for key in display_order}
            mapping = {key: key for key in keys}
        else:
            displayed = {
                display_key: original[keys[(index - shift) % len(keys)]]
                for index, display_key in enumerate(keys)
            }
            by_text = {normalize_option(text): key for key, text in original.items()}
            mapping = {
                display_key: by_text[normalize_option(text)]
                for display_key, text in displayed.items()
            }
            display_order = list(displayed)
        if set(mapping) != set(keys) or set(mapping.values()) != set(keys):
            raise ValueError("display_to_canonical_not_bijective")
        inverse = {canonical: display for display, canonical in mapping.items()}
        displayed_gold = [inverse[key] for key in gold]
        variants.append({
            "variant_id": f"shift_{shift}",
            "shift": shift,
            "display_kind": (
                "item_display_order" if qtype == "ordering" else "option_position"
            ),
            "display_order": display_order,
            field: displayed,
            "display_to_canonical": mapping,
            "displayed_gold": displayed_gold,
            "canonical_gold": list(gold),
        })
    return variants


def audit_rebinding(candidate: Mapping[str, Any], variants: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    qtype = str(candidate.get("question_type") or "")
    field = "items" if qtype == "ordering" else "options"
    container = candidate.get(field) or {}
    keys = set(question_keys(qtype, len(container)))
    gold = canonical_gold(candidate)
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    for variant in variants:
        mapping = {
            str(key): str(value)
            for key, value in (variant.get("display_to_canonical") or {}).items()
        }
        displayed_gold = [str(value) for value in variant.get("displayed_gold") or []]
        rebound = [mapping.get(value, "") for value in displayed_gold]
        bijective = set(mapping) == keys and set(mapping.values()) == keys
        semantic_match = (
            set(rebound) == set(gold) if qtype == "multi_select" else rebound == gold
        )
        if not bijective:
            errors.append(f"{variant.get('variant_id')}:mapping_not_bijective")
        if not semantic_match:
            errors.append(f"{variant.get('variant_id')}:canonical_gold_mismatch")
        rows.append({
            "variant_id": variant.get("variant_id"),
            "displayed_gold": displayed_gold,
            "rebound_canonical_gold": rebound,
            "mapping_bijective": bijective,
            "semantic_gold_preserved": semantic_match,
        })
    expected = len(keys)
    if len(variants) != expected:
        errors.append(f"variant_count:{len(variants)}:{expected}")
    return {
        "version": PERMUTATION_CONTRACT_VERSION,
        "pass": not errors,
        "errors": errors,
        "rows": rows,
    }


def build_prompt(
    tokenizer: Any,
    candidate: Mapping[str, Any],
    variant: Mapping[str, Any],
    dialogue: Sequence[Mapping[str, Any]],
    *,
    context_mode: str,
) -> str:
    if context_mode not in {"no_context", "full_context"}:
        raise ValueError(f"unsupported_context_mode:{context_mode}")
    context = list(dialogue) if context_mode == "full_context" else []
    qtype = str(candidate["question_type"])
    chunks: list[str] = []
    if context:
        label = "Context" if qtype == "single_choice" else "Dialogue"
        chunks.append(label + ":\n" + "\n".join(
            f"[{row['turn_id']}] {row['speaker']}: {row['text']}" for row in context
        ))
    if qtype == "single_choice":
        chunks.append(str(candidate["question"]))
        chunks.append("\n".join(
            f"{key}. {value}" for key, value in (variant.get("options") or {}).items()
        ))
        chunks.append(SOLVER_TRAILING_FORMAT_INSTRUCTION)
        system = SOLVER_POSITION_NEUTRAL_RESPONDER_PROMPT
    else:
        chunks.append(f"Question ({qtype}):\n{candidate['question']}")
        field = "options" if qtype == "multi_select" else "items"
        heading = "Options" if qtype == "multi_select" else "Items"
        chunks.append(heading + ":\n" + "\n".join(
            f"{key}. {value}" for key, value in (variant.get(field) or {}).items()
        ))
        if qtype == "multi_select":
            chunks.append(
                "Output only selected letters separated by commas, e.g. <box>A,C</box>."
            )
        else:
            chunks.append(
                "Output only ordered item IDs separated by commas, "
                "e.g. <box>I2,I4,I1,I3</box>."
            )
        system = STRUCTURED_SYSTEM_PROMPT
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(chunks)},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def structured_choices(question_type: str, variant: Mapping[str, Any]) -> list[str]:
    allowed = (variant.get("options") or variant.get("items") or {}).keys()
    return structured_box_choices(question_type, allowed_item_ids=allowed)


def normalize_prediction(question_type: str, prediction: Iterable[str]) -> list[str]:
    values = [str(value) for value in prediction]
    if question_type == "multi_select":
        return sorted(set(values))
    return values


def score_output(
    raw_output: str,
    candidate: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse displayed labels, map them to canonical IDs, then score Gold."""
    qtype = str(candidate["question_type"])
    field = "items" if qtype == "ordering" else "options"
    displayed = dict(variant.get(field) or {})
    mapping = dict(variant.get("display_to_canonical") or {})
    gold = canonical_gold(candidate)
    if qtype == "single_choice":
        parsed = parse_boxed_mcq_answer(raw_output, displayed)
        if parsed.get("parse_status") != "valid_boxed_letter":
            return {
                "parse_status": parsed.get("parse_status"),
                "parser_version": parsed.get("parser_version"),
                "scored": False,
                "correct": None,
                "display_prediction": [],
                "canonical_prediction": [],
                "failure_type": "parse_failure",
                "parse_reason": parsed.get("reason"),
                "metrics": {},
            }
        display_prediction = [str(parsed["parsed_letter"])]
        metrics: dict[str, Any] = {}
        parser_version = parsed.get("parser_version")
        parse_status = parsed.get("parse_status")
    else:
        parsed = parse_structured_box(
            raw_output,
            question_type=qtype,
            allowed_item_ids=displayed.keys(),
        )
        if parsed.get("parse_status") != "success":
            return {
                "parse_status": parsed.get("parse_status"),
                "parser_version": parsed.get("parser_version"),
                "scored": False,
                "correct": None,
                "display_prediction": [],
                "canonical_prediction": [],
                "failure_type": "parse_failure",
                "parse_reason": parsed.get("parse_reason"),
                "metrics": {},
            }
        display_prediction = [str(value) for value in parsed.get("parsed_answer") or []]
        parser_version = parsed.get("parser_version")
        parse_status = parsed.get("parse_status")
        metrics = {}
    canonical_prediction = normalize_prediction(
        qtype, (mapping[value] for value in display_prediction)
    )
    normalized_gold = normalize_prediction(qtype, gold)
    if qtype == "single_choice":
        correct = canonical_prediction == normalized_gold
    else:
        result = classify_structured_result(
            {
                "parse_status": "success",
                "parsed_answer": canonical_prediction,
                "parser_version": parser_version,
                "failure_type": None,
            },
            gold=normalized_gold,
            question_type=qtype,
        )
        metrics = dict(result.get("metrics") or {})
        correct = bool(
            metrics.get("exact_match") if qtype == "multi_select"
            else metrics.get("exact_order_match")
        )
    return {
        "parse_status": parse_status,
        "parser_version": parser_version,
        "scored": True,
        "correct": correct,
        "display_prediction": display_prediction,
        "canonical_prediction": canonical_prediction,
        "failure_type": None if correct else "reasoning_error",
        "metrics": metrics,
    }


__all__ = [
    "CANONICAL_SCORING_VERSION",
    "PERMUTATION_CONTRACT_VERSION",
    "PROMPT_CONTRACT_VERSION",
    "STRUCTURED_SYSTEM_PROMPT",
    "audit_rebinding",
    "build_prompt",
    "canonical_gold",
    "dialogue_rows",
    "position_variants",
    "score_output",
    "structured_choices",
]
