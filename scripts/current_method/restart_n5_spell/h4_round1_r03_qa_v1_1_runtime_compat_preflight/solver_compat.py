"""Dynamic 4-6 option Solver-interface primitives for offline verification."""

from __future__ import annotations

import re
from itertools import combinations, permutations
from typing import Any, Mapping


def canonical_keys(question_type: str, count: int) -> tuple[str, ...]:
    if question_type == "single_choice":
        if count != 4:
            raise ValueError("single_choice_requires_exactly_four")
        return ("A", "B", "C", "D")
    if question_type == "multi_select":
        if not 4 <= count <= 6:
            raise ValueError("multi_select_requires_four_to_six")
        return tuple(chr(ord("A") + index) for index in range(count))
    if question_type == "ordering":
        if not 4 <= count <= 6:
            raise ValueError("ordering_requires_four_to_six")
        return tuple(f"I{index + 1}" for index in range(count))
    raise ValueError(f"unsupported_question_type:{question_type}")


def _container(candidate: Mapping[str, Any]) -> dict[str, str]:
    name = "items" if candidate.get("question_type") == "ordering" else "options"
    value = candidate.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}_missing")
    return {str(key): str(text) for key, text in value.items()}


def validate_candidate(candidate: Mapping[str, Any]) -> None:
    qtype = str(candidate.get("question_type") or "")
    values = _container(candidate)
    keys = canonical_keys(qtype, len(values))
    if tuple(values) != keys:
        raise ValueError("noncontiguous_or_noncanonical_keys")
    normalized = [" ".join(text.casefold().split()) for text in values.values()]
    if any(not text for text in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError("empty_or_duplicate_option_text")
    gold = [str(key) for key in candidate.get("gold") or []]
    if qtype == "single_choice" and (len(gold) != 1 or gold[0] not in keys):
        raise ValueError("single_gold_invalid")
    if qtype == "multi_select" and not (2 <= len(gold) < len(keys) and len(gold) == len(set(gold)) and set(gold) <= set(keys)):
        raise ValueError("multi_gold_invalid")
    if qtype == "ordering" and not (len(gold) == len(keys) and len(gold) == len(set(gold)) and set(gold) == set(keys)):
        raise ValueError("ordering_gold_invalid")


def _rotate(mapping: Mapping[str, str], shift: int) -> dict[str, str]:
    keys = list(mapping)
    values = [mapping[key] for key in keys]
    return {key: values[(index - shift) % len(keys)] for index, key in enumerate(keys)}


def position_variants(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_candidate(candidate)
    qtype = str(candidate["question_type"])
    original = _container(candidate)
    keys = list(original)
    gold = [str(key) for key in candidate["gold"]]
    variants: list[dict[str, Any]] = []
    if qtype in {"single_choice", "multi_select"}:
        gold_texts = {original[key] for key in gold}
        for shift in range(len(keys)):
            display = _rotate(original, shift)
            rebound = [key for key, text in display.items() if text in gold_texts]
            variants.append({
                "variant": f"shift_{shift}", "shift": shift,
                "options": display, "gold": rebound,
            })
    else:
        for shift in range(len(keys)):
            display_order = keys[shift:] + keys[:shift]
            variants.append({
                "variant": f"shift_{shift}", "shift": shift,
                "items": {key: original[key] for key in display_order}, "gold": list(gold),
            })
    return variants


def audit_rebinding(candidate: Mapping[str, Any], variants: list[Mapping[str, Any]]) -> dict[str, Any]:
    qtype = str(candidate["question_type"])
    original = _container(candidate)
    original_gold = [str(key) for key in candidate["gold"]]
    expected_count = len(original)
    errors: list[str] = []
    coverage: set[str] = set()
    for variant in variants:
        display = variant.get("items") if qtype == "ordering" else variant.get("options")
        rebound = [str(key) for key in variant.get("gold") or []]
        if qtype == "ordering":
            valid = rebound == original_gold and set(rebound) == set(original)
        else:
            wanted = {original[key] for key in original_gold}
            valid = {str(display[key]) for key in rebound} == wanted
            coverage.update(rebound)
        if not valid:
            errors.append(f"{variant.get('variant')}:semantic_gold_mismatch")
    if len(variants) != expected_count:
        errors.append(f"variant_count:{len(variants)}:{expected_count}")
    if qtype in {"single_choice", "multi_select"} and coverage != set(original):
        errors.append("gold_position_coverage_incomplete")
    return {"pass": not errors, "errors": errors, "variant_count": len(variants), "expected_variant_count": expected_count}


def structured_choices(question_type: str, allowed_keys: list[str]) -> list[str]:
    allowed = tuple(str(key).upper() for key in allowed_keys)
    if question_type == "multi_select":
        return [f"<box>{','.join(combo)}</box>" for size in range(1, len(allowed) + 1) for combo in combinations(allowed, size)]
    if question_type == "ordering":
        return [f"<box>{','.join(order)}</box>" for order in permutations(allowed)]
    raise ValueError(f"unsupported_structured_question_type:{question_type}")


def parse_structured(raw: str, question_type: str, allowed_keys: list[str]) -> dict[str, Any]:
    match = re.fullmatch(r"\s*<box>\s*(.*?)\s*</box>\s*", str(raw), flags=re.DOTALL)
    if not match or str(raw).count("<box>") != 1 or str(raw).count("</box>") != 1:
        return {"parse_status": "parse_failure", "parsed_answer": [], "reason": "one_box_required"}
    tokens = [token.strip().upper() for token in match.group(1).split(",")]
    allowed = [str(key).upper() for key in allowed_keys]
    if not tokens or len(tokens) != len(set(tokens)) or not set(tokens) <= set(allowed):
        return {"parse_status": "parse_failure", "parsed_answer": [], "reason": "invalid_or_duplicate_key"}
    if question_type == "ordering" and (len(tokens) != len(allowed) or set(tokens) != set(allowed)):
        return {"parse_status": "parse_failure", "parsed_answer": [], "reason": "incomplete_ordering"}
    return {"parse_status": "success", "parsed_answer": tokens}


def score_structured(parsed: Mapping[str, Any], gold: list[str], question_type: str) -> dict[str, Any]:
    if parsed.get("parse_status") != "success":
        return {"scored": False, "correct": False, "failure_type": "parse_failure"}
    predicted = [str(key) for key in parsed.get("parsed_answer") or []]
    if question_type == "multi_select":
        correct = set(predicted) == set(gold)
    elif question_type == "ordering":
        correct = predicted == list(gold)
    else:
        raise ValueError(f"unsupported_structured_question_type:{question_type}")
    return {"scored": True, "correct": correct, "failure_type": None if correct else "reasoning_error"}


def requests_per_model(candidate: Mapping[str, Any], *, rollouts: int = 3, context_modes: int = 2) -> int:
    validate_candidate(candidate)
    return len(_container(candidate)) * rollouts * context_modes

