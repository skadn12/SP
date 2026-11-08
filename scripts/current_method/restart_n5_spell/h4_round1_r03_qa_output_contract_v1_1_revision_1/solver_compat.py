"""Dynamic downstream validator/rebinding interface for v1.1 candidates."""

from __future__ import annotations

from itertools import combinations, permutations
import re
from typing import Any, Mapping

from .contract import shape_errors


def candidate_container(candidate: Mapping[str, Any]) -> dict[str, str]:
    qtype = str(candidate.get("question_type") or "")
    field = "items" if qtype == "ordering" else "options"
    value = candidate.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field}_missing")
    return {str(key): str(text) for key, text in value.items()}


def validate_candidate_for_ingestion(candidate: Mapping[str, Any]) -> dict[str, Any]:
    qtype = candidate.get("question_type")
    container = candidate_container(candidate)
    gold = candidate.get("gold")
    if qtype == "single_choice" and not isinstance(gold, list):
        gold = [candidate.get("correct_option")]
    errors = shape_errors(qtype, container, gold)
    normalized = [" ".join(value.casefold().split()) for value in container.values()]
    if len(normalized) != len(set(normalized)):
        errors.append("duplicate_option_or_item_text")
    return {"valid": not errors, "errors": sorted(set(errors)), "question_type": qtype,
            "option_count": len(container), "gold_count": len(gold or [])}


def _rotate(mapping: Mapping[str, str], shift: int) -> dict[str, str]:
    keys = list(mapping)
    values = [mapping[key] for key in keys]
    return {key: values[(index - shift) % len(keys)] for index, key in enumerate(keys)}


def position_variants(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    qtype = str(candidate.get("question_type") or "")
    container = candidate_container(candidate)
    errors = shape_errors(qtype, container, candidate.get("gold"))
    if errors:
        raise ValueError("candidate_invalid:" + ",".join(errors))
    keys = list(container)
    gold_value = candidate.get("gold")
    if qtype == "single_choice" and not isinstance(gold_value, list):
        gold_value = [candidate.get("correct_option")]
    gold = [str(key) for key in gold_value or []]
    variants = []
    if qtype in {"single_choice", "multi_select"}:
        gold_texts = {container[key] for key in gold}
        for shift in range(len(keys)):
            display = _rotate(container, shift)
            variants.append({"variant": f"shift_{shift}", "shift": shift, "options": display,
                             "gold": [key for key, text in display.items() if text in gold_texts]})
    else:
        for shift in range(len(keys)):
            order = keys[shift:] + keys[:shift]
            variants.append({"variant": f"shift_{shift}", "shift": shift,
                             "items": {key: container[key] for key in order}, "gold": list(gold)})
    return variants


def audit_gold_rebinding(candidate: Mapping[str, Any], variants: list[Mapping[str, Any]]) -> dict[str, Any]:
    qtype = str(candidate.get("question_type") or "")
    container = candidate_container(candidate)
    gold = [str(key) for key in candidate.get("gold") or []]
    expected = len(container)
    errors = []
    original_texts = {container[key] for key in gold} if qtype != "ordering" else set()
    for variant in variants:
        shown = variant.get("items") if qtype == "ordering" else variant.get("options")
        rebound = [str(key) for key in variant.get("gold") or []]
        if qtype == "ordering":
            valid = rebound == gold and set(rebound) == set(container)
        else:
            valid = {str(shown[key]) for key in rebound} == original_texts
        if not valid:
            errors.append(f"{variant.get('variant')}:gold_mismatch")
    if len(variants) != expected:
        errors.append(f"variant_count:{len(variants)}:{expected}")
    return {"pass": not errors, "errors": errors, "variant_count": len(variants), "expected_variant_count": expected}


def guided_choices(question_type: str, keys: list[str]) -> list[str]:
    if question_type == "multi_select":
        return [f"<box>{','.join(value)}</box>" for size in range(1, len(keys) + 1) for value in combinations(keys, size)]
    if question_type == "ordering":
        return [f"<box>{','.join(value)}</box>" for value in permutations(keys)]
    raise ValueError(f"unsupported_structured_question_type:{question_type}")


def requests_per_model(candidate: Mapping[str, Any], *, rollouts: int = 3, context_modes: int = 2) -> int:
    return len(candidate_container(candidate)) * rollouts * context_modes


def parse_structured(raw: str, question_type: str, keys: list[str]) -> dict[str, Any]:
    match = re.fullmatch(r"\s*<box>\s*(.*?)\s*</box>\s*", str(raw), flags=re.DOTALL)
    if not match or str(raw).count("<box>") != 1 or str(raw).count("</box>") != 1:
        return {"parse_status": "parse_failure", "parsed_answer": [], "reason": "one_box_required"}
    tokens = [part.strip().upper() for part in match.group(1).split(",")]
    allowed = [str(key).upper() for key in keys]
    if not tokens or len(tokens) != len(set(tokens)) or not set(tokens) <= set(allowed):
        return {"parse_status": "parse_failure", "parsed_answer": [], "reason": "invalid_or_duplicate_key"}
    if question_type == "ordering" and (len(tokens) != len(allowed) or set(tokens) != set(allowed)):
        return {"parse_status": "parse_failure", "parsed_answer": [], "reason": "incomplete_ordering"}
    return {"parse_status": "success", "parsed_answer": tokens}


def score_structured(parsed: Mapping[str, Any], gold: list[str], question_type: str) -> dict[str, Any]:
    if parsed.get("parse_status") != "success":
        return {"scored": False, "correct": False, "failure_type": "parse_failure"}
    predicted = [str(key) for key in parsed.get("parsed_answer") or []]
    correct = set(predicted) == set(gold) if question_type == "multi_select" else predicted == list(gold)
    return {"scored": True, "correct": correct, "failure_type": None if correct else "reasoning_error"}
