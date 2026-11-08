"""Probe-only boxed protocol for multi-select and ordering Solver replays.

This module is intentionally not imported by the production single-choice QA
route.  It gives structured QA probes one strict output contract and keeps
format failures separate from semantic/reasoning failures.
"""

from __future__ import annotations

import re
from itertools import combinations, permutations
from typing import Any, Iterable, Mapping


STRUCTURED_SOLVER_MAX_TOKENS = 32
STRUCTURED_SOLVER_TEMPERATURE = 0.0
BOXED_PROTOCOL_VERSION = "structured_qa_box_v1"
STRUCTURED_GUIDED_DECODING_VERSION = "finite_choice_set_v1"

_BOX_OPEN = "<box>"
_BOX_CLOSE = "</box>"
_BOX_WRAPPER = re.compile(r"^\s*<box>\s*(.*?)\s*</box>\s*$", re.DOTALL)


def _allowed_keys(question_type: str, values: Iterable[str] | None) -> tuple[str, ...]:
    if question_type == "multi_select":
        keys = tuple(str(value).upper() for value in (values or ()))
        canonical = tuple(chr(ord("A") + index) for index in range(len(keys)))
    elif question_type == "ordering":
        keys = tuple(str(value).upper() for value in (values or ()))
        canonical = tuple(f"I{index + 1}" for index in range(len(keys)))
    else:
        return ()
    if not (4 <= len(keys) <= 6 and len(set(keys)) == len(keys)):
        return ()
    # Ordering position variants rotate the display keys (I2,I3,I4,I1), but
    # the legal vocabulary is still the same canonical item-ID set.
    if question_type == "ordering":
        return keys if set(keys) == set(canonical) else ()
    return keys if keys == canonical else ()


def structured_box_choices(
    question_type: str, *, allowed_item_ids: Iterable[str] | None = None,
) -> list[str]:
    """Return the finite legal answer vocabulary for structured local decoding.

    This is only a syntax constraint.  It does not score or select the gold
    answer, and therefore cannot turn a reasoning error into a correct answer.
    """
    if question_type == "multi_select":
        letters = _allowed_keys(question_type, allowed_item_ids)
        if not letters:
            return []
        return [
            f"<box>{','.join(combo)}</box>"
            for size in range(1, len(letters) + 1)
            for combo in combinations(letters, size)
        ]
    if question_type == "ordering":
        item_ids = _allowed_keys(question_type, allowed_item_ids)
        if not item_ids:
            return []
        return [f"<box>{','.join(order)}</box>" for order in permutations(item_ids)]
    return []


def _parse_box(raw_output: Any) -> tuple[str | None, str | None]:
    raw = str(raw_output or "")
    if raw.count(_BOX_OPEN) != 1 or raw.count(_BOX_CLOSE) != 1:
        return None, "exactly_one_box_required"
    match = _BOX_WRAPPER.fullmatch(raw)
    if not match:
        return None, "text_outside_box"
    return match.group(1).strip(), None


def parse_structured_box(
    raw_output: Any,
    *,
    question_type: str,
    allowed_item_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Parse one strict boxed answer for a structured QA probe.

    The parser accepts no bare answer and no explanation outside the single
    box. For multi-select it returns unique letters from the supplied A-D,
    A-E, or A-F key set. For ordering it requires a complete permutation of
    the supplied I1-I4, I1-I5, or I1-I6 item IDs.
    """
    raw = str(raw_output or "")
    inner, error = _parse_box(raw)
    base = {
        "raw_output": raw,
        "parsed_answer": [],
        "parse_status": "parse_failure",
        "failure_type": "parse_failure",
        "parser_version": BOXED_PROTOCOL_VERSION,
    }
    if error:
        base["parse_reason"] = error
        return base
    assert inner is not None
    if question_type == "multi_select":
        tokens = [part.strip().upper() for part in inner.split(",")]
        allowed = _allowed_keys(question_type, allowed_item_ids)
        if (
            not allowed
            or not tokens
            or any(token not in allowed for token in tokens)
            or len(tokens) != len(set(tokens))
        ):
            base["parse_reason"] = "multi_select_requires_unique_comma_letters"
            return base
    elif question_type == "ordering":
        tokens = [part.strip().upper() for part in inner.split(",")]
        allowed = [str(value).upper() for value in (allowed_item_ids or ())]
        allowed_set = set(allowed)
        if (
            not allowed
            or len(tokens) != len(allowed)
            or any(not re.fullmatch(r"I\d+", token) for token in tokens)
            or len(tokens) != len(set(tokens))
            or set(tokens) != allowed_set
        ):
            base["parse_reason"] = "ordering_requires_complete_unique_item_permutation"
            return base
    else:
        base["parse_reason"] = f"unsupported_structured_question_type:{question_type}"
        return base
    base.update({
        "parsed_answer": tokens,
        "parse_status": "success",
        "failure_type": None,
    })
    base.pop("parse_reason", None)
    return base


def evaluate_multi_select(predicted: list[str], gold: list[str]) -> dict[str, Any]:
    predicted_set = set(predicted)
    gold_set = set(gold)
    intersection = len(predicted_set & gold_set)
    precision = intersection / len(predicted_set) if predicted_set else 0.0
    recall = intersection / len(gold_set) if gold_set else 0.0
    union = len(predicted_set | gold_set)
    jaccard = intersection / union if union else 1.0
    exact = predicted_set == gold_set
    return {
        "exact_match": exact,
        "precision": precision,
        "recall": recall,
        "jaccard": jaccard,
        "failure_type": None if exact else "reasoning_error",
    }


def evaluate_ordering(predicted: list[str], gold: list[str]) -> dict[str, Any]:
    pairs = list(combinations(range(len(gold)), 2))
    pairwise_correct = sum(
        predicted.index(gold[left]) < predicted.index(gold[right])
        for left, right in pairs
        if gold[left] in predicted and gold[right] in predicted
    )
    pairwise_accuracy = pairwise_correct / len(pairs) if pairs else 1.0
    exact = predicted == gold
    return {
        "exact_order_match": exact,
        "pairwise_accuracy": pairwise_accuracy,
        "failure_type": None if exact else "reasoning_error",
    }


def classify_structured_result(
    parsed: Mapping[str, Any], *, gold: list[str], question_type: str,
) -> dict[str, Any]:
    """Add semantic scoring without relabelling parse failures as reasoning errors."""
    if parsed.get("parse_status") != "success":
        return {
            **dict(parsed),
            "failure_type": "parse_failure",
            "metrics": {},
        }
    answer = list(parsed.get("parsed_answer") or [])
    metrics = (
        evaluate_multi_select(answer, gold)
        if question_type == "multi_select"
        else evaluate_ordering(answer, gold)
    )
    return {
        **dict(parsed),
        "failure_type": metrics.pop("failure_type"),
        "metrics": metrics,
    }
