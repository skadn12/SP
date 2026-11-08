"""Offline regression fixtures for prompt wiring and dynamic Solver support."""

from __future__ import annotations

from .solver_compat import (
    audit_rebinding, parse_structured, position_variants, requests_per_model,
    score_structured, structured_choices,
)


def _multi(count: int) -> dict:
    keys = [chr(ord("A") + index) for index in range(count)]
    return {"question_type": "multi_select", "options": {key: f"text-{key}" for key in keys}, "gold": ["A", "C"]}


def _ordering(count: int) -> dict:
    keys = [f"I{index + 1}" for index in range(count)]
    return {"question_type": "ordering", "items": {key: f"text-{key}" for key in keys}, "gold": list(reversed(keys))}


def test_dynamic_multi_variants_rebinding_and_requests():
    for count in (4, 5, 6):
        candidate = _multi(count)
        variants = position_variants(candidate)
        assert audit_rebinding(candidate, variants)["pass"]
        assert len(variants) == count
        assert requests_per_model(candidate) == count * 6


def test_dynamic_ordering_variants_rebinding_and_requests():
    for count in (4, 5, 6):
        candidate = _ordering(count)
        variants = position_variants(candidate)
        assert audit_rebinding(candidate, variants)["pass"]
        assert len(variants) == count
        assert requests_per_model(candidate) == count * 6


def test_dynamic_structured_choices_and_parsing():
    for count in (4, 5, 6):
        letters = [chr(ord("A") + index) for index in range(count)]
        assert len(structured_choices("multi_select", letters)) == 2 ** count - 1
        parsed = parse_structured("<box>A,C</box>", "multi_select", letters)
        assert score_structured(parsed, ["A", "C"], "multi_select")["correct"]
        item_ids = [f"I{index + 1}" for index in range(count)]
        assert len(structured_choices("ordering", item_ids)) == __import__("math").factorial(count)
        answer = list(reversed(item_ids))
        parsed = parse_structured(f"<box>{','.join(answer)}</box>", "ordering", item_ids)
        assert score_structured(parsed, answer, "ordering")["correct"]


def test_invalid_dynamic_answers_remain_parse_or_reasoning_failures():
    letters = ["A", "B", "C", "D", "E", "F"]
    assert parse_structured("<box>A,G</box>", "multi_select", letters)["parse_status"] == "parse_failure"
    parsed = parse_structured("<box>A,B</box>", "multi_select", letters)
    assert score_structured(parsed, ["A", "C"], "multi_select")["failure_type"] == "reasoning_error"
    items = ["I1", "I2", "I3", "I4", "I5", "I6"]
    assert parse_structured("<box>I1,I2,I3,I4,I5</box>", "ordering", items)["parse_status"] == "parse_failure"

