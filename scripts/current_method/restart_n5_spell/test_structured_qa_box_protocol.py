from __future__ import annotations

from .structured_qa_box_protocol import (
    classify_structured_result,
    parse_structured_box,
    structured_box_choices,
)


def test_multiselect_box_success() -> None:
    parsed = parse_structured_box(
        "<box>A,B,C</box>", question_type="multi_select", allowed_item_ids=list("ABCDE"))
    assert parsed["parse_status"] == "success"
    assert parsed["parsed_answer"] == ["A", "B", "C"]
    assert parsed["failure_type"] is None


def test_multiselect_two_answers() -> None:
    parsed = parse_structured_box(
        "<box>A,C</box>", question_type="multi_select", allowed_item_ids=list("ABCDE"))
    assert parsed["parsed_answer"] == ["A", "C"]


def test_bare_multiselect_is_parse_failure() -> None:
    parsed = parse_structured_box(
        "Answer: A,B,C", question_type="multi_select", allowed_item_ids=list("ABCDE"))
    assert parsed["parse_status"] == "parse_failure"
    assert parsed["failure_type"] == "parse_failure"


def test_text_outside_box_is_parse_failure() -> None:
    parsed = parse_structured_box(
        "<box>A,B,C</box>\nExplanation: ...",
        question_type="multi_select",
        allowed_item_ids=list("ABCDE"),
    )
    assert parsed["parse_status"] == "parse_failure"
    assert parsed["parse_reason"] == "text_outside_box"


def test_multiselect_reasoning_error_is_not_parse_failure() -> None:
    parsed = parse_structured_box(
        "<box>A,B</box>", question_type="multi_select", allowed_item_ids=list("ABCDE"))
    result = classify_structured_result(
        parsed, gold=["A", "B", "C"], question_type="multi_select")
    assert result["parse_status"] == "success"
    assert result["failure_type"] == "reasoning_error"
    assert result["metrics"] == {
        "exact_match": False,
        "precision": 1.0,
        "recall": 2 / 3,
        "jaccard": 2 / 3,
    }


def test_ordering_box_success() -> None:
    parsed = parse_structured_box(
        "<box>I2,I4,I1,I3</box>",
        question_type="ordering",
        allowed_item_ids=["I1", "I2", "I3", "I4"],
    )
    assert parsed["parse_status"] == "success"
    assert parsed["parsed_answer"] == ["I2", "I4", "I1", "I3"]


def test_bare_ordering_is_parse_failure() -> None:
    parsed = parse_structured_box(
        "I2>I4>I1>I3",
        question_type="ordering",
        allowed_item_ids=["I1", "I2", "I3", "I4"],
    )
    assert parsed["failure_type"] == "parse_failure"


def test_ordering_wrong_permutation_is_parse_failure() -> None:
    parsed = parse_structured_box(
        "<box>I2,I4,I1</box>",
        question_type="ordering",
        allowed_item_ids=["I1", "I2", "I3", "I4"],
    )
    assert parsed["failure_type"] == "parse_failure"


def test_ordering_reasoning_error_has_pairwise_metric() -> None:
    parsed = parse_structured_box(
        "<box>I2,I1,I4,I3</box>",
        question_type="ordering",
        allowed_item_ids=["I1", "I2", "I3", "I4"],
    )
    result = classify_structured_result(
        parsed,
        gold=["I2", "I4", "I1", "I3"],
        question_type="ordering",
    )
    assert result["failure_type"] == "reasoning_error"
    assert "exact_order_match" in result["metrics"]
    assert "pairwise_accuracy" in result["metrics"]


def test_guided_choices_are_syntax_only() -> None:
    multi = structured_box_choices("multi_select", allowed_item_ids=list("ABCDE"))
    assert len(multi) == 31
    assert "<box>A,B,C</box>" in multi
    assert "<box>A,B,C,D,E</box>" in multi
    ordering = structured_box_choices(
        "ordering", allowed_item_ids=["I1", "I2", "I3", "I4"])
    assert len(ordering) == 24
    assert "<box>I2,I4,I1,I3</box>" in ordering


def test_multiselect_choices_and_parser_follow_candidate_arity() -> None:
    for count in (4, 5, 6):
        keys = [chr(ord("A") + index) for index in range(count)]
        choices = structured_box_choices("multi_select", allowed_item_ids=keys)
        assert len(choices) == 2 ** count - 1
        full = f"<box>{','.join(keys)}</box>"
        assert full in choices
        parsed = parse_structured_box(
            full, question_type="multi_select", allowed_item_ids=keys,
        )
        assert parsed["parse_status"] == "success"
    rejected = parse_structured_box(
        "<box>A,F</box>", question_type="multi_select",
        allowed_item_ids=["A", "B", "C", "D", "E"],
    )
    assert rejected["parse_status"] == "parse_failure"


def test_ordering_choices_follow_candidate_arity() -> None:
    expected = {4: 24, 5: 120, 6: 720}
    for count, vocabulary_size in expected.items():
        keys = [f"I{index + 1}" for index in range(count)]
        choices = structured_box_choices("ordering", allowed_item_ids=keys)
        assert len(choices) == vocabulary_size
        parsed = parse_structured_box(
            f"<box>{','.join(reversed(keys))}</box>",
            question_type="ordering", allowed_item_ids=keys,
        )
        assert parsed["parse_status"] == "success"
