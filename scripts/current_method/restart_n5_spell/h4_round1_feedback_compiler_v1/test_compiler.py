"""Focused unit tests for transition closure and card redaction."""

from .compiler import _audit_card, recompute_primary


def _model(full, no, bucket):
    return {
        "full_context_accuracy": full,
        "no_context_accuracy": no,
        "difficulty_bucket": bucket,
        "technical_failure_count": 0,
        "parse_failure_count": 0,
        "truncated_request_count": 0,
        "position_mechanism": "none",
    }


def _row(m0, clean, inclusive):
    return {"models": {"M0": m0, "M_clean_policy": clean, "M_inclusive_policy": inclusive}}


def test_m0_zero_transitions_are_mutually_exclusive():
    zero = _model(0, 0, "full_zero_review")
    partial = _model(0.5, 0, "hard_candidate")
    full = _model(1, 0, "full_context_resolved")
    assert recompute_primary(_row(zero, zero, zero)) == "PERSISTENT_FULL_ZERO"
    assert recompute_primary(_row(zero, partial, partial)) == "EMERGING_FRONTIER_BOTH"
    assert recompute_primary(_row(zero, partial, zero)) == "EMERGING_FRONTIER_CLEAN_ONLY"
    assert recompute_primary(_row(zero, zero, partial)) == "EMERGING_FRONTIER_INCLUSIVE_ONLY"
    assert recompute_primary(_row(zero, full, partial)) == "M0_ZERO_TO_RESOLVED"


def test_feedback_card_rejects_question_and_answer_sequences():
    assert _audit_card({"question": "old"}) == ["forbidden_key:question"]
    assert "answer_label_sequence_present" in _audit_card({"target": "A,C,E"})
    assert _audit_card({"target": {"gap_type": "temporal_precedence"}}) == []

