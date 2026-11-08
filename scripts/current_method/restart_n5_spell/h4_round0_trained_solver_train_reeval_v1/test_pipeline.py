from __future__ import annotations

from .pipeline import EXPECTED_TRAIN_REQUESTS, _primary_transition, aggregate_model


def _state(bucket: str, full: int, no: int, total: int = 12) -> dict:
    return {
        "difficulty_bucket": bucket,
        "full_context_correct_count": full,
        "full_context_total_count": total,
        "full_context_accuracy": full / total,
        "no_context_correct_count": no,
        "no_context_total_count": total,
        "no_context_accuracy": no / total,
        "technical_failure_count": 0,
    }


def _row(root: str, full: int, no: int) -> dict:
    full_accuracy, no_accuracy = full / 12, no / 12
    diag = lambda value: {
        "accuracy": value / 12,
        "correct_count": value,
        "total_count": 12,
        "position_accuracy": {"shift_0": value / 12},
    }
    return {
        "root_id": root,
        "difficulty_bucket": "hard_candidate",
        "full_context_correct_count": full,
        "full_context_total_count": 12,
        "full_context_accuracy": full_accuracy,
        "no_context_correct_count": no,
        "no_context_total_count": 12,
        "no_context_accuracy": no_accuracy,
        "technical_failure_count": 0,
        "position_diagnostics": {
            "full_context": diag(full),
            "no_context": diag(no),
        },
        "canonical_flip_rate": {"full_context": 0.0, "no_context": 0.0},
    }


def test_frozen_request_count() -> None:
    assert 1153 * 24 + 572 * 30 + 569 * 24 == EXPECTED_TRAIN_REQUESTS


def test_primary_transition_priority() -> None:
    zero = _state("full_zero_review", 0, 0)
    partial = _state("hard_candidate", 4, 1)
    full = _state("full_context_resolved", 12, 4)
    assert _primary_transition(zero, zero, zero) == "PERSISTENT_FULL_ZERO"
    assert _primary_transition(zero, partial, zero) == "EMERGING_FRONTIER"
    assert _primary_transition(partial, full, full) == "GRADUATED"
    assert _primary_transition(partial, partial, partial) == "CONSENSUS_RESIDUAL"


def test_candidate_weighted_and_root_equal_are_distinct() -> None:
    rows = [_row("r1", 12, 0), _row("r1", 12, 0), _row("r2", 0, 0)]
    report = aggregate_model(rows, "test")
    assert report["candidate_weighted"]["full_context_accuracy"] == 2 / 3
    assert report["root_equal"]["full_context_accuracy"] == 1 / 2
