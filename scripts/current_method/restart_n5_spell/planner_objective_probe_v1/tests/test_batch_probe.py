from __future__ import annotations

from ..batch_probe import _batch_conclusion, _load_cases


def test_batch_inputs_are_root_deduplicated() -> None:
    cases, parents = _load_cases(5)
    assert len(cases) == 5
    assert len({row["root_id"] for row in cases}) == 5
    assert set(parents) == {row["case_id"] for row in cases}
    assert all(row["dialogue_sha256"] and row["memory_sha256"] for row in cases)


def test_batch_signal_requires_multiple_positive_pairs() -> None:
    paired = {"paired_complete_cases": 5, "pair_outcomes": [
        {"objective_gain": 1}, {"objective_gain": 1},
        {"objective_gain": 0}, {"objective_gain": 0}, {"objective_gain": 0},
    ]}
    assert _batch_conclusion(paired)["label"] == "OBJECTIVE_CONTROL_BATCH_SIGNAL"


def test_under_three_complete_pairs_is_insufficient() -> None:
    paired = {"paired_complete_cases": 2, "pair_outcomes": [
        {"objective_gain": 1}, {"objective_gain": 1},
    ]}
    assert _batch_conclusion(paired)["label"] == "INSUFFICIENT_COMPLETE_PAIRS"
