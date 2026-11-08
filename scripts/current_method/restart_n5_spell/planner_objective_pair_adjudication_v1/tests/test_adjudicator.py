from __future__ import annotations

from ..adjudicator import adjudicate_record, conclusion, surface_continuity


def _record(errors: list[str], *, valid: bool = False, left: str = "") -> dict:
    return {
        "contract": {"valid": valid, "errors": errors},
        "plan": {"scenes": [
            {"after_value": left},
            {"before_value": "Co-workers A leaves early to see a doctor; Co-workers B covering remaining shift"},
        ]},
    }


def test_surface_continuity_rejects_intention_to_realization_progress() -> None:
    evidence = surface_continuity(
        "Co-workers A decides to leave work early to see a doctor, with Co-workers B covering the remaining shift",
        "Co-workers A leaves early to see a doctor; Co-workers B covering remaining shift",
    )
    assert not evidence["accepted"]
    assert not evidence["semantic_guards_passed"]


def test_surface_continuity_accepts_light_wording_change() -> None:
    evidence = surface_continuity(
        "Co-workers A will leave early to see a doctor, with Co-workers B covering the remaining shift",
        "Co-workers A will leave early to see the doctor; Co-workers B covers the remaining shift",
    )
    assert evidence["accepted"]


def test_unrelated_value_is_rejected() -> None:
    evidence = surface_continuity("doctor visit with shift coverage", "choose a new hiking trail")
    assert not evidence["accepted"]


def test_only_surface_value_discontinuity_can_be_waived() -> None:
    result = adjudicate_record(_record(
        ["scene_1_to_2_value_discontinuity"],
        left="Co-workers A leaves early to see a doctor, with Co-workers B covering the remaining shift",
    ))
    assert result["effective_valid"]
    assert result["status"] == "adjudicated_surface_continuity"


def test_pob02_shape_is_not_waived() -> None:
    result = adjudicate_record(_record(
        ["scene_1_to_2_value_discontinuity"],
        left="Co-workers A decides to leave work early to see a doctor, with Co-workers B covering the remaining shift",
    ))
    assert not result["effective_valid"]
    assert result["status"] == "rejected"


def test_canonical_speaker_error_is_never_waived() -> None:
    result = adjudicate_record(_record(
        ["noncanonical_speakers:Neighbor A,Neighbor B"],
        left="Co-workers A decides to leave work early to see a doctor, with Co-workers B covering the remaining shift",
    ))
    assert not result["effective_valid"]
    assert result["status"] == "rejected"


def test_batch_threshold_requires_three_pairs() -> None:
    result = conclusion({"paired_complete_cases": 2, "positive_pairs": 2, "negative_pairs": 0, "unchanged_pairs": 0})
    assert result["label"] == "INSUFFICIENT_COMPLETE_PAIRS_AFTER_ADJUDICATION"
