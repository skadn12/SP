from __future__ import annotations

from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.batch_probe_v1 import (
    _profile_audit,
)


def _child(scene_texts: list[str]) -> dict:
    return {
        "scenes": [
            {"dialogue_turns": [{"text": text}]} for text in scene_texts
        ]
    }


def test_future_plan_is_not_counted_as_early_execution() -> None:
    audit = _profile_audit(_child([
        "We are mixing the glaze.",
        "I'll apply it once the ham is in the oven; let's keep this provisional.",
        "I'll brush the glaze and put the ham in the oven; set the timer now.",
    ]), "POB03")
    assert audit["overall_pass"] is True


def test_therapist_is_valid_final_resolution_evidence() -> None:
    audit = _profile_audit(_child([
        "We are still considering therapy.",
        "The family heirloom was broken, so the concern is more urgent.",
        "The therapist recommends weekly sessions and a first appointment.",
    ]), "POS07")
    assert audit["overall_pass"] is True
