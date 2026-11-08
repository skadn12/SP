from __future__ import annotations

from ..contracts import validate_objective_plan
from ..probe import _conclusion, _paired_comparison


def _row(case_id: str, objective: str, *, valid: bool, complex_chain: bool) -> dict:
    return {"case_id": case_id, "objective": objective,
            "contract": {"valid": valid},
            "analysis": {"complex_chain": complex_chain} if valid else {}}


def test_objective_gain_is_supported() -> None:
    paired = _paired_comparison([
        _row("P1", "O0", valid=True, complex_chain=False),
        _row("P1", "O1", valid=True, complex_chain=True),
    ])
    assert paired["mean_objective_gain"] == 1.0
    assert _conclusion(paired)["label"] == "OBJECTIVE_CONTROL_PILOT_SIGNAL"


def test_zero_zero_is_insufficient_prompt() -> None:
    paired = _paired_comparison([
        _row("P1", "O0", valid=True, complex_chain=False),
        _row("P1", "O1", valid=True, complex_chain=False),
    ])
    assert _conclusion(paired)["label"] == "OBJECTIVE_PROMPT_INSUFFICIENT"


def test_invalid_member_excludes_pair() -> None:
    paired = _paired_comparison([
        _row("P1", "O0", valid=True, complex_chain=False),
        _row("P1", "O1", valid=False, complex_chain=True),
    ])
    assert paired["paired_complete_cases"] == 0
    assert paired["excluded_case_ids"] == ["P1"]
    assert _conclusion(paired)["label"] == "INSUFFICIENT_DATA"


def test_noncanonical_role_is_rejected() -> None:
    scenes = []
    states = [("unset", "x", "unset", "tentative"),
              ("x", "y", "tentative", "revised"),
              ("y", "z", "revised", "final")]
    roles = ["proposal", "revision", "resolution"]
    for index, ((before, after, before_stage, after_stage), role) in enumerate(zip(states, roles), 1):
        scenes.append({
            "scene_index": index, "primary_action": role, "state_slot": "choice",
            "before_value": before, "after_value": after,
            "before_stage": before_stage, "after_stage": after_stage,
            "trigger": {"type": "none" if index == 1 else "new_information",
                        "content": "Classmates C supplied concrete new information"},
            "transition_type": role, "required_transition": role,
            "resolution_status": "final" if index == 3 else "open",
            "interaction_beats": ["Classmates A acts", "Classmates B responds", "They record it"],
        })
    audit = validate_objective_plan(
        {"case_id": "P1", "setting": "O1", "scenes": scenes},
        objective_key="O1", case_id="P1",
        valid_speakers=["Classmates A", "Classmates B"],
    )
    assert not audit["valid"]
    assert audit["noncanonical_speakers"] == ["Classmates C"]
