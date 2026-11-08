from __future__ import annotations

from ..analyzer import analyze_chain
from ..contracts import SETTINGS, validate_plan


def _scene(index: int, *, transition: str, before_value: str, after_value: str,
           before_stage: str, after_stage: str, resolution: str,
           trigger_type: str = "new_information") -> dict:
    return {
        "scene_index": index, "primary_action": transition,
        "state_slot": "venue_choice", "before_value": before_value,
        "before_stage": before_stage,
        "trigger": {"type": trigger_type, "content": "a concrete budget constraint" if trigger_type != "none" else "no new trigger exists"},
        "transition_type": transition, "required_transition": f"perform {transition}",
        "after_value": after_value, "after_stage": after_stage,
        "resolution_status": resolution,
        "interaction_beats": ["beat one", "beat two", "beat three"],
    }


def _plan(scenes: list[dict]) -> dict:
    return {"case_id": "SC01", "setting": "A", "round_goal": "test", "scenes": scenes}


def test_s1_final_and_s3_final_is_rejected() -> None:
    plan = _plan([
        _scene(1, transition="proposal", before_value="none", after_value="car", before_stage="unset", after_stage="final", resolution="final"),
        _scene(2, transition="reconsideration", before_value="car", after_value="bike", before_stage="final", after_stage="reconsidering", resolution="open"),
        _scene(3, transition="resolution", before_value="bike", after_value="car", before_stage="reconsidering", after_stage="final", resolution="final"),
    ])
    result = analyze_chain(plan)
    assert result["premature_closure"]
    assert not result["complex_chain"]


def test_tentative_reconsider_final_passes() -> None:
    plan = _plan([
        _scene(1, transition="proposal", before_value="none", after_value="motorcycle", before_stage="unset", after_stage="tentative", resolution="open", trigger_type="none"),
        _scene(2, transition="reconsideration", before_value="motorcycle", after_value="electric bike", before_stage="tentative", after_stage="reconsidering", resolution="provisional"),
        _scene(3, transition="resolution", before_value="electric bike", after_value="car", before_stage="reconsidering", after_stage="final", resolution="final"),
    ])
    assert validate_plan(plan, setting=SETTINGS["A"], case_id="SC01")["valid"]
    assert analyze_chain(plan)["complex_chain"]


def test_same_value_different_stage_is_stage_change_only() -> None:
    plan = _plan([
        _scene(1, transition="proposal", before_value="car", after_value="car", before_stage="unset", after_stage="tentative", resolution="open"),
        _scene(2, transition="reconsideration", before_value="car", after_value="car", before_stage="tentative", after_stage="reconsidering", resolution="provisional"),
        _scene(3, transition="resolution", before_value="car", after_value="car", before_stage="reconsidering", after_stage="final", resolution="final"),
    ])
    result = analyze_chain(plan)
    assert result["state_value_change_count"] == 0
    assert result["state_stage_change_count"] == 3
    assert result["complex_chain"]


def test_value_change_without_trigger_is_rejected() -> None:
    plan = _plan([
        _scene(1, transition="proposal", before_value="none", after_value="motorcycle", before_stage="unset", after_stage="tentative", resolution="open", trigger_type="none"),
        _scene(2, transition="revision", before_value="motorcycle", after_value="car", before_stage="tentative", after_stage="revised", resolution="provisional", trigger_type="none"),
        _scene(3, transition="resolution", before_value="car", after_value="car", before_stage="revised", after_stage="final", resolution="final", trigger_type="none"),
    ])
    result = analyze_chain(plan)
    assert result["trigger_backed_transition_rate"] < 1.0
    assert not result["complex_chain"]


def test_normal_event_progress_is_not_state_evolution() -> None:
    plan = _plan([
        _scene(1, transition="event_progress", before_value="festival", after_value="festival", before_stage="in_progress", after_stage="completed", resolution="open"),
        _scene(2, transition="event_progress", before_value="festival", after_value="festival", before_stage="completed", after_stage="completed", resolution="provisional"),
        _scene(3, transition="resolution", before_value="festival", after_value="festival", before_stage="completed", after_stage="final", resolution="final"),
    ])
    result = analyze_chain(plan)
    assert result["normal_event_progress_count"] == 1
    assert not result["complex_chain"]


def test_discontinuous_scene_state_is_invalid() -> None:
    plan = _plan([
        _scene(1, transition="proposal", before_value="none", after_value="motorcycle", before_stage="unset", after_stage="tentative", resolution="open"),
        _scene(2, transition="revision", before_value="car", after_value="bike", before_stage="contested", after_stage="revised", resolution="provisional"),
        _scene(3, transition="resolution", before_value="bike", after_value="bike", before_stage="revised", after_stage="final", resolution="final"),
    ])
    audit = validate_plan(plan, setting=SETTINGS["A"], case_id="SC01")
    assert not audit["valid"]
    assert "scene_1_to_2_value_discontinuity" in audit["errors"]
    assert "scene_1_to_2_stage_discontinuity" in audit["errors"]


def test_primary_action_reconsideration_counts_with_challenge_transition() -> None:
    plan = _plan([
        _scene(1, transition="proposal", before_value="none", after_value="motorcycle", before_stage="unset", after_stage="tentative", resolution="open", trigger_type="none"),
        _scene(2, transition="challenge", before_value="motorcycle", after_value="electric bike", before_stage="tentative", after_stage="reconsidering", resolution="provisional"),
        _scene(3, transition="resolution", before_value="electric bike", after_value="car", before_stage="reconsidering", after_stage="final", resolution="final"),
    ])
    plan["scenes"][1]["primary_action"] = "reconsideration"
    result = analyze_chain(plan)
    assert result["role_requirement_met"]
    assert result["reconsideration_rate"] == 1 / 3
    assert result["complex_chain"]
