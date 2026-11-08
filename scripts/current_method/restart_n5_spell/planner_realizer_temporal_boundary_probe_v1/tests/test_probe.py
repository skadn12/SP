from __future__ import annotations

from pathlib import Path

from ..probe import SOURCE_A_RUN, _future_boundary_block, _load_a


def test_boundary_block_forbids_future_actions_and_finality() -> None:
    plan = {"scenes": [
        {"after_state": "recipe remains contested", "interaction_beats": ["mix ingredients"], "required_transition": "challenge"},
        {"after_state": "recipe revised", "interaction_beats": ["add water"], "required_transition": "revision"},
        {"after_state": "recipe final", "interaction_beats": ["put ham in oven", "set timer"], "required_transition": "resolution"},
    ]}
    text = _future_boundary_block(plan)
    assert "FORBIDDEN FUTURE CONTENT" in text
    assert "put ham in oven" in text
    assert "MUST REMAIN OPEN/PROVISIONAL" in text


def test_boundary_block_explicitly_preserves_scene_two_order() -> None:
    plan = {"scenes": [
        {"after_state": "open", "interaction_beats": [], "required_transition": "proposal"},
        {"after_state": "provisional", "interaction_beats": ["observe too thick", "add water"], "required_transition": "revision"},
        {"after_state": "final", "interaction_beats": [], "required_transition": "resolution"},
    ]}
    text = _future_boundary_block(plan)
    assert "Scene 2 must first contain the too-thick observation, then water/thinning" in text


def test_frozen_condition_a_is_recomputed_with_strict_audit() -> None:
    condition, _ = _load_a(Path(SOURCE_A_RUN))
    assert condition["mechanical_contract_valid"]
    assert not condition["semantic_audit"]["overall_pass"]
    assert not condition["semantic_audit"]["stage_boundary_checks"]["no_explicit_finality_before_final_scene"]
