from __future__ import annotations

from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.stage_transition_reanalysis_v2 import strict_stage_audit

def _child(texts: list[str]) -> dict:
    return {'scenes': [{'dialogue_turns': [{'text': text}]} for text in texts]}

def test_future_trigger_is_not_allowed_in_early_scene() -> None:
    audit = strict_stage_audit(_child([
        'We propose the charity event and keep the date flexible.',
        'Green Horizon is available this weekend, so we reconsider the weekday plan.',
        "The venue confirmed they're available on weekends, so we finalize the outreach with the environmental group.",
    ]), 'POB04')
    assert audit['overall_pass'] is True

def test_premature_heirloom_trigger_fails() -> None:
    audit = strict_stage_audit(_child([
        "I broke my grandmother's vase and am considering therapy.",
        "I broke my dad's old watch and feel guilty.",
        'The therapist recommends weekly sessions and a first appointment.',
    ]), 'POS07')
    assert audit['overall_pass'] is False
    assert 'scene2_heirloom_severity_leaked_in_scene1' in audit['violations']
