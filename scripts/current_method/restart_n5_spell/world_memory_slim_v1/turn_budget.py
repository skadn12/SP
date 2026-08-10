"""Programmatic Natural Dialogue Turn Budget v2."""

from __future__ import annotations

from typing import Any, Mapping


TURN_BUDGET_RULE_VERSION = "natural_dialogue_turn_budget_v2"


def compile_turn_budget(scene: Mapping[str, Any]) -> dict[str, Any]:
    speakers = len(scene.get("speaker_participants") or [])
    if speakers not in {2, 3, 4}:
        raise ValueError("turn_budget_speaker_count")
    base = {2: 7, 3: 9, 4: 11}[speakers]
    adjustments: list[str] = []
    if len(scene.get("interaction_beats") or []) >= 6:
        adjustments.append("six_or_more_interaction_beats")
    actions = [scene.get("primary_action"), scene.get("secondary_action")]
    if scene.get("secondary_action") is not None:
        adjustments.append("primary_and_secondary_narrative_actions")
    if any((row or {}).get("resolved_action_type", (row or {}).get("action_type")) in {
            "create_or_escalate_conflict", "resolve_or_deescalate_conflict",
            "assign_or_handoff_role"} for row in actions):
        adjustments.append("conflict_or_handoff_narrative_action")
    target = min(base + len(adjustments), 12)
    result = {
        "turn_budget_rule_version": TURN_BUDGET_RULE_VERSION,
        "minimum": max(5, target - 2), "target": target,
        "maximum": min(14, target + 2), "adjustments": adjustments,
    }
    if not result["minimum"] <= target <= result["maximum"]:
        raise AssertionError("turn_budget_internal_invariant")
    return result
