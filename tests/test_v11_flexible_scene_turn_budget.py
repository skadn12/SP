from __future__ import annotations

import unittest

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.dialogue_quality import (
    validate_dialogue_quality,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_canary_v11 import (
    _round_turn_budget,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    REALIZER_RETRY_PROMPT_VERSION_V11,
    build_realizer_retry_prompt_v11,
)


def _scene(*, complex_scene: bool) -> dict:
    speakers = [
        {"participant_ref": "E1", "scene_role": "primary"},
        {"participant_ref": "E2", "scene_role": "primary"},
    ]
    if complex_scene:
        speakers.append({"participant_ref": "E3", "scene_role": "supporting"})
    return {
        "speaker_participants": speakers,
        "interaction_beats": [f"beat-{index}" for index in range(
            6 if complex_scene else 3)],
        "primary_action": {"action_type": "advance_shared_task"},
        "secondary_action": (
            {"action_type": "develop_character_or_task_state"}
            if complex_scene else None),
    }


def _turns(count: int, speaker_refs: tuple[str, ...]) -> list[dict]:
    return [
        {
            "turn_ref": f"R01:S04:T{index:03d}",
            "speaker_ref": speaker_refs[(index - 1) % len(speaker_refs)],
            "text": f"New scene utterance number {index} with concrete progress.",
        }
        for index in range(1, count + 1)
    ]


class FlexibleSceneTurnBudgetTests(unittest.TestCase):
    def test_seven_scene_round_uses_preferred_range_and_flexible_safety_bounds(self):
        budget = _round_turn_budget(_scene(complex_scene=True), 7)
        self.assertEqual((budget["preferred_minimum"], budget["target"],
                          budget["preferred_maximum"]), (5, 6, 6))
        self.assertEqual(budget["minimum"], 4)
        self.assertEqual(budget["maximum"], 12)
        self.assertEqual(
            budget["round_allocation_rule_version"],
            "continuous_round_turn_allocation_v2_flexible")

    def test_turns_above_preferred_maximum_are_diagnostic_not_failure(self):
        scene = _scene(complex_scene=True)
        budget = _round_turn_budget(scene, 7)
        result = validate_dialogue_quality(
            scene_ref="R01:S04",
            dialogue_turns=_turns(10, ("E1", "E2", "E3")),
            speaker_participants=scene["speaker_participants"],
            turn_budget=budget,
            roles_applicable=False,
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertNotIn("turn_count_outside_frozen_budget", result["errors"])

    def test_hard_safety_maximum_still_rejects_runaway_scene(self):
        scene = _scene(complex_scene=True)
        budget = _round_turn_budget(scene, 7)
        result = validate_dialogue_quality(
            scene_ref="R01:S04",
            dialogue_turns=_turns(13, ("E1", "E2", "E3")),
            speaker_participants=scene["speaker_participants"],
            turn_budget=budget,
            roles_applicable=False,
        )
        self.assertFalse(result["valid"])
        self.assertIn("turn_count_outside_frozen_budget", result["errors"])

    def test_compact_retry_repeats_exact_model_owned_turn_schema(self):
        prompt = build_realizer_retry_prompt_v11(
            base_prompt="unused in compact mode",
            frozen_input_sha256="a" * 64,
            errors=["turn_count_outside_frozen_budget"],
            turn_budget={"minimum": 4, "target": 6, "maximum": 12},
            retry_attempt=2,
            retry_limit=2,
            allowed_speaker_refs=("E1", "E2"),
            compact_retry_context={"scene_ref": "R01:S04"},
        )
        self.assertIn('"speaker_ref":"one exact allowed speaker ref"', prompt)
        self.assertIn('"text":"non-empty natural utterance"', prompt)
        self.assertIn("contain exactly speaker_ref and text", prompt)
        self.assertIn("Do not output turn_ref, scene_ref", prompt)
        self.assertIn("retry_v8_flexible_turn_schema",
                      REALIZER_RETRY_PROMPT_VERSION_V11)


if __name__ == "__main__":
    unittest.main()
