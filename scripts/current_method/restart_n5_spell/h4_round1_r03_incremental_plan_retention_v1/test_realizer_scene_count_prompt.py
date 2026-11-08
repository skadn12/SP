"""Offline regressions for the exact Realizer scene-count prompt contract."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from .prompts import TARGET_RULES, render_realizer


class RealizerSceneCountPromptTests(unittest.TestCase):
    def setUp(self):
        for target in ("socket.create_connection", "socket.socket.connect"):
            guard = patch(target, side_effect=AssertionError("offline_prompt_test_network_forbidden"))
            guard.start()
            self.addCleanup(guard.stop)
        self.parent = {"root_id": "MOCK-root", "dialogue": {"turns": [
            {"speaker": "A", "text": "MOCK historical wording must stay outside text-free input."},
            {"speaker": "B", "text": "MOCK second historical turn."},
        ]}}
        self.scenes = [
            {"phase_id": "initial:MOCK-1", "goal": "Arrange a meeting", "required_participants": ["A", "B"]},
            {"phase_id": "initial:MOCK-2", "goal": "Compare completed exercises", "required_participants": ["A", "B"]},
        ]
        self.prefix = [{"turn_ref": "R03:S00:T001", "speaker": "A", "text": "MOCK retained actual progress."}]

    def render(self, count, mode):
        return render_realizer(parent=self.parent, plan={"scenes": self.scenes[:count]},
            mode=mode, continuity_shape={"speakers": ["A", "B"]}, promoted_prefix=self.prefix)

    def test_exact_count_and_example_match_each_request_in_both_modes(self):
        for count in (1, 2):
            for mode in ("text_free", "full_parent"):
                with self.subTest(count=count, mode=mode):
                    built = self.render(count, mode)
                    prompt = built["prompt"]
                    self.assertEqual(built["prompt_version"],
                        "incremental_retention_v1.realizer.successful_prompt_alignment_v3." + mode)
                    self.assertEqual(built["messages"], [{"role": "user", "content": prompt}])
                    self.assertIn(f"exactly {count} input scene plans", prompt)
                    self.assertIn(f"Return exactly {count} separate objects", prompt)
                    self.assertIn(f"len(scenes) == {count}", prompt)
                    self.assertIn("Never merge scenes, omit an input scene, or stop after an earlier scene", prompt)
                    for index in range(count):
                        self.assertIn(f"Output scenes[{index}] must realize only planned_phases_not_facts[{index}]", prompt)
                    example = json.loads(prompt.split("EXACT SCENE-COUNT JSON EXAMPLE\n", 1)[1]
                        .split("\nEND SCENE-COUNT EXAMPLE", 1)[0])
                    self.assertEqual(set(example), {"scenes"})
                    self.assertEqual(len(example["scenes"]), count)
                    for scene in example["scenes"]:
                        self.assertEqual(set(scene), {"dialogue_turns"})
                        self.assertEqual(set(scene["dialogue_turns"][0]), {"speaker", "text"})

    def test_history_boundary_input_order_and_progress_policy_remain(self):
        before = deepcopy((self.parent, self.scenes, self.prefix))
        for count in (1, 2):
            for mode in ("text_free", "full_parent"):
                with self.subTest(count=count, mode=mode):
                    prompt = self.render(count, mode)["prompt"]
                    data = json.loads(prompt.rsplit("\n\nINPUT DATA\n", 1)[1])
                    self.assertEqual(data["planned_phases_not_facts"], self.scenes[:count])
                    self.assertEqual(data["retained_actual_prefix"], self.prefix)
                    self.assertIn(TARGET_RULES, prompt)
                    self.assertIn("Use exactly 14 dialogue turns per scene by default", prompt)
                    self.assertIn("absolute hard contract is 9 to 48 turns for EACH scene", prompt)
                    self.assertIn("Preserve actor bindings and required participants", prompt)
                    self.assertIn("Do not add output fields. Scene/turn IDs are assigned locally.", prompt)
                    self.assertIn("do not claim an unobserved final result", prompt)
                    if mode == "text_free":
                        self.assertNotIn("authoritative_history", data)
                        for turn in self.parent["dialogue"]["turns"]:
                            self.assertNotIn(turn["text"], prompt)
                    else:
                        self.assertEqual(data["authoritative_history"], self.parent["dialogue"]["turns"])
        self.assertEqual((self.parent, self.scenes, self.prefix), before)


if __name__ == "__main__":
    unittest.main()
