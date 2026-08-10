import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.context_compiler_v2 import (
    ContextInvariantFailure,
    compile_scene_context_v2,
    validate_context_manifest_v2,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    build_realizer_retry_prompt_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


CANARY_11 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_11"
)
CANARY_11_TREE_SHA = (
    "4d1ff648765ffb9c4a42d24198f6719651b6ef0c4e78e73bd0f3f3d27da5a49d"
)


def load(relative):
    return json.loads((CANARY_11 / relative).read_text(encoding="utf-8"))


class RealizerFailurePropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = load("planner/compiled_round_plan.json")
        cls.complete = load("complete_dialogue_view.json")
        cls.anchors = load("source_anchor_registry.json")
        cls.placements = load("placement_choice_registry.json")
        cls.planning = load("planning_state_view.json")

    def test_soft_parallel_predecessor_can_be_unavailable(self):
        scene = self.plan["scenes"][6]
        self.assertEqual(scene["dependency_edges"][0]["hard_dependency"], False)
        context = compile_scene_context_v2(
            scene=scene, complete_dialogue=self.complete,
            source_anchor_registry=self.anchors,
            placement_choice_registry=self.placements,
            planning_state=self.planning, same_round_scenes={})
        self.assertEqual(context["declared_same_round_predecessor_refs"], [
            "R01:S06"])
        self.assertEqual(context["same_round_predecessor_refs"], [])
        self.assertEqual(context["unavailable_soft_predecessor_refs"], [
            "R01:S06"])
        self.assertTrue(validate_context_manifest_v2(context)["valid"])

    def test_missing_hard_predecessor_remains_blocking(self):
        scene = self.plan["scenes"][1]
        self.assertEqual(scene["dependency_edges"][0]["hard_dependency"], True)
        with self.assertRaisesRegex(
                ContextInvariantFailure,
                "context_v2_predecessor_not_generated:R01:S01"):
            compile_scene_context_v2(
                scene=scene, complete_dialogue=self.complete,
                source_anchor_registry=self.anchors,
                placement_choice_registry=self.placements,
                planning_state=self.planning, same_round_scenes={})

    def test_retry_prompt_states_exact_frozen_turn_range_last(self):
        prompt = build_realizer_retry_prompt_v11(
            base_prompt="BASE", frozen_input_sha256="frozen",
            errors=["turn_count_outside_frozen_budget"],
            turn_budget={"minimum": 5, "maximum": 6})
        self.assertIn("between 5 and 6 items, inclusive", prompt)
        self.assertIn("Count the array items before returning", prompt)
        self.assertIn("Do not copy any complete utterance", prompt)
        self.assertIn(
            'FINAL OUTPUT CONTRACT — count once more, then output exactly this shape:',
            prompt)
        self.assertIn("contain exactly speaker_ref and text", prompt)
        self.assertIn("Do not output turn_ref, scene_ref", prompt)

    def test_canary11_history_is_unchanged(self):
        self.assertEqual(hash_tree(CANARY_11)["tree_sha256"],
                         CANARY_11_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
