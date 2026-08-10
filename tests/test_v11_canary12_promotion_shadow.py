import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.promotion_v2 import (
    promote_scenes_v2,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


CANARY_12 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_12"
)
CANARY_12_TREE_SHA = (
    "62e3197bdf1dfc86febbe72c8c1080801cb880165de813cadcb133cd9bfa48e3"
)


def load(relative):
    return json.loads((CANARY_12 / relative).read_text(encoding="utf-8"))


class Canary12PromotionShadowTests(unittest.TestCase):
    def test_all_accepted_scenes_promote_despite_partial_memory_commit(self):
        result = promote_scenes_v2(
            realized_scenes=load("round_validator_input.json")["scenes"],
            local_gate_results=load("local_scene_gate_results.json"),
            validator_output=load("round_validator_output.json"),
            scene_decision_closure=load("scene_decision_closure.json"),
            durable_update_compile_result=load("durable_update_compile_result.json"),
            narrative_order_manifest=load("narrative_order_manifest.json"))
        self.assertEqual(len(result["full_child_scenes"]), 6)
        self.assertEqual(len(result["promotable_child_scenes"]), 6)
        self.assertEqual(len(result["accepted_with_update_scenes"]), 2)
        self.assertEqual(len(result["accepted_without_update_scenes"]), 4)
        self.assertTrue(all(row["scene_requires_update_commit_advisory_only"]
                            for row in result["scene_decisions"]))

    def test_canary12_history_is_unchanged(self):
        self.assertEqual(hash_tree(CANARY_12)["tree_sha256"],
                         CANARY_12_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
