import tempfile
import unittest
from pathlib import Path

from tests.test_world_memory_slim_v11_hybrid_runtime import fixture_inputs
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_fresh_e2e_v11 import run_mock_hybrid_v11
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.promotion_v2 import promote_scenes_v2


class PromotionV2Tests(unittest.TestCase):
    def run_case(self, case):
        with tempfile.TemporaryDirectory() as temp:
            return run_mock_hybrid_v11(**fixture_inputs(), run_root=Path(temp), case=case)

    def test_scene_valid_update_absent_is_promotable(self):
        result = self.run_case("accept_no_updates")
        promotion = result["promotion_v2_result"]
        self.assertEqual(len(promotion["accepted_without_update_scenes"]), 6)
        self.assertEqual(len(promotion["promotable_child_scenes"]), 6)

    def test_invalid_independent_update_does_not_delete_scene(self):
        result = self.run_case("one_valid_one_invalid_update")
        promotion = result["promotion_v2_result"]
        self.assertEqual(len(promotion["promotable_child_scenes"]), 6)

    def test_dependency_blocked_separate_from_reject(self):
        result = self.run_case("reject_middle_scene")
        promotion = result["promotion_v2_result"]
        self.assertIn("R02:S03", promotion["dependency_blocked_scenes"])
        self.assertIn("R02:S02", promotion["rejected_scenes"])

    def test_model_update_commit_requirement_cannot_delete_accepted_scene(self):
        result = self.run_case("one_valid_one_invalid_update")
        rejected_source = result["durable_update_compile_result"]["rejected_updates"][0][
            "candidate"]["source_scene_ref"]
        alias = next(row["scene_alias"] for row in result["realized_scenes"]
                     if row["scene_ref"] == rejected_source)
        output = {**result["round_validator_output"],
                  "scene_reviews": [dict(row) for row in result[
                      "round_validator_output"]["scene_reviews"]]}
        next(row for row in output["scene_reviews"] if row["scene_alias"] == alias)[
            "scene_requires_update_commit"] = True
        promotion = promote_scenes_v2(
            realized_scenes=result["realized_scenes"],
            local_gate_results=result["local_scene_gate_results"],
            validator_output=output,
            scene_decision_closure=result["scene_decision_closure"],
            durable_update_compile_result=result["durable_update_compile_result"],
            narrative_order_manifest=result["narrative_order_manifest"])
        self.assertIn(rejected_source,
                      [row["scene_ref"] for row in promotion["promotable_child_scenes"]])
        decision = next(row for row in promotion["scene_decisions"]
                        if row["scene_ref"] == rejected_source)
        self.assertTrue(decision["scene_requires_update_commit"])
        self.assertTrue(decision["scene_requires_update_commit_advisory_only"])


if __name__ == "__main__":
    unittest.main()
