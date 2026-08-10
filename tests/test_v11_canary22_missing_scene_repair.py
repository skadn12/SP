"""Shadow the Canary22 missing-scene-review contract-only recovery."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    compile_round_validator_model_output,
    validate_round_validator_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract_repair_v11 import (
    apply_round_validator_contract_repair,
    build_round_validator_contract_repair_plan,
    pack_round_validator_contract_repair_output,
)


CANARY22 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_22"
)
CANARY22_TREE_SHA = (
    "ff6d64a51a174c7cbf0d27a66d8ddaa828968613a74b134437b15af328497cc5"
)


def load(relative: str):
    return json.loads((CANARY22 / relative).read_text(encoding="utf-8"))


class Canary22MissingSceneRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenes = [load(f"scenes/R01_S0{i}/realized_scene.json")
                      for i in range(1, 7)]
        cls.base = load("attempts/round_validator_attempt_01.json")[
            "parsed_response"]
        compilation = compile_round_validator_model_output(
            cls.base, realized_scenes=cls.scenes)
        cls.validation = validate_round_validator_output(
            compilation["compiled_output"], realized_scenes=cls.scenes,
            review_world_view=load("review_world_view.json"),
            local_gate_results=load("local_scene_gate_results.json"))
        cls.plan = build_round_validator_contract_repair_plan(
            model_output=cls.base, realized_scenes=cls.scenes,
            compilation_errors=cls.validation["errors"])

    def test_missing_scene_coverage_uses_contract_only_repair(self) -> None:
        self.assertIn("validator_scene_review_coverage", self.validation["errors"])
        self.assertTrue(self.plan["valid"], self.plan["errors"])
        self.assertEqual([row["scene_alias"] for row in
                          self.plan["missing_scene_review_targets"]], ["s6"])
        self.assertEqual(self.plan["target_removed_update_keys"],
                         ["DU005", "DU006"])
        self.assertEqual(self.plan["durable_removal_targets"][0]["reason"],
                         "identity_introduction_continuation_not_new_identity")

    def test_strict_missing_scene_fixture_reconstructs_full_valid_review(self) -> None:
        second = load("attempts/round_validator_attempt_02.json")["parsed_response"]
        missing = deepcopy(next(row for row in second["scene_reviews"]
                                if row["scene_alias"] == "s6"))
        target = self.plan["missing_scene_review_targets"][0]
        missing["current_scene_supporting_turn_refs"] = [
            target["allowed_current_scene_turn_refs"][0]]
        missing["historical_supporting_turn_refs"] = [
            target["allowed_historical_turn_refs"][0]]
        missing["uses_scene_refs"] = [
            target["allowed_dependency_scene_refs"][-1]]
        missing["dependency_context_turn_refs"] = [
            target["allowed_dependency_turn_refs"][-1]]
        missing["primary_action_realization"][
            "progression_evidence_turn_refs"] = list(
                missing["primary_action_realization"]["evidence_turn_refs"])
        grouped = {
            "top_level_rehome": None,
            "scene_reference_repairs": [],
            "action_realization_repairs": [],
            "durable_update_evidence_repairs": [{
                "candidate_update_key": row["candidate_update_key"],
                "evidence_turn_refs": [row["allowed_evidence_turn_refs"][0]],
            } for row in self.plan["durable_update_targets"]],
            "observation_reference_repairs": [{
                "observation_key": row["observation_key"],
                "source_scene_refs": [row["allowed_scene_refs"][0]],
                "evidence_turn_refs": [next(iter(
                    row["allowed_evidence_turn_refs_by_scene"].values()))[0]],
            } for row in self.plan["observation_targets"]],
            "unplanned_action_repairs": [],
            "future_knowledge_repairs": [],
            "durable_update_removals": [{
                "candidate_update_key": row["candidate_update_key"],
                "replacement": None,
            } for row in self.plan["durable_removal_targets"]],
            "durable_dependency_repairs": [],
            "verified_connection_repairs": [],
            "verified_connection_removals": [{
                "candidate_update_key": row["candidate_update_key"],
                "replacement": None,
            } for row in self.plan["verified_connection_removal_targets"]],
            "missing_scene_reviews": [{
                "scene_alias": "s6", "scene_review": missing}],
        }
        response = pack_round_validator_contract_repair_output(grouped)
        applied = apply_round_validator_contract_repair(
            base_model_output=self.base, repair_output=response,
            repair_plan=self.plan)
        self.assertTrue(applied["valid"], applied["errors"])
        compilation = compile_round_validator_model_output(
            applied["reconstructed_output"], realized_scenes=self.scenes)
        self.assertTrue(compilation["valid"], compilation["errors"])
        validation = validate_round_validator_output(
            compilation["compiled_output"], realized_scenes=self.scenes,
            review_world_view=load("review_world_view.json"),
            local_gate_results=load("local_scene_gate_results.json"))
        self.assertTrue(validation["valid"], validation["errors"])

    def test_missing_scene_repair_requires_explicit_semantic_submission(self) -> None:
        template = self.plan["missing_scene_review_targets"][0]["output_template"]
        self.assertEqual(template["decision"], "<SCENE_DECISION>")
        self.assertEqual(template["current_scene_supporting_turn_refs"], ["!"])
        self.assertEqual(
            template["primary_action_realization"]["status"], "<STATUS>")
        self.assertEqual(
            template["primary_action_realization"]["evidence_turn_refs"], ["!"])

    def test_historical_canary_is_unchanged(self) -> None:
        self.assertEqual(hash_tree(CANARY22)["tree_sha256"], CANARY22_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
