import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    compile_round_validator_model_output,
    migrate_historical_connected_person_validator_fixture_v5,
    run_round_validator_with_retry,
    validate_round_validator_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract_repair_v11 import (
    apply_round_validator_contract_repair,
    build_round_validator_contract_repair_plan,
    build_round_validator_contract_repair_prompt,
)


CANARY_6 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_6"
)
CANARY_6_TREE_SHA = (
    "bb2a37373ed5d9afaece6ed4838aaa9d0b4c8fc67296a49ac3682c41b1b21d0b"
)


def load(relative):
    return json.loads((CANARY_6 / relative).read_text(encoding="utf-8"))


class RoundValidatorTopLevelRehomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.realized = [json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted((CANARY_6 / "scenes").glob(
                            "*/realized_scene.json"))]
        cls.raw1 = migrate_historical_connected_person_validator_fixture_v5(
            load("attempts/round_validator_attempt_01.json")["parsed_response"],
            realized_scenes=cls.realized)
        cls.raw2 = load("attempts/round_validator_attempt_02.json")[
            "parsed_response"]
        cls.review = load("review_world_view.json")
        cls.gates = load("local_scene_gate_results.json")
        cls.validator_input = load("round_validator_input.json")
        cls.compilation = compile_round_validator_model_output(
            cls.raw1, realized_scenes=cls.realized)
        cls.plan = build_round_validator_contract_repair_plan(
            model_output=cls.raw1, realized_scenes=cls.realized,
            compilation_errors=cls.compilation["errors"])

    def repair_fixture(self):
        target = self.plan["top_level_rehome"]
        return {
            "top_level_rehome": {
                "misplaced_scene_review_index": target[
                    "misplaced_scene_review_index"],
                "durable_updates": target["durable_updates"],
                "semantic_review_observations": target[
                    "semantic_review_observations"],
            },
            "scene_reference_repairs": [],
            "action_realization_repairs": [],
            "durable_update_evidence_repairs": [],
            "observation_reference_repairs": [],
        }

    def test_first_response_is_exact_supported_rehome_shape(self):
        self.assertEqual(self.compilation["errors"], [
            "validator_model_top_level_schema"])
        self.assertTrue(self.plan["valid"], self.plan["errors"])
        target = self.plan["top_level_rehome"]
        self.assertEqual(target["misplaced_scene_review_index"], 6)
        self.assertEqual(target["expected_scene_review_count"], 6)
        self.assertEqual(self.plan["targets"], [])
        self.assertEqual(self.plan["durable_update_targets"], [])

    def test_prompt_is_compact_and_does_not_repeat_full_review(self):
        prompt = build_round_validator_contract_repair_prompt(
            repair_plan=self.plan,
            frozen_input_sha256=self.validator_input[
                "frozen_round_validator_input_sha256"])
        self.assertNotIn("historical_supporting_turn_refs", prompt)
        self.assertNotIn("current_scene_supporting_turn_refs", prompt)
        self.assertNotIn('"round_decision"', prompt)
        self.assertLess(len(prompt), 20_000)

    def test_model_must_resubmit_exact_misplaced_payload(self):
        fixture = self.repair_fixture()
        fixture["top_level_rehome"]["durable_updates"] = []
        result = apply_round_validator_contract_repair(
            base_model_output=self.raw1, repair_output=fixture,
            repair_plan=self.plan)
        self.assertFalse(result["valid"])
        self.assertIn("top_level_rehome:payload_drift", result["errors"])

    def test_explicit_rehome_reconstructs_and_passes_complete_gate(self):
        result = apply_round_validator_contract_repair(
            base_model_output=self.raw1, repair_output=self.repair_fixture(),
            repair_plan=self.plan)
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(result["non_target_fields_unchanged"])
        reconstructed = result["reconstructed_output"]
        self.assertEqual(len(reconstructed["scene_reviews"]), 6)
        self.assertIn("durable_updates", reconstructed)
        self.assertIn("semantic_review_observations", reconstructed)
        compilation = compile_round_validator_model_output(
            reconstructed, realized_scenes=self.realized)
        self.assertTrue(compilation["valid"], compilation["errors"])
        validation = validate_round_validator_output(
            compilation["compiled_output"], realized_scenes=self.realized,
            review_world_view=self.review, local_gate_results=self.gates)
        self.assertTrue(validation["valid"], validation["errors"])

    def test_retry_runner_uses_contract_only_rehome(self):
        calls = []

        def backend(prompt, attempt):
            calls.append(prompt)
            return self.raw1 if attempt == 1 else self.repair_fixture()

        result = run_round_validator_with_retry(
            backend_call=backend, validator_input=self.validator_input,
            realized_scenes=self.realized, review_world_view=self.review,
            local_gate_results=self.gates)
        self.assertTrue(result["completed"])
        self.assertEqual(result["physical_calls"], 2)
        self.assertEqual(result["attempts"][1]["request_mode"],
                         "contract_only_repair")
        self.assertLess(len(calls[1]), len(calls[0]) // 2)

    def test_historical_second_full_retry_remains_invalid(self):
        compilation = compile_round_validator_model_output(
            self.raw2, realized_scenes=self.realized)
        self.assertFalse(compilation["valid"])
        self.assertEqual(compilation["errors"], [
            "validator_model_top_level_schema"])

    def test_canary6_history_is_unchanged(self):
        self.assertEqual(hash_tree(CANARY_6)["tree_sha256"], CANARY_6_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
