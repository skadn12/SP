import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    compile_round_validator_model_output,
    run_round_validator_with_retry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract_repair_v11 import (
    apply_round_validator_contract_repair,
    build_round_validator_contract_repair_plan,
    build_round_validator_contract_repair_prompt,
    validate_round_validator_contract_repair_output,
)


CANARY_5 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_5"
)
CANARY_5_TREE_SHA = (
    "6432259bad24a08bd32d03dccdca2116e236861012c9b3b26d497294e5798c12"
)


def load(relative):
    return json.loads((CANARY_5 / relative).read_text(encoding="utf-8"))


class RoundValidatorContractOnlyRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.realized = [json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted((CANARY_5 / "scenes").glob(
                            "*/realized_scene.json"))]
        cls.attempt = load("attempts/round_validator_attempt_01.json")
        cls.raw = cls.attempt["parsed_response"]
        cls.review = load("review_world_view.json")
        cls.gates = load("local_scene_gate_results.json")
        cls.validator_input = load("round_validator_input.json")
        cls.compilation = compile_round_validator_model_output(
            cls.raw, realized_scenes=cls.realized)
        cls.plan = build_round_validator_contract_repair_plan(
            model_output=cls.raw, realized_scenes=cls.realized,
            compilation_errors=cls.compilation["errors"])

    def repair_fixture(self):
        return {
            "top_level_rehome": None,
            "scene_reference_repairs": [],
            "action_realization_repairs": [{
                "action_ref": row["action_ref"],
                "evidence_turn_refs": row["original_evidence_turn_refs"],
                "progression_evidence_turn_refs": (
                    row["candidate_evidence_turn_refs"][:1]
                    if row["status"] in {"realized", "partially_realized"} else []),
            } for row in self.plan["targets"]],
            "durable_update_evidence_repairs": [{
                "candidate_update_key": row["candidate_update_key"],
                "evidence_turn_refs": row["allowed_evidence_turn_refs"][:1],
            } for row in self.plan["durable_update_targets"]],
            "observation_reference_repairs": [],
        }

    def test_canary5_failure_is_exactly_supported_contract_omission(self):
        self.assertFalse(self.compilation["valid"])
        self.assertTrue(self.plan["valid"], self.plan["errors"])
        self.assertEqual(self.plan["target_action_refs"], [
            "S02.primary_action", "S04.primary_action",
            "S05.primary_action", "S07.primary_action",
        ])
        self.assertEqual(self.plan["target_candidate_update_keys"], [
            "DU001", "DU002", "DU003", "DU004", "DU005", "DU006", "DU007",
        ])

    def test_compact_prompt_omits_full_prior_response(self):
        prompt = build_round_validator_contract_repair_prompt(
            repair_plan=self.plan,
            frozen_input_sha256=self.validator_input[
                "frozen_round_validator_input_sha256"])
        self.assertNotIn("semantic_review_observations", prompt)
        self.assertNotIn("durable_updates", prompt)
        self.assertNotIn('"round_decision"', prompt)
        self.assertLess(len(prompt.encode()), 16_000)
        self.assertTrue(prompt.rstrip().endswith("}"))

    def test_repair_must_be_candidate_subset(self):
        fixture = self.repair_fixture()
        fixture["action_realization_repairs"][0][
            "progression_evidence_turn_refs"] = ["R01:S02:T999"]
        result = validate_round_validator_contract_repair_output(
            fixture, repair_plan=self.plan)
        self.assertFalse(result["valid"])
        self.assertTrue(any("evidence_not_candidate_subset" in row
                            for row in result["errors"]))

    def test_repair_cannot_change_review_semantics(self):
        fixture = self.repair_fixture()
        fixture["action_realization_repairs"][0]["status"] = "realized"
        result = validate_round_validator_contract_repair_output(
            fixture, repair_plan=self.plan)
        self.assertFalse(result["valid"])
        self.assertIn("contract_repair:0:schema", result["errors"])

    def test_durable_update_repair_must_use_source_action_subset(self):
        fixture = self.repair_fixture()
        fixture["durable_update_evidence_repairs"][0]["evidence_turn_refs"] = [
            "R01:S07:T001"]
        result = validate_round_validator_contract_repair_output(
            fixture, repair_plan=self.plan)
        self.assertFalse(result["valid"])
        self.assertTrue(any("evidence_not_source_action_subset" in row
                            for row in result["errors"]))

    def test_durable_evidence_only_failure_uses_same_compact_contract(self):
        progression_only = deepcopy(self.raw)
        for target in self.plan["targets"]:
            review = next(row for row in progression_only["scene_reviews"]
                          if row["scene_alias"] == target["scene_alias"])
            review[f"{target['action_role']}_action_realization"][
                "progression_evidence_turn_refs"] = (
                    target["candidate_evidence_turn_refs"][:1])
        compilation = compile_round_validator_model_output(
            progression_only, realized_scenes=self.realized)
        self.assertTrue(compilation["valid"], compilation["errors"])
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
            validate_round_validator_output,
        )
        validation = validate_round_validator_output(
            compilation["compiled_output"], realized_scenes=self.realized,
            review_world_view=self.review, local_gate_results=self.gates)
        plan = build_round_validator_contract_repair_plan(
            model_output=progression_only, realized_scenes=self.realized,
            compilation_errors=validation["errors"])
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual(plan["targets"], [])
        self.assertEqual(len(plan["durable_update_targets"]), 7)

    def test_apply_changes_only_missing_members(self):
        result = apply_round_validator_contract_repair(
            base_model_output=self.raw, repair_output=self.repair_fixture(),
            repair_plan=self.plan)
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(result["non_target_fields_unchanged"])
        reconstructed = result["reconstructed_output"]
        for review in reconstructed["scene_reviews"]:
            for role in ("primary", "secondary"):
                value = review.get(f"{role}_action_realization")
                if not isinstance(value, dict):
                    continue
                action_ref = next((row["action_ref"] for row in self.plan["targets"]
                                   if row["scene_alias"] == review["scene_alias"] and
                                   row["action_role"] == role), None)
                if action_ref:
                    self.assertIn("progression_evidence_turn_refs", value)

    def test_full_retry_runner_uses_compact_repair_and_accepts_third_party_choice(self):
        calls = []
        fixture = self.repair_fixture()

        def backend(prompt, attempt):
            calls.append((attempt, prompt))
            return self.raw if attempt == 1 else fixture

        result = run_round_validator_with_retry(
            backend_call=backend, validator_input=self.validator_input,
            realized_scenes=self.realized, review_world_view=self.review,
            local_gate_results=self.gates)
        self.assertTrue(result["completed"])
        self.assertTrue(result["contract_only_repair_used"])
        self.assertEqual(result["physical_calls"], 2)
        self.assertEqual(result["attempts"][0]["request_mode"], "full_review")
        self.assertEqual(result["attempts"][1]["request_mode"],
                         "contract_only_repair")
        self.assertLess(len(calls[1][1]), len(calls[0][1]) // 3)
        self.assertTrue(result["attempts"][1]["repair_application"][
            "non_target_fields_unchanged"])

    def test_invalid_repair_exhausts_existing_two_request_budget(self):
        calls = []

        def backend(prompt, attempt):
            calls.append(prompt)
            return self.raw if attempt == 1 else {
                "top_level_rehome": None,
                "scene_reference_repairs": [],
                "action_realization_repairs": [],
                "durable_update_evidence_repairs": [],
                "observation_reference_repairs": [],
            }

        result = run_round_validator_with_retry(
            backend_call=backend, validator_input=self.validator_input,
            realized_scenes=self.realized, review_world_view=self.review,
            local_gate_results=self.gates)
        self.assertFalse(result["completed"])
        self.assertEqual(result["physical_calls"], 2)
        self.assertEqual(len(calls), 2)

    def test_base_output_drift_is_rejected(self):
        changed = deepcopy(self.raw)
        changed["round_decision"] = "reject"
        result = apply_round_validator_contract_repair(
            base_model_output=changed, repair_output=self.repair_fixture(),
            repair_plan=self.plan)
        self.assertFalse(result["valid"])
        self.assertEqual(result["errors"], ["contract_repair_base_output_drift"])

    def test_historical_canary5_is_unchanged(self):
        self.assertEqual(hash_tree(CANARY_5)["tree_sha256"], CANARY_5_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
