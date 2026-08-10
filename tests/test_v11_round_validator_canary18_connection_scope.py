import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    build_round_validator_prompt,
    compile_round_validator_model_output,
    validate_round_validator_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract_repair_v11 import (
    apply_round_validator_contract_repair,
    build_round_validator_contract_repair_plan,
    build_round_validator_contract_repair_prompt,
    validate_round_validator_contract_repair_output,
)


CANARY_18 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_18"
)
CANARY_18_TREE_SHA = (
    "33ccbb4fd68c0c9697c2bee61c0584ec9db735dfb17e9aeabff5b0fadb07dc6b"
)


def load(relative):
    return json.loads((CANARY_18 / relative).read_text(encoding="utf-8"))


class Canary18RoundValidatorConnectionScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.realized = [json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted((CANARY_18 / "scenes").glob(
                            "*/realized_scene.json"))]
        cls.review = load("review_world_view.json")
        cls.gates = load("local_scene_gate_results.json")
        cls.validator_input = load("round_validator_input.json")
        cls.run_audit = load("round_validator_run.json")
        cls.raw1 = load("attempts/round_validator_attempt_01.json")[
            "parsed_response"]
        cls.raw2 = load("attempts/round_validator_attempt_02.json")[
            "parsed_response"]

    def plan(self):
        return build_round_validator_contract_repair_plan(
            model_output=self.raw1, realized_scenes=self.realized,
            compilation_errors=self.run_audit["attempts"][0][
                "validation"]["errors"])

    def complete_repair(self):
        repair = deepcopy(self.raw2)
        person_update = next(
            row for row in self.raw1["durable_updates"]
            if row["candidate_update_key"] == "DU001")
        connection = deepcopy(person_update["verified_connection"])
        connection["evidence_turn_refs"] = ["R01:S01:T003"]
        repair["repairs"].append({
            "repair_kind": "verified_connection",
            "candidate_update_key": "DU001",
            "verified_connection": connection,
        })
        for key in ("DU002", "DU003", "DU004", "DU005", "DU006"):
            repair["repairs"].append({
                "repair_kind": "verified_connection_nullification",
                "candidate_update_key": key,
                "replacement": None,
            })
        return repair

    def test_plan_collects_connection_errors_before_contract_retry(self):
        plan = self.plan()
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual([row["candidate_update_key"] for row in
                          plan["verified_connection_targets"]], ["DU001"])
        self.assertEqual([row["candidate_update_key"] for row in
                          plan["verified_connection_removal_targets"]], [
                              "DU002", "DU003", "DU004", "DU005", "DU006"])
        self.assertIn(
            "durable_update:0:verified_connection_evidence_turn_refs",
            plan["verified_connection_targets"][0]["validation_errors"])

    def test_old_attempt_two_remains_invalid_and_is_not_locally_cleaned(self):
        validation = validate_round_validator_contract_repair_output(
            self.raw2, repair_plan=self.plan())
        self.assertFalse(validation["valid"])
        self.assertTrue(any("verified_connection_repair:missing_keys:DU001" in row
                            for row in validation["errors"]))
        applied = apply_round_validator_contract_repair(
            base_model_output=self.raw1, repair_output=self.raw2,
            repair_plan=self.plan())
        self.assertFalse(applied["valid"])
        self.assertIsNone(applied["reconstructed_output"])

    def test_explicit_connection_repairs_pass_full_strict_shadow(self):
        plan = self.plan()
        repair = self.complete_repair()
        repair_validation = validate_round_validator_contract_repair_output(
            repair, repair_plan=plan)
        self.assertTrue(repair_validation["valid"], repair_validation["errors"])
        applied = apply_round_validator_contract_repair(
            base_model_output=self.raw1, repair_output=repair, repair_plan=plan)
        self.assertTrue(applied["valid"], applied["errors"])
        reconstructed = applied["reconstructed_output"]
        updates = {row["candidate_update_key"]: row
                   for row in reconstructed["durable_updates"]}
        self.assertEqual(updates["DU001"]["verified_connection"][
            "evidence_turn_refs"], ["R01:S01:T003"])
        self.assertTrue(all("verified_connection" not in updates[key]
                            for key in ("DU002", "DU003", "DU004", "DU005", "DU006")))
        compiled = compile_round_validator_model_output(
            reconstructed, realized_scenes=self.realized)
        self.assertTrue(compiled["valid"], compiled["errors"])
        strict = validate_round_validator_output(
            compiled["compiled_output"], realized_scenes=self.realized,
            review_world_view=self.review, local_gate_results=self.gates)
        self.assertTrue(strict["valid"], strict["errors"])
        self.assertTrue(applied["non_target_fields_unchanged"])

    def test_connection_evidence_must_be_exact_action_subset(self):
        repair = self.complete_repair()
        row = next(item for item in repair["repairs"]
                   if item["repair_kind"] == "verified_connection")
        row["verified_connection"]["evidence_turn_refs"] = ["R01:S02:T001"]
        validation = validate_round_validator_contract_repair_output(
            repair, repair_plan=self.plan())
        self.assertFalse(validation["valid"])
        self.assertTrue(any("evidence_not_source_action_subset" in error
                            for error in validation["errors"]))

    def test_prompts_make_connection_scope_per_action_slot_explicit(self):
        prompt = build_round_validator_prompt(self.validator_input)
        self.assertIn("only_slots_eligible_for_durable_update_candidates", prompt)
        repair_prompt = build_round_validator_contract_repair_prompt(
            repair_plan=self.plan(),
            frozen_input_sha256=self.validator_input[
                "frozen_round_validator_input_sha256"])
        self.assertIn("FROZEN VERIFIED CONNECTION TARGETS", repair_prompt)
        self.assertIn("FROZEN VERIFIED CONNECTION NULLIFICATION TARGETS", repair_prompt)

    def test_canary18_history_is_unchanged(self):
        self.assertEqual(hash_tree(CANARY_18)["tree_sha256"], CANARY_18_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
