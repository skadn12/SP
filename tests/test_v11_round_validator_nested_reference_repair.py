import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    compile_round_validator_model_output,
    run_round_validator_with_retry,
    validate_round_validator_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract_repair_v11 import (
    apply_round_validator_contract_repair,
    build_round_validator_contract_repair_plan,
    validate_round_validator_contract_repair_output,
)


CANARY_8 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_8"
)
CANARY_8_TREE_SHA = (
    "dc20a5d46b4975c9dcec035d4a7f1838fff44def23b9298c500fcee2ae37ed5f"
)


def load(relative):
    return json.loads((CANARY_8 / relative).read_text(encoding="utf-8"))


class RoundValidatorNestedReferenceRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.realized = [json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted((CANARY_8 / "scenes").glob(
                            "*/realized_scene.json"))]
        cls.raw = load("attempts/round_validator_attempt_01.json")[
            "parsed_response"]
        cls.review = load("review_world_view.json")
        cls.gates = load("local_scene_gate_results.json")
        cls.validator_input = load("round_validator_input.json")
        cls.errors = load("round_validator_run.json")["attempts"][0][
            "validation"]["errors"]
        cls.plan = build_round_validator_contract_repair_plan(
            model_output=cls.raw, realized_scenes=cls.realized,
            compilation_errors=cls.errors)

    def repair_fixture(self):
        top = self.plan["top_level_rehome"]
        result = {
            "top_level_rehome": {
                "misplaced_scene_review_index": top[
                    "misplaced_scene_review_index"],
                "durable_updates": top["durable_updates"],
                "semantic_review_observations": top[
                    "semantic_review_observations"],
            },
            "scene_reference_repairs": [],
            "action_realization_repairs": [],
            "durable_update_evidence_repairs": [{
                "candidate_update_key": target["candidate_update_key"],
                "evidence_turn_refs": target[
                    "allowed_evidence_turn_refs"][:1],
            } for target in self.plan["durable_update_targets"]],
            "observation_reference_repairs": [],
        }
        for target in self.plan["observation_targets"]:
            source = target["allowed_scene_refs"][0]
            result["observation_reference_repairs"].append({
                "observation_key": target["observation_key"],
                "source_scene_refs": [source],
                "evidence_turn_refs": target[
                    "allowed_evidence_turn_refs_by_scene"][source][:1],
            })
        return result

    def test_rehome_plan_also_detects_nested_reference_targets(self):
        self.assertTrue(self.plan["valid"], self.plan["errors"])
        self.assertIsNotNone(self.plan["top_level_rehome"])
        self.assertEqual(self.plan["target_candidate_update_keys"], ["DU001"])
        self.assertEqual(self.plan["target_observation_keys"], ["OBS001"])

    def test_durable_update_must_use_its_source_action_evidence(self):
        fixture = self.repair_fixture()
        fixture["durable_update_evidence_repairs"][0]["evidence_turn_refs"] = [
            "R01:S02:T001"]
        validation = validate_round_validator_contract_repair_output(
            fixture, repair_plan=self.plan)
        self.assertFalse(validation["valid"])
        self.assertTrue(any("evidence_not_source_action_subset" in row
                            for row in validation["errors"]))

    def test_observation_evidence_must_belong_to_selected_scene(self):
        fixture = self.repair_fixture()
        fixture["observation_reference_repairs"][0].update({
            "source_scene_refs": ["R01:S01"],
            "evidence_turn_refs": ["R01:S02:T001"],
        })
        validation = validate_round_validator_contract_repair_output(
            fixture, repair_plan=self.plan)
        self.assertFalse(validation["valid"])
        self.assertTrue(any("evidence_not_selected_scene_subset" in row
                            for row in validation["errors"]))

    def test_explicit_nested_repairs_pass_complete_gate(self):
        applied = apply_round_validator_contract_repair(
            base_model_output=self.raw, repair_output=self.repair_fixture(),
            repair_plan=self.plan)
        self.assertTrue(applied["valid"], applied["errors"])
        self.assertTrue(applied["non_target_fields_unchanged"])
        compilation = compile_round_validator_model_output(
            applied["reconstructed_output"], realized_scenes=self.realized)
        self.assertTrue(compilation["valid"], compilation["errors"])
        validation = validate_round_validator_output(
            compilation["compiled_output"], realized_scenes=self.realized,
            review_world_view=self.review, local_gate_results=self.gates)
        self.assertTrue(validation["valid"], validation["errors"])

    def test_retry_runner_uses_one_compact_contract_request(self):
        calls = []

        def backend(prompt, attempt):
            calls.append(prompt)
            return self.raw if attempt == 1 else self.repair_fixture()

        result = run_round_validator_with_retry(
            backend_call=backend, validator_input=self.validator_input,
            realized_scenes=self.realized, review_world_view=self.review,
            local_gate_results=self.gates)
        self.assertTrue(result["completed"])
        self.assertTrue(result["contract_only_repair_used"])
        self.assertEqual(result["physical_calls"], 2)
        self.assertLess(len(calls[1]), len(calls[0]) // 2)

    def test_canary8_history_is_unchanged(self):
        self.assertEqual(hash_tree(CANARY_8)["tree_sha256"], CANARY_8_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
