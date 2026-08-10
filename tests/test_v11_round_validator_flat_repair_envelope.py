import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
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
    pack_round_validator_contract_repair_output,
    validate_round_validator_contract_repair_output,
)


CANARY_9 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_9"
)
CANARY_9_TREE_SHA = (
    "bdca2a7fba1b76785c0bfa19d6bb0bff4db28fc8a44222349f64c84a3a90fcd8"
)


def load(relative):
    return json.loads((CANARY_9 / relative).read_text(encoding="utf-8"))


class RoundValidatorFlatRepairEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.realized = [json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted((CANARY_9 / "scenes").glob(
                            "*/realized_scene.json"))]
        cls.raw1 = migrate_historical_connected_person_validator_fixture_v5(
            load("attempts/round_validator_attempt_01.json")["parsed_response"],
            realized_scenes=cls.realized)
        cls.raw2 = load("attempts/round_validator_attempt_02.json")[
            "parsed_response"]
        cls.review = load("review_world_view.json")
        cls.gates = load("local_scene_gate_results.json")
        cls.validator_input = load("round_validator_input.json")
        cls.errors = load("round_validator_run.json")["attempts"][0][
            "validation"]["errors"]
        cls.plan = build_round_validator_contract_repair_plan(
            model_output=cls.raw1, realized_scenes=cls.realized,
            compilation_errors=cls.errors)

    def flat_fixture(self):
        grouped = {
            "top_level_rehome": None,
            "scene_reference_repairs": [],
            "action_realization_repairs": [],
            "durable_update_evidence_repairs": [],
            "observation_reference_repairs": [],
        }
        for target in self.plan["targets"]:
            evidence = (target["allowed_evidence_turn_refs"][:1]
                        if target["evidence_turn_refs_requires_repair"]
                        else target["original_evidence_turn_refs"])
            grouped["action_realization_repairs"].append({
                "action_ref": target["action_ref"],
                "evidence_turn_refs": evidence,
                "progression_evidence_turn_refs": evidence[:1],
            })
        for target in self.plan["durable_update_targets"]:
            grouped["durable_update_evidence_repairs"].append({
                "candidate_update_key": target["candidate_update_key"],
                "evidence_turn_refs": target[
                    "allowed_evidence_turn_refs"][:1],
            })
        return pack_round_validator_contract_repair_output(grouped)

    def test_live_prompt_uses_one_flat_tagged_array(self):
        prompt = build_round_validator_contract_repair_prompt(
            repair_plan=self.plan,
            frozen_input_sha256=self.validator_input[
                "frozen_round_validator_input_sha256"])
        template = json.loads(prompt.split("OUTPUT ONLY THIS OBJECT\n", 1)[1])
        self.assertEqual(set(template), {"repairs"})
        self.assertIsInstance(template["repairs"], list)
        self.assertTrue(all("repair_kind" in row for row in template["repairs"]))

    def test_historical_partial_grouped_response_remains_rejected(self):
        validation = validate_round_validator_contract_repair_output(
            self.raw2, repair_plan=self.plan)
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["errors"], [
            "round_validator_contract_repair_top_level_schema"])

    def test_flat_fixture_passes_complete_gate(self):
        applied = apply_round_validator_contract_repair(
            base_model_output=self.raw1, repair_output=self.flat_fixture(),
            repair_plan=self.plan)
        self.assertTrue(applied["valid"], applied["errors"])
        compilation = compile_round_validator_model_output(
            applied["reconstructed_output"], realized_scenes=self.realized)
        self.assertTrue(compilation["valid"], compilation["errors"])
        validation = validate_round_validator_output(
            compilation["compiled_output"], realized_scenes=self.realized,
            review_world_view=self.review, local_gate_results=self.gates)
        self.assertTrue(validation["valid"], validation["errors"])

    def test_runner_accepts_flat_second_response(self):
        calls = []

        def backend(prompt, attempt):
            calls.append(prompt)
            return self.raw1 if attempt == 1 else self.flat_fixture()

        result = run_round_validator_with_retry(
            backend_call=backend, validator_input=self.validator_input,
            realized_scenes=self.realized, review_world_view=self.review,
            local_gate_results=self.gates)
        self.assertTrue(result["completed"])
        self.assertTrue(result["contract_only_repair_used"])
        self.assertEqual(result["physical_calls"], 2)

    def test_canary9_history_is_unchanged(self):
        self.assertEqual(hash_tree(CANARY_9)["tree_sha256"], CANARY_9_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
