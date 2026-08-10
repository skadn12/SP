import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.mock_hybrid_backends import (
    make_validator_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    build_round_validator_input,
    build_round_validator_prompt,
    build_round_validator_retry_prompt,
    compile_round_validator_model_output,
    run_round_validator_with_retry,
    validate_round_validator_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract import (
    ROUND_VALIDATOR_SCHEMA_VERSION,
)


CANARY_3 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_3"
)
CANARY_3_SHA = "974035b8ddb29f8fdc5aad9c8edeffb7109a4d81582d40cec702162eb593cf38"


def load(relative):
    return json.loads((CANARY_3 / relative).read_text(encoding="utf-8"))


class RoundValidatorContractV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled_plan = load("planner/compiled_round_plan.json")
        cls.realized = [json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted((CANARY_3 / "scenes").glob(
                            "*/realized_scene.json"))]
        cls.review = load("review_world_view.json")
        cls.gates = load("local_scene_gate_results.json")
        cls.dag = load("scene_dependency_dag.json")
        cls.validator_input = build_round_validator_input(
            round_blueprint=cls.compiled_plan,
            review_world_view=cls.review,
            realized_scenes=cls.realized,
            scene_dependency_dag=cls.dag,
            structural_gate_results=cls.gates,
        )

    def valid_model_output(self):
        return make_validator_output(
            "accept_no_updates", self.realized, self.compiled_plan)

    def compile_and_validate(self, value):
        compilation = compile_round_validator_model_output(
            value, realized_scenes=self.realized)
        self.assertTrue(compilation["valid"], compilation["errors"])
        validation = validate_round_validator_output(
            compilation["compiled_output"], realized_scenes=self.realized,
            review_world_view=self.review, local_gate_results=self.gates)
        return compilation["compiled_output"], validation

    def test_program_injects_action_identity(self):
        model = self.valid_model_output()
        raw_action = model["scene_reviews"][0]["primary_action_realization"]
        self.assertEqual(set(raw_action), {"status", "evidence_turn_refs", "reason"})
        compiled, validation = self.compile_and_validate(model)
        action = compiled["scene_reviews"][0]["primary_action_realization"]
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertEqual(action["action_ref"], "S01.primary_action")
        self.assertEqual(action["action_mode"], "new")
        self.assertEqual(action["progression_evidence_turn_refs"], [])

    def test_model_cannot_repeat_program_owned_action_fields(self):
        model = self.valid_model_output()
        model["scene_reviews"][0]["primary_action_realization"][
            "action_ref"] = "S01.primary_action"
        result = compile_round_validator_model_output(
            model, realized_scenes=self.realized)
        self.assertFalse(result["valid"])
        self.assertIn("new_model_schema", result["errors"][0])

    def test_new_action_accepts_uniform_empty_progression_field(self):
        model = self.valid_model_output()
        model["scene_reviews"][0]["primary_action_realization"][
            "progression_evidence_turn_refs"] = []
        result = compile_round_validator_model_output(
            model, realized_scenes=self.realized)
        self.assertTrue(result["valid"], result["errors"])

    def test_legacy_new_action_progression_field_is_ignored(self):
        model = self.valid_model_output()
        model["scene_reviews"][0]["primary_action_realization"][
            "progression_evidence_turn_refs"] = ["R01:S01:T001"]
        result = compile_round_validator_model_output(
            model, realized_scenes=self.realized)
        self.assertTrue(result["valid"], result["errors"])
        action = result["compiled_output"]["scene_reviews"][0][
            "primary_action_realization"]
        self.assertEqual([], action["progression_evidence_turn_refs"])
        self.assertIn(
            "scene_review:0:primary_action_realization",
            result["compatibility_normalization"][
                "ignored_new_action_progression_fields"],
        )

    def test_continuation_requires_progression_field(self):
        model = self.valid_model_output()
        continuation_review = next(
            row for row in model["scene_reviews"]
            if row["scene_alias"] == "s6")["primary_action_realization"]
        self.assertIn("progression_evidence_turn_refs", continuation_review)
        continuation_review.pop("progression_evidence_turn_refs")
        result = compile_round_validator_model_output(
            model, realized_scenes=self.realized)
        self.assertFalse(result["valid"])
        self.assertTrue(any("continuation_model_schema" in error
                            for error in result["errors"]))

    def test_planned_secondary_requires_review_object(self):
        model = self.valid_model_output()
        model["scene_reviews"][0]["secondary_action_realization"] = None
        result = compile_round_validator_model_output(
            model, realized_scenes=self.realized)
        self.assertFalse(result["valid"])
        self.assertTrue(any("planned_action_review_required" in error
                            for error in result["errors"]))

    def test_dependency_context_refs_are_separate_and_ancestry_bound(self):
        model = self.valid_model_output()
        s2 = next(row for row in model["scene_reviews"]
                  if row["scene_alias"] == "s2")
        s2["dependency_context_turn_refs"] = ["R01:S01:T007"]
        s2["uses_scene_refs"] = ["R01:S01"]
        _, valid = self.compile_and_validate(model)
        self.assertTrue(valid["valid"], valid["errors"])
        s2["dependency_context_turn_refs"] = ["R01:S06:T001"]
        compilation = compile_round_validator_model_output(
            model, realized_scenes=self.realized)
        invalid = validate_round_validator_output(
            compilation["compiled_output"], realized_scenes=self.realized,
            review_world_view=self.review, local_gate_results=self.gates)
        self.assertFalse(invalid["valid"])
        self.assertIn("scene_review:1:dependency_context_turn_refs",
                      invalid["errors"])

    def test_dependency_context_source_must_be_declared(self):
        model = self.valid_model_output()
        s2 = next(row for row in model["scene_reviews"]
                  if row["scene_alias"] == "s2")
        s2["dependency_context_turn_refs"] = ["R01:S01:T007"]
        s2["uses_scene_refs"] = []
        compiled = compile_round_validator_model_output(
            model, realized_scenes=self.realized)["compiled_output"]
        invalid = validate_round_validator_output(
            compiled, realized_scenes=self.realized,
            review_world_view=self.review, local_gate_results=self.gates)
        self.assertIn("scene_review:1:dependency_context_source_not_declared",
                      invalid["errors"])

    def test_prompt_ends_with_data_not_schema_instruction(self):
        prompt = build_round_validator_prompt(self.validator_input)
        self.assertNotIn("OUTPUT ONLY THE SCHEMA OBJECT", prompt)
        self.assertTrue(prompt.endswith(
            "NEVER OUTPUT type, properties, required, or additionalProperties."))
        self.assertIn("OUTPUT DATA TEMPLATE", prompt)
        self.assertIn("current_scene_supporting_turn_refs", prompt)
        self.assertIn("dependency_context_turn_refs", prompt)

    def test_retry_keeps_frozen_input_and_data_instruction(self):
        base = build_round_validator_prompt(self.validator_input)
        prompt = build_round_validator_retry_prompt(
            frozen_input_sha256=self.validator_input[
                "frozen_round_validator_input_sha256"],
            contract_errors=["validator_returned_schema_definition"],
            frozen_base_prompt=base,
        )
        self.assertIn(self.validator_input[
            "frozen_round_validator_input_sha256"], prompt)
        self.assertNotIn("OUTPUT ONLY THE SCHEMA OBJECT", prompt)
        self.assertTrue(prompt.endswith(
            "NEVER OUTPUT type, properties, required, or additionalProperties."))

    def test_canary3_schema_definition_retry_is_explicitly_rejected(self):
        raw = load("raw/backend_responses/round_validator_attempt_02.txt")
        result = compile_round_validator_model_output(
            raw, realized_scenes=self.realized)
        self.assertFalse(result["valid"])
        self.assertEqual(result["errors"], ["validator_returned_schema_definition"])

    def test_canary3_first_response_failure_is_contract_not_semantic_reject(self):
        raw = load("raw/backend_responses/round_validator_attempt_01.txt")
        self.assertEqual(raw["round_decision"], "accept")
        self.assertIn("validator_schema_version", raw)
        self.assertTrue(any(
            row["primary_action_realization"].get(
                "progression_evidence_turn_refs")
            for row in raw["scene_reviews"]
            if row["primary_action_realization"].get("action_mode") == "new"))
        result = compile_round_validator_model_output(
            raw, realized_scenes=self.realized)
        self.assertFalse(result["valid"])
        self.assertEqual(result["classification"], "schema_invalid")

    def test_contract_retry_can_accept_valid_semantic_payload(self):
        valid = self.valid_model_output()
        invalid = {"type": "object", "properties": {}, "required": []}
        calls = []

        def backend(prompt, attempt):
            calls.append(prompt)
            return invalid if attempt == 1 else valid

        result = run_round_validator_with_retry(
            backend_call=backend, validator_input=self.validator_input,
            realized_scenes=self.realized, review_world_view=self.review,
            local_gate_results=self.gates)
        self.assertTrue(result["completed"])
        self.assertEqual(result["physical_calls"], 2)
        self.assertEqual(result["valid_output"]["validator_schema_version"],
                         ROUND_VALIDATOR_SCHEMA_VERSION)
        self.assertEqual(len({row["frozen_input_sha256"]
                              for row in result["attempts"]}), 1)

    def test_historical_canary3_is_unchanged(self):
        self.assertEqual(hash_tree(CANARY_3)["tree_sha256"], CANARY_3_SHA)


if __name__ == "__main__":
    unittest.main()
