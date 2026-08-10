import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
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


CANARY = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_23")
CANARY_SHA = "51d52203a89591e0667cf91d0ef095b28d5674e5e2a06f9970cf86e72911bde6"


def load(relative):
    return json.loads((CANARY / relative).read_text(encoding="utf-8"))


class Canary23ValidatorCascadeRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load("attempts/round_validator_attempt_01.json")[
            "parsed_response"]
        cls.run_audit = load("round_validator_run.json")
        cls.validator_input = load("round_validator_input.json")
        cls.review = load("review_world_view.json")
        cls.gates = load("local_scene_gate_results.json")
        cls.realized = [json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted((CANARY / "scenes").glob(
                            "*/realized_scene.json"))]

    def plan(self):
        historical = {
            row["source_anchor_id"]: row["source_refs"]
            for row in self.validator_input["historical_anchor_index"]}
        return build_round_validator_contract_repair_plan(
            model_output=self.base, realized_scenes=self.realized,
            compilation_errors=self.run_audit["attempts"][0]["validation"]["errors"],
            historical_anchor_refs=historical)

    def repair_fixture(self):
        plan = self.plan()
        rows = []
        for target in plan["scene_reference_targets"]:
            rows.append({
                "repair_kind": "scene_reference",
                "scene_alias": target["scene_alias"],
                "historical_supporting_turn_refs": (
                    target["allowed_historical_turn_refs"][:1]
                    if target["repair_historical_supporting_turn_refs"] else
                    target["original_historical_supporting_turn_refs"]),
                "current_scene_supporting_turn_refs": (
                    target["allowed_current_scene_turn_refs"][:1]
                    if target["repair_current_scene_supporting_turn_refs"] else
                    target["original_current_scene_supporting_turn_refs"]),
                "dependency_context_turn_refs": (
                    target["allowed_dependency_turn_refs"][:1]
                    if target["repair_dependency_context_turn_refs"] and
                    target["allowed_dependency_turn_refs"] else
                    target["original_dependency_context_turn_refs"]),
                "uses_scene_refs": target["original_uses_scene_refs"],
            })
        for target in plan["targets"]:
            rows.append({
                "repair_kind": "action_realization",
                "action_ref": target["action_ref"],
                "evidence_turn_refs": target["original_evidence_turn_refs"],
                "progression_evidence_turn_refs":
                    target["candidate_evidence_turn_refs"][:1],
            })
        for target in plan["durable_removal_targets"]:
            rows.append({
                "repair_kind": "durable_update_nullification",
                "candidate_update_key": target["candidate_update_key"],
                "replacement": None,
            })
        return {"repairs": rows}

    def test_plan_covers_scoped_refs_and_malformed_durable_shape(self):
        plan = self.plan()
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual(
            [row["scene_alias"] for row in plan["scene_reference_targets"]],
            ["s2", "s3", "s4", "s5", "s6", "s7"])
        self.assertTrue(all(row["repair_historical_supporting_turn_refs"]
                            for row in plan["scene_reference_targets"]))
        self.assertEqual([(row["candidate_update_key"], row["reason"])
                          for row in plan["durable_removal_targets"]],
                         [("DU001", "verification_status_invalid_or_missing")])

    def test_explicit_shadow_repair_passes_full_strict_gate(self):
        plan = self.plan()
        fixture = self.repair_fixture()
        repair_validation = validate_round_validator_contract_repair_output(
            fixture, repair_plan=plan)
        self.assertTrue(repair_validation["valid"], repair_validation["errors"])
        applied = apply_round_validator_contract_repair(
            base_model_output=self.base, repair_output=fixture,
            repair_plan=plan)
        self.assertTrue(applied["valid"], applied["errors"])
        compiled = compile_round_validator_model_output(
            applied["reconstructed_output"], realized_scenes=self.realized)
        self.assertTrue(compiled["valid"], compiled["errors"])
        strict = validate_round_validator_output(
            compiled["compiled_output"], realized_scenes=self.realized,
            review_world_view=self.review, local_gate_results=self.gates)
        self.assertTrue(strict["valid"], strict["errors"])

    def test_prompt_does_not_present_shape_label_as_output_key(self):
        prompt = build_round_validator_prompt(self.validator_input)
        self.assertIn("durable_update_array_item_shapes", prompt)
        self.assertIn("Durable items are flat", prompt)
        self.assertNotIn('"connected_person_durable_update":', prompt)
        repair_prompt = build_round_validator_contract_repair_prompt(
            repair_plan=self.plan(), frozen_input_sha256=self.validator_input[
                "frozen_round_validator_input_sha256"])
        self.assertIn("allowed_historical_turn_options", repair_prompt)
        self.assertIn("historical_supporting_turn_indices", repair_prompt)
        self.assertIn("verification_status_invalid_or_missing", repair_prompt)

    def test_history_unchanged(self):
        self.assertEqual(hash_tree(CANARY)["tree_sha256"], CANARY_SHA)


if __name__ == "__main__":
    unittest.main()
