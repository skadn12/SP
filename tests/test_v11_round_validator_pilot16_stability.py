from __future__ import annotations

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
    pack_round_validator_contract_repair_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.scene_decision_closure import (
    compute_scene_decision_closure,
)


PILOT16 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_16/roots")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class RoundValidatorPilot16StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.historical_run = PILOT16 / "cc-train-episode-216742/01_expansion"
        cls.tree_before = hash_tree(cls.historical_run)["tree_sha256"]
        cls.raw1 = load(cls.historical_run / "attempts/round_validator_attempt_01.json")[
            "parsed_response"]
        cls.raw2 = load(cls.historical_run / "attempts/round_validator_attempt_02.json")[
            "parsed_response"]
        cls.realized = [load(path) for path in sorted(
            (cls.historical_run / "scenes").glob("*/realized_scene.json"))]
        cls.review = load(cls.historical_run / "review_world_view.json")
        cls.gates = load(cls.historical_run / "local_scene_gate_results.json")
        cls.dag = load(cls.historical_run / "scene_dependency_dag.json")
        cls.validator_input = load(cls.historical_run / "round_validator_input.json")

    def compile_and_validate(self, raw):
        compiled = compile_round_validator_model_output(
            raw, realized_scenes=self.realized)
        self.assertTrue(compiled["valid"], compiled["errors"])
        validated = validate_round_validator_output(
            compiled["compiled_output"], realized_scenes=self.realized,
            review_world_view=self.review, local_gate_results=self.gates)
        return compiled, validated

    def test_five_misplaced_progression_lists_are_rehomed_without_semantic_edit(self):
        compiled, validated = self.compile_and_validate(self.raw1)
        moves = compiled["compatibility_normalization"][
            "unambiguous_action_field_rehomes"]
        self.assertEqual(["s2", "s3", "s5", "s6", "s7"],
                         [row["scene_alias"] for row in moves])
        self.assertFalse(any("model_schema" in row
                             for row in validated["errors"]))
        for move in moves:
            original = next(row for row in self.raw1["scene_reviews"]
                            if row["scene_alias"] == move["scene_alias"])
            normalized = next(row for row in compiled["compiled_output"][
                "scene_reviews"] if row["scene_alias"] == move["scene_alias"])
            self.assertEqual(
                original["progression_evidence_turn_refs"],
                normalized["primary_action_realization"][
                    "progression_evidence_turn_refs"])

    def test_ambiguous_rehome_is_not_guessed(self):
        raw = deepcopy(self.raw1)
        scenes = deepcopy(self.realized)
        scene = next(row for row in scenes if row["scene_alias"] == "s2")
        scene["secondary_action"]["action_mode"] = "continuation"
        scene["secondary_action"]["continuation_of"] = "S01.secondary_action"
        result = compile_round_validator_model_output(raw, realized_scenes=scenes)
        self.assertFalse(result["valid"])
        self.assertIn("scene_review:1:model_schema", result["errors"])

    def test_uniform_prompt_gives_new_actions_an_empty_progression_array(self):
        prompt = build_round_validator_prompt(self.validator_input)
        self.assertIn('"progression_evidence_turn_refs":[]', prompt)
        self.assertIn("Every action review has exactly status", prompt)

    def test_reference_only_repair_closes_contract_and_preserves_blockers(self):
        compiled, validated = self.compile_and_validate(self.raw1)
        self.assertEqual([
            "scene_review:2:current_scene_supporting_turn_refs",
            "scene_review:3:dependency_context_turn_refs",
            "scene_review:3:dependency_context_source_not_declared",
        ], validated["errors"])
        anchors = {
            str(row.get("source_anchor_id") or ""): list(row.get("source_refs") or [])
            for row in self.validator_input.get("historical_anchor_index") or []
        }
        plan = build_round_validator_contract_repair_plan(
            model_output=self.raw1, realized_scenes=self.realized,
            compilation_errors=validated["errors"],
            historical_anchor_refs=anchors,
            eligible_durable_action_refs=[])
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual(["s3", "s4"], plan["target_scene_aliases"])
        self.assertEqual([], plan["target_action_refs"])
        historical_scene_repairs = [
            {key: deepcopy(value) for key, value in row.items()
             if key != "repair_kind"}
            for row in self.raw2["repairs"]
            if row.get("repair_kind") == "scene_reference"
        ]
        repair = pack_round_validator_contract_repair_output({
            "top_level_rehome": None,
            "scene_reference_repairs": historical_scene_repairs,
            "action_realization_repairs": [],
            "durable_update_evidence_repairs": [],
            "observation_reference_repairs": [],
        })
        applied = apply_round_validator_contract_repair(
            base_model_output=self.raw1, repair_output=repair,
            repair_plan=plan)
        self.assertTrue(applied["valid"], applied["errors"])
        repaired, final_validation = self.compile_and_validate(
            applied["reconstructed_output"])
        self.assertTrue(final_validation["valid"], final_validation["errors"])
        self.assertEqual(3, len(final_validation[
            "continuation_consistency_blockers"]))
        closure = compute_scene_decision_closure(
            realized_scenes=self.realized, local_gate_results=self.gates,
            validator_output=repaired["compiled_output"],
            scene_dependency_dag=self.dag)
        s3 = next(row for row in closure["scenes"]
                  if row["scene_ref"] == "R01:S03")
        self.assertEqual("continuation_source_unavailable", s3["closure_reason"])
        self.assertTrue(s3["continuation_action_blockers"])

    def test_historical_run_remains_read_only(self):
        self.assertEqual(self.tree_before,
                         hash_tree(self.historical_run)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
