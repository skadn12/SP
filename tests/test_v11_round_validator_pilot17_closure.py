from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    compile_round_validator_model_output,
    materialize_round_validator_optional_model_fields,
    validate_round_validator_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract_repair_v11 import (
    apply_round_validator_contract_repair,
    build_round_validator_contract_repair_plan,
    build_round_validator_contract_repair_prompt,
    pack_round_validator_contract_repair_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.scene_decision_closure import (
    compute_scene_decision_closure,
)


PILOT17 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_17/roots/"
    "cc-train-episode-41625/01_expansion")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class RoundValidatorPilot17ClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree_before = hash_tree(PILOT17)["tree_sha256"]
        cls.raw = load(PILOT17 / "attempts/round_validator_attempt_01.json")[
            "parsed_response"]
        cls.realized = [load(path) for path in sorted(
            (PILOT17 / "scenes").glob("*/realized_scene.json"))]
        cls.review = load(PILOT17 / "review_world_view.json")
        cls.gates = load(PILOT17 / "local_scene_gate_results.json")
        cls.dag = load(PILOT17 / "scene_dependency_dag.json")
        cls.validator_input = load(PILOT17 / "round_validator_input.json")

    def validation_and_plan(self):
        compiled = compile_round_validator_model_output(
            self.raw, realized_scenes=self.realized)
        self.assertTrue(compiled["valid"], compiled["errors"])
        validation = validate_round_validator_output(
            compiled["compiled_output"], realized_scenes=self.realized,
            review_world_view=self.review, local_gate_results=self.gates)
        base = materialize_round_validator_optional_model_fields(self.raw)
        anchors = {
            str(row.get("source_anchor_id") or ""): list(row.get("source_refs") or [])
            for row in self.validator_input.get("historical_anchor_index") or []
        }
        plan = build_round_validator_contract_repair_plan(
            model_output=base, realized_scenes=self.realized,
            compilation_errors=validation["errors"],
            historical_anchor_refs=anchors, eligible_durable_action_refs=[])
        return compiled, validation, base, plan

    def test_partial_primary_is_a_closure_blocker_not_contract_error(self):
        compiled, validation, _, _ = self.validation_and_plan()
        self.assertNotIn(
            "scene_review:3:primary_action_not_fully_realized_but_accepted",
            validation["errors"])
        output = compiled["compiled_output"]
        # Use a reference-valid fixture so closure can evaluate the frozen semantic status.
        for review in output["scene_reviews"]:
            action = review.get("primary_action_realization")
            if (action or {}).get("action_mode") == "continuation":
                action["progression_evidence_turn_refs"] = list(
                    action["evidence_turn_refs"][:1])
        closure = compute_scene_decision_closure(
            realized_scenes=self.realized, local_gate_results=self.gates,
            validator_output=output, scene_dependency_dag=self.dag)
        s4 = next(row for row in closure["scenes"]
                  if row["scene_ref"] == "R01:S04")
        self.assertEqual("primary_action_not_realized", s4["closure_reason"])
        self.assertEqual("not_realized",
                         s4["primary_action_blockers"][0]["status"])

    def test_progression_reference_errors_get_contract_only_repair_plan(self):
        _, validation, _, plan = self.validation_and_plan()
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual(5, len(validation["errors"]))
        self.assertTrue(all(error.endswith(
            "progression_evidence_not_action_evidence")
            for error in validation["errors"]))
        self.assertEqual([
            "S02.primary_action", "S03.primary_action", "S05.primary_action",
            "S06.primary_action", "S07.primary_action",
        ], plan["target_action_refs"])
        self.assertEqual([], plan["scene_reference_targets"])
        prompt = build_round_validator_contract_repair_prompt(
            repair_plan=plan, frozen_input_sha256="frozen")
        for target in plan["targets"]:
            for invalid_ref in target["original_progression_evidence_turn_refs"]:
                if invalid_ref not in target["original_evidence_turn_refs"]:
                    self.assertNotIn(invalid_ref, prompt)

    def test_reference_repair_reaches_valid_contract_then_semantic_closure(self):
        _, _, base, plan = self.validation_and_plan()
        repairs = [{
            "action_ref": target["action_ref"],
            "progression_evidence_turn_refs": [
                target["original_evidence_turn_refs"][0]],
        } for target in plan["targets"]]
        packed = pack_round_validator_contract_repair_output({
            "top_level_rehome": None,
            "scene_reference_repairs": [],
            "action_realization_repairs": repairs,
            "durable_update_evidence_repairs": [],
            "observation_reference_repairs": [],
        })
        applied = apply_round_validator_contract_repair(
            base_model_output=base, repair_output=packed, repair_plan=plan)
        self.assertTrue(applied["valid"], applied["errors"])
        compiled = compile_round_validator_model_output(
            applied["reconstructed_output"], realized_scenes=self.realized)
        self.assertTrue(compiled["valid"], compiled["errors"])
        validation = validate_round_validator_output(
            compiled["compiled_output"], realized_scenes=self.realized,
            review_world_view=self.review, local_gate_results=self.gates)
        self.assertTrue(validation["valid"], validation["errors"])
        closure = compute_scene_decision_closure(
            realized_scenes=self.realized, local_gate_results=self.gates,
            validator_output=compiled["compiled_output"],
            scene_dependency_dag=self.dag)
        self.assertEqual(["R01:S01", "R01:S02", "R01:S03"],
                         closure["accepted_scene_refs"])
        self.assertEqual(["R01:S04", "R01:S05", "R01:S06", "R01:S07"],
                         closure["rejected_scene_refs"])

    def test_historical_run_is_unchanged(self):
        self.assertEqual(self.tree_before, hash_tree(PILOT17)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
