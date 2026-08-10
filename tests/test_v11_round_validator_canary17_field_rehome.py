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
    pack_round_validator_contract_repair_output,
)


CANARY_17 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_17"
)
CANARY_17_TREE_SHA = (
    "a6285ea9c3244162a3a1ce8724fbe35b23e59d3c2d1b6abe6fe37179a37bc02f"
)


def load(relative):
    return json.loads((CANARY_17 / relative).read_text(encoding="utf-8"))


class Canary17RoundValidatorFieldRehomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.realized = [json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted((CANARY_17 / "scenes").glob(
                            "*/realized_scene.json"))]
        cls.review = load("review_world_view.json")
        cls.gates = load("local_scene_gate_results.json")
        cls.validator_input = load("round_validator_input.json")
        cls.run_audit = load("round_validator_run.json")
        cls.raw1 = load("attempts/round_validator_attempt_01.json")[
            "parsed_response"]
        cls.raw2 = load("attempts/round_validator_attempt_02.json")[
            "parsed_response"]

    def plan(self, raw, attempt_index):
        return build_round_validator_contract_repair_plan(
            model_output=raw, realized_scenes=self.realized,
            compilation_errors=self.run_audit["attempts"][attempt_index][
                "validation"]["errors"])

    @staticmethod
    def repair_fixture(plan):
        top = plan.get("top_level_rehome")
        top_output = None
        if top is not None:
            index_fields = (
                {"misplaced_scene_review_indices": deepcopy(
                    top["misplaced_scene_review_indices"])}
                if "misplaced_scene_review_indices" in top else
                {"misplaced_scene_review_index":
                    top["misplaced_scene_review_index"]})
            top_output = {
                **index_fields,
                "durable_updates": deepcopy(top["durable_updates"]),
                "semantic_review_observations": deepcopy(
                    top["semantic_review_observations"]),
            }
        grouped = {
            "top_level_rehome": top_output,
            "scene_reference_repairs": [],
            "action_realization_repairs": [],
            "durable_update_evidence_repairs": [],
            "observation_reference_repairs": [],
            "unplanned_action_repairs": [],
            "future_knowledge_repairs": [],
            "durable_update_removals": [],
            "durable_dependency_repairs": [],
        }
        for target in plan["scene_reference_targets"]:
            grouped["scene_reference_repairs"].append({
                "scene_alias": target["scene_alias"],
                "current_scene_supporting_turn_refs": (
                    target["allowed_current_scene_turn_refs"]
                    if target["repair_current_scene_supporting_turn_refs"] else
                    target["original_current_scene_supporting_turn_refs"]),
                "dependency_context_turn_refs": (
                    target["allowed_dependency_turn_refs"][:1]
                    if target["repair_dependency_context_turn_refs"] and
                    target["allowed_dependency_turn_refs"] else
                    target["original_dependency_context_turn_refs"]),
            })
        for target in plan["targets"]:
            action_evidence = (
                target["allowed_evidence_turn_refs"][:1]
                if target["evidence_turn_refs_requires_repair"] else
                target["original_evidence_turn_refs"])
            progression_evidence = (
                action_evidence[:1]
                if target["progression_evidence_turn_refs_requires_repair"] else
                target["original_progression_evidence_turn_refs"])
            grouped["action_realization_repairs"].append({
                "action_ref": target["action_ref"],
                "evidence_turn_refs": deepcopy(action_evidence),
                "progression_evidence_turn_refs": deepcopy(
                    progression_evidence),
            })
        for target in plan["observation_targets"]:
            source = target["allowed_scene_refs"][1]
            grouped["observation_reference_repairs"].append({
                "observation_key": target["observation_key"],
                "source_scene_refs": [source],
                "evidence_turn_refs": target[
                    "allowed_evidence_turn_refs_by_scene"][source][:1],
            })
        for target in plan["future_knowledge_targets"]:
            grouped["future_knowledge_repairs"].append({
                "scene_alias": target["scene_alias"], "value": False,
            })
        return pack_round_validator_contract_repair_output(grouped)

    def test_attempt_one_split_top_level_rows_are_explicit_rehome_targets(self):
        plan = self.plan(self.raw1, 0)
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual(
            plan["top_level_rehome"]["misplaced_scene_review_indices"], [6, 7])
        # Action-field placement is now handled before schema validation by the
        # deterministic compiler; the model repair only rehomes top-level arrays.
        self.assertEqual(plan["target_action_refs"], [])

    def test_attempt_two_scene_progression_is_mapped_to_unique_continuations(self):
        plan = self.plan(self.raw2, 1)
        self.assertTrue(plan["valid"], plan["errors"])
        targets = {row["action_ref"]: row for row in plan["targets"]}
        self.assertEqual(targets["S02.secondary_action"][
            "original_progression_evidence_turn_refs"], ["R01:S02:T005"])
        self.assertEqual(targets["S05.primary_action"][
            "original_progression_evidence_turn_refs"], ["R01:S05:T005"])
        self.assertTrue(all(row["progression_rehome_from_scene"]
                            for row in targets.values()))

    def test_attempt_one_shadow_repair_passes_original_strict_validator(self):
        plan = self.plan(self.raw1, 0)
        applied = apply_round_validator_contract_repair(
            base_model_output=self.raw1,
            repair_output=self.repair_fixture(plan), repair_plan=plan)
        self.assertTrue(applied["valid"], applied["errors"])
        reconstructed = applied["reconstructed_output"]
        self.assertEqual(set(reconstructed), {
            "round_decision", "round_reason_codes", "scene_reviews",
            "durable_updates", "semantic_review_observations"})
        compiled = compile_round_validator_model_output(
            reconstructed, realized_scenes=self.realized)
        self.assertTrue(compiled["valid"], compiled["errors"])
        compiled_s2 = compiled["compiled_output"]["scene_reviews"][1]
        self.assertNotIn("progression_evidence_turn_refs", compiled_s2)
        self.assertIn("progression_evidence_turn_refs",
                      compiled_s2["secondary_action_realization"])
        validated = validate_round_validator_output(
            compiled["compiled_output"], realized_scenes=self.realized,
            review_world_view=self.review, local_gate_results=self.gates)
        self.assertTrue(validated["valid"], validated["errors"])

    def test_rehome_requires_exact_original_progression_evidence(self):
        plan = self.plan(self.raw2, 1)
        repair = self.repair_fixture(plan)
        action_row = next(row for row in repair["repairs"]
                          if row["repair_kind"] == "action_realization")
        action_row["progression_evidence_turn_refs"] = ["R01:S02:T004"]
        applied = apply_round_validator_contract_repair(
            base_model_output=self.raw2, repair_output=repair,
            repair_plan=plan)
        self.assertFalse(applied["valid"])
        self.assertTrue(any("valid_progression_evidence_changed" in error
                            for error in applied["errors"]))

    def test_prompt_states_exact_field_placement(self):
        prompt = build_round_validator_prompt(self.validator_input)
        self.assertIn("three arrays as top-level sibling fields", prompt)
        self.assertIn("never directly on a scene review", prompt)
        repair_prompt = build_round_validator_contract_repair_prompt(
            repair_plan=self.plan(self.raw1, 0),
            frozen_input_sha256=self.validator_input[
                "frozen_round_validator_input_sha256"])
        self.assertIn("index or indices", repair_prompt)

    def test_canary17_history_is_unchanged(self):
        self.assertEqual(hash_tree(CANARY_17)["tree_sha256"],
                         CANARY_17_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
