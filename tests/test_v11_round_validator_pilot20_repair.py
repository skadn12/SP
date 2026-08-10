import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    compile_round_validator_model_output,
    normalize_exact_duplicate_scene_reviews,
    validate_round_validator_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract_repair_v11 import (
    apply_round_validator_contract_repair,
    build_round_validator_contract_repair_plan,
    build_round_validator_contract_repair_prompt,
    validate_round_validator_contract_repair_output,
)


PILOT20 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_20"
)
PILOT20_TREE_SHA = (
    "a6c8b6e6d9b8ab25fe157cdd0c213b6d455a3a6aca073f538cac9f56a364e733"
)


class Pilot20RoundValidatorRepairTests(unittest.TestCase):
    def load_case(self, root_id):
        root = PILOT20 / "roots" / root_id / "01_expansion"
        realized = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / "scenes").glob("*/realized_scene.json"))
        ]
        attempt1 = json.loads((
            root / "attempts" / "round_validator_attempt_01.json"
        ).read_text(encoding="utf-8"))["parsed_response"]
        attempt1 = normalize_exact_duplicate_scene_reviews(attempt1)["output"]
        validator_input = json.loads((
            root / "round_validator_input.json").read_text(encoding="utf-8"))
        review_world = json.loads((
            root / "review_world_view.json").read_text(encoding="utf-8"))
        gates = json.loads((
            root / "local_scene_gate_results.json").read_text(encoding="utf-8"))
        compilation = compile_round_validator_model_output(
            attempt1, realized_scenes=realized)
        if compilation["valid"]:
            validation = validate_round_validator_output(
                compilation["compiled_output"], realized_scenes=realized,
                review_world_view=review_world, local_gate_results=gates)
        else:
            validation = compilation
        historical = {
            str(row.get("source_anchor_id") or ""): [
                str(ref) for ref in row.get("source_refs") or []]
            for row in validator_input.get("historical_anchor_index") or []
        }
        plan = build_round_validator_contract_repair_plan(
            model_output=attempt1, realized_scenes=realized,
            compilation_errors=validation["errors"],
            historical_anchor_refs=historical,
            eligible_durable_action_refs=[])
        return root, realized, attempt1, review_world, gates, plan

    def test_progression_only_errors_do_not_create_scene_reference_targets(self):
        _, _, _, _, _, plan = self.load_case("cc-train-episode-41625")
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual(plan["scene_reference_targets"], [])
        self.assertEqual(plan["target_action_refs"], [
            "S02.primary_action", "S03.primary_action", "S04.primary_action",
            "S05.primary_action", "S06.primary_action", "S07.primary_action",
        ])

    def test_pilot20_progression_repair_reconstructs_valid_review(self):
        _, realized, base, review_world, gates, plan = self.load_case(
            "cc-train-episode-41625")
        repair = {"repairs": [{
            "repair_kind": "action_realization",
            "action_ref": row["action_ref"],
            "progression_evidence_turn_refs": [
                row["original_evidence_turn_refs"][0]],
        } for row in plan["targets"]]}
        checked = validate_round_validator_contract_repair_output(
            repair, repair_plan=plan)
        self.assertTrue(checked["valid"], checked["errors"])
        applied = apply_round_validator_contract_repair(
            base_model_output=base, repair_output=repair, repair_plan=plan)
        self.assertTrue(applied["valid"], applied["errors"])
        compilation = compile_round_validator_model_output(
            applied["reconstructed_output"], realized_scenes=realized)
        self.assertTrue(compilation["valid"], compilation["errors"])
        final = validate_round_validator_output(
            compilation["compiled_output"], realized_scenes=realized,
            review_world_view=review_world, local_gate_results=gates)
        self.assertTrue(final["valid"], final["errors"])

    def test_missing_scene_prompt_and_validator_enforce_action_subset_and_enums(self):
        root, realized, base, review_world, gates, plan = self.load_case(
            "cc-train-episode-216742")
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual([row["scene_alias"] for row in
                          plan["missing_scene_review_targets"]], ["s7"])
        prompt = build_round_validator_contract_repair_prompt(
            repair_plan=plan, frozen_input_sha256="frozen")
        self.assertIn("subset of action evidence", prompt)
        self.assertIn("ALLOWED MISSING-SCENE SEMANTIC VALUES", prompt)

        historical_repair = json.loads((
            root / "attempts" / "round_validator_attempt_02.json"
        ).read_text(encoding="utf-8"))["parsed_response"]
        rejected = validate_round_validator_contract_repair_output(
            historical_repair, repair_plan=plan)
        self.assertFalse(rejected["valid"])
        self.assertTrue(any("primary_progression_evidence" in error
                            for error in rejected["errors"]))
        self.assertTrue(any("decision" in error for error in rejected["errors"]))

        target = plan["missing_scene_review_targets"][0]
        turn_ref = target["allowed_current_scene_turn_refs"][0]
        fixture = deepcopy(historical_repair)
        review = fixture["repairs"][0]["scene_review"]
        review.update({
            "decision": "accept",
            "historical_connection_status": "valid",
            "continuity_status": "valid",
            "historical_consistency_status": "valid",
            "semantic_duplicate_status": "not_duplicate",
            "new_development_status": "valid",
            "reason_codes": ["supported", "primary_action_realized"],
            "historical_supporting_turn_refs": [],
            "current_scene_supporting_turn_refs": [turn_ref],
        })
        realization = review["primary_action_realization"]
        realization["evidence_turn_refs"] = [turn_ref]
        realization["progression_evidence_turn_refs"] = [turn_ref]
        checked = validate_round_validator_contract_repair_output(
            fixture, repair_plan=plan)
        self.assertTrue(checked["valid"], checked["errors"])
        applied = apply_round_validator_contract_repair(
            base_model_output=base, repair_output=fixture, repair_plan=plan)
        self.assertTrue(applied["valid"], applied["errors"])
        compilation = compile_round_validator_model_output(
            applied["reconstructed_output"], realized_scenes=realized)
        self.assertTrue(compilation["valid"], compilation["errors"])
        final = validate_round_validator_output(
            compilation["compiled_output"], realized_scenes=realized,
            review_world_view=review_world, local_gate_results=gates)
        self.assertTrue(final["valid"], final["errors"])

    def test_pilot20_tree_is_unchanged(self):
        self.assertEqual(hash_tree(PILOT20)["tree_sha256"], PILOT20_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
