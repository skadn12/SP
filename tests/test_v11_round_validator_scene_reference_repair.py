import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    compile_round_validator_model_output,
    migrate_historical_connected_person_validator_fixture_v5,
    run_round_validator_with_retry,
    validate_round_validator_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.durable_update_compiler import (
    durable_candidate_action_eligible,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract_repair_v11 import (
    apply_round_validator_contract_repair,
    build_round_validator_contract_repair_plan,
    build_round_validator_contract_repair_prompt,
    validate_round_validator_contract_repair_output,
)


CANARY_7 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_7"
)
CANARY_7_TREE_SHA = (
    "3422a93b2a4a3d793d3533d557531e058f608853aa28e58732cb6a737f29dfa1"
)


def load(relative):
    return json.loads((CANARY_7 / relative).read_text(encoding="utf-8"))


class RoundValidatorSceneReferenceRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.realized = [json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted((CANARY_7 / "scenes").glob(
                            "*/realized_scene.json"))]
        cls.raw = migrate_historical_connected_person_validator_fixture_v5(
            load("attempts/round_validator_attempt_01.json")["parsed_response"],
            realized_scenes=cls.realized)
        cls.review = load("review_world_view.json")
        cls.gates = load("local_scene_gate_results.json")
        cls.validator_input = load("round_validator_input.json")
        cls.compilation = compile_round_validator_model_output(
            cls.raw, realized_scenes=cls.realized)
        cls.errors = load("round_validator_run.json")["attempts"][0][
            "validation"]["errors"]
        historical_anchor_refs = {
            row["source_anchor_id"]: row["source_refs"]
            for row in cls.validator_input["historical_anchor_index"]
        }
        eligible_durable_action_refs = [
            action["action_ref"]
            for scene in cls.validator_input["scenes"]
            for role in ("primary", "secondary")
            if isinstance((action := scene.get(f"{role}_action")), dict)
            and durable_candidate_action_eligible(action)
        ]
        cls.plan = build_round_validator_contract_repair_plan(
            model_output=cls.raw, realized_scenes=cls.realized,
            compilation_errors=cls.errors,
            historical_anchor_refs=historical_anchor_refs,
            eligible_durable_action_refs=eligible_durable_action_refs,
        )

    def repair_fixture(self):
        scene_repairs = []
        for target in self.plan["scene_reference_targets"]:
            current = (target["allowed_current_scene_turn_refs"][:1]
                       if target["repair_current_scene_supporting_turn_refs"]
                       else target[
                           "original_current_scene_supporting_turn_refs"])
            if target["repair_dependency_context_turn_refs"]:
                original_uses = set(target.get("original_uses_scene_refs") or [])
                compatible = [
                    ref for ref in target["allowed_dependency_turn_refs"]
                    if ref.rsplit(":T", 1)[0] in original_uses
                ]
                dependency = (compatible or target[
                    "allowed_dependency_turn_refs"])[:1]
            else:
                dependency = target["original_dependency_context_turn_refs"]
            uses = (sorted({ref.rsplit(":T", 1)[0] for ref in dependency})
                    if target.get("repair_uses_scene_refs") else
                    target.get("original_uses_scene_refs") or [])
            repair = {
                "scene_alias": target["scene_alias"],
                "current_scene_supporting_turn_refs": current,
                "dependency_context_turn_refs": dependency,
            }
            if target.get("historical_scope_enabled"):
                repair["historical_supporting_turn_refs"] = (
                    target["allowed_historical_turn_refs"][:1]
                    if target.get("repair_historical_supporting_turn_refs")
                    else target["original_historical_supporting_turn_refs"]
                )
            if target.get("repair_uses_scene_refs"):
                repair["uses_scene_refs"] = uses
            scene_repairs.append(repair)
        action_repairs = []
        for target in self.plan["targets"]:
            evidence = (target["allowed_evidence_turn_refs"][:1]
                        if target["evidence_turn_refs_requires_repair"]
                        else target["original_evidence_turn_refs"])
            action_repairs.append({
                "action_ref": target["action_ref"],
                "evidence_turn_refs": evidence,
                "progression_evidence_turn_refs": evidence[:1],
            })
        return {
            "top_level_rehome": None,
            "scene_reference_repairs": scene_repairs,
            "action_realization_repairs": action_repairs,
            "durable_update_evidence_repairs": [{
                "candidate_update_key": target["candidate_update_key"],
                "evidence_turn_refs": target[
                    "allowed_evidence_turn_refs"][:1],
            } for target in self.plan["durable_update_targets"]],
            "observation_reference_repairs": [],
        }

    def test_canary7_is_exact_supported_compact_repair(self):
        self.assertFalse(self.compilation["valid"])
        self.assertTrue(self.plan["valid"], self.plan["errors"])
        self.assertEqual(self.plan["target_scene_aliases"], ["s4", "s5"])
        s4 = next(row for row in self.plan["targets"]
                  if row["action_ref"] == "S04.primary_action")
        self.assertTrue(s4["evidence_turn_refs_requires_repair"])
        self.assertIn("R01:S04:T006", s4["original_evidence_turn_refs"])
        self.assertNotIn("R01:S04:T006", s4["allowed_evidence_turn_refs"])

    def test_invalid_refs_are_not_locally_replaced(self):
        fixture = self.repair_fixture()
        s4 = next(row for row in fixture["action_realization_repairs"]
                  if row["action_ref"] == "S04.primary_action")
        s4["evidence_turn_refs"] = ["R01:S04:T006"]
        result = validate_round_validator_contract_repair_output(
            fixture, repair_plan=self.plan)
        self.assertFalse(result["valid"])
        self.assertTrue(any("action_evidence_not_scene_subset" in row
                            for row in result["errors"]))

    def test_explicit_reference_resubmission_passes_complete_gate(self):
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

    def test_retry_runner_uses_compact_single_contract_retry(self):
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
        self.assertEqual(result["attempts"][1]["request_mode"],
                         "contract_only_repair")
        self.assertLess(len(calls[1]), len(calls[0]) // 2)

    def test_prompt_exposes_only_exact_allowed_reference_sets(self):
        prompt = build_round_validator_contract_repair_prompt(
            repair_plan=self.plan,
            frozen_input_sha256=self.validator_input[
                "frozen_round_validator_input_sha256"])
        self.assertIn("allowed_current_scene_turn_refs", prompt)
        self.assertIn("allowed_dependency_turn_refs", prompt)
        self.assertNotIn('"round_decision"', prompt)
        self.assertLess(len(prompt), 22_000)

    def test_canary7_history_is_unchanged(self):
        self.assertEqual(hash_tree(CANARY_7)["tree_sha256"], CANARY_7_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
