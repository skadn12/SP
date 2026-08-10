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
    pack_round_validator_contract_repair_output,
    validate_round_validator_contract_repair_output,
)


CANARY_10 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_10"
)
CANARY_10_TREE_SHA = (
    "24ada6b7e107ee3b65505cdfae88044044d4c692e2139281115e045832c626ff"
)


def load(relative):
    return json.loads((CANARY_10 / relative).read_text(encoding="utf-8"))


class RoundValidatorUnplannedFutureDurableRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.realized = load("round_validator_input.json")["scenes"]
        cls.raw = migrate_historical_connected_person_validator_fixture_v5(
            load("attempts/round_validator_attempt_01.json")["parsed_response"],
            realized_scenes=cls.realized)
        cls.review = load("review_world_view.json")
        cls.gates = load("local_scene_gate_results.json")
        cls.validator_input = load("round_validator_input.json")
        cls.errors = load("round_validator_run.json")["attempts"][0][
            "validation"]["errors"]
        cls.plan = build_round_validator_contract_repair_plan(
            model_output=cls.raw, realized_scenes=cls.realized,
            compilation_errors=cls.errors)

    def flat_fixture(self):
        grouped = {
            "top_level_rehome": None,
            "scene_reference_repairs": [],
            "action_realization_repairs": [],
            "durable_update_evidence_repairs": [],
            "observation_reference_repairs": [],
            "unplanned_action_repairs": [],
            "future_knowledge_repairs": [],
            "durable_update_removals": [],
            "durable_dependency_repairs": [],
        }
        for target in self.plan["scene_reference_targets"]:
            current = (target["allowed_current_scene_turn_refs"][:1]
                       if target["repair_current_scene_supporting_turn_refs"]
                       else target["original_current_scene_supporting_turn_refs"])
            dependency = (target["allowed_dependency_turn_refs"][:1]
                          if target["repair_dependency_context_turn_refs"]
                          else target["original_dependency_context_turn_refs"])
            grouped["scene_reference_repairs"].append({
                "scene_alias": target["scene_alias"],
                "current_scene_supporting_turn_refs": current,
                "dependency_context_turn_refs": dependency,
            })
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
                "evidence_turn_refs": target["allowed_evidence_turn_refs"][:1],
            })
        for target in self.plan["observation_targets"]:
            source = next(
                scene_ref for scene_ref in target["allowed_scene_refs"]
                if target["allowed_evidence_turn_refs_by_scene"].get(scene_ref))
            grouped["observation_reference_repairs"].append({
                "observation_key": target["observation_key"],
                "source_scene_refs": [source],
                "evidence_turn_refs": target[
                    "allowed_evidence_turn_refs_by_scene"][source][:1],
            })
        for target in self.plan["unplanned_action_targets"]:
            grouped["unplanned_action_repairs"].append({
                "scene_alias": target["scene_alias"],
                "action_role": target["action_role"],
                "replacement": None,
            })
        for target in self.plan["future_knowledge_targets"]:
            grouped["future_knowledge_repairs"].append({
                "scene_alias": target["scene_alias"], "value": False,
            })
        for target in self.plan["durable_removal_targets"]:
            grouped["durable_update_removals"].append({
                "candidate_update_key": target["candidate_update_key"],
                "replacement": None,
            })
        for target in self.plan["durable_dependency_targets"]:
            grouped["durable_dependency_repairs"].append({
                "candidate_update_key": target["candidate_update_key"],
                "depends_on_candidate_keys": [],
            })
        return pack_round_validator_contract_repair_output(grouped)

    def test_canary10_targets_are_exact(self):
        self.assertTrue(self.plan["valid"], self.plan["errors"])
        self.assertEqual(self.plan["target_unplanned_action_slots"], [
            "s3.secondary"])
        self.assertEqual(self.plan["target_future_knowledge_scenes"], [
            "s1", "s2", "s3", "s4", "s5", "s6", "s7"])
        self.assertEqual(
            self.plan["target_removed_update_keys"],
            ["DU003", "DU004", "DU005", "DU006", "DU007"],
        )
        self.assertEqual(self.plan["durable_dependency_targets"], [])
        reasons = {
            row["candidate_update_key"]: row["reason"]
            for row in self.plan["durable_removal_targets"]
        }
        self.assertEqual(reasons["DU003"], "source_action_not_planned")
        self.assertTrue(all(
            reasons[key] == "dependency_candidate_removed"
            for key in ("DU004", "DU005", "DU006", "DU007")
        ))

    def test_new_repair_kinds_are_present_in_flat_envelope(self):
        kinds = {row["repair_kind"] for row in self.flat_fixture()["repairs"]}
        self.assertIn("unplanned_action_nullification", kinds)
        self.assertIn("future_knowledge_boolean", kinds)
        self.assertIn("durable_update_nullification", kinds)
        self.assertNotIn("durable_update_dependencies", kinds)

    def test_future_knowledge_requires_json_boolean(self):
        fixture = self.flat_fixture()
        item = next(row for row in fixture["repairs"]
                    if row["repair_kind"] == "future_knowledge_boolean")
        item["value"] = "valid"
        result = validate_round_validator_contract_repair_output(
            fixture, repair_plan=self.plan)
        self.assertFalse(result["valid"])
        self.assertTrue(any("value_not_boolean" in row
                            for row in result["errors"]))

    def test_unplanned_update_must_be_nullified(self):
        fixture = self.flat_fixture()
        item = next(row for row in fixture["repairs"]
                    if row["repair_kind"] == "durable_update_nullification")
        item["replacement"] = {}
        result = validate_round_validator_contract_repair_output(
            fixture, repair_plan=self.plan)
        self.assertFalse(result["valid"])
        self.assertTrue(any("replacement_must_be_null" in row
                            for row in result["errors"]))

    def test_dependency_chain_must_be_explicitly_nullified(self):
        fixture = self.flat_fixture()
        item = next(row for row in fixture["repairs"]
                    if row["repair_kind"] == "durable_update_nullification"
                    and row["candidate_update_key"] == "DU004")
        fixture["repairs"].remove(item)
        result = validate_round_validator_contract_repair_output(
            fixture, repair_plan=self.plan)
        self.assertFalse(result["valid"])
        self.assertTrue(any("durable_update_removal:missing_keys:DU004" in row
                            for row in result["errors"]))

    def test_flat_fixture_passes_complete_validator_gate(self):
        applied = apply_round_validator_contract_repair(
            base_model_output=self.raw, repair_output=self.flat_fixture(),
            repair_plan=self.plan)
        self.assertTrue(applied["valid"], applied["errors"])
        reconstructed = applied["reconstructed_output"]
        self.assertFalse({"DU003", "DU004", "DU005", "DU006", "DU007"} & {
            row["candidate_update_key"] for row in reconstructed["durable_updates"]})
        compilation = compile_round_validator_model_output(
            reconstructed, realized_scenes=self.realized)
        self.assertTrue(compilation["valid"], compilation["errors"])
        validation = validate_round_validator_output(
            compilation["compiled_output"], realized_scenes=self.realized,
            review_world_view=self.review, local_gate_results=self.gates)
        self.assertTrue(validation["valid"], validation["errors"])

    def test_runner_uses_contract_only_second_request(self):
        calls = []

        def backend(prompt, attempt):
            calls.append(prompt)
            return self.raw if attempt == 1 else self.flat_fixture()

        result = run_round_validator_with_retry(
            backend_call=backend, validator_input=self.validator_input,
            realized_scenes=self.realized, review_world_view=self.review,
            local_gate_results=self.gates)
        self.assertTrue(result["completed"])
        self.assertTrue(result["contract_only_repair_used"])
        self.assertEqual(result["physical_calls"], 2)

    def test_canary10_history_is_unchanged(self):
        self.assertEqual(hash_tree(CANARY_10)["tree_sha256"], CANARY_10_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
