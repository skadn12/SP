import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from tests.test_world_memory_slim_v11_hybrid_runtime import fixture_inputs
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_fresh_e2e_v11 import run_mock_hybrid_v11
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.durable_update_compiler import compile_durable_updates


class DurableCompilerTests(unittest.TestCase):
    def run_case(self, case):
        with tempfile.TemporaryDirectory() as temp:
            return run_mock_hybrid_v11(**fixture_inputs(), run_root=Path(temp), case=case)

    def test_one_valid_one_invalid_independent(self):
        result = self.run_case("one_valid_one_invalid_update")
        compiled = result["durable_update_compile_result"]
        self.assertEqual(len(compiled["accepted_updates"]), 1)
        self.assertEqual(len(compiled["rejected_updates"]), 1)
        self.assertEqual(len(result["transactions"]), 1)

    def test_atomic_partial_failure_blocks_group(self):
        result = self.run_case("atomic_group_partial_failure")
        compiled = result["durable_update_compile_result"]
        self.assertEqual(len(compiled["accepted_updates"]), 0)
        self.assertGreaterEqual(len(compiled["rejected_updates"]), 2)
        self.assertEqual(result["transactions"], [])

    def test_true_atomic_group_closure_blocks_valid_sibling(self):
        result = self.run_case("all_accept")
        plan = deepcopy(result["compiled_round_plan"])
        candidates = deepcopy(result["round_validator_output"]["durable_updates"])
        for candidate in candidates:
            candidate["atomic_group_key"] = "AG001"
        candidates[1]["verification_status"] = "rejected"
        inputs = fixture_inputs()
        compiled = compile_durable_updates(
            candidates=candidates, compiled_round_plan=plan,
            scene_decision_closure=result["scene_decision_closure"],
            realized_scenes=result["realized_scenes"],
            authoritative_memory=inputs["authoritative_memory"],
            planning_state=inputs["planning_state"],
            action_menu=inputs["action_menu"])
        self.assertEqual(compiled["accepted_updates"], [])
        self.assertTrue(any("candidate_atomic_group_incomplete" in row["errors"]
                            for row in compiled["rejected_updates"]))

    def test_unhinted_observation_never_binds(self):
        result = self.run_case("unhinted_update_observation")
        self.assertEqual(result["transactions"], [])
        self.assertEqual(len(result["round_validator_output"][
            "semantic_review_observations"]), 1)
        self.assertEqual(len(result["durable_update_compile_result"][
            "audit_only_observations"]), 1)

    def test_evidence_from_rejected_scene_not_compiled(self):
        result = self.run_case("reject_middle_scene")
        self.assertFalse(any("R02:S02" in row["source_scene_refs"]
                             for row in result["durable_update_compile_result"][
                                 "accepted_updates"]))

    def test_post_review_duplicate_fingerprint_is_rejected(self):
        result = self.run_case("all_accept")
        candidates = deepcopy(result["round_validator_output"]["durable_updates"][:1])
        duplicate = deepcopy(candidates[0]); duplicate["candidate_update_key"] = "DU999"
        candidates.append(duplicate)
        inputs = fixture_inputs()
        compiled = compile_durable_updates(
            candidates=candidates, compiled_round_plan=result["compiled_round_plan"],
            scene_decision_closure=result["scene_decision_closure"],
            realized_scenes=result["realized_scenes"],
            authoritative_memory=inputs["authoritative_memory"],
            planning_state=inputs["planning_state"], action_menu=inputs["action_menu"])
        self.assertEqual(len(compiled["accepted_updates"]), 1)
        self.assertTrue(any("candidate_semantic_or_fingerprint_duplicate" in row["errors"]
                            for row in compiled["rejected_updates"]))

    def test_post_review_transition_still_checks_pre_state(self):
        result = self.run_case("accept_no_updates")
        inputs = fixture_inputs()
        task = next(row for row in inputs["authoritative_memory"]["authoritative_records"]
                    if row.get("record_type") == "task")
        entities = [row["entity_ref"] for row in inputs["planning_state"]["entities"]]
        plan = deepcopy(result["compiled_round_plan"])
        plan["scenes"][0]["primary_action"] = {
            "action_type": "assign_or_handoff_role", "arguments": {
                "task_or_responsibility": task["memory_ref"],
                "from_participant": entities[0], "to_participant": entities[1],
                "handoff_reason": "a verified workload change"}}
        menu = deepcopy(inputs["action_menu"])
        menu["actions"].append({
            "action_id": "POST001", "operation": "add_role_handoff",
            "operation_class": "transition", "root_id": menu["root_id"],
            "connected_memory_refs": [task["memory_ref"], *entities],
            "allowed_existing_entity_refs": entities,
            "payload_schema": {"target_memory_ref": "memory_ref",
                "source_ref": "entity_ref", "target_ref": "entity_ref",
                "handoff_description": "string"},
            "frozen_payload": {"target_memory_ref": task["memory_ref"],
                "source_ref": entities[0], "target_ref": entities[1]},
            "target_memory_ref": task["memory_ref"],
            "current_value": "deliberately stale pre-state",
            "authoritative_version": task["authoritative_version"],
        })
        scene_ref = plan["scenes"][0]["scene_ref"]
        evidence = result["realized_scenes"][0]["dialogue_turns"][0]["turn_ref"]
        compiled = compile_durable_updates(
            candidates=[{"candidate_update_key": "DU001",
                "source_scene_ref": scene_ref, "source_action_role": "primary",
                "evidence_turn_refs": [evidence], "verification_status": "confirmed",
                "atomic_group_key": None, "depends_on_candidate_keys": []}],
            compiled_round_plan=plan,
            scene_decision_closure=result["scene_decision_closure"],
            realized_scenes=result["realized_scenes"],
            authoritative_memory=inputs["authoritative_memory"],
            planning_state=inputs["planning_state"], action_menu=menu)
        self.assertEqual(compiled["accepted_updates"], [])
        self.assertTrue(any("candidate_pre_state_mismatch" in row["errors"]
                            for row in compiled["rejected_updates"]))


if __name__ == "__main__":
    unittest.main()
