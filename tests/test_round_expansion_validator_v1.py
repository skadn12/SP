import tempfile
import unittest
from pathlib import Path

from tests.test_world_memory_slim_v11_hybrid_runtime import fixture_inputs
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_fresh_e2e_v11 import run_mock_hybrid_v11
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import validate_round_validator_output


class RoundValidatorTests(unittest.TestCase):
    def run_case(self, case):
        with tempfile.TemporaryDirectory() as temp:
            return run_mock_hybrid_v11(**fixture_inputs(), run_root=Path(temp), case=case)

    def test_parse_invalid_retry_freezes_input(self):
        result = self.run_case("validator_parse_invalid_then_valid")
        run = result["round_validator_run"]
        self.assertEqual(run["physical_calls"], 2)
        self.assertEqual(len({row["frozen_input_sha256"] for row in run["attempts"]}), 1)

    def test_invalid_ref_retry(self):
        result = self.run_case("validator_invalid_ref_then_valid")
        self.assertEqual(result["round_validator_run"]["physical_calls"], 2)
        self.assertEqual(result["round_validator_run"]["attempts"][0]["validation"][
            "classification"], "reference_invalid")

    def test_legal_reject_no_confirmation_shopping(self):
        result = self.run_case("validator_legal_reject_no_retry")
        self.assertEqual(result["round_validator_run"]["physical_calls"], 1)

    def test_final_id_forbidden(self):
        result = self.run_case("accept_no_updates")
        output = dict(result["round_validator_output"])
        output["semantic_review_observations"] = [{
            "observation_key": "O", "update_type": "add_event",
            "source_scene_refs": ["R02:S01"],
            "evidence_turn_refs": ["R02:S01:T001"],
            "description": "audit", "disposition": "audit_only_unhinted_update",
            "memory_id": "M999"}]
        validation = validate_round_validator_output(
            output, realized_scenes=result["realized_scenes"],
            review_world_view=result["review_world_view"],
            local_gate_results=result["local_scene_gate_results"])
        self.assertFalse(validation["valid"])

    def test_invented_evidence_turn_ref_rejected(self):
        result = self.run_case("all_accept")
        output = dict(result["round_validator_output"])
        output["durable_updates"] = [dict(row) for row in output["durable_updates"]]
        output["durable_updates"][0]["evidence_turn_refs"] = ["R02:S02:T999"]
        validation = validate_round_validator_output(
            output, realized_scenes=result["realized_scenes"],
            review_world_view=result["review_world_view"],
            local_gate_results=result["local_scene_gate_results"])
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["classification"], "reference_invalid")

    def test_missing_scene_review_rejected(self):
        result = self.run_case("accept_no_updates")
        output = dict(result["round_validator_output"])
        output["scene_reviews"] = output["scene_reviews"][:-1]
        validation = validate_round_validator_output(
            output, realized_scenes=result["realized_scenes"],
            review_world_view=result["review_world_view"],
            local_gate_results=result["local_scene_gate_results"])
        self.assertFalse(validation["valid"])
        self.assertIn("validator_scene_review_coverage", validation["errors"])


if __name__ == "__main__":
    unittest.main()
