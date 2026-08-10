import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_fresh_e2e_v11 import (
    hybrid_request_budget, run_mock_hybrid_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.local_scene_gate_v11 import dependency_unavailable_result


PHASE_A = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_61555_slim_continuous_expansion_v11_phase_a_preflight")
V10 = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v10")


def load(path):
    return json.loads(path.read_text())


def fixture_inputs():
    plan = deepcopy(load(PHASE_A / "round_blueprint_fixture.json")["model_fixture"])
    plan["scenes"][1]["candidate_update_hints"][0]["proposed_content"][
        "task_description"] = "prepare a written contingency schedule for the department activity"
    plan["scenes"][3]["candidate_update_hints"] = [{
        "hint_mode": "single_update", "action_id": "A0001",
        "proposed_content": {"description": "A concrete coordination meeting occurs.",
                             "narrative_time_expression": "the following afternoon"}}]
    return dict(complete_dialogue=load(PHASE_A / "complete_dialogue_view.json"),
                planning_state=load(PHASE_A / "planning_state_view.json"),
                anchor_registry=load(PHASE_A / "anchor_registry.json"),
                action_menu=load(V10 / "affordance_menu.json"), planner_output=plan,
                authoritative_memory=load(V10 / "world_memory_final.json"))


def run_case(case):
    with tempfile.TemporaryDirectory() as temp:
        return run_mock_hybrid_v11(**fixture_inputs(), run_root=Path(temp), case=case)


class HybridRuntimeTests(unittest.TestCase):
    def test_all_accept_two_transactions(self):
        result = run_case("all_accept")
        self.assertTrue(result["completed"])
        self.assertEqual(result["mock_calls"], {"planner": 1, "realizer": 6,
                         "round_validator": 1, "evidence_committer": 0})
        self.assertEqual(len(result["realized_scenes"]), 6)
        self.assertEqual(len(result["promotion_v2_result"]["promotable_child_scenes"]), 6)
        self.assertEqual(len(result["durable_update_compile_result"]["accepted_updates"]), 2)
        self.assertEqual(len(result["transactions"]), 2)
        self.assertTrue(all(row["status"] == "COMMITTED" for row in result["transactions"]))

    def test_accept_no_updates_still_promotes(self):
        result = run_case("accept_no_updates")
        self.assertTrue(result["completed"])
        self.assertEqual(len(result["durable_update_compile_result"]["accepted_updates"]), 0)
        self.assertEqual(result["transactions"], [])
        self.assertEqual(len(result["promotion_v2_result"]["promotable_child_scenes"]), 6)

    def test_reject_middle_closes_dependents(self):
        result = run_case("reject_middle_scene")
        rejected = result["scene_decision_closure"]["rejected_scene_refs"]
        self.assertIn("R02:S02", rejected)
        self.assertIn("R02:S03", rejected)

    def test_uncertain_not_retried(self):
        result = run_case("uncertain_middle_scene")
        self.assertEqual(result["round_validator_run"]["physical_calls"], 1)
        self.assertIn("R02:S02", result["scene_decision_closure"]["rejected_scene_refs"])

    def test_independent_parallel_branch_survives(self):
        result = run_case("independent_parallel_branch_survives")
        self.assertIn("R02:S04", result["scene_decision_closure"]["accepted_scene_refs"])

    def test_parallel_branch_using_rejected_information_is_blocked(self):
        result = run_case("reject_middle_scene")
        self.assertIn("R02:S04", result["scene_decision_closure"]["rejected_scene_refs"])

    def test_dependency_unavailable_does_not_call_realizer_contract(self):
        row = dependency_unavailable_result(
            {"scene_ref": "R02:S03", "scene_alias": "s3"}, ["R02:S02"])
        self.assertEqual(row["status"], "dependency_unavailable")
        self.assertEqual(row["blocking_predecessor_refs"], ["R02:S02"])

    def test_review_view_excludes_utterance_memory_mirror(self):
        result = run_case("accept_no_updates")
        view = result["review_world_view"]
        self.assertEqual(view["excluded_record_types"], ["parent_utterance"])
        self.assertNotIn("authoritative_records", view)
        self.assertTrue(view["historical_anchors"][0]["exact_excerpt"])

    def test_budget(self):
        budget = hybrid_request_budget()
        self.assertEqual(budget["normal_min"]["physical_calls"], 7)
        self.assertEqual(budget["normal_max"]["physical_calls"], 9)
        self.assertEqual(budget["worst_case"]["physical_calls"], 26)
        self.assertEqual(budget["committer_physical_calls"], 0)


if __name__ == "__main__":
    unittest.main()
