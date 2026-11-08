from __future__ import annotations

import unittest

from .planner_state_contract_gate_live import (
    _filter_plan_to_eligible, validate_planner_state_output)


class PlannerStateContractTest(unittest.TestCase):
    def setUp(self):
        self.candidates = [{
            "candidate_ref": "ND01:C01",
            "required_transition": "A receives the school confirmation.",
        }]
        self.refs = {"GM:E001:C001": "School responses are pending."}

    def value(self, **updates):
        row = {
            "candidate_ref": "ND01:C01",
            "current_state_stage": "AWAITING_CONFIRMATION",
            "current_state_summary": "School responses are pending.",
            "state_basis_refs": ["GM:E001:C001"],
            "next_transition": "A receives the school confirmation.",
            "completion_condition": "The school explicitly confirms participation.",
        }
        row.update(updates)
        return {"candidate_state_reviews": [row]}

    def test_pending_state_contract_passes(self):
        result = validate_planner_state_output(
            self.value(), candidates=self.candidates,
            memory_ref_map=self.refs)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["eligible_candidate_refs"], ["ND01:C01"])

    def test_completed_and_ambiguous_are_not_eligible(self):
        completed = validate_planner_state_output(
            self.value(current_state_stage="COMPLETED"),
            candidates=self.candidates, memory_ref_map=self.refs)
        self.assertTrue(completed["valid"], completed["errors"])
        self.assertEqual(completed["eligible_candidate_refs"], [])
        ambiguous = validate_planner_state_output(
            self.value(current_state_stage="AMBIGUOUS", state_basis_refs=[]),
            candidates=self.candidates, memory_ref_map=self.refs)
        self.assertTrue(ambiguous["valid"], ambiguous["errors"])
        self.assertEqual(ambiguous["eligible_candidate_refs"], [])

    def test_unknown_memory_ref_is_rejected(self):
        result = validate_planner_state_output(
            self.value(state_basis_refs=["GM:E999:C001"]),
            candidates=self.candidates, memory_ref_map=self.refs)
        self.assertFalse(result["valid"])
        self.assertIn("planner_state_review_1_basis_refs", result["errors"])

    def test_next_transition_drift_is_rejected(self):
        result = validate_planner_state_output(
            self.value(next_transition="A does something else."),
            candidates=self.candidates, memory_ref_map=self.refs)
        self.assertFalse(result["valid"])
        self.assertIn("planner_state_review_1_next_transition_drift",
                      result["errors"])

    def test_ineligible_candidates_are_removed_before_gate(self):
        plan = {"candidate_pool": [{
            "memory_event_id": "ND01",
            "next_step_candidates": [{
                "required_transition": "old",
                "planner_current_state_stage": "AMBIGUOUS",
            }],
        }, {
            "memory_event_id": "ND02",
            "next_step_candidates": [{
                "required_transition": "new",
                "planner_current_state_stage": "AWAITING_CONFIRMATION",
            }],
        }]}
        filtered = _filter_plan_to_eligible(plan, {"ND02:C01"})
        self.assertEqual(
            [scene["memory_event_id"] for scene in filtered["candidate_pool"]],
            ["ND02"])


if __name__ == "__main__":
    unittest.main()
