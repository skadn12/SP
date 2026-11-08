"""Offline fixtures verify mechanics, not semantic accuracy or training data."""
from copy import deepcopy
import json
import unittest

import event_state as scan
import realization_review as review


class RouteContractsTests(unittest.TestCase):
    def setUp(self):
        self.root = "fixture-root"
        self.route = "dataset_new_event"
        self.history = [
            {"turn_ref": "P01:T001", "speaker": "Emily",
             "text": "I plan to attend the job interview tomorrow."},
            {"turn_ref": "R02:A:S01:T001", "speaker": "Emily",
             "text": "I attended the interview and received an offer."},
        ]
        self.scene = {"scene_id": "S01", "developments": [
            "Emily arranges a dinner to celebrate the offer."
        ]}
        self.role = {"role": self.route, "satisfied": True,
            "primary_event": "Dinner celebrating the job offer",
            "relation_to_history": "distinct_event",
            "reason": "The dinner has its own goal while sharing the interview as motivation.",
            "primary_goal": "Arrange celebratory dinner", "serves_existing_event": False,
            "existing_event": "", "new_progress": True, "correction": "",
            "history_evidence": []}
        self.state = {"root_id": self.root, "route": self.route, "scene_id": "S01",
            "segments": [
                {"segment_id": "P01", "none_reason": "The earlier interview plan is superseded by accepted A."},
                {"segment_id": "R02:A:S01", "none_reason": "The completed interview motivates but does not constrain dinner arrangements."},
            ], "timeline": [],
            "plan_checks": [{"development_index": 1, "verdict": "compatible",
                             "constraint_ids": [], "comparison_kind": "new_development", "reason": "A coherent new dinner may be arranged."}],
            "route_fulfillment": deepcopy(self.role)}
        self.realization = {"root_id": self.root, "route": self.route,
            "scene_id": "S01", "status": "scene", "turns": [
                {"speaker": "Emily", "text": "Let's arrange a dinner to celebrate the offer."},
                {"speaker": "James", "text": "I booked a table for our dinner this evening."},
            ]}
        self.answer = {"root_id": self.root, "route": self.route, "scene_id": "S01",
            "checks": [{"development_index": 1, "realized": True,
                       "evidence": [{"turn_ref": "NEW:T002",
                                     "quote": "I booked a table for our dinner this evening."}]}],
            "new_event_or_progress": True, "consistent_with_history": True,
            "duplicates_existing_event": False, "reason": "Dinner arrangements are made.",
            "route_fulfillment": {**deepcopy(self.role), "new_evidence": [
                {"turn_ref": "NEW:T001", "quote": "Let's arrange a dinner to celebrate the offer."}]}
        }

    def validate_state(self, state=None):
        return scan.validate(self.state if state is None else state, self.root,
                             self.route, self.scene, self.history)

    def validate_review(self, answer=None):
        return review.validate_review(self.answer if answer is None else answer, self.root,
            self.route, self.scene, self.realization, history=self.history)

    def test_distinct_event_can_share_people_and_motivation(self):
        state = self.validate_state()
        self.assertFalse(scan.requires_replan(state))
        result = self.validate_review()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["schema_valid"])
        self.assertFalse(result["semantic_certified"])

    def test_distinct_event_does_not_require_absence_proof(self):
        self.assertEqual(self.state["route_fulfillment"]["history_evidence"], [])
        self.validate_state()
        self.assertEqual(self.validate_review()["status"], "PASS")

    def test_old_event_substep_binds_replanning_even_with_compatible_developments(self):
        role = self.state["route_fulfillment"]
        role.update(satisfied=False, primary_event="Interview transport",
                    relation_to_history="old_event_substep", serves_existing_event=True,
                    existing_event="Interview", correction="Choose an independent new purpose",
                    reason="Adding transport for the old interview is only a supporting step.",
                    history_evidence=[{"turn_ref": "P01:T001",
                                       "quote": "I plan to attend the job interview tomorrow."}])
        checked = self.validate_state()
        original = deepcopy(checked)
        self.assertTrue(scan.requires_replan(checked))
        finding = scan.binding_plan_checks(checked)[0]
        self.assertEqual(finding["development_index"], 0)
        self.assertTrue(finding["route_role_check"])
        self.assertEqual(finding["route_fulfillment"], role)
        self.assertEqual(checked, original)

    def test_wrong_relation_cannot_pass_when_satisfied_is_true(self):
        self.state["route_fulfillment"].update(relation_to_history="continuation",
            history_evidence=[{"turn_ref": "P01:T001", "quote": "I plan to attend the job interview tomorrow."}])
        with self.assertRaises(scan.EventStateValidationError):
            self.validate_state()

    def test_unclear_role_is_substantive_negative_not_schema_error(self):
        self.answer["route_fulfillment"].update(satisfied=False, relation_to_history="unclear",
            reason="It is unclear whether this is a separate event.",
            correction="Clarify the separate event purpose", new_evidence=[])
        result = self.validate_review()
        self.assertEqual(result["status"], "REJECT")
        self.assertTrue(result["schema_valid"])
        self.assertTrue(any(item.startswith("review_route_fulfillment_reject:") for item in result["failures"]))

    def test_missing_role_is_schema_failure(self):
        self.state.pop("route_fulfillment")
        with self.assertRaises(scan.EventStateValidationError):
            self.validate_state()
        self.answer.pop("route_fulfillment")
        self.assertFalse(self.validate_review()["schema_valid"])

    def test_boolean_and_role_identity_are_strict(self):
        for key, invalid in (("satisfied", 1), ("satisfied", "true"),
                             ("role", "old_event_continuation"), ("relation_to_history", "new")):
            with self.subTest(key=key, invalid=invalid):
                state = deepcopy(self.state)
                state["route_fulfillment"][key] = invalid
                with self.assertRaises(scan.EventStateValidationError):
                    self.validate_state(state)
                answer = deepcopy(self.answer)
                answer["route_fulfillment"][key] = invalid
                self.assertFalse(self.validate_review(answer)["schema_valid"])

    def test_accepted_a_can_be_cited_as_b_history(self):
        citation = {"turn_ref": "R02:A:S01:T001",
                    "quote": "I attended the interview and received an offer."}
        self.state["route_fulfillment"]["history_evidence"] = [citation]
        self.answer["route_fulfillment"]["history_evidence"] = [citation]
        self.validate_state()
        self.assertEqual(self.validate_review()["status"], "PASS")

    def test_invalid_or_paraphrased_history_citation_is_schema_failure(self):
        for citation in ({"turn_ref": "P99:T001", "quote": "I attended the interview"},
                         {"turn_ref": "R02:A:S01:T001", "quote": "The interview was completed."}):
            self.state["route_fulfillment"]["history_evidence"] = [citation]
            self.answer["route_fulfillment"]["history_evidence"] = [citation]
            with self.assertRaises(scan.EventStateValidationError):
                self.validate_state()
            self.assertFalse(self.validate_review()["schema_valid"])

    def test_satisfied_realization_needs_exact_new_evidence(self):
        for evidence in ([], [{"turn_ref": "NEW:T001", "quote": "Dinner happened already."}],
                         [{"turn_ref": "R02:A:S01:T001", "quote": "I attended the interview"}]):
            self.answer["route_fulfillment"]["new_evidence"] = evidence
            self.assertFalse(self.validate_review()["schema_valid"])

    def test_a_continuation_requires_history_and_can_execute_old_plan(self):
        self.route = "old_event_continuation"
        self.state["route"] = self.route
        self.state["route_fulfillment"].update(role=self.route, relation_to_history="continuation",
            primary_event="Interview follow-up", serves_existing_event=True, existing_event="Interview", reason="The interview leads to a new offer discussion.",
            history_evidence=[{"turn_ref": "R02:A:S01:T001", "quote": "I attended the interview"}])
        self.assertFalse(scan.requires_replan(self.validate_state()))
        self.state["route_fulfillment"]["history_evidence"] = []
        with self.assertRaises(scan.EventStateValidationError):
            self.validate_state()

    def test_existing_development_gate_is_preserved(self):
        self.state["plan_checks"][0].update(verdict="revise", comparison_kind="historical_assertion",
            reason="Unstated earlier completion is assumed.", conflict={"type":"unsupported_prior_assertion",
            "historical_claim":"There is no historical report that dinner happened",
            "proposed_claim":"Assumes dinner was already completed", "correction":"Introduce dinner as a new event"})
        self.assertTrue(scan.requires_replan(self.validate_state()))
        self.assertEqual(scan.binding_plan_checks(self.state), self.state["plan_checks"])
        self.answer["checks"][0].update(realized=False, evidence=[])
        result = self.validate_review()
        self.assertEqual(result["status"], "REJECT")
        self.assertTrue(result["schema_valid"])

    def test_existing_duplicate_gate_is_preserved(self):
        self.answer["duplicates_existing_event"] = True
        self.assertEqual(self.validate_review()["status"], "REJECT")

    def test_existing_coverage_and_exact_development_evidence_are_preserved(self):
        self.state["segments"].pop()
        with self.assertRaises(scan.EventStateValidationError):
            self.validate_state()
        self.answer["checks"][0]["evidence"][0]["quote"] = "Not in the scene"
        self.assertFalse(self.validate_review()["schema_valid"])

    def test_review_request_contains_full_history_and_complete_check_ids(self):
        prompt = review.build_review_prompt(self.root, self.route, self.scene, self.history, self.realization)
        context = json.loads(prompt.split("\nREVIEW INPUT DATA:\n", 1)[1])
        self.assertEqual(context["branch_history"], self.history)
        self.assertEqual(context["expected_check_ids"]["development_indices"], [1])
        self.assertEqual(context["new_scene"]["turns"][0]["turn_ref"], "NEW:T001")


if __name__ == "__main__":
    unittest.main()
