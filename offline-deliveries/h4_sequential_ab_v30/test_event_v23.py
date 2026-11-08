"""Typed scan observations are model reports, never inferred facts or gates."""
from copy import deepcopy
import unittest

import event_state as scan
import test_event_v19 as event_fixture


class ScanObservationTests(unittest.TestCase):
    def setUp(self):
        fixture = event_fixture.EventV19Tests()
        fixture.setUp()
        self.fixture = fixture
        self.fixture.route = "dataset_new_event"
        self.fixture.result["route"] = self.fixture.route
        self.fixture.result["route_fulfillment"].update(
            role=self.fixture.route, relation_to_history="distinct_event",
            serves_existing_event=False, existing_event="", primary_event="Dinner",
            primary_goal="Arrange a dinner", new_progress=True,
            reason="The dinner is a separate occasion.", correction="")
        self.fixture.result["route_fulfillment"].pop("satisfied", None)

    def validate(self, result=None):
        return scan.validate(self.fixture.result if result is None else result,
            self.fixture.root, self.fixture.route, self.fixture.scene, self.fixture.history)

    def test_empty_timeline_evidence_still_rejected(self):
        result = deepcopy(self.fixture.result)
        result["timeline"][0]["evidence"] = []
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate(result)
        self.assertIn("event_state_evidence_count_invalid", {e["code"] for e in caught.exception.errors})

    def test_empty_evidence_absence_is_not_automatically_moved_to_observations(self):
        result = deepcopy(self.fixture.result)
        result["timeline"][0].update(statement="No earlier celebration was mentioned.", evidence=[])
        before = deepcopy(result)
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate(result)
        repair = scan.build_repair(caught.exception, result, self.fixture.history)
        self.assertEqual(result, before)
        self.assertNotIn("scan_observations", result)
        self.assertNotIn("original_scan_observations", repair)
        self.assertIn("YOU must move that observation", repair["instruction"])
        self.assertIn("not proof of absence", repair["instruction"])

    def test_typed_not_found_observation_is_optional_diagnostic(self):
        result = deepcopy(self.fixture.result)
        result["scan_observations"] = [{
            "observation_id": "o1", "kind": "not_found_in_supplied_history",
            "scope": "cumulative_history", "statement": "No earlier celebration was found."}]
        checked = self.validate(result)
        self.assertEqual(checked["scan_observations"], result["scan_observations"])
        self.assertFalse(scan.requires_replan(checked))
        self.assertEqual(scan.plan_diagnostics(checked), [])

    def test_missing_observations_are_not_filled(self):
        result = self.validate()
        self.assertNotIn("scan_observations", result)

    def test_unknown_observation_kind_scope_or_id_rejected(self):
        for key, value in (("observation_id", "o2"), ("kind", "not_found"),
                           ("scope", "history")):
            with self.subTest(key=key):
                result = deepcopy(self.fixture.result)
                observation = {"observation_id": "o1", "kind": "not_found_in_supplied_history",
                    "scope": "cumulative_history", "statement": "No mention found."}
                observation[key] = value
                result["scan_observations"] = [observation]
                with self.assertRaises(scan.EventStateValidationError) as caught:
                    self.validate(result)
                self.assertIn("event_state_observation_value_invalid",
                              {e["code"] for e in caught.exception.errors})

    def test_unknown_extra_observation_fields_rejected(self):
        result = deepcopy(self.fixture.result)
        result["scan_observations"] = [{"observation_id": "o1", "kind": "not_found_in_supplied_history",
            "scope": "cumulative_history", "statement": "No mention.", "turn_ref": "P01:T001"}]
        with self.assertRaises(scan.EventStateValidationError):
            self.validate(result)

    def test_observation_ids_cannot_be_plan_constraints(self):
        result = deepcopy(self.fixture.result)
        result["scan_observations"] = [{"observation_id": "o1", "kind": "not_found_in_supplied_history",
            "scope": "cumulative_history", "statement": "No prior booking found."}]
        result["plan_checks"][0]["constraint_ids"] = ["o1"]
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate(result)
        self.assertIn("event_state_plan_constraint_ids_invalid",
                      {e["code"] for e in caught.exception.errors})

    def test_repair_keeps_existing_observation_and_negative_plan(self):
        result = deepcopy(self.fixture.result)
        result["scan_observations"] = [{"observation_id": "o1", "kind": "not_found_in_supplied_history",
            "scope": "cumulative_history", "statement": "No prior booking found."}]
        result["plan_checks"][0].update(verdict="revise", comparison_kind="timing_ambiguity",
            conflict={"type": "timing_ambiguity", "historical_claim": "Morning arrival.",
                      "proposed_claim": "Afternoon meeting.", "correction": "Clarify city vs hotel timing."},
            reason="Clarify the timing.")
        broken = deepcopy(result)
        broken["plan_checks"].pop()
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate(broken)
        repair = scan.build_repair(caught.exception, broken, self.fixture.history)
        fixed = deepcopy(broken)
        fixed["plan_checks"].append(deepcopy(self.fixture.result["plan_checks"][1]))
        self.assertTrue(scan.validate_repair_preservation(broken, fixed, repair))
        self.assertEqual(repair["original_scan_observations"], broken["scan_observations"])
        self.assertTrue(scan.requires_replan(self.validate(fixed)))
        erased_negative = deepcopy(fixed)
        erased_negative["plan_checks"][0] = deepcopy(self.fixture.result["plan_checks"][0])
        with self.assertRaises(scan.EventStateValidationError) as negative_error:
            scan.validate_repair_preservation(broken, erased_negative, repair)
        self.assertIn("unaffected_negative", str(negative_error.exception.errors))
        fixed["scan_observations"] = []
        with self.assertRaises(scan.EventStateValidationError) as observation_error:
            scan.validate_repair_preservation(broken, fixed, repair)
        self.assertIn("unaffected_observation", str(observation_error.exception.errors))

    def test_observation_does_not_prove_absence_or_route_identity(self):
        result = deepcopy(self.fixture.result)
        result["scan_observations"] = [{"observation_id": "o1", "kind": "not_found_in_supplied_history",
            "scope": "cumulative_history", "statement": "No earlier interview train booking found."}]
        result["route_fulfillment"].update(primary_event="Interview train booking",
            primary_goal="Reach the interview", relation_to_history="old_event_substep",
            serves_existing_event=True, existing_event="Interview", new_progress=True,
            reason="This serves the interview.", correction="Create an independent event.")
        checked = self.validate(result)
        self.assertTrue(scan.requires_replan(checked))
        self.assertEqual(checked["scan_observations"], result["scan_observations"])

    def test_more_than_four_observations_rejected(self):
        result = deepcopy(self.fixture.result)
        result["scan_observations"] = [{"observation_id": f"o{i}", "kind": "not_found_in_supplied_history",
            "scope": "cumulative_history", "statement": f"No mention {i}."} for i in range(1, 6)]
        with self.assertRaises(scan.EventStateValidationError):
            self.validate(result)


if __name__ == "__main__":
    unittest.main()
