"""Purpose binding and assignment-replan controls; scripted judgments, no API."""
from copy import deepcopy
import json
import unittest

import event_state as scan
import latest_state as state
import realization_review as review
import test_event_v19 as event_fixture
import test_seed_review_v24 as review_fixture


def input_data(request, marker):
    text = request["messages"][0]["content"] if isinstance(request, dict) else request
    return json.JSONDecoder().raw_decode(text.split(marker)[1])[0]


class PurposeReviewV25Tests(unittest.TestCase):
    def setUp(self):
        fixture = review_fixture.SeedReviewV24Tests()
        fixture.setUp()
        self.fixture = fixture
        self.purpose = {"root_id": fixture.root, "route": fixture.route,
            "scene_id": fixture.scene["scene_id"], "dataset_card_id": fixture.card["card_id"],
            "trigger": "A newly lost wallet needs recovery.", "goal": fixture.scene["goal"]}
        self.history = deepcopy(fixture.history) + [
            {"turn_ref": "R02:A:S01:T001", "text": "The previous interview rehearsal is now done."},
            {"turn_ref": "R02:B:S01:T001", "text": "A prior accepted independent dinner is planned."}]

    def scan_fixture(self):
        fixture = event_fixture.EventV19Tests()
        fixture.setUp()
        fixture.as_b(independent=True)
        fixture.scene["goal"] = "Organize an independent weekend hotel stay."
        fixture.scene["developments"] = ["Book a hotel solely to attend the old interview.",
                                          "Ask Morgan to help prepare that old interview."]
        fixture.result["route_fulfillment"].update(
            primary_event="Another independent activity", primary_goal="A scripted distinct goal",
            relation_to_history="distinct_event", serves_existing_event=False, existing_event="", new_progress=True)
        row = fixture.result["plan_checks"][0]
        row.update(verdict="revise", constraint_ids=[], comparison_kind="new_development",
            reason="SYNTHETIC: copied goal label says independent stay; actual developments serve the old interview.",
            conflict={"type": "assigned_event_mismatch", "historical_claim": "",
                      "proposed_claim": "Book interview accommodation instead of the assigned independent stay.",
                      "correction": "Keep the assigned stay's own purpose and remove interview-serving work."})
        return fixture

    def test_all_constructor_inputs_keep_same_purpose_and_complete_history(self):
        f = self.fixture
        purpose_before, history_before = deepcopy(self.purpose), deepcopy(self.history)
        contexts = [
            (input_data(state.build_state_request(f.root, f.route, "S01", self.history, [f.card],
                purpose_input=self.purpose), "\nSTATE_INPUT_JSON\n"), "cumulative_history"),
            (input_data(scan.build_request(f.root, f.route, f.scene, self.history, [f.card], {},
                purpose_input=self.purpose), "\nEVENT_STATE_INPUT_JSON\n"), "cumulative_history"),
            (input_data(review.build_review_prompt(f.root, f.route, f.scene, self.history, f.realization,
                dataset_card=f.card, purpose_input=self.purpose), "\nREVIEW INPUT DATA:\n"), "branch_history"),
        ]
        for context, history_key in contexts:
            self.assertEqual(context["purpose_input"], purpose_before)
            self.assertEqual(context[history_key], history_before)
            self.assertEqual(scan.digest(context["purpose_input"]), scan.digest(purpose_before))
            context["purpose_input"]["goal"] = "Changed validation view only"
        self.assertEqual(self.purpose, purpose_before)
        self.assertEqual(self.history, history_before)

    def test_mismatched_purpose_card_or_scene_rejected_before_request(self):
        f = self.fixture
        for field in ("dataset_card_id", "scene_id", "root_id", "route"):
            purpose = {**self.purpose, field: "WRONG"}
            for constructor in (
                lambda: state.build_state_request(f.root, f.route, "S01", self.history, [f.card], purpose_input=purpose),
                lambda: scan.build_request(f.root, f.route, f.scene, self.history, [f.card], {}, purpose_input=purpose),
                lambda: review.build_review_prompt(f.root, f.route, f.scene, self.history, f.realization,
                                                  dataset_card=f.card, purpose_input=purpose),
            ):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    constructor()

    def test_assignment_failure_requires_replan_even_if_route_tuple_is_positive(self):
        f = self.scan_fixture()
        original = deepcopy(f.result)
        checked = f.validate()
        self.assertTrue(scan.route_fulfilled(checked["route_fulfillment"], f.route))
        self.assertTrue(scan.requires_replan(checked))
        binding = scan.binding_plan_checks(checked)
        self.assertEqual(len(binding), 1)
        self.assertEqual(binding[0]["conflict"]["type"], "assigned_event_mismatch")
        self.assertEqual(binding[0]["constraint_ids"], [])
        self.assertEqual(binding[0]["conflict"]["historical_claim"], "")
        self.assertEqual(f.result, original)

    def test_assignment_repair_does_not_invent_history_or_absorb_soft_constraint(self):
        for mutate in ("cite_history", "historical_claim", "wrong_kind", "wrong_route"):
            f = self.scan_fixture()
            row = f.result["plan_checks"][0]
            if mutate == "cite_history":
                row["constraint_ids"] = ["c1"]
                f.result["timeline"][0]["evidence_basis"] = "inferred"
            elif mutate == "historical_claim":
                row["conflict"]["historical_claim"] = "The purpose happened previously."
            elif mutate == "wrong_kind":
                row["comparison_kind"] = "historical_assertion"
            else:
                f.route = "old_event_continuation"
                f.result["route"] = f.route
            with self.subTest(mutate=mutate), self.assertRaises(scan.EventStateValidationError):
                f.validate()

    def test_schema_repair_cannot_erase_assignment_negative(self):
        f = self.scan_fixture()
        f.result["plan_checks"].pop()
        with self.assertRaises(scan.EventStateValidationError) as caught:
            f.validate()
        repair = scan.build_repair(caught.exception, f.result, f.history)
        updated = deepcopy(f.result)
        updated["plan_checks"][0].update(verdict="compatible", conflict=None,
            reason="A copied goal label alone is sufficient.")
        with self.assertRaisesRegex(scan.EventStateValidationError, "unaffected_negative"):
            scan.validate_repair_preservation(f.result, updated, repair)

    def test_empty_evidence_error_points_to_existing_observation_without_mutation(self):
        f = self.scan_fixture()
        f.result["timeline"][0].update(statement="No prior independent hotel stay was mentioned.", evidence=[])
        before = deepcopy(f.result)
        with self.assertRaises(scan.EventStateValidationError) as caught:
            f.validate()
        errors = [e for e in caught.exception.errors if e["code"] == "event_state_evidence_count_invalid"]
        self.assertEqual(errors[0]["path"], "timeline[0].evidence")
        self.assertEqual(errors[0]["alternative_output_path"], "scan_observations")
        self.assertEqual(errors[0]["statement"], before["timeline"][0]["statement"])
        self.assertIn("Never invent evidence", errors[0]["repair_instruction"])
        self.assertEqual(errors[0]["observation_shape"]["kind"], "not_found_in_supplied_history")
        repair = scan.build_repair(caught.exception, f.result, f.history)
        self.assertEqual(repair["original_negative_findings"], [f.result["plan_checks"][0]])
        self.assertEqual(f.result, before)
        self.assertNotIn("scan_observations", f.result)

    def test_purpose_cannot_be_timeline_or_review_evidence(self):
        f = self.scan_fixture()
        f.result["timeline"][0]["evidence"] = [{"turn_ref": "purpose_input.goal"}]
        with self.assertRaises(scan.EventStateValidationError):
            f.validate()
        raw = deepcopy(self.fixture.raw)
        raw["checks"][0]["evidence"] = [{"turn_ref": "purpose_input.goal"}]
        result = self.fixture.validate(raw)
        self.assertEqual(result["status"], "REJECT")
        self.assertFalse(result["schema_valid"])


if __name__ == "__main__":
    unittest.main()
