"""Authored judgment routing and exact-reference regressions, not model accuracy.

The event purpose labels here are explicit test annotations. No production
keyword classifier or automatic semantic acceptance is exercised or installed.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import event_state as event
import realization_review as scene_review
from test_event_state_v92 import review_fixture
import test_scene_consistency as scene_fixtures


class EventFunctionV97Tests(unittest.TestCase):
    def setUp(self):
        self.history = [{"turn_ref": "r:P01:T001", "text": "We will discuss a new activity."}]
        self.card = {"card_id": "inquiry-card", "actors": [
            {"actor_id": "a1", "role": "visitor"}, {"actor_id": "a2", "role": "local advisor"}],
            "steps": [{"step_id": "s1", "actor_id": "a1", "action": "request local information", "state": "requested"},
                      {"step_id": "s2", "actor_id": "a2", "action": "provide local information", "state": "reported"}],
            "dependencies": [{"before": "s1", "after": "s2", "relation": "respond to request"}]}
        self.scene = {"scene_id": "S01", "dataset_card_id": "inquiry-card",
                      "developments": ["Elena asks Milan where to visit on a first reserve visit.",
                                       "Milan answers with a visitor center recommendation."],
                      "card_development_links": [
                          {"card_step_ids": ["s1"], "dependency_indices": [], "development_indices": [1]},
                          {"card_step_ids": ["s2"], "dependency_indices": [1], "development_indices": [2]}]}
        self.response = review_fixture(self.scene, self.history, [self.card])

    def validate(self, response=None):
        return event.validate(self.response if response is None else response, "r", "dataset_new_event",
                              self.scene, self.history, [self.card])

    def test_function_negative_binds_independently_of_all_step_and_global_positives(self):
        self.response["card_checks"][0]["function_check"].update(
            verdict="revise", reason="Authored negative: the interaction serves another purpose and different roles.")
        result = self.validate()
        self.assertTrue(event.requires_replan(result))
        self.assertEqual([row["check_kind"] for row in event.binding_plan_checks(result)], ["card_event_function"])
        self.assertTrue(all(row["verdict"] == "compatible" for row in result["card_checks"][0]["step_checks"]))

    def test_missing_function_or_nonexact_function_evidence_fails_closed(self):
        for mutation in ("missing", "quote"):
            with self.subTest(mutation=mutation):
                response = deepcopy(self.response)
                if mutation == "missing":
                    del response["card_checks"][0]["function_check"]
                else:
                    response["card_checks"][0]["function_check"]["plan_evidence"][0]["quote"] = "invented phrase"
                with self.assertRaises(event.EventStateValidationError):
                    self.validate(response)

    def test_scene_only_fact_cannot_become_evidenceless_historical_constraint(self):
        self.response["timeline"] = [{"constraint_id": "c1", "status": "uncertain", "evidence_basis": "inferred",
                                      "statement": "The new scene introduces a pet.", "implication": "A new detail.", "evidence": []}]
        with self.assertRaises(event.EventStateValidationError) as caught:
            self.validate()
        self.assertIn("event_state_evidence_count_invalid", [row["code"] for row in caught.exception.errors])
        response = review_fixture(self.scene, self.history, [self.card])
        self.assertEqual(self.validate(response)["timeline"], [])
        self.assertFalse(event.requires_replan(response))

    def test_only_plan_review_compatible_revise_enum_and_object_card_members_are_valid(self):
        for mutation in ("enum", "string"):
            with self.subTest(mutation=mutation):
                response = deepcopy(self.response)
                if mutation == "enum":
                    response["consistency_checks"][0]["verdict"] = "not_applicable"
                else:
                    response["card_checks"].append("For each explicit step return one object")
                with self.assertRaises(event.EventStateValidationError):
                    self.validate(response)

    def test_request_contract_enumerates_actual_shape_without_inserting_prose_array_members(self):
        request = event.build_request("r", "dataset_new_event", self.scene, self.history, [self.card], {})
        self.assertEqual(request["max_tokens"], 4096)
        content = request["messages"][0]["content"]
        context, _ = json.JSONDecoder().raw_decode(content.split("EVENT_STATE_INPUT_JSON\n", 1)[1])
        contract = context["output_contract"]
        self.assertEqual(contract["card_checks"]["length"], 1)
        self.assertEqual(contract["card_checks"]["items"][0]["step_ids"], ["s1", "s2"])
        self.assertEqual(contract["all_plan_review_verdict_values"], ["compatible", "revise"])
        self.assertEqual(context["cumulative_history"], self.history)

    def scene_fixture(self):
        f = scene_fixtures.SceneConsistencyContractTests(); f.setUp()
        card = {"card_id": "booking-card", "steps": [{"step_id": "s1"}], "dependencies": []}
        review = f.review()
        review["card_checks"] = [
            {"kind": "event_function", "id": "booking-card", "realized": True,
             "reason": "Authored comparison: the card and actual dialogue both perform a new booking.", "evidence": [f.span(1), f.span(3)]},
            {"kind": "step", "id": "s1", "realized": True, "evidence": [f.span(3)]}]
        return f, card, review

    def test_scene_function_negative_rejects_with_positive_steps_and_aggregate_flags(self):
        f, card, review = self.scene_fixture()
        review["card_checks"][0].update(realized=False, reason="The dialogue only rehearses a different event.")
        result = scene_review.validate_review(review, f.root, f.route, f.scene, f.realization, history=[], dataset_card=card)
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["status"], "REJECT")
        self.assertIn("card_event_function[booking-card]:not_realized", "\n".join(result["failures"]))

    def test_scene_function_requires_coverage_reason_and_exact_new_dialogue_quote(self):
        for mutation in ("missing", "reason", "quote"):
            with self.subTest(mutation=mutation):
                f, card, review = self.scene_fixture()
                if mutation == "missing": review["card_checks"].pop(0)
                elif mutation == "reason": del review["card_checks"][0]["reason"]
                else: review["card_checks"][0]["evidence"][0]["quote"] = "invented"
                result = scene_review.validate_review(review, f.root, f.route, f.scene, f.realization, history=[], dataset_card=card)
                self.assertFalse(result["schema_valid"])
                self.assertEqual(result["status"], "REJECT")

    def test_authored_positive_with_literal_evidence_stays_uncertified(self):
        f, card, review = self.scene_fixture()
        result = scene_review.validate_review(review, f.root, f.route, f.scene, f.realization, history=[], dataset_card=card)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["semantic_certified"])
        self.assertEqual(result["parsed"], review)

    def test_historical_model_accepted_substitution_lacks_new_function_check_and_is_not_auto_upgraded(self):
        base = Path(__file__).resolve().parents[2] / "h4_controlled_six_root_expansion_v95/expansion_engine/runs/six_roots_B_scene01/requests"
        folder = next(base.glob("*24719*event-state*"))
        raw = json.loads((folder / "RESPONSE.json").read_text())["text"]
        request = json.loads((folder / "REQUEST.json").read_text())
        context, _ = json.JSONDecoder().raw_decode(request["messages"][0]["content"].split("EVENT_STATE_INPUT_JSON\n", 1)[1])
        before = raw.encode()
        with self.assertRaises(event.EventStateValidationError):
            event.validate(raw, context["root_id"], context["route"], context["scene_plan"],
                           context["cumulative_history"], context["dataset_cards"])
        self.assertEqual(raw.encode(), before)


if __name__ == "__main__": unittest.main()
