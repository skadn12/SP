"""Mechanical review-contract regression tests; no semantic/model certification."""

from copy import deepcopy
import json
import unittest

from realization_review import (
    CONSISTENCY_CHECK_IDS, build_review_prompt, validate_review,
)


class SceneConsistencyContractTests(unittest.TestCase):
    def setUp(self):
        self.root = "25652"
        self.route = "B"
        self.scene = {"scene_id": "S01", "developments": ["Arrange the new booking."]}
        self.realization = {
            "root_id": self.root, "route": self.route, "scene_id": "S01",
            "status": "scene", "turns": [
                {"speaker": "Milan", "text": "Shall I book the table for 12:30?"},
                {"speaker": "Elena", "text": "Yes, please book it for 12:30."},
                {"speaker": "Milan", "text": "All set. I have secured the table for 12:30."},
            ],
        }

    def span(self, number, quote=None):
        return {"turn_ref": f"NEW:T{number:03d}",
                "quote": quote or self.realization["turns"][number - 1]["text"]}

    def review(self):
        return {
            "root_id": self.root, "route": self.route, "scene_id": "S01",
            "checks": [{"development_index": 1, "realized": True,
                        "evidence": [self.span(3)]}],
            "new_event_or_progress": True, "consistent_with_history": True,
            "duplicates_existing_event": False, "reason": "A new table is booked.",
            "consistency_checks": [
                {"check_id": "participant_identity", "verdict": "supported",
                 "reason": "Milan offers the booking to Elena and she authorizes it.",
                 "evidence": [self.span(1), self.span(2)]},
                {"check_id": "itinerary_and_time", "verdict": "not_applicable",
                 "reason": "No travel itinerary is asserted.", "evidence": []},
                {"check_id": "action_status_progression", "verdict": "supported",
                 "reason": "Offer, authorization, and explicit new confirmation occur in order.",
                 "evidence": [self.span(1), self.span(2), self.span(3)]},
            ],
        }

    def validate(self, review):
        return validate_review(review, self.root, self.route, self.scene,
                               self.realization, history=[])

    def check(self, review, identifier):
        return next(item for item in review["consistency_checks"]
                    if item["check_id"] == identifier)

    def test_explicit_new_booking_progression_does_not_require_old_completion(self):
        review = self.review()
        result = self.validate(json.dumps(review))
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["schema_valid"])
        self.assertFalse(result["semantic_certified"])
        self.assertEqual(result["scene_consistency_status"], "MODEL_REPORTED_COHERENT")
        self.assertEqual(result["parsed"], review)
        self.assertEqual(result["scene_consistency_checks"], review["consistency_checks"])

    def test_participant_contradiction_rejects_despite_all_positive_flags(self):
        self.realization["turns"][0] = {
            "speaker": "Elena", "text": "Sam, Sam mentioned you wanted lunch."}
        review = self.review()
        check = self.check(review, "participant_identity")
        check.update(verdict="contradicted", reason="Elena addresses Sam while attributing his need to an unexplained third Sam.",
                     evidence=[self.span(1)])
        result = self.validate(review)
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["status"], "REJECT")
        self.assertIn("scene_consistency[participant_identity]:contradicted:",
                      "\n".join(result["failures"]))
        self.assertEqual(result["scene_consistency_status"], "MODEL_REPORTED_CONFLICT")

    def test_travel_direction_and_return_day_contradictions_cite_opposing_turns(self):
        self.realization["turns"] = [
            {"speaker": "Elena", "text": "I need to get to Oakhaven this weekend."},
            {"speaker": "Elena", "text": "I need to leave Oakhaven on Saturday morning."},
            {"speaker": "Milan", "text": "You arrive home on Saturday before your Sunday return."},
        ]
        review = self.review()
        check = self.check(review, "itinerary_and_time")
        check.update(verdict="contradicted", reason="The same trip changes destination and return timing without a correction.",
                     evidence=[self.span(1), self.span(2), self.span(3)])
        result = self.validate(review)
        self.assertEqual(result["status"], "REJECT")
        self.assertTrue(result["schema_valid"])
        self.assertIn("scene_consistency[itinerary_and_time]:contradicted:",
                      "\n".join(result["failures"]))

    def test_premature_booking_then_pending_rejects_despite_positive_flags(self):
        self.realization["turns"] = [
            {"speaker": "Emily", "text": "I will call to lock in the booking."},
            {"speaker": "James", "text": "With our celebration booked, we can relax."},
            {"speaker": "Emily", "text": "I will confirm once the reservation is secured."},
        ]
        review = self.review()
        check = self.check(review, "action_status_progression")
        check.update(verdict="contradicted", reason="The same booking is claimed complete between two pending booking claims.",
                     evidence=[self.span(1), self.span(2), self.span(3)])
        result = self.validate(review)
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["status"], "REJECT")
        self.assertIn("scene_consistency[action_status_progression]:contradicted:",
                      "\n".join(result["failures"]))

    def test_omitted_check_or_duplicate_coverage_fails_closed(self):
        for mutation in ("entire_field", "one_check", "duplicate_check"):
            with self.subTest(mutation=mutation):
                review = self.review()
                if mutation == "entire_field":
                    del review["consistency_checks"]
                elif mutation == "one_check":
                    review["consistency_checks"].pop()
                else:
                    review["consistency_checks"][2] = deepcopy(review["consistency_checks"][0])
                result = self.validate(review)
                self.assertEqual(result["status"], "REJECT")
                self.assertFalse(result["schema_valid"])

    def test_invalid_citations_are_not_valid_blocking_contradiction_evidence(self):
        for bad_span in ({"turn_ref": "NEW:T099", "quote": "Invented text"},
                         {"turn_ref": "NEW:T001", "quote": "Milan agreed"},
                         {"turn_ref": "OLD:T001", "quote": self.realization["turns"][0]["text"]},
                         {"turn_ref": "NEW:T001", "quote": ""}):
            with self.subTest(bad_span=bad_span):
                review = self.review()
                check = self.check(review, "participant_identity")
                check.update(verdict="contradicted", reason="An alleged identity contradiction.",
                             evidence=[bad_span])
                result = self.validate(review)
                self.assertFalse(result["schema_valid"])
                self.assertEqual(result["status"], "REJECT")
                self.assertFalse(any(":contradicted:" in failure
                                     for failure in result["failures"]))

    def test_opposing_claims_need_two_distinct_exact_spans(self):
        for identifier in ("itinerary_and_time", "action_status_progression"):
            for spans in ([self.span(1)], [self.span(1), self.span(1)]):
                with self.subTest(identifier=identifier, spans=spans):
                    review = self.review()
                    self.check(review, identifier).update(
                        verdict="contradicted", reason="Opposing claims are alleged.", evidence=spans)
                    result = self.validate(review)
                    self.assertFalse(result["schema_valid"])
                    self.assertEqual(result["status"], "REJECT")

    def test_supported_and_not_applicable_cannot_omit_required_evidence_or_reason(self):
        for mutation in ("supported_empty", "not_applicable_with_quote", "reason_empty", "unknown_verdict"):
            with self.subTest(mutation=mutation):
                review = self.review()
                check = self.check(review, "participant_identity")
                if mutation == "supported_empty":
                    check["evidence"] = []
                elif mutation == "not_applicable_with_quote":
                    check["verdict"] = "not_applicable"
                elif mutation == "reason_empty":
                    check["reason"] = ""
                else:
                    check["verdict"] = "consistent"
                result = self.validate(review)
                self.assertEqual(result["status"], "REJECT")
                self.assertFalse(result["schema_valid"])

    def test_prompt_includes_entire_new_scene_and_distinguishes_plan_from_claim(self):
        plan = deepcopy(self.scene)
        plan["developments"] = ["Celebrate Elena and Sam's romantic relationship."]
        self.realization["turns"][-1]["text"] = "Since we are celebrating, lunch sounds good."
        prompt = build_review_prompt(self.root, self.route, plan, [], self.realization)
        payload = json.loads(prompt.split("\nREVIEW INPUT DATA:\n", 1)[1])
        self.assertEqual(len(payload["new_scene"]["turns"]), 3)
        self.assertEqual(payload["new_scene"]["turns"][-1]["turn_ref"], "NEW:T003")
        self.assertIn(self.realization["turns"][-1]["text"], prompt)
        for identifier in CONSISTENCY_CHECK_IDS:
            self.assertIn(identifier, prompt)
        self.assertIn("it does NOT prove that the dialogue asserts", prompt)
        self.assertIn("the booking\nneed not already have happened", prompt)
        self.assertIn("Silence in history is not a contradiction", prompt)

    def test_exact_quote_pass_does_not_certify_model_semantics(self):
        # A fabricated positive judgment with exact quotes passes the mechanical
        # contract. It must retain the substantive limitation explicitly.
        self.realization["turns"][0]["text"] = "I will book it later."
        self.realization["turns"][2]["text"] = "It is booked, though we still need to book it."
        result = self.validate(self.review())
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["semantic_certified"])
        self.assertEqual(result["acceptance_basis"], "same_model_review_not_semantic_certification")


if __name__ == "__main__":
    unittest.main()
