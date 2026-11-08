import copy
import json
import unittest

from realization_review import REVIEW_MODEL, build_review_prompt, validate_review


class RealizationReviewTests(unittest.TestCase):
    def setUp(self):
        self.root = "103866"
        self.route = "dataset_new_event"
        self.scene = {
            "scene_id": "S01", "actors": ["Emily", "James"],
            "developments": [
                "Emily accepts one night's accommodation at James's home.",
                "They agree she will arrive at 10pm after the children are asleep.",
            ],
        }
        self.realization = {
            "root_id": self.root, "route": self.route, "scene_id": "S01",
            "status": "scene", "turns": [
                {"speaker": "Emily", "text": "I accept your offer to stay one night."},
                {"speaker": "James", "text": "Come at 10pm, after the children are asleep."},
                {"speaker": "Emily", "text": "Agreed. I'll arrive at 10pm."},
            ],
        }
        self.review = {
            "root_id": self.root, "route": self.route, "scene_id": "S01",
            "checks": [
                {"development_index": 1, "realized": True,
                 "evidence": [{"turn_index": 1, "quote": "I accept your offer to stay one night."}],
                 "reason": "Emily accepts the offer; this is a decision, not claimed execution."},
                {"development_index": 2, "realized": True,
                 "evidence": [
                     {"turn_index": 2, "quote": "Come at 10pm, after the children are asleep."},
                     {"turn_index": 3, "quote": "Agreed. I'll arrive at 10pm."}],
                 "reason": "James specifies the time and condition and Emily agrees."},
            ],
            "new_event_or_progress": True, "consistent_with_history": True,
            "duplicates_existing_event": False, "reason": "The new arrangements are made explicitly.",
        }

    def validate(self, review=None, realization=None):
        return validate_review(self.review if review is None else review, self.root,
                               self.route, self.scene,
                               self.realization if realization is None else realization)

    def assertReject(self, review=None, realization=None, code=None):
        result = self.validate(review, realization)
        self.assertEqual(result["status"], "REJECT")
        if code:
            self.assertTrue(any(code in item for item in result["failures"]), result)
        return result

    def test_complete_supported_review_passes(self):
        result = self.validate(json.dumps(self.review))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["review_model"], "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["parsed"], self.review)

    def test_missing_development_rejected(self):
        self.review["checks"].pop()
        self.assertReject(code="coverage_invalid")

    def test_duplicate_development_rejected(self):
        self.review["checks"][1]["development_index"] = 1
        self.assertReject(code="coverage_invalid")

    def test_out_of_order_complete_checks_allowed(self):
        self.review["checks"].reverse()
        self.assertEqual(self.validate()["status"], "PASS")

    def test_false_development_cannot_pass_with_positive_top_level_flags(self):
        check = self.review["checks"][1]
        check.update(realized=False, evidence=[], reason="Arrival time is never agreed.")
        self.assertReject(code="not_realized: Arrival time is never agreed.")

    def test_empty_evidence_rejected_for_realized_step(self):
        self.review["checks"][0]["evidence"] = []
        self.assertReject(code="realized_without_evidence")

    def test_empty_and_fabricated_quotes_rejected(self):
        for quote in ("", "  ", "Emily already stayed there last night.", "I accept...one night."):
            with self.subTest(quote=quote):
                review = copy.deepcopy(self.review)
                review["checks"][0]["evidence"][0]["quote"] = quote
                self.assertReject(review, code="quote_not_exact")

    def test_quote_bound_to_exact_new_turn(self):
        self.review["checks"][0]["evidence"][0]["turn_index"] = 2
        self.assertReject(code="quote_not_exact")

    def test_invalid_evidence_turn_indexes(self):
        for value in (0, 4, True, "1", 1.0):
            with self.subTest(value=value):
                review = copy.deepcopy(self.review)
                review["checks"][0]["evidence"][0]["turn_index"] = value
                self.assertReject(review, code="turn_index_invalid")

    def test_identity_mismatch_rejected(self):
        for key in ("root_id", "route", "scene_id"):
            with self.subTest(key=key):
                review = copy.deepcopy(self.review)
                review[key] = "different"
                self.assertReject(review, code=f"review_identity_mismatch:{key}")

    def test_realization_identity_mismatch_rejected(self):
        realization = copy.deepcopy(self.realization)
        realization["scene_id"] = "S99"
        self.assertReject(realization=realization, code="realization_identity_mismatch")

    def test_boolean_verdicts_must_be_actual_booleans(self):
        for key in ("new_event_or_progress", "consistent_with_history", "duplicates_existing_event"):
            for value in ("true", "false", 0, 1, None):
                with self.subTest(key=key, value=value):
                    review = copy.deepcopy(self.review)
                    review[key] = value
                    self.assertReject(review, code=f"review_boolean_invalid:{key}")
        for value in ("true", 1, None):
            review = copy.deepcopy(self.review)
            review["checks"][0]["realized"] = value
            self.assertReject(review, code="realized_not_boolean")

    def test_negative_top_level_verdict_rejected(self):
        for key, value in (("new_event_or_progress", False),
                           ("consistent_with_history", False),
                           ("duplicates_existing_event", True)):
            with self.subTest(key=key):
                review = copy.deepcopy(self.review)
                review[key] = value
                self.assertReject(review, code=f"review_verdict_reject:{key}")

    def test_invalid_review_json_rejected(self):
        for value in ("{bad", "[]", [], 0, "null"):
            with self.subTest(value=value):
                self.assertReject(value, code="review_json_invalid")

    def test_duplicate_json_keys_cannot_overwrite_a_negative_verdict(self):
        raw = json.dumps(self.review).replace(
            '"new_event_or_progress": true',
            '"new_event_or_progress": false, "new_event_or_progress": true',
        )
        self.assertReject(raw, code="review_duplicate_json_key")
        raw = json.dumps(self.review).replace(
            '"realized": true', '"realized": false, "realized": true', 1,
        )
        self.assertReject(raw, code="review_duplicate_json_key")

    def test_single_markdown_wrapper_is_lossless(self):
        for wrapper in ("```json\n", "```\n"):
            self.assertEqual(self.validate(wrapper + json.dumps(self.review) + "\n```")["status"], "PASS")

    def test_no_scene_or_no_developments_cannot_pass(self):
        realization = copy.deepcopy(self.realization)
        realization["status"] = "no_scene"
        self.assertReject(realization=realization, code="realization_not_scene")
        self.scene["developments"] = []
        self.assertReject(code="scene_developments_invalid")

    def test_mock_103866_reassurance_without_lodging_progress_rejected(self):
        # Reproduces the observed failure pattern, not a claim of live review:
        # old reassurance is repeated while the planned lodging arrangement is absent.
        realization = copy.deepcopy(self.realization)
        realization["turns"] = [
            {"speaker": "Emily", "text": "I'll keep a low profile."},
            {"speaker": "James", "text": "Stay safe. I'm here for you."},
        ]
        review = copy.deepcopy(self.review)
        for check in review["checks"]:
            check.update(realized=False, evidence=[],
                         reason="Only old reassurance is repeated; the lodging arrangement is absent.")
        review.update(new_event_or_progress=False, duplicates_existing_event=True,
                      reason="The new scene repeats the old hiding-and-reassurance event.")
        result = self.assertReject(review, realization, "not_realized")
        self.assertTrue(any("duplicates_existing_event" in x for x in result["failures"]))

    def test_prompt_distinguishes_branch_history_from_review_only_other_branch(self):
        history = [{"speaker": "James", "text": "Frozen old history."}]
        other = [{"speaker": "Emily", "text": "OTHER_BRANCH_REVIEW_ONLY_SENTINEL"}]
        prompt = build_review_prompt(self.root, self.route, self.scene, history,
                                     self.realization, other)
        payload = json.loads(prompt.split("REVIEW INPUT DATA:\n", 1)[1])
        self.assertEqual(payload["branch_history"], history)
        self.assertEqual(payload["other_branch_added_turns_for_review_only"], other)
        self.assertNotIn("OTHER_BRANCH_REVIEW_ONLY_SENTINEL", json.dumps(payload["branch_history"]))
        self.assertIn("not evidence that an action already happened", prompt)
        self.assertIn("Do not judge anticipated question difficulty", prompt)
        self.assertIn("same Qwen model", prompt)
        self.assertEqual(REVIEW_MODEL, "Qwen/Qwen3.6-35B-A3B")


if __name__ == "__main__":
    unittest.main()
