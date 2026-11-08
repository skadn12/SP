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

    def validate(self, review=None, realization=None, *, dataset_card=None):
        return validate_review(self.review if review is None else review, self.root,
                               self.route, self.scene,
                               self.realization if realization is None else realization,
                               dataset_card=dataset_card)

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


class DatasetCardReviewTests(unittest.TestCase):
    """The card is checked independently of a structurally complete scene plan."""

    validate = RealizationReviewTests.validate

    def setUp(self):
        RealizationReviewTests.setUp(self)
        self.card = {
            "card_id": "lodging_card", "guidance_level": "scaffold",
            "steps": [
                {"step_id": "s1", "action": "accept an offer of one night's accommodation",
                 "state": "reported_action"},
                {"step_id": "s2", "action": "agree arrival after the children sleep",
                 "state": "proposed"},
            ],
            "dependencies": [{"before": "s1", "after": "s2", "relation": "order"}],
        }

    def card_review(self):
        review = copy.deepcopy(self.review)
        review["card_checks"] = [
            {"kind": "step", "id": "s1", "realized": True,
             "evidence": [{"turn_index": 1, "quote": "I accept your offer to stay one night."}],
             "reason": "The guest accepts the one-night lodging offer."},
            {"kind": "step", "id": "s2", "realized": True,
             "evidence": [
                 {"turn_index": 2, "quote": "Come at 10pm, after the children are asleep."},
                 {"turn_index": 3, "quote": "Agreed. I'll arrive at 10pm."}],
             "reason": "The arrival time and condition are agreed, without claiming arrival occurred."},
            {"kind": "dependency", "id": 1, "realized": True,
             "evidence": [
                 {"turn_index": 1, "quote": "I accept your offer to stay one night."},
                 {"turn_index": 3, "quote": "Agreed. I'll arrive at 10pm."}],
             "reason": "Acceptance precedes the agreement on arrival details."},
        ]
        return review

    def assertCardSchemaReject(self, review, code):
        result = self.validate(review, dataset_card=self.card)
        self.assertEqual(result["status"], "REJECT")
        self.assertFalse(result["schema_valid"], result)
        self.assertTrue(any(code in item for item in result["schema_failures"]), result)
        return result

    def test_card_checks_complete_with_exact_quotes_pass(self):
        result = self.validate(json.dumps(self.card_review()), dataset_card=self.card)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["schema_failures"], [])
        self.assertEqual(result["acceptance_basis"], "same_model_review_not_semantic_certification")

    def test_negative_card_check_rejects_despite_all_developments_and_overall_flags(self):
        for index, prefix in ((0, "card_step[s1]"), (2, "card_dependency[1]")):
            with self.subTest(prefix=prefix):
                review = self.card_review()
                review["card_checks"][index].update(
                    realized=False, evidence=[], reason="The required behavior or relation is absent.")
                result = self.validate(review, dataset_card=self.card)
                self.assertEqual(result["status"], "REJECT")
                self.assertTrue(result["schema_valid"], result)
                self.assertEqual(result["schema_failures"], [])
                self.assertTrue(any(item.startswith(prefix + ":not_realized:")
                                    for item in result["failures"]), result)

    def test_card_schema_is_required_only_when_card_supplied(self):
        self.assertCardSchemaReject(self.review, "review_card_checks_not_array")
        result = self.validate(self.card_review())
        self.assertEqual(result["status"], "REJECT")
        self.assertIn("review_fields_invalid", result["schema_failures"])
        self.route = "continuation"
        self.review["route"] = self.route
        self.realization["route"] = self.route
        self.assertEqual(self.validate()["status"], "PASS")

    def test_missing_or_duplicate_card_checks_rejected(self):
        missing_step = self.card_review()
        missing_step["card_checks"].pop(0)
        missing_dependency = self.card_review()
        missing_dependency["card_checks"].pop()
        duplicate = self.card_review()
        duplicate["card_checks"].append(copy.deepcopy(duplicate["card_checks"][0]))
        for review in (missing_step, missing_dependency, duplicate):
            self.assertCardSchemaReject(review, "review_card_coverage_invalid")

    def test_card_checks_may_be_out_of_order(self):
        review = self.card_review()
        review["card_checks"].reverse()
        self.assertEqual(self.validate(review, dataset_card=self.card)["status"], "PASS")

    def test_card_references_must_use_exact_step_id_or_one_based_integer_dependency(self):
        for index, identifier in ((0, "S1"), (0, 1), (0, []), (0, True),
                                  (2, 0), (2, 2), (2, "1"), (2, True), (2, 1.0)):
            with self.subTest(index=index, identifier=identifier):
                review = self.card_review()
                review["card_checks"][index]["id"] = identifier
                self.assertCardSchemaReject(review, "card_reference_invalid")

    def test_unknown_card_check_kind_is_schema_failure(self):
        for kind in ("development", [], None):
            review = self.card_review()
            review["card_checks"][0]["kind"] = kind
            self.assertCardSchemaReject(review, "card_reference_invalid")

    def test_card_false_is_valid_but_invalid_booleans_are_not(self):
        for value in ("false", "true", 1, 0, None):
            review = self.card_review()
            review["card_checks"][0]["realized"] = value
            self.assertCardSchemaReject(review, "realized_not_boolean")

    def test_card_evidence_must_be_nonempty_exact_and_bound_to_new_turn(self):
        for evidence, code in (
            ([], "realized_without_evidence"),
            (None, "evidence_not_array"),
            ([{"turn_index": 1, "quote": ""}], "quote_not_exact"),
            ([{"turn_index": 2, "quote": "I accept your offer to stay one night."}], "quote_not_exact"),
            ([{"turn_index": 1, "quote": "I accept...one night."}], "quote_not_exact"),
            ([{"turn_index": True, "quote": "I accept"}], "turn_index_invalid"),
            ([{"turn_index": 0, "quote": "I accept"}], "turn_index_invalid"),
            ([{"turn_index": 4, "quote": "I accept"}], "turn_index_invalid"),
            ([{"turn_index": 1, "quote": "I accept", "source": "history"}], "fields_invalid"),
        ):
            with self.subTest(evidence=evidence):
                review = self.card_review()
                review["card_checks"][0]["evidence"] = evidence
                self.assertCardSchemaReject(review, code)

    def test_card_false_with_fabricated_quote_still_is_schema_failure(self):
        review = self.card_review()
        review["card_checks"][0].update(realized=False,
            evidence=[{"turn_index": 1, "quote": "A fabricated explanation."}])
        result = self.assertCardSchemaReject(review, "quote_not_exact")
        self.assertTrue(any("card_step[s1]:not_realized:" in item for item in result["failures"]))

    def test_card_check_fields_and_reason_required(self):
        for mutation, code in ((lambda check: check.pop("kind"), "fields_invalid"),
                               (lambda check: check.update(reason=" "), "reason_missing")):
            review = self.card_review()
            mutation(review["card_checks"][0])
            self.assertCardSchemaReject(review, code)
        review = self.card_review()
        review["card_checks"][0] = "not an object"
        self.assertCardSchemaReject(review, "not_object")

    def test_seed_checks_only_explicit_steps_without_inventing_dependencies(self):
        self.card.update(guidance_level="seed", dependencies=[])
        review = self.card_review()
        review["card_checks"].pop()
        self.assertEqual(self.validate(review, dataset_card=self.card)["status"], "PASS")

    def test_malformed_card_identity_contract_fails_closed(self):
        for card in ([], {}, {"steps": [], "dependencies": []},
                     {"steps": [{"step_id": 1}], "dependencies": []},
                     {"steps": [{"step_id": "s1"}, {"step_id": "s1"}], "dependencies": []},
                     {"steps": [{"step_id": "s1"}], "dependencies": "not an array"}):
            with self.subTest(card=card):
                result = self.validate(self.card_review(), dataset_card=card)
                self.assertEqual(result["status"], "REJECT")
                self.assertFalse(result["schema_valid"])
                with self.assertRaisesRegex(ValueError, "review_dataset_card_"):
                    build_review_prompt(self.root, self.route, self.scene, [],
                                        self.realization, dataset_card=card)

    def test_prompt_enumerates_exact_card_checks_with_adaptation_and_state_guidance(self):
        prompt = build_review_prompt(self.root, self.route, self.scene, [],
                                     self.realization, dataset_card=self.card)
        schema = json.loads(prompt.split("OUTPUT SHAPE:\n", 1)[1].split("\nREVIEW INPUT DATA:\n", 1)[0])
        self.assertEqual([(item["kind"], item["id"]) for item in schema["card_checks"]],
                         [("step", "s1"), ("step", "s2"), ("dependency", 1)])
        self.assertIn("both\nseed and scaffold", prompt)
        self.assertIn("conditions and action status must survive adaptation", prompt)
        self.assertIn("unrelated continuation of an old\ndonation/course event", prompt)
        self.assertIn("source event happened to this branch's people", prompt)
        self.assertIn("evidence that both steps were mentioned alone does not establish", prompt)
        no_card_prompt = build_review_prompt(self.root, "continuation", self.scene, [], self.realization)
        no_card_schema = json.loads(no_card_prompt.split("OUTPUT SHAPE:\n", 1)[1].split("\nREVIEW INPUT DATA:\n", 1)[0])
        self.assertNotIn("card_checks", no_card_schema)


if __name__ == "__main__":
    unittest.main()
