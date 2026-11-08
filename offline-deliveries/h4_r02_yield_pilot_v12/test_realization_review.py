import copy
import json
from pathlib import Path
import unittest

from realization_review import REVIEW_MODEL, build_review_prompt, validate_review, review_scene


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
                 "evidence": [{"turn_ref": "NEW:T001", "quote": "I accept your offer to stay one night."}],
                 "reason": "Emily accepts the offer; this is a decision, not claimed execution."},
                {"development_index": 2, "realized": True,
                 "evidence": [
                     {"turn_ref": "NEW:T002", "quote": "Come at 10pm, after the children are asleep."},
                     {"turn_ref": "NEW:T003", "quote": "Agreed. I'll arrive at 10pm."}],
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

    def test_optional_check_reasons_can_be_omitted_without_rewriting_review(self):
        for check in self.review["checks"]:
            check.pop("reason")
        original = copy.deepcopy(self.review)
        result = self.validate()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["parsed"], original)
        self.assertEqual(self.review, original)
        self.assertFalse(result["semantic_certified"])

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
        self.review["checks"][0]["evidence"][0]["turn_ref"] = "NEW:T002"
        self.assertReject(code="quote_not_exact")

    def test_invalid_evidence_turn_references(self):
        for value in (0, 4, True, "1", 1.0, "NEW:T1", "NEW:T004", "R01:T001"):
            with self.subTest(value=value):
                review = copy.deepcopy(self.review)
                review["checks"][0]["evidence"][0]["turn_ref"] = value
                self.assertReject(review, code="turn_ref_invalid")

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
             "evidence": [{"turn_ref": "NEW:T001", "quote": "I accept your offer to stay one night."}],
             "reason": "The guest accepts the one-night lodging offer."},
            {"kind": "step", "id": "s2", "realized": True,
             "evidence": [
                 {"turn_ref": "NEW:T002", "quote": "Come at 10pm, after the children are asleep."},
                 {"turn_ref": "NEW:T003", "quote": "Agreed. I'll arrive at 10pm."}],
             "reason": "The arrival time and condition are agreed, without claiming arrival occurred."},
            {"kind": "dependency", "id": 1, "realized": True,
             "evidence": [
                 {"turn_ref": "NEW:T001", "quote": "I accept your offer to stay one night."},
                 {"turn_ref": "NEW:T003", "quote": "Agreed. I'll arrive at 10pm."}],
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

    def test_card_check_reasons_are_optional_but_overall_reason_remains_required(self):
        review = self.card_review()
        for check in review["card_checks"]:
            check.pop("reason")
        self.assertEqual(self.validate(review, dataset_card=self.card)["status"], "PASS")
        review.pop("reason")
        self.assertCardSchemaReject(review, "review_reason_missing")

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
            ([{"turn_ref": "NEW:T001", "quote": ""}], "quote_not_exact"),
            ([{"turn_ref": "NEW:T002", "quote": "I accept your offer to stay one night."}], "quote_not_exact"),
            ([{"turn_ref": "NEW:T001", "quote": "I accept...one night."}], "quote_not_exact"),
            ([{"turn_ref": True, "quote": "I accept"}], "turn_ref_invalid"),
            ([{"turn_ref": "NEW:T000", "quote": "I accept"}], "turn_ref_invalid"),
            ([{"turn_ref": "NEW:T004", "quote": "I accept"}], "turn_ref_invalid"),
            ([{"turn_ref": "NEW:T001", "quote": "I accept", "source": "history"}], "fields_invalid"),
        ):
            with self.subTest(evidence=evidence):
                review = self.card_review()
                review["card_checks"][0]["evidence"] = evidence
                self.assertCardSchemaReject(review, code)

    def test_card_false_with_fabricated_quote_still_is_schema_failure(self):
        review = self.card_review()
        review["card_checks"][0].update(realized=False,
            evidence=[{"turn_ref": "NEW:T001", "quote": "A fabricated explanation."}])
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


class LatestStateReviewTests(unittest.TestCase):
    def setUp(self):
        RealizationReviewTests.setUp(self)
        self.history = [
            {"turn_ref": "P:T001", "speaker": "James",
             "text": "You may stay at my home for one night."},
            {"turn_ref": "R01:S01:T001", "speaker": "Emily",
             "text": "I have not decided where to stay yet."},
        ]
        self.latest_state = {
            "root_id": self.root, "route": self.route, "scene_id": "S01",
            "facts": [{"fact_id": "f1", "entity": "James and Emily",
                       "event": "one-night accommodation", "status": "planned",
                       "evidence_basis": "reported",
                       "statement": "James offered Emily one night's accommodation.",
                       "evidence": [{"turn_ref": "P:T001",
                                     "quote": "You may stay at my home for one night."}]}],
            "conflicts": [], "unknowns": ["Whether Emily accepts."],
        }
        self.review["state_checks"] = [{
            "fact_id": "f1", "verdict": "consistent",
            "history_evidence": copy.deepcopy(self.latest_state["facts"][0]["evidence"]),
            "new_evidence": [{"turn_ref": "NEW:T001", "quote": "I accept your offer to stay one night."}],
            "reason": "The earlier offer becomes newly accepted.",
        }]

    def validate(self, review=None, *, latest_state=None, history=None):
        return validate_review(self.review if review is None else review,
                               self.root, self.route, self.scene, self.realization,
                               latest_state=self.latest_state if latest_state is None else latest_state,
                               history=self.history if history is None else history)

    def assertSchemaReject(self, review, code):
        result = self.validate(review)
        self.assertEqual(result["status"], "REJECT", result)
        self.assertFalse(result["schema_valid"], result)
        self.assertTrue(any(code in reason for reason in result["schema_failures"]), result)
        return result

    def test_newly_established_decision_is_consistent_with_historical_plan(self):
        result = self.validate(json.dumps(self.review))
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["acceptance_basis"], "same_model_review_not_semantic_certification")

    def test_empty_consistent_evidence_is_diagnostic_without_rewriting_verdict(self):
        self.review["state_checks"][0]["new_evidence"] = []
        original = copy.deepcopy(self.review)
        result = self.validate()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["parsed"], original)
        self.assertEqual(self.review, original)
        self.assertFalse(result["semantic_certified"])
        self.assertEqual(result["soft_findings"], [])
        diagnostic = result["empty_consistent_findings"][0]
        self.assertEqual(diagnostic["fact_id"], "f1")
        self.assertEqual(diagnostic["original_check"], original["state_checks"][0])
        self.assertEqual(diagnostic["code"], "consistent_without_new_evidence")

    def test_empty_consistent_evidence_does_not_bypass_history_or_global_negative(self):
        self.review["state_checks"][0]["new_evidence"] = []
        for history_evidence in ([], [{"turn_ref": "P:T001", "quote": "Invented history."}]):
            review = copy.deepcopy(self.review)
            review["state_checks"][0]["history_evidence"] = history_evidence
            self.assertSchemaReject(review, "history_evidence")
        self.review["consistent_with_history"] = False
        result = self.validate()
        self.assertEqual(result["status"], "REJECT")
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["history_consistency_status"], "MODEL_REPORTED_CONFLICT")

    def test_empty_contradiction_stays_schema_failure_for_hard_and_soft_facts(self):
        for basis in ("reported", "inferred", "uncertain"):
            with self.subTest(basis=basis):
                self.latest_state["facts"][0]["evidence_basis"] = basis
                self.review["state_checks"][0].update(verdict="contradicted", new_evidence=[])
                result = self.assertSchemaReject(self.review, "new_evidence_count_invalid")
                self.assertEqual(result["empty_consistent_findings"], [])

    def test_soft_contradiction_is_preserved_as_unresolved_finding_without_hard_rejection(self):
        for basis in ("inferred", "uncertain"):
            for status in ("completed", "ongoing"):
                with self.subTest(basis=basis, status=status):
                    self.latest_state["facts"][0].update(evidence_basis=basis, status=status)
                    self.review["state_checks"][0]["verdict"] = "contradicted"
                    self.review["consistent_with_history"] = False
                    original = copy.deepcopy((self.review, self.latest_state))
                    result = self.validate()
                    self.assertEqual(result["status"], "PASS", result)
                    self.assertEqual(result["history_consistency_status"], "UNRESOLVED_SOFT_FINDINGS")
                    self.assertFalse(result["semantic_certified"])
                    self.assertFalse(result["parsed"]["consistent_with_history"])
                    self.assertEqual(result["parsed"]["state_checks"][0]["verdict"], "contradicted")
                    self.assertEqual(result["soft_findings"][0]["fact_id"], "f1")
                    self.assertEqual(result["soft_findings"][0]["evidence_basis"], basis)
                    self.assertEqual(result["soft_findings"][0]["status"], status)
                    self.assertEqual((self.review, self.latest_state), original)

    def test_soft_findings_cannot_hide_invalid_evidence_or_missing_checks(self):
        self.latest_state["facts"][0]["evidence_basis"] = "inferred"
        self.review["state_checks"][0]["verdict"] = "contradicted"
        self.review["consistent_with_history"] = False
        for mutation in (
            lambda review: review["state_checks"][0]["new_evidence"][0].update(quote="invented"),
            lambda review: review["checks"].pop(),
        ):
            review = copy.deepcopy(self.review)
            mutation(review)
            result = self.validate(review)
            self.assertEqual(result["status"], "REJECT")
            self.assertFalse(result["schema_valid"])
            self.assertTrue(any("review_verdict_reject:consistent_with_history" in item
                                for item in result["failures"]))

    def test_unattributed_aggregate_history_conflict_is_still_rejected(self):
        self.latest_state["facts"][0]["evidence_basis"] = "uncertain"
        self.review["consistent_with_history"] = False
        result = self.validate()
        self.assertEqual(result["status"], "REJECT")
        self.assertEqual(result["soft_findings"], [])
        self.assertEqual(result["history_consistency_status"], "MODEL_REPORTED_CONFLICT")

    def test_state_check_reason_is_optional_and_prompt_omits_redundant_reasons(self):
        self.review["state_checks"][0].pop("reason")
        self.assertEqual(self.validate()["status"], "PASS")
        prompt = build_review_prompt(self.root, self.route, self.scene, self.history,
                                     self.realization, latest_state=self.latest_state)
        schema = json.loads(prompt.split("OUTPUT SHAPE:\n", 1)[1].split("\nREVIEW INPUT DATA:\n", 1)[0])
        self.assertIn("reason", schema)
        for key in ("checks", "state_checks"):
            self.assertTrue(all("reason" not in check for check in schema[key]))
        self.assertIn("inferred or uncertain completion", prompt)

    def test_missing_or_invalid_snapshot_evidence_basis_fails_closed(self):
        for basis in (None, "verified", []):
            self.latest_state["facts"][0]["evidence_basis"] = basis
            result = self.validate()
            self.assertEqual(result["status"], "REJECT")
            self.assertIn("review_latest_state_evidence_basis_invalid:f1", result["schema_failures"])

    def test_state_contract_is_optional_for_legacy_call_but_required_with_snapshot(self):
        legacy = copy.deepcopy(self.review)
        legacy.pop("state_checks")
        result = validate_review(legacy, self.root, self.route, self.scene, self.realization)
        self.assertEqual(result["status"], "PASS")
        self.assertSchemaReject(legacy, "review_state_checks_not_array")
        result = validate_review(self.review, self.root, self.route, self.scene, self.realization)
        self.assertIn("review_fields_invalid", result["schema_failures"])

    def test_structured_negative_rejects_independent_of_global_positive(self):
        self.review["state_checks"][0].update(verdict="contradicted", reason="The state conflicts.")
        result = self.validate()
        self.assertEqual(result["status"], "REJECT")
        self.assertTrue(result["schema_valid"], result)
        self.assertEqual(result["failures"], ["state_fact[f1]:contradicted: The state conflicts."])

    def test_v6_taxi_arrival_replay_model_negative_rejects_but_quotes_do_not_certify_semantics(self):
        # These spans reproduce the v6 taxi failure. The verdict is a mock Qwen
        # response, not an automated semantic inference or a live API result.
        self.history = [{"turn_ref": "R01:S01:T021", "speaker": "Emily",
                         "text": "I'm officially here! Let's make this work."}]
        self.latest_state["facts"][0].update(
            entity="Emily", event="first day at office", status="completed",
            statement="Emily has arrived at her office.",
            evidence=[{"turn_ref": "R01:S01:T021", "quote": "I'm officially here!"}])
        self.realization["turns"] = [{"speaker": "Emily", "text":
            "David, I'm heading out to the office now, but I'm a bit worried about traffic. "
            "Could you help me arrange a taxi to make sure I get there on time for orientation?"}]
        self.scene["developments"] = ["Emily asks for a taxi to orientation."]
        self.review["checks"] = [{"development_index": 1, "realized": True,
                                  "evidence": [{"turn_ref": "NEW:T001", "quote": "arrange a taxi"}],
                                  "reason": "Emily requests a taxi."}]
        self.review["state_checks"] = [{
            "fact_id": "f1", "verdict": "contradicted",
            "history_evidence": [{"turn_ref": "R01:S01:T021", "quote": "I'm officially here!"}],
            "new_evidence": [{"turn_ref": "NEW:T001", "quote": "I'm heading out to the office now"}],
            "reason": "An already completed first arrival is replayed without a new occasion.",
        }]
        result = self.validate()
        self.assertEqual(result["status"], "REJECT")
        self.assertTrue(result["schema_valid"])
        self.assertTrue(any(x.startswith("state_fact[f1]:contradicted:") for x in result["failures"]))
        self.review["state_checks"][0].update(verdict="consistent",
                                             reason="The model mistakenly regards this as consistent.")
        result = self.validate()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["acceptance_basis"], "same_model_review_not_semantic_certification")

    def test_state_coverage_requires_every_fact_exactly_once(self):
        for checks in ([], self.review["state_checks"] * 2):
            review = copy.deepcopy(self.review)
            review["state_checks"] = checks
            self.assertSchemaReject(review, "review_state_coverage_invalid")
        for value in ("f999", [], 1, None):
            review = copy.deepcopy(self.review)
            review["state_checks"][0]["fact_id"] = value
            self.assertSchemaReject(review, "fact_id_invalid")

    def test_state_checks_may_be_out_of_order(self):
        second = copy.deepcopy(self.latest_state["facts"][0])
        second["fact_id"] = "f2"
        self.latest_state["facts"].append(second)
        check = copy.deepcopy(self.review["state_checks"][0])
        check["fact_id"] = "f2"
        self.review["state_checks"].insert(0, check)
        self.assertEqual(self.validate()["status"], "PASS")

    def test_not_relevant_allows_empty_new_evidence_but_requires_history_and_reason(self):
        check = self.review["state_checks"][0]
        check.update(verdict="not_relevant", new_evidence=[], reason="This fact concerns an unrelated event.")
        self.assertEqual(self.validate()["status"], "PASS")
        review = copy.deepcopy(self.review)
        review["state_checks"][0]["history_evidence"] = []
        self.assertSchemaReject(review, "history_evidence_count_invalid")
        review = copy.deepcopy(self.review)
        review["state_checks"][0]["reason"] = " "
        self.assertSchemaReject(review, "reason_missing")

    def test_each_history_citation_is_exact_own_history_and_at_least_one_binds_fact(self):
        bad_evidence = (
            ([{"turn_ref": "R02:A:S01:T001", "quote": "You may stay at my home for one night."}], "turn_ref_invalid"),
            ([{"turn_ref": "P:T001", "quote": "I have not decided where to stay yet."}], "quote_not_exact"),
            ([{"turn_ref": "P:T001", "quote": "You may...one night."}], "quote_not_exact"),
            ([{"turn_ref": "P:T001", "quote": " "}], "quote_not_exact"),
            ([{"turn_ref": "R01:S01:T001", "quote": "I have not decided where to stay yet."}], "missing_fact_reference"),
            ([{"turn_ref": [], "quote": "stay"}], "turn_ref_invalid"),
            ([{"turn_index": 1, "quote": "stay"}], "fields_invalid"),
            (None, "history_evidence_not_array"),
        )
        for evidence, code in bad_evidence:
            with self.subTest(evidence=evidence):
                review = copy.deepcopy(self.review)
                review["state_checks"][0]["history_evidence"] = evidence
                self.assertSchemaReject(review, code)

    def test_additional_later_history_can_clarify_conflicted_fact(self):
        self.latest_state["conflicts"] = [{"fact_ids": ["f1"], "reason": "Decision is still pending."}]
        self.review["state_checks"][0]["history_evidence"].append(
            {"turn_ref": "R01:S01:T001", "quote": "I have not decided where to stay yet."})
        self.assertEqual(self.validate()["status"], "PASS")
        # No invented hard rejection merely because a fallible snapshot has conflicts.

    def test_state_evidence_array_lengths_and_new_turn_binding(self):
        for field in ("history_evidence", "new_evidence"):
            review = copy.deepcopy(self.review)
            review["state_checks"][0][field] *= 4
            self.assertSchemaReject(review, f"{field}_count_invalid")
        for value in (None, [{"turn_ref": True, "quote": "I accept"}],
                      [{"turn_ref": "NEW:T000", "quote": "I accept"}],
                      [{"turn_ref": "NEW:T002", "quote": "I accept"}],
                      [{"turn_ref": "NEW:T001", "quote": ""}],
                      [{"turn_ref": "NEW:T001", "quote": "I accept", "extra": True}]):
            review = copy.deepcopy(self.review)
            review["state_checks"][0]["new_evidence"] = value
            self.assertSchemaReject(review, "new_evidence")

    def test_invalid_state_verdicts_fields_and_reason_are_schema_errors(self):
        for verdict in (True, 1, [], "PASS", "false", None):
            review = copy.deepcopy(self.review)
            review["state_checks"][0]["verdict"] = verdict
            self.assertSchemaReject(review, "verdict_invalid")
        for mutation, code in ((lambda item: item.update(extra=True), "fields_invalid"),
                               (lambda item: item.update(reason=""), "reason_missing")):
            review = copy.deepcopy(self.review)
            mutation(review["state_checks"][0])
            self.assertSchemaReject(review, code)
        for checks, code in ((None, "checks_not_array"), (["bad"], "not_object")):
            review = copy.deepcopy(self.review)
            review["state_checks"] = checks
            self.assertSchemaReject(review, code)

    def test_history_must_be_supplied_and_unambiguous(self):
        for history in (None, {}, [{"text": "No ref"}], self.history * 2):
            result = validate_review(self.review, self.root, self.route, self.scene,
                                     self.realization, latest_state=self.latest_state, history=history)
            self.assertEqual(result["status"], "REJECT")
            self.assertFalse(result["schema_valid"])
            self.assertTrue(any("review_state_history_" in x for x in result["schema_failures"]))

    def test_malformed_snapshot_and_fabricated_fact_citations_fail_closed(self):
        for snapshot in ([], {}, dict(self.latest_state, facts=None),
                         dict(self.latest_state, root_id="other"),
                         dict(self.latest_state, facts=self.latest_state["facts"] * 2),
                         dict(self.latest_state, facts=[{"fact_id": "f1", "evidence": []}])):
            result = self.validate(latest_state=snapshot)
            self.assertEqual(result["status"], "REJECT")
            self.assertFalse(result["schema_valid"])
        self.latest_state["facts"][0]["evidence"][0]["quote"] = "Invented historical evidence."
        result = self.validate()
        self.assertFalse(result["schema_valid"])
        self.assertIn("review_latest_state_fact_quote_not_exact:f1", result["schema_failures"])

    def test_state_and_card_checks_both_remain_required(self):
        card = {"steps": [{"step_id": "s1"}], "dependencies": []}
        review = copy.deepcopy(self.review)
        review["card_checks"] = [{"kind": "step", "id": "s1", "realized": True,
                                 "evidence": [{"turn_ref": "NEW:T001", "quote": "I accept"}],
                                 "reason": "The offer is accepted."}]
        result = validate_review(review, self.root, self.route, self.scene, self.realization,
                                 dataset_card=card, latest_state=self.latest_state, history=self.history)
        self.assertEqual(result["status"], "PASS")
        review["state_checks"][0]["verdict"] = "contradicted"
        result = validate_review(review, self.root, self.route, self.scene, self.realization,
                                 dataset_card=card, latest_state=self.latest_state, history=self.history)
        self.assertEqual(result["status"], "REJECT")
        self.assertTrue(result["schema_valid"])

    def test_prompt_includes_fallible_state_full_history_targeted_roles_and_complete_shape(self):
        other = [{"turn_ref": "R02:A:T001", "text": "OTHER_BRANCH_ONLY"}]
        prompt = build_review_prompt(self.root, self.route, self.scene, self.history,
                                     self.realization, other, latest_state=self.latest_state)
        payload = json.loads(prompt.split("REVIEW INPUT DATA:\n", 1)[1])
        self.assertEqual(payload["latest_state"], self.latest_state)
        self.assertEqual(payload["branch_history"], self.history)
        self.assertNotIn("OTHER_BRANCH_ONLY", json.dumps(payload["branch_history"]))
        schema = json.loads(prompt.split("OUTPUT SHAPE:\n", 1)[1].split("\nREVIEW INPUT DATA:\n", 1)[0])
        self.assertEqual([x["fact_id"] for x in schema["state_checks"]], ["f1"])
        for phrase in ("fallible extraction", "full branch_history remains\nauthoritative",
                       "actor versus recipient", "origin/pickup versus destination", "planned versus actual",
                       "completed actions", "repeated FIRST occurrence", "conflicts and unknowns",
                       "coherently established as a NEW event", "shortest sufficient exact quote"):
            self.assertIn(phrase, prompt)
        legacy = build_review_prompt(self.root, self.route, self.scene, self.history, self.realization)
        legacy_payload = json.loads(legacy.split("REVIEW INPUT DATA:\n", 1)[1])
        self.assertNotIn("latest_state", legacy_payload)


class ExplicitNewTurnReferenceTests(unittest.TestCase):
    def setUp(self):
        LatestStateReviewTests.setUp(self)

    def test_review_view_adds_refs_without_mutating_original(self):
        before = copy.deepcopy(self.realization)
        view = review_scene(self.realization)
        self.assertEqual([turn["turn_ref"] for turn in view["turns"]],
                         ["NEW:T001", "NEW:T002", "NEW:T003"])
        self.assertEqual(self.realization, before)
        self.assertNotIn("turn_ref", self.realization["turns"][0])
        self.assertEqual(review_scene(view), view)
        prompt = build_review_prompt(self.root, self.route, self.scene, self.history,
                                     self.realization, latest_state=self.latest_state)
        payload = json.loads(prompt.split("REVIEW INPUT DATA:\n", 1)[1])
        self.assertEqual(payload["new_scene"], view)
        self.assertEqual(self.realization, before)
        result = validate_review(self.review, self.root, self.route, self.scene, view,
                                 latest_state=self.latest_state, history=self.history)
        self.assertEqual(result["status"], "PASS")

    def test_wrong_program_turn_ref_is_not_repaired(self):
        view = review_scene(self.realization)
        view["turns"][0]["turn_ref"] = "NEW:T002"
        with self.assertRaisesRegex(ValueError, "turn_ref_mismatch"):
            review_scene(view)
        result = validate_review(self.review, self.root, self.route, self.scene, view,
                                 latest_state=self.latest_state, history=self.history)
        self.assertFalse(result["schema_valid"])
        self.assertIn("review_new_scene_turn_ref_mismatch", result["failures"])
        self.assertEqual(view["turns"][0]["turn_ref"], "NEW:T002")

    def test_old_turn_index_cannot_be_silently_converted(self):
        self.review["checks"][0]["evidence"] = [{"turn_index": 1, "quote": "I accept"}]
        result = validate_review(self.review, self.root, self.route, self.scene, self.realization,
                                 latest_state=self.latest_state, history=self.history)
        self.assertFalse(result["schema_valid"])
        self.assertIn("development[1]:evidence[1]:fields_invalid", result["schema_failures"])
        self.assertIn("turn_index", self.review["checks"][0]["evidence"][0])

    def test_silence_about_unrelated_fact_uses_not_relevant_without_forced_repetition(self):
        self.review["state_checks"][0].update(verdict="not_relevant", new_evidence=[],
                                             reason="This old event does not constrain the new scene.")
        result = validate_review(self.review, self.root, self.route, self.scene, self.realization,
                                 latest_state=self.latest_state, history=self.history)
        self.assertEqual(result["status"], "PASS")
        self.review["state_checks"][0]["verdict"] = "consistent"
        result = validate_review(self.review, self.root, self.route, self.scene, self.realization,
                                 latest_state=self.latest_state, history=self.history)
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["empty_consistent_findings"][0]["fact_id"], "f1")
        prompt = build_review_prompt(self.root, self.route, self.scene, self.history,
                                     self.realization, latest_state=self.latest_state)
        self.assertIn("Silence about an\nold fact is not by itself a contradiction", prompt)
        self.assertIn("repeat every historical fact", prompt)


class EventStateReviewTests(unittest.TestCase):
    def setUp(self):
        LatestStateReviewTests.setUp(self)
        self.event_state = {
            "root_id": self.root, "route": self.route, "scene_id": "S01",
            "segments": [{"segment_id": "fixture", "relevant": True,
                          "reason": "The offer constrains this accommodation arrangement."}],
            "timeline": [{"constraint_id": "c1", "status": "planned", "evidence_basis": "reported",
                          "statement": "James offered Emily one night of accommodation.",
                          "evidence": copy.deepcopy(self.latest_state["facts"][0]["evidence"]),
                          "implication": "Emily may accept this offer as a new decision.",
                          "uncertainty": "Her arrival has not happened."}],
            "plan_checks": [{"development_index": 1, "verdict": "compatible",
                             "constraint_ids": ["c1"], "reason": "The offer may be accepted."},
                            {"development_index": 2, "verdict": "compatible",
                             "constraint_ids": ["c1"], "reason": "They may agree a future arrival."}],
        }
        check = copy.deepcopy(self.review["state_checks"][0])
        check["constraint_id"] = "c1"
        check.pop("fact_id")
        self.review["event_checks"] = [check]

    def validate(self, review=None, *, event_state=None, latest_state=True):
        return validate_review(self.review if review is None else review, self.root,
                               self.route, self.scene, self.realization,
                               latest_state=self.latest_state if latest_state else None,
                               event_state=self.event_state if event_state is None else event_state,
                               history=self.history)

    def test_event_and_latest_state_coexist_with_distinct_reference_spaces(self):
        self.assertEqual(self.validate()["status"], "PASS")
        prompt = build_review_prompt(self.root, self.route, self.scene, self.history,
                                     self.realization, latest_state=self.latest_state,
                                     event_state=self.event_state)
        schema = json.loads(prompt.split("OUTPUT SHAPE:\n", 1)[1].split("\nREVIEW INPUT DATA:\n", 1)[0])
        payload = json.loads(prompt.split("REVIEW INPUT DATA:\n", 1)[1])
        self.assertEqual(payload["event_state"], self.event_state)
        self.assertEqual([item["constraint_id"] for item in schema["event_checks"]], ["c1"])
        self.assertEqual([item["fact_id"] for item in schema["state_checks"]], ["f1"])
        self.assertIn("positive plan check cannot settle a contradiction", prompt)
        self.assertIn("statement, implication and any supplied uncertainty", prompt)
        self.assertEqual(schema["state_checks"][0]["new_evidence"], [])
        self.assertEqual(schema["event_checks"][0]["new_evidence"], [])
        self.assertIn("never for contradicted", prompt)

    def test_event_contradiction_rejects_even_with_positive_plan_and_other_checks(self):
        self.review["event_checks"][0].update(verdict="contradicted",
                                              reason="The scene contradicts a historical event constraint.")
        result = self.validate()
        self.assertEqual(result["status"], "REJECT")
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["failures"],
                         ["event_constraint[c1]:contradicted: The scene contradicts a historical event constraint."])

    def test_event_empty_consistent_is_separate_diagnostic_and_contradicted_still_rejects(self):
        self.review["event_checks"][0]["new_evidence"] = []
        original = copy.deepcopy(self.review)
        result = self.validate()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["parsed"], original)
        self.assertEqual(result["empty_consistent_findings"][0]["constraint_id"], "c1")
        self.assertFalse(result["semantic_certified"])
        for basis in ("reported", "inferred", "uncertain"):
            with self.subTest(basis=basis):
                self.event_state["timeline"][0]["evidence_basis"] = basis
                self.review["event_checks"][0]["verdict"] = "contradicted"
                result = self.validate()
                self.assertEqual(result["status"], "REJECT")
                self.assertFalse(result["schema_valid"])
                self.assertIn("event_constraint[c1]:new_evidence_count_invalid", result["schema_failures"])
                self.assertEqual(result["empty_consistent_findings"], [])

    def test_soft_event_contradiction_is_diagnostic_and_keeps_raw_negative(self):
        self.event_state["timeline"][0].update(evidence_basis="inferred", status="completed")
        self.review["event_checks"][0].update(verdict="contradicted")
        self.review["event_checks"][0].pop("reason")
        self.review["consistent_with_history"] = False
        original = copy.deepcopy((self.review, self.event_state))
        result = self.validate()
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["history_consistency_status"], "UNRESOLVED_SOFT_FINDINGS")
        self.assertEqual(result["soft_findings"][0]["constraint_id"], "c1")
        self.assertEqual(result["parsed"]["event_checks"][0]["verdict"], "contradicted")
        self.assertFalse(result["parsed"]["consistent_with_history"])
        self.assertEqual((self.review, self.event_state), original)

    def test_reported_event_conflict_still_rejects_alongside_soft_state_finding(self):
        self.latest_state["facts"][0]["evidence_basis"] = "uncertain"
        self.review["state_checks"][0]["verdict"] = "contradicted"
        self.review["event_checks"][0]["verdict"] = "contradicted"
        self.review["consistent_with_history"] = False
        result = self.validate()
        self.assertEqual(result["status"], "REJECT")
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["history_consistency_status"], "MODEL_REPORTED_CONFLICT")
        self.assertEqual(len(result["soft_findings"]), 1)
        self.assertTrue(any(item.startswith("event_constraint[c1]:contradicted:")
                            for item in result["failures"]))
        self.assertTrue(any("review_verdict_reject:consistent_with_history" in item
                            for item in result["failures"]))

    def test_event_unknowns_and_scan_verdict_are_not_automatic_hard_rejections(self):
        self.event_state["timeline"][0]["status"] = "uncertain"
        self.event_state["plan_checks"][0]["verdict"] = "revise"
        result = self.validate()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["acceptance_basis"], "same_model_review_not_semantic_certification")

    def test_event_without_latest_state_works_and_empty_timeline_is_explicit(self):
        review = copy.deepcopy(self.review)
        review.pop("state_checks")
        self.assertEqual(self.validate(review, latest_state=False)["status"], "PASS")
        self.event_state["timeline"] = []
        review["event_checks"] = []
        self.assertEqual(self.validate(review, latest_state=False)["status"], "PASS")
        review.pop("event_checks")
        result = self.validate(review, latest_state=False)
        self.assertFalse(result["schema_valid"])
        self.assertIn("review_event_checks_not_array", result["schema_failures"])

    def test_event_missing_duplicate_unknown_and_wrong_space_ids_fail(self):
        for checks in ([], self.review["event_checks"] * 2):
            review = copy.deepcopy(self.review)
            review["event_checks"] = checks
            result = self.validate(review)
            self.assertFalse(result["schema_valid"])
            self.assertTrue(any("review_event_coverage_invalid" in x for x in result["schema_failures"]))
        for identifier in ("f1", "c2", [], True):
            review = copy.deepcopy(self.review)
            review["event_checks"][0]["constraint_id"] = identifier
            result = self.validate(review)
            self.assertIn("event_check[1]:constraint_id_invalid", result["schema_failures"])
        review = copy.deepcopy(self.review)
        review["event_checks"][0]["fact_id"] = review["event_checks"][0].pop("constraint_id")
        self.assertIn("event_check[1]:fields_invalid", self.validate(review)["schema_failures"])

    def test_event_citations_exact_bound_to_both_constraint_and_original_new_turn(self):
        modifications = [
            ("history_evidence", [{"turn_ref": "R01:S01:T001", "quote": "I have not decided"}], "missing_fact_reference"),
            ("history_evidence", [{"turn_ref": "NEW:T001", "quote": "I accept"}], "turn_ref_invalid"),
            ("history_evidence", [{"turn_ref": "P:T001", "quote": "Fabricated text"}], "quote_not_exact"),
            ("new_evidence", [{"turn_ref": "NEW:T002", "quote": "I accept"}], "quote_not_exact"),
            ("new_evidence", [{"turn_ref": "NEW:T001", "quote": "I accept...night"}], "quote_not_exact"),
            ("new_evidence", [{"turn_index": 1, "quote": "I accept"}], "fields_invalid"),
        ]
        for field, value, code in modifications:
            review = copy.deepcopy(self.review)
            review["event_checks"][0][field] = value
            result = self.validate(review)
            self.assertFalse(result["schema_valid"], result)
            self.assertTrue(any(code in x for x in result["schema_failures"]), result)

    def test_unrelated_constraint_does_not_require_new_dialogue_evidence(self):
        self.review["event_checks"][0].update(verdict="not_relevant", new_evidence=[],
            reason="This earlier event does not constrain this newly established occasion.")
        self.assertEqual(self.validate()["status"], "PASS")
        self.review["event_checks"][0]["history_evidence"] = []
        result = self.validate()
        self.assertFalse(result["schema_valid"])
        self.assertIn("event_constraint[c1]:history_evidence_count_invalid", result["schema_failures"])

    def test_malformed_event_snapshot_and_unbound_citations_fail_closed(self):
        for event_state in ([], {}, dict(self.event_state, timeline=None),
                            dict(self.event_state, route="different"),
                            dict(self.event_state, timeline=self.event_state["timeline"] * 2)):
            result = self.validate(event_state=event_state)
            self.assertEqual(result["status"], "REJECT")
            self.assertFalse(result["schema_valid"])
        self.event_state["timeline"][0]["evidence"][0]["quote"] = "Fabricated snapshot citation"
        result = self.validate()
        self.assertIn("review_event_state_fact_quote_not_exact:c1", result["schema_failures"])


class FrozenV11ReviewerReplayTests(unittest.TestCase):
    def test_138343_reviews_keep_original_answers_and_reject_ellipsis_quotes(self):
        previous = (Path(__file__).resolve().parent.parent / "h4_r02_yield_pilot_v11/runs"
                    / "yield_pilot_5roots_01/requests")
        requests = sorted(previous.glob("*-cc-train-episode-138343-B-S01-review-1-*"))
        self.assertEqual(len(requests), 2, "Both frozen v11 reviewer attempts must be available.")
        for folder in requests:
            with self.subTest(request=folder.name):
                paths = [folder / name for name in ("REQUEST.json", "RESPONSE.json", "REVIEW.json")]
                before = {path: path.read_bytes() for path in paths}
                payload, response, old_result = [json.loads(before[path]) for path in paths]
                context = json.JSONDecoder().raw_decode(
                    payload["messages"][0]["content"].split("\nREVIEW INPUT DATA:\n", 1)[1])[0]
                result = validate_review(
                    response["text"], context["root_id"], context["route"],
                    context["scene_plan"], context["new_scene"],
                    dataset_card=context["dataset_card_guidance"], latest_state=context["latest_state"],
                    event_state=context["event_state"], history=context["branch_history"])
                self.assertEqual(result["status"], "REJECT")
                self.assertFalse(result["schema_valid"])
                self.assertFalse(result["semantic_certified"])
                self.assertEqual(result["parsed"], json.loads(response["text"]))
                self.assertEqual(result["schema_failures"], [failure for failure in old_result["schema_failures"]
                    if not failure.endswith(":new_evidence_count_invalid")])
                self.assertIn("event_constraint[c4]:new_evidence[1]:quote_not_exact_nonempty_new_turn_span",
                              result["schema_failures"])
                self.assertEqual({item["fact_id"] for item in result["empty_consistent_findings"]},
                                 {f"f{number}" for number in range(1, 6)})
                self.assertEqual({path: path.read_bytes() for path in paths}, before)


if __name__ == "__main__":
    unittest.main()
