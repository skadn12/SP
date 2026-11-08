"""Offline contracts; model-like fixtures never become production extractions."""
from copy import deepcopy
import json
import unittest

import event_state as scan


class EventStateTests(unittest.TestCase):
    def setUp(self):
        self.root = "cc-train-episode-210046"
        self.route = "dataset_new_event"
        # Exact source excerpts preserve the historical milestones separately.
        self.history = [
            {"turn_ref": "cc-train-episode-210046:P05:T014", "speaker": "David",
             "text": "Of course, welcome to the team!"},
            {"turn_ref": "R01:S01:T015", "speaker": "Emily",
             "text": "I'm just heading into the office now. The email had all the details, so I think I've got everything covered."},
            {"turn_ref": "R01:S01:T021", "speaker": "Emily", "text": "I'm officially here! Let's make this work."},
            {"turn_ref": "R01:S02:T014", "speaker": "David",
             "text": "I look forward to hearing from you. Take care, and I'll see you at orientation."},
            {"turn_ref": "R01:S03:T011", "speaker": "Emily",
             "text": "I've made the donation, David. I sent you the receipt as well. I hope it helps."},
        ]
        self.scene = {"scene_id": "S01", "developments": [
            "Emily and David continue their office welcome and discuss a future trip.",
            "David recommends a hotel for that new trip.",
        ]}
        self.initial = {"root_id": self.root, "route": self.route, "scene_id": "S01",
                        "facts": [{"fact_id": "f1", "statement": "Intentionally incomplete prior summary."}],
                        "conflicts": [], "unknowns": []}
        self.card = {"card_id": "fixture", "guidance_level": "scaffold",
                     "source_text": "PRIVATE SOURCE", "source_id": "PRIVATE SOURCE ID",
                     "actors": [{"actor_id": "a1", "role": "traveler", "source_name": "PRIVATE NAME"}],
                     "steps": [{"step_id": "s1", "actor_id": "a1", "action": "request accommodation",
                                "state": "requested", "evidence": [{"quote": "PRIVATE SOURCE QUOTE"}]}],
                     "dependencies": []}
        self.result = {
            "root_id": self.root, "route": self.route, "scene_id": "S01",
            "segments": [
                {"segment_id": "cc-train-episode-210046:P05",
                 "none_reason": "This earlier welcome adds no separate status update needed for this plan."},
                {"segment_id": "R01:S01", "none_reason": ""},
                {"segment_id": "R01:S02", "none_reason": ""},
                {"segment_id": "R01:S03",
                 "none_reason": "The completed donation does not constrain the proposed accommodation discussion."},
            ],
            "timeline": [
                {"constraint_id": "c1", "status": "ongoing", "evidence_basis": "reported",
                 "statement": "Emily reports heading into the office.",
                 "evidence": [{"turn_ref": "R01:S01:T015", "quote": "I'm just heading into the office now."}],
                 "implication": "This in-transit state is superseded by the later reported arrival in c2.", "uncertainty": ""},
                {"constraint_id": "c2", "status": "completed", "evidence_basis": "reported",
                 "statement": "Emily reports being officially here.",
                 "evidence": [{"turn_ref": "R01:S01:T021", "quote": "I'm officially here!"}],
                 "implication": "Continue after reported arrival; do not replay the same first departure.",
                 "uncertainty": "Arrival does not establish whether orientation has finished."},
                {"constraint_id": "c3", "status": "planned", "evidence_basis": "reported",
                 "statement": "David says he will see Emily at orientation.",
                 "evidence": [{"turn_ref": "R01:S02:T014", "quote": "I'll see you at orientation."}],
                 "implication": "An unfinished orientation may continue without erasing the earlier reported arrival.",
                 "uncertainty": "The exact interval between arrival and this future-oriented remark is unclear."},
            ],
            "plan_checks": [
                {"development_index": 1, "verdict": "compatible", "constraint_ids": ["c1", "c2", "c3"],
                 "reason": "The plan continues after arrival and does not assume orientation is completed."},
                {"development_index": 2, "verdict": "compatible", "constraint_ids": [],
                 "reason": "A new hotel recommendation does not contradict a supplied historical constraint."},
            ],
        }

    def validate(self, raw=None):
        return scan.validate(self.result if raw is None else raw, self.root, self.route, self.scene, self.history)

    def codes(self, raw):
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate(raw)
        return {error["code"] for error in caught.exception.errors}

    def test_reported_arrival_and_pending_orientation_remain_distinct(self):
        original = deepcopy((self.result, self.history))
        checked = self.validate(json.dumps(self.result))
        self.assertEqual(checked, self.result)
        self.assertEqual([row["status"] for row in checked["timeline"]], ["ongoing", "completed", "planned"])
        self.assertTrue(checked["timeline"][2]["uncertainty"])
        self.assertFalse(scan.requires_replan(checked))
        self.assertEqual((self.result, self.history), original)

    def test_segment_index_uses_full_turn_prefix_and_keeps_input_order(self):
        original = deepcopy(self.history)
        self.assertEqual(scan.segment_index(self.history), [
            {"segment_id": "cc-train-episode-210046:P05", "first_turn_ref": "cc-train-episode-210046:P05:T014",
             "last_turn_ref": "cc-train-episode-210046:P05:T014", "turn_count": 1},
            {"segment_id": "R01:S01", "first_turn_ref": "R01:S01:T015", "last_turn_ref": "R01:S01:T021", "turn_count": 2},
            {"segment_id": "R01:S02", "first_turn_ref": "R01:S02:T014", "last_turn_ref": "R01:S02:T014", "turn_count": 1},
            {"segment_id": "R01:S03", "first_turn_ref": "R01:S03:T011", "last_turn_ref": "R01:S03:T011", "turn_count": 1},
        ])
        self.assertEqual(self.history, original)

    def test_missing_duplicate_and_unknown_segments_rejected(self):
        for operation in ("missing", "duplicate", "unknown"):
            with self.subTest(operation=operation):
                result = deepcopy(self.result)
                if operation == "missing":
                    result["segments"].pop()
                elif operation == "duplicate":
                    result["segments"].append(deepcopy(result["segments"][0]))
                else:
                    result["segments"][0]["segment_id"] = "R02:A:S01"
                self.assertTrue(self.codes(result) & {"event_state_segment_coverage_invalid", "event_state_segment_unknown"})

    def test_v11_segment_reference_field_is_rejected_even_when_correct(self):
        for refs in (["R01:S01:T015", "R01:S01:T021"], [], None):
            with self.subTest(refs=refs):
                result = deepcopy(self.result)
                result["segments"][1]["relevant_turn_refs"] = refs
                self.assertIn("event_state_fields_invalid", self.codes(result))
                with self.assertRaises(scan.EventStateValidationError):
                    scan.derive_coverage(result, self.history)

    def test_none_declarations_require_explicit_reason_and_no_refs(self):
        result = deepcopy(self.result)
        result["segments"][0]["none_reason"] = ""
        self.assertIn("event_state_segment_none_reason_invalid", self.codes(result))
        result = deepcopy(self.result)
        result["segments"][1]["none_reason"] = "Still unrelated"
        self.assertIn("event_state_segment_none_reason_invalid", self.codes(result))

    def test_coverage_does_not_certify_that_model_selected_decisive_later_turn(self):
        result = deepcopy(self.result)
        result["timeline"][1]["evidence"] = deepcopy(result["timeline"][0]["evidence"])
        # There is no independently model-declared reference list in v12.
        # A valid derived index cannot prove the model found the later update.
        checked = self.validate(result)
        coverage = scan.derive_coverage(checked, self.history)
        self.assertEqual(coverage["segments"][1]["relevant_turn_refs"], ["R01:S01:T015"])
        self.assertFalse(coverage["semantic_certified"])

    def test_timeline_citation_cannot_be_called_irrelevant_in_segment_reason(self):
        result = deepcopy(self.result)
        result["timeline"][1]["evidence"] = [{"turn_ref": "R01:S03:T011", "quote": "I've made the donation, David."}]
        self.assertIn("event_state_segment_none_reason_invalid", self.codes(result))

    def test_exact_quote_and_own_history_reference_required(self):
        for cite in ({"turn_ref": "R01:S01:T021", "quote": "I finished orientation!"},
                     {"turn_ref": "R02:A:S01:T001", "quote": "I'm officially here!"},
                     {"turn_ref": "R01:S01:T021", "quote": ""}):
            result = deepcopy(self.result)
            result["timeline"][1]["evidence"] = [cite]
            self.assertTrue(self.codes(result) & {"event_state_evidence_quote_invalid", "event_state_evidence_reference_invalid"})

    def test_added_root_prefix_remains_invalid_in_timeline(self):
        result = deepcopy(self.result)
        incorrect = f"{self.root}:R01:S01:T021"
        result["timeline"][1]["evidence"][0]["turn_ref"] = incorrect
        original = deepcopy(result)
        self.assertIn("event_state_evidence_reference_invalid", self.codes(result))
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate(result)
        errors = deepcopy(caught.exception.errors)
        repair = scan.build_repair(caught.exception, result, self.history)
        self.assertEqual(repair["errors"], errors)
        self.assertEqual(repair["previous_raw"], original)
        self.assertEqual(repair["citation_reference_candidates"], [{
            "path": "timeline[1].evidence[0].turn_ref", "actual_turn_ref": incorrect,
            "original_quote": "I'm officially here!", "candidate_turn_refs": ["R01:S01:T021"],
        }])
        self.assertEqual(result, original)
        self.assertEqual(caught.exception.errors, errors)
        self.assertIn("event_state_evidence_reference_invalid", self.codes(result))

    def test_reference_repair_keeps_zero_and_multiple_candidates_without_editing_raw(self):
        self.history.append({"turn_ref": "R01:S04:T001", "speaker": "David",
                             "text": "I'm officially here! This concerns a different visit."})
        for quote, expected in (("I'm officially here!", ["R01:S01:T021", "R01:S04:T001"]),
                                ("i'm officially here!", [])):
            with self.subTest(quote=quote):
                result = deepcopy(self.result)
                result["timeline"][1]["evidence"][0] = {"turn_ref": "unknown", "quote": quote}
                raw = json.dumps(result)
                original = deepcopy((result, self.history))
                with self.assertRaises(scan.EventStateValidationError) as caught:
                    self.validate(result)
                repair = scan.build_repair(caught.exception, raw, self.history)
                candidate = repair["citation_reference_candidates"][0]
                self.assertEqual(candidate["candidate_turn_refs"], expected)
                self.assertEqual(candidate["original_quote"], quote)
                self.assertNotIn("selected_turn_ref", candidate)
                self.assertEqual(repair["previous_raw"], raw)
                self.assertEqual((result, self.history), original)

    def test_each_development_checked_once(self):
        for operation in ("missing", "duplicate", "bool_index"):
            result = deepcopy(self.result)
            if operation == "missing":
                result["plan_checks"].pop()
            elif operation == "duplicate":
                result["plan_checks"].append(deepcopy(result["plan_checks"][0]))
            else:
                result["plan_checks"][0]["development_index"] = True
            self.assertIn("event_state_plan_check_coverage_invalid", self.codes(result))

    def test_negative_development_is_binding_even_if_other_checks_pass(self):
        result = deepcopy(self.result)
        result["plan_checks"][0].update(verdict="revise", reason="The proposed first departure would ignore the reported arrival.")
        checked = self.validate(result)
        self.assertTrue(scan.requires_replan(checked))
        result["overall"] = "PASS"
        self.assertIn("event_state_fields_invalid", self.codes(result))

    def test_all_negative_developments_remain_negative(self):
        result = deepcopy(self.result)
        for check in result["plan_checks"]:
            check["verdict"] = "revise"
        checked = self.validate(result)
        self.assertTrue(scan.requires_replan(checked))
        self.assertEqual(checked["plan_checks"], result["plan_checks"])

    def test_soft_only_revise_is_diagnostic_and_keeps_original_verdict_and_status(self):
        for basis in ("inferred", "uncertain"):
            result = deepcopy(self.result)
            result["timeline"][1]["evidence_basis"] = basis
            result["plan_checks"][0].update(verdict="revise", constraint_ids=["c2"])
            original = deepcopy(result)
            checked = self.validate(result)
            self.assertFalse(scan.requires_replan(checked))
            self.assertEqual(scan.binding_plan_checks(checked), [])
            findings = scan.plan_diagnostics(checked)
            self.assertEqual(findings[0]["verdict"], "revise")
            self.assertFalse(findings[0]["requires_replan"])
            self.assertEqual(findings[0]["evidence_bases"], {"c2": basis})
            self.assertEqual(checked["timeline"][1]["status"], "completed")
            self.assertEqual(checked, original)
            self.assertEqual(result, original)

    def test_mixed_reported_and_soft_revise_remains_binding(self):
        result = deepcopy(self.result)
        result["timeline"][0]["evidence_basis"] = "inferred"
        result["plan_checks"][0].update(verdict="revise", constraint_ids=["c1", "c2"])
        checked = self.validate(result)
        self.assertTrue(scan.requires_replan(checked))
        binding = scan.binding_plan_checks(checked)
        self.assertEqual(binding, [result["plan_checks"][0]])
        binding[0]["constraint_ids"].clear()
        self.assertEqual(checked, result)

    def test_unattributed_revise_remains_binding(self):
        result = deepcopy(self.result)
        result["plan_checks"][1]["verdict"] = "revise"
        checked = self.validate(result)
        self.assertTrue(scan.requires_replan(checked))
        self.assertEqual(scan.binding_plan_checks(checked), [result["plan_checks"][1]])

    def test_evidence_basis_is_required_and_validated_independently(self):
        for basis in sorted(scan.EVIDENCE_BASES):
            result = deepcopy(self.result)
            result["timeline"][1]["evidence_basis"] = basis
            self.assertEqual(self.validate(result), result)
        for basis in ("verified", "", None, []):
            result = deepcopy(self.result)
            result["timeline"][1]["evidence_basis"] = basis
            self.assertIn("event_state_evidence_basis_invalid", self.codes(result))
        result = deepcopy(self.result)
        result["timeline"][1].pop("evidence_basis")
        self.assertIn("event_state_fields_invalid", self.codes(result))

    def test_plan_constraints_require_actual_distinct_ids(self):
        for refs in (["c99"], ["c1", "c1"], [True], None):
            result = deepcopy(self.result)
            result["plan_checks"][0]["constraint_ids"] = refs
            self.assertIn("event_state_plan_constraint_ids_invalid", self.codes(result))
        result = deepcopy(self.result)
        result["plan_checks"][1]["reason"] = ""
        self.assertIn("event_state_text_required", self.codes(result))

    def test_status_ids_implication_and_uncertainty_contract(self):
        result = deepcopy(self.result)
        result["timeline"][0].update(constraint_id="f1", status="unknown", implication="", uncertainty=None)
        self.assertTrue({"event_state_constraint_id_invalid", "event_state_status_invalid",
                         "event_state_text_required", "event_state_uncertainty_invalid"} <= self.codes(result))
        result = deepcopy(self.result)
        result["timeline"][0]["evidence"] = []
        self.assertIn("event_state_evidence_count_invalid", self.codes(result))

    def test_uncertainty_omission_is_preserved_and_never_changes_negative_policy(self):
        # v11 repeatedly omitted uncertainty across every otherwise valid entry.
        # The v12 contract accepts absence without inserting empty strings.
        for basis in ("reported", "inferred", "uncertain"):
            with self.subTest(basis=basis):
                result = deepcopy(self.result)
                result["timeline"][1]["evidence_basis"] = basis
                result["plan_checks"][0].update(verdict="revise", constraint_ids=["c2"])
                expected_diagnostics = scan.plan_diagnostics(self.validate(result))
                for constraint in result["timeline"]:
                    constraint.pop("uncertainty")
                original = deepcopy((result, self.history))
                checked = self.validate(json.dumps(result))
                self.assertEqual(checked, result)
                self.assertTrue(all("uncertainty" not in row for row in checked["timeline"]))
                self.assertEqual(scan.plan_diagnostics(checked), expected_diagnostics)
                self.assertEqual(scan.requires_replan(checked), basis == "reported")
                coverage = scan.derive_coverage(checked, self.history)
                self.assertEqual((result, self.history), original)
                self.assertEqual(checked, result)
                self.assertFalse(coverage["semantic_certified"])

    def test_present_uncertainty_must_be_string_and_other_required_fields_stay_required(self):
        for value in (None, [], {}, False, 0):
            result = deepcopy(self.result)
            result["timeline"][0]["uncertainty"] = value
            with self.subTest(value=value):
                self.assertIn("event_state_uncertainty_invalid", self.codes(result))
        for field in ("constraint_id", "status", "evidence_basis", "statement", "evidence", "implication"):
            result = deepcopy(self.result)
            result["timeline"][0].pop(field)
            with self.subTest(field=field):
                self.assertIn("event_state_fields_invalid", self.codes(result))
        result = deepcopy(self.result)
        result["timeline"][0]["uncertainity"] = result["timeline"][0].pop("uncertainty")
        self.assertIn("event_state_fields_invalid", self.codes(result))

    def test_derived_coverage_is_separate_ordered_and_deduplicated_from_exact_evidence(self):
        result = deepcopy(self.result)
        result["segments"].reverse()
        result["timeline"][1]["evidence"].append(deepcopy(result["timeline"][0]["evidence"][0]))
        result["timeline"][0]["evidence"] = deepcopy(result["timeline"][1]["evidence"])
        original = deepcopy((result, self.history))
        checked = self.validate(result)
        coverage = scan.derive_coverage(checked, self.history)
        self.assertEqual(coverage, {
            "version": scan.EVENT_STATE_VERSION, "derived_from": "timeline.evidence", "semantic_certified": False,
            "root_id": self.root, "route": self.route, "scene_id": "S01",
            "segments": [
                {"segment_id": "cc-train-episode-210046:P05", "relevant_turn_refs": [],
                 "none_reason": self.result["segments"][0]["none_reason"]},
                {"segment_id": "R01:S01", "relevant_turn_refs": ["R01:S01:T015", "R01:S01:T021"], "none_reason": ""},
                {"segment_id": "R01:S02", "relevant_turn_refs": ["R01:S02:T014"], "none_reason": ""},
                {"segment_id": "R01:S03", "relevant_turn_refs": [], "none_reason": self.result["segments"][3]["none_reason"]},
            ],
        })
        self.assertEqual(checked, result)
        self.assertEqual((result, self.history), original)
        self.assertTrue(all("relevant_turn_refs" not in row for row in checked["segments"]))
        coverage["segments"][1]["relevant_turn_refs"].clear()
        self.assertEqual((result, self.history), original)

    def test_derive_coverage_refuses_invalid_evidence_or_segment_declarations(self):
        for mutation in ("bad_quote", "bad_ref", "no_evidence", "too_many_citations", "missing_segment",
                         "duplicate_segment", "foreign_segment", "none_reason_conflict"):
            result = deepcopy(self.result)
            if mutation == "bad_quote":
                result["timeline"][1]["evidence"][0]["quote"] = "I completed orientation."
            elif mutation == "bad_ref":
                result["timeline"][1]["evidence"][0]["turn_ref"] = "R02:A:S01:T001"
            elif mutation == "no_evidence":
                result["timeline"][1]["evidence"] = []
            elif mutation == "too_many_citations":
                result["timeline"][1]["evidence"] *= 4
            elif mutation == "missing_segment":
                result["segments"].pop()
            elif mutation == "duplicate_segment":
                result["segments"].append(deepcopy(result["segments"][0]))
            elif mutation == "foreign_segment":
                result["segments"][0]["segment_id"] = "R02:A:S01"
            else:
                result["segments"][1]["none_reason"] = "This segment has no related update."
            with self.subTest(mutation=mutation), self.assertRaises(scan.EventStateValidationError):
                scan.derive_coverage(result, self.history)

    def test_later_scene_wording_does_not_override_model_reported_hard_negative(self):
        result = deepcopy(self.result)
        result["plan_checks"][0].update(verdict="revise", constraint_ids=["c3"],
                                          reason="The clearly later scene still needs explicit event separation.")
        checked = self.validate(result)
        self.assertTrue(scan.requires_replan(checked))
        self.assertEqual(scan.binding_plan_checks(checked), [result["plan_checks"][0]])

    def test_empty_timeline_requires_a_reason_for_every_segment(self):
        result = deepcopy(self.result)
        result["timeline"] = []
        for entry in result["segments"]:
            entry.update(none_reason="Fixture plan has no related historical state in this segment.")
        for check in result["plan_checks"]:
            check.update(constraint_ids=[], reason="Fixture new event has no historical constraint.")
        self.assertEqual(self.validate(result), result)
        # This is structural acceptance only: the program cannot certify a
        # model's judgment that a segment has no semantically relevant updates.
        self.assertTrue(all(not row["relevant_turn_refs"]
                            for row in scan.derive_coverage(result, self.history)["segments"]))
        result["segments"][1]["none_reason"] = ""
        self.assertIn("event_state_segment_none_reason_invalid", self.codes(result))

    def test_schema_validation_does_not_certify_statement_truth_or_completeness(self):
        result = deepcopy(self.result)
        result["timeline"][1]["statement"] = "Intentionally unsupported claim with valid fixture citation."
        self.assertEqual(self.validate(result), result)

    def test_duplicate_keys_nonfinite_and_nonobject_json_rejected(self):
        raw = json.dumps(self.result)
        bad = raw.replace('"verdict": "compatible"', '"verdict": "revise", "verdict": "compatible"', 1)
        self.assertIn("event_state_duplicate_json_key", self.codes(bad))
        for raw in ("[]", "null", "```json\n{}\n```", '{"timeline":NaN}', 4):
            self.assertTrue(self.codes(raw) & {"event_state_response_not_object", "event_state_response_invalid_json"})

    def test_bad_history_never_silently_drops_a_segment(self):
        for mutation in ("bad_ref", "duplicate", "bad_text"):
            history = deepcopy(self.history)
            if mutation == "bad_ref":
                history[0]["turn_ref"] = "unsegmented"
            elif mutation == "duplicate":
                history.append(deepcopy(history[0]))
            else:
                history[0]["text"] = None
            with self.subTest(mutation=mutation), self.assertRaises(scan.EventStateValidationError):
                scan.segment_index(history)

    def test_request_keeps_entire_history_and_actual_plan_without_source_leakage(self):
        history = deepcopy(self.history)
        history[1:1] = [{"turn_ref": f"fixture:unrelated:T{i:03d}", "speaker": "Emily", "text": f"Fixture {i}."}
                        for i in range(1, 201)]
        original = deepcopy((history, self.scene, self.card, self.initial))
        payload = scan.build_request(self.root, self.route, self.scene, history, [self.card], self.initial)
        self.assertEqual(payload["model"], scan.MODEL)
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(payload["temperature"], .25)
        self.assertFalse(payload["enable_thinking"])
        context = json.loads(payload["messages"][0]["content"].split("\nEVENT_STATE_INPUT_JSON\n", 1)[1])
        self.assertEqual(context["cumulative_history"], history)
        self.assertEqual(context["scene_plan"], self.scene)
        self.assertEqual(context["latest_state"], self.initial)
        self.assertEqual(context["segment_index"], scan.segment_index(history))
        self.assertEqual(context["history_reference_index"], [turn["turn_ref"] for turn in history])
        self.assertEqual(context["history_reference_examples"], [history[0]["turn_ref"], history[1]["turn_ref"]])
        self.assertEqual((history, self.scene, self.card, self.initial), original)
        for forbidden in ("PRIVATE", "source_id", "source_text", "source_name", "other_branch", "memory", "feedback"):
            self.assertNotIn(forbidden, json.dumps(context))

    def test_repair_preserves_detailed_errors_and_original_model_output(self):
        result = deepcopy(self.result)
        result["segments"].pop()
        result["timeline"][0]["evidence"][0]["quote"] = "fabricated"
        try:
            self.validate(result)
        except scan.EventStateValidationError as error:
            repair = scan.build_repair(error, result)
            self.assertEqual(repair["errors"], error.errors)
        self.assertEqual(repair["previous_raw"], result)
        payload = scan.build_request(self.root, self.route, self.scene, self.history, [self.card], self.initial, repair)
        context = json.loads(payload["messages"][0]["content"].split("\nEVENT_STATE_INPUT_JSON\n", 1)[1])
        self.assertEqual(context["targeted_repair"], repair)

    def test_prompt_explicitly_requires_real_updates_and_distinguishes_milestones(self):
        for marker in ("ENTIRE", "six facts", "{segment_id,none_reason}", "NOT automatically event time",
                       "does not imply completed", "see you at orientation", "inferred/uncertain", "NO overall pass"):
            self.assertIn(marker, scan.PROMPT)
        for marker in ("VERBATIM", "NOT a reference prefix", "Mixed namespaces"):
            self.assertIn(marker, " ".join(scan.PROMPT.split()))


if __name__ == "__main__":
    unittest.main()
