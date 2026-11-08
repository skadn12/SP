"""Exposed v97 failures and adversarial correction mutations; no model calls.

Mutated corrected responses are explicitly authored fixtures, never new model
successes. Historical validator replay is kept separate from v98 semantic rules.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import card_selection
import event_state as event


V97 = Path(__file__).resolve().parents[2] / "h4_semantic_review_mixed_pilot_v97"
RUN = V97 / "expansion_engine/runs/four_roots_B_scene01"
REQUESTS = RUN / "requests"
FIRST = "020-cc-train-episode-104814-B-S01-event-state-2-1"
RETRY = "021-cc-train-episode-104814-B-S01-event-state-2-2"
_spec = importlib.util.spec_from_file_location("frozen_v97_event_correction_reference", V97 / "expansion_engine/event_state.py")
frozen_v97 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frozen_v97)


def read(path):
    return json.loads(path.read_text())


def response(name):
    return read(REQUESTS / name / "RESPONSE.json")["text"]


def request_context(name):
    text = read(REQUESTS / name / "REQUEST.json")["messages"][0]["content"]
    return json.JSONDecoder().raw_decode(text.split("EVENT_STATE_INPUT_JSON\n", 1)[1])[0]


class PreciseCardCorrectionV98Tests(unittest.TestCase):
    def setUp(self):
        self.sources = read(RUN / "dataset_sources.json")
        self.subset = card_selection._legacy.select_source_subset(self.sources, ["soda:train:6027861"])
        self.names = ["004-extract-soda-train-6027861-1", "005-extract-soda-train-6027861-2"]

    def test_both_real_rejected_cards_report_exact_enum_and_do_not_mutate(self):
        for name in self.names:
            with self.subTest(name=name):
                raw = response(name)
                before = deepcopy(self.subset)
                rejected = card_selection.select_cards(raw, self.subset)
                self.assertEqual(rejected["accepted_cards"], [])
                errors = rejected["repair_feedback"]["candidate_errors"][0]["field_errors"]
                self.assertEqual(len(errors), 1)
                self.assertEqual(errors[0]["path"], "cards[0].steps[1].state")
                self.assertEqual(errors[0]["actual"], "proposed_action")
                self.assertEqual(errors[0]["step_id"], "s2")
                self.assertEqual(errors[0]["legal_values"], sorted(card_selection._legacy.STEP_STATES))
                self.assertEqual(before, self.subset)
                self.assertEqual(raw, response(name))
                request = card_selection.build_extraction_request(self.subset, rejected["repair_feedback"])
                self.assertEqual(request["max_tokens"], 4096)
                text = request["messages"][0]["content"]
                self.assertIn('"actual": "proposed_action"', text)
                self.assertIn('"path": "cards[0].steps[1].state"', text)
                self.assertIn("complete card again", text)
                self.assertFalse(rejected["repair_feedback"]["automatic_enum_replacement"])

    def test_authored_enum_change_still_receives_complete_card_validation(self):
        raw = json.loads(response(self.names[0]))
        raw["cards"][0]["steps"][1]["state"] = "proposed"
        self.assertEqual(len(card_selection.select_cards(raw, self.subset)["accepted_cards"]), 1)
        raw["cards"][0]["steps"][1]["evidence"][0]["quote"] = "Invented source quotation."
        rejected = card_selection.select_cards(raw, self.subset)
        self.assertEqual(rejected["accepted_cards"], [])
        self.assertEqual(rejected["rejected_records"][0]["error_code"], "EVIDENCE_INVALID")
        self.assertEqual(raw["cards"][0]["steps"][1]["state"], "proposed")

    def test_multiple_enum_diagnostics_and_nonstring_values_are_safe(self):
        raw = json.loads(response(self.names[0]))
        raw["cards"][0]["guidance_level"] = "scaffolding"
        raw["cards"][0]["steps"][0]["state"] = ["proposed"]
        raw["cards"][0]["dependencies"][0]["relation"] = "reply"
        rejected = card_selection.select_cards(raw, self.subset)
        errors = rejected["repair_feedback"]["candidate_errors"][0]["field_errors"]
        self.assertEqual(len(errors), 4)
        self.assertEqual(errors[1]["actual"], ["proposed"])
        self.assertEqual(errors[-1]["legal_values"], sorted(card_selection._legacy.DEPENDENCY_RELATIONS))


class EventCorrectionResponsivenessV98Tests(unittest.TestCase):
    def setUp(self):
        self.context = request_context(FIRST)
        self.bad = json.loads(response(FIRST))
        try:
            self.validate_full(self.bad)
        except frozen_v97.EventStateValidationError as error:
            self.repair = event.build_repair(error, response(FIRST), history=self.context["cumulative_history"])
        else:
            raise AssertionError("The exposed v97 wrong-development quotation must fail the frozen full validator")

    def validate_full(self, value):
        c = self.context
        return frozen_v97.validate(value, c["root_id"], c["route"], c["scene_plan"],
                                   c["cumulative_history"], cards=c["dataset_cards"])

    def audit(self, value):
        return event.correction_progress(value, self.repair)

    def assert_progress_error(self, code, value):
        result = self.audit(value)
        self.assertEqual(result["status"], "REJECT")
        self.assertIn(code, [row["code"] for row in result["errors"]])
        with self.assertRaises(event.EventStateValidationError):
            event.require_correction_progress(result)
        return result

    def fixed_fixture(self):
        corrected = deepcopy(self.bad)
        corrected["consistency_checks"][1]["plan_evidence"][1]["quote"] = "soil preparation instructions"
        return corrected

    def test_real_byte_identical_retry_and_whitespace_reformatting_are_rejected(self):
        self.assertEqual(response(FIRST), response(RETRY))
        self.assert_progress_error("correction_response_unchanged", response(RETRY))
        self.assert_progress_error("correction_response_unchanged", json.dumps(self.bad, sort_keys=True))
        with self.assertRaises(frozen_v97.EventStateValidationError):
            self.validate_full(response(RETRY))

    def test_unrelated_edits_or_row_reordering_do_not_fix_the_reported_evidence(self):
        for mutation in ("reason", "reorder"):
            with self.subTest(mutation=mutation):
                changed = deepcopy(self.bad)
                if mutation == "reason":
                    changed["plan_checks"][0]["reason"] += " An unrelated extra sentence."
                else:
                    changed["consistency_checks"].reverse()
                    changed["consistency_checks"][1]["plan_evidence"].reverse()
                self.assert_progress_error("correction_reported_evidence_unchanged", changed)

    def test_deleting_bad_quote_cannot_turn_full_structure_pass_into_retry_pass(self):
        changed = deepcopy(self.bad)
        changed["consistency_checks"][1]["plan_evidence"].pop()
        # The old full schema admits one exact quote, but the retry must replace
        # the invalid evidence rather than silently dropping its obligation.
        self.validate_full(changed)
        self.assert_progress_error("correction_evidence_or_check_deleted", changed)

    def test_owner_deletion_and_not_applicable_are_not_corrections(self):
        changed = deepcopy(self.bad)
        changed["consistency_checks"].pop(1)
        self.assert_progress_error("correction_evidence_or_check_deleted", changed)
        with self.assertRaises(frozen_v97.EventStateValidationError):
            self.validate_full(changed)
        changed = self.fixed_fixture()
        changed["consistency_checks"][1]["verdict"] = "not_applicable"
        self.assert_progress_error("correction_evidence_owner_verdict_changed", changed)
        with self.assertRaises(frozen_v97.EventStateValidationError):
            self.validate_full(changed)

    def test_evidence_correction_must_not_silently_flip_negative_verdict(self):
        previous = deepcopy(self.bad)
        previous["consistency_checks"][1]["verdict"] = "revise"
        previous["consistency_checks"][1]["reason"] = "A concrete transition is missing in this authored negative fixture."
        repair = event.build_repair(frozen_v97.EventStateValidationError(self.repair["errors"]), previous)
        changed = self.fixed_fixture()
        audit = event.correction_progress(changed, repair)
        self.assertIn("correction_evidence_owner_verdict_changed", [row["code"] for row in audit["errors"]])

    def test_previous_positive_can_become_binding_negative_after_complete_review(self):
        corrected = self.fixed_fixture()
        corrected["consistency_checks"][1].update(
            verdict="revise", reason="This authored re-review identifies a concrete missing transition.")
        self.assertEqual(self.audit(corrected)["status"], "PASS")
        validated = self.validate_full(corrected)
        self.assertTrue(frozen_v97.requires_replan(validated))

    def test_authored_supported_replacement_or_correct_reference_can_progress(self):
        for mutation in ("quote", "reference"):
            with self.subTest(mutation=mutation):
                corrected = self.fixed_fixture() if mutation == "quote" else deepcopy(self.bad)
                if mutation == "reference":
                    corrected["consistency_checks"][1]["plan_evidence"][0]["quote"] = "safe, enclosed patch of soil"
                    corrected["consistency_checks"][1]["plan_evidence"][1]["development_index"] = 1
                result = self.audit(corrected)
                self.assertEqual(result["status"], "PASS")
                self.assertTrue(result["full_validation_required"])
                self.assertFalse(result["semantic_certified"])
                event.require_correction_progress(result)
                self.validate_full(corrected)

    def test_changed_target_does_not_skip_full_response_validation(self):
        corrected = self.fixed_fixture()
        corrected["timeline"].append({"constraint_id": "c4", "status": "uncertain"})
        self.assertEqual(self.audit(corrected)["status"], "PASS")
        with self.assertRaises(frozen_v97.EventStateValidationError):
            self.validate_full(corrected)

    def test_no_retry_and_malformed_response_are_distinct_and_inputs_unchanged(self):
        self.assertEqual(event.correction_progress(self.bad)["status"], "NOT_A_RETRY")
        before = deepcopy((self.repair, self.bad))
        self.audit(self.bad)
        self.assertEqual(before, (self.repair, self.bad))
        self.assert_progress_error("correction_response_not_parseable", "{invalid-json")
        self.assert_progress_error("correction_response_not_parseable", '{"x": 1, "x": 2}')

    def test_retry_with_no_resolvable_error_fields_still_detects_document_repetition(self):
        for repair in ({"previous_raw": response(FIRST), "errors": []},
                       event.build_repair(ValueError("generic_contract_failure"), response(FIRST))):
            with self.subTest(repair_errors=repair["errors"]):
                unchanged = event.correction_progress(self.bad, repair)
                self.assertEqual(unchanged["status"], "REJECT")
                self.assertIn("correction_response_unchanged", [row["code"] for row in unchanged["errors"]])
                corrected = self.fixed_fixture()
                self.assertEqual(event.correction_progress(corrected, repair)["status"], "PASS")
                self.validate_full(corrected)

    def test_complete_retry_request_preserves_frozen_history_plan_budget_and_targets(self):
        c = self.context
        request = event.build_request(c["root_id"], c["route"], c["scene_plan"], c["cumulative_history"],
                                      c["dataset_cards"], c["latest_state"], repair=self.repair)
        self.assertEqual(request["max_tokens"], 4096)
        text = request["messages"][0]["content"]
        current = json.JSONDecoder().raw_decode(text.split("EVENT_STATE_INPUT_JSON\n", 1)[1])[0]
        self.assertEqual(current["cumulative_history"], c["cumulative_history"])
        self.assertEqual(current["scene_plan"], c["scene_plan"])
        self.assertEqual(current["targeted_repair"]["previous_raw"], response(FIRST))
        target = current["targeted_repair"]["correction_contract"]["targets"][0]
        self.assertEqual(target["path"], "consistency_checks[1].plan_evidence[1].quote")
        self.assertEqual(target["previous_value"], "in her backyard")
        self.assertEqual(target["evidence_replacement"]["previous_count"], 2)
        self.assertIn("complete review again", text)

    def test_production_runner_records_progress_and_runs_full_validator_on_both_attempts(self):
        # The unchanged v97 validator isolates this exposed citation regression;
        # it does not auto-upgrade old records to the new v98 semantic schema.
        import runner

        for mutation in ("unchanged", "delete", "replace"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                retry = deepcopy(self.bad)
                if mutation == "delete":
                    retry["consistency_checks"][1]["plan_evidence"].pop()
                elif mutation == "replace":
                    retry = self.fixed_fixture()
                queue, folders, payloads = [deepcopy(self.bad), retry], [], []

                class Caller:
                    def call(_self, label, payload, proof):
                        folder = Path(tmp) / "requests" / str(len(folders) + 1)
                        folder.mkdir(parents=True)
                        folders.append(folder)
                        payloads.append(deepcopy(payload))
                        return json.dumps(queue.pop(0)), folder

                calls = []

                def full(raw, root, route, scene, history, cards=None):
                    calls.append(raw)
                    return frozen_v97.validate(raw, root, route, scene, history, cards=cards)

                c = self.context
                with patch.object(runner, "branch_history_ok", return_value=c["cumulative_history"]), \
                     patch.object(runner.event_contract, "validate", side_effect=full):
                    result = runner.extract_event_state(
                        Caller(), Path(tmp), c["root_id"], c["route"], c["cumulative_history"],
                        c["dataset_cards"], c["scene_plan"], c["latest_state"], plan_attempt=2)
                self.assertEqual(len(calls), 2)
                self.assertEqual(len(folders), runner.EVENT_STATE_ATTEMPTS)
                self.assertTrue(all(payload["max_tokens"] == 4096 for payload in payloads))
                self.assertEqual(read(folders[0] / "CORRECTION_PROGRESS.json")["status"], "NOT_A_RETRY")
                second = read(folders[1] / "VALIDATION.json")
                progress = read(folders[1] / "CORRECTION_PROGRESS.json")
                if mutation == "replace":
                    self.assertIsNotNone(result)
                    self.assertEqual(second["status"], "PASS")
                    self.assertEqual(progress["status"], "PASS")
                else:
                    self.assertIsNone(result)
                    self.assertEqual(second["status"], "REJECT")
                    self.assertEqual(progress["status"], "REJECT")
                    expected = ("event_state_plan_evidence_quote_invalid" if mutation == "unchanged"
                                else "correction_global_evidence_or_check_deleted")
                    self.assertEqual(second["reason"], expected)


class NestedComparisonCorrectionV98Tests(unittest.TestCase):
    """New v98 comparison citations use the same replacement safeguards."""

    def setUp(self):
        import event_identity
        self.identity = event_identity
        self.path = "card_checks[0].function_check.comparison"
        self.card = {"card_id": "fixture", "actors": [{"actor_id": "a1"}, {"actor_id": "a2"}]}
        self.history = [{"turn_ref": "old:1", "speaker": "Emily", "text": "We planted a garden last year."},
                        {"turn_ref": "old:2", "speaker": "James", "text": "I taught that lesson last year."}]
        self.scene = {"actors": ["Emily", "James"], "developments": [
            "Emily asks James to teach a new potting lesson today.",
            "James accepts Emily's request to teach the lesson today."]}
        comparison = {
            "purpose": {"source": "request and accept a lesson", "actual": "request and accept a new potting lesson"},
            "start_state": {"source": "a new request not yet accepted", "actual": "a new request not yet accepted"},
            "end_state": {"source": "lesson request accepted", "actual": "lesson request accepted"},
            "roles": [{"actor_id": "a1", "participant": "Emily", "role": "requester"},
                      {"actor_id": "a2", "participant": "James", "role": "teacher accepting request"}],
            "current_trigger": "Emily requests today's new potting lesson.",
            "nearest_old_event": {"summary": "A completed garden lesson last year.",
                                  "history_evidence": [{"turn_ref": "old:2", "quote": "I taught that lesson last year."}],
                                  "difference": "A newly requested potting lesson today; the prior one was completed last year."},
            "request_acceptance": {"required": True, "verdict": "compatible",
                                   "request_evidence": [{"development_index": 1, "quote": self.scene["developments"][0]}],
                                   "acceptance_evidence": [{"development_index": 2, "quote": self.scene["developments"][1]}]},
            "experience_state": {"source": "future_arrangement", "actual": "future_arrangement", "verdict": "compatible"},
        }
        self.document = {"card_checks": [{"card_id": "fixture", "function_check": {"comparison": comparison}}]}
        self.assertEqual(self.errors(self.document), [])

    def comparison(self, value):
        return value["card_checks"][0]["function_check"]["comparison"]

    def errors(self, value):
        return self.identity.collect_errors(self.comparison(value), self.card, self.history,
                                            path=self.path, comparison=True, scene=self.scene)

    def test_all_new_comparison_evidence_lists_require_replacement_not_deletion(self):
        for key in ("history_evidence", "request_evidence", "acceptance_evidence"):
            with self.subTest(evidence_key=key):
                bad = deepcopy(self.document)
                owner_key = "nearest_old_event" if key == "history_evidence" else "request_acceptance"
                self.comparison(bad)[owner_key][key][0]["quote"] = "This is an invented quotation."
                errors = self.errors(bad)
                self.assertEqual(errors[0]["code"], "event_identity_quote_invalid")
                repair = event.build_repair(event.EventStateValidationError(errors), bad)
                self.assertEqual(event.correction_progress(bad, repair)["status"], "REJECT")
                deleted = deepcopy(bad)
                self.comparison(deleted)[owner_key][key] = []
                if owner_key == "request_acceptance":
                    # A negative legitimately permits an absent action, yet this
                    # citation retry still must not hide its reported fake span.
                    self.comparison(deleted)[owner_key]["verdict"] = "revise"
                self.assertEqual(self.errors(deleted), [])
                audit = event.correction_progress(deleted, repair)
                self.assertIn("correction_evidence_or_check_deleted", [row["code"] for row in audit["errors"]])
                self.assertEqual(event.correction_progress(self.document, repair)["status"], "PASS")
                self.assertEqual(self.errors(self.document), [])

    def test_nested_negative_cannot_be_erased_but_new_negative_is_allowed(self):
        bad = deepcopy(self.document)
        request = self.comparison(bad)["request_acceptance"]
        request["verdict"] = "revise"
        request["acceptance_evidence"][0]["quote"] = "This is an invented quotation."
        repair = event.build_repair(event.EventStateValidationError(self.errors(bad)), bad)
        audit = event.correction_progress(self.document, repair)
        self.assertIn("correction_evidence_owner_verdict_changed", [row["code"] for row in audit["errors"]])
        corrected_negative = deepcopy(self.document)
        self.comparison(corrected_negative)["request_acceptance"]["verdict"] = "revise"
        self.assertEqual(event.correction_progress(corrected_negative, repair)["status"], "PASS")
        self.assertEqual(self.identity.explicit_negatives(self.comparison(corrected_negative)), ["request_acceptance"])


class GlobalRetryProtectionV98Tests(unittest.TestCase):
    """Independent pre-freeze bypasses remain archived, and now fail closed."""

    def test_independently_discovered_plan_negative_and_required_bypass_is_blocked(self):
        fixture = read(Path(__file__).resolve().parents[1] / "evidence/INTEGRATION_INITIAL_CORRECTION_BYPASS.json")
        current = event.validate(fixture["retry_review"], fixture["root"], fixture["route"],
                                 fixture["scene"], fixture["history"], cards=[fixture["card"]])
        self.assertFalse(event.requires_replan(current))
        audit = event.correction_progress(fixture["retry_review"], fixture["repair"])
        self.assertEqual(audit["status"], "REJECT")
        erased = [row for row in audit["errors"] if row["code"] == "correction_protected_judgment_erased"]
        self.assertTrue(any(row["previous"] == "revise" for row in erased))
        self.assertTrue(any(row["previous"] is True for row in erased))

    def test_independently_discovered_scene_negative_and_required_bypass_is_blocked(self):
        import realization_review

        fixture = read(Path(__file__).resolve().parents[1] / "evidence/INTEGRATION_INITIAL_SCENE_CORRECTION_BYPASS.json")
        result = realization_review.validate_review(fixture["retry_review"], fixture["root"], fixture["route"],
                                                   fixture["scene"], fixture["realization"],
                                                   history=fixture["history"], dataset_card=fixture["card"])
        self.assertEqual(result["status"], "PASS")
        repair = {"previous_response": fixture["initial_review"],
                  "errors": [{"code": code, "path": "response"}
                             for code in fixture["initial_validation"]["schema_failures"]]}
        audit = event.correction_progress(fixture["retry_review"], repair)
        self.assertEqual(audit["status"], "REJECT")
        self.assertIn("correction_protected_judgment_erased", [row["code"] for row in audit["errors"]])

    def test_all_scene_negative_types_and_composite_identities_survive_reordering(self):
        previous = {
            "new_event_or_progress": False, "consistent_with_history": False, "duplicates_existing_event": True,
            "checks": [{"development_index": 1, "realized": False, "evidence": [{"turn_ref": "NEW:T001", "quote": "A"}]},
                       {"development_index": 2, "realized": True, "evidence": [{"turn_ref": "NEW:T002", "quote": "B"}]}],
            "state_checks": [{"fact_id": "f1", "verdict": "contradicted"}],
            "event_checks": [{"constraint_id": "c1", "verdict": "contradicted"}],
            "card_checks": [{"kind": "event_function", "id": "same", "realized": False},
                            {"kind": "step", "id": "same", "realized": True}],
            "consistency_checks": [{"check_id": "identity", "verdict": "contradicted"},
                                   {"check_id": "time", "verdict": "consistent"}],
        }
        repair = {"previous_raw": previous, "errors": [{"code": "review_bad_quote", "path": "response"}]}
        reordered = deepcopy(previous)
        for key in ("checks", "card_checks", "consistency_checks"):
            reordered[key].reverse()
        reordered["new_explanation"] = "A complete re-review with rows reordered."
        self.assertEqual(event.correction_progress(reordered, repair)["status"], "PASS")
        for collection, field, value in [
            (None, "new_event_or_progress", True), (None, "consistent_with_history", True),
            (None, "duplicates_existing_event", False), ("checks", "realized", True),
            ("state_checks", "verdict", "consistent"), ("event_checks", "verdict", "consistent"),
            ("card_checks", "realized", True), ("consistency_checks", "verdict", "consistent")]:
            with self.subTest(collection=collection, field=field):
                erased = deepcopy(previous)
                owner = erased if collection is None else erased[collection][0]
                owner[field] = value
                self.assertIn("correction_protected_judgment_erased", [
                    row["code"] for row in event.correction_progress(erased, repair)["errors"]])
        deleted = deepcopy(previous)
        deleted["checks"][1]["evidence"] = []
        self.assertIn("correction_global_evidence_or_check_deleted", [
            row["code"] for row in event.correction_progress(deleted, repair)["errors"]])

    def test_exact_contract_invalid_owner_exemption_never_releases_other_negative(self):
        previous = {"plan_checks": [{"development_index": 1, "verdict": "revise", "reason": "absence only"},
                                    {"development_index": 2, "verdict": "revise", "reason": "concrete conflict"}],
                    "card_checks": [{"card_id": "c", "function_check": {"comparison": {
                        "request_acceptance": {"required": True, "verdict": "revise"}}}}]}
        repair = {"previous_raw": previous, "errors": [{"code": "event_state_history_absence_not_contradiction",
                                                         "path": "plan_checks[0].reason"}]}
        corrected = deepcopy(previous)
        corrected["plan_checks"][0].update(verdict="compatible", reason="Substantively reviewed as a coherent new event.")
        self.assertEqual(event.correction_progress(corrected, repair)["status"], "PASS")
        for other in ("plan", "nested", "required"):
            erased = deepcopy(corrected)
            if other == "plan": erased["plan_checks"][1]["verdict"] = "compatible"
            else:
                nested = erased["card_checks"][0]["function_check"]["comparison"]["request_acceptance"]
                nested["verdict" if other == "nested" else "required"] = "compatible" if other == "nested" else False
            self.assertEqual(event.correction_progress(erased, repair)["status"], "REJECT")


if __name__ == "__main__":
    unittest.main()
