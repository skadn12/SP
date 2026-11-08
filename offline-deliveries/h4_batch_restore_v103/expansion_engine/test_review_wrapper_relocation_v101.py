"""Exact wrapper relocation regression; never rewrite or promote archived output.

The v100 real 001 comparison was returned at function_check instead of nested
inside function_check.comparison. The real 002 wrapper fix preserved its duties.
Only the exact validator pair may relocate those duties; all original complete
validation and negative/evidence protections still apply.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import correction_contract as correction
import event_state as event
import runner


V100 = Path(__file__).resolve().parents[2] / "h4_bound_card_review_feedback_v100"
REQUESTS = V100 / "regression/requests"
FIRST = REQUESTS / "001-cc-train-episode-19053-B-S01-event-state-0-1"
SECOND = REQUESTS / "002-cc-train-episode-19053-B-S01-event-state-0-2"


def read(path):
    return json.loads(path.read_text())


def hashes(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


class RawCaller:
    """Production extract_event_state caller boundary, with exact raw strings."""
    def __init__(self, run, raw):
        self.run, self.raw, self.calls = Path(run), list(raw), []

    def call(self, label, payload, proof):
        if proof["role"] != "event_state_extractor":
            raise AssertionError("Only reviewer-format retry is authorized")
        directory = self.run / "requests" / (f"{len(self.calls) + 1:03d}-" + label)
        directory.mkdir(parents=True)
        self.calls.append({"payload": deepcopy(payload), "proof": deepcopy(proof), "directory": directory})
        return self.raw.pop(0), directory


class ReviewWrapperRelocationV101Tests(unittest.TestCase):
    def setUp(self):
        self.paths = [FIRST / "REQUEST.json", FIRST / "RESPONSE.json", FIRST / "VALIDATION.json",
                      SECOND / "REQUEST.json", SECOND / "RESPONSE.json", SECOND / "VALIDATION.json"]
        self.before_hashes = hashes(self.paths)
        self.first_raw = read(FIRST / "RESPONSE.json")["text"]
        self.second_raw = read(SECOND / "RESPONSE.json")["text"]
        self.first, self.second = json.loads(self.first_raw), json.loads(self.second_raw)
        payload = read(SECOND / "REQUEST.json")
        self.context, _ = json.JSONDecoder().raw_decode(payload["messages"][0]["content"].split("EVENT_STATE_INPUT_JSON\n", 1)[1])
        self.errors = read(FIRST / "VALIDATION.json")["errors"]
        self.repair = deepcopy(self.context["targeted_repair"])

    def tearDown(self):
        self.assertEqual(self.before_hashes, hashes(self.paths))

    def validate(self, value):
        c = self.context
        return event.validate(value, c["root_id"], c["route"], c["scene_plan"],
                              c["cumulative_history"], c["dataset_cards"])

    def repair_for(self, *, previous=None, errors=None):
        return event.build_repair(event.EventStateValidationError(deepcopy(self.errors if errors is None else errors)),
                                  deepcopy(self.first if previous is None else previous),
                                  history=self.context["cumulative_history"])

    def assert_rejected(self, value, repair=None, *, code=None):
        audit = event.correction_progress(value, self.repair if repair is None else repair)
        self.assertEqual(audit["status"], "REJECT", audit)
        if code is not None:
            self.assertIn(code, [row["code"] for row in audit["errors"]])
        with self.assertRaises(event.EventStateValidationError):
            event.require_correction_progress(audit)
        return audit

    def comparison(self, value):
        return value["card_checks"][0]["function_check"]["comparison"]

    def test_real_v100_wrapper_fix_progresses_and_remains_binding_replan(self):
        # Expose the original failure, then test the same immutable actual raw
        # pair with the v101 gate. The old rejection is not silently overwritten.
        spec = importlib.util.spec_from_file_location("frozen_v100_wrapper_correction", V100 / "expansion_engine/correction_contract.py")
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        self.assertEqual(old.assess(self.second_raw, self.repair)["status"], "REJECT")
        self.assertEqual(read(SECOND / "FULL_VALIDATION_BEFORE_CORRECTION_GATE.json")["status"], "PASS")
        result = self.validate(self.second_raw)
        self.assertTrue(event.requires_replan(result))
        audit = event.correction_progress(self.second_raw, self.repair)
        self.assertEqual(audit["status"], "PASS", audit)
        self.assertTrue(audit["full_validation_required"])
        self.assertFalse(audit["semantic_certified"])
        event.require_correction_progress(audit)
        self.assertEqual(self.first["card_checks"][0]["function_check"], self.comparison(self.second))

    def test_original_misplaced_response_still_fails_complete_validator(self):
        with self.assertRaises(event.EventStateValidationError) as caught:
            self.validate(self.first_raw)
        self.assertIn("event_state_fields_invalid", [row["code"] for row in caught.exception.errors])
        self.assert_rejected(self.first_raw)

    def test_relocated_required_true_cannot_be_flipped_or_omitted(self):
        for mode in ("false", "delete"):
            with self.subTest(mode=mode):
                changed = deepcopy(self.second)
                request = self.comparison(changed)["request_acceptance"]
                if mode == "false":
                    request.update(required=False, request_evidence=[], acceptance_evidence=[])
                    self.validate(changed)  # Mechanically legal alone, still erases a protected duty.
                else:
                    del request["required"]
                    with self.assertRaises(event.EventStateValidationError): self.validate(changed)
                self.assert_rejected(changed, code="correction_protected_judgment_erased")

    def test_relocated_nested_negative_cannot_be_made_positive(self):
        previous = deepcopy(self.first)
        previous["card_checks"][0]["function_check"]["experience_state"]["verdict"] = "revise"
        repair = self.repair_for(previous=previous)
        corrected = deepcopy(self.second)
        self.comparison(corrected)["experience_state"]["verdict"] = "revise"
        self.validate(corrected)
        self.assertEqual(event.correction_progress(corrected, repair)["status"], "PASS")
        self.comparison(corrected)["experience_state"]["verdict"] = "compatible"
        self.validate(corrected)
        self.assert_rejected(corrected, repair, code="correction_protected_judgment_erased")

    def test_unrelated_step_negative_remains_binding_after_wrapper_move(self):
        changed = deepcopy(self.second)
        changed["card_checks"][0]["step_checks"][0]["verdict"] = "compatible"
        self.validate(changed)
        self.assert_rejected(changed, code="correction_protected_judgment_erased")

    def test_source_reference_counts_are_preserved_at_relocated_path(self):
        for mode in ("empty", "shorten"):
            with self.subTest(mode=mode):
                changed = deepcopy(self.second)
                refs = self.comparison(changed)["purpose"]["source_paths"]
                self.comparison(changed)["purpose"]["source_paths"] = [] if mode == "empty" else refs[:-1]
                self.validate(changed)
                audit = self.assert_rejected(changed, code="correction_global_evidence_or_check_deleted")
                self.assertTrue(any(row.get("minimum_count") == 4 and "comparison" in row["path"]
                                    for row in audit["errors"]), audit)

    def test_history_evidence_counts_are_preserved_at_relocated_path(self):
        changed = deepcopy(self.second)
        self.comparison(changed)["nearest_old_event"]["history_evidence"].pop()
        self.validate(changed)
        self.assert_rejected(changed, code="correction_global_evidence_or_check_deleted")

    def test_moved_binding_does_not_bypass_current_card_hash_validation(self):
        changed = deepcopy(self.second)
        self.comparison(changed)["source_binding"]["projection_sha256"] = "0" * 64
        with self.assertRaises(event.EventStateValidationError) as caught:
            self.validate(changed)
        self.assertIn("card_source_binding_mismatch", [row["code"] for row in caught.exception.errors])
        # Progress alone never certifies this mutated response; full validation
        # still rejects it even if the wrapper relocation itself was responsive.
        self.assertTrue(event.correction_progress(changed, self.repair)["full_validation_required"])

    def test_relocation_requires_both_exact_validator_error_records(self):
        wrapper = next(row for row in self.errors if row["path"] == "card_checks[0].function_check")
        missing = next(row for row in self.errors if row["path"] == "card_checks[0].function_check.comparison")
        for label, errors in [("none", []), ("wrapper_only", [wrapper]), ("missing_only", [missing])]:
            with self.subTest(label=label):
                self.assert_rejected(self.second, self.repair_for(errors=errors))
        for mode in ("wrapper_code", "wrapper_path", "wrapper_allowed", "missing_code", "missing_path", "missing_actual", "allowed_shape"):
            with self.subTest(mode=mode):
                errors = deepcopy(self.errors)
                w = next(row for row in errors if row["path"] == "card_checks[0].function_check")
                m = next(row for row in errors if row["path"] == "card_checks[0].function_check.comparison")
                if mode == "wrapper_code": w["code"] = "generic_schema_failure"
                elif mode == "wrapper_path": w["path"] = "metadata.function_check"
                elif mode == "wrapper_allowed": w["allowed"] = ["comparison"]
                elif mode == "missing_code": m["code"] = "generic_missing_field"
                elif mode == "missing_path": m["path"] = "card_checks[1].function_check.comparison"
                elif mode == "missing_actual": m["actual"] = {}
                else: m["allowed"].remove("source_binding")
                self.assert_rejected(self.second, self.repair_for(errors=errors))

    def test_relocation_requires_exact_original_comparison_keyset_without_wrapper(self):
        for mode in ("extra", "already_has_comparison", "missing"):
            with self.subTest(mode=mode):
                previous = deepcopy(self.first)
                old = previous["card_checks"][0]["function_check"]
                if mode == "extra": old["foreign"] = {"required": True}
                elif mode == "already_has_comparison": old["comparison"] = deepcopy(old)
                else: old.pop("source_binding")
                self.assert_rejected(self.second, self.repair_for(previous=previous))

    def test_no_recursive_search_or_supplied_contract_can_authorize_foreign_move(self):
        previous, changed = deepcopy(self.first), deepcopy(self.second)
        previous["diagnostic"] = {"function_check": {"request_acceptance": {"required": True},
                                                       "nearest_old_event": {"history_evidence": [{"turn_ref": "not-actual", "quote": "sentinel"}]}}}
        changed["diagnostic"] = {"function_check": {"comparison": deepcopy(previous["diagnostic"]["function_check"])}}
        repair = self.repair_for(previous=previous)
        repair["correction_contract"] = {"protected_judgments": [], "evidence_obligations": [], "targets": []}
        self.assert_rejected(changed, repair, code="correction_protected_judgment_erased")

    def test_exact_error_for_one_card_never_authorizes_adjacent_card_relocation(self):
        previous, changed = deepcopy(self.first), deepcopy(self.second)
        adjacent = deepcopy(previous["card_checks"][0])
        adjacent["card_id"] = "adjacent-card-without-exact-wrapper-error"
        previous["card_checks"].append(adjacent)
        moved = deepcopy(changed["card_checks"][0])
        moved["card_id"] = adjacent["card_id"]
        changed["card_checks"].append(moved)
        repair = self.repair_for(previous=previous)
        audit = self.assert_rejected(changed, repair, code="correction_protected_judgment_erased")
        self.assertTrue(any({"identity": {"card_id": adjacent["card_id"]}} in error["path"]
                            for error in audit["errors"]), audit)

    def test_only_obligations_relocate_error_targets_stay_at_original_paths(self):
        contract = correction.build_contract(self.errors, self.first_raw)
        old_spec = importlib.util.spec_from_file_location("frozen_v100_wrapper_contract_target_reference", V100 / "expansion_engine/correction_contract.py")
        old = importlib.util.module_from_spec(old_spec); old_spec.loader.exec_module(old)
        baseline = old.build_contract(self.errors, self.first_raw)
        self.assertEqual(contract["targets"], baseline["targets"])
        self.assertEqual(len(contract["protected_judgments"]), len(baseline["protected_judgments"]))
        self.assertEqual(len(contract["evidence_obligations"]), len(baseline["evidence_obligations"]))
        required = next(row for row in contract["protected_judgments"] if row["kind"] == "declared_applicability")
        self.assertEqual(required["path"][-4:], ["function_check", "comparison", "request_acceptance", "required"])
        self.assertEqual(required["value"], True)

    def test_production_reviewer_retry_uses_actual_raw_pair_with_existing_budget(self):
        c = self.context
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            caller = RawCaller(run, [self.first_raw, self.second_raw])
            with patch.object(runner, "branch_history_ok", return_value=c["cumulative_history"]):
                result = runner.extract_event_state(caller, run, c["root_id"], c["route"], c["cumulative_history"],
                                                    c["dataset_cards"], c["scene_plan"], c["latest_state"], plan_attempt=0)
            self.assertEqual(len(caller.calls), runner.EVENT_STATE_ATTEMPTS)
            self.assertEqual(caller.raw, [])
            self.assertIsNotNone(result)
            self.assertTrue(event.requires_replan(result))
            last = caller.calls[-1]["directory"]
            self.assertEqual(read(last / "VALIDATION.json")["status"], "PASS")
            self.assertEqual(read(last / "CORRECTION_PROGRESS.json")["status"], "PASS")
            self.assertTrue(all(call["payload"]["max_tokens"] == 4096 for call in caller.calls))

    def test_production_reviewer_retry_still_rejects_bad_current_source_hash(self):
        c = self.context
        changed = deepcopy(self.second)
        self.comparison(changed)["source_binding"]["projection_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            caller = RawCaller(run, [self.first_raw, json.dumps(changed)])
            with patch.object(runner, "branch_history_ok", return_value=c["cumulative_history"]):
                result = runner.extract_event_state(caller, run, c["root_id"], c["route"], c["cumulative_history"],
                                                    c["dataset_cards"], c["scene_plan"], c["latest_state"], plan_attempt=0)
            self.assertIsNone(result)
            self.assertEqual(len(caller.calls), runner.EVENT_STATE_ATTEMPTS)
            validation = read(caller.calls[-1]["directory"] / "VALIDATION.json")
            self.assertEqual(validation["status"], "REJECT")
            self.assertIn("card_source_binding_mismatch", [error["code"] for error in validation["errors"]])


if __name__ == "__main__":
    unittest.main()
