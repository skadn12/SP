"""Offline regressions for retaining useful dialogue without rewriting audits."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from candidate_contract import CORE_CHECKS, decide_candidate


IDENTITY = {"root_id": "mock-root", "route": "dataset_new_event", "scene_id": "S01"}


def compact():
    return {**IDENTITY, "verdict": "acceptable", "core_checks": {
        name: {"status": "ok", "reason": "MOCK ONLY diagnostic finding.",
               "evidence_refs": ["NEW:T001"]} for name in CORE_CHECKS},
        "plan_observations": [], "summary": "MOCK ONLY summary."}


def decide(raw=None, **kwargs):
    return decide_candidate(duplicate_result={"status": "PASS"}, review_raw=raw, **kwargs)


class CandidateDecisionTests(unittest.TestCase):
    def assert_pending(self, decision):
        self.assertEqual(decision["disposition"], "retained_pending_review")
        self.assertTrue(decision["artifact_retained"])
        self.assertTrue(decision["candidate_pool_eligible"])
        self.assertTrue(decision["continue_candidate_history"])
        self.assertFalse(decision["training_admitted"])
        self.assertFalse(decision["training_pool_eligible"])
        self.assertFalse(decision["semantic_certified"])

    def test_complete_positive_is_pending_not_training(self):
        result = decide(compact(), expected_identity=IDENTITY)
        self.assert_pending(result)
        self.assertTrue(result["audit_complete"])
        self.assertEqual(result["raw_review_verdict"], "acceptable")

    def test_missing_reviewer_check_retained(self):
        report = compact()
        del report["core_checks"]["history_consistency"]
        result = decide(report)
        self.assert_pending(result)
        self.assertFalse(result["audit_complete"])

    def test_missing_historical_goal_checks_does_not_discard_legacy_scene(self):
        raw = {"checks": [{"development_index": 1, "realized": True}],
               "new_event_or_progress": True, "consistent_with_history": True,
               "duplicates_existing_event": False,
               "route_fulfillment": {"new_progress": True},
               "reason": "The summary incorrectly says execution finished."}
        validation = {"status": "REJECT", "schema_valid": False,
                      "schema_failures": ["review_history_goal_checks_required:route_fulfillment.history_goal_checks"],
                      "failures": ["review_history_goal_checks_required:route_fulfillment.history_goal_checks"]}
        before = deepcopy((raw, validation))
        result = decide(raw, review_validation=validation)
        self.assert_pending(result)
        self.assertFalse(result["audit_complete"])
        self.assertIsNone(result["raw_review_verdict"])
        self.assertEqual((raw, validation), before)
        self.assertEqual(validation["status"], "REJECT")

    def test_summary_and_partial_plan_coverage_are_not_gates(self):
        report = compact()
        report["verdict"] = "problem"
        report["summary"] = "REJECT: a misleading summary claim, preserved as raw."
        report["plan_observations"] = [{"development": 2, "realized": False}]
        result = decide(report)
        self.assert_pending(result)
        self.assertEqual(result["raw_review_verdict"], "problem")
        self.assertTrue(result["observations"])

    def test_transport_or_parse_failure_preserves_candidate(self):
        for kwargs in ({"review_error": "HTTP 503 after allowed retry"}, {},
                       {"review_raw": "not json"},
                       {"review_raw": '{"verdict":"acceptable","verdict":"problem"}'}):
            result = decide_candidate(duplicate_result={"status": "PASS"}, **kwargs)
            self.assert_pending(result)
            self.assertFalse(result["audit_complete"])

    def test_core_model_problem_quarantined_even_with_other_missing_fields(self):
        for name in CORE_CHECKS:
            with self.subTest(name=name):
                report = compact()
                report["core_checks"][name]["status"] = "problem"
                report["verdict"] = "problem"
                result = decide(report)
                self.assertEqual(result["disposition"], "quarantined")
                self.assertTrue(result["artifact_retained"])
                self.assertFalse(result["candidate_pool_eligible"])
                self.assertFalse(result["continue_candidate_history"])
                self.assertFalse(result["semantic_certified"])
                self.assertFalse(result["training_admitted"])
                self.assertEqual(result["content_risks"][0]["provenance"],
                                 "model_reported_not_independently_verified")

    def test_explicit_uncertainty_retains_for_review(self):
        report = compact()
        report["verdict"] = "uncertain"
        report["core_checks"]["history_consistency"]["status"] = "uncertain"
        result = decide(report)
        self.assert_pending(result)
        self.assertTrue(result["observations"])

    def test_mechanical_source_and_duplicate_failures_remain_hard(self):
        for kwargs in ({"mechanical_failures": ["speaker_not_allowed"]},
                       {"source_failures": ["card_source_sha_mismatch"]},
                       {"duplicate_result": {"status": "REJECT", "reasons": ["copied_turns"]}}):
            result = decide_candidate(review_raw=compact(), **kwargs)
            self.assertEqual(result["disposition"], "hard_reject")
            self.assertFalse(result["candidate_pool_eligible"])
            self.assertFalse(result["training_admitted"])
            self.assertTrue(result["artifact_retained"])

    def test_wrong_scene_review_does_not_quarantine_unrelated_dialogue(self):
        report = compact()
        report["scene_id"] = "S99"
        report["verdict"] = "problem"
        report["core_checks"]["actor_actions"]["status"] = "problem"
        result = decide(report, expected_identity=IDENTITY)
        self.assert_pending(result)
        self.assertFalse(result["audit_complete"])
        self.assertEqual(result["content_risks"], [])
        self.assertEqual(result["raw_review_verdict"], "problem")

    def test_compact_findings_core_problem_and_auxiliary_problem_differ(self):
        raw = {**IDENTITY, "verdict": "REJECT", "findings": [
            {"category": "summary", "verdict": "problem", "reason": "Bad summary.", "evidence": []}]}
        self.assert_pending(decide(raw))
        raw["findings"][0]["category"] = "actor_identity"
        result = decide(raw)
        self.assertEqual(result["disposition"], "quarantined")
        self.assertEqual(result["raw_review_verdict"], "REJECT")

    def test_old_unperformed_action_is_plan_drift_but_wrong_actor_is_risk(self):
        raw = {"consistent_with_history": True, "new_event_or_progress": True,
               "duplicates_existing_event": False, "checks": [{"realized": False,
                   "actor_checks": [{"satisfied": False, "expected_actor": "James", "actor": "James"}]}]}
        self.assert_pending(decide(raw))
        raw["checks"][0]["actor_checks"][0]["actor"] = "Emily"
        self.assertEqual(decide(raw)["disposition"], "quarantined")

    def test_legacy_explicit_history_conflict_quarantined_but_soft_findings_retained(self):
        raw = {"consistent_with_history": False, "new_event_or_progress": True,
               "duplicates_existing_event": False,
               "state_checks": [{"fact_id": "f1", "verdict": "contradicted"}]}
        self.assertEqual(decide(raw)["disposition"], "quarantined")
        validation = {"history_consistency_status": "UNRESOLVED_SOFT_FINDINGS",
                      "soft_findings": [{"fact_id": "f1"}]}
        self.assert_pending(decide(raw, review_validation=validation))

    def test_inputs_and_raw_verdict_immutable(self):
        raw = compact()
        raw["verdict"] = "problem"
        original = deepcopy(raw)
        first = decide(raw)
        first["observations"].append({"external": "mutation"})
        self.assertEqual(raw, original)
        self.assertEqual(decide(raw)["raw_review_sha256"], first["raw_review_sha256"])

    def test_unusable_review_preserves_verdict_without_applying_findings(self):
        report = compact()
        report["verdict"] = "problem"
        report["core_checks"]["actor_actions"]["status"] = "problem"
        result = decide(report, review_error="scene_identity_mismatch")
        self.assert_pending(result)
        self.assertEqual(result["raw_review_verdict"], "problem")
        self.assertEqual(result["content_risks"], [])

    def test_malformed_reviewer_leaf_types_do_not_block_candidate(self):
        report = compact()
        report["verdict"] = []
        report["core_checks"]["source_core"]["status"] = {}
        self.assert_pending(decide(report))
        report = {"verdict": [], "findings": [
            {"category": [], "verdict": {}, "evidence": {}}]}
        self.assert_pending(decide(report))

    def test_recorded_v37_positive_survives_without_becoming_new_training(self):
        delivery = Path(__file__).resolve().parent.parent
        path = delivery / "h4_sequential_ab_v37/experiments/fixed_soda_a3_01/requests/007-cc-train-episode-104814-B-S01-review-1-1/REVIEW.json"
        if not path.exists():
            self.skipTest("frozen historical artifact is unavailable")
        original = path.read_bytes()
        validation = json.loads(original)
        raw = validation.get("raw_parsed") or validation["parsed"]
        result = decide(raw, review_validation=validation)
        self.assert_pending(result)
        self.assertEqual(path.read_bytes(), original)

    def test_recorded_v40_missing_goal_report_keeps_original_candidate(self):
        delivery = Path(__file__).resolve().parent.parent
        directory = delivery / "h4_sequential_ab_v40/probes/two_stage_01/requests/002-full-review-original_decision_scene"
        if not directory.exists():
            self.skipTest("frozen historical artifact is unavailable")
        response_path, evaluation_path = directory / "RESPONSE.json", directory / "EVALUATION.json"
        before = (response_path.read_bytes(), evaluation_path.read_bytes())
        response, evaluation = (json.loads(blob) for blob in before)
        self.assertEqual(evaluation["review"]["status"], "REJECT")
        self.assertIn("review_history_goal_checks_required:route_fulfillment.history_goal_checks",
                      evaluation["review"]["schema_failures"])
        result = decide(response["text"], review_validation=evaluation["review"])
        self.assert_pending(result)
        self.assertFalse(result["audit_complete"])
        self.assertEqual((response_path.read_bytes(), evaluation_path.read_bytes()), before)


if __name__ == "__main__":
    unittest.main()
