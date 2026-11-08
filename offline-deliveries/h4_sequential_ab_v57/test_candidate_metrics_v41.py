"""Offline accounting invariants; fixtures are never production dialogue."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import candidate_metrics as metrics


def count_words(text):
    return len(text.split())


def record(candidate_id, disposition="retained", route="B", provenance="fresh_collection"):
    return {
        "candidate_id": candidate_id, "route": route, "disposition": disposition,
        "provenance": provenance, "realization": {"turns": [
            {"speaker": "fixture", "text": "one two"},
            {"speaker": "fixture", "text": "three four five"},
        ]}, "raw_review": {"status": "REJECT", "failures": ["missing_history_goal_checks"]},
        "training_pool_eligible": False,
    }


class CandidateMetricsTests(unittest.TestCase):
    def summary(self, records, **kwargs):
        return metrics.summarize_records(records, count_text_tokens=count_words, **kwargs)

    def test_generated_dispositions_and_routes_are_separate(self):
        records = [record("a-retained", route="old_event_continuation"),
                   record("b-retained", disposition="RETAINED_PENDING_REVIEW"),
                   record("b-quarantined", disposition="QUARANTINED"),
                   record("b-rejected", disposition="REJECTED")]
        result = self.summary(records, baseline_turns=[{"text": "frozen base"}])
        self.assertEqual(result["generated"], {"scenes": 4, "turns": 8, "body_tokens": 20})
        self.assertEqual(result["retained"]["body_tokens"], 10)
        self.assertEqual(result["quarantined"]["body_tokens"], 5)
        self.assertEqual(result["rejected"]["body_tokens"], 5)
        self.assertEqual(result["admitted"], {"scenes": 0, "turns": 0, "body_tokens": 0})
        self.assertEqual(result["by_route"]["A"]["generated"]["body_tokens"], 5)
        self.assertEqual(result["by_route"]["B"]["generated"]["body_tokens"], 15)
        self.assertEqual(result["formal_accepted_cumulative"], {"turns": 1, "body_tokens": 2})
        self.assertEqual(result["provisional_candidate_cumulative"], {"turns": 5, "body_tokens": 12})
        self.assertFalse(result["training_pool_eligible"])

    def test_reviewer_reject_is_preserved_while_dialogue_is_retained(self):
        source = record("missing-auxiliary-review-field")
        snapshot = deepcopy(source)
        result = self.summary([source])
        self.assertEqual(result["retained"]["scenes"], 1)
        self.assertEqual(result["candidate_rows"][0]["raw_review_status"], "REJECT")
        self.assertEqual(source, snapshot)
        self.assertEqual(result["candidate_rows"][0]["raw_review_sha256"], metrics.digest(source["raw_review"]))

    def test_rewritten_raw_verdict_is_rejected(self):
        source = record("bad-status")
        source["review_status"] = "PASS"
        with self.assertRaisesRegex(ValueError, "raw_review_verdict_rewritten"):
            self.summary([source])

    def test_diagnostic_replay_never_increases_fresh_yield(self):
        result = self.summary([record("live"), record("v37-replay", provenance="diagnostic_replay")])
        self.assertEqual(result["generated"]["body_tokens"], 5)
        self.assertEqual(result["retained"]["body_tokens"], 5)
        self.assertEqual(result["diagnostic_replay"]["generated"]["body_tokens"], 5)
        self.assertEqual(result["diagnostic_replay"]["admitted"]["body_tokens"], 0)
        self.assertEqual(result["provisional_candidate_cumulative"]["body_tokens"], 5)

    def test_offline_diagnostic_run_cannot_claim_fresh(self):
        with self.assertRaisesRegex(ValueError, "diagnostic_run_cannot_claim_fresh_candidate"):
            self.summary([record("not-live")], scope="offline_diagnostic")
        result = self.summary([record("replay", provenance="diagnostic_replay")], scope="offline_diagnostic")
        self.assertEqual(result["generated"]["scenes"], 0)

    def test_candidate_provenance_must_be_explicit(self):
        source = record("no-origin")
        del source["provenance"]
        with self.assertRaisesRegex(ValueError, "candidate_provenance_required"):
            self.summary([source])

    def test_training_admission_is_never_available_in_collection(self):
        for key in ("training_pool_eligible", "admitted"):
            with self.subTest(key=key):
                source = record("would-be-admitted")
                source[key] = True
                with self.assertRaisesRegex(ValueError, "candidate_collection_cannot_admit_training_data"):
                    self.summary([source])

    def test_malformed_output_counts_only_readable_text_as_rejected(self):
        source = record("malformed", disposition="rejected")
        source["realization"]["turns"].append({"text": None})
        result = self.summary([source])
        self.assertEqual(result["generated"]["body_tokens"], 5)
        self.assertEqual(result["rejected"]["body_tokens"], 5)
        self.assertEqual(result["attempts_with_unmeasurable_body"]["fresh_collection"], 1)
        source["disposition"] = "retained"
        with self.assertRaisesRegex(ValueError, "retained_candidate_has_unreadable_dialogue"):
            self.summary([source])

    def test_broken_json_cannot_be_estimated_as_dialogue_tokens(self):
        source = record("broken-json", disposition="rejected")
        source["realization"] = None
        source["raw_response"] = '{"turns": [{"text": "cut off'
        result = self.summary([source])
        self.assertEqual(result["generated"]["body_tokens"], 0)
        self.assertEqual(result["attempt_counts"]["fresh_collection"], 1)
        self.assertEqual(result["attempts_with_unmeasurable_body"]["fresh_collection"], 1)

    def test_no_scene_is_an_attempt_but_zero_dialogue(self):
        source = record("no-scene", disposition="rejected")
        source["realization"] = {"status": "no_scene", "turns": []}
        result = self.summary([source])
        self.assertEqual(result["generated"]["scenes"], 0)
        self.assertEqual(result["attempt_counts"]["fresh_collection"], 1)

    def test_missing_reviewer_is_allowed_without_program_pass(self):
        source = record("review-transport-error")
        source["raw_review"] = None
        result = self.summary([source])
        self.assertIsNone(result["candidate_rows"][0]["raw_review_status"])
        self.assertEqual(result["retained"]["scenes"], 1)

    def test_duplicate_attempt_identifier_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid_or_duplicate_candidate_id"):
            self.summary([record("same"), record("same")])

    def test_replay_detects_changed_count_or_unchanged_count_content_tamper(self):
        records = [record("bound")]
        result = self.summary(records)
        check = metrics.verify_summary(result, records, count_text_tokens=count_words)
        self.assertEqual(check["status"], "ARTIFACTS_CONSISTENT")
        changed_count = deepcopy(result)
        changed_count["retained"]["body_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "candidate_accounting_or_artifact_binding_changed"):
            metrics.verify_summary(changed_count, records, count_text_tokens=count_words)
        records[0]["realization"]["turns"][0]["text"] = "same length"
        with self.assertRaisesRegex(ValueError, "candidate_accounting_or_artifact_binding_changed"):
            metrics.verify_summary(result, records, count_text_tokens=count_words)

    def test_raw_artifact_binding_detects_review_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "RESPONSE.json"
            path.write_text(json.dumps({"status": "REJECT"}))
            bindings = metrics.bind_files([path])
            self.assertEqual(metrics.verify_files(bindings), 1)
            path.write_text(json.dumps({"status": "PASS"}))
            with self.assertRaisesRegex(ValueError, "candidate_artifact_changed"):
                metrics.verify_files(bindings)


if __name__ == "__main__":
    unittest.main()
