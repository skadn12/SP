"""Request provenance tests use temporary offline fixtures only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import candidate_runner as c
from candidate_request_audit import audit_requests


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def append_fixture_feedback(data, *_args, **_kwargs):
    data = deepcopy(data)
    data["messages"][0]["content"] += "\nFIXTURE_FEEDBACK_SUFFIX\n{\"instruction\":\"preserve\"}"
    return data, {"fixture": "offline_feedback_receipt"}


class CandidateRequestAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.feedback = patch.object(c.r.base.feedback, "append_feedback_to_request", append_fixture_feedback)
        self.feedback.start()
        self.addCleanup(self.feedback.stop)
        write(self.run / "feedback/COMPILED_FEEDBACK.json", {"target_solver_identity_sha256": "fixture"})
        self.baseline = [{"turn_ref": "BASE:T001", "speaker": "Fixture", "text": "A frozen fixture history."}]
        self.history = deepcopy(self.baseline)
        self.route = c.r.ROUTES[0]
        self.slot = self.run / "candidates" / c.ROOT / "A" / "S01"
        write(self.slot / "INPUT.json", {**c.identity(c.ROOT, self.route, "S01"), "history": self.history,
              "card": None, "baseline_sha256": c.r.base.digest(self.baseline)})
        write(self.slot / "PLAN_ADVISORY.json", {"plan": None, "issue": "invalid_fixture_plan", "blocks_generation": False})
        self.realization = {**c.identity(c.ROOT, self.route, "S01"), "status": "scene",
                            "turns": [{"speaker": "Fixture", "text": "A new fixture response."}]}
        self.review = json.dumps({**c.identity(c.ROOT, self.route, "S01"), "verdict": "uncertain"})
        self.planner = self.request(1, "planner", response="{}")
        self.realizer = self.request(2, "realizer", response=json.dumps(self.realization))
        self.reviewer = self.request(3, "reviewer", response=self.review, realization=self.realization)
        self.records = [{"candidate_id": "offline/A/S01/1", "route": self.route, "scene_id": "S01",
                         "request": str(self.realizer), "review_request": str(self.reviewer),
                         "raw_review": self.review, "realization": self.realization,
                         "disposition": "retained_pending_review", "mechanical_failures": [],
                         "duplicate_result": {"status": "PASS", "reasons": []}}]

    def request(self, number, role, response=None, realization=None, repair=None):
        folder = self.run / "requests" / f"{number:03d}-{role}"
        request, proof = c.request(self.run, c.ROOT, self.route, "S01", role,
                                   self.history, self.baseline, realization=realization, repair=repair)
        write(folder / "REQUEST.json", request)
        write(folder / "PROOF.json", {**proof, "payload_sha256": c.r.base.digest(request), "capacity": {"status": "PASS"}})
        write(folder / "RESERVATION.json", {"request_number": number})
        if response is not None:
            write(folder / "RESPONSE.json", {"text": response})
            write(folder / "HTTP_RESPONSE_01.json", {"text": response})
        return folder

    def audit(self):
        return audit_requests(self.run, self.records, self.baseline)

    def change_context(self, directory, updater):
        request = c.r.base.read(directory / "REQUEST.json")
        prefix, encoded = request["messages"][0]["content"].split("\nCANDIDATE_INPUT_JSON\n", 1)
        context, end = json.JSONDecoder().raw_decode(encoded)
        updater(context)
        request["messages"][0]["content"] = prefix + "\nCANDIDATE_INPUT_JSON\n" + json.dumps(context, ensure_ascii=False) + encoded[end:]
        write(directory / "REQUEST.json", request)
        proof = c.r.base.read(directory / "PROOF.json")
        proof["payload_sha256"] = c.r.base.digest(request)
        write(directory / "PROOF.json", proof)

    def test_successful_responses_are_fully_replayed_with_feedback_suffix(self):
        result = self.audit()
        self.assertEqual(result["status"], "REQUEST_ARTIFACTS_REPLAYED")
        self.assertEqual(result["roles"], {"planner": 1, "realizer": 1, "reviewer": 1})
        self.assertFalse(result["semantic_certified"])

    def test_omitted_realizer_response_cannot_disappear_from_counts(self):
        self.records = []
        with self.assertRaisesRegex(ValueError, "successful_realizer_response_missing_record"):
            self.audit()

    def test_review_response_cannot_be_silently_omitted(self):
        self.records[0]["review_request"] = None
        self.records[0]["raw_review"] = None
        with self.assertRaisesRegex(ValueError, "successful_reviewer_response_missing_record"):
            self.audit()

    def test_truncated_actual_history_is_detected_even_with_updated_payload_hash(self):
        self.change_context(self.realizer, lambda context: context.update(cumulative_history=[]))
        with self.assertRaisesRegex(ValueError, "candidate_request_full_context_changed"):
            self.audit()

    def test_reviewer_cannot_see_different_dialogue(self):
        self.change_context(self.reviewer, lambda context: context["new_scene"]["turns"][0].update(text="Changed fixture."))
        with self.assertRaisesRegex(ValueError, "candidate_request_full_context_changed"):
            self.audit()

    def test_proof_hash_does_not_replace_actual_input_replay(self):
        proof = c.r.base.read(self.realizer / "PROOF.json")
        proof["history_sha256"] = "0" * 64
        write(self.realizer / "PROOF.json", proof)
        with self.assertRaisesRegex(ValueError, "candidate_request_proof_changed"):
            self.audit()

    def test_response_must_match_actual_http_body(self):
        write(self.realizer / "HTTP_RESPONSE_01.json", {"text": "different raw body"})
        with self.assertRaisesRegex(ValueError, "response_differs_from_raw_http_body"):
            self.audit()

    def test_no_proof_is_only_allowed_for_local_failure_without_http(self):
        directory = self.run / "requests/004-local-error"
        write(directory / "REQUEST.json", {})
        write(directory / "LOCAL_ERROR.json", {"http_attempted": False})
        self.assertEqual(self.audit()["successful_realizer_responses"], 1)
        write(directory / "HTTP_ATTEMPT.json", {})
        with self.assertRaisesRegex(ValueError, "request_missing_proof"):
            self.audit()

    def test_reused_diagnostic_request_path_cannot_claim_fresh(self):
        self.records[0]["request"] = str(self.run / "diagnostics/old-response")
        with self.assertRaisesRegex(ValueError, "candidate_request_outside_current_run"):
            self.audit()

    def test_valid_mechanical_retry_reconstructs_exact_repair(self):
        self.records[0]["disposition"] = "hard_reject"
        self.records[0]["mechanical_failures"] = ["fixture_mechanical_failure"]
        self.records[0]["review_request"] = None
        self.records[0]["raw_review"] = None
        # Remove fixture reviewer receipts; only temporary test files are touched.
        for path in self.reviewer.iterdir():
            path.unlink()
        self.reviewer.rmdir()
        repair = {"instruction": "Fix only the mechanical/duplicate issues, preserving source and history.",
                  "failures": ["fixture_mechanical_failure"], "duplicate_reasons": [],
                  "previous_response": json.dumps(self.realization)}
        second = self.request(3, "realizer", response=json.dumps(self.realization), repair=repair)
        self.records.append({**self.records[0], "candidate_id": "offline/A/S01/2", "request": str(second),
                             "disposition": "retained_pending_review", "mechanical_failures": []})
        self.assertEqual(self.audit()["successful_realizer_responses"], 2)
        self.change_context(second, lambda context: context["mechanical_repair"].update(failures=[]))
        with self.assertRaisesRegex(ValueError, "candidate_request_full_context_changed"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
