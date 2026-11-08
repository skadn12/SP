"""CPU-only regression checks for the observable-reason feedback contract.

Fixtures are synthetic, including the review records; these checks neither
perform nor claim semantic review of actual model output.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import repair


class ReasonContractChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="reason-interface-selfcheck-")
        self.addCleanup(self.temp.cleanup)
        raw = Path(self.temp.name) / "synthetic_raw.json"
        self.raw_path = raw
        self.row = {"labels": ["A", "B"], "question_type": "ordering"}
        self.schema = repair.schema(self.row["labels"], ["T1", "T2"])
        self.parsed = {
            "item_evidence": {
                "A": {"judgment": "supported", "turn_refs": ["T1"],
                      "reason": "Synthetic item marker: the first event appears at T1."},
                "B": {"judgment": "supported", "turn_refs": ["T2"],
                      "reason": "The second event appears at T2."}},
            "answer_reason": "Synthetic relation marker: T1 precedes T2, so A precedes B.",
            "final_answer": "<box>A B</box>"}
        raw.write_text(json.dumps({"raw_output": json.dumps(self.parsed), "status": "RETURNED", "finish_reason": "stop"}))
        self.raw_binding = repair.bind(raw)
        self.source = Path(self.temp.name) / "synthetic_source.json"
        self.source.write_text(json.dumps({"root_id": "synthetic_train", "turns": [
            {"turn_ref": "T1", "speaker": "S", "text": "First event."},
            {"turn_ref": "T2", "speaker": "S", "text": "Second event."}]}))
        self.row.update(request_id="synthetic-interface-fixture", candidate_id="synthetic_candidate", root_id="synthetic_train", question="Order the reported events.", display={"A": "First event.", "B": "Second event."}, source_path=str(self.source), source_sha256=repair.sha(self.source))
        self.schema_path = Path(self.temp.name) / "synthetic_schema.json"
        self.schema_path.write_text(json.dumps(self.schema))
        self.row["schema_path"] = str(self.schema_path)

    def validate(self, obj):
        return repair.validate(json.dumps(obj), self.row, self.schema)

    def rejected(self, obj):
        with self.assertRaises(Exception):
            self.validate(obj)

    def packet(self):
        return {"requests": [{
            "request_id": "synthetic-interface-fixture",
            "candidate_id": self.row["candidate_id"], "root_id": self.row["root_id"],
            "question_type": self.row["question_type"], "request": self.row,
            "source_binding": repair.bind(self.source), "schema_binding": repair.bind(self.schema_path),
            "status": "STRUCTURALLY_VALID_REASONS_SAVED",
            "parsed": copy.deepcopy(self.parsed),
            "raw_binding": self.raw_binding,
            "mechanical_consistency": {"answer_matches_cited_order": True}}]}

    def reviews(self):
        return [{"request_id": "synthetic-interface-fixture",
                 "raw_binding": self.raw_binding,
                 "review_complete": True, "reviewer_type": "ai_agent",
                 "item_reviews": {
                     label: {"source_support": "pass", "reason_support": "pass",
                             "notes": "Synthetic fixture for reader test only."}
                     for label in self.row["labels"]},
                 "answer_reason_consistency": "pass"}]

    def test_valid_box_and_reasons(self):
        parsed, labels = self.validate(self.parsed)
        self.assertEqual(labels, ["A", "B"])
        self.assertEqual(parsed, self.parsed)

    def test_old_evidence_only_output_rejected(self):
        old = copy.deepcopy(self.parsed)
        for item in old["item_evidence"].values():
            del item["reason"]
        self.rejected(old)

    def test_missing_answer_reason_rejected(self):
        obj = copy.deepcopy(self.parsed)
        del obj["answer_reason"]
        self.rejected(obj)

    def test_whitespace_reasons_rejected(self):
        obj = copy.deepcopy(self.parsed)
        obj["item_evidence"]["A"]["reason"] = "  "
        self.rejected(obj)
        obj = copy.deepcopy(self.parsed)
        obj["answer_reason"] = "  "
        self.rejected(obj)

    def test_wrong_box_rejected(self):
        for value in ("A B", "<box>A B", "<box>A B</box> trailing", "<box>A A</box>"):
            with self.subTest(value=value):
                obj = copy.deepcopy(self.parsed)
                obj["final_answer"] = value
                self.rejected(obj)

    def test_answer_before_reasons_rejected(self):
        self.rejected({"final_answer": self.parsed["final_answer"],
                       "item_evidence": self.parsed["item_evidence"],
                       "answer_reason": self.parsed["answer_reason"]})

    def test_incomplete_ordering_rejected(self):
        obj = copy.deepcopy(self.parsed)
        obj["final_answer"] = "<box>A</box>"
        self.rejected(obj)

    def test_unknown_reference_rejected(self):
        obj = copy.deepcopy(self.parsed)
        obj["item_evidence"]["A"]["turn_refs"] = ["INVENTED"]
        self.rejected(obj)

    def test_missing_semantic_review_rejected(self):
        with self.assertRaises(AssertionError):
            repair.feedback_input(self.packet(), [])

    def test_feedback_rejects_missing_reasons_even_with_review_id(self):
        for field in ("reason", "answer_reason"):
            with self.subTest(field=field):
                packet = self.packet()
                parsed = packet["requests"][0]["parsed"]
                if field == "reason":
                    del parsed["item_evidence"]["A"]["reason"]
                else:
                    del parsed["answer_reason"]
                with self.assertRaises((KeyError, AssertionError)):
                    repair.feedback_input(packet, self.reviews())

    def test_reviewed_fixture_actually_exposes_both_reason_levels(self):
        result = repair.feedback_input(self.packet(), self.reviews())
        consumed = result[0]["model_reported_explanations"]
        self.assertEqual(consumed["item_evidence"]["A"]["reason"],
                         self.parsed["item_evidence"]["A"]["reason"])
        self.assertEqual(consumed["answer_reason"], self.parsed["answer_reason"])
        self.assertEqual(result[0]["semantic_review"], self.reviews()[0])

    def test_missing_per_item_semantic_review_rejected(self):
        reviews = self.reviews()
        del reviews[0]["item_reviews"]["A"]
        with self.assertRaises(AssertionError):
            repair.feedback_input(self.packet(), reviews)

    def test_review_raw_binding_mismatch_rejected(self):
        reviews = self.reviews()
        reviews[0]["raw_binding"] = dict(self.raw_binding, sha256="0" * 64)
        with self.assertRaises(AssertionError):
            repair.feedback_input(self.packet(), reviews)

    def test_modified_model_reason_rejected(self):
        packet = self.packet()
        packet["requests"][0]["parsed"]["answer_reason"] = "Invented after the original model response."
        with self.assertRaisesRegex(AssertionError, "MODEL_REASON_WAS_MODIFIED"):
            repair.feedback_input(packet, self.reviews())

    def test_gpu_requires_explicit_flag(self):
        with self.assertRaisesRegex(AssertionError, "GPU_EXECUTION_NOT_REQUESTED"):
            repair.run()

    def test_export_reasons_and_public_task_to_challenger(self):
        from unittest.mock import patch
        here = Path(self.temp.name) / "interface"
        (here / "outputs").mkdir(parents=True)
        repair.save(here / "REQUESTS.json", {"requests": [self.row]})
        repair.save(here / "outputs/REASON_REVIEW_PACKET.json", self.packet())
        review_path = here / "reviews.json"
        repair.save(review_path, {"reviews": self.reviews()})
        with patch.object(repair, "HERE", here):
            repair.export_feedback(review_path, here / "challenger")
        packet = repair.read(here / "challenger/INPUT_PACKET.json")
        text = (here / "challenger/CHALLENGER_INPUT.txt").read_text()
        self.assertEqual(packet["entries"][0]["public_task"]["question"], self.row["question"])
        self.assertIn(self.parsed["item_evidence"]["A"]["reason"], text)
        self.assertIn(self.parsed["answer_reason"], text)
        self.assertFalse(packet["llm_received_packet"])
        self.assertFalse(packet["formal_B_admission"])

    def test_unknown_review_holds_before_export(self):
        from unittest.mock import patch
        here = Path(self.temp.name) / "interface"
        (here / "outputs").mkdir(parents=True)
        repair.save(here / "REQUESTS.json", {"requests": [self.row]})
        repair.save(here / "outputs/REASON_REVIEW_PACKET.json", self.packet())
        reviews = self.reviews()
        reviews[0]["answer_reason_consistency"] = "unknown"
        repair.save(here / "reviews.json", {"reviews": reviews})
        with patch.object(repair, "HERE", here):
            with self.assertRaisesRegex(AssertionError, "NO_REVIEWED_REASONS_READY"):
                repair.export_feedback(here / "reviews.json", here / "challenger")
        self.assertFalse((here / "challenger").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
