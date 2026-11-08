"""Focused CPU checks; synthetic fixtures never become experimental output."""
from __future__ import annotations

import copy
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import run as diagnostic


class DiagnosticChecks(unittest.TestCase):
    def setUp(self):
        self.row = {"labels": ["A", "B"], "question_type": "ordering",
                    "task_operation": "ORDER_EXPLICIT_TIME"}
        self.parsed = {"item_evidence": {
            "A": {"judgment": "supported", "turn_refs": ["T1"], "reason": "The source states noon."},
            "B": {"judgment": "supported", "turn_refs": ["T2"], "reason": "The source states 09:00."}},
            "answer_reason": "09:00 is earlier than noon, regardless of mention order.",
            "final_answer": "<box>B A</box>"}

    def test_all_seven_operation_specific_prompts(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[1]["content"]
        source = {"turns": [{"turn_ref": "T1", "speaker": "S", "text": "Example."}]}
        for _, operation in diagnostic.SELECTION:
            row = {"task_operation": operation, "question_type": "ordering" if operation.startswith("ORDER_") else "multi_select", "question": "Synthetic question."}
            prompt = diagnostic.make_prompt(Tokenizer(), row, {"A": "Synthetic item."}, source)
            self.assertIn(diagnostic.OPERATIONS[operation], prompt)
            self.assertNotIn("requested first-appearance rule", prompt)
            self.assertIn("answer_reason second and final_answer last", prompt)

    def test_missing_and_mismatched_operation_rejected(self):
        with self.assertRaisesRegex(AssertionError, "EXPLICIT_SUPPORTED"):
            diagnostic.operation_text({"question_type": "ordering"})
        with self.assertRaisesRegex(AssertionError, "OPERATION_QUESTION_TYPE"):
            diagnostic.operation_text({"question_type": "multi_select", "task_operation": "ORDER_EXPLICIT_TIME"})

    def test_clock_and_procedure_do_not_use_reference_position_order(self):
        for operation in ("ORDER_EXPLICIT_TIME", "ORDER_EXPLICIT_PROCEDURE", "ORDER_DISCOURSE_RECONSTRUCTION"):
            row = dict(self.row, task_operation=operation)
            result = diagnostic.mechanical_consistency(row, self.parsed, ["B", "A"])
            self.assertIsNone(result["answer_matches_cited_order"])
            self.assertFalse(result["reference_location_order_applicable"])
            self.assertTrue(result["semantic_order_review_required"])

    def test_reference_and_reason_schema_still_applies(self):
        schema = diagnostic.repair.schema(["A", "B"], ["T1", "T2"])
        _, labels = diagnostic.repair.validate(json.dumps(self.parsed), self.row, schema)
        self.assertEqual(labels, ["B", "A"])
        missing = copy.deepcopy(self.parsed)
        del missing["item_evidence"]["A"]["reason"]
        with self.assertRaises(Exception):
            diagnostic.repair.validate(json.dumps(missing), self.row, schema)

    def test_display_content_mapping_is_explicit(self):
        question = {"question_type": "ordering", "items": {"I1": "Earlier.", "I2": "Later."}}
        display, mapping, canonical = diagnostic.relabel_display(question, {"items": {"I1": "Later.", "I2": "Earlier."}})
        self.assertEqual(display, {"A": "Later.", "B": "Earlier."})
        self.assertEqual(mapping, {"I1": "A", "I2": "B"})
        self.assertEqual(canonical, {"A": "I2", "B": "I1"})

    def test_feedback_retains_operation_and_raw_reasons(self):
        with tempfile.TemporaryDirectory(prefix="diversity-reason-cpu-") as temporary:
            raw_path = Path(temporary) / "synthetic_raw.json"
            raw_path.write_text(json.dumps({"raw_output": json.dumps(self.parsed)}))
            binding = diagnostic.bind(raw_path)
            row = dict(self.row, question="Order clock times.", display={"A": "Noon", "B": "09:00"},
                       display_to_canonical={"A": "I2", "B": "I1"})
            record = {"request_id": "synthetic", "candidate_id": "synthetic", "root_id": "synthetic_train",
                "question_type": "ordering", "status": "STRUCTURALLY_VALID_REASONS_SAVED", "request": row,
                "source_binding": binding, "raw_binding": binding, "parsed": self.parsed,
                "mechanical_consistency": diagnostic.mechanical_consistency(row, self.parsed, ["B", "A"])}
            review = {"request_id": "synthetic", "raw_binding": binding, "review_complete": True, "reviewer_type": "ai_agent",
                "item_reviews": {label: {"source_support": "pass", "reason_support": "pass", "notes": "Synthetic test only."} for label in row["labels"]},
                "answer_reason_consistency": "pass"}
            result = diagnostic.feedback_input({"requests": [record]}, [review])[0]
            self.assertEqual(result["public_task"]["task_operation"], "ORDER_EXPLICIT_TIME")
            self.assertEqual(result["model_reported_explanations"], self.parsed)

    def test_mc_supported_set_consistency(self):
        row = dict(self.row, question_type="multi_select", task_operation="MC_EVIDENCE_SELECTION")
        result = diagnostic.mechanical_consistency(row, self.parsed, ["A"])
        self.assertFalse(result["answer_matches_supported_labels"])

    def test_gpu_requires_explicit_flag(self):
        with self.assertRaisesRegex(AssertionError, "GPU_EXECUTION_NOT_REQUESTED"):
            diagnostic.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=diagnostic.HERE / "SELF_CHECK_REPORT.json")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    diagnostic.save(args.report, {
        "status": "PASS" if result.wasSuccessful() else "FAIL", "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "cpu_only": True, "synthetic_test_fixtures": True, "model_calls": 0,
        "bindings": [diagnostic.bind(diagnostic.HERE / name) for name in ("run.py", "selfcheck.py")]})
    raise SystemExit(0 if result.wasSuccessful() else 1)
