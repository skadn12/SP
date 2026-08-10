"""Zero-API regression tests for v19.2 Prompt calibration."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from .structural_contract import (
    QUESTION_CAPACITY_MAX,
    STRUCTURAL_QA_SCHEMA_VERSION,
    validate_structural_writer_output,
)
from .structural_prompts import build_structural_qa_prompt
from .v19_2_preflight import (
    EXPERIMENT_ROOT,
    HISTORICAL_RUNS,
    _dialogue,
    _fixture_results,
    _response,
)


class StructuralQAV192Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expanded = json.loads((
            EXPERIMENT_ROOT / "expansion/expanded_dialogue.json").read_text())
        cls.overview = json.loads((
            EXPERIMENT_ROOT / "expansion/memory/post_generation_overview.json"
        ).read_text())

    def test_positive_true_abstention_and_mixed_fixtures(self):
        fixtures = _fixture_results()
        self.assertTrue(fixtures["positive"]["passed"])
        self.assertTrue(fixtures["true_abstention"]["passed"])
        self.assertFalse(fixtures["true_abstention"]["solver_should_start"])
        self.assertTrue(fixtures["mixed"]["passed"])
        self.assertEqual(fixtures["mixed"]["returned_count"], 1)
        self.assertEqual(fixtures["mixed"]["automatically_added_count"], 0)

    def test_unexplained_empty_live_shape_is_invalid(self):
        dialogue = _dialogue("fixture-empty", [("A", "Hello."), ("B", "Hi.")])
        old_empty = _response([])
        result = validate_structural_writer_output(
            old_empty, expanded_dialogue=dialogue)
        self.assertFalse(result["envelope_valid"])
        self.assertIn(
            "empty_batch_without_abstention_evidence",
            result["envelope_errors"])

    def test_explained_zero_batch_remains_valid(self):
        dialogue = _dialogue("fixture-zero", [("A", "Hello."), ("B", "Hi.")])
        explained = _response(
            [], skipped=2, reasons=["no_discriminative_structure"])
        result = validate_structural_writer_output(
            explained, expanded_dialogue=dialogue)
        self.assertTrue(result["envelope_valid"])
        self.assertTrue(result["zero_question_batch_valid"])
        self.assertEqual(result["structural_valid_count"], 0)

    def test_v19_1_empty_live_response_fails_v19_2_shadow(self):
        raw = json.loads((
            HISTORICAL_RUNS[1] / "qa/writer_output_raw.json").read_text())
        raw["structural_qa_schema_version"] = STRUCTURAL_QA_SCHEMA_VERSION
        result = validate_structural_writer_output(
            raw, expanded_dialogue=self.expanded)
        self.assertIn(
            "empty_batch_without_abstention_evidence",
            result["envelope_errors"])

    def test_prompt_has_no_copyable_empty_envelope_and_ends_with_task(self):
        prompt = build_structural_qa_prompt(self.expanded, self.overview)["prompt"]
        self.assertNotIn('"questions":[]', prompt)
        self.assertNotIn('"questions": []', prompt)
        self.assertIn("questions: array with length 0-12", prompt)
        self.assertIn("12 is a capacity ceiling", prompt)
        tail = prompt[-500:]
        self.assertIn("FINAL TASK", tail)
        self.assertIn("Return zero only when", tail)
        self.assertNotIn("{", tail)
        self.assertNotIn('"questions"', tail)

    def test_capacity_remains_twelve_without_minimum_or_quota(self):
        prompt = build_structural_qa_prompt(
            self.expanded, self.overview)["prompt"].casefold()
        self.assertEqual(QUESTION_CAPACITY_MAX, 12)
        self.assertNotIn("minimum 8", prompt)
        self.assertNotIn("at least 8", prompt)
        self.assertNotIn("coverage quota", prompt)


if __name__ == "__main__":
    unittest.main()
