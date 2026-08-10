"""Focused tests for the structure/evidence-first probe contract."""

from __future__ import annotations

import unittest

from .structure_first_contract import validate_structure_first_writer_output
from .structure_first_fixtures import (
    ambiguous_gold_envelope,
    make_envelope,
    weak_distractor_envelope,
)
from .structure_first_live import DEFAULT_EXPANDED, _load


class StructureFirstContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expanded = _load(DEFAULT_EXPANDED)

    def validate(self, value):
        return validate_structure_first_writer_output(
            value, expanded_dialogue=self.expanded)

    def test_variable_size_batches(self) -> None:
        zero = self.validate(make_envelope(0))
        self.assertTrue(zero["zero_question_batch_valid"])
        for count in (1, 3, 12):
            with self.subTest(count=count):
                result = self.validate(make_envelope(count))
                self.assertEqual(count, result["structural_valid_count"])

    def test_preferred_evidence_range(self) -> None:
        for count in range(2, 7):
            with self.subTest(count=count):
                result = self.validate(make_envelope(1, evidence_count=count))
                self.assertEqual(1, result["structural_valid_count"])
                self.assertEqual(count, result["evidence_statistics"]["min"])

    def test_ambiguous_gold_is_rejected(self) -> None:
        result = self.validate(ambiguous_gold_envelope())
        self.assertEqual(0, result["structural_valid_count"])
        self.assertIn(
            "gold_not_uniquely_supported",
            result["items"][0]["structural_errors"],
        )

    def test_weak_invented_distractor_is_rejected(self) -> None:
        result = self.validate(weak_distractor_envelope())
        self.assertEqual(0, result["structural_valid_count"])
        self.assertIn(
            "weak_invented_distractor",
            result["items"][0]["structural_errors"],
        )

    def test_structure_evidence_mismatch_is_rejected(self) -> None:
        value = make_envelope(1)
        value["questions"][0]["structure_plan"][
            "local_evidence_turn_ids"] = ["D001", "D003"]
        result = self.validate(value)
        self.assertIn(
            "structure_plan_evidence_mismatch",
            result["items"][0]["structural_errors"],
        )

    def test_reasoning_type_binding_is_strict(self) -> None:
        value = make_envelope(1)
        value["questions"][0]["structure_plan"][
            "structure_type"] = "information_visibility"
        result = self.validate(value)
        self.assertIn(
            "structure_type_question_type_mismatch",
            result["items"][0]["structural_errors"],
        )


if __name__ == "__main__":
    unittest.main()
