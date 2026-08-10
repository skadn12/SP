"""Zero-API tests for the minimal Structure Miner contract."""

from __future__ import annotations

import unittest

from .position_inputs import DEFAULT_EXPANDED, load_object
from .structure_miner_contract import validate_structure_miner_output


def candidate(positions=None, structure_type="revision", variable="old vs revised state"):
    return {
        "structure_type": structure_type,
        "evidence_positions": list(positions or [1, 2]),
        "discriminative_variable": variable,
    }


class StructureMinerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expanded = load_object(DEFAULT_EXPANDED)

    def validate(self, structures):
        return validate_structure_miner_output(
            {"structures": structures}, expanded_dialogue=self.expanded)

    def test_two_positions_valid(self):
        self.assertEqual(1, self.validate([candidate([1, 2])])["mechanically_valid_count"])

    def test_six_positions_valid(self):
        self.assertEqual(1, self.validate([candidate([1, 2, 3, 4, 5, 6])])["mechanically_valid_count"])

    def test_one_position_rejected(self):
        row = self.validate([candidate([1])])["items"][0]
        self.assertIn("evidence_below_minimum", row["mechanical_errors"])

    def test_seven_positions_rejected(self):
        row = self.validate([candidate([1, 2, 3, 4, 5, 6, 7])])["items"][0]
        self.assertIn("evidence_above_maximum", row["mechanical_errors"])

    def test_duplicate_position_rejected(self):
        row = self.validate([candidate([1, 1])])["items"][0]
        self.assertIn("evidence_duplicate_positions", row["mechanical_errors"])

    def test_out_of_range_position_rejected(self):
        row = self.validate([candidate([1, 111])])["items"][0]
        self.assertIn("evidence_position_out_of_range", row["mechanical_errors"])

    def test_unknown_type_rejected(self):
        row = self.validate([candidate(structure_type="ordinary_recall")])["items"][0]
        self.assertIn("structure_type_not_allowed", row["mechanical_errors"])

    def test_empty_variable_rejected(self):
        row = self.validate([candidate(variable="  ")])["items"][0]
        self.assertIn("discriminative_variable_empty", row["mechanical_errors"])

    def test_empty_structures_valid(self):
        result = self.validate([])
        self.assertTrue(result["empty_structures_valid"])

    def test_zero_one_and_multiple_candidates_valid(self):
        for count in (0, 1, 4):
            with self.subTest(count=count):
                result = self.validate([candidate([1, 2]) for _ in range(count)])
                self.assertEqual(count, result["mechanically_valid_count"])

    def test_over_hard_max_rejected(self):
        result = self.validate([candidate([1, 2]) for _ in range(11)])
        self.assertIn("structure_capacity_exceeded", result["envelope_errors"])
        self.assertEqual(0, result["mechanically_valid_count"])

    def test_position_maps_to_canonical_turn_id(self):
        row = self.validate([candidate([1, 2])])["valid_structures"][0]
        self.assertEqual([
            "cc-train-episode-146700:P01:T001",
            "cc-train-episode-146700:P01:T002",
        ], row["canonical_turn_ids"])

    def test_cross_container_is_warning_only(self):
        row = self.validate([candidate([1, 34])])["items"][0]
        self.assertTrue(row["valid"])
        self.assertIn("cross_parent_sessions", row["audit_warnings"])


if __name__ == "__main__":
    unittest.main()
