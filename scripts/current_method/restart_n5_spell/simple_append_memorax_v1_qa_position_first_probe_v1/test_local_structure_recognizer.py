"""Zero-API tests for natural windows and local structure recognition."""

from __future__ import annotations

import unittest

from .local_structure_recognizer_contract import validate_local_recognizer_output
from .local_structure_recognizer_prompts import build_local_recognizer_prompt
from .local_structure_windows import compile_natural_windows
from .position_inputs import DEFAULT_EXPANDED, load_object


def candidate(positions, structure_type="revision", variable="old vs revised state"):
    return {
        "structure_type": structure_type,
        "evidence_positions": list(positions),
        "discriminative_variable": variable,
    }


class LocalStructureRecognizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expanded = load_object(DEFAULT_EXPANDED)
        cls.compiled = compile_natural_windows(cls.expanded)
        cls.window = cls.compiled["windows"][0]

    def validate(self, structures, window=None):
        return validate_local_recognizer_output(
            {"structures": structures}, expanded_dialogue=self.expanded,
            window=window or self.window,
        )

    def test_parent_session_windows(self):
        self.assertEqual(5, self.compiled["parent_session_count"])
        self.assertEqual(5, sum(
            window["window_kind"] == "parent_session"
            for window in self.compiled["windows"]
        ))

    def test_child_scene_windows(self):
        self.assertEqual(3, self.compiled["child_scene_count"])
        self.assertEqual(3, sum(
            window["window_kind"] == "child_scene"
            for window in self.compiled["windows"]
        ))

    def test_windows_do_not_overlap_and_cover_all_turns(self):
        positions = [
            pos for window in self.compiled["windows"]
            for pos in window["global_positions"]
        ]
        self.assertEqual(list(range(1, 111)), sorted(positions))
        self.assertEqual(len(positions), len(set(positions)))

    def test_global_positions_and_canonical_ids_are_preserved(self):
        first = self.window["turns"][0]
        self.assertEqual(1, first["global_position"])
        self.assertEqual("cc-train-episode-146700:P01:T001", first["canonical_turn_id"])

    def test_two_and_six_positions_valid(self):
        allowed = self.window["global_positions"]
        self.assertEqual(1, self.validate([candidate(allowed[:2])])["mechanically_valid_count"])
        self.assertEqual(1, self.validate([candidate(allowed[:6])])["mechanically_valid_count"])

    def test_one_and_seven_positions_rejected(self):
        allowed = self.window["global_positions"]
        one = self.validate([candidate(allowed[:1])])["items"][0]
        seven = self.validate([candidate(allowed[:7])])["items"][0]
        self.assertIn("evidence_below_minimum", one["mechanical_errors"])
        self.assertIn("evidence_above_maximum", seven["mechanical_errors"])

    def test_duplicate_position_rejected(self):
        row = self.validate([candidate([1, 1])])["items"][0]
        self.assertIn("evidence_duplicate_positions", row["mechanical_errors"])

    def test_outside_window_position_rejected(self):
        row = self.validate([candidate([1, 9])])["items"][0]
        self.assertIn("evidence_position_outside_window", row["mechanical_errors"])

    def test_out_of_range_position_rejected(self):
        row = self.validate([candidate([1, 111])])["items"][0]
        self.assertIn("evidence_position_out_of_range", row["mechanical_errors"])

    def test_unknown_type_rejected(self):
        row = self.validate([candidate([1, 2], "ordinary_recall")])["items"][0]
        self.assertIn("structure_type_not_allowed", row["mechanical_errors"])

    def test_empty_variable_rejected(self):
        row = self.validate([candidate([1, 2], variable="")])["items"][0]
        self.assertIn("discriminative_variable_empty", row["mechanical_errors"])

    def test_empty_structures_valid(self):
        self.assertTrue(self.validate([])["empty_structures_valid"])

    def test_per_window_capacity_rejected(self):
        result = self.validate([candidate([1, 2]) for _ in range(6)])
        self.assertIn("per_window_capacity_exceeded", result["envelope_errors"])
        self.assertEqual(0, result["mechanically_valid_count"])

    def test_position_maps_to_canonical_id(self):
        row = self.validate([candidate([1, 2])])["valid_structures"][0]
        self.assertEqual([
            "cc-train-episode-146700:P01:T001",
            "cc-train-episode-146700:P01:T002",
        ], row["canonical_turn_ids"])

    def test_prompt_contains_only_current_window(self):
        prompt = build_local_recognizer_prompt(self.window)["prompt"]
        self.assertIn("1 | Co-workers A", prompt)
        self.assertNotIn("9 |", prompt)


if __name__ == "__main__":
    unittest.main()
