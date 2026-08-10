import unittest

from scripts.current_method.restart_n5_spell.balanced_factorial_distractor_canary import (
    deterministic_balanced_factorial_placement,
    validate_and_render_balanced_factorial,
)


class BalancedFactorialDistractorCanaryTests(unittest.TestCase):
    def setUp(self):
        self.answer = "Her aunt's belief in giving back and helping others in need."
        self.valid = {
            "frame": "Her {S1}'s belief in {S2}.",
            "slots": [
                {"slot_id": "S1", "gold_value": "aunt",
                 "contrast_value": "cousin"},
                {"slot_id": "S2",
                 "gold_value": "giving back and helping others in need",
                 "contrast_value":
                 "organizing food deliveries for vulnerable families"},
            ],
        }

    def test_valid_contract_builds_balanced_square(self):
        result = validate_and_render_balanced_factorial(
            self.valid, correct_answer_text=self.answer)
        self.assertEqual(result["decision"], "balanced_factorial_accept")
        self.assertEqual(len(result["distractors"]), 3)
        self.assertFalse(result["gold_is_unique_slotwise_mode"])
        for frequencies in result["value_frequencies_across_four_options"].values():
            self.assertEqual(sorted(frequencies.values()), [2, 2])

    def test_requires_exactly_two_slots(self):
        raw = {**self.valid, "slots": self.valid["slots"][:1]}
        result = validate_and_render_balanced_factorial(
            raw, correct_answer_text=self.answer)
        self.assertIn("slots_count_invalid", result["reasons"])

    def test_gold_must_render_exactly(self):
        raw = {**self.valid, "frame": "Her {S1} believed in {S2}."}
        result = validate_and_render_balanced_factorial(
            raw, correct_answer_text=self.answer)
        self.assertIn("rendered_gold_mismatch", result["reasons"])

    def test_contrast_cannot_equal_gold(self):
        slots = [dict(row) for row in self.valid["slots"]]
        slots[0]["contrast_value"] = slots[0]["gold_value"]
        result = validate_and_render_balanced_factorial(
            {**self.valid, "slots": slots}, correct_answer_text=self.answer)
        self.assertIn("contrast_equals_gold", result["reasons"])

    def test_placement_supports_arbitrary_qa_ids_deterministically(self):
        first = deterministic_balanced_factorial_placement("qa_3")
        second = deterministic_balanced_factorial_placement("qa_3")
        self.assertEqual(first, second)
        self.assertIn(first["target_letter"], "ABCD")
        self.assertFalse(first["content_dependent"])

    def test_placement_rejects_empty_qa_id(self):
        with self.assertRaisesRegex(
                ValueError, "balanced_factorial_placement_input_invalid"):
            deterministic_balanced_factorial_placement("")


if __name__ == "__main__":
    unittest.main()
