import unittest

from scripts.current_method.restart_n5_spell.slot_mutation_distractor_canary import (
    validate_and_render_slot_mutations,
)


class SlotMutationDistractorCanaryTests(unittest.TestCase):
    def setUp(self):
        self.answer = "Her aunt's belief in giving back and helping others in need."
        self.valid = {
            "frame": "Her {S1}'s belief in {S2}.",
            "slots": [
                {"slot_id": "S1", "gold_value": "aunt"},
                {"slot_id": "S2", "gold_value":
                 "giving back and helping others in need"},
            ],
            "mutations": [
                {"changed_slot": "S1", "replacement_value": "mother"},
                {"changed_slot": "S2", "replacement_value":
                 "serving vulnerable families through local outreach"},
                {"changed_slot": "S2", "replacement_value":
                 "organizing meal deliveries for people in need"},
            ],
        }

    def test_valid_contract_renders_three_distractors(self):
        result = validate_and_render_slot_mutations(
            self.valid, correct_answer_text=self.answer)
        self.assertEqual(result["decision"], "slot_mutation_accept")
        self.assertEqual(len(result["distractors"]), 3)
        self.assertEqual(result["rendered_gold"], self.answer)

    def test_gold_must_render_exactly(self):
        raw = {**self.valid, "frame": "Her {S1} believed in {S2}."}
        result = validate_and_render_slot_mutations(
            raw, correct_answer_text=self.answer)
        self.assertIn("rendered_gold_mismatch", result["reasons"])

    def test_fourth_mutation_is_rejected_not_trimmed(self):
        raw = {**self.valid, "mutations": self.valid["mutations"] + [{
            "changed_slot": "S1", "replacement_value": "sister"}]}
        result = validate_and_render_slot_mutations(
            raw, correct_answer_text=self.answer)
        self.assertIn("mutations_count_invalid", result["reasons"])
        self.assertNotIn("distractors", result)

    def test_unknown_slot_is_rejected(self):
        mutations = [dict(row) for row in self.valid["mutations"]]
        mutations[0]["changed_slot"] = "S3"
        result = validate_and_render_slot_mutations(
            {**self.valid, "mutations": mutations},
            correct_answer_text=self.answer)
        self.assertIn("mutation_slot_unknown", result["reasons"])

    def test_duplicate_distractors_are_rejected(self):
        mutations = [dict(row) for row in self.valid["mutations"]]
        mutations[2] = dict(mutations[1])
        result = validate_and_render_slot_mutations(
            {**self.valid, "mutations": mutations},
            correct_answer_text=self.answer)
        self.assertIn("duplicate_rendered_distractor", result["reasons"])


if __name__ == "__main__":
    unittest.main()
