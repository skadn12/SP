from __future__ import annotations

import unittest

from scripts.current_method.restart_n5_spell.semantic_parity_distractor_retry import (
    build_semantic_parity_prompt,
    validate_semantic_parity_bundle,
)


class SemanticParityDistractorRetryTests(unittest.TestCase):
    def test_balanced_bundle_accepts_same_frame_and_two_slots(self) -> None:
        gold = "To capture the group dynamics and the environment for the campaign."
        raw = {
            "frame": "To capture the {S1} and the {S2} for the campaign.",
            "slots": [
                {"slot_id": "S1", "gold_value": "group dynamics",
                 "contrast_value": "individual routines"},
                {"slot_id": "S2", "gold_value": "environment",
                 "contrast_value": "event schedule"},
            ],
        }
        result = validate_semantic_parity_bundle(
            raw, correct_answer_text=gold)
        self.assertEqual(result["decision"], "semantic_parity_accept")
        self.assertEqual(len(result["distractors"]), 3)
        self.assertFalse(result["gold_is_unique_slotwise_mode"])
        for values in result["value_frequencies_across_four_options"].values():
            self.assertEqual(sorted(values.values()), [2, 2])

    def test_low_shared_frame_coverage_is_rejected(self) -> None:
        gold = "He started a community food drive."
        raw = {
            "frame": "He started {S1} {S2}.",
            "slots": [
                {"slot_id": "S1", "gold_value": "a community",
                 "contrast_value": "a neighborhood"},
                {"slot_id": "S2", "gold_value": "food drive",
                 "contrast_value": "job fair"},
            ],
        }
        result = validate_semantic_parity_bundle(
            raw, correct_answer_text=gold)
        self.assertEqual(result["decision"], "semantic_parity_schema_invalid")
        self.assertIn("shared_frame_coverage_below_half", result["reasons"])

    def test_prompt_freezes_source_and_requires_equal_semantic_coverage(self) -> None:
        prompt = build_semantic_parity_prompt(
            base_qa={
                "qa_id": "q1", "question": "What was the purpose?",
                "correct_answer_text": "To document the group and setting.",
                "answer_support_span_ids": ["T1#S0001"],
            },
            evidence_rows=[{"span_id": "T1#S0001", "speaker": "A",
                            "exact_text": "evidence"}],
            previous_distractors=["old one", "old two", "old three"],
            previous_grounding_diagnosis="bad_ground_semantic_leakage",
        )
        self.assertIn("No combination may be a subset", prompt)
        self.assertIn("same number of object, setting, beneficiary", prompt)
        self.assertIn("old one", prompt)
        self.assertNotIn("distractors\": {", prompt)


if __name__ == "__main__":
    unittest.main()
