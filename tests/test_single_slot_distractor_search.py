from __future__ import annotations

import unittest

from scripts.current_method.restart_n5_spell.single_slot_distractor_search import (
    build_single_slot_prompt,
    validate_single_slot_bundles,
)


class SingleSlotDistractorSearchTests(unittest.TestCase):
    def _valid(self) -> dict:
        return {
            "frame": "He started a community {S1}.",
            "gold_value": "food drive",
            "bundles": [{
                "bundle_id": f"B{index + 1}",
                "contrast_values": [
                    f"job fair {index}",
                    f"work clinic {index}",
                    f"hiring drive {index}",
                ],
            } for index in range(4)],
        }

    def test_accepts_exact_gold_frame_and_four_bundles(self) -> None:
        result = validate_single_slot_bundles(
            self._valid(),
            correct_answer_text="He started a community food drive.")
        self.assertEqual(result["decision"], "single_slot_accept")
        self.assertEqual(result["rendered_gold"],
                         "He started a community food drive.")
        self.assertEqual(len(result["bundles"]), 4)

    def test_rejects_frame_that_rewrites_gold(self) -> None:
        raw = self._valid()
        raw["frame"] = "He launched a community {S1}."
        result = validate_single_slot_bundles(
            raw, correct_answer_text="He started a community food drive.")
        self.assertIn("rendered_gold_mismatch", result["reasons"])

    def test_rejects_second_placeholder(self) -> None:
        raw = self._valid()
        raw["frame"] = "He started a {S2} community {S1}."
        result = validate_single_slot_bundles(
            raw, correct_answer_text="He started a community food drive.")
        self.assertIn("unknown_placeholder", result["reasons"])

    def test_prompt_requires_unemployment_interventions(self) -> None:
        prompt = build_single_slot_prompt(
            base_qa={
                "qa_id": "q1", "question": "What did he do?",
                "correct_answer_text": "He started a community food drive.",
                "answer_support_span_ids": ["T1#S0001"],
            }, evidence_rows=[{"span_id": "T1#S0001", "exact_text": "x"}],
            prior_failed_distractors=["old"],
        )
        self.assertIn("plausible concrete unemployment interventions", prompt)
        self.assertIn("old", prompt)


if __name__ == "__main__":
    unittest.main()
