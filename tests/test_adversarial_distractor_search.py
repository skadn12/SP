from __future__ import annotations

import unittest

from scripts.current_method.restart_n5_spell.adversarial_distractor_search import (
    build_adversarial_distractor_prompt,
    deterministic_bundle_target_letter,
    is_grounding_search_success,
    validate_adversarial_bundles,
)


class AdversarialDistractorSearchTests(unittest.TestCase):
    def _valid(self) -> dict:
        return {"bundles": [{
            "bundle_id": f"B{index + 1}",
            "distractors": [
                f"He started a community job drive {index}.",
                f"He started a community aid drive {index}.",
                f"He started a community work drive {index}.",
            ],
        } for index in range(4)]}

    def test_accepts_exactly_four_parallel_bundles(self) -> None:
        result = validate_adversarial_bundles(
            self._valid(),
            correct_answer_text="He started a community food drive.")
        self.assertEqual(result["decision"], "adversarial_bundles_accept")
        self.assertEqual([row["bundle_id"] for row in result["bundles"]],
                         ["B1", "B2", "B3", "B4"])

    def test_rejects_three_bundles(self) -> None:
        raw = self._valid()
        raw["bundles"].pop()
        result = validate_adversarial_bundles(
            raw, correct_answer_text="He started a community food drive.")
        self.assertIn("bundle_count_invalid", result["reasons"])

    def test_rejects_gold_as_distractor(self) -> None:
        raw = self._valid()
        raw["bundles"][0]["distractors"][0] = (
            "He started a community food drive.")
        result = validate_adversarial_bundles(
            raw, correct_answer_text="He started a community food drive.")
        self.assertIn("distractor_equals_correct_answer", result["reasons"])

    def test_rejects_nonparallel_length(self) -> None:
        raw = self._valid()
        raw["bundles"][0]["distractors"][0] = "No."
        result = validate_adversarial_bundles(
            raw, correct_answer_text="He started a community food drive.")
        self.assertIn("distractor_length_not_parallel", result["reasons"])

    def test_placement_is_deterministic_and_covers_letters(self) -> None:
        first = [deterministic_bundle_target_letter("q1", index)
                 for index in range(4)]
        second = [deterministic_bundle_target_letter("q1", index)
                  for index in range(4)]
        self.assertEqual(first, second)
        self.assertEqual({row["target_letter"] for row in first}, set("ABCD"))
        self.assertTrue(all(row["content_dependent"] is False for row in first))

    def test_only_grounded_answerable_outcomes_stop_search(self) -> None:
        self.assertTrue(is_grounding_search_success("reject_easy"))
        self.assertTrue(is_grounding_search_success("retain_frontier"))
        for value in ("bad_ground", "reject_all_wrong",
                      "bad_option_bias", "frontier_audit_inconclusive"):
            self.assertFalse(is_grounding_search_success(value))

    def test_prompt_freezes_qa_and_uses_prior_sets(self) -> None:
        prompt = build_adversarial_distractor_prompt(
            base_qa={
                "qa_id": "q1", "question": "What did he do?",
                "correct_answer_text": "He started a community food drive.",
                "answer_support_span_ids": ["T1#S0001"],
            },
            evidence_rows=[{"span_id": "T1#S0001", "exact_text": "x"}],
            prior_failed_sets=[["old1", "old2", "old3"]],
        )
        self.assertIn("exactly four distinct candidate bundles", prompt)
        self.assertIn("old1", prompt)
        self.assertIn("He started a community food drive.", prompt)


if __name__ == "__main__":
    unittest.main()
