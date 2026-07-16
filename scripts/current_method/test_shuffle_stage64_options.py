from __future__ import annotations

import unittest

from scripts.current_method import shuffle_stage64_options as subject
from scripts.current_method.test_recalibrate_stage64_shuffled_frontier import sample


class ShuffleStage64OptionsTest(unittest.TestCase):
    def test_balances_each_skill_stratum_and_preserves_gold_text(self) -> None:
        rows = [sample(f"s{index}") for index in range(8)]
        original_gold_text = {
            row["sample_id"]: next(option["text"] for option in row["qa"]["options"] if option["id"] == row["qa"]["gold"][0])
            for row in rows
        }
        shuffled, summary = subject.balance_options(rows, seed=11)
        self.assertEqual(summary["gold_distribution"], {"A": 2, "B": 2, "C": 2, "D": 2})
        for row in shuffled:
            gold = row["qa"]["gold"][0]
            text = next(option["text"] for option in row["qa"]["options"] if option["id"] == gold)
            self.assertEqual(text, original_gold_text[row["sample_id"]])
            self.assertNotIn("solver_eval", row)


if __name__ == "__main__":
    unittest.main()
