"""Focused checks for source-bound ordering validation."""

from __future__ import annotations

import copy
import unittest

from generate import check_candidate


ROOT = "cc-train-episode-231899"
LABELS = [f"I{index}" for index in range(1, 7)]
TURNS = [
    {
        "turn_ref": f"{ROOT}:P01:T00{index}" if index <= 3 else f"R01:S01:T00{index}",
        "speaker": "Micah",
        "text": f"Micah states distinct fact {index}.",
    }
    for index in range(1, 7)
]
GOLD = ["I2", "I4", "I1", "I6", "I3", "I5"]
QUESTION = {
    "candidate_id": f"{ROOT}-order-v5-001",
    "root_id": ROOT,
    "question_type": "ordering",
    "topic": "A coherent six-turn sequence",
    "chronology_kind": "conversation_order",
    "question": "In what order do these statements APPEAR IN THE SUPPLIED DIALOGUE?",
    "items": {label: f"Micah states distinct fact {index}." for index, label in enumerate(GOLD, 1)},
    "gold": GOLD,
    "item_evidence": {
        label: {
            "turn_ref": turn["turn_ref"],
            "speaker": turn["speaker"],
            "quote": turn["text"],
            "status": "statement",
            "reason": "Exact statement.",
        }
        for turn, label in zip(TURNS, GOLD)
    },
    "order_reason": "Source turn order.",
}


class ValidationTest(unittest.TestCase):
    def errors(self, candidate):
        return check_candidate(candidate, ROOT, TURNS, {QUESTION["candidate_id"]})

    def test_valid_shuffled_gold(self):
        self.assertEqual(self.errors(QUESTION), [])

    def test_unshuffled_display(self):
        candidate = copy.deepcopy(QUESTION)
        candidate["gold"] = LABELS
        self.assertIn("display_not_sufficiently_shuffled", self.errors(candidate))

    def test_wrong_gold(self):
        candidate = copy.deepcopy(QUESTION)
        candidate["gold"] = GOLD[::-1]
        self.assertIn("gold_conflicts_with_anchor_order", self.errors(candidate))

    def test_speaker_and_ref(self):
        candidate = copy.deepcopy(QUESTION)
        candidate["item_evidence"]["I2"]["speaker"] = "Other"
        candidate["item_evidence"]["I4"]["turn_ref"] = "R02:S01:T004"
        errors = self.errors(candidate)
        self.assertIn("I2:speaker_mismatch", errors)
        self.assertIn("I4:invalid_turn_ref", errors)


if __name__ == "__main__":
    unittest.main()
