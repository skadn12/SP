"""Offline contracts for v12 slots, label positions, and duplicate claims."""

import unittest

from generate import LABELS, check_candidate, dependency, slot_id


ROOT = "cc-train-episode-136202"
TURNS = [
    {"turn_ref": "R02:S01:T001", "text": "First source statement."},
    {"turn_ref": "R02:S01:T002", "text": "Second source statement."},
]


def question(gold="AD", slot=2):
    return {
        "candidate_id": slot_id(ROOT, slot), "root_id": ROOT,
        "question_type": "multi_select", "topic": "test",
        "question": "In R02, which claims hold? Select all that apply.",
        "options": {label: f"claim {label}" for label in LABELS},
        "gold": list(gold),
        "option_evidence": {
            label: {"turn_ref": TURNS[(index // 2) % 2]["turn_ref"],
                    "quote": TURNS[(index // 2) % 2]["text"],
                    "relation": "supports" if label in gold else "contradicts",
                    "reason": "fixture"}
            for index, label in enumerate(LABELS)},
        "answer_reason": "Fixture answer reason",
    }


class GenerateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v8, _ = dependency()

    def test_gold_counts_and_prefix_exception(self):
        for gold in ("AD", "ACE", "ABDF", "ABDEF", "ABCDEF"):
            with self.subTest(gold=gold):
                self.assertEqual(check_candidate(question(gold), ROOT, 2, TURNS, [], self.v8), [])
        self.assertIn("gold_prefix_forbidden",
                      check_candidate(question("ABC"), ROOT, 2, TURNS, [], self.v8))

    def test_prior_position_and_content_dedup(self):
        old = question("AD", 1)
        self.assertIn("gold_position_pattern_repeated_at_same_cardinality",
                      check_candidate(question("AD"), ROOT, 2, TURNS, [old], self.v8))
        self.assertIn("three_or_more_prior_options_reused",
                      check_candidate(question("CF"), ROOT, 2, TURNS, [old], self.v8))
        fresh = question("CF")
        fresh["options"] = {label: f"different {label}" for label in LABELS}
        self.assertEqual(check_candidate(fresh, ROOT, 2, TURNS, [old], self.v8), [])

    def test_slot_identity(self):
        self.assertIn("candidate_id_invalid",
                      check_candidate(question("AD", 3), ROOT, 2, TURNS, [], self.v8))


if __name__ == "__main__":
    unittest.main()
