"""Offline checks for variable-cardinality Gold and deterministic label mapping."""

from __future__ import annotations

import unittest
from pathlib import Path

from generate import LABELS, check_candidate, relabel, runner, target_gold


ROOTS = (
    "cc-train-episode-136202",
    "cc-train-episode-197635",
    "cc-train-episode-254852",
)
TURNS = [
    {"turn_ref": "R02:S01:T001", "text": "First source statement."},
    {"turn_ref": "R02:S01:T002", "text": "Second source statement."},
]


def question(root: str, gold: str) -> dict:
    return {
        "candidate_id": root.rsplit("-", 1)[-1] + "-qwen36-multi-01",
        "root_id": root, "question_type": "multi_select",
        "topic": "test", "question": "In R02, which statements hold? Select all that apply.",
        "options": {label: "claim " + label for label in LABELS},
        "gold": list(gold),
        "option_evidence": {
            label: {
                "turn_ref": TURNS[index % 2]["turn_ref"],
                "quote": TURNS[index % 2]["text"],
                "relation": "supports" if label in gold else "contradicts",
                "reason": "fixture",
            }
            for index, label in enumerate(LABELS)
        },
        "answer_reason": "Fixture labels.",
    }


class GenerateTests(unittest.TestCase):
    def test_all_allowed_gold_counts_and_boundaries(self):
        for count in range(2, 7):
            row = question(ROOTS[0], "ABCDEF"[:count])
            self.assertEqual(check_candidate(row, ROOTS[0], TURNS), [])
        self.assertIn("gold_must_have_2_to_6_distinct_labels",
                      check_candidate(question(ROOTS[0], "A"), ROOTS[0], TURNS))
        duplicate = question(ROOTS[0], "AB")
        duplicate["gold"] = ["A", "A"]
        self.assertIn("gold_must_have_2_to_6_distinct_labels",
                      check_candidate(duplicate, ROOTS[0], TURNS))

    def test_relabel_preserves_claim_and_evidence_content(self):
        for root in ROOTS:
            for count in range(2, 7):
                original = question(root, "ABCDEF"[:count])
                derived, meta = relabel(original)
                self.assertEqual(set(derived["gold"]), set(target_gold(root, count)))
                self.assertEqual(set(meta["source_to_derived_label"]), set(LABELS))
                for old, new in meta["source_to_derived_label"].items():
                    self.assertEqual(derived["options"][new], original["options"][old])
                    self.assertEqual(derived["option_evidence"][new],
                                     original["option_evidence"][old])
                    self.assertEqual(old in original["gold"], new in derived["gold"])
                self.assertEqual(check_candidate(derived, root, TURNS), [])
                self.assertEqual(meta["six_correct_position_exception"], count == 6)
                self.assertEqual(meta["explanation_label_scope"], "source_labels_only")
                self.assertFalse(meta["ready_for_evaluation"])
                self.assertEqual(meta["admission_blocker"],
                                 "DERIVED_EXPLANATIONS_USE_SOURCE_LABELS")

    def test_relabel_keeps_explanation_source_labels_and_blocks_ready(self):
        original = question(ROOTS[0], "ABC")
        original["answer_reason"] = "Option A is supported; Option D is false."
        original["option_evidence"]["A"]["reason"] = "Option A matches the quote."
        derived, meta = relabel(original)
        mapped_a = meta["source_to_derived_label"]["A"]
        self.assertEqual(derived["answer_reason"], original["answer_reason"])
        self.assertEqual(derived["option_evidence"][mapped_a]["reason"],
                         "Option A matches the quote.")
        self.assertEqual(meta["explanation_label_scope"], "source_labels_only")
        self.assertFalse(meta["ready_for_evaluation"])

    def test_position_patterns_differ_by_root_unless_all_correct(self):
        for count in range(2, 6):
            patterns = [target_gold(root, count) for root in ROOTS]
            self.assertEqual(len(set(patterns)), len(ROOTS))
            for pattern in patterns:
                self.assertNotEqual(pattern, LABELS[:count])
        self.assertEqual({target_gold(root, 6) for root in ROOTS}, {LABELS})

    def test_runner_isolated_from_frozen_v4(self):
        batch = runner()
        self.assertEqual(batch.HERE.name, "h4_crossroot_multiselect_batch_v8")
        self.assertEqual(batch.VERSION, batch.HERE.name)
        self.assertIs(batch.check_candidate, check_candidate)

    def test_prompt_spells_out_object_schema_and_abstention(self):
        instructions = (Path(__file__).resolve().parent / "instructions.txt").read_text()
        self.assertIn('"options": {"A":"Mira said', instructions)
        self.assertIn("Never return options as an array", instructions)
        self.assertIn("two to six distinct A-F labels", instructions)
        self.assertIn('"questions": []', instructions)
        self.assertIn("same speaker", instructions)
        self.assertIn("logical opposite", instructions)


if __name__ == "__main__":
    unittest.main()
