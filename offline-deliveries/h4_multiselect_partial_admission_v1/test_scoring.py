"""Focused tests for the independent multi-select partial-admission scorer."""

from __future__ import annotations

import unittest

from scoring import score_response


class ScoreResponseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.question = {
            "candidate_id": "q1", "question_type": "multi_select",
            "options": {key: f"option {key}" for key in "ABCDEF"},
            "gold": ["A", "C", "E"],
        }
        self.request = {
            "request_id": "q1__perm_00__no_context__s00",
            "candidate_id": "q1", "question_type": "multi_select",
            "variant": {
                "options": self.question["options"].copy(),
                "display_to_canonical": {key: key for key in "ABCDEF"},
                "displayed_gold": ["A", "C", "E"],
            },
        }

    def score(self, raw: str) -> dict:
        return score_response(self.question, self.request, {
            "request_id": self.request["request_id"],
            "raw_output": raw,
            "score": {"correct": True, "canonical_prediction": ["A", "C", "E"]},
        })

    def test_exact_full_gold(self) -> None:
        row = self.score("<box>A,C,E</box>")
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (3, 0, 0))
        self.assertEqual((row["exact_match"], row["partial_score"]), (True, 1.0))

    def test_omission_false_positive_and_negative_score(self) -> None:
        omitted = self.score("<box>A,C</box>")
        self.assertEqual((omitted["tp"], omitted["fp"], omitted["fn"]), (2, 0, 1))
        self.assertAlmostEqual(omitted["partial_score"], 2 / 3)
        extra = self.score("<box>A,C,E,F</box>")
        self.assertEqual((extra["tp"], extra["fp"], extra["fn"]), (3, 1, 0))
        self.assertAlmostEqual(extra["partial_score"], 2 / 3)
        negative = self.score("<box>B,D,F</box>")
        self.assertEqual((negative["tp"], negative["fp"], negative["fn"]), (0, 3, 3))
        self.assertEqual(negative["partial_score"], -1.0)

    def test_all_selected_uses_gold_count(self) -> None:
        row = self.score("<box>A,B,C,D,E,F</box>")
        self.assertEqual((row["n"], row["g"]), (6, 3))
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (3, 3, 0))
        self.assertEqual(row["partial_score"], 0.0)

    def test_permuted_display_maps_to_canonical(self) -> None:
        mapping = dict(zip("ABCDEF", "BCDEFA"))
        self.request["variant"] = {
            "display_to_canonical": mapping,
            "options": {display: self.question["options"][canonical]
                        for display, canonical in mapping.items()},
            "displayed_gold": ["F", "B", "D"],
        }
        row = self.score("<box>D,F,B</box>")
        self.assertEqual(row["display_prediction"], ["D", "F", "B"])
        self.assertEqual(row["canonical_prediction"], ["A", "C", "E"])
        self.assertEqual(row["mapping"], mapping)
        self.assertTrue(row["exact_match"])

    def test_empty_is_valid_and_distinct_from_failure(self) -> None:
        empty = self.score("<box> </box>")
        self.assertEqual((empty["parse_status"], empty["canonical_prediction"]), ("success", []))
        self.assertEqual((empty["tp"], empty["fp"], empty["fn"], empty["partial_score"]), (0, 0, 3, 0.0))
        failed = self.score("<box>A,,C</box>")
        self.assertEqual(failed["parse_status"], "parse_failure")
        self.assertIsNone(failed["exact_match"])
        self.assertIsNone(failed["partial_score"])
        self.assertEqual(self.score("A,C")["parse_status"], "parse_failure")

    def test_invalid_id_is_distinct(self) -> None:
        row = self.score("<box>A,Z</box>")
        self.assertEqual(row["parse_status"], "invalid_id")
        self.assertEqual(row["display_prediction"], ["A", "Z"])
        self.assertEqual(row["invalid_ids"], ["Z"])
        self.assertIsNone(row["tp"])
        self.assertEqual(row["scoring_status"], "INVALID_ID")

    def test_missing_raw_output_is_call_failure(self) -> None:
        row = score_response(self.question, self.request, {"request_id": self.request["request_id"]})
        self.assertEqual((row["parse_status"], row["scoring_status"]), ("call_failure", "CALL_FAILURE"))
        self.assertIsNone(row["partial_score"])

    def test_boundary_gold_counts_keep_exact_but_not_j(self) -> None:
        self.question["gold"] = []
        self.request["variant"]["displayed_gold"] = []
        empty_gold = self.score("<box></box>")
        self.assertTrue(empty_gold["exact_match"])
        self.assertIsNone(empty_gold["partial_score"])
        self.assertEqual(empty_gold["scoring_status"], "PARTIAL_NOT_APPLICABLE")
        self.question["gold"] = list("ABCDEF")
        self.request["variant"]["displayed_gold"] = list("ABCDEF")
        full_gold = self.score("<box>A,B,C,D,E,F</box>")
        self.assertTrue(full_gold["exact_match"])
        self.assertIsNone(full_gold["partial_score"])

    def test_gold_count_changes_denominators(self) -> None:
        self.question["gold"] = ["A", "C"]
        self.request["variant"]["displayed_gold"] = ["A", "C"]
        row = self.score("<box>A,B,D</box>")
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (1, 2, 1))
        self.assertEqual(row["partial_score"], 0.0)

    def test_inconsistent_mapping_rejected(self) -> None:
        self.request["variant"]["display_to_canonical"]["B"] = "A"
        with self.assertRaisesRegex(ValueError, "display_mapping_not_bijective"):
            self.score("<box>A</box>")


if __name__ == "__main__":
    unittest.main()
