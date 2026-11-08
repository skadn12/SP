"""Focused checks for inherited review and position binding."""

from __future__ import annotations

from copy import deepcopy
import unittest

import build


class ReviewedPositionPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = build.read(build.PRIOR / "READY_FOR_EVALUATION.json")["questions"][0]
        cls.new = next(row for row in build.read(build.SOURCE_POOL)["questions"]
                       if row["candidate_id"] == cls.old["candidate_id"])

    def test_reviewed_question_maps_by_exact_content(self):
        pool, audit = build.make_pool()
        self.assertEqual(pool["candidate_count"], 1)
        self.assertEqual(pool["questions"], [self.new])
        row = audit["candidates"][0]
        mapping = row["old_to_new_label_by_content"]
        self.assertEqual([mapping[label] for label in self.old["gold"]], self.new["gold"])
        self.assertEqual(set(row["prior_review_items_rebound"]), set(build.REVIEW_FILES))
        for check in row["prior_review_items_rebound"].values():
            self.assertTrue(check["pass"])
            self.assertEqual(set(check["items_by_new_label"]), set(build.LABELS))

    def test_changed_claim_cannot_inherit_review(self):
        modified = deepcopy(self.new)
        modified["items"]["I1"] += " Yesterday."
        with self.assertRaisesRegex(ValueError, "item_or_evidence_content_changed"):
            build.content_label_map(self.old, modified)

    def test_changed_question_scope_cannot_inherit_review(self):
        modified = deepcopy(self.new)
        modified["question"] += " Today."
        with self.assertRaisesRegex(ValueError, "non_label_question_content_changed"):
            build.content_label_map(self.old, modified)

    def test_new_gold_must_match_source_turn_order(self):
        source = build.read(build.BATCH / self.new["root_id"] / "source.json")
        modified = deepcopy(self.new)
        modified["gold"][0], modified["gold"][1] = modified["gold"][1], modified["gold"][0]
        with self.assertRaisesRegex(ValueError, "gold_conflicts_with_source_order"):
            build.binder.source_order(modified, source)


if __name__ == "__main__":
    unittest.main()
