"""Offline checks for source-driven diversity in v13."""

import unittest

from generate import SLOT, allowed_turns, anchor_refs, base, check_candidate, prior_questions


class SourceDiversityTests(unittest.TestCase):
    def test_available_turns_exclude_all_prior_anchors(self):
        _, batch = base.dependency()
        for root in batch.ROOTS:
            row = batch.ordering.source_manifest()[root]
            _, source = batch.ordering.checked_source(root, row, frozen=False)
            prior = prior_questions(root, SLOT)
            allowed = allowed_turns(source, prior)
            self.assertGreaterEqual(len(allowed), 6)
            self.assertTrue(all(t["turn_ref"].startswith("R02:S03:") for t in allowed))
            self.assertFalse({t["turn_ref"] for t in allowed} & anchor_refs(prior))

    def test_recycled_anchor_rejected_even_with_changed_words(self):
        root = "cc-train-episode-136202"
        _, batch = base.dependency()
        _, source = batch.ordering.checked_source(root, batch.ordering.source_manifest()[root], frozen=False)
        prior = prior_questions(root, SLOT)
        visible = allowed_turns(source, prior)[:6]
        gold = "AD"
        row = {"candidate_id": base.slot_id(root, SLOT), "root_id": root,
               "question_type": "multi_select", "topic": "test",
               "question": "In R02, which claims hold? Select all that apply.",
               "options": {label: f"new claim {label}" for label in base.LABELS},
               "gold": list(gold),
               "option_evidence": {label: {
                   "turn_ref": visible[index]["turn_ref"],
                   "quote": visible[index]["text"],
                   "relation": "supports" if label in gold else "contradicts",
                   "reason": "fixture"}
                   for index, label in enumerate(base.LABELS)},
               "answer_reason": "Fixture answer reason"}
        v8, _ = base.dependency()
        self.assertEqual(check_candidate(row, root, SLOT, source["turns"], prior, v8), [])
        row["option_evidence"]["F"]["turn_ref"] = prior[0]["option_evidence"]["A"]["turn_ref"]
        self.assertIn("anchor_not_from_unused_r02_scene_three",
                      check_candidate(row, root, SLOT, source["turns"], prior, v8))


if __name__ == "__main__":
    unittest.main()
