"""Offline contracts for v14 model revision inputs."""

import unittest

from revise import (ROOTS, SLOT, SEEDS, base, check_candidate, older_questions,
                    parent_input, prompt_for, refs, seed_view, visible_turns)


class RevisionTests(unittest.TestCase):
    def test_seed_and_source_scope(self):
        v8, batch = base.dependency()
        instructions = (base.HERE / "instructions.txt").read_text()
        for root in ROOTS:
            with self.subTest(root=root):
                parent, _ = parent_input(root)
                old = older_questions(root)
                _, source = batch.ordering.checked_source(
                    root, batch.ordering.source_manifest()[root], frozen=False)
                seeds = seed_view(root, parent)
                self.assertEqual(len(seeds), len(SEEDS[root]))
                self.assertTrue(all(item["turn_ref"] not in refs(old) for item in seeds))
                visible = visible_turns(source, root, parent, old)
                self.assertTrue(all(t["turn_ref"].startswith("R02:S03:") for t in visible))
                self.assertFalse({t["turn_ref"] for t in visible} & refs(old))
                prompt = prompt_for(root, SLOT, source, instructions, old + [parent], v8)
                self.assertIn("safe_true_seed_claims_and_quotes", prompt)
                self.assertNotIn("previous_questions_avoid_repeating", prompt)
                for label, option in parent["options"].items():
                    if label not in SEEDS[root]:
                        self.assertNotIn(option, prompt)

    def test_duplicate_anchor_rejected(self):
        root = ROOTS[0]
        parent, _ = parent_input(root)
        old = older_questions(root)
        v8, batch = base.dependency()
        _, source = batch.ordering.checked_source(root, batch.ordering.source_manifest()[root], frozen=False)
        allowed = visible_turns(source, root, parent, old)
        seed_by_ref = {item["turn_ref"]: item for item in seed_view(root, parent)}
        chosen = [next(t for t in allowed if t["turn_ref"] == ref) for ref in seed_by_ref]
        chosen += [t for t in allowed if t["turn_ref"] not in seed_by_ref][:3]
        gold = "ABCE"
        row = {"candidate_id": base.slot_id(root, SLOT), "root_id": root,
               "question_type": "multi_select", "topic": "test",
               "question": "In R02, which claims hold? Select all that apply.",
               "options": {label: f"new option {label}" for label in base.LABELS},
               "gold": list(gold),
               "option_evidence": {label: {"turn_ref": chosen[index]["turn_ref"],
                   "quote": chosen[index]["text"],
                   "relation": "supports" if label in gold else "contradicts",
                   "reason": "fixture"} for index, label in enumerate(base.LABELS)},
               "answer_reason": "fixture"}
        self.assertEqual(check_candidate(row, root, SLOT, source["turns"], old + [parent], v8), [])
        row["option_evidence"]["F"]["turn_ref"] = row["option_evidence"]["E"]["turn_ref"]
        self.assertIn("six_distinct_evidence_turns_required",
                      check_candidate(row, root, SLOT, source["turns"], old + [parent], v8))


if __name__ == "__main__":
    unittest.main()
