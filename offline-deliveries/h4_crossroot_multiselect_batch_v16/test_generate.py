"""Offline contracts for one-call, per-candidate batch validation."""

import copy
import unittest

from generate import (base, candidate_id, check_candidate, generation_view,
                      prior_context, profile, prompt_for)


ROOTS = ("cc-train-episode-136202", "cc-train-episode-197635",
         "cc-train-episode-254852")


class BatchContracts(unittest.TestCase):
    def test_each_root_has_eighteen_unused_r02_turns_and_one_large_request(self):
        _, _, batch = base()
        for root_id in ROOTS:
            raw, source = batch.ordering.checked_source(
                root_id, batch.ordering.source_manifest()[root_id], frozen=False)
            self.assertTrue(raw)
            context, _ = prior_context(root_id)
            view = generation_view(source, context)
            self.assertGreaterEqual(len(view["turns"]), 18)
            self.assertTrue(all(t["turn_ref"].startswith("R02:") for t in view["turns"]))
            self.assertFalse({t["turn_ref"] for t in view["turns"]} &
                             set(context["forbidden_turn_refs"]))
            prompt = prompt_for(root_id, context, view)
            self.assertIn(candidate_id(root_id, 3), prompt)
            request = batch.ordering.build_payload(profile(batch), prompt)
            self.assertEqual(request["max_tokens"], 12288)
            self.assertEqual(request["model"], batch.MODEL)

    def test_candidates_checked_independently_and_repeated_labels_diagnostic_only(self):
        _, v8, batch = base()
        root_id = ROOTS[0]
        _, source = batch.ordering.checked_source(
            root_id, batch.ordering.source_manifest()[root_id], frozen=False)
        context, prior = prior_context(root_id)
        turns = generation_view(source, context)["turns"][:12]
        gold = ["A", "C"]
        row = {"candidate_id": candidate_id(root_id, 1), "root_id": root_id,
               "question_type": "multi_select", "topic": "Fixture",
               "question": "In R02, which claims hold? Select all that apply.",
               "options": {label: f"fresh fixture {label}" for label in "ABCDEF"},
               "gold": gold,
               "option_evidence": {label: {
                   "turn_ref": turns[i]["turn_ref"], "quote": turns[i]["text"][:100],
                   "relation": "supports" if label in gold else "contradicts",
                   "reason": "Fixture"} for i, label in enumerate("ABCDEF")},
               "answer_reason": "Fixture"}
        errors, _ = check_candidate(row, root_id, 1, source, context, prior, v8)
        self.assertEqual(errors, [])
        bad = copy.deepcopy(row)
        bad["candidate_id"] = candidate_id(root_id, 2)
        bad["option_evidence"]["F"]["turn_ref"] = context["forbidden_turn_refs"][0]
        errors, _ = check_candidate(bad, root_id, 2, source, context, prior, v8)
        self.assertIn("anchor_not_in_unused_r02_view", errors)
        self.assertEqual(check_candidate(row, root_id, 1, source, context, prior, v8)[0], [])
        context = copy.deepcopy(context)
        context["prior_selected_gold_patterns"].append({"candidate_id": "old", "gold": "AC"})
        errors, diagnostics = check_candidate(row, root_id, 1, source, context, prior, v8)
        self.assertEqual(errors, [])
        self.assertIn("prior_selected_gold_pattern_repeated_same_cardinality", diagnostics)


if __name__ == "__main__":
    unittest.main()
