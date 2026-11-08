import copy
import unittest

from revalidate import inspect_questions, load_v16


class IsolationTests(unittest.TestCase):
    def test_malformed_first_candidate_does_not_block_later_candidate(self):
        module = load_v16()
        _, v8, batch, _, _, source, context, prior, view, _, _ = module.inputs(
            "cc-train-episode-136202")
        root_id = "cc-train-episode-136202"
        turns = view["turns"][:6]
        row = {"candidate_id": module.candidate_id(root_id, 2), "root_id": root_id,
               "question_type": "multi_select", "topic": "Fixture",
               "question": "Which R02 claims? Select all that apply.",
               "options": {label: f"fresh fixture {label}" for label in "ABCDEF"},
               "gold": ["A", "C"],
               "option_evidence": {label: {
                   "turn_ref": turns[i]["turn_ref"], "quote": turns[i]["text"][:100],
                   "relation": "supports" if label in "AC" else "contradicts",
                   "reason": "Fixture"} for i, label in enumerate("ABCDEF")},
               "answer_reason": "Fixture"}
        malformed = copy.deepcopy(row)
        malformed["candidate_id"] = module.candidate_id(root_id, 1)
        malformed["options"] = ["invalid", "shape"]
        malformed["option_evidence"] = [{"turn_ref": []}]
        checks, valid = inspect_questions(module, v8, root_id, source, context, prior,
                                          [malformed, row])
        self.assertTrue(checks[0]["errors"])
        self.assertEqual(checks[1]["errors"], [])
        self.assertEqual([q["candidate_id"] for q in valid], [row["candidate_id"]])


if __name__ == "__main__":
    unittest.main()
