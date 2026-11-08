"""Focused checks for the v10 offline revision and Gold-position rule."""

from copy import deepcopy
import unittest

from recheck import build, check_position, check_revision, dependency


class RecheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v9, _ = dependency()
        _, cls.runtime = cls.v9.dependency()

    def test_all_gold_counts_and_position_exception(self):
        for count in range(2, 6):
            self.assertFalse(check_position(list("ABCDEF"[:count]))["position_rule_pass"])
            self.assertTrue(check_position(list("A" + "CDEF"[:count - 1]))["position_rule_pass"])
        self.assertTrue(check_position(list("ABCDEF"))["position_rule_pass"])
        self.assertTrue(check_position(list("ABCDEF"))["six_correct_position_exception"])

    def test_target_reason_may_stay_unchanged(self):
        for root_id, target in self.v9.TARGETS.items():
            _, _, source, original = self.v9.source_inputs(root_id)
            revised = deepcopy(original)
            revised["options"][target["label"]] += " Revised."
            self.assertEqual(check_revision(revised, original, root_id, source["turns"],
                                            self.runtime.check_candidate, self.v9), [])
            revised["gold"] = list(reversed(original["gold"]))
            self.assertTrue(check_revision(revised, original, root_id, source["turns"],
                                           self.runtime.check_candidate, self.v9))

    def test_frozen_responses_pass_with_distinct_nonprefix_gold(self):
        summary, pool = build()
        self.assertEqual(summary["mechanically_valid_count"], 2)
        self.assertEqual(len(pool["questions"]), 2)
        self.assertEqual([row["position"]["gold_pattern"] for row in summary["roots"]],
                         ["ACEF", "ABDEF"])
        self.assertFalse(pool["ready_for_evaluation"])


if __name__ == "__main__":
    unittest.main()
