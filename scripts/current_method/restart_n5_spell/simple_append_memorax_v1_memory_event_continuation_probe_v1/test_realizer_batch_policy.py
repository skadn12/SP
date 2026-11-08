import unittest

from .workflow import build_realizer_plan_batches
from .profiles import EXPANDED_SINGLE_PLAN_PROFILE, EXPANDED_PROFILE, LEGACY_PROFILE


class RealizerBatchPolicyTests(unittest.TestCase):
    def test_continuation_run_is_kept_together_then_independent_chunks(self):
        plan = {"scenes": [
            {"memory_event_id": "C1", "development_mode": "continuation"},
            {"memory_event_id": "C2", "development_mode": "continuation",
             "dependencies": [{"phase_id": "C1"}]},
            {"memory_event_id": "C3", "development_mode": "continuation"},
            {"memory_event_id": "N1", "development_mode": "new_development"},
            {"memory_event_id": "N2", "development_mode": "new_development"},
            {"memory_event_id": "N3", "development_mode": "new_development"},
        ]}
        batches = build_realizer_plan_batches(
            plan, batch_size=2, round_index=3, batch_policy="continuation_first")
        self.assertEqual([[s["memory_event_id"] for s in b["scenes"]]
                          for b in batches], [["C1", "C2", "C3"], ["N1", "N2"], ["N3"]])
        self.assertEqual([b["batch_policy"] for b in batches], ["continuation_first"] * 3)

    def test_fixed_policy_remains_historical_behavior(self):
        plan = {"scenes": [{"memory_event_id": str(i), "development_mode": "continuation"}
                            for i in range(5)]}
        batches = build_realizer_plan_batches(plan, batch_size=2, batch_policy="fixed_size")
        self.assertEqual([len(b["scenes"]) for b in batches], [2, 2, 1])

    def test_profiles_keep_expanded_memory_fast_and_legacy_reproducible(self):
        self.assertTrue(EXPANDED_SINGLE_PLAN_PROFILE.memory_only_fast)
        self.assertEqual(EXPANDED_SINGLE_PLAN_PROFILE.realizer_batch_policy,
                         "continuation_first")
        self.assertEqual(EXPANDED_PROFILE.realizer_batch_policy, "fixed_size")
        self.assertEqual(LEGACY_PROFILE.realizer_batch_policy, "fixed_size")


if __name__ == "__main__":
    unittest.main()
