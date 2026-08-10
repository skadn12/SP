from __future__ import annotations

import unittest

from scripts.current_method.restart_n5_spell.run_single_item_qa_canary import (
    budget_limits,
    summarize_records,
)


class RunSingleItemQaCanaryTests(unittest.TestCase):
    def test_format_canary_budget_excludes_downstream_requests(self) -> None:
        limits = budget_limits(False)
        self.assertEqual(limits["normal_caps"], {
            "single_item_questioner": 3, "distractor_writer": 0})
        self.assertEqual(limits["format_retry_cap"], 3)
        self.assertEqual(limits["physical_cap"], 6)

    def test_full_pipeline_budget_is_bounded(self) -> None:
        limits = budget_limits(True)
        self.assertEqual(sum(limits["normal_caps"].values()), 6)
        self.assertEqual(limits["format_retry_cap"], 6)
        self.assertEqual(limits["physical_cap"], 12)

    def test_three_of_three_strict_schema_outputs_meet_target(self) -> None:
        rows = [{
            "strict_json_valid": True,
            "questioner_schema_valid": True,
            "core_qa_accepted": True,
            "final_decision": "core_qa_accept_format_canary_stop",
        } for _ in range(3)]
        summary = summarize_records(rows, execute_downstream=False)
        self.assertTrue(summary["format_canary_target_met"])
        self.assertEqual(summary["strict_json_valid"], 3)
        self.assertEqual(summary["questioner_schema_valid"], 3)
        self.assertFalse(summary["execute_downstream"])

    def test_parseable_but_schema_invalid_does_not_meet_target(self) -> None:
        rows = [{
            "strict_json_valid": True,
            "questioner_schema_valid": index != 1,
            "core_qa_accepted": index != 1,
            "final_decision": "core_qa_accept_format_canary_stop",
        } for index in range(3)]
        summary = summarize_records(rows, execute_downstream=False)
        self.assertFalse(summary["format_canary_target_met"])
        self.assertEqual(summary["strict_json_valid"], 3)
        self.assertEqual(summary["questioner_schema_valid"], 2)


if __name__ == "__main__":
    unittest.main()
