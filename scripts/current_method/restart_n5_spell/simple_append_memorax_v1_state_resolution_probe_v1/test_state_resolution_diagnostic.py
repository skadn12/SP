from __future__ import annotations

import unittest

from .audit import (
    EVENT_SPECS,
    EXPANDED_DIALOGUE_PATH,
    FROZEN_RUN_ROOT,
    _sha256_file,
    _tree_sha256,
    build_diagnostic,
)


class StateResolutionDiagnosticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = build_diagnostic()

    def test_scope_is_fixed_to_four_events(self) -> None:
        self.assertEqual(
            [spec["event"] for spec in EVENT_SPECS],
            [
                "soccer",
                "report_mistake",
                "health_doctor_work_coverage",
                "new_cashier",
            ],
        )
        self.assertEqual(self.result["summary"]["event_count"], 4)

    def test_transcript_evidence_is_bound_to_real_turns(self) -> None:
        for event in self.result["events"]:
            for state in event["transcript_state_history"]:
                self.assertIn(state["stage"], {"earlier", "latest"})
                self.assertTrue(state["evidence"])
                for evidence in state["evidence"]:
                    self.assertIsInstance(evidence["position"], int)
                    self.assertTrue(evidence["turn_ref"])
                    self.assertTrue(evidence["speaker"])
                    self.assertTrue(evidence["source_text"])

    def test_each_event_has_exactly_one_latest_state(self) -> None:
        for event in self.result["events"]:
            latest = [
                state
                for state in event["transcript_state_history"]
                if state["stage"] == "latest"
            ]
            self.assertEqual(len(latest), 1)
            self.assertEqual(event["transcript_latest_state"], latest[0])

    def test_memory_coverage_and_cases_match_fixed_audit(self) -> None:
        by_event = {event["event"]: event for event in self.result["events"]}
        self.assertTrue(by_event["soccer"]["memorax_recalled_facts"])
        self.assertTrue(by_event["report_mistake"]["memorax_recalled_facts"])
        self.assertTrue(
            by_event["health_doctor_work_coverage"]["memorax_recalled_facts"]
        )
        self.assertEqual(by_event["new_cashier"]["memorax_recalled_facts"], [])
        self.assertTrue(by_event["new_cashier"]["memory_latest_update_missing"])
        self.assertTrue(by_event["soccer"]["memory_event_identity_unstable"])
        self.assertTrue(by_event["report_mistake"]["memory_summary_state_conflict"])
        self.assertEqual(
            by_event["health_doctor_work_coverage"]["memory_fact_coverage"],
            "sufficient",
        )
        self.assertEqual(
            {event: value["case"] for event, value in by_event.items()},
            {
                "soccer": "C",
                "report_mistake": "C",
                "health_doctor_work_coverage": "A",
                "new_cashier": "B",
            },
        )

    def test_overall_resolution_answer_is_explicit(self) -> None:
        summary = self.result["summary"]
        self.assertEqual(summary["memory_contains_update_facts"], "partial")
        self.assertFalse(summary["memory_alone_supports_latest_state_resolution"])
        self.assertTrue(summary["memory_plus_transcript_supports_latest_state_resolution"])
        self.assertEqual(summary["primary_problem"], "mixed")

    def test_build_is_read_only_for_frozen_run(self) -> None:
        before_tree = _tree_sha256(FROZEN_RUN_ROOT, ignore_caches=False)
        before_dialogue = _sha256_file(EXPANDED_DIALOGUE_PATH)
        build_diagnostic()
        self.assertEqual(
            before_tree, _tree_sha256(FROZEN_RUN_ROOT, ignore_caches=False)
        )
        self.assertEqual(before_dialogue, _sha256_file(EXPANDED_DIALOGUE_PATH))

    def test_no_live_or_pipeline_execution_is_reported(self) -> None:
        self.assertEqual(self.result["scope"]["api_calls"], 0)
        self.assertFalse(self.result["scope"]["state_resolver_implemented"])
        self.assertFalse(self.result["scope"]["state_ledger_implemented"])


if __name__ == "__main__":
    unittest.main()
