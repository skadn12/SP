from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.current_method import stage3_graph_only_smoke as subject


class Stage3GraphOnlySmokeTest(unittest.TestCase):
    def test_semantic_normalization_preserves_bridge_role_order(self) -> None:
        target = subject.native._build_stage64_canonical_targets(
            1, {"bridge": 1.0}, None
        )[0]["stage64_target"]
        raw_graph = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "child",
            "parent_graph_id": "parent",
            "operation": "bridge",
            "events": [
                {"event_id": "E1", "entity": "Maya", "slot": "route", "value": "north", "status": "active", "time_index": 1},
                {"event_id": "E2", "entity": "Maya", "slot": "permit", "value": "harbor", "status": "active", "time_index": 1},
                {"event_id": "E3", "entity": "Maya", "slot": "approved_route", "value": "harbor route", "status": "active", "time_index": 2, "depends_on": ["E1", "E2"]},
                {"event_id": "E4", "entity": "Maya", "slot": "alternate_route", "value": "ridge", "status": "outdated", "time_index": 1},
                {"event_id": "E5", "entity": "Noah", "slot": "destination", "value": "Lyon", "status": "active", "time_index": 3},
                {"event_id": "E6", "entity": "Li", "slot": "meeting_date", "value": "June 8", "status": "report", "time_index": 4},
            ],
        }
        plan = subject.native.normalize_current_event_plan(
            raw_graph,
            {"sample_id": "parent"},
            question_type="single_choice",
            attack_skill="conditional_plan",
            policy=subject.policy_for("bridge", target),
            sample_index=1,
            round_id="test",
        )

        self.assertEqual(
            [event["origin_event_id"] for event in plan["events"][:3]],
            ["E1", "E2", "E3"],
        )
        self.assertEqual(
            plan["events"][2]["depends_on"],
            ["test:S0001:E01", "test:S0001:E02"],
        )
        self.assertEqual(
            subject.native.semantic_graph_plan_quality_issues(
                plan, "bridge", target
            ),
            [],
        )

    def test_quality_gate_receives_frozen_stage64_target(self) -> None:
        captured: list[dict] = []

        def fake_quality(plan, operation, target=None):
            captured.append({"operation": operation, "target": target})
            return []

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                subject.native,
                "call_api_graph_expansion",
                return_value=({}, {"used": True}, {}),
            ), mock.patch.object(
                subject.native,
                "semantic_graph_plan_quality_issues",
                side_effect=fake_quality,
            ):
                summary = subject.run_graph_smoke(
                    parents=[{"sample_id": "parent-1", "dialogue": {"turns": []}}],
                    api_args=object(),
                    operations=["comparison"],
                    n=1,
                    seed=7,
                    round_id="test",
                    max_attempts_per_sample=1,
                    output_dir=Path(tmp),
                    minimum_pass_rate=1.0,
                )
                valid_rows = subject.native.read_jsonl_if_exists(
                    Path(tmp) / "graph_valid.jsonl"
                )

        self.assertTrue(summary["gate_passed"])
        self.assertEqual(summary["graph_gate_version"], "semantic_graph_v17")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["operation"], "comparison")
        self.assertEqual(captured[0]["target"]["qa_operation"], "comparison")
        self.assertIn("reasoning_subtype", captured[0]["target"])
        self.assertEqual(len(valid_rows), 1)
        self.assertEqual(valid_rows[0]["stage64_target"], captured[0]["target"])
        self.assertTrue(valid_rows[0]["stage64_target_identity"])
        self.assertTrue(valid_rows[0]["stage64_target_plan_sha256"])

    def test_smoke_stops_when_minimum_pass_rate_is_unreachable(self) -> None:
        def reject(*args, **kwargs):
            raise RuntimeError("synthetic graph failure")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(subject.native, "call_api_graph_expansion", side_effect=reject):
                summary = subject.run_graph_smoke(
                    parents=[{"sample_id": "parent-1", "dialogue": {"turns": []}}],
                    api_args=object(),
                    operations=list(subject.native.STAGE64_QA_OPERATIONS),
                    n=7,
                    seed=7,
                    round_id="test-early-stop",
                    max_attempts_per_sample=2,
                    output_dir=Path(tmp),
                    minimum_pass_rate=0.70,
                )

        self.assertTrue(summary["early_stopped"])
        self.assertEqual(summary["attempted_target_count"], 3)
        self.assertEqual(summary["unattempted_target_count"], 4)
        self.assertFalse(summary["gate_passed"])


if __name__ == "__main__":
    unittest.main()
