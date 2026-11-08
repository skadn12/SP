from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.current_method import replay_stage3_claim_first_target_adapter as replay


class OfflineGuardTests(unittest.TestCase):
    def test_socket_attempt_is_rejected_and_counted(self) -> None:
        with replay.offline_network_guard() as attempts:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                with self.assertRaises(replay.OfflineNetworkAttempt):
                    sock.connect(("127.0.0.1", 9))
            finally:
                sock.close()
        self.assertEqual(len(attempts), 1)


class IndependentSiblingFilteringTests(unittest.TestCase):
    def test_one_failed_sibling_does_not_delete_dialogue_or_other_qas(self) -> None:
        qas = [{"qa_id": "good-a"}, {"qa_id": "bad"}, {"qa_id": "good-b"}]

        def semantic_report(_claim_ir, qa, _dialogue):
            passed = qa["qa_id"] != "bad"
            return {
                "pass": passed,
                "issue_counts_by_code": {} if passed else {"test_failure": 1},
                "issues": [] if passed else [{"code": "test_failure"}],
            }

        with (
            mock.patch.object(
                replay, "_claim_validator_report", return_value={"pass": True}
            ),
            mock.patch.object(
                replay.stage3_claim_semantic_validator,
                "validate_qa_semantics",
                side_effect=semantic_report,
            ),
        ):
            accepted, rejected = replay.independently_filter_qas(
                claim_ir={}, qas=qas, dialogue={"turns": []}
            )
        self.assertEqual([item["qa_id"] for item in accepted], ["good-a", "good-b"])
        self.assertEqual([item["qa_id"] for item in rejected], ["bad"])
        self.assertTrue(rejected[0]["dialogue_retained"])
        self.assertTrue(rejected[0]["sibling_qas_retained"])
        self.assertEqual(qas, [{"qa_id": "good-a"}, {"qa_id": "bad"}, {"qa_id": "good-b"}])


@unittest.skipUnless(
    replay.DEFAULT_SOURCE_RUN.is_dir(), "run224 integration artifact is unavailable"
)
class Run224OfflineReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_manifest_before = replay.source_manifest(replay.DEFAULT_SOURCE_RUN)
        cls.result = replay.build_replay(replay.DEFAULT_SOURCE_RUN)

    @classmethod
    def tearDownClass(cls) -> None:
        if replay.source_manifest(replay.DEFAULT_SOURCE_RUN) != cls.source_manifest_before:
            raise AssertionError("run224 source tree changed during replay tests")

    def test_all_offline_replay_acceptance_criteria_pass(self) -> None:
        report = self.result["report"]
        self.assertTrue(report["all_acceptance_criteria_pass"])
        self.assertTrue(report["primary_matches_immutable_target"])
        self.assertTrue(report["primary_legacy_stage64_quality_pass"])
        self.assertGreaterEqual(report["semantic_clean_qa_count"], 3)
        self.assertEqual(report["api_calls_made"], 0)
        self.assertEqual(report["network_attempt_count"], 0)
        self.assertTrue(report["source_run_immutable"])
        self.assertTrue(report["graph_dialogue_alignment_hashes_unchanged"])
        self.assertTrue(report["claim_bundle_hash_stable"])

    def test_primary_exactly_matches_target_and_has_lossless_query_plan(self) -> None:
        sample = self.result["clean_samples"][0]
        qa = sample["qa"]
        target = sample["stage64_target"]
        self.assertEqual(sample["claim_first_primary_qa_id"], qa["qa_id"])
        self.assertEqual(
            sample["claim_first_multiqa_bundle"]["target_fulfillment_qa_id"],
            qa["qa_id"],
        )
        for field in (
            "qa_operation",
            "reasoning_subtype",
            "reasoning_template",
            "hop_count",
            "hard_negative_strategy",
        ):
            self.assertEqual(qa[field], target[field], field)
        self.assertEqual(qa["answer_type"], target["preferred_answer_type"])
        self.assertTrue(qa["forbid_max_time_shortcut"])
        event_ids = {
            event["event_instance_id"] for event in sample["graph_update"]["events"]
        }
        claim_event_map = qa["claim_event_map"]
        self.assertTrue(claim_event_map)
        self.assertEqual(len(claim_event_map), len(set(claim_event_map.values())))
        self.assertTrue(set(claim_event_map.values()).issubset(event_ids))
        query_plan = qa["query_plan"]
        self.assertTrue(query_plan["reasoning_anchor_path"])
        self.assertTrue(query_plan["resolved_required_event_instance_ids"])
        self.assertTrue(query_plan["resolved_gold_event_instance_ids"])
        self.assertTrue(query_plan["resolved_distractor_event_instance_ids"])
        for option in qa["options"]:
            self.assertIn(option["source_event_instance_id"], event_ids)

    def test_semantic_failures_are_only_sibling_rejects(self) -> None:
        report = self.result["report"]
        self.assertEqual(report["compiled_qa_count"], 6)
        self.assertEqual(report["semantic_clean_qa_count"], 4)
        self.assertEqual(report["semantic_rejected_qa_count"], 2)
        self.assertEqual(len(self.result["clean_samples"]), 1)
        self.assertEqual(len(self.result["sibling_qas_clean"]), 4)
        self.assertEqual(len(self.result["qa_rejects"]), 2)
        self.assertTrue(report["dialogue_retained_after_sibling_rejection"])

    def test_write_result_creates_only_new_output_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "replay"
            replay.write_result(self.result, output)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "clean_samples.jsonl",
                    "sibling_qas_clean.jsonl",
                    "qa_rejects.jsonl",
                    "report.json",
                    "report.md",
                },
            )
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["api_calls_made"], 0)
            self.assertTrue(report["all_acceptance_criteria_pass"])


if __name__ == "__main__":
    unittest.main()
