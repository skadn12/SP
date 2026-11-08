from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import build_stage3_claim_first_replay as replay223  # noqa: E402


RUN175 = (
    ROOT
    / "base_restart_current_method_validation"
    / "175_stage3_backfilled_smoke_n10"
)


class Stage3ClaimFirstReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = replay223.build_replay(
            RUN175,
            expected_candidate_count=10,
            min_qas=3,
            max_qas=6,
        )

    def test_all_ten_frozen_candidates_replay_without_api(self) -> None:
        report = self.result["report"]
        self.assertEqual(report["frozen_final_qa_candidate_count"], 10)
        self.assertEqual(report["replay_pass_count"], 10)
        self.assertEqual(report["api_calls_made"], 0)
        self.assertTrue(report["offline_only"])
        self.assertTrue(
            all(row["api_calls_made"] == 0 for row in self.result["dialogue_rows"])
        )

    def test_claim_truth_is_frozen_before_support_binding(self) -> None:
        report = self.result["report"]
        self.assertEqual(report["claim_bundle_hash_stable_count"], 10)
        self.assertEqual(len(self.result["claim_bundles"]), 10)
        for row in self.result["claim_bundles"]:
            self.assertTrue(row["claim_bundle_hash_stable"])
            self.assertEqual(
                row["pre_claim_bundle"]["claim_bundle_hash"],
                row["post_claim_bundle"]["claim_bundle_hash"],
            )
            self.assertEqual(
                row["claim_bundle_hash"],
                row["pre_claim_bundle"]["claim_bundle_hash"],
            )

    def test_every_compiled_dialogue_has_three_to_six_pre_qas(self) -> None:
        compiled = [
            row for row in self.result["dialogue_rows"] if row["claim_ir_compiled"]
        ]
        self.assertEqual(len(compiled), 10)
        for row in compiled:
            self.assertGreaterEqual(row["precompiled_qa_count"], 3)
            self.assertLessEqual(row["precompiled_qa_count"], 6)

    def test_each_qa_is_independent_and_proof_ready_gate_is_per_dialogue(self) -> None:
        ready_by_candidate: dict[str, list[dict]] = defaultdict(list)
        for row in self.result["proof_ready_qas"]:
            ready_by_candidate[row["candidate_id"]].append(row)
            self.assertTrue(row["validator_report"]["pass"])
            self.assertEqual(row["api_calls_made"], 0)
            self.assertTrue(row["qa"]["semantic_contract_frozen"])
            self.assertTrue(row["qa_instance_id"].startswith("qa_instance_"))
        self.assertEqual(
            len(
                {
                    row["qa_instance_id"]
                    for row in self.result["proof_ready_qas"]
                }
            ),
            len(self.result["proof_ready_qas"]),
        )
        for dialogue in self.result["dialogue_rows"]:
            accepted = ready_by_candidate.get(dialogue["candidate_id"], [])
            if dialogue["proof_ready"]:
                self.assertGreaterEqual(dialogue["validator_pass_qa_count"], 3)
                self.assertEqual(
                    len(accepted), dialogue["validator_pass_qa_count"]
                )
            else:
                self.assertEqual(accepted, [])
                self.assertLess(dialogue["validator_pass_qa_count"], 3)

    def test_failed_qa_does_not_delete_valid_siblings_or_dialogue(self) -> None:
        dialogue_ids = {row["candidate_id"] for row in self.result["dialogue_rows"]}
        self.assertEqual(len(dialogue_ids), 10)
        rejected_by_candidate: dict[str, int] = defaultdict(int)
        ready_by_candidate: dict[str, int] = defaultdict(int)
        for row in self.result["qa_rejects"]:
            rejected_by_candidate[row["candidate_id"]] += 1
        for row in self.result["proof_ready_qas"]:
            ready_by_candidate[row["candidate_id"]] += 1
        for candidate_id in dialogue_ids:
            if rejected_by_candidate[candidate_id] and ready_by_candidate[candidate_id]:
                self.assertGreater(ready_by_candidate[candidate_id], 0)
        self.assertFalse(
            self.result["report"]["single_qa_failure_deletes_siblings"]
        )

    def test_injected_single_qa_failure_keeps_siblings(self) -> None:
        original = replay223.claim_validator.validation_report
        call_count = 0

        def reject_first(bundle: dict) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "schema_version": "test_injected_validator",
                    "pass": False,
                    "issue_count": 1,
                    "issue_counts_by_code": {"injected_single_qa_failure": 1},
                    "issues": [
                        {
                            "code": "injected_single_qa_failure",
                            "path": "$.proof",
                            "message": "test-only independent QA rejection",
                        }
                    ],
                }
            return original(bundle)

        with mock.patch.object(
            replay223.claim_validator,
            "validation_report",
            side_effect=reject_first,
        ):
            result = replay223.build_replay(RUN175, expected_candidate_count=10)

        first = result["dialogue_rows"][0]
        self.assertTrue(first["proof_ready"])
        self.assertEqual(first["validator_reject_qa_count"], 1)
        self.assertGreaterEqual(first["validator_pass_qa_count"], 3)
        self.assertEqual(
            len(
                [
                    row
                    for row in result["proof_ready_qas"]
                    if row["candidate_id"] == first["candidate_id"]
                ]
            ),
            first["validator_pass_qa_count"],
        )
        injected = [
            row
            for row in result["qa_rejects"]
            if row["rejection_reason"] == "claim_validator_rejected_qa"
            and (row["validator_report"] or {}).get("issue_counts_by_code", {}).get(
                "injected_single_qa_failure"
            )
        ]
        self.assertEqual(len(injected), 1)

    def test_family_split_is_stable_and_has_no_leakage(self) -> None:
        report = self.result["report"]
        self.assertEqual(report["family_split_leak_count"], 0)
        splits: dict[str, set[str]] = defaultdict(set)
        for row in self.result["dialogue_rows"]:
            if row.get("dialogue_family_id"):
                splits[row["dialogue_family_id"]].add(
                    row["dialogue_family_split"]
                )
        self.assertTrue(splits)
        self.assertTrue(all(len(values) == 1 for values in splits.values()))
        for row in self.result["proof_ready_qas"]:
            self.assertEqual(
                row["dialogue_family_split"],
                row["qa"]["dialogue_family_split"],
            )

    def test_run175_tree_hashes_are_identical_before_and_after(self) -> None:
        report = self.result["report"]
        self.assertTrue(report["source_run_immutable"])
        self.assertEqual(
            report["source_tree_sha256_before"],
            report["source_tree_sha256_after"],
        )
        self.assertEqual(
            report["source_file_sha256_before"],
            report["source_file_sha256_after"],
        )

    def test_writer_emits_all_required_artifacts_outside_source(self) -> None:
        before = replay223.source_tree_manifest(RUN175)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "223"
            replay223.write_result(self.result, output_dir)
            expected = {
                "claim_bundles.jsonl",
                "proof_ready_qas.jsonl",
                "qa_rejects.jsonl",
                "dialogue_rows.jsonl",
                "report.json",
                "report.md",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()}, expected
            )
            report = json.loads(
                (output_dir / "report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["api_calls_made"], 0)
            dialogue_lines = [
                line
                for line in (output_dir / "dialogue_rows.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(dialogue_lines), 10)
        self.assertEqual(before, replay223.source_tree_manifest(RUN175))

    def test_writer_rejects_output_inside_source_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the immutable source"):
            replay223.write_result(
                self.result,
                RUN175 / "forbidden-223-output",
            )


if __name__ == "__main__":
    unittest.main()
