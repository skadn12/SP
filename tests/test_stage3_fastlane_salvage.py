from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import audit_stage3_fastlane_salvage as audit  # noqa: E402


RUN175 = (
    ROOT
    / "base_restart_current_method_validation"
    / "175_stage3_backfilled_smoke_n10"
)


class Stage3FastlaneSalvageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit.audit_run(RUN175, expected_candidate_count=10)

    def test_all_frozen_candidates_reconstruct_exactly(self) -> None:
        report = self.result["report"]
        self.assertEqual(report["frozen_final_qa_candidate_count"], 10)
        self.assertEqual(report["replay_pass_count"], 10)
        self.assertEqual(report["frozen_hash_triple_match_count"], 10)
        self.assertEqual(report["dialogue_alignment_dispersion_pass_count"], 10)

    def test_accepted_rows_match_persisted_oracle(self) -> None:
        checks = self.result["report"]["accepted_oracle_checks"]
        self.assertEqual(len(checks), 2)
        self.assertTrue(all(item["pass"] for item in checks))
        self.assertEqual(len(self.result["strict_clean_rows"]), 2)

    def test_truth_strict_classification(self) -> None:
        counts = self.result["report"]["classification_counts"]
        self.assertEqual(counts[audit.CLASS_STRICT_CLEAN], 2)
        self.assertEqual(counts[audit.CLASS_REPAIRABLE], 2)
        self.assertEqual(counts[audit.CLASS_DIAGNOSTIC], 1)
        self.assertEqual(counts[audit.CLASS_INVALID_TRUTH], 5)
        self.assertNotIn(audit.CLASS_UNRECONSTRUCTABLE, counts)

    def test_only_comparison_is_representation_repairable(self) -> None:
        repairable = [
            row
            for row in self.result["matrix"]
            if row["classification"] == audit.CLASS_REPAIRABLE
        ]
        self.assertEqual([row["qa_operation"] for row in repairable], ["comparison", "comparison"])
        self.assertTrue(
            all(row["truth_audit"]["answer_truth_pass"] for row in repairable)
        )

    def test_count_graph_only_membership_is_not_promoted(self) -> None:
        count_row = next(row for row in self.result["matrix"] if row["qa_operation"] == "count")
        self.assertEqual(count_row["classification"], audit.CLASS_INVALID_TRUTH)
        self.assertTrue(count_row["truth_audit"]["unique_graph_membership_contract"])
        self.assertFalse(count_row["truth_audit"]["dialogue_membership_decision_explicit"])

    def test_writer_keeps_repairable_rows_out_of_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit"
            audit.write_result(self.result, out, overwrite=False)
            strict = [
                json.loads(line)
                for line in (out / "strict_clean.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            repairable = [
                json.loads(line)
                for line in (out / "repairable_candidates.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(strict), 2)
            self.assertEqual(len(repairable), 2)
            self.assertTrue(all(row["classification"] == audit.CLASS_REPAIRABLE for row in repairable))


if __name__ == "__main__":
    unittest.main()
