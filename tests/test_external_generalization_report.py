from __future__ import annotations

import unittest

from scripts.current_method import external_generalization_report as subject


def payload(name: str, correctness: list[bool], labels: list[str] | None = None) -> dict:
    labels = labels or ["A"] * len(correctness)
    return {
        "model": name,
        "temperature": 0.0,
        "top_p": 1.0,
        "summary": {},
        "results": [
            {
                "record_id": f"r{index}",
                "scored": True,
                "correct": correct,
                "gold_label": labels[index],
                "pred_label": labels[index] if correct else "D",
                "dataset": "longbench_v2",
                "prompt_bucket": "16k-24k",
            }
            for index, correct in enumerate(correctness)
        ],
    }


class ExternalReportTest(unittest.TestCase):
    def test_paired_counts_and_exact_test(self) -> None:
        baseline = payload("base", [True, True, False, False])
        candidate = payload("candidate", [True, False, True, False])
        stats = subject.paired_stats(baseline, candidate)
        self.assertEqual((stats["both_correct"], stats["baseline_only"], stats["candidate_only"], stats["neither_correct"]), (1, 1, 1, 1))
        self.assertEqual(stats["mcnemar_exact_p"], 1.0)

    def test_summary_contains_label_strata(self) -> None:
        summary = subject.build_summary(
            {"r2": payload("r2", [True, False], ["A", "B"]), "stage64": payload("stage64", [False, True], ["A", "B"])},
            baseline="r2",
        )
        self.assertEqual(summary["models"]["stage64"]["label_stats"]["by_gold"]["B"]["accuracy"], 1.0)

    def test_metadata_mismatch_is_rejected(self) -> None:
        baseline = payload("base", [True])
        candidate = payload("candidate", [True])
        candidate["results"][0]["gold_label"] = "B"
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            subject.paired_stats(baseline, candidate)

    def test_decode_configuration_mismatch_fails_health_gate(self) -> None:
        baseline = payload("base", [True])
        candidate = payload("candidate", [True])
        baseline["max_new_tokens"] = 16
        candidate["max_new_tokens"] = 32
        with self.assertRaisesRegex(RuntimeError, "max_new_tokens differs"):
            subject.build_summary({"r2": baseline, "candidate": candidate}, baseline="r2")


if __name__ == "__main__":
    unittest.main()
