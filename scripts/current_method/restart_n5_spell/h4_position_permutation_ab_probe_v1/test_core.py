from __future__ import annotations

import unittest

from .core import (
    SOURCE_RUN,
    build_permutations,
    load_hard_pool,
    profile_hard,
    stratified_sample,
)


class PositionProbeOfflineTests(unittest.TestCase):
    def test_frozen_hard_pool_contract(self) -> None:
        rows, report = load_hard_pool(SOURCE_RUN)
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(len(rows), 71)
        self.assertEqual(report["observed"]["question_type"], {
            "single_choice": 9, "multi_select": 35, "ordering": 27,
        })
        self.assertEqual(report["observed"]["generation_source"], {
            "contract_guided": 27, "free_exploration": 44,
        })

    def test_profile_variance_decomposition_and_sample_quotas(self) -> None:
        rows, report = load_hard_pool(SOURCE_RUN)
        self.assertTrue(report["passed"])
        profiles = {row["candidate_id"]: profile_hard(row) for row in rows}
        sample = stratified_sample(rows, profiles, source_run_id=SOURCE_RUN.name)
        self.assertEqual(len(sample), 30)
        self.assertEqual(
            {q: sum(row["candidate"]["question_type"] == q for row in sample)
             for q in ("single_choice", "multi_select", "ordering")},
            {"single_choice": 6, "multi_select": 12, "ordering": 12},
        )
        roots = {row["root_id"] for row in sample}
        self.assertEqual(len(roots), 8)
        self.assertTrue(all(2 <= sum(row["root_id"] == root for row in sample) <= 5 for root in roots))
        for profile in profiles.values():
            full = profile["full_context"]
            self.assertAlmostEqual(
                full["total_variance"],
                full["within_position_uncertainty"] + full["between_position_sensitivity"],
                places=9,
            )

    def test_permutation_counts_and_cyclic_subset(self) -> None:
        rows, report = load_hard_pool(SOURCE_RUN)
        self.assertTrue(report["passed"])
        profiles = {row["candidate_id"]: profile_hard(row) for row in rows}
        sample = stratified_sample(rows, profiles, source_run_id=SOURCE_RUN.name)
        counts = {"single_choice": 0, "multi_select": 0, "ordering": 0}
        cyclic = {"single_choice": 0, "multi_select": 0, "ordering": 0}
        for row in sample:
            permutations = build_permutations(row, source_run_id=SOURCE_RUN.name)
            qtype = row["candidate"]["question_type"]
            counts[qtype] += len(permutations)
            cyclic[qtype] += sum(bool(item["is_cyclic"]) for item in permutations)
            self.assertEqual(len({item["permutation_hash"] for item in permutations}), len(permutations))
            self.assertTrue(all(item["canonical_gold"] for item in permutations))
            for item in permutations:
                expected = [item["canonical_to_display_mapping"][gold] for gold in item["canonical_gold"]]
                self.assertEqual(item["displayed_gold"], expected)
        self.assertEqual(counts, {"single_choice": 144, "multi_select": 360, "ordering": 288})
        self.assertEqual(cyclic, {"single_choice": 24, "multi_select": 60, "ordering": 48})


if __name__ == "__main__":
    unittest.main()
