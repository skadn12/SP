from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.mcq_option_diagnostics import (
    diagnose_option_shape,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


PILOT26 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_26"
)
PILOT26_SHA = "93a6ee0e093b0a68141b337f30bfdcfb7a244e2a2ebf553c7b5b3e2dac276eed"


class BatchPilot26RecoveryTests(unittest.TestCase):
    def _mcq(self, candidate: str) -> dict:
        path = PILOT26 / (
            "roots/cc-train-episode-216742/qa_candidates/" + candidate +
            "/04_mcq_materializer/qa/final_mcq.json")
        return json.loads(path.read_text())

    def test_candidate_three_two_neighbour_mutations_are_now_rejected(self) -> None:
        mcq = self._mcq("candidate_03")
        report = diagnose_option_shape(
            options=mcq["options"],
            reference_answer_letter=mcq["reference_answer_letter"],
            question=mcq["question"],
        )
        self.assertEqual(2, report["small_local_mutation_distractor_count"])
        self.assertTrue(report["single_mutation_star"])
        self.assertTrue(report["answer_shape_leakage_risk"])

    def test_candidate_four_shared_false_component_is_now_rejected(self) -> None:
        mcq = self._mcq("candidate_04")
        report = diagnose_option_shape(
            options=mcq["options"],
            reference_answer_letter=mcq["reference_answer_letter"],
            question=mcq["question"],
        )
        self.assertIn("more careless",
                      report["shared_distractor_only_bigrams"])
        self.assertTrue(report["shared_false_trajectory_risk"])
        self.assertTrue(report["answer_shape_leakage_risk"])

    def test_historical_spell_failure_remains_grounding_inconclusive(self) -> None:
        path = PILOT26 / (
            "roots/cc-train-episode-216742/qa_candidates/candidate_03/"
            "05_spell_filter/reports/SPELL_FILTER_SUMMARY.json")
        summary = json.loads(path.read_text())
        self.assertEqual("grounding_inconclusive", summary["final_decision"])
        self.assertFalse(summary["qa_pipeline_ready"])

    def test_pilot26_history_is_unchanged(self) -> None:
        self.assertEqual(PILOT26_SHA, hash_tree(PILOT26)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
