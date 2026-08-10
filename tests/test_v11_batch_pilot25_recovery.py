from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_expansion_qa import (
    validate_cross_boundary_core_object,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


PILOT25 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_25"
)
PILOT25_SHA = "392cf075af4a91c721b9b1965f6578a2a4e356ba7a999deb28469b9ad20f05c7"


class BatchPilot25RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = PILOT25 / (
            "roots/cc-train-episode-41625/qa_candidates/candidate_01")
        cls.qa = json.loads((
            cls.base / "02_questioner/qa/core_qa.json").read_text())
        cls.view = json.loads((
            cls.base / "02_questioner/frozen/qa_source_view.json").read_text())
        cls.catalog = json.loads((
            cls.base / "02_questioner/frozen/span_catalog.json").read_text())

    def test_historical_from_to_stem_is_rejected_as_endpoint_leakage(self):
        result = validate_cross_boundary_core_object(
            {
                "question": self.qa["question"],
                "answer_support_span_ids": self.qa["answer_support_span_ids"],
            },
            span_catalog=self.catalog,
            partition_by_turn=self.view["partition_by_turn"],
            lineage_pair_registry=self.view["lineage_pair_registry"],
            evidence_mismatch_policy="mark_only",
        )
        self.assertEqual("cross_boundary_qa_contract_reject", result["decision"])
        diagnostic = result["question_support_detail_leakage"]
        self.assertTrue(diagnostic["states_both_transition_endpoints"])

    def test_historical_distractors_are_all_small_gold_mutations(self):
        summary = json.loads((self.base / (
            "04_mcq_materializer/reports/MCQ_MATERIALIZER_SUMMARY.json"
        )).read_text())
        attempts = summary["distractor_result"]["attempts"]
        self.assertEqual(2, len(attempts))
        for attempt in attempts:
            self.assertIn("gold_plus_three_small_mutations",
                          attempt["validation"]["reasons"])

    def test_semantic_gold_is_not_the_extractive_source_copy(self):
        gold = json.loads((
            self.base / "03_semantic_audit/qa/admitted_gold.json").read_text())
        self.assertNotEqual(gold["correct_answer_text"],
                            gold["source_extractive_answer_text"])

    def test_pilot25_history_is_unchanged(self):
        self.assertEqual(PILOT25_SHA, hash_tree(PILOT25)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
