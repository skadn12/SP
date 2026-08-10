from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_expansion_qa import (
    MAX_ELIGIBLE_PAIRS_PER_LINEAGE_ROW,
    build_cross_boundary_lineage_pair_registry,
    build_qa_source_view,
    validate_cross_boundary_core_object,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.mcq_option_diagnostics import (
    GOLD_LENGTH_ADVANTAGE_RATIO,
    diagnose_option_shape,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


PILOT23 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_23"
)
PILOT23_SHA = "c19e335d3a0eb5a237f5de27fe87b80fae1653d45e1e9eac86681beb71721aab"


class BatchPilot23RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expansion = PILOT23 / "roots/cc-train-episode-216742/01_expansion"
        expanded = json.loads((
            cls.expansion / "expanded_dialogue_round_1.json").read_text())
        cls.view = build_qa_source_view(expanded)
        cls.registry = build_cross_boundary_lineage_pair_registry(
            round_validator_input=json.loads((
                cls.expansion / "round_validator_input.json").read_text()),
            span_catalog=cls.view["span_catalog"],
            partition_by_turn=cls.view["partition_by_turn"],
        )
        cls.catalog = {
            span["span_id"]: span
            for span in cls.view["span_catalog"]["spans"]}

    def test_registry_exposes_only_finite_exact_same_speaker_pairs(self):
        self.assertTrue(self.registry["rows"])
        for row in self.registry["rows"]:
            pairs = row["eligible_same_speaker_pairs"]
            self.assertGreater(len(pairs), 0)
            self.assertLessEqual(len(pairs),
                                 MAX_ELIGIBLE_PAIRS_PER_LINEAGE_ROW)
            for pair in pairs:
                parent = self.catalog[pair["parent_span_id"]]
                child = self.catalog[pair["child_span_id"]]
                self.assertEqual(parent["speaker"], child["speaker"])
                self.assertEqual(parent["speaker"], pair["speaker"])

    def test_historical_mixed_speaker_selection_remains_rejected(self):
        raw_path = next((PILOT23 / (
            "roots/cc-train-episode-216742/qa_candidates/candidate_01/"
            "02_questioner/raw/question_answer_writer")).glob("*a00.json"))
        raw = json.loads(json.loads(raw_path.read_text())["raw_response"])
        result = validate_cross_boundary_core_object(
            raw,
            span_catalog=self.view["span_catalog"],
            partition_by_turn=self.view["partition_by_turn"],
            lineage_pair_registry=self.registry,
            evidence_mismatch_policy="mark_only",
        )
        self.assertEqual("cross_boundary_qa_contract_reject", result["decision"])
        self.assertIn("cross_boundary_qa_requires_same_speaker",
                      result["reasons"])

    def test_pilot23_gold_length_outlier_is_detected_before_solver(self):
        mcq = json.loads((PILOT23 / (
            "roots/cc-train-episode-41625/qa_candidates/candidate_02/"
            "04_mcq_materializer/qa/final_mcq.json")).read_text())
        diagnostic = diagnose_option_shape(
            options=mcq["options"],
            reference_answer_letter=mcq["reference_answer_letter"],
            question=mcq["question"],
        )
        self.assertTrue(diagnostic["gold_length_leakage"])
        self.assertGreaterEqual(diagnostic["gold_length_advantage_ratio"],
                                GOLD_LENGTH_ADVANTAGE_RATIO)
        self.assertTrue(diagnostic["answer_shape_leakage_risk"])

    def test_pilot23_history_is_unchanged(self):
        self.assertEqual(PILOT23_SHA, hash_tree(PILOT23)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
