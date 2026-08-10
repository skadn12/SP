from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_expansion_qa import (
    build_cross_boundary_lineage_pair_registry,
    build_cross_boundary_questioner_prompt,
    validate_cross_boundary_core_object,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


PILOT22 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_22"
)
PILOT22_SHA = "fc3b3d6689244f17c4f4cb39cad01af3b01220ff58b7e7a33973c6cc77759181"


class BatchPilot22QARecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = PILOT22 / "roots/cc-train-episode-41625"
        frozen = cls.root / "qa_candidates/candidate_01/02_questioner/frozen"
        cls.view = json.loads((frozen / "qa_source_view.json").read_text())
        cls.catalog = json.loads((frozen / "span_catalog.json").read_text())
        cls.catalog_by_id = {
            span["span_id"]: span for span in cls.catalog["spans"]}
        expansion = cls.root / "01_expansion"
        cls.current_registry = build_cross_boundary_lineage_pair_registry(
            round_validator_input=json.loads((
                expansion / "round_validator_input.json").read_text()),
            span_catalog=cls.catalog,
            partition_by_turn=cls.view["partition_by_turn"],
        )
        cls.historical = [
            json.loads((cls.root / (
                f"qa_candidates/candidate_{index:02d}/02_questioner/qa/core_qa.json"
            )).read_text())
            for index in (1, 2)
        ]

    def validate(self, raw, **kwargs):
        return validate_cross_boundary_core_object(
            raw,
            span_catalog=self.catalog,
            partition_by_turn=self.view["partition_by_turn"],
            lineage_pair_registry=self.view["lineage_pair_registry"],
            evidence_mismatch_policy="mark_only",
            **kwargs,
        )

    def test_both_historical_candidates_are_rejected_for_mixed_speakers(self):
        for core in self.historical:
            result = self.validate({
                "question": core["question"],
                "answer_support_span_ids": core["answer_support_span_ids"],
            })
            self.assertEqual("cross_boundary_qa_contract_reject",
                             result["decision"])
            self.assertIn("cross_boundary_qa_requires_same_speaker",
                          result["reasons"])

    def test_second_candidate_cannot_reuse_first_support_pair(self):
        first_pair = tuple(self.historical[0]["answer_support_span_ids"])
        second = self.historical[1]
        result = self.validate({
            "question": second["question"],
            "answer_support_span_ids": second["answer_support_span_ids"],
        }, excluded_support_pairs=(first_pair,))
        self.assertIn("cross_boundary_qa_repeats_excluded_support_pair",
                      result["reasons"])

    def test_excluded_pair_is_order_independent(self):
        pair = tuple(self.historical[0]["answer_support_span_ids"])
        result = self.validate({
            "question": "What changed across the two moments?",
            "answer_support_span_ids": list(reversed(pair)),
        }, excluded_support_pairs=(pair,))
        self.assertIn("cross_boundary_qa_repeats_excluded_support_pair",
                      result["reasons"])

    def test_same_speaker_lineage_pair_passes(self):
        row = next(row for row in self.current_registry["rows"]
                   if row["source_anchor_id"] == "PA0004")
        pair = row["eligible_same_speaker_pairs"][0]
        parent_id = pair["parent_span_id"]
        child_id = pair["child_span_id"]
        result = validate_cross_boundary_core_object(
            {
                "question": (
                    "How did Classmates A's earlier uncertainty change in the "
                    "later development?"),
                "answer_support_span_ids": [parent_id, child_id],
            },
            span_catalog=self.catalog,
            partition_by_turn=self.view["partition_by_turn"],
            lineage_pair_registry=self.current_registry,
            evidence_mismatch_policy="mark_only",
        )
        self.assertEqual("cross_boundary_qa_accept", result["decision"])
        self.assertEqual(
            self.catalog_by_id[parent_id]["speaker"],
            self.catalog_by_id[child_id]["speaker"],
        )

    def test_prompt_freezes_support_pair_exclusions(self):
        pair = tuple(self.historical[0]["answer_support_span_ids"])
        prompt = build_cross_boundary_questioner_prompt(
            root_id="cc-train-episode-41625",
            span_catalog=self.catalog,
            partition_by_turn=self.view["partition_by_turn"],
            lineage_pair_registry=self.view["lineage_pair_registry"],
            excluded_support_pairs=(pair,),
        )
        self.assertIn("EXCLUDED PRIOR SUPPORT PAIRS", prompt)
        self.assertIn("same exact speaker label", prompt)
        self.assertIn(pair[0], prompt)
        self.assertIn(pair[1], prompt)

    def test_pilot22_history_is_unchanged(self):
        self.assertEqual(PILOT22_SHA, hash_tree(PILOT22)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
