from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_expansion_qa import (
    validate_cross_boundary_core_object,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.qa_semantic_audit import (
    validate_qa_semantic_audit_output,
)


PILOT24 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_24"
)
PILOT24_SHA = "aa157ecf0d0cc15b06a4722f933393098253c8b7cc9c6985be71564701f065a7"


class BatchPilot24RecoveryTests(unittest.TestCase):
    def _qa_paths(self, root_id: str) -> tuple[Path, dict, dict, dict]:
        run = PILOT24 / f"roots/{root_id}/qa_candidates/candidate_01/02_questioner"
        qa = json.loads((run / "qa/core_qa.json").read_text())
        view = json.loads((run / "frozen/qa_source_view.json").read_text())
        catalog = json.loads((run / "frozen/span_catalog.json").read_text())
        return run, qa, view, catalog

    def test_both_historical_question_stems_are_now_rejected_for_detail_leakage(self):
        for root_id in ("cc-train-episode-216742", "cc-train-episode-41625"):
            _, qa, view, catalog = self._qa_paths(root_id)
            result = validate_cross_boundary_core_object(
                {
                    "question": qa["question"],
                    "answer_support_span_ids": qa["answer_support_span_ids"],
                },
                span_catalog=catalog,
                partition_by_turn=view["partition_by_turn"],
                lineage_pair_registry=view["lineage_pair_registry"],
                evidence_mismatch_policy="mark_only",
            )
            self.assertEqual("cross_boundary_qa_contract_reject",
                             result["decision"])
            self.assertIn("cross_boundary_qa_question_repeats_support_details",
                          result["reasons"])

    def test_semantic_auditor_accepts_a_concise_nonextractive_gold(self):
        _, qa, _, _ = self._qa_paths("cc-train-episode-216742")
        raw = {
            "audit_decision": "accept",
            "question_answerable": True,
            "answer_entailment": "fully_supported",
            "speaker_attribution": "correct",
            "temporality_and_modality": "correct",
            "cross_boundary_same_development": True,
            "cross_boundary_relation_entailment": "explicitly_supported",
            "support_set_sufficient": True,
            "question_answer_leakage": "clean",
            "grounded_answer": (
                "A recognized that pursuing a major creative goal still required "
                "taking care of routine responsibilities."),
            "unsupported_claims": [],
            "evidence_span_ids_checked": qa["answer_support_span_ids"],
            "reason": "The synthesis is supported by both selected moments.",
        }
        result = validate_qa_semantic_audit_output(
            raw,
            expected_span_ids=qa["answer_support_span_ids"],
            submitted_extractive_answer=qa["correct_answer_text"],
        )
        self.assertEqual("qa_semantic_audit_contract_accept",
                         result["decision"])

    def test_pilot24_spell_failures_remain_historical_failures(self):
        decisions = []
        for summary in sorted(PILOT24.glob(
                "roots/*/qa_candidates/candidate_*/05_spell_filter/reports/"
                "SPELL_FILTER_SUMMARY.json")):
            decisions.append(json.loads(summary.read_text())["final_decision"])
        self.assertEqual(["no_context_position_bias", "reject_easy"],
                         sorted(decisions))

    def test_pilot24_history_is_unchanged(self):
        self.assertEqual(PILOT24_SHA, hash_tree(PILOT24)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
