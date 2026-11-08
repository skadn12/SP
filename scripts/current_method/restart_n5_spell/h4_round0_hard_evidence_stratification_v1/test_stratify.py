from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .stratify import classify_evidence


def _row(candidate: dict, *, split: str = "train", generation_source: str = "free_exploration") -> dict:
    return {
        "candidate_id": candidate["candidate_id"], "root_id": "cc-train-episode-1",
        "split": split, "question_type": "single_choice",
        "generation_source": generation_source, "difficulty_bucket": "hard_candidate",
        "position_mechanism": "mixed", "program_gate_status": "PROGRAM_PASS",
        "program_errors": [], "candidate_pool_path": "/pool.json",
        "candidate_pool_sha256": "p" * 64, "dialogue_path": "/dialogue.json",
        "dialogue_sha256": "d" * 64, "candidate": candidate,
    }


class ClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.refs = {"cc-train-episode-1:P01:T001", "R01:S01:T001"}
        self.base = {
            "candidate_id": "cc-train-episode-1:I001-C-Q01", "root_id": "cc-train-episode-1",
            "evidence_refs": ["cc-train-episode-1:P01:T001", "R01:S01:T001"],
            "evidence_groups": {"evidence": ["cc-train-episode-1:P01:T001", "R01:S01:T001"]},
            "contract_group_alignment": {"evidence": True}, "required_reasoning": ["evidence"],
            "reasoning_groups_used": ["evidence"], "schema_valid": True, "schema_errors": [],
        }

    def _classify(self, candidate: dict, *, gen: str = "free_exploration") -> dict:
        return classify_evidence(
            _row(candidate, generation_source=gen), valid_refs=self.refs,
            difficulty_manifest_path=Path("/difficulty_manifest.jsonl"),
            difficulty_manifest_sha256="f" * 64,
        )

    def test_clean_free_exploration_alignment_is_not_a_risk(self) -> None:
        result = self._classify(self.base)
        self.assertEqual(result["evidence_status"], "EVIDENCE_CLEAN")
        self.assertEqual(result["contract_alignment_status"], "NOT_APPLICABLE")
        self.assertFalse(result["strict_eligible"] is False and result["split"] == "train")

    def test_group_mismatch_is_risk(self) -> None:
        candidate = dict(self.base, evidence_groups={"evidence": ["cc-train-episode-1:P01:T001"]})
        result = self._classify(candidate)
        self.assertEqual(result["evidence_status"], "EVIDENCE_RISK")
        self.assertIn("flat_group_union_mismatch", result["evidence_risk_reasons"])

    def test_wrong_root_is_invalid(self) -> None:
        candidate = dict(self.base, evidence_refs=["cc-train-episode-2:P01:T001"])
        candidate["evidence_groups"] = {"evidence": ["cc-train-episode-2:P01:T001"]}
        result = self._classify(candidate)
        self.assertEqual(result["evidence_status"], "EVIDENCE_INVALID")
        self.assertIn("evidence_ref_wrong_root", result["evidence_invalid_reasons"])

    def test_missing_groups_is_unclassifiable(self) -> None:
        candidate = dict(self.base)
        candidate.pop("evidence_groups")
        result = self._classify(candidate)
        self.assertEqual(result["evidence_status"], "EVIDENCE_UNCLASSIFIABLE")

    def test_contract_false_is_risk(self) -> None:
        candidate = dict(self.base, contract_group_alignment={"evidence": False})
        result = self._classify(candidate, gen="contract_guided")
        self.assertEqual(result["evidence_status"], "EVIDENCE_RISK")
        self.assertEqual(result["contract_alignment_status"], "PARTIAL_OR_FALSE")

    def test_dev_never_gets_training_eligibility(self) -> None:
        result = classify_evidence(
            _row(self.base, split="dev"), valid_refs=self.refs,
            difficulty_manifest_path=Path("/difficulty_manifest.jsonl"),
            difficulty_manifest_sha256="f" * 64,
        )
        self.assertFalse(result["strict_eligible"])
        self.assertFalse(result["broad_only_eligible"])


if __name__ == "__main__":
    unittest.main()
