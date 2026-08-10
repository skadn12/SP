from __future__ import annotations

import unittest

from scripts.current_method.restart_n5_spell.questioner_contract import (
    build_span_catalog,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_expansion_qa import (
    build_cross_boundary_questioner_prompt,
    validate_cross_boundary_core_object,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_qa_semantic_audit_1 import (
    classify_semantic_audit_admission,
)


class QuestionerOwnedEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        turns = [
            {"turn_id": "root:P01:T001", "session_id": "P01",
             "speaker": "A", "text": "A initially postponed the meeting."},
            {"turn_id": "root:P01:T002", "session_id": "P01",
             "speaker": "B", "text": "B noted that the report was incomplete."},
            {"turn_id": "R01:S01:T001", "session_id": "R01:S01",
             "speaker": "B", "text": "B later finished the report."},
            {"turn_id": "R01:S01:T002", "session_id": "R01:S01",
             "speaker": "A", "text": "A then scheduled the meeting."},
        ]
        self.catalog = build_span_catalog(turns)
        self.partitions = {
            "root:P01:T001": "original_parent",
            "root:P01:T002": "original_parent",
            "R01:S01:T001": "promoted_child",
            "R01:S01:T002": "promoted_child",
        }

    def raw(self, support: list[str] | None = None) -> dict:
        return {
            "question": "How did the preparation change before the meeting?",
            "correct_answer_text": (
                "B completed the report before A scheduled the meeting."),
            "answer_support_span_ids": support or [
                "root:P01:T002#S0001", "R01:S01:T001#S0001",
                "R01:S01:T002#S0001"],
        }

    def test_prompt_assigns_evidence_and_answer_to_questioner(self) -> None:
        prompt = build_cross_boundary_questioner_prompt(
            root_id="root", span_catalog=self.catalog,
            partition_by_turn=self.partitions)
        self.assertIn('"correct_answer_text"', prompt)
        self.assertIn("Select 2 to 6 exact evidence spans", prompt)
        self.assertIn("program will not choose evidence for you", prompt)
        self.assertNotIn("FROZEN SUPPORT PAIR", prompt)
        self.assertNotIn("ALLOWED LINEAGE PAIRS", prompt)

    def test_cross_speaker_three_span_evidence_is_reference_valid(self) -> None:
        result = validate_cross_boundary_core_object(
            self.raw(), span_catalog=self.catalog,
            partition_by_turn=self.partitions)
        self.assertEqual("cross_boundary_qa_accept", result["decision"])
        self.assertTrue(result["reference_integrity_verified"])
        self.assertTrue(result["questioner_selected_evidence"])
        self.assertFalse(result["gold_materialized_locally"])
        self.assertEqual(self.raw()["correct_answer_text"],
                         result["core_qa"]["correct_answer_text"])

    def test_unknown_evidence_id_is_still_blocking(self) -> None:
        result = validate_cross_boundary_core_object(
            self.raw(["root:P01:T002#S0001", "unknown#S0001"]),
            span_catalog=self.catalog, partition_by_turn=self.partitions,
            evidence_mismatch_policy="mark_only")
        self.assertEqual("cross_boundary_qa_schema_invalid",
                         result["decision"])

    def test_parent_only_evidence_is_valid_and_classified(self) -> None:
        result = validate_cross_boundary_core_object(
            self.raw(["root:P01:T001#S0001", "root:P01:T002#S0001"]),
            span_catalog=self.catalog, partition_by_turn=self.partitions)
        self.assertEqual("cross_boundary_qa_accept", result["decision"])
        self.assertEqual("parent_only", result["evidence_scope"])
        self.assertFalse(result["child_involved"])
        self.assertFalse(result["scope_is_quality_gate"])

    def test_child_only_evidence_is_valid_and_classified(self) -> None:
        result = validate_cross_boundary_core_object(
            self.raw(["R01:S01:T001#S0001", "R01:S01:T002#S0001"]),
            span_catalog=self.catalog, partition_by_turn=self.partitions)
        self.assertEqual("cross_boundary_qa_accept", result["decision"])
        self.assertEqual("child_only", result["evidence_scope"])
        self.assertTrue(result["child_involved"])

    def test_answer_valid_evidence_insufficient_is_marked_not_training_ready(self) -> None:
        audit = {
            "audit_decision": "accept",
            "support_set_sufficient": False,
        }
        admission = classify_semantic_audit_admission(
            qa={}, audit=audit, evidence_mismatch_policy="mark_only")
        self.assertTrue(admission["downstream_observation_ready"])
        self.assertTrue(admission["evidence_mismatch_marked"])
        self.assertFalse(admission["training_ready"])
        self.assertFalse(admission["strictly_verified"])

    def test_answer_invalid_is_not_reclassified_as_evidence_only(self) -> None:
        audit = {
            "audit_decision": "reject",
            "support_set_sufficient": False,
        }
        admission = classify_semantic_audit_admission(
            qa={}, audit=audit, evidence_mismatch_policy="mark_only")
        self.assertFalse(admission["admitted"])
        self.assertFalse(admission["answer_semantically_verified"])


if __name__ == "__main__":
    unittest.main()
