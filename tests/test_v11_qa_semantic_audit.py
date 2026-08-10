from __future__ import annotations

import unittest

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.qa_semantic_audit import (
    build_qa_semantic_audit_prompt,
    validate_qa_semantic_audit_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_qa_semantic_audit_1 import (
    classify_semantic_audit_admission,
)


SPAN_IDS = ["P05:T005#S0001", "R01:S04:T001#S0001"]


def accepted() -> dict:
    return {
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
            "The speaker moved from uncertainty to a concrete plan supported "
            "by the later development."),
        "unsupported_claims": [],
        "evidence_gaps": [],
        "evidence_span_ids_checked": list(SPAN_IDS),
        "reason": "Both clauses are directly supported.",
    }


class QASemanticAuditTest(unittest.TestCase):
    def test_accept_contract(self) -> None:
        result = validate_qa_semantic_audit_output(
            accepted(), expected_span_ids=SPAN_IDS)
        self.assertEqual("qa_semantic_audit_contract_accept", result["decision"])
        self.assertTrue(result["semantic_entailment_verified"])

    def test_non_cross_boundary_scope_does_not_require_cross_relation(self) -> None:
        value = accepted()
        value["cross_boundary_same_development"] = False
        value["cross_boundary_relation_entailment"] = "unrelated"
        result = validate_qa_semantic_audit_output(
            value, expected_span_ids=SPAN_IDS,
            requires_cross_boundary=False)
        self.assertEqual("qa_semantic_audit_contract_accept", result["decision"])
        self.assertTrue(result["semantic_entailment_verified"])

    def test_valid_reject_is_not_schema_error(self) -> None:
        value = accepted()
        value.update({"audit_decision": "reject",
                      "answer_entailment": "partially_supported",
                      "grounded_answer": "",
                      "unsupported_claims": ["The answer adds an unsupported state."],
                      "reason": "One clause is overstated."})
        result = validate_qa_semantic_audit_output(
            value, expected_span_ids=SPAN_IDS)
        self.assertEqual("qa_semantic_audit_contract_accept", result["decision"])
        self.assertFalse(result["semantic_entailment_verified"])

    def test_auditor_cannot_change_evidence(self) -> None:
        value = accepted()
        value["evidence_span_ids_checked"] = [SPAN_IDS[0]]
        result = validate_qa_semantic_audit_output(
            value, expected_span_ids=SPAN_IDS)
        self.assertEqual("qa_semantic_audit_schema_invalid", result["decision"])
        self.assertIn("audit_evidence_set_changed", result["errors"])

    def test_accept_requires_all_semantic_gates(self) -> None:
        value = accepted()
        value["temporality_and_modality"] = "incorrect"
        result = validate_qa_semantic_audit_output(
            value, expected_span_ids=SPAN_IDS)
        self.assertEqual("qa_semantic_audit_schema_invalid", result["decision"])
        self.assertIn("audit_accept_conditions_not_met", result["errors"])

    def test_prompt_limits_auditor_to_submitted_evidence(self) -> None:
        prompt = build_qa_semantic_audit_prompt(
            core_qa={"question": "Q", "correct_answer_text": "A",
                     "answer_support_span_ids": SPAN_IDS},
            selected_evidence=[
                {"span_id": SPAN_IDS[0], "source_partition": "original_parent",
                 "session_id": "P05", "speaker": "A", "exact_text": "old"},
                {"span_id": SPAN_IDS[1], "source_partition": "promoted_child",
                 "session_id": "S04", "speaker": "A", "exact_text": "new"},
            ],
            complete_dialogue=[
                {"turn_id": "P05:T005", "source_partition": "original_parent",
                 "session_id": "P05", "speaker": "A", "text": "old"},
                {"turn_id": "R01:S04:T001",
                 "source_partition": "promoted_child", "session_id": "S04",
                 "speaker": "A", "text": "new"},
            ])
        self.assertIn("Do not rewrite the question, replace evidence", prompt)
        self.assertIn("planned/intended/possible", prompt)
        self.assertIn("8-40 words", prompt)
        self.assertIn("question_answer_leakage", prompt)
        self.assertIn("Temporal order, the same speaker", prompt)
        self.assertIn("COMPLETE_DIALOGUE", prompt)
        self.assertIn("support_set_sufficient and evidence_gaps describe only",
                      prompt)

    def test_accept_cannot_upgrade_inferred_causality(self) -> None:
        value = accepted()
        value["cross_boundary_relation_entailment"] = "requires_inference"
        result = validate_qa_semantic_audit_output(
            value, expected_span_ids=SPAN_IDS)
        self.assertEqual("qa_semantic_audit_schema_invalid", result["decision"])
        self.assertIn("audit_accept_conditions_not_met", result["errors"])

    def test_mark_only_does_not_admit_semantic_reject(self) -> None:
        audit = accepted()
        audit.update({"audit_decision": "reject",
                      "answer_entailment": "partially_supported",
                      "grounded_answer": "",
                      "unsupported_claims": ["Question wording is not supported."],
                      "reason": "The evidence does not support one clause."})
        result = classify_semantic_audit_admission(
            qa={"evidence_mismatch": True,
                "evidence_mismatch_reasons": ["lineage mismatch"]},
            audit=audit, evidence_mismatch_policy="mark_only")
        self.assertFalse(result["admitted"])
        self.assertFalse(result["strictly_verified"])
        self.assertFalse(result["evidence_mismatch_marked"])

    def test_mark_only_admits_verified_answer_and_marks_lineage_mismatch(self) -> None:
        result = classify_semantic_audit_admission(
            qa={"evidence_mismatch": True,
                "evidence_mismatch_reasons": ["lineage mismatch"]},
            audit=accepted(), evidence_mismatch_policy="mark_only")
        self.assertTrue(result["admitted"])
        self.assertFalse(result["strictly_verified"])
        self.assertTrue(result["evidence_mismatch_marked"])
        self.assertFalse(result["training_ready"])
        self.assertTrue(result["downstream_observation_ready"])

    def test_mark_only_continues_when_answer_is_verified_but_evidence_is_insufficient(self) -> None:
        audit = accepted()
        audit["support_set_sufficient"] = False
        audit["evidence_gaps"] = [
            "The selected evidence omits the earlier supporting turn."]
        validation = validate_qa_semantic_audit_output(
            audit, expected_span_ids=SPAN_IDS)
        self.assertEqual("qa_semantic_audit_contract_accept",
                         validation["decision"])
        self.assertTrue(validation["answer_semantically_verified"])
        self.assertFalse(validation["submitted_evidence_sufficient"])
        self.assertFalse(validation["semantic_entailment_verified"])
        admission = classify_semantic_audit_admission(
            qa={}, audit=audit, evidence_mismatch_policy="mark_only")
        self.assertTrue(admission["admitted"])
        self.assertTrue(admission["evidence_mismatch_marked"])
        self.assertFalse(admission["training_ready"])

    def test_strict_policy_blocks_verified_answer_with_insufficient_evidence(self) -> None:
        audit = accepted()
        audit["support_set_sufficient"] = False
        audit["evidence_gaps"] = [
            "The selected evidence omits the earlier supporting turn."]
        admission = classify_semantic_audit_admission(
            qa={}, audit=audit, evidence_mismatch_policy="strict")
        self.assertFalse(admission["admitted"])
        self.assertTrue(admission["answer_semantically_verified"])
        self.assertFalse(admission["submitted_evidence_sufficient"])

    def test_strict_policy_keeps_semantic_reject_blocking(self) -> None:
        audit = accepted()
        audit.update({"audit_decision": "uncertain",
                      "answer_entailment": "uncertain",
                      "grounded_answer": "",
                      "unsupported_claims": ["Connection is unclear."],
                      "reason": "Insufficient evidence."})
        result = classify_semantic_audit_admission(
            qa={}, audit=audit, evidence_mismatch_policy="strict")
        self.assertFalse(result["admitted"])
        self.assertFalse(result["evidence_mismatch_marked"])

    def test_accept_rejects_extractive_gold_copy(self) -> None:
        value = accepted()
        extractive = value["grounded_answer"]
        result = validate_qa_semantic_audit_output(
            value, expected_span_ids=SPAN_IDS,
            submitted_extractive_answer=extractive)
        self.assertEqual("qa_semantic_audit_schema_invalid", result["decision"])
        self.assertIn("audit_grounded_answer_is_extractive_copy",
                      result["errors"])

    def test_accept_rejects_question_answer_leakage(self) -> None:
        value = accepted()
        value["question_answer_leakage"] = "leaking"
        result = validate_qa_semantic_audit_output(
            value, expected_span_ids=SPAN_IDS)
        self.assertEqual("qa_semantic_audit_schema_invalid", result["decision"])
        self.assertIn("audit_accept_conditions_not_met", result["errors"])

    def test_contract_failure_is_not_reclassified_as_evidence_mismatch(self) -> None:
        result = classify_semantic_audit_admission(
            qa={}, audit=None, evidence_mismatch_policy="mark_only")
        self.assertFalse(result["admitted"])
        self.assertFalse(result["evidence_mismatch_marked"])

    def test_fully_supported_answer_cannot_list_unsupported_claims(self) -> None:
        value = accepted()
        value["audit_decision"] = "reject"
        value["grounded_answer"] = ""
        value["unsupported_claims"] = [
            "A claim allegedly missing only from selected evidence."]
        value["support_set_sufficient"] = False
        value["evidence_gaps"] = ["The selected evidence omits its support."]
        result = validate_qa_semantic_audit_output(
            value, expected_span_ids=SPAN_IDS)
        self.assertEqual("qa_semantic_audit_schema_invalid", result["decision"])
        self.assertIn("audit_full_dialogue_entailment_claims_inconsistent",
                      result["errors"])

    def test_answer_accept_with_evidence_gap_has_split_decisions(self) -> None:
        value = accepted()
        value["support_set_sufficient"] = False
        value["evidence_gaps"] = [
            "The selected evidence omits the initial-state turn."]
        result = validate_qa_semantic_audit_output(
            value, expected_span_ids=SPAN_IDS)
        self.assertEqual("qa_semantic_audit_contract_accept", result["decision"])
        self.assertEqual("accept", result["answer_decision"])
        self.assertEqual("insufficient", result["evidence_decision"])
        self.assertTrue(result["answer_semantically_verified"])
        self.assertFalse(result["submitted_evidence_sufficient"])

    def test_evidence_sufficiency_and_gap_list_must_agree(self) -> None:
        missing_gap = accepted()
        missing_gap["support_set_sufficient"] = False
        result = validate_qa_semantic_audit_output(
            missing_gap, expected_span_ids=SPAN_IDS)
        self.assertIn("audit_evidence_gaps_missing_when_insufficient",
                      result["errors"])
        unexpected_gap = accepted()
        unexpected_gap["evidence_gaps"] = ["No gap should be present."]
        result = validate_qa_semantic_audit_output(
            unexpected_gap, expected_span_ids=SPAN_IDS)
        self.assertIn("audit_evidence_gaps_present_when_sufficient",
                      result["errors"])

    def test_semantic_reject_is_not_rescued_by_mark_only(self) -> None:
        value = accepted()
        value.update({
            "audit_decision": "reject",
            "answer_entailment": "partially_supported",
            "cross_boundary_relation_entailment": "requires_inference",
            "support_set_sufficient": False,
            "grounded_answer": "",
            "unsupported_claims": [
                "The answer treats an omitted topic as a known earlier state."],
            "evidence_gaps": [
                "No selected turn explicitly states that earlier state."],
            "reason": "The answer relies on absence rather than stated fact.",
        })
        validation = validate_qa_semantic_audit_output(
            value, expected_span_ids=SPAN_IDS)
        self.assertEqual("qa_semantic_audit_contract_accept",
                         validation["decision"])
        admission = classify_semantic_audit_admission(
            qa={}, audit=value, evidence_mismatch_policy="mark_only")
        self.assertFalse(admission["admitted"])
        self.assertFalse(admission["evidence_mismatch_marked"])

    def test_old_insufficient_output_requires_explicit_gap_migration(self) -> None:
        value = accepted()
        value.pop("evidence_gaps")
        value["support_set_sufficient"] = False
        result = validate_qa_semantic_audit_output(
            value, expected_span_ids=SPAN_IDS)
        self.assertEqual("qa_semantic_audit_schema_invalid", result["decision"])
        self.assertIn("audit_field_set_invalid", result["errors"])


if __name__ == "__main__":
    unittest.main()
