"""Admission tiers for Solver-filtered MCQs.

Training difficulty is governed by the SPELL grounding and K=5 full-context
frontier gates. Evidence-reference defects remain explicit metadata, while an
executed audit that disproves the answer still quarantines the item.
"""

from __future__ import annotations

from typing import Any, Mapping


QA_ADMISSION_TIER_VERSION = "memorax_qa_admission_tier_v2_spell_frontier"


def classify_qa_admission_tier(item: Mapping[str, Any]) -> dict[str, Any]:
    solver_admitted = bool(item.get("admitted"))
    audit_executed = bool(item.get("semantic_evidence_audit_executed"))
    mismatch = bool(item.get("evidence_mismatch"))
    answer_verified = bool(item.get("answer_semantically_verified"))
    evidence_sufficient = bool(item.get("submitted_evidence_sufficient"))

    if not solver_admitted:
        tier = "rejected_by_solver_filter"
        semantic_status = "not_applicable"
        training_ready = False
    elif audit_executed and not answer_verified:
        tier = "semantic_quarantine"
        semantic_status = "answer_audit_not_accepted"
        training_ready = False
    elif mismatch or (audit_executed and not evidence_sufficient):
        tier = "training_candidate_evidence_mismatch_marked"
        semantic_status = "evidence_mismatch_marked"
        training_ready = True
    elif audit_executed and answer_verified and evidence_sufficient:
        tier = "training_candidate"
        semantic_status = "semantically_verified"
        training_ready = True
    else:
        tier = "training_candidate_unaudited_evidence"
        semantic_status = "evidence_not_audited"
        training_ready = True

    return {
        "qa_admission_tier_version": QA_ADMISSION_TIER_VERSION,
        "admission_tier": tier,
        "observation_ready": solver_admitted,
        "training_ready": training_ready,
        "evidence_semantic_status": semantic_status,
    }
