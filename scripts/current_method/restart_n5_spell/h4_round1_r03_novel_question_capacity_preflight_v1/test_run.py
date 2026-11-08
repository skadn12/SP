"""Direct regression tests for the offline R03 capacity preflight."""

from __future__ import annotations

from functools import lru_cache

from .capacity import build_all, load_inputs


@lru_cache(maxsize=1)
def _built():
    return build_all(load_inputs())


def test_01_status_and_next_stage():
    report = _built()["artifacts"]["final_r03_novel_question_capacity_preflight.json"]
    assert report["status"] == "R03_NOVEL_QUESTION_CAPACITY_PREFLIGHT_PASS"
    assert report["next_status"] == "READY_FOR_TARGETED_EVOLUTION_MULTI_QA_AND_FRESH_EXPLORATION"


def test_02_root_disposition_closure():
    report = _built()["artifacts"]["final_r03_novel_question_capacity_preflight.json"]
    assert report["success_root_count"] == 7
    assert report["quarantine_root_count"] == 1
    assert report["memory_writes"] == 0


def test_03_structural_counts():
    report = _built()["artifacts"]["final_r03_novel_question_capacity_preflight.json"]
    assert report["r03_turn_count"] == 296
    assert report["new_r03_evidence_ref_count"] == 296
    assert report["independent_evidence_bundle_count"] == 21
    assert report["joint_cross_round_bundle_candidate_count"] == 63


def test_04_estimates_are_explicitly_bounded():
    report = _built()["artifacts"]["final_r03_novel_question_capacity_preflight.json"]
    assert report["estimated_new_prompt_stem_lower_bound"] == 21
    assert report["estimated_new_prompt_stem_upper_bound"] == 63


def test_05_semantic_limits_and_calls():
    report = _built()["artifacts"]["final_r03_novel_question_capacity_preflight.json"]
    assert report["semantic_state_event_verified_count"] == 0
    assert report["semantic_family_claimed"] is False
    assert not report["semantic_auditor_called"]
    assert all(int(value or 0) == 0 for value in report["forbidden_call_counts"].values())


def test_06_all_structural_refs_valid():
    report = _built()["artifacts"]["final_r03_novel_question_capacity_preflight.json"]
    assert report["structurally_valid_r03_evidence_ref_count"] == report["new_r03_evidence_ref_count"]
    assert report["gate_results"]["all_success_scene_refs_structurally_valid"]


def test_07_quarantine_not_promoted():
    rows = _built()["artifacts"]["r03_root_capacity_summary.json"]["roots"]
    quarantined = [row for row in rows if row["disposition"] == "R03_LOCAL_QUARANTINE"]
    assert len(quarantined) == 1 and quarantined[0]["scene_count"] == 0


def test_08_mechanical_repeat_does_not_claim_semantics():
    audit = _built()["artifacts"]["r03_old_family_repeat_audit.json"]
    assert audit["semantic_family_claimed"] is False
    assert audit["semantic_auditor_called"] is False

