"""Regression tests for the pure-offline lifecycle and weight freeze."""

from __future__ import annotations

import json
from functools import lru_cache

from . import EPS, MODELS
from .run import (
    ALLOWED_WEIGHT_FEATURES,
    FORBIDDEN_WEIGHT_FEATURES,
    build_all,
    classify_numeric,
    composition_components,
    evidence_multiplier,
    lifecycle_state,
    load_inputs,
    m0_transition_class,
    replay_components,
    sha256_file,
    targeted_components,
)


def _metric(full: float, no: float, bucket: str, position: str = "none") -> dict:
    total = 10
    return {
        "full": full,
        "no": no,
        "full_correct": round(full * total),
        "full_total": total,
        "no_correct": round(no * total),
        "no_total": total,
        "difficulty_bucket": bucket,
        "context_helpful": full > no,
        "position_mechanism": position,
        "technical_failure_count": 0,
        "technical_exclusion_reason": None,
    }


def _pair(clean: dict, inclusive: dict | None = None) -> dict:
    return {MODELS[0]: clean, MODELS[1]: inclusive or clean}


def _features(**overrides) -> dict:
    value = {
        "clean_full": 0.5,
        "clean_no": 0.0,
        "clean_difficulty_bucket": "hard_candidate",
        "inclusive_full": 0.5,
        "inclusive_no": 0.0,
        "inclusive_difficulty_bucket": "hard_candidate",
        "context_gain": {MODELS[0]: 0.5, MODELS[1]: 0.5},
        "m0_transition": None,
        "training_exposure": "UNKNOWN",
        "evidence_status": "EVIDENCE_CLEAN",
        "position_mechanism": "none",
        "r02_touch": False,
        "cross_round": False,
        "canonical_representative": True,
        "normalized_stem_family_cluster_count": 1,
        "seed_cooldown": False,
        "replay_cooldown": False,
        "replay_history": {"replay_count": 0, "last_replayed_round": None},
    }
    value.update(overrides)
    return value


@lru_cache(maxsize=1)
def _built() -> dict:
    return build_all(load_inputs())


def test_01_active_train_scope_complete() -> None:
    report = _built()["artifacts"]["final_question_bank_lifecycle_freeze_report.json"]
    assert report["active_train_candidates"] == 3062


def test_02_normalized_arm_counts() -> None:
    rows = _built()["artifacts"]["active_train_scope_manifest.jsonl"]
    counts = {}
    for row in rows:
        counts[row["normalized_experiment_arm"]] = counts.get(row["normalized_experiment_arm"], 0) + 1
    assert counts == {"ROUND0_PARENT_PLUS_R01": 2294, "E0Q0_CONTROL_BLIND": 276, "E1Q0_FEEDBACK_BLIND": 254, "E1Q1_FEEDBACK_GUIDED": 238}


def test_03_dev_test_locked_and_zero() -> None:
    rows = _built()["artifacts"]["eval_locked_scope_manifest.jsonl"]
    assert len(rows) == 992 and all(row["sampling_weight"] == 0 for row in rows)


def test_04_untouched_probe_is_eval_locked() -> None:
    hard = _metric(0.5, 0.0, "hard_candidate")
    assert lifecycle_state(_pair(hard), untouched_probe=True) == "EVAL_LOCKED"


def test_05_one_representative_per_cluster() -> None:
    audit = _built()["artifacts"]["representative_weight_audit.json"]
    assert audit["one_representative_per_cluster"] and audit["cluster_count"] == 3048


def test_06_alias_weights_zero() -> None:
    assert _built()["artifacts"]["representative_weight_audit.json"]["alias_sampling_weights_all_zero"]


def test_07_external_observation_not_independent() -> None:
    rows = _built()["artifacts"]["cluster_observation_origin_manifest.jsonl"]
    external = [row for row in rows if row["observation_origin"] == "E1Q0_REPRESENTATIVE"]
    assert external and all(not row["counted_as_independent_observation"] for row in external)


def test_08_difficulty_numeric_contract() -> None:
    assert classify_numeric(_metric(0.5, 0.0, "hard_candidate")) == "hard_candidate"
    assert _built()["artifacts"]["difficulty_numeric_contract_audit.json"]["mismatch_count"] == 0


def test_09_lifecycle_mutually_exclusive_and_exhaustive() -> None:
    rows = _built()["artifacts"]["cluster_lifecycle_manifest.jsonl"]
    assert len(rows) == 3048 and all(isinstance(row["lifecycle_state"], str) for row in rows)


def test_10_position_only_precedes_content_frontier() -> None:
    hard = _metric(0.5, 0.0, "hard_candidate", "position_only")
    assert lifecycle_state(_pair(hard)) == "POSITION_ONLY_DIAGNOSTIC"


def test_11_context_worse_blocked() -> None:
    bad = _metric(0.2, 0.8, "full_context_worse")
    assert lifecycle_state(_pair(bad)) == "CONTEXT_WORSE_DIAGNOSTIC"


def test_12_both_zero_bridge_only() -> None:
    zero = _metric(0.0, 0.0, "full_zero_review")
    assert lifecycle_state(_pair(zero)) == "BRIDGE_REQUIRED_BOTH_ZERO"


def test_13_active_frontier_both() -> None:
    hard = _metric(0.5, 0.0, "hard_candidate")
    assert lifecycle_state(_pair(hard)) == "ACTIVE_FRONTIER_BOTH"


def test_14_mastered_exits_hard_training() -> None:
    report = _built()["artifacts"]["model_specific_training_eligibility.jsonl"]
    assert all(not row["training_eligible_clean"] and not row["training_eligible_inclusive"] for row in report if row["lifecycle_state"] == "EASY_MASTERED")


def test_15_missing_m0_not_fabricated() -> None:
    assert m0_transition_class(None) is None


def test_16_feedback_target_forbidden_from_weight() -> None:
    audit = _built()["artifacts"]["weight_feature_usage_audit.json"]
    assert audit["formula_forbidden_field_read_counts"]["feedback_target_hit"] == 0


def test_17_experiment_arm_forbidden_from_weight() -> None:
    audit = _built()["artifacts"]["weight_feature_usage_audit.json"]
    assert audit["formula_forbidden_field_read_counts"]["experiment_arm"] == 0


def test_18_evidence_risk_multiplier() -> None:
    assert evidence_multiplier("EVIDENCE_RISK") == 0.5
    clean = targeted_components(_features(evidence_status="EVIDENCE_CLEAN"), "ACTIVE_FRONTIER_BOTH")["raw_weight"]
    risk = targeted_components(_features(evidence_status="EVIDENCE_RISK"), "ACTIVE_FRONTIER_BOTH")["raw_weight"]
    assert abs(risk - clean / 2.0) <= EPS


def test_19_family_saturation_multiplier() -> None:
    row = targeted_components(_features(normalized_stem_family_cluster_count=4), "ACTIVE_FRONTIER_BOTH")
    assert row["family_saturation_multiplier"] == 0.5


def test_20_cooldown_multiplier() -> None:
    row = targeted_components(_features(seed_cooldown=True), "ACTIVE_FRONTIER_BOTH")
    assert row["raw_weight"] == 0.0


def test_21_targeted_root_normalization() -> None:
    audit = _built()["artifacts"]["weight_normalization_audit.json"]["questioner_targeted"]
    assert audit["within_root_sums_pass"]


def test_22_replay_model_normalization() -> None:
    audits = _built()["artifacts"]["weight_normalization_audit.json"]["solver_replay"]
    assert all(row["global_sum_pass"] for row in audits.values())


def test_23_composition_contract_avoids_semantic_guessing() -> None:
    contract = _built()["artifacts"]["cooldown_and_family_cap_contract.json"]["composition_pairing_contract"]
    assert contract["semantic_family_claimed"] is False


def test_24_source_files_unchanged_during_build() -> None:
    built = _built()
    assert all(sha256_file(__import__("pathlib").Path(path)) == digest for path, digest in built["source_hashes"].items())


def test_25_online_calls_zero() -> None:
    calls = _built()["artifacts"]["final_question_bank_lifecycle_freeze_report.json"]["online_calls"]
    assert calls and all(value == 0 for value in calls.values())


def test_26_forbidden_and_allowed_feature_sets_disjoint() -> None:
    assert set(ALLOWED_WEIGHT_FEATURES).isdisjoint(FORBIDDEN_WEIGHT_FEATURES)


def test_27_evidence_invalid_zeroes_composition() -> None:
    row = composition_components(_features(evidence_status="EVIDENCE_INVALID"), "ACTIVE_FRONTIER_BOTH", 2)
    assert row["raw_weight"] == 0.0


def test_28_alias_multiplier_zeroes_replay() -> None:
    row = replay_components(_features(canonical_representative=False), 6.0)
    assert row["raw_weight"] == 0.0


def test_29_immutable_build_is_deterministic() -> None:
    first = _built()["artifacts"]["lifecycle_summary.json"]
    second = build_all(load_inputs())["artifacts"]["lifecycle_summary.json"]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_30_e1q1_frozen_counts_match() -> None:
    report = _built()["artifacts"]["final_question_bank_lifecycle_freeze_report.json"]
    assert report["e1q1_source_consistency"]["mismatches"] == {}
