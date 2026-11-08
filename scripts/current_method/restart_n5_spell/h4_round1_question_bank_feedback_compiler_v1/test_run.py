"""Regression checks for the offline feedback compiler artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .compiler import evidence_jaccard, open_unit, weighted_priority

OUT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_question_bank_feedback_compiler_v1_live_13")


def _json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _jsonl(name):
    return [json.loads(x) for x in (OUT / name).read_text(encoding="utf-8").splitlines() if x.strip()]


def test_01_eight_roots_unique():
    assert len(_jsonl("r03_root_scope_manifest.jsonl")) == 8


def test_02_quarantine_disjoint():
    roots = {x["root_id"] for x in _jsonl("r03_root_scope_manifest.jsonl")}
    quarantine = {x["root_id"] for x in _jsonl("quarantine_root_manifest.jsonl")}
    assert roots.isdisjoint(quarantine)


def test_03_96462_present():
    assert "cc-train-episode-96462" in {x["root_id"] for x in _jsonl("r03_root_scope_manifest.jsonl")}


def test_04_control_not_selected():
    assert not any(x["normalized_experiment_arm"] == "E0Q0_CONTROL_BLIND" for x in _jsonl("r03_expansion_seed_selection.jsonl"))


def test_05_fallback_refs_present():
    assert all(x["evidence_refs_present"] or x["branch_compatibility"] in {"CONTROL_BRANCH_EXCLUDED", "OUTSIDE_R03_ROOT_SCOPE"} for x in _jsonl("branch_compatibility_audit.jsonl"))


def test_06_tier_order():
    for root in {x["root_id"] for x in _jsonl("r03_expansion_seed_selection.jsonl")}:
        assert all(x["tier"] == min(y["tier"] for y in _jsonl("r03_expansion_seed_candidate_pool.jsonl") if y["root_id"] == root) for x in _jsonl("r03_expansion_seed_selection.jsonl") if x["root_id"] == root)


def test_07_weighted_sampling_stable():
    args = ("compiler", "R03", "root", "cluster", "op", "seed")
    assert weighted_priority(2.0, *args) == weighted_priority(2.0, *args)
    assert 0 < open_unit(*args) < 1


def test_08_primary_reserve_unique():
    rows = _jsonl("r03_expansion_seed_selection.jsonl") + _jsonl("r03_expansion_seed_reserves.jsonl")
    assert len({x["solver_prompt_fingerprint"] for x in rows}) == len(rows)


def test_09_alias_weight_zero():
    assert all(float(x.get("alias_sampling_weight") or 0) == 0 for name in ("questioner_targeted_pool.jsonl", "questioner_bridge_pool.jsonl", "questioner_composition_parent_pool.jsonl") for x in _jsonl(name))


def test_10_position_only_not_expansion():
    assert not any(x["lifecycle_state"] == "POSITION_ONLY_DIAGNOSTIC" for x in _jsonl("r03_expansion_seed_selection.jsonl"))


def test_11_context_worse_not_expansion():
    assert not any(x["lifecycle_state"] == "CONTEXT_WORSE_DIAGNOSTIC" for x in _jsonl("r03_expansion_seed_selection.jsonl"))


def test_12_both_zero_bridge_only():
    assert all(x["lifecycle_state"] == "BRIDGE_REQUIRED_BOTH_ZERO" for x in _jsonl("questioner_bridge_pool.jsonl"))
    assert not any(x["lifecycle_state"] == "BRIDGE_REQUIRED_BOTH_ZERO" for x in _jsonl("r03_expansion_seed_selection.jsonl"))


def test_13_prompt_view_redacted():
    for row in _jsonl("r03_expansion_feedback_cards_prompt_view.jsonl"):
        assert not ({"gold", "options", "items", "raw_predictions", "solver_output", "semantic_auditor_label"} & set(row))
        assert "canonical_confusion" in row and "operation_instruction" in row


def test_14_confusion_is_canonical():
    assert all(x["canonical_confusion"]["basis"] == "canonical_ids" for x in _jsonl("r03_expansion_feedback_cards_prompt_view.jsonl"))


def test_15_questioner_budget():
    p = _json("questioner_future_budget_plan.json")
    assert (p["TARGETED_EVOLUTION"], p["MULTI_QA_COMPOSITION"], p["FRESH_EXPLORATION"]) == (0.4, 0.2, 0.4)


def test_16_bridge_separate_budget():
    assert _json("questioner_future_budget_plan.json")["BRIDGE_GENERATION"]["deducted_from_main_budget"] is False


def test_17_pair_jaccard():
    assert all(0 <= float(x["evidence_jaccard"]) <= 0.5 for x in _jsonl("questioner_composition_pair_feasibility.jsonl"))


def test_18_fresh_is_signature_only():
    assert all(set(x) <= {"root_id", "normalized_stem_signature", "prompt_cluster_count", "candidate_count", "family_saturation_multiplier", "fresh_avoidance_only"} for x in _jsonl("fresh_exploration_negative_signature_view.jsonl"))


def test_19_replay_models_independent():
    assert all(x["model_name"] == "M_clean_policy" for x in _jsonl("solver_replay_ranked_pool_clean.jsonl"))
    assert all(x["model_name"] == "M_inclusive_policy" for x in _jsonl("solver_replay_ranked_pool_inclusive.jsonl"))


def test_20_lineage_prefix():
    assert all(x["source_prefix"] == "Parent+R01+R02-feedback" for x in _jsonl("r03_cumulative_lineage_plan.jsonl"))


def test_21_add_forbidden():
    assert all(not x["parent_add_allowed"] and not x["r01_add_allowed"] and not x["r02_add_allowed"] for x in _jsonl("r03_cumulative_lineage_plan.jsonl"))


def test_22_token_not_solver_limit():
    assert _json("r03_expansion_token_preflight.json")["context_ceiling"] == 40960


def test_23_forbidden_features_unused():
    assert _json("feedback_feature_usage_audit.json")["forbidden_features_observed"] == []


def test_24_sources_unchanged():
    assert _json("source_sha_audit.json")["source_manifests_modified"] == 0


def test_25_online_zero():
    assert all(int(v) == 0 for v in _json("global_question_bank_scope_audit.json")["online_calls"].values())
