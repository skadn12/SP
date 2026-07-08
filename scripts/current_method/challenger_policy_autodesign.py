from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BOUNDARY_BUCKETS = {"1/5", "2/5", "3/5", "4/5"}
EASY_BUCKET = "5/5"
ZERO_BUCKET = "0/5"


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_mapping(values: dict[str, float], *, digits: int = 6) -> dict[str, float]:
    rounded = {k: round(float(v), digits) for k, v in values.items() if float(v) > 0}
    total = sum(rounded.values())
    if not rounded or total <= 0:
        return rounded
    # Keep sums stable after rounding without hiding the non-50/50 decision.
    keys = list(rounded)
    drift = round(1.0 - sum(rounded.values()), digits)
    rounded[keys[0]] = round(rounded[keys[0]] + drift, digits)
    return rounded


def _normalize(scores: dict[str, float], *, floor: float = 0.0, cap: float = 1.0) -> dict[str, float]:
    cleaned = {k: max(floor, min(cap, float(v))) for k, v in scores.items()}
    total = sum(cleaned.values())
    if total <= 0:
        return {k: round(1.0 / len(cleaned), 6) for k in cleaned} if cleaned else {}
    return _round_mapping({k: v / total for k, v in cleaned.items()})


def _bucket_stats(counter: Counter[str]) -> dict[str, Any]:
    n = sum(counter.values())
    boundary = sum(counter.get(bucket, 0) for bucket in BOUNDARY_BUCKETS)
    easy = counter.get(EASY_BUCKET, 0)
    zero = counter.get(ZERO_BUCKET, 0)
    return {
        "n": n,
        "bucket_counts": dict(counter),
        "boundary_count": boundary,
        "easy_5of5_count": easy,
        "zero_0of5_count": zero,
        "boundary_rate": round(boundary / n, 6) if n else 0.0,
        "easy_rate": round(easy / n, 6) if n else 0.0,
        "zero_rate": round(zero / n, 6) if n else 0.0,
    }


def _learnability_score(stats: dict[str, Any]) -> float:
    """Score a cell for next-round allocation.

    Boundary rows are useful, 5/5 rows are too easy, and 0/5 rows are often too
    hard or brittle. The weights intentionally favor stable boundary yield over
    raw difficulty.
    """

    boundary = _safe_float(stats.get("boundary_rate"))
    easy = _safe_float(stats.get("easy_rate"))
    zero = _safe_float(stats.get("zero_rate"))
    n = int(stats.get("n") or 0)
    support_bonus = min(n, 8) / 80.0
    return max(0.01, 0.10 + 0.75 * boundary + 0.25 * (1.0 - easy) - 0.65 * zero + support_bonus)


def _reward_v3_score(stats: dict[str, Any]) -> float:
    """Opt-in boundary-aware cell reward.

    Unlike the legacy score, this treats boundary density as the primary target
    and explicitly penalizes item-surface shortcuts found in Stage18/19.
    """

    boundary = _safe_float(stats.get("boundary_rate"))
    easy = _safe_float(stats.get("easy_rate"))
    zero = _safe_float(stats.get("zero_rate"))
    bad_or_brittle_zero = _safe_float(stats.get("bad_or_brittle_zero_rate"), 0.0)
    clean_zero = _safe_float(stats.get("clean_zero_rate"), max(0.0, zero - bad_or_brittle_zero))
    obvious = _safe_float(stats.get("obvious_pattern_rate"))
    duplicate = _safe_float(stats.get("duplicate_option_rate"))
    gold_only_active = _safe_float(stats.get("gold_only_active_pattern_rate"))
    identical_gold_pair = _safe_float(stats.get("identical_gold_pair_rate"))
    qa_dependency_gate = _safe_float(stats.get("qa_dependency_gate_pass_rate"), 0.0)
    n = int(stats.get("n") or 0)
    support_bonus = min(n, 8) / 80.0
    return max(
        0.01,
        1.00 * boundary
        - 0.80 * easy
        - 0.40 * clean_zero
        - 1.00 * bad_or_brittle_zero
        - 1.20 * obvious
        - 1.20 * duplicate
        - 1.00 * gold_only_active
        - 1.00 * identical_gold_pair
        + 0.20 * qa_dependency_gate
        + support_bonus,
    )


def _autodesign_score(stats: dict[str, Any], *, reward_v3: bool = False) -> float:
    return _reward_v3_score(stats) if reward_v3 else _learnability_score(stats)


def _load_eval_buckets(clean_samples_path: Path) -> dict[str, dict[str, Any]]:
    eval_path = clean_samples_path.parent / "solver_k5_eval.jsonl"
    if not eval_path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for item in read_jsonl(eval_path):
        sample_id = str(item.get("sample_id") or "")
        solver_eval = item.get("solver_eval") if isinstance(item.get("solver_eval"), dict) else {}
        if sample_id:
            rows[sample_id] = {
                "bucket": solver_eval.get("bucket"),
                "exact_correct_count": solver_eval.get("exact_correct_count"),
                "soft_accuracy": solver_eval.get("soft_accuracy"),
            }
    return rows


def _sample_qtype(sample: dict[str, Any]) -> str:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    return str(sample.get("question_type") or qa.get("question_type") or "unknown")


def _sample_skill(sample: dict[str, Any]) -> str:
    return str(sample.get("attack_skill") or (sample.get("metadata") or {}).get("attack_skill") or "unknown")


def _sample_template(sample: dict[str, Any]) -> str:
    return str(
        sample.get("hard_single_choice_template")
        or (sample.get("metadata") or {}).get("hard_single_choice_template")
        or "none"
    )


def _sample_parent_id(sample: dict[str, Any]) -> str:
    seed_source = sample.get("seed_source") if isinstance(sample.get("seed_source"), dict) else {}
    return str(sample.get("source_id") or (sample.get("metadata") or {}).get("source_seed_id") or seed_source.get("sample_id") or "unknown")


def _collect_bucket_tables(samples: list[dict[str, Any]], eval_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_qtype: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_skill: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_template: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_parent: defaultdict[str, Counter[str]] = defaultdict(Counter)
    missing_eval = 0
    for sample in samples:
        sample_id = str(sample.get("sample_id") or "")
        bucket = eval_rows.get(sample_id, {}).get("bucket")
        if not bucket:
            missing_eval += 1
            continue
        bucket = str(bucket)
        by_qtype[_sample_qtype(sample)][bucket] += 1
        by_skill[_sample_skill(sample)][bucket] += 1
        by_template[_sample_template(sample)][bucket] += 1
        by_parent[_sample_parent_id(sample)][bucket] += 1
    return {
        "qtype": {k: _bucket_stats(v) for k, v in by_qtype.items()},
        "skill": {k: _bucket_stats(v) for k, v in by_skill.items()},
        "template": {k: _bucket_stats(v) for k, v in by_template.items()},
        "parent": {k: _bucket_stats(v) for k, v in by_parent.items()},
        "missing_eval_count": missing_eval,
    }


def _design_qtype_allocation(qtype_stats: dict[str, dict[str, Any]], *, reward_v3: bool = False) -> dict[str, float]:
    allowed = {k: v for k, v in qtype_stats.items() if k in {"single_choice", "multi_select"}}
    scores = {qtype: _autodesign_score(stats, reward_v3=reward_v3) for qtype, stats in allowed.items()}
    shares = _normalize(scores, floor=0.05)
    # Keep currently unsafe or off-scope qtypes explicit at zero.
    for qtype in ("ordering", "state_reconstruction", "evidence_selection"):
        shares[qtype] = 0.0
    return _round_mapping({k: v for k, v in shares.items() if v > 0 or k in {"ordering", "state_reconstruction", "evidence_selection"}})


def _design_skill_allocation(skill_stats: dict[str, dict[str, Any]], *, reward_v3: bool = False) -> dict[str, float]:
    allowed = {
        k: v
        for k, v in skill_stats.items()
        if k in {"conditional_plan", "delayed_activation", "permission_visibility_scope"}
    }
    scores = {skill: _autodesign_score(stats, reward_v3=reward_v3) for skill, stats in allowed.items()}
    return _normalize(scores, floor=0.05)


def _design_template_allocation(template_stats: dict[str, dict[str, Any]], *, reward_v3: bool = False) -> dict[str, float]:
    allowed = {
        k: v
        for k, v in template_stats.items()
        if k in {"minimal_difference", "latest_valid_state", "guaranteed_vs_possible", "override_sensitive"}
    }
    scores = {template: _autodesign_score(stats, reward_v3=reward_v3) for template, stats in allowed.items()}
    if "guaranteed_vs_possible" in scores:
        scores["guaranteed_vs_possible"] *= 1.15
    shares = _normalize(scores, floor=0.03)
    shares["sequence_as_single_choice"] = 0.0
    return _round_mapping(shares)


def _design_parent_preferences(parent_stats: dict[str, dict[str, Any]], *, reward_v3: bool = False) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    for parent_id, stats in parent_stats.items():
        score = _autodesign_score(stats, reward_v3=reward_v3)
        zero = _safe_float(stats.get("zero_rate"))
        easy = _safe_float(stats.get("easy_rate"))
        boundary = _safe_float(stats.get("boundary_rate"))
        if zero >= 0.35:
            tier = "downweight_zero_risk"
        elif boundary >= 0.45 and zero == 0:
            tier = "prefer_boundary_clean"
        elif easy >= 0.75:
            tier = "harden_if_used_too_easy"
        else:
            tier = "neutral_probe"
        ranked.append(
            {
                "parent_id": parent_id,
                "tier": tier,
                "score": round(score, 6),
                "stats": stats,
            }
        )
    ranked.sort(key=lambda item: (item["score"], item["stats"].get("boundary_rate", 0)), reverse=True)
    return {
        "ranked_parents": ranked,
        "selection_rule": (
            "Prefer boundary-clean parent_v2 rows; keep a small exploration floor for neutral rows; "
            "downweight parents whose descendants produced high 0/5 or all-easy outcomes."
        ),
    }


def _cell_weights_from_skill_allocation(skill_allocation: dict[str, float]) -> dict[str, float]:
    allowed_cells = {
        "single_choice|conditional_plan": skill_allocation.get("conditional_plan", 0.0),
        "single_choice|delayed_activation": skill_allocation.get("delayed_activation", 0.0),
        "multi_select|conditional_plan": skill_allocation.get("conditional_plan", 0.0),
        "multi_select|permission_visibility_scope": skill_allocation.get("permission_visibility_scope", 0.0),
    }
    return {cell: round(max(0.0, float(weight)), 6) for cell, weight in allowed_cells.items() if weight > 0}


def _compare_expected_actual(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(expected) | set(actual))
    rows: dict[str, Any] = {}
    max_abs_drift = 0.0
    for key in keys:
        exp = _safe_float(expected.get(key))
        act = _safe_float(actual.get(key))
        drift = round(act - exp, 6)
        max_abs_drift = max(max_abs_drift, abs(drift))
        rows[key] = {
            "expected": round(exp, 6),
            "actual": round(act, 6),
            "drift_actual_minus_expected": drift,
            "status": "over_sampled" if drift > 0.05 else "under_sampled" if drift < -0.05 else "close",
        }
    return {
        "rows": rows,
        "max_abs_drift": round(max_abs_drift, 6),
        "close": max_abs_drift <= 0.05,
    }


def _stage_solver_profile(summary: dict[str, Any], stage_key: str) -> dict[str, Any]:
    solver_eval = summary.get(stage_key) if isinstance(summary.get(stage_key), dict) else {}
    buckets = solver_eval.get("soft_buckets") if isinstance(solver_eval.get("soft_buckets"), dict) else {}
    total = sum(int(v) for v in buckets.values()) or int(summary.get("clean_count") or 0) or 1
    boundary = sum(int(buckets.get(bucket) or 0) for bucket in BOUNDARY_BUCKETS)
    easy = int(buckets.get(EASY_BUCKET) or 0)
    zero = int(buckets.get(ZERO_BUCKET) or 0)
    return {
        "k1": solver_eval.get("k1"),
        "k5_soft": solver_eval.get("k5_soft"),
        "soft_buckets": buckets,
        "easy_5of5_count": easy,
        "easy_5of5_rate": round(easy / total, 6),
        "zero_0of5_count": zero,
        "zero_0of5_rate": round(zero / total, 6),
        "boundary_1to4_count": boundary,
        "boundary_1to4_rate": round(boundary / total, 6),
    }


def _stage14_solver_profile(summary_14: dict[str, Any]) -> dict[str, Any]:
    solver_eval = summary_14.get("solver_eval") if isinstance(summary_14.get("solver_eval"), dict) else {}
    buckets = solver_eval.get("soft_buckets") if isinstance(solver_eval.get("soft_buckets"), dict) else {}
    total = sum(int(v) for v in buckets.values()) or int(summary_14.get("clean_count") or 0) or 1
    boundary = sum(int(buckets.get(bucket) or 0) for bucket in BOUNDARY_BUCKETS)
    easy = int(buckets.get(EASY_BUCKET) or 0)
    zero = int(buckets.get(ZERO_BUCKET) or 0)
    return {
        "k1": solver_eval.get("k1"),
        "k5_soft": solver_eval.get("k5_soft"),
        "soft_buckets": buckets,
        "easy_5of5_count": easy,
        "easy_5of5_rate": round(easy / total, 6),
        "zero_0of5_count": zero,
        "zero_0of5_rate": round(zero / total, 6),
        "boundary_1to4_count": boundary,
        "boundary_1to4_rate": round(boundary / total, 6),
    }


def _apply_stage16_controller_patch(
    *,
    previous_policy: dict[str, Any],
    summary_16: dict[str, Any],
    summary_14: dict[str, Any],
    report_16_text: str,
) -> dict[str, Any]:
    """Return a v2 policy that corrects the stage16 easy-policy failure.

    This path is intentionally opt-in. The v1 autodesign behavior remains the
    default unless callers provide stage16 diagnostics and request this patch.
    """

    policy = copy.deepcopy(previous_policy)
    next_policy = policy.setdefault("next_policy", {})
    difficulty_plan = next_policy.setdefault("difficulty_plan", {})

    expected_qtype = summary_16.get("policy_qtype_allocation") or next_policy.get("question_type_allocation") or {}
    actual_qtype = summary_16.get("actual_qtype_share") or {}
    expected_skill = summary_16.get("policy_skill_allocation") or next_policy.get("attack_skill_allocation") or {}
    actual_skill = summary_16.get("actual_skill_share") or {}
    expected_template = summary_16.get("policy_template_allocation") or next_policy.get("hard_single_choice_template_allocation") or {}
    actual_template = summary_16.get("actual_template_share") or {}

    allocation_diagnostic = {
        "question_type": _compare_expected_actual(expected_qtype, actual_qtype),
        "attack_skill": _compare_expected_actual(expected_skill, actual_skill),
        "hard_single_choice_template": _compare_expected_actual(expected_template, actual_template),
    }

    stage14_profile = _stage14_solver_profile(summary_14)
    stage16_profile = _stage_solver_profile(summary_16, "solver_eval_stage16")
    comparison = summary_16.get("solver_eval_comparison_with_14") if isinstance(summary_16.get("solver_eval_comparison_with_14"), dict) else {}
    delta = comparison.get("delta") if isinstance(comparison.get("delta"), dict) else {}
    failure_reasons = summary_16.get("failure_reasons") if isinstance(summary_16.get("failure_reasons"), list) else []

    # Stage16 showed qtype allocation was close, so keep qtype nearly stable but
    # move a small amount toward multi_select because permission_visibility_scope
    # can only be expressed there in the current generator cell map.
    qtype_v2 = {
        "single_choice": 0.48,
        "multi_select": 0.52,
        "ordering": 0.0,
        "state_reconstruction": 0.0,
        "evidence_selection": 0.0,
    }

    # Conditional_plan was over-sampled by +0.18 and the run became easier. Push
    # the mass toward the two cells with better prior boundary behavior while
    # keeping a small conditional_plan floor for continuity.
    skill_v2 = {
        "conditional_plan": 0.12,
        "delayed_activation": 0.44,
        "permission_visibility_scope": 0.44,
    }

    # From v1/Stage14, guaranteed_vs_possible was the only hard-single template
    # with strong boundary yield and zero 0/5. Minimal_difference and
    # override_sensitive were mostly/all 5/5, so they receive sharp penalties.
    template_v2 = {
        "guaranteed_vs_possible": 0.72,
        "latest_valid_state": 0.20,
        "minimal_difference": 0.06,
        "override_sensitive": 0.02,
        "sequence_as_single_choice": 0.0,
    }

    cell_weights_v2 = {
        "single_choice|conditional_plan": 0.04,
        "single_choice|delayed_activation": 0.44,
        "multi_select|conditional_plan": 0.08,
        "multi_select|permission_visibility_scope": 0.44,
    }

    cell_level_adjustments = {
        "single_choice|conditional_plan": {
            "previous_weight": (difficulty_plan.get("cell_weights") or {}).get("single_choice|conditional_plan"),
            "new_weight": cell_weights_v2["single_choice|conditional_plan"],
            "reward_penalty": "strong_penalty",
            "reason": (
                "conditional_plan was over-sampled in Stage16 (+0.182 actual-minus-expected) while "
                "5/5 increased and boundary did not improve; keep only a small continuity floor."
            ),
        },
        "multi_select|conditional_plan": {
            "previous_weight": (difficulty_plan.get("cell_weights") or {}).get("multi_select|conditional_plan"),
            "new_weight": cell_weights_v2["multi_select|conditional_plan"],
            "reward_penalty": "controlled_penalty",
            "reason": (
                "Do not remove conditional_plan entirely, but prevent the controller from reducing 0/5 by "
                "collapsing into easier conditional_plan rows."
            ),
        },
        "single_choice|delayed_activation": {
            "previous_weight": (difficulty_plan.get("cell_weights") or {}).get("single_choice|delayed_activation"),
            "new_weight": cell_weights_v2["single_choice|delayed_activation"],
            "reward_penalty": "boundary_reward",
            "reason": (
                "delayed_activation was slightly under-sampled and remains a useful cross-state-change skill; "
                "raise it while keeping sequence_as_single_choice frozen."
            ),
        },
        "multi_select|permission_visibility_scope": {
            "previous_weight": (difficulty_plan.get("cell_weights") or {}).get("multi_select|permission_visibility_scope"),
            "new_weight": cell_weights_v2["multi_select|permission_visibility_scope"],
            "reward_penalty": "strong_boundary_reward",
            "reason": (
                "permission_visibility_scope was under-sampled by -0.141 and is the main route to harder "
                "multi_select boundary without re-enabling ordering."
            ),
        },
    }

    parent_distribution = summary_16.get("parent_distribution") if isinstance(summary_16.get("parent_distribution"), dict) else {}
    parent_constraints: dict[str, Any] = {}
    total_parent_samples = sum(int(v) for v in parent_distribution.values()) or 1
    prior_ranked = (
        next_policy.get("parent_selection_preference", {}).get("ranked_parents", [])
        if isinstance(next_policy.get("parent_selection_preference"), dict)
        else []
    )
    prior_by_parent = {str(item.get("parent_id")): item for item in prior_ranked if isinstance(item, dict)}
    for parent_id, count in sorted(parent_distribution.items(), key=lambda item: (-int(item[1]), item[0])):
        actual_share = int(count) / total_parent_samples
        prior_stats = prior_by_parent.get(parent_id, {}).get("stats", {}) if isinstance(prior_by_parent.get(parent_id, {}), dict) else {}
        prior_easy = _safe_float(prior_stats.get("easy_rate"))
        prior_boundary = _safe_float(prior_stats.get("boundary_rate"))
        prior_zero = _safe_float(prior_stats.get("zero_rate"))
        if actual_share >= 0.25:
            action = "cap_concentration"
            max_share = 0.12
            reason = "Stage16 concentrated too much mass on this parent without improving boundary density."
        elif prior_easy >= 0.75 and prior_boundary == 0:
            action = "all_easy_parent_penalty"
            max_share = 0.04
            reason = "Prior descendants were all-easy; use only as rare hardened exploration."
        elif prior_zero >= 0.35:
            action = "zero_risk_controlled_exploration"
            max_share = 0.06
            reason = "Prior descendants carried high 0/5 risk; do not chase difficulty into brittleness."
        else:
            action = "diversify_boundary_probe"
            max_share = 0.16
            reason = "Keep diverse boundary probes instead of overusing a single promising parent."
        parent_constraints[parent_id] = {
            "stage16_actual_share": round(actual_share, 6),
            "prior_boundary_rate": prior_boundary,
            "prior_easy_rate": prior_easy,
            "prior_zero_rate": prior_zero,
            "action": action,
            "max_share": max_share,
            "reason": reason,
        }

    objective_v2 = {
        "primary": "maximize_1of5_to_4of5_boundary_density",
        "secondary": [
            "reduce_5of5_easy_share",
            "keep_0of5_controlled_without_policy_collapse_to_easy",
            "preserve_chained_qa_dependency_targeting",
        ],
        "not_primary": "minimize_0of5_alone",
        "target_bucket_movement_next_probe": {
            "boundary_1to4": "increase above Stage14/16 count=10 at n=30",
            "easy_5of5": "decrease below Stage14 count=15 and Stage16 count=16",
            "zero_0of5": "keep <= Stage14 count=5 unless validator labels are clean and boundary also increases",
        },
    }

    next_policy["question_type_allocation"] = qtype_v2
    next_policy["attack_skill_allocation"] = skill_v2
    next_policy["hard_single_choice_template_allocation"] = template_v2
    next_policy["boundary_objective_v2"] = objective_v2
    next_policy["cell_level_reward_penalty"] = cell_level_adjustments
    next_policy["parent_selection_preference_v2"] = {
        "selection_rule": (
            "Diversify parent_v2 selection, cap Stage16-concentrated parents, sharply penalize all-easy "
            "parent/cell combinations, and keep zero-heavy parents as controlled exploration only."
        ),
        "parent_constraints": parent_constraints,
    }
    next_policy["generator_enforcement"] = {
        "expected_vs_actual_feedback_enabled": True,
        "skill_allocation_enforcement": "strict",
        "max_skill_abs_drift_target": 0.05,
        "if_conditional_plan_actual_minus_expected_gt_0.05": "apply negative feedback multiplier before next allocation",
        "if_permission_visibility_scope_actual_minus_expected_lt_minus_0.05": "increase requested mass and monitor actual accepted share",
    }
    next_policy["dependency_targeting_strength"] = {
        **(next_policy.get("dependency_targeting_strength") or {}),
        "chained_qa_dependency_targeting": True,
        "min_qa_used_dependency_count": 8,
        "target_qa_used_dependency_count": "8-12",
        "min_evidence_turn_distance": 1000,
    }
    next_policy["distractor_strategy"] = {
        **(next_policy.get("distractor_strategy") or {}),
        "banned_false_reason_types": sorted(set((next_policy.get("distractor_strategy") or {}).get("banned_false_reason_types", [])) | {"weak_obvious_status"}),
        "weak_obvious_status_distractor_count_target": 0,
    }
    next_policy["retention_guard"] = {
        **(next_policy.get("retention_guard") or {}),
        "sequence_as_single_choice": {
            "allocation": 0.0,
            "reason": "Still frozen by retention guard; do not use it to create harder delayed_activation rows.",
            "known_risk_sample": "native_current_method_graph_dialogue_0375_d51d9075cd",
        },
        "ordering": {
            "allocation": 0.0,
            "reason": "Still frozen; do not use ordering as the easy way to increase 0/5 difficulty.",
        },
    }
    difficulty_plan["question_type_shares"] = {
        "single_choice": qtype_v2["single_choice"],
        "multi_select": qtype_v2["multi_select"],
        "ordering": 0.0,
        "state_reconstruction": 0.0,
    }
    difficulty_plan["allowed_question_types"] = ["single_choice", "multi_select"]
    difficulty_plan["single_choice_subtype_shares"] = {
        "hard_single_choice": 1.0,
        "ordinary_single_choice": 0.0,
    }
    difficulty_plan["hard_single_choice_template_shares"] = template_v2
    difficulty_plan["cell_weights"] = cell_weights_v2
    difficulty_plan["cell_hardness_mix"] = {
        "single_choice|conditional_plan": {"h0": 0.05, "h1": 0.45, "h2": 0.50},
        "single_choice|delayed_activation": {"h0": 0.05, "h1": 0.45, "h2": 0.50},
        "multi_select|conditional_plan": {"h0": 0.10, "h1": 0.45, "h2": 0.45},
        "multi_select|permission_visibility_scope": {"h0": 0.05, "h1": 0.50, "h2": 0.45},
    }
    difficulty_plan["qtype_curriculum"] = {
        "stage": "17_challenger_policy_autodesign_controller_patch",
        "autodesigned_from": "16_autodesigned_chained_generation_probe",
        "manual_policy_correction": None,
        "controller_patch_v2": True,
    }

    policy["policy_id"] = "challenger_autodesigned_chained_policy_v2"
    policy["created_by"] = "current_method_challenger_policy_autodesign_controller_patch_v2"
    policy["target_use"] = "next_autodesigned_chained_generation_or_mining_design_only"
    source = policy.setdefault("source_diagnostics", {})
    source["stage16_summary"] = summary_16.get("run_root")
    source["stage16_clean_count"] = summary_16.get("clean_count")
    source["stage16_report_loaded"] = bool(report_16_text.strip())
    source["stage16_failure_reasons"] = failure_reasons
    source["stage16_solver_profile"] = stage16_profile
    source["stage14_solver_profile"] = stage14_profile
    source["stage16_vs_stage14_delta"] = delta
    source["stage16_expected_vs_actual_allocation"] = allocation_diagnostic

    policy["controller_patch_v2"] = {
        "enabled": True,
        "why_stage16_became_easier": [
            "Stage16 kept qtype close to policy but accepted skill mix drifted: conditional_plan was over-sampled by +0.182.",
            "The controller treated lower 0/5 as useful, but the bucket movement came with 5/5 increasing 15->16 and k1 increasing 0.733333->0.766667.",
            "Parent sampling was concentrated, especially one parent around 30 percent, so the policy did not sufficiently diversify boundary-producing cells.",
            "The previous objective under-penalized all-easy templates/cells such as minimal_difference and override_sensitive.",
        ],
        "expected_vs_actual_allocation_diagnostic": allocation_diagnostic,
        "cell_level_reward_penalty": cell_level_adjustments,
        "parent_concentration_and_all_easy_penalty": parent_constraints,
        "objective": objective_v2,
        "quality_and_dependency_targeting_preserved": {
            "chained_qa_dependency_targeting": True,
            "weak_obvious_status_disabled": True,
            "sequence_as_single_choice_allocation": 0.0,
            "ordering_allocation": 0.0,
            "default_logic_unchanged": True,
            "opt_in_required": True,
        },
    }

    policy["rationale"] = {
        **(policy.get("rationale") or {}),
        "v2_response_to_stage16_easier_probe": (
            "The objective now prioritizes boundary density over lowering 0/5. It sharply lowers "
            "conditional_plan and all-easy template/cell mass, increases permission_visibility_scope, "
            "and caps over-concentrated/all-easy parents."
        ),
        "response_to_all_easy_parent_cell": (
            "All-easy parent/cell combinations receive low max share and score multipliers; they may only be used "
            "as rare hardened exploration, never as the dominant allocation."
        ),
        "response_to_skill_allocation_drift": (
            "The v2 generator-facing cell weights encode the desired skill mix directly and add expected-vs-actual "
            "feedback diagnostics for the next probe."
        ),
        "response_to_boundary_as_primary_objective": (
            "The policy target is now explicit: increase 1/5-4/5 density first; only reduce 0/5 when 5/5 does not rise."
        ),
    }
    pass_conditions = policy.setdefault("pass_conditions", {})
    pass_conditions["controller_patch_v2_generated_from_stage16"] = True
    pass_conditions["default_logic_unchanged"] = True
    pass_conditions["must_be_enabled_explicitly"] = True
    pass_conditions["does_not_start_mining"] = True
    pass_conditions["does_not_start_train_mix"] = True
    pass_conditions["does_not_start_training"] = True
    return policy


def _extract_stage13_markers(report_text: str) -> dict[str, Any]:
    markers: dict[str, Any] = {}
    patterns = {
        "generated_valid_clean": r"generated/valid/clean:\s*([0-9]+/[0-9]+/[0-9]+)",
        "qa_used_event_count_min_mean_max": r"qa_used_event_count min/mean/max:\s*([^\n]+)",
        "qa_used_dependency_count_min_mean_max": r"qa_used_dependency_count min/mean/max:\s*([^\n]+)",
        "evidence_turn_distance_min_mean_max": r"evidence_turn_distance min/mean/max:\s*([^\n]+)",
        "weak_obvious_status_distractor_count_total": r"weak_obvious_status_distractor_count_total:\s*([0-9]+)",
        "smoke_pass": r"smoke_pass:\s*(True|False|true|false)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, report_text)
        if match:
            markers[key] = match.group(1).strip()
    return markers


def build_autodesigned_policy(
    *,
    summary_14: dict[str, Any],
    samples_14: list[dict[str, Any]],
    eval_rows: dict[str, dict[str, Any]],
    audit_12: dict[str, Any],
    report_14_text: str,
    smoke_13_text: str,
    reward_v3: bool = False,
) -> dict[str, Any]:
    tables = _collect_bucket_tables(samples_14, eval_rows)
    qtype_allocation = _design_qtype_allocation(tables["qtype"], reward_v3=reward_v3)
    skill_allocation = _design_skill_allocation(tables["skill"], reward_v3=reward_v3)
    template_allocation = _design_template_allocation(tables["template"], reward_v3=reward_v3)
    parent_preferences = _design_parent_preferences(tables["parent"], reward_v3=reward_v3)
    cell_weights = _cell_weights_from_skill_allocation(skill_allocation)
    stage13_markers = _extract_stage13_markers(smoke_13_text)

    soft_buckets = summary_14.get("solver_eval", {}).get("soft_buckets") if isinstance(summary_14.get("solver_eval"), dict) else {}
    total = int(summary_14.get("clean_count") or 0) or max(1, sum(int(v) for v in soft_buckets.values()))
    easy_count = int((soft_buckets or {}).get(EASY_BUCKET) or 0)
    zero_count = int((soft_buckets or {}).get(ZERO_BUCKET) or 0)
    boundary_count = sum(int((soft_buckets or {}).get(bucket) or 0) for bucket in BOUNDARY_BUCKETS)
    easy_rate = easy_count / total
    zero_rate = zero_count / total
    boundary_rate = boundary_count / total

    dependency_mean = _safe_float((summary_14.get("qa_used_dependency_count") or {}).get("mean"))
    evidence_mean = _safe_float((summary_14.get("evidence_turn_distance") or {}).get("mean"))
    weak_count = int(summary_14.get("weak_obvious_status_distractor_count_total") or 0)

    main_causes = audit_12.get("main_causes") if isinstance(audit_12.get("main_causes"), list) else []
    policy = {
        "policy_id": "challenger_autodesigned_chained_policy_v1",
        "created_by": "current_method_challenger_policy_autodesign_v1",
        "autodesign_opt_in_required": True,
        "reward_v3_opt_in": bool(reward_v3),
        "method": "current_method",
        "native_challenger_mode": "graph_conditioned_api_dialogue",
        "target_use": "next_calibrated_chained_mining_or_generation_design_only",
        "source_diagnostics": {
            "stage14_summary": summary_14.get("run_root"),
            "stage14_clean_count": summary_14.get("clean_count"),
            "stage14_qtype_distribution": summary_14.get("qtype_distribution"),
            "stage14_solver_eval": summary_14.get("solver_eval"),
            "stage14_qa_used_dependency_count": summary_14.get("qa_used_dependency_count"),
            "stage14_evidence_turn_distance": summary_14.get("evidence_turn_distance"),
            "stage14_false_reason_types_distribution": summary_14.get("false_reason_types_distribution"),
            "stage12_main_causes": main_causes,
            "stage13_smoke_report_loaded": bool(smoke_13_text.strip()),
            "stage13_dependency_targeting_markers": stage13_markers,
            "stage14_report_loaded": bool(report_14_text.strip()),
            "adjacent_solver_k5_eval_loaded": bool(eval_rows),
            "missing_eval_count": tables["missing_eval_count"],
            "reward_function": "reward_v3" if reward_v3 else "legacy_learnability_score",
        },
        "observed_bucket_profile": {
            "total": total,
            "easy_5of5_count": easy_count,
            "easy_5of5_rate": round(easy_rate, 6),
            "zero_0of5_count": zero_count,
            "zero_0of5_rate": round(zero_rate, 6),
            "boundary_1to4_count": boundary_count,
            "boundary_1to4_rate": round(boundary_rate, 6),
            "by_question_type": tables["qtype"],
            "by_attack_skill": tables["skill"],
            "by_hard_single_choice_template": tables["template"],
        },
        "next_policy": {
            "question_type_allocation": qtype_allocation,
            "attack_skill_allocation": skill_allocation,
            "hard_single_choice_template_allocation": template_allocation,
            "dependency_targeting_strength": {
                "chained_qa_dependency_targeting": True,
                "min_qa_used_event_count": 2,
                "target_qa_used_event_count": "2-3",
                "min_qa_used_dependency_count": 8 if dependency_mean >= 8 else 6,
                "target_qa_used_dependency_count": "8-12",
                "require_parent_context_used": True,
                "require_child_append_context_used": True,
                "min_evidence_turn_distance": 1000,
                "target_evidence_turn_distance": ">=1000 and far above Stage12 local append mean 4.4",
            },
            "distractor_strategy": {
                "allowed_false_reason_types": [
                    "cross_context_dependency_mismatch",
                    "outdated",
                    "later_state_override",
                    "invalidated_parent_dependency",
                    "same_entity_wrong_later_state",
                ],
                "preferred_false_reason_types": [
                    "cross_context_dependency_mismatch",
                    "later_state_override",
                    "invalidated_parent_dependency",
                ],
                "banned_false_reason_types": ["weak_obvious_status"],
                "weak_obvious_status_distractor_count_target": 0,
                "rule": (
                    "Distractors should be near the gold dependency chain but invalidated by later state, "
                    "override, or cross-context mismatch; do not use local obvious status negatives."
                ),
            },
            "parent_selection_preference": parent_preferences,
            "boundary_too_easy_too_hard_handling": {
                "too_easy_5of5_response": {
                    "observed_rate": round(easy_rate, 6),
                    "action": (
                        "Do not simply increase volume. Keep qtypes with boundary signal, but harden easy cells by "
                        "requiring >=8 dependencies, cross-parent-child evidence, and stronger invalidation distractors."
                    ),
                },
                "too_hard_0of5_response": {
                    "observed_rate": round(zero_rate, 6),
                    "action": (
                        "Prevent 0/5 instability by capping multi_select and downweighting parent/skill cells whose "
                        "zero_rate exceeds 0.35; freeze ordering until a separate retention-safe probe passes."
                    ),
                },
                "boundary_response": {
                    "observed_rate": round(boundary_rate, 6),
                    "action": (
                        "Allocate more mass to cells with high 1/5-4/5 yield and low zero rate, especially "
                        "delayed_activation, permission_visibility_scope, and guaranteed_vs_possible."
                    ),
                },
            },
            "retention_guard": {
                "sequence_as_single_choice": {
                    "allocation": 0.0,
                    "reason": "Stage2/Stage3 retention guard still has a known delayed_activation sequence regression.",
                    "known_risk_sample": "native_current_method_graph_dialogue_0375_d51d9075cd",
                },
                "ordering": {
                    "allocation": 0.0,
                    "reason": "Ordering remained high-zero in prior stages and was not part of the calibrated stable probe.",
                },
                "required_before_training": [
                    "sequence_as_single_choice / delayed_activation clean retention guard",
                    "stage1 clean replay regression guard",
                    "stage2 accepted validation guard",
                ],
            },
            "hard_gates": {
                "no_evidence_window_prompt": True,
                "no_seed_parent_reset": True,
                "full_dialogue_prompt": True,
                "qa_last": True,
                "graph_derived_gold_evidence": True,
                "overflow_count": 0,
                "truncated_count": 0,
                "anchor_leakage_count": 0,
                "solver_visible_meta_leakage_count": 0,
                "contradiction_count": 0,
                "accepted_parser_failure_count": 0,
                "accepted_empty_prediction_count": 0,
            },
            "difficulty_plan": {
                "question_type_shares": {
                    "single_choice": qtype_allocation.get("single_choice", 0.0),
                    "multi_select": qtype_allocation.get("multi_select", 0.0),
                    "ordering": 0.0,
                    "state_reconstruction": 0.0,
                },
                "allowed_question_types": ["single_choice", "multi_select"],
                "single_choice_subtype_shares": {
                    "hard_single_choice": 1.0,
                    "ordinary_single_choice": 0.0,
                },
                "hard_single_choice_template_shares": template_allocation,
                "cell_weights": cell_weights,
                "cell_hardness_mix": {
                    "single_choice|conditional_plan": {"h0": 0.15, "h1": 0.55, "h2": 0.30},
                    "single_choice|delayed_activation": {"h0": 0.10, "h1": 0.55, "h2": 0.35},
                    "multi_select|conditional_plan": {"h0": 0.20, "h1": 0.60, "h2": 0.20},
                    "multi_select|permission_visibility_scope": {"h0": 0.15, "h1": 0.65, "h2": 0.20},
                },
                "qtype_curriculum": {
                    "stage": "15_challenger_policy_autodesign_patch",
                    "autodesigned_from": "14_calibrated_chained_generation_probe",
                    "manual_policy_correction": None,
                },
            },
        },
        "rationale": {
            "why_not_fixed_50_50": (
                "Stage14 used 15/15, but qtype bucket profiles differ: multi_select has higher 0/5 risk while "
                "single_choice has more 5/5 easy rows. Allocation is therefore computed from boundary/easy/zero rates."
            ),
            "response_to_5of5_too_many": (
                "The policy hardens dependencies and distractors instead of increasing easy-volume cells."
            ),
            "response_to_0of5": (
                "The policy caps high-zero qtypes/parents and freezes ordering; 0/5 is treated as too-hard/brittle risk."
            ),
            "response_to_boundary": (
                "The policy favors cells with 1/5-4/5 yield and low zero rate."
            ),
            "response_to_dependency": (
                "The policy preserves the Stage13/14 fix: parent+child evidence, >=8 dependencies when feasible, and long evidence distance."
            ),
            "response_to_distractors": (
                "The policy bans weak_obvious_status and prefers cross-context invalidation/override distractors."
            ),
            "response_to_retention": (
                "The policy keeps sequence_as_single_choice and ordering frozen until explicit retention probes pass."
            ),
        },
        "pass_conditions": {
            "based_on_12_13_14_diagnostics": True,
            "does_not_start_mining": True,
            "does_not_start_train_mix": True,
            "does_not_start_training": True,
            "default_logic_unchanged": True,
            "must_be_enabled_explicitly": True,
        },
    }
    return policy


def _format_stats_table(stats: dict[str, dict[str, Any]]) -> str:
    lines = ["| cell | n | boundary | zero | easy | buckets |", "|---|---:|---:|---:|---:|---|"]
    for key, item in sorted(stats.items()):
        lines.append(
            f"| `{key}` | {item.get('n', 0)} | {item.get('boundary_count', 0)} "
            f"({item.get('boundary_rate', 0)}) | {item.get('zero_0of5_count', 0)} "
            f"({item.get('zero_rate', 0)}) | {item.get('easy_5of5_count', 0)} "
            f"({item.get('easy_rate', 0)}) | `{item.get('bucket_counts', {})}` |"
        )
    return "\n".join(lines)


def write_report(path: str | Path, policy: dict[str, Any]) -> None:
    report = Path(path)
    report.parent.mkdir(parents=True, exist_ok=True)
    observed = policy["observed_bucket_profile"]
    next_policy = policy["next_policy"]
    rationale = policy["rationale"]
    md = f"""# Challenger Policy Autodesign Report

## Scope

- policy_id: `{policy['policy_id']}`
- created_by: `{policy['created_by']}`
- autodesign_opt_in_required: {policy['autodesign_opt_in_required']}
- target_use: `{policy['target_use']}`
- evidence_window_prompt: forbidden
- seed_parent_reset: forbidden
- no mining, no train_mix, no training was run.

## Source Diagnostics

- stage14 clean count: {policy['source_diagnostics'].get('stage14_clean_count')}
- stage14 solver eval: `{policy['source_diagnostics'].get('stage14_solver_eval')}`
- stage14 dependency diagnostics: `{policy['source_diagnostics'].get('stage14_qa_used_dependency_count')}`
- stage14 evidence distance: `{policy['source_diagnostics'].get('stage14_evidence_turn_distance')}`
- stage12 main causes: `{policy['source_diagnostics'].get('stage12_main_causes')}`

## Observed Bucket Profile

- total: {observed['total']}
- 5/5 easy: {observed['easy_5of5_count']} ({observed['easy_5of5_rate']})
- 0/5 too-hard/brittle: {observed['zero_0of5_count']} ({observed['zero_0of5_rate']})
- 1/5-4/5 boundary: {observed['boundary_1to4_count']} ({observed['boundary_1to4_rate']})

### By Question Type

{_format_stats_table(observed['by_question_type'])}

### By Skill

{_format_stats_table(observed['by_attack_skill'])}

### By Hard Single-Choice Template

{_format_stats_table(observed['by_hard_single_choice_template'])}

## Autodesigned Next Policy

- qtype allocation: `{next_policy['question_type_allocation']}`
- skill allocation: `{next_policy['attack_skill_allocation']}`
- hard_single_choice template allocation: `{next_policy['hard_single_choice_template_allocation']}`
- generator cell_weights: `{next_policy['difficulty_plan']['cell_weights']}`
- dependency targeting: `{next_policy['dependency_targeting_strength']}`
- distractor strategy: `{next_policy['distractor_strategy']}`
- retention guard: `{next_policy['retention_guard']}`

## Required Responses

1. 5/5 too many: {rationale['response_to_5of5_too_many']}
2. 0/5 control: {rationale['response_to_0of5']}
3. Boundary increase: {rationale['response_to_boundary']}
4. QA dependency retention: {rationale['response_to_dependency']}
5. Distractor strengthening: {rationale['response_to_distractors']}
6. Retention guard: {rationale['response_to_retention']}

## Parent Selection

{next_policy['parent_selection_preference']['selection_rule']}

Top parent recommendations:

| parent_id | tier | score | buckets |
|---|---|---:|---|
"""
    for item in next_policy["parent_selection_preference"]["ranked_parents"][:10]:
        md += f"| `{item['parent_id']}` | `{item['tier']}` | {item['score']} | `{item['stats'].get('bucket_counts', {})}` |\n"
    md += f"""

## Hard Gates

`{next_policy['hard_gates']}`

## Decision

- policy_generated: true
- policy_is_diagnostic_autodesign_only: true
- default_logic_unchanged: true
- must_be_enabled_explicitly: true
- recommendation: use this policy only after human review to design calibrated chained mining; do not auto-run mining/training.

## Output

- policy json: `{Path(path).parents[1] / 'challenger_autodesigned_policy.json'}`
"""
    report.write_text(md, encoding="utf-8")


def write_controller_patch_report(path: str | Path, policy: dict[str, Any]) -> None:
    report = Path(path)
    report.parent.mkdir(parents=True, exist_ok=True)
    patch = policy.get("controller_patch_v2") if isinstance(policy.get("controller_patch_v2"), dict) else {}
    next_policy = policy.get("next_policy") if isinstance(policy.get("next_policy"), dict) else {}
    source = policy.get("source_diagnostics") if isinstance(policy.get("source_diagnostics"), dict) else {}
    objective = patch.get("objective") if isinstance(patch.get("objective"), dict) else {}
    allocation = patch.get("expected_vs_actual_allocation_diagnostic") if isinstance(patch.get("expected_vs_actual_allocation_diagnostic"), dict) else {}
    stage16_profile = source.get("stage16_solver_profile") if isinstance(source.get("stage16_solver_profile"), dict) else {}
    stage14_profile = source.get("stage14_solver_profile") if isinstance(source.get("stage14_solver_profile"), dict) else {}
    stage_delta = source.get("stage16_vs_stage14_delta") if isinstance(source.get("stage16_vs_stage14_delta"), dict) else {}

    def allocation_table(name: str, payload: dict[str, Any]) -> str:
        rows = payload.get("rows") if isinstance(payload.get("rows"), dict) else {}
        lines = [f"### {name}", "", "| item | expected | actual | drift | status |", "|---|---:|---:|---:|---|"]
        for key, item in sorted(rows.items()):
            lines.append(
                f"| `{key}` | {item.get('expected')} | {item.get('actual')} | "
                f"{item.get('drift_actual_minus_expected')} | `{item.get('status')}` |"
            )
        lines.append(f"\nmax_abs_drift={payload.get('max_abs_drift')}, close={payload.get('close')}\n")
        return "\n".join(lines)

    md = f"""# Challenger Policy Autodesign Controller Patch Report

## Scope

- policy_id: `{policy.get('policy_id')}`
- created_by: `{policy.get('created_by')}`
- output policy: `{report.parents[1] / 'challenger_autodesigned_policy_v2.json'}`
- source v1 policy: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/15_challenger_policy_autodesign_patch/challenger_autodesigned_policy.json`
- source stage16 summary: `{source.get('stage16_summary')}`
- no generation, no mining, no train_mix, no training, no accepted solver was run.
- default logic unchanged: {policy.get('pass_conditions', {}).get('default_logic_unchanged')}
- opt-in required: {policy.get('autodesign_opt_in_required')}

## Why Stage16 Became Easier

Stage16 preserved the chained QA dependency targeting and quality gates, but the solver profile moved easier:

| metric | Stage14 | Stage16 | delta |
|---|---:|---:|---:|
| k1 | {stage14_profile.get('k1')} | {stage16_profile.get('k1')} | {stage_delta.get('k1')} |
| k5_soft | {stage14_profile.get('k5_soft')} | {stage16_profile.get('k5_soft')} | {stage_delta.get('k5_soft')} |
| 5/5 easy | {stage14_profile.get('easy_5of5_count')} | {stage16_profile.get('easy_5of5_count')} | {stage_delta.get('easy_5of5')} |
| 1/5-4/5 boundary | {stage14_profile.get('boundary_1to4_count')} | {stage16_profile.get('boundary_1to4_count')} | {stage_delta.get('boundary_1to4')} |
| 0/5 | {stage14_profile.get('zero_0of5_count')} | {stage16_profile.get('zero_0of5_count')} | {stage_delta.get('zero_0of5')} |

Main causes:

"""
    for item in patch.get("why_stage16_became_easier", []):
        md += f"- {item}\n"

    md += f"""
## Expected vs Actual Allocation

{allocation_table('Question Type', allocation.get('question_type', {}))}

{allocation_table('Attack Skill', allocation.get('attack_skill', {}))}

{allocation_table('Hard Single-Choice Template', allocation.get('hard_single_choice_template', {}))}

## V2 Objective

- primary: `{objective.get('primary')}`
- secondary: `{objective.get('secondary')}`
- not_primary: `{objective.get('not_primary')}`
- target_bucket_movement_next_probe: `{objective.get('target_bucket_movement_next_probe')}`

This is the important correction: reducing 0/5 is no longer enough. The controller must increase 1/5-4/5 boundary density and reduce 5/5 easy rows at the same time.

## V2 Policy

- qtype allocation: `{next_policy.get('question_type_allocation')}`
- skill allocation: `{next_policy.get('attack_skill_allocation')}`
- hard_single_choice template allocation: `{next_policy.get('hard_single_choice_template_allocation')}`
- generator cell_weights: `{next_policy.get('difficulty_plan', {}).get('cell_weights')}`
- cell_hardness_mix: `{next_policy.get('difficulty_plan', {}).get('cell_hardness_mix')}`

## Cell-Level Reward / Penalty

| cell | previous | new | adjustment | reason |
|---|---:|---:|---|---|
"""
    for cell, item in (patch.get("cell_level_reward_penalty") or {}).items():
        md += (
            f"| `{cell}` | {item.get('previous_weight')} | {item.get('new_weight')} | "
            f"`{item.get('reward_penalty')}` | {item.get('reason')} |\n"
        )

    md += """
## Parent / All-Easy Penalties

| parent_id | stage16 share | prior boundary | prior easy | prior zero | action | max_share | reason |
|---|---:|---:|---:|---:|---|---:|---|
"""
    for parent_id, item in (patch.get("parent_concentration_and_all_easy_penalty") or {}).items():
        md += (
            f"| `{parent_id}` | {item.get('stage16_actual_share')} | {item.get('prior_boundary_rate')} | "
            f"{item.get('prior_easy_rate')} | {item.get('prior_zero_rate')} | `{item.get('action')}` | "
            f"{item.get('max_share')} | {item.get('reason')} |\n"
        )

    md += f"""
## Preserved Hard Gates

- chained_qa_dependency_targeting: {next_policy.get('dependency_targeting_strength', {}).get('chained_qa_dependency_targeting')}
- min_qa_used_dependency_count: {next_policy.get('dependency_targeting_strength', {}).get('min_qa_used_dependency_count')}
- min_evidence_turn_distance: {next_policy.get('dependency_targeting_strength', {}).get('min_evidence_turn_distance')}
- weak_obvious_status banned: {'weak_obvious_status' in next_policy.get('distractor_strategy', {}).get('banned_false_reason_types', [])}
- sequence_as_single_choice allocation: {next_policy.get('retention_guard', {}).get('sequence_as_single_choice', {}).get('allocation')}
- ordering allocation: {next_policy.get('retention_guard', {}).get('ordering', {}).get('allocation')}
- no evidence-window prompt: {next_policy.get('hard_gates', {}).get('no_evidence_window_prompt')}
- no seed-parent reset: {next_policy.get('hard_gates', {}).get('no_seed_parent_reset')}

## Pass Check

- explains Stage16 easier outcome: true
- lowers all-easy conditional_plan oversampling: true
- raises permission_visibility_scope / boundary-likely cells: true
- boundary density is primary objective: true
- 0/5 handling is controlled, not the main objective: true
- default logic unchanged: {policy.get('pass_conditions', {}).get('default_logic_unchanged')}
- opt-in required: {policy.get('pass_conditions', {}).get('must_be_enabled_explicitly')}

## Decision

Controller patch v2 is generated only. It does not run data generation or training. The next safe step would be a small opt-in probe using this v2 policy, not mining or train_mix by default.
"""
    report.write_text(md, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autodesign next chained challenger policy from diagnostics.")
    parser.add_argument("--stage14-summary", required=True)
    parser.add_argument("--stage14-clean-samples", required=True)
    parser.add_argument("--stage14-report", required=True)
    parser.add_argument("--stage12-audit", required=True)
    parser.add_argument("--stage13-smoke-report", required=True)
    parser.add_argument("--previous-policy", default=None)
    parser.add_argument("--stage16-summary", default=None)
    parser.add_argument("--stage16-report", default=None)
    parser.add_argument("--controller-patch-v2", action="store_true")
    parser.add_argument(
        "--reward-v3",
        action="store_true",
        help="Opt-in boundary-aware reward. Default keeps legacy learnability score unchanged.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report-md", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clean_path = Path(args.stage14_clean_samples)
    summary_14 = read_json(args.stage14_summary)
    samples_14 = read_jsonl(clean_path)
    eval_rows = _load_eval_buckets(clean_path)
    audit_12 = read_json(args.stage12_audit)
    report_14 = read_text(args.stage14_report)
    smoke_13 = read_text(args.stage13_smoke_report)
    policy = build_autodesigned_policy(
        summary_14=summary_14,
        samples_14=samples_14,
        eval_rows=eval_rows,
        audit_12=audit_12,
        report_14_text=report_14,
        smoke_13_text=smoke_13,
        reward_v3=bool(args.reward_v3),
    )
    if args.controller_patch_v2:
        if not args.previous_policy or not args.stage16_summary or not args.stage16_report:
            raise SystemExit("--controller-patch-v2 requires --previous-policy, --stage16-summary, and --stage16-report")
        previous_policy = read_json(args.previous_policy)
        summary_16 = read_json(args.stage16_summary)
        report_16 = read_text(args.stage16_report)
        policy = _apply_stage16_controller_patch(
            previous_policy=previous_policy,
            summary_16=summary_16,
            summary_14=summary_14,
            report_16_text=report_16,
        )
    write_json(args.output_json, policy)
    if args.controller_patch_v2:
        write_controller_patch_report(args.report_md, policy)
    else:
        write_report(args.report_md, policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
