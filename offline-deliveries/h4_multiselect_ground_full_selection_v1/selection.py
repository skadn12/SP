"""Trial selection policy; no Solver, training, or legacy admission dependencies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction


VERSION = "multi_select_ground_full_selection_v1"
GROUND_VERSION = "exact_k3_adaptive_k5_v1"
VARIANTS = tuple(f"perm_{i:02d}" for i in range(6))
FULL_ROLLOUTS = 3


def fraction(value: Fraction) -> dict:
    return {"fraction": str(value), "value": float(value)}


def ground_decision(initial: list[bool], extra: list[bool] | None = None) -> dict:
    if len(initial) != 3 or any(type(value) is not bool for value in initial):
        return {"ground_pass": None, "status": "EVALUATION_INCOMPLETE"}
    correct = sum(initial)
    if correct <= 1:
        passed, total = True, 3
    elif correct == 3:
        passed, total = False, 3
    elif extra is None:
        return {"ground_pass": None, "status": "NEEDS_EXTENSION",
                "exact_correct": correct, "evaluated_count": 3}
    elif len(extra) != 2 or any(type(value) is not bool for value in extra):
        return {"ground_pass": None, "status": "EVALUATION_INCOMPLETE"}
    else:
        correct += sum(extra)
        passed, total = correct < 3, 5
    return {"ground_pass": passed, "status": "PASS" if passed else "FILTERED",
            "exact_correct": correct, "evaluated_count": total}


def select_question(
    ground: dict,
    full_scores: Mapping[str, Sequence[Fraction | None]],
    *,
    data_errors: Sequence[str] = (),
) -> dict:
    errors = list(data_errors)
    if set(full_scores) != set(VARIANTS):
        errors.append("FULL_VARIANT_SET_MISMATCH")
    for variant in VARIANTS:
        scores = full_scores.get(variant, ())
        if len(scores) != FULL_ROLLOUTS:
            errors.append(f"FULL_ROLLOUT_COUNT:{variant}")
        elif any(not isinstance(score, Fraction) or not -1 <= score <= 1 for score in scores):
            errors.append(f"FULL_SCORE_INVALID:{variant}")
    result = {
        "policy_version": VERSION, "ground_pass": ground.get("ground_pass"),
        "selected_by_policy": None, "selection_reason": None,
        "full_partial_mean": None, "permutation_means": {},
        "permutation_score_min": None, "permutation_score_max": None,
        "permutation_score_range": None, "data_errors": sorted(set(errors)),
    }
    if errors:
        result["selection_reason"] = "EVALUATION_INCOMPLETE_OR_INVALID"
        return result
    means = {variant: sum(full_scores[variant], Fraction()) / FULL_ROLLOUTS for variant in VARIANTS}
    mean = sum(means.values(), Fraction()) / len(VARIANTS)
    result.update({
        "full_partial_mean": fraction(mean),
        "permutation_means": {variant: fraction(score) for variant, score in means.items()},
        "permutation_score_min": fraction(min(means.values())),
        "permutation_score_max": fraction(max(means.values())),
        "permutation_score_range": fraction(max(means.values()) - min(means.values())),
    })
    if ground.get("ground_pass") is None:
        result["selection_reason"] = "GROUND_EVALUATION_INCOMPLETE"
    elif ground["ground_pass"] is False:
        result.update(selected_by_policy=False, selection_reason="GROUND_FILTERED")
    elif 0 < mean < 1:
        result.update(selected_by_policy=True, selection_reason="GROUND_PASS_AND_FULL_PARTIAL_IN_RANGE")
    else:
        result.update(selected_by_policy=False, selection_reason=(
            "FULL_PARTIAL_NONPOSITIVE" if mean <= 0 else "FULL_PARTIAL_PERFECT_NO_NEW_HARD_ITEM"
        ))
    return result
