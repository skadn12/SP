"""Full-only selection policy. No ground, context gain, or stability veto exists."""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping, Sequence

VERSION = "groundless_full_only_random_position_v1"


def select(full_scores: Mapping[str, Sequence[Fraction | None]], expected_variants: Sequence[str],
           *, exact_rate: Fraction | None = None, no_context: Any = None,
           data_errors: Sequence[str] = ()) -> dict[str, Any]:
    errors = list(data_errors)
    if set(full_scores) != set(expected_variants):
        errors.append("FULL_VARIANT_SET_MISMATCH")
    means: dict[str, Fraction] = {}
    for variant in expected_variants:
        scores = full_scores.get(variant, ())
        if not scores or any(not isinstance(score, Fraction) for score in scores):
            errors.append("FULL_EVALUATION_INCOMPLETE:" + variant)
        else:
            means[variant] = sum(scores, Fraction()) / len(scores)
    result = {"policy_version": VERSION, "ground_used": False, "selected_by_policy": None,
              "selection_reason": None, "full_partial_mean": None,
              "permutation_means": {key: str(value) for key, value in means.items()},
              "permutation_score_min": None, "permutation_score_max": None,
              "permutation_score_range": None, "full_exact_rate": str(exact_rate) if exact_rate is not None else None,
              "context_gain": None, "data_errors": sorted(set(errors))}
    if errors:
        result["selection_reason"] = "EVALUATION_INCOMPLETE_OR_INVALID"
        return result
    mean = sum(means.values(), Fraction()) / len(means)
    result["full_partial_mean"] = str(mean)
    result["permutation_score_min"] = str(min(means.values()))
    result["permutation_score_max"] = str(max(means.values()))
    result["permutation_score_range"] = str(max(means.values()) - min(means.values()))
    result["selected_by_policy"] = 0 < mean < 1
    result["selection_reason"] = "FULL_PARTIAL_IN_OPEN_INTERVAL" if 0 < mean < 1 else (
        "FULL_PARTIAL_NONPOSITIVE" if mean <= 0 else "FULL_PARTIAL_PERFECT_NO_NEW_HARD_ITEM")
    return result

