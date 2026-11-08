"""Full-only selection policy. No ground, context gain, or stability veto exists."""
from __future__ import annotations

from fractions import Fraction
from collections import Counter
from typing import Any, Mapping, Sequence

VERSION = "r02_groundless_full_only_random_position_v2"


def audit_complete_call_set(manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                            candidate_id: str | None = None) -> list[str]:
    """Require exactly one valid response for every frozen candidate/variant/slot.

    The manifest is the planned call set, not a count inferred from responses.
    With a candidate filter, a record carrying its request ID but a wrong
    candidate ID remains in scope so identity corruption cannot be hidden.
    """
    requests = manifest.get("requests")
    if not isinstance(requests, list) or not requests:
        return ["PLANNED_CALL_SET_MISSING"]
    if any(not isinstance(row, Mapping) for row in requests):
        return ["PLANNED_CALL_SCHEMA_INVALID"]
    planned = [row for row in requests if candidate_id is None or row.get("candidate_id") == candidate_id]
    if not planned:
        return ["PLANNED_CANDIDATE_CALLS_MISSING"]
    errors: list[str] = []
    expected: dict[str, Mapping[str, Any]] = {}
    positions: set[tuple[str, str, int]] = set()
    for row in planned:
        rid, cid = row.get("request_id"), row.get("candidate_id")
        variant = row.get("variant")
        vid = variant.get("variant_id") if isinstance(variant, Mapping) else None
        slot = row.get("slot")
        if (not isinstance(rid, str) or not rid or not isinstance(cid, str) or not cid
                or not isinstance(vid, str) or not vid or type(slot) is not int or slot < 0
                or row.get("mode") != "full_context"):
            errors.append("PLANNED_CALL_SCHEMA_INVALID")
            continue
        if rid in expected:
            errors.append("DUPLICATE_PLANNED_REQUEST:" + rid)
        expected[rid] = row
        position = (cid, vid, slot)
        if position in positions:
            errors.append("DUPLICATE_PLANNED_SLOT:" + ":".join(map(str, position)))
        positions.add(position)
    observed: Counter[str] = Counter()
    for row in records:
        if not isinstance(row, Mapping):
            errors.append("RESPONSE_SCHEMA_INVALID")
            continue
        rid = row.get("request_id")
        if (candidate_id is not None and row.get("candidate_id") != candidate_id
                and (not isinstance(rid, str) or rid not in expected)):
            continue
        if not isinstance(rid, str) or rid not in expected:
            errors.append("UNPLANNED_RESPONSE:" + str(rid))
            continue
        observed[rid] += 1
        plan = expected[rid]
        bindings = {"candidate_id": plan["candidate_id"], "variant_id": plan["variant"]["variant_id"],
                    "slot": plan["slot"]}
        bindings.update({key: plan[key] for key in ("root_id", "question_type", "seed") if key in plan})
        for key, value in bindings.items():
            if row.get(key) != value or (key == "slot" and type(row.get(key)) is not int):
                errors.append("RESPONSE_PLAN_MISMATCH:" + rid + ":" + key)
        if row.get("evaluation_valid") is not True:
            errors.append("INVALID_RESPONSE:" + rid)
    for rid in expected:
        if observed[rid] == 0:
            errors.append("MISSING_RESPONSE:" + rid)
        elif observed[rid] != 1:
            errors.append("DUPLICATE_RESPONSE:" + rid)
    return sorted(set(errors))


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
