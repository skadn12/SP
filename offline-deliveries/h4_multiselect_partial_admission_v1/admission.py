"""Offline-only multiselect candidate diagnosis; no production admission hook."""

from __future__ import annotations

EPS = 1e-9
VERSION = "multi_select_partial_admission_v1"


def numeric_reason(metrics: dict) -> str:
    """Classify observed paired means without treating them as training approval."""
    full_exact = metrics["full_exact_rate"]
    no_exact = metrics["no_exact_rate"]
    full_partial = metrics["full_partial_mean"]
    no_partial = metrics["no_partial_mean"]
    if None in (full_exact, no_exact, full_partial, no_partial):
        return "PAIRING_INCOMPLETE"
    if full_exact > 1 - EPS:
        return "OBSERVED_FULL_RESOLVED"
    exact_gain = full_exact - no_exact
    partial_gain = full_partial - no_partial
    if exact_gain < -EPS or partial_gain < -EPS:
        return "CONTEXT_WORSE" if exact_gain <= EPS and partial_gain <= EPS else "METRIC_DIRECTION_CONFLICT"
    if exact_gain <= EPS and partial_gain <= EPS:
        return "NO_OBSERVED_CONTEXT_GAIN"
    if full_exact <= EPS and no_exact <= EPS:
        return "FULL_ZERO_PARTIAL_GAIN"
    return "EXACT_GAIN" if exact_gain > EPS else "PARTIAL_GAIN"


def candidate_decision(
    metrics: dict,
    *,
    quality_check_status: str,
    permutation_check_status: str,
    pairing_check_status: str,
    split_check_status: str,
    confirmed_quarantine: bool = False,
) -> dict:
    reason = numeric_reason(metrics)
    observed_numeric_pattern = reason in {"EXACT_GAIN", "FULL_ZERO_PARTIAL_GAIN", "PARTIAL_GAIN"}
    numeric_candidate = observed_numeric_pattern and pairing_check_status == "PASS"
    blockers = []
    if confirmed_quarantine:
        blockers.append("CONFIRMED_QUARANTINE")
    for name, value in (
        ("split", split_check_status),
        ("quality", quality_check_status),
        ("pairing", pairing_check_status),
        ("permutation", permutation_check_status),
    ):
        if value != "PASS":
            blockers.append(f"{name.upper()}:{value}")
    return {
        "policy_version": VERSION,
        "observed_numeric_pattern": observed_numeric_pattern,
        "numeric_candidate": numeric_candidate,
        "new_policy_reason": reason,
        "new_policy_candidate_status": (
            "TRAINING_CANDIDATE" if numeric_candidate and not blockers
            else "DIAGNOSTIC_PENDING" if observed_numeric_pattern
            else "NOT_NUMERIC_CANDIDATE"
        ),
        "blockers": blockers,
        "quality_check_status": quality_check_status,
        "permutation_check_status": permutation_check_status,
        "pairing_check_status": pairing_check_status,
        "split_check_status": split_check_status,
    }
