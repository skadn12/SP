"""K=3 aggregation and diagnostic classification."""

from __future__ import annotations

from typing import Any, Mapping


MAIN_CATEGORIES = (
    "robustly_solved",
    "position_invariant_failure",
    "mixed_position_behavior",
)

_PARSE_FAILURES = {"parse_failure", "parse_error", "protocol_failure"}
_RUNTIME_FAILURES = {
    "runtime_error", "runtime_failure", "technical_failure", "timeout",
    "oom", "out_of_memory", "guided_decoding_error", "backend_failure",
    "request_failure", "transport_failure",
}


def _failure_type(row: Mapping[str, Any]) -> str:
    return str(row.get("failure_type") or "").strip().casefold()


def _technical_kind(row: Mapping[str, Any]) -> str | None:
    """Return the technical-failure class, independent of ``scored`` flags."""

    failure_type = _failure_type(row)
    if failure_type in _PARSE_FAILURES:
        return "parse_failure"
    if failure_type in _RUNTIME_FAILURES:
        return "runtime_failure"
    parse_status = str(row.get("parse_status") or "").strip().casefold()
    if parse_status in {"failure", "failed", "error"}:
        return "parse_failure"
    if row.get("runtime_error") or row.get("transport_error"):
        return "runtime_failure"
    return None


def aggregate_position(rollouts: list[Mapping[str, Any]]) -> dict[str, Any]:
    attempted = len(rollouts)
    technical = [_technical_kind(row) for row in rollouts]
    scored = [
        row for row, kind in zip(rollouts, technical)
        if kind is None and bool(row.get("scored"))
    ]
    correct = sum(bool(row.get("correct")) for row in scored)
    parse_failures = sum(kind == "parse_failure" for kind in technical)
    runtime_failures = sum(kind == "runtime_failure" for kind in technical)
    reasoning_errors = sum(
        kind is None and row.get("failure_type") == "reasoning_error"
        for row, kind in zip(rollouts, technical)
    )
    unscored = sum(kind is None and not bool(row.get("scored")) for row, kind in zip(rollouts, technical))
    return {
        "requested_total": 3,
        "attempted_total": attempted,
        "scored_total": len(scored),
        "correct": correct,
        "accuracy": correct / len(scored) if scored else None,
        "parse_failure_count": parse_failures,
        "runtime_failure_count": runtime_failures,
        "technical_failure_count": parse_failures + runtime_failures,
        "reasoning_error_count": reasoning_errors,
        "unscored_count": unscored,
        "complete": attempted == 3 and len(scored) == 3,
        "nonunanimous": len(scored) == 3 and correct not in (0, 3),
        "rollouts": [dict(row) for row in rollouts],
    }


def classify_candidate(positions: list[Mapping[str, Any]]) -> dict[str, Any]:
    complete = bool(positions) and all(bool(row.get("complete")) for row in positions)
    accuracies = [row.get("accuracy") for row in positions if row.get("accuracy") is not None]
    if not complete:
        category = "diagnostic_incomplete"
    elif accuracies and min(accuracies) == 1.0:
        category = "robustly_solved"
    elif accuracies and max(accuracies) == 0.0:
        category = "position_invariant_failure"
    else:
        category = "mixed_position_behavior"
    position_range = max(accuracies) - min(accuracies) if complete and accuracies else None
    strong = bool(
        complete and accuracies and max(accuracies) == 1.0 and min(accuracies) <= (1 / 3)
    )
    weak = bool(complete and position_range is not None and 0 < position_range <= (1 / 3))
    sampling_unstable = bool(
        complete and any(bool(row.get("nonunanimous")) for row in positions)
        and not strong
    )
    return {
        "category": category,
        "complete": complete,
        "position_accuracies": accuracies,
        "min_position_accuracy": min(accuracies) if accuracies else None,
        "max_position_accuracy": max(accuracies) if accuracies else None,
        "position_accuracy_range": position_range,
        # Keep the requested labels; they are explicitly diagnostic only.
        "position_difference": bool(position_range is not None and position_range > 0),
        "strong_position_sensitive": strong,
        "weak_position_difference": weak,
        "sampling_or_position_unstable": sampling_unstable,
        "nonunanimous_position_count": sum(bool(row.get("nonunanimous")) for row in positions),
    }


def context_diagnostic(
    no_positions: list[Mapping[str, Any]], full_positions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    no_by_variant = {str(row.get("variant")): row for row in no_positions}
    full_by_variant = {str(row.get("variant")): row for row in full_positions}
    paired: list[dict[str, Any]] = []
    for variant in sorted(set(no_by_variant) & set(full_by_variant)):
        no = no_by_variant[variant]
        full = full_by_variant[variant]
        if not no.get("complete") or not full.get("complete"):
            continue
        paired.append({
            "variant": variant,
            "no_accuracy": no.get("accuracy"),
            "full_accuracy": full.get("accuracy"),
            "delta": float(full["accuracy"]) - float(no["accuracy"]),
            "helpful": float(full["accuracy"]) > float(no["accuracy"]),
        })
    no_correct = sum(int(row.get("correct") or 0) for row in no_positions)
    no_scored = sum(int(row.get("scored_total") or 0) for row in no_positions)
    full_correct = sum(int(row.get("correct") or 0) for row in full_positions)
    full_scored = sum(int(row.get("scored_total") or 0) for row in full_positions)
    no_accuracy = no_correct / no_scored if no_scored else None
    full_accuracy = full_correct / full_scored if full_scored else None
    return {
        "no_context": {"correct": no_correct, "scored_total": no_scored, "accuracy": no_accuracy},
        "full_context": {"correct": full_correct, "scored_total": full_scored, "accuracy": full_accuracy},
        "paired_position_count": len(paired),
        "paired_positions": paired,
        "context_gap": (full_accuracy - no_accuracy)
        if full_accuracy is not None and no_accuracy is not None else None,
        "context_helpful": bool(
            full_accuracy is not None and no_accuracy is not None and full_accuracy > no_accuracy
        ),
        "context_helpful_position_count": sum(bool(row["helpful"]) for row in paired),
        "context_helpful_position_rate": (
            sum(bool(row["helpful"]) for row in paired) / len(paired) if paired else None
        ),
    }
