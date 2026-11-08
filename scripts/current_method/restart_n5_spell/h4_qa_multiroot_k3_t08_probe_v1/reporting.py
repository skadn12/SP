"""Offline aggregation for the isolated K=3/T=0.8 probe.

The report deliberately distinguishes an observed K=3 outcome from a claim
that a question is intrinsically difficult.  No semantic evidence judgment is
performed here; structural grounding comes from the deterministic validator.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def _accuracy(result: Mapping[str, Any], mode: str) -> float | None:
    return ((result.get("position_diagnostics") or {}).get(mode) or {}).get("max_position_accuracy")


def _overall_accuracy(result: Mapping[str, Any], mode: str) -> float | None:
    values = [
        row.get("summary", {}).get("accuracy")
        for row in result.get("variants") or []
        if row.get("mode") == mode and row.get("summary", {}).get("complete")
        and row.get("summary", {}).get("accuracy") is not None
    ]
    if not values:
        return None
    # Every display variant is intentionally weighted equally.  The raw
    # rollout rows remain available for a micro-level aggregate if needed.
    return sum(float(value) for value in values) / len(values)


def classify_result(result: Mapping[str, Any], *, generation_source: str,
                    structural_grounding_pass: bool = True) -> dict[str, Any]:
    full = _overall_accuracy(result, "full_context")
    no = _overall_accuracy(result, "no_context")
    full_diag = (result.get("position_diagnostics") or {}).get("full_context") or {}
    no_diag = (result.get("position_diagnostics") or {}).get("no_context") or {}
    complete = bool(full_diag.get("complete") and no_diag.get("complete"))
    if not structural_grounding_pass:
        bucket = "structural_invalid"
    elif not complete or full is None or no is None:
        bucket = "solver_not_run"
    elif full == 0:
        bucket = "invalid_or_ambiguous"
    elif full < no:
        bucket = "full_context_worse"
    elif 0 < full < 1 and no < 1:
        bucket = "hard_candidate"
    elif full == 1 and no < 1:
        bucket = "full_context_resolved"
    elif no == 1:
        bucket = "easy"
    else:
        bucket = "unresolved"
    gap = full - no if full is not None and no is not None else None
    position = full_diag
    row = {
        "candidate_id": result.get("candidate_id"),
        "root_id": result.get("root_id"),
        "question_type": result.get("question_type"),
        "generation_source": generation_source,
        "structural_grounding_pass": bool(structural_grounding_pass),
        "solver_complete": complete,
        "full_context_accuracy": full,
        "no_context_accuracy": no,
        "context_gap": gap,
        # One auxiliary label is used; context_dependent is deliberately not
        # a second bucket with the same meaning.
        "context_helpful": bool(gap is not None and gap > 0),
        "strong_context_helpful": bool(gap is not None and gap >= 0.2),
        "difficulty_bucket": bucket,
        "position_category": position.get("category"),
        "position_accuracy_range": position.get("position_accuracy_range"),
        "position_difference": position.get("position_difference"),
        "strong_position_sensitive": position.get("strong_position_sensitive"),
        "sampling_or_position_unstable": position.get("sampling_or_position_unstable"),
        "technical_incomplete": not complete,
    }
    return row


def summarize_root(*, root_id: str, candidates: list[Mapping[str, Any]],
                   validity_rows: list[Mapping[str, Any]],
                   solver_results: list[Mapping[str, Any]]) -> dict[str, Any]:
    source_by_id = {
        str(candidate.get("candidate_id")): str(candidate.get("generation_source") or "unknown")
        for candidate in candidates
    }
    valid_by_id = {
        str(row.get("candidate_id")): str(row.get("decision")) == "pass"
        for row in validity_rows
    }
    rows = [classify_result(
        result,
        generation_source=source_by_id.get(str(result.get("candidate_id")), "unknown"),
        structural_grounding_pass=valid_by_id.get(str(result.get("candidate_id")), True),
    ) for result in solver_results]
    counts = Counter(str(row["difficulty_bucket"]) for row in rows)
    by_source: dict[str, Any] = {}
    for source in sorted({str(x.get("generation_source") or "unknown") for x in candidates}):
        selected = [row for row in rows if row["generation_source"] == source]
        grounded = [row for row in selected if row["structural_grounding_pass"]]
        source_counts = Counter(str(row["difficulty_bucket"]) for row in selected)
        by_source[source] = {
            "candidate_count": len(selected),
            "structurally_grounded_count": len(grounded),
            "grounding_rate": len(grounded) / len(selected) if selected else None,
            "hard_candidate_count": source_counts.get("hard_candidate", 0),
            "valid_hard_rate": (
                source_counts.get("hard_candidate", 0) / len(grounded) if grounded else None
            ),
            "context_helpful_count": sum(bool(x["context_helpful"]) for x in selected),
            "context_helpful_rate": (
                sum(bool(x["context_helpful"]) for x in selected) / len(selected)
                if selected else None
            ),
            "bucket_counts": dict(sorted(source_counts.items())),
        }
    return {
        "report_version": "h4_k3_difficulty_report_v1",
        "root_id": root_id,
        "candidate_count": len(candidates),
        "solver_result_count": len(solver_results),
        "candidate_summaries": rows,
        "bucket_counts": dict(sorted(counts.items())),
        "hard_candidate_count": counts.get("hard_candidate", 0),
        "context_helpful_count": sum(bool(x["context_helpful"]) for x in rows),
        "invalid_or_ambiguous_count": counts.get("invalid_or_ambiguous", 0),
        "structural_invalid_count": counts.get("structural_invalid", 0),
        "incomplete_count": counts.get("solver_not_run", 0),
        "by_source": by_source,
        "difficulty_policy": {
            "hard_candidate": "0 < full_context_accuracy < 1 and no_context_accuracy < 1",
            "context_helpful": "full_context_accuracy > no_context_accuracy (auxiliary label)",
            "invalid_or_ambiguous": "full_context_accuracy == 0",
            "position_sensitive": "position_accuracy_range > 0.3 (diagnostic only)",
            "semantic_grounding": "not judged by this probe",
        },
    }


def summarize_batch(root_reports: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(root_reports)
    merged = Counter()
    root_rates: list[dict[str, Any]] = []
    for report in rows:
        merged.update(report.get("bucket_counts") or {})
        total = int(report.get("candidate_count") or 0)
        hard = int(report.get("hard_candidate_count") or 0)
        root_rates.append({
            "root_id": report.get("root_id"),
            "candidate_count": total,
            "hard_candidate_count": hard,
            "hard_rate": hard / total if total else None,
            "context_helpful_count": report.get("context_helpful_count", 0),
            "bucket_counts": report.get("bucket_counts") or {},
        })
    candidate_total = sum(int(report.get("candidate_count") or 0) for report in rows)
    hard_total = sum(int(report.get("hard_candidate_count") or 0) for report in rows)
    return {
        "report_version": "h4_k3_batch_summary_v1",
        "root_count": len(rows),
        "roots": root_rates,
        "micro": {
            "candidate_count": candidate_total,
            "hard_candidate_count": hard_total,
            "hard_rate": hard_total / candidate_total if candidate_total else None,
            "bucket_counts": dict(sorted(merged.items())),
        },
        "macro": {
            "root_hard_rate_mean": (
                sum(float(row["hard_rate"]) for row in root_rates if row["hard_rate"] is not None)
                / sum(row["hard_rate"] is not None for row in root_rates)
                if any(row["hard_rate"] is not None for row in root_rates) else None
            ),
            "root_level_hard_presence": sum(
                bool(row["hard_candidate_count"]) for row in root_rates
            ),
        },
        "interpretation": (
            "Exploratory K=3 observations only; no claim of stable position sensitivity, "
            "temperature superiority, or semantic correctness."
        ),
    }


__all__ = ["classify_result", "summarize_batch", "summarize_root"]
