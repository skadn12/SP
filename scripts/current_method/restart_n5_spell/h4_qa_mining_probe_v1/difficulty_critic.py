"""Deterministic diagnostic labels for Phase 0.

The critic is intentionally a report-only layer.  It does not reject a QA
candidate on a keyword heuristic and it does not call the production writer.
"""

from __future__ import annotations

from typing import Any, Mapping


def critic_labels(candidate: Mapping[str, Any], summary: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    if summary.get("full_context_zero_manual_review"):
        labels.append("full_context_zero_manual_review")
    if summary.get("difficulty_bucket") == "easy_no_context":
        labels.append("surface_fact_or_weak_distractor")
    if summary.get("difficulty_bucket") == "full_context_worse":
        labels.append("context_conflict_or_solver_noise")
    if summary.get("difficulty_bucket") == "hard_candidate":
        full = (summary.get("full_context") or {}).get("accuracy")
        no = (summary.get("no_context") or {}).get("accuracy")
        if full is not None and no is not None:
            labels.append("context_gain" if full > no else "no_context_advantage")
    if summary.get("position_accuracy_range") is not None and summary["position_accuracy_range"] >= 0.8:
        labels.append("position_sensitivity")
    if any("gold_uniquely_longest" == warning for warning in candidate.get("static_warnings") or []):
        labels.append("gold_length_leakage_risk")
    if any("question_option_keyword_overlap" == warning for warning in candidate.get("static_warnings") or []):
        labels.append("question_option_overlap_risk")
    if any("possible_lexical_temporal_leakage" == warning for warning in candidate.get("static_warnings") or []):
        labels.append("temporal_wording_leakage_risk")
    if not labels:
        labels.append("no_phase0_diagnostic_flag")
    return sorted(set(labels))


def build_critic_report(candidates: list[Mapping[str, Any]], difficulty: Mapping[str, Any]) -> dict[str, Any]:
    by_id = {str(row.get("candidate_id")): row for row in candidates}
    reports = []
    for summary in difficulty.get("candidate_summaries") or []:
        candidate = dict(by_id.get(str(summary.get("candidate_id"))) or {})
        candidate["static_warnings"] = summary.get("static_warnings") or []
        reports.append({
            "candidate_id": summary.get("candidate_id"),
            "difficulty_bucket": summary.get("difficulty_bucket"),
            "labels": critic_labels(candidate, summary),
            "manual_review": bool(summary.get("full_context_zero_manual_review")),
        })
    return {"critic_version": "h4_phase0_deterministic_v1", "items": reports}
