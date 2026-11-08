"""Phase-0 difficulty buckets for the isolated H4 QA mining probe.

This module deliberately does not make an admission decision for production
QA.  It only summarizes the frozen Solver replay and keeps the user's
``full-context = 0/5`` cases visible for manual inspection.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping


def _mode_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def _correct_count(row: Mapping[str, Any]) -> int:
        value = row.get("correct")
        # Single-choice rows store a five-rollout integer; structured rows
        # store a one-rollout boolean.  Do not collapse the former to 1.
        return int(value) if isinstance(value, (int, float, bool)) else 0

    correct = sum(_correct_count(row) for row in rows)
    total = sum(int(row.get("total") or 0) for row in rows)
    parse_failures = 0
    reasoning_errors = 0
    for row in rows:
        if row.get("failure_type") == "parse_failure":
            parse_failures += 1
        elif row.get("failure_type") == "reasoning_error":
            reasoning_errors += 1
        if row.get("rollouts"):
            parse_failures += sum(
                int(rollout.get("local_reward") is None)
                for rollout in row.get("rollouts") or []
            )
        else:
            parse_failures += int(row.get("parse_failure") or 0)
    return {
        "correct": correct,
        "total": total,
        "accuracy": (correct / total if total else None),
        "parse_failure_count": parse_failures,
        "reasoning_error_count": reasoning_errors,
    }


def summarize_candidate(result: Mapping[str, Any]) -> dict[str, Any]:
    by_mode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_variant: dict[str, dict[str, Any]] = {}
    for row in result.get("variants") or []:
        mode = str(row.get("mode") or "unknown")
        by_mode[mode].append(row)
        variant = str(row.get("variant") or "unknown")
        bucket = by_variant.setdefault(variant, {"variant": variant, "modes": {}})
        bucket["modes"][mode] = _mode_summary([row])
    mode_summary = {mode: _mode_summary(rows) for mode, rows in by_mode.items()}
    full = mode_summary.get("full_context", {"correct": 0, "total": 0, "accuracy": None})
    no = mode_summary.get("no_context", {"correct": 0, "total": 0, "accuracy": None})
    full_acc = full.get("accuracy")
    no_acc = no.get("accuracy")
    if not full.get("total"):
        bucket = "solver_not_run"
    elif full.get("correct", 0) == 0:
        bucket = "unresolved_full_zero"
    elif no_acc == 1:
        bucket = "easy_no_context"
    elif full_acc is not None and no_acc is not None and full_acc < no_acc:
        bucket = "full_context_worse"
    elif full_acc == 1 and no_acc < 1:
        bucket = "context_dependent"
    elif (full_acc is not None and no_acc is not None and
          0 < full_acc < 1 and no_acc < 1 and full_acc >= no_acc):
        bucket = "hard_candidate"
    else:
        bucket = "unresolved"
    ranges_by_mode: dict[str, float | None] = {}
    for mode in ("no_context", "full_context"):
        rates = [float(item["modes"][mode]["accuracy"])
                 for item in by_variant.values()
                 if mode in item["modes"] and item["modes"][mode].get("accuracy") is not None]
        ranges_by_mode[mode] = max(rates) - min(rates) if rates else None
    position_range = ranges_by_mode.get("full_context")
    if position_range is None:
        position_range = ranges_by_mode.get("no_context")
    return {
        "candidate_id": result.get("candidate_id"),
        "question_type": result.get("question_type"),
        "difficulty_bucket": bucket,
        "hard_candidate": bucket == "hard_candidate",
        "context_dependent": bucket == "context_dependent",
        "full_context_zero_manual_review": bucket == "unresolved_full_zero",
        "full_context": full,
        "no_context": no,
        "position_accuracy_range": position_range,
        "position_accuracy_ranges": ranges_by_mode,
        "position_audit_variant_count": len(by_variant),
        "variants": list(by_variant.values()),
    }


def summarize_all(results: list[Mapping[str, Any]], static_rows: Mapping[str, Any] | None = None) -> dict[str, Any]:
    static_rows = static_rows or {}
    summaries = []
    for result in results:
        item = summarize_candidate(result)
        static = static_rows.get(str(result.get("candidate_id")), {})
        item["static_decision"] = static.get("decision", "unknown")
        item["static_warnings"] = list(static.get("warnings") or [])
        if static.get("decision") == "reject":
            item["difficulty_bucket"] = "static_reject"
            item["hard_candidate"] = False
        summaries.append(item)
    counts: dict[str, int] = defaultdict(int)
    for item in summaries:
        counts[str(item["difficulty_bucket"])] += 1
    return {"candidate_summaries": summaries, "bucket_counts": dict(sorted(counts.items()))}
