"""Local difficulty bucket correction for the isolated guided probe.

The shared Phase-0 summarizer is reused for metrics, but this probe applies
the hard/easy ordering explicitly so a full-context regression is not hidden
by the ``no_context == 1`` easy label.
"""

from __future__ import annotations

from typing import Any


def apply_policy(difficulty: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in difficulty.get("candidate_summaries") or []:
        full = (row.get("full_context") or {}).get("accuracy")
        no = (row.get("no_context") or {}).get("accuracy")
        if full is None or no is None:
            bucket = "solver_not_run"
        elif full == 0:
            bucket = "unresolved_full_zero"
        elif full < no:
            bucket = "full_context_worse"
        elif no == 1:
            bucket = "easy_no_context"
        elif full == 1:
            bucket = "context_dependent"
        elif 0 < full < 1 and full >= no:
            bucket = "hard_candidate"
        else:
            bucket = "unresolved"
        row["difficulty_bucket"] = bucket
        row["hard_candidate"] = bucket == "hard_candidate"
        row["context_dependent"] = bucket == "context_dependent"
        row["full_context_zero_manual_review"] = bucket == "unresolved_full_zero"
        counts[bucket] = counts.get(bucket, 0) + 1
    difficulty["bucket_counts"] = dict(sorted(counts.items()))
    difficulty["difficulty_policy"] = "h4_contract_guided_ordered_v1"
    return difficulty
