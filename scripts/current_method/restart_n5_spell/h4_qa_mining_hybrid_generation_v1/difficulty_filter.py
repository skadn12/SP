from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ..h4_qa_mining_probe_v1.difficulty_filter import summarize_all


def _bucket(full: float | None, no: float | None) -> str:
    # User-defined hard target: neither context is perfect, while full context
    # remains non-zero.  Categories are ordered to avoid overlap.
    if full is None or no is None:
        return "solver_not_run"
    if full == 0:
        return "invalid_or_ambiguous"
    if full < no:
        return "full_context_worse"
    if 0 < full < 1 and no < 1:
        return "hard_candidate"
    if full == 1 and no < 1:
        return "full_context_resolved"
    if no == 1:
        return "easy"
    return "unresolved"


def apply_hybrid_policy(raw: Mapping[str, Any], candidates: list[Mapping[str, Any]],
                        basic_rows: Mapping[str, Any] | None = None) -> dict[str, Any]:
    base = summarize_all(list(raw.get("results") or []), basic_rows or {})
    source_by_id = {str(x.get("candidate_id")): str(x.get("generation_source") or "unknown") for x in candidates}
    candidate_rows = []
    for row in base.get("candidate_summaries") or []:
        full = (row.get("full_context") or {}).get("accuracy")
        no = (row.get("no_context") or {}).get("accuracy")
        row = dict(row)
        row["generation_source"] = source_by_id.get(str(row.get("candidate_id")), "unknown")
        row["difficulty_bucket"] = _bucket(full, no)
        row["hard_candidate"] = row["difficulty_bucket"] == "hard_candidate"
        row["context_helpful"] = bool(full is not None and no is not None and full > no)
        row["context_gap"] = (full - no) if full is not None and no is not None else None
        row["strong_context_helpful"] = bool(row["context_gap"] is not None and row["context_gap"] >= 0.2)
        row["position_sensitive"] = bool((row.get("position_accuracy_range") or 0) > 0.3)
        row["full_context_zero_manual_review"] = full == 0
        candidate_rows.append(row)
    counts = Counter(str(x["difficulty_bucket"]) for x in candidate_rows)
    by_source: dict[str, dict[str, Any]] = {}
    for source in sorted({str(x.get("generation_source") or "unknown") for x in candidates}):
        rows = [x for x in candidate_rows if x["generation_source"] == source]
        grounded = [x for x in candidates if x.get("generation_source") == source and
                     (basic_rows or {}).get(str(x.get("candidate_id")), {}).get("structural_grounding_pass")]
        src_counts = Counter(str(x["difficulty_bucket"]) for x in rows)
        aligned = sum(1 for x in candidates if x.get("generation_source") == source and
                      all(bool(v) for v in (x.get("contract_group_alignment") or {}).values()))
        by_source[source] = {
            "candidate_count": len(rows), "structurally_grounded_count": len(grounded),
            "grounding_rate": len(grounded) / len(rows) if rows else None,
            "hard_candidate_count": src_counts.get("hard_candidate", 0),
            "valid_hard_rate": src_counts.get("hard_candidate", 0) / len(grounded) if grounded else None,
            "context_helpful_count": sum(bool(x.get("context_helpful")) for x in rows),
            "context_helpful_rate": sum(bool(x.get("context_helpful")) for x in rows) / len(rows) if rows else None,
            "alignment_metadata_all_true_count": aligned,
            "bucket_counts": dict(sorted(src_counts.items())),
        }
    return {
        "candidate_summaries": candidate_rows,
        "bucket_counts": dict(sorted(counts.items())),
        "difficulty_policy": "h4_hybrid_user_policy_v1",
        "labels": {
            "hard_candidate": "0 < full_context_accuracy < 1 and no_context_accuracy < 1",
            "context_helpful": "full_context_accuracy > no_context_accuracy (auxiliary, not a separate bucket)",
            "invalid_or_ambiguous": "full_context_accuracy == 0",
            "position_sensitive": "position_accuracy_range > 0.3 (diagnostic only)",
        },
        "by_source": by_source,
    }

