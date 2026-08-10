"""Deterministic paid-request budgets and diagnostics for v11 batches."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


BATCH_COST_CONTROL_VERSION = "slim_v11_batch_cost_control_v1"


class PaidRequestBudgetExhausted(RuntimeError):
    def __init__(self, *, root_id: str, stage: str, scope: str):
        self.root_id = root_id
        self.stage = stage
        self.scope = scope
        super().__init__(f"budget_exhausted_before_{stage}:{scope}:{root_id}")


def normalize_error_code(value: Any) -> str:
    """Keep the contract code while removing volatile indexes and references."""
    text = str(value or "unknown_error").strip().lower()
    text = re.sub(r"r\d{2}:s\d{2}:t\d{3}", "<turn_ref>", text)
    text = re.sub(r"r\d{2}:s\d{2}", "<scene_ref>", text)
    text = re.sub(r"scene_review:\d+", "scene_review:<index>", text)
    text = re.sub(r"(attempt|candidate)[_: -]?\d+", r"\1:<index>", text)
    return text or "unknown_error"


def stable_error_signature(
    *, stage: str, item_ref: str, errors: Sequence[Any],
) -> str:
    codes = sorted({normalize_error_code(value) for value in errors})
    return f"{stage}|{item_ref}|{'&'.join(codes) if codes else 'unknown_error'}"


@dataclass
class BatchRequestBudget:
    root_ids: Sequence[str]
    per_root_paid_limit: int = 24
    batch_paid_limit: int | None = None
    root_paid_counts: dict[str, int] = field(default_factory=dict)
    stage_paid_counts: Counter[str] = field(default_factory=Counter)
    root_stage_paid_counts: dict[str, Counter[str]] = field(default_factory=dict)
    retry_request_count: int = 0
    cached_request_count: int = 0
    error_signatures: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        roots = tuple(dict.fromkeys(str(value) for value in self.root_ids))
        if not roots:
            raise ValueError("batch_budget_requires_roots")
        if self.per_root_paid_limit < 1:
            raise ValueError("per_root_paid_limit_invalid")
        self.root_ids = roots
        if self.batch_paid_limit is None:
            self.batch_paid_limit = len(roots) * self.per_root_paid_limit
        if int(self.batch_paid_limit) < 1:
            raise ValueError("batch_paid_limit_invalid")
        self.root_paid_counts = {root_id: 0 for root_id in roots}
        self.root_stage_paid_counts = {
            root_id: Counter() for root_id in roots
        }

    @property
    def paid_request_count(self) -> int:
        return sum(self.root_paid_counts.values())

    def reserve(
        self, *, root_id: str, stage: str, item_ref: str,
        is_retry: bool = False,
    ) -> None:
        if root_id not in self.root_paid_counts:
            raise ValueError(f"batch_budget_unknown_root:{root_id}")
        clean_stage = str(stage or "unknown_stage")
        if self.root_paid_counts[root_id] >= self.per_root_paid_limit:
            raise PaidRequestBudgetExhausted(
                root_id=root_id, stage=clean_stage, scope="root")
        if self.paid_request_count >= int(self.batch_paid_limit):
            raise PaidRequestBudgetExhausted(
                root_id=root_id, stage=clean_stage, scope="batch")
        self.root_paid_counts[root_id] += 1
        self.stage_paid_counts[clean_stage] += 1
        self.root_stage_paid_counts[root_id][clean_stage] += 1
        if is_retry:
            self.retry_request_count += 1

    def record_errors(
        self, *, stage: str, item_ref: str, errors: Sequence[Any],
    ) -> str:
        signature = stable_error_signature(
            stage=stage, item_ref=item_ref, errors=errors)
        self.error_signatures[signature] += 1
        return signature

    def snapshot(self) -> dict[str, Any]:
        return {
            "budget_version": BATCH_COST_CONTROL_VERSION,
            "per_root_paid_request_limit": self.per_root_paid_limit,
            "batch_paid_request_limit": int(self.batch_paid_limit),
            "paid_request_count": self.paid_request_count,
            "retry_request_count": self.retry_request_count,
            "cached_request_count": self.cached_request_count,
            "request_count_by_stage": dict(sorted(self.stage_paid_counts.items())),
            "request_count_by_root": dict(self.root_paid_counts),
            "request_count_by_root_and_stage": {
                root_id: dict(sorted(counts.items()))
                for root_id, counts in self.root_stage_paid_counts.items()
            },
            "wasted_requests_by_error_signature": dict(
                sorted(self.error_signatures.items())),
        }


def candidate_terminal_status(
    *, stage: str, final_decision: str | None = None,
    admitted: bool = False,
) -> str:
    if admitted:
        return "admitted"
    decision = str(final_decision or "").lower()
    if stage == "questioner":
        return "parse_invalid" if "json" in decision else "rejected_format"
    if stage == "semantic_audit":
        return "rejected_semantic"
    if stage == "distractor":
        return "parse_invalid" if "json" in decision else "rejected_format"
    if "position" in decision:
        return "position_sensitive"
    if "semantic_leakage" in decision or "too_easy" in decision:
        return "solver_too_easy"
    if "full_context" in decision or "too_hard" in decision:
        return "solver_too_hard"
    return "solver_rejected"
