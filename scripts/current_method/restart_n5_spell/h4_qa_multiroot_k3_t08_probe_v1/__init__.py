"""Isolated K=3/T=0.8 H4 multi-root diagnostics."""

from .diagnostics import aggregate_position, classify_candidate, context_diagnostic
from .permutation import audit_gold_rebinding, position_variants
from .seed_schedule import SEED_SCHEDULE_VERSION, seed_schedule, stable_seed

__all__ = [
    "SEED_SCHEDULE_VERSION",
    "aggregate_position",
    "audit_gold_rebinding",
    "classify_candidate",
    "context_diagnostic",
    "position_variants",
    "seed_schedule",
    "stable_seed",
]
