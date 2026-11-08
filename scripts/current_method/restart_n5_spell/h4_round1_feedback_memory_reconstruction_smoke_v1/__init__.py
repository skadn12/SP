"""Offline preflight for the Round-1 R01-only Memory reconstruction smoke."""

from pathlib import Path


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
COMPILER_RUN = RUN_BASE / "h4_round1_feedback_compiler_v1_live_1"
CONTINUATION_RUNS = RUN_BASE
DEFAULT_RUN_ROOT = RUN_BASE / (
    "h4_round1_feedback_memory_reconstruction_smoke_v1_live_1")

__all__ = ["COMPILER_RUN", "CONTINUATION_RUNS", "DEFAULT_RUN_ROOT"]
