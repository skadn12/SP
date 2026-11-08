"""Offline evidence stratification for the frozen B0 K=3 hard pool."""

from pathlib import Path


STRATIFICATION_VERSION = "h4_round0_hard_evidence_stratification_v1"
CLASSIFICATION_ALGORITHM_VERSION = "h4_round0_evidence_status_rules_v1"
RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_SOURCE_RUN = RUN_BASE / "h4_round0_b0_k3_difficulty_v1_live_1"
DEFAULT_RUN_ROOT = RUN_BASE / "h4_round0_hard_evidence_stratification_v1_live_1"

EXPECTED_HARD_COUNT = 761
EXPECTED_SPLIT_COUNTS = {"train": 544, "dev": 99, "test": 118}
EXPECTED_HARD_BY_SPLIT = EXPECTED_SPLIT_COUNTS.copy()

__all__ = [
    "CLASSIFICATION_ALGORITHM_VERSION",
    "DEFAULT_RUN_ROOT",
    "DEFAULT_SOURCE_RUN",
    "EXPECTED_HARD_COUNT",
    "EXPECTED_HARD_BY_SPLIT",
    "EXPECTED_SPLIT_COUNTS",
    "STRATIFICATION_VERSION",
]
