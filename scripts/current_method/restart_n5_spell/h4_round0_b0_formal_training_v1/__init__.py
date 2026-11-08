"""Formal Round-0 B0 Clean versus Inclusive training launcher."""

from pathlib import Path


FORMAL_VERSION = "h4_round0_b0_formal_training_v1"
RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DATA_RUN = RUN_BASE / "h4_round0_b0_matched_training_v1_live_2"
SMOKE_RUN = RUN_BASE / "h4_round0_b0_training_smoke_v1_live_1"
DEFAULT_RUN_ROOT = RUN_BASE / "h4_round0_b0_formal_training_v1_live_2"
BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
REPO_ROOT = Path(__file__).resolve().parents[4]
REQUIRED_MAX_LENGTH = 8192
EPOCHS = 2.0
EXPECTED_TRAIN_RECORDS = 386
EXPECTED_OPTIMIZER_STEPS = 193
EXPECTED_BACKWARD_CALLS = 772

__all__ = [
    "BASE_MODEL", "DATA_RUN", "DEFAULT_RUN_ROOT", "EPOCHS", "EXPECTED_BACKWARD_CALLS",
    "EXPECTED_OPTIMIZER_STEPS", "EXPECTED_TRAIN_RECORDS", "FORMAL_VERSION",
    "REPO_ROOT", "REQUIRED_MAX_LENGTH", "SMOKE_RUN",
]
