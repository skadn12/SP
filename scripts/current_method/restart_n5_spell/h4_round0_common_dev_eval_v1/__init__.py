"""Common frozen B0 K=3 Dev evaluation for M0 and two trained adapters."""

from pathlib import Path


VERSION = "h4_round0_common_dev_eval_v1"
RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DATA_RUN = RUN_BASE / "h4_round0_b0_matched_training_v1_live_2"
DIFFICULTY_RUN = RUN_BASE / "h4_round0_b0_k3_difficulty_v1_live_1"
FORMAL_RUN = RUN_BASE / "h4_round0_b0_formal_training_v1_live_2"
DEFAULT_RUN_ROOT = RUN_BASE / "h4_round0_common_dev_eval_v1_live_3"
BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
REPO_ROOT = Path(__file__).resolve().parents[4]
MAX_MODEL_LEN = 16_384
MAX_TOKENS = 32
ROLLOUTS = 3
EXPECTED_DEV = 485
EXPECTED_DEV_ROOTS = 15

__all__ = [
    "BASE_MODEL", "DATA_RUN", "DEFAULT_RUN_ROOT", "DIFFICULTY_RUN", "EXPECTED_DEV",
    "EXPECTED_DEV_ROOTS", "FORMAL_RUN", "MAX_MODEL_LEN", "MAX_TOKENS", "REPO_ROOT",
    "ROLLOUTS", "VERSION",
]
