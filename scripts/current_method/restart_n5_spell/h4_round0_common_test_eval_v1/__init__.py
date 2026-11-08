"""Frozen B0 K=3 Test evaluation for the trained adapters."""

from pathlib import Path

VERSION = "h4_round0_common_test_eval_v1"
RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DATA_RUN = RUN_BASE / "h4_round0_b0_matched_training_v1_live_2"
DIFFICULTY_RUN = RUN_BASE / "h4_round0_b0_k3_difficulty_v1_live_1"
FORMAL_RUN = RUN_BASE / "h4_round0_b0_formal_training_v1_live_2"
DEV_RUN = RUN_BASE / "h4_round0_common_dev_eval_v1_live_3"
DEFAULT_RUN_ROOT = RUN_BASE / "h4_round0_common_test_eval_v1_live_1"
BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
MAX_MODEL_LEN = 16_384
EXPECTED_TEST = 507
