"""Offline Round-1 feedback compiler for frozen Round-0 Train results."""

from pathlib import Path


VERSION = "h4_round1_feedback_compiler_v1"
ALGORITHM_VERSION = "round1_feedback_compiler_v1"
RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")

TRAIN_REEVAL_RUN = RUN_BASE / "h4_round0_trained_solver_train_reeval_v1_live_1"
M0_DIFFICULTY_RUN = RUN_BASE / "h4_round0_b0_k3_difficulty_v1_live_1"
TRAINING_RUN = RUN_BASE / "h4_round0_b0_matched_training_v1_live_2"
EVIDENCE_RUN = RUN_BASE / "h4_round0_hard_evidence_stratification_v1_live_1"
DEV_RUN = RUN_BASE / "h4_round0_common_dev_eval_v1_live_3"
ROUND1_PLAN_RUN = RUN_BASE / "h4_round1_cumulative_continuation_fresh_mining_v1_live_47"

DEFAULT_RUN_ROOT = RUN_BASE / "h4_round1_feedback_compiler_v1_live_1"
TOKENIZER_PATH = Path("/root/autodl-tmp/Qwen3-4B")
API_CONFIG_PATH = Path("/root/autodl-tmp/gxy-selfplay/config.json")

EXPECTED_CANDIDATES = 2_294
EXPECTED_ROOTS = 70
EXPECTED_TECHNICAL = 0
FORMAL_ROOT_COUNT = 10
RESERVE_ROOT_COUNT = 5
ENDPOINT_CONTEXT_LIMIT = 40_960
EXPANSION_OUTPUT_RESERVE = 8_192
FUTURE_R02_DIALOGUE_RESERVE = 4_096
FUTURE_SOLVER_LIMIT = 16_384
FUTURE_SOLVER_OUTPUT_RESERVE = 32
FUTURE_SOLVER_FIXED_OVERHEAD = 512
FUTURE_SFT_LIMIT = 8_192

