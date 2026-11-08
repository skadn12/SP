"""Train-only post-training Solver reevaluation for the frozen Round-0 corpus."""

from pathlib import Path


VERSION = "h4_round0_trained_solver_train_reeval_v1"
RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DIFFICULTY_RUN = RUN_BASE / "h4_round0_b0_k3_difficulty_v1_live_1"
FORMAL_RUN = RUN_BASE / "h4_round0_b0_formal_training_v1_live_2"
TRAINING_RUN = RUN_BASE / "h4_round0_b0_matched_training_v1_live_2"
DEV_IDENTITY_REPORT = (
    RUN_BASE / "h4_round0_common_dev_eval_v1_live_3" / "model_identity_report.json"
)
DEFAULT_RUN_ROOT = RUN_BASE / "h4_round0_trained_solver_train_reeval_v1_live_1"
BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
REPO_ROOT = Path(__file__).resolve().parents[4]

MAX_MODEL_LEN = 16_384
MAX_TOKENS = 32
ROLLOUTS = 3
EXPECTED_TRAIN_CANDIDATES = 2_294
EXPECTED_TRAIN_ROOTS = 70
EXPECTED_M0_HARD = 544
EXPECTED_M0_FULL_ZERO = 312
EXPECTED_M0_FULL_CONTEXT_WORSE = 161
EXPECTED_EXPOSURE_COUNTS = {
    "SHARED_TRAINED": 309,
    "CLEAN_ONLY_TRAINED": 77,
    "INCLUSIVE_ONLY_TRAINED": 77,
    "UNEXPOSED_TO_BOTH": 1_831,
}
QTYPE_REQUESTS = {"single_choice": 24, "multi_select": 30, "ordering": 24}


__all__ = [
    "BASE_MODEL",
    "DEFAULT_RUN_ROOT",
    "DEV_IDENTITY_REPORT",
    "DIFFICULTY_RUN",
    "EXPECTED_EXPOSURE_COUNTS",
    "EXPECTED_M0_FULL_CONTEXT_WORSE",
    "EXPECTED_M0_FULL_ZERO",
    "EXPECTED_M0_HARD",
    "EXPECTED_TRAIN_CANDIDATES",
    "EXPECTED_TRAIN_ROOTS",
    "FORMAL_RUN",
    "MAX_MODEL_LEN",
    "MAX_TOKENS",
    "QTYPE_REQUESTS",
    "REPO_ROOT",
    "ROLLOUTS",
    "RUN_BASE",
    "TRAINING_RUN",
    "VERSION",
]
