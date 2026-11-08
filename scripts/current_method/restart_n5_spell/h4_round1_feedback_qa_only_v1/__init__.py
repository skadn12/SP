"""Round-1 QA-only generation conditioned on frozen Round-0 feedback."""

from pathlib import Path


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
REPO_ROOT = Path("/root/autodl-tmp/gxy-selfplay/current_method_github")
API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")

FEEDBACK_COMPILER_RUN = RUN_BASE / "h4_round1_feedback_compiler_v1_live_1"
E1Q0_QA_RUN = RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_8"
ROUND0_DIFFICULTY_RUN = RUN_BASE / "h4_round0_b0_k3_difficulty_v1_live_1"
ROUND0_TRANSITION_RUN = RUN_BASE / "h4_round0_trained_solver_train_reeval_v1_live_1"

DEFAULT_SMOKE_RUN = RUN_BASE / "h4_round1_feedback_qa_only_v1_smoke_live_1"
DEFAULT_BATCH_RUN = RUN_BASE / "h4_round1_feedback_qa_only_v1_live_1"

ROOTS = (
    "cc-train-episode-112812",
    "cc-train-episode-122711",
    "cc-train-episode-136202",
    "cc-train-episode-197635",
    "cc-train-episode-198480",
    "cc-train-episode-254852",
    "cc-train-episode-94289",
    "cc-train-episode-96462",
)

SLOT_TYPES = (
    "TARGETED_FEEDBACK_QA",
    "CONTRASTIVE_FEEDBACK_QA",
    "COVERAGE_FEEDBACK_QA",
)

GENERATOR_VERSION = "h4_round1_feedback_qa_only_v1"
GENERATOR_TEMPERATURE = 0.1
GENERATOR_MAX_TOKENS = 8192
INTENT_MAX_TOKENS = 4096
