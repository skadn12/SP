"""Offline preflight compiler for the Round-1 QA-feedback composite arm."""

from pathlib import Path


VERSION = "h4_round1_memory_bucket_qa_feedback_compiler_v1"
ALGORITHM_VERSION = "round1_memory_bucket_qa_feedback_compiler_v1"
RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")

ROUND0_REEVAL_RUN = RUN_BASE / "h4_round0_trained_solver_train_reeval_v1_live_1"
ROUND0_DIFFICULTY_RUN = RUN_BASE / "h4_round0_b0_k3_difficulty_v1_live_1"
ROUND0_EVIDENCE_RUN = RUN_BASE / "h4_round0_hard_evidence_stratification_v1_live_1"
ROUND0_TRAINING_RUN = RUN_BASE / "h4_round0_b0_matched_training_v1_live_2"
EXPANSION_COMPILER_RUN = RUN_BASE / "h4_round1_feedback_compiler_v1_live_1"
E1Q0_QA_RUN = RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_8"
SOLVER_COMPARE_RUN = RUN_BASE / "h4_round1_paired_blind_solver_compare_v1_live_1"
SOLVER_EVAL_RUN = RUN_BASE / "h4_round1_paired_blind_solver_eval_v1_live_1"
TOKENIZER_PATH = Path("/root/autodl-tmp/Qwen3-4B")
API_CONFIG_PATH = Path("/root/autodl-tmp/gxy-selfplay/config.json")

DEFAULT_RUN_ROOT = RUN_BASE / "h4_round1_memory_bucket_qa_feedback_compiler_v1_live_1"
EXPECTED_CANDIDATES = 2294
EXPECTED_ROUND0_CANDIDATES = 3286
EXPECTED_E1Q0_CANDIDATES = 530
EXPECTED_ROOTS = 8
EVIDENCE_STATUSES = (
    "EVIDENCE_CLEAN",
    "EVIDENCE_RISK",
    "EVIDENCE_INVALID",
    "EVIDENCE_UNCLASSIFIABLE",
    "NOT_APPLICABLE_NON_HARD",
)
SELECTED_ROOT_IDS = frozenset({
    "cc-train-episode-122711",
    "cc-train-episode-254852",
    "cc-train-episode-112812",
    "cc-train-episode-136202",
    "cc-train-episode-94289",
    "cc-train-episode-96462",
    "cc-train-episode-198480",
    "cc-train-episode-197635",
})
QUARANTINE_ROOT_IDS = frozenset({
    "cc-train-episode-259192",
    "cc-train-episode-218519",
})

