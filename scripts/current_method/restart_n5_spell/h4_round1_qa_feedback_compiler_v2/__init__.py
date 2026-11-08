"""Offline Round-1 QA-feedback compiler without expansion/Memory prerequisites."""

from pathlib import Path


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
ROUND0_REEVAL_RUN = RUN_BASE / "h4_round0_trained_solver_train_reeval_v1_live_1"
ROUND0_DIFFICULTY_RUN = RUN_BASE / "h4_round0_b0_k3_difficulty_v1_live_1"
ROUND0_EVIDENCE_RUN = RUN_BASE / "h4_round0_hard_evidence_stratification_v1_live_1"
ROUND0_TRAINING_RUN = RUN_BASE / "h4_round0_b0_matched_training_v1_live_2"
EXPANSION_COMPILER_RUN = RUN_BASE / "h4_round1_feedback_compiler_v1_live_1"
E1Q0_QA_RUN = RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_8"
TOKENIZER_PATH = Path("/root/autodl-tmp/Qwen3-4B")

DEFAULT_RUN_ROOT = RUN_BASE / "h4_round1_qa_feedback_compiler_v2_live_1"
VERSION = "h4_round1_qa_feedback_compiler_v2"
ALGORITHM_VERSION = "round1_qa_feedback_compiler_v2"

SELECTED_ROOT_IDS = frozenset({
    "cc-train-episode-112812", "cc-train-episode-122711",
    "cc-train-episode-136202", "cc-train-episode-197635",
    "cc-train-episode-198480", "cc-train-episode-254852",
    "cc-train-episode-94289", "cc-train-episode-96462",
})
QUARANTINE_ROOT_IDS = frozenset({"cc-train-episode-259192", "cc-train-episode-218519"})
EXPECTED_ROUND0_CANDIDATES = 3286
EXPECTED_TRAIN_CANDIDATES = 2294
EXPECTED_E1Q0_CANDIDATES = 530
CONTEXT_LIMIT = 16384
WRITER_MAX_TOKENS = 8192
INTENT_MAX_TOKENS = 4096

