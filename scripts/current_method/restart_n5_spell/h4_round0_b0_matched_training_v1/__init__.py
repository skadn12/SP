"""Matched Round-0 B0 answer-only dataset construction without training."""

from pathlib import Path


DATASET_VERSION = "h4_round0_b0_matched_training_v1"
MATCHING_ALGORITHM_VERSION = "h4_round0_clean_risk_min_cost_match_v1"
PERMUTATION_ALGORITHM_VERSION = "h4_round0_paired_display_target_balance_v1"
RENDER_ALGORITHM_VERSION = "h4_round0_b0_answer_only_chat_v1"
EVAL_FREEZE_VERSION = "h4_round0_common_b0_eval_freeze_v1"

RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_DIFFICULTY_RUN = RUN_BASE / "h4_round0_b0_k3_difficulty_v1_live_1"
DEFAULT_STRATIFICATION_RUN = RUN_BASE / "h4_round0_hard_evidence_stratification_v1_live_1"
DEFAULT_RUN_ROOT = RUN_BASE / "h4_round0_b0_matched_training_v1_live_2"
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_BLACKLIST = Path(__file__).resolve().parent / "confirmed_invalid_blacklist_v1.json"

__all__ = [
    "DATASET_VERSION",
    "DEFAULT_BLACKLIST",
    "DEFAULT_DIFFICULTY_RUN",
    "DEFAULT_RUN_ROOT",
    "DEFAULT_STRATIFICATION_RUN",
    "DEFAULT_TOKENIZER",
    "EVAL_FREEZE_VERSION",
    "MATCHING_ALGORITHM_VERSION",
    "PERMUTATION_ALGORITHM_VERSION",
    "RENDER_ALGORITHM_VERSION",
]
