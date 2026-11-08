"""Controlled Composition/Fresh QA generation comparison pilot."""

from pathlib import Path


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
VERSION = "h4_round1_r03_multi_pass_qa_generation_comparison_pilot_v1"
DEFAULT_OUTPUT = RUN_BASE / f"{VERSION}_live_1"
API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
CONTEXT_LIMIT = 16384
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.1

CLOSURE_RUN = RUN_BASE / "h4_round1_r03_local_batch_closure_v1_live_2"
CAPACITY_RUN = RUN_BASE / "h4_round1_r03_novel_question_capacity_preflight_v1_live_2"
COMPILER_RUN = RUN_BASE / "h4_round1_question_bank_feedback_compiler_v1_live_14"
LIFECYCLE_RUN = RUN_BASE / "h4_round1_question_bank_lifecycle_weight_freeze_v1_live_1"
REROUTE_RUN = RUN_BASE / "h4_round1_qa_feedback_multilabel_rerouting_v1_live_9"
TARGETED_RUN = RUN_BASE / "h4_round1_r03_feedback_guided_targeted_evolution_pilot_v1_live_1"
TARGETED_CLOSURE_RUN = RUN_BASE / "h4_round1_r03_feedback_guided_targeted_evolution_pilot_v1_offline_closure_review_v1_live_4"

SUCCESS_ROOTS = (
    "cc-train-episode-122711", "cc-train-episode-254852",
    "cc-train-episode-136202", "cc-train-episode-94289",
    "cc-train-episode-96462", "cc-train-episode-198480",
    "cc-train-episode-197635",
)
QUARANTINE_ROOT = "cc-train-episode-112812"

ALLOWED_LIFECYCLE = {
    "ACTIVE_FRONTIER_BOTH", "ACTIVE_FRONTIER_ONE", "RESOLVED_WITH_CONTEXT",
}

