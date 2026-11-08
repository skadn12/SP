"""E1Q1 QA generation using the v2 offline QA-feedback packets."""

from pathlib import Path

RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
COMPILER_RUN = RUN_BASE / "h4_round1_qa_feedback_compiler_v2_live_1"
E1Q0_QA_RUN = RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_8"
API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_SMOKE_RUN = RUN_BASE / "h4_round1_qa_feedback_generation_v2_smoke_live_1"
DEFAULT_BATCH_RUN = RUN_BASE / "h4_round1_qa_feedback_generation_v2_live_1"
GENERATOR_VERSION = "h4_round1_qa_feedback_generation_v2"
TEMPERATURE = 0.1
ROOTS = (
    "cc-train-episode-112812", "cc-train-episode-122711",
    "cc-train-episode-136202", "cc-train-episode-197635",
    "cc-train-episode-198480", "cc-train-episode-254852",
    "cc-train-episode-94289",
)
SLOTS = ("TARGETED_EVOLUTION", "CONTRASTIVE_QA", "R02_COVERAGE_QA")

