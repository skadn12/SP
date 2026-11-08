"""Offline preparation for the revision-1 Composition/Fresh comparison."""

from pathlib import Path


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
OLD_PILOT = RUN_BASE / "h4_round1_r03_multi_pass_qa_generation_comparison_pilot_v1_live_2"
REVISION_PREFLIGHT = RUN_BASE / "h4_round1_r03_qa_output_contract_v1_1_revision_1_preflight_live_7"
TEMPLATE_RUN = RUN_BASE / "h4_round1_r03_qa_feedback_template_v1_preflight_v1_live_4"
OUTPUT = RUN_BASE / "h4_round1_r03_multi_pass_qa_generation_comparison_revision_1_preflight_live_1"
SUCCESS_ROOTS = (
    "cc-train-episode-122711", "cc-train-episode-254852",
    "cc-train-episode-136202", "cc-train-episode-94289",
    "cc-train-episode-96462", "cc-train-episode-198480",
    "cc-train-episode-197635",
)
QUARANTINE_ROOT = "cc-train-episode-112812"
MAX_REQUESTS = 14
CONTEXT_LIMIT = 16384
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.1
