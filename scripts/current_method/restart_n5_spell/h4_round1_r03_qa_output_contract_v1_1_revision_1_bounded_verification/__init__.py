"""Bounded live verification for the v1.1 revision-1 QA output contract."""

from pathlib import Path


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
SOURCE_BASE = Path("/root/autodl-tmp/gxy-selfplay/current_method_github/scripts/current_method/restart_n5_spell")
TEMPLATE_RUN = RUN_BASE / "h4_round1_r03_qa_feedback_template_v1_preflight_v1_live_4"
CLOSURE_RUN = RUN_BASE / "h4_round1_r03_local_batch_closure_v1_live_2"
REVISION_PREFLIGHT_RUN = RUN_BASE / "h4_round1_r03_qa_output_contract_v1_1_revision_1_preflight_live_7"
API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
OUTPUT = RUN_BASE / "h4_round1_r03_qa_output_contract_v1_1_revision_1_bounded_verification_live_1"
ROOTS = ("cc-train-episode-122711", "cc-train-episode-198480")
MAX_REQUESTS = 2
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.1
FROZEN_TEMPLATE_SHA256 = "ed5bc3a9bed801ba83709bf51ae409e4d7fca92ef5e49cc71766ecea64d3a9a0"
