"""Offline preflight for the R03 QA output contract v1.1."""

from pathlib import Path


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
SOURCE_RUN = RUN_BASE / "h4_round1_r03_multi_pass_qa_generation_comparison_pilot_v1_live_2"
TARGETED_TEMPLATE_RUN = RUN_BASE / "h4_round1_r03_qa_feedback_template_v1_preflight_v1_live_4"
TARGETED_PILOT_RUN = RUN_BASE / "h4_round1_r03_feedback_guided_targeted_evolution_pilot_v1_live_1"
DEFAULT_OUTPUT = RUN_BASE / "h4_round1_r03_qa_output_contract_v1_1_preflight_live_4"
VERSION = "h4_round1_r03_qa_output_contract_v1_1_preflight"
TEMPLATE_SHA_POLICY = "PRESERVED_FROZEN_FEEDBACK_TEMPLATE_SHA"
FROZEN_FEEDBACK_TEMPLATE_SHA256 = "ed5bc3a9bed801ba83709bf51ae409e4d7fca92ef5e49cc71766ecea64d3a9a0"
