"""Offline prompt and Solver-interface preflight for R03 QA contract v1.1."""

from pathlib import Path


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
CONTRACT_RUN = RUN_BASE / "h4_round1_r03_qa_output_contract_v1_1_preflight_live_4"
TARGETED_RUN = RUN_BASE / "h4_round1_r03_feedback_guided_targeted_evolution_pilot_v1_live_1"
TARGETED_TEMPLATE_RUN = RUN_BASE / "h4_round1_r03_qa_feedback_template_v1_preflight_v1_live_4"
MULTI_PASS_RUN = RUN_BASE / "h4_round1_r03_multi_pass_qa_generation_comparison_pilot_v1_live_2"
DEFAULT_OUTPUT = RUN_BASE / "h4_round1_r03_qa_v1_1_runtime_compat_preflight_live_1"
VERSION = "h4_round1_r03_qa_v1_1_runtime_compat_preflight"
FROZEN_FEEDBACK_TEMPLATE_SHA256 = "ed5bc3a9bed801ba83709bf51ae409e4d7fca92ef5e49cc71766ecea64d3a9a0"
ROLLOUTS = 3
CONTEXT_MODES = ("no_context", "full_context")

