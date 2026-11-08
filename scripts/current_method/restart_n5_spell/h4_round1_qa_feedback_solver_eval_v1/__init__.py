"""B0 K=3 Solver evaluation for the frozen E1Q1 primary QA pool."""

from pathlib import Path

RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
FILTER_RUN = RUN_BASE / "h4_round1_qa_feedback_filter_v1_live_1"
GENERATION_RUN = RUN_BASE / "h4_round1_qa_feedback_generation_v2_live_1"
IDENTITY_REPORT = RUN_BASE / "h4_round0_common_dev_eval_v1_live_3" / "model_identity_report.json"
BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
OUTPUT_RUN = RUN_BASE / "h4_round1_qa_feedback_solver_eval_v1_live_1"
PROBE_VERSION = "h4_round1_qa_feedback_solver_eval_v1"
MODELS = ("M_clean_policy", "M_inclusive_policy")

