"""E1Q1 missing canonical Solver completion."""

from pathlib import Path

RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
REROUTE_RUN = RUN_BASE / "h4_round1_qa_feedback_multilabel_rerouting_v1_live_9"
GENERATION_RUN = RUN_BASE / "h4_round1_qa_feedback_generation_v2_live_1"
EXISTING_SOLVER_RUN = RUN_BASE / "h4_round1_qa_feedback_solver_eval_v1_live_1"
IDENTITY_REPORT = RUN_BASE / "h4_round0_common_dev_eval_v1_live_3" / "model_identity_report.json"
OUTPUT_RUN = RUN_BASE / "h4_round1_e1q1_missing_solver_eval_v1_live_1"
PROBE_VERSION = "h4_round1_e1q1_missing_solver_eval_v1"

