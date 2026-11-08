"""Offline comparison of E1Q1 QA-feedback questions with the frozen E1Q0 baseline."""

from pathlib import Path

RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
BASELINE_RUN = RUN_BASE / "h4_round1_paired_blind_solver_eval_v1_live_1"
E1Q1_RUN = RUN_BASE / "h4_round1_qa_feedback_solver_eval_v1_live_1"
E1Q1_GENERATION_RUN = RUN_BASE / "h4_round1_qa_feedback_generation_v2_live_1"
OLD_QA_RUN = RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_8"
OUTPUT_RUN = RUN_BASE / "h4_round1_qa_feedback_solver_compare_v1_live_1"
MODELS = ("M_clean_policy", "M_inclusive_policy")
VERSION = "h4_round1_qa_feedback_solver_compare_v1"

