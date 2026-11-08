"""Offline value analysis for the closed E1Q1 question pool."""

from pathlib import Path

RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
REROUTE_RUN = RUN_BASE / "h4_round1_qa_feedback_multilabel_rerouting_v1_live_9"
E1Q1_GENERATION_RUN = RUN_BASE / "h4_round1_qa_feedback_generation_v2_live_1"
E1Q1_CLOSURE_RUN = RUN_BASE / "h4_round1_e1q1_missing_solver_eval_v1_live_1"
E1Q1_OLD_SOLVER_RUN = RUN_BASE / "h4_round1_qa_feedback_solver_eval_v1_live_1"
E1Q0_SOLVER_RUN = RUN_BASE / "h4_round1_paired_blind_solver_eval_v1_live_1"
OUTPUT_RUN = RUN_BASE / "h4_round1_e1q1_complete_pool_value_analysis_v1_live_1"
MODELS = ("M_clean_policy", "M_inclusive_policy")
VERSION = "h4_round1_e1q1_complete_pool_value_analysis_v1"
