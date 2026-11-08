"""Offline scope freeze for the Round-1 question bank bootstrap."""

from pathlib import Path


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")

ROUND0_DIFFICULTY_RUN = RUN_BASE / "h4_round0_b0_k3_difficulty_v1_live_1"
ROUND0_REEVAL_RUN = RUN_BASE / "h4_round0_trained_solver_train_reeval_v1_live_1"
E1Q0_RUN = RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_8"
E1Q1_COMPILER_RUN = RUN_BASE / "h4_round1_qa_feedback_compiler_v2_live_1"
E1Q1_GENERATION_RUN = RUN_BASE / "h4_round1_qa_feedback_generation_v2_live_1"
E1Q1_FILTER_RUN = RUN_BASE / "h4_round1_qa_feedback_filter_v1_live_1"
E1Q1_SOLVER_RUN = RUN_BASE / "h4_round1_qa_feedback_solver_eval_v1_live_1"
E1Q0_SOLVER_RUN = RUN_BASE / "h4_round1_paired_blind_solver_eval_v1_live_1"

DEFAULT_RUN = RUN_BASE / "h4_round1_question_bank_bootstrap_v1_live_1"
DEFAULT_REROUTE_RUN = RUN_BASE / "h4_round1_qa_feedback_multilabel_rerouting_v1_live_1"

VERSION = "h4_round1_question_bank_bootstrap_v1"
SCOPE_VERSION = "round1_question_bank_scope_v1"
REROUTE_VERSION = "h4_round1_qa_feedback_multilabel_rerouting_v1"
