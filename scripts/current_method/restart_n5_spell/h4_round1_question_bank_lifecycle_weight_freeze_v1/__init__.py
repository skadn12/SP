"""Round-1 Question Bank lifecycle and weight freeze (pure offline)."""

from pathlib import Path


VERSION = "h4_round1_question_bank_lifecycle_weight_freeze_v1"
RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_OUTPUT = RUNS / f"{VERSION}_live_1"

SCOPE_RUN = RUNS / "h4_round1_question_bank_bootstrap_v1_live_1"
REROUTE_RUN = RUNS / "h4_round1_qa_feedback_multilabel_rerouting_v1_live_9"
E1Q1_CLOSURE_RUN = RUNS / "h4_round1_e1q1_missing_solver_eval_v1_live_1"
E1Q1_VALUE_RUN = RUNS / "h4_round1_e1q1_complete_pool_value_analysis_v1_live_1"
ROUND0_REEVAL_RUN = RUNS / "h4_round0_trained_solver_train_reeval_v1_live_1"
PAIRED_COMPARE_RUN = RUNS / "h4_round1_paired_blind_solver_compare_v1_live_1"
PAIRED_SOLVER_RUN = RUNS / "h4_round1_paired_blind_solver_eval_v1_live_1"
ROUND0_DIFFICULTY_RUN = RUNS / "h4_round0_b0_k3_difficulty_v1_live_1"

MODELS = ("M_clean_policy", "M_inclusive_policy")
EPS = 1e-9
