"""Round-1 paired blind B0 K=3 Solver evaluation."""

from pathlib import Path

RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
QA_RUN = RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_8"
IDENTITY_REPORT = RUN_BASE / "h4_round0_common_dev_eval_v1_live_3" / "model_identity_report.json"
BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
OUTPUT_RUN = RUN_BASE / "h4_round1_paired_blind_solver_eval_v1_live_1"
PROBE_VERSION = "h4_round1_paired_blind_solver_eval_v1"
MODELS = ("M_clean_policy", "M_inclusive_policy")
BRANCHES = ("control", "feedback")

__all__ = ["BASE_MODEL", "BRANCHES", "IDENTITY_REPORT", "MODELS", "OUTPUT_RUN", "PROBE_VERSION", "QA_RUN", "RUN_BASE"]
