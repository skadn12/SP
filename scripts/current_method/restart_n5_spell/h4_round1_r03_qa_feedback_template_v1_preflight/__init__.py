"""Offline freeze and preflight for the common R03 QA-feedback template."""

from pathlib import Path


VERSION = "h4_round1_r03_qa_feedback_template_v1_preflight"
RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_OUTPUT = RUNS / f"{VERSION}_v1_live_1"

PACKET_RUN = RUNS / "h4_round1_qa_feedback_compiler_v2_live_1"
PACKET_PATH = PACKET_RUN / "source_qa_packets.jsonl"
SEED_PATH = PACKET_RUN / "qa_feedback_seed_selection.jsonl"
SOLVER_CLEAN_PATH = RUNS / "h4_round1_e1q1_missing_solver_eval_v1_live_1" / "clean_complete_difficulty_manifest.jsonl"
SOLVER_INCLUSIVE_PATH = RUNS / "h4_round1_e1q1_missing_solver_eval_v1_live_1" / "inclusive_complete_difficulty_manifest.jsonl"
SOLVER_COMPLETE_PATH = RUNS / "h4_round1_e1q1_missing_solver_eval_v1_live_1" / "e1q1_complete_candidate_manifest.jsonl"
LIFECYCLE_PATH = RUNS / "h4_round1_question_bank_lifecycle_weight_freeze_v1_live_1" / "cluster_lifecycle_manifest.jsonl"
CLUSTER_MAP_PATH = RUNS / "h4_round1_question_bank_lifecycle_weight_freeze_v1_live_1" / "candidate_to_cluster_map.jsonl"
ACTIVE_SCOPE_PATH = RUNS / "h4_round1_question_bank_lifecycle_weight_freeze_v1_live_1" / "active_train_scope_manifest.jsonl"
TOKENIZER_PATH = Path("/root/autodl-tmp/Qwen3-4B")
CAPACITY_REPORT_PATH = RUNS / "h4_round1_r03_novel_question_capacity_preflight_v1_live_2" / "final_r03_novel_question_capacity_preflight.json"
CAPACITY_SCENES_PATH = RUNS / "h4_round1_r03_novel_question_capacity_preflight_v1_live_2" / "r03_scene_evidence_manifest.jsonl"
CAPACITY_BUNDLES_PATH = RUNS / "h4_round1_r03_novel_question_capacity_preflight_v1_live_2" / "r03_joint_evidence_bundle_manifest.jsonl"
CLOSURE_REPORT_PATH = RUNS / "h4_round1_r03_local_batch_closure_v1_live_2" / "final_r03_local_batch_closure_report.json"
CLOSURE_ROOTS_PATH = RUNS / "h4_round1_r03_local_batch_closure_v1_live_2" / "r03_local_batch_root_manifest.jsonl"
CLOSURE_CALLS_PATH = RUNS / "h4_round1_r03_local_batch_closure_v1_live_2" / "r03_local_batch_call_closure.json"

AUTHORITATIVE_ROOTS = (
    "cc-train-episode-122711", "cc-train-episode-254852",
    "cc-train-episode-112812", "cc-train-episode-136202",
    "cc-train-episode-94289", "cc-train-episode-96462",
    "cc-train-episode-198480", "cc-train-episode-197635",
)
QUARANTINE_ROOTS = frozenset({"cc-train-episode-112812"})
UNSAFE_TARGETED_SEED_ROOTS = frozenset({"cc-train-episode-96462"})
SAFE_ROOTS = tuple(root for root in AUTHORITATIVE_ROOTS if root not in QUARANTINE_ROOTS and root not in UNSAFE_TARGETED_SEED_ROOTS)
