"""Versioned, offline-only feedback views for the next R03 branch.

The package is deliberately separate from historical R03 runners.  It only
compiles frozen local records and renders prompts; it has no service clients.
"""

from pathlib import Path

VERSION = "h4_round1_r03_feedback_view_v2"
RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_OUTPUT = RUNS / f"{VERSION}_offline_preflight_live_22"

COMPILER_RUN = RUNS / "h4_round1_question_bank_feedback_compiler_v1_live_13"
REROUTE_RUN = RUNS / "h4_round1_qa_feedback_multilabel_rerouting_v1_live_9"
R02_INPUT_RUN = RUNS / "h4_round1_r03_memory_readonly_trial_preflight_v2_live_3"
QA_SCHEMA_RUN = RUNS / "h4_round1_r03_qa_output_contract_v1_1_revision_1_preflight_live_7"
FROZEN_TARGETED_TEMPLATE_RUN = RUNS / "h4_round1_r03_qa_feedback_template_v1_preflight_v1_live_4"
HISTORICAL_FEEDBACK_RUN = RUNS / "h4_round1_r03_feedback_guided_targeted_evolution_pilot_v1_live_1"

R03_ROOTS = (
    "cc-train-episode-122711", "cc-train-episode-254852",
    "cc-train-episode-112812", "cc-train-episode-136202",
    "cc-train-episode-94289", "cc-train-episode-96462",
    "cc-train-episode-198480", "cc-train-episode-197635",
)

HIDDEN_BACKEND_KEYS = {
    "lifecycle", "frontier", "transition", "difficulty_bucket",
    "training_exposure", "position_mechanism", "file_path", "file_paths",
    "input_dialogue_path", "feedback_path",
    "duplicate_provenance", "r03_full_refs", "independent_bundle_hash",
    "joint_bundle_hash", "diagnostic_boundary", "target_dialogue_sha256",
    "target_namespace", "source_sha256", "selection_weight", "operation",
}

EXTERNAL_CALLS = {
    "memorax_search": 0, "memorax_status": 0, "memorax_add": 0,
    "memorax_mutation": 0, "qa_writer": 0, "expansion_planner": 0,
    "expansion_realizer": 0, "solver": 0, "semantic_auditor": 0,
    "training": 0,
}
