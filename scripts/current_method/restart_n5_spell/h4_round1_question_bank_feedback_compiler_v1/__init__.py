"""Pure-offline Round-1 Question Bank feedback compiler."""

from pathlib import Path

VERSION = "h4_round1_question_bank_feedback_compiler_v1"
RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_OUTPUT = RUNS / f"{VERSION}_live_1"

FREEZE_RUN = RUNS / "h4_round1_question_bank_lifecycle_weight_freeze_v1_live_1"
REROUTE_RUN = RUNS / "h4_round1_qa_feedback_multilabel_rerouting_v1_live_9"
VALUE_RUN = RUNS / "h4_round1_e1q1_complete_pool_value_analysis_v1_live_1"
PAIRED_QA_RUN = RUNS / "h4_round1_paired_blind_fresh_qa_v1_live_8"

ACTIVE_CANDIDATES = 3062
PROMPT_CLUSTERS = 3048
ALIASES = 14
R03_ROOTS = (
    "cc-train-episode-122711", "cc-train-episode-254852",
    "cc-train-episode-112812", "cc-train-episode-136202",
    "cc-train-episode-94289", "cc-train-episode-96462",
    "cc-train-episode-198480", "cc-train-episode-197635",
)
QUARANTINE_ROOTS = ("cc-train-episode-218519", "cc-train-episode-259192")
ENDPOINT_CONTEXT_LIMIT = 40960
EXPANSION_OUTPUT_RESERVE = 8192
FROZEN_SAMPLING_SEED = "round1_question_bank_feedback_compiler_sampling_v1"
TARGET_ROUND = "R03"
