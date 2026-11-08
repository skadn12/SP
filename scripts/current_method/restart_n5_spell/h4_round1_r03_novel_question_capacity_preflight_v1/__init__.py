"""Offline capacity preflight for the authoritative local-only R03 batch."""

from pathlib import Path


VERSION = "h4_round1_r03_novel_question_capacity_preflight_v1"
RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_OUTPUT = RUNS / f"{VERSION}_live_1"

CLOSURE_RUN = RUNS / "h4_round1_r03_local_batch_closure_v1_live_2"
LIFECYCLE_RUN = RUNS / "h4_round1_question_bank_lifecycle_weight_freeze_v1_live_1"
REROUTE_RUN = RUNS / "h4_round1_qa_feedback_multilabel_rerouting_v1_live_9"
R03_RUN_PREFIX = "h4_round1_r03_memory_readonly_single_root_smoke_v1_live_"

# The order is frozen by the local batch closure.  Development live_1--live_5
# are intentionally absent from this list.
AUTHORITATIVE_RUNS = (
    ("cc-train-episode-122711", 6),
    ("cc-train-episode-254852", 7),
    ("cc-train-episode-112812", 8),
    ("cc-train-episode-136202", 9),
    ("cc-train-episode-94289", 10),
    ("cc-train-episode-96462", 11),
    ("cc-train-episode-198480", 12),
    ("cc-train-episode-197635", 13),
)

SUCCESS_DISPOSITION = "R03_LOCAL_SUCCESS"
QUARANTINE_DISPOSITION = "R03_LOCAL_QUARANTINE"

