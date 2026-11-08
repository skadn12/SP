"""Offline closure for the eight-root local-only R03 dialogue batch."""

from pathlib import Path


VERSION = "h4_round1_r03_local_batch_closure_v1"
RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_OUTPUT = RUNS / f"{VERSION}_live_1"

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

DEVELOPMENT_RUN_NUMBERS = tuple(range(1, 6))
R03_RUN_PREFIX = "h4_round1_r03_memory_readonly_single_root_smoke_v1_live_"
SUCCESS_STATUS = "R03_SINGLE_ROOT_READONLY_MEMORY_SMOKE_PASS"

