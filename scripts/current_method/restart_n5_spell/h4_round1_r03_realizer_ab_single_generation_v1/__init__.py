"""One-shot live Realizer A/B comparison using frozen messages."""

from pathlib import Path

from ..h4_round1_r03_feedback_view_v2 import RUNS


VERSION = "h4_round1_r03_realizer_ab_single_generation_v1"
DEFAULT_OUTPUT = RUNS / f"{VERSION}_live_1"
PREPARATION_RUN = RUNS / "h4_round1_r03_realizer_ab_input_preparation_v1_live_3"
SOURCE_RUN = RUNS / "h4_round1_r03_feedback_view_v2_single_root_debug_live_1"
CONFIG_PATH = Path("/root/autodl-tmp/gxy-selfplay/config.json")
ROOT_ID = "cc-train-episode-122711"
TIMEOUT_SECONDS = 600

