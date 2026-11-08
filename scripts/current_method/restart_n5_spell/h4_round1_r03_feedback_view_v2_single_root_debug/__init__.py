"""One authorized live Expansion debug run using Feedback View v2."""

from pathlib import Path

from ..h4_round1_r03_feedback_view_v2 import RUNS

VERSION = "h4_round1_r03_feedback_view_v2_single_root_debug"
DEFAULT_OUTPUT = RUNS / f"{VERSION}_live_1"
ROOT_ID = "cc-train-episode-122711"
CONFIG_PATH = Path("/root/autodl-tmp/gxy-selfplay/config.json")
TOKENIZER_PATH = Path("/root/autodl-tmp/Qwen3-4B")
PREFLIGHT_OUTPUT = RUNS / "h4_round1_r03_feedback_view_v2_offline_preflight_live_5"
MEMORY_PREFLIGHT = RUNS / "h4_round1_r03_memory_readonly_trial_preflight_v2_live_3"
CONTEXT_CEILING = 40960
MODEL = "Qwen/Qwen3-14B"
TEMPERATURE = 0.25
PLANNER_MAX_TOKENS = 4096
REALIZER_MAX_TOKENS = 8192
TIMEOUT_SECONDS = 600
