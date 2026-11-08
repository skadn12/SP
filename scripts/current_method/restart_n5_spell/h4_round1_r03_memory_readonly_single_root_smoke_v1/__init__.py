"""Single-root local-only R03 smoke using a frozen R02 Memory snapshot."""

from pathlib import Path


VERSION = "h4_round1_r03_memory_readonly_single_root_smoke_v1"
RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PREFLIGHT_RUN = RUNS / "h4_round1_r03_memory_readonly_trial_preflight_v2_live_3"
COMPILER_RUN = RUNS / "h4_round1_question_bank_feedback_compiler_v1_live_13"
DEFAULT_OUTPUT = RUNS / f"{VERSION}_live_1"
CONFIG_PATH = Path("/root/autodl-tmp/gxy-selfplay/config.json")
TOKENIZER_PATH = Path("/root/autodl-tmp/Qwen3-4B")

ROOT_ID = "cc-train-episode-122711"
MODEL = "Qwen/Qwen3-14B"
TEMPERATURE = 0.25
CONTEXT_CEILING = 40_960
PLANNER_MAX_TOKENS = 4_096
REALIZER_MAX_TOKENS = 8_192

