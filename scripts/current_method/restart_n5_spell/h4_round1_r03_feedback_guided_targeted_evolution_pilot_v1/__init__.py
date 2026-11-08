"""One-attempt R03 feedback-guided targeted-evolution pilot."""

from pathlib import Path


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
TEMPLATE_RUN = RUN_BASE / "h4_round1_r03_qa_feedback_template_v1_preflight_v1_live_4"
CLOSURE_RUN = RUN_BASE / "h4_round1_r03_local_batch_closure_v1_live_2"
API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
TOKENIZER_PATH = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_OUTPUT = RUN_BASE / "h4_round1_r03_feedback_guided_targeted_evolution_pilot_v1_live_1"
VERSION = "h4_round1_r03_feedback_guided_targeted_evolution_pilot_v1"
TEMPLATE_SHA256 = "ed5bc3a9bed801ba83709bf51ae409e4d7fca92ef5e49cc71766ecea64d3a9a0"
CONTEXT_LIMIT = 16384
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.1
ELIGIBLE_ROOTS = (
    "cc-train-episode-122711", "cc-train-episode-254852",
    "cc-train-episode-136202", "cc-train-episode-94289",
    "cc-train-episode-198480", "cc-train-episode-197635",
)
QUARANTINE_ROOT = "cc-train-episode-112812"
UNSAFE_ROOT = "cc-train-episode-96462"
