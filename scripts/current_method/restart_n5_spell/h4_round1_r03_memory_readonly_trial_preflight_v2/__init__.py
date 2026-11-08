"""R03 existing-Memory read-only dialogue trial preflight."""

from pathlib import Path


VERSION = "h4_round1_r03_memory_readonly_trial_preflight_v2"
RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
COMPILER_RUN = RUNS / "h4_round1_question_bank_feedback_compiler_v1_live_13"
PAIRED_QA_RUN = RUNS / "h4_round1_paired_blind_fresh_qa_v1_live_8"
DEFAULT_OUTPUT = RUNS / f"{VERSION}_live_1"
TOKENIZER_PATH = Path("/root/autodl-tmp/Qwen3-4B")
MEMORAX_CREDENTIALS = Path("/root/.config/memorax/credentials.env")

TRIAL_ROOTS = (
    "cc-train-episode-122711",
    "cc-train-episode-254852",
    "cc-train-episode-112812",
    "cc-train-episode-136202",
    "cc-train-episode-94289",
    "cc-train-episode-96462",
    "cc-train-episode-198480",
    "cc-train-episode-197635",
)
QUARANTINE_ROOTS = (
    "cc-train-episode-259192",
    "cc-train-episode-218519",
)

CONTEXT_CEILING = 40_960
OUTPUT_RESERVE = 8_192

