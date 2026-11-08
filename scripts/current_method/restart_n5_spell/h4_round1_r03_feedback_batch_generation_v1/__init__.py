"""R03 batch QA generation under the human-authored feedback policy v2."""

from pathlib import Path

VERSION = "h4_round1_r03_feedback_batch_generation_v1"
HERE = Path(__file__).parent
RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
BASELINE = RUNS / "h4_round1_r03_multi_pass_qa_generation_comparison_pilot_v1_live_2"
COMMON = RUNS / "h4_round1_r03_qa_feedback_template_v1_preflight_v1_live_4"
BANK = RUNS / "h4_round1_question_bank_lifecycle_weight_freeze_v1_live_1"
ROUND1 = RUNS / "h4_round1_qa_feedback_generation_v2_live_1"
API_CONFIG = RUNS.parent / "config.json"
SLOTS = ("SUCCESSOR_FRONTIER", "COVERAGE_GAP", "FRESH_R03")
ROOTS = ("cc-train-episode-122711", "cc-train-episode-254852",
         "cc-train-episode-136202", "cc-train-episode-94289",
         "cc-train-episode-96462", "cc-train-episode-198480",
         "cc-train-episode-197635")
MODEL = "Qwen/Qwen3-14B"
TEMPERATURE = 0.1
INTENT_TOKENS = 4096
GENERATOR_TOKENS = 8192
MAX_CALLS_PER_ROOT = 10  # 1 miner + 6 generators + at most 3 missing-type repairs
