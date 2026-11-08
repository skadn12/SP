"""Round-0 matched B0 LoRA training smoke gate."""

from pathlib import Path


SMOKE_VERSION = "h4_round0_b0_training_smoke_v1"
RECIPE_VERSION = "qwen3_4b_lora_sft_verified_recipe_v1"
SMOKE_MANIFEST_VERSION = "h4_round0_b0_training_smoke_manifest_v1"
LOSS_MASK_AUDIT_VERSION = "h4_round0_b0_training_loss_mask_audit_v1"
RELOAD_TEST_VERSION = "h4_round0_b0_adapter_reload_test_v1"
REQUIRED_MAX_LENGTH = 8192
SMOKE_COUNT = 16
SHARED_COUNT = 12
REPLACEMENT_COUNT = 4
SMOKE_EPOCHS = 1.0
EXPECTED_OPTIMIZER_STEPS = 4

REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DATA_RUN = RUN_BASE / "h4_round0_b0_matched_training_v1_live_2"
DEFAULT_RUN_ROOT = RUN_BASE / "h4_round0_b0_training_smoke_v1_live_1"
BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
TRAINER = REPO_ROOT / "train_qwen3_lora_sft.py"
SOURCE_RECIPE_REPORT = REPO_ROOT / "data/base_restart_current_method_validation/01_mining500_train/training/reports/training_summary.json"

__all__ = [
    "BASE_MODEL", "DATA_RUN", "DEFAULT_RUN_ROOT", "EXPECTED_OPTIMIZER_STEPS",
    "LOSS_MASK_AUDIT_VERSION", "RECIPE_VERSION", "RELOAD_TEST_VERSION",
    "REPO_ROOT", "REPLACEMENT_COUNT", "REQUIRED_MAX_LENGTH", "SHARED_COUNT",
    "SMOKE_COUNT", "SMOKE_EPOCHS", "SMOKE_MANIFEST_VERSION", "SMOKE_VERSION",
    "SOURCE_RECIPE_REPORT", "TRAINER",
]
