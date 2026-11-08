"""Full-pool B0 K=3 difficulty screening on the frozen QA-only batch."""

from pathlib import Path
from typing import Mapping, Any


PROBE_VERSION = "h4_round0_b0_k3_difficulty_v1"
RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_SOURCE_RUN = RUN_BASE / "h4_unexpanded122_qa_generation_only_v1"
DEFAULT_RUN_ROOT = RUN_BASE / "h4_round0_b0_k3_difficulty_v1_live_1"
BASE_SOLVER_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
BASE_TOKENIZER = BASE_SOLVER_MODEL
FORBIDDEN_SOLVER_MODEL = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "05_stage2_repair_topup_from_stage1/training/models/merged"
)

EXPECTED_SOURCE_ROOTS = 102
EXPECTED_READY_ROOTS = 101
EXPECTED_CANDIDATES = 3_286
INCOMPLETE_ROOT_ID = "cc-train-episode-122126"
EXPECTED_QUESTION_TYPES = {
    "single_choice": 1_652,
    "multi_select": 820,
    "ordering": 814,
}

ROOT_SPLIT_COUNTS = {"train": 70, "dev": 15, "test": 16}
ROOT_SPLIT_VERSION = "h4_round0_root_split_unexposed_test_v1"
ROOT_SPLIT_NAMESPACE = "gxy-selfplay:h4-round0-b0-k3-difficulty-v1"

ROLLOUTS = 3
SINGLE_TEMPERATURE = 0.7
STRUCTURED_TEMPERATURE = 0.8
TOP_P = 0.9
MAX_TOKENS = 32
MAX_MODEL_LEN = 16_384
SEED_SCHEDULE_VERSION = "h4_k3_paired_seed_v1"
PARSER_SINGLE = "boxed_letter_v1"
PARSER_STRUCTURED = "structured_qa_box_v1"
STRUCTURED_OUTPUT_VERSION = "finite_choice_set_v1"

QUESTION_KEYS = {
    "single_choice": ("A", "B", "C", "D"),
    "multi_select": ("A", "B", "C", "D", "E"),
    "ordering": ("I1", "I2", "I3", "I4"),
}
REQUESTS_PER_CANDIDATE = {
    "single_choice": 24,
    "multi_select": 30,
    "ordering": 24,
}


def question_keys(question_type: str, count: int) -> tuple[str, ...]:
    """Return canonical keys for the active 4-6-item QA contract.

    QUESTION_KEYS and REQUESTS_PER_CANDIDATE remain frozen B0 metadata. Runtime
    callers that can receive R03 v1.1 candidates must use these dynamic helpers.
    """
    if question_type == "single_choice":
        return ("A", "B", "C", "D") if count == 4 else ()
    if question_type == "multi_select" and 4 <= count <= 6:
        return tuple(chr(ord("A") + index) for index in range(count))
    if question_type == "ordering" and 4 <= count <= 6:
        return tuple(f"I{index + 1}" for index in range(count))
    return ()


def candidate_position_count(candidate: Mapping[str, Any]) -> int:
    question_type = str(candidate.get("question_type") or "")
    field = "items" if question_type == "ordering" else "options"
    value = candidate.get(field)
    return len(value) if isinstance(value, Mapping) else 0


def candidate_request_count(candidate: Mapping[str, Any]) -> int:
    return candidate_position_count(candidate) * 2 * ROLLOUTS

# Only these historical artifact families define analysis exposure. Additional
# manual-review artifacts must be passed explicitly with --exposure-artifact.
DEFAULT_EXPOSURE_GLOBS = (
    "h4_position_permutation_ab_*/sample_manifest.jsonl",
    "h4_qa_multiroot_k3_t08_semantic_audit_v1_live_*/frozen/candidates.json",
    "h4_qa_multiroot_k3_t08_blind_semantic_recheck_v1_live_*/frozen/candidates.json",
    "conversation_chronicles_episode_*_qa_*/*/STRUCTURAL_QA_*_MANUAL_AUDIT.json",
)


__all__ = ["PROBE_VERSION"]
