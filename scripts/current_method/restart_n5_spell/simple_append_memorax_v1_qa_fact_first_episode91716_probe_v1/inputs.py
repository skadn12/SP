"""Frozen input paths for the episode-91716 QA-only probe."""

from __future__ import annotations

from pathlib import Path

from ..simple_append_memorax_v1_qa_fact_first_probe_v1.integrity import sha256_file


EXPANSION_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_91716_event_bucket_only_fallback_expansion_probe_2"
)
DEFAULT_EXPANDED = EXPANSION_RUN_ROOT / "expanded_dialogue.json"
DEFAULT_MEMORY_OVERVIEW = EXPANSION_RUN_ROOT / "memory/post_generation_overview.json"
TARGET_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_91716_qa_fact_first_probe_v1_live_1"
)
SUBJECT_BINDING_TARGET_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_91716_qa_subject_binding_probe_v1_live_1"
)
# This is the file SHA. The JSON's embedded complete_dialogue_sha256 is a
# separate content identity and is recorded from the frozen input at runtime.
EXPECTED_EXPANDED_SHA256 = "a9663db7cff6c55683db739b445d0712d886d2ba2c938ebb7bca61459a9e9c15"


def frozen_input_manifest() -> dict[str, object]:
    return {
        "expanded_path": str(DEFAULT_EXPANDED),
        "expanded_sha256": sha256_file(DEFAULT_EXPANDED)
        if DEFAULT_EXPANDED.is_file() else None,
        "expanded_expected_sha256": EXPECTED_EXPANDED_SHA256,
        "expanded_unchanged": DEFAULT_EXPANDED.is_file()
        and sha256_file(DEFAULT_EXPANDED) == EXPECTED_EXPANDED_SHA256,
        "memory_overview_path": str(DEFAULT_MEMORY_OVERVIEW),
        "memory_overview_sha256": sha256_file(DEFAULT_MEMORY_OVERVIEW)
        if DEFAULT_MEMORY_OVERVIEW.is_file() else None,
        "target_run_root": str(TARGET_RUN_ROOT),
        "target_run_absent": not TARGET_RUN_ROOT.exists(),
        "subject_binding_target_run_root": str(SUBJECT_BINDING_TARGET_RUN_ROOT),
        "subject_binding_target_run_absent": not SUBJECT_BINDING_TARGET_RUN_ROOT.exists(),
    }
