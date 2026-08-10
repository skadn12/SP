"""Frozen input paths and read helpers for the position-only branch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


INPUT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_structural_"
    "v19_2_budget16384_probe_v1_live_1")
DEFAULT_EXPANDED = INPUT_RUN_ROOT / "frozen/expanded_dialogue.json"
DEFAULT_MEMORY_OVERVIEW = INPUT_RUN_ROOT / "frozen/memory_overview.json"
EXPECTED_EXPANDED_SHA256 = (
    "97cf7b5aef3b9e6b7a04e9323f015d7d3cbef6f82fc985da97b89cbc70f0f7f5")
EXPECTED_MEMORY_OVERVIEW_SHA256 = (
    "0a494b2495b77cf1e4d54fd006cef45dee5e905d8debe637f786902fecdbb962")


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]
