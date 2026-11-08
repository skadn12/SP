"""Small atomic checkpoint state machine for the isolated batch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core import atomic_json


STAGES = (
    "ROOT_STARTED",
    "EXPANSION_READY",
    "QA_READY",
    "QA_VALIDATED",
    "NO_CONTEXT_STARTED",
    "NO_CONTEXT_COMPLETED",
    "FULL_CONTEXT_STARTED",
    "FULL_CONTEXT_COMPLETED",
    "DIAGNOSTICS_COMPLETED",
    "REPORT_COMPLETED",
)


def write_checkpoint(path: Path, *, root_id: str, stage: str, **extra: Any) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"checkpoint_stage_invalid:{stage}")
    payload = {"checkpoint_version": "h4_k3_checkpoint_v1", "root_id": root_id,
               "stage": stage, **extra}
    atomic_json(path, payload)
    return payload


def read_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None
