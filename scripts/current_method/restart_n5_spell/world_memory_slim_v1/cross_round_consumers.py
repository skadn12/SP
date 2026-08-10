"""V7 cross-round consumers are hard-wired to the promotable child surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .promotion import load_promotable_child


def load_round_parent(previous_run_root: Path) -> dict[str, Any]:
    return load_promotable_child(previous_run_root)


def load_lineage_surface(previous_run_root: Path) -> dict[str, Any]:
    return load_promotable_child(previous_run_root)


def load_resume_surface(previous_run_root: Path) -> dict[str, Any]:
    return load_promotable_child(previous_run_root)


def load_report_surface(previous_run_root: Path) -> dict[str, Any]:
    return load_promotable_child(previous_run_root)
