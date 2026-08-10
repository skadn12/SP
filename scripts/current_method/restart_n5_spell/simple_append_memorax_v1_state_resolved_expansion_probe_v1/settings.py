"""Frozen paths, budgets, and integrity helpers for the isolated probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..core import sha256_file


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots/cc-train-episode-146700/unified_root.json"
)
FROZEN_MEMORY_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_simple_append_memorax_"
    "fullr1_durable_devdiv_local_anchor_span_probe_v1_live_1"
)
DEFAULT_OVERVIEW = FROZEN_MEMORY_RUN / "memory/pre_generation_overview.json"
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_simple_append_memorax_"
    "state_resolved_expansion_probe_v1_live_1"
)
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_REPORT_ROOT = MODULE_DIR / "reports"

FROZEN_INPUT_SHA256 = "a7199f528b7382e716098bbee7b635f5e43c787cb0624ad96a57cbd3ba689818"
FROZEN_OVERVIEW_SHA256 = "35fb79d782d422e3f18602d8b73fca54bd4e1afd6737977f5ef1ebf030da5865"
RESOLVER_MAX_TOKENS = 4096
PLANNER_MAX_TOKENS = 4096
ROUND_REALIZER_MAX_TOKENS = 8192
CONTEXT_LIMIT = 40_960
LIVE_TIMEOUT_SECONDS = 600

PROTECTED_SOURCE_DIRS = (
    REPO_ROOT / "scripts/current_method/restart_n5_spell/simple_append_memorax_v1",
    REPO_ROOT / "scripts/current_method/restart_n5_spell/"
    "simple_append_memorax_v1_fullr1_durable_devdiv_probe_v1",
    REPO_ROOT / "scripts/current_method/restart_n5_spell/"
    "simple_append_memorax_v1_state_resolution_probe_v1",
    REPO_ROOT / "scripts/current_method/restart_n5_spell/"
    "simple_append_memorax_v1_qa_fact_first_probe_v1",
)


def tree_sha256(root: Path, *, ignore_caches: bool = True) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if ignore_caches and (
            "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        digest.update(
            f"{sha256_file(path)}  {relative.as_posix()}\n".encode("utf-8")
        )
    return digest.hexdigest()


def source_manifest() -> dict[str, Any]:
    simple_append = REPO_ROOT / "scripts/current_method/restart_n5_spell/simple_append_v1"
    baseline = (
        REPO_ROOT / "scripts/current_method/restart_n5_spell/"
        "simple_append_memorax_v1_fullr1_durable_devdiv_probe_v1"
    )
    shared = REPO_ROOT / "scripts/current_method/restart_n5_spell"
    files = list(MODULE_DIR.glob("*.py"))
    files.extend(simple_append.glob("*.py"))
    files.extend((baseline / name) for name in ("context.py", "memory_context.py"))
    files.extend((shared / name) for name in (
        "backends.py", "core.py", "ledger.py", "live_backends.py",
        "unified_dialogue_root.py",
    ))
    unique = sorted({path.resolve() for path in files if path.is_file()})
    rows = [{"path": str(path), "sha256": sha256_file(path)} for path in unique]
    payload = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {"files": rows, "manifest_sha256": hashlib.sha256(payload).hexdigest()}


def protected_integrity_snapshot() -> dict[str, Any]:
    return {
        "input_sha256": sha256_file(DEFAULT_INPUT),
        "overview_sha256": sha256_file(DEFAULT_OVERVIEW),
        "frozen_memory_run_tree_sha256": tree_sha256(
            FROZEN_MEMORY_RUN, ignore_caches=False),
        "protected_source_manifests": {
            str(path): tree_sha256(path) for path in PROTECTED_SOURCE_DIRS
        },
    }


__all__ = [
    "CONTEXT_LIMIT", "DEFAULT_API_CONFIG", "DEFAULT_INPUT", "DEFAULT_OVERVIEW",
    "DEFAULT_REPORT_ROOT", "DEFAULT_RUN_ROOT", "DEFAULT_TOKENIZER",
    "FROZEN_INPUT_SHA256", "FROZEN_MEMORY_RUN", "FROZEN_OVERVIEW_SHA256",
    "LIVE_TIMEOUT_SECONDS", "MODULE_DIR", "PLANNER_MAX_TOKENS",
    "PROTECTED_SOURCE_DIRS", "REPO_ROOT", "RESOLVER_MAX_TOKENS",
    "ROUND_REALIZER_MAX_TOKENS", "protected_integrity_snapshot",
    "source_manifest", "tree_sha256",
]
