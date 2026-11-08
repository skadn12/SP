"""Create a non-destructive source manifest for the current route baseline."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .preflight import ROOT, SPELL, _sha256


OUT = SPELL / "route_integrity/reports/ROUTE_BASELINE_MANIFEST.json"

# The manifest includes the two canonical routes, their shared QA core, the
# experimental v19 sibling (to prevent contract leakage), and shared runtime
# modules imported by those routes. Generated reports and caches are excluded.
SOURCE_ROOTS = (
    SPELL / "simple_append_memorax_v1",
    SPELL / "simple_append_v1",
    SPELL / "simple_append_memorax_v1_memory_event_continuation_probe_v1",
    SPELL / "simple_append_memorax_v1_fullr1_devdiv_qa_structural_v19_probe_v1",
    SPELL / "route_integrity",
)
SHARED_FILES = (
    SPELL / "core.py",
    SPELL / "ledger.py",
    SPELL / "live_backends.py",
    SPELL / "minimal_admission.py",
    SPELL / "questioner_contract.py",
)


def _source_files() -> list[Path]:
    files: set[Path] = set(path for path in SHARED_FILES if path.is_file())
    for root in SOURCE_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if "reports" in path.parts or path.suffix not in {".py", ".md"}:
                continue
            files.add(path)
    return sorted(files)


def _status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _status_sha(lines: Iterable[str]) -> str:
    value = "\n".join(lines) + "\n"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def freeze() -> dict[str, object]:
    paths = _source_files()
    status = _status()
    manifest = {
        "manifest_version": "route_baseline_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "working_tree_status_sha256": _status_sha(status),
        "working_tree_status_count": len(status),
        "working_tree_was_dirty": bool(status),
        "source_files": {
            str(path.relative_to(ROOT)): _sha256(path) for path in paths
        },
        "source_file_count": len(paths),
        "excluded_from_source_manifest": [
            "run outputs", "reports", "__pycache__", "model weights",
            "API credentials", "historical run roots",
        ],
        "api_calls": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2, sort_keys=True))
