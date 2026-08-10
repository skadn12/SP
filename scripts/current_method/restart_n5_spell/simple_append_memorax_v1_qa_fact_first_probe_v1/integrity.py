"""Read-only SHA manifest helpers for the fact-first sibling."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_manifest(root: Path) -> dict[str, Any]:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    rows = [{
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    } for path in files]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"  ./")
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\n")
    return {
        "root": str(root.resolve()),
        "file_count": len(rows),
        "tree_sha256": digest.hexdigest(),
        "files": rows,
    }


def git_status(repo: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=repo,
        check=True, capture_output=True, text=True,
    )
    return result.stdout

