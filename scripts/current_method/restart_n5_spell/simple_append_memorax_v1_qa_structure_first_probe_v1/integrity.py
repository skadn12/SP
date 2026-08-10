"""Read-only integrity helpers for the structure-first QA probe."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


MODULE_NAME = (
    "simple_append_memorax_v1_qa_structure_first_probe_v1")
PROTECTED_RELATIVE_PATHS = (
    "scripts/current_method/restart_n5_spell/simple_append_memorax_v1",
    "scripts/current_method/restart_n5_spell/"
    "simple_append_memorax_v1_fullr1_devdiv_probe_v1",
    "scripts/current_method/restart_n5_spell/"
    "simple_append_memorax_v1_fullr1_devdiv_qa_structural_v19_probe_v1",
)
EXPECTED_TREE_SHA256 = {
    PROTECTED_RELATIVE_PATHS[0]:
        "4abc9c534ff28a1a8094906247b881f64449fb19d32a8f80868656ae343fd9d1",
    PROTECTED_RELATIVE_PATHS[1]:
        "bc058bdd2fccd06ee1c648aaca6a8ae001aedf81bc61b61ff63b0771db2ee144",
    PROTECTED_RELATIVE_PATHS[2]:
        "c16e8212b8b93a48b17d89f38117f51b5fa5d63e21806d56ab834bf6d309b3e7",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_manifest(root: Path) -> dict[str, Any]:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts)
    rows = [{
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha(path),
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


def protected_manifests(repo_root: Path) -> dict[str, Any]:
    manifests = {
        relative: tree_manifest(repo_root / relative)
        for relative in PROTECTED_RELATIVE_PATHS
    }
    errors = [
        f"{relative}:tree_sha256_mismatch"
        for relative, manifest in manifests.items()
        if manifest["tree_sha256"] != EXPECTED_TREE_SHA256[relative]
    ]
    return {
        "expected": dict(EXPECTED_TREE_SHA256),
        "manifests": manifests,
        "errors": errors,
        "unchanged": not errors,
    }


def initial_git_status(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=repo_root,
        check=True, capture_output=True, text=True)
    return "\n".join(
        line for line in result.stdout.splitlines()
        if MODULE_NAME not in line) + "\n"
