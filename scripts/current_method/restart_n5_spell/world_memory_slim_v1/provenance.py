"""Generic provenance helpers isolated from legacy generation orchestration."""

from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path
from typing import Any

PROVENANCE_VERSION = "world_memory_slim_provenance_v1"
BANNED_LEGACY_IMPORTS = frozenset({
    "stateful_graph_expansion",
    "stateful_outline_planner",
    "stateful_graph_orchestrator",
    "dynamic_generation_canary",
})


def git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def project_diff_sha256(root: Path) -> str:
    """Hash tracked diff plus the bodies of all untracked project files."""
    material = bytearray(git_value(root, "diff", "--binary", "HEAD").encode())
    names = git_value(root, "ls-files", "--others", "--exclude-standard").splitlines()
    for relative in sorted(value for value in names if value):
        path = root / relative
        if path.is_file():
            material.extend(relative.encode() + b"\0" + path.read_bytes() + b"\0")
    return hashlib.sha256(material).hexdigest()


def project_diff_sha256_excluding(
    root: Path, excluded_relatives: set[str] | frozenset[str],
) -> str:
    """Hash the project diff while excluding deterministic generated reports."""
    excluded = {Path(value).as_posix() for value in excluded_relatives}
    pathspecs = [".", *(f":(exclude){value}" for value in sorted(excluded))]
    material = bytearray(
        git_value(root, "diff", "--binary", "HEAD", "--", *pathspecs).encode())
    names = git_value(root, "ls-files", "--others", "--exclude-standard").splitlines()
    for relative in sorted(value for value in names
                           if value and Path(value).as_posix() not in excluded):
        path = root / relative
        if path.is_file():
            material.extend(relative.encode() + b"\0" + path.read_bytes() + b"\0")
    return hashlib.sha256(material).hexdigest()


def ensure_fresh_output(path: Path) -> None:
    """Apply a no-resume boundary without writing into historical run roots."""
    if path.exists():
        raise FileExistsError(f"fresh_output_already_exists:{path}")


def hash_tree(root: Path) -> dict[str, Any]:
    """Hash a tree as sorted path/size/content-hash records."""
    root = root.resolve()
    digest = hashlib.sha256()
    file_count = total_bytes = 0
    for path in sorted(candidate for candidate in root.rglob("*")
                       if candidate.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        file_digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(block)
                size += len(block)
        digest.update(relative + b"\0" + str(size).encode() + b"\0" +
                      file_digest.hexdigest().encode() + b"\0")
        file_count += 1
        total_bytes += size
    return {
        "root": str(root),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
        "tree_sha256_scope": "sorted(relative_path\\0size\\0file_sha256\\0)",
    }


def hash_tree_excluding(root: Path, excluded: Path) -> dict[str, Any]:
    """Hash pre-existing history while excluding the current new run root."""
    root = root.resolve()
    excluded = excluded.resolve()
    digest = hashlib.sha256()
    file_count = total_bytes = 0
    for path in sorted(candidate for candidate in root.rglob("*")
                       if candidate.is_file() and
                       excluded != candidate and excluded not in candidate.parents):
        relative = path.relative_to(root).as_posix().encode()
        file_digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(block)
                size += len(block)
        digest.update(relative + b"\0" + str(size).encode() + b"\0" +
                      file_digest.hexdigest().encode() + b"\0")
        file_count += 1
        total_bytes += size
    return {
        "root": str(root), "excluded": str(excluded),
        "file_count": file_count, "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
        "tree_sha256_scope": "sorted(relative_path\\0size\\0file_sha256\\0)",
    }


def audit_legacy_import_isolation(package_root: Path) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    for path in sorted(package_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                leaf = module.rsplit(".", 1)[-1]
                if leaf in BANNED_LEGACY_IMPORTS:
                    findings.append({"file": path.name, "module": module})
    return {
        "audit_version": "world_memory_slim_legacy_import_audit_v1",
        "package_root": str(package_root.resolve()),
        "banned_modules": sorted(BANNED_LEGACY_IMPORTS),
        "findings": findings,
        "isolated": not findings,
    }


__all__ = [
    "PROVENANCE_VERSION", "audit_legacy_import_isolation", "ensure_fresh_output",
    "git_value", "hash_tree", "hash_tree_excluding", "project_diff_sha256",
    "project_diff_sha256_excluding",
]
