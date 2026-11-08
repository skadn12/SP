#!/usr/bin/env python3
"""Read-only SHA-256 verification of explicitly supplied JSON manifests.

This checks dictionaries containing both path and sha256, not other binding
formats. It never imports experiment code or follows references to manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
CHUNK_SIZE = 1024 * 1024


def bindings(value: Any, recursive: bool = True) -> Iterable[dict[str, Any]]:
    """Yield binding dictionaries; nonrecursive accepts a root object/list."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value
        if recursive:
            for child in value.values():
                yield from bindings(child, True)
    elif isinstance(value, list):
        for child in value:
            if recursive:
                yield from bindings(child, True)
            elif isinstance(child, dict) and "path" in child and "sha256" in child:
                yield child


def parse_mapping(raw: str) -> tuple[Path, Path]:
    if "=" not in raw:
        raise ValueError("mapping must have OLD=NEW format")
    source, destination = raw.split("=", 1)
    if not source or not destination:
        raise ValueError("mapping endpoints must be nonempty")
    source_path, destination_path = Path(source), Path(destination)
    if not source_path.is_absolute() or not destination_path.is_absolute():
        raise ValueError("mapping endpoints must be absolute")
    if ".." in source_path.parts or ".." in destination_path.parts:
        raise ValueError("mapping endpoints cannot contain parent traversal")
    return source_path, destination_path


def resolve_binding(path: str, relative_base: Path | None, mappings: list[tuple[Path, Path]]) -> Path:
    if not path or "\x00" in path:
        raise ValueError("invalid_path")
    original = Path(path)
    if not original.is_absolute():
        if relative_base is None:
            raise ValueError("relative_base_required")
        if ".." in original.parts:
            raise ValueError("relative_path_escape")
        base = relative_base.resolve()
        resolved = (base / original).resolve()
        if not resolved.is_relative_to(base):
            raise ValueError("relative_path_escape")
        return resolved
    if ".." in original.parts:
        raise ValueError("absolute_path_parent_traversal")
    for old, new in mappings:
        if original.is_relative_to(old):
            return new / original.relative_to(old)
    return original


def digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def verify(manifests: list[Path], relative_base: Path | None = None,
           mappings: list[tuple[Path, Path]] | None = None,
           recursive: bool = True) -> dict[str, Any]:
    mappings = sorted(mappings or [], key=lambda pair: len(pair[0].parts), reverse=True)
    if len({str(old) for old, _ in mappings}) != len(mappings):
        raise ValueError("duplicate_mapping_source")
    issues: list[dict[str, str]] = []
    expected: dict[Path, tuple[str, int | None]] = {}
    conflicts: set[Path] = set()
    occurrence_count = 0
    for manifest in manifests:
        try:
            with manifest.open("r", encoding="utf-8") as stream:
                document = json.load(stream)
        except (OSError, ValueError, UnicodeError):
            issues.append({"path": str(manifest), "problem": "manifest_unreadable_or_invalid"})
            continue
        manifest_count = 0
        for record in bindings(document, recursive):
            manifest_count += 1
            occurrence_count += 1
            raw_path, raw_hash = record["path"], record["sha256"]
            shown_path = raw_path if isinstance(raw_path, str) else "<invalid-path>"
            if not isinstance(raw_path, str) or not isinstance(raw_hash, str) or not SHA256.fullmatch(raw_hash):
                issues.append({"path": shown_path, "problem": "invalid_binding"})
                continue
            size = record.get("size_bytes")
            if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
                issues.append({"path": shown_path, "problem": "invalid_size"})
                continue
            try:
                path = resolve_binding(raw_path, relative_base, mappings)
            except (OSError, RuntimeError, ValueError) as exc:
                problem = str(exc) if isinstance(exc, ValueError) else "path_resolution_error"
                issues.append({"path": shown_path, "problem": problem})
                continue
            previous = expected.get(path)
            normalized = (raw_hash.lower(), size)
            if previous:
                if previous[0] != normalized[0] or (previous[1] is not None and size is not None and previous[1] != size):
                    conflicts.add(path)
                elif previous[1] is None and size is not None:
                    expected[path] = normalized
            else:
                expected[path] = normalized
        if manifest_count == 0:
            issues.append({"path": str(manifest), "problem": "no_bindings_found"})
    matched = 0
    for path, (expected_hash, expected_size) in sorted(expected.items(), key=lambda item: str(item[0])):
        if path in conflicts:
            issues.append({"path": str(path), "problem": "conflicting_bindings"})
            continue
        try:
            if not path.exists():
                issues.append({"path": str(path), "problem": "missing"})
                continue
            if not path.is_file():
                issues.append({"path": str(path), "problem": "not_regular_file"})
                continue
            actual_hash, actual_size = digest_file(path)
        except OSError:
            issues.append({"path": str(path), "problem": "unreadable"})
            continue
        if actual_hash != expected_hash:
            issues.append({"path": str(path), "problem": "sha256_mismatch"})
        elif expected_size is not None and actual_size != expected_size:
            issues.append({"path": str(path), "problem": "size_mismatch"})
        else:
            matched += 1
    return {
        "status": "PASS" if not issues else "FAIL",
        "manifests": len(manifests),
        "binding_occurrences": occurrence_count,
        "unique_resolved_paths": len(expected),
        "matched_files": matched,
        "issues": issues,
        "scope": "Only explicitly supplied JSON manifests and dictionaries with path and sha256 were checked.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path, help="JSON manifests to inspect")
    parser.add_argument("--relative-base", type=Path, help="base directory for all relative binding paths")
    parser.add_argument("--map", action="append", default=[], metavar="OLD=NEW", help="map absolute path prefixes")
    parser.add_argument("--no-recursive", action="store_true", help="inspect only the root binding or root list")
    args = parser.parse_args(argv)
    try:
        mappings = [parse_mapping(item) for item in args.map]
        if args.relative_base is not None and not args.relative_base.is_absolute():
            raise ValueError("relative base must be absolute")
        report = verify(args.manifests, args.relative_base, mappings, not args.no_recursive)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
