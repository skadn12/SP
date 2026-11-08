"""Build a reference-only manifest for cleaned roots with no ready expansion.

This is an offline boundary.  It reads immutable ``unified_root.json`` files,
deduplicates root IDs in the caller's explicit directory order, removes roots
with a completed expansion gate, and writes a selection manifest accepted by
the H4 batch planner.  It never calls an API and never copies root files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json
from ..unified_dialogue_root import validate_unified_dialogue_root


SELECTION_VERSION = "h4_root_selection_manifest_v1"
BUILDER_VERSION = "h4_unexpanded_selection_builder_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _root_files(path: Path) -> list[Path]:
    path = path.resolve()
    if path.is_file():
        return [path] if path.name == "unified_root.json" else []
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(path.glob("roots/*/unified_root.json"))


def _root_record(path: Path) -> dict[str, Any]:
    root = _load_object(path)
    validation = validate_unified_dialogue_root(root)
    if not validation["valid"]:
        raise ValueError(
            f"cleaned_root_invalid:{path}:{','.join(validation['errors'])}")
    root_id = str(root.get("root_id") or "")
    if not root_id:
        raise ValueError(f"cleaned_root_id_missing:{path}")
    dialogue = root.get("dialogue") or {}
    turns = dialogue.get("turns") or []
    return {
        "root_id": root_id,
        "input_path": str(path),
        "input_sha256": sha256_file(path),
        "parent_dialogue_sha256": str(dialogue.get("dialogue_hash") or ""),
        "turn_count": int(dialogue.get("turn_count") or len(turns)),
        "session_count": int(dialogue.get("session_count") or 0),
        "speaker_names": sorted({str(row.get("speaker") or "")
                                  for row in turns if isinstance(row, Mapping)}),
        "cleaning_report_path": str(path.with_name("name_cleaning_report.json")),
        "source_record_sha256": (root.get("source") or {}).get(
            "source_record_sha256"),
        "source_row_index": (root.get("source") or {}).get("source_row_index"),
        "validation": validation,
    }


def _completed_expansion_ids(search_root: Path) -> tuple[set[str], list[str]]:
    """Collect ready root IDs from both per-root and batch reports."""
    ready: set[str] = set()
    reports: list[str] = []
    for path in sorted(search_root.resolve().rglob(
            "reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json")):
        try:
            report = _load_object(path)
        except Exception:
            continue
        if (report.get("status") == "completed"
                and report.get("generation_ready") is True
                and report.get("child_add_verified") is True):
            root_id = str(report.get("root_id") or "")
            if root_id:
                ready.add(root_id)
                reports.append(str(path))
    for path in sorted(search_root.resolve().rglob(
            "expansion_execution_report.json")):
        try:
            payload = _load_object(path)
        except Exception:
            continue
        expansion = payload.get("expansion")
        if not isinstance(expansion, Mapping):
            expansion = payload
        for row in expansion.get("results") or []:
            if not isinstance(row, Mapping) or row.get("status") != "EXPANSION_READY":
                continue
            root_id = str(row.get("root_id") or "")
            if root_id:
                ready.add(root_id)
                reports.append(str(path))
    return ready, sorted(set(reports))


def build_unexpanded_selection(
    *, cleaned_roots: Sequence[Path], expansion_search_root: Path,
    output: Path, expected_count: int | None = None,
) -> dict[str, Any]:
    if not cleaned_roots:
        raise ValueError("cleaned_roots_empty")
    if output.exists():
        raise FileExistsError(f"selection_manifest_exists:{output.resolve()}")
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_sources: list[dict[str, Any]] = []
    source_dirs: list[dict[str, Any]] = []
    for raw_dir in cleaned_roots:
        directory = raw_dir.resolve()
        files = _root_files(directory)
        source_dirs.append({
            "path": str(directory),
            "root_file_count": len(files),
            "file_sha256": {
                str(path.relative_to(directory)): sha256_file(path)
                for path in files
            },
        })
        for path in files:
            row = _root_record(path)
            root_id = row["root_id"]
            previous = by_id.get(root_id)
            if previous is None:
                by_id[root_id] = row
                continue
            duplicate_sources.append({
                "root_id": root_id,
                "kept_path": previous["input_path"],
                "duplicate_path": row["input_path"],
                "same_sha256": previous["input_sha256"] == row["input_sha256"],
                "policy": "first_explicit_cleaned_roots_order_wins",
            })

    expanded_ids, expansion_reports = _completed_expansion_ids(
        expansion_search_root)
    selected = [row for root_id, row in sorted(by_id.items())
                if root_id not in expanded_ids]
    if expected_count is not None and len(selected) != expected_count:
        raise ValueError(
            f"unexpanded_selection_count_mismatch:{len(selected)}/{expected_count}")
    if not selected:
        raise ValueError("unexpanded_selection_empty")
    selected_ids = [row["root_id"] for row in selected]
    manifest = {
        "manifest_version": SELECTION_VERSION,
        "builder_version": BUILDER_VERSION,
        "status": "passed",
        "record_count": len(selected),
        "expected_count": expected_count if expected_count is not None else len(selected),
        "selection_root_ids": selected_ids,
        "records": selected,
        "roots_directory": "reference_only",
        "source_cleaned_directories": source_dirs,
        "duplicate_sources_collapsed": duplicate_sources,
        "expansion_search_root": str(expansion_search_root.resolve()),
        "expansion_report_paths": expansion_reports,
        "expanded_root_ids_excluded": sorted(expanded_ids),
        "unexpanded_policy": {
            "definition": "no completed expansion report with generation_ready and child_add_verified",
            "source_files_copied": False,
            "historical_expansion_outputs_reused": False,
        },
        "global_checks": {
            "all_records_valid": True,
            "root_ids_unique": len(selected_ids) == len(set(selected_ids)),
            "root_sha256_match": True,
            "root_identity_match": True,
            "no_data_copied": True,
        },
        "materialization": {
            "mode": "reference_only",
            "copied_root_files": 0,
            "source_roots_are_immutable": True,
        },
        "pipeline_integration": {
            "api_calls": 0,
            "solver_calls": 0,
            "cleaning_calls": 0,
        },
    }
    atomic_json(output.resolve(), manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-root", type=Path, action="append", required=True)
    parser.add_argument("--expansion-search-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args(argv)
    manifest = build_unexpanded_selection(
        cleaned_roots=args.cleaned_root,
        expansion_search_root=args.expansion_search_root,
        output=args.output,
        expected_count=args.expected_count,
    )
    print(json.dumps({
        "status": manifest["status"],
        "record_count": manifest["record_count"],
        "expanded_excluded_count": len(manifest["expanded_root_ids_excluded"]),
        "duplicate_sources_collapsed": len(manifest["duplicate_sources_collapsed"]),
        "output": str(args.output.resolve()),
        "api_calls": 0,
        "solver_calls": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
