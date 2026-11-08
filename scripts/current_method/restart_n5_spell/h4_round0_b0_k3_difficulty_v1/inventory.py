"""Read-only source inventory and exposure-aware root split freezing."""

from __future__ import annotations

import glob
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core import sha256_file
from . import (
    DEFAULT_EXPOSURE_GLOBS,
    EXPECTED_CANDIDATES,
    EXPECTED_QUESTION_TYPES,
    EXPECTED_READY_ROOTS,
    EXPECTED_SOURCE_ROOTS,
    INCOMPLETE_ROOT_ID,
    ROOT_SPLIT_COUNTS,
    ROOT_SPLIT_NAMESPACE,
    ROOT_SPLIT_VERSION,
)
from .io_utils import load_json, load_jsonl, stable_hash_int


def _root_dirs(source_run: Path) -> list[Path]:
    return sorted(source_run.glob("shards/*/roots/*"))


def build_source_inventory(source_run: Path) -> dict[str, Any]:
    source_run = source_run.resolve()
    report_path = source_run / "generation_only_report.json"
    source_inventory_path = source_run / "source_inventory.json"
    if not report_path.is_file() or not source_inventory_path.is_file():
        raise RuntimeError("source_run_missing_generation_inventory")
    generation_report = load_json(report_path)
    upstream_inventory = load_json(source_inventory_path)
    roots: list[dict[str, Any]] = []
    candidate_ids: list[str] = []
    type_counts: Counter[str] = Counter()
    for root_dir in _root_dirs(source_run):
        root_id = root_dir.name
        pool_path = root_dir / "candidate_pool.json"
        dialogue_path = root_dir / "expanded_dialogue_clean.json"
        qa_report_path = root_dir / "qa_generation_report.json"
        source_metadata_path = root_dir / "source_metadata.json"
        required = (pool_path, dialogue_path, qa_report_path, source_metadata_path)
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"source_root_artifact_missing:{root_id}")
        pool = load_json(pool_path)
        qa_report = load_json(qa_report_path)
        dialogue = load_json(dialogue_path)
        if not isinstance(pool, list):
            raise RuntimeError(f"candidate_pool_not_list:{root_id}")
        if str(qa_report.get("root_id") or "") != root_id:
            raise RuntimeError(f"qa_report_root_mismatch:{root_id}")
        if str(dialogue.get("root_id") or "") != root_id:
            raise RuntimeError(f"dialogue_root_mismatch:{root_id}")
        counts = Counter(str(row.get("question_type") or "") for row in pool)
        type_counts.update(counts)
        candidate_ids.extend(str(row.get("candidate_id") or "") for row in pool)
        roots.append({
            "root_id": root_id,
            "shard": root_dir.parent.parent.name,
            "qa_status": str(qa_report.get("status") or ""),
            "exclusion_status": (
                "QA_INCOMPLETE_EXCLUDED" if root_id == INCOMPLETE_ROOT_ID else None
            ),
            "candidate_count": len(pool),
            "question_type_counts": dict(sorted(counts.items())),
            "turn_count": len(dialogue.get("turns") or []),
            "candidate_pool_path": str(pool_path.resolve()),
            "candidate_pool_sha256": sha256_file(pool_path),
            "dialogue_path": str(dialogue_path.resolve()),
            "dialogue_sha256": sha256_file(dialogue_path),
            "qa_generation_report_path": str(qa_report_path.resolve()),
            "qa_generation_report_sha256": sha256_file(qa_report_path),
            "source_metadata_path": str(source_metadata_path.resolve()),
            "source_metadata_sha256": sha256_file(source_metadata_path),
        })
    duplicates = sorted(key for key, count in Counter(candidate_ids).items() if count > 1)
    empty_ids = sum(not value for value in candidate_ids)
    incomplete = [row for row in roots if row["root_id"] == INCOMPLETE_ROOT_ID]
    bearing = [row for row in roots if row["candidate_count"] > 0]
    blockers: list[str] = []
    if len(roots) != EXPECTED_SOURCE_ROOTS:
        blockers.append(f"source_root_count:{len(roots)}:{EXPECTED_SOURCE_ROOTS}")
    if len(bearing) != EXPECTED_READY_ROOTS:
        blockers.append(f"candidate_bearing_root_count:{len(bearing)}:{EXPECTED_READY_ROOTS}")
    if len(candidate_ids) != EXPECTED_CANDIDATES:
        blockers.append(f"candidate_count:{len(candidate_ids)}:{EXPECTED_CANDIDATES}")
    if dict(type_counts) != EXPECTED_QUESTION_TYPES:
        blockers.append(f"question_type_counts:{dict(type_counts)}:{EXPECTED_QUESTION_TYPES}")
    if len(incomplete) != 1 or incomplete[0]["candidate_count"] != 0:
        blockers.append("incomplete_root_identity_or_candidate_count")
    if incomplete and incomplete[0]["qa_status"] != "incomplete":
        blockers.append(f"incomplete_root_status:{incomplete[0]['qa_status']}")
    if any(row["qa_status"] != "ready" for row in bearing):
        blockers.append("candidate_bearing_root_not_ready")
    if duplicates:
        blockers.append(f"duplicate_candidate_ids:{len(duplicates)}")
    if empty_ids:
        blockers.append(f"empty_candidate_ids:{empty_ids}")
    if int(generation_report.get("root_count", -1)) != EXPECTED_SOURCE_ROOTS:
        blockers.append("generation_report_root_count_mismatch")
    if int(generation_report.get("candidate_count", -1)) != EXPECTED_CANDIDATES:
        blockers.append("generation_report_candidate_count_mismatch")
    if generation_report.get("status") != "completed":
        blockers.append(f"generation_report_not_completed:{generation_report.get('status')}")
    if int(generation_report.get("solver_calls", -1)) != 0:
        blockers.append("generation_source_solver_calls_not_zero")
    if generation_report.get("difficulty_filter_applied") is not False:
        blockers.append("generation_source_difficulty_filter_applied")
    if generation_report.get("semantic_admission_applied") is not False:
        blockers.append("generation_source_semantic_admission_applied")
    if int(upstream_inventory.get("ready_count", -1)) != EXPECTED_SOURCE_ROOTS:
        blockers.append("upstream_inventory_ready_count_mismatch")
    return {
        "inventory_version": "h4_round0_source_inventory_v1",
        "source_run": str(source_run),
        "source_run_read_only": True,
        "source_generation_report": str(report_path.resolve()),
        "source_generation_report_sha256": sha256_file(report_path),
        "upstream_source_inventory": str(source_inventory_path.resolve()),
        "upstream_source_inventory_sha256": sha256_file(source_inventory_path),
        "source_root_count": len(roots),
        "candidate_bearing_root_count": len(bearing),
        "candidate_count": len(candidate_ids),
        "question_type_counts": dict(sorted(type_counts.items())),
        "incomplete_root_id": INCOMPLETE_ROOT_ID,
        "duplicate_candidate_ids": duplicates,
        "roots": sorted(roots, key=lambda row: row["root_id"]),
        "blockers": blockers,
        "status": "BLOCKED" if blockers else "READY",
        "api_calls": 0,
        "solver_calls": 0,
    }


def _walk_root_ids(value: Any) -> set[str]:
    roots: set[str] = set()
    if isinstance(value, Mapping):
        root_id = value.get("root_id")
        if isinstance(root_id, str) and root_id:
            roots.add(root_id)
        candidate_id = value.get("candidate_id")
        if isinstance(candidate_id, str) and ":" in candidate_id:
            roots.add(candidate_id.split(":", 1)[0])
        for child in value.values():
            roots.update(_walk_root_ids(child))
    elif isinstance(value, list):
        for child in value:
            roots.update(_walk_root_ids(child))
    return roots


def discover_exposure_artifacts(
    run_base: Path, explicit: Iterable[Path] = (),
) -> list[Path]:
    paths: set[Path] = set()
    for pattern in DEFAULT_EXPOSURE_GLOBS:
        paths.update(Path(value).resolve() for value in glob.glob(str(run_base / pattern)))
    paths.update(Path(value).resolve() for value in explicit)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"exposure_artifact_missing:{missing}")
    return sorted(paths)


def build_exposure_inventory(paths: Iterable[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    union: set[str] = set()
    for path in sorted(Path(value).resolve() for value in paths):
        value = load_jsonl(path) if path.suffix == ".jsonl" else load_json(path)
        roots = sorted(_walk_root_ids(value))
        union.update(roots)
        rows.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "root_count": len(roots),
            "root_ids": roots,
        })
    return {
        "exposure_inventory_version": "h4_round0_analysis_exposure_v1",
        "artifact_count": len(rows),
        "artifacts": rows,
        "exposed_root_count": len(union),
        "exposed_root_ids": sorted(union),
    }


def freeze_root_split(
    inventory: Mapping[str, Any], exposure: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [
        dict(row) for row in inventory.get("roots") or []
        if int(row.get("candidate_count", 0)) > 0
    ]
    exposed = set(str(value) for value in exposure.get("exposed_root_ids") or [])
    unexposed = [row for row in eligible if row["root_id"] not in exposed]
    test_count = ROOT_SPLIT_COUNTS["test"]
    if len(unexposed) < test_count:
        raise RuntimeError(f"insufficient_unexposed_test_roots:{len(unexposed)}:{test_count}")
    test_rows = sorted(
        unexposed,
        key=lambda row: (
            stable_hash_int(ROOT_SPLIT_NAMESPACE, "test", row["root_id"]),
            row["root_id"],
        ),
    )[:test_count]
    test_ids = {row["root_id"] for row in test_rows}
    remaining = [row for row in eligible if row["root_id"] not in test_ids]
    dev_count = ROOT_SPLIT_COUNTS["dev"]
    dev_rows = sorted(
        remaining,
        key=lambda row: (
            stable_hash_int(ROOT_SPLIT_NAMESPACE, "dev", row["root_id"]),
            row["root_id"],
        ),
    )[:dev_count]
    dev_ids = {row["root_id"] for row in dev_rows}
    assignments: list[dict[str, Any]] = []
    for row in sorted(eligible, key=lambda value: value["root_id"]):
        root_id = str(row["root_id"])
        split = "test" if root_id in test_ids else "dev" if root_id in dev_ids else "train"
        assignments.append({
            "split_version": ROOT_SPLIT_VERSION,
            "split_namespace": ROOT_SPLIT_NAMESPACE,
            "root_id": root_id,
            "split": split,
            "analysis_exposed": root_id in exposed,
            "candidate_count": row["candidate_count"],
            "candidate_pool_sha256": row["candidate_pool_sha256"],
            "dialogue_sha256": row["dialogue_sha256"],
        })
    counts = Counter(row["split"] for row in assignments)
    if dict(counts) != ROOT_SPLIT_COUNTS:
        raise RuntimeError(f"root_split_count_mismatch:{dict(counts)}:{ROOT_SPLIT_COUNTS}")
    if any(row["analysis_exposed"] for row in assignments if row["split"] == "test"):
        raise RuntimeError("analysis_exposed_root_assigned_to_test")
    summary = {
        "split_version": ROOT_SPLIT_VERSION,
        "split_namespace": ROOT_SPLIT_NAMESPACE,
        "root_counts": dict(sorted(counts.items())),
        "candidate_counts": dict(sorted(Counter({
            split: sum(row["candidate_count"] for row in assignments if row["split"] == split)
            for split in ROOT_SPLIT_COUNTS
        }).items())),
        "analysis_exposed_root_count": len(exposed & {row["root_id"] for row in eligible}),
        "unexposed_eligible_root_count": len(unexposed),
        "test_selection": "stable_hash_from_unexposed_roots",
        "dev_selection": "stable_hash_from_remaining_roots",
        "train_selection": "all_remaining_roots",
    }
    return assignments, summary


__all__ = [
    "build_exposure_inventory",
    "build_source_inventory",
    "discover_exposure_artifacts",
    "freeze_root_split",
]
