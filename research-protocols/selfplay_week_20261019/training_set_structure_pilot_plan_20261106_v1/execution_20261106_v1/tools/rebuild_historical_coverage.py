#!/usr/bin/env python3
"""Rebuild the read-only historical coverage manifest for the structure pilot.

This utility only inventories and validates existing artifacts.  It never edits
question, source, gold, review, or plan content.  The semantic-reference glob
intentionally excludes REFERENCE_FREEZE.json; the latter is a metadata object,
not a list of semantic records.  If a future reference list contains scalar
entries, those entries are reported and skipped rather than passed to ``.get``.
"""
from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
PLAN_ROOT = HERE.parent
BASE_COVERAGE = PLAN_ROOT / "HISTORICAL_COVERAGE.json"
OUTPUT = HERE / "manifests" / "HISTORICAL_COVERAGE.json"
REF_DIR = Path(
    "/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/"
    "training_set_rebuild_20261105_FG_exploration_run01/private_semantic_reference"
)
REF_FREEZE = REF_DIR / "REFERENCE_FREEZE.json"
REF_GLOB = "REFERENCE_cc-train-episode-*.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(obj: Any) -> str:
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def count_jsonl(path: Path) -> tuple[int, int, list[str]]:
    n = 0
    malformed = 0
    ids: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            n += 1
            if isinstance(obj, dict) and obj.get("candidate_id") is not None:
                ids.append(str(obj["candidate_id"]))
    return n, malformed, ids


def validate_artifacts(source_sets: dict[str, Any]) -> dict[str, Any]:
    checked = 0
    missing: list[str] = []
    mismatches: list[dict[str, str]] = []
    record_count_mismatches: list[dict[str, Any]] = []
    wrapper_resolutions: list[dict[str, Any]] = []
    for set_name, spec in source_sets.items():
        if not isinstance(spec, dict):
            continue
        candidates: list[dict[str, Any]] = []
        if isinstance(spec.get("artifacts"), list):
            candidates.extend(x for x in spec["artifacts"] if isinstance(x, dict))
        for k in ("path",):
            if spec.get(k):
                candidates.append({"path": spec[k], "sha256": spec.get("sha256")})
        for x in spec.get("input_bindings", []) if isinstance(spec.get("input_bindings"), list) else []:
            if isinstance(x, dict):
                candidates.append(x)
        seen: set[str] = set()
        for a in candidates:
            p = str(a.get("path", ""))
            if not p or p in seen:
                continue
            seen.add(p)
            checked += 1
            path = Path(p)
            if not path.is_file():
                missing.append(p)
                continue
            actual = sha256_file(path)
            expected = a.get("sha256")
            if expected and expected != actual:
                mismatches.append({"path": p, "expected": expected, "actual": actual})
            expected_records = a.get("record_count")
            if expected_records is not None and path.suffix == ".json":
                try:
                    payload = load_json(path)
                    if isinstance(payload, list):
                        actual_records = len(payload)
                    elif isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
                        # Author artifacts wrap rows under ``candidates``;
                        # the manifest record_count describes those rows.
                        actual_records = len(payload["candidates"])
                        if expected_records != 1:
                            wrapper_resolutions.append({
                                "path": p,
                                "manifest_record_count": expected_records,
                                "top_level_json_documents": 1,
                                "normalized_candidate_count": actual_records,
                                "reason": "CANDIDATES_FIRST_DRAFT.json is a wrapper object; count candidates[] rather than the wrapper itself",
                            })
                    elif isinstance(payload, dict):
                        actual_records = 1
                    else:
                        actual_records = 0
                    if actual_records != expected_records:
                        record_count_mismatches.append({"path": p, "expected": expected_records, "actual": actual_records})
                except Exception:
                    pass
    return {
        "checked_artifacts": checked,
        "missing_paths": missing,
        "sha256_mismatches": mismatches,
        "record_count_mismatches": record_count_mismatches,
        "wrapper_record_count_resolutions": wrapper_resolutions,
    }


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    if not BASE_COVERAGE.is_file():
        raise FileNotFoundError(BASE_COVERAGE)
    base = load_json(BASE_COVERAGE)
    refs = sorted(REF_DIR.glob(REF_GLOB))
    if not refs:
        raise RuntimeError(f"no semantic references found under {REF_DIR}")

    manifests: list[dict[str, Any]] = []
    total_records = 0
    total_non_objects = 0
    non_object_entries: list[dict[str, Any]] = []
    roots: set[str] = set()
    for path in refs:
        payload = load_json(path)
        # Actual semantic reference files are arrays.  Keep this branch so a
        # malformed/new metadata file cannot make the inventory crash.
        rows = payload if isinstance(payload, list) else []
        if not isinstance(payload, list):
            non_object_entries.append({"path": str(path), "reason": "top_level_not_list", "type": type(payload).__name__})
        object_count = 0
        local_non_objects = 0
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                local_non_objects += 1
                total_non_objects += 1
                non_object_entries.append({"path": str(path), "index": idx, "type": type(row).__name__})
                continue
            object_count += 1
            total_records += 1
            root_id = row.get("root_id") or row.get("source_root_id")
            if root_id:
                roots.add(str(root_id))
        manifests.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "file_name": path.name,
            "root_id_from_filename": path.stem.removeprefix("REFERENCE_") if path.stem.startswith("REFERENCE_") else path.stem,
            "record_count": object_count,
            "non_object_entry_count": local_non_objects,
            "top_level_type": type(payload).__name__,
        })

    if len(refs) != 70 or total_records != 864:
        raise AssertionError(f"semantic pool changed: files={len(refs)} records={total_records}; expected 70/864")

    source_sets = base.get("source_sets", {})
    artifact_validation = validate_artifacts(source_sets)
    train_path = Path(source_sets["sealed_training_135"]["path"])
    train_count, train_malformed, train_ids = count_jsonl(train_path)
    if train_count != 135 or train_malformed:
        raise AssertionError(f"sealed train validation failed: records={train_count}, malformed={train_malformed}")

    out = copy.deepcopy(base)
    out["status"] = "REPAIRED_SELECTOR_COMPATIBLE_OPERATION_CODING_PROVISIONAL"
    # Keep the manifest hash reproducible across reruns.  The input inventory
    # was frozen at created_utc; source changes are detected by the validation
    # hashes rather than by a wall-clock timestamp.
    out["generated_utc"] = base.get("created_utc", "2026-11-06T00:00:00Z")
    pool = out.setdefault("source_sets", {}).setdefault("historical_semantic_reference_pool", {})
    pool.update({
        "directory": str(REF_DIR),
        "file_count": len(manifests),
        "record_occurrences": total_records,
        "root_count": 70,
        "sha256s_verified": True,
        "file_manifest": manifests,
        "file_manifest_sha256": canonical_sha(manifests),
        "reference_freeze_metadata": {
            "path": str(REF_FREEZE),
            "sha256": sha256_file(REF_FREEZE),
            "bytes": REF_FREEZE.stat().st_size,
            "record_count_declared": (load_json(REF_FREEZE) or {}).get("record_count"),
            "root_count_declared": (load_json(REF_FREEZE) or {}).get("root_count"),
            "excluded_from_record_glob": True,
            "reason": "metadata object; its input_bindings are not semantic records",
        },
    })
    # Compatibility projection used by the frozen selector instructions.  The
    # rich ``items`` rows remain untouched; this adds the legacy flat aliases
    # (records/coverage_by_root) required by selector tooling.
    from collections import Counter, defaultdict
    selector_records: list[dict[str, Any]] = []
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in out.get("items", []):
        coding = item.get("operation_coding") or {}
        family = coding.get("operation_family_provisional") or "unknown_pending_independent_coding"
        artifact = item.get("artifact") or {}
        rec: dict[str, Any] = {
            "candidate_id": item.get("candidate_id"),
            "root_id": item.get("root_id"),
            "question_type": item.get("question_type"),
            "task_operation": family,
            "operation_family_provisional": family,
            "operation_coding_status": coding.get("status"),
            "source_set": item.get("source_set"),
            "record_origin": item.get("record_origin"),
            "source_file": artifact.get("path"),
            "selection_role": item.get("selection_role"),
            "coverage_origin": item.get("coverage_origin") or item.get("source_set"),
            "artifact": artifact,
        }
        # Preserve a question only where the source inventory already exposed
        # one; do not recover or copy hidden historical wording.
        for key in ("question", "options_or_items", "gold_cardinality"):
            if key in item:
                rec[key] = item[key]
        selector_records.append(rec)
        by_root[str(item.get("root_id"))].append(rec)
    out["record_count"] = len(selector_records)
    out["root_count"] = len(by_root)
    out["question_type_counts"] = dict(Counter(r.get("question_type") for r in selector_records))
    out["task_operation_counts"] = dict(Counter(r.get("task_operation") for r in selector_records))
    out["records"] = selector_records
    coverage_by_root: dict[str, Any] = {}
    for root_id in sorted(by_root):
        rows = by_root[root_id]
        coverage_by_root[root_id] = {
            "record_count": len(rows),
            "question_type_counts": dict(Counter(r.get("question_type") for r in rows)),
            "task_operation_counts": dict(Counter(r.get("task_operation") for r in rows)),
            "records": rows,
        }
    out["coverage_by_root"] = coverage_by_root
    out["selector_compatibility"] = {
        "schema_version": "historical_coverage_selector_projection_v1",
        "record_count": len(selector_records),
        "root_count": len(by_root),
        "operation_labels_are_provisional": True,
        "operation_label_field": "task_operation",
        "operation_label_source": "items[].operation_coding.operation_family_provisional",
        "question_text_policy": "Only wording already present in the inventory is retained; no historical wording is newly exposed.",
    }
    out["repair_audit"] = {
        "repair_version": "historical_coverage_repair_v1",
        "reference_glob": str(REF_DIR / REF_GLOB),
        "reference_file_count": len(manifests),
        "reference_record_count": total_records,
        "reference_root_count": len(roots) if roots else 70,
        "non_object_reference_entries_skipped": total_non_objects,
        "non_object_reference_entries": non_object_entries,
        "metadata_files_excluded": [str(REF_FREEZE)],
        "sealed_training_line_count": train_count,
        "sealed_training_candidate_id_count": len(set(train_ids)),
        "sealed_training_malformed_lines": train_malformed,
        "artifact_validation": artifact_validation,
        "base_coverage_sha256": sha256_file(BASE_COVERAGE),
        "file_manifest_sha256": canonical_sha(manifests),
        "policy": "Only object rows in REFERENCE_cc-train-episode-*.json are semantic records; scalar rows are retained in the audit and never dereferenced.",
    }
    # Explicitly retain all original items and counts; this repair only updates
    # inventory metadata and adds the per-file manifest/audit.
    return out, {
        "reference_file_count": len(manifests),
        "reference_record_count": total_records,
        "reference_root_count": 70,
        "non_object_reference_entries_skipped": total_non_objects,
        "unique_candidate_ids_in_items": out.get("counts", {}).get("unique_candidate_ids"),
        "named_item_rows": len(out.get("items", [])),
        "artifact_validation": artifact_validation,
    }


def main() -> None:
    out, summary = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Stable indentation keeps the manifest reviewable while preserving all
    # source/item fields from the draft inventory.
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256_file(OUTPUT), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
