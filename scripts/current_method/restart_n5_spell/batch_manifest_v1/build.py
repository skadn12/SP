"""Build a deduplicated batch manifest without copying root data.

Cleaning outputs are immutable data stores.  This module creates a small
selection manifest that references their existing ``unified_root.json`` files
and records their hashes.  It never calls an API and never writes into a
cleaning output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, canonical_json
from ..unified_dialogue_root import validate_unified_dialogue_root


SELECTION_MANIFEST_VERSION = "h4_root_selection_manifest_v1"
CLEANING_MANIFEST_VERSION = "named_input_cleaning_batch_manifest_v1"
SUPPORTED_SOURCE_MANIFEST_VERSIONS = {
    CLEANING_MANIFEST_VERSION,
    SELECTION_MANIFEST_VERSION,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"json_load_failed:{path}:{type(error).__name__}") from error


def _resolve_path(raw: Any, *, base: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _source_records(manifest_path: Path) -> list[dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    value = _load_json(manifest_path)
    if not isinstance(value, Mapping):
        raise ValueError(f"source_manifest_not_object:{manifest_path}")
    manifest_version = value.get("manifest_version")
    if manifest_version not in SUPPORTED_SOURCE_MANIFEST_VERSIONS:
        raise ValueError(
            f"source_manifest_version_invalid:{manifest_path}:"
            f"{manifest_version}"
        )
    if value.get("status") != "passed":
        raise ValueError(f"source_manifest_not_passed:{manifest_path}")
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"source_manifest_records_missing:{manifest_path}")

    result: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"source_record_not_object:{manifest_path}:{index}")
        root_id = str(record.get("root_id") or "")
        # Cleaning manifests use output_path; selection manifests preserve the
        # same reference under input_path.  Both are immutable source paths.
        raw_path = record.get("output_path") or record.get("input_path")
        if not root_id or not raw_path:
            raise ValueError(f"source_record_identity_missing:{manifest_path}:{index}")
        path = _resolve_path(raw_path, base=manifest_path.parent)
        if not path.is_file():
            raise FileNotFoundError(f"source_root_missing:{root_id}:{path}")
        actual_sha = sha256_file(path)
        expected_sha = str(
            record.get("output_sha256") or record.get("input_sha256") or "")
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(f"source_root_hash_drift:{root_id}:{path}")
        root = _load_json(path)
        if not isinstance(root, Mapping):
            raise ValueError(f"source_root_not_object:{root_id}:{path}")
        if str(root.get("root_id") or "") != root_id:
            raise ValueError(f"source_root_identity_drift:{root_id}:{path}")
        validation = validate_unified_dialogue_root(root)
        if not validation["valid"]:
            raise ValueError(
                f"source_root_invalid:{root_id}:{','.join(validation['errors'])}"
            )
        dialogue = root.get("dialogue") or {}
        result.append({
            "root_id": root_id,
            "input_path": str(path),
            "input_sha256": actual_sha,
            "parent_dialogue_sha256": str(dialogue.get("dialogue_hash") or ""),
            "turn_count": int(dialogue.get("turn_count") or 0),
            "session_count": int(dialogue.get("session_count") or 0),
            "speaker_names": sorted({
                str(turn.get("speaker") or "")
                for turn in (dialogue.get("turns") or [])
                if isinstance(turn, Mapping)
            }),
            "source_manifest_path": str(manifest_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_record_sha256": record.get("source_record_sha256"),
            "source_row_index": record.get("source_row_index"),
            "cleaning_report_path": str(path.with_name("name_cleaning_report.json")),
            "validation": validation,
        })
    return result


def build_selection_manifest(
    source_manifests: Sequence[Path],
    *,
    expected_count: int | None = None,
    allow_identical_duplicates: bool = True,
    root_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a unique-root manifest referencing existing files only.

    Identical references to the same root may occur when one newer cleaning
    manifest is cumulative.  They are collapsed.  A root ID with two
    different paths or hashes is rejected as version ambiguity.
    """

    if not source_manifests:
        raise ValueError("source_manifests_empty")
    seen: dict[str, dict[str, Any]] = {}
    duplicate_sources: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for raw_manifest in source_manifests:
        manifest_path = Path(raw_manifest).resolve()
        rows = _source_records(manifest_path)
        source_rows.append({
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "record_count": len(rows),
        })
        for row in rows:
            root_id = str(row["root_id"])
            previous = seen.get(root_id)
            if previous is None:
                seen[root_id] = row
                continue
            same_reference = (
                previous["input_path"] == row["input_path"]
                and previous["input_sha256"] == row["input_sha256"]
            )
            if not same_reference or not allow_identical_duplicates:
                raise ValueError(
                    f"root_id_conflict:{root_id}:"
                    f"{previous['input_path']}:{row['input_path']}"
                )
            duplicate_sources.append({
                "root_id": root_id,
                "kept_path": previous["input_path"],
                "duplicate_path": row["input_path"],
                "duplicate_manifest": row["source_manifest_path"],
            })

    requested_root_ids = {str(root_id) for root_id in (root_ids or [])}
    unknown_root_ids = sorted(requested_root_ids - set(seen))
    if unknown_root_ids:
        raise ValueError("selection_root_ids_missing:" + ",".join(unknown_root_ids))
    selected_ids = requested_root_ids or set(seen)
    records = [seen[root_id] for root_id in sorted(selected_ids)]
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"selection_count_mismatch:{len(records)}/{expected_count}")
    if not records:
        raise ValueError("selection_records_empty")
    return {
        "manifest_version": SELECTION_MANIFEST_VERSION,
        "status": "passed",
        "record_count": len(records),
        "expected_count": expected_count,
        "selection_root_ids": sorted(selected_ids),
        "records": records,
        "roots_directory": "reference_only",
        "source_manifests": source_rows,
        "duplicate_sources_collapsed": duplicate_sources,
        "global_checks": {
            "all_records_valid": True,
            "root_ids_unique": len(records) == len({row["root_id"] for row in records}),
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


def write_selection_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    """Atomically write only the new selection manifest."""

    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"selection_manifest_exists:{output}")
    atomic_json(output, dict(manifest))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--root-id", action="append", default=[],
        help="Restrict the selection to these root IDs; repeat for multiple roots.",
    )
    args = parser.parse_args(argv)
    manifest = build_selection_manifest(
        args.source_manifest, expected_count=args.expected_count,
        root_ids=args.root_id,
    )
    output = write_selection_manifest(args.output, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "manifest_version": manifest["manifest_version"],
        "record_count": manifest["record_count"],
        "output": str(output),
        "duplicate_sources_collapsed": len(manifest["duplicate_sources_collapsed"]),
        "api_calls": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
