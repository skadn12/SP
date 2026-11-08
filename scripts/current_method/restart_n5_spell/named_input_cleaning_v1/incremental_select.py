"""Select the next unseen Conversation Chronicles roots in source order.

This is an offline selection boundary, before name mapping and cleaning.  It
reads the source JSONL and prior manifests, writes only a small selection
manifest (and an optional mapping template), and never modifies source or
cleaned-root files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..core import atomic_json, sha256_file
from ..conversation_chronicles_adapter import validate_cc_source_record
from .batch_clean import _root_id, _source_speakers, record_format


SELECTION_VERSION = "named_input_sequential_selection_v1"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"manifest_json_invalid:{path}:{type(error).__name__}") from error


def _root_from_identity(value: Any) -> str | None:
    identity = str(value or "").strip()
    if not identity:
        return None
    if identity.startswith("cc-train-"):
        return identity
    suffix = re.sub(r"[^a-z0-9]+", "-", identity.casefold()).strip("-")
    return f"cc-train-{suffix}" if suffix else None


def load_excluded_root_ids(manifest_paths: Sequence[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    """Load root IDs from cleaning, audit, selection, or batch manifests."""

    excluded: set[str] = set()
    sources: list[dict[str, Any]] = []
    for raw_path in manifest_paths:
        path = Path(raw_path).resolve()
        value = _load_json(path)
        if not isinstance(value, Mapping):
            raise ValueError(f"excluded_manifest_not_object:{path}")
        rows = value.get("records")
        if not isinstance(rows, list):
            rows = value.get("items")
        if not isinstance(rows, list):
            # The earlier 10-record cleaning route calls its root inventory
            # ``roots`` rather than ``records``.
            rows = value.get("roots")
        found: set[str] = set()
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                root = _root_from_identity(row.get("root_id"))
                if root is None:
                    root = _root_from_identity(row.get("source_id"))
                if root is not None:
                    found.add(root)
        for raw_id in value.get("root_ids") or []:
            root = _root_from_identity(raw_id)
            if root is not None:
                found.add(root)
        sources.append({
            "path": str(path), "sha256": sha256_file(path),
            "root_count": len(found),
        })
        excluded.update(found)
    return excluded, sources


def _source_rows(path: Path, start_row: int) -> Iterable[dict[str, Any]]:
    if path.suffix.casefold() != ".jsonl":
        raise ValueError("sequential_selection_requires_jsonl")
    with path.open("rb") as handle:
        for row_index, raw_line in enumerate(handle):
            if row_index < start_row or not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                yield {"row_index": row_index, "skip": "invalid_json"}
                continue
            if not isinstance(value, Mapping):
                yield {"row_index": row_index, "skip": "record_not_object"}
                continue
            try:
                fmt = record_format(value)
                root_id = _root_id(value, fmt)
                speakers = _source_speakers(value, fmt)
            except (ValueError, KeyError) as error:
                yield {
                    "row_index": row_index,
                    "skip": f"record_contract:{type(error).__name__}:{error}",
                }
                continue
            # Validate the complete Conversation Chronicles source contract
            # before spending a mapping-model call.  The cleaner/adapter would
            # reject the same record later (for example an empty dialogue
            # string), so selecting it would make an otherwise valid batch
            # fail only after partial materialization.
            if fmt == "conversation_chronicles_record":
                source_check = validate_cc_source_record(value)
                if not source_check["valid"]:
                    yield {
                        "row_index": int(row_index),
                        "skip": "source_contract:" + ",".join(
                            source_check["errors"]),
                    }
                    continue
            yield {
                "row_index": row_index,
                "raw": raw_line,
                "value": value,
                "format": fmt,
                "root_id": root_id,
                "source_id": str(value.get("dataID") or "").strip(),
                "speakers": speakers,
            }


def _cursor_start(cursor_manifest: Path, source_path: Path, source_sha256: str) -> int:
    value = _load_json(cursor_manifest.resolve())
    if not isinstance(value, Mapping):
        raise ValueError("cursor_manifest_not_object")
    if value.get("selection_version") != SELECTION_VERSION:
        raise ValueError("cursor_manifest_version_invalid")
    if Path(str(value.get("source_path") or "")).resolve() != source_path.resolve():
        raise ValueError("cursor_source_path_mismatch")
    if str(value.get("source_sha256") or "") != source_sha256:
        raise ValueError("cursor_source_sha256_mismatch")
    cursor = value.get("next_cursor_row_index")
    if not isinstance(cursor, int) or cursor < 0:
        raise ValueError("cursor_value_invalid")
    return cursor


def select_next(
    *,
    input_path: Path,
    count: int,
    exclude_manifests: Sequence[Path] = (),
    start_row: int = 0,
    cursor_manifest: Path | None = None,
) -> dict[str, Any]:
    """Select exactly ``count`` unseen roots, preserving source order."""

    source_path = input_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if count <= 0:
        raise ValueError("selection_count_must_be_positive")
    if start_row < 0:
        raise ValueError("selection_start_row_invalid")
    source_sha256 = sha256_file(source_path)
    if cursor_manifest is not None:
        if start_row:
            raise ValueError("cursor_and_start_row_are_mutually_exclusive")
        start_row = _cursor_start(cursor_manifest, source_path, source_sha256)
    excluded, exclusion_sources = load_excluded_root_ids(exclude_manifests)
    selected: list[dict[str, Any]] = []
    seen_in_scan: set[str] = set()
    skip_counts: dict[str, int] = {}
    last_scanned = start_row - 1
    for row in _source_rows(source_path, start_row):
        last_scanned = int(row["row_index"])
        reason = row.get("skip")
        if reason:
            key = str(reason).split(":", 1)[0]
            skip_counts[key] = skip_counts.get(key, 0) + 1
            continue
        root_id = str(row["root_id"])
        if root_id in excluded:
            skip_counts["excluded_root"] = skip_counts.get("excluded_root", 0) + 1
            continue
        if root_id in seen_in_scan:
            skip_counts["duplicate_source_root"] = skip_counts.get(
                "duplicate_source_root", 0) + 1
            continue
        seen_in_scan.add(root_id)
        value = row["value"]
        source_id = str(row.get("source_id") or "").strip()
        selected.append({
            "root_id": root_id,
            "source_id": source_id,
            "source_row_index": int(row["row_index"]),
            "source_record_sha256": hashlib.sha256(row["raw"]).hexdigest(),
            "source_speakers": list(row["speakers"]),
            "relationship": value.get("relationship"),
            "format": row["format"],
            "mapping_key": source_id or root_id,
            "mapping_template": {
                str(speaker): "" for speaker in row["speakers"]
            },
        })
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"selection_insufficient:{len(selected)}/{count}:"
            f"start_row={start_row}:last_scanned={last_scanned}"
        )
    next_cursor = int(selected[-1]["source_row_index"]) + 1
    return {
        "selection_version": SELECTION_VERSION,
        "status": "passed",
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "requested_count": count,
        "selected_count": len(selected),
        "scan_start_row_index": start_row,
        "scan_end_row_index": last_scanned,
        "next_cursor_row_index": next_cursor,
        "excluded_root_count": len(excluded),
        "excluded_manifests": exclusion_sources,
        "records": selected,
        "skip_counts": skip_counts,
        "mapping_stage": {
            "required": True,
            "model_assistance_allowed": True,
            "mapping_template_values_are_blank": True,
            "template_is_not_cleaner_ready": True,
        },
        "pipeline_integration": {
            "api_calls": 0, "cleaning_calls": 0, "planner_calls": 0,
            "realizer_calls": 0, "qa_writer_calls": 0, "solver_calls": 0,
        },
    }


def write_selection(path: Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"selection_output_exists:{output}")
    atomic_json(output, dict(payload))
    return output


def write_mapping_template(path: Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"mapping_template_exists:{output}")
    mappings: dict[str, Any] = {}
    for row in payload.get("records") or []:
        mappings[str(row["mapping_key"])] = {
            "source_row_index": row["source_row_index"],
            "speaker_map": dict(row["mapping_template"]),
        }
    atomic_json(output, {
        "schema_version": "named_input_cleaning_mapping_template_v1",
        "status": "template_only",
        "source_path": payload["source_path"],
        "source_sha256": payload["source_sha256"],
        "selection_manifest": payload.get("selection_version"),
        "mappings": mappings,
        "warning": "Fill unique concrete names and validate before batch_clean.",
    })
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--cursor-manifest", type=Path)
    parser.add_argument("--mapping-template-output", type=Path)
    args = parser.parse_args(argv)
    payload = select_next(
        input_path=args.input, count=args.count,
        exclude_manifests=args.exclude_manifest,
        start_row=args.start_row, cursor_manifest=args.cursor_manifest,
    )
    output = write_selection(args.output, payload)
    template = None
    if args.mapping_template_output is not None:
        template = write_mapping_template(args.mapping_template_output, payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_count": payload["selected_count"],
        "scan_start_row_index": payload["scan_start_row_index"],
        "next_cursor_row_index": payload["next_cursor_row_index"],
        "output": str(output),
        "mapping_template": str(template) if template else None,
        "api_calls": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
