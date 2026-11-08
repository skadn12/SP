"""Offline, manifest-driven name-cleaning batch for expansion inputs.

The batch boundary is intentionally before Planner/Realizer.  The command
reads a frozen Conversation Chronicles JSONL, emits named unified roots, and
never imports or calls an expansion, QA, Memorax, or Solver entrypoint.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..conversation_chronicles_adapter import (
    CONVERSATION_CHRONICLES_ADAPTER_VERSION,
    build_unified_root,
    validate_cc_source_record,
)
from ..core import atomic_json, atomic_text, canonical_json, sha256_file
from ..unified_dialogue_root import validate_unified_dialogue_root
from .batch_clean import (
    CLEANING_VERSION,
    clean_conversation_chronicles_record,
    validate_cleaned_record,
)


DEFAULT_MANIFEST = Path(__file__).with_name("audit10_alias_manifest.json")
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-cleaned/"
    "named_batch_v1_audit10_alias_default")


def _read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("cleaning_manifest_must_be_object")
    source_path = value.get("source_path")
    rows = value.get("records")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("cleaning_manifest_source_path_missing")
    if not isinstance(rows, list) or not rows:
        raise ValueError("cleaning_manifest_records_missing")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("cleaning_manifest_record_invalid")
        source_id = str(row.get("source_id") or "").strip()
        names = row.get("speaker_map")
        if not source_id or source_id in seen:
            raise ValueError(f"cleaning_manifest_source_id_invalid:{source_id}")
        if not isinstance(names, Mapping) or len(names) != 2:
            raise ValueError(f"cleaning_manifest_speaker_map_invalid:{source_id}")
        seen.add(source_id)
    return dict(value)


def _load_selected_records(
    source_path: Path, source_ids: set[str],
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Read the source once and retain raw-line provenance for selected rows."""
    import hashlib

    digest = hashlib.sha256()
    found: dict[str, dict[str, Any]] = {}
    with source_path.open("rb") as handle:
        for row_index, raw_line in enumerate(handle):
            digest.update(raw_line)
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            source_id = str(record.get("dataID") or "").strip()
            if source_id not in source_ids:
                continue
            if source_id in found:
                raise ValueError(f"source_id_duplicate:{source_id}")
            found[source_id] = {
                "record": record,
                "source_row_index": row_index,
                "source_record_sha256": hashlib.sha256(raw_line).hexdigest(),
            }
    missing = sorted(source_ids - set(found))
    if missing:
        raise FileNotFoundError("source_ids_missing:" + ",".join(missing))
    return digest.hexdigest(), found


def _root_with_cleaning_metadata(
    root: Mapping[str, Any], *, canonical_names: list[str],
    source_record_sha256: str,
) -> dict[str, Any]:
    """Add only new-name metadata inside the permissive source_metadata field."""
    result = copy.deepcopy(dict(root))
    metadata = dict(result.get("source_metadata") or {})
    metadata.update({
        "name_cleaning_version": CLEANING_VERSION,
        "name_cleaning_canonical_names": list(canonical_names),
        "name_cleaning_source_record_sha256": source_record_sha256,
    })
    result["source_metadata"] = metadata
    return result


def _stable_payload(value: Any) -> str:
    return canonical_json(value)


def _render_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Named Input Cleaning Batch v1 (10-record stability audit)", "",
        f"- status: **{report['status']}**",
        f"- source: `{report['source_path']}`",
        f"- source SHA256: `{report['source_file_sha256']}`",
        f"- source unchanged: `{str(report['source_unchanged']).lower()}`",
        f"- records: `{report['record_count']}`",
        f"- passed: `{report['passed_count']}`",
        f"- failed: `{report['failed_count']}`",
        f"- API calls: `{report['api_calls']}`", "",
        "| Source ID | Relation | Turns | Repeated output equal | Valid |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["records"]:
        lines.append(
            f"| `{row['source_id']}` | {row.get('relationship', '')} | "
            f"{row.get('turn_count', '')} | "
            f"`{str(row.get('repeat_equal')).lower()}` | "
            f"`{str(row.get('valid')).lower()}` |")
    lines.extend([
        "", "## Contract", "",
        "- Cleaning is deterministic and manifest-driven.",
        "- Source JSONL is read-only; no historical run is modified.",
        "- Full aliases and explicit A-and-B pairs are replaced; isolated short A/B tokens are replaced only as direct addresses.",
        "- Unresolved aliases fail the record instead of being guessed.",
        "- The emitted `roots/*/unified_root.json` files are the only inputs intended for the expansion flow.",
        "- No Planner, Realizer, QA Writer, Memorax, Solver, or external API call was made.", "",
    ])
    return "\n".join(lines)


def run_batch(*, manifest_path: Path, output_root: Path) -> dict[str, Any]:
    manifest = _read_manifest(manifest_path.resolve())
    source_path = Path(str(manifest["source_path"])).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source_missing:{source_path}")
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output_root_exists:{output_root}")

    manifest_rows = [dict(row) for row in manifest["records"]]
    source_ids = {str(row["source_id"]).strip() for row in manifest_rows}
    source_file_sha256, selected = _load_selected_records(source_path, source_ids)
    records: list[dict[str, Any]] = []
    output_root.mkdir(parents=True)

    for row in manifest_rows:
        source_id = str(row["source_id"]).strip()
        speaker_map = {str(old): str(new) for old, new in
                       dict(row["speaker_map"]).items()}
        item = selected[source_id]
        original = item["record"]
        source_check = validate_cc_source_record(original)
        if not source_check["valid"]:
            raise ValueError(f"source_record_invalid:{source_id}:"
                             + ",".join(source_check["errors"]))

        cleaned, counts, clean_meta = clean_conversation_chronicles_record(
            original, speaker_map)
        validation = validate_cleaned_record(
            original, cleaned, speaker_map,
            fmt="conversation_chronicles_record")

        # Repeat the pure transformation in a fresh object.  This catches
        # accidental stateful counters, unordered substitutions, and mapping
        # mutation before any output is admitted.
        repeated, repeated_counts, _ = clean_conversation_chronicles_record(
            original, speaker_map)
        repeat_equal = _stable_payload(cleaned) == _stable_payload(repeated)
        repeat_counts_equal = counts == repeated_counts

        root = build_unified_root(
            cleaned, source_path=source_path,
            source_file_sha256=source_file_sha256,
            source_row_index=int(item["source_row_index"]),
            source_record_sha256=str(item["source_record_sha256"]),
        )
        root = _root_with_cleaning_metadata(
            root, canonical_names=clean_meta["canonical_names"],
            source_record_sha256=str(item["source_record_sha256"]),
        )
        root_check = validate_unified_dialogue_root(
            root, expected_source_path=source_path,
            expected_source_file_sha256=source_file_sha256,
            expected_session_count=5)
        valid = bool(validation["valid"] and repeat_equal and
                     repeat_counts_equal and root_check["valid"])

        root_dir = output_root / "roots" / str(root["root_id"])
        root_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(root_dir / "cleaned_record.json", cleaned)
        atomic_json(root_dir / "unified_root.json", root)
        row_report = {
            "source_id": source_id,
            "root_id": root["root_id"],
            "relationship": cleaned.get("relationship"),
            "source_row_index": item["source_row_index"],
            "source_record_sha256": item["source_record_sha256"],
            "source_file_sha256": source_file_sha256,
            "source_path": str(source_path),
            "speaker_map": speaker_map,
            "canonical_names": clean_meta["canonical_names"],
            "replacement_counts": counts,
            "turn_count": len(root["dialogue"]["turns"]),
            "clean_validation": validation,
            "root_validation": root_check,
            "repeat_equal": repeat_equal,
            "repeat_counts_equal": repeat_counts_equal,
            "valid": valid,
            "output_path": str((root_dir / "unified_root.json").resolve()),
        }
        atomic_json(root_dir / "cleaning_report.json", row_report)
        records.append(row_report)

    # Re-hash the immutable source boundary after every output is written.
    # A source mutation must fail the batch even if every derived root passed.
    source_file_sha256_after = sha256_file(source_path)
    source_unchanged = source_file_sha256 == source_file_sha256_after
    passed = sum(bool(row["valid"]) for row in records)
    report: dict[str, Any] = {
        "schema_version": "named_input_cleaning_batch_report_v1",
        "cleaning_version": CLEANING_VERSION,
        "adapter_version": CONVERSATION_CHRONICLES_ADAPTER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.resolve()),
        "source_path": str(source_path),
        "source_file_sha256": source_file_sha256,
        "source_file_sha256_after": source_file_sha256_after,
        "source_unchanged": source_unchanged,
        "record_count": len(records),
        "passed_count": passed,
        "failed_count": len(records) - passed,
        "status": ("passed" if passed == len(records) and source_unchanged
                   else "failed"),
        "api_calls": 0,
        "planner_calls": 0,
        "realizer_calls": 0,
        "qa_writer_calls": 0,
        "memorax_calls": 0,
        "solver_calls": 0,
        "records": records,
        "output_root": str(output_root),
    }
    output_manifest = {
        "schema_version": "named_input_cleaning_batch_manifest_v1",
        "cleaning_version": CLEANING_VERSION,
        "source_path": str(source_path),
        "source_file_sha256": source_file_sha256,
        "source_file_sha256_after": source_file_sha256_after,
        "source_unchanged": source_unchanged,
        "record_count": len(records),
        "status": report["status"],
        "roots": [{key: row[key] for key in (
            "source_id", "root_id", "source_row_index", "turn_count",
            "output_path", "source_record_sha256")}
                  for row in records],
        "expansion_input_glob": str(output_root / "roots" / "*" /
                                    "unified_root.json"),
        "cleaning_is_not_in_expansion_mainline": True,
    }
    atomic_json(output_root / "cleaning_manifest.json", output_manifest)
    atomic_json(output_root / "reports/batch_report.json", report)
    atomic_text(output_root / "reports/batch_report.md", _render_report(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run_batch(manifest_path=args.manifest,
                       output_root=args.output_root)
    print(json.dumps({key: report[key] for key in (
        "status", "record_count", "passed_count", "failed_count",
        "api_calls", "output_root")}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
