"""Prepare ten Conversation Chronicles roots without invoking generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .conversation_chronicles_adapter import (
    CONVERSATION_CHRONICLES_ADAPTER_VERSION, DEFAULT_ROOT_COUNT,
    DEFAULT_SELECTION_SEED, prepare_cc_roots,
)
from .core import atomic_json, atomic_text, canonical_json, sha256_file
from .unified_dialogue_root import read_unified_parent


DEFAULT_SOURCE = Path(
    "/root/autodl-tmp/gxy-selfplay/data/conversation_chronicles/train.jsonl")
DEFAULT_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1")


def _audit_source_file(
    source_path: Path, roots: list[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {root["source"]["source_row_index"]:
                root["source"]["source_record_sha256"] for root in roots}
    found: dict[int, str] = {}
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for row_index, raw_line in enumerate(handle):
            digest.update(raw_line)
            if row_index in expected:
                found[row_index] = hashlib.sha256(raw_line).hexdigest()
    checks = {str(index): found.get(index) == value
              for index, value in sorted(expected.items())}
    return {"source_file_sha256": digest.hexdigest(),
            "source_record_sha256_checks": checks,
            "all_source_records_reverified": all(checks.values())}


def _root_reader_has_minimum_fields(parent: Any) -> bool:
    if not isinstance(parent, dict) or not all(
            key in parent for key in ("sample_id", "dialogue")):
        return False
    dialogue = parent.get("dialogue")
    if (not isinstance(dialogue, dict) or
            not isinstance(dialogue.get("dialogue_hash"), str) or
            not isinstance(dialogue.get("turns"), list) or
            not dialogue["turns"]):
        return False
    required = {"turn_id", "session_id", "speaker", "text", "partition"}
    return all(isinstance(turn, dict) and required <= set(turn)
               for turn in dialogue["turns"])


def _render_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Conversation Chronicles Adapter Validation", "",
        f"- adapter_ready: `{str(report['adapter_ready']).lower()}`",
        f"- schema_ready: `{str(report['schema_ready']).lower()}`",
        f"- root_reader_ready: `{str(report['root_reader_ready']).lower()}`",
        f"- generation_ready: `{str(report['generation_ready']).lower()}`",
        f"- source: `{report['source_path']}`",
        f"- source SHA256: `{report['source_file_sha256']}`",
        f"- selected roots: `{len(report['roots'])}`", "",
        "## Global Gates", "",
        "| Gate | Pass |", "|---|---:|",
    ]
    for name, passed in report["global_checks"].items():
        lines.append(f"| `{name}` | `{str(bool(passed)).lower()}` |")
    lines.extend(["", "## Roots", "",
                  "| Root | Source ID | Sessions | Turns | Schema | Reader |",
                  "|---|---|---:|---:|---:|---:|"])
    for row in report["roots"]:
        lines.append(
            f"| `{row['root_id']}` | `{row['source_id']}` | "
            f"{row['session_count']} | {row['turn_count']} | "
            f"`{str(row['schema_valid']).lower()}` | "
            f"`{str(row['root_reader_valid']).lower()}` |")
    lines.extend(["", "## Historical Snapshot Review", ""])
    for name, value in report["historical_snapshot_logic_review"].items():
        lines.append(f"- `{name}`: {value}")
    lines.extend(["", "## Remaining Integration Dependencies", ""])
    lines.extend(f"- {value}" for value in
                 report["remaining_dynamic_planner_hardcoded_dependencies"])
    lines.extend(["", "No Planner, Realizer, Questioner, Solver, verifier, "
                  "or training call was made.", ""])
    return "\n".join(lines)


def run(*, source_path: Path, output_root: Path,
        count: int = DEFAULT_ROOT_COUNT,
        seed: int = DEFAULT_SELECTION_SEED) -> dict[str, Any]:
    source_path = source_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output_root_exists:{output_root}")
    if not source_path.is_file():
        raise FileNotFoundError(f"cc_source_missing:{source_path}")
    prepared = prepare_cc_roots(source_path, count=count, seed=seed)
    output_root.mkdir(parents=True)
    root_records: list[dict[str, Any]] = []
    reader_checks: list[bool] = []
    serialization_checks: list[bool] = []
    for index, root in enumerate(prepared["roots"]):
        path = output_root / "roots" / root["root_id"] / "unified_root.json"
        atomic_json(path, root)
        reloaded = json.loads(path.read_text(encoding="utf-8"))
        serialization_valid = canonical_json(reloaded) == canonical_json(root)
        parent = read_unified_parent(path)
        reader_valid = _root_reader_has_minimum_fields(parent)
        serialization_checks.append(serialization_valid)
        reader_checks.append(reader_valid)
        schema_result = prepared["collection_validation"]["roots"][index]
        mapping = prepared["source_mapping_checks"][index]
        root_records.append({
            "root_id": root["root_id"],
            "source_id": root["source"]["source_id"],
            "source_row_index": root["source"]["source_row_index"],
            "session_count": root["dialogue"]["session_count"],
            "turn_count": root["dialogue"]["turn_count"],
            "dialogue_hash": root["dialogue"]["dialogue_hash"],
            "source_record_sha256": root["source"]["source_record_sha256"],
            "path": str(path), "file_sha256": sha256_file(path),
            "schema_valid": schema_result["valid"],
            "schema_errors": schema_result["errors"],
            "schema_checks": schema_result["checks"],
            "source_mapping_checks": mapping,
            "source_mapping_valid": all(mapping.values()),
            "serialization_round_trip_valid": serialization_valid,
            "root_reader_valid": reader_valid,
        })
    source_audit = _audit_source_file(source_path, prepared["roots"])
    source_file_reverified = (
        source_audit["source_file_sha256"] ==
        prepared["source_file_sha256"])
    global_checks = {
        **prepared["collection_validation"]["global_checks"],
        "source_file_sha256_reverified": source_file_reverified,
        "source_record_sha256_reverified": source_audit[
            "all_source_records_reverified"],
        "source_mapping_exact": all(
            all(row.values()) for row in prepared["source_mapping_checks"]),
        "serialization_round_trip_lossless": all(serialization_checks),
        "root_reader_minimum_parent_contract": all(reader_checks),
        "qa_fields_not_read_or_used": True,
        "planner_not_called": True, "realizer_not_called": True,
        "questioner_not_called": True, "solver_not_called": True,
    }
    adapter_ready = all((
        source_file_reverified,
        source_audit["all_source_records_reverified"],
        all(all(row.values()) for row in prepared["source_mapping_checks"]),
    ))
    schema_ready = prepared["collection_validation"]["valid"]
    root_reader_ready = all(serialization_checks) and all(reader_checks)
    report = {
        "schema_version": "conversation_chronicles_adapter_report_v1",
        "adapter_version": CONVERSATION_CHRONICLES_ADAPTER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "source_file_sha256": prepared["source_file_sha256"],
        "source_record_count": prepared["source_record_count"],
        "valid_source_record_count": prepared["valid_source_record_count"],
        "invalid_json_record_count": prepared["invalid_json_record_count"],
        "invalid_schema_record_count": prepared["invalid_schema_record_count"],
        "selection_seed": seed,
        "selection_algorithm": prepared["selection_algorithm"],
        "unified_root_schema_version": prepared[
            "unified_root_schema_version"],
        "unified_root_schema_sha256": prepared[
            "unified_root_schema_sha256"],
        "adapter_ready": adapter_ready, "schema_ready": schema_ready,
        "root_reader_ready": root_reader_ready,
        "generation_ready": False,
        "global_checks": global_checks, "roots": root_records,
        "historical_snapshot_logic_review": {
            "general_root_loading": (
                "Only the source-session/turn normalization and content-hash "
                "boundary were retained conceptually."),
            "planner_scheduling": "Not restored or copied.",
            "static_outline_compatibility": "Not restored or copied.",
            "run_and_report_management": (
                "Not restored; this adapter writes an independent offline "
                "selection manifest and validation report."),
        },
        "remaining_dynamic_planner_hardcoded_dependencies": [
            "run_stateful_expansion_qa_solver_canary.py hardcodes the conv-41 source root",
            "run_stateful_expansion_qa_solver_canary.py requires exactly 237 parent and 134 excluded child turns",
            "stateful_outline_planner.py emits a conv-41-specific outline_id",
            "run_stateful_expansion_qa_solver_canary.py writes conv-41 into the Questioner context metadata",
            "run_graph_expansion.py still loads static expansion_outline.json and parent_anchor_snapshot.json through load_lineage",
            "input_adapter.load_lineage still depends on round_dataset_train_seed_rows.jsonl for legacy parents",
        ],
        "api_calls": 0, "planner_calls": 0, "realizer_calls": 0,
        "questioner_calls": 0, "solver_calls": 0,
        "verifier_calls": 0, "training_runs": 0,
    }
    manifest = {
        "schema_version": "conversation_chronicles_selection_manifest_v1",
        "adapter_version": CONVERSATION_CHRONICLES_ADAPTER_VERSION,
        "created_at_utc": report["created_at_utc"],
        "source_path": str(source_path),
        "source_file_sha256": prepared["source_file_sha256"],
        "source_record_sha256_scope":
            "raw_jsonl_line_bytes_including_line_terminator",
        "source_split": "train", "selection_seed": seed,
        "selection_algorithm": prepared["selection_algorithm"],
        "requested_root_count": count, "selected_root_count": len(root_records),
        "source_ids": [row["source_id"] for row in root_records],
        "root_ids": [row["root_id"] for row in root_records],
        "roots": [{key: row[key] for key in (
            "root_id", "source_id", "source_row_index", "session_count",
            "turn_count", "dialogue_hash", "source_record_sha256", "path",
            "file_sha256")}
            for row in root_records],
        "qa_fields_read_or_used": False,
        "summary_and_time_interval_used_as_metadata_only": True,
        "conversation_windowing_used": False,
        "conversation_splitting_used": False,
        "cross_conversation_concatenation_used": False,
        "external_api_used": False, "generation_run_created": False,
        "adapter_ready": adapter_ready, "schema_ready": schema_ready,
        "root_reader_ready": root_reader_ready, "generation_ready": False,
    }
    atomic_json(output_root / "selection_manifest.json", manifest)
    atomic_json(output_root / "reports/adapter_validation_report.json", report)
    atomic_text(output_root / "reports/adapter_validation_report.md",
                _render_report(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--count", type=int, default=DEFAULT_ROOT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    args = parser.parse_args(argv)
    report = run(source_path=args.source, output_root=args.output_root,
                 count=args.count, seed=args.seed)
    print(json.dumps({key: report[key] for key in (
        "adapter_ready", "schema_ready", "root_reader_ready",
        "generation_ready")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
