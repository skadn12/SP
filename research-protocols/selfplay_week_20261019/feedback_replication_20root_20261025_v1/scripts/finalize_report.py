#!/usr/bin/env python3
"""Build the final report from observed F/G replication artifacts.

This script never calls a model. ``--self-check`` is pure in-memory and writes
nothing. ``--write-report`` publishes the report and a final original-integrity
receipt but deliberately does not create the manifest. After an independent
reviewer writes the audit for the exact report hash, ``--finalize-manifest``
may bind the report, audit, frozen inputs, and all non-transient task artifacts.
Published outputs are immutable: a differing existing file is an error.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
PHASES = (
    "frozen_design_and_inputs",
    "shared_r03_source_generation_and_review",
    "isolated_fg_plans_questions_and_review",
    "content_freeze_and_solver_preflight",
    "fixed_b1_solver_and_scoring",
    "new16_and_combined20_analysis_and_delivery",
)
SLOTS = ("MC2", "MC3", "MC4", "MC5", "MC6", "ORDER1", "ORDER2", "ORDER3")
ARMS = ("F", "G")
TRANSIENT_NAMES = {
    "FINAL_ARTIFACT_MANIFEST.json", "supervisor.lock", "supervisor.pid",
    ".source_driver.lock", "TASK_STATUS.json", "SUPERVISOR_STATUS.json",
    "PROGRESS.jsonl", "supervisor.log",
}
TRANSIENT_PARTS = {"worker_events", "process_registry"}
RUNTIME_SNAPSHOT_NAME = "FINAL_RUNTIME_SNAPSHOT.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def bind(path: Path) -> dict[str, Any]:
    path = path.resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": sha256_bytes(raw), "bytes": len(raw)}


def publish_immutable(path: Path, value: Any) -> dict[str, Any]:
    """Exclusive-create an artifact, accepting a byte-identical prior result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json_bytes(value)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"IMMUTABLE_OUTPUT_DRIFT:{path}")
    else:
        with path.open("xb") as stream:
            stream.write(raw)
    return bind(path)


def publish_text_immutable(path: Path, value: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value.encode("utf-8")
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"IMMUTABLE_OUTPUT_DRIFT:{path}")
    else:
        with path.open("xb") as stream:
            stream.write(raw)
    return bind(path)


def validator_module(base: Path):
    spec = importlib.util.spec_from_file_location("fgr20_final_validator_for_report", base / "validate_final.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("VALIDATOR_IMPORT_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_optional(base: Path, relative: str, issues: list[str]) -> Any | None:
    path = base / relative
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        issues.append(f"INVALID_JSON:{relative}:{type(exc).__name__}:{exc}")
        return None


def as_rows(value: Any, key: str | None = None) -> list[dict[str, Any]]:
    if key is not None and isinstance(value, dict):
        value = value.get(key)
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def normalize_root(value: Any) -> str:
    text = str(value)
    return text if text.startswith("cc-train-episode-") else "cc-train-episode-" + text


def status_for_phase(complete: bool, attempted: bool, blocked: bool = False) -> str:
    if blocked:
        return "BLOCKED"
    if complete:
        return "COMPLETED"
    return "PARTIAL" if attempted else "NOT_STARTED"


def actual_original_receipt(base: Path) -> dict[str, Any]:
    """Recheck the frozen original manifest and all listed files at closeout."""
    validator = validator_module(base)
    plan: dict[str, Any] = {}
    config: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    errors: list[str] = []
    try:
        plan = read_json(base / "PLAN.json")
    except Exception as exc:
        errors.append(f"PLAN_UNREADABLE:{type(exc).__name__}:{exc}")
    try:
        config = read_json(base / "CONFIG.json")
    except Exception as exc:
        errors.append(f"CONFIG_UNREADABLE:{type(exc).__name__}:{exc}")
    try:
        selection = read_json(base / "SELECTION_FREEZE.json")
        frozen = selection["original_four_reused_read_only"]["bindings"]["FINAL_MANIFEST.json"]
    except Exception as exc:
        frozen = None
        errors.append(f"FROZEN_ORIGINAL_MANIFEST_BINDING_UNREADABLE:{type(exc).__name__}:{exc}")
    roots = plan.get("original_train_roots", config.get("original_roots", []))
    original_roots = [normalize_root(root) for root in roots]
    receipt: dict[str, Any] = {
        "schema_version": "original_integrity_receipt_v1",
        "status": "FAIL", "unchanged": False,
        "original_root_ids": original_roots,
        "original_manifest_binding": frozen,
        "verified_artifacts": 0,
        "mismatch_count": None,
    }
    started = time.monotonic()
    if isinstance(frozen, dict):
        try:
            frozen_manifest = validator.verify(frozen, base)
            original_base = Path(config["original_experiment"]).resolve()
            if frozen_manifest != original_base / "FINAL_MANIFEST.json":
                raise ValueError("ORIGINAL_MANIFEST_PATH_DRIFT")
            manifest = read_json(frozen_manifest)
            artifacts = manifest["files"]
            if not artifacts or len(artifacts) != manifest["file_count"]:
                raise ValueError("ORIGINAL_MANIFEST_FILE_COUNT_DRIFT")
            paths = []
            for artifact in artifacts:
                artifact_path = validator.verify(artifact, original_base)
                if not artifact_path.is_relative_to(original_base):
                    raise ValueError("ORIGINAL_MANIFEST_PATH_ESCAPE")
                paths.append(artifact_path)
            if len(paths) != len(set(paths)):
                raise ValueError("DUPLICATE_ORIGINAL_MANIFEST_PATH")
            receipt.update(status="PASS", unchanged=True,
                           verified_artifacts=len(paths), mismatch_count=0)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    if errors:
        receipt["verification_error"] = "; ".join(errors)
    receipt["verification_elapsed_seconds"] = round(time.monotonic() - started, 6)
    return receipt


def integrity_receipt_identity(receipt: dict[str, Any]) -> dict[str, Any]:
    """Reverification duration is not part of the sealed-artifact identity."""
    return {key: value for key, value in receipt.items() if key != "verification_elapsed_seconds"}


def matching_original_receipts(before: Any, after: Any) -> bool:
    """Compare the contract fields of the independently checked before/after receipts."""
    return bool(
        isinstance(before, dict) and isinstance(after, dict)
        and before.get("status") == after.get("status") == "PASS"
        and before.get("unchanged") is after.get("unchanged") is True
        and before.get("original_manifest_binding") == after.get("original_manifest_binding")
        and sorted(normalize_root(root) for root in before.get("original_root_ids", []))
            == sorted(normalize_root(root) for root in after.get("original_root_ids", []))
        and before.get("verified_artifacts") == after.get("verified_artifacts")
    )


def frozen_analysis_integrity(base: Path, comparison: Any, issues: list[str]) -> dict[str, Any]:
    """Validate frozen analysis provenance without changing or recomputing metrics."""
    document = read_optional(base, "ANALYSIS_DRIVER_FREEZE.json", issues)
    result = {"status": "NOT_VERIFIED", "freeze_binding": None, "script_binding": None,
              "analysis_plan_binding": None, "comparison_provenance_verified": False}
    if not isinstance(document, dict):
        issues.append("ANALYSIS_DRIVER_FREEZE_MISSING_OR_INVALID")
        return result
    result["freeze_binding"] = bind(base / "ANALYSIS_DRIVER_FREEZE.json")
    try:
        if document.get("status") != "FROZEN_BEFORE_SOLVER":
            raise ValueError("ANALYSIS_DRIVER_NOT_FROZEN_BEFORE_SOLVER")
        for key, filename in (("script_binding", "analysis_replication.py"),
                              ("analysis_plan_binding", "ANALYSIS_PLAN.json")):
            if document.get(key) != bind(base / filename):
                raise ValueError("ANALYSIS_FROZEN_BINDING_DRIFT:" + key)
            result[key] = document[key]
        if comparison is not None:
            if not isinstance(comparison, dict):
                raise ValueError("COMPARISON_NOT_OBJECT")
            if comparison.get("script_binding") != document["script_binding"]:
                raise ValueError("COMPARISON_SCRIPT_DIFFERS_FROM_PRE_SOLVER_FREEZE")
            if (comparison.get("models_called_by_analysis") is not False
                    or comparison.get("training") is not False
                    or comparison.get("primary_cohort") != "new16"
                    or comparison.get("primary_group") != "NON_MC6_PRIMARY"
                    or set(comparison.get("cohorts", {})) != {"new16", "original4", "combined20"}):
                raise ValueError("COMPARISON_FROZEN_DESIGN_PROVENANCE_DRIFT")
            plan = read_json(base / "ANALYSIS_PLAN.json")
            if comparison.get("metric_policies", {}).get("bootstrap") != plan["bootstrap"]:
                raise ValueError("COMPARISON_BOOTSTRAP_POLICY_DRIFT")
            for row in comparison.get("input_bindings", []):
                path = Path(row["path"])
                # Runtime status is an explicitly recorded analysis snapshot;
                # it may continue updating after statistics have been written.
                if path.name in TRANSIENT_NAMES:
                    continue
                if bind(path) != row:
                    raise ValueError("COMPARISON_INPUT_BINDING_DRIFT:" + str(path))
            result["comparison_provenance_verified"] = True
        result["status"] = "PASS"
    except (OSError, KeyError, TypeError, ValueError) as exc:
        result["status"] = "FAIL"
        issues.append(f"ANALYSIS_PROVENANCE_CHECK_FAILED:{type(exc).__name__}:{exc}")
    return result


def task_hint(base: Path) -> dict[str, Any]:
    try:
        value = read_json(base / "TASK_STATUS.json")
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def phase_hint(hints: dict[str, Any], phase: str) -> str:
    row = hints.get("phases", {}).get(phase, {}) if isinstance(hints.get("phases"), dict) else {}
    return str(row.get("status", "PENDING")) if isinstance(row, dict) else "PENDING"


def parse_timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        return None


def phase_from_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.lower().replace("-", "_").replace(" ", "_")
    for phase in PHASES:
        if normalized == phase or normalized in phase or phase in normalized:
            return phase
    aliases = {
        "design": PHASES[0], "input": PHASES[0], "source": PHASES[1],
        "question": PHASES[2], "authoring": PHASES[2], "content": PHASES[2],
        "preflight": PHASES[3], "solver": PHASES[4], "scoring": PHASES[4],
        "analysis": PHASES[5], "delivery": PHASES[5], "report": PHASES[5],
    }
    return aliases.get(normalized)


def progress_rows(base: Path, issues: list[str]) -> list[dict[str, Any]]:
    path = base / "PROGRESS.jsonl"
    if not path.is_file():
        return []
    rows = []
    try:
        raw_text = path.read_text(encoding="utf-8")
        # A startup writer escaped a JSONL object separator; normalize that boundary only.
        raw_text = raw_text.replace("}\\n{", "}\n{")
        for line_number, line in enumerate(raw_text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                issues.append(f"INVALID_PROGRESS_ROW:{line_number}:{exc}")
                continue
            if isinstance(value, dict):
                rows.append(value)
    except OSError as exc:
        issues.append(f"PROGRESS_UNREADABLE:{type(exc).__name__}:{exc}")
    return rows


def recorded_timing(base: Path, task: dict[str, Any], issues: list[str]) -> dict[str, Any]:
    """Derive wall durations only from logged UTC timestamps, never file mtimes."""
    supervisor = read_optional(base, "SUPERVISOR_STATUS.json", issues)
    supervisor = supervisor if isinstance(supervisor, dict) else {}
    events = progress_rows(base, issues)
    started = parse_timestamp(supervisor.get("started_at_utc"))
    starts: dict[str, dt.datetime] = {}
    ends: dict[str, dt.datetime] = {}

    phase_rows = task.get("phases", {}) if isinstance(task.get("phases"), dict) else {}
    for phase in PHASES:
        row = phase_rows.get(phase, {})
        if not isinstance(row, dict):
            continue
        for key in ("started_at_utc", "start_at_utc", "phase_started_at_utc"):
            parsed = parse_timestamp(row.get(key))
            if parsed:
                starts[phase] = parsed
                break
        for key in ("ended_at_utc", "end_at_utc", "completed_at_utc", "phase_ended_at_utc"):
            parsed = parse_timestamp(row.get(key))
            if parsed:
                ends[phase] = parsed
                break

    for event in events:
        at = parse_timestamp(event.get("at_utc"))
        if not at:
            continue
        kind = str(event.get("kind", "")).upper()
        phase = phase_from_label(event.get("phase") or event.get("stage") or event.get("phase_name"))
        if kind == "SOURCE_STAGE_START":
            phase = PHASES[1]
        elif phase is None:
            for marker, mapped in (
                ("SOURCE", PHASES[1]), ("QUESTION", PHASES[2]), ("AUTHOR", PHASES[2]),
                ("CONTENT", PHASES[2]), ("PREFLIGHT", PHASES[3]), ("SOLVER", PHASES[4]),
                ("SCOR", PHASES[4]), ("ANALYSIS", PHASES[5]), ("DELIVERY", PHASES[5]),
            ):
                if marker in kind and ("STAGE" in kind or "PHASE" in kind):
                    phase = mapped
                    break
        if phase is None:
            continue
        if kind.endswith("START") or kind in {"PHASE_START", "STAGE_START"}:
            starts.setdefault(phase, at)
        elif kind.endswith("END") or kind.endswith("FINISH") or kind in {"PHASE_COMPLETED", "STAGE_COMPLETED"}:
            ends.setdefault(phase, at)

    if PHASES[0] not in starts and started:
        starts[PHASES[0]] = started

    recorded_endpoints = [
        parse_timestamp(supervisor.get(key))
        for key in ("ended_at_utc", "updated_at_utc", "last_event_at_utc")
    ]
    recorded_endpoints.extend(parse_timestamp(row.get("at_utc")) for row in events)
    recorded_endpoints.append(parse_timestamp(task.get("updated_at_utc")))
    endpoint = max((value for value in recorded_endpoints if value), default=None)
    supervisor_ended = parse_timestamp(supervisor.get("ended_at_utc"))
    if not endpoint and started and type(supervisor.get("elapsed_seconds")) in (int, float):
        endpoint = started + dt.timedelta(seconds=float(supervisor["elapsed_seconds"]))

    phase_timing: dict[str, dict[str, Any]] = {}
    for index, phase in enumerate(PHASES):
        phase_start = starts.get(phase)
        phase_end = ends.get(phase)
        next_start = next((starts.get(later) for later in PHASES[index + 1:] if starts.get(later)), None)
        status_row = phase_rows.get(phase, {}) if isinstance(phase_rows, dict) else {}
        status = str(status_row.get("status", "PENDING")) if isinstance(status_row, dict) else "PENDING"
        if phase_end is None and next_start:
            phase_end = next_start
        if phase_end is None and status in {"IN_PROGRESS", "RUNNING"} and phase_start:
            phase_end = endpoint
        if phase_end is None and status == "COMPLETED" and phase == PHASES[-1] and supervisor_ended:
            phase_end = supervisor_ended
        if phase_start and phase_end and phase_end >= phase_start:
            seconds: float | None = round((phase_end - phase_start).total_seconds(), 3)
            basis = "recorded_start_and_end"
        else:
            seconds = None
            basis = "stage_boundary_not_recorded"
        phase_timing[phase] = {
            "elapsed_seconds": seconds,
            "started_at_utc": phase_start.isoformat() if phase_start else None,
            "ended_at_utc": phase_end.isoformat() if phase_end else None,
            "basis": basis,
        }

    total_seconds = None
    if started and endpoint and endpoint >= started:
        total_seconds = round((endpoint - started).total_seconds(), 3)
    return {
        "started_at_utc": started.isoformat() if started else None,
        "measured_through_utc": endpoint.isoformat() if endpoint else None,
        "total_elapsed_seconds": total_seconds,
        "per_phase": phase_timing,
        "measurement_basis": "SUPERVISOR_STATUS.started_at_utc plus recorded supervisor/progress/task timestamps; file mtimes are not used",
        "unavailable_phase_boundaries": [phase for phase, row in phase_timing.items() if row["elapsed_seconds"] is None],
    }


def checked_rows(bindings: Any, issues: list[str], label: str) -> list[dict[str, Any]]:
    rows = as_rows(bindings)
    good = []
    for row in rows:
        try:
            actual = bind(Path(row["path"]))
            expected = {key: row.get(key) for key in ("path", "sha256", "bytes")}
            if actual != expected:
                issues.append(f"BINDING_MISMATCH:{label}:{row.get('path')}")
            else:
                good.append(row)
        except (OSError, KeyError, TypeError) as exc:
            issues.append(f"BINDING_UNVERIFIED:{label}:{row.get('path')}:{type(exc).__name__}:{exc}")
    return good


def nested_bindings(value: Any) -> list[dict[str, Any]]:
    """Read explicit provenance only; do not infer missing bindings."""
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= set(value):
            return [{key: value[key] for key in ("path", "sha256", "bytes")}]
        return [row for child in value.values() for row in nested_bindings(child)]
    if isinstance(value, list):
        return [row for child in value for row in nested_bindings(child)]
    return []


def schema_adapter_launch_disclosures(base: Path, issues: list[str]) -> dict[str, Any]:
    records = []
    for family in ("review_schema_adapter", "review_schema_adapter_v2"):
        for path in sorted((base / family / "launches").glob("*.json")):
            doc = read_optional(base, str(path.relative_to(base)), issues)
            if not isinstance(doc, dict):
                continue
            references = nested_bindings(doc)
            verified = checked_rows(references, issues, "ACTUAL_SCHEMA_ADAPTER_LAUNCH")
            records.append({
                "launch_receipt_binding": bind(path), "adapter_family": family,
                "recorded_schema_version": doc.get("schema_version"),
                "driver": doc.get("driver"), "root_id": doc.get("root_id"), "arm": doc.get("arm"),
                "driver_argv": doc.get("driver_argv"),
                "reference_count": len(references), "verified_reference_count": len(verified),
                "references": verified,
                "future_unstarted_stage_only_reported": doc.get("future_unstarted_stage_only"),
                "extra_semantic_budget_reported": doc.get("extra_semantic_budget"),
                "semantic_admission_asserted": doc.get("semantic_admission_asserted"),
                "raw_and_old_validation_and_review_artifacts_preserved_reported": doc.get("raw_and_old_validation_and_review_artifacts_preserved"),
            })
    return {"observed_launch_receipt_count": len(records), "records": records,
            "scope": "Actual recorded driver/adapter identities and immutable launch receipts; no semantic acceptance or additional author/reviewer budget inferred. V2 applies only to its four exact previously unstarted targets; the earlier F179186 v1 work is retained."}


def summarize_recovery_sources(freeze: Any, selection: Any, verified_rows: list[dict[str, Any]], all_ids: list[str]) -> dict[str, Any]:
    freeze = freeze if isinstance(freeze, dict) else None
    selection = selection if isinstance(selection, dict) else None
    verified_ids = sorted({normalize_root(row.get("root_id", "")) for row in verified_rows if row.get("root_id")})
    return {
        "selection_path": "SOURCE_SELECTION_RECOVERY.json" if selection else None,
        "freeze_path": "SOURCE_FREEZE_RECOVERY.json" if freeze else None,
        "freeze_status": freeze.get("status") if freeze else None,
        "reported_passed_root_count": freeze.get("passed_root_count") if freeze else None,
        "selection_root_count": len(as_rows(selection, "roots")),
        "verified_selection_root_count": len(verified_ids),
        "verified_root_ids": verified_ids,
        "unverified_root_ids": sorted(set(all_ids) - set(verified_ids)),
        "counted_as_canonical_source_selection": False,
    }


def summarize_shape_changes(changes: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in as_rows(changes):
        mapping = str(row.get("mapping", "UNSPECIFIED_MAPPING"))
        result[mapping] = result.get(mapping, 0) + 1
    return dict(sorted(result.items()))


def engineering_disclosures(base: Path, issues: list[str]) -> dict[str, Any]:
    """Describe saved engineering adaptations, without semantic judgments."""
    normalization_rows = []
    plan_trees = ("question_plan_continuation", "question_plan_continuation_v3")
    normalization_paths = [path for tree in plan_trees for path in
                           (base / tree).glob("*/*/MECHANICAL_PLAN_NORMALIZATION.json")]
    for path in sorted(normalization_paths):
        doc = read_optional(base, str(path.relative_to(base)), issues)
        if not isinstance(doc, dict):
            continue
        work = path.parent
        reference = read_optional(base, str((work / "ORIGINAL_PLAN_RECEIPT_REFERENCE.json").relative_to(base)), issues)
        references = [value for value in (reference or {}).values()
                      if isinstance(value, dict) and {"path", "sha256", "bytes"} <= set(value)]
        verified = checked_rows(references, issues, "ORIGINAL_PLAN_SCHEMA_RECOVERY")
        files = [path, work / "TASK_PLANS.json", work / "TASK_PLANS_FREEZE.json"]
        files.extend(work / name for name in ("TASK_PLAN_RESPONSE.json", "TASK_PLAN_CALL.json",
                                              "ORIGINAL_PLAN_RECEIPT_REFERENCE.json") if (work / name).is_file())
        changes = as_rows(doc.get("changes"))
        normalization_rows.append({
            "root_id": doc.get("root_id"), "arm": doc.get("arm"),
            "actual_engineering_stage_directory": str(work.resolve()),
            "normalization_binding": bind(path),
            "structural_change_count": len(changes),
            "changes_by_recorded_mapping": summarize_shape_changes(changes),
            "strict_validation_errors": doc.get("strict_validation_errors"),
            "raw_response_is_preserved_reported": doc.get("raw_response_is_preserved"),
            "no_semantic_generation_or_acceptance_reported": doc.get("no_semantic_generation_or_acceptance"),
            "all_logged_changes_marked_nonsemantic": all(row.get("semantic_acceptance") is False for row in changes),
            "original_plan_call_repeated_reported": (reference or {}).get("plan_call_repeated"),
            "original_artifact_reference_count": len(references),
            "original_artifact_bindings_verified": len(verified),
            "original_artifact_bindings": verified,
            "saved_stage_bindings": [bind(value) for value in files if value.is_file()],
            "driver_adaptation_bindings": [bind(work / name) for name in
                ("DRIVER_ADAPTATION.json", "DRIVER_ADAPTATION_V2.json", "DRIVER_ADAPTATION_V3.json") if (work / name).is_file()],
        })
    consumed_plan_failures = []
    disposition_paths = [path for tree in plan_trees for path in
                         (base / tree).glob("*/*/PLAN_FAILURE_DISPOSITION.json")]
    for path in sorted(disposition_paths):
        doc = read_optional(base, str(path.relative_to(base)), issues)
        if not isinstance(doc, dict):
            continue
        references = [doc[key] for key in ("original_outcome_binding", "semantic_failure_binding", "actual_terminal_binding")
                      if isinstance(doc.get(key), dict)]
        references.extend(as_rows(doc.get("actual_plan_call_artifacts")))
        references = list({row.get("path"): row for row in references}.values())
        verified = checked_rows(references, issues, "CONSUMED_PLAN_FAILURE_DISPOSITION")
        consumed_plan_failures.append({"root_id": doc.get("root_id"), "arm": doc.get("arm"),
            "disposition_binding": bind(path), "recorded_status": doc.get("status"),
            "new_plan_calls": doc.get("new_plan_calls"), "new_candidate_calls": doc.get("new_candidate_calls"),
            "semantic_failure_was_not_reclassified_as_transport": doc.get("semantic_failure_was_not_reclassified_as_transport"),
            "actual_failure_artifact_count": len(references), "verified_failure_artifact_count": len(verified),
            "failure_artifact_bindings": verified,
            "driver_adaptation_bindings": [bind(path.parent / name) for name in
                ("DRIVER_ADAPTATION.json", "DRIVER_ADAPTATION_V2.json", "DRIVER_ADAPTATION_V3.json") if (path.parent / name).is_file()]})
    initial_assembly = read_optional(base, "INITIAL_CONTENT_ASSEMBLY.json", issues)
    serialization_recoveries = []
    for row in as_rows(initial_assembly.get("same_initial_draft_serialization_recoveries")) if isinstance(initial_assembly, dict) else []:
        reference = row.get("original_draft_reference_binding")
        verified_reference = checked_rows([reference] if isinstance(reference, dict) else [], issues,
                                          "ORIGINAL_DRAFT_SERIALIZATION_REFERENCE")
        serialization_recoveries.append({
            "root_id": row.get("root_id"), "arm": row.get("arm"),
            "original_draft_reference_binding": reference,
            "original_reference_binding_verified": len(verified_reference) == 1,
            "recorded_provenance": row.get("recorded_provenance"),
            "raw_initial_serialization_failure_surfaces": row.get("raw_initial_serialization_failure_surfaces"),
            "denominator_policy": row.get("denominator_policy"),
        })
    runtime_schema_rows = []
    runtime_stage_trees = ("question_authoring", "question_plan_continuation", "question_plan_continuation_v3",
                           "question_candidate_shape_continuation", "question_unfilled_continuation",
                           "question_unfilled_slot_continuation", "question_saved_revision_verification")
    for tree in runtime_stage_trees:
        for path in sorted((base / tree).rglob("MECHANICAL_*_SCHEMA_NORMALIZATION*.json")):
            doc = read_optional(base, str(path.relative_to(base)), issues)
            if not isinstance(doc, dict):
                continue
            references = [value for value in doc.values()
                          if isinstance(value, dict) and {"path", "sha256", "bytes"} <= set(value)]
            verified = checked_rows(references, issues, "RAM_REVIEW_OR_CANDIDATE_SCHEMA_NORMALIZATION")
            runtime_schema_rows.append({
                "normalization_binding": bind(path), "actual_stage_directory": str(path.parent.resolve()),
                "kind": doc.get("kind"), "stage_suffix": doc.get("stage_suffix"),
                "recorded_change_count": len(as_rows(doc.get("changes"))),
                "recorded_reference_count": len(references), "verified_reference_count": len(verified),
                "references": verified,
                "input_object_matches_saved_authored_response": doc.get("input_object_matches_saved_authored_response"),
                "exact_round_trip_verified": doc.get("exact_round_trip_verified"),
                "question_text_gold_evidence_decisions_and_all_values_changed": doc.get("question_text_gold_evidence_decisions_and_all_values_changed"),
                "semantic_calls_or_revision_budget_consumed_by_adapter": doc.get("semantic_calls_or_revision_budget_consumed_by_adapter"),
                "semantic_admission_asserted": doc.get("semantic_admission_asserted"),
                "earlier_raw_validation_and_review_files_overwritten": doc.get("earlier_raw_validation_and_review_files_overwritten"),
            })
    runtime_doc = read_optional(base, "RUNTIME_INTERPRETER_ADAPTATION.json", issues)
    runtime = None
    if isinstance(runtime_doc, dict):
        code_rows = as_rows(runtime_doc.get("code_bindings"))
        verified = checked_rows(code_rows, issues, "RUNTIME_INTERPRETER_ADAPTATION")
        runtime = {
            "receipt_binding": bind(base / "RUNTIME_INTERPRETER_ADAPTATION.json"),
            "recorded_status": runtime_doc.get("status"),
            "adaptation": runtime_doc.get("adaptation"),
            "recorded_settings": runtime_doc.get("settings"),
            "code_binding_count": len(code_rows), "verified_code_binding_count": len(verified),
            "gpu_model_started_by_cpu_check": runtime_doc.get("gpu_model_started"),
            "cuda_context_initialized_by_cpu_check": runtime_doc.get("cuda_context_initialized"),
            "model_calls_by_cpu_check": runtime_doc.get("model_calls"),
        }
    saved_revision_verifications = []
    for path in sorted((base / "question_saved_revision_verification").glob("*/*/RECOVERY_INPUT_FREEZE.json")):
        doc = read_optional(base, str(path.relative_to(base)), issues)
        if not isinstance(doc, dict):
            continue
        references = as_rows(doc.get("input_bindings"))
        verified = checked_rows(references, issues, "SAVED_FINAL_REVISION_GOLD_INPUT")
        work = path.parent
        ready = read_optional(base, str((work / "CONTINUATION_FINALIZER_READY.json").relative_to(base)), issues)
        gates = read_optional(base, str((work / "FINAL_REVIEW_GATE.json").relative_to(base)), issues)
        call_path = work / "CONTENT_VERIFIER_CALL_revision_01.json"
        final_candidate_doc = read_optional(base, str((work / "CANDIDATES_FINAL.json").relative_to(base)), issues)
        final_review_doc = read_optional(base, str((work / "CONTENT_REVIEW_FINAL.json").relative_to(base)), issues)
        saved_revision_verifications.append({
            "root_id": doc.get("root_id"), "arm": doc.get("arm"),
            "input_freeze_binding": bind(path), "original_work_dir": doc.get("input_work_dir"),
            "saved_gold_call_id": doc.get("saved_gold_call_id"),
            "only_allowed_new_call_id": doc.get("allowed_new_call_id"),
            "saved_gold_receipt_binding": doc.get("saved_gold_receipt_binding"),
            "input_binding_count": len(references), "verified_input_binding_count": len(verified),
            "saved_gold_values_unchanged_reported": doc.get("independent_gold_values_unchanged"),
            "no_new_gold_call_reported": doc.get("no_new_gold_call"),
            "no_new_plan_or_author_call_reported": doc.get("no_new_plan_or_author_call"),
            "no_second_revision_reported": doc.get("no_second_revision"),
            "new_verifier_return_receipt_binding": bind(call_path) if call_path.is_file() else None,
            "terminal_ready_binding": bind(work / "CONTINUATION_FINALIZER_READY.json") if isinstance(ready, dict) else None,
            "terminal_review_stage": ready.get("review_stage") if isinstance(ready, dict) else None,
            "serialization_variant": doc.get("serialization_variant"),
            "saved_gold_covered_slot_ids": doc.get("covered_slot_ids"),
            "uncovered_slot_ids": doc.get("uncovered_slot_ids", []),
            "published_final_candidate_count": len(as_rows(final_candidate_doc)) if final_candidate_doc is not None else None,
            "published_final_review_count": len(as_rows(final_review_doc, "reviews")) if final_review_doc is not None else None,
            "original_failure_archive_binding": ready.get("original_final_candidates_archive_binding") if isinstance(ready, dict) else None,
            "uncovered_original_failure_and_HOLD_retained_reported": ready.get("uncovered_ORDER2_original_failure_notice_and_review_preserved") if isinstance(ready, dict) else None,
            "recorded_final_gate_admitted_slots": sum(row.get("admitted") is True for row in as_rows(gates, "results")) if isinstance(gates, dict) else None,
            "initial_statistics_replaced": False,
        })
    artifact_names = ("SOURCE_RECOVERY_CANONICAL_PROMOTION_V3.json", "SOURCE_FILE_PUBLICATION.json",
                      "QUESTION_DRIVER_AUDIT_READY.json", "CONTENT_FINALIZER_ADAPTATION.json",
                      "QUESTION_PLAN_SCHEMA_CONTINUATION_READY.json", "QUESTION_PLAN_SCHEMA_CONTINUATION_V2_READY.json",
                      "QUESTION_PLAN_SCHEMA_CONTINUATION_V3_READY.json",
                      "QUESTION_CANDIDATE_SHAPE_CONTINUATION_READY.json",
                      "REVIEW_SCHEMA_ADAPTER_READY.json",
                      "REVIEW_SCHEMA_ADAPTER_V2_READY.json",
                      "QUESTION_DRIVER_ENGINEERING_HANDOFF_V2.json",
                      "UNUSED_FAILED_LINEAGE_ADAPTER_READY.json",
                      "G136202_SAVED_GLOBAL_GOLD_VERIFICATION_READY.json",
                      "G109164_SAVED_GLOBAL_GOLD_VERIFICATION_READY.json",
                      "F179186_SAVED_SHAPE_GOLD_VERIFICATION_READY.json",
                      "PRODUCTION_QUESTION_PROCESS_GUARD_HISTORY_LOCATOR.json",
                      "PRODUCTION_G109_QUESTION_PROCESS_GUARD_READY.json",
                      "PRODUCTION_G109_QUESTION_PROCESS_GUARD_HISTORY_LOCATOR.json",
                      "SAVED_REVISION_GOLD_VERIFICATION_READY.json",
                      "PRODUCTION_QUESTION_PROCESS_GUARD_ENGINEERING_READY.json",
                      "QUESTION_UNFILLED_SLOT_CONTINUATION_READY.json",
                      "INITIAL_CONTENT_ASSEMBLY.json", "CPU_LENGTH_PRECHECK.json",
                      "scripts/question_plan_schema_continuation.py", "scripts/question_plan_schema_continuation_v2.py",
                      "scripts/question_plan_schema_continuation_v3.py",
                      "scripts/question_candidate_shape_continuation.py",
                      "scripts/review_schema_adapter.py",
                      "scripts/review_schema_adapter_v2.py",
                      "scripts/unused_failed_lineage_adapter.py",
                      "scripts/saved_revision_gold_verification.py",
                      "scripts/g136202_saved_global_gold_verification.py",
                      "scripts/g109164_saved_global7_gold_verification.py",
                      "scripts/f179186_saved_shape_gold_verification.py",
                      "scripts/question_unfilled_slot_continuation.py",
                      "scripts/content_continuation_adapter.py", "scripts/solver_runtime_adapter.py",
                      "scripts/solver_cpu_length_precheck.py", "scripts/production_driver.py",
                      "scripts/closeout_audit_driver.py", "scripts/audit_call_receipts.py")
    return {
        "scope": "Recorded artifact/schema/interpreter-path engineering only; never a source, question, Gold or content-review semantic PASS.",
        "plan_schema_recovery": {"observed_stage_count": len(normalization_rows),
                                 "unique_root_arm_count": len({(row["root_id"], row["arm"]) for row in normalization_rows}),
                                 "structural_change_count": sum(row["structural_change_count"] for row in normalization_rows),
                                 "records": normalization_rows,
                                 "budget_policy": "Existing raw plan responses and original freezes/outcomes are retained. Shape normalization is not another planner call or semantic acceptance. Actual independent author/reviewer receipts govern content admission."},
        "consumed_plan_failures_without_replanning": consumed_plan_failures,
        "same_initial_draft_serialization_recovery": {
            "observed_root_arm_count": len(serialization_recoveries),
            "records": serialization_recoveries,
            "interpretation": "Pure structural recovery of the already consumed initial author response is disclosed separately from content revision. Initial validity uses only the first actual independent reviews of the normalized same draft; the original serialization gate failures are retained and do not create extra draft/slot denominators.",
        },
        "runtime_review_and_candidate_schema_adaptation": {
            "recorded_normalization_surface_count": len(runtime_schema_rows),
            "records": runtime_schema_rows,
            "interpretation": "These are recorded reversible serialization/field-alias normalizations, not new Gold, reviewer decisions, extra author calls or retrospective admission. Previously saved initial raw validation/HOLD remains preserved; revision reviews never replace initial-review statistics.",
        },
        "actual_schema_adapter_launch_provenance": schema_adapter_launch_disclosures(base, issues),
        "saved_final_revision_gold_first_verifier": {
            "observed_preparation_count": len(saved_revision_verifications),
            "records": saved_revision_verifications,
            "interpretation": "Only actually saved final-revision blind Gold with a separately approved exact reversible schema variant can be reused for its first fresh verifier. Existing call-global read attestations may be copied into otherwise absent row fields only under a bound exact round-trip variant; their values and Gold remain unchanged. No planner, author, Gold reconstruction or second revision is created. Original initial responses/reviews and their failed yield remain unchanged; final verification never fills initial statistics. An original author failure notice without question/items/Gold is archived, never counted as a published question or an invented review.",
        },
        "runtime_interpreter_adaptation": runtime,
        "associated_artifact_bindings": [bind(base / name) for name in artifact_names if (base / name).is_file()],
    }


def observe(base: Path, receipt_binding: dict[str, Any]) -> dict[str, Any]:
    """Read current artifacts and derive status/counts without writing files."""
    issues: list[str] = []
    evidence: dict[str, list[str]] = {phase: [] for phase in PHASES}
    plan = read_optional(base, "PLAN.json", issues)
    config = read_optional(base, "CONFIG.json", issues)
    input_lock = read_optional(base, "INPUT_LOCK.json", issues)
    task = task_hint(base)
    design_ok = False
    lock_ok = False
    new_roots: list[str] = []
    old_roots: list[str] = []
    validator = validator_module(base)
    try:
        _, _, new_roots, old_roots, _ = validator.design(base)
        design_ok = True
        evidence[PHASES[0]].append("PLAN.json and CONFIG.json satisfy frozen 16+4 design")
    except Exception as exc:
        issues.append(f"DESIGN_CHECK_FAILED:{type(exc).__name__}:{exc}")
    try:
        input_binding_count = validator.verify_input_lock(base)
        lock_ok = True
        evidence[PHASES[0]].append(f"INPUT_LOCK verified: {input_binding_count} bindings")
    except Exception as exc:
        input_binding_count = 0
        issues.append(f"INPUT_LOCK_CHECK_FAILED:{type(exc).__name__}:{exc}")

    before = read_optional(base, "ORIGINAL_INTEGRITY_BEFORE.json", issues)
    receipt_ok = receipt_binding.get("sha256") is not None
    try:
        current_receipt_binding = bind(Path(receipt_binding["path"]))
        receipt_ok = current_receipt_binding == receipt_binding
        after = read_json(Path(receipt_binding["path"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        after = None
        receipt_ok = False
        issues.append(f"AFTER_RECEIPT_UNREADABLE:{type(exc).__name__}:{exc}")
    receipt_pass = bool(isinstance(after, dict) and after.get("status") == "PASS" and after.get("unchanged") is True)
    if not receipt_ok:
        issues.append("AFTER_RECEIPT_BINDING_MISMATCH")
    if not receipt_pass:
        issues.append("ORIGINAL_AFTER_RECEIPT_NOT_PASS")
    before_pass = matching_original_receipts(before, after)
    if not before_pass:
        issues.append("ORIGINAL_BEFORE_RECEIPT_MISSING_OR_NOT_PASS")
    frozen_design_block = not design_ok or not lock_ok or not before_pass or not receipt_pass
    evidence[PHASES[0]].append("ORIGINAL_INTEGRITY_BEFORE.json" if before else "ORIGINAL_INTEGRITY_BEFORE.json missing")
    evidence[PHASES[0]].append(str(receipt_binding.get("path", "ORIGINAL_INTEGRITY_AFTER.json")))

    sources = read_optional(base, "SOURCE_FREEZE.json", issues)
    selection = read_optional(base, "SOURCE_SELECTION_FINAL.json", issues)
    source_rows = as_rows(selection, "roots")
    source_root_ids = sorted({normalize_root(row.get("root_id", "")) for row in source_rows if row.get("root_id")})
    source_binding_rows = checked_rows(source_rows, issues, "SOURCE_SELECTION_FINAL")
    recovery_sources = read_optional(base, "SOURCE_FREEZE_RECOVERY.json", issues)
    recovery_selection = read_optional(base, "SOURCE_SELECTION_RECOVERY.json", issues)
    recovery_source_rows = as_rows(recovery_selection, "roots")
    recovery_source_ids_all = sorted({normalize_root(row.get("root_id", "")) for row in recovery_source_rows if row.get("root_id")})
    recovery_source_bindings = checked_rows(recovery_source_rows, issues, "SOURCE_SELECTION_RECOVERY")
    recovery_ids_verified = sorted({normalize_root(row.get("root_id", "")) for row in recovery_source_bindings if row.get("root_id")})
    recovery_provenance = summarize_recovery_sources(
        recovery_sources, recovery_selection, recovery_source_bindings, recovery_source_ids_all)
    source_complete = bool(sources and sources.get("status") == "PASS" and len(source_rows) == 16
                           and set(source_root_ids) == {"cc-train-episode-" + root for root in new_roots}
                           and len(source_binding_rows) == 16)
    source_attempted = bool(sources or selection or recovery_sources or recovery_selection
                            or list((base / "source_authoring").glob("*/OUTCOME.json"))
                            or phase_hint(task, PHASES[1]) != "PENDING")
    source_status = status_for_phase(source_complete, source_attempted)
    if source_complete:
        evidence[PHASES[1]].append("SOURCE_FREEZE passed with all 16 planned roots")
    elif sources or selection:
        evidence[PHASES[1]].append(f"observed final source roots: {len(source_root_ids)}/16; source freeze status={sources.get('status') if sources else 'missing'}")
    if recovery_sources or recovery_selection:
        evidence[PHASES[1]].append(
            f"separate recovery provenance: {len(recovery_ids_verified)}/{len(recovery_source_rows)} verified roots; "
            f"recovery freeze status={recovery_sources.get('status') if recovery_sources else 'missing'}; not counted as canonical selection"
        )
    if (base / "SOURCE_FAILURES.json").is_file():
        evidence[PHASES[1]].append("SOURCE_FAILURES.json retained")

    candidates = read_optional(base, "CANDIDATES_PRIVATE.json", issues)
    reviews_doc = read_optional(base, "CONTENT_REVIEW_AI.json", issues)
    gate = read_optional(base, "CONTENT_GATE_FINAL.json", issues)
    content_freeze = read_optional(base, "CONTENT_FREEZE.json", issues)
    candidate_rows = as_rows(candidates)
    review_rows = as_rows(reviews_doc, "reviews")
    gate_rows = as_rows(gate, "records")
    candidate_map = {r.get("candidate_id"): r for r in candidate_rows if r.get("candidate_id")}
    review_map = {r.get("candidate_id"): r for r in review_rows if r.get("candidate_id")}
    gate_map = {r.get("candidate_id"): r for r in gate_rows if r.get("candidate_id")}
    admitted_ids = {
        cid for cid, candidate in candidate_map.items()
        if review_map.get(cid, {}).get("decision") == "PASS"
        and gate_map.get(cid, {}).get("eligible_for_solver") is True
    }
    admitted_count = len(admitted_ids)
    expected_candidate_slots = {
        ("cc-train-episode-" + root, slot, arm)
        for root in new_roots for slot in SLOTS for arm in ARMS
    }
    candidate_slots = {
        (row.get("root_id"), row.get("slot_id"), row.get("arm")) for row in candidate_rows
        if row.get("candidate_id") in admitted_ids
    }
    content_complete = bool(
        len(candidate_rows) == 256 and len(candidate_map) == 256
        and len(review_rows) == len(review_map) == 256
        and len(gate_rows) == len(gate_map) == 256
        and admitted_count == 256 and candidate_slots == expected_candidate_slots
        and content_freeze and content_freeze.get("status") == "FROZEN_BEFORE_SOLVER"
        and content_freeze.get("admitted_candidate_count") == 256
        and content_freeze.get("planned_slot_count") == 256
    )
    content_attempted = bool(candidate_rows or reviews_doc or gate or content_freeze
                              or phase_hint(task, PHASES[2]) != "PENDING")
    content_status = status_for_phase(content_complete, content_attempted)
    if content_freeze:
        evidence[PHASES[2]].append(
            f"candidate rows={len(candidate_rows)}, unique IDs={len(candidate_map)}, "
            f"PASS review + eligible gate={admitted_count}/256"
        )
        evidence[PHASES[2]].append("CONTENT_FREEZE.json present")
    for name in ("SLOT_FAILURES.json", "DUPLICATE_DISPOSITIONS.json"):
        if (base / name).is_file():
            evidence[PHASES[2]].append(name + " retained")

    prep = read_optional(base, "execution/PREPARATION.json", issues)
    cpu_api = read_optional(base, "execution/CPU_API_CHECK.json", issues)
    prep_requests = prep.get("requests", 0) if isinstance(prep, dict) and type(prep.get("requests")) is int else 0
    preflight_complete = bool(
        content_freeze and prep and cpu_api
        and prep_requests == admitted_count * 2
        and prep.get("planned_requests") == 512
        and prep.get("planned_semantic_questions") == 256
        and cpu_api.get("gpu_model_started") is False
    )
    prep_attempted = bool(prep or cpu_api or phase_hint(task, PHASES[3]) != "PENDING")
    preflight_status = status_for_phase(preflight_complete, prep_attempted)
    if prep:
        evidence[PHASES[3]].append(f"execution/PREPARATION.json requests={prep_requests}")
    if cpu_api:
        evidence[PHASES[3]].append("execution/CPU_API_CHECK.json present")

    raw_freeze = read_optional(base, "execution/RAW_RESPONSE_FREEZE.json", issues)
    score_doc = read_optional(base, "execution/SCORES.json", issues)
    analysis_summary = read_optional(base, "execution/ANALYSIS_SUMMARY.json", issues)
    raw_returned_ids: set[str] = set()
    raw_total = 0
    raw_bindings = as_rows(raw_freeze.get("bindings")) if isinstance(raw_freeze, dict) else []
    if raw_freeze:
        raw_total = raw_freeze.get("requests", 0) if type(raw_freeze.get("requests")) is int else 0
        good_raw_bindings = checked_rows(raw_bindings, issues, "RAW_RESPONSE_FREEZE")
        for binding_row in good_raw_bindings:
            try:
                response = read_json(Path(binding_row["path"]))
                if response.get("status") == "RETURNED" and isinstance(response.get("request_id"), str):
                    raw_returned_ids.add(response["request_id"])
            except (OSError, ValueError, AttributeError) as exc:
                issues.append(f"RAW_RESPONSE_UNREADABLE:{binding_row.get('path')}:{type(exc).__name__}:{exc}")
        if raw_freeze.get("returned") != len(raw_returned_ids):
            issues.append(f"RAW_RETURNED_COUNT_DISAGREES:freeze={raw_freeze.get('returned')}, verified={len(raw_returned_ids)}")

    score_rows = as_rows(score_doc, "records")
    score_by_id = {row.get("request_id"): row for row in score_rows if isinstance(row.get("request_id"), str)}
    scored_returned_ids = set(score_by_id) & raw_returned_ids
    returned_count = len(raw_returned_ids)
    scored_count = len(scored_returned_ids)
    if score_doc and len(score_rows) != len(score_by_id):
        issues.append("DUPLICATE_OR_UNIDENTIFIED_SCORE_ROWS")
    solver_attempted = bool(raw_freeze or score_doc or (base / "execution/STARTED.json").exists()
                             or phase_hint(task, PHASES[4]) != "PENDING")
    denom = analysis_summary.get("denominators", {}) if isinstance(analysis_summary, dict) else {}
    solver_complete = bool(
        prep and raw_freeze and score_doc and analysis_summary
        and raw_total == 512 and raw_freeze.get("returned") == 512
        and raw_freeze.get("failure") is None
        and raw_freeze.get("submitted_ids") == raw_freeze.get("request_ids")
        and len(raw_returned_ids) == 512 and len(score_by_id) == 512
        and scored_count == 512 and denom.get("planned_semantic_questions") == 256
        and denom.get("content_admitted_semantic_questions") == 256
        and denom.get("planned_solver_requests") == 512
        and denom.get("frozen_requests") == denom.get("returned_requests") == 512
        and denom.get("held_or_unavailable_semantic_slots") == 0
    )
    solver_status = status_for_phase(solver_complete, solver_attempted)
    if raw_freeze:
        evidence[PHASES[4]].append(f"raw requests={raw_total}; verified RETURNED={returned_count}")
    if score_doc:
        evidence[PHASES[4]].append(f"score rows={len(score_rows)}; unique returned-and-scored={scored_count}")

    replication_paths = ("REPLICATION_COMPARISON.json", "REPLICATION_TABLES_ZH.md", "REPLICATION_ROOT_DIFFERENCES.csv")
    replication_existing = [name for name in replication_paths if (base / name).is_file() and (base / name).stat().st_size > 0]
    comparison = read_optional(base, replication_paths[0], issues)
    analysis_integrity = frozen_analysis_integrity(base, comparison, issues)
    analysis_complete = bool(
        analysis_summary and len(replication_existing) == 3 and isinstance(comparison, dict)
        and analysis_integrity["status"] == "PASS"
        and analysis_integrity["comparison_provenance_verified"] is True
        and (base / replication_paths[1]).stat().st_size > 0
        and (base / replication_paths[2]).stat().st_size > 0
    )
    analysis_attempted = bool(analysis_summary or replication_existing or phase_hint(task, PHASES[5]) != "PENDING")
    analysis_status = status_for_phase(analysis_complete, analysis_attempted,
                                       analysis_integrity["status"] == "FAIL")
    if replication_existing:
        evidence[PHASES[5]].extend(replication_existing)

    phase_statuses = {
        PHASES[0]: status_for_phase(design_ok and lock_ok and receipt_pass, True, frozen_design_block),
        PHASES[1]: source_status,
        PHASES[2]: content_status,
        PHASES[3]: preflight_status,
        PHASES[4]: solver_status,
        PHASES[5]: analysis_status,
    }
    for phase, status in list(phase_statuses.items()):
        hint = phase_hint(task, phase)
        if hint == "BLOCKED" and status != "COMPLETED":
            phase_statuses[phase] = "BLOCKED"
            continue
        if status == "NOT_STARTED" and hint in {"IN_PROGRESS", "COMPLETED"}:
            phase_statuses[phase] = "PARTIAL"
    blocked = frozen_design_block or task.get("status") == "BLOCKED" or any(x == "BLOCKED" for x in phase_statuses.values())
    report_status = derive_overall(phase_statuses, admitted_count, returned_count, scored_count, blocked)
    timing = recorded_timing(base, task, issues)

    failures: dict[str, Any] = {}
    for relative in ("SOURCE_FAILURES.json", "SLOT_FAILURES.json", "execution/RAW_RESPONSE_FREEZE.json"):
        document = read_optional(base, relative, issues)
        if not document:
            continue
        if relative == "SOURCE_FAILURES.json":
            failed = as_rows(document.get("failed_roots"))
            failures[relative] = {"failed_root_count": len(failed), "rows": failed}
        elif relative == "SLOT_FAILURES.json":
            rows = as_rows(document.get("failures"))
            failures[relative] = {"failure_count": document.get("failure_count", len(rows)), "rows": rows}
        else:
            failures[relative] = {"runner_failure": document.get("failure"), "submitted": len(document.get("submitted_ids", [])),
                                  "requests": document.get("requests"), "returned": document.get("returned")}

    usage = collect_usage(base)
    analysis_bindings = {}
    for name in replication_paths:
        path = base / name
        if path.is_file():
            analysis_bindings[name] = bind(path)
    engineering = engineering_disclosures(base, issues)
    planned_source_ids = [normalize_root(root) for root in new_roots]
    source_shortfalls = {"planned_root_count": 16, "selected_root_count": len(source_root_ids),
                         "selected_root_ids": source_root_ids,
                         "missing_root_ids": [root for root in planned_source_ids if root not in source_root_ids],
                         "all_planned_sources_completed": source_complete,
                         "selection_bias_warning": "Downstream paired difficulty uses only real, independently admitted common slots and returned displays. Source or content deficits remain missing and can select the analyzable roots; missing roots are not imputed as zero."}

    report = {
        "schema_version": "fgr20_final_report_v1",
        "overall_status": report_status,
        "objective_completed": report_status == "COMPLETED",
        "training": False,
        "rerun_original_four": False,
        "phases": {phase: {"status": phase_statuses[phase],
                            "elapsed_seconds": timing["per_phase"][phase]["elapsed_seconds"]}
                   for phase in PHASES},
        "timing": timing,
        "total_elapsed_seconds": timing["total_elapsed_seconds"],
        "phase_evidence": evidence,
        "counts": {
            "new_roots": 16, "original_roots": 4, "total_roots": 20,
            "planned_new_semantic": 256, "planned_new_responses": 512,
            "planned_combined_semantic": 320, "planned_combined_responses": 640,
            "new_semantic_admitted": admitted_count,
            "new_responses_returned": returned_count,
            "new_responses_scored": scored_count,
        },
        "new_source_roots_selected": len(source_root_ids),
        "new_source_root_ids": source_root_ids,
        "source_completion_and_selection_limits": source_shortfalls,
        "recovery_source_provenance": recovery_provenance,
        "engineering_disclosures": engineering,
        "content_candidate_count": len(candidate_rows),
        "content_review_count": len(review_rows),
        "content_admitted_candidate_ids": sorted(admitted_ids),
        "input_lock_bindings_verified": input_binding_count,
        "original_integrity_receipt": receipt_binding,
        "independent_closeout_audit": {"path": "INDEPENDENT_CLOSEOUT_AUDIT.json"},
        "closeout_audit_state_at_report_freeze": "PENDING_INDEPENDENT_REVIEW",
        "replication_analysis_bindings": analysis_bindings,
        "frozen_analysis_integrity": analysis_integrity,
        "usage": usage,
        "failures_and_shortfalls": failures,
        "observed_issues": issues,
        "design_limits": [
            "The same neutral R03 source extension is shared by F and G; this design does not estimate a causal benefit of source expansion or feedback training.",
            "The sixteen added Train roots are the frozen replication sample, not claims of M-clean unseen roots.",
            "Question authoring and review use separate fresh Codex-family AI sessions; this is not human review.",
            "Two display permutations are dependent repeated measurements and cannot establish robust generalization across permutations.",
            "The four original roots are read-only discovery data; new16 is primary and combined20 is secondary.",
            "Prospective feedback targets and target realization are not evidence that the original weakness was reproduced.",
        ],
        "frozen_inputs": {"plan": bind(base / "PLAN.json") if (base / "PLAN.json").is_file() else None,
                          "config": bind(base / "CONFIG.json") if (base / "CONFIG.json").is_file() else None,
                          "input_lock": bind(base / "INPUT_LOCK.json") if (base / "INPUT_LOCK.json").is_file() else None},
    }
    if report_status == "COMPLETED":
        report["completion_basis"] = (
            "All observed experiment phases and 256 admitted candidates / 512 returned and scored new requests "
            "meet the frozen denominators. The independent closeout reviewer must still PASS this frozen report "
            "hash before the final manifest validates as COMPLETED."
        )
    else:
        report["completion_basis"] = "One or more frozen phases or planned denominators are incomplete or blocked; missing outputs are not counted as successes."

    md = render_markdown(report, base, failures, usage, issues, after)
    return report, md, {"phases": phase_statuses, "counts": report["counts"], "issues": issues}


def derive_overall(phases: dict[str, str], admitted: int, returned: int, scored: int, blocked: bool) -> str:
    if blocked or any(status == "BLOCKED" for status in phases.values()):
        return "BLOCKED"
    if all(phases[p] == "COMPLETED" for p in PHASES) and (admitted, returned, scored) == (256, 512, 512):
        return "COMPLETED"
    return "PARTIAL"


def collect_usage(base: Path) -> dict[str, Any]:
    result_rows: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    calls_root = base / "cli_calls"
    if calls_root.is_dir():
        for path in sorted(calls_root.glob("*/RESULT.json")):
            try:
                row = read_json(path)
            except (OSError, ValueError):
                continue
            result_rows.append({"call_id": path.parent.name, "receipt_kind": "RESULT", "receipt": row})
        for path in sorted(calls_root.glob("*/attempt_*/TERMINAL.json")):
            try:
                row = read_json(path)
            except (OSError, ValueError):
                continue
            terminal_rows.append({"call_id": path.parents[1].name, "receipt_kind": "TERMINAL",
                                  "receipt_path": str(path.resolve()), "receipt": row})
    return aggregate_usage_records(result_rows, terminal_rows)


def aggregate_usage_records(result_rows: list[dict[str, Any]], terminal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count one usage receipt per call ID, preferring RESULT over TERMINAL."""
    results: dict[str, dict[str, Any]] = {}
    terminals: dict[str, list[dict[str, Any]]] = {}
    for row in result_rows:
        call_id = row.get("call_id")
        if isinstance(call_id, str):
            results[call_id] = row
    for row in terminal_rows:
        call_id = row.get("call_id")
        if isinstance(call_id, str):
            terminals.setdefault(call_id, []).append(row)

    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    known = {key: False for key in total}
    calls = []
    for call_id in sorted(set(results) | set(terminals)):
        result_row = results.get(call_id)
        terminal_options = terminals.get(call_id, [])
        if result_row:
            receipt_row = result_row
            usage_source = "RESULT"
            receipt = result_row.get("receipt", {})
            usage = receipt.get("usage") if isinstance(receipt, dict) else None
            if not isinstance(usage, dict):
                bound_path = receipt.get("terminal_binding", {}).get("path") if isinstance(receipt, dict) else None
                bound = next((item for item in terminal_options
                              if item.get("receipt_path") == bound_path), None)
                if bound:
                    usage = bound.get("receipt", {}).get("usage")
                    usage_source = "RESULT_TERMINAL_FALLBACK"
        else:
            def terminal_key(item: dict[str, Any]) -> tuple[int, str]:
                value = item.get("receipt", {})
                has_usage = isinstance(value.get("usage"), dict)
                return (1 if has_usage else 0, str(value.get("finished_at_utc", "")))
            receipt_row = max(terminal_options, key=terminal_key)
            usage_source = "TERMINAL"
            receipt = receipt_row.get("receipt", {})
            usage = receipt.get("usage") if isinstance(receipt, dict) else None
        receipt = receipt if isinstance(receipt, dict) else {}
        usage = usage if isinstance(usage, dict) else {}
        values = {}
        if isinstance(usage, dict):
            for key in ("input_tokens", "output_tokens"):
                value = usage.get(key)
                if type(value) is int and value >= 0:
                    total[key] += value
                    known[key] = True
                    values[key] = value
            explicit_total = usage.get("total_tokens")
            if type(explicit_total) is int and explicit_total >= 0:
                values["total_tokens"] = explicit_total
                total["total_tokens"] += explicit_total
                known["total_tokens"] = True
            elif type(values.get("input_tokens")) is int and type(values.get("output_tokens")) is int:
                values["total_tokens"] = values["input_tokens"] + values["output_tokens"]
                total["total_tokens"] += values["total_tokens"]
                known["total_tokens"] = True
        backend = receipt.get("backend", {}) if isinstance(receipt.get("backend"), dict) else {}
        calls.append({"call_id": call_id, "receipt_kind": receipt_row.get("receipt_kind"),
                      "usage_source": usage_source, "thread_id": backend.get("thread_id"),
                      "model": backend.get("model"), "model_provider": backend.get("model_provider"),
                      "effort": backend.get("effort"), "usage": values or None})
    return {"completed_call_receipts_observed": len(calls),
            "unique_terminal_call_receipts_observed": len(calls),
            "returned_result_receipts_observed": len(results),
            "terminal_only_call_receipts_observed": len(set(terminals) - set(results)),
            "calls_with_token_usage_observed": sum(row["usage"] is not None for row in calls),
            "receipt_count_policy": "Unique call_id with RESULT or TERMINAL, including returned semantic failures and interrupted/failed terminal-only calls. The legacy completed_call_receipts_observed field counts terminal receipts, not successful content or experimental completion.",
            "usage_snapshot_policy": "Through final-report freeze; the subsequent independent closeout call has its own raw/usage receipt and metadata in the final artifact manifest, and is not retroactively included in this frozen report.",
            "token_totals": {key: total[key] if known[key] else None for key in total},
            "per_call": calls, "cost_reported_by_receipts": False}


def render_markdown(report: dict[str, Any], base: Path, failures: dict[str, Any], usage: dict[str, Any],
                    issues: list[str], after: Any) -> str:
    phases = report["phases"]
    counts = report["counts"]
    lines = [
        "# F/G 20-root 复现最终报告", "",
        f"整体状态：**{report['overall_status']}**。本报告仅按已落盘并可读取的产物计算；未生成、未返回或未评分的项目不会视为成功。", "",
        "本任务固定新增 16 个 Train roots，并只读复用原 4 个 roots。未训练模型，也未重跑原 4 个 root 的评估。新增问题共 256 个计划语义槽，预定 512 次 B1 请求；合计计划 320 个语义题和 640 次展示请求。", "",
        "## 阶段与数量", "",
        f"记录总耗时：**{format_duration(report.get('total_elapsed_seconds'))}**；测量截止 `{report.get('timing', {}).get('measured_through_utc') or '无可用时间戳'}`。{report.get('timing', {}).get('measurement_basis', '')}", "",
        "| 阶段 | 状态 | 记录耗时 | 观察到的证据 |", "| --- | --- | ---: | --- |",
    ]
    evidence = report.get("phase_evidence", {})
    for phase in PHASES:
        items = evidence.get(phase, [])
        detail = "; ".join(str(x).replace("|", "\\|") for x in items) or "没有阶段产物"
        lines.append(f"| `{phase}` | {phases[phase]['status']} | {format_duration(phases[phase].get('elapsed_seconds'))} | {detail} |")
    unavailable = report.get("timing", {}).get("unavailable_phase_boundaries", [])
    if unavailable:
        lines += ["", "未记录足够的阶段起止时间戳，因此这些阶段耗时留空：" + ", ".join(f"`{phase}`" for phase in unavailable) + "."]
    lines += ["", "| 数量 | 计划 | 实际可核验 |", "| --- | ---: | ---: |",
              f"| 新增 roots | 16 | {report.get('new_source_roots_selected', 0)} 最终来源被选中 |",
              f"| 恢复来源 roots | 未并入 canonical | {report.get('recovery_source_provenance', {}).get('verified_selection_root_count', 0)}（独立 provenance） |",
              f"| 新增语义题 admitted | 256 | {counts['new_semantic_admitted']} |",
              f"| 新增 Solver 返回 | 512 | {counts['new_responses_returned']} |",
              f"| 新增 Solver 已评分 | 512 | {counts['new_responses_scored']} |",
              "| 原始 roots | 4 | 只读复用 |", "| 合计 roots | 20 | 4 原始 + 16 新增 |", ""]
    shortfalls = report.get("source_completion_and_selection_limits", {})
    if shortfalls.get("selected_root_count", 0) < shortfalls.get("planned_root_count", 16):
        lines += [f"来源阶段实际只有 **{shortfalls.get('selected_root_count', 0)}/16** 个计划 root 的最终共享来源被选中；其余 {len(shortfalls.get('missing_root_ids', []))} 个 root 保留失败或缺额，不能把后续有效子集视为完整新增 16-root 复现。缺额 root：" + ", ".join(f"`{root}`" for root in shortfalls.get("missing_root_ids", [])) + "。",
                  "来源或内容缺额会选择后续可分析 root；配对统计只使用真实通过内容审核且两臂各两排列已返回的共同槽，缺失 root/槽不填零。", ""]
    engineering = report.get("engineering_disclosures", {})
    shape = engineering.get("plan_schema_recovery", {})
    runtime = engineering.get("runtime_interpreter_adaptation")
    lines += ["## 工程恢复与运行环境", "",
              "管理脚本仅组织调用、保全原始产物、做机械 schema/路径适配和统计聚合；这些操作不构成来源、题目、Gold 或审核的语义 PASS。语义内容和准入判断仍须来自实际保存的独立 fresh Codex 调用与内容 gate。", ""]
    if shape.get("records"):
        lines += [f"实际记录了 {shape.get('unique_root_arm_count', 0)} 个 root/arm 的计划结构适配，保留 {shape.get('observed_stage_count', 0)} 份工程阶段记录、共 {shape.get('structural_change_count', 0)} 项已记录逐字段映射。同一原始计划在不同工程恢复目录可重复保存映射记录，这不增加计划调用数。原始 planner 响应、旧计划 freeze 和旧 OUTCOME 均保留；每项映射同时记录原值与规范值。规范化只处理已有作者值的表示形式或明确枚举别名，不等于新计划调用、触发已实现或内容有效。", "",
                  "| root / arm | 结构变更 | 原产物绑定核验 | 严格 schema 错误 |",
                  "| --- | ---: | ---: | --- |"]
        for row in shape["records"]:
            lines.append(f"| `{row.get('root_id')}` / {row.get('arm')} | {row.get('structural_change_count')} | {row.get('original_artifact_bindings_verified')}/{row.get('original_artifact_reference_count')} | `{json.dumps(row.get('strict_validation_errors'), ensure_ascii=False)}` |")
        lines.append("")
    else:
        lines += ["当前未观察到已保存的计划结构适配记录；不据此假定出题阶段成功。", ""]
    consumed = engineering.get("consumed_plan_failures_without_replanning", [])
    if consumed:
        lines += ["已发生且已消耗预算的 planner 失败单独保留，不重开计划或把语义失败改记为 transport retry：", "",
                  "| root / arm | 保存的处置 | 新计划调用 | 新候选调用 | 失败产物绑定核验 |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for row in consumed:
            lines.append(f"| `{row.get('root_id')}` / {row.get('arm')} | `{row.get('recorded_status')}` | {row.get('new_plan_calls')} | {row.get('new_candidate_calls')} | {row.get('verified_failure_artifact_count')}/{row.get('actual_failure_artifact_count')} |")
        lines += ["", "这些 root/arm 的空最终候选/审核集合仅记录失败后的真实完成面，不是伪造题目，也不会成为内容 PASS。v2 工程说明和实际 `DRIVER_ADAPTATION_V2.json` 同旧版本一并保留。", ""]
    draft_shape = engineering.get("same_initial_draft_serialization_recovery", {})
    if draft_shape.get("records"):
        lines += [f"另有 {draft_shape.get('observed_root_arm_count')} 个 root/arm 对同一已返回初稿进行可逆序列化结构恢复。原始 author 响应、原形状门失败与规范化日志独立保留；恢复不增加初稿数、题目槽分母或语义修订次数。初稿准入率使用该同一初稿规范化之后保存的首次初审表面与机械门；没有保存的初审表面仍为未知，已记录的缺失/无效独立审核 HOLD 只表示初稿未准入，不声称题义为假，也不填成 Solver 零分。", "",
                  "| root / arm | 原始初稿回执引用 | 引用绑定核验 | 原序列化失败表面数 |",
                  "| --- | --- | --- | ---: |"]
        for row in draft_shape["records"]:
            reference = row.get("original_draft_reference_binding") or {}
            lines.append(f"| `{row.get('root_id')}` / {row.get('arm')} | `{reference.get('path')}` | {row.get('original_reference_binding_verified')} | {len(row.get('raw_initial_serialization_failure_surfaces') or [])} |")
        lines.append("")
    review_shape = engineering.get("runtime_review_and_candidate_schema_adaptation", {})
    if review_shape.get("records"):
        lines += [f"后续 RAM 适配另外保存 {review_shape.get('recorded_normalization_surface_count')} 份候选/审核 schema 规范化表面：只按日志记录原值和规范值，不据此产出 Gold 或语义 PASS。既有初审 raw、validation、HOLD 和已消耗的唯一实质修订保留；修订后的审核不能回填初稿统计。", "",
                  "| 实际阶段目录 | 类型 / 阶段 | 记录映射数 | 引用绑定核验 | 适配器消耗语义预算 |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for row in review_shape["records"]:
            lines.append(f"| `{row.get('actual_stage_directory')}` | {row.get('kind')} / {row.get('stage_suffix')} | {row.get('recorded_change_count')} | {row.get('verified_reference_count')}/{row.get('recorded_reference_count')} | {row.get('semantic_calls_or_revision_budget_consumed_by_adapter')} |")
        lines.append("")
    launch_provenance = engineering.get("actual_schema_adapter_launch_provenance", {})
    if launch_provenance.get("records"):
        lines += ["实际 RAM 适配器启动凭据按版本逐次保留。v2 仅适用于事前批准的四个未启动阶段，先前 F179186 的 v1 工作保留；启动凭据及其显式嵌套绑定在内容冻结前逐一核验，工程回执不代表语义准入。", "",
                  "| 适配器 | driver / root / arm | 启动凭据 | 绑定核验 | 记录额外语义预算 |",
                  "| --- | --- | --- | ---: | ---: |"]
        for row in launch_provenance["records"]:
            lines.append(f"| `{row.get('adapter_family')}` | {row.get('driver')} / `{row.get('root_id')}` / {row.get('arm')} | `{row.get('launch_receipt_binding', {}).get('path')}` | {row.get('verified_reference_count')}/{row.get('reference_count')} | {row.get('extra_semantic_budget_reported')} |")
        lines.append("")
    saved_revision = engineering.get("saved_final_revision_gold_first_verifier", {})
    if saved_revision.get("records"):
        lines += ["已保存的最终修订独立 Gold 如仅存在单独绑定批准的精确、可逆 schema 变体，工程续接只允许其尚未使用的首次 fresh verifier；包括把既有 call-global 完整读取/未读 Solver 声明的相同值复制到原本缺字段的各行，原声明和全部 Gold/evidence 值保留。工程不重建 Gold、不重开作者或计划，也不增加第二次修订。下表只记录真实保存的准备/调用/最终门产物；没有最终门时保持未观察，不补 PASS。初稿的旧失败和产出率不被修订核验覆盖。", "",
                  "| root / arm | 保存的 Gold call | 新 verifier 回执 | 最终审核阶段 | 发布真实题/审核 | 最终机械准入槽 | 输入绑定核验 |",
                  "| --- | --- | --- | --- | ---: | ---: | ---: |"]
        for row in saved_revision["records"]:
            lines.append(f"| `{row.get('root_id')}` / {row.get('arm')} | `{row.get('saved_gold_call_id')}` | {row.get('new_verifier_return_receipt_binding') is not None} | {row.get('terminal_review_stage')} | {row.get('published_final_candidate_count')}/{row.get('published_final_review_count')} | {row.get('recorded_final_gate_admitted_slots')} | {row.get('verified_input_binding_count')}/{row.get('input_binding_count')} |")
        lines.append("")
        for row in saved_revision["records"]:
            if row.get("uncovered_slot_ids"):
                lines.append(f"未覆盖槽 `{row.get('root_id')}` / {row.get('arm')}: `{row.get('uncovered_slot_ids')}`；原失败/HOLD 保留声明={row.get('uncovered_original_failure_and_HOLD_retained_reported')}，原候选字节归档=`{row.get('original_failure_archive_binding')}`。未生成题、Gold 或审核补位，也不进入 Solver 题数。")
        lines.append("")
    if isinstance(runtime, dict):
        adaptation = runtime.get("adaptation") or {}
        settings = runtime.get("recorded_settings") or {}
        lines += [f"运行环境回执 `{runtime.get('recorded_status')}`：使用 `{adaptation.get('python_executable')}`，只读导入 `{adaptation.get('readonly_dependency_path')}`；内存适配解释器前缀守卫和实际 structured-API 文件路径，缓存定向到本任务目录。代码绑定核验 {runtime.get('verified_code_binding_count')}/{runtime.get('code_binding_count')}。回执记录未安装/更新依赖、未改冻结文件，也未改科学模型或解码设置。", "",
                  f"固定设置按该回执逐项披露：`{json.dumps(settings, ensure_ascii=False, sort_keys=True)}`。环境 CPU 检查自身 model_calls=`{runtime.get('model_calls_by_cpu_check')}`、GPU model started=`{runtime.get('gpu_model_started_by_cpu_check')}`；真实 Solver 返回数仍以本报告的 raw/score 产物为准。", ""]
    lines += ["工程映射、运行环境、CPU 预算预检及续接目录的实际回执/脚本均由最终内容 freeze 与 artifact manifest 保留。原始失败和缺额不会因为工程恢复而被删除。", ""]
    lines += ["## 复现结果", ""]
    table_path = base / "REPLICATION_TABLES_ZH.md"
    if table_path.is_file() and table_path.stat().st_size:
        lines += ["以下为实际统计脚本生成的完整 F/G 对比表：", "", table_path.read_text(encoding="utf-8").rstrip(), ""]
    else:
        lines += ["统计表尚未生成，因此本报告不推断 F/G 难度差、置信区间或子组结果。", ""]
    comparison = base / "REPLICATION_COMPARISON.json"
    if comparison.is_file():
        try:
            comp = read_json(comparison)
            original_rows = comp.get("cohorts", {}).get("original4", {}).get("groups", {}).get("NON_MC6_PRIMARY", {}).get("per_root", [])
            if original_rows:
                def displayed_rate(value: Any) -> str:
                    return "—" if type(value) not in (int, float) else f"{100 * value:.2f}%"
                def displayed_pp(value: Any) -> str:
                    return "—" if type(value) not in (int, float) else f"{100 * value:+.2f}"
                lines += ["原 4 个 discovery roots 的冻结主比较逐 root 结果（直接列出统计产物字段）：", "",
                          "| root | 共同主槽/7 | F exact | G exact | F−G pp |",
                          "| --- | ---: | ---: | ---: | ---: |"]
                for row in original_rows:
                    lines.append(f"| `{row.get('root_id')}` | {row.get('eligible_common_slots')}/7 | {displayed_rate(row.get('F_strict_exact'))} | {displayed_rate(row.get('G_strict_exact'))} | {displayed_pp(row.get('F_minus_G_strict_exact'))} |")
                lines.append("")
            limits = comp.get("interpretation_limits") if isinstance(comp, dict) else None
            if limits:
                lines += ["分析脚本的解释边界：", "", *(f"- {x}" for x in limits if isinstance(x, str)), ""]
        except (OSError, ValueError):
            lines += ["REPLICATION_COMPARISON.json 存在但无法解析；相关结果不在此报告复述。", ""]
    lines += ["## 缺额与失败", ""]
    if failures:
        for path, detail in failures.items():
            lines.append(f"- `{path}`：`{json.dumps(detail, ensure_ascii=False, sort_keys=True)}`")
    else:
        lines.append("当前未观察到独立失败记录；这不替代各阶段产物的完整性检查。")
    if issues:
        lines += ["", "本次观察到的完整性或解析问题：", "", *(f"- `{issue}`" for issue in issues)]
    lines += ["", "## 来源、审核与解释限制", "",
              "F 与 G 共用同一份中性 R03 扩充来源；因此这个实验不能把差异归因于来源扩充的因果收益或训练收益。新增 roots 是按冻结名单选出的 Train roots，并非未见于 M-clean 的 roots。题目作者与审核者通过隔离的新 Codex-family AI 会话执行，不是人工审核。", "",
              "两个固定展示排列是同一道语义题的重复测量，不能证明弱点在任意排列下都稳健。Gold 判断与最终选择不一致不应简单归为 box 格式问题。F 卡只形成事前目标；目标命中不等于已复现原始弱点，也不证明全部卡片都被正式复现。原 4 roots 是 discovery 数据，新 16 为主比较，20-root 合并分析为次要结果。", "",
              f"截至报告冻结可读取到 {usage['unique_terminal_call_receipts_observed']} 个唯一 call_id 的终态回执（包括失败和中断）：其中 RESULT {usage['returned_result_receipts_observed']} 个、仅 TERMINAL {usage['terminal_only_call_receipts_observed']} 个，{usage['calls_with_token_usage_observed']} 个有可读取 token 用量。token 汇总为 `{json.dumps(usage['token_totals'], ensure_ascii=False)}`；终态回执数量不代表内容通过或实验完成。回执未提供可验证费用字段，因此费用没有推算。后续独立收口审计的用量单独保存在其实际调用回执和 metadata，并由最终 manifest 绑定，不回写这份冻结报告。", "",
              "原实验完整性复核 receipt："]
    if isinstance(after, dict):
        lines.append(f"`{after.get('status')}`，verified_artifacts=`{after.get('verified_artifacts')}`，unchanged=`{after.get('unchanged')}`。")
    else:
        lines.append("没有可读取的最终原实验完整性 receipt。")
    lines += ["", "报告仅记录独立 closeout audit 的路径。关闭审计必须核对本报告的精确 SHA256；`FINAL_ARTIFACT_MANIFEST.json` 只应在审计文件写入且其 `report_sha256` 与本报告完全匹配后生成。管理用 TASK_STATUS、PROGRESS、SUPERVISOR_STATUS、process_registry 与 worker_events 可继续写入；收口 manifest 绑定其逐文件原始字节快照 `FINAL_RUNTIME_SNAPSHOT.json`，不绑定这些活路径。原记录保留，实际 CLI 请求、raw、events、terminal 和 usage 回执另行封存。快照和后续审计用量不会回写本报告。", ""]
    return "\n".join(lines)


def format_duration(seconds: Any) -> str:
    if type(seconds) not in (int, float) or seconds < 0:
        return "未记录"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}小时{minutes}分{secs}秒" if hours else f"{minutes}分{secs}秒"


def include_manifest_path(relative: Path) -> bool:
    """Management event streams stay live after a finalizer call returns."""
    return not (any(part in TRANSIENT_PARTS or part == "__pycache__" for part in relative.parts)
                or relative.suffix == ".pyc" or relative.name in TRANSIENT_NAMES
                or relative.name.endswith(".tmp"))


def runtime_snapshot_path(relative: Path) -> bool:
    return (any(part in TRANSIENT_PARTS for part in relative.parts)
            or relative.name in TRANSIENT_NAMES - {"FINAL_ARTIFACT_MANIFEST.json"}) and not (
                "__pycache__" in relative.parts or relative.suffix == ".pyc" or relative.name.endswith(".tmp"))


def capture_runtime_snapshot(base: Path) -> dict[str, Any]:
    """Preserve live management records without binding their future bytes."""
    path = base / RUNTIME_SNAPSHOT_NAME
    if path.is_file():
        saved = read_json(path)
        if saved.get("schema_version") != "fgr20_final_runtime_byte_snapshot_v1":
            raise RuntimeError("FINAL_RUNTIME_SNAPSHOT_SCHEMA_DRIFT")
        for row in saved.get("records", []):
            raw = base64.b64decode(row["content_base64"], validate=True)
            if len(raw) != row["bytes"] or sha256_bytes(raw) != row["sha256"]:
                raise RuntimeError("FINAL_RUNTIME_SNAPSHOT_CONTENT_DRIFT")
        return bind(path)
    live = []
    for registry in sorted((base / "process_registry").glob("*.json")):
        row = read_json(registry)
        pid, token = row.get("pid"), row.get("start_token")
        if row.get("active") is not True or type(pid) is not int or token is None:
            continue
        try:
            actual = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        except (OSError, IndexError):
            continue
        if actual == str(token):
            live.append({"call_id": row.get("call_id"), "pid": pid, "start_token": str(token)})
    if live:
        raise RuntimeError(f"FINAL_RUNTIME_SNAPSHOT_REQUIRES_NO_LIVE_SEMANTIC_CALLS:{live}")
    records = []
    for source in sorted(base.rglob("*"), key=str):
        if not source.is_file() or not runtime_snapshot_path(source.relative_to(base)):
            continue
        raw = source.read_bytes()
        records.append({"original_relative_path": str(source.relative_to(base)),
                        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "sha256": sha256_bytes(raw), "bytes": len(raw),
                        "content_base64": base64.b64encode(raw).decode("ascii")})
    snapshot = {"schema_version": "fgr20_final_runtime_byte_snapshot_v1",
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "records": records, "files": len(records), "total_source_bytes": sum(row["bytes"] for row in records),
        "original_files_modified_or_removed": False,
        "policy": "Exact per-file byte observations of live management records, including task/supervisor/progress/process registry and worker events. Captures are sequential, not one globally atomic instant; a live log is the captured prefix. Original files remain in place and may continue to append. Only this immutable snapshot is bound by the final manifest; actual immutable CLI request/raw/event/terminal/usage receipts are separately bound.",
        "no_live_semantic_calls_at_snapshot_start": True,
        "report_json_or_usage_rewritten": False}
    return publish_immutable(path, snapshot)


def manifest_artifact_paths(base: Path) -> list[Path]:
    paths = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        if not include_manifest_path(relative):
            continue
        paths.append(path.resolve())
    return sorted(set(paths), key=str)


def finalize_manifest(base: Path) -> dict[str, Any]:
    report_path = base / "FINAL_REPORT.json"
    markdown_path = base / "FINAL_REPORT_ZH.md"
    audit_path = base / "INDEPENDENT_CLOSEOUT_AUDIT.json"
    report = read_json(report_path)
    reference = report.get("independent_closeout_audit")
    if not isinstance(reference, dict) or set(reference) != {"path"}:
        raise ValueError("REPORT_AUDIT_REFERENCE_MUST_BE_PATH_ONLY")
    expected_path = Path(reference["path"])
    if not expected_path.is_absolute():
        expected_path = base / expected_path
    if expected_path.resolve() != audit_path.resolve() or not audit_path.is_file():
        raise ValueError("CLOSEOUT_AUDIT_NOT_PRESENT_AT_FROZEN_PATH")
    audit = read_json(audit_path)
    report_binding = bind(report_path)
    if audit.get("report_sha256") != report_binding["sha256"]:
        raise ValueError("CLOSEOUT_AUDIT_REPORT_HASH_MISMATCH")
    if audit.get("verdict") not in {"PASS", "FAIL"}:
        raise ValueError("CLOSEOUT_AUDIT_VERDICT_INVALID")
    if type(audit.get("independent_of_report_author")) is not bool:
        raise ValueError("CLOSEOUT_AUDIT_INDEPENDENCE_FIELD_INVALID")
    if not isinstance(audit.get("checks"), (list, dict)) or not isinstance(audit.get("issues"), (list, dict)):
        raise ValueError("CLOSEOUT_AUDIT_CHECKS_OR_ISSUES_INVALID")
    if not markdown_path.is_file() or markdown_path.stat().st_size == 0:
        raise ValueError("FINAL_MARKDOWN_MISSING")
    validator = validator_module(base)
    validator.verify_input_lock(base)
    existing_manifest = base / "FINAL_ARTIFACT_MANIFEST.json"
    if existing_manifest.is_file():
        result = validator.validate(base)
        return {"manifest": bind(existing_manifest), "idempotent_reuse": True,
                "validator": {key: value for key, value in result.items() if key != "report"}}
    runtime_snapshot_binding = capture_runtime_snapshot(base)
    artifacts = [bind(path) for path in manifest_artifact_paths(base)]
    required = {report_path.resolve(), markdown_path.resolve(), base.joinpath("PLAN.json").resolve(),
                base.joinpath("CONFIG.json").resolve(), base.joinpath("SELECTION_FREEZE.json").resolve(),
                base.joinpath("INPUT_LOCK.json").resolve(), audit_path.resolve()}
    actual_paths = {Path(row["path"]).resolve() for row in artifacts}
    if not required <= actual_paths:
        raise ValueError("FINAL_MANIFEST_REQUIRED_FILE_MISSING")
    receipt_ref = report.get("original_integrity_receipt", {})
    receipt_path = Path(receipt_ref.get("path", ""))
    if not receipt_path.is_absolute():
        receipt_path = base / receipt_path
    if receipt_path.resolve() not in actual_paths:
        raise ValueError("FINAL_MANIFEST_ORIGINAL_RECEIPT_MISSING")
    manifest = {"schema_version": "fgr20_final_artifact_manifest_v1", "artifacts": artifacts}
    out = base / "FINAL_ARTIFACT_MANIFEST.json"
    publish_immutable(out, manifest)
    result = validator.validate(base)
    return {"manifest": bind(out), "runtime_snapshot": runtime_snapshot_binding,
            "validator": {key: value for key, value in result.items() if key != "report"}}


def self_check() -> dict[str, Any]:
    fixture = {
        "all_complete": {phase: "COMPLETED" for phase in PHASES},
        "partial_solver": {phase: "COMPLETED" for phase in PHASES},
        "not_started": {phase: "NOT_STARTED" for phase in PHASES},
    }
    partial = dict(fixture["partial_solver"])
    partial[PHASES[4]] = "PARTIAL"
    recovery_fixture = summarize_recovery_sources(
        {"status": "PARTIAL", "passed_root_count": 1},
        {"roots": [{"root_id": "cc-train-episode-fixture"}]},
        [{"root_id": "cc-train-episode-fixture"}],
        ["cc-train-episode-fixture"],
    )
    usage_fixture = aggregate_usage_records(
        [{"call_id": "same-call", "receipt_kind": "RESULT", "receipt": {
            "backend": {"thread_id": "result-thread", "model": "fixture"},
            "usage": {"input_tokens": 10, "output_tokens": 3},
        }}],
        [{"call_id": "same-call", "receipt_kind": "TERMINAL", "receipt_path": "/fixture/same-call/TERMINAL.json", "receipt": {
            "backend": {"thread_id": "terminal-thread"}, "usage": {"input_tokens": 100, "output_tokens": 30},
        }}, {"call_id": "terminal-only", "receipt_kind": "TERMINAL", "receipt_path": "/fixture/terminal-only/TERMINAL.json", "receipt": {
            "backend": {"thread_id": "terminal-only-thread"}, "usage": {"input_tokens": 7, "output_tokens": 2},
        }}],
    )
    integrity_fixture = {"status": "PASS", "unchanged": True,
                         "original_root_ids": ["218519", "222937", "225575", "229686"],
                         "original_manifest_binding": {"path": "/fixture/FINAL_MANIFEST.json", "sha256": "fixture", "bytes": 1},
                         "verified_artifacts": 772, "verification_elapsed_seconds": 1.0}
    matching_after = {**integrity_fixture, "verification_elapsed_seconds": 2.0}
    checks = {
        "completed_requires_every_phase_and_denominator": derive_overall(fixture["all_complete"], 256, 512, 512, False) == "COMPLETED",
        "partial_if_any_phase_incomplete": derive_overall(partial, 256, 512, 512, False) == "PARTIAL",
        "partial_on_missing_semantic_slot": derive_overall(fixture["all_complete"], 255, 512, 512, False) == "PARTIAL",
        "partial_on_missing_return": derive_overall(fixture["all_complete"], 256, 511, 512, False) == "PARTIAL",
        "blocked_on_integrity_failure": derive_overall(fixture["all_complete"], 256, 512, 512, True) == "BLOCKED",
        "utc_timestamp_parsing": parse_timestamp("2026-10-24T18:48:28.295796+00:00") == dt.datetime(2026, 10, 24, 18, 48, 28, 295796, tzinfo=dt.timezone.utc),
        "duration_formatting": format_duration(3661.9) == "1小时1分1秒" and format_duration(None) == "未记录",
        "recovery_provenance_is_separate": recovery_fixture["verified_selection_root_count"] == 1 and recovery_fixture["counted_as_canonical_source_selection"] is False,
        "usage_prefers_result_without_double_counting": usage_fixture["completed_call_receipts_observed"] == 2 and usage_fixture["token_totals"] == {"input_tokens": 17, "output_tokens": 5, "total_tokens": 22},
        "before_after_receipts_use_actual_after_object": matching_original_receipts(integrity_fixture, matching_after),
        "receipt_reverification_duration_does_not_drift_identity": integrity_receipt_identity(integrity_fixture) == integrity_receipt_identity(matching_after),
        "receipt_artifact_count_drift_rejected": not matching_original_receipts(integrity_fixture, {**matching_after, "verified_artifacts": 771}),
        "live_worker_stream_excluded_but_semantic_calls_bound": not include_manifest_path(Path("worker_events/round_02.jsonl")) and include_manifest_path(Path("cli_calls/fresh_call/attempt_01/EVENTS.jsonl")),
        "live_task_supervisor_registry_excluded_but_exact_snapshot_bound": all(not include_manifest_path(Path(name)) for name in ("TASK_STATUS.json", "PROGRESS.jsonl", "SUPERVISOR_STATUS.json", "process_registry/fresh_call.json")) and include_manifest_path(Path(RUNTIME_SNAPSHOT_NAME)) and runtime_snapshot_path(Path("process_registry/fresh_call.json")),
        "report_json_round_trip": json.loads(json_bytes({"é": "实际观察"}).decode("utf-8")) == {"é": "实际观察"},
        "engineering_mapping_counts_do_not_adjudicate_semantics": summarize_shape_changes([
            {"mapping": "shape_alias", "semantic_acceptance": False},
            {"mapping": "shape_alias", "semantic_acceptance": False},
            {"mapping": "canonical_json", "semantic_acceptance": False}]) == {"canonical_json": 1, "shape_alias": 2},
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"status": "PASS", "checks": checks, "files_written": False, "model_calls": 0}


def write_report(base: Path) -> dict[str, Any]:
    fresh_receipt = actual_original_receipt(base)
    receipt_path = base / "ORIGINAL_INTEGRITY_AFTER.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if integrity_receipt_identity(receipt) != integrity_receipt_identity(fresh_receipt):
            raise RuntimeError("ORIGINAL_INTEGRITY_CHANGED_AFTER_RECEIPT_FREEZE")
        receipt_binding = bind(receipt_path)
    else:
        receipt_binding = publish_immutable(receipt_path, fresh_receipt)
    build_path = base / "FINAL_REPORT_BUILD.json"
    if build_path.exists():
        build = read_json(build_path)
        if (build.get("assembler_binding") != bind(Path(__file__))
                or build.get("original_integrity_receipt") != receipt_binding):
            raise RuntimeError("FROZEN_REPORT_BUILD_PROVENANCE_DRIFT")
        report, markdown, diagnostics = build["report"], build["markdown"], build["diagnostics"]
    else:
        if (base / "FINAL_REPORT.json").exists() or (base / "FINAL_REPORT_ZH.md").exists():
            raise RuntimeError("FROZEN_REPORT_EXISTS_WITHOUT_BUILD_RECEIPT")
        report, markdown, diagnostics = observe(base, receipt_binding)
        build = {"schema_version": "fgr20_report_build_freeze_v1",
                 "assembler_binding": bind(Path(__file__)),
                 "original_integrity_receipt": receipt_binding,
                 "report": report, "markdown": markdown, "diagnostics": diagnostics,
                 "model_calls": 0,
                 "purpose": "Write-once exact report payload for interrupted publication recovery; no semantic audit verdict is generated here."}
        publish_immutable(build_path, build)
    report_binding = publish_immutable(base / "FINAL_REPORT.json", report)
    markdown_binding = publish_text_immutable(base / "FINAL_REPORT_ZH.md", markdown)
    return {"status": report["overall_status"], "report": report_binding,
            "markdown": markdown_binding, "receipt": receipt_binding,
            "build_freeze": bind(build_path),
            "diagnostics": diagnostics, "manifest": "PENDING_CLOSEOUT_AUDIT"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-check", action="store_true")
    group.add_argument("--write-report", action="store_true")
    group.add_argument("--finalize-manifest", action="store_true")
    args = parser.parse_args()
    base = args.base.resolve()
    if args.self_check:
        result = self_check()
    elif args.write_report:
        result = write_report(base)
    else:
        result = finalize_manifest(base)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if "validator" in result:
        return 0 if result["validator"].get("valid") is True else 2
    return 0 if result.get("status") in ("PASS", "COMPLETED", "PARTIAL", "BLOCKED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
