"""Resolve verified QA artifacts from normal or recovered Memorax runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


QA_SOURCE_ADAPTER_VERSION = "memorax_qa_source_adapter_v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def resolve_qa_artifacts(
    *, source_run: Path, memory_artifacts: Path,
) -> dict[str, Any]:
    """Return one normalized, read-only QA handoff.

    A normal successful single-root run owns all artifacts itself.  A run whose
    asynchronous Child Add completed outside the original polling window uses
    a separate recovery directory for only the Memory overview and readiness
    attestation.  Both routes expose the same downstream contract.
    """
    expanded_path = source_run / "expanded_dialogue.json"
    overview_path = memory_artifacts / "memory/post_generation_overview.json"
    prompt_manifest_path = memory_artifacts / "qa/mcq_prompt_manifest.json"
    legacy_prompt_path = memory_artifacts / "qa/mcq_prompt.json"
    recovery_report = (
        memory_artifacts / "reports/MEMORAX_ASYNC_RECOVERY_REPORT.json")
    normal_report = (
        memory_artifacts / "reports/SIMPLE_APPEND_MEMORAX_LIVE_REPORT.json")

    if not expanded_path.is_file():
        raise FileNotFoundError(f"expanded_dialogue_missing:{expanded_path}")
    if not overview_path.is_file():
        raise FileNotFoundError(f"memory_overview_missing:{overview_path}")
    if prompt_manifest_path.is_file():
        prompt_artifact_path = prompt_manifest_path
    elif legacy_prompt_path.is_file():
        # The first batch wrote the complete prompt envelope as JSON.  Accept
        # it read-only for recovery, while all new runs write the split .txt
        # plus manifest contract.
        prompt_artifact_path = legacy_prompt_path
    else:
        raise FileNotFoundError(
            f"mcq_prompt_artifact_missing:{prompt_manifest_path}")

    if recovery_report.is_file():
        report = _load(recovery_report)
        source_kind = "async_recovery"
        # Recovery is a Memory-only handoff.  A completed Memory write is
        # useful metadata, but is not required for the complete-dialogue QA
        # source.  This branch still requires the recovery artifacts and QA
        # readiness, and preserves the write status for auditing.
        verified = all((
            report.get("status") == "completed",
            report.get("qa_prompt_ready") is True,
        ))
        readiness = {
            "generation_ready": bool(report.get("generation_ready")),
            "memory_write_verified": bool(report.get("memory_write_verified")),
            "memory_context_degraded": bool(report.get("memory_context_degraded", False)),
            "qa_prompt_ready": bool(report.get("qa_prompt_ready")),
        }
        report_path = recovery_report
    elif normal_report.is_file() and memory_artifacts.resolve() == source_run.resolve():
        report = _load(normal_report)
        source_kind = "normal_single_root"
        integrity = report.get("integrity") or {}
        verified = all((
            report.get("status") == "completed",
            report.get("generation_ready") is True,
            report.get("qa_prompt_ready") is True,
            bool(integrity) and all(bool(value) for value in integrity.values()),
        ))
        readiness = {
            "generation_ready": bool(report.get("generation_ready")),
            "memory_write_verified": bool(
                report["memory_write_verified"]
                if "memory_write_verified" in report
                else report.get("failure_stage") is None),
            "memory_context_degraded": bool(report.get("memory_context_degraded", False)),
            "qa_prompt_ready": bool(report.get("qa_prompt_ready")),
        }
        report_path = normal_report
    else:
        raise FileNotFoundError(
            "verified_memorax_qa_source_report_missing")

    if not verified:
        raise ValueError(f"memorax_qa_source_not_verified:{source_kind}")
    return {
        "qa_source_adapter_version": QA_SOURCE_ADAPTER_VERSION,
        "source_kind": source_kind,
        "expanded_dialogue_path": expanded_path,
        "memory_overview_path": overview_path,
        "mcq_prompt_manifest_path": prompt_artifact_path,
        "readiness_report_path": report_path,
        "readiness": readiness,
    }
