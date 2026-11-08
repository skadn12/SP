"""Retry the two quarantined feedback R02 roots without Memorax writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_PLAN_ROOT = RUN_BASE / "h4_round1_feedback_r02_batch_v1_live_4"
DEFAULT_BOOTSTRAP_ROOT = (
    RUN_BASE / "h4_round1_feedback_r02_batch_execution_v1_live_2")
DEFAULT_QUARANTINE_REPORT = (
    RUN_BASE
    / "h4_round1_paired_blind_fresh_qa_v1_live_8"
    / "final_paired_qa_batch_report.json"
)
DEFAULT_OUTPUT_ROOT = (
    RUN_BASE
    / "h4_round1_feedback_r02_failed2_retry_no_memorax_v1_live_1"
)
DEFAULT_CREDENTIALS = Path("/root/.config/memorax/credentials.env")
RETRY_ROOT_IDS = (
    "cc-train-episode-218519",
    "cc-train-episode-259192",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _plan_rows(plan_root: Path) -> dict[str, dict[str, Any]]:
    path = plan_root / "r02_feedback_execution_plan.jsonl"
    rows = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    selected = {
        str(row.get("root_id") or ""): row
        for row in rows
        if str(row.get("root_id") or "") in RETRY_ROOT_IDS
    }
    if set(selected) != set(RETRY_ROOT_IDS):
        raise RuntimeError("retry_roots_missing_from_feedback_plan")
    return selected


def _preflight(*, plan_root: Path, bootstrap_root: Path,
               quarantine_report: Path, output_root: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "output_root_absent": not output_root.exists(),
        "plan_exists": (plan_root / "r02_feedback_execution_plan.jsonl").is_file(),
        "quarantine_report_exists": quarantine_report.is_file(),
        "memorax_write_mode_disabled": True,
        "exactly_two_retry_roots": len(RETRY_ROOT_IDS) == 2,
    }
    details: dict[str, Any] = {}
    rows: dict[str, dict[str, Any]] = {}
    try:
        rows = _plan_rows(plan_root)
        quarantined = tuple(_load(quarantine_report).get("quarantine_roots") or [])
        checks["retry_roots_match_quarantine"] = set(quarantined) == set(
            RETRY_ROOT_IDS)
        details["quarantine_roots"] = list(quarantined)
        for root_id, row in rows.items():
            dialogue = Path(str(row.get("canonical_dialogue_path") or ""))
            bootstrap = (
                bootstrap_root / "roots" / root_id
                / "final_r01_bootstrap_report.json")
            checks[f"{root_id}:canonical_dialogue_exists"] = dialogue.is_file()
            checks[f"{root_id}:canonical_dialogue_hash"] = (
                dialogue.is_file()
                and _sha(dialogue) == row.get("canonical_dialogue_sha256"))
            checks[f"{root_id}:bootstrap_pass"] = (
                bootstrap.is_file() and _load(bootstrap).get("status") == "PASS")
    except Exception as error:
        details["preflight_error"] = f"{type(error).__name__}:{error}"
        checks["source_contract_valid"] = False
    return {
        "preflight_version": "h4_round1_feedback_r02_failed2_no_memorax_v1",
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "details": details,
        "retry_root_ids": list(RETRY_ROOT_IDS),
        "no_memorax_writes": True,
        "memorax_add_budget": 0,
        "output_root": str(output_root.resolve()),
    }


def _root_success(root_output: Path) -> tuple[bool, dict[str, Any]]:
    report_path = root_output / "reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json"
    audit_path = root_output / "no_memorax_write_audit.json"
    payload_path = root_output / "feedback_payload_audit.json"
    if not all(path.is_file() for path in (report_path, audit_path, payload_path)):
        return False, {"reason": "required_result_artifact_missing"}
    report = _load(report_path)
    audit = _load(audit_path)
    payload = _load(payload_path)
    add_records = [
        row for row in report.get("memorax_call_records") or []
        if isinstance(row, Mapping)
        and str(row.get("path") or "") == "/v1/memories/add"
    ]
    checks = {
        "expansion_completed": report.get("status") == "completed",
        "generation_ready": report.get("generation_ready") is True,
        "no_write_audit_pass": audit.get("status") == "PASS",
        "feedback_payload_audit_pass": payload.get("status") == "PASS",
        "child_add_calls_zero": int(report.get("child_add_calls") or 0) == 0,
        "memorax_add_records_zero": not add_records,
        "no_memorax_writes_recorded": report.get("no_memorax_writes") is True,
    }
    return all(checks.values()), {
        "status": report.get("status"),
        "failure_stage": report.get("failure_stage"),
        "runtime_error": report.get("runtime_error"),
        "checks": checks,
        "llm_calls": (report.get("llm_usage") or {}).get("total_requests"),
        "memorax_calls": report.get("memorax_calls"),
        "memorax_read_calls": audit.get("memorax_read_call_count"),
        "memorax_add_calls": len(add_records),
        "promoted_scene_count": report.get("promoted_scene_count"),
        "full_round_ready": report.get("full_round_ready"),
        "expanded_dialogue": str((root_output / "expanded_dialogue.json").resolve()),
        "promotable_expanded_dialogue": str(
            (root_output / "promotion/promotable_expanded_dialogue.json").resolve()),
    }


def run(*, plan_root: Path, bootstrap_root: Path,
        quarantine_report: Path, output_root: Path,
        credentials_path: Path, allow_insecure_http: bool,
        live_mode: bool) -> dict[str, Any]:
    plan_root = plan_root.resolve()
    bootstrap_root = bootstrap_root.resolve()
    quarantine_report = quarantine_report.resolve()
    output_root = output_root.resolve()
    preflight = _preflight(
        plan_root=plan_root, bootstrap_root=bootstrap_root,
        quarantine_report=quarantine_report, output_root=output_root)
    if preflight["decision"] != "pass":
        raise RuntimeError(
            "retry_preflight_failed:" + ",".join(preflight["failed_checks"]))
    if not live_mode:
        return {**preflight, "status": "OFFLINE_PLAN_ONLY", "live_started": False}

    output_root.mkdir(parents=True, exist_ok=False)
    _write(output_root / "preflight_report.json", preflight)
    _write(output_root / "source_manifest.json", {
        "manifest_version": "h4_round1_feedback_r02_failed2_no_memorax_source_v1",
        "created_at": _now(),
        "plan": {
            "path": str((plan_root / "r02_feedback_execution_plan.jsonl").resolve()),
            "sha256": _sha(plan_root / "r02_feedback_execution_plan.jsonl"),
        },
        "bootstrap_root": str(bootstrap_root),
        "quarantine_report": {
            "path": str(quarantine_report),
            "sha256": _sha(quarantine_report),
        },
        "retry_root_ids": list(RETRY_ROOT_IDS),
        "no_memorax_writes": True,
    })
    _write(output_root / "run_status.json", {
        "status": "RUNNING", "started_at": _now(),
        "retry_root_ids": list(RETRY_ROOT_IDS),
        "no_memorax_writes": True, "memorax_add_budget": 0,
        "automatic_retry": False,
    })

    root_results: list[dict[str, Any]] = []
    for root_id in RETRY_ROOT_IDS:
        root_output = output_root / "roots" / root_id
        command = [
            sys.executable, "-m",
            "scripts.current_method.restart_n5_spell."
            "h4_round1_feedback_r02_batch_v1.feedback_runtime",
            "--plan-root", str(plan_root),
            "--bootstrap-root", str(bootstrap_root),
            "--root-id", root_id,
            "--output-root", str(root_output),
            "--memorax-credentials", str(credentials_path),
            "--no-memorax-writes",
            "--live",
        ]
        if allow_insecure_http:
            command.append("--allow-insecure-memorax-http")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get(
            "PYTHONPATH", "")
        completed = subprocess.run(command, env=env, check=False)
        success, details = _root_success(root_output)
        root_results.append({
            "root_id": root_id,
            "success": success,
            "returncode": completed.returncode,
            "output_root": str(root_output.resolve()),
            **details,
        })
        _write(output_root / "progress.json", {
            "status": "RUNNING",
            "completed_root_ids": [
                row["root_id"] for row in root_results if row["success"]],
            "failed_root_ids": [
                row["root_id"] for row in root_results if not row["success"]],
            "remaining_root_ids": list(RETRY_ROOT_IDS[len(root_results):]),
            "no_memorax_writes": True,
        })

    successful = [row["root_id"] for row in root_results if row["success"]]
    failed = [row["root_id"] for row in root_results if not row["success"]]
    result = {
        "report_version": "h4_round1_feedback_r02_failed2_no_memorax_v1",
        "status": "PASS" if len(successful) == len(RETRY_ROOT_IDS) else "PARTIAL_FAILURE",
        "completed_at": _now(),
        "retry_root_ids": list(RETRY_ROOT_IDS),
        "successful_root_ids": successful,
        "failed_root_ids": failed,
        "root_results": root_results,
        "r02_generation_count": len(successful),
        "memorax_read_calls": sum(int(row.get("memorax_read_calls") or 0)
                                  for row in root_results),
        "memorax_add_calls": sum(int(row.get("memorax_add_calls") or 0)
                                 for row in root_results),
        "no_memorax_writes": True,
        "qa_calls": 0,
        "solver_calls": 0,
        "training": False,
        "automatic_retry": False,
        "output_root": str(output_root),
    }
    _write(output_root / "final_retry_report.json", result)
    _write(output_root / "run_status.json", {
        "status": result["status"], "completed_at": result["completed_at"],
        "successful_root_ids": successful, "failed_root_ids": failed,
        "memorax_add_calls": result["memorax_add_calls"],
        "no_memorax_writes": True,
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--bootstrap-root", type=Path,
                        default=DEFAULT_BOOTSTRAP_ROOT)
    parser.add_argument("--quarantine-report", type=Path,
                        default=DEFAULT_QUARANTINE_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--memorax-credentials", type=Path,
                        default=DEFAULT_CREDENTIALS)
    parser.add_argument("--allow-insecure-memorax-http", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        plan_root=args.plan_root,
        bootstrap_root=args.bootstrap_root,
        quarantine_report=args.quarantine_report,
        output_root=args.output_root,
        credentials_path=args.memorax_credentials,
        allow_insecure_http=args.allow_insecure_memorax_http,
        live_mode=args.live,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] in {"OFFLINE_PLAN_ONLY", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
