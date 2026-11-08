"""Serial, resumable R02-feedback generation for bootstrapped pilot roots."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_PLAN_ROOT = RUN_BASE / "h4_round1_feedback_r02_batch_v1_live_4"
DEFAULT_BOOTSTRAP_ROOT = RUN_BASE / "h4_round1_feedback_r02_batch_execution_v1_live_2"
DEFAULT_OUTPUT_ROOT = RUN_BASE / "h4_round1_feedback_r02_batch_execution_v2_live_1"
DEFAULT_CREDENTIALS = Path("/root/.config/memorax/credentials.env")
COMPLETED_ROOT_IDS = {
    "cc-train-episode-254852",  # historical live_6 smoke
    "cc-train-episode-122711",  # dialogue-only payload contract smoke
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _is_success(output_root: Path) -> bool:
    report = output_root / "reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json"
    audit = output_root / "feedback_payload_audit.json"
    if not report.is_file() or not audit.is_file():
        return False
    try:
        return (_load(report).get("status") == "completed"
                and _load(audit).get("status") == "PASS")
    except (OSError, ValueError, TypeError):
        return False


def run(*, plan_root: Path, bootstrap_root: Path, output_root: Path,
        credentials_path: Path, allow_insecure_http: bool,
        live_mode: bool, skip_root_ids: set[str] | None = None) -> dict[str, Any]:
    plan_root = plan_root.resolve()
    bootstrap_root = bootstrap_root.resolve()
    output_root = output_root.resolve()
    skip_root_ids = set(skip_root_ids or ())
    rows = _rows(plan_root / "r02_feedback_execution_plan.jsonl")
    rows = [row for row in rows if row.get("root_id") not in COMPLETED_ROOT_IDS]
    plan_root_ids = {str(row.get("root_id") or "") for row in rows}
    unknown_skips = skip_root_ids - plan_root_ids
    if unknown_skips:
        raise RuntimeError(f"skip_root_not_in_plan:{sorted(unknown_skips)}")
    rows = [row for row in rows if str(row.get("root_id") or "") not in skip_root_ids]
    if not rows:
        raise RuntimeError("no_pending_roots_after_skip")
    pending = []
    for row in rows:
        root_id = str(row.get("root_id") or "")
        bootstrap = bootstrap_root / "roots" / root_id / "final_r01_bootstrap_report.json"
        if not bootstrap.is_file() or _load(bootstrap).get("status") != "PASS":
            raise RuntimeError(f"r01_bootstrap_not_pass:{root_id}")
        root_output = output_root / "roots" / root_id
        if _is_success(root_output):
            continue
        if root_output.exists() and any(root_output.iterdir()):
            raise RuntimeError(f"partial_root_requires_manual_reconciliation:{root_id}")
        pending.append(row)
    if not live_mode:
        return {"status": "OFFLINE_PLAN_ONLY", "total_roots": len(rows),
                "pending_root_ids": [str(row["root_id"]) for row in pending],
                "skipped_root_ids": sorted(skip_root_ids),
                "live_started": False}
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".run.lock"
    lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError("feedback_batch_already_running") from error
    completed: list[str] = []
    failures: list[dict[str, Any]] = []
    _write(output_root / "run_status.json", {
        "status": "LIVE_STARTED", "started_at": _now(), "serial": True,
        "total_roots": len(rows), "pending_root_ids": [str(row["root_id"]) for row in pending],
        "skipped_root_ids": sorted(skip_root_ids),
        "r02_generation_only": True, "qa_calls": 0, "solver_calls": 0,
        "automatic_retry": False,
    })
    try:
        for index, row in enumerate(pending):
            root_id = str(row["root_id"])
            root_output = output_root / "roots" / root_id
            command = [
                sys.executable, "-m",
                "scripts.current_method.restart_n5_spell.h4_round1_feedback_r02_batch_v1.feedback_runtime",
                "--plan-root", str(plan_root),
                "--bootstrap-root", str(bootstrap_root),
                "--root-id", root_id,
                "--output-root", str(root_output),
                "--memorax-credentials", str(credentials_path),
            ]
            if allow_insecure_http:
                command.append("--allow-insecure-memorax-http")
            command.append("--live")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
            completed_process = subprocess.run(command, env=env, check=False)
            if completed_process.returncode != 0 or not _is_success(root_output):
                failures.append({"root_id": root_id,
                                 "returncode": completed_process.returncode,
                                 "output_root": str(root_output.resolve()),
                                 "automatic_retry": False})
                _write(output_root / "progress.json", {
                    "status": "STOPPED_ON_FAILURE", "completed_root_ids": completed,
                    "failed_root_ids": [root_id],
                    "remaining_root_ids": [str(item["root_id"]) for item in pending[index + 1:]],
                    "automatic_retry": False,
                })
                break
            completed.append(root_id)
            _write(output_root / "progress.json", {
                "status": "RUNNING", "completed_root_ids": completed,
                "remaining_root_ids": [str(item["root_id"]) for item in pending[index + 1:]],
                "automatic_retry": False,
            })
        status = "PASS" if not failures and len(completed) == len(pending) else "PARTIAL_FAILURE"
        result = {
            "status": status, "completed_at": _now(), "serial": True,
            "total_roots": 10, "already_completed_root_ids": sorted(COMPLETED_ROOT_IDS),
            "skipped_root_ids": sorted(skip_root_ids),
            "newly_completed_root_ids": completed, "failed_roots": failures,
            "r02_generation": len(completed), "r02_child_adds": len(completed),
            "qa_calls": 0, "solver_calls": 0, "automatic_retry": False,
            "output_root": str(output_root.resolve()),
        }
        _write(output_root / "final_feedback_r02_batch_report.json", result)
        return result
    finally:
        lock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--bootstrap-root", type=Path, default=DEFAULT_BOOTSTRAP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--memorax-credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--allow-insecure-memorax-http", action="store_true")
    parser.add_argument("--skip-root-id", action="append", default=[],
                        help="Root already quarantined in another immutable run; do not retry it")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    result = run(plan_root=args.plan_root, bootstrap_root=args.bootstrap_root,
                 output_root=args.output_root,
                 credentials_path=args.memorax_credentials,
                 allow_insecure_http=args.allow_insecure_memorax_http,
                 live_mode=args.live,
                 skip_root_ids=set(args.skip_root_id))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"OFFLINE_PLAN_ONLY", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
