#!/usr/bin/env python3
"""Run one authorized engineering stage with immutable invocation receipts.

There is no automatic stage loop, repair, semantic retry, or model content here.
Management logs live under worker_events so final manifest creation never hashes
a log which its own subprocess is still appending to.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
PYTHON = "/root/miniconda3/bin/python"
EVENTS = BASE / "worker_events" / "production_driver"
STAGES = ("question", "content", "cpu-length", "prepare", "run", "analyze",
          "replication", "report", "manifest", "validate")
CLOSEOUT = {"content", "analyze", "replication", "report", "manifest", "validate"}
PHASES = {
    "question": "isolated_fg_plans_questions_and_review",
    "content": "content_freeze_and_solver_preflight",
    "cpu-length": "content_freeze_and_solver_preflight",
    "prepare": "content_freeze_and_solver_preflight",
    "run": "fixed_b1_solver_and_scoring", "analyze": "fixed_b1_solver_and_scoring",
    "replication": "new16_and_combined20_analysis_and_delivery",
    "report": "new16_and_combined20_analysis_and_delivery",
    "manifest": "new16_and_combined20_analysis_and_delivery",
    "validate": "new16_and_combined20_analysis_and_delivery",
}
REPLICATION = ("REPLICATION_COMPARISON.json", "REPLICATION_TABLES_ZH.md",
               "REPLICATION_ROOT_DIFFERENCES.csv")
QUESTION_WORKFLOW = "plan_schema_continuation_v2"
QUESTION_WRAPPER = BASE / "scripts/question_plan_schema_continuation_v2.py"
QUESTION_READY = BASE / "QUESTION_PLAN_SCHEMA_CONTINUATION_V2_READY.json"


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bind(path: Path | str) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def verify(binding: dict[str, Any]) -> None:
    expected = {key: binding[key] for key in ("path", "sha256", "bytes")}
    if bind(expected["path"]) != expected:
        raise RuntimeError("Immutable binding drift: " + expected["path"])


def verify_nested(value: Any) -> None:
    if isinstance(value, dict):
        if all(key in value for key in ("path", "sha256", "bytes")):
            verify(value)
        for child in value.values():
            verify_nested(child)
    elif isinstance(value, list):
        for child in value:
            verify_nested(child)


def immutable(path: Path, value: Any) -> dict[str, Any]:
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError("Immutable invocation artifact drift: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(raw)
    return bind(path)


def replace_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def progress(kind: str, **fields: Any) -> None:
    row = {"at_utc": utc(), "kind": kind, **fields}
    with (BASE / "PROGRESS.jsonl").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def process_start(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def live_semantic_calls() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((BASE / "process_registry").glob("*.json")):
        row = read(path)
        pid, token = row.get("pid"), row.get("start_token")
        if (row.get("active") is True and isinstance(pid, int) and token is not None
                and process_start(pid) == token):
            rows.append({"call_id": row.get("call_id"), "pid": pid, "start_token": token})
    return rows


def active_source_workers() -> list[dict[str, Any]]:
    names = {"source_driver.py", "source_driver_recovery.py", "source_call_recovery.py",
             "source_recovery_continuation.py", "source_recovery_continuation_v2.py",
             "source_recovery_nonjson_continuation.py"}
    result = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = [part.decode() for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
            cwd = (proc / "cwd").resolve()
        except (OSError, UnicodeDecodeError):
            continue
        scripts = [Path(arg) for arg in args if Path(arg).name in names]
        if any((path if path.is_absolute() else cwd / path).resolve().is_relative_to(BASE)
               for path in scripts):
            result.append({"pid": int(proc.name), "arguments": args})
    return result


def owned_question_script(argument: str, cwd: Path) -> bool:
    expected = {BASE / "scripts" / name for name in
                ("question_driver.py", "question_driver_recovery.py", "question_driver_continued.py",
                 "question_plan_schema_continuation.py", "question_plan_schema_continuation_v2.py",
                 "question_plan_schema_continuation_v3.py", "question_candidate_shape_continuation.py",
                 "question_unfilled_slot_continuation.py", "review_schema_adapter.py",
                 "saved_revision_gold_verification.py", "unused_failed_lineage_adapter.py",
                 "g136202_saved_global_gold_verification.py", "g109164_saved_global7_gold_verification.py",
                 "production_driver.py")}
    candidate = Path(argument)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    return candidate.resolve() in expected


def active_question_workers() -> list[dict[str, Any]]:
    """Catch an owned driver between calls, when its CLI registry is momentarily idle."""
    result = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = [part.decode() for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
            cwd = (proc / "cwd").resolve()
        except (OSError, UnicodeDecodeError):
            continue
        if not args or not Path(args[0]).name.startswith("python"):
            continue
        if any(owned_question_script(argument, cwd) for argument in args[1:]
               if argument and not argument.startswith("-")):
            result.append({"pid": int(proc.name), "start_token": process_start(int(proc.name)),
                           "arguments": args})
    return result


def deadline() -> dt.datetime:
    value = dt.datetime.fromisoformat(read(BASE / "SUPERVISOR_STATUS.json")["deadline_at_utc"])
    if value.tzinfo is None:
        raise RuntimeError("Supervisor deadline must include its UTC offset")
    return value


def cutoff_reason(stage: str, now: dt.datetime, end: dt.datetime,
                  stop: bool, finalize: bool, errors: int) -> str | None:
    if stage in CLOSEOUT:
        return None
    if stop:
        return "STOP_REQUEST prevents a new experimental stage"
    if errors >= 3:
        return "Three consecutive engineering failures; only closeout stages remain"
    if stage == "question" and (finalize or now >= end - dt.timedelta(minutes=45)):
        return "No new semantic generation in the final 45 minutes or after FINALIZE_REQUEST"
    if stage in {"cpu-length", "prepare", "run"} and now >= end - dt.timedelta(minutes=20):
        return "No new Solver/preflight stage in the final 20 minutes"
    if now >= end:
        return "Experiment deadline reached"
    return None


def task_path(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(BASE):
        raise ValueError("Stage selection inputs must remain inside the task directory")
    return path


def question_targets(config_roots: list[str], selection: dict[str, Any],
                     requested_roots: list[str] | None,
                     requested_arms: list[str] | None) -> tuple[list[str], list[str]]:
    """Resolve explicit wrapper targets from approved sources, preserving frozen order."""
    def number(value: str) -> str:
        return str(value).removeprefix("cc-train-episode-")
    configured = [number(value) for value in config_roots]
    approved = [number(row["root_id"]) for row in selection.get("roots", [])]
    if len(approved) != len(set(approved)) or any(root not in configured for root in approved):
        raise RuntimeError("Approved source selection has duplicate or unplanned roots")
    wanted = {number(value) for value in requested_roots} if requested_roots else set(approved)
    invalid = wanted - set(approved)
    if invalid:
        raise RuntimeError("Question roots lack an approved source: " + ", ".join(sorted(invalid)))
    roots = [root for root in configured if root in wanted]
    if not roots:
        raise RuntimeError("No approved source roots are available for the question stage")
    arms_wanted = set(requested_arms or ("F", "G"))
    if not arms_wanted <= {"F", "G"}:
        raise RuntimeError("Unsupported question arm")
    return roots, [arm for arm in ("F", "G") if arm in arms_wanted]


def command_for(args: argparse.Namespace) -> list[str]:
    stage = args.stage
    if stage == "question":
        if not args.root or not args.arm:
            raise RuntimeError("Question targets must be resolved before command construction")
        command = [PYTHON, str(QUESTION_WRAPPER), "run",
                   "--selection", str(args.selection), "--source-freeze", str(args.source_freeze),
                   "--workers", str(getattr(args, "workers", 3))]
        for root in args.root or []:
            command.extend(("--root", root))
        for arm in args.arm or []:
            command.extend(("--arm", arm))
        return command
    fixed = {
        "content": ["scripts/content_assembly_driver.py"],
        "cpu-length": ["scripts/solver_cpu_length_precheck.py", "--runtime-wrapper",
                       str(BASE / "scripts/solver_runtime_adapter.py")],
        "prepare": ["scripts/solver_runtime_adapter.py", "prepare"],
        "run": ["scripts/solver_runtime_adapter.py", "run"],
        "analyze": ["scripts/solver_runtime_adapter.py", "analyze"],
        "replication": ["analysis_replication.py"],
        "report": ["scripts/finalize_report.py", "--write-report"],
        "manifest": ["scripts/finalize_report.py", "--finalize-manifest"],
        "validate": ["validate_final.py"],
    }
    parts = fixed[stage]
    return [PYTHON, str(BASE / parts[0]), *parts[1:]]


def stage_inputs(args: argparse.Namespace, command: list[str]) -> list[dict[str, Any]]:
    paths = [Path(__file__), Path(command[1]), BASE / "INPUT_LOCK.json", BASE / "CONFIG.json", BASE / "PLAN.json"]
    required = {
        "question": [args.selection, args.source_freeze, BASE / "scripts/question_driver.py",
                     BASE / "scripts/codex_calls.py", BASE / "question_authoring/COMMON_AUTHOR_CONTRACT_ZH.md",
                     QUESTION_READY],
        "content": [BASE / "SOURCE_SELECTION_FINAL.json", BASE / "SOURCE_FREEZE.json",
                    BASE / "scripts/question_driver.py", BASE / "scripts/question_driver_recovery.py"],
        "cpu-length": [BASE / "CONTENT_FREEZE.json", BASE / "scripts/solver_runtime_adapter.py",
                       BASE / "solver_runner.py", BASE / "ANALYSIS_DRIVER_FREEZE.json"],
        "prepare": [BASE / "CONTENT_FREEZE.json", BASE / "CPU_LENGTH_PRECHECK.json",
                    BASE / "solver_runner.py", BASE / "ANALYSIS_DRIVER_FREEZE.json"],
        "run": [BASE / "execution/PREPARATION.json", BASE / "CONTENT_FREEZE.json", BASE / "solver_runner.py"],
        "analyze": [BASE / "execution/PREPARATION.json", BASE / "execution/RAW_RESPONSE_FREEZE.json",
                    BASE / "CONTENT_FREEZE.json", BASE / "solver_runner.py"],
        "replication": [BASE / "ANALYSIS_PLAN.json", BASE / "ANALYSIS_DRIVER_FREEZE.json"],
        "report": [BASE / "scripts/finalize_report.py", BASE / "ANALYSIS_PLAN.json"],
        "manifest": [BASE / "FINAL_REPORT.json", BASE / "FINAL_REPORT_ZH.md", BASE / "INDEPENDENT_CLOSEOUT_AUDIT.json"],
        "validate": [BASE / "FINAL_ARTIFACT_MANIFEST.json", BASE / "FINAL_REPORT.json", BASE / "FINAL_REPORT_ZH.md"],
    }
    paths.extend(required[args.stage])
    lock = read(BASE / "INPUT_LOCK.json")
    if args.stage not in {"report", "manifest", "validate"}:
        verify_nested(lock)
    if args.stage in {"question", "content"}:
        selection_path = args.selection if args.stage == "question" else BASE / "SOURCE_SELECTION_FINAL.json"
        selection = read(selection_path)
        for row in selection.get("roots", []):
            verify(row)
            paths.append(Path(row["path"]))
    if args.stage == "question":
        ready = read(QUESTION_READY)
        verify_nested(ready)
        if (ready.get("status") != "PASS" or ready.get("wrapper_binding") != bind(QUESTION_WRAPPER)
                or ready.get("frozen_driver_binding") != bind(BASE / "scripts/question_driver.py")):
            raise RuntimeError("Plan-schema continuation readiness does not bind the actual wrapper and frozen driver")
    if args.stage == "content":
        paths.extend(path for path in (BASE / "question_authoring").glob("*/roots/*/*") if path.is_file())
    if args.stage in {"cpu-length", "prepare", "run", "analyze"}:
        verify_nested(read(BASE / "CONTENT_FREEZE.json"))
    if args.stage in {"prepare", "run", "analyze", "replication"}:
        analysis = read(BASE / "ANALYSIS_DRIVER_FREEZE.json")
        verify_nested(analysis)
        if analysis.get("status") != "FROZEN_BEFORE_SOLVER" or analysis["script_binding"] != bind(BASE / "analysis_replication.py"):
            raise RuntimeError("Analysis implementation was not frozen before Solver")
    return [bind(path) for path in sorted(set(path.resolve() for path in paths), key=str)]


def product_paths(stage: str, args: argparse.Namespace) -> list[Path]:
    names = {
        "content": ("CANDIDATES_PRIVATE.json", "CONTENT_REVIEW_AI.json", "CONTENT_GATE_FINAL.json", "CONTENT_FREEZE.json"),
        "cpu-length": ("CPU_LENGTH_PRECHECK.json",),
        "prepare": ("execution/PREPARATION.json", "execution/PUBLIC_REQUESTS.json", "execution/PRIVATE_SCORING.json"),
        "run": ("execution/STARTED.json", "execution/RAW_RESPONSE_FREEZE.json"),
        "analyze": ("execution/SCORES.json", "execution/ANALYSIS_SUMMARY.json"),
        "replication": REPLICATION,
        "report": ("FINAL_REPORT.json", "FINAL_REPORT_ZH.md", "ORIGINAL_INTEGRITY_AFTER.json"),
        "manifest": ("FINAL_ARTIFACT_MANIFEST.json",), "validate": ("FINAL_ARTIFACT_MANIFEST.json",),
    }
    if stage != "question":
        return [BASE / name for name in names[stage]]
    if not args.root or not args.arm:
        raise RuntimeError("Question products require resolved explicit root and arm targets")
    return [BASE / "question_plan_continuation" / root.removeprefix("cc-train-episode-") / arm / name
            for root in args.root for arm in args.arm
            for name in ("CONTINUATION_FINALIZER_READY.json", "CONTINUATION_INTEGRATION_OUTCOME.json", "OUTCOME.json")]


def collect_products(stage: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = product_paths(stage, args)
    if stage == "question":
        for path in paths:
            if path.is_file():
                verify_nested(read(path))
    return [bind(path) for path in paths if path.is_file()]


def existing_products(stage: str, args: argparse.Namespace) -> bool:
    if stage in {"question", "validate"}:
        return False
    paths = product_paths(stage, args)
    if not paths or not all(path.is_file() for path in paths):
        return False
    for path in paths:
        if path.suffix == ".json":
            verify_nested(read(path))
    return True


def guard(args: argparse.Namespace, errors: int) -> None:
    reason = cutoff_reason(args.stage, dt.datetime.now(dt.timezone.utc), deadline(),
                           (BASE / "STOP_REQUEST").exists(), (BASE / "FINALIZE_REQUEST").exists(), errors)
    if reason:
        raise RuntimeError(reason)
    if args.stage == "question":
        active, workers, question_workers = (live_semantic_calls(), active_source_workers(),
                                             active_question_workers())
        if active or workers or question_workers:
            raise RuntimeError("Full question stage requires zero active semantic calls, source workers, and question workers: "
                               + json.dumps({"calls": active, "source_workers": workers,
                                             "question_workers": question_workers}, ensure_ascii=False))
        if (BASE / "CONTENT_FREEZE.json").exists():
            raise RuntimeError("Content is already frozen; no question generation after freeze")
    if args.stage == "content" and (live_semantic_calls() or active_source_workers() or active_question_workers()):
        raise RuntimeError("Content aggregation waits for all active semantic/source/question processes")
    if args.stage == "prepare":
        checked = read(BASE / "CPU_LENGTH_PRECHECK.json")
        verify_nested(checked)
        if checked.get("status") != "PASS":
            raise RuntimeError("CPU length precheck did not pass; no automatic exclusion or rewrite")
        if not (BASE / "execution/PREPARATION.json").exists():
            execution = BASE / "execution"
            partial = [path for path in (execution / "CPU_API_CHECK.json", execution / "PUBLIC_REQUESTS.json",
                       execution / "PROMPT_TOKEN_IDS.json", execution / "PRIVATE_SCORING.json") if path.exists()]
            partial.extend(path for path in (execution / "prompts").glob("*") if path.is_file())
            if partial:
                raise RuntimeError("Partial write-once preparation artifacts exist; refusing automatic replay")
    if args.stage == "run":
        if (BASE / "execution/STARTED.json").exists() and not (BASE / "execution/RAW_RESPONSE_FREEZE.json").exists():
            raise RuntimeError("Solver STARTED without RAW_RESPONSE_FREEZE: replay is forbidden; preserve PARTIAL")
    if args.stage == "analyze" and not (BASE / "execution/RAW_RESPONSE_FREEZE.json").exists():
        raise RuntimeError("No frozen raw responses to analyze")


def update_status(stage: str, event: str, invocation: str, result: dict[str, Any] | None = None) -> None:
    path = BASE / "TASK_STATUS.json"
    value = read(path) if path.is_file() else {"status": "RUNNING", "phases": {}}
    value.update(updated_at_utc=utc(), current_phase=PHASES[stage], training=False)
    phase = value.setdefault("phases", {}).setdefault(PHASES[stage], {})
    phase["production_driver_stage"] = stage
    phase["production_driver_invocation"] = invocation
    if event == "START":
        phase["status"] = "IN_PROGRESS"
    elif event == "RESULT" and result is not None:
        phase["last_stage_status"] = result["status"]
        phase["status"] = "BLOCKED" if result["status"] == "BLOCKED" else "PARTIAL"
        if stage == "validate" and result.get("exit_code") == 0:
            report = read(BASE / "FINAL_REPORT.json")
            value["status"] = report["overall_status"]
            value["phases"] = report["phases"]
    value["engineering_heartbeat"] = {"at_utc": utc(), "kind": "PRODUCTION_DRIVER_" + event,
                                     "stage": stage, "invocation_id": invocation,
                                     "active_semantic_calls": live_semantic_calls(),
                                     "finalize_request": (BASE / "FINALIZE_REQUEST").exists(),
                                     "stop_request": (BASE / "STOP_REQUEST").exists()}
    replace_json(path, value)


def descendants(pid: int) -> list[tuple[int, str]]:
    relations = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            relations[int(proc.name)] = (int(fields[1]), fields[19])
        except (OSError, ValueError, IndexError):
            continue
    ordered, parents = [], [pid]
    while parents:
        parent = parents.pop(0)
        children = [child for child, (ppid, _) in relations.items() if ppid == parent]
        ordered.extend((child, relations[child][1]) for child in children)
        parents.extend(children)
    return ordered


def stop_child(process: subprocess.Popen, token: str | None) -> list[dict[str, Any]]:
    owned = descendants(process.pid) + ([(process.pid, token)] if token is not None else [])
    actions = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, start in reversed(owned):
            if process_start(pid) == start:
                try:
                    os.kill(pid, sig)
                    actions.append({"pid": pid, "start_token": start, "signal": sig.name})
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
    process.wait(timeout=10)
    return actions


def run_stage(args: argparse.Namespace) -> int:
    if args.stage == "question":
        args.root, args.arm = question_targets(read(BASE / "CONFIG.json")["roots"], read(args.selection),
                                               args.root, args.arm)
    command = command_for(args)
    identity = {"stage": args.stage, "root": args.root or [], "arm": args.arm or [],
                "selection": str(args.selection), "source_freeze": str(args.source_freeze)}
    if args.stage == "question":
        identity.update(question_workflow=QUESTION_WORKFLOW, workers=args.workers)
    key = args.invocation_id or args.stage + "__" + hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", key):
        raise ValueError("Unsafe invocation ID")
    directory = EVENTS / key
    directory.mkdir(parents=True, exist_ok=True)
    state_path = EVENTS / "STATE.json"
    state = read(state_path) if state_path.exists() else {"consecutive_engineering_failures": 0, "invocations": []}
    result_path, start_path = directory / "RESULT.json", directory / "START.json"
    if result_path.exists():
        saved = read(result_path)
        verify_nested(saved)
        if saved.get("identity") != identity or saved.get("command") != command:
            raise RuntimeError("Completed invocation identity/command differs")
        print(json.dumps({"idempotent_reuse": True, "result": saved}, ensure_ascii=False, indent=2))
        return saved["exit_code"]
    if start_path.exists():
        prior = read(start_path)
        verify_nested(prior)
        child_path = directory / "CHILD_PID.json"
        child = read(child_path) if child_path.exists() else {}
        live = isinstance(child.get("pid"), int) and process_start(child["pid"]) == child.get("start_token")
        immutable(directory / "INTERRUPTED_CLASSIFICATION.json", {
            "classification": "EXISTING_STAGE_STILL_LIVE" if live else "INTERRUPTED_STAGE_NO_AUTOMATIC_REPLAY",
            "start_binding": bind(start_path), "child_receipt": child, "replayed": False})
        raise RuntimeError("Existing invocation START has no RESULT; it is explicitly classified and will not be replayed")
    start = {"schema_version": "fgr20_production_stage_start_v1", "started_at_utc": utc(),
             "identity": identity, "command": command, "cwd": str(BASE), "python": PYTHON,
             "driver_binding": bind(__file__), "supervisor_deadline_at_utc": deadline().isoformat(),
             "automatic_stage_loop": False, "automatic_repair": False, "automatic_semantic_retry": False}
    immutable(start_path, start)
    update_status(args.stage, "START", key)
    progress("PRODUCTION_STAGE_START", stage=args.stage, invocation_id=key, command=command)
    started = time.monotonic()
    result = {"schema_version": "fgr20_production_stage_result_v1", "identity": identity,
              "command": command, "start_binding": bind(start_path), "status": "BLOCKED",
              "exit_code": 3, "engineering_failure": False, "stage_replayed": False}
    process = None
    token = None
    try:
        guard(args, int(state.get("consecutive_engineering_failures", 0)))
        result["input_bindings"] = stage_inputs(args, command)
        if existing_products(args.stage, args):
            result.update(status="REUSED_EXISTING_PRODUCTS", exit_code=0, subprocess_started=False,
                          product_bindings=collect_products(args.stage, args))
            if args.stage == "cpu-length" and read(BASE / "CPU_LENGTH_PRECHECK.json").get("status") != "PASS":
                result.update(status="PARTIAL", exit_code=2,
                              note="Existing failed CPU length result preserved without exclusion or retry")
        else:
            stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
            with stdout_path.open("xb") as out, stderr_path.open("xb") as err:
                process = subprocess.Popen(command, cwd=BASE, stdout=out, stderr=err, start_new_session=True)
                token = process_start(process.pid)
                immutable(directory / "CHILD_PID.json", {"pid": process.pid, "start_token": token,
                          "pgid": process.pid, "command": command, "started_at_utc": utc()})
                heartbeat = time.monotonic()
                stop_reason = None
                while process.poll() is None:
                    if (BASE / "STOP_REQUEST").exists() and args.stage not in CLOSEOUT:
                        stop_reason = "STOP_REQUEST"
                    elif dt.datetime.now(dt.timezone.utc) >= deadline() and args.stage not in CLOSEOUT:
                        stop_reason = "EXPERIMENT_DEADLINE"
                    if stop_reason:
                        result["termination_actions"] = stop_child(process, token)
                        break
                    if time.monotonic() - heartbeat >= 30:
                        update_status(args.stage, "HEARTBEAT", key)
                        progress("PRODUCTION_STAGE_HEARTBEAT", stage=args.stage, invocation_id=key, child_pid=process.pid)
                        heartbeat = time.monotonic()
                    time.sleep(1)
                code = process.wait()
            result.update(subprocess_started=True, subprocess_exit_code=code,
                          stdout_binding=bind(stdout_path), stderr_binding=bind(stderr_path),
                          child_binding=bind(directory / "CHILD_PID.json"),
                          product_bindings=collect_products(args.stage, args))
            if stop_reason:
                result.update(status="STOPPED_PARTIAL", exit_code=3, stop_reason=stop_reason)
            elif code == 0:
                required = product_paths(args.stage, args)
                missing = [str(path) for path in required if not path.is_file()]
                if args.stage == "question" and not required:
                    missing.append("No question root/arm outcome was produced")
                if missing:
                    result.update(status="FAILED", exit_code=3, engineering_failure=True,
                                  error="Subprocess exited zero without required artifacts: " + json.dumps(missing))
                else:
                    result.update(status="COMPLETED", exit_code=0)
            elif code == 2 and args.stage in {"question", "cpu-length"}:
                result.update(status="PARTIAL", exit_code=2,
                              note="Truthful semantic/content/budget shortfall; not retried or counted as an engineering failure")
            else:
                result.update(status="FAILED", exit_code=code if code > 0 else 3, engineering_failure=True)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if process is not None and process.poll() is None:
            result["termination_actions"] = stop_child(process, token)
            result.update(status="FAILED", engineering_failure=True)
        for name in ("stdout", "stderr"):
            path = directory / (name + ".log")
            if path.is_file():
                result[name + "_binding"] = bind(path)
    result.update(ended_at_utc=utc(), wall_seconds=time.monotonic() - started)
    immutable(result_path, result)
    if result["engineering_failure"]:
        state["consecutive_engineering_failures"] = int(state.get("consecutive_engineering_failures", 0)) + 1
    elif result["status"] in {"COMPLETED", "PARTIAL", "REUSED_EXISTING_PRODUCTS"}:
        state["consecutive_engineering_failures"] = 0
    state.setdefault("invocations", []).append(bind(result_path))
    state["updated_at_utc"] = utc()
    replace_json(state_path, state)
    update_status(args.stage, "RESULT", key, result)
    progress("PRODUCTION_STAGE_END", stage=args.stage, invocation_id=key, status=result["status"],
             exit_code=result["exit_code"], consecutive_engineering_failures=state["consecutive_engineering_failures"])
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result["exit_code"]


def self_check() -> dict[str, Any]:
    now = dt.datetime(2026, 10, 25, tzinfo=dt.timezone.utc)
    end = now + dt.timedelta(hours=1)
    assert cutoff_reason("question", now, end, False, False, 0) is None
    assert cutoff_reason("question", now + dt.timedelta(minutes=16), end, False, False, 0)
    assert cutoff_reason("run", now + dt.timedelta(minutes=41), end, False, False, 0)
    assert cutoff_reason("question", now, end, False, True, 0)
    assert cutoff_reason("run", now, end, True, False, 0)
    assert cutoff_reason("prepare", now, end, False, False, 3)
    assert cutoff_reason("report", end, end, True, True, 3) is None
    assert owned_question_script("scripts/question_driver_recovery.py", BASE)
    assert owned_question_script(str(BASE / "scripts/question_driver.py"), Path("/"))
    assert owned_question_script(str(QUESTION_WRAPPER), Path("/"))
    assert not owned_question_script("/tmp/question_driver.py", BASE)
    # A separate production parent must also block aggregation during the gap
    # before its child starts. active_question_workers excludes this process by
    # PID before applying the path allowlist.
    assert owned_question_script(str(Path(__file__)), BASE)
    for name in ("question_plan_schema_continuation_v3.py", "question_candidate_shape_continuation.py",
                 "question_unfilled_slot_continuation.py", "review_schema_adapter.py",
                 "saved_revision_gold_verification.py", "unused_failed_lineage_adapter.py",
                 "g136202_saved_global_gold_verification.py", "g109164_saved_global7_gold_verification.py"):
        assert owned_question_script("scripts/" + name, BASE)
        assert owned_question_script(str(BASE / "scripts" / name), Path("/"))
        assert not owned_question_script("/tmp/" + name, BASE)
    assert all(worker["pid"] != os.getpid() for worker in active_question_workers())
    roots, arms = question_targets(["75241", "55788", "761", "77882", "45849"],
        {"roots": [{"root_id": "cc-train-episode-45849"}, {"root_id": "cc-train-episode-55788"}]}, None, None)
    assert roots == ["55788", "45849"] and arms == ["F", "G"]
    try:
        question_targets(["55788"], {"roots": []}, None, None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Empty selection must fail closed")
    namespace = argparse.Namespace(stage="question", selection=BASE / "SOURCE_SELECTION_FINAL.json",
        source_freeze=BASE / "SOURCE_FREEZE.json", root=["55788"], arm=["F"], workers=3)
    command = command_for(namespace)
    assert command[0] == PYTHON and command[-4:] == ["--root", "55788", "--arm", "F"]
    assert command[1:3] == [str(QUESTION_WRAPPER), "run"]
    assert command[command.index("--workers") + 1] == "3"
    expected_products = product_paths("question", namespace)
    assert len(expected_products) == 3
    assert all(path.parent == BASE / "question_plan_continuation/55788/F" for path in expected_products)
    for stage in STAGES:
        namespace.stage = stage
        assert command_for(namespace)[0] == PYTHON
    return {"status": "PASS", "model_calls": 0, "subprocesses_started": 0, "files_written": 0,
            "checks": ["stage command allowlist", "generation 45-minute cutoff", "Solver 20-minute cutoff",
                       "STOP_REQUEST", "FINALIZE_REQUEST", "three engineering failures", "closeout remains possible",
                       "exact original and v3/shape/unused/review/saved-verifier/failed-lineage question-parent script paths",
                       "separate production parent recognized; foreign paths and current process PID excluded",
                       "approved roots only in CONFIG order", "explicit continuation roots/arms/workers",
                       "continuation READY/integration/outcome products; original outcomes preserved"],
            "stages": list(STAGES)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=STAGES)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--invocation-id", help="Explicit immutable invocation label; defaults to stage/filter hash")
    parser.add_argument("--root", action="append", help="Question stage only; repeatable")
    parser.add_argument("--arm", action="append", choices=("F", "G"), help="Question stage only; repeatable")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3,
                        help="Concurrent question continuation jobs (default: 3; global call guard remains authoritative)")
    parser.add_argument("--selection", default=str(BASE / "SOURCE_SELECTION_FINAL.json"))
    parser.add_argument("--source-freeze", default=str(BASE / "SOURCE_FREEZE.json"))
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), ensure_ascii=False, indent=2))
        return 0
    if args.stage is None:
        parser.error("stage is required unless --self-check is used")
    if args.stage != "question" and (args.root or args.arm):
        parser.error("--root/--arm are only permitted for the question stage")
    args.selection, args.source_freeze = task_path(args.selection), task_path(args.source_freeze)
    EVENTS.mkdir(parents=True, exist_ok=True)
    with (EVENTS / ".driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
