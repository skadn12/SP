#!/usr/bin/env python3
"""Serialize exactly seven authorized remaining semantic stages.

Stops before content assembly, Solver, analysis and report.  No semantic
prompt, question, Gold, verdict or retry is created here.  Delivery and
``self-check`` never launch a semantic child.  ``run`` is a separate explicit
operation, guarded by its exact engineering READY and current input bindings.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
PYTHON = "/root/miniconda3/bin/python"
READY = BASE / "AUTHORIZED_PIPELINE_RESUME_READY.json"
EVENTS = BASE / "worker_events" / "authorized_pipeline_resume"
V2_READY = BASE / "REVIEW_SCHEMA_ADAPTER_V2_READY.json"
V2_SHA = "3d8bf15b6e42655fb4ca5d53fc0fde00f2da7970f3dafae1660c53987b469645"
HEARTBEAT_SECONDS = 35
PREREQUISITE = BASE / "question_candidate_shape_continuation/179186/F/CONTINUATION_FINALIZER_READY.json"
STEPS = (
    ("shape_77882_G", "shape", "77882", "G"),
    ("shape_136202_F", "shape", "136202", "F"),
    ("saved_179186_G", "saved", "179186", "G"),
    ("saved_136202_G", "global8", "136202", "G"),
    ("saved_109164_G", "global7", "109164", "G"),
    ("unused_77882_F", "unused", "77882", "F"),
    ("unused_failed_109164_F", "unused_failed", "109164", "F"),
)
PARENT_NAMES = {
    "source_driver.py", "source_driver_recovery.py", "source_call_recovery.py",
    "question_driver.py", "question_driver_recovery.py", "question_driver_continued.py",
    "question_plan_schema_continuation.py", "question_plan_schema_continuation_v2.py",
    "question_plan_schema_continuation_v3.py", "question_candidate_shape_continuation.py",
    "question_unfilled_slot_continuation.py", "review_schema_adapter.py",
    "review_schema_adapter_v2.py", "unused_failed_lineage_adapter.py",
    "saved_revision_gold_verification.py", "g136202_saved_global_gold_verification.py",
    "g109164_saved_global7_gold_verification.py", "production_driver.py",
    "resume_authorized_pipeline.py",
}
STOPPING = False


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def bind(path: Path | str) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def verify(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or bind(value["path"]) != {key: value[key] for key in ("path", "sha256", "bytes")}:
        raise RuntimeError("BINDING_DRIFT:" + str(value))


def immutable(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError("IMMUTABLE_ORCHESTRATOR_OUTPUT_DRIFT:" + str(path))
    else:
        with path.open("xb") as handle:
            handle.write(raw)
    return bind(path)


def replace_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def process_start(pid: int) -> str | None:
    try:
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if tail[0] == "Z" else tail[19]
    except (OSError, IndexError):
        return None


def live_calls(base: Path = BASE) -> list[dict[str, Any]]:
    result = []
    for path in sorted((base / "process_registry").glob("*.json")):
        row = read(path)
        if (row.get("active") is True and type(row.get("pid")) is int
                and process_start(row["pid"]) == str(row.get("start_token"))):
            result.append({**row, "registry_path": str(path)})
    return result


def active_parents() -> list[dict[str, Any]]:
    result = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [part.decode(errors="replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
            if not argv or "python" not in Path(argv[0]).name:
                continue
            script = next((arg for arg in argv[1:] if arg.endswith(".py")), None)
            if not script or Path(script).name not in PARENT_NAMES:
                continue
            actual = Path(script)
            if not actual.is_absolute():
                actual = (proc / "cwd").resolve() / actual
            if actual.resolve() != BASE / "scripts" / Path(script).name:
                continue
            token = process_start(int(proc.name))
            if token is not None:
                result.append({"pid": int(proc.name), "start_token": token, "argv": argv})
        except OSError:
            continue
    return result


def cutoff(base: Path = BASE, now: dt.datetime | None = None) -> str | None:
    if STOPPING or (base / "STOP_REQUEST").exists():
        return "STOP_REQUEST"
    deadline = dt.datetime.fromisoformat(read(base / "SUPERVISOR_STATUS.json")["deadline_at_utc"])
    current = now or dt.datetime.now(dt.timezone.utc)
    if current >= deadline - dt.timedelta(minutes=20):
        return "CLOSEOUT_ONLY_NO_NEW_ENGINEERING"
    if (base / "FINALIZE_REQUEST").exists():
        return "FINALIZE_REQUEST"
    if current >= deadline - dt.timedelta(minutes=45):
        return "GENERATION_CUTOFF"
    return None


def heartbeat(phase: str, **fields: Any) -> None:
    row = {"at_utc": utc(), "kind": "AUTHORIZED_REMAINING_PIPELINE_HEARTBEAT",
           "phase": phase, "orchestrator_pid": os.getpid(), "training": False, **fields}
    with (BASE / "PROGRESS.jsonl").open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    path = BASE / "TASK_STATUS.json"
    state = read(path)
    state["updated_at_utc"] = row["at_utc"]
    state["authorized_remaining_pipeline"] = row
    replace_json(path, state)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def output_dir(step: tuple[str, str, str, str]) -> Path:
    _, kind, number, arm = step
    tree = ("question_candidate_shape_continuation" if kind == "shape" else
            "question_unfilled_continuation" if kind in {"unused", "unused_failed"} else
            "question_saved_revision_verification")
    return BASE / tree / number / arm


def command(step: tuple[str, str, str, str]) -> list[str]:
    _, kind, number, arm = step
    source = BASE / "question_plan_continuation" / number / arm
    if kind in {"shape", "unused", "unused_failed"}:
        tail = ["run"] if kind == "unused_failed" else ["run", "--root", number, "--arm", arm,
                "--input-work" if kind == "shape" else "--work-dir", str(source)]
        return [PYTHON, str(BASE / "scripts/review_schema_adapter_v2.py"), "launch", "--driver", kind, "--", *tail]
    script = {"saved": "saved_revision_gold_verification.py",
              "global8": "g136202_saved_global_gold_verification.py",
              "global7": "g109164_saved_global7_gold_verification.py"}[kind]
    tail = ["run", "--root", number, "--arm", arm, "--input-work", str(source)] if kind == "saved" else ["run"]
    return [PYTHON, str(BASE / "scripts" / script), *tail]


def family_call_ids(step: tuple[str, str, str, str], base: Path = BASE) -> list[str]:
    _, kind, number, arm = step
    suffix = "candidate_shape_01" if kind == "shape" else "unused_content_revision_01" if kind in {"unused", "unused_failed"} else "saved_revision_gold_01"
    pattern = re.compile(rf"q20_(?:plan|author|gold|verify|revision)_{arm.lower()}_{number}_{suffix}(?:_|$)")
    return sorted(path.name for path in (base / "cli_calls").glob("q20_*") if pattern.match(path.name)
                  and any(path.glob("attempt_*/REQUEST.txt")))


def validated_output(step: tuple[str, str, str, str]) -> dict[str, Any] | None:
    path = output_dir(step) / "CONTINUATION_FINALIZER_READY.json"
    if not path.is_file():
        return None
    import content_continuation_adapter as adapter
    checker = adapter.Continuations.__new__(adapter.Continuations)
    checker.inputs = []
    row = checker.validate_ready(path)
    if (row["root_id"], row["arm"]) != ("cc-train-episode-" + step[2], step[3]):
        raise RuntimeError("FINAL_READY_WRONG_STAGE_IDENTITY")
    return {"ready_binding": bind(path), "input_bindings_verified": len(checker.inputs),
            "review_semantics_adjudicated_by_orchestrator": False}


def verify_v2() -> dict[str, Any]:
    row = read(V2_READY)
    expected = BASE / "scripts/review_schema_adapter_v2.py"
    if (row.get("status") != "READY_ENGINEERING_ONLY" or row.get("adapter_binding") != bind(expected)
            or row["adapter_binding"]["sha256"] != V2_SHA
            or row.get("semantic_calls_started_by_engineering") != 0
            or row.get("semantic_admission_asserted") is not False):
        raise RuntimeError("FUTURE_V2_READY_OR_CODE_DRIFT")
    for field in ("adapter_binding", "frozen_v1_adapter_binding", "engineering_checks_binding"):
        verify(row[field])
    verify(row["independent_engineering_review_binding"])
    expected = [(step[1], "cc-train-episode-" + step[2], step[3], command(step)) for step in STEPS
                if step[1] in {"shape", "unused", "unused_failed"}]
    recorded = [(target.get("driver"), target.get("root_id"), target.get("arm"), shlex.split(target.get("command", "")))
                for target in row.get("future_targets", []) if target.get("stage_request_count_at_freeze") == 0]
    if recorded != expected:
        raise RuntimeError("FUTURE_V2_READY_TARGETS_OR_COMMANDS_DRIFT")
    return bind(V2_READY)


def unused_budget(step: tuple[str, str, str, str]) -> dict[str, Any] | None:
    if step[1] not in {"unused", "unused_failed"}:
        return None
    import question_driver as q
    import question_unfilled_slot_continuation as unused
    q.SOURCE_SELECTION_PATH = BASE / "SOURCE_SELECTION_FINAL.json"
    q.SOURCE_FREEZE_PATH = BASE / "SOURCE_FREEZE.json"
    _, sources = q.load_selection(q.SOURCE_SELECTION_PATH)
    root = q.root_id(step[2])
    if step[1] == "unused_failed":
        import unused_failed_lineage_adapter as failed
        data = failed.inspect_budget(sources[root])
        expected = ["MC2", "MC3", "MC4", "MC6", "ORDER1", "ORDER3"]
    else:
        data = unused.inspect_budget(root, step[3], sources[root], BASE / "question_plan_continuation" / step[2] / step[3])
        expected = list(q.SLOTS)
    if data.get("eligible_slot_ids") != expected or data.get("revision_requests"):
        raise RuntimeError("UNUSED_REVISION_BUDGET_CHANGED_NO_REPLAY")
    return {"eligible_slot_ids": expected, "inspection_sha256": hashlib.sha256(
        json.dumps(data, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "revision_requests": [], "read_only_inspection": True}


def orphan_starts(base: Path = BASE) -> list[str]:
    return [str(path) for path in sorted((base / "cli_calls").glob("*/attempt_*/START.json"))
            if not (path.parent / "TERMINAL.json").exists()
            and not (path.parent / "TRANSPORT_FAILURE.json").exists()
            and not (path.parent.parent / "RESULT.json").exists()]


def wait_idle(require_prerequisite: bool = False) -> None:
    last = 0.0
    while True:
        reason = cutoff()
        if reason:
            raise RuntimeError(reason)
        calls, parents = live_calls(), active_parents()
        prerequisite_ok = PREREQUISITE.is_file() or not require_prerequisite
        if prerequisite_ok and not calls and not parents:
            if orphan_starts():
                raise RuntimeError("UNCLASSIFIED_START_WITHOUT_TERMINAL_NO_AUTOMATIC_RESTART:" + json.dumps(orphan_starts()))
            return
        if time.monotonic() - last >= HEARTBEAT_SECONDS:
            heartbeat("waiting_for_current_stage_and_idle", prerequisite_ready=prerequisite_ok,
                      active_semantic_calls=[{k: x.get(k) for k in ("call_id", "pid")} for x in calls],
                      active_driver_parents=parents)
            last = time.monotonic()
        time.sleep(1)


def stop_owned(child: subprocess.Popen, token: str | None, before: set[str], step: tuple[str, str, str, str]) -> None:
    # Codex children use their own sessions. Stop only new registry calls born
    # during this one serialized child, after token/task-directory validation.
    owned = []
    actual_family = set(family_call_ids(step))
    for row in live_calls():
        if (row["registry_path"] in before or row.get("task_directory") != str(BASE)
                or row.get("call_id") not in actual_family):
            continue
        if process_start(row["pid"]) != str(row.get("start_token")):
            continue
        try:
            os.killpg(row.get("pgid", row["pid"]), signal.SIGTERM)
            owned.append(row)
        except ProcessLookupError:
            pass
    if child.poll() is None and process_start(child.pid) == token:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    until = time.monotonic() + 8
    while child.poll() is None and time.monotonic() < until:
        time.sleep(0.2)
    if child.poll() is None and process_start(child.pid) == token:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
    for row in owned:
        if process_start(row["pid"]) == str(row.get("start_token")):
            try:
                os.killpg(row.get("pgid", row["pid"]), signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_step(step: tuple[str, str, str, str]) -> dict[str, Any]:
    target = EVENTS / "stages" / step[0]
    existing = validated_output(step)
    if existing:
        result = {"status": "REUSED_TERMINAL_OUTPUT", **existing, "semantic_calls_started_by_orchestrator": 0}
        recovery = target / "RECOVERED_FROM_EXISTING_OUTPUT.json"
        if (target / "START.json").exists() and not (target / "RESULT.json").exists() and not recovery.exists():
            immutable(recovery, {"at_utc": utc(), "classification": "child_final_output_saved_before_coordinator_result", **result})
        return result
    if (target / "START.json").exists() or (target / "RESULT.json").exists() or family_call_ids(step):
        raise RuntimeError("CONSUMED_OR_INTERRUPTED_STAGE_WITHOUT_READY_NO_REPLAY:" + step[0])
    wait_idle()
    verify_v2()
    budget = unused_budget(step)
    if cutoff():
        raise RuntimeError(str(cutoff()))
    before = {str(path) for path in (BASE / "process_registry").glob("*.json")}
    argv = command(step)
    target.mkdir(parents=True, exist_ok=True)
    immutable(target / "START.json", {"at_utc": utc(), "step": list(step), "argv": argv,
              "script_binding": bind(Path(__file__)), "v2_ready_binding": bind(V2_READY),
              "unused_budget_inspection": budget, "automatic_child_retry": False,
              "output_ready_required": str(output_dir(step) / "CONTINUATION_FINALIZER_READY.json")})
    with (target / "STDOUT.log").open("xb") as out, (target / "STDERR.log").open("xb") as err:
        child = subprocess.Popen(argv, cwd=BASE, stdout=out, stderr=err, start_new_session=True)
        token = process_start(child.pid)
        immutable(target / "PID.json", {"pid": child.pid, "start_token": token, "pgid": child.pid,
                  "task_directory": str(BASE), "argv": argv, "orchestrator_pid": os.getpid()})
        last = 0.0
        try:
          while child.poll() is None:
            reason = cutoff()
            # At -45min wrappers stop opening new calls on their own. Preserve
            # the currently active attempt; STOP/-20min terminate owned work.
            if reason in {"STOP_REQUEST", "CLOSEOUT_ONLY_NO_NEW_ENGINEERING"}:
                stop_owned(child, token, before, step)
                break
            if time.monotonic() - last >= HEARTBEAT_SECONDS:
                heartbeat("running_authorized_remaining_stage", step=step[0], child_pid=child.pid,
                          child_start_token=token, cutoff=reason, active_calls=live_calls())
                last = time.monotonic()
            time.sleep(1)
        finally:
            if child.poll() is None:
                stop_owned(child, token, before, step)
        child.wait()
    observed = validated_output(step)
    result = {"at_utc": utc(), "step": step[0], "exit_code": child.returncode,
              "status": "TERMINAL_OUTPUT_OBSERVED" if observed else "ENGINEERING_OR_SEMANTIC_STAGE_FAILED_NO_REPLAY",
              "output_validation": observed, "stdout_binding": bind(target / "STDOUT.log"),
              "stderr_binding": bind(target / "STDERR.log"), "automatic_retry": False}
    immutable(target / "RESULT.json", result)
    return result


def run() -> int:
    EVENTS.mkdir(parents=True, exist_ok=True)
    with (EVENTS / "orchestrator.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ready = read(READY)
        if ready.get("script_binding") != bind(Path(__file__)) or ready.get("scope") != "SEVEN_REMAINING_SEMANTIC_STEPS_STOP_BEFORE_CONTENT_FREEZE":
            raise RuntimeError("ORCHESTRATOR_ENGINEERING_READY_DRIFT")
        for value in ready["dependency_bindings"]:
            verify(value)
        spec = importlib.util.spec_from_file_location("fgr20_orchestrator_input_validator", BASE / "validate_final.py")
        validator = importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
        validator.verify_input_lock(BASE)
        if (BASE / "CONTENT_FREEZE.json").exists():
            raise RuntimeError("CONTENT_ALREADY_FROZEN_NO_NEW_SEMANTICS")
        wait_idle(require_prerequisite=True)
        verify_v2()
        # Verify actual current-stage final bytes; its v1 semantic work is never
        # rerun or retrofitted with v2.
        import content_continuation_adapter as adapter
        checker = adapter.Continuations.__new__(adapter.Continuations); checker.inputs = []
        checker.validate_ready(PREREQUISITE)
        outputs = []
        consecutive_failures = 0
        for step in STEPS:
            if cutoff():
                break
            try:
                value = run_step(step)
                outputs.append({"step": step[0], **value})
                consecutive_failures = 0 if value["status"] in {"REUSED_TERMINAL_OUTPUT", "TERMINAL_OUTPUT_OBSERVED"} else consecutive_failures + 1
            except Exception as exc:
                value = {"step": step[0], "status": "FAILED_WITHOUT_RETRY", "error": f"{type(exc).__name__}: {exc}"}
                outputs.append(value)
                consecutive_failures += 1
                heartbeat("stage_failed_without_retry", **value, consecutive_engineering_failures=consecutive_failures)
                if "NO_REPLAY" in str(exc) or "NO_AUTOMATIC_RESTART" in str(exc) or "DRIFT" in str(exc):
                    break
            if consecutive_failures >= 3:
                break
        # No content, CPU, GPU, scoring or report subprocess can be reached.
        reason = cutoff()
        if reason is None:
            wait_idle()
        summary = {"at_utc": utc(), "status": "SEMANTIC_SEQUENCE_TERMINAL_HANDOFF" if len(outputs) == len(STEPS) and not reason else "PARTIAL_HANDOFF",
                   "steps": outputs, "cutoff": reason, "consecutive_engineering_failures": consecutive_failures,
                   "stops_before_content_freeze": True, "content_freeze_created": False, "solver_started": False,
                   "training": False, "rerun_original_four": False}
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        immutable(EVENTS / ("HANDOFF_" + run_id + ".json"), summary)
        heartbeat("semantic_sequence_handoff_before_content_freeze", summary=summary)
        return 0 if summary["status"] == "SEMANTIC_SEQUENCE_TERMINAL_HANDOFF" else 2


def self_check() -> dict[str, Any]:
    assert len(STEPS) == 7 and len({step[0] for step in STEPS}) == 7
    assert all(step[2:] != ("179186", "F") for step in STEPS)
    assert [step[0] for step in STEPS] == ["shape_77882_G", "shape_136202_F", "saved_179186_G",
        "saved_136202_G", "saved_109164_G", "unused_77882_F", "unused_failed_109164_F"]
    commands = [command(step) for step in STEPS]
    assert all(argv[0] == PYTHON for argv in commands)
    assert all(not any(token in {"solver_runner.py", "prepare", "analyze", "report", "manifest"} for token in argv) for argv in commands)
    with tempfile.TemporaryDirectory(prefix="fgr20_orchestrator_selfcheck_") as directory:
        base = Path(directory); (base / "cli_calls").mkdir()
        deadline = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)
        (base / "SUPERVISOR_STATUS.json").write_text(json.dumps({"deadline_at_utc": deadline.isoformat()}))
        assert cutoff(base, deadline - dt.timedelta(minutes=46)) is None
        assert cutoff(base, deadline - dt.timedelta(minutes=45)) == "GENERATION_CUTOFF"
        assert cutoff(base, deadline - dt.timedelta(minutes=20)) == "CLOSEOUT_ONLY_NO_NEW_ENGINEERING"
        (base / "FINALIZE_REQUEST").touch()
        assert cutoff(base, deadline - dt.timedelta(minutes=46)) == "FINALIZE_REQUEST"
        assert cutoff(base, deadline - dt.timedelta(minutes=20)) == "CLOSEOUT_ONLY_NO_NEW_ENGINEERING"
        (base / "FINALIZE_REQUEST").unlink()
        (base / "STOP_REQUEST").touch(); assert cutoff(base) == "STOP_REQUEST"; (base / "STOP_REQUEST").unlink()
        attempt = base / "cli_calls/q20_gold_g_77882_candidate_shape_01_initial/attempt_01"; attempt.mkdir(parents=True)
        (attempt / "REQUEST.txt").write_text("mock request bytes without semantics"); (attempt / "START.json").write_text("{}")
        assert family_call_ids(STEPS[0], base) == ["q20_gold_g_77882_candidate_shape_01_initial"]
        assert len(orphan_starts(base)) == 1
        (attempt / "TERMINAL.json").write_text("{}"); assert not orphan_starts(base)
        immutable(base / "receipt.json", {"fixture": True}); verify(bind(base / "receipt.json"))
        try: immutable(base / "receipt.json", {"fixture": False})
        except RuntimeError: pass
        else: raise AssertionError("Differing receipt overwritten")
    return {"status": "PASS", "semantic_calls_started": 0, "subprocesses_started": 0,
            "mock_only": True, "fixed_seven_step_order": True, "current_v1_stage_never_relaunched": True,
            "cutoff_and_STOP_checked": True, "existing_REQUEST_and_orphan_START_detected": True,
            "immutable_receipt_drift_rejected": True, "stops_before_content_freeze": True,
            "commands": commands}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["describe", "self-check", "run"])
    args = parser.parse_args()
    if args.command == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False, indent=2)); return 0
    if args.command == "describe":
        print(json.dumps({"scope": "SEVEN_REMAINING_SEMANTIC_STEPS_STOP_BEFORE_CONTENT_FREEZE",
                          "prerequisite": str(PREREQUISITE), "steps": [{"id": step[0], "argv": command(step),
                          "output_ready": str(output_dir(step) / "CONTINUATION_FINALIZER_READY.json")} for step in STEPS],
                          "v2_ready_required": str(V2_READY), "automatic_stage_retries": 0, "heartbeat_seconds": HEARTBEAT_SECONDS}, indent=2)); return 0
    global STOPPING
    def interrupted(signum, frame):
        global STOPPING
        STOPPING = True
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        return run()
    except Exception as exc:
        heartbeat("orchestrator_stopped_without_automatic_restart", error=f"{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
