"""Finite authorized first-wave finish, data freeze and CPU preflight only.

Waits for the already running v1 to exit, then invokes one v4 first-wave pass.
No second wave, GPU, supervisor, semantic retries, or existing content edits.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parents[1]
TRAIN_PYTHON = "/root/miniconda3/bin/python"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


driver = module("oet1_finish_status_helper", BASE / "scripts/production_driver.py")


def processes():
    old, data = [], []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit(): continue
        try:
            args = [x.decode(errors="replace") for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if len(args) > 1 and "python" in Path(args[0]).name and Path(args[1]).name == "production_driver.py":
            old.append(int(proc.name))
        if args and Path(args[0]).name == "codex" and "exec" in args and "read-only" in args and any(str(BASE / "cli_calls") in arg for arg in args):
            data.append(int(proc.name))
    return old, data


def heartbeat(stage, **details):
    old, data = processes()
    value = {"at_utc": driver.now(), "owner": "question_engineering", "finite_finish_stage": stage,
             "original_v1_pids": old, "active_data_cli_pids": data, **details}
    driver.status(heartbeat=value)
    driver.log("FINITE_FIRST_WAVE_FINISH_HEARTBEAT", **{k: v for k, v in value.items() if k != "at_utc"})


def run(command, label):
    out = BASE / "finite_finish_logs_v2"; out.mkdir(exist_ok=True)
    log_path = out / (label + ".log")
    started = out / (label + "_START.json")
    driver.write(started, {"started_at_utc": driver.now(), "command": command, "stage": label})
    with log_path.open("xb") as stream:
        proc = subprocess.Popen(command, cwd=BASE, stdout=stream, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            heartbeat(label, child_pid=proc.pid)
            time.sleep(30)
    terminal = out / (label + "_TERMINAL.json")
    driver.write(terminal, {"finished_at_utc": driver.now(), "returncode": proc.returncode,
                           "command_binding": driver.bind(started), "log_binding": driver.bind(log_path)})
    assert proc.returncode == 0, ("FINITE_STAGE_FAILED", label, proc.returncode, str(log_path))
    return driver.bind(terminal)


def main():
    lock = (BASE / "finite_first_wave_finish_v2.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (BASE / "FIRST_WAVE_FINISH_RECEIPT_V2.json").exists(), "FINISH_ALREADY_RECORDED"
    driver.write(BASE / "FIRST_WAVE_FINISH_STARTED_V2.json", {"schema_version": "oet1_finite_first_wave_finish_v2",
                 "started_at_utc": driver.now(), "pid": os.getpid(), "code_binding": driver.bind(__file__),
                 "max_data_cli_workers": 3, "gpu_allowed": False, "second_wave_allowed": False,
                 "wait_for_original_v1_natural_exit": True})
    stages = []
    try:
        driver.status("different_root_paired_source_extension", "IN_PROGRESS")
        while True:
            old, data = processes()
            if not old and not data: break
            heartbeat("WAIT_ORIGINAL_V1_NATURAL_EXIT", finalize_requested=(BASE / "FINALIZE_REQUEST").exists())
            time.sleep(30)
        # Verify every engineering upgrade dependency before the sole continuation.
        frozen = driver.read(BASE / "ENGINEERING_WAVE_RESULTS_UPGRADE_V4.json")
        assert frozen["status"] == "PASS"
        for binding in [frozen["entrypoint"]] + frozen["inherited_bindings"]:
            assert driver.bind(binding["path"]) == binding, ("UPGRADE_CODE_CHANGED", binding["path"])
        stages.append(run([sys.executable, str(BASE / "scripts/recover_pending_native_ready_v2.py")], "RECOVER_NATIVE_READY_METADATA"))
        stages.append(run([sys.executable, str(BASE / "scripts/production_driver_v4.py"), "--action", "verify-only"], "VERIFY_V4_BEFORE_CONTINUATION"))
        deadline = datetime.fromisoformat(driver.read(BASE / "SUPERVISOR_STATUS.json")["deadline_at_utc"])
        generation_allowed = not (BASE / "FINALIZE_REQUEST").exists() and datetime.now(timezone.utc) < deadline - timedelta(minutes=90)
        if generation_allowed:
            assert processes() == ([], []), "CONCURRENT_V1_OR_DATA_CLI_PREVENTS_CONTINUATION"
            driver.status("independent_source_and_question_review", "IN_PROGRESS")
            stages.append(run([sys.executable, str(BASE / "scripts/production_driver_v4.py"), "--action", "first-wave"], "FIRST_WAVE_V4_ONLY"))
        else:
            driver.write(BASE / "FIRST_WAVE_CONTINUATION_SKIPPED_FINALIZE_V2.json", {"status": "SHORTFALL",
                         "at_utc": driver.now(), "finalize_requested": (BASE / "FINALIZE_REQUEST").exists(),
                         "reason": "No new generation within the reserved closeout window; preserve all planned slots."})
        assert processes() == ([], []), "DATA_CLI_STILL_ACTIVE_AT_FREEZE"
        stages.append(run([sys.executable, str(BASE / "scripts/production_integrity_v2.py"), "--repair-metadata",
                           "--output", str(BASE / "PRODUCTION_ENGINEERING_AUDIT_FINAL.json")], "FINAL_PRODUCTION_INTEGRITY"))
        integrity = driver.read(BASE / "PRODUCTION_ENGINEERING_AUDIT_FINAL.json")
        assert integrity["verdict"] == "PASS" and not integrity["active_calls"] and not integrity["actual_tool_events"] and not integrity["failures"], "PRODUCTION_PROVENANCE_BLOCKS_FREEZE"
        qm = module("oet1_finish_frozen_questions", BASE / "scripts/questions.py")
        gate = qm.engineering_gate()
        assert gate["status"] == "FAIL", "KNOWN_STOP_RECEIPT_REQUIRES_FAILED_EXPANSION_GATE"
        assert (BASE / "EXPANSION_STOP_RECEIPT.json").exists()
        qm.save(BASE / "FIRST_WAVE_ENGINEERING_GATE.json", gate)
        export = qm.export()
        assert export["planned"] == 112
        driver.status("different_root_paired_source_extension", "PARTIAL")
        driver.status("independent_source_and_question_review", "PARTIAL")
        driver.status("freeze_matched_training_pool", "IN_PROGRESS")
        pool_binding = qm.freeze_training(include_qfp2=True)
        pool = qm.read(pool_binding["path"])
        assert pool["mode"] == "QFP2_ONLY_PILOT_NEW_ROOT_GATE_FAILED" and pool["question_counts"] == {"GENERIC": 24, "TARGETED": 24}
        driver.status("freeze_matched_training_pool", "COMPLETED", training_scope=pool["mode"])
        stages.append(run([TRAIN_PYTHON, str(BASE / "training/build_schedule.py"), "--pool-freeze", pool_binding["path"]], "BUILD_PILOT_SCHEDULE_CPU"))
        for arm in ("GENERIC", "TARGETED"):
            stages.append(run([TRAIN_PYTHON, str(BASE / "training/runner.py"), "preflight", "--arm", arm,
                               "--output", str(BASE / f"training/PREFLIGHT_{arm}.json")], "CPU_PREFLIGHT_"+arm))
        schedule = qm.read(BASE / "training/schedule/SCHEDULE_FREEZE.json")
        assert schedule["fixed_final_step"] == 24 and all(v["questions"] == 24 and v["microsteps"] == 96 for v in schedule["arms"].values())
        receipt = {"schema_version": "oet1_first_wave_finish_receipt_v2", "status": "COMPLETED_THROUGH_CPU_PREFLIGHT_ONLY",
                   "finished_at_utc": driver.now(), "production_scope": "FIRST_FOUR_ONLY_SECOND_WAVE_STOPPED",
                   "new_root_main_objective_completed": False, "gpu_started_by_this_stage": False,
                   "engineering_gate": qm.bind(BASE / "FIRST_WAVE_ENGINEERING_GATE.json"),
                   "question_freeze": qm.bind(BASE / "QUESTION_FREEZE.json"), "export": export,
                   "training_pool": pool_binding, "schedule": qm.bind(BASE / "training/schedule/SCHEDULE_FREEZE.json"),
                   "cpu_preflights": {arm: qm.bind(BASE / f"training/PREFLIGHT_{arm}.json") for arm in ("GENERIC", "TARGETED")},
                   "question_counts": pool["question_counts"], "fixed_final_optimizer_steps": 24,
                   "stages": stages, "finite_finish_code": qm.bind(__file__)}
        driver.write(BASE / "FIRST_WAVE_FINISH_RECEIPT_V2.json", receipt)
        driver.status(cpu_preflight_ready=True, training_scope=pool["mode"], current_phase="paired_training_with_metrics_and_dev_every20steps")
        driver.log("FINITE_FIRST_WAVE_FINISH_COMPLETE", receipt=qm.bind(BASE / "FIRST_WAVE_FINISH_RECEIPT_V2.json"))
    except Exception as error:
        failure = {"schema_version": "oet1_first_wave_finish_failure_v2", "status": "BLOCKED_ENGINEERING_STAGE",
                   "failed_at_utc": driver.now(), "error": str(error), "traceback": traceback.format_exc(),
                   "completed_stage_receipts": stages, "gpu_started": False, "semantic_retry": False}
        driver.write(BASE / "FIRST_WAVE_FINISH_FAILURE_V2.json", failure)
        driver.status(finite_first_wave_finish_failure=failure)
        raise


if __name__ == "__main__":
    main()
