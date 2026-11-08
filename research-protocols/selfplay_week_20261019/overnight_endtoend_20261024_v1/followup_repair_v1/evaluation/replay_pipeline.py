#!/usr/bin/env python3
"""Serial, durable replay of the fixed OET1 fallback after a config repair."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE / "corrected_replay"
REPO = HERE.parents[4]
RUNNER = HERE / "corrected_runner_v2.py"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def save(path, value, replace=False):
    path = Path(path)
    if replace:
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
        with tmp.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    else:
        with path.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")


def main():
    smoke = json.loads((HERE / "smoke_v2/RESULT.json").read_text())
    if smoke["status"] != "PASS":
        raise RuntimeError("Target-task engineering smoke must pass before replay")
    ROOT.mkdir(exist_ok=True)
    status_path = ROOT / "REPLAY_STATUS.json"
    if status_path.exists():
        raise RuntimeError("Replay already started; inspect its status instead of duplicating it")
    tasks = [
        ("B1_COMMON_DEV", ["evaluate", "--role", "B1"]),
        ("GENERIC_FIXED_TRAINING", ["train", "--arm", "GENERIC"]),
        ("GENERIC_COMMON_DEV", ["evaluate", "--role", "B1_GENERIC"]),
        ("TARGETED_FIXED_TRAINING", ["train", "--arm", "TARGETED"]),
        ("TARGETED_COMMON_DEV", ["evaluate", "--role", "B1_TARGETED"]),
    ]
    state = {
        "status": "RUNNING", "started_at_utc": now(), "last_update_at_utc": now(),
        "scope": "Corrected runtime replay of old-QFP2 fallback only",
        "main_new_root_objective_complete": False,
        "configuration_amendment": binding(HERE / "CONFIG_BRIDGE_AMENDMENT_V2.json"),
        "pipeline": binding(Path(__file__)), "smoke": binding(HERE / "smoke_v2/RESULT.json"),
        "completed_tasks": [], "active_task": None,
    }
    save(status_path, state)
    for name, argv in tasks:
        logfile = ROOT / (name + ".log")
        state.update(active_task=name, last_update_at_utc=now())
        save(status_path, state, True)
        print(json.dumps({"event": "task_start", "name": name, "at_utc": now()}), flush=True)
        started = time.monotonic()
        with logfile.open("x") as log:
            process = subprocess.Popen([sys.executable, str(RUNNER), *argv], cwd=REPO,
                                       stdout=log, stderr=subprocess.STDOUT)
            last_heartbeat = started
            while process.poll() is None:
                time.sleep(1)
                if time.monotonic() - last_heartbeat >= 45:
                    state.update(last_update_at_utc=now(), active_pid=process.pid)
                    save(status_path, state, True)
                    print(json.dumps({"event": "heartbeat", "task": name,
                                      "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
                    last_heartbeat = time.monotonic()
        receipt = {"name": name, "exit_code": process.returncode, "log": binding(logfile),
                   "wall_seconds": round(time.monotonic() - started, 3), "finished_at_utc": now()}
        if process.returncode:
            state.update(status="STOPPED_ENGINEERING_FAILURE", failure=receipt,
                         last_update_at_utc=now(), active_task=None)
            save(status_path, state, True)
            raise RuntimeError("Replay task failed: " + name)
        state["completed_tasks"].append(receipt)
        state.update(last_update_at_utc=now(), active_task=None)
        save(status_path, state, True)
        print(json.dumps({"event": "task_complete", **receipt}), flush=True)

    models = {}
    for name in ("B1", "B1_GENERIC", "B1_TARGETED"):
        directory = ROOT / "final_evaluation" / name
        summary = json.loads((directory / "SUMMARY.json").read_text())
        raw = [json.loads(line) for line in (directory / "RAW.jsonl").read_text().splitlines()]
        models[name] = {
            "summary": binding(directory / "SUMMARY.json"),
            "corrected_run_receipt": binding(directory / "CORRECTED_RUN_RECEIPT.json"),
            "metrics": summary["metrics"], "requests": len(raw),
            "outputs_at_32_token_cap": sum(r["output_tokens"] == 32 for r in raw),
            "pure_whitespace_outputs": sum(not r["raw_output"].strip() for r in raw),
        }
    training = {name: binding(ROOT / "runs" / name / "RESULT.json") for name in ("GENERIC", "TARGETED")}
    report = {
        "status": "CORRECTED_FALLBACK_REPLAY_COMPLETED", "completed_at_utc": now(),
        "configuration_amendment": binding(HERE / "CONFIG_BRIDGE_AMENDMENT_V2.json"),
        "models": models, "training": training,
        "new_source_causal_benefit_established": False, "main_new_root_objective_complete": False,
        "short_control_all_correct": smoke["short_control_all_correct"],
        "old_wrong_rope_adapters_reused": False, "test_requests": 0,
        "interpretation": "Same fixed fallback data/schedules/strict scoring, corrected config; historical Dev regression only.",
    }
    save(ROOT / "CORRECTED_COMPARISON.json", report)
    state.update(status="COMPLETED", active_task=None, last_update_at_utc=now(),
                 comparison=binding(ROOT / "CORRECTED_COMPARISON.json"))
    save(status_path, state, True)
    print(json.dumps({"event": "completed", "comparison": state["comparison"]}), flush=True)


if __name__ == "__main__":
    main()
