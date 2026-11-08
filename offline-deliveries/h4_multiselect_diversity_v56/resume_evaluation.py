"""Explicit recovery after c01 finished all 18 Solver calls before interruption.

Preserves the original evaluation ERROR and reuses completed calls unchanged.
No generation, repeated Solver calls, or automatic replay is permitted.
"""
from __future__ import annotations

import json
from pathlib import Path
import signal
import sys
import traceback

sys.dont_write_bytecode = True
import run_collection as r

VERSION = "h4_v56_explicit_evaluation_recovery_v1"
STEM = r.REPORT / "EVALUATION_RECOVERY"


def marker(suffix):
    return Path(str(STEM) + "." + suffix + ".json")


def verify_completed_solver(pipeline, run):
    outer = pipeline.verify(run)
    solver = run / "solver"
    complete = r.read(solver / "COMPLETE.json")
    manifest = r.read(solver / "manifest.json")
    pipeline.evaluator._verify_manifest(solver, manifest)
    pipeline._scope(manifest)
    records = [r.read(path) for path in sorted((solver / "responses").glob("*.json"))]
    expected = [row["request_id"] for row in manifest["requests"]]
    actual = [row["request_id"] for row in records]
    if (complete["status"] != "COMPLETE" or complete["actual_call_count"] != len(expected)
            or len(actual) != len(expected) or len(set(actual)) != len(expected)
            or set(actual) != set(expected) or len(actual) != outer["full_calls_planned"]):
        raise ValueError("completed_solver_call_set_mismatch")
    return {"calls": len(records), "solver_started": r.binding(solver / "STARTED.json"),
            "solver_complete": r.binding(solver / "COMPLETE.json"),
            "responses": [r.binding(path) for path in sorted((solver / "responses").glob("*.json"))]}


def main():
    plan = r.verify_plan()
    r.completion("generation")
    if any(marker(suffix).exists() for suffix in ("STARTED", "COMPLETE", "ERROR")):
        raise ValueError("explicit_recovery_already_recorded_no_replay")
    error = r.read(r.marker("evaluation", "ERROR"))
    if error["error_type"] != "KeyboardInterrupt" or error["current_operation"] != {"chunk": 1, "action": "live"}:
        raise ValueError("unexpected_original_evaluation_interruption")
    audit_path = r.REPORT / "INDEPENDENT_CONTENT_AUDIT.json"
    audit = r.read(audit_path)
    _, pipeline = r.modules()
    first_run = Path(plan["chunks"][0]["evaluation_run"])
    first = verify_completed_solver(pipeline, first_run)
    if first["calls"] != 18 or (first_run / "training_pool").exists():
        raise ValueError("c01_expected_complete_unfinalized_18_calls")
    for chunk in plan["chunks"][1:]:
        if Path(chunk["evaluation_run"]).exists():
            raise ValueError("unstarted_evaluation_run_already_exists")
    started = {"version": VERSION, "started_at_utc": r.now(),
               "orchestrator": r.binding(Path(__file__)),
               "experiment_plan": r.binding(r.REPORT / "EXPERIMENT_PLAN.json"),
               "generation_completion": r.binding(r.marker("generation", "COMPLETE")),
               "original_evaluation_error": r.binding(r.marker("evaluation", "ERROR")),
               "content_audit": r.binding(audit_path), "c01_completed_before_recovery": first,
               "reason": "c01 completed all 18 responses; interruption occurred during backend.close",
               "c01_live_is_not_repeated": True,
               "audit_completion_before_all_solver_calls": False,
               "original_error_preserved": True, "automatic_replay": False}
    r.save(marker("STARTED"), started)
    summaries, current, total_calls = [], None, 0
    try:
        for chunk in plan["chunks"]:
            r.verify_plan()
            r.verify_binding(started["content_audit"])
            r.completion("generation")
            generation_run, run = Path(chunk["generation_run"]), Path(chunk["evaluation_run"])
            current = {"chunk": chunk["index"], "action": "finalize_existing_complete" if chunk["index"] == 1 else "preflight"}
            logs = []
            if chunk["index"] != 1:
                logs.append(r.command("evaluate_and_pool.py", "preflight", run, generation_run))
                manifest = r.read(run / "MANIFEST.json")
                if manifest["full_calls_planned"] > 36 or total_calls + manifest["full_calls_planned"] > r.MAX_SOLVER:
                    raise ValueError("solver_budget_exceeded_before_live")
                current["action"] = "live"
                logs.append(r.command("evaluate_and_pool.py", "live", run))
            completed = verify_completed_solver(pipeline, run)
            for action in ("finalize", "verify"):
                current["action"] = action
                logs.append(r.command("evaluate_and_pool.py", action, run))
            pipeline.verify(run)
            total_calls += completed["calls"]
            report_path = run / "training_pool/REPORT.json"
            report = r.read(report_path)
            summaries.append({"index": chunk["index"], "evaluation_run": str(run),
                "candidate_count": r.read(run / "MANIFEST.json")["candidate_count"],
                "full_solver_calls": completed["calls"], "reused_preinterruption_calls": 18 if chunk["index"] == 1 else 0,
                "pool_report": report,
                "bindings": [r.binding(run / "MANIFEST.json"), r.binding(report_path),
                             r.binding(run / "training_pool/MANIFEST.json"),
                             completed["solver_complete"], *completed["responses"], *logs]})
            print(json.dumps({"event": "recovery_evaluation_chunk_complete", "index": chunk["index"],
                              "full_solver_calls": completed["calls"], "selected": report["new_unique_added"]}), flush=True)
        r.verify_binding(started["original_evaluation_error"])
        r.save(marker("COMPLETE"), {"version": VERSION, "completed_at_utc": r.now(),
            "recovery_start": r.binding(marker("STARTED")),
            "experiment_plan": started["experiment_plan"],
            "original_evaluation_error": started["original_evaluation_error"],
            "generation_completion": started["generation_completion"],
            "content_audit": started["content_audit"], "chunks": summaries,
            "full_solver_calls": total_calls, "reused_completed_calls": 18,
            "new_solver_calls": total_calls - 18, "repeated_solver_calls": 0,
            "generation_calls_during_recovery": 0, "training_calls": 0,
            "original_evaluation_error_preserved": True,
            "audit_completion_before_all_solver_calls": False})
    except BaseException as exc:
        r.save(marker("ERROR"), {"version": VERSION, "failed_at_utc": r.now(),
               "error_type": type(exc).__name__, "error": str(exc),
               "traceback": traceback.format_exc(), "current_operation": current,
               "completed_chunks": summaries, "automatic_replay": False})
        raise


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt("interrupted_by_signal_" + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    main()
