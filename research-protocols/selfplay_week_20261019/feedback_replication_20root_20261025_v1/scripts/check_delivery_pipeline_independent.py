#!/usr/bin/env python3
"""Pure temporary control-flow checks of the finite delivery coordinator."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
import delivery_pipeline as p
import question_driver as q


def controller_scenario(base: Path, mode: str) -> dict:
    calls = []
    ready = base / "READY.json"; ready.write_text('{"engineering_fixture_only":true}')
    def execute(stage):
        calls.append(stage)
        if mode == "three_failures" and stage in {"content", "cpu-length", "replication"}:
            raise RuntimeError("ENGINEERING_FIXTURE_FAILURE_NO_REAL_STAGE")
        if stage == "report":
            (base / "FINAL_REPORT.json").write_text('{"fixture_only":true}')
        if stage == "closeout-run":
            (base / "INDEPENDENT_CLOSEOUT_AUDIT.json").write_text('{"fixture_only":true}')
        return {"stage": stage, "status": "TERMINAL_PRODUCTS_OBSERVED",
                "exit_code": 2 if mode == "CPU_FAIL" and stage == "cpu-length" else 0,
                "subprocess_started": False}
    def products(stage):
        return {"observed_CPU_status": "FAIL" if mode == "CPU_FAIL" else "PASS"}
    original_policy = p.may_launch
    with patch.multiple(p, BASE=base, EVENTS=base / "events", READY=ready, verify_launch_inputs=lambda: {},
                        wait_idle=lambda: None, prerequisites=lambda: {"fixture_only": True},
                        cutoff=lambda: None, heartbeat=lambda *args, **kwargs: None,
                        execute_stage=execute, checked_products=products,
                        may_launch=lambda stage, blocked, failures: original_policy(stage, blocked, failures, base)):
        assert p.run() == 0
    handoffs = list((base / "events").glob("HANDOFF_*.json")); assert len(handoffs) == 1
    handoff = q.read(handoffs[0])
    assert "prepare" not in calls and "run" not in calls and "analyze" not in calls
    assert calls.count("closeout-run") == 1 and calls.count("report") == 1
    assert handoff["response_imputation_performed"] is False
    if mode == "three_failures":
        assert handoff["new_generation_solver_and_engineering_disabled"] is True
        assert handoff["only_once_existing_frozen_report_closeout_remains_authorized"] is True
    return {"scenario": mode, "mock_dispatches": calls,
            "prepare_run_analyze_never_dispatched": True,
            "unique_existing_report_closeout_still_dispatched": True,
            "actual_subprocesses": 0}


def main() -> int:
    assert p.bind(p.__file__)["sha256"] == "7461b74a73503e6921add4f46fb5ff13f4cf4c61039d3809fd4e5bdd5f3bc1a9"
    cache = BASE / "runtime_cache/delivery_independent"; cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="delivery_", dir=cache) as directory:
        root = Path(directory); scenarios = []
        for mode in ("CPU_FAIL", "three_failures"):
            base = root / mode; base.mkdir()
            scenarios.append(controller_scenario(base, mode))
        base = root / "guards"; base.mkdir(); events = base / "events"
        (base / "execution").mkdir(); (base / "execution/STARTED.json").write_text("{}")
        def forbidden(*args, **kwargs):
            raise AssertionError("ACTUAL_SUBPROCESS_OR_SIGNAL_REACHED_BY_INDEPENDENT_CHECK")
        with patch.multiple(p, BASE=base, EVENTS=events, wait_idle=lambda: None,
                            production_receipts=lambda stage: [], checked_products=lambda stage: None), \
                patch.object(p.subprocess, "Popen", forbidden):
            try: p.execute_stage("run")
            except RuntimeError as exc: assert "SOLVER_STARTED_NO_RAW_NO_REPLAY" in str(exc)
            else: raise AssertionError("Unfrozen prior Solver start replayed")
            with patch.object(p, "cutoff", return_value="DEADLINE"):
                try: p.execute_stage("cpu-length")
                except RuntimeError as exc: assert "FINAL20_NO_NEW_SOLVER_OR_PREFLIGHT" in str(exc)
                else: raise AssertionError("Deadline crossing opened CPU engineering")
            with patch.object(p, "cutoff", return_value="STOP_REQUEST"):
                try: p.execute_stage("report")
                except RuntimeError as exc: assert "STOP_REQUEST" in str(exc)
                else: raise AssertionError("STOP opened report stage")
        stable = base / "scientific.json"; stable.write_text("{}")
        script = p.bind(p.__file__)
        review = base / "independent_engineering.json"
        review.write_text(json.dumps({"status": "PASS_ENGINEERING_ONLY", "script_binding": script,
                                      "reviewer_agent": "/root/analysis_audit"}))
        ready = base / "READY.json"
        ready.write_text(json.dumps({"status": "READY_ENGINEERING_ONLY", "script_binding": script,
            "scope": "SETTLED_REPLICATION_DELIVERY_ONLY", "semantic_calls_started_by_engineering": 0,
            "dependency_bindings": [p.bind(stable)], "independent_engineering_review_binding": p.bind(review)}))
        (base / "INPUT_LOCK.json").write_text(json.dumps({"bindings": [p.bind(stable)]}))
        with patch.multiple(p, BASE=base, EVENTS=events, READY=ready):
            p.verify_launch_inputs()
            stable.write_text('{"fixture_drift":true}')
            try: p.verify_launch_inputs()
            except RuntimeError as exc: assert "BINDING_DRIFT" in str(exc)
            else: raise AssertionError("Scientific input drift accepted at launch")
    checks = p.self_check()
    assert checks["semantic_calls_started"] == checks["solver_calls_started"] == checks["subprocesses_started"] == 0
    parents = p.active_parents()
    assert all(row["script"] != str(BASE / "scripts/status_heartbeat.py") for row in parents)
    now = dt.datetime.now(dt.timezone.utc)
    receipt = {"schema_version": "fgr20_delivery_pipeline_independent_engineering_checks_v1",
        "at_utc": now.isoformat(), "status": "PASS_ENGINEERING_ONLY", "reviewer_agent": "/root/analysis_audit",
        "check_script_binding": q.binding(__file__), "script_binding": q.binding(p.__file__),
        "owner_engineering_checks_binding": q.binding(BASE / "mechanical_audits/DELIVERY_PIPELINE_ENGINEERING_CHECKS_20261024T221204Z.json"),
        "controller_scenarios": scenarios, "unfrozen_Solver_START_refuses_replay": True,
        "pre_spawn_deadline_crossing_blocks_CPU_and_STOP_blocks_all": True,
        "every_launch_revalidates_scientific_inputs_and_independent_review": True,
        "actual_parent_scope_excludes_exact_persistent_status_heartbeat": True,
        "existing_self_check_reexecuted": checks,
        "temporary_controller_outputs_removed": True,
        "actual_semantic_calls": 0, "actual_solver_calls": 0, "actual_pipeline_launches": 0,
        "semantic_judgments_authored": 0, "existing_frozen_files_modified": False}
    path = BASE / "mechanical_audits" / ("DELIVERY_PIPELINE_INDEPENDENT_CHECKS_" + now.strftime("%Y%m%dT%H%M%SZ") + ".json")
    q.publish(path, receipt); print(json.dumps(q.binding(path), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
