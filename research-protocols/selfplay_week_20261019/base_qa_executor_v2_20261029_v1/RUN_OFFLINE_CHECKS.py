"""Reproducible offline tests and synthetic smoke, never a model/API runner.

Run from the project directory with Python -B. Output is exclusive: choose a new
--output directory for a repeat. Synthetic fixture QA is not new dataset QA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

ARTIFACTS = Path(__file__).resolve().parent
PROJECT = ARTIFACTS.parents[2]
sys.path.insert(0, str(PROJECT))


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def offline_smoke(out):
    from scripts.current_method import base_qa_executor_v2 as ex
    from scripts.current_method.test_base_qa_executor_v2 import make_selection, ScriptedMock

    fixtures = out / "MOCK_SYNTHETIC_ONLY"
    fixtures.mkdir()
    selection = make_selection(fixtures)
    scenarios = []
    cases = [("normal_three_roots", None, list(ex.ROOTS), 12, "PASS_STATIC_REVIEW"),
             ("gold_conflict", "gold-conflict", [ex.ROOTS[0]], 3, "HOLD"),
             ("invalid_reviewer_json", "invalid-review", [ex.ROOTS[0]], 3, "HOLD"),
             ("incomplete_reviewer", "incomplete-review", [ex.ROOTS[0]], 3, "HOLD"),
             ("reviewer_ambiguity", "ambiguous", [ex.ROOTS[0]], 3, "HOLD"),
             ("reviewer_insufficient", "insufficient", [ex.ROOTS[0]], 3, "HOLD"),
             ("timeout", "timeout", [ex.ROOTS[0]], 1, "HOLD"),
             ("connection_reset", "disconnect", [ex.ROOTS[0]], 1, "HOLD"),
             ("no_safe_qa", "no-safe", [ex.ROOTS[0]], 1, "NO_SAFE_QA"),
             ("invalid_author", "bad-author", [ex.ROOTS[0]], 2, "HOLD")]
    for name, failure, roots, count, status in cases:
        state = fixtures / name
        ex.initialize(state, selection)
        mock = ScriptedMock(failure)
        engine = ex.Executor(state, mode="mock", transport=mock, token_counter=lambda _: 100)
        results = [engine.run_root(root) for root in roots]
        assert all(r["status"] == status for r in results), results
        snapshot = engine.ledger.snapshot()
        assert snapshot["attempt_count"] == count == len(mock.calls)
        save(state / "LEDGER_EXPORT.json", snapshot)
        engine.ledger.close()
        restarted = ex.Executor(state, mode="mock", transport=mock, token_counter=lambda _: 100)
        try:
            try:
                restarted.run_root(roots[0])
            except ex.BudgetError:
                pass
            else:
                raise AssertionError("restart reopened a root opportunity")
            assert restarted.ledger.snapshot()["attempt_count"] == count == len(mock.calls)
        finally:
            restarted.ledger.close()
        if status != "PASS_STATIC_REVIEW":
            assert not (state / "roots" / roots[0] / "COMPARISON_PACKET.json").exists()
        scenarios.append({"scenario": name, "fixture_only": True,
            "results": results, "mock_requests": count, "real_requests": 0,
            "restart_blocked": True, "ledger_export": str(state / "LEDGER_EXPORT.json")})

    # Real reviewed sources are only bound and dry-prepared, never passed to a
    # model or synthetic author. The canonical live directory is never created.
    real_selection = ARTIFACTS.parent / "base_qa_p0_20261027_v1/offline_smoke_v1/SOURCE_SELECTION_THREE_ROOTS.json"
    real_state = out / "REAL_SOURCES_DRY_RUN_ONLY"
    manifest = ex.initialize(real_state, real_selection)
    engine = ex.Executor(real_state)
    dry = [engine.run_root(root) for root in ex.ROOTS]
    assert all(r["model_calls"] == 0 for r in dry)
    try:
        ex.Executor(real_state, mode="live")
    except ex.ExecutionHold:
        pass
    else:
        raise AssertionError("unapproved live initialization was admitted")
    assert not (real_state / "attempts.sqlite3").exists()
    return {"synthetic_scenarios": scenarios, "real_source_dry_run": dry,
            "real_source_bindings": manifest["sources"], "live_without_approval": "HOLD_BEFORE_LEDGER",
            "actual_new_dataset_questions": 0, "actual_model_calls": 0,
            "solver_runs": 0, "training_runs": 0, "source_expansions": 0,
            "empirical_quality_established": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "offline_checks_v1")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit_path = output / "NETWORK_EVENTS.jsonl"
    audit_path.touch(exist_ok=False)
    os.environ.update({"QA_OFFLINE_NETWORK_AUDIT": str(audit_path),
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    guard = ARTIFACTS / "offline_guard"
    os.environ["PYTHONPATH"] = os.pathsep.join([str(guard), str(PROJECT)])
    sys.path.insert(0, str(guard))
    import sitecustomize  # installs the same guard in this parent process
    started = time.monotonic()
    modules = ["scripts.current_method.test_base_qa_executor_v2",
               "scripts.current_method.test_qa_api_budget_v2",
               "scripts.current_method.test_qa_api_transport_v2",
               "scripts.current_method.test_base_qa_executor_v2_adversarial",
               "scripts.current_method.test_base_qa_pipeline_v1",
               "scripts.current_method.test_question_presentation_gate",
               "scripts.current_method.test_production_question_gate"]
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    with (output / "UNITTEST_OUTPUT.txt").open("x", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit("Offline tests failed; inspect UNITTEST_OUTPUT.txt")
    smoke = offline_smoke(output)
    save(output / "SMOKE_RESULT.json", smoke)
    baseline = json.loads((ARTIFACTS / "BASELINE_BINDINGS.json").read_text())
    checked = []
    for row in baseline["files"]:
        raw = Path(row["path"]).read_bytes()
        ok = len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"]
        checked.append({"path": row["path"], "unchanged": ok})
    assert all(row["unchanged"] for row in checked), "frozen historical file drift"
    save(output / "BASELINE_INTEGRITY.json", {"checked_count": len(checked),
         "all_unchanged": True, "files": checked})
    events = audit_path.read_text().splitlines()
    assert not events, "A real network attempt was blocked: offline verification failed"
    summary = {"version": "base_qa_executor_v2_20261029_v1", "tests_run": result.testsRun,
        "tests_passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "modules": modules, "seconds": round(time.monotonic() - started, 3),
        "network_attempts": len(events), "network_guard": "Python audit hook; inherited by Python child processes",
        "real_api_calls": 0, "real_solver_calls": 0, "training_runs": 0,
        "historical_files_checked": len(checked), "historical_files_unchanged": True,
        "actual_new_dataset_questions": 0, "smoke_scenarios": len(smoke["synthetic_scenarios"]),
        "live_readiness": "HOLD_UNVERIFIED_SERVICE_CAPABILITY_AND_NO_USER_APPROVAL"}
    save(output / "TEST_RESULT.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
