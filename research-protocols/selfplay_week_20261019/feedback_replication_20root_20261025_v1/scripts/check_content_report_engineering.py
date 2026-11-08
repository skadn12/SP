#!/usr/bin/env python3
"""Append an engineering-only integration receipt; never dispatch a model.

All generated scientific-shape fixtures live in TemporaryDirectory and are
deleted.  Persistent writes are timestamped engineering CHECKS and READY only.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
import assemble_initial_content as initial
import closeout_audit_driver as closeout
import content_continuation_adapter as continuation
import content_assembly_driver as assembly
import finalize_report as report
import g109164_saved_global7_gold_verification as g109
import g136202_saved_global_gold_verification as g136
import question_driver as q


def saved_wrapper_integration() -> list[dict]:
    q.SOURCE_SELECTION_PATH = BASE / "SOURCE_SELECTION_FINAL.json"
    q.SOURCE_FREEZE_PATH = BASE / "SOURCE_FREEZE.json"
    _, sources = q.load_selection(q.SOURCE_SELECTION_PATH)
    result = []
    for module in (g136, g109):
        frozen = q.read(module.saved.work_dir(module.ROOT, module.ARM) / "RECOVERY_INPUT_FREEZE.json")
        calls = []

        class Mock:
            def call(self, key, prompt):
                calls.append({"call_id": key, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
                raise RuntimeError("ENGINEERING_INTEGRATION_MOCK_NO_DISPATCH")

        old = q.normalize_blind, q.normalize_candidates, q.validate_g_prompt
        try:
            module.rsa.install()
            with tempfile.TemporaryDirectory(prefix="saved_final_integration_", dir=BASE / "runtime_cache") as directory:
                out = Path(directory)
                q.publish(out / "RECOVERY_INPUT_FREEZE.json", frozen)
                if module is g136:
                    reviewer, publisher, _ = module.build_review()
                    candidates = q.read(module.INPUT_WORK / "CANDIDATES_FINAL.json")
                    data = reviewer(module.ROOT, module.ARM, candidates, sources[module.ROOT], module.SELECTION, out, frozen, Mock())
                    publisher(module.ROOT, module.ARM, module.INPUT_WORK, out, frozen, data)
                else:
                    reviewer, _ = module.build_review()
                    candidates = [candidate for candidate in q.read(module.INPUT_WORK / "CANDIDATES_FINAL.json")
                                  if candidate["slot_id"] in module.SLOTS]
                    data = reviewer(module.ROOT, module.ARM, candidates, sources[module.ROOT], module.SELECTION, out, frozen, Mock())
                    module.publish_ready(out, frozen, data)
                checker = continuation.Continuations.__new__(continuation.Continuations)
                checker.inputs = []
                checker.validate_ready(out / continuation.READY_NAME)
                assert len(calls) == 1
                result.append({"root_id": module.ROOT, "arm": module.ARM, "status": "PASS",
                               "actual_saved_candidates": len(candidates),
                               "published_final_candidates": len(q.read(out / "CANDIDATES_FINAL.json")),
                               "published_final_reviews": len(q.read(out / "CONTENT_REVIEW_FINAL.json")["reviews"]),
                               "actual_verifier_calls_started": 0, "mock_attempts": calls,
                               "ready_passes_actual_adapter": True, "retained_input_bindings": len(checker.inputs),
                               "temporary_outputs_removed": True})
        finally:
            q.normalize_blind, q.normalize_candidates, q.validate_g_prompt = old
    return result


def runtime_snapshot_integration() -> dict:
    with tempfile.TemporaryDirectory(prefix="runtime_snapshot_regression_", dir=BASE / "runtime_cache") as directory:
        base = Path(directory)
        for name in ("process_registry", "worker_events", "cli_calls"):
            (base / name).mkdir()
        contents = {"TASK_STATUS.json": b'{"status":"fixture"}\n',
                    "SUPERVISOR_STATUS.json": b'{"status":"fixture"}\n',
                    "PROGRESS.jsonl": b'{"event":"fixture"}\n',
                    "process_registry/test.json": b'{"active":false}\n',
                    "worker_events/round_01.jsonl": b'{"type":"fixture"}\n'}
        for name, raw in contents.items():
            (base / name).write_bytes(raw)
        (base / "cli_calls/frozen.txt").write_bytes(b"fixture scientific CLI artifact\n")
        first = report.capture_runtime_snapshot(base)
        snapshot = report.read_json(base / report.RUNTIME_SNAPSHOT_NAME)
        observed = {row["original_relative_path"]: base64.b64decode(row["content_base64"]) for row in snapshot["records"]}
        assert observed == contents
        for row in snapshot["records"]:
            raw = observed[row["original_relative_path"]]
            assert row["bytes"] == len(raw) and row["sha256"] == hashlib.sha256(raw).hexdigest()
        (base / "PROGRESS.jsonl").write_bytes(contents["PROGRESS.jsonl"] + b'{"event":"later"}\n')
        assert report.capture_runtime_snapshot(base) == first
        listed = {str(path.relative_to(base)) for path in report.manifest_artifact_paths(base)}
        assert report.RUNTIME_SNAPSHOT_NAME in listed and "cli_calls/frozen.txt" in listed and not listed.intersection(contents)
        assert all((base / name).is_file() for name in contents)
    with tempfile.TemporaryDirectory(prefix="runtime_live_rejection_", dir=BASE / "runtime_cache") as directory:
        base = Path(directory)
        (base / "process_registry").mkdir()
        token = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
        (base / "process_registry/live.json").write_text(json.dumps({"active": True, "pid": os.getpid(), "start_token": token, "call_id": "fixture"}))
        try:
            report.capture_runtime_snapshot(base)
        except RuntimeError as exc:
            assert "NO_LIVE_SEMANTIC_CALLS" in str(exc)
        else:
            raise AssertionError("Live semantic fixture was not rejected")
    return {"exact_management_files_captured": True, "all_snapshot_contents_round_trip": True,
            "snapshot_reused_after_live_updates": True, "manifest_binds_snapshot_and_cli_only": True,
            "original_management_records_retained": True, "live_semantic_registry_rejected": True,
            "temporary_fixture_files_removed": True}


def v2_launch_provenance_integration() -> dict:
    actual_paths = assembly.adapter_launch_bindings()
    actual_launches = sorted((BASE / "review_schema_adapter_v2/launches").glob("*.json"))
    assert actual_launches, "Parent-started future v2 launch receipt must exist"
    actual_ready = q.read(BASE / "REVIEW_SCHEMA_ADAPTER_V2_READY.json")
    with tempfile.TemporaryDirectory(prefix="v2_launch_provenance_", dir=BASE / "runtime_cache") as directory:
        base = Path(directory)
        (base / "scripts").mkdir()
        (base / "review_schema_adapter_v2/launches").mkdir(parents=True)
        script = base / "scripts/review_schema_adapter_v2.py"
        script.write_bytes((BASE / "scripts/review_schema_adapter_v2.py").read_bytes())
        ready = json.loads(json.dumps(actual_ready))
        ready["adapter_binding"] = report.bind(script)
        for target in ready["future_targets"]:
            argv = shlex.split(target["command"]); argv[1] = str(script)
            target["command"] = shlex.join(argv)
        ready_path = base / "REVIEW_SCHEMA_ADAPTER_V2_READY.json"
        ready_path.write_text(json.dumps(ready))
        launch = json.loads(json.dumps(q.read(actual_launches[0])))
        launch["adapter_binding"] = report.bind(script)
        launch_path = base / "review_schema_adapter_v2/launches/fixture.json"
        launch_path.write_text(json.dumps(launch))
        accepted = assembly.adapter_launch_bindings(base)
        assert launch_path in accepted and ready_path in accepted and script in accepted
        issues = []
        disclosed = report.schema_adapter_launch_disclosures(base, issues)
        assert not issues and disclosed["observed_launch_receipt_count"] == 1
        assert disclosed["records"][0]["verified_reference_count"] == len(report.nested_bindings(launch))
        rejected = []
        for name in ("wrong_root", "changed_argv", "claimed_extra_budget", "semantic_admission", "nested_binding_drift"):
            bad = json.loads(json.dumps(launch))
            if name == "wrong_root": bad["root_id"] = "cc-train-episode-179186"
            elif name == "changed_argv": bad["driver_argv"] += ["UNAUTHORIZED_EXTRA_ARGUMENT"]
            elif name == "claimed_extra_budget": bad["extra_semantic_budget"] = 1
            elif name == "semantic_admission": bad["semantic_admission_asserted"] = True
            else: bad["fixture_nested"] = {"binding": {**report.bind(script), "sha256": "0" * 64}}
            launch_path.write_text(json.dumps(bad))
            try:
                assembly.adapter_launch_bindings(base)
            except (RuntimeError, ValueError):
                rejected.append(name)
            else:
                raise AssertionError("Unsafe v2 launch fixture accepted: " + name)
        launch_path.write_text(json.dumps(launch))
        ready["adapter_binding"]["sha256"] = "0" * 64
        ready_path.write_text(json.dumps(ready))
        try:
            assembly.adapter_launch_bindings(base)
        except RuntimeError:
            rejected.append("ready_adapter_binding_drift")
        else:
            raise AssertionError("Changed v2 approval adapter accepted")
    return {"status": "PASS", "actual_v2_launch_count_read": len(actual_launches),
            "all_actual_launch_and_nested_bindings_verified": True,
            "actual_launch_provenance_path_count": len(actual_paths),
            "rejected_fixture_cases": rejected, "report_discloses_actual_launch_identity": True,
            "temporary_fixtures_removed": True, "semantic_calls": 0}


def main() -> int:
    codes = ["scripts/assemble_initial_content.py", "scripts/content_continuation_adapter.py",
             "scripts/content_assembly_driver.py", "scripts/finalize_report.py", "scripts/closeout_audit_driver.py",
             "scripts/g136202_saved_global_gold_verification.py", "scripts/g109164_saved_global7_gold_verification.py",
             "scripts/production_driver.py", "scripts/check_content_report_engineering.py"]
    subprocess.run(["/root/miniconda3/bin/python", "-m", "py_compile", *[str(BASE / name) for name in codes]], check=True)
    actual = continuation.Continuations()
    heads = [{"root_id": root, "arm": arm, "continuation_kind": row["continuation_kind"],
              "ready_binding": row["ready_binding"], "initial_work_dir": str(actual.initial_work(
                  root, arm, BASE / "question_authoring" / arm / "roots" / q.root_number(root)))}
             for (root, arm), row in sorted(actual.selected.items())]
    now = dt.datetime.now(dt.timezone.utc)
    stamp, at = now.strftime("%Y%m%dT%H%M%SZ"), now.isoformat()
    input_count = report.validator_module(BASE).verify_input_lock(BASE)
    checks = {"schema_version": "fgr20_content_report_final_engineering_checks_v1", "at_utc": at,
              "status": "PASS", "py_compile_pass": True,
              "self_checks": {"initial": initial.self_check(), "continuation_adapter": continuation.self_check(),
                              "report": report.self_check(), "closeout": closeout.self_check()},
              "real_saved_ready_interface_check": {"terminal_ready_records_read": len(actual.records),
                  "selected_heads": heads, "generation_may_still_be_live": True,
                  "candidate_review_duplicate_gate_logic_changed": False,
                  "real_saved_wrapper_integration": saved_wrapper_integration()},
              "runtime_snapshot_fixture_checks": runtime_snapshot_integration(),
              "v2_launch_provenance_integration": v2_launch_provenance_integration(),
              "input_lock_binding": report.bind(BASE / "INPUT_LOCK.json"),
              "input_lock_bindings_verified": input_count, "input_lock_mismatches": [],
              "semantic_calls_started_by_engineering": 0, "semantic_judgments_authored": 0,
              "final_content_or_report_freezes_published": False}
    check_path = BASE / "mechanical_audits" / ("CONTENT_REPORT_ENGINEERING_CHECKS_" + stamp + ".json")
    report.publish_immutable(check_path, checks)
    names = ["QUESTION_DRIVER_AUDIT_READY.json", "REVIEW_SCHEMA_ADAPTER_READY.json",
             "REVIEW_SCHEMA_ADAPTER_V2_READY.json", "QUESTION_DRIVER_ENGINEERING_HANDOFF_V2.json",
             "SAVED_REVISION_GOLD_VERIFICATION_READY.json", "UNUSED_FAILED_LINEAGE_ADAPTER_READY.json",
             "G136202_SAVED_GLOBAL_GOLD_VERIFICATION_READY.json", "G109164_SAVED_GLOBAL_GOLD_VERIFICATION_READY.json",
             "PRODUCTION_G109_QUESTION_PROCESS_GUARD_READY.json", "PRODUCTION_G109_QUESTION_PROCESS_GUARD_HISTORY_LOCATOR.json",
             "PRODUCTION_QUESTION_PROCESS_GUARD_HISTORY_LOCATOR.json", "ANALYSIS_DRIVER_FREEZE.json"]
    ready = {"schema_version": "fgr20_content_report_closeout_engineering_ready_v1", "at_utc": at, "status": "PASS",
             "code_bindings": [report.bind(BASE / name) for name in codes],
             "engineering_checks_binding": report.bind(check_path),
             "frozen_interface_bindings": [report.bind(BASE / name) for name in names],
             "py_compile_pass": True, "model_calls": 0, "semantic_judgments_authored": 0,
             "content_or_report_freezes_published": False,
             "supersedes_engineering_receipt_path": str(BASE / "CONTENT_REPORT_ENGINEERING_READY_20261024T210528Z.json"),
             "interface_pending": "Actual semantic continuations must become terminal, all driver parents exit, and registry idle before content assembly. G109 final seven true questions must supersede its original eight-object archive; original ORDER2 failure/HOLD remains only in archive and failure ledger. All actual future READY artifacts are revalidated; Mock tests never admit semantic content.",
             "runtime_manifest_policy": "After frozen-report closeout audit, manifest captures FINAL_RUNTIME_SNAPSHOT.json with exact management-file bytes and excludes live TASK_STATUS/PROGRESS/SUPERVISOR_STATUS/process_registry/worker_events paths. Original management records remain; immutable CLI scientific artifacts stay directly bound.",
             "historical_code_policy": "Old production bindings are time-specific historical identities, resolved by both append-only history locators to exact archived bytes; historical old SHA is not the current owned production path.",
             "execution_order": ["After every authorized semantic driver is terminal and registry idle: /root/miniconda3/bin/python scripts/content_assembly_driver.py",
                 "/root/miniconda3/bin/python scripts/solver_cpu_length_precheck.py --runtime-wrapper scripts/solver_runtime_adapter.py",
                 "Fixed Solver prepare/run/analyze and frozen analysis script; no semantic changes after content freeze.",
                 "/root/miniconda3/bin/python scripts/finalize_report.py --write-report",
                 "/root/miniconda3/bin/python scripts/closeout_audit_driver.py prepare",
                 "/root/miniconda3/bin/python scripts/closeout_audit_driver.py run",
                 "/root/miniconda3/bin/python scripts/finalize_report.py --finalize-manifest",
                 "/root/miniconda3/bin/python validate_final.py"]}
    ready_path = BASE / ("CONTENT_REPORT_ENGINEERING_READY_" + stamp + ".json")
    report.publish_immutable(ready_path, ready)
    print(json.dumps({"ready": report.bind(ready_path), "checks": report.bind(check_path),
                      "terminal_records_read": len(actual.records), "input_lock_verified": input_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
