"""Independent stdlib-only audit of v99 exposed paired Planner regression.

No engine imports, provider calls, source reservations or raw edits.  This
checks frozen inputs and artifact accounting, not semantic correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "regression"
OLD = BASE.parent / "h4_event_identity_correction_pilot_v98"
OLD_RUN = OLD / "expansion_engine/runs/four_roots_B_scene01"
ROOTS = ["cc-train-episode-19053", "cc-train-episode-25652"]
CELL_ORDER = [(ROOTS[0], "control"), (ROOTS[0], "treatment"),
              (ROOTS[1], "treatment"), (ROOTS[1], "control")]
LEGACY = BASE.parent / "h4_r02_dualroute_feedback_v1"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def context(payload: dict[str, Any], marker: str = "\nINPUT_JSON\n") -> dict[str, Any]:
    messages = payload["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user", "request_message_shape"
    text = messages[0]["content"]
    assert text.count(marker) == 1, "request_marker_count"
    value = json.JSONDecoder().raw_decode(text.split(marker, 1)[1])[0]
    assert isinstance(value, dict), "request_context_shape"
    return value


def old_request(root: str) -> Path:
    paths = list((OLD_RUN / "requests").glob(f"*-{root}-B-S01-planner-1/REQUEST.json"))
    assert len(paths) == 1, "old_request_not_unique"
    return paths[0]


def check(condition: bool, code: str, failures: list[dict[str, Any]], **details: Any) -> None:
    if not condition:
        failures.append({"code": code, **details})


def audit_preflight(manifest: dict[str, Any], failures: list[dict[str, Any]], *, live: bool) -> dict[str, Any]:
    expected = {"version": "v99_exposed_paired_planner_regression_v1", "roots": ROOTS,
                "max_planner_attempts_per_cell": 2, "max_event_review_attempts_per_cell": 2,
                "max_logical_requests": 16, "http_max": 32, "max_tokens": 4096,
                "first_attempt_contexts_paired": True, "retry_contexts_paired": False,
                "fresh_source_count": 0, "new_source_reservations": 0, "training_allowed": False}
    for field, value in expected.items():
        check(manifest.get(field) == value, "manifest_value_changed", failures,
              field=field, expected=value, actual=manifest.get(field))
    check([(x.get("root_id"), x.get("arm")) for x in manifest.get("cells", [])] == CELL_ORDER,
          "cell_order_changed", failures)
    bindings = manifest.get("files", [])
    by_path = {row["path"]: row["sha256"] for row in bindings}
    check(len(by_path) == len(bindings), "duplicate_manifest_path", failures)
    changed = []
    for path, expected_hash in by_path.items():
        p = Path(path)
        actual = sha(p) if p.is_file() else None
        if actual != expected_hash:
            changed.append({"path": path, "expected": expected_hash, "actual": actual})
    check(not changed, "frozen_manifest_file_changed", failures, files=changed)
    required = {BASE / "evidence/regression_driver.py"}
    required.update((BASE / "expansion_engine").glob("*.py"))
    required.update((OLD / "expansion_engine").glob("*.py"))
    required.update(LEGACY.glob("*.py"))
    for root in ROOTS:
        required.update({RUN / "sources" / f"{root}.json", OLD_RUN / "sources" / f"{root}.json",
                         OLD_RUN / "roots" / root / "B/S01.LATEST_STATE.json"})
        required.update(old_request(root).parent / name for name in
                        ("REQUEST.json", "PROOF.json", "RESPONSE.json", "VALIDATION.json"))
    missing = sorted(str(p) for p in required if str(p.resolve()) not in by_path)
    check(not missing, "required_dependency_not_frozen", failures, paths=missing)

    prepared = []
    for cell in manifest.get("cells", []):
        root, arm = cell["root_id"], cell["arm"]
        binding = cell["prepared_request"]
        path = Path(binding["path"])
        check(sha(path) == binding["sha256"] == by_path.get(str(path.resolve())),
              "prepared_binding_mismatch", failures, root=root, arm=arm)
        record = read(path)
        payload, proof = record["payload"], record["proof"]
        old_payload = read(old_request(root))
        ctx, old_ctx = context(payload), context(old_payload)
        check(ctx == old_ctx, "first_context_not_exact_old_input", failures, root=root, arm=arm)
        metadata = lambda p: {k: v for k, v in p.items() if k != "messages"}
        check(metadata(payload) == metadata(old_payload), "model_parameters_changed", failures,
              root=root, arm=arm)
        if arm == "control":
            check(payload == old_payload, "control_first_payload_not_exact_old", failures, root=root)
        else:
            check(payload != old_payload, "treatment_prompt_did_not_change", failures, root=root)
            text = payload["messages"][0]["content"]
            check("COMPLETE OUTPUT SCHEMA" in text and "Add exactly one extra field" not in text,
                  "treatment_schema_instruction_unified_check_failed", failures, root=root)
        history = read(RUN / "sources" / f"{root}.json")["turns"]
        check(ctx.get("cumulative_history") == history, "full_history_not_frozen", failures,
              root=root, arm=arm)
        check(ctx.get("latest_state") == read(OLD_RUN / "roots" / root / "B/S01.LATEST_STATE.json"),
              "latest_state_not_frozen", failures, root=root, arm=arm)
        fb = (RUN / "feedback" / root / "dataset_new_event__planner.txt").read_text(encoding="utf-8")
        check(bool(fb) and payload["messages"][0]["content"].count(fb) == 1,
              "feedback_exact_once_failed", failures, root=root, arm=arm)
        check(proof.get("payload_sha256") == digest(payload), "prepared_proof_hash_failed", failures,
              root=root, arm=arm)
        prepared.append({"root_id": root, "arm": arm, "path": str(path),
                         "file_sha256": sha(path), "payload_sha256": digest(payload),
                         "context_sha256": digest(ctx), "history_turns": len(history),
                         "branch_hash": ctx["branch_base_sha256"]})

    test_path = BASE / "evidence/REGRESSION_SCHEMA_TEST_RESULTS_FINAL.json"
    test = read(test_path)
    check(test.get("status") == "PASS" and test.get("live") is False
          and test.get("provider_calls") == test.get("logical_requests") == 0,
          "offline_schema_test_not_pass_no_live", failures)
    test_changed = []
    for relative, expected_hash in test.get("engine_bindings", {}).items():
        actual = sha(BASE / relative)
        if actual != expected_hash:
            test_changed.append({"path": relative, "expected": expected_hash, "actual": actual})
    check(bool(test.get("engine_bindings")) and not test_changed,
          "offline_schema_test_code_changed", failures, files=test_changed)

    capacity_path = RUN / "CAPACITY_PREFLIGHT.json"
    if not capacity_path.exists():
        check(False, "capacity_preflight_missing", failures)
        capacity = None
    else:
        capacity = read(capacity_path)
        check(capacity.get("status") == "PASS", "capacity_preflight_failed", failures)
        rows = capacity.get("cells", [])
        check(len(rows) == 4, "capacity_preflight_cell_count", failures)
        for cell in rows:
            key = (cell.get("root_id"), cell.get("arm"))
            original = next((x for x in prepared if (x["root_id"], x["arm"]) == key), None)
            check(original is not None, "capacity_preflight_unknown_cell", failures, cell=list(key))
            check(cell.get("capacity", {}).get("status") == "PASS", "cell_capacity_not_pass", failures,
                  cell=list(key))
            if original:
                check(cell.get("prepared_request_sha256") in {original["file_sha256"], original["payload_sha256"]},
                      "capacity_prepared_request_binding", failures, cell=list(key))
    forbidden = []
    for name in ("SOURCE_RESERVATIONS.json", "UNIQUE_EVENT_ALLOCATION.json", "dataset_cards_result.json"):
        if (RUN / name).exists(): forbidden.append(str(RUN / name))
    for name in ("roots", "shards", "source_reviews", "materialized_sources"):
        if (RUN / name).exists(): forbidden.append(str(RUN / name))
    check(not forbidden, "regression_contains_production_source_artifacts", failures, paths=forbidden)
    if not live:
        check(not (RUN / "LIVE_STARTED.json").exists() and not (RUN / "requests").exists(),
              "preflight_already_has_live_artifacts", failures)
    return {"frozen_file_count": len(bindings), "prepared_cells": prepared,
            "offline_schema_test_sha256": sha(test_path),
            "capacity_preflight_sha256": sha(capacity_path) if capacity_path.exists() else None}


def audit_live(manifest: dict[str, Any], failures: list[dict[str, Any]]) -> dict[str, Any]:
    request_dirs = sorted((RUN / "requests").glob("*"))
    check(len(request_dirs) <= 16, "logical_budget_exceeded", failures, actual=len(request_dirs))
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    candidate_plans: dict[tuple[str, str], dict[str, Any]] = {}
    prior_planner_raw: dict[tuple[str, str], str] = {}
    prior_event_context: dict[tuple[str, str], dict[str, Any]] = {}
    rows = []
    http_numbers = []
    for number, directory in enumerate(request_dirs, 1):
        check(directory.name.startswith(f"{number:03d}-"), "logical_number_gap", failures,
              directory=directory.name)
        payload = read(directory / "REQUEST.json")
        proof = read(directory / "PROOF.json")
        role = proof.get("role")
        root, arm = proof.get("regression_cell_root"), proof.get("experiment_arm")
        key = (root, arm)
        check(key in CELL_ORDER, "live_unknown_cell", failures, directory=directory.name,
              root=root, arm=arm)
        check(proof.get("regression_only") is True and proof.get("fresh_source") is False,
              "live_exposure_labels_missing", failures, directory=directory.name)
        check(role in {"planner", "event_state_extractor"}, "forbidden_role_called", failures,
              directory=directory.name, role=role)
        counts[key][role] += 1
        check(proof.get("root_id") == root and proof.get("route") == "dataset_new_event",
              "request_root_route_not_cell", failures, directory=directory.name)
        check(payload.get("max_tokens") == 4096, "live_max_tokens_changed", failures, directory=directory.name)
        check(proof.get("payload_sha256") == digest(payload), "live_payload_proof_mismatch", failures,
              directory=directory.name)
        if role == "planner":
            check(counts[key]["event_state_extractor"] == 0, "planner_called_after_event_endpoint", failures,
                  directory=directory.name)
            ctx = context(payload)
            old_ctx = context(read(old_request(root)))
            repair = ctx.get("targeted_repair")
            if counts[key][role] > 1:
                check(isinstance(repair, dict) and repair.get("previous_response") == prior_planner_raw.get(key),
                      "planner_retry_previous_response_not_actual", failures, directory=directory.name)
            ctx.pop("targeted_repair", None)
            check(ctx == old_ctx, "live_planner_context_outside_repair_changed", failures,
                  directory=directory.name)
            if counts[key][role] == 1:
                prepared = next(x for x in manifest["cells"] if (x["root_id"], x["arm"]) == key)
                check(payload == read(Path(prepared["prepared_request"]["path"]))["payload"],
                      "live_first_request_not_prepared", failures, directory=directory.name)
            candidate_file = directory / "CANDIDATE_PLAN.json"
            if candidate_file.exists():
                plan = read(candidate_file)
                scenes = plan.get("scenes")
                if isinstance(scenes, list) and len(scenes) == 1:
                    candidate_plans[key] = scenes[0]
            raw_file = directory / "RESPONSE.json"
            if raw_file.exists(): prior_planner_raw[key] = read(raw_file)["text"]
        else:
            ctx = context(payload, "\nEVENT_STATE_INPUT_JSON\n")
            old_ctx = context(read(old_request(root)))
            check(ctx.get("cumulative_history") == old_ctx["cumulative_history"],
                  "live_event_full_history_changed", failures, directory=directory.name)
            check(ctx.get("latest_state") == old_ctx["latest_state"],
                  "live_event_latest_state_changed", failures, directory=directory.name)
            check(proof.get("scene_plan_sha256") == digest(ctx.get("scene_plan")),
                  "live_event_plan_proof_changed", failures, directory=directory.name)
            check(ctx.get("scene_plan") == candidate_plans.get(key),
                  "live_event_plan_not_actual_valid_planner_candidate", failures, directory=directory.name)
            event_common = dict(ctx); event_common.pop("targeted_repair", None)
            if key in prior_event_context:
                check(event_common == prior_event_context[key], "event_retry_context_outside_repair_changed",
                      failures, directory=directory.name)
            else:
                prior_event_context[key] = event_common
        attempts = sorted(directory.glob("HTTP_ATTEMPT*.json"))
        check(1 <= len(attempts) <= 2, "http_attempt_count_invalid", failures, directory=directory.name)
        http_statuses = []
        for n, attempt_file in enumerate(attempts, 1):
            attempt = read(attempt_file)
            http_numbers.append(attempt.get("http_attempt_number"))
            check(attempt.get("actual_payload_equals_frozen") is True
                  and attempt.get("payload_sha256") == digest(payload),
                  "http_payload_changed", failures, directory=directory.name, attempt=n)
            result_file = directory / f"HTTP_RESULT_{n:02d}.json"
            result = read(result_file)
            http_statuses.append(result.get("http_status"))
            if n > 1:
                check(http_statuses[n - 2] in {503, 504}, "unallowed_transport_retry", failures,
                      directory=directory.name, previous_http_status=http_statuses[n - 2])
        response_file = directory / "RESPONSE.json"
        if response_file.exists():
            response = read(response_file)
            check(response.get("metadata", {}).get("finish_reason") == "stop",
                  "incomplete_success_response", failures, directory=directory.name)
        rows.append({"request": directory.name, "root_id": root, "arm": arm,
                     "role": role, "http_attempts": len(attempts), "http_statuses": http_statuses})
    check(sorted(http_numbers) == list(range(1, len(http_numbers) + 1)),
          "http_sequence_not_complete_unique", failures)
    check(len(http_numbers) <= 32, "http_budget_exceeded", failures, actual=len(http_numbers))
    for (root, arm), count in counts.items():
        check(count["planner"] <= 2 and count["event_state_extractor"] <= 2,
              "cell_attempt_budget_exceeded", failures, root=root, arm=arm, counts=dict(count))

    report = read(RUN / "REPORT.json")
    check(report.get("status") == "COMPLETE" and len(report.get("cells", [])) == 4,
          "report_not_complete_four_cells", failures)
    check(report.get("logical_requests") == len(request_dirs), "report_logical_count_wrong", failures)
    check(report.get("http_attempts") == len(http_numbers), "report_http_count_wrong", failures)
    for field in ("new_sources", "new_dialogue_turns", "new_questions", "solver_calls", "training_jobs"):
        check(report.get(field) == 0, "regression_claims_production_output", failures,
              field=field, actual=report.get(field))
    check(report.get("fresh_yield_measured") is False and report.get("semantic_reliability_certified") is False,
          "regression_overclaims_reliability", failures)
    for cell in report.get("cells", []):
        key = (cell.get("root_id"), cell.get("arm"))
        actual = counts[key]
        check(cell.get("event_review_logical_requests") == actual["event_state_extractor"],
              "report_event_request_count_wrong", failures, cell=list(key))
        check(cell.get("event_review_reached") == (actual["event_state_extractor"] > 0),
              "report_event_reached_wrong", failures, cell=list(key))
        check(len(cell.get("attempts", [])) <= actual["planner"],
              "report_planner_attempt_count_wrong", failures, cell=list(key))
    return {"logical_requests": len(request_dirs), "http_attempts": len(http_numbers),
            "requests": rows, "per_cell": [{"root_id": k[0], "arm": k[1], **dict(v)} for k, v in counts.items()],
            "report_sha256": sha(RUN / "REPORT.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "live"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_path = RUN / "REGRESSION_MANIFEST.json"
    failures: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    try:
        manifest = read(manifest_path)
        summary["preflight"] = audit_preflight(manifest, failures, live=args.phase == "live")
        if args.phase == "live": summary["live"] = audit_live(manifest, failures)
    except Exception as error:
        failures.append({"code": "audit_exception", "error": f"{type(error).__name__}:{error}"})
    output = {"version": "v99_independent_paired_regression_audit_v1", "phase": args.phase,
              "status": "PASS" if not failures else "FAIL", "manifest_path": str(manifest_path),
              "manifest_sha256": sha(manifest_path) if manifest_path.is_file() else None,
              "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "classification": "exposed paired regression; fresh yield is not measured",
              "summary": summary, "failures": failures, "semantic_certified": False,
              "audit_uses_engine_imports": False, "provider_calls": 0}
    output_path = args.output or BASE / "evidence" / f"REGRESSION_PAIRED_{args.phase.upper()}_AUDIT.json"
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({"status": output["status"], "phase": args.phase, "failures": len(failures),
                      "manifest_sha256": output["manifest_sha256"], "output": str(output_path)}))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
