"""Four frozen v28 event/review regressions; no dialogue or source generation.

prepare is offline. live makes at most two logical judgment calls per case;
the second call may repair schema/citations only. audit replays recorded raw
responses deterministically, including repair merges, and verifies all frozen
input/code/transport bindings. Nothing edits or promotes the historical run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import runner as r

ORIGIN = r.HERE.parent / "h4_sequential_ab_v28/runs/inherited_a2_b3_01"
VERSION = "v31_four_frozen_case_regression_probe"
CASES = (
    ("event_s02_old_dinner_support", "015", "event", "REJECT", None),
    ("event_s01_independent_control", "008", "event", "PASS", None),
    ("review_s03_development_actor", "026", "review", "REJECT", "025"),
    ("review_s01_valid_control", "010", "review", "PASS", "009"),
)
LOGICAL_CALLS_PER_CASE = 2


def _decode(payload, marker):
    content = payload["messages"][0]["content"]
    r.base.require(content.count(marker) == 1, "historical_prompt_marker_invalid")
    return json.JSONDecoder().raw_decode(content.split(marker, 1)[1])[0]


def _directory(prefix):
    found = list((ORIGIN / "requests").glob(prefix + "-*"))
    r.base.require(len(found) == 1, "historical_request_not_unique:" + prefix)
    return found[0]


def _freeze(source, destination):
    source, destination = Path(source), Path(destination)
    r.base.require(source.is_file(), "historical_input_missing:" + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    r.base.require(r.base.digest(source.read_bytes()) == r.base.digest(destination.read_bytes()),
                   "frozen_copy_changed")
    return {"original_path": str(source), "frozen_path": str(destination),
            "sha256": r.base.digest(destination.read_bytes())}


def _make_request(case, *, repair=None):
    c = case["inputs"]
    if case["kind"] == "event":
        payload = r.event_contract.build_request(c["root_id"], c["route"], c["scene_plan"],
            c["history"], [c["dataset_card"]], c["latest_state"], repair=repair,
            purpose_input=c["purpose_input"], require_goal_checks=True)
        projected = r.event_contract.scan_plan_context(c["scene_plan"])
        sent = _decode(payload, "\nEVENT_STATE_INPUT_JSON\n")
        r.base.require(sent["scene_plan"] == projected, "scanner_projection_payload_mismatch")
        r.base.require(sent["cumulative_history"] == c["history"], "scanner_complete_history_changed")
        proof = {"root_id": c["root_id"], "route": c["route"], "scene_id": c["scene_plan"]["scene_id"],
                 "role": "historical_event_regression_judgment",
                 "projected_scene_plan_sha256": r.base.digest(projected),
                 "require_goal_checks": True,
                 "scanner_plan_policy": "actual_work_trigger_timing_without_planner_route_judgments"}
    else:
        payload, proof = r.review_request(c["root_id"], c["route"], c["scene_plan"], c["history"],
            c["realization"], c["other_turns"], card=c["dataset_card"], repair=repair,
            latest_state=c["latest_state"], event_state=c["event_state"], purpose_input=c["purpose_input"])
    proof.update(regression_case=case["name"], kind=case["kind"],
                 history_sha256=r.base.digest(c["history"]),
                 scene_plan_sha256=r.base.digest(c["scene_plan"]),
                 frozen_inputs_sha256=r.base.digest(c),
                 expected_verdict_withheld_from_model=True, prior_response_withheld_from_initial_judgment=True,
                 schema_repair_only=repair is not None,
                 repair_spec_sha256=r.base.digest(repair))
    return payload, proof


def _evaluate(case, candidate):
    c = case["inputs"]
    if case["kind"] == "event":
        try:
            checked = r.event_contract.validate(candidate, c["root_id"], c["route"], c["scene_plan"], c["history"],
                                                require_goal_checks=True)
        except (ValueError, TypeError) as error:
            return {"status": "FORMAT_ERROR", "schema_valid": False, "error": str(error),
                    "errors": deepcopy(getattr(error, "errors", [])), "semantic_certified": False}
        return {"status": "REJECT" if r.event_contract.requires_replan(checked) else "PASS",
                "schema_valid": True, "validated": checked, "semantic_certified": False}
    return r.realization_review.validate_review(candidate, c["root_id"], c["route"], c["scene_plan"],
        c["realization"], dataset_card=c["dataset_card"], latest_state=c["latest_state"],
        history=c["history"], event_state=c["event_state"], require_actor_checks=True, require_goal_checks=True)


def _format_repair(case, candidate, checked):
    r.base.require(checked.get("schema_valid") is False, "semantic_result_must_not_be_retried")
    c = case["inputs"]
    if case["kind"] == "event":
        try:
            r.event_contract.validate(candidate, c["root_id"], c["route"], c["scene_plan"], c["history"],
                                      require_goal_checks=True)
        except (ValueError, TypeError) as error:
            return r.event_contract.build_repair(error, candidate, history=c["history"])
        raise ValueError("event_schema_error_did_not_reproduce")
    return r.realization_review.build_format_repair(checked, candidate, c["scene_plan"], c["history"],
        c["realization"], dataset_card=c["dataset_card"], latest_state=c["latest_state"], event_state=c["event_state"],
        require_actor_checks=True, require_goal_checks=True)


def _merge(case, previous, raw, repair):
    if case["kind"] == "event":
        merged = (r.event_contract.merge_event_repair(previous, raw, repair, case["inputs"]["history"])
                  if repair.get("repair_mode") == "timeline_entry_patch" else raw)
        r.event_contract.validate_repair_preservation(previous, merged, repair)
        return merged
    return r.realization_review.merge_review_repair(previous, raw, repair)


def _expected_match(case, checked):
    """Assess explicit fixed regression assertions, never substitute a model verdict."""
    if checked.get("schema_valid") is not True or checked.get("status") != case["expected_status"]:
        return False
    if case["name"] == "event_s02_old_dinner_support":
        role = checked["validated"].get("route_fulfillment", {})
        # v31 history_goal_checks may veto the global tuple independently.
        goal_rows = role.get("history_goal_checks", [])
        dinner_ref = "R02:B:S01:T004"
        dinner_ids = {x["constraint_id"] for x in checked["validated"].get("timeline", [])
                      if any(isinstance(e, dict) and e.get("turn_ref") == dinner_ref
                             for e in x.get("evidence", []))}
        serving = [x for x in goal_rows if x.get("relation") == "serves_existing_goal"
                   and x.get("constraint_id") in dinner_ids]
        old_dinner_evidence = any(isinstance(e, dict) and e.get("turn_ref") == dinner_ref
                                  for e in role.get("history_evidence", []))
        return bool(serving) or (role.get("relation_to_history") == "old_event_substep"
                                 and role.get("serves_existing_event") is True and old_dinner_evidence)
    if case["name"] == "review_s03_development_actor":
        checks = (checked.get("parsed") or {}).get("checks", [])
        fourth = next((x for x in checks if x.get("development_index") == 4), {})
        failures = checked.get("failures", [])
        return (fourth.get("realized") is False or any(
            "development[4]" in str(x) and ("actor" in str(x) or "not_realized" in str(x))
            for x in failures))
    return True


def _summary(case, checked, requests, *, transport_error=None):
    return {"case": case["name"], "kind": case["kind"], "expected_status": case["expected_status"],
            "status": checked.get("status", "ERROR"), "schema_valid": checked.get("schema_valid", False),
            "expected_match": _expected_match(case, checked), "requests": requests,
            "logical_calls": len(requests), "transport_error": transport_error,
            "history_turns": len(case["inputs"]["history"]),
            "history_sha256": r.base.digest(case["inputs"]["history"]),
            "result_sha256": r.base.digest(checked), "semantic_certified": False}


def prepare(run):
    r.base.require(not run.exists(), "probe_already_exists")
    r.base.require(run.parent == r.HERE / "probes", "regression_probe_must_be_in_v31_probes")
    r.base.require(r.MIN_HTTP_GAP_SECONDS >= 35, "regression_gap_below_existing_policy")
    run.mkdir(parents=True)
    card_file = ORIGIN / "dataset_cards_result.json"
    cards = {x["card_id"]: x for x in r.base.read(card_file)["accepted_cards"]}
    cases, sources = [], []
    sources.append(_freeze(card_file, run / "frozen_origin/dataset_cards_result.json"))
    for name, prefix, kind, expected, realization_prefix in CASES:
        directory = _directory(prefix)
        payload = r.base.read(directory / "REQUEST.json")
        c = _decode(payload, "\nEVENT_STATE_INPUT_JSON\n" if kind == "event" else "\nREVIEW INPUT DATA:\n")
        root, scene_id = c["root_id"], c["scene_id"]
        root_dir = ORIGIN / "roots" / root / "B"
        plan = r.base.read(root_dir / (scene_id + ".PLAN.json"))["scenes"]
        r.base.require(len(plan) == 1 and plan[0]["scene_id"] == scene_id, "historical_scene_plan_invalid")
        scene = plan[0]
        purpose = r.base.read(root_dir / (scene_id + ".PURPOSE.json"))
        latest = r.base.read(root_dir / (scene_id + ".LATEST_STATE.json"))
        r.base.require(c["purpose_input"] == purpose and c["latest_state"] == latest,
                       "historical_context_root_artifact_mismatch")
        history = deepcopy(c["cumulative_history" if kind == "event" else "branch_history"])
        card = cards[scene["dataset_card_id"]]
        r.base.require(c["root_id"] == root and scene["scene_id"] == scene_id, "case_identity_invalid")
        if kind == "event":
            r.base.require(c["scene_plan"] == scene, "event_original_full_plan_mismatch")
        inputs = {"root_id": root, "route": c["route"], "scene_plan": scene,
                  "history": history, "dataset_card": card, "latest_state": latest, "purpose_input": purpose}
        paths = [directory / "REQUEST.json", directory / "RESPONSE.json",
                 root_dir / (scene_id + ".PLAN.json"), root_dir / (scene_id + ".PURPOSE.json"),
                 root_dir / (scene_id + ".LATEST_STATE.json"),
                 root_dir / (scene_id + ".EVENT_STATE.json")]
        if kind == "review":
            event = r.base.read(root_dir / (scene_id + ".EVENT_STATE.json"))
            rd = _directory(realization_prefix)
            realization_raw = r.base.read(rd / "RESPONSE.json")["text"]
            realization = r.base.validate_realization(r.parse_object(realization_raw), root, c["route"], scene)
            r.base.require(r.realization_review.review_scene(realization) == c["new_scene"],
                           "original_realization_prompt_projection_mismatch")
            inputs.update(realization=realization, event_state=event,
                          other_turns=deepcopy(c["other_branch_added_turns_for_review_only"]))
            paths += [root_dir / (scene_id + ".EVENT_STATE.json"), rd / "REQUEST.json", rd / "RESPONSE.json"]
        receipts = []
        for path in paths:
            receipt = _freeze(path, run / "frozen_origin" / path.relative_to(ORIGIN))
            receipts.append(receipt)
            if receipt not in sources:
                sources.append(receipt)
        case = {"name": name, "kind": kind, "expected_status": expected, "origin_request": str(directory),
                "original_context": c, "inputs": inputs,
                "previous_raw": r.base.read(directory / "RESPONSE.json")["text"],
                "source_bindings": receipts, "original_history_sha256": r.base.digest(history)}
        case_path = run / "cases" / (name + ".json")
        r.base.write(case_path, case)
        first, proof = _make_request(case)
        r.base.write(run / "previews" / (name + ".json"), {"payload": first, "proof": proof})
        cases.append(str(case_path))
    profile, transport, tokenizer = r.transport_inputs()
    r.base.write(run / "provider_profile.json", profile)
    code = sorted(set(list(r.HERE.glob("*.py")) + list(r.LEGACY.glob("*.py")) + list(transport)))
    snapshots = [_freeze(p, run / "execution_code" / f"{i:03d}_{p.name}") for i, p in enumerate(code)]
    r.base.write(run / "EXECUTION_CODE_SNAPSHOT.json", {"files": snapshots})
    paths = code + list(tokenizer) + [r.HERE / "POLICY.md"]
    paths += [Path(x[key]) for x in sources for key in ("original_path", "frozen_path")]
    paths += list(run.rglob("*.json")) + [Path(x["frozen_path"]) for x in snapshots]
    manifest = {"version": VERSION, "model": r.MODEL, "review_model": r.MODEL,
        "prepared_at_utc": r.now(), "origin_run": str(ORIGIN), "cases": cases,
        "max_logical_calls_per_case": LOGICAL_CALLS_PER_CASE, "logical_call_max": len(CASES)*2,
        "http_max": len(CASES)*2*(r.TRANSPORT_RETRIES+1),
        "automatic_transport_retries": r.TRANSPORT_RETRIES,
        "minimum_http_gap_after_response_seconds": r.MIN_HTTP_GAP_SECONDS,
        "source_events_reassigned": False, "historical_records_modified": False,
        "new_scene_generation": False, "question_calls": 0, "solver_calls": 0, "training_calls": 0,
        "format_repairs_only": True, "prior_verdict_withheld_from_initial_requests": True,
        "files": r.base.bindings(sorted(set(paths)))}
    r.base.write(run / "MANIFEST.json", manifest)
    return {"status": "PREPARED", "cases": len(cases), "logical_call_max": 8, "http_max": manifest["http_max"],
            "http_calls": 0, "new_scenes": 0}


def verify(run):
    manifest = r.base.read(run / "MANIFEST.json")
    r.base.require(manifest["version"] == VERSION and manifest["model"] == r.MODEL, "wrong_probe_version_or_model")
    r.base.require(manifest["max_logical_calls_per_case"] == 2 and manifest["logical_call_max"] == 8,
                   "logical_budget_changed")
    r.base.require(manifest["http_max"] == 8*(r.TRANSPORT_RETRIES+1), "http_budget_changed")
    r.base.require(manifest["minimum_http_gap_after_response_seconds"] == r.MIN_HTTP_GAP_SECONDS,
                   "probe_http_gap_changed")
    r.base.verify_bindings(manifest["files"])
    frozen = r.base.read(run / "EXECUTION_CODE_SNAPSHOT.json")["files"]
    for row in frozen:
        r.base.require(r.base.digest(Path(row["frozen_path"]).read_bytes()) == row["sha256"], "execution_snapshot_changed")
    expected = [x[0] for x in CASES]
    r.base.require([r.base.read(p)["name"] for p in manifest["cases"]] == expected, "case_inventory_changed")
    for path in manifest["cases"]:
        case = r.base.read(path)
        history = case["inputs"]["history"]
        original = case["original_context"]["cumulative_history" if case["kind"] == "event" else "branch_history"]
        r.base.require(history == original and r.base.digest(history) == case["original_history_sha256"],
                       "complete_case_history_changed")
        payload, proof = _make_request(case)
        r.base.require(r.base.read(run / "previews" / (case["name"] + ".json")) ==
                       {"payload": payload, "proof": proof}, "frozen_preview_replay_mismatch:" + case["name"])
    return manifest


def live(run):
    manifest = verify(run)
    r.base.require(not (run / "LIVE_STARTED.json").exists(), "probe_cannot_restart")
    r.base.write(run / "LIVE_STARTED.json", {"started_at_utc": r.now(), "manifest_sha256": r.base.digest(manifest)})
    caller, rows = r.Caller(run), []
    for path in manifest["cases"]:
        case = r.base.read(path)
        repair, previous, requests, checked, transport_error = None, None, [], {}, None
        for attempt in range(1, LOGICAL_CALLS_PER_CASE + 1):
            payload, proof = _make_request(case, repair=repair)
            proof.update(logical_attempt=attempt, previous_candidate_sha256=r.base.digest(previous))
            before = caller.request_counter
            try:
                raw, directory = caller.call(f"regression-{case['name']}-{attempt}", payload, proof)
            except (RuntimeError, ValueError) as error:
                transport_error = str(error)
                if caller.request_counter > before:
                    directory = run / "requests" / f"{caller.request_counter:03d}-regression-{case['name']}-{attempt}"
                    requests.append(str(directory))
                checked = {"status": "ERROR", "schema_valid": False, "error": str(error), "semantic_certified": False}
                break
            requests.append(str(directory))
            r.base.write(directory / "REGRESSION_STEP.json", {"case": case["name"], "logical_attempt": attempt,
                "repair": repair, "previous_candidate_sha256": r.base.digest(previous)})
            try:
                candidate = raw if repair is None else _merge(case, previous, raw, repair)
            except (ValueError, TypeError) as error:
                checked = {"status": "FORMAT_ERROR", "schema_valid": False, "error": str(error), "semantic_certified": False}
                r.base.write(directory / "EVALUATION.json", checked)
                break
            r.base.write(directory / "CANDIDATE.json", candidate)
            checked = _evaluate(case, candidate)
            r.base.write(directory / "EVALUATION.json", checked)
            if checked.get("schema_valid") is not False:
                break  # A semantic rejection is never retried in this probe.
            if attempt < LOGICAL_CALLS_PER_CASE:
                repair = _format_repair(case, candidate, checked)
                previous = candidate
                r.base.write(directory / "FORMAT_REPAIR_SPEC.json", repair)
        row = _summary(case, checked, requests, transport_error=transport_error)
        rows.append(row)
        r.base.write(run / "results" / (case["name"] + ".json"), row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    verify(run)
    report = {"status": "PASS" if all(x["expected_match"] for x in rows) else "REGRESSION_NOT_ESTABLISHED",
              "cases": rows, "logical_calls": caller.request_counter, "http_attempts": caller.http_counter,
              "new_scenes": 0, "question_calls": 0, "solver_calls": 0, "training_calls": 0,
              "semantic_certified": False, "completed_at_utc": r.now()}
    r.base.write(run / "REPORT.json", report)
    return report


def audit(run):
    manifest = verify(run)
    failures, rows, request_inventory = [], [], []
    def check(condition, code):
        if not condition:
            failures.append(code)
    for path in manifest["cases"]:
        case = r.base.read(path)
        result_path = run / "results" / (case["name"] + ".json")
        if not result_path.exists():
            rows.append({"case": case["name"], "status": "NO_LIVE_RESULT",
                         "old_raw_replay": _evaluate(case, case["previous_raw"]), "expected_match": False})
            if (run / "LIVE_STARTED.json").exists():
                failures.append("case_result_missing:" + case["name"])
            continue
        row = r.base.read(result_path)
        directories = [Path(p) for p in row["requests"]]
        check(1 <= len(directories) <= 2, "case_request_budget_invalid:" + case["name"])
        repair, previous, checked = None, None, {}
        for attempt, directory in enumerate(directories, 1):
            request_inventory.append(directory.resolve())
            payload, proof = _make_request(case, repair=repair)
            proof.update(logical_attempt=attempt, previous_candidate_sha256=r.base.digest(previous))
            check(r.base.read(directory / "REQUEST.json") == payload, "request_replay_mismatch:" + directory.name)
            if not (directory / "RESPONSE.json").exists():
                check((directory / "ERROR.json").exists() or (directory / "LOCAL_ERROR.json").exists(),
                      "missing_transport_or_local_error:" + directory.name)
                checked = {"status": "ERROR", "schema_valid": False,
                           "error": row["transport_error"], "semantic_certified": False}
                break
            actual_proof = r.base.read(directory / "PROOF.json")
            check(all(actual_proof.get(k) == v for k, v in proof.items()), "proof_replay_mismatch:" + directory.name)
            check(actual_proof.get("payload_sha256") == r.base.digest(payload), "proof_payload_hash_mismatch:" + directory.name)
            raw = r.base.read(directory / "RESPONSE.json")["text"]
            step = r.base.read(directory / "REGRESSION_STEP.json")
            check(step == {"case": case["name"], "logical_attempt": attempt, "repair": repair,
                          "previous_candidate_sha256": r.base.digest(previous)}, "regression_step_mismatch:" + directory.name)
            try:
                candidate = raw if repair is None else _merge(case, previous, raw, repair)
            except (ValueError, TypeError) as error:
                checked = {"status": "FORMAT_ERROR", "schema_valid": False, "error": str(error), "semantic_certified": False}
                check(r.base.read(directory / "EVALUATION.json") == checked, "merge_failure_replay_mismatch:" + directory.name)
                check(attempt == len(directories), "request_after_failed_repair_merge:" + directory.name)
                break
            check(r.base.read(directory / "CANDIDATE.json") == candidate, "candidate_replay_mismatch:" + directory.name)
            checked = _evaluate(case, candidate)
            check(r.base.read(directory / "EVALUATION.json") == checked, "evaluation_replay_mismatch:" + directory.name)
            if checked.get("schema_valid") is not False:
                check(attempt == len(directories), "semantic_result_retried:" + directory.name)
                break
            if attempt < len(directories):
                repair = _format_repair(case, candidate, checked)
                previous = candidate
                check(r.base.read(directory / "FORMAT_REPAIR_SPEC.json") == repair, "repair_spec_replay_mismatch:" + directory.name)
        replay_row = _summary(case, checked, row["requests"], transport_error=row.get("transport_error"))
        check(replay_row == row, "case_result_replay_mismatch:" + case["name"])
        rows.append(replay_row)
    actual_requests = sorted(p.resolve() for p in (run / "requests").glob("*") if p.is_dir())
    check(sorted(request_inventory) == actual_requests, "unexpected_or_missing_request_directory")
    http_attempts = sorted((run / "requests").glob("*/HTTP_ATTEMPT*.json"))
    check(len(http_attempts) <= manifest["http_max"], "http_budget_exceeded")
    for directory in actual_requests:
        attempts = sorted(directory.glob("HTTP_ATTEMPT*.json"))
        check(len(attempts) <= r.TRANSPORT_RETRIES+1, "transport_retry_budget_exceeded:" + directory.name)
        payload = r.base.read(directory / "REQUEST.json")
        for f in attempts:
            attempt = r.base.read(f)
            check(attempt.get("payload_sha256") == r.base.digest(payload)
                  and attempt.get("actual_payload_equals_frozen") is True
                  and attempt.get("requested_model") == r.MODEL, "http_payload_or_model_mismatch:" + str(f))
            number = attempt.get("request_attempt", 1)
            throttle = r.base.read(directory / f"THROTTLE_{number:02d}.json")
            check(throttle.get("minimum_gap_after_previous_response_seconds") == r.MIN_HTTP_GAP_SECONDS,
                  "http_gap_policy_mismatch:" + directory.name)
            check(type(throttle.get("waited_seconds")) in (int, float)
                  and throttle["waited_seconds"] >= 0 and isinstance(throttle.get("http_started_at_utc"), str),
                  "http_throttle_record_invalid:" + directory.name)
            if number > 1:
                previous_result = r.base.read(directory / f"HTTP_RESULT_{number-1:02d}.json")
                check(attempt.get("same_payload_retry") is True
                      and previous_result.get("outcome") == "error"
                      and previous_result.get("http_status") in (503, 504),
                      "transport_retry_policy_mismatch:" + directory.name)
        if (directory / "RESPONSE.json").exists():
            metadata = r.base.read(directory / "RESPONSE.json").get("metadata", {})
            check(metadata.get("returned_model") == r.MODEL and metadata.get("finish_reason") == "stop",
                  "response_model_or_finish_mismatch:" + directory.name)
    if (run / "REPORT.json").exists():
        report = r.base.read(run / "REPORT.json")
        check(report.get("cases") == rows, "report_case_replay_mismatch")
        check(report.get("http_attempts") == len(http_attempts) and report.get("logical_calls") == len(actual_requests),
              "report_call_count_mismatch")
        for k in ("new_scenes", "question_calls", "solver_calls", "training_calls"):
            check(report.get(k) == 0, "forbidden_generation_count:" + k)
    result = {"status": "PASS" if not failures else "FAIL", "run": str(run), "failures": failures,
              "regression_status": "PREPARED_NO_LIVE_CALLS" if not actual_requests else (
                  "PASS" if len(rows) == len(CASES) and all(x.get("expected_match") for x in rows)
                  else "REGRESSION_NOT_ESTABLISHED"),
              "cases": rows, "logical_calls": len(actual_requests), "http_attempts": len(http_attempts),
              "new_scenes": 0, "question_calls": 0, "solver_calls": 0, "training_calls": 0,
              "semantic_certified": False, "audited_at_utc": r.now()}
    # Audits are append-only as well: a prepared-only audit must not block the
    # later live-response replay, and no previous audit record is replaced.
    output = run / "REGRESSION_AUDIT.json"
    if output.exists():
        index = 2
        while (run / f"REGRESSION_AUDIT_{index:03d}.json").exists():
            index += 1
        output = run / f"REGRESSION_AUDIT_{index:03d}.json"
    r.base.write(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "live", "audit"])
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    result = globals()[args.command](args.run.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "audit" and result["status"] != "PASS":
        raise SystemExit(1)
