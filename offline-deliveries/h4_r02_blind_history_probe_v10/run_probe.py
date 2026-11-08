"""Frozen diagnostic of summary influence; never produces expansion data."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
V9 = HERE.parent / "h4_r02_reference_contract_v9"
sys.path.insert(0, str(V9))
import runner as runtime
import probe_contract as contract

VERSION = "h4_r02_blind_history_probe_v10"
ROOT = "cc-train-episode-210046"
HISTORICAL_RUN = V9 / "runs/r02_reference_contract_v9_B_one_01"
SCHEDULE = [("raw_history_only", 1), ("with_prior_summary", 1),
            ("with_prior_summary", 2), ("raw_history_only", 2),
            ("raw_history_only", 3), ("with_prior_summary", 3)]


def frozen_cases(historical_run=HISTORICAL_RUN):
    read = runtime.base.read
    folder = historical_run / "roots" / ROOT / "B"
    state_path = folder / "S01.LATEST_STATE.json"
    event_path = folder / "S01.EVENT_STATE.json"
    state, event = read(state_path), read(event_path)
    by_fact = {item["fact_id"]: item for item in state["facts"]}
    by_constraint = {item["constraint_id"]: item for item in event["timeline"]}
    reason = event["plan_checks"][0]["reason"]
    end = reason.index(", making a business trip request coherent.")
    specifications = [
        ("c01", event_path, "plan_checks[0].reason", reason[:end],
         by_constraint["c2"]["status"], {"start": 0, "end": end}),
        ("c02", state_path, "facts[f4].statement", by_fact["f4"]["statement"],
         by_fact["f4"]["status"], None),
        ("c03", event_path, "timeline[c4].statement", by_constraint["c4"]["statement"],
         by_constraint["c4"]["status"], None),
        ("c04", state_path, "facts[f1].statement", by_fact["f1"]["statement"],
         by_fact["f1"]["status"], None),
        ("c05", event_path, "timeline[c3].statement", by_constraint["c3"]["statement"],
         by_constraint["c3"]["status"], None),
    ]
    claims, origins = [], []
    for identifier, path, field, statement, status, span in specifications:
        claims.append({"claim_id": identifier, "statement": statement, "claimed_status": status})
        origins.append({"claim_id": identifier, "path": str(path), "field": field,
                        "character_span": span, "source_file_sha256": runtime.base.digest(path.read_bytes()),
                        "statement_sha256": runtime.base.digest(statement.encode())})
    history = read(historical_run / "sources" / f"{ROOT}.json")["turns"]
    return {"root_id": ROOT, "cumulative_history": history, "claims": claims,
            "prior_summary": state}, origins


def assert_isolated_pair(inputs):
    pair = {}
    for condition in ("raw_history_only", "with_prior_summary"):
        pair[condition] = contract.build_request(
            inputs["root_id"], inputs["cumulative_history"], inputs["claims"],
            prior_summary=inputs["prior_summary"] if condition == "with_prior_summary" else None)
    marker = "\nPROBE_INPUT_JSON\n"
    contexts = {}
    prefixes = {}
    for condition, payload in pair.items():
        text = payload["messages"][0]["content"]
        prefixes[condition], suffix = text.split(marker, 1)
        contexts[condition] = json.loads(suffix)
    raw, prior = contexts["raw_history_only"], contexts["with_prior_summary"]
    runtime.base.require(raw.pop("prior_summary") is None, "raw_condition_contains_prior")
    runtime.base.require(prior.pop("prior_summary") == inputs["prior_summary"], "prior_condition_changed")
    runtime.base.require(raw == prior, "conditions_differ_beyond_summary")
    runtime.base.require(len(set(prefixes.values())) == 1, "condition_prompt_differs")
    for condition in pair:
        runtime.base.require(set(pair[condition]) == set(pair["raw_history_only"]), "request_config_fields_differ")
    runtime.base.require({k: v for k, v in pair["raw_history_only"].items() if k != "messages"}
                         == {k: v for k, v in pair["with_prior_summary"].items() if k != "messages"},
                         "condition_decoding_config_differs")
    return pair


def prepare(run):
    require, write = runtime.base.require, runtime.base.write
    require(run.parent == HERE / "diagnostics", "diagnostic_directory_required")
    require(not run.exists(), "diagnostic_already_exists")
    runtime.verify(HISTORICAL_RUN)
    inputs, origins = frozen_cases()
    pair = assert_isolated_pair(inputs)
    run.mkdir(parents=True)
    write(run / "INPUT.json", inputs)
    write(run / "CASE_ORIGINS.json", {"origins": origins, "model_authored_claims": True,
          "expected_answers_sent_to_model": False, "new_training_questions": False})
    for condition, payload in pair.items():
        write(run / "previews" / f"{condition}.json", payload)
    historical_inputs = [HISTORICAL_RUN / "MANIFEST.json", HISTORICAL_RUN / "REPORT.json",
                         HISTORICAL_RUN / "sources" / f"{ROOT}.json"]
    historical_inputs.extend(Path(item["path"]) for item in origins)
    files = [*HERE.glob("*.py"), *V9.glob("*.py"), *runtime.LEGACY.glob("*.py"),
             HERE.parent / "h4_r02_source_unique_v2/unique_events.py", *historical_inputs,
             *run.rglob("*.json")]
    prior = runtime.source_inventory.collect_prior()
    manifest = {"version": VERSION, "kind": "frozen_history_summary_influence_diagnostic",
        "model": runtime.MODEL, "http_max": len(SCHEDULE) * (runtime.TRANSPORT_RETRIES + 1),
        "schedule": [{"condition": condition, "replica": replica} for condition, replica in SCHEDULE],
        "logical_requests": len(SCHEDULE), "repetitions_per_condition": 3,
        "semantic_repair_attempts": 0, "schema_repair_attempts": 0,
        "transport_retry_statuses": [503, 504], "transport_retries_per_request": 1,
        "condition_difference": "prior_summary_value_only", "files": runtime.base.bindings(files),
        "inventory_before": prior, "source_reservations_created": False,
        "new_expansion": False, "question_calls": 0, "solver_calls": 0,
        "training_started": False, "semantic_certified": False}
    manifest["manifest_sha256"] = runtime.base.digest(manifest)
    write(run / "MANIFEST.json", manifest)
    snapshots = []
    for index, row in enumerate(manifest["files"]):
        path = Path(row["path"])
        if path.suffix != ".py":
            continue
        destination = run / "execution_code" / f"{index:03d}_{path.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        require(runtime.base.digest(destination.read_bytes()) == row["sha256"], "code_snapshot_mismatch")
        snapshots.append({"original_path": str(path), "snapshot_path": str(destination), "sha256": row["sha256"]})
    write(run / "EXECUTION_CODE_SNAPSHOT.json", {"files": snapshots})
    verify(run)
    return {"status": "PREPARED", "claims": len(inputs["claims"]), "logical_requests": len(SCHEDULE),
            "http_max": manifest["http_max"], "new_sources": 0}


def verify(run):
    manifest = runtime.base.read(run / "MANIFEST.json")
    runtime.base.require(manifest["version"] == VERSION, "diagnostic_version_changed")
    runtime.base.require(manifest["manifest_sha256"] == runtime.base.digest(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}), "manifest_changed")
    runtime.base.verify_bindings(manifest["files"])
    inputs = runtime.base.read(run / "INPUT.json")
    expected_inputs, origins = frozen_cases()
    runtime.base.require(inputs == expected_inputs, "frozen_diagnostic_inputs_changed")
    runtime.base.require(runtime.base.read(run / "CASE_ORIGINS.json")["origins"] == origins, "claim_origin_changed")
    for condition, payload in assert_isolated_pair(inputs).items():
        runtime.base.require(runtime.base.read(run / "previews" / f"{condition}.json") == payload,
                             "preview_not_current_contract")
    for row in runtime.base.read(run / "EXECUTION_CODE_SNAPSHOT.json")["files"]:
        runtime.base.require(runtime.base.digest(Path(row["snapshot_path"]).read_bytes()) == row["sha256"],
                             "code_snapshot_changed")
    runtime.base.require(not (run / "SOURCE_RESERVATIONS.json").exists(), "diagnostic_reserved_source")
    return manifest, inputs


def summarize(rows, claims):
    result = {}
    for condition in ("raw_history_only", "with_prior_summary"):
        subset = [row for row in rows if row["condition"] == condition]
        result[condition] = {
            "request_statuses": dict(Counter(row["status"] for row in subset)),
            "claim_verdict_counts": {
                claim["claim_id"]: dict(Counter(check["verdict"] for row in subset
                    if row["status"] == "SCHEMA_PASS" for check in row["parsed"]["checks"]
                    if check["claim_id"] == claim["claim_id"])) for claim in claims}}
    return result


def live(run):
    require, write, digest = runtime.base.require, runtime.base.write, runtime.base.digest
    manifest, inputs = verify(run)
    require(not (run / "LIVE_STARTED.json").exists(), "diagnostic_already_started")
    require(runtime.source_inventory.collect_prior() == manifest["inventory_before"], "inventory_changed_before_probe")
    write(run / "LIVE_STARTED.json", {"started_at": runtime.now(), "diagnostic_only": True})
    caller = runtime.Caller(run)
    pair = assert_isolated_pair(inputs)
    rows = []
    for item in manifest["schedule"]:
        condition, replica = item["condition"], item["replica"]
        proof = {"role": "frozen_history_diagnostic", "condition": condition, "replica": replica,
                 "history_sha256": digest(inputs["cumulative_history"]),
                 "claims_sha256": digest(inputs["claims"]), "independent_single_message": True,
                 "prior_answers_supplied": False, "expected_answers_supplied": False,
                 "source_reservation": "none_diagnostic_only"}
        try:
            raw, directory = caller.call(f"{condition}-{replica}", deepcopy(pair[condition]), proof)
        except RuntimeError as error:
            rows.append({**item, "status": "TRANSPORT_FAILURE", "error": str(error)})
            continue
        try:
            parsed = contract.validate_response(raw, inputs["root_id"], inputs["cumulative_history"], inputs["claims"])
        except (ValueError, TypeError) as error:
            validation = {"status": "SCHEMA_REJECT", "error": str(error),
                          "errors": deepcopy(getattr(error, "errors", [])), "semantic_certified": False}
        else:
            validation = {"status": "SCHEMA_PASS", "parsed": parsed, "semantic_certified": False}
        write(directory / "VALIDATION.json", validation)
        rows.append({**item, "request": str(directory), **validation})
    verify(run)
    inventory_unchanged = runtime.source_inventory.collect_prior() == manifest["inventory_before"]
    require(inventory_unchanged, "source_inventory_changed_during_diagnostic")
    report = {"version": VERSION, "status": "COMPLETE", "model": runtime.MODEL,
        "rows": rows, "summary": summarize(rows, inputs["claims"]),
        "http_attempts": caller.http_counter, "logical_requests": caller.request_counter,
        "inventory_unchanged": inventory_unchanged, "source_reservations_created": False,
        "accepted_expansion_scenes": 0, "question_calls": 0, "solver_calls": 0,
        "training_started": False, "semantic_certified": False,
        "limitations": ["One root and three repeats per condition; no accuracy estimate.",
                        "Both conditions contain identical candidate claims; raw-only is not free of all anchoring.",
                        "Specialized diagnostic prompt is not the production Reviewer.",
                        "Model verdicts are observations, not certified ground truth."]}
    write(run / "REPORT.json", report)
    return {k: v for k, v in report.items() if k != "rows"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify", "live"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.action == "prepare":
        result = prepare(run)
    elif args.action == "live":
        result = live(run)
    else:
        verify(run)
        result = {"status": "PASS"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
