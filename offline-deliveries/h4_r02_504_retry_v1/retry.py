"""One HTTP retry for each 504 request in the completed R02 pilot_02 run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "h4_r02_dualroute_feedback_v1"
PILOT = PARENT / "runs" / "pilot_02"
FAILED = (
    "001-dataset-1",
    "003-103866-A-planner-1",
    "006-103866-B-realizer-S02-1",
    "010-104814-A-realizer-S02-1",
    "011-104814-A-realizer-S03-1",
    "016-104814-B-realizer-S03-1",
    "019-210046-A-realizer-S02-1",
)
sys.path.insert(0, str(PARENT))
import dataset_cards  # noqa: E402
import runner  # noqa: E402


def preflight():
    runner.require((PILOT / "REPORT.json").is_file(), "original_pilot_incomplete")
    actual_failed = {p.parent.name for p in (PILOT / "requests").glob("*/ERROR.json")
                     if runner.read(p).get("metadata", {}).get("http_status") == 504}
    runner.require(actual_failed == set(FAILED), "504_inventory_changed")
    records = []
    for name in FAILED:
        source = PILOT / "requests" / name
        payload, proof = runner.read(source / "REQUEST.json"), runner.read(source / "PROOF.json")
        attempt, error = runner.read(source / "HTTP_ATTEMPT.json"), runner.read(source / "ERROR.json")
        runner.require(not (source / "RESPONSE.json").exists(), "504_has_response:" + name)
        runner.require(error["metadata"]["http_status"] == 504 and
                       error["metadata"]["transport_attempt_count"] == 1 and
                       error["metadata"]["transport_retry_count"] == 0,
                       "not_single_504:" + name)
        runner.require(attempt["actual_payload_equals_frozen"] is True and
                       attempt["payload_sha256"] == runner.digest(payload) and
                       proof["payload_sha256"] == runner.digest(payload), "payload_mismatch:" + name)
        runner.require(payload["model"] == dataset_cards.MODEL and
                       len(payload["messages"]) == 1 and
                       payload["messages"][0]["role"] == "user", "model_or_message_changed:" + name)
        records.append({"original_id": name, "payload_sha256": runner.digest(payload),
                        "files": runner.bindings([source / x for x in
                                  ("REQUEST.json", "PROOF.json", "HTTP_ATTEMPT.json", "ERROR.json")])})
    return records


def validate_response(name, raw, proof):
    if proof["role"] == "dataset_extractor":
        source_manifest = runner.read(PILOT / "dataset_sources.json")
        validation = dataset_cards.validate_extracted_cards(raw, source_manifest, proof["source_ids"])
        return {"kind": "dataset_extractor", "mechanical_status": "PASS" if not validation["rejected_records"] else "PARTIAL_OR_FAIL",
                "accepted_cards": len(validation["accepted_cards"]), "rejected_records": validation["rejected_records"],
                "source_ids": proof["source_ids"], "integration_status": "DIAGNOSTIC_ONLY"}
    root, route, role = proof["root_id"], proof["route"], proof["role"]
    letter = "A" if route == runner.ROUTES[0] else "B"
    route_dir = PILOT / "roots" / root / letter
    try:
        if role == "planner":
            history = runner.read(route_dir / "INPUT.json")["turns"]
            cards = runner.read(route_dir / "INPUT.json")["cards"]
            parsed = runner.validate_plan(raw, root, route, history, cards)
            detail = {"planned_scenes": len(parsed["scenes"])}
        else:
            scene_id = name.split("-realizer-")[1].split("-")[0]
            scene = next(s for s in runner.read(route_dir / "PLAN.json")["scenes"] if s["scene_id"] == scene_id)
            parsed = runner.validate_realization(raw, root, route, scene)
            detail = {"status": parsed["status"], "turns": len(parsed["turns"]),
                      "dataset_card_id": scene["dataset_card_id"]}
        return {"kind": role, "root_id": root, "route": route, "mechanical_status": "PASS",
                **detail, "integration_status": "DIAGNOSTIC_ONLY_REQUIRES_HISTORY_AND_EVENT_RECHECK"}
    except (ValueError, TypeError, KeyError, AttributeError, StopIteration) as exc:
        return {"kind": role, "root_id": root, "route": route, "mechanical_status": "FAIL",
                "reason": str(exc), "integration_status": "NOT_ACCEPTED"}


def live(run):
    records = preflight()
    runner.require(not run.exists(), "retry_run_already_exists")
    run.mkdir(parents=True)
    runner.write(run / "provider_profile.json", runner.read(PILOT / "provider_profile.json"))
    runner.write(run / "MANIFEST.json", {
        "version": "h4_r02_504_retry_v1", "original_run": str(PILOT),
        "original_report_sha256": runner.digest((PILOT / "REPORT.json").read_bytes()),
        "requests": records, "maximum_http_attempts": len(FAILED), "automatic_transport_retries": 0,
        "historical_run_unchanged": True, "retries_are_not_auto_integrated": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    caller = runner.Caller(run)
    outcomes = []
    for record in records:
        runner.verify_bindings(record["files"])
        name = record["original_id"]
        source = PILOT / "requests" / name
        payload, proof = runner.read(source / "REQUEST.json"), runner.read(source / "PROOF.json")
        runner.require(runner.digest(payload) == record["payload_sha256"], "retry_payload_changed")
        try:
            raw, directory = caller.call(name, payload, proof)
            validation = validate_response(name, raw, proof)
            runner.write(directory / "VALIDATION.json", validation)
            outcome = {"original_id": name, "http_status": "SUCCESS", "retry_request": directory.name,
                       "validation": validation}
        except RuntimeError:
            directory = sorted((run / "requests").glob("*"))[-1]
            error = runner.read(directory / "ERROR.json")
            outcome = {"original_id": name, "http_status": error.get("metadata", {}).get("http_status"),
                       "error_type": error["error_type"], "retry_request": directory.name,
                       "integration_status": "NOT_ACCEPTED"}
        runner.write(run / "outcomes" / f"{name}.json", outcome)
        outcomes.append(outcome)
        print(json.dumps({"retry": name, "http_status": outcome["http_status"]}), flush=True)
    report = {"version": "h4_r02_504_retry_v1", "original_run": str(PILOT),
              "requested_model": dataset_cards.MODEL, "attempts": len(outcomes),
              "successes": sum(x["http_status"] == "SUCCESS" for x in outcomes),
              "repeat_504": sum(x["http_status"] == 504 for x in outcomes),
              "other_failures": sum(x["http_status"] not in ("SUCCESS", 504) for x in outcomes),
              "auto_integrated_scenes": 0, "outcomes": outcomes,
              "finished_at": datetime.now(timezone.utc).isoformat()}
    runner.write(run / "REPORT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "live"))
    parser.add_argument("--run", type=Path, default=HERE / "runs" / "pilot_02_retry_01")
    args = parser.parse_args()
    result = preflight() if args.action == "check" else live(args.run.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
