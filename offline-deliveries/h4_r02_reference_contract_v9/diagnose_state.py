"""Bounded citation-interface replay; produces no expansion or source use."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import runner as r


def replay(output, historical_run):
    output, historical_run = Path(output).resolve(), Path(historical_run).resolve()
    read, write, require, digest = r.base.read, r.base.write, r.base.require, r.base.digest
    require(not output.exists(), "diagnostic_already_exists")
    requests = sorted((historical_run / "requests").glob("*-state-*"))
    require(bool(requests), "historical_state_request_missing")
    previous = requests[-1]
    request = read(previous / "REQUEST.json")
    context = json.JSONDecoder().raw_decode(
        request["messages"][0]["content"].split("\nSTATE_INPUT_JSON\n", 1)[1])[0]
    root, route, scene_id = (context[key] for key in ("root_id", "route", "scene_id"))
    history = context["cumulative_history"]
    require(history == read(historical_run / "sources" / f"{root}.json")["turns"],
            "diagnostic_not_bound_to_frozen_history")
    cards = read(historical_run / "dataset_cards_result.json")["accepted_cards"]
    require(len(cards) == 1, "diagnostic_requires_one_historical_card")
    require(context["dataset_cards"] == [r.planner_contract.mapping_context(cards[0])["card_guidance"]],
            "diagnostic_card_differs_from_historical_input")
    raw = read(previous / "RESPONSE.json")["text"]
    try:
        r.state_contract.validate_state(raw, root, route, scene_id, history)
    except r.state_contract.StateValidationError as error:
        require(error.errors == read(previous / "VALIDATION.json")["errors"],
                "historical_failure_does_not_reproduce")
        repair = r.state_contract.build_state_repair(error, raw, history=history)
    else:
        raise ValueError("historical_response_was_not_a_validation_failure")

    output.mkdir(parents=True)
    files = [*r.HERE.glob("*.py"), *r.LEGACY.glob("*.py"),
             r.HERE.parent / "h4_r02_source_unique_v2/unique_events.py",
             historical_run / "MANIFEST.json", previous / "REQUEST.json",
             previous / "RESPONSE.json", previous / "VALIDATION.json",
             historical_run / "dataset_cards_result.json",
             historical_run / "sources" / f"{root}.json"]
    bindings = r.base.bindings(files)
    write(output / "MANIFEST.json", {
        "version": r.VERSION, "kind": "historical_state_citation_diagnostic_only",
        "model": r.MODEL, "http_max": r.STATE_ATTEMPTS * (r.TRANSPORT_RETRIES + 1),
        "files": bindings, "source_reservations_created": False,
        "new_expansion": False, "question_calls": 0, "solver_calls": 0,
        "training_started": False, "semantic_certified": False})
    snapshots = []
    for index, row in enumerate(bindings):
        path = Path(row["path"])
        if path.suffix != ".py":
            continue
        destination = output / "execution_code" / f"{index:03d}_{path.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        require(digest(destination.read_bytes()) == row["sha256"], "snapshot_hash_mismatch")
        snapshots.append({"original_path": row["path"], "snapshot_path": str(destination),
                          "sha256": row["sha256"]})
    write(output / "EXECUTION_CODE_SNAPSHOT.json", {"files": snapshots})
    write(output / "INPUT.json", {"historical_run": str(historical_run),
        "historical_request": str(previous), "history_sha256": digest(history),
        "previous_raw_sha256": digest(raw), "initial_targeted_repair": repair})
    r.base.verify_bindings(bindings)
    write(output / "LIVE_STARTED.json", {"started_at": r.now(), "diagnostic_only": True})
    caller = r.Caller(output)
    status, attempts = "REJECT", []
    try:
        for attempt in range(1, r.STATE_ATTEMPTS + 1):
            payload = r.state_contract.build_state_request(
                root, route, scene_id, history, cards, repair=repair)
            answer, directory = caller.call(f"historical-state-repair-{attempt}", payload, {
                "role": "state_citation_diagnostic", "history_sha256": digest(history),
                "historical_request": str(previous), "source_use": "none_diagnostic_only"})
            try:
                snapshot = r.state_contract.validate_state(answer, root, route, scene_id, history)
            except (ValueError, TypeError) as error:
                repair = r.state_contract.build_state_repair(error, answer, history=history)
                validation = {"status": "REJECT", "errors": repair["errors"]}
            else:
                validation = {"status": "PASS", "semantic_certified": False,
                              "snapshot_sha256": digest(snapshot)}
                write(output / "DIAGNOSTIC_STATE.json", snapshot)
                status = "PASS"
            write(directory / "VALIDATION.json", validation)
            attempts.append({"request": str(directory), **validation})
            if status == "PASS":
                break
    except RuntimeError as error:
        status = "TRANSPORT_FAILURE"
        attempts.append({"error": str(error)})
    r.base.verify_bindings(bindings)
    report = {"status": status, "scope": "citation_interface_only", "model": r.MODEL,
        "attempts": attempts, "http_attempts": caller.http_counter,
        "source_reservations_created": False, "accepted_expansion_scenes": 0,
        "question_calls": 0, "solver_calls": 0, "training_started": False,
        "semantic_certified": False, "historical_inputs_unchanged": True}
    write(output / "REPORT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.output, args.historical_run), ensure_ascii=False, indent=2))
