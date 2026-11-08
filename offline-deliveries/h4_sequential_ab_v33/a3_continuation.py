"""One new A S03 after exact accepted v24 A2; no B or downstream generation.

The wrapper freezes its own execution manifest. Audit replays current frozen
stage functions with recorded raw responses and intercepts all artifact writes;
it neither calls a model nor edits original request/acceptance records.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import runner as r

VERSION = "v33_exact_A2_to_A3_single_slot"
ROOT = "cc-train-episode-104814"
ROUTE = r.ROUTES[0]
SCENE = "S03"
HISTORY_SHA = "70fcbddce04e6e873d1a0a866108a8244099d00d260bdd785511c184289ec6be"
LOGICAL_MAX = (r.STATE_ATTEMPTS + r.PLAN_ATTEMPTS * (1 + r.EVENT_STATE_ATTEMPTS)
               + r.SCENE_ATTEMPTS * (r.REVIEW_RETRIES + 2))


def _history(run):
    record = r.base.read(run / "INHERITED_A.json")
    history = r.base.read(Path(record["local_path"]))["turns"]
    r.base.require(len(history) == 120 and r.base.digest(history) == HISTORY_SHA,
                   "A2_exact_history_mismatch")
    r.base.require(not any(":B:" in turn["turn_ref"] for turn in history), "B_in_A_history")
    return deepcopy(history)


def _scope_payload(payload, proof, history):
    r.base.require(proof.get("root_id") == ROOT and proof.get("route") == ROUTE
                   and proof.get("scene_id") == SCENE, "continuation_request_scope_changed")
    r.base.require(proof.get("role") in {"state_extractor", "planner", "event_state_extractor",
                                       "realizer", "scene_reviewer"}, "forbidden_A3_stage")
    content = payload["messages"][0]["content"]
    markers = ("\nSTATE_INPUT_JSON\n", "\nEVENT_STATE_INPUT_JSON\n", "\nINPUT_JSON\n", "\nREVIEW INPUT DATA:\n")
    marker = next((m for m in markers if m in content), None)
    r.base.require(marker is not None, "continuation_request_context_missing")
    context = json.JSONDecoder().raw_decode(content.split(marker, 1)[1])[0]
    r.base.require(context.get("cumulative_history", context.get("branch_history")) == history,
                   "continuation_full_history_changed")
    r.base.require(not context.get("dataset_cards") and not context.get("dataset_card")
                   and not context.get("purpose_input"), "continuation_used_dataset_guidance")
    return {**proof, "continuation_version": VERSION, "inherited_A2_history_sha256": HISTORY_SHA,
            "input_accepted_A_scenes": 2, "only_new_slot": SCENE,
            "old_failed_plan_supplied": False}


class ScopedCaller:
    def __init__(self, caller, history):
        self.caller, self.history = caller, history

    def call(self, label, payload, proof):
        r.base.require(f"{ROOT}-A-{SCENE}-" in label, "continuation_label_invalid")
        return self.caller.call(label, payload, _scope_payload(payload, proof, self.history))


def _flow(run, caller):
    """Same bounded stages as runner, with exactly the missing slot identity."""
    history = _history(run)
    caller = ScopedCaller(caller, history)
    folder = run / "roots" / ROOT / "A"
    inherited = r.base.read(run / "INHERITED_RESULT.json")
    accepted, failure, realization = None, None, None
    snapshot = plan = event = None
    r.base.write(folder / "INPUT.json", {"root_id": ROOT, "route": ROUTE, "turns": history,
        "input_turns": len(history), "history_sha256": HISTORY_SHA, "new_scene_slots": [SCENE],
        "inherited_scene_ids": ["S01", "S02"]})
    try:
        snapshot = r.extract_latest_state(caller, run, ROOT, ROUTE, history, [], SCENE)
        if snapshot is None:
            failure = "state_audit_incomplete_after_retry"
        else:
            r.base.write(folder / f"{SCENE}.LATEST_STATE.json", snapshot)
            repair = None
            for attempt in range(1, r.PLAN_ATTEMPTS + 1):
                payload, proof = r.make_request(run, ROOT, ROUTE, "planner", history, [], SCENE,
                                                repair=repair, latest_state=snapshot, event_state=event)
                raw, directory = caller.call(f"{ROOT}-A-{SCENE}-planner-{attempt}", payload, proof)
                try:
                    candidate = r.validate_plan(raw, ROOT, ROUTE, history, [], SCENE, latest_state=snapshot)
                except (ValueError, TypeError) as error:
                    failure = "plan_rejected_after_retry"
                    repair = r.planner_contract.build_repair_feedback(error, raw, None, route=ROUTE)
                    r.base.write(directory / "VALIDATION.json", {"status": "REJECT", "reason": str(error),
                                                               "errors": repair["errors"]})
                    continue
                r.base.write(directory / "VALIDATION.json", {"status": "PASS"})
                r.base.write(directory / "PLAN_CITATION_PROVENANCE.json",
                             r.evidence_refs.citation_provenance(r.parse_object(raw), history))
                r.base.write(directory / "CANDIDATE_PLAN.json", candidate)
                if not candidate["scenes"]:
                    plan, failure = candidate, candidate["shortfall_reason"]
                    break
                event = r.extract_event_state(caller, run, ROOT, ROUTE, history, [],
                                              candidate["scenes"][0], snapshot, attempt)
                if event is None:
                    failure = "event_state_audit_incomplete_after_retry"
                    break
                replan = r.event_contract.requires_replan(event)
                r.base.write(directory / "EVENT_STATE_DECISION.json", {"status": "REPLAN" if replan else "PASS",
                    "event_state_sha256": r.base.digest(event),
                    "plan_diagnostics": r.event_contract.plan_diagnostics(event)})
                if replan:
                    failure = "event_state_requires_replan_after_limit"
                    repair = {"instruction": "Revise the complete A plan to resolve every binding finding. "
                              "Advance an existing historical event with genuine new progress. Preserve historical "
                              "actors, event states and uncertainty; do not replay completed milestones.",
                              "failure": "event_state_requires_replan", "previous_response": candidate,
                              "plan_checks": r.event_contract.binding_plan_checks(event)}
                    continue
                plan, failure = candidate, None
                r.base.write(folder / f"{SCENE}.EVENT_STATE.json", event)
                r.base.write(folder / f"{SCENE}.EVENT_COVERAGE.json", r.event_contract.derive_coverage(event, history))
                break
            if plan is not None:
                r.base.write(folder / f"{SCENE}.PLAN.json", plan)
            if plan is not None and plan["scenes"]:
                realization, audits = r.produce_scene(caller, run, ROOT, ROUTE, history, [], plan["scenes"][0], [],
                                                      latest_state=snapshot, event_state=event)
                r.base.write(folder / f"{SCENE}.ATTEMPTS.json", audits)
                if realization is None:
                    failure = "audit_incomplete" if audits and audits[-1].get("failure_category") == "audit_incomplete" else "no_accepted_scene"
                else:
                    r.base.write(folder / f"{SCENE}.json", realization)
                    accepted = {"scene_id": SCENE, "dataset_card_id": None, "turns": len(realization["turns"]),
                                "qwen_review": "PASS", "duplicate_check": "PASS",
                                "latest_state_sha256": r.base.digest(snapshot), "event_state_sha256": r.base.digest(event),
                                "acceptance": deepcopy(audits[-1]), "semantic_certified": False}
    except RuntimeError as error:
        if str(error) != "provider_request_failed_after_retry_policy" and not isinstance(error, r.LocalRequestError):
            raise
        failure = str(error)
    final_history = r.base.append_scene(history, ROOT, ROUTE, realization) if accepted else history
    r.base.require(final_history[:120] == history and all(t["turn_ref"].startswith("R02:A:S03:")
                   for t in final_history[120:]), "A3_append_scope_changed")
    result = {"root_id": ROOT, "route": ROUTE, "target_scenes": 3, "available_new_slots": [SCENE],
              "inherited_scenes": 2, "new_accepted_scenes": int(accepted is not None),
              "accepted_scenes": 2 + int(accepted is not None),
              "scene_metadata": deepcopy(inherited["scene_metadata"]) + ([accepted] if accepted else []),
              "failures": [] if accepted else [{"scene_id": SCENE, "reason": failure or "plan_rejected_after_retry"}],
              "shortfall_scenes": int(accepted is None), "input_turns": 120,
              "input_history_sha256": HISTORY_SHA, "output_history_sha256": r.base.digest(final_history),
              "added_turns": len(final_history) - 120, "semantic_certified": False}
    r.base.write(folder / "RESULT.json", result)
    cumulative = {"root_id": ROOT, "route": ROUTE, "turns": final_history,
                  "version": VERSION, "model": r.MODEL, "input_history_sha256": HISTORY_SHA,
                  "sequential_root_history": True, "accepted_A_scenes": result["accepted_scenes"]}
    r.base.write(folder / "cumulative_dialogue.json", cumulative)
    r.base.write(run / "sources_merged" / f"{ROOT}.json", {**cumulative, "turn_count": len(final_history),
        "accepted_scene_order": ["R02:A:S01", "R02:A:S02"] + (["R02:A:S03"] if accepted else []),
        "new_accepted_scenes": int(accepted is not None), "source_context_policy": "base_then_exact_A2_then_new_A_S03"})
    return result


def prepare(run):
    r.base.require(not run.exists() and run.parent == r.HERE / "runs", "new_A3_run_required")
    parent = r.checked_parent_a(r.PARENT_RUN, ROOT)
    inherited = r.base.read(parent["cumulative_path"])["turns"]
    r.base.require(len(inherited) == 120 and r.base.digest(inherited) == HISTORY_SHA, "wrong_A2_parent")
    r.base.require([row["scene_id"] for row in parent["result"]["scene_metadata"]] == ["S01", "S02"],
                   "wrong_A2_accepted_scenes")
    run.mkdir(parents=True)
    r.base.feedback.compile_feedback(run / "feedback", [ROOT])
    r.base.write(run / "sources" / f"{ROOT}.json", parent["original"])
    receipts = []
    for source in parent["paths"]:
        target = run / "frozen_parent" / source.relative_to(parent["parent"])
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        receipts.append({"original_path": str(source), "frozen_path": str(target),
                         "sha256": r.base.digest(target.read_bytes())})
    local = run / "frozen_parent" / parent["cumulative_path"].relative_to(parent["parent"])
    record = {"root_id": ROOT, "parent_run": str(parent["parent"]), "accepted_scenes": 2,
        "source_path": str(parent["cumulative_path"]), "local_path": str(local),
        "source_sha256": r.base.digest(local.read_bytes()), "history_sha256": HISTORY_SHA,
        "new_A_calls": 0, "files": r.base.bindings(parent["paths"]), "semantic_certified": False,
        "acceptance_scope": "exact inherited A2; old failed S03 evidence preserved only in frozen_parent"}
    r.base.write(run / "INHERITED_A.json", record)
    r.base.write(run / "INHERITED_RESULT.json", parent["result"])
    r.base.write(run / "FROZEN_PARENT_INDEX.json", {"files": receipts})
    old_folder = parent["cumulative_path"].parent
    for source in old_folder.iterdir():
        if source.name.startswith(("S01.", "S02.")):
            target = run / "roots" / ROOT / "A" / source.name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    provider, transport_code, tokenizer = r.transport_inputs()
    r.base.write(run / "provider_profile.json", provider)
    preview = r.state_contract.build_state_request(ROOT, ROUTE, SCENE, inherited, [])
    r.base.write(run / "previews/S03.state.json", preview)
    code = sorted(set(list(r.HERE.glob("*.py")) + list(r.LEGACY.glob("*.py")) + transport_code))
    snapshots = []
    for index, source in enumerate(code):
        target = run / "execution_code" / f"{index:03d}_{source.name}"
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        snapshots.append({"original_path": str(source), "frozen_path": str(target),
                          "sha256": r.base.digest(target.read_bytes())})
    r.base.write(run / "EXECUTION_CODE_SNAPSHOT.json", {"files": snapshots})
    paths = code + list(tokenizer) + parent["paths"] + [r.HERE / "POLICY.md"]
    paths += [p for p in run.rglob("*") if p.is_file()]
    manifest = {"version": VERSION, "model": r.MODEL, "review_model": r.MODEL, "root_id": ROOT,
        "route": ROUTE, "new_scene_slots": [SCENE], "inherited_scenes": 2, "inherited_history_sha256": HISTORY_SHA,
        "logical_call_max": LOGICAL_MAX, "http_max": LOGICAL_MAX * (r.TRANSPORT_RETRIES + 1),
        "minimum_http_gap_after_response_seconds": r.MIN_HTTP_GAP_SECONDS,
        "question_calls": 0, "solver_calls": 0, "training_calls": 0, "B_calls": 0,
        "source_allocation_calls": 0, "purpose_calls": 0, "old_failed_plan_reused": False,
        "prepared_at_utc": r.now(), "files": r.base.bindings(sorted(set(paths)))}
    r.base.write(run / "MANIFEST.json", manifest)
    return {"status": "PREPARED", "input_turns": 120, "inherited_A_scenes": 2,
            "new_scene_slots": [SCENE], "logical_call_max": LOGICAL_MAX, "http_max": manifest["http_max"], "http_calls": 0}


def verify(run):
    m = r.base.read(run / "MANIFEST.json")
    r.base.require(m["version"] == VERSION and m["model"] == r.MODEL and m["new_scene_slots"] == [SCENE],
                   "wrong_A3_manifest")
    r.base.require(m["logical_call_max"] == LOGICAL_MAX and m["http_max"] == LOGICAL_MAX * (r.TRANSPORT_RETRIES + 1)
                   and m["minimum_http_gap_after_response_seconds"] == r.MIN_HTTP_GAP_SECONDS, "A3_budget_changed")
    r.base.verify_bindings(m["files"]); r.verify_inherited_a(run); _history(run)
    for row in r.base.read(run / "FROZEN_PARENT_INDEX.json")["files"]:
        r.base.require(r.base.digest(Path(row["frozen_path"]).read_bytes()) == row["sha256"], "frozen_parent_changed")
    for row in r.base.read(run / "EXECUTION_CODE_SNAPSHOT.json")["files"]:
        r.base.require(r.base.digest(Path(row["frozen_path"]).read_bytes()) == row["sha256"], "execution_snapshot_changed")
    return m


def live(run):
    manifest = verify(run)
    r.base.require(not (run / "LIVE_STARTED.json").exists(), "A3_run_cannot_restart")
    r.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with r.LOCK_PATH.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r.base.write(run / "LIVE_STARTED.json", {"started_at_utc": r.now(), "manifest_sha256": r.base.digest(manifest)})
        caller = r.Caller(run)
        result = _flow(run, caller)
        verify(run)
        report = {"status": "A3_COMPLETE" if result["accepted_scenes"] == 3 else "A3_SHORTFALL", "result": result,
            "logical_calls": caller.request_counter, "http_attempts": caller.http_counter,
            "new_accepted_A_scenes": result["new_accepted_scenes"], "accepted_A_scenes": result["accepted_scenes"],
            "B_calls": 0, "purpose_calls": 0, "question_calls": 0, "solver_calls": 0, "training_calls": 0,
            "semantic_certified": False, "completed_at_utc": r.now()}
        r.base.write(run / "REPORT.json", report)
        paths = [p for p in run.rglob("*") if p.is_file() and p.name != "MERGE_MANIFEST.json"]
        r.base.write(run / "MERGE_MANIFEST.json", {"version": VERSION, "root_id": ROOT,
            "accepted_A_scenes": result["accepted_scenes"], "files": r.base.bindings(paths)})
        return report


def audit(run):
    manifest = verify(run)
    directories = sorted(p for p in (run / "requests").glob("*") if p.is_dir())
    failures, recorded, replay = [], {}, None
    def check(condition, code):
        if not condition:
            failures.append(code)
    class ReplayCaller:
        request_counter = 0
        def call(self, label, payload, proof):
            self.request_counter += 1
            directory = run / "requests" / f"{self.request_counter:03d}-{label}"
            r.base.require(directory in directories, "replay_request_missing:" + directory.name)
            r.base.require(r.base.read(directory / "REQUEST.json") == payload, "request_replay_mismatch:" + directory.name)
            if (directory / "PROOF.json").exists():
                actual = r.base.read(directory / "PROOF.json")
                r.base.require(all(actual.get(k) == v for k, v in proof.items()), "proof_replay_mismatch:" + directory.name)
            if (directory / "LOCAL_ERROR.json").exists():
                raise r.LocalRequestError(r.base.read(directory / "LOCAL_ERROR.json")["error_code"])
            if not (directory / "RESPONSE.json").exists():
                r.base.require((directory / "ERROR.json").exists(), "missing_response_or_error")
                raise RuntimeError("provider_request_failed_after_retry_policy")
            r.base.require((directory / "PROOF.json").is_file(), "response_without_proof")
            return r.base.read(directory / "RESPONSE.json")["text"], directory
    if (run / "LIVE_STARTED.json").exists():
        try:
            r.base.require((run / "REPORT.json").exists() and (run / "MERGE_MANIFEST.json").exists(), "A3_live_incomplete")
            r.base.verify_bindings(r.base.read(run / "MERGE_MANIFEST.json")["files"])
            caller = ReplayCaller()
            with patch.object(r.base, "write", side_effect=lambda path, value: recorded.__setitem__(Path(path), deepcopy(value))):
                replay = _flow(run, caller)
            check(caller.request_counter == len(directories), "unexpected_request_directory")
            for path, value in recorded.items():
                check(path.is_file() and r.base.read(path) == value, "artifact_replay_mismatch:" + str(path.relative_to(run)))
            report = r.base.read(run / "REPORT.json")
            check(report["result"] == replay, "report_result_mismatch")
            check(report["status"] == ("A3_COMPLETE" if replay["accepted_scenes"] == 3 else "A3_SHORTFALL"), "report_status_mismatch")
            for key in ("B_calls", "purpose_calls", "question_calls", "solver_calls", "training_calls"):
                check(report.get(key) == 0, "forbidden_stage:" + key)
            check(report["logical_calls"] == len(directories), "logical_count_mismatch")
            check(report["accepted_A_scenes"] == replay["accepted_scenes"]
                  and report["new_accepted_A_scenes"] == replay["new_accepted_scenes"], "acceptance_count_mismatch")
        except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
            failures.append("replay:" + str(error))
    else:
        check(not directories and not (run / "REPORT.json").exists(), "requests_without_live_start")
    attempts = list((run / "requests").glob("*/HTTP_ATTEMPT*.json"))
    check(len(directories) <= manifest["logical_call_max"] and len(attempts) <= manifest["http_max"], "call_budget_exceeded")
    for directory in directories:
        payload = r.base.read(directory / "REQUEST.json")
        http = sorted(directory.glob("HTTP_ATTEMPT*.json"))
        check(len(http) <= r.TRANSPORT_RETRIES + 1, "transport_budget_exceeded")
        for path in http:
            row = r.base.read(path)
            check(row.get("payload_sha256") == r.base.digest(payload) and row.get("actual_payload_equals_frozen") is True
                  and row.get("requested_model") == r.MODEL, "http_payload_changed")
            number = row.get("request_attempt", 1)
            throttle = r.base.read(directory / f"THROTTLE_{number:02d}.json")
            check(throttle.get("minimum_gap_after_previous_response_seconds") == r.MIN_HTTP_GAP_SECONDS,
                  "http_throttle_policy_changed")
            if number > 1:
                before = r.base.read(directory / f"HTTP_RESULT_{number-1:02d}.json")
                check(row.get("same_payload_retry") is True and before.get("outcome") == "error"
                      and before.get("http_status") in (503, 504), "retry_policy_changed")
        if (directory / "RESPONSE.json").exists():
            check(bool(http), "response_without_http_attempt")
            metadata = r.base.read(directory / "RESPONSE.json")["metadata"]
            check(metadata.get("returned_model") == r.MODEL and metadata.get("finish_reason") == "stop", "response_model_changed")
    if (run / "REPORT.json").exists():
        check(r.base.read(run / "REPORT.json")["http_attempts"] == len(attempts), "http_count_mismatch")
    result = {"status": "PASS" if not failures else "FAIL", "run": str(run), "version": VERSION, "failures": failures,
        "generation_status": "PREPARED_NO_LIVE_CALLS" if replay is None else ("A3_COMPLETE" if replay["accepted_scenes"] == 3 else "A3_SHORTFALL"),
        "accepted_A_scenes": replay["accepted_scenes"] if replay else 2,
        "logical_calls": len(directories), "http_attempts": len(attempts), "semantic_certified": False,
        "audited_at_utc": r.now()}
    # An offline smoke audit must not masquerade as the final live audit.
    stem = "FINAL_AUDIT" if (run / "LIVE_STARTED.json").exists() else "PREPARE_AUDIT"
    output = run / f"{stem}.json"
    if output.exists():
        index = 2
        while (run / f"{stem}_{index:03d}.json").exists():
            index += 1
        output = run / f"{stem}_{index:03d}.json"
    r.base.write(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "live", "audit"])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](args.run.resolve()), ensure_ascii=False, indent=2))
