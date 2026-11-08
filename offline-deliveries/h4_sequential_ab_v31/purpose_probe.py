"""Bounded same-Qwen purpose/planner probe on three frozen v30 materials.

No card/source allocation, state extraction, event review, Realizer, QA, Solver,
or training. Structural acceptance is never a semantic event acceptance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil

import runner as r
import regression_probe as freeze_helpers

ORIGIN = r.HERE.parent / "h4_sequential_ab_v30/runs/inherited_a2_b3_01"
VERSION = "v31_frozen_purpose_planner_probe"
CASES = (("S01", "004", "007"), ("S02", "012", "015"), ("S03", "017", "020"))
ROOT = "cc-train-episode-104814"
ROUTE = "dataset_new_event"


def _find(prefix):
    matches = list((ORIGIN / "requests").glob(prefix + "-*"))
    r.base.require(len(matches) == 1, "original_request_not_unique:" + prefix)
    return matches[0]


def _decode(payload, marker="\nINPUT_JSON\n"):
    return freeze_helpers._decode(payload, marker)


def _purpose_request(run, case):
    c = case["inputs"]
    speakers = sorted({x["speaker"] for x in c["history"]})
    payload = r.purpose_contract.build_request(ROOT, case["scene_id"], c["card"], speakers, r.MODEL)
    projection = r.purpose_contract.project_feedback(run / "feedback", ROOT, r.base.feedback)
    r.base.require(projection == r.base.read(run / "PURPOSE_FEEDBACK_PROJECTION.json"), "purpose_projection_changed")
    suffix = r.purpose_contract.feedback_suffix(projection)
    before = payload["messages"][0]["content"]
    payload["messages"][0]["content"] += suffix
    receipt = {"feedback_kind": "purpose_guidance_projection", "full_feedback_appended": False,
        "projection_sha256": projection["projected_feedback_sha256"], "source_sha256": projection["source_sha256"],
        "suffix_sha256": r.base.digest(suffix.encode()), "before_user_text_sha256": r.base.digest(before.encode()),
        "request_payload_sha256": r.base.digest(payload), "current_generation_feedback_used": False}
    proof = {"root_id": ROOT, "route": ROUTE, "scene_id": case["scene_id"], "role": "purpose_proposer",
        "probe_version": VERSION, "frozen_case_sha256": r.base.digest(case),
        "dataset_card_sha256": r.base.digest(c["card"]), "allowed_speakers": speakers,
        "input_scope": "card_allowed_speakers_frozen_feedback_only", "history_or_state_supplied": False,
        "source_context_sha256": r.base.digest(c["history"]), "feedback_receipt": receipt,
        "old_purpose_or_plan_supplied": False}
    return payload, proof


def _planner_request(run, case, purpose, repair=None):
    c = case["inputs"]
    payload, proof = r.make_request(run, ROOT, ROUTE, "planner", c["history"], [c["card"]], case["scene_id"],
                                  latest_state=c["latest_state"], purpose_input=purpose, repair=repair)
    sent = _decode(payload)
    r.base.require(sent["cumulative_history"] == c["history"], "planner_history_changed")
    r.base.require(sent["dataset_cards"] == [c["card"]] and sent["purpose_input"] == purpose,
                   "planner_card_or_new_purpose_changed")
    proof.update(probe_version=VERSION, frozen_case_sha256=r.base.digest(case),
                 schema_local_repair_only=repair is not None, repair_spec_sha256=r.base.digest(repair),
                 old_purpose_or_plan_supplied=False)
    return payload, proof


def _plan_checked(case, raw):
    c = case["inputs"]
    try:
        plan = r.validate_plan(raw, ROOT, ROUTE, c["history"], [c["card"]], case["scene_id"], latest_state=c["latest_state"])
    except (ValueError, TypeError) as error:
        return {"status": "FORMAT_ERROR", "schema_valid": False, "error": str(error),
                "errors": deepcopy(getattr(error, "errors", [])), "semantic_certified": False}
    return {"status": "STRUCTURE_PASS" if plan["scenes"] else "STRUCTURE_PASS_NO_PLAN", "schema_valid": True,
            "plan": plan, "semantic_certified": False}


def _normalized_path(path):
    if path.startswith("scene."):
        path = "response.scenes[0]." + path[len("scene."):]
    if "[*]" in path:
        # Only mapping/reference errors use wildcard diagnostics. Preserve all
        # event prose while allowing the model to repair its reference group.
        path = path.split("[*]", 1)[0]
    return path


def _repair_plan(case, raw, checked):
    """Build a format-only repair with immutable unaffected response values."""
    try:
        previous = r.parse_object(raw)
    except (ValueError, TypeError):
        return None  # Invalid JSON has no trustworthy local repair boundary.
    paths = [_normalized_path(x.get("path", "")) for x in checked.get("errors", [])]
    paths = [x for x in paths if x.startswith("response.")]
    # Runner's exact-citation checks use a scalar error; derive the precise
    # faulty citation leaves from equality checks, never semantic inference.
    if (checked.get("error") in ("anchor_ref_missing", "anchor_quote_not_exact", "anchor_schema_invalid")
            or str(checked.get("error", "")).startswith("citation_reference_not_in_context:")):
        by_ref = {x["turn_ref"]: x for x in case["inputs"]["history"]}
        for i, scene in enumerate(previous.get("scenes", [])):
            for j, anchor in enumerate(scene.get("source_anchors", [])):
                path = f"response.scenes[{i}].source_anchors[{j}]"
                if not isinstance(anchor, dict) or not {"turn_ref"} <= set(anchor) <= {"turn_ref", "quote"}:
                    paths.append(path)
                elif anchor.get("turn_ref") not in by_ref:
                    paths.extend([path + ".turn_ref", path + ".quote"])
                elif "quote" in anchor and (not isinstance(anchor["quote"], str) or not anchor["quote"] or anchor["quote"] not in by_ref[anchor["turn_ref"]]["text"]):
                    paths.append(path + ".quote")
    if not paths:
        return None
    # A repair may not replace the entire response, scene or valid narrative
    # container. Missing/ill-typed semantic scalar fields are authorable by Qwen.
    forbidden = {"response.scenes", "response.scenes[0]", "response.scenes[0].developments"}
    if any(path in forbidden for path in paths):
        return None
    return {"mode": "complete_response_preserve_unaffected_fields", "failure": checked["error"],
        "errors": checked.get("errors", []), "allowed_paths": sorted(set(paths)),
        "previous_response": previous, "previous_response_sha256": r.base.digest(previous),
        "instruction": "Return the complete corrected JSON for this same scene. Change ONLY the listed allowed_paths; preserve every other field and array entry exactly. Fix schema/ID/citation errors only. Do not replan, rewrite valid developments, alter the card or new purpose, drop a scene, or change unaffected text. Historical citations must copy existing references and exact substrings from cumulative_history. The program rejects every change outside allowed_paths. No event acceptance or semantic verdict is being repaired.",
        "semantic_certified": False}


def _preserve_plan(raw, repair):
    current = r.parse_object(raw)
    previous = repair["previous_response"]
    r.base.require(r.base.digest(previous) == repair["previous_response_sha256"], "repair_previous_changed")
    allowed = repair["allowed_paths"]
    def visit(old, new, path):
        if any(path == a or path.startswith(a + ".") or path.startswith(a + "[") for a in allowed):
            return
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                sub = path + "." + key
                if key not in old or key not in new:
                    r.base.require(sub in allowed, "planner_repair_protected_field_changed:" + sub)
                else:
                    visit(old[key], new[key], sub)
        elif isinstance(old, list) and isinstance(new, list):
            r.base.require(len(old) == len(new), "planner_repair_protected_array_changed:" + path)
            for i, (a, b) in enumerate(zip(old, new)):
                visit(a, b, f"{path}[{i}]")
        else:
            r.base.require(type(old) is type(new) and old == new, "planner_repair_protected_value_changed:" + path)
    visit(previous, current, "response")
    return current


def _case_flow(run, case, call, save):
    """Shared live/replay control flow; all model outcomes remain explicit."""
    requests = []
    def invoke(stage, attempt, payload, proof):
        proof.update(logical_stage=stage, logical_attempt=attempt)
        label = f"purpose-probe-{case['scene_id']}-{stage}-{attempt}"
        raw, directory = call(label, payload, proof)
        requests.append(str(directory))
        return raw, directory
    try:
        payload, proof = _purpose_request(run, case)
        raw, directory = invoke("purpose", 1, payload, proof)
    except (RuntimeError, ValueError) as error:
        return {"scene_id": case["scene_id"], "status": "TRANSPORT_OR_LOCAL_ERROR", "error": str(error),
                "requests": requests, "semantic_certified": False}
    try:
        purpose = r.purpose_contract.validate_purpose(raw, ROOT, ROUTE, case["scene_id"], case["inputs"]["card"])
        checked = {"status": "STRUCTURE_PASS", "schema_valid": True, "purpose": purpose, "semantic_certified": False}
    except (ValueError, TypeError) as error:
        checked = {"status": "FORMAT_ERROR", "schema_valid": False, "error": str(error),
                   "errors": getattr(error, "errors", []), "semantic_certified": False}
        save(directory / "EVALUATION.json", checked)
        return {"scene_id": case["scene_id"], "status": "PURPOSE_FORMAT_ERROR", "purpose_result": checked,
                "requests": requests, "semantic_certified": False}
    save(directory / "EVALUATION.json", checked)
    purpose_result = checked
    repair = None
    for attempt in range(1, 3):
        payload, proof = _planner_request(run, case, purpose, repair)
        try:
            raw, directory = invoke("planner", attempt, payload, proof)
        except (RuntimeError, ValueError) as error:
            checked = {"status": "TRANSPORT_OR_LOCAL_ERROR", "schema_valid": False, "error": str(error), "semantic_certified": False}
            break
        save(directory / "REPAIR_SPEC.json", repair)
        try:
            candidate = raw if repair is None else _preserve_plan(raw, repair)
        except (ValueError, TypeError) as error:
            checked = {"status": "FORMAT_ERROR", "schema_valid": False, "error": str(error), "semantic_certified": False}
            save(directory / "EVALUATION.json", checked)
            break
        save(directory / "CANDIDATE.json", candidate)
        checked = _plan_checked(case, candidate)
        save(directory / "EVALUATION.json", checked)
        if checked["schema_valid"]:
            save(directory / "PLAN_CITATION_PROVENANCE.json",
                 r.evidence_refs.citation_provenance(r.parse_object(candidate), case["inputs"]["history"]))
            break
        if attempt == 1:
            repair = _repair_plan(case, candidate, checked)
            save(directory / "LOCAL_REPAIR_SPEC.json", repair)
            if repair is None:
                break
    return {"scene_id": case["scene_id"], "status": checked["status"], "purpose_result": purpose_result,
            "planner_result": checked, "requests": requests, "logical_calls": len(requests),
            "history_sha256": r.base.digest(case["inputs"]["history"]),
            "card_sha256": r.base.digest(case["inputs"]["card"]), "semantic_certified": False}


def prepare(run):
    r.base.require(not run.exists() and run.parent == r.HERE / "probes", "probe_path_invalid_or_exists")
    run.mkdir(parents=True)
    sources, case_paths = [], []
    def copy(source, destination):
        receipt = freeze_helpers._freeze(source, destination)
        sources.append(receipt)
    for source in sorted((ORIGIN / "feedback").rglob("*")):
        if source.is_file():copy(source, run / "feedback" / source.relative_to(ORIGIN / "feedback"))
    for relative in [f"sources/{ROOT}.json", f"roots/{ROOT}/A/cumulative_dialogue.json", "INHERITED_A.json"]:
        copy(ORIGIN / relative, run / relative)
    projection = r.purpose_contract.project_feedback(run / "feedback", ROOT, r.base.feedback)
    original_projection = r.base.read(ORIGIN / "PURPOSE_FEEDBACK_PROJECTION.json")
    r.base.require({k:v for k,v in projection.items() if k != "source_path"} ==
                   {k:v for k,v in original_projection.items() if k != "source_path"}, "old_feedback_projection_changed")
    r.base.write(run / "PURPOSE_FEEDBACK_PROJECTION.json", projection)
    copy(ORIGIN / "PURPOSE_FEEDBACK_PROJECTION.json", run / "frozen_origin/PURPOSE_FEEDBACK_PROJECTION.json")
    for scene_id, purpose_prefix, planner_prefix in CASES:
        pd, dd = _find(purpose_prefix), _find(planner_prefix)
        old_context = _decode(r.base.read(dd / "REQUEST.json"))
        history, cards = old_context["cumulative_history"], old_context["dataset_cards"]
        r.base.require(len(cards) == 1 and old_context["target_scene_id"] == scene_id, "old_case_identity_invalid")
        state_path = ORIGIN / f"roots/{ROOT}/B/{scene_id}.LATEST_STATE.json"
        state = r.base.read(state_path)
        r.base.require(old_context["latest_state"] == state, "old_state_mismatch")
        r.branch_history_ok(run, ROOT, ROUTE, history)
        inputs = {"history": history, "card": cards[0], "latest_state": state}
        case = {"scene_id": scene_id, "inputs": inputs, "old_planner_context": old_context,
                "old_purpose_raw": r.base.read(pd / "RESPONSE.json")["text"],
                "old_planner_raw": r.base.read(dd / "RESPONSE.json")["text"],
                "purpose_request_origin": str(pd), "planner_request_origin": str(dd),
                "new_materials": False, "semantic_certified": False}
        for source in [pd / "REQUEST.json", pd / "RESPONSE.json", dd / "REQUEST.json", dd / "RESPONSE.json", state_path]:
            copy(source, run / "frozen_origin" / source.relative_to(ORIGIN))
        path = run / "cases" / (scene_id + ".json")
        r.base.write(path, case); case_paths.append(str(path))
        payload, proof = _purpose_request(run, case)
        r.base.write(run / "previews" / (scene_id + ".purpose.json"), {"payload": payload, "proof": proof})
        # This is only a capacity/shape preview; live always uses the NEW purpose.
        payload, proof = _planner_request(run, case, old_context["purpose_input"])
        r.base.write(run / "previews" / (scene_id + ".planner.json"), {"payload": payload, "proof": proof,
            "offline_only_uses_historical_purpose_for_shape": True})
    profile, transport, tokenizer = r.transport_inputs()
    r.base.write(run / "provider_profile.json", profile)
    code = sorted(set(r.HERE.glob("*.py")) | set(r.LEGACY.glob("*.py")) | set(transport))
    snapshots = [freeze_helpers._freeze(p, run / "execution_code" / f"{i:03d}_{p.name}") for i,p in enumerate(code)]
    r.base.write(run / "EXECUTION_CODE_SNAPSHOT.json", {"files": snapshots})
    files = code + list(tokenizer) + [r.HERE / "POLICY.md"]
    files += [Path(row[key]) for row in sources for key in ("original_path", "frozen_path")]
    files += list(run.rglob("*.json")) + [Path(row["frozen_path"]) for row in snapshots]
    manifest = {"version": VERSION, "origin_run": str(ORIGIN), "model": r.MODEL, "cases": case_paths,
                "max_purpose_calls_per_case": 1, "max_planner_calls_per_case": 2,
                "logical_call_max": 9, "http_max": 9*(r.TRANSPORT_RETRIES+1),
                "minimum_http_gap_after_response_seconds": r.MIN_HTTP_GAP_SECONDS,
                "files": r.base.bindings(files), "sources_reassigned": False, "new_scenes": 0,
                "question_calls": 0, "solver_calls": 0, "training_calls": 0,
                "semantic_certified": False, "acceptance_scope": "purpose_and_plan_structure_only",
                "prepared_at_utc": r.now()}
    r.base.write(run / "MANIFEST.json", manifest)
    return {"status": "PREPARED", "cases": 3, "logical_call_max": 9, "http_max": manifest["http_max"], "http_calls": 0}


def verify(run):
    manifest = r.base.read(run / "MANIFEST.json")
    r.base.require(manifest["version"] == VERSION and manifest["model"] == r.MODEL, "probe_version_model_changed")
    r.base.require(manifest["logical_call_max"] == 9 and manifest["max_purpose_calls_per_case"] == 1
        and manifest["max_planner_calls_per_case"] == 2 and manifest["http_max"] == 9*(r.TRANSPORT_RETRIES+1), "probe_budget_changed")
    r.base.require(manifest["minimum_http_gap_after_response_seconds"] == r.MIN_HTTP_GAP_SECONDS, "http_gap_changed")
    r.base.verify_bindings(manifest["files"])
    for path in manifest["cases"]:
        case = r.base.read(path); c = case["inputs"]
        r.base.require(c["history"] == case["old_planner_context"]["cumulative_history"], "case_history_changed")
        r.base.require([c["card"]] == case["old_planner_context"]["dataset_cards"], "case_card_changed")
        r.base.require(c["latest_state"] == case["old_planner_context"]["latest_state"], "case_state_changed")
        payload, proof = _purpose_request(run, case)
        r.base.require(r.base.read(run / "previews" / (case["scene_id"] + ".purpose.json")) == {"payload":payload,"proof":proof}, "purpose_preview_changed")
        payload, proof = _planner_request(run, case, case["old_planner_context"]["purpose_input"])
        r.base.require(r.base.read(run / "previews" / (case["scene_id"] + ".planner.json")) ==
            {"payload":payload,"proof":proof,"offline_only_uses_historical_purpose_for_shape":True}, "planner_preview_changed")
    return manifest


def _report(rows, logical_calls, http_attempts):
    return {"status": "STRUCTURE_PASS" if all(x["status"] in ("STRUCTURE_PASS", "STRUCTURE_PASS_NO_PLAN") for x in rows) else "STRUCTURE_INCOMPLETE",
        "cases": rows, "logical_calls": logical_calls, "http_attempts": http_attempts,
        "new_scenes": 0, "sources_reassigned": False, "question_calls": 0, "solver_calls": 0, "training_calls": 0,
        "semantic_certified": False, "acceptance_scope": "purpose_and_plan_structure_only_not_card_fidelity_or_training_admission"}


def live(run):
    manifest = verify(run)
    r.base.require(not (run / "LIVE_STARTED.json").exists(), "probe_cannot_restart")
    r.base.write(run / "LIVE_STARTED.json", {"started_at_utc":r.now(),"manifest_sha256":r.base.digest(manifest)})
    caller = r.Caller(run)
    rows = []
    for path in manifest["cases"]:
        case = r.base.read(path)
        row = _case_flow(run, case, caller.call, r.base.write)
        rows.append(row); r.base.write(run / "results" / (case["scene_id"] + ".json"),row)
        print(json.dumps({"scene_id":row["scene_id"],"status":row["status"],"logical_calls":len(row["requests"])},ensure_ascii=False),flush=True)
    verify(run)
    report = _report(rows, caller.request_counter, caller.http_counter)
    r.base.write(run / "REPORT.json", report)
    return report


def audit(run):
    manifest = verify(run)
    failures, rows, observed, counter = [], [], [], 0
    def check(ok, code):
        if not ok:failures.append(code)
    def save(path,value):
        check(path.exists() and r.base.read(path) == value,"replay_artifact_mismatch:"+str(path.relative_to(run)))
    def call(label,payload,proof):
        nonlocal counter
        counter += 1; directory=run/"requests"/f"{counter:03d}-{label}"
        observed.append(directory.resolve())
        check(r.base.read(directory/"REQUEST.json")==payload,"request_replay_mismatch:"+directory.name)
        if (directory/"PROOF.json").exists():
            actual=r.base.read(directory/"PROOF.json")
            check(all(actual.get(k)==v for k,v in proof.items()),"proof_replay_mismatch:"+directory.name)
        if not (directory/"RESPONSE.json").exists():
            check((directory/"ERROR.json").exists() or (directory/"LOCAL_ERROR.json").exists(),"missing_failure_artifact:"+directory.name)
            if (directory/"LOCAL_ERROR.json").exists():raise r.LocalRequestError(r.base.read(directory/"LOCAL_ERROR.json")["error_code"])
            raise RuntimeError("provider_request_failed_after_retry_policy")
        return r.base.read(directory/"RESPONSE.json")["text"],directory
    started=(run/"LIVE_STARTED.json").exists()
    if started:
        for path in manifest["cases"]:
            case=r.base.read(path);row=_case_flow(run,case,call,save);rows.append(row)
            check(r.base.read(run/"results"/(case["scene_id"]+".json"))==row,"case_result_mismatch:"+case["scene_id"])
    actual=sorted(p.resolve() for p in (run/"requests").glob("*") if p.is_dir())
    check(sorted(observed)==actual,"unexpected_request_directory")
    check(counter<=9,"logical_budget_exceeded")
    attempts=list((run/"requests").glob("*/HTTP_ATTEMPT*.json"))
    check(len(attempts)<=manifest["http_max"],"http_budget_exceeded")
    for directory in actual:
        payload=r.base.read(directory/"REQUEST.json")
        http=list(directory.glob("HTTP_ATTEMPT*.json"))
        check(len(http)<=r.TRANSPORT_RETRIES+1,"transport_retry_budget_exceeded:"+directory.name)
        for path in http:
            record=r.base.read(path);n=record["request_attempt"]
            check(record["payload_sha256"]==r.base.digest(payload) and record["actual_payload_equals_frozen"] is True
                  and record["requested_model"]==r.MODEL,"http_payload_or_model_mismatch:"+directory.name)
            throttle=r.base.read(directory/f"THROTTLE_{n:02d}.json")
            check(throttle["minimum_gap_after_previous_response_seconds"]==r.MIN_HTTP_GAP_SECONDS,"http_gap_mismatch:"+directory.name)
            if n>1:
                prior=r.base.read(directory/f"HTTP_RESULT_{n-1:02d}.json")
                check(prior["outcome"]=="error" and prior["http_status"] in (503,504),"invalid_transport_retry:"+directory.name)
        if (directory/"RESPONSE.json").exists():
            metadata=r.base.read(directory/"RESPONSE.json")["metadata"]
            check(metadata["returned_model"]==r.MODEL and metadata["finish_reason"]=="stop","response_model_finish_mismatch:"+directory.name)
    if started:
        check(r.base.read(run/"REPORT.json")==_report(rows,counter,len(attempts)),"report_replay_mismatch")
    result={"status":"PASS" if not failures else "FAIL","failures":failures,"run":str(run),
        "probe_status":_report(rows,counter,len(attempts))["status"] if started else "PREPARED_NO_LIVE_CALLS",
        "cases":rows,"logical_calls":counter,"http_attempts":len(attempts),"new_scenes":0,
        "semantic_certified":False,"audited_at_utc":r.now()}
    destination=run/"PURPOSE_PROBE_AUDIT.json";index=2
    while destination.exists():destination=run/f"PURPOSE_PROBE_AUDIT_{index:03d}.json";index+=1
    r.base.write(destination,result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","live","audit"])
    parser.add_argument("--run",required=True,type=Path)
    args=parser.parse_args();result=globals()[args.command](args.run.resolve())
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.command=="audit" and result["status"]!="PASS":raise SystemExit(1)
