"""Bounded same-Qwen re-review of frozen v18 failures; never generates data.

Old cards are audit input for the SAME historical scenes, not reassigned event
guides. Nothing from this probe changes an old acceptance decision or pool.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path

import runner as r

HERE = Path(__file__).resolve().parent
HISTORICAL = HERE.parent / "h4_sequential_ab_v18/runs/sequential_pilot_3roots_01"
CASES = (("104814_B_scan", "011", "event"),
         ("138343_A_progress", "030", "event"),
         ("135466_B_old_step", "024", "review"),
         ("138343_B_lost_wallet", "036", "review"))


def prepare(run):
    r.base.require(not run.exists(), "probe_already_exists")
    run.mkdir(parents=True)
    files = list(HERE.glob("*.py")) + [HERE / "POLICY.md"]
    cases = []
    for name, prefix, kind in CASES:
        origin = next((HISTORICAL / "requests").glob(prefix + "-*")) / "REQUEST.json"
        request = r.base.read(origin)
        marker = "\nEVENT_STATE_INPUT_JSON\n" if kind == "event" else "\nREVIEW INPUT DATA:\n"
        context = json.JSONDecoder().raw_decode(request["messages"][0]["content"].split(marker, 1)[1])[0]
        path = run / "cases" / (name + ".json")
        r.base.write(path, {"name": name, "kind": kind, "context": context,
                            "origin": str(origin), "origin_sha256": r.base.digest(origin.read_bytes())})
        cases.append(str(path))
        files.extend((origin, path))
    profile, transport, tokenizer = r.transport_inputs()
    r.base.write(run / "provider_profile.json", profile)
    files.extend((*transport, *tokenizer, run / "provider_profile.json"))
    manifest = {"version": "v20_historical_review_probe_1", "model": r.MODEL,
        "cases": cases, "http_max": len(CASES) * 4,
        "files": r.base.bindings(files), "new_generation_calls": 0,
        "dataset_events_reassigned": False, "old_decisions_modified": False,
        "scope": "same historical scenes and plans re-reviewed under new contract; not a new training pool"}
    r.base.write(run / "MANIFEST.json", manifest)
    return {"status": "PREPARED", "cases": len(cases), "http_calls": 0}


def raw_scene(context):
    scene = deepcopy(context["new_scene"])
    for turn in scene["turns"]:
        turn.pop("turn_ref", None)
    return scene


def expectations(name, verdict):
    if not verdict.get("schema_valid"):
        return {"status": "AUDIT_INCOMPLETE", "passed": False}
    parsed = verdict["parsed"]
    if name == "104814_B_scan":
        passed = sorted(row["development_index"] for row in parsed["plan_checks"]) == [1, 2, 3, 4, 5]
        detail = "All five checks present and evidence valid; does not assert plan must pass."
    elif name == "138343_A_progress":
        rows = {row["development_index"]: row for row in parsed["plan_checks"]}
        passed = all(rows[index]["verdict"] == "compatible" for index in (1, 2))
        detail = "New ticket/hotel completion should be compatible; timing clarification elsewhere may still be required."
    elif name == "135466_B_old_step":
        role = parsed["route_fulfillment"]
        passed = role["satisfied"] is False and role["relation_to_history"] == "old_event_substep"
        detail = "Train travel for the established interview must be identified as an old-event substep."
    else:
        passed = verdict["status"] == "PASS"
        detail = "The historical lost-wallet scene should remain acceptable with exact reference lookup."
    return {"status": "PASS" if passed else "EXPECTATION_NOT_MET", "passed": passed, "detail": detail}


def live(run):
    manifest = r.base.read(run / "MANIFEST.json")
    r.base.verify_bindings(manifest["files"])
    r.base.require(not (run / "LIVE_STARTED.json").exists(), "probe_cannot_replay")
    r.base.write(run / "LIVE_STARTED.json", {"at": r.now(), "scope": manifest["scope"]})
    caller, results = r.Caller(run), []
    for path in manifest["cases"]:
        case = r.base.read(path)
        c, repair, candidate, verdict = case["context"], None, None, None
        for attempt in (1, 2):
            if case["kind"] == "event":
                payload = r.event_contract.build_request(c["root_id"], c["route"], c["scene_plan"],
                    c["cumulative_history"], c["dataset_cards"], c["latest_state"], repair=repair)
            else:
                payload, _ = r.review_request(c["root_id"], c["route"], c["scene_plan"], c["branch_history"],
                    raw_scene(c), c["other_branch_added_turns_for_review_only"], card=c["dataset_card_guidance"],
                    repair=repair, latest_state=c["latest_state"], event_state=c["event_state"])
            proof = {"case": case["name"], "kind": case["kind"], "historical_context_sha256": r.base.digest(c),
                     "attempt": attempt, "old_scene_or_plan_unchanged": True}
            try:
                raw, directory = caller.call(case["name"] + f"-{attempt}", payload, proof)
            except RuntimeError as error:
                verdict = {"status": "TRANSPORT_ERROR", "schema_valid": False, "error": str(error)}
                break
            if case["kind"] == "event":
                try:
                    if repair:
                        r.event_contract.validate_repair_preservation(repair["previous_raw"], raw, repair)
                    parsed = r.event_contract.validate(raw, c["root_id"], c["route"], c["scene_plan"], c["cumulative_history"])
                    verdict = {"schema_valid": True, "status": "REPLAN" if r.event_contract.requires_replan(parsed) else "PASS",
                               "parsed": parsed, "citation_provenance": r.evidence_refs.citation_provenance(
                                   r.event_contract._parse(raw), c["cumulative_history"])}
                except (ValueError, TypeError) as error:
                    verdict = {"schema_valid": False, "status": "AUDIT_INCOMPLETE", "error": str(error),
                               "errors": getattr(error, "errors", [])}
                    repair = r.event_contract.build_repair(error, raw, history=c["cumulative_history"])
            else:
                try:
                    previous = candidate
                    candidate = raw if repair is None else r.realization_review.merge_review_repair(candidate, raw, repair)
                    if repair is not None:
                        r.base.write(directory / "FORMAT_REPAIR_MERGE.json", {"previous": previous,
                            "raw_repair": raw, "repair_spec": repair, "candidate": candidate})
                    verdict = r.realization_review.validate_review(candidate, c["root_id"], c["route"],
                        c["scene_plan"], raw_scene(c), dataset_card=c["dataset_card_guidance"],
                        latest_state=c["latest_state"], history=c["branch_history"], event_state=c["event_state"])
                except (ValueError, TypeError) as error:
                    verdict = {"status": "AUDIT_INCOMPLETE", "schema_valid": False, "error": str(error)}
                if verdict.get("schema_valid") is False and attempt == 1:
                    repair = r.realization_review.build_format_repair(verdict, candidate, c["scene_plan"],
                        c["branch_history"], raw_scene(c), dataset_card=c["dataset_card_guidance"],
                        latest_state=c["latest_state"], event_state=c["event_state"])
            r.base.write(directory / "VALIDATION.json", verdict)
            if verdict.get("schema_valid"):
                break
            if attempt == 1:
                r.base.write(directory / "REPAIR_SPEC.json", repair)
        row = {"case": case["name"], "attempts": attempt, "validation": verdict,
               "expectation": expectations(case["name"], verdict)}
        results.append(row)
        r.base.write(run / "results" / (case["name"] + ".json"), row)
        print(json.dumps({"case": row["case"], "attempts": attempt, "expectation": row["expectation"]}, ensure_ascii=False), flush=True)
    r.base.verify_bindings(manifest["files"])
    report = {"model": r.MODEL, "cases": results, "expectations_met": sum(x["expectation"]["passed"] for x in results),
              "total_cases": len(results), "http_attempts": caller.http_counter,
              "new_generation_calls": 0, "solver_calls": 0, "training_started": False,
              "old_decisions_modified": False, "dataset_events_reassigned": False,
              "limit": "One bounded repeat of known historical cases is not proof of general semantic reliability."}
    r.base.write(run / "REPORT.json", report)
    return {key: value for key, value in report.items() if key != "cases"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "live"))
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    r.base.require(run.parent == HERE / "probes", "probe_path_invalid")
    print(json.dumps((prepare if args.command == "prepare" else live)(run), ensure_ascii=False, indent=2))
