"""Three bounded same-Qwen patch requests on unchanged v26 failed records."""
import argparse
import json
from pathlib import Path
import runner as r

ORIGIN = r.HERE.parent / "h4_sequential_ab_v26/runs/inherited_a2_b3_01"
CASES = (("state_references", "005", "state"), ("absence_timeline", "010", "event"),
         ("new_development_timeline", "016", "event"))


def decode(payload, marker):
    return json.JSONDecoder().raw_decode(payload["messages"][0]["content"].split(marker, 1)[1])[0]


def prepare(run):
    r.base.require(not run.exists(), "probe_already_exists")
    cases, files = [], list(r.HERE.glob("*.py")) + [r.HERE / "POLICY.md"]
    for name, prefix, kind in CASES:
        directory = next((ORIGIN / "requests").glob(prefix + "-*"))
        marker = "\nSTATE_INPUT_JSON\n" if kind == "state" else "\nEVENT_STATE_INPUT_JSON\n"
        context = decode(r.base.read(directory / "REQUEST.json"), marker)
        raw = r.base.read(directory / "RESPONSE.json")["text"]
        module = r.state_contract if kind == "state" else r.event_contract
        try:
            if kind == "state":
                module.validate_state(raw, context["root_id"], context["route"], context["scene_id"], context["cumulative_history"])
            else:
                module.validate(raw, context["root_id"], context["route"], context["scene_plan"], context["cumulative_history"])
        except ValueError as error:
            builder = module.build_state_repair if kind == "state" else module.build_repair
            repair = builder(error, raw, history=context["cumulative_history"])
        else:
            raise ValueError("historical_case_no_longer_reproduces")
        path = run / "cases" / (name + ".json")
        r.base.write(path, {"name": name, "kind": kind, "context": context,
                           "previous_raw": raw, "repair": repair, "origin": str(directory)})
        cases.append(str(path))
        files += [path, directory / "REQUEST.json", directory / "RESPONSE.json"]
    profile, transport, tokenizer = r.transport_inputs()
    r.base.write(run / "provider_profile.json", profile)
    files += [*transport, *tokenizer, run / "provider_profile.json"]
    r.base.write(run / "MANIFEST.json", {"version": "v32_three_local_patch_probe", "http_max": 6,
        "cases": cases, "files": r.base.bindings(files), "new_scene_generation": False,
        "source_events_reassigned": False, "old_acceptances_modified": False})
    return {"status": "PREPARED", "cases": 3}


def live(run):
    manifest = r.base.read(run / "MANIFEST.json")
    r.base.verify_bindings(manifest["files"])
    r.base.require(not (run / "LIVE_STARTED.json").exists(), "probe_cannot_restart")
    r.base.write(run / "LIVE_STARTED.json", {"started_at": r.now()})
    caller, rows = r.Caller(run), []
    for path in manifest["cases"]:
        case = r.base.read(path); c, repair = case["context"], case["repair"]
        history = c["cumulative_history"]
        if case["kind"] == "state":
            payload = r.state_contract.build_state_request(c["root_id"], c["route"], c["scene_id"], history,
                c["dataset_cards"], repair=repair, purpose_input=c.get("purpose_input"))
        else:
            payload = r.event_contract.build_request(c["root_id"], c["route"], c["scene_plan"], history,
                c["dataset_cards"], c["latest_state"], repair=repair, purpose_input=c.get("purpose_input"))
        proof = {"role": "historical_local_patch_probe", "kind": case["kind"],
                 "history_sha256": r.base.digest(history), "previous_raw_sha256": r.base.digest(case["previous_raw"]),
                 "repair_spec_sha256": r.base.digest(repair)}
        raw, directory = caller.call(case["name"], payload, proof)
        row = {"case": case["name"], "request": str(directory), "semantic_certified": False}
        try:
            if case["kind"] == "state":
                merged = r.state_contract.merge_state_repair(case["previous_raw"], raw, repair, history)
                checked = r.state_contract.validate_state(merged, c["root_id"], c["route"], c["scene_id"], history)
            else:
                merged = r.event_contract.merge_event_repair(case["previous_raw"], raw, repair, history)
                r.event_contract.validate_repair_preservation(case["previous_raw"], merged, repair)
                checked = r.event_contract.validate(merged, c["root_id"], c["route"], c["scene_plan"], history)
        except (ValueError, TypeError) as error:
            row.update(status="REJECT", reason=str(error))
        else:
            r.base.write(directory / "MERGED.json", merged)
            r.base.write(directory / "VALIDATED.json", checked)
            row.update(status="PASS", merged_sha256=r.base.digest(merged))
        r.base.write(directory / "PROBE_RESULT.json", row)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    r.base.verify_bindings(manifest["files"])
    result = {"status": "PASS" if all(row["status"] == "PASS" for row in rows) else "INCOMPLETE",
              "cases": rows, "http_attempts": caller.http_counter, "new_scenes": 0,
              "question_calls": 0, "solver_calls": 0, "training_calls": 0, "semantic_certified": False}
    r.base.write(run / "REPORT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "live"])
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(globals()[args.command](args.run.resolve()), ensure_ascii=False, indent=2))
