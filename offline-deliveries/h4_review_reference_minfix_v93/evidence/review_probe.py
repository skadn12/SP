"""Bounded actual-model diagnostic review of unchanged v91 primary material.

This is not an expansion run. It allocates no event, generates no dialogue,
and never changes v91 decisions or source bytes. Expected observations are
kept outside model requests. All original responses and failed attempts stay.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
DELIVERY = Path(__file__).resolve().parents[1]
ENGINE = DELIVERY / "expansion_engine"
sys.path.insert(0, str(ENGINE))
import runner as r

OLD = DELIVERY.parent / "h4_planner_cardinality_expansion_batch_v91"
OLD_RUN = OLD / "expansion_engine/runs/ten_roots_B_scene01"
RUN = DELIVERY / "evidence/model_review_probe"
CASES = ((19053, "plan"), (194259, "plan"), (20432, "plan"),
         (38601, "plan"), (25652, "scene"), (38601, "scene"))


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": r.base.digest(path.read_bytes())}


def inputs(number, kind, by_source, allocation):
    root = f"cc-train-episode-{number}"
    folder = OLD_RUN / "roots" / root / "B"
    source = OLD_RUN / "sources" / (root + ".json")
    state_path = folder / "S01.LATEST_STATE.json"
    if number == 20432:
        plan_path = next((OLD_RUN / "requests").glob("037-*")) / "RESPONSE.json"
        plan = r.parse_object(r.base.read(plan_path)["text"])
    else:
        plan_path = folder / "S01.PLAN.json"
        plan = r.base.read(plan_path)
    assigned = allocation["assignments"][root]
    r.base.require(len(assigned) == 1, "probe_requires_original_single_slot")
    card = r.base.dataset_cards.build_realization_guidance(by_source[assigned[0]["source_id"]])
    item = {"root_id": root, "kind": kind,
            "history": r.base.read(source)["turns"],
            "state": r.base.read(state_path), "scene": plan["scenes"][0], "card": card,
            "input_bindings": [artifact(source), artifact(state_path), artifact(plan_path)]}
    if kind == "scene":
        prefix = "043-" if number == 25652 else "048-"
        path = next((OLD_RUN / "requests").glob(prefix + "*")) / "RESPONSE.json"
        item["realization"] = r.parse_object(r.base.read(path)["text"])
        item["input_bindings"].append(artifact(path))
    return item


def expected_observation(number, kind, parsed):
    """A narrow diagnostic, not an independent certificate of all semantics."""
    if kind == "plan":
        findings = r.event_contract.plan_diagnostics(parsed)
        if number in (19053, 194259):
            return any(x["requires_replan"] and x["check_kind"] in ("card_step", "card_dependency")
                       for x in findings)
        if number == 20432:
            return all(check["verdict"] == "compatible" for check in parsed["plan_checks"])
        return any(x["requires_replan"] and
                   (x["check_kind"] == "plan_consistency" and x.get("category") == "participant_identity"
                    or x["check_kind"] == "development" and x.get("issue_type") == "historical_contradiction")
                   for x in findings)
    target = "itinerary_and_time" if number == 25652 else "participant_identity"
    return any(x["check_id"] == target and x["verdict"] == "contradicted"
               for x in parsed["consistency_checks"])


def run():
    r.base.require(not RUN.exists(), "diagnostic_probe_already_started_no_replay")
    cards_path = OLD_RUN / "dataset_cards_result.json"
    allocation_path = OLD_RUN / "UNIQUE_EVENT_ALLOCATION.json"
    cards = r.base.read(cards_path)["accepted_cards"]
    allocation = r.base.read(allocation_path)
    by_source = {card["source_id"]: card for card in cards}
    cases = [inputs(number, kind, by_source, allocation) for number, kind in CASES]
    profile, transport, tokenizer = r.transport_inputs()
    original = r.base.read(DELIVERY / "PRESERVATION_BEFORE.json")
    r.base.verify_bindings(original["files"])
    ancestor = r.base.read(DELIVERY.parent / "h4_card_fidelity_event_consistency_minfix_v92/PRESERVATION_BEFORE.json")
    r.base.verify_bindings(ancestor["files"])
    code_paths = list(ENGINE.glob("*.py")) + list(r.LEGACY.glob("*.py")) + transport
    files = [cards_path, allocation_path, DELIVERY / "PRESERVATION_BEFORE.json",
             Path(__file__), *code_paths, *tokenizer]
    files += [Path(binding["path"]) for case in cases for binding in case["input_bindings"]]
    manifest = {"version": "v93_actual_model_review_probe_v1", "model": r.MODEL,
                "http_max": 24, "logical_case_count": len(cases), "review_attempts_per_case": 2,
                "transport_retries": 1, "purpose": "diagnostic_review_of_original_v91_material",
                "new_event_allocations": 0, "new_dialogue_turns": 0, "question_calls": 0,
                "root20432_observation": "historical development checks allow coherent new event; independent card negatives may still bind",
                "solver_calls": 0, "training_started": False, "semantic_certified": False,
                "capacity_tokenizer_model": "Qwen/Qwen3-14B",
                "prior_v91_verdicts_in_model_request": False,
                "expected_observations_are_not_in_model_requests": True,
                "files": r.base.bindings(files),
                "cases": [{"root_id": case["root_id"], "kind": case["kind"],
                           "input_bindings": case["input_bindings"]} for case in cases]}
    r.base.write(RUN / "MANIFEST.json", manifest)
    r.base.write(RUN / "provider_profile.json", profile)
    snapshots = []
    for index, binding in enumerate(manifest["files"]):
        path = Path(binding["path"])
        if path.suffix == ".py":
            destination = RUN / "execution_code" / f"{index:03d}_{path.name}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
            snapshots.append({"original": binding, "snapshot": artifact(destination)})
    r.base.write(RUN / "EXECUTION_CODE_SNAPSHOT.json", {"files": snapshots})
    caller = r.Caller(RUN)
    rows = []
    stopped = False
    for (number, kind), case in zip(CASES, cases):
        repair = None
        parsed = None
        attempts = []
        for attempt in range(1, 3):
            r.base.verify_bindings(manifest["files"])
            root, scene, history, state = (case[key] for key in ("root_id", "scene", "history", "state"))
            if kind == "plan":
                payload = r.event_contract.build_request(root, r.ROUTES[1], scene, history,
                                                        [case["card"]], state, repair=repair)
                proof = {"root_id": root, "role": "diagnostic_plan_reviewer", "scene_id": "S01"}
            else:
                payload, proof = r.review_request(root, r.ROUTES[1], scene, history,
                    case["realization"], [], card=case["card"], latest_state=state, repair=repair)
            proof.update({"purpose": "review_only_not_expansion", "prior_model_verdict_provided": False,
                          "new_event_allocation": False, "original_inputs": case["input_bindings"]})
            try:
                raw, directory = caller.call(f"{root}-{kind}-review-{attempt}", payload, proof)
            except RuntimeError as error:
                attempts.append({"attempt": attempt, "status": "TRANSPORT_FAILURE", "reason": str(error)})
                stopped = True
                break
            if kind == "plan":
                try:
                    parsed = r.event_contract.validate(raw, root, r.ROUTES[1], scene, history,
                                                      cards=[case["card"]])
                    validation = {"status": "VALID_REVIEW", "requires_replan": r.event_contract.requires_replan(parsed),
                                  "diagnostics": r.event_contract.plan_diagnostics(parsed),
                                  "semantic_certified": False}
                except (ValueError, TypeError) as error:
                    parsed = None
                    repair = r.event_contract.build_repair(error, raw, history=history)
                    validation = {"status": "INVALID_REVIEW", "errors": repair["errors"],
                                  "semantic_certified": False}
            else:
                checked = r.realization_review.validate_review(raw, root, r.ROUTES[1], scene,
                    case["realization"], dataset_card=case["card"], latest_state=state, history=history)
                validation = checked
                parsed = checked["parsed"] if checked["schema_valid"] else None
                if parsed is None:
                    repair = {"failure_codes": checked["schema_failures"], "previous_response": raw,
                              "instruction": "Fix only this review's required field/citation errors. Reassess unchanged original dialogue; preserve genuine conflicts."}
            r.base.write(directory / "VALIDATION.json", validation)
            attempts.append({"attempt": attempt, "request": str(directory),
                             "validation": validation, "original_response": artifact(directory / "RESPONSE.json")})
            if parsed is not None:
                break
        row = {"root_id": case["root_id"], "kind": kind, "attempts": attempts,
               "review_complete": parsed is not None,
               "expected_narrow_observation_found": expected_observation(number, kind, parsed) if parsed else None,
               "semantic_certified": False, "original_source_changed": False}
        rows.append(row)
        r.base.write(RUN / f"root{number}_{kind}_RESULT.json", row)
        print(json.dumps({"root": number, "kind": kind, "review_complete": row["review_complete"],
                          "expected_found": row["expected_narrow_observation_found"]}, ensure_ascii=False), flush=True)
        if stopped:
            break
    r.base.verify_bindings(manifest["files"])
    r.base.verify_bindings(original["files"])
    r.base.verify_bindings(ancestor["files"])
    report = {"status": "STOPPED_AFTER_TRANSPORT_FAILURE" if stopped else "COMPLETE",
              "model": r.MODEL, "cases_requested": len(CASES), "cases_attempted": len(rows),
              "reviews_completed": sum(row["review_complete"] for row in rows),
              "expected_narrow_observations_found": sum(row["expected_narrow_observation_found"] is True for row in rows),
              "http_attempts": caller.http_counter, "rows": rows, "new_dialogue_turns": 0,
              "new_event_allocations": 0, "usable_expansion_roots": 0,
              "question_calls": 0, "solver_calls": 0, "training_started": False,
              "semantic_certified": False, "preservation_bindings_verified": original["protected_count"],
              "v91_preservation_bindings_verified": ancestor["protected_count"],
              "limitations": "Diagnostic re-review of selected previously inspected failures/HOLDs, not blind root selection, dialogue generation, acceptance-rate benchmark or proof of stable expansion."}
    r.base.write(RUN / "REPORT.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
