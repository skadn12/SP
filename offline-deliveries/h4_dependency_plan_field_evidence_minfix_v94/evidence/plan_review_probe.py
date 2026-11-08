"""Bounded diagnostic of unchanged v91/v93 plans; allocates no source events."""
from pathlib import Path
import json
import sys

sys.dont_write_bytecode = True
DELIVERY = Path(__file__).resolve().parents[1]
ENGINE = DELIVERY / "expansion_engine"
sys.path.insert(0, str(ENGINE))
import runner as r

RUN = DELIVERY / "evidence/model_plan_review_probe"
ARCHIVES = {
    "v91": DELIVERY.parent / "h4_planner_cardinality_expansion_batch_v91/expansion_engine/runs/ten_roots_B_scene01",
    "v93": DELIVERY.parent / "h4_review_reference_minfix_v93/expansion_engine/runs/four_other_roots_B_scene01",
}
CASES = (
    ("v93", 14304, "007-", "card_action"),
    ("v93", 20432, "011-", "history_new_event"),
    ("v93", 25652, "016-", "format_only"),
    ("v93", 38601, "020-", "card_action"),
    ("v91", 20432, "037-", "history_new_event"),
    ("v91", 38601, None, "metadata_identity"),
    ("v91", 194259, None, "card_action"),
    ("v91", 19053, None, "card_action"),
)


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": r.base.digest(path.read_bytes())}


def case_inputs(version, number, prefix, observation):
    old = ARCHIVES[version]
    root = f"cc-train-episode-{number}"
    cards_path, allocation_path = old / "dataset_cards_result.json", old / "UNIQUE_EVENT_ALLOCATION.json"
    allocation = r.base.read(allocation_path)
    assigned = allocation["assignments"][root]
    r.base.require(len(assigned) == 1, "diagnostic_original_single_slot_required")
    by_source = {card["source_id"]: card for card in r.base.read(cards_path)["accepted_cards"]}
    card = r.base.dataset_cards.build_realization_guidance(by_source[assigned[0]["source_id"]])
    source = old / "sources" / (root + ".json")
    state_path = old / "roots" / root / "B/S01.LATEST_STATE.json"
    if prefix is None:
        plan_path = old / "roots" / root / "B/S01.PLAN.json"
        plan = r.base.read(plan_path)
    else:
        plan_path = next((old / "requests").glob(prefix + "*")) / "RESPONSE.json"
        plan = r.parse_object(r.base.read(plan_path)["text"])
    return {"case_id": f"{version}_root{number}", "version": version, "root_id": root,
            "history": r.base.read(source)["turns"], "state": r.base.read(state_path),
            "scene": plan["scenes"][0], "card": card, "observation": observation,
            "input_bindings": [artifact(path) for path in (cards_path, allocation_path, source, state_path, plan_path)]}


def expected_observation(case, result):
    """Selected, narrow observations; never a semantic acceptance certificate."""
    kind = case["observation"]
    if kind == "format_only":
        return None
    if kind == "history_new_event":
        return all(check["verdict"] == "compatible" for check in result["plan_checks"])
    findings = r.event_contract.plan_diagnostics(result)
    if kind == "card_action":
        return any(item["check_kind"] in ("card_step", "card_dependency") and item["requires_replan"]
                   for item in findings)
    return any(item["check_kind"] == "plan_consistency" and
               item.get("category") == "participant_identity" and item["requires_replan"] for item in findings)


def protected_inputs():
    paths = [DELIVERY / "PRESERVATION_BEFORE.json",
             DELIVERY.parent / "h4_review_reference_minfix_v93/PRESERVATION_BEFORE.json",
             DELIVERY.parent / "h4_card_fidelity_event_consistency_minfix_v92/PRESERVATION_BEFORE.json"]
    protection = [r.base.read(path) for path in paths]
    for item in protection:
        r.base.verify_bindings(item["files"])
    return paths, protection


def run():
    r.base.require(not RUN.exists(), "diagnostic_probe_already_started_no_replay")
    cases = [case_inputs(*case) for case in CASES]
    paths, protection = protected_inputs()
    profile, transport, tokenizer = r.transport_inputs()
    files = [Path(__file__), DELIVERY / "FIX_PLAN.json", ENGINE / "POLICY.md", *paths,
             *ENGINE.glob("*.py"), *r.LEGACY.glob("*.py"), *transport, *tokenizer]
    files += [Path(row["path"]) for case in cases for row in case["input_bindings"]]
    manifest = {"version": "v94_original_plan_review_probe", "model": r.MODEL,
                "http_max": 32, "logical_case_count": 8, "attempts_per_case": 2,
                "new_event_allocations": 0, "new_dialogue_turns": 0, "question_calls": 0,
                "solver_calls": 0, "training_started": False, "semantic_certified": False,
                "old_verdicts_or_expected_observations_in_model_requests": False,
                "files": r.base.bindings(files),
                "cases": [{"case_id": case["case_id"], "input_bindings": case["input_bindings"]}
                          for case in cases]}
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
    rows, stopped = [], False
    for case in cases:
        repair, result = None, None
        attempts = []
        for attempt in range(1, 3):
            r.base.verify_bindings(manifest["files"])
            payload = r.event_contract.build_request(case["root_id"], r.ROUTES[1], case["scene"],
                case["history"], [case["card"]], case["state"], repair=repair)
            try:
                raw, directory = caller.call(f"{case['case_id']}-plan-review-{attempt}", payload, {
                    "role": "diagnostic_plan_reviewer", "root_id": case["root_id"], "case_id": case["case_id"],
                    "review_only": True, "new_event_allocation": False,
                    "prior_verdict_and_expected_observation_provided": False,
                    "input_bindings": case["input_bindings"]})
            except RuntimeError as error:
                attempts.append({"attempt": attempt, "status": "TRANSPORT_FAILURE", "reason": str(error)})
                stopped = True
                break
            try:
                result = r.event_contract.validate(raw, case["root_id"], r.ROUTES[1], case["scene"],
                                                  case["history"], cards=[case["card"]])
                validation = {"status": "VALID_REVIEW", "requires_replan": r.event_contract.requires_replan(result),
                              "diagnostics": r.event_contract.plan_diagnostics(result), "semantic_certified": False}
            except (ValueError, TypeError) as error:
                result = None
                repair = r.event_contract.build_repair(error, raw, history=case["history"])
                validation = {"status": "INVALID_REVIEW", "errors": repair["errors"], "semantic_certified": False}
            r.base.write(directory / "VALIDATION.json", validation)
            attempts.append({"attempt": attempt, "request": str(directory), "validation": validation,
                             "original_response": artifact(directory / "RESPONSE.json")})
            if result is not None:
                break
        row = {"case_id": case["case_id"], "root_id": case["root_id"], "attempts": attempts,
               "review_complete": result is not None,
               "observation_scope": case["observation"],
               "expected_narrow_observation_found": expected_observation(case, result) if result else None,
               "semantic_certified": False, "original_inputs_changed": False}
        rows.append(row)
        r.base.write(RUN / (case["case_id"] + "_RESULT.json"), row)
        print(json.dumps({key: value for key, value in row.items() if key != "attempts"}, ensure_ascii=False), flush=True)
        if stopped:
            break
    r.base.verify_bindings(manifest["files"])
    for item in protection:
        r.base.verify_bindings(item["files"])
    report = {"status": "STOPPED_TRANSPORT_FAILURE" if stopped else "COMPLETE",
              "cases_requested": len(cases), "cases_attempted": len(rows),
              "reviews_completed": sum(row["review_complete"] for row in rows),
              "narrow_observations_found": sum(row["expected_narrow_observation_found"] is True for row in rows),
              "http_attempts": caller.http_counter, "rows": rows,
              "prior_protected_file_counts": [item["protected_count"] for item in protection],
              "new_event_allocations": 0, "new_dialogue_turns": 0, "question_calls": 0,
              "solver_calls": 0, "training_started": False, "semantic_certified": False,
              "limitations": "Selected prior failures/HOLDs; format-only case and narrow semantic observations are distinct. This is not blind accuracy, new expansion or production readiness."}
    r.base.write(RUN / "REPORT.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
