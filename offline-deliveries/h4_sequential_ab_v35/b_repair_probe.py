"""Five frozen B regressions, at most eight logical / sixteen HTTP calls.

The S02 scanner and accepted S03 dialogue receive fresh independent judgments.
The historical S02 review receives one exact local format patch, retaining its
negative judgments. No source allocation, dialogue, QA, Solver or training runs.
prepare is offline; audit replays immutable inputs, requests, merges and counts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import runner as r

DELIVERY_BASE = r.HERE.parent
ORIGIN = r.HERE.parent / "h4_sequential_ab_v32/runs/inherited_a2_b3_01"
V33_PROBE = r.HERE.parent / "h4_sequential_ab_v33/probes/frozen_b_repair_01"
V28_ORIGIN = r.HERE.parent / "h4_sequential_ab_v28/runs/inherited_a2_b3_01"
VERSION = "v34_frozen_parent_goal_and_local_repair_regressions"
CASES = (
    ("event_s02_card_replacement_old_goal", "015", "event", "REJECT", None, 2),
    ("review_s03_accepted_dialogue_control", "024", "review", "PASS", "023", 2),
    ("repair_s02_negative_review_preserved", "017", "review_repair", "REJECT", "016", 1),
    ("repair_s03_missing_history_goal_group", "v33_s03", "review_repair", "PASS", None, 1),
    ("event_s01_same_theme_independent_control", "v28_s01", "event", "PASS", None, 2),
)
REPAIR_MODE = "exact_leaf_patch_v34"
LOGICAL_MAX = sum(x[5] for x in CASES)


def _decode(payload, marker):
    content = payload["messages"][0]["content"]
    r.base.require(content.count(marker) == 1, "historical_prompt_marker_invalid")
    return json.JSONDecoder().raw_decode(content.split(marker, 1)[1])[0]


def _directory(prefix, origin=ORIGIN):
    found = list((origin / "requests").glob(prefix + "-*"))
    r.base.require(len(found) == 1, "historical_request_not_unique:" + prefix)
    return found[0]


def _freeze(source, destination):
    source, destination = Path(source), Path(destination)
    data = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A shared source can belong to two cases; its first immutable copy is reused.
    if destination.exists():
        r.base.require(destination.read_bytes() == data, "frozen_copy_changed")
    else:
        with destination.open("xb") as out:
            out.write(data)
    return {"original_path": str(source), "frozen_path": str(destination),
            "sha256": r.base.digest(data)}


def _load_case(spec, cards):
    name, prefix, kind, expected, realization_prefix, limit = spec
    if prefix == "v33_s03":
        original_case_path = V33_PROBE / "cases/review_s03_accepted_dialogue_control.json"
        original_case = r.base.read(original_case_path)
        result = r.base.read(V33_PROBE / "results/review_s03_accepted_dialogue_control.json")
        r.base.require(len(result["requests"]) == 1, "frozen_v33_S03_request_inventory_changed")
        directory = Path(result["requests"][0])
        raw = r.base.read(directory / "RESPONSE.json")["text"]
        context = _decode(r.base.read(directory / "REQUEST.json"), "\nREVIEW INPUT DATA:\n")
        r.base.require("history_goal_checks" not in r.parse_object(raw)["route_fulfillment"],
                       "frozen_v33_S03_missing_group_not_reproduced")
        inputs = deepcopy(original_case["inputs"])
        r.base.require(context["branch_history"] == inputs["history"], "v33_frozen_history_changed")
        case = {"name": name, "kind": kind, "expected_status": expected, "logical_call_max": limit,
                "origin_request": str(directory), "original_context": context, "inputs": inputs,
                "previous_raw": raw, "original_history_sha256": r.base.digest(inputs["history"])}
        paths = [Path(row["original_path"]) for row in original_case["source_bindings"]]
        paths += [original_case_path, directory / "REQUEST.json", directory / "RESPONSE.json",
                  directory / "EVALUATION.json", V33_PROBE / "MANIFEST.json", V33_PROBE / "REPORT.json",
                  V33_PROBE / "B_REPAIR_AUDIT.json"]
        return case, sorted(set(paths))
    origin = V28_ORIGIN if prefix == "v28_s01" else ORIGIN
    prefix = "008" if prefix == "v28_s01" else prefix
    if origin != ORIGIN:
        cards = {row["card_id"]: row for row in r.base.read(origin / "dataset_cards_result.json")["accepted_cards"]}
    directory = _directory(prefix, origin)
    payload = r.base.read(directory / "REQUEST.json")
    marker = "\nEVENT_STATE_INPUT_JSON\n" if kind == "event" else "\nREVIEW INPUT DATA:\n"
    context = _decode(payload, marker)
    root, scene_id = context["root_id"], context["scene_id"]
    folder = origin / "roots" / root / "B"
    plan = r.base.read(folder / (scene_id + ".PLAN.json"))["scenes"]
    r.base.require(len(plan) == 1 and plan[0]["scene_id"] == scene_id, "historical_scene_plan_invalid")
    scene = plan[0]
    purpose = r.base.read(folder / (scene_id + ".PURPOSE.json"))
    latest = r.base.read(folder / (scene_id + ".LATEST_STATE.json"))
    event = r.base.read(folder / (scene_id + ".EVENT_STATE.json"))
    r.base.require(context["purpose_input"] == purpose and context["latest_state"] == latest,
                   "historical_context_root_artifact_mismatch")
    history = deepcopy(context["cumulative_history" if kind == "event" else "branch_history"])
    card = cards[scene["dataset_card_id"]]
    projection = (r.event_contract.scan_plan_context(scene) if kind == "event"
                  else r.realization_review.review_plan_context(scene))
    r.base.require(context["scene_plan"] == projection or context["scene_plan"] == scene,
                   "historical_plan_projection_mismatch")
    inputs = {"root_id": root, "route": context["route"], "scene_plan": scene,
              "history": history, "dataset_card": card, "latest_state": latest,
              "purpose_input": purpose, "event_state": event}
    paths = [directory / "REQUEST.json", directory / "RESPONSE.json",
             folder / (scene_id + ".PLAN.json"), folder / (scene_id + ".PURPOSE.json"),
             folder / (scene_id + ".LATEST_STATE.json"), folder / (scene_id + ".EVENT_STATE.json")]
    if origin != ORIGIN:
        paths += [origin / name for name in ("dataset_cards_result.json", "MANIFEST.json", "REPORT.json", "FINAL_AUDIT.json")]
    if kind != "event":
        rd = _directory(realization_prefix, origin)
        realization = r.base.validate_realization(
            r.parse_object(r.base.read(rd / "RESPONSE.json")["text"]), root, context["route"], scene)
        r.base.require(r.realization_review.review_scene(realization) == context["new_scene"],
                       "original_realization_projection_mismatch")
        r.base.require(r.realization_review.review_event_context(event) == context["event_state"],
                       "original_event_projection_mismatch")
        inputs.update(realization=realization,
                      other_turns=deepcopy(context["other_branch_added_turns_for_review_only"]))
        paths += [rd / "REQUEST.json", rd / "RESPONSE.json"]
    case = {"name": name, "kind": kind, "expected_status": expected, "logical_call_max": limit,
            "origin_request": str(directory), "original_context": context, "inputs": inputs,
            "previous_raw": r.base.read(directory / "RESPONSE.json")["text"],
            "original_history_sha256": r.base.digest(history)}
    return case, paths


def _make_request(case, *, repair=None):
    c = case["inputs"]
    if case["kind"] == "event":
        payload = r.event_contract.build_request(c["root_id"], c["route"], c["scene_plan"], c["history"],
            [c["dataset_card"]], c["latest_state"], repair=repair, purpose_input=c["purpose_input"],
            require_goal_checks=True, require_card_checks=True, require_parent_goal_checks=True)
        projected = r.event_contract.scan_plan_context(c["scene_plan"])
        sent = _decode(payload, "\nEVENT_STATE_INPUT_JSON\n")
        r.base.require(sent["scene_plan"] == projected and sent["cumulative_history"] == c["history"],
                       "scanner_complete_input_changed")
        proof = {"root_id": c["root_id"], "route": c["route"], "scene_id": c["scene_plan"]["scene_id"],
                 "role": "frozen_B_event_regression", "require_goal_checks": True,
                 "require_card_checks": True, "require_parent_goal_checks": True,
                 "projected_scene_plan_sha256": r.base.digest(projected)}
    else:
        payload, proof = r.review_request(c["root_id"], c["route"], c["scene_plan"], c["history"],
            c["realization"], c["other_turns"], card=c["dataset_card"], repair=repair,
            latest_state=c["latest_state"], event_state=c["event_state"], purpose_input=c["purpose_input"])
    proof.update(probe_version=VERSION, regression_case=case["name"], kind=case["kind"],
                 history_sha256=r.base.digest(c["history"]), scene_plan_sha256=r.base.digest(c["scene_plan"]),
                 frozen_inputs_sha256=r.base.digest(c), expected_verdict_withheld_from_model=True,
                 prior_response_withheld_from_initial_judgment=case["kind"] != "review_repair",
                 schema_repair_only=repair is not None, repair_spec_sha256=r.base.digest(repair))
    return payload, proof


def _evaluate(case, candidate):
    c = case["inputs"]
    if case["kind"] == "event":
        try:
            checked = r.event_contract.validate(candidate, c["root_id"], c["route"], c["scene_plan"],
                c["history"], require_goal_checks=True, dataset_cards=[c["dataset_card"]],
                require_card_checks=True, require_parent_goal_checks=True)
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
                require_goal_checks=True, dataset_cards=[c["dataset_card"]], require_card_checks=True,
                require_parent_goal_checks=True)
        except (ValueError, TypeError) as error:
            return r.event_contract.build_repair(error, candidate, history=c["history"])
        raise ValueError("event_schema_error_did_not_reproduce")
    repair = r.realization_review.build_format_repair(checked, candidate, c["scene_plan"], c["history"],
        c["realization"], dataset_card=c["dataset_card"], latest_state=c["latest_state"],
        event_state=c["event_state"], require_actor_checks=True, require_goal_checks=True, mode=REPAIR_MODE)
    # Unsupported diagnostics cannot justify a speculative full re-review or
    # consume a provider call whose patch is guaranteed to fail closed.
    r.base.require(not repair.get("unrepairable_failures"), "review_format_repair_has_unrepairable_failures")
    return repair


def _initial(case):
    if case["kind"] != "review_repair":
        return None, None
    previous = case["previous_raw"]
    return _format_repair(case, previous, _evaluate(case, previous)), previous


def _merge(case, previous, raw, repair):
    if case["kind"] == "event":
        merged = (r.event_contract.merge_event_repair(previous, raw, repair, case["inputs"]["history"])
                  if repair.get("repair_mode") == "timeline_entry_patch" else raw)
        r.event_contract.validate_repair_preservation(previous, merged, repair)
        return merged
    return r.realization_review.merge_review_repair(previous, raw, repair)


def _negative_card_preserved(case, parsed):
    original = r.parse_object(case["previous_raw"])
    negative = [row for row in original.get("card_checks", []) if row.get("realized") is False]
    new = {(row.get("kind"), row.get("id")): row for row in parsed.get("card_checks", [])}
    if not negative:
        return False
    for old in negative:
        current = new.get((old.get("kind"), old.get("id")), {})
        if current.get("realized") is not False or current.get("evidence") != old.get("evidence"):
            return False
        old_actors, new_actors = old.get("actor_checks", []), current.get("actor_checks", [])
        if len(old_actors) != len(new_actors):
            return False
        for before, after in zip(old_actors, new_actors):
            if before.get("satisfied") is False and after.get("satisfied") is not False:
                return False
    return True


def _expected_match(case, checked):
    if checked.get("schema_valid") is not True or checked.get("status") != case["expected_status"]:
        return False
    if case["kind"] == "event" and case["expected_status"] == "REJECT":
        validated = checked["validated"]
        card_rejection = any(row.get("verdict") == "revise" for row in validated.get("card_plan_checks", []))
        role = validated.get("route_fulfillment", {})
        goal_rejection = any(row.get("relation") == "serves_existing_goal"
                             for row in role.get("history_goal_checks", []))
        global_rejection = (role.get("relation_to_history") in ("old_event_substep", "continuation")
                            and role.get("serves_existing_event") is True)
        parent_rejection = any(row.get("functional_relation") in ("executes_or_prepares", "verifies_or_follows_up")
                               for row in validated.get("parent_goal_checks", []))
        return card_rejection and (goal_rejection or global_rejection or parent_rejection)
    if case["kind"] == "review_repair":
        if case["name"] == "repair_s03_missing_history_goal_group":
            old = _evaluate(case, case["previous_raw"]).get("parsed") or {}
            updated = deepcopy(checked.get("parsed") or {})
            checks = updated.get("route_fulfillment", {}).pop("history_goal_checks", None)
            if "satisfied" not in old.get("route_fulfillment", {}):
                # Successful schema validation compiles this derived field;
                # it is not a newly authored or repaired judgment.
                updated.get("route_fulfillment", {}).pop("satisfied", None)
            return bool(checks) and updated == old
        return _negative_card_preserved(case, checked.get("parsed") or {})
    return True


def _summary(case, checked, requests, *, transport_error=None, repair_error=None):
    return {"case": case["name"], "kind": case["kind"], "expected_status": case["expected_status"],
            "status": checked.get("status", "ERROR"), "schema_valid": checked.get("schema_valid", False),
            "expected_match": _expected_match(case, checked), "requests": requests,
            "logical_calls": len(requests), "transport_error": transport_error, "repair_error": repair_error,
            "history_turns": len(case["inputs"]["history"]),
            "history_sha256": r.base.digest(case["inputs"]["history"]),
            "result_sha256": r.base.digest(checked), "semantic_certified": False}


def _parent_provenance(case, candidate, checked):
    if case["kind"] != "event" or checked.get("schema_valid") is not True:
        return None
    return r.event_contract.parent_goal_decision_provenance(r.parse_object(candidate), case["inputs"]["history"])


def prepare(run):
    r.base.require(not run.exists(), "probe_already_exists")
    r.base.require(run.parent == r.HERE / "probes", "B_repair_probe_must_be_in_v34_probes")
    r.base.require(r.MIN_HTTP_GAP_SECONDS >= 35, "probe_gap_below_existing_policy")
    run.mkdir(parents=True)
    cards_file = ORIGIN / "dataset_cards_result.json"
    cards = {row["card_id"]: row for row in r.base.read(cards_file)["accepted_cards"]}
    sources = [_freeze(cards_file, run / "frozen_origin/dataset_cards_result.json")]
    # Preserve original transport counts and audit results, without promoting
    # any historical verdict to the fresh initial judgment inputs.
    for filename in ("MANIFEST.json", "REPORT.json", "FINAL_AUDIT.json", "provider_profile.json",
                     "PURPOSE_FEEDBACK_PROJECTION.json", "INHERITED_A.json"):
        sources.append(_freeze(ORIGIN / filename, run / "frozen_origin" / filename))
    cases = []
    for spec in CASES:
        case, paths = _load_case(spec, cards)
        receipts = [_freeze(path, (run / "frozen_origin" / path.relative_to(ORIGIN)
                    if path.is_relative_to(ORIGIN) else
                    run / "frozen_sources" / path.relative_to(DELIVERY_BASE))) for path in paths]
        case["source_bindings"] = receipts
        sources.extend(row for row in receipts if row not in sources)
        repair, previous = _initial(case)
        case["initial_repair"] = repair
        case["initial_previous_sha256"] = r.base.digest(previous)
        case_path = run / "cases" / (case["name"] + ".json")
        r.base.write(case_path, case)
        payload, proof = _make_request(case, repair=repair)
        r.base.write(run / "previews" / (case["name"] + ".json"), {"payload": payload, "proof": proof})
        cases.append(str(case_path))
    profile, transport, tokenizer = r.transport_inputs()
    r.base.write(run / "provider_profile.json", profile)
    # Freeze imported runtime code, including the legacy runner loaded through
    # importlib rather than sys.modules. Unused tests and future wrappers are not
    # dependencies of this bounded probe and do not invalidate it when added.
    code = {Path(__file__).resolve(), Path(r.__file__).resolve(), Path(r.base.__file__).resolve(),
            *map(Path, transport)}
    runtime_roots = {Path(r.__file__).resolve().parent, r.LEGACY.resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and any(path.is_relative_to(root) for root in runtime_roots):
                code.add(path)
    code = sorted(code)
    snapshots = [_freeze(path, run / "execution_code" / f"{i:03d}_{path.name}") for i, path in enumerate(code)]
    r.base.write(run / "EXECUTION_CODE_SNAPSHOT.json", {"files": snapshots})
    paths = code + list(tokenizer) + [r.HERE / "POLICY.md"]
    paths += [Path(row[key]) for row in sources for key in ("original_path", "frozen_path")]
    paths += list(run.rglob("*.json")) + [Path(row["frozen_path"]) for row in snapshots]
    manifest = {"version": VERSION, "model": r.MODEL, "review_model": r.MODEL,
        "prepared_at_utc": r.now(), "origin_run": str(ORIGIN), "cases": cases,
        "logical_call_max": LOGICAL_MAX, "case_logical_limits": {spec[0]: spec[5] for spec in CASES},
        "http_max": LOGICAL_MAX * (r.TRANSPORT_RETRIES + 1),
        "automatic_transport_retries": r.TRANSPORT_RETRIES,
        "minimum_http_gap_after_response_seconds": r.MIN_HTTP_GAP_SECONDS,
        "source_events_reassigned": False, "historical_records_modified": False,
        "new_scene_generation": False, "question_calls": 0, "solver_calls": 0, "training_calls": 0,
        "format_repairs_only": True, "initial_judgments_exclude_expected_and_prior_verdicts": True,
        "case_origin_requests": {r.base.read(path)["name"]: r.base.read(path)["origin_request"] for path in cases},
        "historical_origins": [str(ORIGIN), str(V28_ORIGIN), str(V33_PROBE)],
        "original_run_http_attempts": len(list((ORIGIN / "requests").glob("*/HTTP_ATTEMPT*.json"))),
        "v28_origin_http_attempts": len(list((V28_ORIGIN / "requests").glob("*/HTTP_ATTEMPT*.json"))),
        "v33_probe_http_attempts": len(list((V33_PROBE / "requests").glob("*/HTTP_ATTEMPT*.json"))),
        "files": r.base.bindings(sorted(set(paths)))}
    original_report = r.base.read(run / "frozen_origin/REPORT.json")
    original_audit = r.base.read(run / "frozen_origin/FINAL_AUDIT.json")
    r.base.require(original_report["http_attempts"] == original_audit["http_attempts"]
                   == manifest["original_run_http_attempts"], "historical_http_count_mismatch")
    r.base.write(run / "MANIFEST.json", manifest)
    return {"status": "PREPARED", "cases": len(cases), "logical_call_max": LOGICAL_MAX,
            "http_max": manifest["http_max"], "http_calls": 0, "new_scenes": 0}


def verify(run):
    manifest = r.base.read(run / "MANIFEST.json")
    r.base.require(manifest["version"] == VERSION and manifest["model"] == r.MODEL, "wrong_probe_version_or_model")
    r.base.require(manifest["logical_call_max"] == LOGICAL_MAX and manifest["case_logical_limits"] ==
                   {spec[0]: spec[5] for spec in CASES}, "logical_budget_changed")
    r.base.require(manifest["http_max"] == LOGICAL_MAX * (r.TRANSPORT_RETRIES + 1), "http_budget_changed")
    r.base.require(manifest["minimum_http_gap_after_response_seconds"] == r.MIN_HTTP_GAP_SECONDS,
                   "probe_http_gap_changed")
    r.base.verify_bindings(manifest["files"])
    for row in r.base.read(run / "EXECUTION_CODE_SNAPSHOT.json")["files"]:
        r.base.require(r.base.digest(Path(row["frozen_path"]).read_bytes()) == row["sha256"], "execution_snapshot_changed")
    r.base.require([r.base.read(path)["name"] for path in manifest["cases"]] == [spec[0] for spec in CASES],
                   "case_inventory_changed")
    for path in manifest["cases"]:
        case = r.base.read(path)
        history = case["inputs"]["history"]
        original = case["original_context"]["cumulative_history" if case["kind"] == "event" else "branch_history"]
        r.base.require(history == original and r.base.digest(history) == case["original_history_sha256"],
                       "complete_case_history_changed")
        repair, previous = _initial(case)
        r.base.require(repair == case["initial_repair"] and r.base.digest(previous) == case["initial_previous_sha256"],
                       "initial_repair_replay_mismatch")
        payload, proof = _make_request(case, repair=repair)
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
        repair, previous = _initial(case)
        requests, checked, transport_error, repair_error = [], {}, None, None
        for attempt in range(1, case["logical_call_max"] + 1):
            payload, proof = _make_request(case, repair=repair)
            proof.update(logical_attempt=attempt, previous_candidate_sha256=r.base.digest(previous))
            before = caller.request_counter
            label = f"B-repair-{case['name']}-{attempt}"
            try:
                raw, directory = caller.call(label, payload, proof)
            except (RuntimeError, ValueError) as error:
                transport_error = str(error)
                if caller.request_counter > before:
                    requests.append(str(run / "requests" / f"{caller.request_counter:03d}-{label}"))
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
            provenance = _parent_provenance(case, candidate, checked)
            if provenance is not None:
                r.base.write(directory / "PARENT_GOAL_DECISION_PROVENANCE.json", provenance)
            if checked.get("schema_valid") is not False:
                break
            if attempt < case["logical_call_max"]:
                try:
                    repair = _format_repair(case, candidate, checked)
                except (ValueError, TypeError) as error:
                    repair_error = str(error)
                    r.base.write(directory / "FORMAT_REPAIR_ERROR.json", {"error": repair_error})
                    break
                previous = candidate
                r.base.write(directory / "FORMAT_REPAIR_SPEC.json", repair)
        row = _summary(case, checked, requests, transport_error=transport_error, repair_error=repair_error)
        rows.append(row)
        r.base.write(run / "results" / (case["name"] + ".json"), row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    verify(run)
    report = {"status": "PASS" if all(row["expected_match"] for row in rows) else "REGRESSION_NOT_ESTABLISHED",
              "cases": rows, "logical_calls": caller.request_counter, "http_attempts": caller.http_counter,
              "new_scenes": 0, "question_calls": 0, "solver_calls": 0, "training_calls": 0,
              "semantic_certified": False, "completed_at_utc": r.now()}
    r.base.write(run / "REPORT.json", report)
    return report


def audit(run):
    manifest = verify(run)
    failures, rows, inventory = [], [], []
    def check(condition, code):
        if not condition:
            failures.append(code)
    for path in manifest["cases"]:
        case = r.base.read(path)
        result_path = run / "results" / (case["name"] + ".json")
        if not result_path.exists():
            rows.append({"case": case["name"], "status": "NO_LIVE_RESULT", "expected_match": False})
            check(not (run / "LIVE_STARTED.json").exists(), "case_result_missing:" + case["name"])
            continue
        row = r.base.read(result_path)
        directories = [Path(p) for p in row["requests"]]
        check(1 <= len(directories) <= case["logical_call_max"], "case_request_budget_invalid:" + case["name"])
        repair, previous = _initial(case)
        checked, repair_error = {}, None
        for attempt, directory in enumerate(directories, 1):
            inventory.append(directory.resolve())
            check(directory.resolve().parent == (run / "requests").resolve(), "request_outside_probe")
            payload, proof = _make_request(case, repair=repair)
            proof.update(logical_attempt=attempt, previous_candidate_sha256=r.base.digest(previous))
            check(r.base.read(directory / "REQUEST.json") == payload, "request_replay_mismatch:" + directory.name)
            if not (directory / "RESPONSE.json").exists():
                check((directory / "ERROR.json").exists() or (directory / "LOCAL_ERROR.json").exists(),
                      "missing_transport_or_local_error:" + directory.name)
                checked = {"status": "ERROR", "schema_valid": False, "error": row["transport_error"],
                           "semantic_certified": False}
                check(attempt == len(directories), "request_after_transport_failure")
                break
            actual_proof = r.base.read(directory / "PROOF.json")
            check(all(actual_proof.get(k) == v for k, v in proof.items()), "proof_replay_mismatch:" + directory.name)
            check(actual_proof.get("payload_sha256") == r.base.digest(payload), "proof_payload_hash_mismatch:" + directory.name)
            raw = r.base.read(directory / "RESPONSE.json")["text"]
            check(r.base.read(directory / "REGRESSION_STEP.json") == {
                "case": case["name"], "logical_attempt": attempt, "repair": repair,
                "previous_candidate_sha256": r.base.digest(previous)}, "regression_step_mismatch:" + directory.name)
            try:
                candidate = raw if repair is None else _merge(case, previous, raw, repair)
            except (ValueError, TypeError) as error:
                checked = {"status": "FORMAT_ERROR", "schema_valid": False, "error": str(error), "semantic_certified": False}
                check(r.base.read(directory / "EVALUATION.json") == checked, "merge_failure_replay_mismatch:" + directory.name)
                check(attempt == len(directories), "request_after_failed_merge")
                break
            check(r.base.read(directory / "CANDIDATE.json") == candidate, "candidate_replay_mismatch:" + directory.name)
            checked = _evaluate(case, candidate)
            check(r.base.read(directory / "EVALUATION.json") == checked, "evaluation_replay_mismatch:" + directory.name)
            provenance = _parent_provenance(case, candidate, checked)
            if provenance is not None:
                check(r.base.read(directory / "PARENT_GOAL_DECISION_PROVENANCE.json") == provenance,
                      "parent_goal_provenance_replay_mismatch:" + directory.name)
            if checked.get("schema_valid") is not False:
                check(attempt == len(directories), "semantic_result_retried:" + directory.name)
                break
            if attempt < case["logical_call_max"]:
                try:
                    next_repair = _format_repair(case, candidate, checked)
                except (ValueError, TypeError) as error:
                    repair_error = str(error)
                    check(r.base.read(directory / "FORMAT_REPAIR_ERROR.json") == {"error": repair_error},
                          "format_repair_error_replay_mismatch")
                    check(attempt == len(directories), "request_after_repair_spec_error")
                    break
                check(r.base.read(directory / "FORMAT_REPAIR_SPEC.json") == next_repair, "repair_spec_replay_mismatch")
                repair, previous = next_repair, candidate
        replay_row = _summary(case, checked, row["requests"], transport_error=row.get("transport_error"), repair_error=repair_error)
        check(replay_row == row, "case_result_replay_mismatch:" + case["name"])
        rows.append(replay_row)
    actual = sorted(path.resolve() for path in (run / "requests").glob("*") if path.is_dir())
    check(sorted(inventory) == actual, "unexpected_or_missing_request_directory")
    check(len(actual) <= manifest["logical_call_max"], "logical_budget_exceeded")
    attempts = sorted((run / "requests").glob("*/HTTP_ATTEMPT*.json"))
    check(len(attempts) <= manifest["http_max"], "http_budget_exceeded")
    global_numbers = [r.base.read(path).get("http_attempt_number") for path in attempts]
    check(all(type(number) is int for number in global_numbers)
          and sorted(global_numbers) == list(range(1, len(attempts) + 1)),
          "global_http_attempt_sequence_invalid")
    for directory in actual:
        payload = r.base.read(directory / "REQUEST.json")
        http = sorted(directory.glob("HTTP_ATTEMPT*.json"))
        check(len(http) <= r.TRANSPORT_RETRIES + 1, "transport_retry_budget_exceeded:" + directory.name)
        check(sorted(r.base.read(path).get("request_attempt", 0) for path in http) == list(range(1, len(http) + 1)),
              "request_http_attempt_sequence_invalid:" + directory.name)
        for path in http:
            entry = r.base.read(path)
            check(entry.get("payload_sha256") == r.base.digest(payload)
                  and entry.get("actual_payload_equals_frozen") is True and entry.get("requested_model") == r.MODEL,
                  "http_payload_or_model_mismatch:" + str(path))
            number = entry.get("request_attempt", 1)
            throttle = r.base.read(directory / f"THROTTLE_{number:02d}.json")
            check(throttle.get("minimum_gap_after_previous_response_seconds") == r.MIN_HTTP_GAP_SECONDS
                  and type(throttle.get("waited_seconds")) in (int, float)
                  and throttle["waited_seconds"] >= 0 and isinstance(throttle.get("http_started_at_utc"), str),
                  "http_throttle_record_invalid:" + directory.name)
            if number > 1:
                previous_result = r.base.read(directory / f"HTTP_RESULT_{number-1:02d}.json")
                check(entry.get("same_payload_retry") is True and previous_result.get("outcome") == "error"
                      and previous_result.get("http_status") in (503, 504), "transport_retry_policy_mismatch:" + directory.name)
        if (directory / "RESPONSE.json").exists():
            response = r.base.read(directory / "RESPONSE.json")
            metadata = response.get("metadata", {})
            check(metadata.get("returned_model") == r.MODEL and metadata.get("finish_reason") == "stop",
                  "response_model_or_finish_mismatch:" + directory.name)
            check(bool(http), "response_without_http_attempt:" + directory.name)
            final_response_path = directory / f"HTTP_RESPONSE_{len(http):02d}.json"
            final_result_path = directory / f"HTTP_RESULT_{len(http):02d}.json"
            check(final_response_path.exists() and final_result_path.exists(),
                  "response_http_artifact_missing:" + directory.name)
            if final_response_path.exists() and final_result_path.exists():
                final_response, final_result = r.base.read(final_response_path), r.base.read(final_result_path)
                check(response.get("text") == final_response.get("text")
                      and metadata == final_response.get("metadata"), "response_not_final_http_body:" + directory.name)
                check(final_result.get("outcome") == "response" and final_result.get("http_status") in (None, 200),
                      "final_http_result_not_response:" + directory.name)
    if (run / "REPORT.json").exists():
        report = r.base.read(run / "REPORT.json")
        check(report.get("cases") == rows, "report_case_replay_mismatch")
        expected_status = "PASS" if all(row.get("expected_match") for row in rows) else "REGRESSION_NOT_ESTABLISHED"
        check(report.get("status") == expected_status, "report_regression_status_mismatch")
        check(report.get("http_attempts") == len(attempts) and report.get("logical_calls") == len(actual),
              "report_call_count_mismatch")
        for key in ("new_scenes", "question_calls", "solver_calls", "training_calls"):
            check(report.get(key) == 0, "forbidden_generation_count:" + key)
    result = {"status": "PASS" if not failures else "FAIL", "run": str(run), "failures": failures,
              "regression_status": "PREPARED_NO_LIVE_CALLS" if not actual else (
                  "PASS" if len(rows) == len(CASES) and all(row.get("expected_match") for row in rows)
                  else "REGRESSION_NOT_ESTABLISHED"),
              "cases": rows, "logical_calls": len(actual), "http_attempts": len(attempts),
              "original_run_http_attempts": manifest["original_run_http_attempts"],
              "v28_origin_http_attempts": manifest["v28_origin_http_attempts"],
              "v33_probe_http_attempts": manifest["v33_probe_http_attempts"],
              "new_scenes": 0, "question_calls": 0, "solver_calls": 0, "training_calls": 0,
              "semantic_certified": False, "audited_at_utc": r.now()}
    output, index = run / "B_REPAIR_AUDIT.json", 2
    while output.exists():
        output = run / f"B_REPAIR_AUDIT_{index:03d}.json"
        index += 1
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
