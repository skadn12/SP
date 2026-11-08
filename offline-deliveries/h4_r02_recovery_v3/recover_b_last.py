"""Final bounded B-only recovery after the explicit-scene-budget delta."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("b_budget_parent", HERE / "recover.py")
parent_adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_adapter)
base, v2 = parent_adapter.base, parent_adapter.v2
unique = v2.unique_events
VERSION = "h4_r02_recovery_b_last_v1"
DEFAULT_PARENT = parent_adapter.DEFAULT_RUN
DEFAULT_PREVIOUS = v2.V2_RUNS / "recovery_b_budget_01"
DEFAULT_RUN = v2.V2_RUNS / "recovery_b_last_01"
COMBINED_HTTP_MAX = 22
ROUTE = base.ROUTES[1]

B_PLANNER = """Plan at most ONE natural dataset-guided scene after cumulative_history.
This recovery has a hard scene budget of 1 and exactly one allocated dataset card.
Return one scene if coherent, or zero scenes with a concrete reason. Do not add
old-event continuation scenes, setup scenes, filler scenes, or scenes with a null card.
The one scene will later have 12-16 short conversational turns.
Only use allowed_speakers as direct speakers. They may discuss other people without
inventing a prior relationship. Every scene needs at least one exact quoted anchor
from cumulative_history with its turn_ref. Anchors justify continuity; proposed new
developments are not claims that those facts already occurred.
Return exactly {root_id, route, scenes, shortfall_reason}. A scene has exactly:
scene_id (S01), actors (2 or more allowed speakers),
source_anchors (1-3 {turn_ref,quote}, each short quote copied exactly),
goal (short string), developments (EXACTLY 3 short concrete strings), dataset_card_id,
feedback_topic_ids (0-4 topic IDs copied from feedback, empty if none apply).
The previous planner failed because it returned 6 developments. The array must have
EXACTLY 3 elements. Combine related developments while planning; do not return 6 items.
route is dataset_new_event. dataset_card_id MUST equal the card_id of the single
supplied card; null is invalid. Introduce a new event fitting this root using that card.
A scaffold transfers abstract relationships/condition changes; a seed supplies only
an initial premise. Neither imports source people's identities, source places, or
source outcomes into this root. Adapt details and establish them in future dialogue.
shortfall_reason must be nonempty even for one scene: explain that this recovery is
limited to the one unused event assigned to this root. Do not try to reach three scenes.
"""


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(source.read_bytes())


def inventory():
    return unique.collect_prior_usage(unique.prior_runs(v2.V1_RUNS, v2.V2_RUNS))


def prepare(run, parent=DEFAULT_PARENT, previous=DEFAULT_PREVIOUS):
    base.require(run.parent == v2.V2_RUNS and not run.exists(), "invalid_or_existing_delta_run")
    base.require(parent.parent == v2.V2_RUNS and (parent / "RECOVERY_REPORT.json").is_file(),
                 "parent_recovery_not_completed")
    parent_adapter.verify(parent)
    base.require(previous.parent == v2.V2_RUNS and (previous / "REPORT.json").is_file(),
                 "previous_B_delta_not_completed")
    previous_report = base.read(previous / "REPORT.json")
    base.require(previous_report["parent_run"] == str(parent) and previous_report["combined_http_max"] == COMBINED_HTTP_MAX,
                 "previous_B_delta_parent_or_budget_mismatch")
    previous_manifest = base.read(previous / "DELTA_MANIFEST.json")
    base.verify_bindings(previous_manifest["files"])
    prior = inventory()
    parent_report = base.read(parent / "REPORT.json")
    source_allocation = base.read(parent / "UNIQUE_EVENT_ALLOCATION.json")
    routes = {row["root_id"]: row for row in parent_report["routes"] if row["route"] == ROUTE}
    base.require(set(routes) == set(base.ROOTS), "parent_B_routes_incomplete")
    base.require(all(row["accepted_scenes"] in (0, 1) for row in routes.values()), "unexpected_parent_B_count")
    previous_routes = {row["root_id"]: row for row in previous_report["routes"] if row["route"] == ROUTE}
    roots = [root for root in base.ROOTS if routes[root]["accepted_scenes"] == 0
             and root in previous_routes and previous_routes[root]["accepted_scenes"] == 0]
    base.require(roots == ["cc-train-episode-104814"], "final_recovery_must_target_only_104814")
    consumed = {unique.event_key(row) for row in prior["accepted_uses"]}
    assignments = {root: source_allocation["assignments"][root] for root in roots}
    base.require(all(len(rows) == 1 for rows in assignments.values()), "expected_one_original_assignment")
    assigned_keys = [unique.event_key(rows[0]) for rows in assignments.values()]
    base.require(len(set(assigned_keys)) == len(assigned_keys) and not (set(assigned_keys) & consumed),
                 "original_assignment_already_consumed_or_duplicate")
    parent_http = previous_report["combined_http_attempts"]
    budget = min(4 * len(roots), max(0, COMBINED_HTTP_MAX - parent_http))
    all_cards = base.read(parent / "dataset_cards_result.json")["accepted_cards"]
    cards = [card for card in all_cards if unique.event_key(card) in set(assigned_keys)]
    base.require(len(cards) == len(roots), "missing_original_card")
    for root in roots:
        unique.verify_cards_for_root([card for card in cards if unique.event_key(card) == unique.event_key(assignments[root][0])],
                                     {"assignments": assignments}, root)
    run.mkdir(parents=True)
    for name in ("provider_profile.json", "dataset_sources.json"):
        copy_file(parent / name, run / name)
    for path in sorted((parent / "feedback").rglob("*")):
        if path.is_file():
            copy_file(path, run / "feedback" / path.relative_to(parent / "feedback"))
    for root in roots:
        copy_file(parent / "sources" / f"{root}.json", run / "sources" / f"{root}.json")
        copy_file(parent / "roots" / root / "A" / "cumulative_dialogue.json", run / "parent_A" / f"{root}.json")
    base.write(run / "dataset_cards_result.json", {"accepted_cards": cards, "rejected_records": [],
               "reused_from": str(parent / "dataset_cards_result.json"), "new_extraction_calls": 0})
    frozen = [HERE / "recover_b_last.py", HERE / "recover_b_budget.py", HERE / "recover.py", v2.HERE / "runner.py", v2.HERE / "unique_events.py"]
    frozen += list(previous.rglob("*.json")) + list(previous.rglob("*.txt"))
    frozen += [Path(row["path"]) for row in base.read(parent / "manifest.json")["files"]]
    frozen += [Path(row["path"]) for row in prior["prior_evidence"]]
    frozen += list(run.rglob("*.json")) + list(run.rglob("*.txt"))
    frozen += [parent / "RECOVERY_REPORT.json", parent / "UNIQUE_EVENT_ALLOCATION.json"]
    for root in base.ROOTS:
        for letter in ("A", "B"):
            frozen += [parent / "roots" / root / letter / name for name in ("RESULT.json", "cumulative_dialogue.json")]
    base.write(run / "DELTA_MANIFEST.json", {
        "version": VERSION, "parent_run": str(parent), "previous_delta": str(previous), "roots": roots, "assignments": assignments,
        "prior": prior, "files": base.bindings(frozen), "scene_budget_per_root": 1,
        "new_http_max": budget, "combined_http_max": COMBINED_HTTP_MAX, "parent_http_attempts": parent_http,
        "original_parent_http_attempts": parent_report["http_attempts"],
        "normal_new_http_requests": 2 * len(roots), "per_root_http_max": 4,
        "automatic_transport_retries": 0, "planner_prompt": B_PLANNER,
        "only_delta_B_routes_registered": True, "successful_parent_A_and_B_referenced_only": True,
    })
    verify(run)
    return {"status": "PASS", "roots": roots, "http_calls": 0, "new_http_max": budget,
            "normal_new_http_requests": 2 * len(roots), "parent_http_attempts": parent_http}


def verify(run):
    manifest = base.read(run / "DELTA_MANIFEST.json")
    base.require(run.parent == v2.V2_RUNS and manifest["version"] == VERSION, "delta_contract_changed")
    base.require(manifest["planner_prompt"] == B_PLANNER and manifest["scene_budget_per_root"] == 1,
                 "delta_scene_budget_changed")
    base.require(manifest["combined_http_max"] == COMBINED_HTTP_MAX and
                 manifest["new_http_max"] == min(4 * len(manifest["roots"]),
                    max(0, COMBINED_HTTP_MAX - manifest["parent_http_attempts"])), "delta_http_budget_changed")
    base.verify_bindings(manifest["files"])
    return manifest


def make_request(run, root, role, history, cards, scene=None, repair=None):
    base.require(role in ("planner", "realizer") and len(cards) == 1, "invalid_B_request")
    source = base.read(run / "sources" / f"{root}.json")
    accepted_A = base.read(run / "parent_A" / f"{root}.json")["turns"]
    base.require(history == accepted_A and history[:len(source["turns"])] == source["turns"],
                 "B_must_receive_exact_parent_A_history")
    context = {"root_id": root, "route": ROUTE, "role": role, "mock_only_offline_preview": False,
               "allowed_speakers": sorted({turn["speaker"] for turn in source["turns"]}),
               "cumulative_history": history, "cumulative_history_sha256": base.digest(history),
               "dataset_cards": cards, "scene_plan": scene, "scene_budget": 1}
    if repair:
        context["mechanical_schema_repair"] = repair
    prompt = base.COMMON + (B_PLANNER if role == "planner" else base.REALIZER)
    prompt += "\nINPUT_JSON\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    payload = {"model": base.dataset_cards.MODEL, "temperature": .25, "max_tokens": 4096,
               "enable_thinking": False, "response_format": {"type": "json_object"},
               "messages": [{"role": "user", "content": prompt}]}
    identity = base.read(run / "feedback" / "COMPILED_FEEDBACK.json")["target_solver_identity_sha256"]
    payload, receipt = base.feedback.append_feedback_to_request(payload, run / "feedback", root, ROUTE, role,
        target_solver_model=base.TARGET, expected_target_identity_sha256=identity)
    return payload, {"root_id": root, "route": ROUTE, "role": role, "scene_budget": 1,
        "mock_only_offline_preview": False, "source_prefix_preserved": True,
        "history_turns": len(history), "history_sha256": base.digest(history),
        "history_refs": [turn["turn_ref"] for turn in history], "feedback_receipt": receipt,
        "payload_sha256": base.digest(payload), "exact_parent_A_history": True}


def validate_plan(raw, root, history, cards):
    parsed = base.parse_object(raw)
    base.require(isinstance(parsed.get("scenes"), list) and len(parsed["scenes"]) <= 1,
                 "B_scene_count_exceeds_explicit_budget_1")
    return base.validate_plan(parsed, root, ROUTE, history, cards)


class DeltaCaller(base.Caller):
    def __init__(self, run, manifest):
        super().__init__(run)
        self.budget = manifest["new_http_max"]

    def call(self, label, payload, proof):
        base.require(self.http_counter < self.budget, "combined_recovery_http_budget_exhausted")
        return super().call(label, payload, proof)


def validated_call(caller, run, root, role, history, cards, scene=None):
    repair = None
    for attempt in (1, 2):
        payload, proof = make_request(run, root, role, history, cards, scene, repair)
        label = f"{root.split('-')[-1]}-B-{role}-{attempt}"
        raw, directory = caller.call(label, payload, proof)
        try:
            parsed = validate_plan(raw, root, history, cards) if role == "planner" else base.validate_realization(raw, root, ROUTE, scene)
            base.write(directory / "VALIDATION.json", {"status": "MECHANICAL_PASS", "semantic_certified": False})
            return parsed
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            reason = str(error) if isinstance(error, ValueError) else type(error).__name__
            base.write(directory / "VALIDATION.json", {"status": "MECHANICAL_FAIL", "reason": reason, "semantic_certified": False})
            repair = {"instruction": "Correct only the reported JSON/schema/reference error. Respect the one-scene budget and supplied card.",
                      "failure": reason, "previous_response": raw}
    raise ValueError("mechanical_validation_failed_after_one_repair")


def run_root(caller, run, root, card):
    history = base.read(run / "parent_A" / f"{root}.json")["turns"]
    before = deepcopy(history)
    out = run / "roots" / root / "B"
    cards = [base.dataset_cards.build_realization_guidance(card)]
    base.write(out / "INPUT.json", {"root_id": root, "route": ROUTE, "turns": before, "cards": cards, "scene_budget": 1})
    result = {"root_id": root, "route": ROUTE, "target_scenes": 1, "accepted_scenes": 0,
        "added_turns": 0, "failures": [], "input_turns": len(before), "input_history_sha256": base.digest(before),
        "scene_metadata": [], "mechanical_acceptance_only": True, "semantic_certified": False,
        "A_to_B_handoff_verified": True, "A_output_history_sha256": base.digest(before)}
    try:
        plan = validated_call(caller, run, root, "planner", history, cards)
        base.write(out / "PLAN.json", plan)
        result.update(planned_scenes=len(plan["scenes"]), shortfall_reason=plan["shortfall_reason"])
        for scene in plan["scenes"]:
            try:
                realization = validated_call(caller, run, root, "realizer", history, cards, scene)
                base.write(out / f"{scene['scene_id']}.json", realization)
                if realization["status"] == "scene":
                    history = base.append_scene(history, root, ROUTE, realization)
                    result["accepted_scenes"] += 1
                    result["scene_metadata"].append({"scene_id": scene["scene_id"], "dataset_card_id": card["card_id"],
                        "turns": len(realization["turns"]), "reported_feedback_topic_ids": realization["feedback_topic_ids"],
                        "model_reported_not_certified": True})
                else:
                    result["failures"].append({"scene_id": scene["scene_id"], "reason": realization["no_scene_reason"]})
            except (ValueError, RuntimeError) as error:
                result["failures"].append({"scene_id": scene["scene_id"], "reason": str(error)})
    except (ValueError, RuntimeError) as error:
        result["failures"].append({"stage": "planner", "reason": str(error)})
    base.require(history[:len(before)] == before, "parent_A_prefix_changed")
    result.update(added_turns=len(history)-len(before), output_turns=len(history), output_history_sha256=base.digest(history),
                  source_prefix_preserved=True, shortfall_scenes=1-result["accepted_scenes"])
    base.write(out / "RESULT.json", result)
    base.write(out / "cumulative_dialogue.json", {"root_id": root, "route": ROUTE, "turns": history,
        "history_sha256": base.digest(history), "model": base.dataset_cards.MODEL, "original_prefix_preserved": True,
        "qa_fact_source": "full_cumulative_dialogue", "memory_is_qa_fact_source": False, "semantic_certified": False})
    print(json.dumps({"event": "delta_B_done", "root": root, "scenes": result["accepted_scenes"],
                      "failures": result["failures"]}), flush=True)
    return result


def finish(run, manifest, results):
    verify(run)
    parent = Path(manifest["parent_run"])
    previous = Path(manifest["previous_delta"])
    previous_roots = {row["root_id"] for row in base.read(previous / "REPORT.json")["routes"] if row["route"] == ROUTE}
    cards = {card["card_id"]: card for card in base.read(run / "dataset_cards_result.json")["accepted_cards"]}
    prior_keys = {unique.event_key(row) for row in manifest["prior"]["accepted_uses"]}
    accepted, keys = [], set()
    for result in results:
        root = result["root_id"]
        base.verify_handoff(base.read(parent / "roots" / root / "A" / "cumulative_dialogue.json")["turns"],
                            base.read(run / "roots" / root / "B" / "INPUT.json")["turns"])
        root_cards = [cards[row["dataset_card_id"]] for row in result["scene_metadata"]]
        unique.verify_cards_for_root(root_cards, manifest, root)
        for scene, card in zip(result["scene_metadata"], root_cards):
            key = unique.event_key(card)
            base.require(key not in keys and key not in prior_keys, "delta_source_reused")
            keys.add(key)
            accepted.append({"root_id": root, "scene_id": scene["scene_id"], "card_id": card["card_id"],
                "source_id": card["source_id"], "dataset": key[0], "split": key[1], "item_id": key[2]})
    requests = sorted((run / "requests").glob("*"))
    http = [path for path in requests if (path / "HTTP_ATTEMPT.json").exists()]
    responses = [base.read(path / "RESPONSE.json") for path in http if (path / "RESPONSE.json").exists()]
    for path in requests:
        payload, proof = base.read(path / "REQUEST.json"), base.read(path / "PROOF.json")
        suffix = (run / "feedback" / proof["root_id"] / f"{ROUTE}__{proof['role']}.txt").read_text()
        base.require(proof["payload_sha256"] == base.digest(payload) and payload["messages"][0]["content"].endswith(suffix)
            and proof["feedback_receipt"]["suffix_sha256"] == base.digest(suffix.encode()), "delta_request_or_feedback_changed")
        if path in http:
            attempt = base.read(path / "HTTP_ATTEMPT.json")
            base.require(attempt["actual_payload_equals_frozen"] is True and attempt["payload_sha256"] == base.digest(payload),
                         "delta_actual_payload_changed")
    base.require(len(http) <= manifest["new_http_max"], "delta_http_cap_exceeded")
    report = {"version": VERSION, "parent_run": str(parent), "roots": manifest["roots"], "routes": results,
        "status": "COMPLETED_WITH_RECORDED_SHORTFALLS" if any(row["shortfall_scenes"] for row in results) else "COMPLETED",
        "requested_model": base.dataset_cards.MODEL, "returned_models": sorted({row["metadata"].get("returned_model", "UNKNOWN") for row in responses}),
        "accepted_scenes": sum(row["accepted_scenes"] for row in results), "added_turns": sum(row["added_turns"] for row in results),
        "http_attempts": len(http), "http_max": manifest["new_http_max"], "parent_http_attempts": manifest["parent_http_attempts"],
        "combined_http_attempts": manifest["parent_http_attempts"] + len(http), "combined_http_max": COMBINED_HTTP_MAX,
        "responses": len(responses), "new_run_input_tokens": sum(row["metadata"].get("input_tokens") or 0 for row in responses),
        "new_run_output_tokens": sum(row["metadata"].get("output_tokens") or 0 for row in responses),
        "all_B_handoffs_verified": True, "global_unique_event_check": "PASS", "semantic_certification": False,
        "question_generation_calls": 0, "solver_calls": 0, "training_calls": 0, "memory_calls": 0,
        "automatic_transport_retries": 0, "only_delta_B_routes_registered": True}
    base.write(run / "UNIQUE_EVENT_REPORT.json", {"version": VERSION, "accepted_uses": accepted,
        "global_unique_event_check": "PASS", "only_delta_B_routes_registered": True})
    consolidated = []
    for root in base.ROOTS:
        row = {"root_id": root}
        for letter in ("A", "B"):
            source_run = (run if root in manifest["roots"] else previous if root in previous_roots else parent) if letter == "B" else parent
            result_path = source_run / "roots" / root / letter / "RESULT.json"
            dialogue_path = result_path.with_name("cumulative_dialogue.json")
            row[letter] = {"accepted_scenes": base.read(result_path)["accepted_scenes"], "result_path": str(result_path),
                "result_sha256": unique.sha256(result_path), "dialogue_path": str(dialogue_path), "dialogue_sha256": unique.sha256(dialogue_path)}
        consolidated.append(row)
    base.write(run / "CONSOLIDATED_REPORT.json", {"version": VERSION, "parent_run": str(parent), "previous_delta": str(previous), "delta_run": str(run),
        "per_root": consolidated, "accepted_scenes": sum(row[letter]["accepted_scenes"] for row in consolidated for letter in ("A", "B")),
        "combined_http_attempts": report["combined_http_attempts"], "combined_http_max": COMBINED_HTTP_MAX,
        "references_only_not_additional_source_consumption": True, "semantic_certification": False})
    # This completion marker exposes only the new B scenes to the shared source inventory.
    base.write(run / "REPORT.json", report)
    return report


def live(run):
    with (v2.V2_RUNS / ".unique_event_live.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest = verify(run)
        current = inventory()
        consumed = {unique.event_key(row) for row in current["accepted_uses"]}
        assigned = {unique.event_key(row) for rows in manifest["assignments"].values() for row in rows}
        base.require(not (assigned & consumed), "delta_source_consumed_after_prepare")
        base.write(run / "LIVE_STARTED.json", {"version": VERSION, "started_at": datetime.now(timezone.utc).isoformat(),
            "authorization": "USER_REQUEST_RESOLVE_RECOVERED_R02_RESULTS", "new_http_max": manifest["new_http_max"],
            "combined_http_max": COMBINED_HTTP_MAX, "automatic_transport_retries": 0})
        caller = DeltaCaller(run, manifest)
        cards = base.read(run / "dataset_cards_result.json")["accepted_cards"]
        results = []
        for root in manifest["roots"]:
            key = unique.event_key(manifest["assignments"][root][0])
            card = next(card for card in cards if unique.event_key(card) == key)
            results.append(run_root(caller, run, root, card))
        return finish(run, manifest, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify", "live"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    args = parser.parse_args()
    run = args.run.resolve()
    result = prepare(run, args.parent.resolve(), args.previous.resolve()) if args.action == "prepare" else live(run) if args.action == "live" else {"status": "PASS", "roots": verify(run)["roots"]}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
