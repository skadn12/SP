"""Read-only structural audit of an R02 v8 run; never changes model decisions.

Only FINAL_AUDIT.json is written. A PASS certifies artifact consistency, exact
citations and request isolation, not Qwen's semantic judgments or data quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import runner as r


def decode_context(payload, marker):
    return json.JSONDecoder().raw_decode(payload["messages"][0]["content"].split(marker, 1)[1])[0]


def audit_run(run):
    run = Path(run).resolve()
    read, digest = r.base.read, r.base.digest
    failures, checks, request_rows = [], [], []
    counters = Counter()
    roles, statuses, models = Counter(), Counter(), Counter()
    numbers, slots, state_outputs, final_realizations = [], {}, {}, {}
    event_outputs, plan_outputs, event_plan_hashes = {}, {}, {}
    latest_realizations = {}

    def require(condition, code):
        if not condition:
            raise ValueError(code)

    def stage(name, function):
        try:
            result = function()
        except Exception as error:
            failures.append({"check": name, "error": f"{type(error).__name__}: {error}"})
            return None
        checks.append(name)
        return result

    manifest = read(run / "MANIFEST.json")
    allocation = read(run / "UNIQUE_EVENT_ALLOCATION.json")
    report = read(run / "REPORT.json")
    source_manifest = read(run / "dataset_sources.json")
    stage("frozen_inputs_and_allocation", lambda: r.verify(run))

    def code_bindings():
        index = read(run / "EXECUTION_CODE_SNAPSHOT.json")
        bound = {row["path"]: row["sha256"] for row in manifest["files"]}
        for row in index["files"]:
            require(digest(Path(row["snapshot_path"]).read_bytes()) == row["sha256"], "code_snapshot_changed")
            require(bound.get(row["original_path"]) == row["sha256"], "code_snapshot_not_bound_to_manifest")
        counters["code_snapshots"] = len(index["files"])
    stage("execution_code_bindings", code_bindings)

    def inventory():
        reservations = read(run / "SOURCE_RESERVATIONS.json")
        require(reservations["consumption_policy"] == "reservation_before_model_request_never_release", "source_released_policy")
        assigned = r.source_inventory._assignment_rows(run, allocation["assignments"])
        require(reservations["reservations"] == assigned, "source_reservation_differs_from_allocation")
        keys = {r.source_inventory.event_key(row) for row in assigned}
        prior = {r.source_inventory.event_key(row) for row in allocation["prior"]["accepted_uses"]}
        require(not keys.intersection(prior), "source_reused_from_prior_inventory")
        occupied = r.source_inventory.collect_prior()["accepted_uses"]
        by_key = {r.source_inventory.event_key(row): row for row in occupied}
        for row in assigned:
            found = by_key.get(r.source_inventory.event_key(row))
            require(found is not None, "reserved_source_not_permanently_occupied")
            require(all(found[key] == row[key] for key in ("run", "root_id", "scene_id", "source_item_sha256")),
                    "reserved_source_owner_changed")
        launch_path = run / "LAUNCH_CHECKS.json"
        if launch_path.exists():
            launch = read(launch_path)
            require(launch["source_overlap_count"] == 0, "launch_source_overlap")
            if "previously_consumed_source_identities" in launch:
                before = {tuple(item) for item in launch["previously_consumed_source_identities"]}
                require(before == prior, "launch_prior_identities_differ_from_allocation")
                require(not before.intersection(keys), "launch_reused_source")
        counters["occupied_sources_before"] = len(prior)
        counters["occupied_sources_after"] = len(by_key)
        counters["reserved_sources"] = len(assigned)
        counters["source_overlap_count"] = len(keys.intersection(prior))
        return [list(key) for key in sorted(keys)]
    source_identities = stage("permanent_source_reservations", inventory)

    def route_chain():
        total_scenes, total_turns = 0, 0
        for row in report["routes"]:
            root, route = row["root_id"], row["route"]
            letter = "A" if route == r.ROUTES[0] else "B"
            folder = run / "roots" / root / letter
            history = read(run / "sources" / f"{root}.json")["turns"]
            frozen = list(history)
            require(read(folder / "INPUT.json")["turns"] == frozen, "route_input_changed")
            accepted = {item["scene_id"]: item for item in row["scene_metadata"]}
            for number in range(1, row["available_slots"] + 1):
                scene_id = f"S{number:02d}"
                slot = (root, route, scene_id)
                slots[slot] = {"history": list(history), "folder": folder}
                state_path = folder / f"{scene_id}.LATEST_STATE.json"
                if state_path.exists():
                    snapshot = read(state_path)
                    r.state_contract.validate_state(snapshot, root, route, scene_id, history)
                    slots[slot]["latest_state"] = snapshot
                    counters["saved_state_snapshots"] += 1
                plan_path = folder / f"{scene_id}.PLAN.json"
                if plan_path.exists():
                    slots[slot]["plan"] = read(plan_path)
                event_path = folder / f"{scene_id}.EVENT_STATE.json"
                if event_path.exists():
                    event_snapshot = read(event_path)
                    r.event_contract.validate(event_snapshot, root, route,
                                               slots[slot]["plan"]["scenes"][0], history)
                    require(not r.event_contract.requires_replan(event_snapshot), "final_plan_requires_replan")
                    slots[slot]["event_state"] = event_snapshot
                    counters["saved_event_snapshots"] += 1
                if scene_id in accepted:
                    realization = read(folder / f"{scene_id}.json")
                    require("latest_state" in slots[slot], "accepted_scene_missing_state")
                    require(accepted[scene_id]["latest_state_sha256"] == digest(slots[slot]["latest_state"]),
                            "accepted_state_binding_mismatch")
                    require(accepted[scene_id]["event_state_sha256"] == digest(slots[slot]["event_state"]),
                            "accepted_event_state_binding_mismatch")
                    history = r.base.append_scene(history, root, route, realization)
                    final_realizations[slot] = realization
            final_history = read(folder / "cumulative_dialogue.json")["turns"]
            require(final_history == history, "final_history_not_accepted_scene_chain")
            r.branch_history_ok(run, root, route, final_history)
            require(row["output_history_sha256"] == digest(history), "route_output_hash_mismatch")
            require(row["branch_base_sha256"] == digest(frozen), "route_base_hash_mismatch")
            require(row["accepted_scenes"] == len(accepted), "accepted_scene_count_mismatch")
            require(row["added_turns"] == len(history) - len(frozen), "added_turn_count_mismatch")
            require(read(folder / "RESULT.json") == row, "route_report_mismatch")
            total_scenes += len(accepted)
            total_turns += row["added_turns"]
        require(total_scenes == report["accepted_scenes"], "report_total_scene_mismatch")
        counters["accepted_scenes"] = total_scenes
        counters["added_turns"] = total_turns
    stage("accepted_branch_chains_and_state_citations", route_chain)

    def other_turns(root, route):
        if route == r.ROUTES[0]:
            return []  # A executes before B; its comparison corpus starts empty.
        path = run / "comparison_corpora" / f"{root}.A.json"
        if path.exists():
            return read(path)["turns"]
        path = run / "roots" / root / "A/cumulative_dialogue.json"
        if path.exists():
            frozen = read(run / "sources" / f"{root}.json")["turns"]
            return read(path)["turns"][len(frozen):]
        return []

    accepted_requests = {}
    for directory in sorted((run / "requests").iterdir()):
        if not directory.is_dir():
            continue

        def audit_request(directory=directory):
            payload, proof = read(directory / "REQUEST.json"), read(directory / "PROOF.json")
            role = proof["role"]
            roles[role] += 1
            require(role in ("dataset_extractor", "state_extractor", "event_state_extractor", "planner", "realizer", "scene_reviewer"), "unexpected_role")
            require(payload["model"] == manifest["model"] == r.MODEL, "requested_model_mismatch")
            require(proof["payload_sha256"] == digest(payload), "request_payload_hash_mismatch")
            attempts = sorted(directory.glob("HTTP_ATTEMPT*.json"))
            require(len(attempts) <= manifest["automatic_transport_retries"] + 1, "transport_retry_limit_exceeded")
            for index, path in enumerate(attempts, 1):
                attempt = read(path)
                require(attempt["actual_payload_equals_frozen"] is True, "actual_payload_not_frozen")
                require(attempt["payload_sha256"] == digest(payload), "attempt_payload_hash_mismatch")
                require(attempt["requested_model"] == r.MODEL, "attempt_model_mismatch")
                require(attempt["request_attempt"] == index, "attempt_sequence_invalid")
                require(attempt["same_payload_retry"] == (index > 1), "retry_flag_mismatch")
                numbers.append(attempt["http_attempt_number"])
                result = read(directory / f"HTTP_RESULT_{index:02d}.json")
                statuses[str(result["http_status"])] += 1
                if index > 1:
                    previous = read(directory / f"HTTP_RESULT_{index - 1:02d}.json")
                    require(previous["http_status"] in (503, 504), "non_503_504_transport_retry")
                response_path = directory / f"HTTP_RESPONSE_{index:02d}.json"
                if response_path.exists():
                    metadata = read(response_path)["metadata"]
                    models[str(metadata.get("returned_model"))] += 1
                    if result["outcome"] == "response":
                        require(metadata.get("returned_model") == r.MODEL, "returned_model_mismatch")
                        require(metadata.get("finish_reason") == "stop", "incomplete_success_response")
            require(len(list(directory.glob("HTTP_RESULT_*.json"))) == len(attempts), "http_result_count_mismatch")
            response_path = directory / "RESPONSE.json"
            response = read(response_path) if response_path.exists() else None
            if response is not None:
                require(bool(attempts), "response_without_http_attempt")
                last = read(directory / f"HTTP_RESPONSE_{len(attempts):02d}.json")
                require(response["text"] == last["text"] and response["metadata"] == last["metadata"],
                        "final_response_differs_from_attempt")
            else:
                require((directory / "ERROR.json").exists(), "request_missing_outcome")
            request_rows.append({"request": directory.name, "role": role,
                                 "http_attempts": len(attempts), "has_response": response is not None})
            if role == "dataset_extractor":
                subset = r.base.dataset_cards.select_source_subset(source_manifest, [proof["source_id"]])
                require(tuple(proof["source_event_identity"]) == r.source_inventory.event_key(subset["items"][0]),
                        "extraction_source_identity_mismatch")
                if response is not None:
                    require(r.card_selection.select_cards(response["text"], subset) == read(directory / "VALIDATION.json"),
                            "card_validation_replay_changed")
                    counters["card_validation_replays"] += 1
                return
            root, route, scene_id = proof["root_id"], proof["route"], proof["scene_id"]
            slot = (root, route, scene_id)
            require(slot in slots, "request_slot_missing_from_report")
            if role == "scene_reviewer":
                context = decode_context(payload, "\nREVIEW INPUT DATA:\n")
                history = context["branch_history"]
                require(context["other_branch_added_turns_for_review_only"] == other_turns(root, route),
                        "review_comparison_corpus_mismatch")
                require(proof["other_branch_review_only"] is True, "review_only_marker_missing")
                raw_scene = deepcopy(context["new_scene"])
                for turn in raw_scene["turns"]:
                    turn.pop("turn_ref", None)
                require(r.realization_review.review_scene(raw_scene) == context["new_scene"],
                        "review_new_turn_ids_not_canonical")
                require(proof["realization_sha256"] == digest(raw_scene), "review_realization_hash_mismatch")
                require(latest_realizations.get(slot) == raw_scene, "review_not_latest_realizer_response")
                if response is not None:
                    rechecked = r.realization_review.validate_review(response["text"], root, route,
                        context["scene_plan"], context["new_scene"], dataset_card=context["dataset_card_guidance"],
                        latest_state=context["latest_state"], history=history, event_state=context["event_state"])
                    require(rechecked == read(directory / "REVIEW.json"), "review_exact_citation_replay_changed")
                    counters["review_validation_replays"] += 1
                    state_checks = (rechecked.get("parsed") or {}).get("state_checks", [])
                    state_checks = state_checks if isinstance(state_checks, list) else []
                    counters["review_state_checks"] += len(state_checks)
                    counters["negative_state_checks"] += sum(c.get("verdict") == "contradicted"
                        for c in state_checks if isinstance(c, dict))
                    event_checks = (rechecked.get("parsed") or {}).get("event_checks", [])
                    event_checks = event_checks if isinstance(event_checks, list) else []
                    counters["review_event_checks"] += len(event_checks)
                    counters["negative_event_checks"] += sum(c.get("verdict") == "contradicted"
                        for c in event_checks if isinstance(c, dict))
            else:
                marker = {"state_extractor": "\nSTATE_INPUT_JSON\n",
                          "event_state_extractor": "\nEVENT_STATE_INPUT_JSON\n"}.get(role, "\nINPUT_JSON\n")
                context = decode_context(payload, marker)
                history = context["cumulative_history"]
                require(proof["other_branch_in_generator_context"] is False, "generator_branch_isolation_flag")
                require("other_branch_added_turns_for_review_only" not in context
                        and "other_branch_accepted_additions" not in context, "comparison_in_generator_context")
                if role in ("planner", "realizer"):
                    receipt = proof["feedback_receipt"]
                    text = payload["messages"][0]["content"]
                    suffix = (run / "feedback" / root / f"{route}__{role}.txt").read_text()
                    require(receipt["feedback_appended"] is True and receipt["current_generation_feedback_used"] is False,
                            "feedback_round_invalid")
                    require(text.endswith(suffix), "feedback_suffix_missing")
                    require(receipt["request_payload_sha256"] == digest(payload), "feedback_payload_mismatch")
                    require(receipt["suffix_sha256"] == digest(suffix.encode()), "feedback_suffix_hash_mismatch")
                    require(receipt["request_user_text_sha256"] == digest(text.encode()), "feedback_text_hash_mismatch")
                    require(receipt["before_user_text_sha256"] == digest(text[:-len(suffix)].encode()), "feedback_prefix_hash_mismatch")
                    counters["feedback_receipts_verified"] += 1
                if role == "state_extractor" and response is not None:
                    saved = read(directory / "VALIDATION.json")
                    try:
                        snapshot = r.state_contract.validate_state(response["text"], root, route, scene_id, history)
                    except (ValueError, TypeError):
                        require(saved["status"] == "REJECT", "state_rejection_replay_changed")
                    else:
                        require(saved["status"] == "PASS", "state_pass_replay_changed")
                        require(saved["latest_state_sha256"] == digest(snapshot), "state_response_hash_changed")
                        require(slots[slot]["latest_state"] == snapshot, "saved_state_differs_from_response")
                        state_outputs[slot] = snapshot
                    counters["state_validation_replays"] += 1
                if role == "event_state_extractor":
                    scene = context["scene_plan"]
                    require(proof["scene_plan_sha256"] == digest(scene), "event_scene_proof_mismatch")
                    require(plan_outputs.get(slot, {}).get("scenes") == [scene], "event_scan_not_current_candidate")
                    if response is not None:
                        saved = read(directory / "VALIDATION.json")
                        try:
                            event_snapshot = r.event_contract.validate(response["text"], root, route, scene, history)
                        except (ValueError, TypeError):
                            require(saved["status"] == "REJECT", "event_rejection_replay_changed")
                        else:
                            require(saved["status"] == "PASS", "event_pass_replay_changed")
                            require(saved["event_state_sha256"] == digest(event_snapshot), "event_response_hash_changed")
                            require(saved["requires_replan"] == r.event_contract.requires_replan(event_snapshot),
                                    "event_replan_replay_changed")
                            event_outputs[slot] = event_snapshot
                            event_plan_hashes[slot] = digest(scene)
                        counters["event_validation_replays"] += 1
                if role == "planner" and response is not None:
                    saved = read(directory / "VALIDATION.json")
                    try:
                        plan = r.validate_plan(response["text"], root, route, history, context["dataset_cards"], scene_id,
                                               latest_state=context["latest_state"])
                    except (ValueError, TypeError):
                        require(saved["status"] == "REJECT", "plan_rejection_replay_changed")
                    else:
                        require(saved["status"] == "PASS", "plan_pass_replay_changed")
                        require(read(directory / "CANDIDATE_PLAN.json") == plan, "candidate_plan_differs_from_response")
                        plan_outputs[slot] = plan
                    counters["plan_validation_replays"] += 1
                if role == "realizer" and response is not None:
                    accepted = read(directory / "ACCEPTANCE.json")
                    if (directory / "DUPLICATE_CHECK.json").exists():
                        realization = r.base.validate_realization(r.parse_object(response["text"]), root, route, context["scene_plan"])
                        latest_realizations[slot] = realization
                        duplicate = r.duplicate_check.inspect_duplicates(realization["turns"],
                            {"branch_history": history, "other_branch_accepted_additions": other_turns(root, route)})
                        require(duplicate == read(directory / "DUPLICATE_CHECK.json"), "duplicate_check_replay_changed")
                        counters["duplicate_check_replays"] += 1
                    if accepted["status"] == "PASS":
                        require(r.parse_object(response["text"]) == final_realizations[slot], "accepted_realization_response_changed")
                        review_dir = Path(accepted["review_request"])
                        require(read(review_dir / "REVIEW.json")["status"] == "PASS", "accepted_without_positive_review")
                        review_context = decode_context(read(review_dir / "REQUEST.json"), "\nREVIEW INPUT DATA:\n")
                        require(review_context["new_scene"] == r.realization_review.review_scene(final_realizations[slot]),
                                "accepted_review_for_other_realization")
                        accepted_requests[slot] = str(directory)
            require(history == slots[slot]["history"], "request_history_differs_from_accepted_chain")
            require(proof["history_sha256"] == digest(history), "request_history_hash_mismatch")
            require(context["root_id"] == root and context["route"] == route, "context_identity_mismatch")
            if role != "state_extractor":
                require(context["latest_state"] == slots[slot]["latest_state"], "snapshot_pipeline_mismatch")
                require(proof["latest_state_sha256"] == digest(context["latest_state"]), "snapshot_proof_mismatch")
                require(state_outputs.get(slot) == context["latest_state"], "state_not_from_earlier_success_response")
                counters["snapshot_pipeline_links_verified"] += 1
            if role in ("planner", "realizer", "scene_reviewer"):
                event_snapshot = context.get("event_state")
                require(proof["event_state_sha256"] == digest(event_snapshot), "event_proof_mismatch")
                if event_snapshot is not None:
                    require(event_outputs.get(slot) == event_snapshot, "event_not_from_previous_scan")
                    counters["event_pipeline_links_verified"] += 1
            if role in ("realizer", "scene_reviewer"):
                require(context["scene_plan"] == slots[slot]["plan"]["scenes"][0], "scene_plan_pipeline_mismatch")
                require(context["event_state"] == slots[slot]["event_state"], "final_event_pipeline_mismatch")
                require(event_plan_hashes.get(slot) == digest(context["scene_plan"]), "final_event_scan_for_other_plan")
                require(not r.event_contract.requires_replan(context["event_state"]), "realized_plan_required_revision")
        stage("request:" + directory.name, audit_request)

    def totals():
        require(sum(roles.values()) == report["http_request_reservations"], "logical_request_count_mismatch")
        require(len(numbers) == report["http_attempts"] == report["http_attempt_log_count"], "http_count_mismatch")
        require(sorted(numbers) == list(range(1, len(numbers) + 1)), "global_http_sequence_invalid")
        require(sum(statuses.values()) == len(numbers), "http_status_count_mismatch")
        require(len(numbers) <= manifest["http_max"], "http_budget_exceeded")
        require(set(accepted_requests) == set(final_realizations), "accepted_output_missing_successful_request")
        require(report["question_calls"] == report["solver_calls"] == 0 and report["training_started"] is False,
                "unexpected_downstream_activity")
    stage("request_totals_and_accepted_output_binding", totals)
    result = {
        "audit_version": "r02_event_state_v8_artifact_audit_v1",
        "audit_time_utc": datetime.now(timezone.utc).isoformat(), "run": str(run),
        "status": "FAIL" if failures else "PASS", "failures": failures,
        "completed_checks": checks, "model": manifest["model"],
        "run_status": report["status"], "roots": manifest["roots"],
        "selected_routes": manifest["selected_routes"], "source_identities": source_identities,
        **dict(counters), "logical_requests_by_role": dict(roles), "http_attempts": len(numbers),
        "http_statuses": dict(statuses), "returned_models": dict(models), "requests": request_rows,
        "question_calls": report["question_calls"], "solver_calls": report["solver_calls"],
        "training_started": report["training_started"], "model_decisions_modified": False,
        "semantic_certified": False,
        "limitation": "Artifact and exact-quote checks only; Qwen state extraction and semantic verdicts remain fallible. Shortfall or negative reviews do not by themselves fail this audit.",
    }
    (run / "FINAL_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = audit_run(args.run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
