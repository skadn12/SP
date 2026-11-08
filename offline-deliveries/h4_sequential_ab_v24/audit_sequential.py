"""Independent artifact audit of sequential A/B generation; no model calls.

Only FINAL_AUDIT.json is written. A PASS certifies artifact consistency, exact
citations and request isolation, not Qwen's semantic judgments or data quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("r02_v24_audit_runtime", HERE / "runner.py")
assert _spec and _spec.loader
r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r)
for _module in (r.source_inventory, r.state_contract, r.event_contract, r.realization_review):
    if Path(_module.__file__).resolve().parent != HERE:
        raise RuntimeError("audit_dependency_outside_current_delivery:" + str(_module.__file__))


def decode_context(payload, marker):
    return json.JSONDecoder().raw_decode(payload["messages"][0]["content"].split(marker, 1)[1])[0]


def audit_run(run, *, count_tokens=False):
    run = Path(run).resolve()
    if run.parent != HERE / "runs":
        raise ValueError("audit_run_must_belong_to_current_v24_runs")
    read, digest = r.base.read, r.base.digest
    failures, checks, request_rows = [], [], []
    counters = Counter()
    roles, statuses, models = Counter(), Counter(), Counter()
    numbers, slots, state_outputs, final_realizations = [], {}, {}, {}
    event_outputs, plan_outputs, event_plan_hashes = {}, {}, {}
    latest_realizations = {}
    reference_repairs = {}
    reviewer_candidates, reviewer_repairs = {}, {}

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

    # A live directory is not a successful run: missing final report/merge
    # manifests must fail before any PASS artifact can be written.
    for name in ("MANIFEST.json", "REPORT.json", "MERGE_MANIFEST.json", "SOURCE_RESERVATIONS.json"):
        require((run / name).is_file(), "incomplete_run_missing:" + name)
    manifest = read(run / "MANIFEST.json")
    allocation = read(run / "UNIQUE_EVENT_ALLOCATION.json")
    report = read(run / "REPORT.json")
    source_manifest = read(run / "dataset_sources.json")
    stage("frozen_inputs_and_allocation", lambda: r.verify(run))

    def code_bindings():
        index = read(run / "EXECUTION_CODE_SNAPSHOT.json")
        bound = {row["path"]: row["sha256"] for row in manifest["files"]}
        snapshots = {row["original_path"] for row in index["files"]}
        require(len(snapshots) == len(index["files"]), "duplicate_code_snapshot")
        require(snapshots == {path for path in bound if Path(path).suffix == ".py"},
                "code_snapshot_index_incomplete")
        for row in index["files"]:
            require(digest(Path(row["snapshot_path"]).read_bytes()) == row["sha256"], "code_snapshot_changed")
            require(bound.get(row["original_path"]) == row["sha256"], "code_snapshot_not_bound_to_manifest")
        counters["code_snapshots"] = len(index["files"])
    stage("execution_code_bindings", code_bindings)

    def frozen_feedback():
        lock = read(run / "feedback/source_lock.json")
        historical = lock["historical_source_lock"]
        require(lock["merge_current_generation_feedback"] is False, "current_feedback_in_lock")
        require(lock["automatic_latest_run_selection"] is False, "automatic_feedback_selection")
        require(historical["status"] == "EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY", "feedback_not_previous_round")
        require(historical["merge_current_generation_feedback"] is False, "current_feedback_in_historical_lock")
        bound = {row["path"] for row in manifest["files"]}
        require(str(run / "feedback/source_lock.json") in bound, "feedback_lock_not_frozen")
        return historical["evaluation_run_id"]
    feedback_run = stage("frozen_previous_round_feedback", frozen_feedback)

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
        require((run / "SOURCE_RESERVATIONS.json").stat().st_mtime_ns <=
                (run / "LIVE_STARTED.json").stat().st_mtime_ns, "reservation_after_live_started")
        for path in (run / "requests").glob("*/HTTP_ATTEMPT*.json"):
            require((run / "SOURCE_RESERVATIONS.json").stat().st_mtime_ns <= path.stat().st_mtime_ns,
                    "reservation_after_http_attempt")
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
            frozen = read(run / "sources" / f"{root}.json")["turns"]
            a_path = run / "roots" / root / "A/cumulative_dialogue.json"
            route_base = read(a_path)["turns"] if letter == "B" and a_path.exists() else list(frozen)
            require(route_base[:len(frozen)] == frozen, "route_base_frozen_prefix_changed")
            history = list(route_base)
            require(read(folder / "INPUT.json")["turns"] == route_base, "route_input_changed")
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
                    for fact in snapshot["facts"]:
                        counters["state_evidence_basis_" + fact["evidence_basis"]] += 1
                plan_path = folder / f"{scene_id}.PLAN.json"
                if plan_path.exists():
                    slots[slot]["plan"] = read(plan_path)
                event_path = folder / f"{scene_id}.EVENT_STATE.json"
                if event_path.exists():
                    event_snapshot = read(event_path)
                    r.event_contract.validate(event_snapshot, root, route,
                                               slots[slot]["plan"]["scenes"][0], history)
                    require(read(folder / f"{scene_id}.EVENT_COVERAGE.json") ==
                            r.event_contract.derive_coverage(event_snapshot, history),
                            "saved_event_coverage_changed")
                    require(not r.event_contract.requires_replan(event_snapshot), "final_plan_requires_replan")
                    slots[slot]["event_state"] = event_snapshot
                    counters["saved_event_snapshots"] += 1
                    for constraint in event_snapshot["timeline"]:
                        counters["event_evidence_basis_" + constraint["evidence_basis"]] += 1
                if scene_id in accepted:
                    realization = read(folder / f"{scene_id}.json")
                    require("latest_state" in slots[slot], "accepted_scene_missing_state")
                    require(accepted[scene_id]["latest_state_sha256"] == digest(slots[slot]["latest_state"]),
                            "accepted_state_binding_mismatch")
                    require(accepted[scene_id]["event_state_sha256"] == digest(slots[slot]["event_state"]),
                            "accepted_event_state_binding_mismatch")
                    require(r.event_contract.route_fulfilled(slots[slot]["event_state"]["route_fulfillment"], route),
                            "accepted_plan_does_not_fulfill_route")
                    require(r.event_contract.route_fulfilled(accepted[scene_id]["route_fulfillment"], route),
                            "accepted_scene_does_not_fulfill_route")
                    history = r.base.append_scene(history, root, route, realization)
                    final_realizations[slot] = realization
            final_history = read(folder / "cumulative_dialogue.json")["turns"]
            require(final_history == history, "final_history_not_accepted_scene_chain")
            r.branch_history_ok(run, root, route, final_history)
            require(row["output_history_sha256"] == digest(history), "route_output_hash_mismatch")
            require(row["branch_base_sha256"] == digest(route_base), "route_base_hash_mismatch")
            require(row["route_input_history_sha256"] == digest(route_base), "route_input_hash_mismatch")
            require(row["frozen_base_sha256"] == digest(frozen), "frozen_base_hash_mismatch")
            require(row["accepted_A_in_input"] == (letter == "B" and len(route_base) > len(frozen)),
                    "accepted_A_input_flag_mismatch")
            require(row["accepted_scenes"] == len(accepted), "accepted_scene_count_mismatch")
            require(row["added_turns"] == len(history) - len(route_base), "added_turn_count_mismatch")
            require(row["shortfall_scenes"] == row["target_scenes"] - len(accepted), "shortfall_count_mismatch")
            require(len({turn["turn_ref"] for turn in history}) == len(history), "duplicate_turn_ids")
            require(read(folder / "RESULT.json") == row, "route_report_mismatch")
            total_scenes += len(accepted)
            total_turns += row["added_turns"]
        require(total_scenes == report["accepted_scenes"], "report_total_scene_mismatch")
        counters["accepted_scenes"] = total_scenes
        counters["added_turns"] = total_turns
    stage("accepted_sequential_chains_and_state_citations", route_chain)

    def other_turns(root, route):
        # Accepted A is already in B's authoritative history, not a second corpus.
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
                require(proof["history_policy"] == "complete_current_root_including_accepted_A_for_B",
                        "review_history_policy_mismatch")
                raw_scene = deepcopy(context["new_scene"])
                for turn in raw_scene["turns"]:
                    turn.pop("turn_ref", None)
                require(r.realization_review.review_scene(raw_scene) == context["new_scene"],
                        "review_new_turn_ids_not_canonical")
                require(proof["realization_sha256"] == digest(raw_scene), "review_realization_hash_mismatch")
                require(latest_realizations.get(slot) == raw_scene, "review_not_latest_realizer_response")
                if response is not None:
                    original_plan = slots[slot]["plan"]["scenes"][0]
                    original_event = slots[slot]["event_state"]
                    review_key = (slot, digest(raw_scene))
                    prompt_text = payload["messages"][0]["content"]
                    repair = (json.JSONDecoder().raw_decode(prompt_text.split("\nREVIEW_FORMAT_REPAIR:\n", 1)[1])[0]
                              if "\nREVIEW_FORMAT_REPAIR:\n" in prompt_text else None)
                    require(repair == reviewer_repairs.get(review_key), "review_repair_not_from_failed_response")
                    require(proof["review_format_repair"] == (repair is not None), "review_repair_proof_mismatch")
                    candidate = response["text"]
                    merge_error = None
                    if repair is not None:
                        previous = reviewer_candidates[review_key]
                        merge_record = read(directory / "FORMAT_REPAIR_MERGE.json")
                        require(merge_record["previous_candidate_sha256"] == digest(previous), "review_merge_previous_mismatch")
                        require(merge_record["raw_repair_sha256"] == digest(response["text"]), "review_merge_response_mismatch")
                        require(merge_record["repair_spec_sha256"] == digest(repair), "review_merge_spec_mismatch")
                        try:
                            candidate = r.realization_review.merge_review_repair(previous, response["text"], repair)
                        except ValueError as error:
                            merge_error = str(error)
                            require(merge_record["status"] == "REJECT" and merge_record["reason"] == merge_error,
                                    "review_merge_rejection_mismatch")
                            rechecked = {"status": "REJECT", "schema_valid": False, "schema_failures": [merge_error],
                                         "failures": [merge_error], "parsed": None, "semantic_certified": False,
                                         "failure_category": "audit_incomplete"}
                            candidate = previous
                        else:
                            require(merge_record["status"] == "PASS" and merge_record["candidate"] == candidate,
                                    "review_merged_candidate_mismatch")
                        counters["review_repair_merge_replays"] += 1
                    else:
                        require(not (directory / "FORMAT_REPAIR_MERGE.json").exists(), "unexpected_review_merge")
                    if merge_error is None:
                        rechecked = r.realization_review.validate_review(candidate, root, route,
                            original_plan, raw_scene, dataset_card=context["dataset_card_guidance"],
                            latest_state=context["latest_state"], history=history, event_state=original_event)
                    require(rechecked == read(directory / "REVIEW.json"), "review_exact_citation_replay_changed")
                    reviewer_candidates[review_key] = candidate
                    if (directory / "FORMAT_REPAIR_SPEC.json").exists():
                        require(rechecked["schema_valid"] is False, "repair_after_valid_review")
                        specification = r.realization_review.build_format_repair(rechecked, candidate,
                            original_plan, history, raw_scene, dataset_card=context["dataset_card_guidance"],
                            latest_state=context["latest_state"], event_state=original_event)
                        require(specification == read(directory / "FORMAT_REPAIR_SPEC.json"), "review_repair_spec_changed")
                        reviewer_repairs[review_key] = specification
                    else:
                        reviewer_repairs.pop(review_key, None)
                    counters["review_validation_replays"] += 1
                    counters["review_soft_findings"] += len(rechecked.get("soft_findings", []))
                    counters["review_hard_rejections"] += rechecked["status"] == "REJECT"
                    counters["review_history_status_" + rechecked.get("history_consistency_status", "UNASSESSED")] += 1
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
                frozen = read(run / "sources" / f"{root}.json")["turns"]
                accepted_a_in_history = route == r.ROUTES[1] and any(
                    turn["turn_ref"].startswith("R02:A:") for turn in history[len(frozen):])
                require(proof["sequential_history_verified"] is True, "sequential_history_flag_missing")
                require(proof["accepted_A_in_history"] == accepted_a_in_history, "accepted_A_proof_mismatch")
                require(proof["other_branch_in_generator_context"] == accepted_a_in_history,
                        "accepted_A_generator_context_flag_mismatch")
                require(proof["frozen_base_sha256"] == digest(frozen), "request_frozen_base_mismatch")
                require("other_branch_added_turns_for_review_only" not in context
                        and "other_branch_accepted_additions" not in context, "comparison_in_generator_context")
                if role in ("state_extractor", "event_state_extractor"):
                    references = r.state_contract.history_reference_context(root, history)
                    require(all(context.get(key) == value for key, value in references.items()),
                            "history_reference_context_not_from_request_history")
                    counters["history_reference_context_bindings_verified"] += 1
                    repair_key = (slot, role, proof.get("plan_attempt") if role == "event_state_extractor" else None)
                    require(context.get("targeted_repair") == reference_repairs.get(repair_key),
                            "citation_repair_not_from_previous_failed_response")
                    if context.get("targeted_repair") is not None:
                        counters["citation_repair_bindings_verified"] += 1
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
                    except (ValueError, TypeError) as error:
                        require(saved["status"] == "REJECT", "state_rejection_replay_changed")
                        expected_repair = r.state_contract.build_state_repair(error, response["text"], history=history)
                        require(saved["reason"] == str(error) and saved["errors"] == expected_repair["errors"],
                                "state_rejection_errors_replay_changed")
                        reference_repairs[repair_key] = expected_repair
                    else:
                        require(saved["status"] == "PASS", "state_pass_replay_changed")
                        require(saved["latest_state_sha256"] == digest(snapshot), "state_response_hash_changed")
                        require(slots[slot]["latest_state"] == snapshot, "saved_state_differs_from_response")
                        state_outputs[slot] = snapshot
                        reference_repairs.pop(repair_key, None)
                    counters["state_validation_replays"] += 1
                if role == "event_state_extractor":
                    scene = context["scene_plan"]
                    require(proof["scene_plan_sha256"] == digest(scene), "event_scene_proof_mismatch")
                    require(plan_outputs.get(slot, {}).get("scenes") == [scene], "event_scan_not_current_candidate")
                    if response is not None:
                        saved = read(directory / "VALIDATION.json")
                        try:
                            if context.get("targeted_repair") is not None:
                                repair = context["targeted_repair"]
                                r.event_contract.validate_repair_preservation(repair["previous_raw"], response["text"], repair)
                                preservation = read(directory / "FORMAT_REPAIR_PRESERVATION.json")
                                require(preservation == {"status": "PASS",
                                    "previous_raw_sha256": digest(repair["previous_raw"]),
                                    "raw_repair_sha256": digest(response["text"]),
                                    "repair_spec_sha256": digest(repair),
                                    "unchanged_negative_findings_required": True}, "event_repair_preservation_changed")
                            event_snapshot = r.event_contract.validate(response["text"], root, route, scene, history)
                        except (ValueError, TypeError) as error:
                            require(saved["status"] == "REJECT", "event_rejection_replay_changed")
                            expected_repair = r.event_contract.build_repair(error, response["text"], history=history)
                            require(saved["reason"] == str(error) and saved["errors"] == expected_repair["errors"],
                                    "event_rejection_errors_replay_changed")
                            reference_repairs[repair_key] = expected_repair
                        else:
                            require(saved["status"] == "PASS", "event_pass_replay_changed")
                            require(read(directory / "EVIDENCE_PROVENANCE.json") ==
                                    r.evidence_refs.citation_provenance(r.event_contract._parse(response["text"]), history),
                                    "event_evidence_provenance_changed")
                            require(read(directory / "ROUTE_DECISION_PROVENANCE.json") ==
                                    r.event_contract.route_decision_provenance(
                                        r.event_contract._parse(response["text"])["route_fulfillment"], route),
                                    "event_route_decision_provenance_changed")
                            require(read(directory / "DERIVED_COVERAGE.json") ==
                                    r.event_contract.derive_coverage(event_snapshot, history),
                                    "derived_event_coverage_changed")
                            require(saved["event_state_sha256"] == digest(event_snapshot), "event_response_hash_changed")
                            require(saved["requires_replan"] == r.event_contract.requires_replan(event_snapshot),
                                    "event_replan_replay_changed")
                            event_outputs[slot] = event_snapshot
                            event_plan_hashes[slot] = digest(scene)
                            reference_repairs.pop(repair_key, None)
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
                        require(read(directory / "DUPLICATE_CHECK.json")["status"] == "PASS",
                                "accepted_exact_duplicate_scene")
                        require(r.parse_object(response["text"]) == final_realizations[slot], "accepted_realization_response_changed")
                        review_dir = Path(accepted["review_request"])
                        require(read(review_dir / "REVIEW.json")["status"] == "PASS", "accepted_without_positive_review")
                        review_context = decode_context(read(review_dir / "REQUEST.json"), "\nREVIEW INPUT DATA:\n")
                        require(review_context["new_scene"] == r.realization_review.review_scene(final_realizations[slot]),
                                "accepted_review_for_other_realization")
                        final_review = read(review_dir / "REVIEW.json")
                        require(r.event_contract.route_fulfilled(final_review["parsed"]["route_fulfillment"], route),
                                "accepted_raw_review_route_not_fulfilled")
                        scene_metadata = next(item for row in report["routes"]
                            if row["root_id"] == root and row["route"] == route
                            for item in row["scene_metadata"] if item["scene_id"] == scene_id)
                        require(scene_metadata["route_fulfillment"] == final_review["parsed"]["route_fulfillment"],
                                "route_verdict_not_bound_to_raw_Qwen")
                        accepted_requests[slot] = str(directory)
            require(history == slots[slot]["history"], "request_history_differs_from_accepted_chain")
            require(proof["history_sha256"] == digest(history), "request_history_hash_mismatch")
            require(proof["history_turns"] == len(history), "request_history_count_mismatch")
            require(context["root_id"] == root and context["route"] == route, "context_identity_mismatch")
            if role != "state_extractor":
                require(context["latest_state"] == slots[slot]["latest_state"], "snapshot_pipeline_mismatch")
                require(proof["latest_state_sha256"] == digest(context["latest_state"]), "snapshot_proof_mismatch")
                require(state_outputs.get(slot) == context["latest_state"], "state_not_from_earlier_success_response")
                counters["snapshot_pipeline_links_verified"] += 1
            if role in ("planner", "realizer", "scene_reviewer"):
                event_snapshot = slots[slot]["event_state"] if role == "scene_reviewer" else context.get("event_state")
                require(proof["event_state_sha256"] == digest(event_snapshot), "event_proof_mismatch")
                if role == "scene_reviewer":
                    require(context["event_state"] == r.realization_review.review_event_context(event_snapshot),
                            "review_event_context_projection_mismatch")
                    require(proof["projected_event_state_sha256"] == digest(context["event_state"]),
                            "review_event_context_hash_mismatch")
                if event_snapshot is not None:
                    require(event_outputs.get(slot) == event_snapshot, "event_not_from_previous_scan")
                    counters["event_pipeline_links_verified"] += 1
            if role in ("realizer", "scene_reviewer"):
                original_plan = slots[slot]["plan"]["scenes"][0]
                expected_plan = (r.realization_review.review_plan_context(original_plan)
                                 if role == "scene_reviewer" else original_plan)
                require(context["scene_plan"] == expected_plan, "scene_plan_pipeline_mismatch")
                if role == "scene_reviewer":
                    require(proof["scene_plan_sha256"] == digest(original_plan), "review_original_plan_hash_mismatch")
                    require(proof["projected_scene_plan_sha256"] == digest(expected_plan), "review_plan_context_hash_mismatch")
                else:
                    require(context["event_state"] == slots[slot]["event_state"], "final_event_pipeline_mismatch")
                require(event_plan_hashes.get(slot) == digest(original_plan), "final_event_scan_for_other_plan")
                require(not r.event_contract.requires_replan(slots[slot]["event_state"]), "realized_plan_required_revision")
        stage("request:" + directory.name, audit_request)

    def merged_roots():
        merge = read(run / "MERGE_MANIFEST.json")
        require(merge["status"] == "PASS", "merge_not_complete")
        require(merge["sequence"] == "frozen_base -> accepted_A -> accepted_B", "merge_sequence_mismatch")
        r.base.verify_bindings(merge["files"])
        require({row["root_id"] for row in merge["source_rows"]} == set(manifest["roots"]), "merge_root_set_mismatch")
        for source_row in merge["source_rows"]:
            root = source_row["root_id"]
            path = Path(source_row["path"])
            require(path == run / "sources_merged" / f"{root}.json", "merged_source_path_changed")
            require(digest(path.read_bytes()) == source_row["sha256"], "merged_source_hash_mismatch")
            source = read(path)
            index = read(run / "roots" / root / "BRANCHES.json")
            require(index["sequential_routes"] is True and index["independent_branches"] is False,
                    "old_separate_branch_policy")
            require(index["downstream_requires_merged_root_source"] is True, "downstream_not_merged")
            require(index["merged_dialogue"] == str(path) and index["merged_sha256"] == source_row["sha256"],
                    "merge_index_binding_mismatch")
            frozen = read(run / "sources" / f"{root}.json")["turns"]
            history = list(frozen)
            rows = [row for row in report["routes"] if row["root_id"] == root]
            for route in r.ROUTES:
                route_rows = [row for row in rows if row["route"] == route]
                for row in route_rows:
                    for metadata in row["scene_metadata"]:
                        realization = final_realizations[(root, route, metadata["scene_id"])]
                        history = r.base.append_scene(history, root, route, realization)
            require(source["turns"] == history, "merged_source_not_exact_accepted_chain")
            require(len({turn["turn_ref"] for turn in history}) == len(history), "merged_source_duplicate_ids")
            require(source["base_occurrences"] == 1, "merged_source_base_count_mismatch")
            require(source["accepted_scenes"] == sum(row["accepted_scenes"] for row in rows) ==
                    source_row["accepted_scenes"], "merged_source_scene_count_mismatch")
            for letter, route in zip(("A", "B"), r.ROUTES):
                require(source_row[f"accepted_{letter}_scenes"] ==
                        sum(row["accepted_scenes"] for row in rows if row["route"] == route),
                        "merged_route_scene_count_mismatch")
            counters["merged_roots_verified"] += 1
    stage("one_root_merged_context_and_downstream_bindings", merged_roots)

    def totals():
        require(sum(roles.values()) == report["http_request_reservations"], "logical_request_count_mismatch")
        require(len(numbers) == report["http_attempts"] == report["http_attempt_log_count"], "http_count_mismatch")
        require(sorted(numbers) == list(range(1, len(numbers) + 1)), "global_http_sequence_invalid")
        require(sum(statuses.values()) == len(numbers), "http_status_count_mismatch")
        require(len(numbers) <= manifest["http_max"], "http_budget_exceeded")
        require(set(accepted_requests) == set(final_realizations), "accepted_output_missing_successful_request")
        require(report["question_calls"] == report["solver_calls"] == 0 and report["training_started"] is False,
                "unexpected_downstream_activity")
        require(report["branches_remain_separate"] is False, "report_old_branch_policy")
    stage("request_totals_and_accepted_output_binding", totals)
    result = {
        "audit_version": "r02_sequential_AB_v23_independent_artifact_audit_v1",
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
        "feedback_evaluation_run": feedback_run,
        "limitation": "Artifact and exact-quote checks only; Qwen state extraction and semantic verdicts remain fallible. Shortfall or negative reviews do not by themselves fail this audit.",
    }
    r.base.write(run / "FINAL_AUDIT.json", result)
    write_summary(run, result, report, count_tokens=count_tokens)
    return result


def write_summary(run, audit, report, *, count_tokens=False):
    tokenizer = None
    if count_tokens:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(r.base.TOKENIZER), local_files_only=True)
    rows = []
    for root in audit["roots"]:
        frozen = r.base.read(run / "sources" / f"{root}.json")["turns"]
        final = r.base.read(run / "sources_merged" / f"{root}.json")["turns"]
        row = {"root_id": root, "original_body_turns": len(frozen),
               "final_body_turns": len(final), "added_turns": len(final) - len(frozen)}
        for letter, route in zip(("A", "B"), r.ROUTES):
            result = next((item for item in report["routes"] if item["root_id"] == root and item["route"] == route), None)
            requests, role_checks = [], []
            for directory in sorted((run / "requests").iterdir()):
                proof = r.base.read(directory / "PROOF.json")
                response = directory / "RESPONSE.json"
                if proof.get("root_id") == root and proof.get("route") == route and response.exists():
                    try:
                        parsed = r.parse_object(r.base.read(response)["text"])
                    except (ValueError, TypeError):
                        continue
                    if proof["role"] == "realizer" and parsed.get("status") == "scene":
                        requests.append(directory.name)
                    if isinstance(parsed.get("route_fulfillment"), dict):
                        role_checks.append({"request": str(directory), "stage": proof["role"],
                                            "scene_id": proof["scene_id"], "verdict": parsed["route_fulfillment"]})
            metadata = result["scene_metadata"] if result else []
            row[letter] = {"generated_scene_responses": len(requests),
                          "accepted_scenes": result["accepted_scenes"] if result else 0,
                          "added_turns": result["added_turns"] if result else 0,
                          "route_verdicts": [item["route_fulfillment"] for item in metadata],
                          "all_role_checks": role_checks,
                          "failures": result["failures"] if result else ["route_not_requested"]}
        if tokenizer:
            count = lambda turns: sum(len(tokenizer.encode(turn["text"], add_special_tokens=False)) for turn in turns)
            a_n = row["A"]["added_turns"]
            row["body_tokens"] = {"frozen": count(frozen), "A_added": count(final[len(frozen):len(frozen)+a_n]),
                                  "B_added": count(final[len(frozen)+a_n:]), "final": count(final)}
        rows.append(row)
    summary = {"audit_status": audit["status"], "model": audit["model"], "rows": rows,
               "tokenizer": str(r.base.TOKENIZER) if tokenizer else None,
               "token_scope": "sum of per-turn text only; no speaker labels, IDs, prompts, or feedback",
               "generation_count_scope": "raw scene responses, including retries; not unique accepted scenes",
               "training_started": False, "question_calls": 0, "solver_calls": 0}
    r.base.write(run / "SEQUENTIAL_SUMMARY.json", summary)
    lines = [f"模型：{audit['model']}。独立产物审计：{audit['status']}。", "",
             "生成数为返回场景的次数（含修复）；通过数为最终保留场景数。", "",
             "| root | A 生成 / 通过 | B 生成 / 通过 | 新增回合 | A/B 判定 |",
             "|---|---:|---:|---:|---|"]
    for row in rows:
        roles = []
        for letter in ("A", "B"):
            item = row[letter]
            reasons = [verdict["reason"] for verdict in item["route_verdicts"]]
            if not reasons and item["all_role_checks"]:
                verdict = item["all_role_checks"][-1]["verdict"]
                reasons = [f"{verdict.get('relation_to_history')}: {verdict.get('reason')}"]
            if not reasons:
                reasons = [str(failure.get("reason", failure)) if isinstance(failure, dict) else failure for failure in item["failures"]]
            roles.append(letter + ": " + "; ".join(reasons))
        lines.append(f"| {row['root_id'].rsplit('-', 1)[-1]} | {row['A']['generated_scene_responses']} / {row['A']['accepted_scenes']} | "
                     f"{row['B']['generated_scene_responses']} / {row['B']['accepted_scenes']} | {row['added_turns']} | " + "；".join(roles).replace("|", "/").replace("\n", " ") + " |")
    if tokenizer:
        lines += ["", "Token 仅计逐回合正文，原文累计上下文只计一次。", "",
                  "| root | 冻结原文 | A 新增 | B 新增 | 累计正文 |", "|---|---:|---:|---:|---:|"]
        for row in rows:
            values = row["body_tokens"]
            lines.append(f"| {row['root_id'].rsplit('-', 1)[-1]} | {values['frozen']} | {values['A_added']} | {values['B_added']} | {values['final']} |")
    lines += ["", "Qwen 同模型判定仍可能出错；产物审计验证调用、证据与累计上下文一致性，不宣称语义绝对正确。未启动出题、Solver 或训练。"]
    (run / "SEQUENTIAL_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--count-tokens", action="store_true")
    args = parser.parse_args()
    result = audit_run(args.run, count_tokens=args.count_tokens)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
