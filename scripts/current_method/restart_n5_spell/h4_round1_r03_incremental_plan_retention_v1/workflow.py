"""Explicit new-policy execution seam; no imports of live/Memory backends.

The same request construction runs with a fixture transport offline and a
single-shot explicitly authorized transport live. Old execute_round is untouched.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import POLICY_ID
from .budget import BudgetLedger, validate_budget
from .bundle import validate_bundle_feedback
from .capacity import local_counter, measure, future_audit_reserve, VERSION
from .planning import prepare_original, merge_patch, sha, PROFILE
from .prompts import render_supplement, render_gate, render_realizer
from .supplement_contract_alignment import select_supplement_version
from .root_scope import (is_root_scope, prepare_bundle, render_for_bundle, merge_for_bundle,
                         supplement_format_version)
from .retention import ADHERENCE_VERSION, render_adherence, assess_retention, dependencies_satisfied, local_fact_input
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    validate_semantic_gate_output, normalize_realizer_output, validate_event_binding_alignment,
    validate_planner_outline_distinctness,
)
from ..simple_append_v1.assembly import append_expansion_round, validate_appended_round


def conservative_tokens(value):
    """UTF-8 bytes upper bound for the frozen byte-level tokenizer, not an estimate.

    Real transport must declare a verified tokenizer identity. No truncation.
    """
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")) + 256


def prepare_saved_resume(bundle, saved_patch, *, normalization_version, source_identity=None):
    from .plan_normalization import VERSION as NORMALIZATION_VERSION, merged_queue, normalize_queue
    from .task_selection import VERSION as SELECTION_VERSION, select_same_task
    if not is_root_scope(bundle) or normalization_version not in {NORMALIZATION_VERSION, SELECTION_VERSION}:
        raise ValueError("explicit_root_resume_normalization_version_required")
    prepared = prepare_bundle(bundle)
    merged = merge_for_bundle(bundle, prepared, saved_patch)
    result = {"prepared": prepared, "merged": merged, "normalization_version": normalization_version}
    if not merged["valid"]:
        return {**result, "valid": False, "status": "SAVED_PATCH_INVALID_STOP"}
    queue = merged_queue(prepared, merged)
    result["unnormalized_queue"] = queue
    if normalization_version == SELECTION_VERSION:
        normalization = select_same_task(prepared, merged, source_identity=source_identity)
    else:
        normalization = normalize_queue(queue, version=normalization_version)
    normalization["source_identity"] = source_identity
    result["normalization"] = normalization
    if not normalization["valid"]:
        return {**result, "valid": False, "status": "PLAN_NORMALIZATION_CONFLICT_STOP"}
    outlines = validate_planner_outline_distinctness({"scenes": [e["gate_scene"] for e in normalization["queue"]]})
    result["outlines"] = outlines
    return {**result, "valid": outlines["valid"],
            "status": "SAVED_PATCH_RESUME_OFFLINE_READY" if outlines["valid"] else "NORMALIZED_OUTLINE_REJECT_STOP"}


def execute_policy(*, bundle, budget, backend, state_registry, artifact_dir, live=False,
                   experiment_scope=None, saved_patch=None, resume_context=None,
                   normalization_version=None, gate_reuse=None,
                   fresh_normalization_version=None):
    root_mode = is_root_scope(bundle)
    if fresh_normalization_version is not None:
        from .task_selection import VERSION as SELECTION_VERSION
        if (fresh_normalization_version != SELECTION_VERSION or not root_mode
                or saved_patch is not None or resume_context is not None
                or normalization_version is not None or gate_reuse is not None):
            raise ValueError("fresh_selection_requires_explicit_fresh_root_entry")
    if live and not getattr(backend, "single_shot_authorized", False):
        raise ValueError("real_execution_requires_authorized_single_shot_transport")
    if not live and not getattr(backend, "offline_mock", False):
        raise ValueError("offline_entry_requires_mock_no_real_transport")
    feedback_check = validate_bundle_feedback(bundle)
    budget = validate_budget(budget)
    if budget["role_requests"]["planner"] != 0:
        raise ValueError("frozen_plan_entry_requires_zero_new_initial_planner_budget")
    counter = getattr(backend, "counter", None) or local_counter()
    parent = deepcopy(bundle["parent"])
    if bundle["identity"]["root_id"] != parent["root_id"] or bundle["identity"]["round_index"] != 3:
        raise ValueError("root_round_identity")
    if not bundle["identity"].get("branch_id") or bundle["history_sha256"] != sha(parent):
        raise ValueError("branch_history_identity")
    if bundle["original_plan_sha256"] != sha(bundle["original_plan"]):
        raise ValueError("frozen_plan_sha_mismatch")
    source = bundle["frozen_sources"]["history"]
    prior_bytes = Path(source["path"]).read_bytes()
    if hashlib.sha256(prior_bytes).hexdigest() != source["sha256"]:
        raise ValueError("frozen_prior_file_sha_mismatch")
    prior_expanded = json.loads(prior_bytes)
    if bundle.get("retained_r03_turns") != []:
        raise ValueError("pre_gate_entry_requires_empty_retained_r03")
    if any(str(t.get("turn_id", "")).startswith(("R03:", "R3:")) for t in parent["dialogue"]["turns"]):
        raise ValueError("old_r03_not_allowed")
    prepared = prepare_bundle(bundle)
    resuming = saved_patch is not None
    reusing_gate = gate_reuse is not None
    if reusing_gate and (not resuming or experiment_scope is None):
        raise ValueError("gate_reuse_requires_explicit_same_experiment_resume")
    if resuming != (resume_context is not None) or (not resuming and normalization_version is not None):
        raise ValueError("complete_explicit_saved_patch_resume_required")
    resume_preflight = None
    if resuming:
        if (experiment_scope is None or resume_context.get("experiment_scope") != experiment_scope
                or resume_context.get("identity") != bundle["identity"]
                or resume_context.get("patch_sha256") != sha(saved_patch)):
            raise ValueError("saved_patch_resume_identity_mismatch")
        resume_preflight = prepare_saved_resume(bundle, saved_patch,
            normalization_version=normalization_version, source_identity=resume_context["source_identity"])
        if not resume_preflight["valid"]:
            # A rejected normalization is read-only with respect to call state.
            output = Path(artifact_dir)
            output.mkdir(parents=True, exist_ok=False)
            result = {"status": resume_preflight["status"], "policy_id": bundle["policy_id"],
                "identity": bundle["identity"], "execution_scope": experiment_scope,
                "resume_context": resume_context, "external_call_count": 0, "mock_call_count": 0,
                "new_calls": [], "attempted_request_counts": {role: 0 for role in budget["role_requests"]},
                "retained_scene_count": 0, "retained_turn_count": 0, "offline_fixture_only": not live,
                "resume_claimed": False, "supplement_previously_consumed": True,
                "remote_memory_calls": 0, "other_model_calls": 0, "execution_ended": True}
            for name, value in (("resume_preflight", resume_preflight), ("final_report", result)):
                with (output / (name + ".json")).open("x", encoding="utf-8") as handle:
                    json.dump(value, handle, ensure_ascii=False, indent=2)
            return result
        if reusing_gate:
            from .gate_result_resume import VERSION as GATE_REUSE_VERSION
            queue = resume_preflight["normalization"]["queue"]
            if (gate_reuse.get("version") != GATE_REUSE_VERSION
                    or gate_reuse.get("selected_queue_sha256") != sha(queue)
                    or gate_reuse.get("history_sha256") != bundle["history_sha256"]
                    or gate_reuse.get("manifest_sha256") != resume_context["source_identity"]["source_manifest_sha256"]):
                raise ValueError("gate_reuse_workflow_binding_mismatch")
            entries = {entry["scene"]["phase_id"]: entry for entry in queue}
            for phase, record in gate_reuse["by_phase"].items():
                if phase not in entries or record["entry_sha256"] != sha(entries[phase]):
                    raise ValueError("gate_reuse_phase_binding_mismatch")
    supplement_prompt = render_for_bundle(bundle, budget, prepared)
    needs_supplement = root_mode or bool(prepared["goal_diagnostics"])
    capacity_checks = []
    def checked(role, prompt, label):
        result = measure(prompt["messages"], counter=counter, budget=budget, role=role)
        capacity_checks.append({"label": label, **result})
        return result["fits"]
    # Known inability to reach generation/audit must fail before creating a
    # registry, claiming the run, or consuming the single supplement attempt.
    if prepared["valid_for_new_policy"]:
        first_entries = list(prepared["candidate_registry"].items())[:2]
        first_scenes = []
        for number, (ref, entry) in enumerate(first_entries, 1):
            scene = deepcopy(entry["scene"])
            scene.update(phase_id="initial:" + ref, source_candidate_id=ref,
                         dependencies=deepcopy(scene.get("dependencies", [])), scene_ref=f"R03:S{number:02d}")
            first_scenes.append(scene)
            if not reusing_gate:
                checked("gate", render_gate(parent=parent, gate_scene=entry["gate_scene"], candidates=bundle["candidates"]), "before_claim_gate")
        first_plan = {"scenes": first_scenes}
        checked("realizer", render_realizer(parent=parent, plan=first_plan, mode=bundle["realizer_mode"],
                continuity_shape=bundle.get("continuity_shape", {})), "before_claim_realizer")
        capacity_checks.append({"label": "before_claim_audit_reserve", **future_audit_reserve(parent=parent,
            plan=first_plan, conditions={}, budget=budget, counter=counter)})
        if needs_supplement and not resuming:
            checked("supplement", supplement_prompt, "before_claim_supplement")
        if not all(row["fits"] for row in capacity_checks):
            raise ValueError("known_capacity_blocked_before_attempt:" + json.dumps([
                {"label": r["label"], "excess_tokens": r["excess_tokens"]} for r in capacity_checks if not r["fits"]]))
        if counter.dialogue_tokens(parent["dialogue"]["turns"]) > budget["max_cumulative_tokens"]:
            raise ValueError("known_cumulative_length_blocked_before_attempt")
        minimum = {"closure": 1, "supplement": int(needs_supplement),
                   "gate": len(first_scenes), "realizer": 1, "adherence": 1}
        if (any(budget["role_requests"][r] < n for r, n in minimum.items())
            or sum(minimum.values()) > budget["total_requests"]
            or sum(budget["max_output_tokens"][r] * n for r, n in minimum.items()) > budget["max_generated_tokens"]):
            raise ValueError("known_first_batch_budget_blocked_before_attempt")
    # Experiment-only local rejections must precede a durable execution claim.
    if experiment_scope is not None:
        if not prepared["valid_for_new_policy"]:
            raise ValueError("experiment_original_plan_invalid_before_claim")
        alignment = validate_event_binding_alignment(prepared["compiled"], candidates=bundle["candidates"])
        if not alignment["valid"]:
            raise ValueError("experiment_binding_invalid_before_claim")
    if resuming:
        key = hashlib.sha256(json.dumps({"experiment_id": experiment_scope["experiment_id"]}, sort_keys=True).encode()).hexdigest()
        existing_ledger = (Path(state_registry) / (key + ".jsonl")).resolve()
        if (str(existing_ledger) != resume_context["source_ledger_path"] or not existing_ledger.is_file()
                or hashlib.sha256(existing_ledger.read_bytes()).hexdigest() != resume_context["source_ledger_sha256"]):
            raise ValueError("resume_requires_unchanged_existing_experiment_ledger")
    ledger = BudgetLedger(state_registry, root_id=parent["root_id"], round_index=3, budget=budget,
                          execution_scope=experiment_scope)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        path = output / (name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
    save("input_freeze", bundle)
    save("execution_budget", budget)
    if experiment_scope is not None:
        save("experiment_scope", experiment_scope)
    save("feedback_source_to_bundle_check", feedback_check)
    save("supplement_input_projection_mapping", supplement_prompt["input_projection_mapping"])
    supplement_identity = supplement_prompt.get("supplement_contract_identity")
    if supplement_identity is not None:
        save("supplement_contract_identity", supplement_identity)
    if live:
        # Configuration contains only the credential environment VARIABLE NAME,
        # never its value; headers are not saved.
        save("model_config", backend.config)
    if resuming:
        save("resume_context", resume_context)
        save("resume_preflight", resume_preflight)
        if reusing_gate:
            generation_slots = ledger.claim_gate_result_resume(
                {"identity": bundle["identity"], "resume_context": resume_context,
                 "gate_reuse_manifest_sha256": gate_reuse["manifest_sha256"]},
                source_ledger_sha256=resume_context["source_ledger_sha256"],
                source_ledger_header=resume_context["source_ledger_header"])
            save("formal_gate_reuse_records", gate_reuse["formal_reuse_records"])
            save("gate_reuse_generation_slots", generation_slots)
        else:
            ledger.claim_resume({"identity": bundle["identity"], "resume_context": resume_context},
                source_ledger_sha256=resume_context["source_ledger_sha256"],
                source_ledger_header=resume_context["source_ledger_header"])
    else:
        ledger.claim_execution(bundle["identity"])
    calls = []
    retained_scenes = []
    states = {}
    decisions = []
    goal_register = {}
    received_outputs = {}
    profile = {**PROFILE, "name": bundle["policy_id"], "incremental_scope": bundle["incremental_scope"]} if root_mode else PROFILE
    report = {"policy_id": bundle["policy_id"], "profile": profile, "identity": bundle["identity"],
              "readiness_version": VERSION, "capacity_checks": capacity_checks,
              "offline_fixture_only": not live, "historical_initial_planner_calls": 1,
              "new_calls": calls, "scene_decisions": decisions,
              "remote_memory_calls": 0, "other_model_calls": 0,
              "derived_scene_safety_limit": ledger.derived_scene_safety_limit()}
    if supplement_identity is not None:
        report["supplement_contract_identity"] = supplement_identity
    if experiment_scope is not None:
        report.update(execution_scope=experiment_scope, extra_version_validation=True,
                      first_generation_result=False, automatic_training_admission=False,
                      persistent_ledger_path=str(ledger.path))
    if resuming:
        report.update(resume_context=resume_context, resume_claimed=True,
                      normalization_version=normalization_version, supplement_previously_consumed=True)
    if fresh_normalization_version is not None:
        report.update(fresh_normalization_version=fresh_normalization_version,
                      normalization_version=fresh_normalization_version)
    if reusing_gate:
        report.update(gate_result_reuse=True, gate_reuse_manifest_sha256=gate_reuse["manifest_sha256"],
                      loaded_gate_result_count=len(gate_reuse["by_phase"]), used_gate_result_phases=[],
                      reused_gate_snapshot_sha256=bundle["history_sha256"],
                      reused_gate_judgments_are_not_semantic_truth=True)
    def finish(status):
        journal = ledger.snapshot()
        attempted_counts = {r: sum(c["role"] == r for c in calls) for r in budget["role_requests"]}
        reserved_counts = {r: sum(row.get("requests", {}).get(r, 0) for row in journal if row["kind"] == "reserve") for r in budget["role_requests"]}
        report.update(status=status, execution_ended=True, retained_scene_count=len(retained_scenes),
                      retained_turn_count=sum(len(s["dialogue_turns"]) for s in retained_scenes),
                      goal_statuses={k: v["goal_status"] for k, v in states.items()},
                      original_goals_and_phase_statuses=[{**item, "phase_id": phase,
                          "goal_status": states.get(phase, {}).get("goal_status", "UNKNOWN"),
                          "retained": states.get(phase, {}).get("retained", False)} for phase, item in goal_register.items()],
                      all_goals_complete=bool(goal_register) and all(states.get(phase, {}).get("goal_status") == "realized" and states.get(phase, {}).get("retained") for phase in goal_register),
                      external_call_count=len(calls) if live else 0,
                      mock_call_count=0 if live else len(calls),
                      attempted_request_counts=attempted_counts,
                      reserved_request_counts=reserved_counts,
                      unused_reserved_request_counts={r: reserved_counts[r]-attempted_counts[r] for r in attempted_counts},
                      untested=["real model quality", "server 8192 acceptance", "audit semantic accuracy"])
        if resuming:
            report["experiment_total_attempted_counts"] = {role: sum(row.get("kind") == "attempt"
                and row.get("role") == role for row in journal) for role in budget["role_requests"]}
            report["unused_reserved_request_counts"] = {role: reserved_counts[role]
                - report["experiment_total_attempted_counts"][role] for role in reserved_counts}
        save("local_retained_dialogue_fixture" if not live else "local_retained_dialogue", local_fact_input(retained_scenes))
        if live and retained_scenes:
            cumulative = append_expansion_round(prior_expanded=prior_expanded,
                child={"root_id": parent["root_id"], "round_index": 3, "scenes": retained_scenes})
            save("local_cumulative_dialogue", {"policy_id": bundle["policy_id"], "identity": bundle["identity"],
                "source_history_sha256": bundle["history_sha256"], "complete_dialogue": cumulative,
                "snapshot_sha256": sha({"identity": bundle["identity"], "complete_dialogue": cumulative})})
        save("ledger_snapshot", journal)
        save("final_report", report)
        return report
    def request(role, prompt, reservation):
        if reusing_gate and role not in {"realizer", "adherence"}:
            raise ValueError("gate_result_resume_only_realizer_and_adherence_may_send")
        if resuming and role in {"planner", "supplement"}:
            raise ValueError("resume_never_sends_planner_or_supplement")
        if not checked(role, prompt, "before_send_" + role):
            raise ValueError("actual_final_messages_capacity_exceeded_no_send")
        input_count = capacity_checks[-1]["known_final_messages_tokens"]
        call_id = ledger.begin(reservation, role, input_tokens=input_count)
        request = {"role": role, "call_id": call_id, "messages": prompt["messages"],
                   "max_tokens": budget["max_output_tokens"][role],
                   "prompt_version": prompt["prompt_version"]}
        request.update(identity=bundle["identity"], capacity=capacity_checks[-1])
        if experiment_scope is not None:
            request["execution_scope"] = experiment_scope
        if resuming:
            request["resume_source_identity"] = resume_context["source_identity"]
            request["plan_normalization_version"] = normalization_version
        elif fresh_normalization_version is not None:
            request["plan_normalization_version"] = fresh_normalization_version
        if role == "supplement" and supplement_identity is not None:
            request["supplement_contract_identity"] = supplement_identity
        if role == "gate":
            request["gate_contract_identity"] = deepcopy(prompt["gate_contract_identity"])
        if live:
            request.update(model=backend.config["model"], temperature=backend.config["temperature"],
                           timeout_seconds=backend.config["timeout_seconds"], thinking=False)
        save(f"requests/{call_id:03d}_{role}", request)
        row = {"role": role, "call_id": call_id, "messages_sha256": sha(prompt["messages"]), "status": "ATTEMPTED"}
        if experiment_scope is not None:
            row["execution_scope"] = experiment_scope
        calls.append(row)
        def evidence_sink(evidence):
            # Each transport event is saved before its associated parsing step.
            event = evidence.get("evidence_event", "response")
            path = f"responses/{call_id:03d}_{role}_http_{event}"
            if not (output / (path + ".json")).exists():
                save(path, evidence)
        backend.response_evidence_sink = evidence_sink
        try:
            result = backend.request(request)
            save(f"responses/{call_id:03d}_{role}", result)
            envelope = getattr(backend, "last_response_envelope", None)
            if envelope is not None:
                save(f"responses/{call_id:03d}_{role}_envelope", envelope)
            formatting_raw = role == "supplement" and supplement_format_version(bundle) is not None
            if formatting_raw:
                # The adapter parses the original text, including duplicate-key checks.
                parsed = result
            elif role == "supplement" and prompt.get("planner_contract_identity"):
                from .shared_planner_contract import parse_output
                parsed = parse_output(result) if isinstance(result, str) else result
            else:
                parsed = json.loads(result) if isinstance(result, str) else result
            if not isinstance(parsed, dict) and not (formatting_raw and isinstance(parsed, str)):
                raise ValueError("response_object_required")
            received_outputs[role] = result
        except Exception as exc:
            # No secret exception messages from transports are persisted.
            row["status"] = "TECHNICAL_OR_PARSE_FAILURE_NO_RETRY"
            envelope = getattr(backend, "last_response_envelope", None)
            if envelope is not None and not (output / f"responses/{call_id:03d}_{role}_envelope.json").exists():
                save(f"responses/{call_id:03d}_{role}_envelope", envelope)
            evidence = getattr(backend, "last_response_evidence", None)
            if evidence is not None:
                save(f"failures/{call_id:03d}_http_evidence", evidence)
            save(f"failures/{call_id:03d}", {"exception_type": type(exc).__name__, "no_retry": True,
                 "response_unavailable": evidence is None,
                 "classification": "BUSINESS_JSON_PARSE_OR_TRANSPORT_FAILURE"})
            ledger.finish(call_id, status=row["status"])
            raise
        row["status"] = "RESPONSE_RECEIVED"
        ledger.finish(call_id, status=row["status"])
        return parsed
    def save_formatting(merged):
        if "formatting" in merged:
            save("supplement_formatting", merged["formatting"])
        if "formatted_patch" in merged:
            save("formatted_patch", merged["formatted_patch"])
    save("original_plan_new_policy_diagnostics", prepared)
    if not prepared["valid_for_new_policy"]:
        return finish("BLOCKED_HARD_PLAN_ERRORS")
    compiled = prepared["compiled"]
    alignment = validate_event_binding_alignment(compiled, candidates=bundle["candidates"])
    save("binding_alignment", alignment)
    if not alignment["valid"]:
        return finish("BLOCKED_EVENT_BINDING")
    cumulative = counter.dialogue_tokens(parent["dialogue"]["turns"])
    # Reserve necessary closeout capacity before any supplement/generation.
    if not resuming and not ledger.reserve("closure", {"closure": 1}, cumulative_tokens=cumulative):
        return finish("DEFERRED_BUDGET")
    added = []
    if resuming:
        merged = resume_preflight["merged"]
        save("saved_patch", saved_patch)
        save("increment_patch_and_diff", merged)
        save_formatting(merged)
        added = merged["merged"]["added_phases"]
    elif needs_supplement:
        if not ledger.reserve("supplement", {"supplement": 1}, cumulative_tokens=cumulative):
            return finish("DEFERRED_BUDGET")
        try:
            patch = request("supplement", supplement_prompt, "supplement")
        except Exception:
            return finish("SUPPLEMENT_FAILED_STOP_NO_RETRY")
        merged = merge_for_bundle(bundle, prepared, patch)
        save("increment_patch_and_diff", merged)
        save_formatting(merged)
        if not merged["valid"]:
            return finish("SUPPLEMENT_INVALID_STOP")
        if not root_mode and patch["status"] == "NO_SUPPLEMENT":
            return finish("NO_SUPPLEMENT_STOP")
        added = merged["merged"]["added_phases"]
    initial = []
    for ref, entry in prepared["candidate_registry"].items():
        entry = deepcopy(entry)
        entry["scene"].update(phase_id="initial:" + ref, source_candidate_id=ref,
                              dependencies=deepcopy(entry["scene"].get("dependencies", [])))
        initial.append(entry)
    queue = initial + added
    if resuming or fresh_normalization_version is not None:
        save("unnormalized_execution_queue", queue)
        if resuming:
            normalization = resume_preflight["normalization"]
        else:
            from .task_selection import select_same_task
            normalization = select_same_task(prepared, merged, source_identity={
                "entry": "fresh_root_after_supplement", "identity": bundle["identity"],
                "original_plan_sha256": bundle["original_plan_sha256"],
                "supplement_response_sha256": sha(merged["raw_patch"]),
            })
        save("plan_normalization", normalization)
        if not normalization["valid"]:
            return finish("PLAN_NORMALIZATION_CONFLICT_STOP")
        for item in normalization.get("not_selected", []) + normalization.get("deferred", []):
            decisions.append(deepcopy(item))
        report["selection_summary"] = {key: normalization.get(key, [])
            for key in ("aliases", "not_selected", "deferred")}
        queue = deepcopy(normalization["queue"])
        initial = [entry for entry in queue if entry["scene"]["phase_id"].startswith("initial:")]
        added = [entry for entry in queue if not entry["scene"]["phase_id"].startswith("initial:")]
        save("normalized_execution_queue", queue)
    goal_register.update({e["scene"]["phase_id"]: {"original_goal": e["scene"]["goal"],
                         "original_planned_target": e["scene"]["after_state"],
                         "source_candidate_id": e["scene"]["source_candidate_id"]} for e in queue})
    if root_mode:
        for entry in queue:
            scene = entry["scene"]
            row = goal_register[scene["phase_id"]]
            row["goal_origin"] = scene.get("goal_origin", "original_candidate_goal")
            if scene.get("increment_type") == "independent_new_development":
                row["new_goal"] = row.pop("original_goal")
                row["new_planned_target"] = row.pop("original_planned_target")
                row["materialized_candidate_id"] = row.pop("source_candidate_id")
    outlines = validate_planner_outline_distinctness({"scenes": [e["gate_scene"] for e in queue]})
    save("merged_outline_check", outlines)
    if not outlines["valid"]:
        return finish("MERGED_OUTLINE_REJECT_STOP")
    expected_conditions = {}
    for entry in queue:
        for dep in entry["scene"]["dependencies"]:
            values = expected_conditions.setdefault(dep["phase_id"], [])
            if dep["condition"] not in values:
                values.append(dep["condition"])
    initial_selected = 0
    scene_number = 0
    cursor = 0
    generation_slot_index = 0
    while cursor < len(queue):
        batch = []
        while cursor < len(queue) and len(batch) < 2:
            entry = queue[cursor]
            is_initial = entry["scene"]["phase_id"].startswith("initial:")
            if is_initial and initial_selected + len(batch) >= 6:
                decisions.append({"phase_id": entry["scene"]["phase_id"], "status": "UNSELECTED_INITIAL_LIMIT"})
                cursor += 1
                continue
            if not dependencies_satisfied(entry["scene"], states):
                # If waiting on the pending batch, flush/audit it first.
                if any(dep["phase_id"] in {x["scene"]["phase_id"] for x in batch} for dep in entry["scene"].get("dependencies", [])):
                    break
                decisions.append({"phase_id": entry["scene"]["phase_id"], "status": "DEFERRED_DEPENDENCY"})
                cursor += 1
                continue
            if reusing_gate and entry["scene"]["phase_id"] not in gate_reuse["by_phase"]:
                decisions.append({"phase_id": entry["scene"]["phase_id"],
                    "status": "DEFERRED_GATE_RESULT_UNAVAILABLE", "reason": "no_matching_saved_Gate_and_new_Gate_budget_zero"})
                cursor += 1
                continue
            batch.append(entry)
            cursor += 1
        if not batch:
            continue
        reservation = f"batch_{cursor}"
        if reusing_gate:
            if generation_slot_index >= len(generation_slots):
                for entry in batch + queue[cursor:]:
                    decisions.append({"phase_id": entry["scene"]["phase_id"], "status": "DEFERRED_BUDGET"})
                return finish("EXECUTION_ENDED_WITH_DEFERRED_BUDGET")
            reservation = generation_slots[generation_slot_index]
        cumulative = counter.dialogue_tokens(parent["dialogue"]["turns"])
        projected_scenes = [{**e["scene"], "scene_ref": f"R03:S{scene_number + i:02d}"} for i, e in enumerate(batch, 1)]
        reserve_check = future_audit_reserve(parent=parent, plan={"scenes": projected_scenes},
            conditions={s["scene_ref"]: expected_conditions.get(s["phase_id"], []) for s in projected_scenes},
            budget=budget, counter=counter)
        capacity_checks.append({"label": reservation + "_future_audit", "retained_prefix_turn_count": len(local_fact_input(retained_scenes)), **reserve_check})
        if not reserve_check["fits"]:
            for entry in batch + queue[cursor:]:
                decisions.append({"phase_id": entry["scene"]["phase_id"], "status": "DEFERRED_BUDGET", "reason": "audit_input_reserve", "excess_tokens": reserve_check["excess_tokens"]})
            return finish("EXECUTION_ENDED_WITH_DEFERRED_BUDGET")
        slot_available = (ledger.check_reused_generation_slot(reservation, cumulative_tokens=cumulative,
                          added_scene_count=len(batch)) if reusing_gate else
                          ledger.reserve(reservation, {"gate": len(batch), "realizer": 1, "adherence": 1},
                                         cumulative_tokens=cumulative, added_scene_count=len(batch)))
        if not slot_available:
            for entry in batch + queue[cursor:]:
                decisions.append({"phase_id": entry["scene"]["phase_id"], "status": "DEFERRED_BUDGET"})
            return finish("EXECUTION_ENDED_WITH_DEFERRED_BUDGET")
        if reusing_gate:
            generation_slot_index += 1
        passed = []
        try:
            for entry in batch:
                if reusing_gate:
                    saved_gate = gate_reuse["by_phase"][entry["scene"]["phase_id"]]
                    gate = deepcopy(saved_gate["validation"])
                    report["used_gate_result_phases"].append(entry["scene"]["phase_id"])
                    save(f"checks/{reservation}_{entry['scene']['phase_id'].replace(':','_')}_gate_reuse",
                         {"original_snapshot_sha256": saved_gate["history_sha256"],
                          "entry_sha256": saved_gate["entry_sha256"], "response_unchanged": saved_gate["response"],
                          "new_gate_call": False, "validation": gate})
                else:
                    gate_prompt = render_gate(parent=parent, gate_scene=entry["gate_scene"], candidates=bundle["candidates"])
                    raw = request("gate", gate_prompt, reservation)
                    gate = validate_semantic_gate_output(raw, plan={"scenes": [entry["gate_scene"]]}, parent=parent,
                        **gate_prompt["gate_contract_identity"]["validator_options"])
                save(f"checks/{reservation}_{entry['scene']['phase_id'].replace(':','_')}_gate", gate)
                if gate["valid"] and gate["realizer_allowed"]:
                    scene_number += 1
                    scene = deepcopy(entry["scene"])
                    # Keep legacy canonical refs; branch/snapshot identity is
                    # recorded separately, never smuggled into turn-ref syntax.
                    scene["scene_ref"] = f"R03:S{scene_number:02d}"
                    passed.append(scene)
                    initial_selected += int(scene["phase_id"].startswith("initial:"))
                else:
                    decisions.append({"phase_id": entry["scene"]["phase_id"], "status": "GATE_REJECT_OR_INVALID"})
            if not passed:
                continue
            plan = {"scenes": passed}
            continuity_shape = bundle.get("continuity_shape", {})
            if root_mode:
                # Only observed participants, never declarations from pending plans.
                continuity_shape = {**continuity_shape, "speakers": sorted({
                    turn["speaker"] for turn in parent["dialogue"]["turns"]})}
            prompt = render_realizer(parent=parent, plan=plan, mode=bundle["realizer_mode"],
                                     continuity_shape=continuity_shape,
                                     promoted_prefix=local_fact_input(retained_scenes))
            raw = request("realizer", prompt, reservation)
            normalized = normalize_realizer_output(raw, plan=plan, parent=parent, round_index=3)
            save(f"checks/{reservation}_normalization", normalized)
            if not normalized["valid"]:
                decisions.extend({"phase_id": s["phase_id"], "status": "STRUCTURE_COPY_OR_BINDING_REJECT"} for s in passed)
                continue
            child = normalized["child"]
            for s in child["scenes"]:
                s["session_ref"] = s["scene_ref"]
            conditions = {s["scene_ref"]: expected_conditions.get(s["phase_id"], []) for s in passed}
            audit = request("adherence", render_adherence(parent=parent, plan=plan, child=child,
                requested_conditions=conditions, generated_response=received_outputs["realizer"]), reservation)
            retained = assess_retention(parent=parent, plan=plan, child=child, raw_audit=audit,
                requested_conditions=conditions, audit_version=ADHERENCE_VERSION)
            save(f"checks/{reservation}_retention", retained)
            accepted = retained["mechanical_promotion"].get("promotable_child") or {"scenes": []}
            if accepted["scenes"]:
                tentative = append_expansion_round(prior_expanded=prior_expanded,
                    child={"root_id": parent["root_id"], "round_index": 3,
                           "scenes": retained_scenes + accepted["scenes"]})
                closure = validate_appended_round(tentative, prior_expanded=prior_expanded)
                save(f"checks/{reservation}_append_closure", closure)
                if not closure["valid"]:
                    raise ValueError("legacy_append_closure_failed")
            # Whole dialogue never truncated; overflow rejects this batch.
            actual_text = local_fact_input(accepted["scenes"])
            if counter.dialogue_tokens(parent["dialogue"]["turns"] + actual_text) > budget["max_cumulative_tokens"]:
                decisions.extend({"phase_id": s["phase_id"], "status": "ACTUAL_LENGTH_OVERFLOW_QUARANTINE"} for s in passed)
                continue
            for s, decision in zip(passed, retained["decisions"]):
                states[s["phase_id"]] = decision
                decisions.append({"phase_id": s["phase_id"], **decision})
            retained_scenes.extend(accepted["scenes"])
            parent["dialogue"]["turns"].extend({**t, "turn_id": t["turn_ref"]} for t in actual_text)
        except Exception:
            return finish("STAGE_FAILURE_STOP_NO_RETRY")
    report["initial_selected_scene_count"] = initial_selected
    report["supplement_phase_count"] = len(added)
    return finish("EXECUTION_ENDED_GOALS_RECORDED_SEPARATELY")
