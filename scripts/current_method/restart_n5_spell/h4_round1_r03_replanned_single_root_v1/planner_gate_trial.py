"""Run exactly one fresh Planner call followed by the existing Gate.

This entrypoint deliberately does not import the frozen downstream workflow.
It consumes the rebound offline Planner request, uses the shared Planner
contract and relation compiler, then invokes the existing Gate once per
candidate.  Realizer, supplement, adherence, QA, Solver, training and Memory
writes are outside this entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

from ..h4_round1_r03_incremental_plan_retention_v1 import prompts
from ..h4_round1_r03_incremental_plan_retention_v1.capacity import local_counter, measure
from ..h4_round1_r03_incremental_plan_retention_v1.planner_relations import compile_initial_output
from ..h4_round1_r03_incremental_plan_retention_v1.planning import sha
from ..h4_round1_r03_incremental_plan_retention_v1.shared_planner_contract import parse_output
from ..h4_round1_r03_incremental_plan_retention_v1.planner_relations import validate_output as validate_relation_output
from ..h4_round1_r03_incremental_plan_retention_v1.transport_config import (
    credential_environment,
)
from ..h4_round1_r03_incremental_plan_retention_v1.run import SingleShotTransport
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    validate_semantic_gate_output,
)


from ..h4_round1_r03_incremental_plan_retention_v1 import planner_candidate_policy as candidate_policy
from ..h4_round1_r03_incremental_plan_retention_v1 import planner_relations, planner_templates, planner_preparation

RUN_VERSION = "h4_round1_r03_initial_only_relation_repair_v1"
CAP9_RUN_VERSION = "h4_round1_r03_initial_only_target8_cap9_v1"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_raw(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _quality_observation(raw: dict, relation_sidecar: dict) -> dict:
    rows = []
    generic = re.compile(r"\b(discuss|share|talk|chat|continue discussing|discuss further)\b", re.I)
    action = re.compile(r"\b(choose|decide|record|schedule|practice|prepare|check|ask|offer|agree|track|start|review|confirm|arrange|plan)\b", re.I)
    for collection in ("selected_continuations", "new_development_candidates"):
        for index, item in enumerate(raw.get(collection) or []):
            text = " ".join(str(item.get(k) or "") for k in (
                "goal", "transition_from_previous", "required_transition", "after_state",
                "before_state", "source_basis", "interaction_beats"))
            deps = item.get("dependencies") or []
            rows.append({
                "output_ref": f"/{collection}/{index}",
                "generic_discussion_only": "NEEDS_REVIEW" if generic.search(text) and not action.search(text) else "UNKNOWN",
                "concrete_state_change_language": "OBSERVED" if action.search(str(item.get("required_transition") or "")) else "NEEDS_REVIEW",
                "person_time_state_relation_distinction": "NEEDS_REVIEW",
                "cross_event_relation": "OBSERVED" if len(item.get("source_basis") or []) > 1 else "UNKNOWN",
                "continuity_relation": "OBSERVED" if any(d.get("relation_kind") == "natural_continuity" for d in deps) else "UNKNOWN",
                "hard_prerequisite_relation": "OBSERVED" if any(d.get("relation_kind") == "hard_prerequisite" for d in deps) else "UNKNOWN",
                "old_fact_paraphrase": "NEEDS_REVIEW",
                "history_qa_repair_focus": "NEEDS_REVIEW",
                "evidence": {"required_transition": item.get("required_transition"), "dependency_count": len(deps)},
            })
    return {
        "version": "planner_quality_observation_v1",
        "semantic_truth": "NOT_ASSERTED",
        "unknown_policy": "UNKNOWN_OR_NEEDS_REVIEW_REQUIRES_HUMAN_REVIEW",
        "items": rows,
        "relation_summary": {
            "declared_relation_count": len(relation_sidecar.get("relations") or []),
            "natural_continuity_count": sum(r.get("relation_kind") == "natural_continuity" for r in relation_sidecar.get("relations") or []),
            "hard_prerequisite_count": sum(r.get("relation_kind") == "hard_prerequisite" for r in relation_sidecar.get("relations") or []),
        },
    }


def _post_gate_projection(entries, gate_records, relation_sidecar=None):
    passed = {r["candidate_ref"] for r in gate_records if r.get("decision") == "pass"}
    rejected = {r["candidate_ref"] for r in gate_records if r.get("decision") == "reject"}
    phase_by_candidate = {}
    for index, (candidate_id, entry) in enumerate(entries.items(), 1):
        phase_by_candidate[candidate_id] = f"initial:{candidate_id}"
    relations = []
    blocked = []
    declared = (relation_sidecar or {}).get("relations")
    if declared is None:
        declared = []
        for candidate_id, entry in entries.items():
            for dep in (entry.get("scene") or {}).get("dependencies") or []:
                declared.append({"owner_phase_id": phase_by_candidate.get(candidate_id), **dep})
    for dep in declared:
        owner_phase = dep.get("owner_phase_id")
        owner = owner_phase.removeprefix("initial:") if isinstance(owner_phase, str) else None
        pred_phase = dep.get("predecessor_phase_id") or dep.get("phase_id")
        pred_candidate = pred_phase.removeprefix("initial:") if isinstance(pred_phase, str) else None
        status = "VALID" if owner in passed and pred_candidate in passed else (
            "BLOCKED_BY_REJECTED_PREREQUISITE" if pred_candidate in rejected else "NOT_APPLICABLE")
        relation = {"owner_candidate_ref": owner, "owner_phase_id": owner_phase,
                    "predecessor_phase_id": pred_phase, "predecessor_candidate_ref": pred_candidate,
                    "relation_kind": dep.get("relation_kind", "hard_prerequisite"),
                    "condition": dep.get("condition"), "status": status}
        relations.append(relation)
        if status == "BLOCKED_BY_REJECTED_PREREQUISITE":
            blocked.append(relation)
    return {
        "version": "post_gate_relation_projection_v1",
        "candidate_identity_policy": "stable_candidate_identity_until_materialization",
        "passed_candidate_refs": sorted(passed),
        "rejected_candidate_refs": sorted(rejected),
        "relations": relations,
        "blocked_dependencies": blocked,
        "natural_continuity_and_hard_prerequisite_separate": True,
        "formal_scene_ids_generated": False,
    }


def run(bundle_dir: Path, run_dir: Path, model_config_path: Path, credential_config_path: Path):
    prepared = bundle_dir / "prepared"
    bundle = _read(prepared / "initial/bundle.json")
    messages = _read(prepared / "initial/messages.json")
    root_id = bundle.get("identity", {}).get("root_id")
    if not isinstance(root_id, str) or not root_id:
        raise ValueError("planner_bundle_root_id_required")
    limit = candidate_policy.candidate_limit(bundle)
    run_version = CAP9_RUN_VERSION if limit == 9 else RUN_VERSION
    if limit == 9:
        planner_preparation._validate_bundle(bundle)
        if (bundle.get("output_contract_version") != planner_relations.CAP9_VERSION
                or bundle.get("template_version") != planner_templates.CAP9_VERSION
                or sha(messages) != bundle.get("request_sha256")):
            raise ValueError("cap9_prepared_request_contract_mismatch")
        from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import build_speaker_registry
        expected = planner_templates.render_initial_planner(
            feedback_view=bundle["shared_feedback_view"], dialogue=bundle["dialogue"],
            memory_overview=bundle["memory_overview"], memory_candidates=bundle["candidates"],
            verified_tracks=bundle["verified_tracks"], speaker_registry=build_speaker_registry(bundle["parent"]),
            max_new_candidates=limit)
        if expected["messages"] != messages:
            raise ValueError("cap9_prepared_messages_differ_from_current_template")
    model = _read(model_config_path)
    credential_source = {"path": str(credential_config_path.resolve()), "sha256": _sha_bytes(credential_config_path)}
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "planner").mkdir()
    (run_dir / "gate").mkdir()
    _write_json(run_dir / "authorization_scope.json", {
        "policy": run_version, "root_id": root_id, "round_index": 3,
        "authorized_calls": {"planner": 1, "gate": "one_per_generated_candidate"},
        "forbidden_calls": ["planner_retry", "alternative_planner", "supplement", "realizer", "adherence", "qa", "solver", "training", "memorax_write"],
    })
    _write_json(run_dir / "input_manifest.json", {
        "bundle_path": str(bundle_dir.resolve()),
        "bundle_sha256": _sha_bytes(prepared / "initial/bundle.json"),
        "request_sha256": bundle.get("request_sha256"),
        "input_files": bundle.get("input_files"),
        "root_id": bundle.get("identity", {}).get("root_id"),
        "round_index": bundle.get("identity", {}).get("round_index"),
        "old_r03_plan_or_dialogue_consumed": False,
        "old_aligned_runtime_reference_in_request": False,
    })
    if bundle.get("identity", {}).get("round_index") != 3:
        raise ValueError("rebound_identity_mismatch")
    if bundle.get("status") != "INITIAL_REQUEST_PREPARED_NOT_GENERATED":
        raise ValueError("rebound_bundle_not_fresh")
    if any("h4_round1_r03_aligned_prompts_single_root_v1_live_1" in str(x) for x in messages):
        raise ValueError("old_aligned_runtime_reference_in_planner_messages")
    if any("h4_round1_r03_aligned_prompts_single_root_v1_live_1" in str(x) for x in bundle.get("parent", {}).get("dialogue", {}).get("turns", [])):
        raise ValueError("old_aligned_runtime_reference_in_history")

    counter = local_counter(model["tokenizer_path"], model["tokenizer_json_sha256"])
    # Build a bounded budget view: all role fields remain explicit, but only the
    # authorized Planner and Gate slots are nonzero.
    budget = candidate_policy.gate_budget(bundle["capacity_budget"], max_new_candidates=limit)
    _write_json(run_dir / "budget.json", budget)
    _write_json(run_dir / "model_config.json", {k: v for k, v in model.items() if k != "api_key_env"} | {"api_key_env": model["api_key_env"]})

    planner_request = {"role": "planner", "call_id": "planner-001", "messages": messages,
                       "max_tokens": 8192, "prompt_version": "rebound_prepared_initial_planner",
                       "identity": {"root_id": root_id, "round_index": 3, "run_version": run_version},
                       "model": model["model"], "temperature": model["temperature"], "thinking": False}
    _write_json(run_dir / "planner/request.json", planner_request)
    _write_raw(run_dir / "planner/prompt.txt", messages[0]["content"])
    _write_json(run_dir / "planner/request_manifest.json", {
        "purpose": "single_fresh_initial_planner", "request_sha256": sha(planner_request),
        "candidate_policy_version": bundle.get("candidate_policy_version"),
        "target_new_candidates": 8, "max_new_candidates": limit,
        "gate_request_limit": budget["role_requests"]["gate"],
        "messages_sha256": sha(messages), "message_count": len(messages), "max_tokens": 8192,
        "retry_policy": "NO_RETRY", "source_bundle_sha256": _sha_bytes(prepared / "initial/bundle.json"),
    })
    calls = {"planner": 0, "gate": 0, "other": 0}
    planner_raw = None
    planner_error = None
    backend = None
    with credential_environment(model, credential_source):
        backend = SingleShotTransport(model, budget)
        backend.response_evidence_sink = lambda ev: _write_json(run_dir / "planner" / ("http_" + ev.get("evidence_event", "response") + ".json"), ev)
        calls["planner"] = 1
        try:
            planner_raw = backend.request(planner_request)
            _write_raw(run_dir / "planner/raw_response.json", planner_raw)
            if backend.last_response_envelope is not None:
                _write_json(run_dir / "planner/response_envelope.json", backend.last_response_envelope)
        except Exception as exc:
            planner_error = {"exception_type": type(exc).__name__, "no_retry": True}
            _write_json(run_dir / "planner/validation.json", {"valid": False, "status": "TECHNICAL_ERROR_NO_RETRY", **planner_error})
            calls["other"] = 0
            _write_json(run_dir / "run_summary.json", {"status": "PLANNER_TECHNICAL_ERROR", "calls": calls,
                "realizer": 0, "qa": 0, "solver": 0, "training": 0, "memorax_write": 0})
            _write_raw(run_dir / "HANDOFF.md", "# Handoff\n\nPlanner technical error; no retry and no Gate/Realizer execution.\n")
            return 2

        try:
            parsed = parse_output(planner_raw)
        except Exception as exc:
            validation = {"valid": False, "status": "SCHEMA_OR_JSON_ERROR_STOP", "errors": [type(exc).__name__ + ":" + str(exc)]}
            _write_json(run_dir / "planner/validation.json", validation)
            _write_json(run_dir / "run_summary.json", {"status": "PLANNER_SCHEMA_INVALID", "calls": calls,
                "realizer": 0, "qa": 0, "solver": 0, "training": 0, "memorax_write": 0})
            _write_raw(run_dir / "HANDOFF.md", "# Handoff\n\nPlanner response was not valid JSON; stopped without retry or Gate.\n")
            return 3
        _write_json(run_dir / "planner/parsed_plan.json", parsed)
        structural = (planner_relations.validate_initial_output_cap9(parsed) if limit == 9
                      else validate_relation_output(parsed))
        compiled = compile_initial_output(parsed, parent=bundle["parent"], candidates=bundle["candidates"], verified_tracks=bundle["verified_tracks"], max_new_candidates=limit)
        validation = {"structural": structural, "compiled": {k: v for k, v in compiled.items() if k not in {"prepared", "value"}},
                      "valid": bool(structural.get("valid") and compiled.get("valid")), "no_retry": True}
        _write_json(run_dir / "planner/validation.json", validation)
        if not validation["valid"]:
            _write_json(run_dir / "run_summary.json", {"status": "PLANNER_CONTRACT_INVALID", "calls": calls,
                "continuations": len(parsed.get("selected_continuations") or []),
                "new_development_candidates": len(parsed.get("new_development_candidates") or []),
                "realizer": 0, "qa": 0, "solver": 0, "training": 0, "memorax_write": 0})
            _write_raw(run_dir / "HANDOFF.md", "# Handoff\n\nPlanner contract validation failed; stopped before Gate.\n")
            return 4
        relation_metadata = compiled.get("relation_sidecar") or {}
        _write_json(run_dir / "planner/relation_metadata.json", relation_metadata)
        _write_json(run_dir / "planner/lineage.json", {"root_id": root_id, "round_index": 3,
            "history_sha256": sha(bundle["parent"]), "input_bundle_sha256": _sha_bytes(prepared / "initial/bundle.json"),
            "feedback_source": bundle.get("feedback_source"), "old_r03_consumed": False})
        _write_json(run_dir / "planner_quality_observation.json", _quality_observation(parsed, relation_metadata))

        entries = compiled["prepared"]["candidate_registry"]
        gate_records = []
        gate_errors = []
        for ordinal, (candidate_id, entry) in enumerate(entries.items(), 1):
            if ordinal > budget["role_requests"]["gate"]:
                gate_errors.append("gate_budget_exceeded_candidate_count")
                break
            gate_scene = deepcopy(entry["gate_scene"])
            gate_prompt = prompts.render_gate(parent=bundle["parent"], gate_scene=gate_scene, candidates=bundle["candidates"])
            gate_request = {"role": "gate", "call_id": f"gate-{ordinal:03d}", "messages": gate_prompt["messages"],
                            "max_tokens": 2048, "prompt_version": gate_prompt["prompt_version"],
                            "identity": {"root_id": root_id, "round_index": 3, "candidate_id": candidate_id, "run_version": run_version},
                            "model": model["model"], "temperature": model["temperature"], "thinking": False,
                            "gate_contract_identity": gate_prompt.get("gate_contract_identity")}
            _write_json(run_dir / f"gate/requests/{ordinal:03d}_gate.json", gate_request)
            _write_json(run_dir / f"gate/request_manifests/{ordinal:03d}_gate.json", {
                "request_sha256": sha(gate_request), "candidate_id": candidate_id,
                "retry_policy": "NO_RETRY", "history_sha256": sha(bundle["parent"]),
            })
            calls["gate"] += 1
            backend.response_evidence_sink = lambda ev, n=ordinal: _write_json(run_dir / f"gate/http_{n:03d}_{ev.get('evidence_event', 'response')}.json", ev)
            try:
                raw_gate = backend.request(gate_request)
                _write_raw(run_dir / f"gate/raw_response_{ordinal:03d}.json", raw_gate)
                parsed_gate = parse_output(raw_gate)
                _write_json(run_dir / f"gate/parsed_response_{ordinal:03d}.json", parsed_gate)
                gate_validation = validate_semantic_gate_output(parsed_gate, plan={"candidate_pool": [gate_scene]},
                    parent=bundle["parent"], minimum_pass_candidates=1)
                _write_json(run_dir / f"gate/validation_{ordinal:03d}.json", gate_validation)
                if not gate_validation.get("valid"):
                    gate_errors.append({"candidate_id": candidate_id, "errors": gate_validation.get("errors")})
                    break
                for review in gate_validation.get("normalized_candidate_reviews") or []:
                    gate_records.append({**review, "candidate_id": candidate_id})
            except Exception as exc:
                gate_errors.append({"candidate_id": candidate_id, "exception_type": type(exc).__name__, "no_retry": True})
                break
        _write_json(run_dir / "gate/candidate_decisions.json", gate_records)
        _write_json(run_dir / "gate/selected_candidates.json", [r for r in gate_records if r.get("decision") == "pass"])
        _write_json(run_dir / "gate/rejected_candidates.json", [r for r in gate_records if r.get("decision") == "reject"])
        projection = _post_gate_projection(entries, gate_records, relation_sidecar=relation_metadata)
        if gate_errors:
            projection["status"] = "GATE_STOPPED_ON_ERROR"
            projection["errors"] = gate_errors
        _write_json(run_dir / "gate/post_gate_relation_projection.json", projection)
        _write_json(run_dir / "gate/lineage.json", {"root_id": root_id, "round_index": 3,
            "planner_parsed_plan_sha256": _sha_bytes(run_dir / "planner/parsed_plan.json"),
            "candidate_decisions_sha256": _sha_bytes(run_dir / "gate/candidate_decisions.json"),
            "old_gate_results_reused": False})
        status = "PLANNER_AND_GATE_COMPLETE" if not gate_errors else "GATE_STOPPED_ON_ERROR"
        _write_json(run_dir / "run_summary.json", {"status": status, "run_version": run_version,
            "root_id": root_id, "planner_real_calls": calls["planner"], "gate_real_calls": calls["gate"],
            "other_real_calls": calls["other"], "continuations": len(parsed.get("selected_continuations") or []),
            "new_development_candidates": len(parsed.get("new_development_candidates") or []),
            "gate_passed": sum(r.get("decision") == "pass" for r in gate_records),
            "gate_rejected": sum(r.get("decision") == "reject" for r in gate_records),
            "blocked_by_rejected_prerequisite": len(projection.get("blocked_dependencies") or []),
            "realizer": 0, "qa": 0, "solver": 0, "training": 0, "memorax_write": 0,
            "realizer_static_input_readiness": "STATIC_CHECK_ONLY"})
        _write_raw(run_dir / "HANDOFF.md", f"# Handoff\n\nStatus: {status}\n\nPlanner and Gate only. Realizer, QA, Solver, training and Memorax write were not run.\n")
        return 0 if not gate_errors else 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--credential-config", type=Path, required=True)
    args = parser.parse_args()
    return run(args.bundle, args.run_dir, args.model_config, args.credential_config)


if __name__ == "__main__":
    sys.exit(main())
