"""Bounded live retry for selected v3 Realizer batches that failed copy checks.

The retry consumes exactly one Realizer request per root.  It never invokes
Planner, Gate, QA, Solver, or any promotion path.  The v4 prompt is loaded from
the offline preflight and the full prior dialogue remains provenance/validator
context only.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse, hashlib, json, time
from pathlib import Path

from ..h4_round1_r03_incremental_plan_retention_v1.capacity import local_counter, measure
from ..h4_round1_r03_incremental_plan_retention_v1.budget import validate_budget
from ..h4_round1_r03_incremental_plan_retention_v1.run import SingleShotTransport
from ..h4_round1_r03_incremental_plan_retention_v1.shared_planner_contract import parse_output
from ..h4_round1_r03_incremental_plan_retention_v1.transport_config import credential_environment
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import normalize_realizer_output

VERSION = "h4_r03_realizer_v4_turn_contract_fix_v1"
ROOTS = ("cc-train-episode-122711", "cc-train-episode-112812")


def read(path):
    return json.loads(Path(path).read_text())


def sha(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def source_parent(source_run):
    identity = read(Path(source_run) / "input_manifest.json")
    bundle = read(Path(identity["bundle_path"]) / "prepared/initial/bundle.json")
    return bundle["parent"]


def run_root(preflight_root, source_run, output, credential_source, *, prepare_only=False):
    preflight_root, source_run, output = map(Path, (preflight_root, source_run, output))
    model_input = read(preflight_root / "model_input.json")
    prompt = (preflight_root / "prompt.txt").read_text()
    preflight_manifest = read(preflight_root / "request_manifest.json") if (preflight_root / "request_manifest.json").exists() else {}
    source_run = source_run.resolve()
    selected = read(source_run / "../h4_round1_r03_other_roots_realizer_v1_live_1/roots/" + preflight_root.name + "/selected_plan.json") if False else None
    # The source selected plan is stored in the completed v3 Realizer run.
    v3root = source_run.parent / "h4_round1_r03_other_roots_realizer_v1_live_1" / "roots" / preflight_root.name
    selected = read(v3root / "selected_plan.json")["scenes"][4:6]
    accepted = []
    for n in (1, 2):
        accepted.extend(t for scene in read(v3root / "batches" / f"{n:03d}/normalized_child.json")["scenes"]
                        for t in scene["dialogue_turns"])
    model = read(source_run / "model_config.json")
    budget = read(source_run / "budget.json")
    budget["role_requests"] = {role: (1 if role == "realizer" else 0) for role in budget["role_requests"]}
    budget["total_requests"] = 1
    # A versioned preflight may raise only this request's output token cap.
    # Recompute both capacity ceilings; changing request.max_tokens alone would
    # leave the transport and context validation bound to the old reservation.
    if "max_output_tokens" in preflight_manifest:
        cap = preflight_manifest["max_output_tokens"]
        if type(cap) is not int or cap <= 0:
            raise ValueError("positive_realizer_output_token_cap_required")
        budget["max_output_tokens"]["realizer"] = cap
        budget["max_input_tokens"] = min(
            budget["max_input_tokens"],
            budget["context_window_tokens"] - max(budget["max_output_tokens"].values()),
        )
        if "role_capacity" in budget:
            spec = budget["role_capacity"]["realizer"]
            spec["max_input_tokens"] = min(
                spec["max_input_tokens"], spec["context_window_tokens"] - cap
                - spec["thinking_reserve_tokens"] - spec["safety_margin_tokens"],
            )
            budget["max_input_tokens"] = min(budget["max_input_tokens"], spec["max_input_tokens"])
    budget["max_generated_tokens"] = budget["max_output_tokens"]["realizer"]
    validate_budget(budget)
    request = {
        "role": "realizer", "call_id": "realizer-v4-retry-001", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": budget["max_output_tokens"]["realizer"], "prompt_version": preflight_manifest.get("prompt_version", "incremental_retention_v1.realizer.minimal_state_v4_schema_v3_turn14.text_free"),
        "model": model["model"], "temperature": model["temperature"], "thinking": False,
        "identity": {"root_id": read(v3root / "run_summary.json")["root_id"], "round_index": 3,
                      "batch_index": 3, "candidate_refs": [s["selected_candidate_ref"] for s in selected],
                      "run_version": VERSION},
    }
    counter = local_counter(model["tokenizer_path"], model["tokenizer_json_sha256"])
    capacity = measure(request["messages"], counter=counter, budget=budget, role="realizer")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "request.json", request)
    (output / "prompt.txt").write_text(prompt)
    write(output / "model_input.json", model_input)
    write(output / "plan.json", {"scenes": selected})
    write(output / "capacity.json", capacity)
    write(output / "budget.json", budget)
    write(output / "request_manifest.json", {"request_sha256": sha(request), "prompt_version": request["prompt_version"],
        "adapter_version": VERSION, "retry_count": 0 if prepare_only else 1,
        "max_output_tokens": request["max_tokens"], "preflight_source": str(preflight_root.resolve()),
        "authorized_max_calls": 0 if prepare_only else 1,
        "prefix_turn_count": 0, "full_retained_actual_prefix_included": False,
        "failed_output_included": False, "historical_dialogue_included": False,
        "candidate_refs": request["identity"]["candidate_refs"]})
    write(output / "authorization_scope.json", {"run_version": VERSION, "realizer_max": 0 if prepare_only else 1,
        "other_roles_max": 0, "automatic_retry": False, "promotion": False})
    if prepare_only:
        result = {"status": "OFFLINE_REQUEST_PREPARED_NOT_GENERATED" if capacity["fits"] else "OFFLINE_CAPACITY_REJECT",
                  "root_id": request["identity"]["root_id"], "realizer_real_calls": 0,
                  "max_output_tokens": request["max_tokens"], "capacity_fits": capacity["fits"]}
        write(output / "call_attempts.json", [])
        write(output / "run_summary.json", result)
        return result
    ledger = [{"role": "realizer", "call_id": request["call_id"], "attempt_count": 1,
               "started_at": datetime.now(timezone.utc).isoformat()}]
    write(output / "call_attempts.json", ledger)
    result_summary = None
    with credential_environment(model, credential_source):
        backend = SingleShotTransport(model, budget)
        backend.response_evidence_sink = lambda ev: write(output / ("http_" + ev["evidence_event"] + ".json"), ev)
        if not capacity["fits"]:
            result_summary = {"status": "RETRY_STOPPED_ON_CAPACITY", "realizer_real_calls": 0, "error": "INPUT_CAPACITY_EXCEEDED"}
        else:
            try:
                print(request["identity"]["root_id"], "REQUEST_STARTED", flush=True)
                raw = backend.request(request)
                (output / "raw_response.json").write_text(raw)
                if backend.last_response_envelope is not None: write(output / "response_envelope.json", backend.last_response_envelope)
                parsed = parse_output(raw); write(output / "parsed_response.json", parsed)
                parent = deepcopy(source_parent(source_run)); parent["dialogue"]["turns"].extend(accepted)
                normalized = normalize_realizer_output(parsed, plan={"scenes": selected}, parent=parent,
                    round_index=3, scene_index_offset=4)
                write(output / "normalization_result.json", normalized)
                write(output / "validation.json", {k: v for k, v in normalized.items() if k != "child"})
                if not normalized["valid"]:
                    result_summary = {"status": "REALIZER_OUTPUT_INVALID_STOP", "realizer_real_calls": 1,
                                      "error": {"type": "STRUCTURE_OR_COPY_REJECT", "errors": normalized["errors"]}}
                else:
                    child = normalized["child"]
                    changes=[]
                    for scene in child["scenes"]:
                        before=scene["session_ref"]; scene["session_ref"]=scene["scene_ref"]
                        changes.append({"scene_ref":scene["scene_ref"],"before":before,"after":scene["session_ref"]})
                    write(output / "normalization_diff.json", {"raw_sha256": sha(parsed),"normalized_child_sha256": sha(child),"session_ref_changes":changes,"raw_and_pre_normalization_preserved":True})
                    write(output / "normalized_child.json", child)
                    write(output / "generated_dialogue.json", {"root_id": request["identity"]["root_id"], "round_index":3,"authority_status":"GENERATED_DRAFT_NOT_PROMOTED","turns":[t for s in child["scenes"] for t in s["dialogue_turns"]]})
                    print(request["identity"]["root_id"], "STRUCTURE_VALID", flush=True)
                    result_summary = {"status": "REALIZER_GENERATION_COMPLETE_STRUCTURE_VALID", "realizer_real_calls": 1,
                                      "scene_count": len(child["scenes"]), "turn_count": sum(len(s["dialogue_turns"]) for s in child["scenes"]), "error": None}
            except Exception as exc:
                write(output / "error.json", {"type": type(exc).__name__, "retry": False})
                result_summary = {"status": "REALIZER_TECHNICAL_OR_PARSE_ERROR_STOP", "realizer_real_calls": 1,
                                  "error": {"type": type(exc).__name__, "retry": False}}
    result_summary.update({"root_id": request["identity"]["root_id"], "run_version": VERSION,
        "planner_real_calls": 0, "gate_real_calls": 0, "other_real_calls": 0, "qa": 0, "solver": 0,
        "training": 0, "memorax_write": 0, "adherence": 0, "promotion": 0, "semantic_validation": "NOT_RUN",
        "automatic_retry": False, "input_mode": "minimal_state_v4"})
    write(output / "run_summary.json", result_summary)
    return result_summary


def main():
    p=argparse.ArgumentParser(); p.add_argument("--preflight",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--credential-config",type=Path); p.add_argument("--source-run",action="append",type=Path,required=True)
    p.add_argument("--prepare-only", action="store_true", help="Save the request and capacity checks without credentials or model calls")
    a=p.parse_args(); pre=a.preflight.resolve(); out=a.output.resolve()
    if not a.prepare_only and a.credential_config is None:
        p.error("--credential-config is required for a live request")
    supplied_roots = []
    for path in a.source_run:
        try:
            supplied_roots.append(read(path / "run_summary.json")["root_id"])
        except Exception:
            supplied_roots.append(next((root for root in ROOTS if root in str(path)), None))
    if (not supplied_roots or len(set(supplied_roots)) != len(supplied_roots)
            or not set(supplied_roots).issubset(ROOTS)):
        raise ValueError("retry_scope_requires_unique_supported_roots")
    out.mkdir(parents=True,exist_ok=False)
    credential=None if a.prepare_only else {"path":str(a.credential_config.resolve()),"sha256":sha(a.credential_config.read_bytes())}
    results=[]
    for index, source in enumerate(a.source_run):
        if index and not a.prepare_only:
            print("COOLDOWN 65 seconds", flush=True)
            time.sleep(65)
        root = supplied_roots[index]
        results.append(run_root(pre/"roots"/root,source,out/"roots"/root,credential,prepare_only=a.prepare_only))
    status = "OFFLINE_REQUESTS_PREPARED_NOT_GENERATED" if a.prepare_only else "REALIZER_V4_RETRY_FINISHED"
    if a.prepare_only and not all(x["capacity_fits"] for x in results):
        status = "OFFLINE_CAPACITY_REJECT"
    write(out/"run_summary.json",{"status":status,"run_version":VERSION,"results":results,"realizer_real_calls":sum(x["realizer_real_calls"] for x in results),"planner_real_calls":0,"gate_real_calls":0,"other_real_calls":0,"automatic_retries":0,"qa":0,"solver":0,"training":0,"memorax_write":0,"adherence":0,"promotion":0})
    write(out/"scope.json",{"run_version":VERSION,"roots":supplied_roots,"authorized_max_realizer_calls":0 if a.prepare_only else len(supplied_roots),"realizer_real_calls":sum(x["realizer_real_calls"] for x in results),"other_role_calls":0,"automatic_retry":False})


if __name__ == "__main__": main()
