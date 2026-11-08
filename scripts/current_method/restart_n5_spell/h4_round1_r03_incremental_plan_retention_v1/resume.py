"""Resume saved root patches without reopening the supplement sending path."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .planning import sha
from .plan_normalization import VERSION as NORMALIZATION_VERSION
from .task_selection import VERSION as SELECTION_VERSION
from .root_scope import prepare_bundle, render_for_bundle
from .root_planning import merge_root_patch
from .workflow import execute_policy, prepare_saved_resume
from .run import (validate_authorization, validate_frozen_sources, production_registry,
                  SingleShotTransport, offline_network_guard)
from .experiment_authorization import load_experiment_scope, experiment_registry
from .transport_config import credential_environment

VERSION = "saved_root_patch_resume_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_resume_manifest(source_run, *, source_authorization, normalization_version=NORMALIZATION_VERSION):
    """Freeze existing evidence references; no state or authorization is created."""
    source_run = Path(source_run).resolve()
    report = read(source_run / "final_report.json")
    ledger = Path(report["persistent_ledger_path"]).resolve()
    return {"version": VERSION, "source_run": str(source_run),
        "source_authorization": {"path": str(Path(source_authorization).resolve()),
                                 "sha256": digest(source_authorization)},
        "files": {str(path.relative_to(source_run)): digest(path)
                  for path in sorted(source_run.rglob("*.json"))},
        "ledger_path": str(ledger), "ledger_sha256": digest(ledger),
        "identity": report["identity"], "execution_scope": report["execution_scope"],
        "source_status": report["status"], "normalization_version": normalization_version}


def validate_saved_source(manifest):
    """Check saved HTTP, parsed patch, original merge and persistent lineage."""
    if (manifest.get("version") != VERSION or
            manifest.get("normalization_version") not in {NORMALIZATION_VERSION, SELECTION_VERSION}):
        raise ValueError("saved_resume_manifest_version_mismatch")
    source = Path(manifest["source_run"]).resolve()
    for name, expected in manifest["files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or digest(path) != expected:
            raise ValueError("saved_resume_source_file_sha_mismatch:" + name)
    required = {"final_report.json", "input_freeze.json", "execution_budget.json", "model_config.json",
                "experiment_scope.json", "increment_patch_and_diff.json", "original_plan_new_policy_diagnostics.json",
                "ledger_snapshot.json", "merged_outline_check.json", "supplement_contract_identity.json"}
    if not required <= set(manifest["files"]):
        raise ValueError("saved_resume_required_evidence_missing")
    values = {name: read(source / name) for name in required}
    report = values["final_report.json"]
    bundle, budget, model = (values[name] for name in ("input_freeze.json", "execution_budget.json", "model_config.json"))
    scope = values["experiment_scope.json"]
    if (report["status"] != "MERGED_OUTLINE_REJECT_STOP" or manifest["source_status"] != report["status"]
            or report["identity"] != bundle["identity"] or manifest["identity"] != bundle["identity"]
            or report["execution_scope"] != scope or manifest["execution_scope"] != scope
            or report["retained_scene_count"] != 0 or report["retained_turn_count"] != 0):
        raise ValueError("saved_resume_source_stop_identity_mismatch")
    calls = report["new_calls"]
    if len(calls) != 1 or calls[0]["role"] != "supplement" or calls[0]["status"] != "RESPONSE_RECEIVED":
        raise ValueError("saved_resume_requires_exactly_one_received_supplement")
    call = calls[0]
    requests = [(name, read(source / name)) for name in manifest["files"] if name.startswith("requests/")]
    if len(requests) != 1:
        raise ValueError("saved_resume_request_evidence_count")
    request_name, request = requests[0]
    if (request["role"] != "supplement" or request["call_id"] != call["call_id"]
            or request["identity"] != bundle["identity"] or request["execution_scope"] != scope
            or sha(request["messages"]) != call["messages_sha256"]
            or request.get("model") != model["model"] or request.get("temperature") != model["temperature"]
            or request.get("max_tokens") != budget["max_output_tokens"]["supplement"]):
        raise ValueError("saved_resume_actual_request_identity_mismatch")
    prepared = prepare_bundle(bundle)
    rendered = render_for_bundle(bundle, budget, prepared)
    if (request["messages"] != rendered["messages"]
            or request["prompt_version"] != rendered["prompt_version"]
            or request["supplement_contract_identity"] != rendered["supplement_contract_identity"]
            or values["supplement_contract_identity.json"] != rendered["supplement_contract_identity"]
            or values["original_plan_new_policy_diagnostics.json"] != prepared):
        raise ValueError("saved_resume_frozen_prompt_or_prepared_mismatch")
    response_base = "responses/" + Path(request_name).stem
    required_response = {response_base + ".json", response_base + "_envelope.json"}
    if not required_response <= set(manifest["files"]):
        raise ValueError("saved_resume_response_or_envelope_missing")
    http = [read(source / name) for name in manifest["files"]
            if name.startswith(response_base + "_http_")]
    completed = [row for row in http if row.get("evidence_event") == "completed"]
    if len(completed) != 1:
        raise ValueError("saved_resume_completed_http_evidence_required")
    completed = completed[0]
    if (completed["http_status"] != 200 or completed.get("response_body_complete") is not True
            or completed.get("response_unavailable") is not False
            or completed["request_identity"]["messages_sha256"] != call["messages_sha256"]
            or completed["request_identity"]["execution_identity"] != bundle["identity"]
            or hashlib.sha256(completed["response_body"].encode()).hexdigest() != completed["saved_body_sha256"]):
        raise ValueError("saved_resume_original_http_response_mismatch")
    envelope = read(source / (response_base + "_envelope.json"))
    if json.loads(completed["response_body"]) != envelope:
        raise ValueError("saved_resume_http_envelope_mismatch")
    raw = read(source / (response_base + ".json"))
    if envelope["choices"][0]["message"]["content"] != raw:
        raise ValueError("saved_resume_model_response_mismatch")
    patch = json.loads(raw) if isinstance(raw, str) else raw
    merged = merge_root_patch(prepared, patch)
    if not merged["valid"] or merged != values["increment_patch_and_diff.json"]:
        raise ValueError("saved_resume_original_patch_or_contract_mismatch")
    from .plan_normalization import merged_queue
    from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import validate_planner_outline_distinctness
    outline = validate_planner_outline_distinctness({"scenes": [entry["gate_scene"] for entry in merged_queue(prepared, merged)]})
    if outline["valid"] or outline != values["merged_outline_check.json"]:
        raise ValueError("saved_resume_original_outline_stop_mismatch")
    source_authorization = manifest["source_authorization"]
    if digest(source_authorization["path"]) != source_authorization["sha256"]:
        raise ValueError("saved_resume_source_authorization_sha_mismatch")
    authorization = read(source_authorization["path"])
    expected = {"explicitly_authorized": True, "policy_id": bundle["policy_id"],
                "identity": bundle["identity"], "bundle_sha256": sha(bundle), "budget_sha256": sha(budget),
                "model_config_sha256": sha(model), "experiment_scope": scope}
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise ValueError("saved_resume_original_authorization_binding_mismatch")
    unchanged_contracts = {"planning.py", "root_planning.py", "root_scope.py", "root_prompts.py",
                           "prompts.py", "contracts.py", "promotion.py", "retention.py", "assembly.py"}
    code_identity = authorization.get("implementation_manifest", {})
    protected = {path: expected for path, expected in code_identity.items()
                 if Path(path).name in unchanged_contracts}
    if not {"root_planning.py", "planning.py", "contracts.py"} <= {Path(path).name for path in protected}:
        raise ValueError("saved_resume_original_validator_identity_missing")
    if any(digest(path) != expected for path, expected in protected.items()):
        raise ValueError("saved_resume_original_contract_code_changed")
    ledger_path = Path(manifest["ledger_path"]).resolve()
    if str(ledger_path) != report["persistent_ledger_path"] or digest(ledger_path) != manifest["ledger_sha256"]:
        raise ValueError("saved_resume_persistent_ledger_changed")
    journal = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    if journal != values["ledger_snapshot.json"]:
        raise ValueError("saved_resume_original_ledger_snapshot_mismatch")
    header = {"kind": "identity", "root_id": bundle["identity"]["root_id"],
              "round_index": bundle["identity"]["round_index"], "execution_scope": scope, "budget": budget}
    attempts = [row for row in journal if row["kind"] == "attempt"]
    if journal[0] != header or len(attempts) != 1 or attempts[0]["role"] != "supplement":
        raise ValueError("saved_resume_ledger_consumption_mismatch")
    validate_frozen_sources(bundle)
    context = {"version": VERSION, "identity": bundle["identity"], "experiment_scope": scope,
        "source_ledger_sha256": manifest["ledger_sha256"], "source_ledger_header": header,
        "source_ledger_path": str(ledger_path), "patch_sha256": sha(patch),
        "source_identity": {"source_run": str(source), "source_manifest_sha256": sha(manifest),
                            "old_execution_authorization_sha256": source_authorization["sha256"],
                            "original_stop": report["status"]}}
    return {"bundle": bundle, "budget": budget, "model_config": model,
            "saved_patch": patch, "resume_context": context, "source_manifest": deepcopy(manifest)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-manifest", type=Path)
    source_group.add_argument("--gate-reuse-manifest", type=Path,
                              help="Explicit approved Gate compatibility checkpoint; no new Gate requests")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--normalization-version", required=True, choices=(NORMALIZATION_VERSION, SELECTION_VERSION))
    parser.add_argument("--credential-config", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    with offline_network_guard():
        manifest = read(args.gate_reuse_manifest or args.source_manifest)
        if manifest.get("normalization_version") != args.normalization_version:
            raise ValueError("resume_manifest_selected_version_mismatch")
        if args.gate_reuse_manifest:
            from .gate_result_resume import validate_gate_reuse_source
            source = validate_gate_reuse_source(manifest)
        else:
            from .formatted_resume_source import VERSION as FORMAT_RESUME_VERSION, validate_formatted_source
            source = (validate_formatted_source(manifest) if manifest.get("version") == FORMAT_RESUME_VERSION
                      else validate_saved_source(manifest))
        bundle, budget, model = (source[name] for name in ("bundle", "budget", "model_config"))
        scope = load_experiment_scope(args.experiment_id, bundle=source.get("original_bundle", bundle), budget=budget,
                                     model_config=model, production_registry=production_registry())
        if scope != source["resume_context"]["experiment_scope"]:
            raise ValueError("resume_registered_experiment_scope_mismatch")
        registry = experiment_registry()
        key = hashlib.sha256(json.dumps({"experiment_id": args.experiment_id}, sort_keys=True).encode()).hexdigest()
        if str((registry / (key + ".jsonl")).resolve()) != source["resume_context"]["source_ledger_path"]:
            raise ValueError("resume_requires_same_persistent_experiment_ledger")
        preflight = prepare_saved_resume(bundle, source["saved_patch"],
            normalization_version=args.normalization_version, source_identity=source["resume_context"]["source_identity"])
        if not preflight["valid"] or not args.execute:
            if args.credential_config and not args.execute:
                parser.error("offline resume must not supply credentials")
            args.run_dir.mkdir(parents=True, exist_ok=False)
            result = {"status": preflight["status"], "external_call_count": 0,
                "resume_claimed": False, "supplement_previously_consumed": True,
                "experiment_scope": scope, "normalization_version": args.normalization_version,
                "new_calls": [], "retained_scene_count": 0, "retained_turn_count": 0,
                "source_context": source["resume_context"], "offline_only": True}
            for name, value in (("resume_preflight", preflight), ("final_report", result), ("source_manifest", source["source_manifest"])):
                with (args.run_dir / (name + ".json")).open("x", encoding="utf-8") as handle:
                    json.dump(value, handle, ensure_ascii=False, indent=2)
            print(json.dumps({"status": result["status"], "external_call_count": 0}))
            return result
        if not args.authorization:
            parser.error("explicit current execution authorization required")
        authorization = read(args.authorization)
        validate_authorization(authorization=authorization, bundle=bundle, budget=budget, model_config=model)
        if (authorization.get("experiment_scope") != scope
                or authorization.get("resume_source_manifest_sha256") != sha(source["source_manifest"])
                or authorization.get("normalization_version") != args.normalization_version
                or authorization.get("resume_explicitly_authorized") is not True):
            raise ValueError("current_saved_patch_resume_authorization_mismatch")
        if "original_bundle" in source and (
                authorization.get("supplement_format_version") != bundle.get("supplement_format_version")
                or authorization.get("formatted_supplement_resume_explicitly_authorized") is not True):
            raise ValueError("explicit_formatted_supplement_resume_authorization_required")
        if args.gate_reuse_manifest and (
                authorization.get("gate_result_reuse_explicitly_authorized") is not True
                or authorization.get("gate_reuse_manifest_sha256") != sha(source["source_manifest"])):
            raise ValueError("explicit_gate_result_reuse_authorization_required")
        credential_source = authorization.get("credential_source")
        if args.credential_config:
            if not isinstance(credential_source, dict) or credential_source.get("path") != str(args.credential_config.resolve()):
                raise ValueError("resume_credential_source_mismatch")
        elif credential_source is not None:
            raise ValueError("authorized_credential_source_requires_explicit_cli_argument")
        source["resume_context"]["current_execution_authorization"] = {
            "path": str(args.authorization.resolve()), "sha256": digest(args.authorization),
            "implementation_manifest": authorization["implementation_manifest"],
            "source_manifest_sha256": sha(source["source_manifest"])}
    with credential_environment(model, credential_source):
        backend = SingleShotTransport(model, budget)
        result = execute_policy(bundle=bundle, budget=budget, backend=backend, state_registry=registry,
            artifact_dir=args.run_dir, live=True, experiment_scope=scope,
            saved_patch=source["saved_patch"], resume_context=source["resume_context"],
            normalization_version=args.normalization_version, gate_reuse=source.get("gate_reuse"))
    print(json.dumps({"status": result["status"], "external_call_count": result["external_call_count"]}))
    return result


if __name__ == "__main__":
    main()
