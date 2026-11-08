"""Resume a saved format failure through the deterministic supplement adapter."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import ModuleType

from .planning import sha
from .root_scope import prepare_bundle, render_for_bundle, with_supplement_formatting
from .root_planning import merge_root_patch
from .supplement_format import VERSION as FORMAT_VERSION
from .task_selection import VERSION as SELECTION_VERSION
from .run import validate_frozen_sources

VERSION = "formatted_root_patch_resume_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _original_planning_identity(source, authorization):
    current = Path(__file__).resolve().with_name("root_planning.py")
    expected = authorization.get("implementation_manifest", {}).get(str(current))
    if not expected:
        raise ValueError("formatted_resume_original_validator_identity_missing")
    snapshot = source.parent / "method_snapshot" / current.relative_to(current.parents[4])
    if digest(snapshot) != expected:
        raise ValueError("formatted_resume_original_validator_snapshot_mismatch")
    return {"path": str(snapshot), "sha256": expected, "original_path": str(current)}


def build_formatted_resume_manifest(source_run, *, source_authorization):
    """Pin existing source evidence and its old merger without creating state."""
    source = Path(source_run).resolve()
    authorization_path = Path(source_authorization).resolve()
    report = read(source / "final_report.json")
    ledger = Path(report["persistent_ledger_path"]).resolve()
    return {"version": VERSION, "supplement_format_version": FORMAT_VERSION,
        "normalization_version": SELECTION_VERSION, "source_run": str(source),
        "source_authorization": {"path": str(authorization_path), "sha256": digest(authorization_path)},
        "original_planning": _original_planning_identity(source, read(authorization_path)),
        "files": {str(path.relative_to(source)): digest(path)
                  for path in sorted(source.rglob("*.json"))},
        "ledger_path": str(ledger), "ledger_sha256": digest(ledger),
        "identity": report["identity"], "execution_scope": report["execution_scope"],
        "source_status": report["status"]}


def _original_planning(manifest, authorization):
    identity = manifest["original_planning"]
    if identity != _original_planning_identity(Path(manifest["source_run"]), authorization):
        raise ValueError("formatted_resume_original_validator_identity_mismatch")
    # Dependencies that materialize and validate the old plan must still match
    # the authorization. Forwarding prompts and scope dispatch are reproduced
    # against their exact saved outputs below instead of blocking intentional edits.
    protected_names = {"planning.py", "root_prompts.py", "contracts.py", "promotion.py",
                       "retention.py", "assembly.py"}
    protected = {path: expected for path, expected in authorization["implementation_manifest"].items()
                 if Path(path).name in protected_names}
    if protected_names != {Path(path).name for path in protected}:
        raise ValueError("formatted_resume_original_validator_identity_missing")
    if any(digest(path) != expected for path, expected in protected.items()):
        raise ValueError("formatted_resume_original_contract_code_changed")
    module = ModuleType(__package__ + "._saved_root_planning")
    module.__package__ = __package__
    module.__file__ = identity["path"]
    content = Path(identity["path"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != identity["sha256"]:
        raise ValueError("formatted_resume_original_validator_snapshot_mismatch")
    exec(compile(content, identity["path"], "exec"), module.__dict__)
    return module


def validate_formatted_source(manifest):
    """Verify the failed request, replay its exact failure, then format offline."""
    from .workflow import prepare_saved_resume
    if (manifest.get("version") != VERSION
            or manifest.get("supplement_format_version") != FORMAT_VERSION
            or manifest.get("normalization_version") != SELECTION_VERSION):
        raise ValueError("formatted_resume_manifest_version_mismatch")
    source = Path(manifest["source_run"]).resolve()
    for name, expected in manifest["files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or digest(path) != expected:
            raise ValueError("formatted_resume_source_file_sha_mismatch:" + name)
    source_authorization = manifest["source_authorization"]
    if digest(source_authorization["path"]) != source_authorization["sha256"]:
        raise ValueError("formatted_resume_source_authorization_sha_mismatch")
    if manifest != build_formatted_resume_manifest(source, source_authorization=source_authorization["path"]):
        raise ValueError("formatted_resume_manifest_or_ledger_changed")
    required = {"final_report.json", "input_freeze.json", "execution_budget.json", "model_config.json",
        "experiment_scope.json", "increment_patch_and_diff.json", "original_plan_new_policy_diagnostics.json",
        "ledger_snapshot.json", "supplement_contract_identity.json"}
    if not required <= set(manifest["files"]):
        raise ValueError("formatted_resume_required_evidence_missing")
    values = {name: read(source / name) for name in required}
    report, bundle, budget, model, scope = (values[name] for name in (
        "final_report.json", "input_freeze.json", "execution_budget.json", "model_config.json", "experiment_scope.json"))
    if (report["status"] != "SUPPLEMENT_INVALID_STOP" or manifest["source_status"] != report["status"]
            or "supplement_format_version" in bundle
            or report["identity"] != bundle["identity"] or manifest["identity"] != bundle["identity"]
            or report["execution_scope"] != scope or manifest["execution_scope"] != scope
            or report["retained_scene_count"] != 0 or report["retained_turn_count"] != 0):
        raise ValueError("formatted_resume_source_stop_identity_mismatch")
    calls = report["new_calls"]
    if len(calls) != 1 or calls[0]["role"] != "supplement" or calls[0]["status"] != "RESPONSE_RECEIVED":
        raise ValueError("formatted_resume_requires_exactly_one_received_supplement")
    call = calls[0]
    requests = [(name, read(source / name)) for name in manifest["files"] if name.startswith("requests/")]
    if len(requests) != 1:
        raise ValueError("formatted_resume_request_evidence_count")
    request_name, request = requests[0]
    if (request["role"] != "supplement" or request["call_id"] != call["call_id"]
            or request["identity"] != bundle["identity"] or request["execution_scope"] != scope
            or sha(request["messages"]) != call["messages_sha256"]
            or request.get("model") != model["model"] or request.get("temperature") != model["temperature"]
            or request.get("max_tokens") != budget["max_output_tokens"]["supplement"]):
        raise ValueError("formatted_resume_actual_request_identity_mismatch")
    authorization = read(source_authorization["path"])
    expected_authorization = {"explicitly_authorized": True, "policy_id": bundle["policy_id"],
        "identity": bundle["identity"], "bundle_sha256": sha(bundle), "budget_sha256": sha(budget),
        "model_config_sha256": sha(model), "experiment_scope": scope}
    if any(authorization.get(key) != value for key, value in expected_authorization.items()):
        raise ValueError("formatted_resume_original_authorization_binding_mismatch")
    old_planning = _original_planning(manifest, authorization)
    prepared = prepare_bundle(bundle)
    rendered = render_for_bundle(bundle, budget, prepared)
    if (old_planning.prepare_root_original(bundle) != prepared
            or values["original_plan_new_policy_diagnostics.json"] != prepared
            or request["messages"] != rendered["messages"]
            or request["prompt_version"] != rendered["prompt_version"]
            or request["supplement_contract_identity"] != rendered["supplement_contract_identity"]
            or values["supplement_contract_identity.json"] != rendered["supplement_contract_identity"]):
        raise ValueError("formatted_resume_frozen_prompt_or_prepared_mismatch")
    response_base = "responses/" + Path(request_name).stem
    required_response = {response_base + ".json", response_base + "_envelope.json"}
    if not required_response <= set(manifest["files"]):
        raise ValueError("formatted_resume_response_or_envelope_missing")
    completed = [read(source / name) for name in manifest["files"]
                 if name.startswith(response_base + "_http_")]
    completed = [row for row in completed if row.get("evidence_event") == "completed"]
    if len(completed) != 1:
        raise ValueError("formatted_resume_completed_http_evidence_required")
    completed = completed[0]
    expected_http_identity = {"purpose": "supplement", "call_id": call["call_id"],
        "prompt_version": request["prompt_version"], "model": model["model"],
        "messages_sha256": call["messages_sha256"], "execution_identity": bundle["identity"]}
    if (completed["http_status"] != 200 or completed.get("response_body_complete") is not True
            or completed.get("response_unavailable") is not False
            or completed["request_identity"] != expected_http_identity
            or hashlib.sha256(completed["response_body"].encode()).hexdigest() != completed["saved_body_sha256"]):
        raise ValueError("formatted_resume_original_http_response_mismatch")
    envelope = read(source / (response_base + "_envelope.json"))
    if json.loads(completed["response_body"]) != envelope:
        raise ValueError("formatted_resume_http_envelope_mismatch")
    raw = read(source / (response_base + ".json"))
    if not isinstance(raw, str) or envelope["choices"][0]["message"]["content"] != raw:
        raise ValueError("formatted_resume_model_response_mismatch")
    patch = json.loads(raw)
    merged = old_planning.merge_root_patch(prepared, patch)
    if (merged["valid"] or merged != values["increment_patch_and_diff.json"]
            or merge_root_patch(prepared, patch) != merged):
        raise ValueError("formatted_resume_original_failure_not_reproduced")
    derived = with_supplement_formatting(bundle)
    if prepare_bundle(derived) != prepared or render_for_bundle(derived, budget, prepared) != rendered:
        raise ValueError("formatted_resume_adapter_changed_prompt_or_prepared")
    preflight = prepare_saved_resume(derived, raw, normalization_version=SELECTION_VERSION)
    if not preflight["valid"]:
        raise ValueError("formatted_resume_formatted_patch_or_selection_invalid")
    ledger_path = Path(manifest["ledger_path"]).resolve()
    if str(ledger_path) != report["persistent_ledger_path"] or digest(ledger_path) != manifest["ledger_sha256"]:
        raise ValueError("formatted_resume_persistent_ledger_changed")
    journal = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    if journal != values["ledger_snapshot.json"]:
        raise ValueError("formatted_resume_original_ledger_snapshot_mismatch")
    header = {"kind": "identity", "root_id": bundle["identity"]["root_id"],
        "round_index": bundle["identity"]["round_index"], "execution_scope": scope, "budget": budget}
    attempts = [row for row in journal if row["kind"] == "attempt"]
    if (journal[0] != header or len(attempts) != 1 or attempts[0]["role"] != "supplement"
            or attempts[0]["call_id"] != call["call_id"]
            or any(row["kind"] == "resume_execution" for row in journal)
            or [row.get("status") for row in journal if row["kind"] == "result"] != ["RESPONSE_RECEIVED"]):
        raise ValueError("formatted_resume_ledger_consumption_mismatch")
    validate_frozen_sources(bundle)
    context = {"version": VERSION, "identity": bundle["identity"], "experiment_scope": scope,
        "source_ledger_sha256": manifest["ledger_sha256"], "source_ledger_header": header,
        "source_ledger_path": str(ledger_path), "patch_sha256": sha(raw),
        "source_identity": {"source_run": str(source), "source_manifest_sha256": sha(manifest),
            "old_execution_authorization_sha256": source_authorization["sha256"],
            "original_stop": report["status"], "supplement_format_version": FORMAT_VERSION,
            "original_bundle_sha256": sha(bundle), "execution_bundle_sha256": sha(derived)}}
    return {"original_bundle": bundle, "bundle": derived, "budget": budget, "model_config": model,
        "saved_patch": raw, "resume_context": context, "source_manifest": deepcopy(manifest),
        "original_failure": merged, "format_preflight": preflight}
