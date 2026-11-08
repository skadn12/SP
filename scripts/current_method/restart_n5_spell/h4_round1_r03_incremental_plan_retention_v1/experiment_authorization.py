"""Explicit, pinned extra experiments. Loading approval never claims a request."""
import hashlib
import json
import fcntl
import os
from pathlib import Path
import re
import tempfile

from . import POLICY_ID
from .planning import prepare_original, sha
from .prompts import render_supplement
from .supplement_contract_alignment import select_supplement_version
from .root_scope import is_root_scope, prepare_bundle, render_for_bundle

VERSION = "extra_experiment_scope_v1"


def approval_directory():
    return Path(__file__).resolve().parent / "experiment_authorizations"


def authorization_index_path():
    return approval_directory() / "approved_records.jsonl"


def _read_index(handle):
    handle.seek(0)
    result = {}
    for line in handle:
        row = json.loads(line)
        if (not isinstance(row, dict) or set(row) != {"experiment_id", "record_sha256"}
                or not isinstance(row["experiment_id"], str)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", row["experiment_id"])
                or not isinstance(row["record_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["record_sha256"])
                or row["experiment_id"] in result):
            raise ValueError("invalid_experiment_approval_index")
        result[row["experiment_id"]] = row["record_sha256"]
    return result


def approved_records():
    """Read only; a missing index never creates an authorization or a ledger."""
    path = authorization_index_path()
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        return _read_index(handle)


def experiment_registry():
    return Path(__file__).resolve().parents[5] / "restart-n5-runs/policy_state/extra_experiment_scope_v1"


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding_for(bundle, budget, model_config):
    """Use the same parsed candidate registry and renderer as the workflow."""
    root_mode = is_root_scope(bundle)
    prepared = prepare_bundle(bundle)
    if not prepared["valid_for_new_policy"]:
        raise ValueError("experiment_original_plan_invalid")
    prompt = render_for_bundle(bundle, budget, prepared)
    template = prompt.get("supplement_contract_identity")
    if template is None:
        raise ValueError("experiment_explicit_aligned_template_required")
    result = {"policy_id": bundle["policy_id"], "identity": bundle["identity"],
        "bundle_sha256": sha(bundle), "budget_sha256": sha(budget),
        "model_config_sha256": sha(model_config), "budget": budget,
        "history_sha256": bundle["history_sha256"],
        "history_turn_count": len(bundle["parent"]["dialogue"]["turns"]),
        "original_plan_sha256": bundle["original_plan_sha256"],
        "frozen_sources": bundle["frozen_sources"],
        "shared_feedback_sha256": sha(bundle["shared_feedback_view"]),
        "template": template, "realizer_mode": bundle["realizer_mode"],
        "allowed_source_ids": prepared["eligible_source_candidate_ids"] if root_mode else prepared["affected_candidate_ids"],
        "real_request_limit": budget["total_requests"] - budget["role_requests"]["closure"]}
    if root_mode:
        result.update(incremental_scope=bundle["incremental_scope"], source_scope=prepared["source_scope"])
    return result


def load_experiment_scope(experiment_id, *, bundle, budget, model_config, production_registry):
    """Read approved immutable binding and old lineage; never load credentials."""
    expected_sha = approved_records().get(experiment_id)
    if expected_sha is None:
        raise ValueError("unknown_extra_experiment_authorization")
    path = approval_directory() / (experiment_id + ".json")
    if file_sha(path) != expected_sha:
        raise ValueError("experiment_approval_record_sha_mismatch")
    record = json.loads(path.read_text())
    return _validate_record(record, experiment_id, expected_sha, bundle=bundle, budget=budget,
                            model_config=model_config, production_registry=production_registry)


def _validate_record(record, experiment_id, expected_sha, *, bundle, budget, model_config, production_registry):
    if (record["version"] != VERSION or record["experiment_id"] != experiment_id
            or record["explicit_extra_validation_authorized"] is not True):
        raise ValueError("experiment_approval_identity_mismatch")
    if record["binding"] != binding_for(bundle, budget, model_config):
        raise ValueError("experiment_binding_mismatch")
    prior = record["prior_production"]
    identity = {k: bundle["identity"][k] for k in ("root_id", "round_index")}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    ledger_path = Path(production_registry) / (key + ".jsonl")
    if str(ledger_path.resolve()) != prior["ledger_path"] or file_sha(ledger_path) != prior["ledger_sha256"]:
        raise ValueError("experiment_prior_production_ledger_mismatch")
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    if not any(r.get("kind") == "attempt" and r.get("role") == "supplement" for r in rows):
        raise ValueError("experiment_prior_consumption_evidence_missing")
    for item in prior["evidence_files"]:
        if file_sha(item["path"]) != item["sha256"]:
            raise ValueError("experiment_prior_evidence_sha_mismatch")
    return {"mode": "extra_experiment", "experiment_id": experiment_id,
        "grant_sha256": expected_sha, "source_run": prior["run_path"],
        "production_ledger_path": prior["ledger_path"],
        "production_ledger_sha256": prior["ledger_sha256"]}


def register_experiment_record(record_path, *, expected_sha256, explicitly_authorized,
                               bundle, budget, model_config, production_registry):
    """Explicit operator action after user approval, never called by run.main.

    The fixed local index is the trust boundary, not an approval inference engine.
    Atomic record publication precedes the durable index append. No call-state I/O.
    """
    if explicitly_authorized is not True:
        raise ValueError("explicit_registration_authorization_required")
    content = Path(record_path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("registration_source_sha_mismatch")
    record = json.loads(content)
    experiment_id = record.get("experiment_id")
    if not isinstance(experiment_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
        raise ValueError("invalid_experiment_id")
    expected_sha = hashlib.sha256(content).hexdigest()
    _validate_record(record, experiment_id, expected_sha, bundle=bundle, budget=budget,
                     model_config=model_config, production_registry=production_registry)
    directory = approval_directory()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (experiment_id + ".json")
    with authorization_index_path().open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        index = _read_index(handle)
        if path.exists():
            if json.loads(path.read_text()) != record:
                raise ValueError("experiment_registration_conflict")
            expected_sha = file_sha(path)
        if experiment_id in index:
            if not path.exists() or index[experiment_id] != expected_sha:
                raise ValueError("experiment_registration_conflict")
            return {"experiment_id": experiment_id, "record_sha256": expected_sha,
                    "status": "ALREADY_REGISTERED", "opportunity_consumed": False}
        if not path.exists():
            descriptor, temporary = tempfile.mkstemp(prefix=".registration-", dir=directory)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
                os.link(temporary, path)
            finally:
                os.unlink(temporary)
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps({"experiment_id": experiment_id, "record_sha256": expected_sha}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {"experiment_id": experiment_id, "record_sha256": expected_sha,
            "status": "REGISTERED", "opportunity_consumed": False}
