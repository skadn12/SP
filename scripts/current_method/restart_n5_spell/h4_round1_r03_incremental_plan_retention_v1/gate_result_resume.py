"""Load explicitly approved, frozen Gate compatibility views for downstream use.

This is a checkpoint loader, not a second Gate or a new authorization registry.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .planning import sha
from .task_selection import VERSION as SELECTION_VERSION
from .run import validate_frozen_sources
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import validate_semantic_gate_output

VERSION = "approved_gate_result_resume_v1"
ADAPTER_VERSION = "gate_full_history_empty_pass_field_alias_candidate_v1"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_gate_reuse_manifest(delivery, source_run):
    """Pin the already reviewed files; registration and counters are untouched."""
    delivery, source = Path(delivery).resolve(), Path(source_run).resolve()
    report = read(source / "execution/final_report.json")
    review = read(delivery / "old_response_reuse_review.json")
    paths = [path for path in (source / "execution").rglob("*.json")]
    paths += [source / name for name in ("selection_sidecar.json", "selected_execution_plan.json",
                                        "resume_execution_authorization.json")]
    paths += [delivery / name for name in ("old_response_reuse_review.json", "response_reuse_closure.md",
        "final_report.json", "final_report.md", "method_version.json", "full_history_output_contract.json")]
    for row in review:
        paths += [Path(row[key]) for key in ("request_path", "raw_response_path", "http_response_path", "format_candidate_path")]
    ledger = Path(report["persistent_ledger_path"]).resolve()
    return {"version": VERSION, "delivery": str(delivery), "source_run": str(source),
            "files": {str(path.resolve()): digest(path) for path in sorted(set(paths))},
            "identity": report["identity"], "execution_scope": report["execution_scope"],
            "ledger_path": str(ledger), "ledger_sha256": digest(ledger),
            "normalization_version": SELECTION_VERSION,
            "selected_queue_sha256": sha(read(source / "selected_execution_plan.json"))}


def validate_gate_reuse_source(manifest):
    """Identity checks plus unchanged structural validation; no semantic review."""
    from .workflow import prepare_saved_resume
    if manifest.get("version") != VERSION or manifest.get("normalization_version") != SELECTION_VERSION:
        raise ValueError("gate_reuse_manifest_version_mismatch")
    source, delivery = Path(manifest["source_run"]), Path(manifest["delivery"])
    for path, expected in manifest["files"].items():
        if digest(path) != expected:
            raise ValueError("gate_reuse_frozen_file_sha_mismatch:" + path)
    expected_manifest = build_gate_reuse_manifest(delivery, source)
    if expected_manifest != manifest:
        raise ValueError("gate_reuse_manifest_or_ledger_changed")
    execution = source / "execution"
    report = read(execution / "final_report.json")
    bundle, budget, model = (read(execution / name) for name in (
        "input_freeze.json", "execution_budget.json", "model_config.json"))
    scope = report["execution_scope"]
    identity = bundle["identity"]
    if (manifest["identity"] != identity or report["identity"] != identity
            or manifest["execution_scope"] != scope
            or report["retained_scene_count"] or report["retained_turn_count"]
            or report["status"] != "EXECUTION_ENDED_WITH_DEFERRED_BUDGET"
            or report["normalization_version"] != SELECTION_VERSION
            or bundle["realizer_mode"] != "text_free"):
        raise ValueError("gate_reuse_source_checkpoint_identity")
    validate_frozen_sources(bundle)
    patch = read(execution / "saved_patch.json")
    queue = read(source / "selected_execution_plan.json")
    if queue != read(execution / "normalized_execution_queue.json") or sha(queue) != manifest["selected_queue_sha256"]:
        raise ValueError("gate_reuse_selected_queue_changed")
    source_identity = {"source_run": str(execution), "source_manifest_sha256": sha(manifest),
                       "original_stop": report["status"], "reuse_delivery": str(delivery)}
    prepared = prepare_saved_resume(bundle, patch, normalization_version=SELECTION_VERSION,
                                   source_identity=source_identity)
    if not prepared["valid"] or prepared["normalization"]["queue"] != queue:
        raise ValueError("gate_reuse_plan_not_identical_to_saved_selection")
    ledger = Path(manifest["ledger_path"])
    journal = [json.loads(line) for line in ledger.read_text().splitlines()]
    if journal != read(execution / "ledger_snapshot.json"):
        raise ValueError("gate_reuse_ledger_not_original_checkpoint")
    attempts = [row for row in journal if row["kind"] == "attempt"]
    if ({role: sum(row["role"] == role for row in attempts) for role in budget["role_requests"]}
            != {"planner": 0, "supplement": 1, "gate": 8, "realizer": 0, "adherence": 0, "closure": 0}):
        raise ValueError("gate_reuse_downstream_already_attempted_or_checkpoint_mismatch")
    header = {"kind": "identity", "root_id": identity["root_id"], "round_index": identity["round_index"],
              "execution_scope": scope, "budget": budget}
    if journal[0] != header:
        raise ValueError("gate_reuse_ledger_header_mismatch")
    review = read(delivery / "old_response_reuse_review.json")
    if len(review) != 8:
        raise ValueError("gate_reuse_expected_review_count")
    by_phase, records = {}, []
    for row in review:
        if (row["request_identity"] != identity or row["history_sha256"] != bundle["history_sha256"]
                or row["format_compatibility_possible"] is not True
                or row["candidate_and_context_equal_to_new_render"] is not True):
            raise ValueError("gate_reuse_review_identity_mismatch")
        for path_key, sha_key in (("request_path", "request_sha256"), ("raw_response_path", "raw_response_sha256"),
                                 ("http_response_path", "http_response_sha256"), ("format_candidate_path", "format_candidate_sha256")):
            if digest(row[path_key]) != row[sha_key]:
                raise ValueError("gate_reuse_review_file_changed:" + path_key)
        entry = next((item for item in queue if item["scene"]["phase_id"] == row["phase_id"]), None)
        if entry is None or row["phase_id"] in by_phase or entry["scene"]["source_candidate_id"] != row["candidate_id"]:
            raise ValueError("gate_reuse_candidate_not_selected_or_duplicate")
        request = read(row["request_path"])
        call_id = row["call_id"]
        if (request["role"] != "gate" or request["call_id"] != call_id or request["identity"] != identity
                or request["execution_scope"] != scope or sha(request["messages"]) != row["messages_sha256"]
                or len([item for item in attempts if item["role"] == "gate" and item["call_id"] == call_id]) != 1):
            raise ValueError("gate_reuse_request_or_call_identity_mismatch")
        if [item.get("status") for item in journal if item["kind"] == "result" and item.get("call_id") == call_id] != ["RESPONSE_RECEIVED"]:
            raise ValueError("gate_reuse_response_not_confirmed")
        original = json.loads(read(row["raw_response_path"]))
        candidate_file = read(row["format_candidate_path"])
        candidate = candidate_file["response_candidate"]
        if (candidate_file["adapter_version"] != ADAPTER_VERSION
                or candidate_file["source_raw_response_sha256"] != row["raw_response_sha256"]):
            raise ValueError("gate_reuse_adapter_identity_mismatch")
        transformed = deepcopy(original)
        reviews = transformed.get("candidate_reviews")
        if not isinstance(reviews, list) or len(reviews) != 1 or "evidence_turn_refs" in reviews[0]:
            raise ValueError("gate_reuse_ambiguous_original_fields")
        reviews[0]["evidence_turn_refs"] = reviews[0].pop("supporting_memory_refs")
        if transformed != candidate or reviews[0]["candidate_ref"] != row["candidate_id"]:
            raise ValueError("gate_reuse_not_only_approved_field_transform")
        validation = validate_semantic_gate_output(candidate, plan={"scenes": [entry["gate_scene"]]},
            parent=bundle["parent"], memory_only_fast=False, minimum_pass_candidates=1)
        if not validation["valid"] or validation != candidate_file["local_contract_revalidation_only"]:
            raise ValueError("gate_reuse_current_structure_check_failed")
        loaded = {"phase_id": row["phase_id"], "candidate_id": row["candidate_id"],
                  "entry_sha256": sha(entry), "gate_scene_sha256": sha(entry["gate_scene"]),
                  "history_sha256": bundle["history_sha256"], "response": deepcopy(candidate), "validation": validation}
        by_phase[row["phase_id"]] = loaded
        records.append({**loaded, "status": "AUTHORIZED_FORMAT_COMPATIBLE_GATE_RESULT_LOADED",
            "original_call_id": call_id, "original_request_path": row["request_path"], "original_request_sha256": row["request_sha256"],
            "original_response_path": row["raw_response_path"], "original_response_sha256": row["raw_response_sha256"],
            "compatibility_path": row["format_candidate_path"], "compatibility_sha256": row["format_candidate_sha256"],
            "adapter_version": ADAPTER_VERSION, "original_gate_failure_unchanged": True,
            "new_model_call": False, "semantic_truth": "NOT_ESTABLISHED", "reused_in_scene_execution": False})
    context = {"version": VERSION, "identity": identity, "experiment_scope": scope,
        "source_ledger_sha256": manifest["ledger_sha256"], "source_ledger_header": header,
        "source_ledger_path": str(ledger), "patch_sha256": sha(patch), "source_identity": source_identity}
    return {"bundle": bundle, "budget": budget, "model_config": model, "saved_patch": patch,
            "resume_context": context, "source_manifest": deepcopy(manifest),
            "gate_reuse": {"version": VERSION, "manifest_sha256": sha(manifest),
                           "selected_queue_sha256": sha(queue), "history_sha256": bundle["history_sha256"],
                           "by_phase": by_phase, "formal_reuse_records": records}}
