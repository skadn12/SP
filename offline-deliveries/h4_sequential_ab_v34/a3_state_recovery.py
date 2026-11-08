"""New run: one model-selected reference patch, then the frozen A S03 flow.

The source A3_SHORTFALL and its two state calls remain immutable. This extension
does not reissue those calls, edit their responses, or select replacement IDs.
Only the model may return the existing reference_patch contract. All downstream
state/plan/event/dialogue admission and replay uses the frozen A3 wrapper.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import a3_continuation as a

r = a.r
ENTRYPOINT = str(Path(__file__).resolve())
MODE = "one_model_reference_patch_then_frozen_A_S03_flow"
SOURCE = r.HERE / "runs/a3_continuation_01"
SOURCE_REQUEST = "002-cc-train-episode-104814-A-S03-state-2"
SOURCE_CALLS = 2
RECOVERY_CALL_MAX = 1


def _derive(raw, context, validation):
    history = context["cumulative_history"]
    r.base.require(context["root_id"] == a.ROOT and context["route"] == a.ROUTE
                   and context["scene_id"] == a.SCENE and not context["dataset_cards"], "recovery_source_scope_invalid")
    r.base.require(len(history) == 120 and r.base.digest(history) == a.HISTORY_SHA, "recovery_source_history_changed")
    try:
        r.state_contract.validate_state(raw, a.ROOT, a.ROUTE, a.SCENE, history)
    except (ValueError, TypeError) as error:
        expected = {"status": "REJECT", "reason": str(error), "errors": getattr(error, "errors", [])}
        r.base.require(validation == expected, "recovery_source_failure_did_not_reproduce")
        repair = r.state_contract.build_state_repair(error, raw, history=history)
    else:
        raise ValueError("recovery_source_already_valid")
    r.base.require(repair.get("repair_mode") == "reference_patch" and len(repair["required_paths"]) == 8,
                   "recovery_requires_exact_eight_reference_only_faults")
    r.base.require(all(row["candidate_turn_refs"] for row in repair["citation_reference_candidates"]),
                   "recovery_original_quote_missing_from_history")
    return repair, history


def _source():
    manifest = a.verify(SOURCE)
    report = r.base.read(SOURCE / "REPORT.json")
    merge = r.base.read(SOURCE / "MERGE_MANIFEST.json")
    audit = r.base.read(SOURCE / "FINAL_AUDIT.json")
    r.base.verify_bindings(merge["files"])
    r.base.require(audit["status"] == "PASS" and not audit["failures"]
                   and audit["generation_status"] == "A3_SHORTFALL"
                   and audit["manifest_sha256"] == r.base.digest(manifest)
                   and audit["merge_manifest_sha256"] == r.base.digest(merge), "recovery_source_audit_invalid")
    directories = sorted(p for p in (SOURCE / "requests").iterdir() if p.is_dir())
    r.base.require(len(directories) == SOURCE_CALLS == report["logical_calls"] == audit["logical_calls"],
                   "recovery_source_call_inventory_changed")
    r.base.require(report["status"] == "A3_SHORTFALL" and report["accepted_A_scenes"] == 2
                   and report["new_accepted_A_scenes"] == 0, "recovery_source_not_A2_shortfall")
    r.base.require(report["result"]["failures"] == [{"scene_id": a.SCENE,
                   "reason": "state_audit_incomplete_after_retry"}], "recovery_source_shortfall_changed")
    r.base.require(not (SOURCE / "roots" / a.ROOT / "A/S03.LATEST_STATE.json").exists(),
                   "recovery_source_contains_accepted_state")
    for directory in directories:
        proof = r.base.read(directory / "PROOF.json")
        r.base.require(proof["root_id"] == a.ROOT and proof["route"] == a.ROUTE
                       and proof["scene_id"] == a.SCENE and proof["role"] == "state_extractor",
                       "recovery_source_non_state_call")
    directory = SOURCE / "requests" / SOURCE_REQUEST
    request = r.base.read(directory / "REQUEST.json")
    context = json.JSONDecoder().raw_decode(request["messages"][0]["content"].split("\nSTATE_INPUT_JSON\n", 1)[1])[0]
    raw = r.base.read(directory / "RESPONSE.json")["text"]
    proof = r.base.read(directory / "PROOF.json")
    r.base.require(proof["payload_sha256"] == r.base.digest(request)
                   and proof["history_sha256"] == a.HISTORY_SHA, "recovery_source_proof_mismatch")
    repair, history = _derive(raw, context, r.base.read(directory / "VALIDATION.json"))
    files = [SOURCE / name for name in ("MANIFEST.json", "REPORT.json", "MERGE_MANIFEST.json", "FINAL_AUDIT.json")]
    files += [p for directory in directories for p in directory.iterdir() if p.is_file()]
    return {"repair": repair, "history": history, "raw": raw, "request": request, "files": sorted(files),
            "source_http_attempts": report["http_attempts"]}


def _load(run):
    meta = r.base.read(run / "STATE_RECOVERY.json")
    r.base.require(meta["execution_entrypoint"] == ENTRYPOINT and meta["recovery_mode"] == MODE
                   and meta["source_run"] == str(SOURCE) and meta["source_request"] == SOURCE_REQUEST,
                   "recovery_metadata_scope_changed")
    for row in meta["frozen_source_files"]:
        source, frozen = Path(row["original_path"]), Path(row["frozen_path"])
        r.base.require(r.base.digest(source.read_bytes()) == row["sha256"]
                       == r.base.digest(frozen.read_bytes()), "recovery_frozen_source_changed")
    directory = run / "frozen_recovery_source/requests" / SOURCE_REQUEST
    request = r.base.read(directory / "REQUEST.json")
    context = json.JSONDecoder().raw_decode(request["messages"][0]["content"].split("\nSTATE_INPUT_JSON\n", 1)[1])[0]
    raw = r.base.read(directory / "RESPONSE.json")["text"]
    repair, history = _derive(raw, context, r.base.read(directory / "VALIDATION.json"))
    r.base.require(repair == meta["reference_repair"] and r.base.digest(raw) == meta["source_raw_sha256"],
                   "recovery_spec_or_raw_changed")
    r.base.require(history == a._history(run), "recovery_history_not_new_run_A2")
    return meta, repair, raw, history


def prepare(run):
    source = _source()
    original_write = r.base.write
    def write(path, value):
        if Path(path) == run / "MANIFEST.json":
            receipts = []
            for original in source["files"]:
                frozen = run / "frozen_recovery_source" / original.relative_to(SOURCE)
                frozen.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(original, frozen)
                receipts.append({"original_path": str(original), "frozen_path": str(frozen),
                                 "sha256": r.base.digest(frozen.read_bytes())})
            meta = {"execution_entrypoint": ENTRYPOINT, "recovery_mode": MODE, "source_run": str(SOURCE),
                "source_request": SOURCE_REQUEST, "source_raw_sha256": r.base.digest(source["raw"]),
                "source_request_sha256": r.base.digest(source["request"]), "source_history_sha256": a.HISTORY_SHA,
                "old_source_state_requests": SOURCE_CALLS, "old_source_http_attempts": source["source_http_attempts"],
                "new_reference_patch_logical_call_max": RECOVERY_CALL_MAX,
                "reference_repair": source["repair"], "frozen_source_files": receipts,
                "source_failed_calls_reissued": False, "program_selects_replacement_references": False,
                "substantive_state_fields_immutable": True, "semantic_certified": False}
            original_write(run / "STATE_RECOVERY.json", meta)
            original_write(run / "previews/S03.reference_recovery.json",
                r.state_contract.build_state_request(a.ROOT, a.ROUTE, a.SCENE, source["history"], [], repair=source["repair"]))
            value = deepcopy(value)
            value.update(execution_entrypoint=ENTRYPOINT, recovery_mode=MODE,
                         state_recovery_sha256=r.base.digest(meta), old_source_state_requests=SOURCE_CALLS,
                         new_reference_patch_logical_call_max=RECOVERY_CALL_MAX)
            paths = {Path(row["path"]) for row in value["files"]}
            paths.update(source["files"])
            paths.update(Path(row["frozen_path"]) for row in receipts)
            paths.update((Path(ENTRYPOINT), run / "STATE_RECOVERY.json", run / "previews/S03.reference_recovery.json"))
            value["files"] = r.base.bindings(sorted(paths))
        original_write(path, value)
    with patch.object(r.base, "write", side_effect=write):
        result = a.prepare(run)
    verify(run)
    return {**result, "execution_entrypoint": ENTRYPOINT, "recovery_mode": MODE,
            "old_source_state_requests": SOURCE_CALLS, "new_reference_patch_logical_call_max": RECOVERY_CALL_MAX}


def verify(run):
    manifest = a.verify(run)
    r.base.require(manifest.get("execution_entrypoint") == ENTRYPOINT and manifest.get("recovery_mode") == MODE
                   and manifest.get("old_source_state_requests") == SOURCE_CALLS
                   and manifest.get("new_reference_patch_logical_call_max") == RECOVERY_CALL_MAX,
                   "not_an_A3_reference_recovery_run")
    meta, repair, _raw, history = _load(run)
    r.base.require(r.base.digest(meta) == manifest["state_recovery_sha256"], "recovery_metadata_hash_changed")
    r.base.require(r.base.read(run / "previews/S03.reference_recovery.json") ==
        r.state_contract.build_state_request(a.ROOT, a.ROUTE, a.SCENE, history, [], repair=repair), "recovery_preview_changed")
    return manifest


@contextmanager
def recovery_hook(run):
    """One scoped logical call in live or deterministic original-wrapper replay."""
    meta, repair, old_raw, frozen_history = _load(run)
    calls = 0
    def recover(caller, actual_run, root, route, history, cards, scene_id, *, purpose_input=None):
        nonlocal calls
        r.base.require(Path(actual_run) == run and root == a.ROOT and route == a.ROUTE and scene_id == a.SCENE
                       and cards == [] and purpose_input is None and history == frozen_history,
                       "reference_recovery_hook_scope_changed")
        calls += 1
        r.base.require(calls <= RECOVERY_CALL_MAX, "reference_recovery_call_limit_exceeded")
        payload = r.state_contract.build_state_request(root, route, scene_id, history, [], repair=repair)
        proof = {"root_id": root, "route": route, "role": "state_extractor", "scene_id": scene_id,
            **r.history_proof(run, root, route, history), "purpose_input_sha256": r.base.digest(None),
            "validation_scope": "model_reference_patch_then_unchanged_state_contract",
            "execution_entrypoint": ENTRYPOINT, "recovery_mode": MODE,
            "state_recovery_sha256": r.base.digest(meta), "source_state_raw_sha256": r.base.digest(old_raw),
            "repair_spec_sha256": r.base.digest(repair), "old_source_state_requests": SOURCE_CALLS,
            "new_reference_patch_logical_call_max": RECOVERY_CALL_MAX}
        raw, directory = caller.call(f"{root}-A-{scene_id}-state-reference-recovery-1", payload, proof)
        merge_record = {"previous_raw_sha256": r.base.digest(old_raw), "raw_patch_sha256": r.base.digest(raw),
                        "repair_spec_sha256": r.base.digest(repair), "source_state_request": str(SOURCE / "requests" / SOURCE_REQUEST)}
        try:
            candidate = r.state_contract.merge_state_repair(old_raw, raw, repair, history)
        except (ValueError, TypeError) as error:
            r.base.write(directory / "FORMAT_REPAIR_MERGE.json", {**merge_record, "status": "REJECT", "reason": str(error)})
            r.base.write(directory / "VALIDATION.json", {"status": "REJECT", "reason": str(error),
                "errors": [], "merge_failed": True, "reference_recovery_exhausted": True})
            return None
        r.base.write(directory / "FORMAT_REPAIR_MERGE.json", {**merge_record, "status": "PASS", "candidate": candidate,
            "changed_paths": repair["required_paths"], "semantic_certified": False})
        try:
            snapshot = r.state_contract.validate_state(candidate, root, route, scene_id, history)
        except (ValueError, TypeError) as error:
            r.base.write(directory / "VALIDATION.json", {"status": "REJECT", "reason": str(error),
                "errors": getattr(error, "errors", []), "reference_recovery_exhausted": True})
            return None
        r.base.write(directory / "VALIDATION.json", {"status": "PASS", "semantic_certified": False,
            "latest_state_sha256": r.base.digest(snapshot), "reference_recovery_completed": True,
            "changed_paths": repair["required_paths"]})
        return snapshot
    with patch.object(r, "extract_latest_state", side_effect=recover):
        yield


def _accounting(run, value):
    meta = r.base.read(run / "STATE_RECOVERY.json")
    state_requests = [p for p in (run / "requests").glob("*/PROOF.json")
                      if r.base.read(p).get("role") == "state_extractor"]
    return {**value, "execution_entrypoint": ENTRYPOINT, "recovery_mode": MODE,
            "state_recovery_sha256": r.base.digest(meta),
            "old_source_state_requests": SOURCE_CALLS, "old_source_http_attempts": meta["old_source_http_attempts"],
            "new_state_reference_patch_requests": len(state_requests),
            "old_source_calls_in_new_run_count": False,
            "total_lineage_logical_calls": SOURCE_CALLS + value.get("logical_calls", 0)}


def live(run):
    verify(run)
    original_write = r.base.write
    def write(path, value):
        if Path(path) == run / "REPORT.json":
            value = _accounting(run, value)
        original_write(path, value)
    with recovery_hook(run), patch.object(r.base, "write", side_effect=write):
        a.live(run)
    verify(run)
    return r.base.read(run / "REPORT.json")


def audit(run):
    verify(run)
    report_path = run / "REPORT.json"
    if report_path.exists():
        report = r.base.read(report_path)
        expected = _accounting(run, report)
        r.base.require(report == expected, "recovery_report_accounting_changed")
        r.base.require(report["new_state_reference_patch_requests"] <= RECOVERY_CALL_MAX,
                       "recovery_state_call_inventory_exceeded")
    with recovery_hook(run):
        result = a.audit(run)
    augmented = _accounting(run, result)
    destination = run / ("STATE_RECOVERY_AUDIT.json" if (run / "LIVE_STARTED.json").exists() else "STATE_RECOVERY_PREPARE_AUDIT.json")
    if destination.exists():
        index = 2
        while destination.with_name(destination.stem + f"_{index:03d}.json").exists():
            index += 1
        destination = destination.with_name(destination.stem + f"_{index:03d}.json")
    r.base.write(destination, augmented)
    return augmented


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "live", "audit"])
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](args.run.resolve()), ensure_ascii=False, indent=2))
