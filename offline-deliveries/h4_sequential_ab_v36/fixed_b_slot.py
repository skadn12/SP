"""Fixed consumed-source B experiment; never reserves or publishes training data.

Prepare freezes the actual audited A3, unchanged source card and receipts. Live
uses runner.run_root and its real Caller. Audit replays that same bounded flow
against captured responses; scripted test responses are never live evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import runner as r

VERSION = "v36_fixed_consumed_source_B_single_slot"
ROOT = "cc-train-episode-104814"
SOURCE_ID = "soda:train:6028014"
CARD_ID = "soda:train:6028014_card_1"
ORIGINAL_SCENE = "S02"
SCENE = "S01"
ROUTE = r.ROUTES[1]
PARENT = r.HERE.parent / "h4_sequential_ab_v34/runs/a3_reference_recovery_01"
SOURCE_RUN = r.HERE.parent / "h4_sequential_ab_v34/runs/inherited_a3_b3_01"
LOGICAL_MAX = 15
HTTP_MAX = 30
FORBIDDEN_OUTPUTS = ("SOURCE_RESERVATIONS.json", "UNIQUE_EVENT_ALLOCATION.json",
                     "MERGE_MANIFEST.json", "sources_merged", "REPORT.json")
ROLES = {"purpose_proposer", "state_extractor", "planner", "event_state_extractor", "realizer", "scene_reviewer"}


def checked_a3(parent):
    """Cross-delivery admission with exact audited chain and raw model bindings."""
    parent = Path(parent).resolve()
    manifest, audit, report, merge = (r.base.read(parent / name) for name in
        ("MANIFEST.json", "FINAL_AUDIT.json", "REPORT.json", "MERGE_MANIFEST.json"))
    r.base.require(manifest["version"] == "v34_exact_A2_to_A3_single_slot"
        and manifest["root_id"] == ROOT and manifest["new_scene_slots"] == ["S03"], "parent_A3_scope_invalid")
    r.base.require(audit["status"] == "PASS" and not audit["failures"]
        and audit["generation_status"] == report["status"] == "A3_COMPLETE"
        and audit["version"] == manifest["version"] and audit["accepted_A_scenes"] == 3
        and Path(audit["run"]).resolve() == parent, "parent_A3_not_audited")
    r.base.require(audit["manifest_sha256"] == r.base.digest(manifest)
        and audit["merge_manifest_sha256"] == r.base.digest(merge)
        and audit["logical_calls"] == report["logical_calls"]
        and audit["http_attempts"] == report["http_attempts"], "parent_A3_stale_audit")
    r.base.verify_bindings(manifest["files"])
    r.base.verify_bindings(merge["files"])
    inherited = r.verify_inherited_a(parent)
    folder = parent / "roots" / ROOT / "A"
    result = r.base.read(folder / "RESULT.json")
    r.base.require(result == report["result"] and result["accepted_scenes"] == 3
        and report["new_accepted_A_scenes"] == 1
        and [row["scene_id"] for row in result["scene_metadata"]] == ["S01", "S02", "S03"], "parent_A3_inventory_invalid")
    original = r.base.read(parent / "sources" / f"{ROOT}.json")
    history = deepcopy(original["turns"])
    old_folder = Path(inherited["local_path"]).parent
    r.base.require(result["scene_metadata"][:2] == r.base.read(parent / "INHERITED_RESULT.json")["scene_metadata"],
        "parent_A3_inherited_metadata_changed")
    for scene_id in ("S01", "S02", "S03"):
        scene = r.base.read(folder / f"{scene_id}.json")
        accepted = next(row for row in reversed(r.base.read(folder / f"{scene_id}.ATTEMPTS.json")) if row["status"] == "PASS")
        request, review_dir = Path(accepted["request"]), Path(accepted["review_request"])
        r.base.require(scene == r.parse_object(r.base.read(request / "RESPONSE.json")["text"])
            and r.base.read(request / "ACCEPTANCE.json") == accepted, "parent_A3_not_raw_accepted_output")
        review = r.base.read(review_dir / "REVIEW.json")
        r.base.require(review["status"] == "PASS", "parent_A3_review_failed")
        if scene_id in ("S01", "S02"):
            for suffix in (".json", ".ATTEMPTS.json"):
                r.base.require((folder / (scene_id + suffix)).read_bytes() == (old_folder / (scene_id + suffix)).read_bytes(),
                    "parent_A3_inherited_scene_changed")
        else:
            r.base.require(request.is_relative_to(parent / "requests") and review_dir.is_relative_to(parent / "requests"),
                "parent_A3_request_outside_parent")
            context = decode(r.base.read(review_dir / "REQUEST.json"), "\nREVIEW INPUT DATA:\n")
            r.base.require(review.get("schema_valid") is True and context["branch_history"] == history
                and context["new_scene"] == r.realization_review.review_scene(scene), "parent_A3_review_context_changed")
        history = r.base.append_scene(history, ROOT, r.ROUTES[0], scene)
    cumulative = folder / "cumulative_dialogue.json"
    r.base.require(history == r.base.read(cumulative)["turns"] and r.base.digest(history) == result["output_history_sha256"]
        and len(history) == 132 and len(original["turns"]) == 94
        and all(t["turn_ref"].startswith("R02:A:") for t in history[94:]), "parent_A3_exact_chain_changed")
    paths = {p for p in parent.rglob("*") if p.is_file()}
    paths.update(Path(row["path"]) for row in manifest["files"])
    return {"parent": parent, "root_id": ROOT, "paths": sorted(paths), "result": result,
            "cumulative_path": cumulative, "original": original}


def checked_source(source_run):
    source_run = Path(source_run).resolve()
    manifest = r.base.read(source_run / "MANIFEST.json")
    r.base.require(manifest["version"] == "h4_sequential_ab_v34", "wrong_source_run_version")
    r.base.verify_bindings(manifest["files"])
    r.base.verify_bindings(r.base.read(source_run / "MERGE_MANIFEST.json")["files"])
    audit = r.base.read(source_run / "FINAL_AUDIT.json")
    r.base.require(audit["status"] == "PASS" and not audit["failures"] and Path(audit["run"]).resolve() == source_run,
        "source_run_not_audited")
    sources = r.base.read(source_run / "dataset_sources.json")
    subset = r.base.dataset_cards.select_source_subset(sources, [SOURCE_ID])
    r.base.require(len(subset["items"]) == 1, "fixed_source_missing")
    source = subset["items"][0]
    cards_result = r.base.read(source_run / "dataset_cards_result.json")
    cards = [c for c in cards_result["accepted_cards"] if c["source_id"] == SOURCE_ID]
    r.base.require(len(cards) == 1 and cards[0]["card_id"] == CARD_ID, "fixed_card_missing")
    card = cards[0]
    reservations = r.base.read(source_run / "SOURCE_RESERVATIONS.json")
    occupied = [row for row in reservations["reservations"] if row["source_id"] == SOURCE_ID]
    r.base.require(len(occupied) == 1 and occupied[0]["scene_id"] == ORIGINAL_SCENE
        and occupied[0]["root_id"] == ROOT and Path(occupied[0]["run"]).resolve() == source_run
        and occupied[0]["source_item_sha256"] == source["source_item_sha256"], "original_source_reservation_changed")
    lineage = next(row for row in cards_result["source_attempts"] if row["source_id"] == SOURCE_ID)
    directory = Path(lineage["attempts"][-1]["request"])
    r.base.require(directory.is_relative_to(source_run / "requests"), "source_extraction_outside_run")
    raw = r.base.read(directory / "RESPONSE.json")
    validation = r.card_selection.select_cards(raw["text"], subset)
    r.base.require(validation == r.base.read(directory / "VALIDATION.json") and validation["accepted_cards"][0] == card,
        "fixed_card_raw_extraction_changed")
    proof = r.base.read(directory / "PROOF.json")
    r.base.require(proof["source_id"] == SOURCE_ID and tuple(proof["source_event_identity"]) == r.source_inventory.event_key(source),
        "fixed_source_identity_changed")
    paths = {source_run / name for name in ("MANIFEST.json", "FINAL_AUDIT.json", "MERGE_MANIFEST.json",
        "dataset_sources.json", "dataset_cards_result.json", "SOURCE_RESERVATIONS.json")}
    paths.update(p for p in directory.rglob("*") if p.is_file())
    for row in sources["files"]:
        if row["local_sha256"] != source["source_file_sha256"]:
            continue
        for key, hash_key in (("local_path", "local_sha256"), ("source_receipt_path", "source_receipt_sha256")):
            path = Path(row[key]); paths.add(path)
            r.base.require(r.base.digest(path.read_bytes()) == row[hash_key], "fixed_source_file_changed")
    return {"card": card, "source": source, "reservation": occupied[0], "paths": sorted(paths), "extraction": str(directory)}


def decode(payload, marker):
    return json.JSONDecoder().raw_decode(payload["messages"][0]["content"].split(marker, 1)[1])[0]


def prepare(run, parent_a3=PARENT, source_run=SOURCE_RUN):
    run = Path(run).resolve()
    r.base.require(run.parent == r.HERE / "experiments" and not run.exists(), "fresh_experiment_directory_required")
    parent, source = checked_a3(parent_a3), checked_source(source_run)
    inventory = r.source_inventory.collect_prior()
    r.base.require(any(r.source_inventory.event_key(row) == r.source_inventory.event_key(source["source"])
        and row["scene_id"] == ORIGINAL_SCENE and row["run"] == str(Path(source_run).resolve())
        for row in inventory["accepted_uses"]), "fixed_source_not_previously_consumed")
    run.mkdir(parents=True)
    r.base.write(run / "sources" / f"{ROOT}.json", parent["original"])
    r.import_parent_a(run, ROOT, parent)
    r.base.feedback.compile_feedback(run / "feedback", [ROOT])
    r.base.write(run / "PURPOSE_FEEDBACK_PROJECTION.json", r.purpose_contract.project_feedback(run / "feedback", ROOT, r.base.feedback))
    r.base.write(run / "FROZEN_CARD.json", source["card"])
    r.base.write(run / "FROZEN_SOURCE.json", source["source"])
    r.base.write(run / "SOURCE_REUSE_RECEIPT.json", {"source_run": str(Path(source_run).resolve()),
        "source_id": SOURCE_ID, "card_id": CARD_ID, "original_reserved_scene_id": ORIGINAL_SCENE,
        "experimental_scene_id": SCENE, "original_reservation": source["reservation"],
        "original_extraction": source["extraction"], "card_sha256": r.base.digest(source["card"]),
        "source_sha256": r.base.digest(source["source"]), "source_files": r.base.bindings(source["paths"]),
        "new_source_consumption": 0, "card_reextracted": False, "training_pool_eligible": False})
    r.base.write(run / "SOURCE_INVENTORY_BEFORE.json", inventory)
    provider, transport, tokenizer = r.transport_inputs()
    r.base.write(run / "provider_profile.json", provider)
    code = sorted(set(list(r.HERE.glob("*.py")) + list(r.LEGACY.glob("*.py")) + transport))
    snapshots = []
    for number, path in enumerate(code):
        target = run / "execution_code" / f"{number:03d}_{path.name}"
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        snapshots.append({"original_path": str(path), "snapshot_path": str(target), "sha256": r.base.digest(path.read_bytes())})
    r.base.write(run / "EXECUTION_CODE_SNAPSHOT.json", {"files": snapshots})
    paths = set(code + tokenizer + parent["paths"] + source["paths"] + [r.HERE / "POLICY.md",
        Path(r.source_inventory.legacy.__file__).resolve()])
    paths.update(Path(row["path"]) for row in inventory["prior_evidence"])
    paths.update(p for p in run.rglob("*") if p.is_file())
    manifest = {"version": VERSION, "runtime_version": r.VERSION, "model": r.MODEL, "root_id": ROOT,
        "route": ROUTE, "scene_id": SCENE, "source_id": SOURCE_ID, "card_id": CARD_ID,
        "original_reserved_scene_id": ORIGINAL_SCENE, "inherited_A_scenes": 3,
        "inherited_history_sha256": parent["result"]["output_history_sha256"],
        "B_card_grounding_required": True, "B_scanner_contract": r.SCANNER_CONTRACT,
        "logical_call_max": LOGICAL_MAX, "http_max": HTTP_MAX,
        "stage_attempts": {"purpose": 1, "state": 2, "planner": 2, "scanner_per_plan": 2,
                           "realizer": 2, "reviewer_per_realizer": 2},
        "minimum_http_gap_after_response_seconds": r.MIN_HTTP_GAP_SECONDS,
        "automatic_transport_retries": 1, "retryable_transport_statuses": [503, 504],
        "source_reuse_policy": "fixed_consumed_source_experimental_replay", "training_pool_eligible": False,
        "new_source_consumption": 0, "new_A_calls": 0, "card_extraction_calls": 0,
        "question_calls": 0, "solver_calls": 0, "training_calls": 0,
        "transport_dependency_paths": [str(p) for p in transport], "capacity_tokenizer_paths": [str(p) for p in tokenizer],
        "prepared_at_utc": r.now(), "files": r.base.bindings(sorted(paths))}
    r.base.write(run / "MANIFEST.json", manifest)
    verify(run)
    return {"status": "PREPARED_EXPERIMENT", "input_turns": 132, "inherited_A_scenes": 3,
        "source_id": SOURCE_ID, "logical_call_max": LOGICAL_MAX, "http_max": HTTP_MAX,
        "new_source_consumption": 0, "training_pool_eligible": False, "http_calls": 0}


def verify(run):
    run = Path(run).resolve()
    m = r.base.read(run / "MANIFEST.json")
    r.base.require(run.parent == r.HERE / "experiments", "experiment_outside_quarantine")
    r.base.require(m["version"] == VERSION and m["runtime_version"] == r.VERSION
        and m["model"] == r.MODEL == "Qwen/Qwen3.6-35B-A3B"
        and m["root_id"] == ROOT and m["scene_id"] == SCENE and m["source_id"] == SOURCE_ID
        and m["B_card_grounding_required"] is True and m["B_scanner_contract"] == r.SCANNER_CONTRACT,
        "experiment_scope_changed")
    r.base.require(m["logical_call_max"] == LOGICAL_MAX and m["http_max"] == HTTP_MAX
        and m["minimum_http_gap_after_response_seconds"] == r.MIN_HTTP_GAP_SECONDS
        and m["automatic_transport_retries"] == r.TRANSPORT_RETRIES == 1
        and (r.PURPOSE_ATTEMPTS, r.STATE_ATTEMPTS, r.PLAN_ATTEMPTS, r.EVENT_STATE_ATTEMPTS,
             r.SCENE_ATTEMPTS, r.REVIEW_RETRIES) == (1, 2, 2, 2, 2, 1), "experiment_budget_changed")
    r.base.require(m["training_pool_eligible"] is False and m["new_source_consumption"] == 0
        and not any((run / name).exists() for name in FORBIDDEN_OUTPUTS), "experiment_production_leak")
    r.base.verify_bindings(m["files"])
    inherited = r.verify_inherited_a(run)
    r.base.require(inherited["accepted_scenes"] == 3 and inherited["history_sha256"] == m["inherited_history_sha256"],
        "experiment_A3_changed")
    receipt = r.base.read(run / "SOURCE_REUSE_RECEIPT.json")
    r.base.verify_bindings(receipt["source_files"])
    card = r.base.read(run / "FROZEN_CARD.json")
    r.base.require(card["card_id"] == CARD_ID and card["source_id"] == SOURCE_ID
        and r.base.digest(card) == receipt["card_sha256"]
        and r.base.digest(r.base.read(run / "FROZEN_SOURCE.json")) == receipt["source_sha256"], "fixed_card_or_source_changed")
    for row in r.base.read(run / "EXECUTION_CODE_SNAPSHOT.json")["files"]:
        r.base.require(r.base.digest(Path(row["snapshot_path"]).read_bytes()) == row["sha256"], "execution_snapshot_changed")
    profile, _factory, _annotate, _capacity, _dev = r.base.transport()
    r.base.require(profile() == r.base.read(run / "provider_profile.json"), "provider_profile_changed")
    return m


class ScopedCaller:
    def __init__(self, caller, run):
        self.caller, self.run, self.calls = caller, run, 0
        self.history = r.route_base_history(run, ROOT, ROUTE)
    def call(self, label, payload, proof):
        r.base.require(self.calls < LOGICAL_MAX, "experiment_logical_budget_exhausted")
        r.base.require(proof.get("root_id") == ROOT and proof.get("route") == ROUTE
            and proof.get("scene_id") == SCENE and proof.get("role") in ROLES
            and label.startswith(f"{ROOT}-B-{SCENE}-"), "experiment_request_scope_changed")
        role = proof["role"]
        if role == "purpose_proposer":
            r.base.require(proof.get("history_or_state_supplied") is False, "purpose_scope_changed")
        else:
            marker = {"state_extractor": "\nSTATE_INPUT_JSON\n", "event_state_extractor": "\nEVENT_STATE_INPUT_JSON\n",
                      "scene_reviewer": "\nREVIEW INPUT DATA:\n"}.get(role, "\nINPUT_JSON\n")
            context = decode(payload, marker)
            r.base.require(context.get("cumulative_history", context.get("branch_history")) == self.history,
                "experiment_full_A3_history_changed")
        if role in ("planner", "event_state_extractor"):
            r.base.require(proof.get("require_card_grounding") is True, "experiment_card_grounding_disabled")
        self.calls += 1
        return self.caller.call(label, payload, {**proof, "experiment_version": VERSION,
            "training_pool_eligible": False, "source_reuse_receipt_sha256": r.base.digest(r.base.read(self.run / "SOURCE_REUSE_RECEIPT.json"))})


def flow(run, caller):
    return r.run_root(ScopedCaller(caller, run), run, ROOT, [r.base.read(run / "FROZEN_CARD.json")],
        target_scenes=1, routes=(ROUTE,), output_scope="experiment", require_card_grounding=True)


def live(run):
    manifest = verify(run)
    r.base.require(not (run / "LIVE_STARTED.json").exists(), "experiment_cannot_restart")
    r.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with r.LOCK_PATH.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = r.base.read(run / "SOURCE_INVENTORY_BEFORE.json")
        r.base.require(r.source_inventory.collect_prior() == before, "inventory_changed_since_prepare")
        r.base.write(run / "LIVE_STARTED.json", {"started_at_utc": r.now(), "manifest_sha256": r.base.digest(manifest),
            "logical_call_max": LOGICAL_MAX, "http_max": HTTP_MAX, "training_pool_eligible": False})
        caller = r.Caller(run)
        results, error = None, None
        try:
            results = flow(run, caller)
        except Exception as exc:
            error = {"type": type(exc).__name__, "code": str(exc)}
        after = r.source_inventory.collect_prior()
        r.base.write(run / "SOURCE_INVENTORY_AFTER.json", after)
        unchanged = before == after
        accepted = sum(row["accepted_scenes"] for row in (results or []))
        report = {"version": VERSION, "status": "EXPERIMENT_INTERRUPTED" if error else
                  "EXPERIMENT_B_ACCEPTED" if accepted else "EXPERIMENT_B_SHORTFALL",
            "routes": results, "error": error, "logical_calls": caller.request_counter, "http_attempts": caller.http_counter,
            "experimental_accepted_B_scenes": accepted, "production_added_B_scenes": 0,
            "source_inventory_unchanged": unchanged, "new_source_consumption": 0,
            "new_A_calls": 0, "card_extraction_calls": 0, "question_calls": 0, "solver_calls": 0, "training_calls": 0,
            "training_pool_eligible": False, "semantic_certified": False, "completed_at_utc": r.now()}
        r.base.write(run / "EXPERIMENT_REPORT.json", report)
        verify(run)
        r.base.write(run / "EXPERIMENT_OUTPUT_MANIFEST.json", {"version": VERSION, "training_pool_eligible": False,
            "files": r.base.bindings(sorted(p for p in run.rglob("*") if p.is_file()))})
        r.base.require(unchanged, "experiment_changed_source_inventory")
        return report


def audit(run):
    failures, captured, replay, replay_error = [], {}, None, None
    def check(value, code):
        if not value: failures.append(code)
    try:
        manifest = verify(run)
    except Exception as error:
        return {"status": "FAIL", "run": str(run), "failures": ["immutable_bindings:" + str(error)], "semantic_certified": False}
    directories = sorted(p for p in (run / "requests").glob("*") if p.is_dir())
    started = (run / "LIVE_STARTED.json").exists()
    class ReplayCaller:
        request_counter = 0
        def call(self, label, payload, proof):
            self.request_counter += 1
            directory = run / "requests" / f"{self.request_counter:03d}-{label}"
            r.base.require(directory in directories, "replay_request_missing:" + directory.name)
            r.base.require(r.base.read(directory / "REQUEST.json") == payload, "request_replay_mismatch:" + directory.name)
            if (directory / "PROOF.json").exists():
                actual = r.base.read(directory / "PROOF.json")
                r.base.require(all(actual.get(key) == value for key, value in proof.items()), "proof_replay_mismatch:" + directory.name)
            if (directory / "LOCAL_ERROR.json").exists():
                raise r.LocalRequestError(r.base.read(directory / "LOCAL_ERROR.json")["error_code"])
            r.base.require((directory / "PROOF.json").is_file(), "replay_response_without_proof")
            if not (directory / "RESPONSE.json").exists():
                r.base.require((directory / "ERROR.json").exists(), "replay_missing_response_or_error")
                raise RuntimeError("provider_request_failed_after_retry_policy")
            return r.base.read(directory / "RESPONSE.json")["text"], directory
    caller = ReplayCaller()
    if started:
        try:
            output = r.base.read(run / "EXPERIMENT_OUTPUT_MANIFEST.json")
            r.base.verify_bindings(output["files"])
            check(r.base.read(run / "LIVE_STARTED.json")["manifest_sha256"] == r.base.digest(manifest), "live_manifest_changed")
            with patch.object(r.base, "write", side_effect=lambda path, value: captured.__setitem__(Path(path), deepcopy(value))):
                try: replay = flow(run, caller)
                except Exception as exc: replay_error = {"type": type(exc).__name__, "code": str(exc)}
            check(caller.request_counter == len(directories), "unexpected_request_directory")
            for path, value in captured.items():
                check(path.is_file() and r.base.read(path) == value, "artifact_replay_mismatch:" + str(path.relative_to(run)))
            report = r.base.read(run / "EXPERIMENT_REPORT.json")
            accepted = sum(row["accepted_scenes"] for row in replay or [])
            check(report["routes"] == replay and report["error"] == replay_error, "report_replay_mismatch")
            check(report["experimental_accepted_B_scenes"] == accepted and report["production_added_B_scenes"] == 0,
                "report_acceptance_changed")
            expected_status = "EXPERIMENT_INTERRUPTED" if replay_error else "EXPERIMENT_B_ACCEPTED" if accepted else "EXPERIMENT_B_SHORTFALL"
            check(report["status"] == expected_status, "report_status_changed")
            check(report["logical_calls"] == len(directories), "logical_count_changed")
            for key in ("new_source_consumption", "new_A_calls", "card_extraction_calls", "question_calls", "solver_calls", "training_calls"):
                check(report[key] == 0, "forbidden_stage:" + key)
            check(r.base.read(run / "SOURCE_INVENTORY_BEFORE.json") == r.base.read(run / "SOURCE_INVENTORY_AFTER.json")
                and report["source_inventory_unchanged"] is True, "source_inventory_changed")
            check(report["training_pool_eligible"] is False and output["training_pool_eligible"] is False, "training_pool_leak")
        except Exception as error:
            failures.append("replay:" + str(error))
    else:
        check(not directories and not (run / "EXPERIMENT_REPORT.json").exists(), "requests_without_live_start")
    attempts, roles, models, statuses = [], Counter(), Counter(), Counter()
    for directory in directories:
        try:
            payload = r.base.read(directory / "REQUEST.json")
            check(payload["model"] == r.MODEL, "request_model_changed")
            proof_path = directory / "PROOF.json"
            if proof_path.exists():
                proof = r.base.read(proof_path); roles[proof.get("role")] += 1
                check(proof.get("capacity", {}).get("status") == "PASS"
                    and proof.get("payload_sha256") == r.base.digest(payload), "capacity_or_payload_proof_changed")
            http = sorted(directory.glob("HTTP_ATTEMPT*.json"))
            check(len(http) <= 2, "transport_retry_budget_exceeded")
            reservation = r.base.read(directory / "RESERVATION.json")
            check(reservation["http_max"] == HTTP_MAX and reservation["max_attempts"] == 2
                and reservation["retryable_statuses"] == [503, 504], "request_reservation_changed")
            for index, path in enumerate(http, 1):
                row = r.base.read(path)
                throttle = r.base.read(directory / f"THROTTLE_{index:02d}.json")
                result = r.base.read(directory / f"HTTP_RESULT_{index:02d}.json")
                check(row["request_attempt"] == index and row["payload_sha256"] == r.base.digest(payload)
                    and row["actual_payload_equals_frozen"] is True and row["requested_model"] == r.MODEL,
                    "http_payload_changed")
                check(throttle["minimum_gap_after_previous_response_seconds"] == r.MIN_HTTP_GAP_SECONDS, "throttle_policy_changed")
                if index > 1:
                    previous = r.base.read(directory / f"HTTP_RESULT_{index-1:02d}.json")
                    check(row["same_payload_retry"] is True and previous["outcome"] == "error"
                        and previous["http_status"] in (503, 504), "retry_policy_changed")
                attempts.append((row["http_attempt_number"], throttle["http_started_at_utc"], result.get("http_completed_at_utc")))
                statuses[str(result["http_status"])] += 1
                raw_path = directory / f"HTTP_RESPONSE_{index:02d}.json"
                if result["outcome"] == "response":
                    raw = r.base.read(raw_path); metadata = raw["metadata"]
                    models[metadata.get("returned_model")] += 1
                    check(metadata.get("returned_model") == r.MODEL and metadata.get("finish_reason") == "stop"
                        and result["http_status"] in (None, 200), "http_success_model_or_status_changed")
            check(len(list(directory.glob("HTTP_RESULT_*.json"))) == len(http), "http_result_count_changed")
            response_path = directory / "RESPONSE.json"
            if response_path.exists():
                check(bool(http), "response_without_http_attempt")
                response = r.base.read(response_path)
                check(r.base.read(directory / f"HTTP_RESPONSE_{len(http):02d}.json") ==
                      {"text": response["text"], "metadata": response["metadata"]}, "raw_http_response_changed")
                check(r.base.read(directory / f"HTTP_RESULT_{len(http):02d}.json")["outcome"] == "response", "response_after_error")
            else:
                check((directory / "LOCAL_ERROR.json").exists() or (directory / "ERROR.json").exists(), "request_missing_outcome")
        except Exception as error:
            failures.append("http_capture:" + directory.name + ":" + str(error))
    attempts.sort()
    check([row[0] for row in attempts] == list(range(1, len(attempts) + 1)), "http_numbering_changed")
    for index, (_number, began, ended) in enumerate(attempts):
        try:
            check(datetime.fromisoformat(ended) >= datetime.fromisoformat(began), "http_completion_before_start")
            if index:
                gap = (datetime.fromisoformat(began) - datetime.fromisoformat(attempts[index-1][2])).total_seconds()
                check(gap >= r.MIN_HTTP_GAP_SECONDS - .1, "http_gap_under_minimum")
        except (TypeError, ValueError):
            failures.append("http_completion_timestamp_missing")
    check(len(directories) <= LOGICAL_MAX and len(attempts) <= HTTP_MAX, "experiment_budget_exceeded")
    if started and (run / "EXPERIMENT_REPORT.json").exists():
        check(r.base.read(run / "EXPERIMENT_REPORT.json")["http_attempts"] == len(attempts), "http_count_changed")
    result = {"status": "PASS" if not failures else "FAIL", "version": VERSION, "run": str(run), "failures": failures,
        "manifest_sha256": r.base.digest(manifest), "generation_status": r.base.read(run / "EXPERIMENT_REPORT.json")["status"]
            if (run / "EXPERIMENT_REPORT.json").exists() else "PREPARED_NO_LIVE_CALLS",
        "logical_calls": len(directories), "http_attempts": len(attempts), "roles": dict(roles),
        "http_statuses": dict(statuses), "returned_models": dict(models), "artifact_replays": len(captured),
        "new_source_consumption": 0, "production_added_B_scenes": 0, "training_pool_eligible": False,
        "semantic_certified": False, "audited_at_utc": r.now()}
    stem = "FINAL_AUDIT" if started else "PREPARE_AUDIT"
    target = run / (stem + ".json")
    number = 2
    while target.exists():
        target = run / f"{stem}_{number:03d}.json"; number += 1
    r.base.write(target, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "live", "audit"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent-a3", type=Path, default=PARENT)
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    args = parser.parse_args()
    value = prepare(args.run.resolve(), args.parent_a3, args.source_run) if args.action == "prepare" else globals()[args.action](args.run.resolve())
    if args.action == "verify": value = {"status": "VERIFIED", "version": value["version"]}
    print(json.dumps(value, ensure_ascii=False, indent=2))
    if value.get("status") == "FAIL": raise SystemExit(1)
