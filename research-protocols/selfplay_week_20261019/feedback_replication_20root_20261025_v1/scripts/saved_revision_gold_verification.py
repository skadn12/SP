#!/usr/bin/env python3
"""Use an actual saved final-revision blind Gold for its first verifier only.

This append-only engineering continuation never calls a planner, author, Gold
reconstructor, or revision author.  Its sole possible new semantic request is
the still-unused final-revision content verifier, through codex_calls.py.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import question_driver as q
import question_candidate_shape_continuation as shape
import question_plan_schema_continuation_v3 as v3
import review_schema_adapter as rsa

KIND = "saved_revision_gold_first_verifier"
ROOT = BASE / "question_saved_revision_verification"
EXPECTED_DRIVER = "33ab6b7350bbaa1c9003d66f9c15a7c0c87fd855cf373ae3810664ab4701886d"
EXPECTED_ADAPTER = "7fb0938e8bf007ee62bfdbe6938fd26baff28568c35b7eec92eafa59124f44ec"
EXPECTED_SHAPE = "70b44b835a314fc9a89c2fb88533f7e03dbb0c6d2ca047b22bd58b9613ad3796"
ORIGINAL_NORMALIZE_BLIND = q.normalize_blind
ORIGINAL_NORMALIZE_CANDIDATES = q.normalize_candidates
BACKEND = {"model_override": None, "model": "gpt-6.1-sol", "model_provider": "hua", "effort": "ultra"}


def local(path: str | Path) -> Path:
    result = Path(path).resolve()
    result.relative_to(BASE)
    return result


def work_dir(root: str, arm: str) -> Path:
    return ROOT / q.root_number(root) / arm


def call_id(root: str, arm: str) -> str:
    return q.call_id("verify", arm, root, "saved_revision_gold_01_revision_01")


def verify_code() -> None:
    for path, digest in ((q.__file__, EXPECTED_DRIVER), (rsa.__file__, EXPECTED_ADAPTER),
                         (shape.__file__, EXPECTED_SHAPE)):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Required frozen engineering code drift: {path}")
    ready = q.read(BASE / "REVIEW_SCHEMA_ADAPTER_READY.json")
    # READY has its own schema; independently find and verify its exact script binding.
    def bindings(obj):
        if isinstance(obj, dict):
            if set(("path", "sha256", "bytes")) <= set(obj):
                yield obj
            for value in obj.values(): yield from bindings(value)
        elif isinstance(obj, list):
            for value in obj: yield from bindings(value)
    matched = [b for b in bindings(ready) if b.get("path") == str(Path(rsa.__file__).resolve())]
    if not matched: raise RuntimeError("RAM adapter READY does not bind the actual adapter")
    for value in matched: q.verify_binding(value)


def verify_receipt(path: Path) -> tuple[dict[str, Any], str]:
    receipt = q.read(path)
    for name in ("request", "raw_response", "parsed_response", "terminal_binding", "events", "stderr"):
        q.verify_binding(receipt[name])
    if receipt.get("event_classification_binding"):
        q.verify_binding(receipt["event_classification_binding"])
    directory = Path(receipt["parsed_response"]["path"]).parent
    directory.relative_to(BASE / "cli_calls")
    if q.read(directory / "RESULT.json") != receipt:
        raise RuntimeError("Saved call receipt is not the actual immutable RESULT")
    if receipt.get("response_received") is not True:
        raise RuntimeError("Saved semantic response was not actually received")
    backend = receipt.get("backend", {})
    if not backend.get("thread_id") or any(backend.get(k) != v for k, v in BACKEND.items()):
        raise RuntimeError("Saved actual default backend metadata differs from frozen research backend")
    terminal = q.read(receipt["terminal_binding"]["path"])
    classification = q.classify_cli_events(Path(receipt["events"]["path"]), terminal)
    if classification.get("status") != "PASS":
        raise RuntimeError("Saved events fail the frozen no-tools transport adapter")
    return receipt, directory.name


def actual_requests(root: str, arm: str) -> dict[str, list[dict[str, Any]]]:
    return shape.stage_requests(root, arm)


def verify_budget(requests: dict[str, Any], root: str, arm: str,
                  saved_gold_call: str | None = None) -> None:
    for stage, count in (("plan", 1), ("author", 1), ("revision", 1), ("gold", 2)):
        ids = {row["call_id"] for row in requests[stage]}
        if len(ids) != count:
            raise RuntimeError(f"Expected exactly {count} consumed {stage} call IDs, got {sorted(ids)}")
    for stage in ("plan", "author", "revision"):
        if any(set(row.get("slot_ids") or []) != set(q.SLOTS) for row in requests[stage]):
            raise RuntimeError(f"Saved {stage} assignment must cover the exact eight fixed slots")
    if saved_gold_call and saved_gold_call not in {row["call_id"] for row in requests["gold"]}:
        raise RuntimeError("Saved final-revision Gold is not one of the two actual consumed Gold calls")
    other = {row["call_id"] for row in requests["verify"]} - {call_id(root, arm)}
    if other:
        raise RuntimeError(f"A verifier was already requested for this root/arm: {sorted(other)}")


def inspect_inputs(root: str, arm: str, source: dict[str, Any], input_work: Path,
                   selection: Path) -> dict[str, Any]:
    verify_code()
    root = q.root_id(root)
    configured = {q.root_id(value) for value in q.read(BASE / "CONFIG.json")["roots"]}
    if root not in configured or arm not in q.ARMS:
        raise RuntimeError("Recovery assignment is outside the frozen roster/arms")
    input_work = local(input_work)
    required = ("OUTCOME.json", "CONTINUATION_FINALIZER_READY.json", "TASK_PLANS.json", "TASK_PLANS_FREEZE.json",
        "CANDIDATES_DRAFT_CALL.json", "CANDIDATES_REVISION_01_RESPONSE.json", "CANDIDATES_REVISION_01_CALL.json",
        "CANDIDATES_REVISION_01.json", "CANDIDATES_FINAL.json", "CONTENT_REVIEW_FINAL.json", "SLOT_FAILURES.json",
        "BLIND_GOLD_revision_01.json", "BLIND_GOLD_CALL_revision_01.json", "BLIND_GOLD_VALIDATION_revision_01.json")
    if any(not (input_work / name).is_file() for name in required):
        raise RuntimeError("Final-revision saved inputs lack a terminal frozen artifact")
    outcome = q.read(input_work / "OUTCOME.json")
    if ((outcome.get("root_id"), outcome.get("arm"), outcome.get("source_binding")) != (root, arm, source["binding"])
            or outcome.get("status") not in {"COMPLETED_WITH_HOLES", "COMPLETED"}
            or outcome.get("max_content_revisions") != 1):
        raise RuntimeError("Prior final outcome is not the bounded terminal revision for this root/arm/source")
    ready = q.read(input_work / "CONTINUATION_FINALIZER_READY.json")
    if (ready.get("root_id"), ready.get("arm"), ready.get("final_work_dir")) != (root, arm, str(input_work)):
        raise RuntimeError("Prior finalizer READY identity drift")
    for field in ("final_outcome_binding", "driver_outcome_binding", "final_candidates_binding", "final_review_binding", "slot_failures_binding"):
        q.verify_binding(ready[field])
    if ready["final_candidates_binding"] != q.binding(input_work / "CANDIDATES_FINAL.json"):
        raise RuntimeError("Prior READY does not bind the actual final candidates")
    reviews = q.read(input_work / "CONTENT_REVIEW_FINAL.json").get("reviews", [])
    if len(reviews) != len(q.SLOTS) or any(r.get("decision") == "PASS" for r in reviews):
        raise RuntimeError("This whole-revision continuation cannot alter or rereview a previously PASS slot")
    initial_receipt, initial_id = verify_receipt(input_work / "CANDIDATES_DRAFT_CALL.json")
    revision_receipt, revision_id = verify_receipt(input_work / "CANDIDATES_REVISION_01_CALL.json")
    gold_receipt, gold_id = verify_receipt(input_work / "BLIND_GOLD_CALL_revision_01.json")
    requests = actual_requests(root, arm)
    verify_budget(requests, root, arm, gold_id)
    if {row["call_id"] for row in requests["author"]} != {initial_id}:
        raise RuntimeError("Initial-author receipt is not the sole consumed author")
    if {row["call_id"] for row in requests["revision"]} != {revision_id}:
        raise RuntimeError("Revision-author receipt is not the sole consumed revision")
    # Fresh continuation may reuse only its own exact existing verifier receipt.
    if requests["verify"] and not (work_dir(root, arm) / "RECOVERY_INPUT_FREEZE.json").exists():
        raise RuntimeError("Verifier REQUEST predates the append-only recovery input freeze")
    if any(row["call_id"] == call_id(root, arm) for row in requests["gold"]):
        raise RuntimeError("Recovery verifier ID must never be used as a Gold request")
    plan_path = input_work / "TASK_PLANS.json"
    plan_freeze_path = input_work / "TASK_PLANS_FREEZE.json"
    plan = q.read(plan_path)
    frozen = q.read(plan_freeze_path)
    q.verify_binding(frozen["plans_binding"])
    if frozen["plans_binding"] != q.binding(plan_path) or frozen.get("errors"):
        raise RuntimeError("The actual preauthoring plan is not valid and frozen")
    if plan.get("root_id") != root or plan.get("arm") != arm or {p.get("slot_id") for p in plan.get("plans", [])} != set(q.SLOTS):
        raise RuntimeError("Saved plan identity or exact eight-slot allocation drift")
    revision_raw = q.read(input_work / "CANDIDATES_REVISION_01_RESPONSE.json")
    if revision_raw != q.read(revision_receipt["parsed_response"]["path"]):
        raise RuntimeError("Saved revision raw differs from the real parsed author response")
    normalized, candidate_errors, _ = ORIGINAL_NORMALIZE_CANDIDATES(revision_raw, arm, root,
        initial_receipt["backend"]["thread_id"], source, plan, q.binding(selection),
        revision_receipt["backend"]["thread_id"])
    candidates = q.read(input_work / "CANDIDATES_FINAL.json")
    if candidate_errors or candidates != q.read(input_work / "CANDIDATES_REVISION_01.json") or candidates != normalized:
        raise RuntimeError("Final candidates must equal the actual normalized final revision exactly; no initial fallback")
    if len(candidates) != len(q.SLOTS) or {c.get("slot_id") for c in candidates} != set(q.SLOTS):
        raise RuntimeError("The bounded whole-revision recovery requires the exact eight final revision candidates")
    if any(not q.basic_solver_shape(c, source) or c.get("revision_author_thread_id") != revision_receipt["backend"]["thread_id"]
           or c.get("content_revision_count") != 1 for c in candidates):
        raise RuntimeError("Saved final revision candidate shape or actual revision lineage drift")
    raw = q.read(input_work / "BLIND_GOLD_revision_01.json")
    if raw != q.read(gold_receipt["parsed_response"]["path"]):
        raise RuntimeError("Saved revision blind Gold differs from its actual immutable parsed response")
    payload = shape.request_payload(Path(gold_receipt["request"]["path"]).read_text())
    expected_public = [q.anonymize_candidate(c)[0] for c in candidates]
    if (payload.get("root_id") != root or payload.get("complete_source_and_warnings") != q.source_view(source)
            or payload.get("anonymous_questions") != expected_public):
        raise RuntimeError("Saved revision blind request does not exactly bind the full current source and final public candidates")
    prior_map, prior_errors = ORIGINAL_NORMALIZE_BLIND(copy.deepcopy(raw), candidates, source["source"])
    expected_errors = ["blind_r03_invalid:" + c["candidate_id"] for c in candidates]
    if prior_errors != expected_errors or q.read(input_work / "BLIND_GOLD_VALIDATION_revision_01.json").get("errors") != expected_errors:
        raise RuntimeError("Saved final Gold defect is not bounded to the exact R03 serialization alias")
    adapted, changes = rsa.normalize_blind_aliases(raw)
    blind_map, errors = ORIGINAL_NORMALIZE_BLIND(copy.deepcopy(adapted), candidates, source["source"])
    if len(changes) != len(candidates) or errors or set(blind_map) != {c["candidate_id"] for c in candidates}:
        raise RuntimeError("Exact reversible alias adaptation did not satisfy every frozen blind-Gold check")
    if {cid: row["independent_gold"] for cid, row in prior_map.items()} != {cid: row["independent_gold"] for cid, row in blind_map.items()}:
        raise RuntimeError("Mechanical alias adaptation changed a saved independent Gold")
    threads = [r["backend"]["thread_id"] for r in (initial_receipt, revision_receipt, gold_receipt)]
    if len(set(threads)) != len(threads):
        raise RuntimeError("Saved author, revision author, and blind reviewer do not have independent fresh threads")
    paths = [input_work / name for name in required]
    paths += [selection, q.SOURCE_FREEZE_PATH, BASE / "REVIEW_SCHEMA_ADAPTER_READY.json"]
    return {"schema_version": "fgr20_saved_revision_gold_first_verifier_input_v1", "root_id": root, "arm": arm,
        "source_binding": source["binding"], "input_work_dir": str(input_work), "input_bindings": [q.binding(p) for p in paths],
        "actual_plan_path": str(plan_path), "actual_plan_freeze_path": str(plan_freeze_path),
        "actual_stage_requests_before_recovery": {k: v for k, v in requests.items() if k != "verify"},
        "existing_verifier_call_ids": sorted({r["call_id"] for r in requests["verify"]}),
        "allowed_new_call_id": call_id(root, arm), "saved_gold_call_id": gold_id,
        "saved_gold_receipt_binding": q.binding(input_work / "BLIND_GOLD_CALL_revision_01.json"),
        "actual_initial_author_thread_id": threads[0], "actual_revision_author_thread_id": threads[1],
        "actual_saved_gold_thread_id": threads[2], "prior_blind_validation_errors": prior_errors,
        "alias_transformations": changes, "adapted_blind_validation_errors": errors,
        "final_candidate_count": len(candidates), "independent_gold_values_unchanged": True,
        "final_revision_retained_without_initial_fallback": True, "no_new_gold_call": True,
        "no_new_plan_or_author_call": True, "no_second_revision": True, "semantic_admission_asserted": False}


def prepare(root: str, arm: str, source: dict[str, Any], input_work: Path, selection: Path) -> dict[str, Any]:
    out = work_dir(root, arm)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "RECOVERY_INPUT_FREEZE.json"
    if path.exists():
        saved = q.read(path)
        if (saved.get("root_id"), saved.get("arm"), saved.get("source_binding"), saved.get("input_work_dir")) != (
                root, arm, source["binding"], str(input_work)):
            raise RuntimeError("Existing saved-Gold recovery input identity drift")
        for value in saved["input_bindings"]: q.verify_binding(value)
        verify_budget(actual_requests(root, arm), root, arm, saved["saved_gold_call_id"])
        return saved
    result = inspect_inputs(root, arm, source, input_work, selection)
    q.publish(path, result)
    return result


def active_driver_processes() -> list[dict[str, Any]]:
    """Reject overlap across driver families rather than sharing unsafe locks."""
    names = {"production_driver.py", "question_driver.py", "question_driver_continued.py", "question_driver_recovery.py",
        "question_plan_schema_continuation.py", "question_plan_schema_continuation_v2.py", "question_plan_schema_continuation_v3.py",
        "question_candidate_shape_continuation.py", "question_unfilled_slot_continuation.py", "saved_revision_gold_verification.py",
        "review_schema_adapter.py"}
    rows = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid(): continue
        try:
            argv = [x.decode(errors="replace") for x in (path / "cmdline").read_bytes().split(b"\0") if x]
            if not argv or "python" not in Path(argv[0]).name: continue
            script = next((arg for arg in argv[1:] if arg.endswith(".py")), None)
            if not script or Path(script).name not in names: continue
            actual_script = Path(script)
            if not actual_script.is_absolute(): actual_script = (path / "cwd").resolve() / actual_script
            if str(BASE) not in str(actual_script.resolve()) and str(BASE) not in " ".join(argv): continue
            if "run" not in argv and Path(script).name not in {"production_driver.py", "question_driver.py"}: continue
            if len((path / "stat").read_text().split()) > 2 and (path / "stat").read_text().split()[2] != "Z":
                rows.append({"pid": int(path.name), "argv": argv})
        except (OSError, ValueError): continue
    return rows


def assert_idle() -> None:
    live = v3.active_calls()
    parents = active_driver_processes()
    if live or parents:
        raise RuntimeError(f"Serialized driver-family guard requires registry idle and all other parents exited: calls={live}, parents={parents}")


def build_verify_rows(candidates: list[dict[str, Any]], blind_map: dict[str, Any], selection: Path):
    result = []
    for candidate in candidates:
        cid = candidate["candidate_id"]
        result.append({"question_key": blind_map[cid]["question_key"],
            "anonymous_candidate": {key: candidate.get(key) for key in ("question", "question_type", "task_operation", "options", "items",
                "gold", "item_analysis", "ordering_analysis", "r03_necessity")},
            "saved_independent_gold": blind_map[cid],
            "mechanical_review_packet": q.anonymize_mechanical_packet(q.gate.build_review_packet(candidate, selection))})
    return result


def review_batch(root: str, arm: str, candidates: list[dict[str, Any]], source: dict[str, Any],
                 selection: Path, work: Path, frozen: dict[str, Any], caller: Any) -> dict[str, Any]:
    # The RAM adapter logs the concrete review stage through this local context.
    suffix = "revision_01"
    input_work = Path(frozen["input_work_dir"])
    raw_gold = q.read(input_work / "BLIND_GOLD_revision_01.json")
    adapted_gold, changes = rsa.normalize_blind_aliases(raw_gold)
    blind_map, blind_errors = ORIGINAL_NORMALIZE_BLIND(copy.deepcopy(adapted_gold), candidates, source["source"])
    if blind_errors: raise RuntimeError("Saved Gold no longer passes the frozen parser after alias-only adaptation")
    q.publish(work / "MECHANICAL_SAVED_REVISION_GOLD_INPUT.json", adapted_gold)
    q.publish(work / "MECHANICAL_SAVED_REVISION_GOLD_NORMALIZATION.json", {
        "schema_version": "fgr20_saved_final_gold_alias_normalization_v1", "saved_raw_binding": q.binding(input_work / "BLIND_GOLD_revision_01.json"),
        "saved_call_receipt_binding": frozen["saved_gold_receipt_binding"], "adapter_binding": q.binding(rsa.__file__),
        "normalized_input_binding": q.binding(work / "MECHANICAL_SAVED_REVISION_GOLD_INPUT.json"), "changes": changes,
        "original_strict_parser_errors": frozen["prior_blind_validation_errors"], "adapted_strict_parser_errors": blind_errors,
        "exact_round_trip_verified": True, "gold_decisions_evidence_and_all_values_changed": False,
        "no_new_gold_call": True, "semantic_admission_asserted": False})
    q.publish(work / "SAVED_REVISION_GOLD_REFERENCE.json", {
        "raw_binding": q.binding(input_work / "BLIND_GOLD_revision_01.json"),
        "call_receipt_binding": frozen["saved_gold_receipt_binding"], "actual_call_id": frozen["saved_gold_call_id"],
        "actual_thread_id": frozen["actual_saved_gold_thread_id"], "copied_as_new_call": False})
    prompt = q.verify_prompt(root, q.source_view(source), build_verify_rows(candidates, blind_map, selection))
    if arm == "G": q.validate_g_prompt(prompt, "content_verifier", root)
    key = call_id(root, arm)
    response = receipt = None
    failure = None
    try:
        response, receipt = caller.call(key, prompt)
        q.publish(work / "CONTENT_VERIFIER_revision_01.json", response)
        q.publish(work / "CONTENT_VERIFIER_CALL_revision_01.json", receipt)
    except Exception as exc:
        failure = {"error": f"{type(exc).__name__}: {exc}", "call_id": key,
            "verifier_attempt_consumed_or_unavailable": True, "no_new_gold_or_revision_permitted": True}
        q.publish(work / "CONTENT_VERIFIER_FAILURE_revision_01.json", failure)
    adapted_response, response_changes = rsa.normalize_blind_aliases(response)
    if response is not None:
        q.publish(work / "MECHANICAL_VERIFIER_SCHEMA_INPUT_revision_01.json", adapted_response)
        q.publish(work / "MECHANICAL_VERIFIER_SCHEMA_NORMALIZATION_revision_01.json", {
            "raw_response_binding": q.binding(work / "CONTENT_VERIFIER_revision_01.json"),
            "normalized_input_binding": q.binding(work / "MECHANICAL_VERIFIER_SCHEMA_INPUT_revision_01.json"),
            "adapter_binding": q.binding(rsa.__file__), "changes": response_changes, "exact_round_trip_verified": True,
            "decisions_gold_evidence_and_all_values_changed": False, "semantic_admission_asserted": False})
    reviewer = receipt.get("backend", {}).get("thread_id") if receipt else None
    previous_threads = {frozen["actual_initial_author_thread_id"], frozen["actual_revision_author_thread_id"], frozen["actual_saved_gold_thread_id"]}
    thread_errors = []
    if receipt:
        if not reviewer or reviewer in previous_threads: thread_errors.append("verifier_thread_not_fresh_independent")
        if any(receipt.get("backend", {}).get(k) != v for k, v in BACKEND.items()): thread_errors.append("actual_verifier_backend_drift")
    reviews, audits, gates, failures = [], [], [], []
    for candidate in candidates:
        cid = candidate["candidate_id"]
        row = q.check_for_candidate(adapted_response, candidate)
        review = q.normalized_review(candidate, blind_map[cid], row, reviewer, source["source"])
        if thread_errors:
            review["decision"] = "HOLD"
            review["issues"] = list(review.get("issues", [])) + thread_errors
            review["effective_decision_origin"] = "engineering_hold_independence_or_backend"
        audit = q.audit_candidate(candidate, selection)
        validation = q.gate.validate_review(candidate, review, selection)
        if not validation["admitted"] and review.get("decision") == "PASS":
            review["semantic_reviewer_decision"] = "PASS"
            review["decision"] = "HOLD"
            review["issues"] = list(review.get("issues", [])) + validation.get("review_errors", [])
            review["effective_decision_origin"] = "engineering_hold_mechanical_gate"
            validation = q.gate.validate_review(candidate, review, selection)
        reviews.append(review); audits.append(audit); gates.append(validation)
        if not validation["admitted"]:
            details = [str(x) for x in review.get("issues", [])] + [str(x) for x in validation.get("review_errors", [])]
            if not audit.get("mechanical_pass"): details += [str(x) for x in audit.get("mechanical_errors", [])]
            details += [f"{name}:{check.get('rationale', '')}" for name, check in review.get("checks", {}).items()
                        if isinstance(check, dict) and check.get("decision") != "PASS"]
            failures.append(q.slot_failure(candidate["slot_id"], "final revision not admitted: " + "; ".join(details), root, arm, "saved_revision_first_verifier"))
    q.publish(work / "CONTENT_REVIEW_NORMALIZED_revision_01.json", {"reviews": reviews, "mechanical_audits": audits,
        "saved_blind_call_receipt_reference": frozen["saved_gold_receipt_binding"], "verification_call": receipt,
        "review_stage": suffix, "blind_reconstruction_repeated": False})
    return {"reviews": reviews, "mechanical_audits": audits, "gate_validations": gates, "slot_failures": failures,
            "verification_call": receipt, "failure": failure, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def publish_ready(root: str, arm: str, input_work: Path, out: Path, frozen: dict[str, Any], data: dict[str, Any]):
    candidates = q.read(input_work / "CANDIDATES_FINAL.json")
    q.publish(out / "CANDIDATES_FINAL.json", candidates)
    q.publish(out / "CONTENT_REVIEW_FINAL.json", {"reviews": data["reviews"], "mechanical_audits": data["mechanical_audits"],
        "all_reviews_ai_family_not_human": True, "review_stage": "revision_01", "initial_reviews_retained_at": str(input_work)})
    q.publish(out / "FINAL_REVIEW_GATE.json", {"results": data["gate_validations"], "semantic_admission_basis": "fresh independent AI verifier judgment plus frozen mechanical gate"})
    q.publish(out / "SLOT_FAILURES.json", data["slot_failures"])
    outcome = {"schema_version": "fgr20_question_root_arm_v1", "root_id": root, "arm": arm, "source_binding": frozen["source_binding"],
        "status": "COMPLETED_WITH_HOLES" if data["slot_failures"] else "COMPLETED", "candidate_ids": [c["candidate_id"] for c in candidates],
        "candidate_bindings": [{"candidate_id": c["candidate_id"], "candidate_sha256": q.gate.fingerprint(c)} for c in candidates],
        "slot_failures": data["slot_failures"], "calls": [data["verification_call"]] if data["verification_call"] else [],
        "max_content_revisions": 1, "review_stage": "revision_01", "continuation_kind": KIND,
        "prior_outcome_binding": q.binding(input_work / "OUTCOME.json"), "prior_slot_failures_binding": q.binding(input_work / "SLOT_FAILURES.json"),
        "final_candidates_binding": q.binding(out / "CANDIDATES_FINAL.json"), "review_binding": q.binding(out / "CONTENT_REVIEW_FINAL.json"),
        "slot_failures_binding": q.binding(out / "SLOT_FAILURES.json"), "plan_freeze_binding": q.binding(frozen["actual_plan_freeze_path"]),
        "saved_blind_gold_receipt_binding": frozen["saved_gold_receipt_binding"], "no_new_gold_call": True,
        "no_new_plan_or_author_call": True, "no_second_revision": True, "final_revision_retained_without_initial_fallback": True}
    q.publish(out / "OUTCOME.json", outcome)
    q.publish(out / "CONTINUATION_INTEGRATION_OUTCOME.json", outcome)
    ready = {"schema_version": "fgr20_question_continuation_finalizer_ready_v1", "root_id": root, "arm": arm,
        "continuation_kind": KIND, "review_stage": "revision_01", "original_work_dir": str(input_work), "initial_work_dir": str(input_work),
        "main_original_work_dir": str(BASE / "question_authoring" / arm / "roots" / q.root_number(root)), "final_work_dir": str(out),
        "actual_plan_path": frozen["actual_plan_path"], "actual_plan_freeze_path": frozen["actual_plan_freeze_path"],
        "original_outcome_binding": q.binding(input_work / "OUTCOME.json"), "supersedes_ready_binding": q.binding(input_work / "CONTINUATION_FINALIZER_READY.json"),
        "final_outcome_binding": q.binding(out / "CONTINUATION_INTEGRATION_OUTCOME.json"), "driver_outcome_binding": q.binding(out / "OUTCOME.json"),
        "final_candidates_binding": q.binding(out / "CANDIDATES_FINAL.json"), "final_review_binding": q.binding(out / "CONTENT_REVIEW_FINAL.json"),
        "slot_failures_binding": q.binding(out / "SLOT_FAILURES.json"), "input_freeze_binding": q.binding(out / "RECOVERY_INPUT_FREEZE.json"),
        "saved_gold_receipt_reference_binding": q.binding(out / "SAVED_REVISION_GOLD_REFERENCE.json"),
        "normalization_binding": q.binding(out / "MECHANICAL_SAVED_REVISION_GOLD_NORMALIZATION.json"),
        "gate_validation_binding": q.binding(out / "FINAL_REVIEW_GATE.json"), "driver_binding": q.binding(__file__), "adapter_binding": q.binding(rsa.__file__),
        "no_new_gold_call": True, "no_new_plan_or_author_call": True, "no_second_revision": True,
        "initial_candidate_and_reviews_not_overwritten": True, "final_revision_retained_without_initial_fallback": True}
    q.publish(out / "CONTINUATION_FINALIZER_READY.json", ready)
    return ready


def run(root: str, arm: str, source: dict[str, Any], input_work: Path, selection: Path, do_run: bool):
    out = work_dir(root, arm)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".continuation.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ready_path = out / "CONTINUATION_FINALIZER_READY.json"
        if ready_path.exists():
            ready = q.read(ready_path)
            if (ready.get("root_id"), ready.get("arm"), ready.get("continuation_kind"), ready.get("original_work_dir")) != (root, arm, KIND, str(input_work)):
                raise RuntimeError("Cached saved-Gold recovery READY identity drift")
            for key, value in ready.items():
                if key.endswith("_binding"): q.verify_binding(value)
            return ready
        if do_run and (BASE / "CONTENT_FREEZE.json").exists(): raise RuntimeError("Content has already frozen")
        frozen = prepare(root, arm, source, input_work, selection)
        q.publish(out / "DRIVER_ADAPTATION.json", {"schema_version": "fgr20_saved_revision_first_verifier_engineering_v1",
            "wrapper_binding": q.binding(__file__), "frozen_driver_binding": q.binding(q.__file__), "adapter_binding": q.binding(rsa.__file__),
            "adapter_ready_binding": q.binding(BASE / "REVIEW_SCHEMA_ADAPTER_READY.json"),
            "input_freeze_binding": q.binding(out / "RECOVERY_INPUT_FREEZE.json"), "only_possible_new_call_id": call_id(root, arm),
            "no_new_gold_call": True, "no_new_plan_or_author_call": True, "no_second_revision": True,
            "review_stage": "revision_01", "serialized_driver_families_required": True, "semantic_acceptance_asserted": False})
        if not do_run: return {"root_id": root, "arm": arm, "preparation_binding": q.binding(out / "RECOVERY_INPUT_FREEZE.json"), "semantic_calls_started": 0}
        if q.cc.stop_generation(): raise RuntimeError("Generation cutoff before first final-revision verifier")
        assert_idle()
        # This family is intentionally serialized; same-family processes share one
        # global lock. Parent/registry guards additionally reject other families.
        with (ROOT / ".family_call_start.lock").open("a+") as family:
            fcntl.flock(family, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert_idle()
            rsa.install(q.binding(out / "DRIVER_ADAPTATION.json"))
            data = review_batch(root, arm, q.read(input_work / "CANDIDATES_FINAL.json"), source, selection, out, frozen, q.cc)
        verify_budget(actual_requests(root, arm), root, arm, frozen["saved_gold_call_id"])
        ready = publish_ready(root, arm, input_work, out, frozen, data)
        q.cc.progress("SAVED_REVISION_FIRST_VERIFIER_COMPLETE", root_id=root, arm=arm, status=q.read(out / "OUTCOME.json")["status"])
        return ready


def self_check():
    root, arm = "cc-train-episode-179186", "G"
    selection = BASE / "SOURCE_SELECTION_FINAL.json"
    q.SOURCE_FREEZE_PATH = BASE / "SOURCE_FREEZE.json"
    _, sources = q.load_selection(selection)
    source = sources[root]
    input_work = BASE / "question_plan_continuation/179186/G"
    before = actual_requests(root, arm)
    frozen = inspect_inputs(root, arm, source, input_work, selection)
    assert frozen["final_candidate_count"] == 8 and not frozen["adapted_blind_validation_errors"]
    assert len(frozen["alias_transformations"]) == 8 and not frozen["existing_verifier_call_ids"]
    assert frozen["no_second_revision"] and frozen["final_revision_retained_without_initial_fallback"]
    bad = copy.deepcopy(before)
    bad["verify"].append({"call_id": "q20_verify_g_179186_forbidden_second_verifier"})
    try: verify_budget(bad, root, arm)
    except RuntimeError: pass
    else: raise AssertionError("Existing other verifier was not rejected")
    fixture = BASE / "runtime_cache/saved_revision_verifier_selfcheck"
    fixture.mkdir(parents=True, exist_ok=True)
    class MockCalls:
        reached = False
        prompt_sha256 = None
        def call(self, key, prompt):
            assert key == call_id(root, arm)
            payload = shape.request_payload(prompt)
            assert payload["complete_source_and_warnings"] == q.source_view(source)
            assert len(payload["review_cases"]) == 8
            assert all(c["saved_independent_gold"]["solver_results_read"] is False for c in payload["review_cases"])
            self.reached = True
            self.prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
            raise RuntimeError("MOCK_FIRST_FINAL_REVISION_VERIFIER_REACHED_NO_MODEL_CALL")
    mock = MockCalls()
    with tempfile.TemporaryDirectory(prefix="saved_gold_", dir=fixture) as directory:
        out = Path(directory)
        rsa.install()
        data = review_batch(root, arm, q.read(input_work / "CANDIDATES_FINAL.json"), source, selection, out, frozen, mock)
        assert mock.reached and data["verification_call"] is None
        assert all(r["decision"] == "HOLD" for r in data["reviews"])
        assert len(data["slot_failures"]) == 8 and all(not x["admitted"] for x in data["gate_validations"])
        assert not (out / "CONTENT_VERIFIER_CALL_revision_01.json").exists()
        validation = q.read(out / "G_VERIFIER_ISOLATION_ATTESTATION_VALIDATION_revision_01.json")
        assert validation["actual_prompt_modified"] is False
        assert validation["actual_prompt_sha256"] == mock.prompt_sha256
        assert len(validation["exact_allowed_false_attestation_paths"]) == 8
    assert actual_requests(root, arm) == before
    return {"status": "PASS", "semantic_calls_started": 0, "real_saved_final_revision_candidates_verified": 8,
        "real_saved_revision_gold_rows_alias_restored": 8, "saved_blind_parser_errors_after_alias_mapping": [],
        "full_saved_anonymous_request_exactly_bound_to_final_candidates": True, "gold_values_unchanged": True,
        "mock_reached_only_first_final_revision_verifier": True, "mock_no_response_fail_closed_hold_count": 8,
        "G_actual_prompt_unchanged_and_false_nonaccess_attestation_retained": True,
        "existing_verifier_budget_rejected": True, "plan_author_gold_revision_requests_repeated": 0,
        "content_semantic_verdict": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "prepare", "run", "self-check"))
    parser.add_argument("--root", default="179186")
    parser.add_argument("--arm", choices=q.ARMS, default="G")
    parser.add_argument("--input-work")
    parser.add_argument("--selection", default=str(BASE / "SOURCE_SELECTION_FINAL.json"))
    parser.add_argument("--source-freeze", default=str(BASE / "SOURCE_FREEZE.json"))
    args = parser.parse_args()
    if args.command == "self-check": print(json.dumps(self_check(), ensure_ascii=False)); return 0
    root = q.root_id(args.root)
    selection = local(args.selection)
    q.SOURCE_FREEZE_PATH = local(args.source_freeze)
    _, sources = q.load_selection(selection)
    if root not in sources: raise RuntimeError("Requested root has no canonical independently passed source")
    input_work = local(args.input_work or (BASE / "question_plan_continuation" / q.root_number(root) / args.arm))
    result = inspect_inputs(root, args.arm, sources[root], input_work, selection) if args.command == "inspect" else run(root, args.arm, sources[root], input_work, selection, args.command == "run")
    print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
