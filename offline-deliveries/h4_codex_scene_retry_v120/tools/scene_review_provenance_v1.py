"""Metadata-only verification of new scene read/render/raw/save chains.

This function never imports or calls a scene controller, so a controller can
invoke it before consuming SOURCE_REVIEW without recursive state inspection.
Historical imported reviews are checked by the explicit prefix-import adapter.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from public_anchor_v1 import prefix_input_paths, validate_runtime
from review_provenance_gate_v2 import Audit, binding, canonical_identity, helper_literals, result, sha, timestamp

SCENES = tuple(f"R02:{r}:S{i:02d}" for r in "AB" for i in (1, 2, 3))
CHECKS = {"time", "location", "actor_identity", "plan_vs_completion", "cross_scene_continuity", "semantic_repetition", "naturalness", "event_fidelity"}
STATUSES = {"PASS", "WARNING", "PASS_WITH_WARNING", "FAIL"}


def check_scene(batch, root, scene):
    audit = Audit()
    batch = Path(batch).resolve()
    audit.need(isinstance(root, str) and re.fullmatch(r"cc-train-episode-\d+", root) is not None, "INVALID_ROOT")
    audit.need(scene in SCENES, "INVALID_SCENE")
    if audit.errors:
        return result(audit, batch, root, "scene")
    try:
        validate_runtime(batch, __file__, {"review_provenance_gate_v2.py": helper_literals,
                                          "public_anchor_v1.py": prefix_input_paths})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        audit.need(False, "SCENE_RUNTIME:NOT_FROZEN_OR_LOADED_CODE_CHANGED", str(exc))
        return result(audit, batch, root, "scene")
    unit = batch / "units" / ("root" + root.rsplit("-", 1)[-1])
    folder = unit / "roots" / ("root" + root.rsplit("-", 1)[-1])
    target = folder / "scenes" / scene.removeprefix("R02:").replace(":", "_")
    role_path = batch / "ROLE_ASSIGNMENTS.json"
    helper, anchor_helper = batch / "tools/review_v4.py", batch / "tools/public_anchor_v1.py"
    adapter = batch / "tools/scene_controller_v120.py"
    roles = audit.read(role_path, "ROLE_ASSIGNMENTS")
    freeze = audit.read(batch / "MASTER_FREEZE.json", "MASTER_FREEZE")
    frozen = freeze.get("files", [])
    for path in (role_path, helper, anchor_helper, adapter, batch / "tools/review_provenance_gate_v2.py",
                 batch / "tools/scene_review_provenance_v1.py"):
        matches = [r for r in frozen if isinstance(r, dict) and r.get("path") == str(path.resolve())]
        if audit.need(len(matches) == 1, "SCENE_RUNTIME:UNIQUE_FROZEN_BINDING_REQUIRED", str(path)):
            audit.bound(matches[0], path, "SCENE_RUNTIME")
    assigned = roles.get("roots", [])
    matches = [r for r in assigned if isinstance(r, dict) and r.get("root_id") == root]
    audit.need(len(matches) == 1, "ROLE_ASSIGNMENTS:ROOT_NOT_UNIQUELY_ASSIGNED")
    role = matches[0] if len(matches) == 1 else {}
    author, reviewer = role.get("author"), role.get("reviewer")
    audit.need(canonical_identity(author) and canonical_identity(reviewer) and author != reviewer,
               "ROLE_ASSIGNMENTS:INVALID_OR_SELF_REVIEW")
    digest = hashlib.sha256(str(reviewer).encode()).hexdigest()[:16]
    directory = unit / "evidence/reads" / f"scene_{scene.replace(':', '_')}_{digest}"
    context_path, render_path = directory / "READ_CONTEXT.json", directory / "INPUT_RENDER.txt"
    raw_path, receipt_path = directory / "RAW_JUDGMENT.json", directory / "SAVE_RECEIPT.json"
    review_path = target / "SOURCE_REVIEW.json"
    source_path, draft_path = target / "CUMULATIVE_SOURCE.json", target / "EXPANSION_DRAFT.json"
    constraints_path = unit / "inputs/CONTINUATION_CONSTRAINTS.json"
    review, context, raw, receipt = [audit.read(p, label) for p, label in (
        (review_path, "REVIEW"), (context_path, "READ_CONTEXT"), (raw_path, "RAW_JUDGMENT"), (receipt_path, "SAVE_RECEIPT"))]
    source, draft, constraints = [audit.read(p, label) for p, label in (
        (source_path, "SOURCE"), (draft_path, "DRAFT"), (constraints_path, "CONSTRAINTS"))]
    for obj, label, values in (
        (review, "REVIEW", {"schema_version": "codex_assigned_source_review_v3", "root_id": root, "scene_ref": scene,
                            "reviewer": reviewer, "review_role": "non_author_complete_cumulative_source_and_native_event_reviewer",
                            "backend": "CURRENT_CODEX_SESSION", "external_semantic_certification": False,
                            "semantic_review_is_programmatic": False}),
        (context, "READ_CONTEXT", {"schema_version": "codex_assigned_read_context_v3", "root_id": root,
                                    "phase": "scene", "scene_ref": scene, "executor_id": reviewer}),
        (receipt, "SAVE_RECEIPT", {"status": "RECORDED_EXECUTOR_JUDGMENT_NOT_EXTERNAL_CERTIFICATION",
                                   "executor_id": reviewer, "root_id": root, "phase": "scene",
                                   "semantic_judgment_supplied_by_executor": True,
                                   "private_file_access_proven_by_program": False, "internal_reading_proven_by_program": False})):
        for key, value in values.items():
            audit.need(key in obj and type(obj[key]) is type(value) and obj[key] == value, label + ":FIELD_MISMATCH:" + key)
    audit.need(draft.get("author") == author and reviewer != author, "DRAFT:AUTHOR_NOT_FROZEN_OR_SELF_REVIEW")
    rows = source.get("turns", [])
    refs = [r.get("turn_ref") for r in rows if isinstance(r, dict)]
    audit.need(source.get("root_id") == root and len(refs) == len(rows) and bool(refs) and
               all(isinstance(r, str) and r for r in refs) and len(set(refs)) == len(refs), "SOURCE:INVALID_TURN_REFS")
    source_hash = sha(source_path) if source_path.is_file() else None
    audit.need(source_hash is not None and review.get("source_sha256") == source_hash, "REVIEW:SOURCE_HASH_MISMATCH")
    audit.need(context.get("turn_refs_rendered") == refs and review.get("turn_refs_read") == refs, "REVIEW:SOURCE_COVERAGE_MISMATCH")
    texts = constraints.get("continuation_constraints", [])
    audit.need(constraints.get("root_id") == root and constraints.get("decision") in {"PASS", "WARNING"} and
               isinstance(texts, list) and bool(texts), "CONSTRAINTS:INELIGIBLE")
    audit.need(context.get("constraint_count_rendered") == len(texts), "READ_CONTEXT:CONSTRAINT_COVERAGE_MISMATCH")
    for obj, label, fields in ((review, "REVIEW", (("helper_binding", helper), ("read_context_binding", context_path),
                                                 ("input_render_binding", render_path), ("raw_judgment_binding", raw_path),
                                                 ("role_assignments_binding", role_path), ("constraints_binding", constraints_path))),
                                (context, "READ_CONTEXT", (("source_binding", source_path), ("constraints_binding", constraints_path),
                                                          ("role_assignments_binding", role_path), ("input_render_binding", render_path))),
                                (receipt, "SAVE_RECEIPT", (("read_context_binding", context_path), ("raw_judgment_binding", raw_path),
                                                          ("output_binding", review_path)))):
        for key, path in fields:
            audit.bound(obj.get(key), path, label + ":" + key)
    for key in ("public_binding", "asset_judgments_binding", "ordering_anchor_plan_binding"):
        audit.need(key in context and context[key] is None, "READ_CONTEXT:UNEXPECTED_QA_INPUT:" + key)
    audit.need(context.get("ordering_anchors_rendered") == [] and receipt.get("ordering_anchor_plan_binding") is None,
               "READ_CONTEXT:UNEXPECTED_QA_ANCHORS")
    expected = [role_path, unit / "RUN_CONFIG.json", constraints_path, source_path, helper, adapter, anchor_helper,
                batch / "MASTER_FREEZE.json", unit / "inputs/BASELINE.json", draft_path]
    try:
        expected.extend(prefix_input_paths(unit))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        audit.need(False, "PREFIX_INPUTS:INVALID_OR_PRIVATE_LINEAGE", str(exc))
    audit.bound(constraints.get("baseline_binding"), unit / "inputs/BASELINE.json", "CONSTRAINTS:BASELINE")
    baseline_refs = constraints.get("review_bindings", [])
    audit.need(isinstance(baseline_refs, list) and len(baseline_refs) == 2, "CONSTRAINTS:TWO_BASELINE_REVIEWS_REQUIRED")
    actual_baseline = []
    for i, item in enumerate(baseline_refs):
        try:
            p = Path(item["path"]); p = p if p.is_absolute() else unit / p
            audit.bound({**item, "path": str(p.resolve())}, p, f"BASELINE_REVIEW:{i}")
            record = audit.read(p, f"BASELINE_REVIEW:{i}")
            actual = item.get("reviewer", record.get("reviewer"))
            audit.need(actual == record.get("reviewer"), "BASELINE_REVIEW:IDENTITY_MISMATCH")
            actual_baseline.append(actual); expected.append(p)
        except (KeyError, TypeError) as exc:
            audit.need(False, "BASELINE_REVIEW:INVALID_BINDING", str(exc))
    audit.need(len(set(actual_baseline)) == 2 and set(actual_baseline) == set(role.get("baseline_reviewers", [])),
               "BASELINE_REVIEW:NOT_FROZEN_ASSIGNEES")
    expected.extend(folder / "scenes" / prior.removeprefix("R02:").replace(":", "_") / "SOURCE_REVIEW.json"
                    for prior in SCENES[:SCENES.index(scene)])
    native = selection = None
    if ":B:" in scene:
        reservation_path = unit / "SOURCE_RESERVATIONS.json"
        reservations = audit.read(reservation_path, "SOURCE_RESERVATIONS")
        matches = [r for r in reservations.get("allocation", []) if r.get("root_id") == root and r.get("scene_ref") == scene]
        audit.need(len(matches) == 1, "NATIVE_EVENT:RESERVATION_REQUIRED")
        if len(matches) == 1:
            row = matches[0]
            native_path = Path(row["source_item_path"])
            audit.bound({"path": str(native_path), "sha256": row["source_item_sha256"]}, native_path, "NATIVE_EVENT")
            native = audit.read(native_path, "NATIVE_EVENT")
            expected.extend((reservation_path, native_path))
            audit.bound(context.get("native_event_binding"), native_path, "READ_CONTEXT:native_event_binding")
            audit.need(review.get("native_event_full_read") is True and review.get("native_event_path") == str(native_path.resolve())
                       and review.get("native_event_sha256") == sha(native_path) and
                       review.get("native_turn_ids_read") == [t["turn_id"] for t in native.get("turns", [])], "REVIEW:NATIVE_EVENT_COVERAGE")
            audit.need(review.get("source_id") == draft.get("source_id"), "REVIEW:NATIVE_SOURCE_ID")
            if row.get("selection_path"):
                p = Path(row["selection_path"])
                audit.bound({"path": str(p), "sha256": row["selection_sha256"]}, p, "SELECTION")
                selection = audit.read(p, "SELECTION"); expected.append(p)
                audit.bound(context.get("selection_binding"), p, "READ_CONTEXT:selection_binding")
                audit.need(review.get("selection_path") == str(p.resolve()) and review.get("selection_sha256") == sha(p), "REVIEW:SELECTION_BINDING")
            else:
                audit.need(context.get("selection_binding") is None, "READ_CONTEXT:UNEXPECTED_SELECTION")
    else:
        audit.need(context.get("native_event_binding") is None, "READ_CONTEXT:UNEXPECTED_NATIVE_INPUT")
        selection_path = unit / "selection.json"
        selection = audit.read(selection_path, "SELECTION"); expected.append(selection_path)
        audit.bound(context.get("selection_binding"), selection_path, "READ_CONTEXT:selection_binding")
        continuations = selection.get("a_continuations")
        audit.need(selection.get("root_id") == root and isinstance(continuations, list) and len(continuations) == 3 and
                   [row.get("scene_ref") for row in continuations] == list(SCENES[:3]) and
                   all(isinstance(row, dict) and {"scene_ref", "baseline_scene_ref", "thread", "evidence"} <= set(row)
                       for row in continuations), "SELECTION:A_PLAN_REQUIRED")
    expected_by_path = {str(p.resolve()): p for p in expected}
    actual = context.get("input_bindings")
    actual_paths = [r.get("path") for r in actual if isinstance(r, dict)] if isinstance(actual, list) else []
    audit.need(isinstance(actual, list) and len(actual_paths) == len(actual) and len(set(actual_paths)) == len(actual_paths)
               and set(actual_paths) == set(expected_by_path), "READ_CONTEXT:INPUT_SET_MISMATCH")
    for item in actual if isinstance(actual, list) else []:
        if isinstance(item, dict) and item.get("path") in expected_by_path:
            audit.bound(item, expected_by_path[item["path"]], "READ_CONTEXT:INPUT")
    constants = helper_literals(audit, helper)
    if len(constants) == 2:
        for obj, label in ((review, "REVIEW"), (context, "READ_CONTEXT")):
            for key, constant in (("identity_verification_scope", "IDENTITY_LIMITATION"), ("read_attestation_scope", "READ_LIMITATION")):
                audit.need(obj.get(key) == constants[constant], label + ":LIMITATION_SCOPE_MISMATCH:" + key)
        lines = ["REVIEW_ASSIGNMENT", json.dumps({"root_id": root, "phase": "scene", "scene_ref": scene,
                  "executor_id": reviewer, "assigned_roles": role}, ensure_ascii=False),
                 "ALL_CONTINUATION_CONSTRAINTS", json.dumps(constraints, ensure_ascii=False, indent=2), "FULL_CUMULATIVE_SOURCE"]
        try:
            lines.extend(t["turn_ref"] + " | " + t["speaker"] + ": " + t["text"] for t in rows)
            lines.extend(["DRAFT_NOTES", json.dumps({k: v for k, v in draft.items() if k != "turns"}, ensure_ascii=False, indent=2)])
            for label, value in (("NATIVE_EVENT_FULL", native), ("SELECTION_FULL", selection)):
                if value is not None:
                    lines.extend([label, json.dumps(value, ensure_ascii=False, indent=2)])
            lines.extend(["EVIDENCE_LIMITATIONS", constants["IDENTITY_LIMITATION"], constants["READ_LIMITATION"]])
            audit.need(render_path.is_file() and render_path.read_text(encoding="utf-8") == "\n".join(lines) + "\n", "INPUT_RENDER:NOT_EXACT_CURRENT_INPUT_RENDER")
        except (TypeError, KeyError) as exc:
            audit.need(False, "INPUT_RENDER:INVALID_SOURCE", str(exc))
    attest = raw.get("read_attestation", {})
    for key in ("full_source_read", "all_continuation_constraints_read"):
        audit.need(attest.get(key) is True and review.get(key) is True, "REVIEW:EXPLICIT_READ_ATTESTATION_REQUIRED:" + key)
    if native is not None:
        audit.need(attest.get("native_event_full_read") is True, "RAW_JUDGMENT:NATIVE_READ_ATTESTATION_REQUIRED")
    audit.need(review.get("executor_read_attestation") == attest and bool(attest), "REVIEW:RAW_ATTESTATION_MISMATCH")
    for key in ("decision", "checks", "findings"):
        audit.need(key in raw and raw.get(key) == review.get(key), "REVIEW:RAW_JUDGMENT_FIELD_MISMATCH:" + key)
    constraint_review = raw.get("baseline_constraint_review")
    audit.need(isinstance(constraint_review, dict), "REVIEW:CONSTRAINT_JUDGMENT_REQUIRED")
    constraint_review = constraint_review if isinstance(constraint_review, dict) else {}
    normalized_constraints = deepcopy(constraint_review)
    normalized_constraints.update(constraints_binding=binding(constraints_path), baseline_gate_version=batch.name,
                                  inherited_baseline_decision=constraints.get("decision"))
    audit.need(review.get("baseline_constraint_review") == normalized_constraints, "REVIEW:RAW_CONSTRAINT_JUDGMENT_MISMATCH")
    audit.need(isinstance(review.get("checks"), dict) and set(review["checks"]) == CHECKS, "REVIEW:EIGHT_CHECKS_REQUIRED")
    checks = review.get("checks") if isinstance(review.get("checks"), dict) else {}
    audit.need(all(isinstance(v, dict) and v.get("status") in STATUSES and isinstance(v.get("reason"), str) and v["reason"].strip()
                   for v in checks.values()), "REVIEW:CHECK_STATUS_OR_REASON_INVALID")
    constraint_rows = constraint_review.get("constraint_results")
    audit.need(constraint_review.get("status") in STATUSES and isinstance(constraint_review.get("reason"), str)
               and bool(constraint_review["reason"].strip()), "REVIEW:CONSTRAINT_STATUS_OR_REASON_INVALID")
    audit.need(isinstance(constraint_rows, list) and len(constraint_rows) == len(texts), "REVIEW:CONSTRAINT_COVERAGE_INCOMPLETE")
    constraint_rows = constraint_rows if isinstance(constraint_rows, list) else []
    for index, row in enumerate(constraint_rows, 1):
        if not audit.need(isinstance(row, dict), "REVIEW:CONSTRAINT_ROW_REQUIRED"):
            continue
        audit.need(index <= len(texts) and row.get("index") == index and row.get("constraint") == texts[index - 1],
                   "REVIEW:CONSTRAINT_ORDER_OR_TEXT_CHANGED")
        audit.need(row.get("status") in STATUSES and isinstance(row.get("reason"), str) and bool(row["reason"].strip()),
                   "REVIEW:CONSTRAINT_ROW_STATUS_OR_REASON_INVALID")
        evidence = row.get("evidence_turn_refs")
        audit.need(isinstance(evidence, list) and bool(evidence) and all(isinstance(ref, str) for ref in evidence)
                   and len(set(evidence)) == len(evidence) and set(evidence) <= set(refs), "REVIEW:CONSTRAINT_EVIDENCE_REFS_INVALID")
    blocking = constraint_review.get("new_blocking_findings")
    audit.need(isinstance(blocking, list), "REVIEW:CONSTRAINT_BLOCKING_FINDINGS_REQUIRED")
    failed_constraint = any(isinstance(row, dict) and row.get("status") == "FAIL" for row in constraint_rows) or bool(blocking)
    audit.need(not failed_constraint or constraint_review.get("status") == "FAIL", "REVIEW:CONSTRAINT_FAILURE_NOT_AGGREGATED")
    audit.need(constraint_review.get("status") != "PASS" or all(isinstance(row, dict) and row.get("status") == "PASS" for row in constraint_rows),
               "REVIEW:CONSTRAINT_PASS_NOT_AGGREGATED")
    findings = review.get("findings")
    audit.need(isinstance(findings, list) and all(isinstance(row, dict) for row in findings), "REVIEW:FINDINGS_LIST_REQUIRED")
    audit.need(review.get("decision") in {"READY", "HOLD"}, "REVIEW:DECISION_INVALID")
    if review.get("decision") == "READY":
        audit.need(not any(isinstance(v, dict) and v.get("status") == "FAIL" for v in checks.values()), "REVIEW:READY_WITH_FAILED_CHECK")
        audit.need(constraint_review.get("status") != "FAIL" and not failed_constraint, "REVIEW:READY_WITH_FAILED_CONSTRAINT")
        audit.need(not any(isinstance(row, dict) and row.get("severity") == "BLOCKING" for row in findings or []), "REVIEW:READY_WITH_BLOCKING_FINDING")
    audit.need(not (directory / "SAVE_FAILURE.json").exists(), "SAVE_RECEIPT:FAILURE_RECORD_PRESENT")
    started = timestamp(audit, context.get("started_at_utc"), "READ_CONTEXT:STARTED")
    finished = timestamp(audit, review.get("finished_at_utc"), "REVIEW:FINISHED")
    saved = timestamp(audit, receipt.get("created_at_utc"), "SAVE_RECEIPT:CREATED")
    audit.need(review.get("started_at_utc") == context.get("started_at_utc") and review.get("created_at_utc") == review.get("finished_at_utc"),
               "REVIEW:READ_INTERVAL_MISMATCH")
    if started is not None and finished is not None and saved is not None:
        audit.need(started <= finished <= saved, "REVIEW:TIME_ORDER_INVALID")
    outcome = result(audit, batch, root, "scene")
    outcome["scene_ref"] = scene
    return outcome
