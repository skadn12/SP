#!/usr/bin/env python3
"""Aggregate settled exact-five F136 verification without changing frozen code.

This wrapper performs artifact validation and invokes the existing initial
aggregator and frozen finalizer.  It has no author, Gold, verifier, or Solver
dispatch.  A missing new terminal output requires an explicit parent-owned
missingness disposition and retains the prior saved reviews.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
import question_driver as q
import assemble_initial_content as initial
import content_continuation_adapter as continuation
import content_assembly_driver as existing

ROOT = "cc-train-episode-136202"
ARM = "F"
SLOTS = ("MC2", "MC3", "MC4", "MC5", "MC6")
RETAIN = ("ORDER1", "ORDER2", "ORDER3")
KIND = "saved_final_MC_five_first_verifier"
VARIANT = "f136202_final_revision_exact_five_MC_first_verifier_list_keyed"
TREE = BASE / "question_saved_mc5_first_verification"
INPUT = BASE / "question_candidate_shape_continuation/136202/F"
OUT = TREE / "136202/F"
TERMINAL = OUT / continuation.READY_NAME
READY = BASE / "CONTENT_ASSEMBLY_DELIVERY_READY.json"
ADAPTATION = BASE / "CONTENT_ASSEMBLY_DELIVERY_ADAPTATION.json"
DISPOSITIONS = BASE / "DELIVERY_TERMINAL_DISPOSITIONS.json"
COMPONENT = "f136202_mc5_first_verifier"
NEW_CALL = "q20_verify_f_136202_saved_mc5_first_01_revision_01"
WRAPPER = BASE / "scripts/f136202_saved_mc5_first_verification.py"


def nested_bindings(value: Any) -> list[dict[str, Any]]:
    rows = []
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= set(value):
            rows.append({key: value[key] for key in ("path", "sha256", "bytes")})
        for child in value.values():
            rows.extend(nested_bindings(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(nested_bindings(child))
    return rows


def strict_local(row: dict[str, Any], expected: Path | None = None) -> Path:
    if not isinstance(row, dict):
        raise RuntimeError("DELIVERY_CONTENT_BINDING_NOT_OBJECT")
    path = continuation.local_path(row.get("path", ""))
    if expected is not None and path != expected.resolve():
        raise RuntimeError("DELIVERY_CONTENT_BINDING_PATH_DRIFT:" + str(expected))
    q.verify_binding(row)
    return path


def exact_preservation(candidates: list[dict[str, Any]], old_candidates: list[dict[str, Any]],
                       reviews: list[dict[str, Any]], old_reviews: list[dict[str, Any]]) -> None:
    """Compare saved objects only; never manufacture or improve a decision."""
    final = continuation.by_id(candidates, "delivery-final")
    original = continuation.by_id(old_candidates, "delivery-original")
    current_reviews = continuation.by_id(reviews, "delivery-final-review")
    previous = continuation.by_id(old_reviews, "delivery-original-review")
    if final != original or len(final) != 8 or set(current_reviews) != set(final):
        raise RuntimeError("DELIVERY_F136_EXACT_EIGHT_CANDIDATE_OR_REVIEW_SET_DRIFT")
    if {row.get("slot_id") for row in final.values()} != set(SLOTS + RETAIN):
        raise RuntimeError("DELIVERY_F136_EXACT_EIGHT_SLOT_SET_DRIFT")
    for cid, candidate in final.items():
        slot = candidate.get("slot_id")
        if (candidate.get("root_id"), candidate.get("arm")) != (ROOT, ARM):
            raise RuntimeError("DELIVERY_F136_CANDIDATE_IDENTITY_DRIFT:" + cid)
        if slot in RETAIN:
            if previous.get(cid, {}).get("decision") != "PASS" or current_reviews[cid] != previous[cid]:
                raise RuntimeError("DELIVERY_F136_EXISTING_ORDER_REVIEW_CHANGED:" + cid)
        elif previous.get(cid, {}).get("decision") != "HOLD":
            raise RuntimeError("DELIVERY_F136_PREVIOUS_MC_WAS_ALREADY_REVIEWED:" + cid)


def missingness_disposition(base: Path = BASE) -> dict[str, Any]:
    path = base / DISPOSITIONS.name
    if not path.is_file():
        raise RuntimeError("DELIVERY_F136_TERMINAL_OR_EXPLICIT_PARENT_MISSINGNESS_REQUIRED")
    doc = q.read(path)
    if (doc.get("schema_version") != "fgr20_delivery_terminal_dispositions_v1"
            or doc.get("issuer_agent") != "/root"
            or doc.get("authorizes_no_new_semantic_calls") is not True
            or not isinstance(doc.get("dispositions"), list)):
        raise RuntimeError("DELIVERY_F136_PARENT_MISSINGNESS_SCHEMA_INVALID")
    found = [row for row in doc["dispositions"] if row.get("component") == COMPONENT]
    if len(found) != 1:
        raise RuntimeError("DELIVERY_F136_EXACT_PARENT_MISSINGNESS_REQUIRED")
    row = found[0]
    if (row.get("status") not in {"FAILED_TERMINAL", "NOT_RUN_TERMINAL"}
            or row.get("allow_delivery_with_missingness") is not True
            or not isinstance(row.get("reason"), str) or not row["reason"].strip()
            or not isinstance(row.get("evidence_bindings"), list) or not row["evidence_bindings"]):
        raise RuntimeError("DELIVERY_F136_PARENT_MISSINGNESS_FIELDS_INVALID")
    for binding in row["evidence_bindings"]:
        q.verify_binding(binding)
    return {"binding": q.binding(path), "record": copy.deepcopy(row),
            "missingness_preserved_not_replaced_by_PASS": True}


def require_approval() -> dict[str, Any]:
    doc = q.read(READY)
    if (doc.get("schema_version") != "fgr20_content_assembly_delivery_ready_v1"
            or doc.get("status") != "READY_ENGINEERING_ONLY"
            or doc.get("wrapper_binding") != q.binding(Path(__file__).resolve())
            or doc.get("semantic_calls_started_by_engineering") != 0
            or doc.get("semantic_admission_asserted") is not False
            or not isinstance(doc.get("dependency_bindings"), list)
            or not isinstance(doc.get("independent_review_binding"), dict)):
        raise RuntimeError("CONTENT_ASSEMBLY_DELIVERY_APPROVAL_INVALID")
    for binding in doc["dependency_bindings"] + [doc["independent_review_binding"]]:
        q.verify_binding(binding)
    review = q.read(doc["independent_review_binding"]["path"])
    if (review.get("status") != "PASS_ENGINEERING_ONLY"
            or review.get("script_binding") != doc["wrapper_binding"]
            or review.get("semantic_calls_started") != 0):
        raise RuntimeError("CONTENT_ASSEMBLY_DELIVERY_INDEPENDENT_REVIEW_INVALID")
    return doc


def validate_f136_terminal_ready(path: Path | None = None) -> dict[str, Any]:
    """Independently check the exact new terminal and its retained lineage."""
    path = (path or TERMINAL).resolve()
    if path != TERMINAL.resolve():
        raise RuntimeError("DELIVERY_F136_TERMINAL_PATH_NOT_APPROVED")
    owner = importlib.import_module("f136202_saved_mc5_first_verification")
    if Path(owner.__file__).resolve() != WRAPPER.resolve():
        raise RuntimeError("DELIVERY_F136_WRAPPER_IMPORT_PATH_DRIFT")
    owner_checked = owner.validate_terminal_ready(path)
    row = q.read(path)
    if (row.get("schema_version") != "fgr20_question_continuation_finalizer_ready_v1"
            or (row.get("root_id"), row.get("arm"), row.get("continuation_kind"),
                row.get("serialization_variant")) != (ROOT, ARM, KIND, VARIANT)
            or Path(row.get("original_work_dir", "")).resolve() != INPUT.resolve()
            or Path(row.get("final_work_dir", "")).resolve() != OUT.resolve()
            or row.get("new_verifier_exact_slot_ids") != list(SLOTS)
            or row.get("retained_existing_review_slot_ids") != list(RETAIN)):
        raise RuntimeError("DELIVERY_F136_TERMINAL_IDENTITY_DRIFT")
    strict_local(row["supersedes_ready_binding"], INPUT / continuation.READY_NAME)
    strict_local(row["driver_binding"], WRAPPER)
    for key, expected in (("final_candidates_binding", OUT / "CANDIDATES_FINAL.json"),
                          ("final_review_binding", OUT / "CONTENT_REVIEW_FINAL.json"),
                          ("slot_failures_binding", OUT / "SLOT_FAILURES.json"),
                          ("prior_final_reviews_archive_binding", OUT / "ORIGINAL_FINAL_REVIEWS_ARCHIVE.json"),
                          ("retained_ORDER_reviews_binding", OUT / "RETAINED_ORDER_REVIEWS.json"),
                          ("existing_ORDER_verifier_receipt_binding", INPUT / "CONTENT_VERIFIER_CALL_revision_01.json")):
        strict_local(row[key], expected)
    if (OUT / "CANDIDATES_FINAL.json").read_bytes() != (INPUT / "CANDIDATES_FINAL.json").read_bytes():
        raise RuntimeError("DELIVERY_F136_FINAL_CANDIDATE_BYTES_CHANGED")
    if (OUT / "ORIGINAL_FINAL_REVIEWS_ARCHIVE.json").read_bytes() != (INPUT / "CONTENT_REVIEW_FINAL.json").read_bytes():
        raise RuntimeError("DELIVERY_F136_ORIGINAL_REVIEW_ARCHIVE_BYTES_CHANGED")
    candidates = q.read(OUT / "CANDIDATES_FINAL.json")
    reviews = q.read(OUT / "CONTENT_REVIEW_FINAL.json")["reviews"]
    previous = q.read(INPUT / "CONTENT_REVIEW_FINAL.json")["reviews"]
    exact_preservation(candidates, q.read(INPUT / "CANDIDATES_FINAL.json"), reviews, previous)
    retained = [review for review in previous if review["candidate_id"] in
                {c["candidate_id"] for c in candidates if c["slot_id"] in RETAIN}]
    saved_retained = q.read(OUT / "RETAINED_ORDER_REVIEWS.json")
    if isinstance(saved_retained, dict):
        saved_retained = saved_retained.get("reviews")
    expected_retained = ({row["candidate_id"]: row for row in retained}
                         if isinstance(saved_retained, dict) else retained)
    if saved_retained != expected_retained:
        raise RuntimeError("DELIVERY_F136_RETAINED_ORDER_ARCHIVE_OBJECT_DRIFT")
    selection_path = BASE / "SOURCE_SELECTION_FINAL.json"
    q.SOURCE_SELECTION_PATH = selection_path
    q.SOURCE_FREEZE_PATH = BASE / "SOURCE_FREEZE.json"
    _, sources = q.load_selection(selection_path)
    source = sources[ROOT]
    review_by_id = continuation.by_id(reviews, "delivery-final-review")
    validations = []
    for candidate in candidates:
        review = review_by_id[candidate["candidate_id"]]
        if (candidate.get("source_binding") != source["binding"]
                or review.get("candidate_sha256") != q.gate.fingerprint(candidate)
                or review.get("source_binding") != candidate["source_binding"]
                or review.get("author_id") != candidate.get("author_id")):
            raise RuntimeError("DELIVERY_F136_FINGERPRINT_SOURCE_AUTHOR_DRIFT:" + candidate["candidate_id"])
        audit = q.audit_candidate(candidate, selection_path)
        validation = q.gate.validate_review(candidate, review, selection_path)
        if review.get("decision") == "PASS" and not validation.get("admitted"):
            raise RuntimeError("DELIVERY_F136_PASS_REVIEW_FAILED_FROZEN_GATE:" + candidate["candidate_id"])
        validations.append({"candidate_id": candidate["candidate_id"],
                            "mechanical_audit": audit, "review_validation": validation})
    inputs = nested_bindings(row) + owner_checked.get("input_bindings", [])
    for artifact in sorted(OUT.glob("*.json")):
        inputs.extend(nested_bindings(q.read(artifact)))
    inputs.extend([q.binding(path), q.binding(WRAPPER)])
    for binding in inputs:
        strict_local(binding)
    return {"record": {**row, "ready_binding": q.binding(path), "ready_path": str(path)},
            "input_bindings": sorted({row["path"]: row for row in inputs}.values(), key=lambda row: row["path"]),
            "frozen_gate_validations": validations,
            "candidate_bytes_preserved": True, "existing_ORDER_review_objects_preserved": True,
            "semantic_admission_asserted_by_engineering": False}


class DeliveryContinuations(continuation.Continuations):
    def __init__(self) -> None:
        super().__init__()
        self.delivery_observation: dict[str, Any]
        self.tree_files.extend(path for path in sorted(TREE.rglob("*"))
                               if path.is_file() and "__pycache__" not in path.parts
                               and not path.name.startswith(".") and path.suffix not in {".lock", ".pyc"})
        if TERMINAL.is_file():
            checked = validate_f136_terminal_ready()
            self.records.append(checked["record"])
            self.inputs.extend(checked["input_bindings"])
            self.select_heads()
            if self.selected[(ROOT, ARM)]["ready_binding"] != q.binding(TERMINAL):
                raise RuntimeError("DELIVERY_F136_NEW_TERMINAL_NOT_SELECTED_HEAD")
            self.delivery_observation = {"status": "OBSERVED_TERMINAL_EXACT_FIVE_FIRST_VERIFIER",
                                         **checked}
        else:
            disposition = missingness_disposition()
            self.inputs.append(disposition["binding"])
            self.inputs.extend(disposition["record"]["evidence_bindings"])
            self.delivery_observation = {"status": "EXPLICIT_PARENT_TERMINAL_MISSINGNESS",
                                         **disposition}


def self_check() -> dict[str, Any]:
    candidates = [{"candidate_id": "FIXTURE-" + slot, "root_id": ROOT, "arm": ARM,
                   "slot_id": slot} for slot in SLOTS + RETAIN]
    reviews = [{"candidate_id": c["candidate_id"], "decision": "PASS" if c["slot_id"] in RETAIN else "HOLD"}
               for c in candidates]
    exact_preservation(candidates, copy.deepcopy(candidates), reviews, copy.deepcopy(reviews))
    for defect in ("candidate", "order_review", "prior_mc_pass", "missing_review", "slot_identity"):
        args = [copy.deepcopy(candidates), copy.deepcopy(candidates), copy.deepcopy(reviews), copy.deepcopy(reviews)]
        if defect == "candidate": args[0][0]["question"] = "FIXTURE_CHANGED"
        elif defect == "order_review": args[2][-1]["rationale"] = "FIXTURE_CHANGED"
        elif defect == "prior_mc_pass": args[3][0]["decision"] = "PASS"
        elif defect == "missing_review": args[2].pop()
        else: args[0][0]["root_id"] = "FIXTURE_OTHER_ROOT"; args[1][0]["root_id"] = "FIXTURE_OTHER_ROOT"
        try:
            exact_preservation(*args)
        except RuntimeError:
            continue
        raise AssertionError("UNSAFE_PRESERVATION_FIXTURE_ACCEPTED:" + defect)
    cache = BASE / "runtime_cache/content_assembly_delivery_selfcheck"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fgr20_delivery_disposition_", dir=cache) as directory:
        base = Path(directory)
        try:
            missingness_disposition(base)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Missing parent disposition accepted")
        evidence = base / "evidence.json"; evidence.write_text("{}\n")
        doc = {"schema_version": "fgr20_delivery_terminal_dispositions_v1", "issuer_agent": "/root",
               "authorizes_no_new_semantic_calls": True, "dispositions": [{"component": COMPONENT,
               "status": "NOT_RUN_TERMINAL", "allow_delivery_with_missingness": True,
               "reason": "ENGINEERING_FIXTURE_ONLY", "evidence_bindings": [q.binding(evidence)]}]}
        (base / DISPOSITIONS.name).write_text(json.dumps(doc))
        assert missingness_disposition(base)["missingness_preserved_not_replaced_by_PASS"] is True
        for defect in ("issuer", "component", "empty_evidence", "new_calls"):
            bad = copy.deepcopy(doc)
            if defect == "issuer": bad["issuer_agent"] = "FIXTURE_OTHER"
            elif defect == "component": bad["dispositions"][0]["component"] = "FIXTURE_OTHER"
            elif defect == "empty_evidence": bad["dispositions"][0]["evidence_bindings"] = []
            else: bad["authorizes_no_new_semantic_calls"] = False
            (base / DISPOSITIONS.name).write_text(json.dumps(bad))
            try:
                missingness_disposition(base)
            except RuntimeError:
                continue
            raise AssertionError("UNSAFE_MISSINGNESS_FIXTURE_ACCEPTED:" + defect)
    return {"status": "PASS_ENGINEERING_ONLY", "models_called": False, "semantic_calls_started": 0,
            "semantic_judgments_authored": 0, "actual_content_or_solver_launched": False,
            "exact_preservation_negative_fixtures": 5, "missingness_negative_fixtures": 5,
            "temporary_fixture_outputs_removed": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), ensure_ascii=False, indent=2)); return 0
    approval = require_approval()
    audit = q.read(BASE / "QUESTION_DRIVER_AUDIT_READY.json"); q.verify_binding(audit["driver_binding"])
    analysis = q.read(BASE / "ANALYSIS_DRIVER_FREEZE.json")
    if analysis.get("status") != "FROZEN_BEFORE_SOLVER":
        raise RuntimeError("ANALYSIS_DRIVER_NOT_FROZEN")
    q.verify_binding(analysis["script_binding"]); q.verify_binding(analysis["analysis_plan_binding"])
    q.SOURCE_SELECTION_PATH = BASE / "SOURCE_SELECTION_FINAL.json"
    q.SOURCE_FREEZE_PATH = BASE / "SOURCE_FREEZE.json"
    actual = DeliveryContinuations()
    observation = {"schema_version": "fgr20_content_assembly_delivery_adaptation_v1",
        "status": "ENGINEERING_ONLY_NO_NEW_SEMANTIC_ADMISSION", "wrapper_binding": q.binding(Path(__file__).resolve()),
        "engineering_ready_binding": q.binding(READY), "existing_wrapper_binding": q.binding(existing.__file__),
        "frozen_adapter_binding": q.binding(continuation.__file__),
        "exact_five_first_verifier": actual.delivery_observation,
        "frozen_candidate_review_duplicate_and_gate_logic_changed": False,
        "semantic_calls_started": 0, "semantic_judgments_authored": 0,
        "input_bindings": actual.inputs}
    if args.inspect:
        print(json.dumps(observation, ensure_ascii=False, indent=2)); return 0
    initial_receipt = initial.assemble(q.SOURCE_SELECTION_PATH, actual.initial_work,
        actual.inputs + [initial.binding(Path(continuation.__file__).resolve()), q.binding(Path(__file__).resolve())])
    continuation.publish(ADAPTATION, observation)
    new_paths = [Path(__file__).resolve(), READY, ADAPTATION,
                 *[Path(row["path"]) for row in approval["dependency_bindings"]],
                 Path(approval["independent_review_binding"]["path"]), *actual.tree_files]
    q.EXTRA_DRIVER_BINDINGS = list(q.EXTRA_DRIVER_BINDINGS) + existing.extra_bindings() + new_paths
    freeze = continuation.finalize(actual)
    print(json.dumps({"initial_assembly": initial_receipt, "final_content_freeze": freeze,
                      "delivery_adaptation_binding": q.binding(ADAPTATION)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
