#!/usr/bin/env python3
"""Preserve actual initial drafts and saved initial reviews for yield analysis.

This is an engineering-only aggregator. It makes no model calls, authors no
content, and supplies no semantic judgment. Missing reviews remain unknown.
Only unsuffixed initial-review files are read; revision files are never used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import question_driver as q  # noqa: E402

OUTPUT_NAMES = (
    "CANDIDATES_INITIAL_PRIVATE.json",
    "CONTENT_REVIEW_INITIAL.json",
    "CONTENT_GATE_INITIAL.json",
    "INITIAL_CONTENT_ASSEMBLY.json",
)


def raw_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def verify_binding(record: dict[str, Any]) -> None:
    if binding(Path(record["path"])) != record:
        raise RuntimeError(f"INITIAL_INPUT_BINDING_DRIFT:{record.get('path')}")


def publish(path: Path, value: Any) -> dict[str, Any]:
    data = raw_json(value)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"IMMUTABLE_INITIAL_OUTPUT_DRIFT:{path}")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    return binding(path)


def unique_map(rows: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise RuntimeError(f"INITIAL_SAVED_ROWS_NOT_LIST:{label}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str):
            raise RuntimeError(f"INITIAL_SAVED_ROW_WITHOUT_ID:{label}")
        if row["candidate_id"] in result:
            raise RuntimeError(f"INITIAL_DUPLICATE_ID:{label}:{row['candidate_id']}")
        result[row["candidate_id"]] = row
    return result


def nested_bindings(value: Any) -> list[dict[str, Any]]:
    """Extract only explicit saved bindings; never infer missing provenance."""
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= set(value):
            return [{key: value[key] for key in ("path", "sha256", "bytes")}]
        return [row for child in value.values() for row in nested_bindings(child)]
    if isinstance(value, list):
        return [row for child in value for row in nested_bindings(child)]
    return []


def initial_validation(candidate: dict[str, Any], review: dict[str, Any] | None,
                       saved_audit: dict[str, Any] | None,
                       expected_sha: str,
                       validator: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """Reproduce engineering eligibility without inventing a reviewer decision."""
    if review is None:
        return None, "UNKNOWN_NO_SAVED_INITIAL_REVIEW"
    if saved_audit is None:
        return None, "UNKNOWN_NO_SAVED_INITIAL_MECHANICAL_AUDIT"
    for label, row in (("review", review), ("mechanical_audit", saved_audit)):
        if row.get("candidate_id") != candidate["candidate_id"] or row.get("candidate_sha256") != expected_sha:
            raise RuntimeError(f"INITIAL_{label.upper()}_FINGERPRINT_DRIFT:{candidate['candidate_id']}")
        if row.get("source_binding") != candidate.get("source_binding"):
            raise RuntimeError(f"INITIAL_{label.upper()}_SOURCE_BINDING_DRIFT:{candidate['candidate_id']}")
    if not isinstance(saved_audit.get("mechanical_pass"), bool):
        return None, "UNKNOWN_SAVED_INITIAL_MECHANICAL_STATUS_MISSING"
    try:
        current = validator(candidate, review)
    except Exception as exc:
        current = {"candidate_id": candidate["candidate_id"], "candidate_sha256": expected_sha,
                   "admitted": False, "mechanical_pass": False,
                   "review_record_complete": False,
                   "review_errors": [f"mechanical_validation_exception:{type(exc).__name__}:{exc}"]}
    result = {
        "candidate_id": candidate["candidate_id"], "candidate_sha256": expected_sha,
        "admitted": bool(saved_audit["mechanical_pass"] and current.get("admitted")),
        "saved_initial_review_decision": review.get("decision"),
        "saved_initial_mechanical_pass": saved_audit["mechanical_pass"],
        "saved_initial_mechanical_audit": saved_audit,
        "recomputed_review_schema_validation": current,
        "admission_basis": "Saved initial review surface and saved mechanical audit, mechanically checked against the unchanged candidate and final shared source. A saved missing/invalid-review HOLD is initial nonadmission, not an authored semantic rejection or a Solver zero. No new semantic judgment.",
    }
    return result, "KNOWN_FROM_SAVED_INITIAL_STAGE"


def assemble(selection_path: Path | None = None,
             stage_directory: Callable[[str, str, Path], Path] | None = None,
             extra_input_bindings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    selection_path = (selection_path or BASE / "SOURCE_SELECTION_FINAL.json").resolve()
    config_path = BASE / "CONFIG.json"
    config = q.read(config_path)
    # This verifies the actual canonical source freeze and all its provenance.
    q.SOURCE_SELECTION_PATH = selection_path
    _, source_rows = q.load_selection(selection_path)
    roots = [q.root_id(root) for root in config["roots"]]
    candidates: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    serialization_recoveries: list[dict[str, Any]] = []
    inputs = [binding(config_path), binding(selection_path), binding(q.SOURCE_FREEZE_PATH),
              binding(Path(__file__).resolve()), binding(Path(q.__file__).resolve()),
              binding(q.GATE_DIR / "production_question_gate.py")]
    inputs.extend(extra_input_bindings or [])
    all_ids: set[str] = set()
    for root in roots:
        for arm in q.ARMS:
            work = BASE / "question_authoring" / arm / "roots" / q.root_number(root)
            if stage_directory is not None:
                work = stage_directory(root, arm, work)
            draft_path = work / "CANDIDATES_DRAFT.json"
            review_path = work / "CONTENT_REVIEW_NORMALIZED.json"
            reference_path = work / "ORIGINAL_DRAFT_RECEIPT_REFERENCE.json"
            serialization_recovery = None
            if reference_path.is_file():
                reference = q.read(reference_path)
                if not isinstance(reference, dict):
                    raise RuntimeError(f"INITIAL_DRAFT_SERIALIZATION_REFERENCE_NOT_OBJECT:{reference_path}")
                reference_bindings = nested_bindings(reference)
                for row in reference_bindings:
                    verify_binding(row)
                inputs.extend([binding(reference_path), *reference_bindings])
                original_review_rows = [row for row in reference_bindings
                                        if Path(row["path"]).name == "CONTENT_REVIEW_NORMALIZED.json"]
                serialization_recovery = {
                    "root_id": root, "arm": arm,
                    "original_draft_reference_binding": binding(reference_path),
                    "recorded_provenance": reference,
                    "raw_initial_serialization_failure_surfaces": [
                        {"binding": row, "saved_original_surface": q.read(Path(row["path"]))}
                        for row in original_review_rows],
                    "denominator_policy": "The mechanically normalized candidate is the same consumed initial author draft. Its first unsuffixed independent reviews govern initial content validity; saved original serialization failures are retained separately, not an extra draft, content revision, or automatic semantic zero.",
                }
                serialization_recoveries.append(serialization_recovery)
            draft_exists, review_exists = draft_path.is_file(), review_path.is_file()
            initial = q.read(draft_path) if draft_exists else []
            if not isinstance(initial, list):
                raise RuntimeError(f"INITIAL_DRAFT_NOT_LIST:{draft_path}")
            doc = q.read(review_path) if review_exists else {}
            if not isinstance(doc, dict):
                raise RuntimeError(f"INITIAL_REVIEW_NOT_OBJECT:{review_path}")
            review_map = unique_map(doc.get("reviews", []), str(review_path))
            audit_map = unique_map(doc.get("mechanical_audits", []), str(review_path))
            for name in ("CANDIDATES_DRAFT.json", "CANDIDATES_DRAFT_RESPONSE.json",
                         "CANDIDATES_DRAFT_CALL.json", "DRAFT_NORMALIZATION.json",
                         "CONTENT_REVIEW_NORMALIZED.json", "BLIND_GOLD.json", "BLIND_GOLD_CALL.json",
                         "BLIND_GOLD_VALIDATION.json", "BLIND_GOLD_FAILURE.json",
                         "CONTENT_VERIFIER.json", "CONTENT_VERIFIER_CALL.json", "CONTENT_VERIFIER_FAILURE.json",
                         "TASK_PLANS.json", "TASK_PLANS_FREEZE.json", "MECHANICAL_PLAN_NORMALIZATION.json",
                         "CANDIDATE_SHAPE_NORMALIZATION.json", "ORIGINAL_DRAFT_RECEIPT_REFERENCE.json",
                         "MECHANICAL_CANDIDATE_NORMALIZATION_INITIAL.json", "RECOVERY_INPUT_FREEZE.json",
                         "ORIGINAL_PLAN_RECEIPT_REFERENCE.json", "DRIVER_ADAPTATION.json",
                         "DRIVER_ADAPTATION_V2.json", "DRIVER_ADAPTATION_V3.json", "PLAN_FAILURE_DISPOSITION.json"):
                path = work / name
                if path.is_file():
                    inputs.append(binding(path))
            for path in sorted(work.glob("MECHANICAL_*_SCHEMA_*.json")):
                if "revision_01" in path.name.lower():
                    continue
                inputs.append(binding(path))
                document = q.read(path)
                for row in nested_bindings(document):
                    verify_binding(row)
                    inputs.append(row)
            for path in sorted(work.glob("G_VERIFIER_ISOLATION_ATTESTATION_VALIDATION*.json")):
                if "revision_01" in path.name.lower():
                    continue
                inputs.append(binding(path))
                for row in nested_bindings(q.read(path)):
                    verify_binding(row)
                    inputs.append(row)
            by_slot: dict[str, dict[str, Any]] = {}
            for candidate in initial:
                if not isinstance(candidate, dict) or not isinstance(candidate.get("candidate_id"), str):
                    raise RuntimeError(f"INITIAL_DRAFT_ROW_WITHOUT_ID:{draft_path}")
                cid, slot = candidate["candidate_id"], candidate.get("slot_id")
                if candidate.get("root_id") != root or candidate.get("arm") != arm or slot not in q.SLOTS:
                    raise RuntimeError(f"INITIAL_ASSIGNMENT_DRIFT:{cid}")
                if root not in source_rows or candidate.get("source_binding") != source_rows[root]["binding"]:
                    raise RuntimeError(f"INITIAL_SOURCE_NOT_IN_FINAL_SELECTION:{cid}")
                if cid in all_ids or slot in by_slot:
                    raise RuntimeError(f"INITIAL_DUPLICATE_CANDIDATE_OR_SLOT:{cid}")
                all_ids.add(cid)
                by_slot[slot] = candidate
                candidates.append(candidate)
            if set(review_map) - {c["candidate_id"] for c in initial} or set(audit_map) - {c["candidate_id"] for c in initial}:
                raise RuntimeError(f"INITIAL_REVIEW_WITHOUT_SAVED_DRAFT:{review_path}")
            for slot in q.SLOTS:
                candidate = by_slot.get(slot)
                entry: dict[str, Any] = {
                    "root_id": root, "arm": arm, "slot_id": slot,
                    "actual_initial_stage_directory": str(work.resolve()),
                    "candidate_id": candidate.get("candidate_id") if candidate else None,
                    "saved_initial_draft_file_present": draft_exists,
                    "saved_initial_review_file_present": review_exists,
                    "candidate_present": candidate is not None,
                    "initial_content_valid": None,
                }
                if serialization_recovery is not None:
                    entry["initial_serialization_recovery_reference"] = serialization_recovery["original_draft_reference_binding"]
                    entry["raw_initial_serialization_failure_preserved_separately"] = True
                if candidate is None:
                    entry["status"] = "UNKNOWN_NO_SAVED_INITIAL_CANDIDATE"
                else:
                    cid = candidate["candidate_id"]
                    review = review_map.get(cid)
                    audit = audit_map.get(cid)
                    if review is not None:
                        reviews.append(review)
                    validation, status = initial_validation(candidate, review, audit,
                        q.gate.fingerprint(candidate),
                        lambda c, r: q.gate.validate_review(c, r, selection_path))
                    entry.update(status=status, saved_initial_review_decision=review.get("decision") if review else None,
                                 saved_initial_mechanical_pass=audit.get("mechanical_pass") if audit else None)
                    if validation is not None:
                        validations.append(validation)
                        entry["initial_content_valid"] = validation["admitted"]
                ledger.append(entry)
    order = lambda c: (roots.index(c["root_id"]), list(q.SLOTS).index(c["slot_id"]), q.ARMS.index(c["arm"]))
    candidates.sort(key=order)
    cid_order = {c["candidate_id"]: i for i, c in enumerate(candidates)}
    reviews.sort(key=lambda row: cid_order[row["candidate_id"]])
    validations.sort(key=lambda row: cid_order[row["candidate_id"]])
    inputs = sorted({row["path"]: row for row in inputs}.values(), key=lambda row: row["path"])
    review_doc = {"schema_version": "fgr20_content_review_initial_v1", "reviews": reviews,
                  "provenance": "Exact saved unsuffixed CONTENT_REVIEW_NORMALIZED.json records; no revision reviews and no generated review decisions.",
                  "reviewers_are_same_codex_family_ai_not_human": True}
    gate_doc = {"schema_version": "fgr20_content_gate_initial_v1", "validations": validations,
                "planned_slot_count": len(roots) * len(q.ARMS) * len(q.SLOTS),
                "candidate_count": len(candidates), "known_initial_admission_count": len(validations),
                "admitted_count": sum(row["admitted"] for row in validations),
                "unknown_initial_status_count": sum(row["initial_content_valid"] is None for row in ledger),
                "slot_ledger": ledger,
                "unknown_policy": "No validation row is emitted when a saved initial review surface or mechanical audit is absent; downstream missingness remains null. A saved engineering HOLD for missing/invalid independent review is observed initial gate nonadmission, not a semantic content-false judgment and never an imputed Solver score.",
                "model_calls": 0, "semantic_judgments_authored": 0}
    output_bindings = [publish(BASE / OUTPUT_NAMES[0], candidates), publish(BASE / OUTPUT_NAMES[1], review_doc),
                       publish(BASE / OUTPUT_NAMES[2], gate_doc)]
    receipt = {"schema_version": "fgr20_initial_content_assembly_v1", "status": "ASSEMBLED_FROM_ACTUAL_INITIAL_STAGE",
               "input_bindings": inputs, "output_bindings": output_bindings,
               "candidate_count": len(candidates), "saved_initial_review_count": len(reviews),
               "known_initial_admission_count": len(validations), "admitted_count": gate_doc["admitted_count"],
               "unknown_initial_status_count": gate_doc["unknown_initial_status_count"],
               "same_initial_draft_serialization_recoveries": serialization_recoveries,
               "model_calls": 0, "semantic_judgments_authored": 0}
    publish(BASE / OUTPUT_NAMES[3], receipt)
    return receipt


def self_check() -> dict[str, Any]:
    candidate = {"candidate_id": "fixture", "source_binding": {"fixture": True}}
    review = {**candidate, "candidate_sha256": "hash", "decision": "PASS"}
    audit = {**candidate, "candidate_sha256": "hash", "mechanical_pass": True}
    checks = []
    validate = lambda c, r: {"admitted": r.get("decision") == "PASS", "mechanical_pass": True}
    assert initial_validation(candidate, None, audit, "hash", validate)[0] is None
    checks.append("missing_review_is_unknown")
    assert initial_validation(candidate, review, None, "hash", validate)[0] is None
    checks.append("missing_initial_mechanical_audit_is_unknown")
    assert initial_validation(candidate, review, audit, "hash", validate)[0]["admitted"]
    checks.append("saved_pass_and_mechanical_gate_required")
    assert not initial_validation(candidate, review, {**audit, "mechanical_pass": False}, "hash", validate)[0]["admitted"]
    checks.append("saved_initial_mechanical_failure_not_upgraded")
    assert not initial_validation(candidate, {**review, "decision": "HOLD"}, audit, "hash", validate)[0]["admitted"]
    checks.append("saved_hold_not_upgraded")
    try:
        initial_validation(candidate, {**review, "candidate_sha256": "wrong"}, audit, "hash", validate)
    except RuntimeError:
        checks.append("fingerprint_drift_rejected")
    else:
        raise AssertionError("fingerprint drift accepted")
    explicit = {"path": "/fixture/raw", "sha256": "fixture", "bytes": 1}
    assert nested_bindings({"saved": explicit, "missing": {"path": "/fixture/incomplete"},
                            "policy": "no inferred provenance"}) == [explicit]
    checks.append("serialization_provenance_requires_explicit_full_binding")
    return {"status": "PASS", "checks": checks, "models_called": False, "real_inputs_read": False, "files_written": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    result = self_check() if args.self_check else assemble()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
