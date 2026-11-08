"""Add hard closeout checks for frozen Day 3 supplement evidence."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def verify_bindings(refs, errors, label):
    for ref in refs:
        try:
            actual = bind(ref["path"])
        except OSError as exc:
            errors.append({"check": label, "path": ref.get("path"), "error": str(exc)})
            continue
        if actual != ref:
            errors.append({"check": label, "path": ref["path"], "expected": ref, "actual": actual})


def main():
    errors = []
    freeze_path = OUT / "EXECUTION_FREEZE.json"
    admission_path = OUT / "CONTENT_ADMISSION.json"
    qa_path = OUT / "review/CONTENT_QA.json"
    family_path = OUT / "review/FAMILY_BOUNDARY_AUDIT.json"
    mechanical_path = OUT / "review/MECHANICAL_SOURCE_VALIDATION_V2.json"
    request_freeze_path = OUT / "requests/REQUEST_FREEZE.json"
    preflight_path = OUT / "requests/CPU_PREFLIGHT.json"
    coverage_path = OUT / "SCOPED_COVERAGE_DECISION.json"

    freeze = read(freeze_path)
    admission = read(admission_path)
    qa = read(qa_path)
    family = read(family_path)
    mechanical = read(mechanical_path)
    request_freeze = read(request_freeze_path)
    preflight = read(preflight_path)
    coverage = read(coverage_path)

    ordering_checks = []
    accepted = [r for r in admission["records"] if r.get("status") == "CONTENT_ACCEPTED"]
    accepted_ids = {r["candidate_id"] for r in accepted}
    qa_rows = {r["candidate_id"]: r for r in qa["records"]}
    family_rows = {r["candidate_id"]: r for r in family["records"]}

    if len(accepted) != 12 or len(accepted_ids) != 12:
        errors.append({"check": "ACCEPTED_SET", "expected": 12, "actual": len(accepted_ids)})
    if set(qa_rows) != accepted_ids or set(family_rows) != accepted_ids:
        errors.append({"check": "REVIEW_ID_SET", "accepted": sorted(accepted_ids),
                       "qa": sorted(qa_rows), "family": sorted(family_rows)})
    if mechanical.get("status") != "PASS":
        errors.append({"check": "MECHANICAL_SOURCE_VALIDATION", "status": mechanical.get("status")})
    if request_freeze.get("question_count") != 12 or request_freeze.get("variant_count") != 108 or request_freeze.get("request_count_per_model") != 180:
        errors.append({"check": "REQUEST_COUNTS", "question_count": request_freeze.get("question_count"),
                       "variant_count": request_freeze.get("variant_count"),
                       "request_count_per_model": request_freeze.get("request_count_per_model")})
    if request_freeze.get("execution_permitted") is not False or request_freeze.get("training_start_go") is not False:
        errors.append({"check": "EXECUTION_GATE", "execution_permitted": request_freeze.get("execution_permitted"),
                       "training_start_go": request_freeze.get("training_start_go")})
    if preflight.get("status") != "PASS" or preflight.get("length_holds") or preflight.get("zero_truncation") is not True:
        errors.append({"check": "CPU_PREFLIGHT", "status": preflight.get("status"),
                       "length_holds": preflight.get("length_holds"), "zero_truncation": preflight.get("zero_truncation")})
    if coverage.get("status") != "SCOPED_PREPARATION_GO" or coverage.get("all_gates_pass") is not True:
        errors.append({"check": "SCOPED_COVERAGE", "status": coverage.get("status"),
                       "all_gates_pass": coverage.get("all_gates_pass")})

    for row in accepted:
        cid = row["candidate_id"]
        qa_row = qa_rows.get(cid, {})
        family_row = family_rows.get(cid, {})
        for ref in (row.get("public_question"), row.get("private_record")):
            if ref:
                try:
                    if bind(ref["path"]) != ref:
                        errors.append({"check": "ADMITTED_FILE_HASH", "candidate_id": cid, "path": ref["path"]})
                except OSError as exc:
                    errors.append({"check": "ADMITTED_FILE_MISSING", "candidate_id": cid, "path": ref["path"], "error": str(exc)})
        if family_row.get("status") != "PASS" or family_row.get("whole_task_equivalent_ids"):
            errors.append({"check": "WHOLE_TASK_EQUIVALENCE", "candidate_id": cid,
                           "status": family_row.get("status"),
                           "whole_task_equivalent_ids": family_row.get("whole_task_equivalent_ids")})
        if qa_row.get("status") != "PASS":
            errors.append({"check": "CONTENT_QA", "candidate_id": cid, "status": qa_row.get("status")})
        if row.get("question_type") != "ordering":
            continue
        labels = [e.get("label") for e in qa_row.get("label_evidence", [])]
        first_indices = {e.get("label"): e.get("first_semantic_report_index_1_based")
                         for e in qa_row.get("label_evidence", [])}
        collisions = {}
        for label, idx in first_indices.items():
            collisions.setdefault(idx, []).append(label)
        duplicate_positions = {str(idx): labels_at_idx for idx, labels_at_idx in collisions.items()
                               if len(labels_at_idx) > 1}
        valid_indices = all(isinstance(idx, int) and idx > 0 for idx in first_indices.values())
        source_ref = next((r["source"] for r in freeze["roots"] if r["root_id"] == row["root_id"]), None)
        source = read(source_ref["path"]) if source_ref else {"turns": []}
        anchor_errors = []
        for evidence in qa_row.get("label_evidence", []):
            idx = evidence.get("first_semantic_report_index_1_based")
            anchors = evidence.get("source_evidence", [])
            if not isinstance(idx, int) or not anchors or anchors[0].get("turn_index_1_based") != idx:
                anchor_errors.append(evidence.get("label"))
                continue
            if idx < 1 or idx > len(source["turns"]):
                anchor_errors.append(evidence.get("label"))
                continue
            turn = source["turns"][idx - 1]
            anchor = anchors[0]
            if (turn.get("turn_ref") != anchor.get("turn_ref") or
                    turn.get("speaker") != anchor.get("speaker") or
                    turn.get("scene_ref") != anchor.get("scene_ref") or
                    turn.get("text") != anchor.get("exact_quote")):
                anchor_errors.append(evidence.get("label"))
        expected_order = sorted(first_indices, key=first_indices.get) if valid_indices else []
        if duplicate_positions:
            errors.append({"check": "ORDERING_FIRST_POSITION_UNIQUE", "candidate_id": cid,
                           "duplicate_positions": duplicate_positions})
        if not valid_indices or set(labels) != set(first_indices) or len(labels) != len(first_indices):
            errors.append({"check": "ORDERING_FIRST_POSITION_SCHEMA", "candidate_id": cid,
                           "labels": labels, "first_positions": first_indices})
        if anchor_errors:
            errors.append({"check": "ORDERING_FIRST_ANCHORS", "candidate_id": cid,
                           "labels": anchor_errors})
        if qa_row.get("reconstructed_gold") != expected_order:
            errors.append({"check": "ORDERING_GOLD_ORDER", "candidate_id": cid,
                           "expected": expected_order, "actual": qa_row.get("reconstructed_gold")})
        ordering_checks.append({"candidate_id": cid, "item_count": len(labels),
                                "unique_first_positions": not duplicate_positions,
                                "first_positions": first_indices,
                                "source_anchors_exact": not anchor_errors,
                                "gold_matches_first_report_order": qa_row.get("reconstructed_gold") == expected_order})

    required_bindings = [freeze_path, admission_path, qa_path, family_path, mechanical_path,
                         request_freeze_path, preflight_path, coverage_path,
                         OUT / "requests/PROMPTS.jsonl", OUT / "requests/REQUESTS.jsonl",
                         OUT / "requests/DISPLAY_MAPS.json", OUT / "requests/FINITE_CHOICES.json"]
    verify_bindings(request_freeze.get("evidence_bindings", []), errors, "REQUEST_EVIDENCE_BINDING")
    verify_bindings(coverage.get("input_bindings", []), errors, "COVERAGE_INPUT_BINDING")
    file_bindings = [bind(p) for p in required_bindings]
    script_binding = bind(__file__)
    report = {
        "schema_version": "day3_supplement_closeout_preflight_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "scope": "Independent technical closeout check for the frozen finite supplement; not a model outcome or generalization test.",
        "accepted_questions": len(accepted_ids),
        "ordering_questions_checked": len(ordering_checks),
        "ordering_checks": ordering_checks,
        "request_question_count": request_freeze.get("question_count"),
        "request_display_variant_count": request_freeze.get("variant_count"),
        "request_count_per_model": request_freeze.get("request_count_per_model"),
        "prompt_tokens_min": preflight.get("prompt_tokens_min"),
        "prompt_tokens_max": preflight.get("prompt_tokens_max"),
        "length_holds": len(preflight.get("length_holds", [])),
        "solver_calls": 0,
        "training_runs": 0,
        "execution_permitted": request_freeze.get("execution_permitted"),
        "training_start_go": request_freeze.get("training_start_go"),
        "errors": errors,
        "script_binding": script_binding,
        "input_bindings": file_bindings,
        "historical_root_exposure": True,
        "unseen_root_claim": False,
    }
    output = OUT / "review/ORDERING_UNIQUENESS_PREFLIGHT.json"
    if output.exists():
        raise RuntimeError("Refusing to overwrite: " + str(output))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "accepted_questions", "ordering_questions_checked",
                                              "request_count_per_model", "errors"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
