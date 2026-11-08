"""Append v70 closure, retaining the failed audit and its scoped correction."""
from pathlib import Path
import datetime
import hashlib
import json

R = Path(__file__).resolve().parent.parent
E = R / "execution"
A = E / "independent_audit"
C = A / "corrected" / "final_v3"
CACHE = {}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    path = path.resolve()
    if path not in CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        CACHE[path] = digest.hexdigest()
    return CACHE[path]


def bind(path):
    return {"path": str(path.resolve()), "sha256": sha(path)}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    receipt_path = A / "APPEND_ONLY_AUDIT_RECEIPT.json"
    complete_path = E / "TRIAL.COMPLETE.json"
    assert not receipt_path.exists() and not complete_path.exists()
    initial_path = A / "INDEPENDENT_RESULT_AUDIT.json"
    corrected_path = C / "INDEPENDENT_RESULT_AUDIT.json"
    summary = read(R / "RESULT_SUMMARY.json")
    initial, audit = read(initial_path), read(corrected_path)
    assert initial["status"] == "FAIL" and initial["failures"]
    assert audit["status"] == "PASS" and not audit["failures"]
    assert audit["result_summary_sha256"] == sha(R / "RESULT_SUMMARY.json")
    assert summary["training_started"] is False
    assert summary["automatic_followup"] is False
    for name in ("AUDIT_CORRECTION.json", "AUDIT_CORRECTION.md"):
        assert (A / name).is_file()

    ledgers, verified, binding_rows = {}, {}, 0
    pending = [R / "RESULT_BINDINGS.json"]
    while pending:
        path = pending.pop().resolve()
        if str(path) in ledgers:
            continue
        ledgers[str(path)] = sha(path)
        for row in read(path)["files"]:
            target = Path(row["path"])
            assert target.is_absolute(), row["path"]
            digest = sha(target)
            assert digest == row["sha256"], str(target)
            verified[str(target.resolve())] = digest
            binding_rows += 1
            if target.name == "RESULT_BINDINGS.json":
                pending.append(target)

    append_files = {R / "DECISION_RESULT.md", Path(__file__).resolve()}
    append_files.update(path for path in A.rglob("*")
                        if path.is_file() and path.suffix in {".json", ".md", ".py"}
                        and str(path.resolve()) not in verified)
    receipt = {
        "version": "v70_append_only_corrected_audit_receipt_v1",
        "completed_at_utc": now(), "status": "PASS",
        "original_result_bindings": bind(R / "RESULT_BINDINGS.json"),
        "existing_result_bound_files_verified_unchanged": len(read(R / "RESULT_BINDINGS.json")["files"]),
        "recursive_result_binding_ledgers": ledgers,
        "recursive_binding_rows_verified": binding_rows,
        "recursive_unique_bound_files_verified": len(verified),
        "initial_audit": bind(initial_path),
        "initial_audit_status": initial["status"],
        "initial_audit_findings_retained": len(initial["failures"]),
        "corrected_audit": bind(corrected_path),
        "correction": bind(A / "AUDIT_CORRECTION.json"),
        "all_audit_attempts_retained": [
            {**bind(path), "status": read(path)["status"],
             "checks_completed": read(path)["checks_completed"],
             "findings": len(read(path)["failures"])}
            for path in sorted(A.rglob("INDEPENDENT_RESULT_AUDIT.json"))
        ],
        "historical_environment_reproducibility_claimed": False,
        "append_only_files": [bind(path) for path in sorted(append_files)],
        "new_model_calls": 0, "frozen_files_modified": 0, "training_started": False,
    }
    write(receipt_path, receipt)
    complete = {
        "version": "v70_trial_complete_corrected_audit_v1",
        "completed_at_utc": now(), "status": "COMPLETE",
        "planned_families": summary["family_count"],
        "generation_http_attempts": summary["generation_cost"]["http_attempts"],
        "assembled": summary["assembled_count"],
        "content_usable": summary["content_usable_count"],
        "selected": summary["selected_count"], "added": summary["added_count"],
        "merged_candidate_pool": summary["merged_count"],
        "calls_per_model": summary["calls_per_model"],
        "raw_responses": 2 * summary["calls_per_model"],
        "independent_checks": audit["checks_completed"],
        "independent_differences": len(audit["failures"]),
        "initial_audit_status": initial["status"],
        "initial_audit_findings_retained": len(initial["failures"]),
        "corrected_audit": bind(corrected_path),
        "result_summary": bind(R / "RESULT_SUMMARY.json"),
        "original_result_bindings": bind(R / "RESULT_BINDINGS.json"),
        "append_only_receipt": bind(receipt_path),
        "training_allowed": False, "training_started": False, "automatic_followup": False,
        "required_count_coverage": summary["required_count_coverage"],
        "remaining_merged_count_gaps": summary["distributions"]["merged"]["missing_multiselect_gold_counts"],
    }
    write(complete_path, complete)
    print(json.dumps({"status": "COMPLETE", "binding_ledgers": len(ledgers),
                      "binding_rows": binding_rows, "unique_files": len(verified),
                      "selected": summary["selected_count"],
                      "independent_checks": audit["checks_completed"]}))


if __name__ == "__main__":
    main()
