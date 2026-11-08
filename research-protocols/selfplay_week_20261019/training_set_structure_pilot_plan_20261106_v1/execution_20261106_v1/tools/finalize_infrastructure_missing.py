#!/usr/bin/env python3
"""Read-only reconciliation of the failed attempt; never score or run inference."""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib
import json

E = Path(__file__).resolve().parents[1]
S = E / "solver"
V = S / "evaluation"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def binding(path):
    return {"path": str(path), "sha256": digest(path)}


def write_new(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def summarize(rows):
    return {
        "planned_requests": len(rows), "returned_requests": 0,
        "executed_inference_requests": 0, "infrastructure_missing_requests": len(rows),
        "invalid_model_answers": None, "exact_accuracy": None,
        "planned_denominator_accuracy": None, "J": None,
        "directed_pair_precision": None, "directed_pair_recall": None,
        "directed_pair_F1": None, "feedback_error_patterns": None,
        "status": "NO_BEHAVIORAL_OBSERVATIONS_INFRASTRUCTURE_FAILURE",
    }


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [dict(zip(keys, key), **summarize(values))
            for key, values in sorted(groups.items())]


def main():
    plan_path = V / "PLAN_FREEZE.json"
    seal_path = S / "INPUT_REVIEW_SEAL.json"
    plan, seal = read(plan_path), read(seal_path)
    expected = {}
    for b in plan["files"] + seal["files"]:
        old = expected.setdefault(b["path"], b["sha256"])
        assert old == b["sha256"], ("INCONSISTENT_BINDING", b["path"])
    for path, sha in expected.items():
        assert digest(Path(path)) == sha, ("FROZEN_INPUT_CHANGED", path)
    req_path = V / "REQUESTS.json"
    raw_path = V / "RAW_ANSWERS.json"
    requests, raw = read(req_path), read(raw_path)
    req_ids = [x["request_id"] for x in requests]
    raw_ids = [x["request_id"] for x in raw]
    assert len(set(req_ids)) == len(req_ids) == 1008
    assert len(set(raw_ids)) == len(raw_ids) == 1008
    assert set(req_ids) == set(raw_ids)
    assert all(x["status"] == "MISSING" and x["automatic_retry"] is False for x in raw)
    raw_files = [p for p in (V / "raw").rglob("*") if p.is_file()]
    assert not raw_files, "RETURNED_RAW_EXISTS_REQUIRES_DIFFERENT_ANALYSIS"
    assert not (V / "SCORES.json").exists(), "SCORES_ALREADY_EXIST"
    interruption = read(V / "INTERRUPTED.json")
    assert interruption["error"]["type"] == "ImportError"
    assert "StructuredOutputsParams" in interruption["error"]["message"]
    stdout = (S / "RUN_STDOUT.log").read_text(encoding="utf-8")
    assert "ZeroDivisionError: division by zero" in stdout
    attempt = read(S / "RUN_ATTEMPT.json")
    assert attempt["plan_freeze"]["sha256"] == digest(plan_path)
    assert plan["planned_calls"] == len(requests)
    preflight = read(V / "PREFLIGHT.json")
    assert preflight["request_count"] == len(requests)
    assert preflight["requests_sha256"] == digest(req_path)
    assert len(preflight["holds"]) == 2
    assert {x["candidate_id"] for x in preflight["holds"]} == {"Q_24", "Q_29"}
    out = {
        "schema_version": "infrastructure_missing_summary_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "namespace": plan["namespace"], "final": True,
        "status": "SOLVER_ATTEMPT_FAILED_BEFORE_INFERENCE_SCORING_ALSO_FAILED",
        "solver_stage_complete": False, "process_exit_code": 1,
        "overall": summarize(requests),
        "counts_by_context": dict(Counter(x["context_mode"] for x in requests)),
        "eligible_first_drafts": len(seal["items"]),
        "planned_candidate_count": len({x["candidate_id"] for x in requests}),
        "evaluation_holds": preflight["holds"],
        "holds_change_quality_status": False,
        "failure": {
            "inference_stage": interruption,
            "scoring_stage": {"type": "ZeroDivisionError", "message": "division by zero",
                "cause": "eager dict.get fallback invokes invalid() with empty display map; inherited completeness divides by zero"},
            "inference_execution_basis": "Import failed before inherited LLM construction/generation; no per-request raw files returned. Missing records are collector placeholders, not model outputs.",
        },
        "by_group": grouped(requests, ["group"]),
        "by_root": grouped(requests, ["root_id"]),
        "by_item": grouped(requests, ["candidate_id", "root_id", "group", "question_type"]),
        "by_type": grouped(requests, ["question_type"]),
        "by_context": grouped(requests, ["context_mode"]),
        "by_group_type_context": grouped(requests, ["group", "question_type", "context_mode"]),
        "interpretation": {
            "missing_is_model_error": False,
            "no_context_shortcut_analysis": "UNAVAILABLE_NO_BEHAVIORAL_OBSERVATIONS",
            "card_pattern_solver_error_alignment": "UNAVAILABLE_NO_BEHAVIORAL_OBSERVATIONS",
            "model_or_feedback_failure_claim_supported": False,
            "resumed_or_retried": False,
            "future_inference_requires_explicit_protocol_change": True,
            "this_artifact_is_scores": False,
        },
        "integrity": {"plan_and_seal_unique_bound_files_checked": len(expected),
                      "all_bound_bytes_unchanged": True, "returned_raw_file_count": 0,
                      "scores_file_present": False},
        "source_artifacts": [binding(p) for p in [plan_path, seal_path, req_path, raw_path,
            V / "PREFLIGHT.json", V / "INTERRUPTED.json", S / "RUN_STDOUT.log",
            S / "RUN_ATTEMPT.json", S / "ATTEMPT_LEDGER.json", Path(__file__)]],
    }
    output = V / "INFRASTRUCTURE_MISSING_SUMMARY.json"
    write_new(output, out)
    print(json.dumps({"output": str(output), "sha256": digest(output),
        "status": out["status"], "planned": len(requests), "returned": 0,
        "missing": len(raw), "all_metrics": None,
        "bound_files_checked": len(expected)}))


if __name__ == "__main__":
    main()
