"""Emit the preregistered finite Day 3 coverage GO/NO_GO decision."""
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent


def bind(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def read(path):
    return json.loads(Path(path).read_text())


def verify(ref):
    if bind(ref["path"]) != ref:
        raise RuntimeError("INPUT_BINDING_MISMATCH: " + ref["path"])


def write_new(path, obj):
    path = Path(path)
    if path.exists():
        raise RuntimeError("Refusing to overwrite: " + str(path))
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    freeze = read(OUT / "EXECUTION_FREEZE.json")
    admission = read(OUT / "CONTENT_ADMISSION.json")
    requests = read(OUT / "requests/REQUEST_FREEZE.json")
    preflight = read(OUT / "requests/CPU_PREFLIGHT.json")
    assert admission.get("status") == "CONTENT_ADMISSION_FINAL"
    assert requests.get("status") == "CONTENT_REQUEST_BYTES_FROZEN_FUTURE_MODEL_RUNTIME_PENDING"
    assert requests.get("execution_permitted") is False and requests.get("training_start_go") is False
    assert requests.get("content_admission") == bind(OUT / "CONTENT_ADMISSION.json")
    assert requests.get("execution_freeze") == bind(OUT / "EXECUTION_FREEZE.json")
    for ref in requests.get("evidence_bindings", []):
        verify(ref)
    assert preflight.get("status") == "PASS" and preflight.get("zero_truncation") is True
    assert preflight.get("length_holds") == []
    prompt_rows = [json.loads(line) for line in (OUT / "requests/PROMPTS.jsonl").read_text().splitlines() if line.strip()]
    request_rows = [json.loads(line) for line in (OUT / "requests/REQUESTS.jsonl").read_text().splitlines() if line.strip()]
    assert requests.get("variant_count") == len(prompt_rows)
    assert requests.get("request_count_per_model") == len(request_rows)
    assert len({r["candidate_id"] for r in prompt_rows}) == requests.get("question_count")
    assert {r["prompt_id"] for r in request_rows} <= {r["prompt_id"] for r in prompt_rows}
    assert all(r.get("execution_status") == "NOT_EXECUTED" for r in request_rows)
    held = {r["candidate_id"] for r in preflight["length_holds"]}
    rows = [r for r in admission["records"] if r["status"] == "CONTENT_ACCEPTED" and r["candidate_id"] not in held]
    expected_ids = {r["candidate_id"] for r in rows}
    request_ids = {r["candidate_id"] for r in request_rows}
    per_root = defaultdict(lambda: {"multi_select": 0, "ordering": 0, "mc_relation_classes": set(), "ordering_arities": set()})
    for r in rows:
        v = per_root[r["root_id"]]
        v[r["question_type"]] += 1
        if r["question_type"] == "multi_select":
            v["mc_relation_classes"].add(r["relation_class"])
        else:
            v["ordering_arities"].add(r["arity"])
    requirements = freeze["admission"]["minimum_for_scoped_preparation_GO"]
    root_ids = [r["root_id"] for r in freeze["roots"]]
    gates = {
        "root_count_with_both_types": sum(v["multi_select"] > 0 and v["ordering"] > 0 for v in per_root.values()) >= requirements["new_roots_with_both_types"],
        "each_root_mc_minimum": all(per_root[root]["multi_select"] >= requirements["MC_per_new_root"] for root in root_ids),
        "each_root_ordering_minimum": all(per_root[root]["ordering"] >= requirements["ordering_per_new_root"] for root in root_ids),
        "each_root_mc_relation_class_minimum": all(len(per_root[root]["mc_relation_classes"]) >= requirements["MC_relation_classes"] for root in root_ids),
        "each_root_ordering_arity_minimum": all(len(per_root[root]["ordering_arities"]) >= requirements["ordering_arities"] for root in root_ids),
        "requests_present_for_all_accepted_unheld": requests["question_count"] == len(rows) and request_ids == expected_ids,
        "exact_prompt_candidate_set": {r["candidate_id"] for r in prompt_rows} == expected_ids,
        "request_prompt_hashes_match": all(
            r["prompt_sha256"] == next(p["prompt_sha256"] for p in prompt_rows if p["prompt_id"] == r["prompt_id"])
            for r in request_rows),
        "synthetic_roundtrip_preflight_pass": preflight.get("synthetic_roundtrip_all_types_arities") is True and preflight.get("synthetic_roundtrip_cases", 0) > 0,
        "no_length_holds": not held,
    }
    evidence = {
        "schema_version": "day3_supplement_scoped_coverage_decision_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SCOPED_PREPARATION_GO" if all(gates.values()) else "SCOPED_PREPARATION_NO_GO",
        "scope": "Finite prospective coverage only; both roots are historically exposed heldout TEST roots. No unseen-root or effect claim.",
        "requirements": requirements,
        "counts": {root: {"multi_select": per_root[root]["multi_select"], "ordering": per_root[root]["ordering"],
                           "mc_relation_classes": sorted(per_root[root]["mc_relation_classes"]),
                           "ordering_arities": sorted(per_root[root]["ordering_arities"])} for root in root_ids},
        "length_hold_ids": sorted(held), "gates": gates,
        "all_gates_pass": all(gates.values()),
        "input_bindings": [bind(p) for p in [OUT / "EXECUTION_FREEZE.json", OUT / "CONTENT_ADMISSION.json",
                                                OUT / "requests/REQUEST_FREEZE.json", OUT / "requests/CPU_PREFLIGHT.json"]],
        "new_solver_calls": 0, "new_training_runs": 0,
        "claims_forbidden": ["unseen_or_pristine_root_generalization", "causal_feedback_benefit", "training_benefit", "statistically_reliable_multitype_effect"]
    }
    write_new(OUT / "SCOPED_COVERAGE_DECISION.json", evidence)
    print(json.dumps({"status": evidence["status"], "gates": gates, "counts": evidence["counts"]}))


if __name__ == "__main__":
    main()
