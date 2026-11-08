"""Bind the existing, separate evaluation cohorts before any new training."""
import json
from pathlib import Path

from build_schedule import bind, read, lines, verify_ref, save

OUT = Path(__file__).resolve().parent
WEEK = OUT.parent


def main():
    regression_path = WEEK / "day3_mainline_readiness_v1/common_eval/COMMON_REGRESSION_REQUEST_FREEZE.json"
    old4_path = WEEK / "day3_eval_clearance_v1/requests/REQUEST_FREEZE.json"
    supplement_path = WEEK / "day3_eval_supplement_v2/requests/REQUEST_FREEZE.json"
    regression, old4, supplement = map(read, [regression_path, old4_path, supplement_path])
    public_regression = lines(WEEK / "day3_mainline_readiness_v1/common_eval/evidence/ADMITTED_PUBLIC.jsonl")
    old_admission = read(WEEK / "day3_eval_clearance_v1/content/CONTENT_ADMISSION.json")
    public4 = [read(r["public_question"]["path"]) for r in old_admission["items"] if r["status"] == "CONTENT_ACCEPTED"]
    sup_admission = read(WEEK / "day3_eval_supplement_v2/CONTENT_ADMISSION.json")
    public12 = [read(r["public_question"]["path"]) for r in sup_admission["records"] if r["status"] == "CONTENT_ACCEPTED"]
    train_rows = {a: lines(OUT / f"TRAIN_POOL_{a}.jsonl") for a in ["A", "B"]}
    training_roots = {r["root_id"] for rows in train_rows.values() for r in rows}
    cohorts = []
    for name, freeze_path, freeze, public, questions, requests, role in [
        ("historical115", regression_path, regression, public_regression, 115, 1854, "HISTORICAL_REGRESSION_ONLY"),
        ("previous4MC", old4_path, old4, public4, 4, 72, "PREVIOUS_SEPARATELY_FROZEN_MC_SUPPLEMENT"),
        ("supplement12", supplement_path, supplement, public12, 12, 180, "BOUNDED_MULTITYPE_PREPARATION_ON_EXPOSED_TEST_ROOTS")
    ]:
        assert len(public) == questions
        refs = freeze.get("evidence_bindings", [])
        if name == "previous4MC":
            refs = [freeze["requests"], freeze["display_maps"]]
        for ref in refs:
            verify_ref(ref)
        root_ids = sorted({r["root_id"] for r in public})
        assert not (set(root_ids) & training_roots), "Train/evaluation root overlap"
        cohorts.append({"cohort_id": name, "role": role, "request_freeze": bind(freeze_path),
                        "question_count": questions, "requests_per_model": requests,
                        "root_ids": root_ids, "question_ids": [r["candidate_id"] for r in public],
                        "request_evidence_bindings": refs, "train_root_overlap": [],
                        "separate_denominator": True, "historically_exposed": True})
    save("EVALUATION_LOCK.json", {
        "schema_version": "day4_preserved_common_evaluation_lock_v1",
        "status": "EXISTING_COHORTS_AND_REQUEST_BYTES_LOCKED_NO_EXECUTION",
        "cohorts": cohorts,
        "checkpoint_selection": "fixed final optimizer step, no DEV/TEST selection",
        "dev": "report historical regression aggregates only; no feedback to Challenger",
        "test": "evaluate only final frozen endpoints after model hashes are bound; no interim execution",
        "primary_historical_controls": "regression primary99 (63 MC gold2-5 and36 ordering);16 gold6 controls separate",
        "model_lineage": {"S0": "existing day1 bound M-clean", "A1": "independently trained frozen A pool from S0",
                          "B1": "independently trained frozen B pool from S0"},
        "future_checkpoint_binding": "Actual adapter/merged model hashes must be added in a separate execution manifest after training; no invented future hashes",
        "evaluation_runtime": {"python": "/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python",
                               "torch": "2.11.0", "transformers": "5.9.0", "vllm": "0.21.0",
                               "adapter": bind(WEEK / "day2/solver/adapter.py"),
                               "new_execution_wrapper": "separate per-model public execution manifest required before Solver, no new Solver in D4"},
        "unchanged_sampling_policy": bind(WEEK / "day3_common_evaluation_v1/protocol/REQUEST_POLICY_V1.json"),
        "training_pools": {a: bind(OUT / f"TRAIN_POOL_{a}.jsonl") for a in ["A", "B"]},
        "required_boundary_reviews": [bind(WEEK / "day3_mainline_readiness_v1/common_eval/FAMILY_DEDUP_LEDGER.json"),
                                      bind(WEEK / "day3_eval_clearance_v1/FAMILY_SELECTION.json"),
                                      bind(WEEK / "day3_eval_supplement_v2/review/FAMILY_BOUNDARY_AUDIT.json")],
        "content_review_limit": "inherits bounded upstream necessary-fact/family review; root non-overlap alone is not proof of fact independence",
        "new_solver_calls": 0, "execution_permitted": False, "no_unseen_root_claim": True,
        "script": bind(__file__)})


if __name__ == "__main__":
    main()
