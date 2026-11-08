"""Run the frozen Test Solver evaluation for both trained adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from ..h4_round0_b0_k3_difficulty_v1.difficulty import summarize_candidate_result
from ..h4_round0_b0_k3_difficulty_v1.gate import directory_identity
from ..h4_round0_b0_k3_difficulty_v1.io_utils import RunLock, load_json, load_jsonl, verify_sha_sidecar
from ..h4_round0_common_dev_eval_v1 import evaluator
from ..h4_round0_common_dev_eval_v1.evaluator import MODEL_SPECS, adapter_identity, merge_adapter, run_model, write_json
from ..h4_ordering_full_review_v1.offline_rescore import aggregate, corrected_gold, load_review, rescore_result
from . import DATA_RUN, DEFAULT_RUN_ROOT, DEV_RUN, DIFFICULTY_RUN, EXPECTED_TEST, FORMAL_RUN, VERSION


def _candidate_rows(manifest: list[dict]) -> dict[str, dict]:
    return {str(row["candidate_id"]): row for row in manifest}


def _m0_results(candidates: dict[str, dict]) -> dict[str, dict]:
    result = {}
    for path in (DIFFICULTY_RUN / "candidate_results").glob("*.json"):
        row = load_json(path)
        cid = str(row.get("candidate_id"))
        if cid in candidates:
            result[cid] = row
    if set(result) != set(candidates):
        raise RuntimeError(f"m0_test_result_identity_mismatch:{len(result)}:{len(candidates)}")
    return result


def _rescored_rows(results: dict[str, dict], candidates: dict[str, dict], review: dict[str, dict]) -> list[dict]:
    rows = []
    for cid in sorted(candidates):
        candidate = candidates[cid]
        gold, source, inclusion = corrected_gold(candidate, review)
        if inclusion == "exclude":
            continue
        payload = candidate.get("candidate") or candidate
        rows.append({
            "candidate_id": cid,
            "root_id": candidate["root_id"],
            "question_type": payload["question_type"],
            "generation_source": payload.get("generation_source"),
            "original_gold": payload.get("gold"),
            "rescored_gold": gold,
            "gold_source": source,
            "review_gold_status": review.get(cid, {}).get("gold_status"),
            "review_evidence_status": review.get(cid, {}).get("evidence_status"),
            "score": rescore_result(results[cid], candidate, gold),
        })
    return rows


def main() -> int:
    run_root = Path(os.environ.get("H4_TEST_EVAL_RUN_ROOT", str(DEFAULT_RUN_ROOT))).resolve()
    if run_root.exists():
        raise RuntimeError(f"test_eval_output_exists:{run_root}")
    run_root.mkdir(parents=True)
    data_run, difficulty_run, formal_run = DATA_RUN.resolve(), DIFFICULTY_RUN.resolve(), FORMAL_RUN.resolve()
    manifest = load_jsonl(data_run / "test_full_eval_manifest.jsonl")
    if len(manifest) != EXPECTED_TEST or len(_candidate_rows(manifest)) != EXPECTED_TEST:
        raise RuntimeError(f"test_manifest_count:{len(manifest)}")
    candidates = _candidate_rows(manifest)
    gates = {str(row["candidate_id"]): row for row in load_jsonl(difficulty_run / "program_gate_results.jsonl") if row.get("candidate_id") in candidates}
    if set(gates) != set(candidates) or any(row.get("gate_status") != "PROGRAM_PASS" for row in gates.values()):
        raise RuntimeError("test_gate_identity_or_status_mismatch")
    contract_path = difficulty_run / "solver_contract.json"
    contract_sha = verify_sha_sidecar(contract_path)
    contract = load_json(contract_path)
    tokenizer = AutoTokenizer.from_pretrained(contract["tokenizer"], trust_remote_code=True, local_files_only=True)
    base_identity = directory_identity(Path("/root/autodl-tmp/Qwen3-4B"))
    identities = {}
    models = {}
    for name in ("M_clean_policy", "M_inclusive_policy"):
        models[name] = DEV_RUN / ("clean_merged_model" if name == "M_clean_policy" else "inclusive_merged_model")
        identities[name] = adapter_identity(name=name, adapter_path=MODEL_SPECS[name]["adapter"], base_identity=base_identity, formal_run=formal_run)
        identities[name]["merged_model_path"] = str(models[name])
        identities[name]["merged_model_identity"] = directory_identity(models[name])
    write_json(run_root / "test_solver_plan.json", {
        "version": VERSION + "_plan_v1", "split": "test", "candidate_count": EXPECTED_TEST,
        "root_count": len({str(row["root_id"]) for row in manifest}),
        "question_type_counts": {q: sum((row.get("candidate") or row).get("question_type") == q for row in manifest) for q in ("single_choice", "multi_select", "ordering")},
        "requests_per_candidate": {"single_choice": 24, "multi_select": 30, "ordering": 24},
        "planned_request_count_per_model": sum({"single_choice": 24, "multi_select": 30, "ordering": 24}[(row.get("candidate") or row)["question_type"]] for row in manifest),
        "m0_new_request_count": 0, "solver_contract_sha256": contract_sha,
    })
    write_json(run_root / "model_identity_report.json", {"version": VERSION + "_identity_v1", "models": identities, "solver_contract_sha256": contract_sha})
    outputs = {}
    for name in ("M_clean_policy", "M_inclusive_policy"):
        with RunLock(run_root):
            outputs[name] = run_model(model_name=name, model_path=models[name], run_root=run_root, dev_eval=manifest, gates=gates, tokenizer=tokenizer, contract_sha=contract_sha, contract=contract, identity=identities[name], split_name="test", expected_candidate_count=EXPECTED_TEST)
        print(json.dumps({"model": name, **{k: outputs[name]["report"][k] for k in ("status", "candidate_count", "request_count", "technical_failure_count")}}), flush=True)
    review = load_review(Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_dev_test_ordering_review_v1_live_1/final_manual_review.tsv"))
    m0 = _m0_results(candidates)
    all_results = {"M0": m0, **{name: {str(row["candidate_id"]): row for row in outputs[name]["results"]} for name in outputs}}
    summaries = {}
    for name, results in all_results.items():
        rows = _rescored_rows(results, candidates, review)
        scope = run_root / ("m0_test" if name == "M0" else "clean_test" if name == "M_clean_policy" else "inclusive_test")
        scope.mkdir(parents=True, exist_ok=True)
        (scope / "rescored_rows.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        summaries[name] = aggregate(rows)
    write_json(run_root / "test_model_summary.json", {"version": VERSION + "_summary_v1", "models": summaries})
    final = {"version": VERSION + "_final_v1", "status": "COMPLETE", "models": ["M0", "M_clean_policy", "M_inclusive_policy"], "test_candidate_count": EXPECTED_TEST, "test_root_count": len({str(row["root_id"]) for row in manifest}), "m0_reused": True, "m0_new_solver_generation": 0, "clean_request_count": outputs["M_clean_policy"]["report"]["request_count"], "inclusive_request_count": outputs["M_inclusive_policy"]["report"]["request_count"], "clean_technical_failure_count": outputs["M_clean_policy"]["report"]["technical_failure_count"], "inclusive_technical_failure_count": outputs["M_inclusive_policy"]["report"]["technical_failure_count"], "solver_calls_new_in_this_run": outputs["M_clean_policy"]["report"]["request_count"] + outputs["M_inclusive_policy"]["report"]["request_count"], "external_api_calls": 0, "semantic_auditor_calls": 0, "training_started": False, "test_prompts_accessed": True, "excluded_after_manual_review": {"m0": EXPECTED_TEST - summaries["M0"]["candidate_count"], "clean": EXPECTED_TEST - summaries["M_clean_policy"]["candidate_count"], "inclusive": EXPECTED_TEST - summaries["M_inclusive_policy"]["candidate_count"]}, "summaries": summaries}
    write_json(run_root / "final_test_evaluation_report.json", final)
    print(json.dumps({k: final[k] for k in ("status", "test_candidate_count", "test_root_count", "clean_request_count", "inclusive_request_count", "clean_technical_failure_count", "inclusive_technical_failure_count", "excluded_after_manual_review")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
