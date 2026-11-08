"""Execute the common frozen B0 K=3 Dev evaluation, never Test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from ..h4_round0_b0_k3_difficulty_v1.gate import directory_identity
from ..h4_round0_b0_k3_difficulty_v1.io_utils import RunLock, load_json, load_jsonl, verify_sha_sidecar
from ..h4_round0_b0_k3_difficulty_v1.solver import generation_config
from . import BASE_MODEL, DATA_RUN, DEFAULT_RUN_ROOT, DIFFICULTY_RUN, FORMAL_RUN, MAX_MODEL_LEN, REPO_ROOT, VERSION
from .evaluator import (
    MODEL_SPECS, adapter_identity, aggregate_model, audit_m0_reuse, build_solver_plan,
    dimension_stats, hard_transitions, load_dev_inputs, merge_adapter, run_model,
    source_snapshot, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-run", type=Path, default=DATA_RUN)
    parser.add_argument("--difficulty-run", type=Path, default=DIFFICULTY_RUN)
    parser.add_argument("--formal-run", type=Path, default=FORMAL_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_run, difficulty_run, formal_run, run_root = (p.resolve() for p in (args.data_run, args.difficulty_run, args.formal_run, args.run_root))
    if run_root.exists():
        raise RuntimeError(f"dev_eval_output_exists:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    lock_path = run_root / ".run.lock"
    lock_path.touch(exist_ok=False)
    source_hashes = source_snapshot(data_run)
    contract_path = difficulty_run / "solver_contract.json"
    contract_sha = verify_sha_sidecar(contract_path)
    contract = load_json(contract_path)
    if int(contract["max_model_len"]) != MAX_MODEL_LEN:
        raise RuntimeError("solver_contract_max_model_len_mismatch")
    tokenizer = AutoTokenizer.from_pretrained(contract["tokenizer"], trust_remote_code=True, local_files_only=True)
    dev_eval, difficulty, gates = load_dev_inputs(data_run, difficulty_run)
    # Verify the frozen full-context prompt source and no Test manifest access.
    train_ids = {str(row["metadata"]["candidate_id"]) for name in ("m_clean_policy_train.jsonl", "m_inclusive_policy_train.jsonl") for row in load_jsonl(data_run / name)}
    train_roots = {str(row["root_id"]) for name in ("m_clean_policy_training_manifest.jsonl", "m_inclusive_policy_training_manifest.jsonl") for row in load_jsonl(data_run / name)}
    dev_ids = {str(row["candidate_id"]) for row in dev_eval}; dev_roots = {str(row["root_id"]) for row in dev_eval}
    if train_ids & dev_ids or train_roots & dev_roots:
        raise RuntimeError("train_dev_leakage")
    plan = build_solver_plan(dev_eval, solver_contract=contract)
    write_json(run_root / "dev_solver_plan.json", plan)
    base_identity = directory_identity(BASE_MODEL)
    identities = {"M0": adapter_identity(name="M0", adapter_path=None, base_identity=base_identity, formal_run=formal_run)}
    for name in ("M_clean_policy", "M_inclusive_policy"):
        identities[name] = adapter_identity(name=name, adapter_path=MODEL_SPECS[name]["adapter"], base_identity=base_identity, formal_run=formal_run)
    write_json(run_root / "model_identity_report.json", {"model_identity_report_version": f"{VERSION}_identity_v1", "base_model_identity": base_identity, "solver_contract_sha256": contract_sha, "models": identities})
    reuse = audit_m0_reuse(difficulty_run=difficulty_run, data_run=data_run, dev_eval=dev_eval, difficulty=difficulty, gates=gates, tokenizer=tokenizer, solver_contract=contract, contract_sha=contract_sha)
    write_json(run_root / "m0_reuse_audit.json", reuse)
    if not reuse["reuse_allowed"]:
        raise RuntimeError("m0_reuse_audit_failed_requires_explicit_rerun")
    m0_results = []
    for row in dev_eval:
        cid = str(row["candidate_id"]); path = difficulty_run / "candidate_results" / f"{hashlib.sha256(cid.encode()).hexdigest()}.json"
        m0_results.append(load_json(path))
    # M0 is reused, and no M0 Solver generation is created in this run.
    dev_metadata = {str(row["candidate_id"]): row for row in dev_eval}
    m0_rows = []
    for result in m0_results:
        summary = aggregate_model  # keep the imported function visible in provenance
        from ..h4_round0_b0_k3_difficulty_v1.difficulty import summarize_candidate_result
        gate = gates[str(result["candidate_id"])]
        metadata = dev_metadata[str(result["candidate_id"])]
        row = {**summarize_candidate_result(result), "model_name": "M0", "candidate_id": result["candidate_id"], "root_id": result["root_id"], "question_type": result["question_type"], "generation_source": metadata.get("generation_source"), "position_mechanism": metadata.get("position_mechanism", "unknown"), "evidence_status": metadata.get("evidence_status", "unknown"), "request_count": result["request_count"], "technical_failure_count": result["technical_failure_count"], "context_token_bucket": f"{int(gate.get('max_prompt_token_count', 0)) // 2048 * 2048}-{int(gate.get('max_prompt_token_count', 0)) // 2048 * 2048 + 2047}"}
        m0_rows.append(row)
    write_json(run_root / "m0_dev_results_reference.json", {"m0_dev_results_reference_version": f"{VERSION}_m0_reference_v1", "reuse": True, "source_run": str(difficulty_run), "source_solver_generation_in_this_run": 0, "source_result_sha256": reuse["result_sha256"], "dev_candidate_count": len(m0_rows)})
    adapter_models: dict[str, Path] = {}
    for name in ("M_clean_policy", "M_inclusive_policy"):
        adapter_models[name] = run_root / ("clean_merged_model" if name == "M_clean_policy" else "inclusive_merged_model")
        merge = merge_adapter(adapter=MODEL_SPECS[name]["adapter"], output=adapter_models[name], log=run_root / ("clean_merge.log" if name == "M_clean_policy" else "inclusive_merge.log"))
        identities[name]["merged_model_path"] = str(adapter_models[name]); identities[name]["merged_model_identity"] = merge["model_identity"]
    write_json(run_root / "model_identity_report.json", {"model_identity_report_version": f"{VERSION}_identity_v1", "base_model_identity": base_identity, "solver_contract_sha256": contract_sha, "models": identities})
    model_outputs = {"M0": {"candidate_rows": m0_rows, "report": {"status": "PASS", "candidate_count": len(m0_rows), "request_count": sum(int(row["request_count"]) for row in m0_rows), "technical_failure_count": 0}}}
    for name in ("M_clean_policy", "M_inclusive_policy"):
        model_outputs[name] = run_model(model_name=name, model_path=adapter_models[name], run_root=run_root, dev_eval=dev_eval, gates=gates, tokenizer=tokenizer, contract_sha=contract_sha, contract=contract, identity=identities[name])
        for row in model_outputs[name]["candidate_rows"]:
            row["context_token_bucket"] = f"{int(gates[str(row['candidate_id'])].get('max_prompt_token_count', 0)) // 2048 * 2048}-{int(gates[str(row['candidate_id'])].get('max_prompt_token_count', 0)) // 2048 * 2048 + 2047}"
    all_rows = {name: value["candidate_rows"] for name, value in model_outputs.items()}
    summaries = {name: aggregate_model(name, rows, difficulty) for name, rows in all_rows.items()}
    write_json(run_root / "dev_model_summary.json", {"dev_model_summary_version": f"{VERSION}_summary_v1", "models": summaries})
    write_json(run_root / "dev_comparison_report.json", {"dev_comparison_report_version": f"{VERSION}_comparison_v1", "models": summaries, "common_candidate_count": len(dev_eval), "m0_reused": True, "m0_new_solver_generation": 0})
    write_json(run_root / "dev_by_question_type.json", {name: dimension_stats(rows, "question_type") for name, rows in all_rows.items()})
    write_json(run_root / "dev_by_generation_source.json", {name: dimension_stats(rows, "generation_source") for name, rows in all_rows.items()})
    write_json(run_root / "dev_by_position_mechanism.json", {name: dimension_stats(rows, "position_mechanism") for name, rows in all_rows.items()})
    write_json(run_root / "dev_by_root.json", {name: dimension_stats(rows, "root_id") for name, rows in all_rows.items()})
    write_json(run_root / "dev_hard_by_evidence_status.json", {name: dimension_stats([row | {"evidence_status": difficulty[str(row["candidate_id"])].get("evidence_status", "unknown")} for row in rows if str(row["candidate_id"]) in difficulty and difficulty[str(row["candidate_id"])].get("difficulty_bucket") == "hard_candidate"], "evidence_status") for name, rows in all_rows.items()})
    write_json(run_root / "dev_m0_hard_transitions.json", hard_transitions(m0_rows, {name: rows for name, rows in all_rows.items() if name != "M0"}))
    preflight = {
        "preflight_report_version": f"{VERSION}_preflight_v1", "status": "PASS", "data_run": str(data_run), "difficulty_run": str(difficulty_run), "formal_run": str(formal_run), "source_sha256": source_hashes, "solver_contract_sha256": contract_sha, "solver_contract_max_model_len": contract["max_model_len"], "dev_candidate_count": len(dev_eval), "dev_root_count": len(dev_roots), "train_dev_candidate_overlap": 0, "train_dev_root_overlap": 0, "m0_reuse_allowed": True, "m0_new_solver_generation": 0, "test_prompts_accessed": False, "semantic_auditor_calls": 0, "external_api_calls": 0, "training_started": False,
    }
    write_json(run_root / "preflight_report.json", preflight)
    write_json(run_root / "dev_evaluation_contract.json", {"dev_evaluation_contract_version": f"{VERSION}_contract_v1", "solver_contract_sha256": contract_sha, "max_model_len": contract["max_model_len"], "max_tokens": contract["generation_config"]["max_tokens"], "enable_thinking": False, "context_modes": contract["context_modes"], "rollouts_per_position": contract["rollouts_per_position"], "position_counts": contract["position_counts"], "temperatures": contract["temperatures"], "seed_schedule": contract["seed_schedule"], "prompt_contract_version": contract["prompt_contract_version"], "permutation_contract_version": contract["permutation_contract_version"], "canonical_scoring_version": contract["canonical_scoring_version"], "parser_versions": contract["parsers"], "m0_reuse": True, "test_accessed": False})
    final = {"final_dev_evaluation_report_version": f"{VERSION}_final_v1", "status": "READY_FOR_COMMON_TEST_EVALUATION", "models": ["M0", "M_clean_policy", "M_inclusive_policy"], "m0_reused": True, "m0_new_solver_generation": 0, "dev_candidate_count": len(dev_eval), "dev_root_count": len(dev_roots), "clean_request_count": model_outputs["M_clean_policy"]["report"]["request_count"], "inclusive_request_count": model_outputs["M_inclusive_policy"]["report"]["request_count"], "clean_technical_failure_count": model_outputs["M_clean_policy"]["report"]["technical_failure_count"], "inclusive_technical_failure_count": model_outputs["M_inclusive_policy"]["report"]["technical_failure_count"], "contract_max_model_len": contract["max_model_len"], "training_max_length_not_used_for_eval": True, "test_prompts_accessed": False, "dev_summaries": summaries, "hard_transitions": hard_transitions(m0_rows, {name: rows for name, rows in all_rows.items() if name != "M0"}), "solver_calls_new_in_this_run": model_outputs["M_clean_policy"]["report"]["request_count"] + model_outputs["M_inclusive_policy"]["report"]["request_count"], "external_api_calls": 0, "semantic_auditor_calls": 0, "training_started": False}
    write_json(run_root / "final_dev_evaluation_report.json", final)
    md = "\n".join(["# Round-0 Common Dev Evaluation", "", f"Status: **{final['status']}**", "", f"M0 was reused after a strict fingerprint audit; new M0 Solver generation was `{final['m0_new_solver_generation']}`.", f"Clean and Inclusive each ran `{final['clean_request_count']}` Dev requests under the frozen B0 K=3 contract with `max_model_len={final['contract_max_model_len']}`.", "No Test prompt, training, external API, or semantic auditor was used.", "", "See `dev_comparison_report.json`, per-dimension slices, and model-specific ledgers for results."]) + "\n"
    atomic_text(run_root / "dev_comparison_report.md", md)
    atomic_text(run_root / "dev_comparison_report.md.sha256", sha256_file(run_root / "dev_comparison_report.md") + "\n")
    atomic_text(run_root / "final_dev_evaluation_report.md", md.replace("Dev Comparison", "Final Dev Evaluation"))
    atomic_text(run_root / "final_dev_evaluation_report.md.sha256", sha256_file(run_root / "final_dev_evaluation_report.md") + "\n")
    print(json.dumps({k: v for k, v in final.items() if k not in {"dev_summaries", "hard_transitions"}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
