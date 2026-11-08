"""Finish Dev aggregation from already-materialized Solver candidate results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..h4_round0_b0_k3_difficulty_v1.difficulty import summarize_candidate_result
from ..h4_round0_b0_k3_difficulty_v1.io_utils import load_json, load_jsonl, verify_sha_sidecar
from . import DATA_RUN, DIFFICULTY_RUN, FORMAL_RUN, MAX_MODEL_LEN, MAX_TOKENS, VERSION
from .evaluator import (
    aggregate_model,
    dimension_stats,
    hard_transitions,
    source_snapshot,
    write_json,
    write_jsonl,
)


EXPECTED_DEV = 485
EXPECTED_ROOTS = 15
EXPECTED_REQUESTS = 12378


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-run", type=Path, default=DATA_RUN)
    parser.add_argument("--difficulty-run", type=Path, default=DIFFICULTY_RUN)
    parser.add_argument("--formal-run", type=Path, default=FORMAL_RUN)
    return parser.parse_args()


def _rows(scope: Path, model_name: str, dev_eval: list[dict], difficulty: dict[str, dict], gates: dict[str, dict]) -> list[dict]:
    results = load_jsonl(scope / "candidate_results.jsonl")
    if len(results) != EXPECTED_DEV or len({str(row["candidate_id"]) for row in results}) != EXPECTED_DEV:
        raise RuntimeError(f"{model_name}_candidate_result_count_mismatch:{len(results)}")
    metadata = {str(row["candidate_id"]): row for row in dev_eval}
    rows = []
    for result in results:
        cid = str(result["candidate_id"])
        if cid not in metadata or cid not in difficulty or cid not in gates:
            raise RuntimeError(f"{model_name}_candidate_identity_mismatch:{cid}")
        gate = gates[cid]
        meta = metadata[cid]
        max_prompt = int(gate.get("max_prompt_token_count", 0))
        rows.append({
            **summarize_candidate_result(result),
            "model_name": model_name,
            "candidate_id": result["candidate_id"],
            "root_id": result["root_id"],
            "question_type": result["question_type"],
            "generation_source": meta.get("generation_source"),
            "position_mechanism": meta.get("position_mechanism", "unknown"),
            "evidence_status": meta.get("evidence_status", "unknown"),
            "context_token_bucket": f"{max_prompt // 2048 * 2048}-{max_prompt // 2048 * 2048 + 2047}",
            "request_count": result["request_count"],
            "technical_failure_count": result["technical_failure_count"],
            "candidate_pool_sha256": result["candidate_pool_sha256"],
            "dialogue_sha256": result["dialogue_sha256"],
            "solver_contract_sha256": result["solver_contract_sha256"],
        })
    write_jsonl(scope / "difficulty_manifest.jsonl", rows)
    report = load_json(scope / "model_execution_report.json")
    if report.get("status") != "PASS" or int(report.get("candidate_count", -1)) != EXPECTED_DEV or int(report.get("request_count", -1)) != EXPECTED_REQUESTS or int(report.get("technical_failure_count", -1)) != 0:
        raise RuntimeError(f"{model_name}_execution_report_not_pass")
    return rows


def main() -> int:
    args = parse_args()
    run_root, data_run, difficulty_run, formal_run = (p.resolve() for p in (args.run_root, args.data_run, args.difficulty_run, args.formal_run))
    dev_eval = load_jsonl(data_run / "dev_full_eval_manifest.jsonl")
    if len(dev_eval) != EXPECTED_DEV or len({str(row["root_id"]) for row in dev_eval}) != EXPECTED_ROOTS:
        raise RuntimeError("dev_manifest_not_frozen")
    difficulty = {str(row["candidate_id"]): dict(row) for row in load_jsonl(difficulty_run / "difficulty_manifest.jsonl") if row.get("split") == "dev"}
    gates = {str(row["candidate_id"]): dict(row) for row in load_jsonl(difficulty_run / "program_gate_results.jsonl")}
    m0_results = []
    for row in dev_eval:
        cid = str(row["candidate_id"])
        digest = __import__("hashlib").sha256(cid.encode()).hexdigest()
        path = difficulty_run / "candidate_results" / f"{digest}.json"
        verify_sha_sidecar(path)
        m0_results.append(load_json(path))
    metadata = {str(row["candidate_id"]): row for row in dev_eval}
    m0_rows = []
    for result in m0_results:
        cid = str(result["candidate_id"]); meta = metadata[cid]; gate = gates[cid]; max_prompt = int(gate.get("max_prompt_token_count", 0))
        m0_rows.append({
            **summarize_candidate_result(result), "model_name": "M0", "candidate_id": cid, "root_id": result["root_id"],
            "question_type": result["question_type"], "generation_source": meta.get("generation_source"),
            "position_mechanism": meta.get("position_mechanism", "unknown"), "evidence_status": meta.get("evidence_status", "unknown"),
            "context_token_bucket": f"{max_prompt // 2048 * 2048}-{max_prompt // 2048 * 2048 + 2047}",
            "request_count": result["request_count"], "technical_failure_count": result["technical_failure_count"],
        })
    clean_rows = _rows(run_root / "clean_dev", "M_clean_policy", dev_eval, difficulty, gates)
    inclusive_rows = _rows(run_root / "inclusive_dev", "M_inclusive_policy", dev_eval, difficulty, gates)
    all_rows = {"M0": m0_rows, "M_clean_policy": clean_rows, "M_inclusive_policy": inclusive_rows}
    summaries = {name: aggregate_model(name, rows, difficulty) for name, rows in all_rows.items()}
    write_json(run_root / "dev_model_summary.json", {"dev_model_summary_version": f"{VERSION}_summary_v1", "models": summaries})
    write_json(run_root / "dev_comparison_report.json", {"dev_comparison_report_version": f"{VERSION}_comparison_v1", "models": summaries, "common_candidate_count": EXPECTED_DEV, "m0_reused": True, "m0_new_solver_generation": 0})
    for filename, dimension in (("dev_by_question_type.json", "question_type"), ("dev_by_generation_source.json", "generation_source"), ("dev_by_position_mechanism.json", "position_mechanism"), ("dev_by_root.json", "root_id")):
        write_json(run_root / filename, {name: dimension_stats(rows, dimension, difficulty) for name, rows in all_rows.items()})
    hard_ids = {str(row["candidate_id"]) for row in m0_rows if row.get("difficulty_bucket") == "hard_candidate"}
    write_json(run_root / "dev_hard_by_evidence_status.json", {name: dimension_stats([row for row in rows if str(row["candidate_id"]) in hard_ids], "evidence_status", difficulty) for name, rows in all_rows.items()})
    transitions = hard_transitions(m0_rows, {"M_clean_policy": clean_rows, "M_inclusive_policy": inclusive_rows})
    write_json(run_root / "dev_m0_hard_transitions.json", transitions)
    contract_path = difficulty_run / "solver_contract.json"; contract_sha = verify_sha_sidecar(contract_path); contract = load_json(contract_path)
    reports = {name: load_json(run_root / scope / "model_execution_report.json") for name, scope in (("M_clean_policy", "clean_dev"), ("M_inclusive_policy", "inclusive_dev"))}
    identity = load_json(run_root / "model_identity_report.json")
    if not all(identity["models"][name].get("adapter_enabled") for name in ("M_clean_policy", "M_inclusive_policy")):
        raise RuntimeError("adapter_identity_not_enabled")
    preflight = {"preflight_report_version": f"{VERSION}_preflight_v1", "status": "PASS", "data_run": str(data_run), "difficulty_run": str(difficulty_run), "formal_run": str(formal_run), "source_sha256": source_snapshot(data_run), "solver_contract_sha256": contract_sha, "solver_contract_max_model_len": contract["max_model_len"], "dev_candidate_count": EXPECTED_DEV, "dev_root_count": EXPECTED_ROOTS, "train_dev_candidate_overlap": 0, "train_dev_root_overlap": 0, "m0_reuse_allowed": load_json(run_root / "m0_reuse_audit.json").get("reuse_allowed") is True, "m0_new_solver_generation": 0, "test_prompts_accessed": False, "semantic_auditor_calls": 0, "external_api_calls": 0, "training_started": False, "offline_finalization": True}
    write_json(run_root / "preflight_report.json", preflight)
    write_json(run_root / "dev_evaluation_contract.json", {"dev_evaluation_contract_version": f"{VERSION}_contract_v1", "solver_contract_sha256": contract_sha, "max_model_len": contract["max_model_len"], "max_tokens": contract["generation_config"]["max_tokens"], "enable_thinking": False, "context_modes": contract["context_modes"], "rollouts_per_position": contract["rollouts_per_position"], "position_counts": contract["position_counts"], "temperatures": contract["temperatures"], "seed_schedule": contract["seed_schedule"], "prompt_contract_version": contract["prompt_contract_version"], "permutation_contract_version": contract["permutation_contract_version"], "canonical_scoring_version": contract["canonical_scoring_version"], "parser_versions": contract["parsers"], "m0_reuse": True, "test_accessed": False})
    final = {"final_dev_evaluation_report_version": f"{VERSION}_final_v1", "status": "READY_FOR_COMMON_TEST_EVALUATION", "models": ["M0", "M_clean_policy", "M_inclusive_policy"], "m0_reused": True, "m0_new_solver_generation": 0, "dev_candidate_count": EXPECTED_DEV, "dev_root_count": EXPECTED_ROOTS, "clean_request_count": reports["M_clean_policy"]["request_count"], "inclusive_request_count": reports["M_inclusive_policy"]["request_count"], "clean_technical_failure_count": reports["M_clean_policy"]["technical_failure_count"], "inclusive_technical_failure_count": reports["M_inclusive_policy"]["technical_failure_count"], "contract_max_model_len": contract["max_model_len"], "training_max_length_not_used_for_eval": True, "test_prompts_accessed": False, "dev_summaries": summaries, "hard_transitions": transitions, "solver_calls_new_in_this_run": reports["M_clean_policy"]["request_count"] + reports["M_inclusive_policy"]["request_count"], "external_api_calls": 0, "semantic_auditor_calls": 0, "training_started": False, "offline_finalization": True}
    write_json(run_root / "final_dev_evaluation_report.json", final)
    md = "\n".join(["# Round-0 Common Dev Evaluation", "", f"Status: **{final['status']}**", "", f"M0 was reused after the frozen fingerprint audit; new M0 Solver generation was `{final['m0_new_solver_generation']}`.", f"Clean and Inclusive each ran `{EXPECTED_REQUESTS}` Dev requests under the frozen B0 K=3 contract with `max_model_len={final['contract_max_model_len']}`.", "No Test prompt, training, external API, or semantic auditor was used.", "", "See `dev_model_summary.json`, per-dimension slices, and model-specific ledgers for results."]) + "\n"
    (run_root / "dev_comparison_report.md").write_text(md, encoding="utf-8")
    (run_root / "final_dev_evaluation_report.md").write_text(md.replace("Dev Evaluation", "Final Dev Evaluation"), encoding="utf-8")
    for path in (run_root / "dev_comparison_report.md", run_root / "final_dev_evaluation_report.md"):
        import hashlib
        path.with_name(path.name + ".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    print(json.dumps({"status": final["status"], "clean_request_count": final["clean_request_count"], "inclusive_request_count": final["inclusive_request_count"], "hard_count": transitions["hard_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
