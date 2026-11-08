"""Run the frozen B0/K=3 Dev Solver with an explicitly supplied local model.

This runner reads the already-rendered Clean Dev prompts and the frozen
Inclusive ledgers. It never regenerates prompts with a different tokenizer,
never accesses Test, and never calls the TJU API. Smoke must pass before batch.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_text, canonical_json, sha256_bytes, sha256_file
from ..vllm_backends import VLLMSolverBackend
from ..backends import SolverRequest
from ..h4_round0_b0_k3_difficulty_v1 import (
    MAX_MODEL_LEN,
    MAX_TOKENS,
    PARSER_SINGLE,
    PARSER_STRUCTURED,
    ROLLOUTS,
    SEED_SCHEDULE_VERSION,
    SINGLE_TEMPERATURE,
    STRUCTURED_TEMPERATURE,
    TOP_P,
)
from ..h4_round0_b0_k3_difficulty_v1.contract import position_variants, score_output
from ..h4_round0_b0_k3_difficulty_v1.io_utils import RunLock, load_json, verify_sha_sidecar
from ..h4_round0_b0_k3_difficulty_v1.request_ledger import Round0RequestLedger, request_cache_key
from ..h4_round0_b0_k3_difficulty_v1.solver import _structured_generate, _truncated_score
from ..h4_round0_common_dev_eval_v1.evaluator import aggregate_model, dimension_stats, hard_transitions
from ..h4_round0_b0_k3_difficulty_v1.difficulty import summarize_candidate_result
from ..h4_round0_strong_api_common_dev_eval_v1.run import load_inputs, _candidate_context
from . import VERSION


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_RUN_ROOT = RUN_BASE / "h4_round0_local_qwen3_14b_common_dev_eval_v1_live_1"
EXPECTED = {"candidates": 485, "roots": 15, "single_choice": 243,
            "multi_select": 123, "ordering": 119, "requests": 12378}
REQUESTS_PER_CANDIDATE = {"single_choice": 24, "multi_select": 30, "ordering": 24}


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ))
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "\n")


def model_identity(model_path: Path, tokenizer_path: Path) -> dict[str, Any]:
    from importlib.metadata import version

    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise RuntimeError(f"local_model_missing_or_invalid:{model_path}")
    if not tokenizer_path.is_dir() or not (tokenizer_path / "tokenizer_config.json").is_file():
        raise RuntimeError(f"local_tokenizer_missing_or_invalid:{tokenizer_path}")
    config = load_json(model_path / "config.json")
    architectures = [str(value) for value in config.get("architectures") or []]
    qwen3_14b = (
        "Qwen3ForCausalLM" in architectures
        and int(config.get("hidden_size", -1)) == 5120
        and int(config.get("num_hidden_layers", -1)) == 40
    )
    if not qwen3_14b:
        raise RuntimeError(
            "model_is_not_expected_qwen3_14b_architecture:"
            f"{architectures}:{config.get('hidden_size')}:{config.get('num_hidden_layers')}"
        )
    model_files = _directory_identity(model_path)
    tokenizer_files = model_files if tokenizer_path == model_path else _directory_identity(tokenizer_path)
    return {
        "identity_version": VERSION + "_model_identity_v1",
        "role": "M_strong_local_solver",
        "provider": "local_vllm",
        "model_path": str(model_path.resolve()),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "model_directory_sha256": sha256_bytes(canonical_json(model_files).encode()),
        "tokenizer_directory_sha256": sha256_bytes(canonical_json(tokenizer_files).encode()),
        "architecture": architectures,
        "hidden_size": int(config["hidden_size"]),
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "dtype": "auto",
        "vllm_version": version("vllm"),
        "generation": {
            "thinking": False,
            "max_model_len": MAX_MODEL_LEN,
            "max_tokens": MAX_TOKENS,
            "top_p": TOP_P,
            "temperatures": {"single_choice": SINGLE_TEMPERATURE,
                             "multi_select": STRUCTURED_TEMPERATURE,
                             "ordering": STRUCTURED_TEMPERATURE},
            "seed_schedule": SEED_SCHEDULE_VERSION,
            "structured_output": "finite_choice_set_v1",
        },
    }


def _directory_identity(path: Path) -> list[dict[str, Any]]:
    rows = []
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        rows.append({"name": child.relative_to(path).as_posix(),
                     "size_bytes": child.stat().st_size,
                     "sha256": sha256_file(child)})
    return rows


def _request_contract(record: Mapping[str, Any], lineage: Mapping[str, Any], model: Mapping[str, Any], contract_sha: str) -> dict[str, Any]:
    qtype = str(record["question_type"])
    return {
        "request_contract_version": "h4_round0_b0_request_key_v1",
        "local_runner_version": VERSION,
        "solver_contract_sha256": contract_sha,
        "root_id": str(record["root_id"]),
        "candidate_id": str(record["candidate_id"]),
        "question_type": qtype,
        "variant_id": str(record["variant_id"]),
        "context_mode": str(record["context_mode"]),
        "slot_id": int(record["slot_id"]),
        "seed": int(record["seed"]),
        "seed_schedule_version": SEED_SCHEDULE_VERSION,
        "prompt_sha256": str(record["prompt_sha256"]),
        "candidate_pool_sha256": str(lineage["candidate_pool_sha256"]),
        "dialogue_sha256": str(lineage["dialogue_sha256"]),
        "display_to_canonical": dict(record["display_to_canonical"] or {}),
        "canonical_gold": list(record["canonical_gold"] or []),
        "backend_identity": {"backend": "VLLMSolverBackend", "model": model["model_path"],
                             "tokenizer": model["tokenizer_path"], "max_model_len": MAX_MODEL_LEN,
                             "temperature": float(record["temperature"]), "top_p": TOP_P,
                             "max_tokens": MAX_TOKENS},
        "parser_version": PARSER_SINGLE if qtype == "single_choice" else PARSER_STRUCTURED,
        "generation_config": {"temperature": float(record["temperature"]), "top_p": TOP_P,
                               "max_tokens": MAX_TOKENS, "enable_thinking": False,
                               "structured_output_version": "finite_choice_set_v1" if qtype != "single_choice" else None},
    }


def _result_path(scope: Path, candidate_id: str) -> Path:
    return scope / "candidate_results" / (sha256_bytes(candidate_id.encode()) + ".json")


def _record_result(record: Mapping[str, Any], candidate: Mapping[str, Any], variant: Mapping[str, Any], response: Any) -> dict[str, Any]:
    qtype = str(record["question_type"])
    finish_reason = str((response.metadata or {}).get("finish_reason") or "").casefold()
    parsed = _truncated_score(qtype) if finish_reason == "length" else score_output(response.raw_text, candidate, variant)
    return {
        "request_id": f"{record['candidate_id']}__{record['variant_id']}__{record['context_mode']}__s{int(record['slot_id']):02d}",
        "root_id": str(record["root_id"]), "candidate_id": str(record["candidate_id"]),
        "question_type": qtype, "variant_id": str(record["variant_id"]),
        "context_mode": str(record["context_mode"]), "slot_id": int(record["slot_id"]),
        "seed": int(record["seed"]), "prompt_sha256": str(record["prompt_sha256"]),
        "prompt_token_count": int((response.metadata or {}).get("prompt_token_count", 0)),
        "display_to_canonical": dict(record["display_to_canonical"] or {}),
        "displayed_gold": list(variant["displayed_gold"]), "canonical_gold": list(record["canonical_gold"] or []),
        "raw_output": response.raw_text, "backend_metadata": dict(response.metadata or {}),
        **parsed,
    }


def _make_request(record: Mapping[str, Any], prompt: str, variant: Mapping[str, Any]) -> SolverRequest:
    return SolverRequest(
        request_id=f"{record['candidate_id']}__{record['variant_id']}__{record['context_mode']}__s{int(record['slot_id']):02d}",
        qa_id=str(record["candidate_id"]), evaluation_mode=str(record["context_mode"]),
        order_id=str(record["variant_id"]), target_slot=int(record["slot_id"]), retry_index=0,
        sampling_seed=int(record["seed"]), options=dict(variant.get("options") or variant.get("items") or {}),
        canonical_answer_text=",".join(str(x) for x in variant["displayed_gold"]),
        prompt_sha256=str(record["prompt_sha256"]), prompt=prompt,
    )


def _run_one(backend: VLLMSolverBackend, ledger: Round0RequestLedger, record: Mapping[str, Any], candidate: Mapping[str, Any], lineage: Mapping[str, Any], variant: Mapping[str, Any], model: Mapping[str, Any], contract_sha: str) -> dict[str, Any]:
    prompt_path = Path(str(record["prompt_path"]))
    prompt = prompt_path.read_text(encoding="utf-8")
    if sha256_bytes(prompt.encode()) != str(record["prompt_sha256"]):
        raise RuntimeError(f"frozen_prompt_sha_mismatch:{record['candidate_id']}:{record['variant_id']}:{record['context_mode']}:{record['slot_id']}")
    contract = _request_contract(record, lineage, model, contract_sha)
    qtype = str(record["question_type"])
    request = _make_request(record, prompt, variant)
    if qtype == "single_choice":
        call = lambda: backend.generate(request)
    else:
        call = lambda: _structured_generate(backend, request, question_type=qtype)
    response, _ = ledger.invoke(contract=contract, prompt=prompt, call=call)
    result = _record_result(record, candidate, variant, response)
    cache_key = request_cache_key(contract)
    result["request_cache_key"] = cache_key
    ledger.record_decision(str(record["root_id"]), cache_key, {
        "parser_version": result["parser_version"],
        "parse_status": result["parse_status"],
        "scored": result["scored"],
        "correct": result["correct"],
        "canonical_prediction": result["canonical_prediction"],
        "failure_type": result["failure_type"],
    })
    return result


def _load_old_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def _prepare(run_root: Path, inputs: Mapping[str, Any], model: Mapping[str, Any]) -> str:
    if run_root.exists():
        raise RuntimeError(f"run_root_exists:{run_root}")
    run_root.mkdir(parents=True)
    scope = run_root / "strong_dev"
    for name in ("raw_prompts", "raw_responses", "request_ledger", "candidate_results"):
        (scope / name).mkdir(parents=True)
    contract = {
        "contract_version": VERSION + "_solver_contract_v1",
        "source_dev_evaluation_contract_sha256": inputs["source_contract_sha"],
        "model_identity": model,
        "max_model_len": MAX_MODEL_LEN, "max_tokens": MAX_TOKENS, "top_p": TOP_P,
        "temperatures": {"single_choice": SINGLE_TEMPERATURE, "multi_select": STRUCTURED_TEMPERATURE, "ordering": STRUCTURED_TEMPERATURE},
        "thinking": False, "seed_schedule": SEED_SCHEDULE_VERSION,
        "parser_versions": {"single_choice": PARSER_SINGLE, "multi_select": PARSER_STRUCTURED, "ordering": PARSER_STRUCTURED},
        "prompt_source": "common_dev_eval_v1/clean_dev/raw_prompts",
        "inclusive_prompt_sha_verified": True, "test_accessed": False,
    }
    write_json(run_root / "local_solver_contract.json", contract)
    write_json(run_root / "strong_local_model_identity.json", model)
    write_json(run_root / "strong_local_solver_plan.json", {
        "dev_candidates": EXPECTED["candidates"], "dev_roots": EXPECTED["roots"],
        "question_type_counts": {key: EXPECTED[key] for key in ("single_choice", "multi_select", "ordering")},
        "planned_requests": EXPECTED["requests"], "smoke_candidates": 3,
        "smoke_requests": 78, "batch_reuses_smoke_ledger": True,
        "context_modes": ["no_context", "full_context"],
        "rollouts_per_position": ROLLOUTS,
    })
    return sha256_file(run_root / "local_solver_contract.json")


def _run_phase(args: argparse.Namespace, inputs: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    model_path = args.model.resolve()
    tokenizer_path = (args.tokenizer or args.model).resolve()
    model = model_identity(model_path, tokenizer_path)
    if phase == "preflight":
        contract_sha = _prepare(run_root, inputs, model)
        report = {"status": "PASS", "model": model, "dev_candidates": EXPECTED["candidates"], "dev_roots": EXPECTED["roots"], "question_type_counts": {k: EXPECTED[k] for k in ("single_choice", "multi_select", "ordering")}, "planned_requests": EXPECTED["requests"], "inclusive_prompt_sha_verified": True, "test_accessed": False, "api_calls": 0, "solver_calls": 0, "solver_contract_sha256": contract_sha}
        write_json(run_root / "preflight_report.json", report)
        return report
    if not run_root.is_dir() or not (run_root / "preflight_report.json").is_file():
        raise RuntimeError("preflight_required")
    verify_sha_sidecar(run_root / "preflight_report.json")
    preflight = load_json(run_root / "preflight_report.json")
    if preflight.get("status") != "PASS":
        raise RuntimeError("preflight_not_pass")
    contract_sha = verify_sha_sidecar(run_root / "local_solver_contract.json")
    frozen_contract = load_json(run_root / "local_solver_contract.json")
    if frozen_contract.get("model_identity") != model:
        raise RuntimeError("local_model_identity_drift_since_preflight")
    scope = run_root / "strong_dev"
    selected = list(inputs["prompt_records"])
    if phase == "smoke":
        selected_ids = {}
        for row in sorted(inputs["manifest"], key=lambda r: str(r["candidate_id"])):
            selected_ids.setdefault(str(row["question_type"]), str(row["candidate_id"]))
        selected = [r for r in selected if str(r["candidate_id"]) in set(selected_ids.values())]
    elif phase == "batch":
        smoke_path = run_root / "smoke_report.json"
        if not smoke_path.is_file() or load_json(smoke_path).get("status") != "PASS":
            raise RuntimeError("passing_smoke_required_before_batch")
    candidate_by_id = {str(row["candidate_id"]): row for row in inputs["manifest"]}
    metadata = candidate_by_id
    candidates = {cid: _candidate_context(inputs, cid)[0] for cid in candidate_by_id}
    variants_by_id = {cid: {str(v["variant_id"]): v for v in position_variants(candidates[cid])} for cid in candidate_by_id}
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with RunLock(run_root):
        ledger = Round0RequestLedger(scope, solver_contract_sha256=contract_sha)
        ledger.recover_orphaned_requests()
        backend = VLLMSolverBackend(model_path=model_path, tokenizer_path=tokenizer_path, max_model_len=MAX_MODEL_LEN, temperature=SINGLE_TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, gpu_memory_utilization=float(args.gpu_memory_utilization))
        try:
            for record in sorted(selected, key=lambda r: (str(r["candidate_id"]), str(r["variant_id"]), str(r["context_mode"]), int(r["slot_id"]))):
                cid = str(record["candidate_id"])
                variant = variants_by_id[cid][str(record["variant_id"])]
                by_candidate[cid].append(_run_one(backend, ledger, record, candidates[cid], metadata[cid], variant, model, contract_sha))
        finally:
            backend.close()
            gc.collect()
    completed_candidates = 0
    completed_results = []
    candidate_rows = []
    for cid, requests in sorted(by_candidate.items()):
        qtype = str(metadata[cid]["question_type"])
        if len(requests) != REQUESTS_PER_CANDIDATE[qtype]:
            continue
        candidate = candidates[cid]
        result = {"candidate_result_version": "h4_round0_b0_k3_candidate_result_v1", "probe_version": "h4_round0_local_qwen3_14b_common_dev_eval_v1", "solver_contract_sha256": contract_sha, "root_id": str(metadata[cid]["root_id"]), "candidate_id": cid, "question_type": qtype, "candidate_pool_sha256": metadata[cid]["candidate_pool_sha256"], "dialogue_sha256": metadata[cid]["dialogue_sha256"], "request_count": len(requests), "technical_failure_count": sum(not bool(x.get("scored")) for x in requests), "requests": requests}
        write_json(_result_path(scope, cid), result)
        completed_results.append(result)
        summary = summarize_candidate_result(result)
        candidate_rows.append({**summary, "model_name": "M_strong_local", "candidate_id": cid, "root_id": result["root_id"], "question_type": qtype, "generation_source": metadata[cid].get("generation_source"), "position_mechanism": metadata[cid].get("position_mechanism", "unknown"), "evidence_status": metadata[cid].get("evidence_status", "unknown"), "request_count": len(requests), "technical_failure_count": result["technical_failure_count"], "candidate_pool_sha256": result["candidate_pool_sha256"], "dialogue_sha256": result["dialogue_sha256"], "solver_contract_sha256": contract_sha})
        completed_candidates += 1
    write_jsonl(scope / "candidate_results.jsonl", completed_results)
    write_jsonl(scope / "difficulty_manifest.jsonl", candidate_rows)
    expected = 3 if phase == "smoke" else EXPECTED["candidates"]
    expected_requests = 78 if phase == "smoke" else EXPECTED["requests"]
    technical = sum(int(r["technical_failure_count"]) for r in candidate_rows)
    report = {"status": "PASS" if completed_candidates == expected and sum(len(v) for v in by_candidate.values()) == expected_requests and technical == 0 else "BLOCKED", "phase": phase, "model": model, "candidate_count": completed_candidates, "expected_candidate_count": expected, "request_count": sum(len(v) for v in by_candidate.values()), "expected_request_count": expected_requests, "technical_failure_count": technical, "api_calls": 0, "test_accessed": False, "candidate_rows": candidate_rows}
    write_json(run_root / ("smoke_report.json" if phase == "smoke" else "batch_report.json"), report)
    return report


def finalize(args: argparse.Namespace, inputs: Mapping[str, Any]) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    scope = run_root / "strong_dev"
    batch = load_json(run_root / "batch_report.json")
    if batch.get("status") != "PASS":
        raise RuntimeError("passing_batch_required_before_finalize")
    strong_rows = batch.get("candidate_rows") or []
    difficulty = {str(row["candidate_id"]): row for row in inputs["difficulty"].values()}
    old_root = RUN_BASE / "h4_round0_common_dev_eval_v1_live_3"
    manifest_by_id = {str(row["candidate_id"]): row for row in inputs["manifest"]}
    old_models = {}
    for name, path in (("M_clean_policy", old_root / "clean_dev" / "candidate_results.jsonl"), ("M_inclusive_policy", old_root / "inclusive_dev" / "candidate_results.jsonl")):
        old_models[name] = [summarize_candidate_result(row) | {"model_name": name, "candidate_id": row["candidate_id"], "root_id": row["root_id"], "question_type": row["question_type"], "generation_source": manifest_by_id[str(row["candidate_id"])].get("generation_source"), "position_mechanism": manifest_by_id[str(row["candidate_id"])].get("position_mechanism", "unknown"), "technical_failure_count": row.get("technical_failure_count", 0)} for row in _load_old_rows(path)]
    m0_dir = RUN_BASE / "h4_round0_b0_k3_difficulty_v1_live_1" / "candidate_results"
    m0_models = []
    for row in inputs["manifest"]:
        p = m0_dir / (sha256_bytes(str(row["candidate_id"]).encode()) + ".json")
        result = load_json(p)
        m0_models.append(summarize_candidate_result(result) | {"model_name": "M0", "candidate_id": result["candidate_id"], "root_id": result["root_id"], "question_type": result["question_type"], "generation_source": row.get("generation_source"), "position_mechanism": row.get("position_mechanism", "unknown"), "technical_failure_count": result.get("technical_failure_count", 0)})
    all_rows = {"M0": m0_models, **old_models, "M_strong_local": strong_rows}
    summaries = {name: aggregate_model(name, rows, difficulty) for name, rows in all_rows.items()}
    write_json(run_root / "strong_dev_summary.json", summaries["M_strong_local"])
    write_json(run_root / "four_model_dev_comparison.json", {"status": "READY_FOR_REVIEW", "models": summaries, "dev_candidates": EXPECTED["candidates"], "dev_roots": EXPECTED["roots"], "strong_model": "M_strong_local", "old_models_reused": ["M0", "M_clean_policy", "M_inclusive_policy"], "test_accessed": False, "api_calls": 0})
    for dimension in ("question_type", "generation_source", "position_mechanism", "root_id"):
        write_json(run_root / f"strong_vs_trained_by_{dimension}.json", {name: dimension_stats(rows, dimension) for name, rows in all_rows.items()})
    transitions = hard_transitions(m0_models, {"M_clean_policy": old_models["M_clean_policy"], "M_inclusive_policy": old_models["M_inclusive_policy"], "M_strong_local": strong_rows})
    write_json(run_root / "strong_m0_hard_transitions.json", transitions)
    final = {"status": "READY_FOR_REVIEW", "strong_model": "M_strong_local", "dev_candidates": EXPECTED["candidates"], "dev_roots": EXPECTED["roots"], "actual_strong_requests": batch["request_count"], "technical_failure_count": batch["technical_failure_count"], "api_calls": 0, "test_accessed": False, "comparison_path": str(run_root / "four_model_dev_comparison.json")}
    write_json(run_root / "final_strong_local_dev_report.json", final)
    atomic_text(run_root / "final_strong_local_dev_report.md", "# Strong Local Qwen3-14B Dev Evaluation\n\nStatus: **READY_FOR_REVIEW**\n\nThe strong local model ran only the frozen 485-candidate Dev set. M0, Clean, and Inclusive results were reused; no API, Test, QA generation, or training was used. See `four_model_dev_comparison.json`.\n")
    atomic_text(run_root / "final_strong_local_dev_report.md.sha256", sha256_file(run_root / "final_strong_local_dev_report.md") + "\n")
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "smoke", "batch", "finalize"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    args = parser.parse_args(argv)
    inputs = load_inputs()
    if args.phase == "finalize":
        result = finalize(args, inputs)
    else:
        result = _run_phase(args, inputs, phase=args.phase)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"PASS", "READY_FOR_REVIEW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
