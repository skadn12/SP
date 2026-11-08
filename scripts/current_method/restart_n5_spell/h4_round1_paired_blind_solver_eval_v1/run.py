"""Paired Control/Feedback Solver evaluation for the Round-1 fresh QA batch.

The runner reuses the frozen Round-0 B0 K=3 protocol and evaluates the two
trained policies independently.  It never reads Dev/Test prompts and never
changes the QA or dialogue artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_round0_b0_k3_difficulty_v1.contract import build_prompt, dialogue_rows, position_variants
from ..h4_round0_b0_k3_difficulty_v1.difficulty import summarize_candidate_result
from ..h4_round0_b0_k3_difficulty_v1.gate import (
    PROGRAM_PASS,
    build_solver_contract,
    directory_identity,
    gate_candidate,
)
from ..h4_round0_b0_k3_difficulty_v1.io_utils import (
    RunLock,
    load_json,
    load_jsonl,
    stable_seed,
    verify_sha_sidecar,
    write_once_json,
    write_once_jsonl,
)
from ..h4_round0_b0_k3_difficulty_v1.request_ledger import Round0RequestLedger
from ..h4_round0_b0_k3_difficulty_v1.solver import (
    ensure_vllm_environment,
    evaluate_candidate,
    load_or_recover_candidate_result,
)
from ..h4_round0_b0_k3_difficulty_v1 import (
    MAX_MODEL_LEN,
    MAX_TOKENS,
    ROLLOUTS,
    SINGLE_TEMPERATURE,
    STRUCTURED_TEMPERATURE,
    TOP_P,
    candidate_request_count,
)
from ..h4_round0_common_dev_eval_v1.evaluator import composite_directory_sha
from ..h4_round0_b0_k3_difficulty_v1.run import assert_gpu_idle
from ..vllm_backends import VLLMSolverBackend
from . import (
    BASE_MODEL,
    BRANCHES,
    IDENTITY_REPORT,
    MODELS,
    OUTPUT_RUN,
    PROBE_VERSION,
    QA_RUN,
)


def _write_json(path: Path, value: Any) -> None:
    write_once_json(path, value)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    write_once_jsonl(path, rows)


def _verify_source_artifact(path: Path) -> dict[str, Any]:
    """Verify source bytes without requiring one particular sidecar syntax.

    The frozen QA derivation used both ``<sha>`` and ``<sha>  <filename>``
    sidecars.  The first token is the actual digest in either format.  Some
    copied per-root artifacts have no sidecar; their digest is still captured
    in this run's immutable manifests and checked again before execution.
    """
    if not path.is_file():
        raise RuntimeError(f"source_artifact_missing:{path}")
    observed = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        return {"path": str(path), "sha256": observed, "sidecar": "missing"}
    tokens = sidecar.read_text(encoding="utf-8").strip().split()
    if not tokens or tokens[0] != observed:
        raise RuntimeError(f"source_artifact_sha_mismatch:{path}:{tokens[:1]}:{observed}")
    return {"path": str(path), "sha256": observed, "sidecar": "verified"}


def _load_source_rows() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    report = load_json(QA_RUN / "final_paired_qa_batch_report.json")
    if report.get("status") != "ROUND1_PAIRED_BLIND_QA_BATCH_PASS":
        raise RuntimeError(f"qa_batch_not_pass:{report.get('status')}")
    _verify_source_artifact(QA_RUN / "final_paired_qa_batch_report.json")
    _verify_source_artifact(QA_RUN / "all_candidates.jsonl")
    rows = load_jsonl(QA_RUN / "all_candidates.jsonl")
    if len(rows) != int(report.get("candidate_count_total", -1)) or len(rows) != 530:
        raise RuntimeError(f"qa_candidate_count_mismatch:{len(rows)}")
    ids = [str(row.get("candidate_id") or "") for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("qa_candidate_ids_not_unique")
    pools: dict[tuple[str, str], dict[str, Any]] = {}
    dialogues: dict[tuple[str, str], dict[str, Any]] = {}
    for branch in BRANCHES:
        for root_dir in sorted((QA_RUN / branch / "roots").glob("*")):
            root_id = root_dir.name
            pool_path = root_dir / "candidate_pool.json"
            dialogue_path = root_dir / "dialogue_source.json"
            gate_path = root_dir / "program_gate_results.jsonl"
            for path in (pool_path, dialogue_path, gate_path):
                _verify_source_artifact(path)
            pool_rows = load_json(pool_path)
            gate_rows = load_jsonl(gate_path)
            if any(row.get("branch") != branch or row.get("root_id") != root_id for row in pool_rows):
                raise RuntimeError(f"qa_pool_lineage_mismatch:{branch}:{root_id}")
            if len(pool_rows) != len(gate_rows) or any(row.get("decision") != "pass" for row in gate_rows):
                raise RuntimeError(f"qa_gate_not_all_pass:{branch}:{root_id}")
            pools[(branch, root_id)] = {str(row["candidate_id"]): dict(row) for row in pool_rows}
            if len(pools[(branch, root_id)]) != len(pool_rows):
                raise RuntimeError(f"qa_pool_duplicate_ids:{branch}:{root_id}")
            dialogues[(branch, root_id)] = load_json(dialogue_path)
    row_keys = {(str(row.get("branch")), str(row.get("root_id")), str(row.get("candidate_id"))) for row in rows}
    pool_keys = {(branch, root, cid) for (branch, root), pool in pools.items() for cid in pool}
    if row_keys != pool_keys:
        raise RuntimeError("qa_aggregate_pool_identity_mismatch")
    return rows, pools, dialogues


def _identity_preflight() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    verify_sha_sidecar(IDENTITY_REPORT)
    source = load_json(IDENTITY_REPORT)
    base_current = directory_identity(BASE_MODEL)
    if base_current != source.get("base_model_identity"):
        raise RuntimeError("base_model_identity_drift")
    models: dict[str, dict[str, Any]] = {}
    for name in MODELS:
        frozen = dict(source.get("models", {}).get(name) or {})
        if frozen.get("adapter_enabled") is not True:
            raise RuntimeError(f"adapter_not_enabled:{name}")
        merged_path = Path(str(frozen.get("merged_model_path") or ""))
        adapter_path = Path(str(frozen.get("adapter_path") or ""))
        if not merged_path.is_dir() or not adapter_path.is_dir():
            raise RuntimeError(f"model_or_adapter_missing:{name}")
        merged_identity = directory_identity(merged_path)
        if merged_identity != frozen.get("merged_model_identity"):
            raise RuntimeError(f"merged_model_identity_drift:{name}")
        adapter_config = adapter_path / "adapter_config.json"
        adapter_state = adapter_path / "adapter_model.pt"
        if not adapter_config.is_file() or not adapter_state.is_file():
            raise RuntimeError(f"adapter_files_missing:{name}")
        if sha256_file(adapter_config) != frozen.get("adapter_config_sha256") or sha256_file(adapter_state) != frozen.get("adapter_state_sha256"):
            raise RuntimeError(f"adapter_sha_drift:{name}")
        if composite_directory_sha(adapter_path) != frozen.get("adapter_composite_sha256"):
            raise RuntimeError(f"adapter_composite_sha_drift:{name}")
        models[name] = {
            "model_name": name,
            "adapter_name": frozen.get("adapter_name"),
            "adapter_enabled": True,
            "adapter_path": str(adapter_path),
            "adapter_config_sha256": frozen.get("adapter_config_sha256"),
            "adapter_state_sha256": frozen.get("adapter_state_sha256"),
            "adapter_composite_sha256": frozen.get("adapter_composite_sha256"),
            "merged_model_path": str(merged_path),
            "merged_model_identity": merged_identity,
            "base_model_identity": base_current,
            "lora_config": frozen.get("lora_config"),
        }
    return {"source_identity_report": str(IDENTITY_REPORT), "source_identity_sha256": sha256_file(IDENTITY_REPORT), "base_model_identity": base_current, "models": models}, models


def _build_inputs(rows: list[dict[str, Any]], pools: Mapping[tuple[str, str], Mapping[str, Any]], dialogues: Mapping[tuple[str, str], Mapping[str, Any]], tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    gates: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for row in rows:
        branch, root, cid = str(row["branch"]), str(row["root_id"]), str(row["candidate_id"])
        candidate = dict(pools[(branch, root)][cid])
        if candidate != row:
            raise RuntimeError(f"qa_aggregate_candidate_drift:{cid}")
        gate = gate_candidate(candidate, dialogues[(branch, root)], tokenizer, root_id=root)
        gate.update({
            "candidate_pool_path": str(QA_RUN / branch / "roots" / root / "candidate_pool.json"),
            "candidate_pool_sha256": sha256_file(QA_RUN / branch / "roots" / root / "candidate_pool.json"),
            "dialogue_path": str(QA_RUN / branch / "roots" / root / "dialogue_source.json"),
            "dialogue_sha256": sha256_file(QA_RUN / branch / "roots" / root / "dialogue_source.json"),
            "branch": branch,
        })
        gates.append(gate)
        candidate_rows.append(candidate)
    if any(row.get("gate_status") != PROGRAM_PASS for row in gates):
        bad = [row["candidate_id"] for row in gates if row.get("gate_status") != PROGRAM_PASS]
        raise RuntimeError(f"round1_solver_program_gate_failed:{bad[:10]}")
    return candidate_rows, {str(row["candidate_id"]): row for row in gates}


def _plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_branch: dict[str, Counter[str]] = {branch: Counter() for branch in BRANCHES}
    for row in rows:
        by_branch[str(row["branch"])][str(row["question_type"])] += 1
    branch_plan = {}
    for branch, counts in by_branch.items():
        request_counts = {
            q: sum(candidate_request_count(row) for row in rows
                   if row["branch"] == branch and row["question_type"] == q)
            for q in ("single_choice", "multi_select", "ordering")
        }
        branch_plan[branch] = {
            "candidate_count": sum(counts.values()),
            "question_type_counts": dict(sorted(counts.items())),
            "request_count_by_question_type": request_counts,
            "planned_request_count": sum(request_counts.values()),
            "closed": sum(counts.values()) > 0 and all(q in counts for q in request_counts),
        }
    total = sum(int(value["planned_request_count"]) for value in branch_plan.values())
    if total != 13518 or any(not value["closed"] for value in branch_plan.values()):
        raise RuntimeError(f"solver_plan_not_closed:{total}:{branch_plan}")
    return {
        "solver_plan_version": f"{PROBE_VERSION}_plan_v1",
        "qa_run": str(QA_RUN),
        "candidate_count": len(rows),
        "root_count": len({str(row["root_id"]) for row in rows}),
        "branch_plan": branch_plan,
        "per_model_request_count": total,
        "all_model_request_count": total * len(MODELS),
        "requests_per_candidate": "candidate_position_count * 2_context_modes * 3_rollouts",
        "position_counts": {"single_choice": {"fixed": 4}, "multi_select": {"min": 4, "max": 6, "source": "candidate"}, "ordering": {"min": 4, "max": 6, "source": "candidate"}},
        "rollouts_per_position": ROLLOUTS,
        "context_modes": ["no_context", "full_context"],
        "temperatures": {"single_choice": SINGLE_TEMPERATURE, "multi_select": STRUCTURED_TEMPERATURE, "ordering": STRUCTURED_TEMPERATURE},
        "max_model_len": MAX_MODEL_LEN,
        "max_tokens": MAX_TOKENS,
        "only_b0": True,
        "dev_test_accessed": False,
    }


def _raw_request_files(scope: Path, candidate: Mapping[str, Any], result: Mapping[str, Any], tokenizer: Any, expanded: Mapping[str, Any]) -> None:
    by_variant = {str(row["variant_id"]): row for row in position_variants(candidate)}
    for request in result.get("requests") or []:
        request_id = str(request["request_id"])
        digest = hashlib.sha256(request_id.encode()).hexdigest()
        variant = by_variant[str(request["variant_id"])]
        prompt = build_prompt(tokenizer, candidate, variant, dialogue_rows(expanded), context_mode=str(request["context_mode"]))
        prompt_path = scope / "raw_prompts" / f"{digest}.txt"
        response_path = scope / "raw_responses" / f"{digest}.json"
        if not prompt_path.exists():
            atomic_text(prompt_path, prompt)
        if not response_path.exists():
            atomic_text(response_path, json.dumps({"request_id": request_id, "raw_output": request.get("raw_output"), "backend_metadata": request.get("backend_metadata"), "parse_status": request.get("parse_status"), "scored": request.get("scored"), "correct": request.get("correct")}, ensure_ascii=False, sort_keys=True) + "\n")


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def acc(context: str) -> float | None:
        total = sum(int(row["position_diagnostics"][context]["total_count"]) for row in rows)
        correct = sum(int(row["position_diagnostics"][context]["correct_count"]) for row in rows)
        return correct / total if total else None
    def root_equal(context: str) -> float | None:
        roots: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            value = row["position_diagnostics"][context].get("accuracy")
            if value is not None:
                roots[str(row["root_id"])].append(float(value))
        values = [sum(values) / len(values) for values in roots.values() if values]
        return sum(values) / len(values) if values else None
    buckets = Counter(str(row.get("difficulty_bucket")) for row in rows)
    qtypes = Counter(str(row.get("question_type")) for row in rows)
    return {
        "candidate_count": len(rows),
        "candidate_weighted": {"full_context_accuracy": acc("full_context"), "no_context_accuracy": acc("no_context"), "context_gain": (acc("full_context") - acc("no_context")) if acc("full_context") is not None and acc("no_context") is not None else None},
        "root_equal": {"full_context_accuracy": root_equal("full_context"), "no_context_accuracy": root_equal("no_context"), "context_gain": (root_equal("full_context") - root_equal("no_context")) if root_equal("full_context") is not None and root_equal("no_context") is not None else None},
        "canonical_flip_rate_mean": {context: (sum(float(row["canonical_flip_rate"][context]) for row in rows if row["canonical_flip_rate"][context] is not None) / sum(row["canonical_flip_rate"][context] is not None for row in rows)) if any(row["canonical_flip_rate"][context] is not None for row in rows) else None for context in ("no_context", "full_context")},
        "worst_position_accuracy_mean": {context: (sum(min((row["position_diagnostics"][context].get("position_accuracy") or {}).values()) for row in rows if row["position_diagnostics"][context].get("position_accuracy")) / sum(bool(row["position_diagnostics"][context].get("position_accuracy")) for row in rows)) if any(row["position_diagnostics"][context].get("position_accuracy") for row in rows) else None for context in ("no_context", "full_context")},
        "difficulty_bucket_counts": dict(sorted(buckets.items())),
        "question_type_counts": dict(sorted(qtypes.items())),
        "technical_failure_count": sum(int(row.get("technical_failure_count") or 0) for row in rows),
    }


def _run_model(*, model_name: str, model_identity: Mapping[str, Any], rows: list[dict[str, Any]], gates: Mapping[str, Mapping[str, Any]], pools: Mapping[tuple[str, str], Mapping[str, Any]], dialogues: Mapping[tuple[str, str], Mapping[str, Any]], tokenizer: Any, protocol: Mapping[str, Any], run_root: Path, gpu_memory_utilization: float) -> dict[str, Any]:
    model_dir = run_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_contract = {
        "solver_contract_version": f"{PROBE_VERSION}_model_contract_v1",
        "model_name": model_name,
        "model_path": model_identity["merged_model_path"],
        "adapter_path": model_identity["adapter_path"],
        "adapter_enabled": True,
        "adapter_sha256": model_identity["adapter_composite_sha256"],
        "base_solver_protocol_sha256": sha256_bytes(canonical_json(protocol).encode()),
        "tokenizer": str(BASE_MODEL),
        "max_model_len": MAX_MODEL_LEN,
        "temperatures": protocol["temperatures"],
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "rollouts_per_position": ROLLOUTS,
        "seed_schedule": protocol["seed_schedule"],
        "prompt_contract_version": protocol["prompt_contract_version"],
        "permutation_contract_version": protocol["permutation_contract_version"],
        "canonical_scoring_version": protocol["canonical_scoring_version"],
        "output_mode": "B0_label",
    }
    contract_sha = sha256_bytes(canonical_json(model_contract).encode())
    _write_json(model_dir / "solver_contract.json", {**model_contract, "solver_contract_sha256": contract_sha})
    summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    assert_gpu_idle()
    ensure_vllm_environment()
    backend = VLLMSolverBackend(
        model_path=Path(str(model_identity["merged_model_path"])),
        tokenizer_path=BASE_MODEL,
        max_model_len=MAX_MODEL_LEN,
        temperature=SINGLE_TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    try:
        for branch in BRANCHES:
            scope = model_dir / branch
            scope.mkdir(parents=True, exist_ok=True)
            branch_rows = [row for row in rows if str(row["branch"]) == branch]
            ledger = Round0RequestLedger(scope, solver_contract_sha256=contract_sha)
            recovery = ledger.recover_orphaned_requests()
            results: list[dict[str, Any]] = []
            metadata_rows: list[dict[str, Any]] = []
            for row in sorted(branch_rows, key=lambda value: str(value["candidate_id"])):
                cid, root = str(row["candidate_id"]), str(row["root_id"])
                candidate = pools[(branch, root)][cid]
                result = load_or_recover_candidate_result(
                    run_root=scope,
                    candidate=candidate,
                    expanded=dialogues[(branch, root)],
                    gate_row=gates[cid],
                    tokenizer=tokenizer,
                    backend=backend,
                    ledger=ledger,
                    solver_contract_sha256=contract_sha,
                )
                if result is None:
                    result = evaluate_candidate(
                        candidate=candidate,
                        expanded=dialogues[(branch, root)],
                        gate_row=gates[cid],
                        tokenizer=tokenizer,
                        backend=backend,
                        ledger=ledger,
                        solver_contract_sha256=contract_sha,
                    )
                    write_once_json(scope / "candidate_results" / f"{sha256_bytes(cid.encode())}.json", result)
                if result.get("solver_contract_sha256") != contract_sha or result.get("candidate_pool_sha256") != gates[cid]["candidate_pool_sha256"] or result.get("dialogue_sha256") != gates[cid]["dialogue_sha256"]:
                    raise RuntimeError(f"solver_result_identity_drift:{model_name}:{branch}:{cid}")
                _raw_request_files(scope, candidate, result, tokenizer, dialogues[(branch, root)])
                results.append(result)
                metadata_rows.append({**summarize_candidate_result(result), "model_name": model_name, "branch": branch, "candidate_id": cid, "root_id": root, "question_type": candidate["question_type"], "generation_source": candidate.get("generation_source"), "request_count": result["request_count"], "technical_failure_count": result["technical_failure_count"], "solver_contract_sha256": contract_sha, "candidate_pool_sha256": gates[cid]["candidate_pool_sha256"], "dialogue_sha256": gates[cid]["dialogue_sha256"]})
            expected = sum(candidate_request_count(row) for row in branch_rows)
            observed = sum(int(result["request_count"]) for result in results)
            ledger_summary = ledger.summary()
            if len(results) != len(branch_rows) or observed != expected or ledger_summary["ambiguous_request_count"] != 0:
                raise RuntimeError(f"solver_branch_not_closed:{model_name}:{branch}:{len(results)}:{len(branch_rows)}:{observed}:{expected}:{ledger_summary}")
            _write_jsonl(scope / "candidate_results.jsonl", results)
            _write_jsonl(scope / "difficulty_manifest.jsonl", metadata_rows)
            branch_report = {"status": "PASS", "model_name": model_name, "branch": branch, "candidate_count": len(results), "request_count": observed, "expected_request_count": expected, "technical_failure_count": sum(int(result["technical_failure_count"]) for result in results), "ledger": ledger_summary, "orphan_recovery": recovery, "solver_contract_sha256": contract_sha}
            _write_json(scope / "branch_report.json", branch_report)
            summaries[branch] = branch_report
            all_rows.extend(metadata_rows)
    finally:
        backend.close()
    model_summary = {"status": "PASS", "model_name": model_name, "identity": dict(model_identity), "solver_contract_sha256": contract_sha, "branches": summaries, "summary": _summary(all_rows)}
    _write_json(model_dir / "model_summary.json", model_summary)
    return {"model_summary": model_summary, "rows": all_rows}


def preflight(run_root: Path) -> dict[str, Any]:
    if run_root.exists() and (run_root / "preflight_report.json").is_file():
        verify_sha_sidecar(run_root / "preflight_report.json")
    run_root.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True)
    rows, pools, dialogues = _load_source_rows()
    identity, models = _identity_preflight()
    candidate_rows, gates = _build_inputs(rows, pools, dialogues, tokenizer)
    protocol = build_solver_contract(tokenizer, model_path=BASE_MODEL, tokenizer_path=BASE_MODEL)
    plan = _plan(candidate_rows)
    _write_json(run_root / "model_identity_report.json", identity)
    _write_json(run_root / "solver_protocol.json", protocol)
    _write_json(run_root / "solver_plan.json", plan)
    _write_jsonl(run_root / "candidate_manifest.jsonl", candidate_rows)
    _write_jsonl(run_root / "program_gate_results.jsonl", list(gates.values()))
    report = {"status": "PASS", "probe_version": PROBE_VERSION, "qa_run": str(QA_RUN), "candidate_count": len(candidate_rows), "root_count": len({str(row["root_id"]) for row in candidate_rows}), "branch_counts": {branch: sum(str(row["branch"]) == branch for row in candidate_rows) for branch in BRANCHES}, "models": list(models), "plan_sha256": sha256_file(run_root / "solver_plan.json"), "solver_protocol_sha256": sha256_file(run_root / "solver_protocol.json"), "base_model": str(BASE_MODEL), "tokenizer": str(BASE_MODEL), "max_model_len": MAX_MODEL_LEN, "temperatures": protocol["temperatures"], "program_gate_pass_count": sum(gate.get("gate_status") == PROGRAM_PASS for gate in gates.values()), "program_gate_reject_count": sum(gate.get("gate_status") != PROGRAM_PASS for gate in gates.values()), "dev_test_accessed": False, "solver_started": False}
    _write_json(run_root / "preflight_report.json", report)
    return report


def live(run_root: Path, gpu_memory_utilization: float) -> dict[str, Any]:
    pre = load_json(run_root / "preflight_report.json")
    if pre.get("status") != "PASS":
        raise RuntimeError("solver_preflight_required")
    verify_sha_sidecar(run_root / "solver_protocol.json")
    verify_sha_sidecar(run_root / "solver_plan.json")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True)
    rows, pools, dialogues = _load_source_rows()
    _, models = _identity_preflight()
    _, gates = _build_inputs(rows, pools, dialogues, tokenizer)
    protocol = load_json(run_root / "solver_protocol.json")
    model_results: dict[str, Any] = {}
    with RunLock(run_root):
        for model_name in MODELS:
            result = _run_model(model_name=model_name, model_identity=models[model_name], rows=rows, gates=gates, pools=pools, dialogues=dialogues, tokenizer=tokenizer, protocol=protocol, run_root=run_root, gpu_memory_utilization=gpu_memory_utilization)
            model_results[model_name] = result["model_summary"]
    comparison = {
        "candidate_count": len(rows),
        "models": {name: summary["summary"] for name, summary in model_results.items()},
        "by_model_branch": {name: summary["branches"] for name, summary in model_results.items()},
        "paired_difference": "same QA batch, branch-specific cumulative dialogue is the treatment difference",
    }
    _write_json(run_root / "paired_solver_comparison.json", comparison)
    atomic_text(run_root / "paired_solver_comparison.md", "# Round-1 paired blind Solver comparison\n\n" + json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    atomic_text(run_root / "paired_solver_comparison.md.sha256", sha256_file(run_root / "paired_solver_comparison.md") + "\n")
    final = {
        "status": "ROUND1_PAIRED_BLIND_SOLVER_COMPLETE",
        "candidate_count": len(rows),
        "models": list(MODELS),
        "model_summaries": model_results,
        "candidate_evaluations": sum(
            summary["summary"]["candidate_count"]
            for summary in model_results.values()
        ),
        "request_count": sum(
            branch["request_count"]
            for summary in model_results.values()
            for branch in summary["branches"].values()
        ),
        "dev_test_accessed": False,
        "semantic_auditor_calls": 0,
        "next_step": "offline compare Control versus Feedback; do not use this run to modify QA or training manifests",
    }
    _write_json(run_root / "final_solver_report.json", final)
    atomic_text(run_root / "final_solver_report.md", "# Round-1 paired blind Solver evaluation\n\n" + json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    atomic_text(run_root / "final_solver_report.md.sha256", sha256_file(run_root / "final_solver_report.md") + "\n")
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "live", "all"), required=True)
    parser.add_argument("--run-root", type=Path, default=OUTPUT_RUN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    if args.phase in {"preflight", "all"}:
        print(json.dumps(preflight(run_root), ensure_ascii=False, indent=2), flush=True)
    if args.phase in {"live", "all"}:
        if not args.live:
            raise RuntimeError("solver_live_requires_explicit_live")
        print(json.dumps(live(run_root, args.gpu_memory_utilization), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
