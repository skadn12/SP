"""Preflight, M0 reuse audit, adapter evaluation, and Dev aggregation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core import atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_round0_b0_k3_difficulty_v1.contract import build_prompt, dialogue_rows, position_variants
from ..h4_round0_b0_k3_difficulty_v1.difficulty import summarize_candidate_result
from ..h4_round0_b0_k3_difficulty_v1.io_utils import (
    RunLock,
    load_json,
    load_jsonl,
    verify_sha_sidecar,
    write_once_json,
    write_once_jsonl,
)
from ..h4_round0_b0_k3_difficulty_v1.request_ledger import Round0RequestLedger
from ..h4_round0_b0_k3_difficulty_v1.solver import (
    evaluate_candidate,
    load_or_recover_candidate_result,
)
from ..vllm_backends import VLLMSolverBackend
from ..h4_round0_b0_k3_difficulty_v1.gate import directory_identity
from . import (
    BASE_MODEL,
    DATA_RUN,
    DIFFICULTY_RUN,
    EXPECTED_DEV,
    EXPECTED_DEV_ROOTS,
    FORMAL_RUN,
    MAX_MODEL_LEN,
    MAX_TOKENS,
    REPO_ROOT,
    ROLLOUTS,
    VERSION,
)


MODEL_SPECS = {
    "M0": {"model": BASE_MODEL, "adapter": None, "adapter_name": "none"},
    "M_clean_policy": {"adapter": FORMAL_RUN / "clean_adapter", "adapter_name": "clean_adapter"},
    "M_inclusive_policy": {"adapter": FORMAL_RUN / "inclusive_adapter", "adapter_name": "inclusive_adapter"},
}
QTYPE_REQUESTS = {"single_choice": 24, "multi_select": 30, "ordering": 24}


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "\n")


def file_sha(path: Path) -> str:
    return sha256_file(path)


def composite_directory_sha(path: Path) -> str:
    rows = []
    for child in sorted(path.rglob("*")):
        if child.is_file() and not child.name.endswith(".sha256"):
            rows.append({"name": child.relative_to(path).as_posix(), "size": child.stat().st_size, "sha256": sha256_file(child)})
    return sha256_bytes(canonical_json(rows).encode("utf-8"))


def verify_sha(path: Path) -> str:
    verify_sha_sidecar(path)
    return sha256_file(path)


def source_snapshot(data_run: Path) -> dict[str, str]:
    names = [
        "dev_full_eval_manifest.jsonl", "dev_m0_hard_eval_manifest.jsonl",
        "m_clean_policy_train.jsonl", "m_inclusive_policy_train.jsonl",
        "m_clean_policy_training_manifest.jsonl", "m_inclusive_policy_training_manifest.jsonl",
        "training_interface_contract.json",
    ]
    return {name: verify_sha(data_run / name) for name in names}


def load_dev_inputs(data_run: Path, difficulty_run: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    dev_eval = load_jsonl(data_run / "dev_full_eval_manifest.jsonl")
    if len(dev_eval) != EXPECTED_DEV or len({str(row["candidate_id"]) for row in dev_eval}) != EXPECTED_DEV:
        raise RuntimeError(f"dev_candidate_count_mismatch:{len(dev_eval)}")
    if len({str(row["root_id"]) for row in dev_eval}) != EXPECTED_DEV_ROOTS:
        raise RuntimeError("dev_root_count_mismatch")
    difficulty_rows = load_jsonl(difficulty_run / "difficulty_manifest.jsonl")
    gate_rows = load_jsonl(difficulty_run / "program_gate_results.jsonl")
    difficulty = {str(row["candidate_id"]): dict(row) for row in difficulty_rows if row.get("split") == "dev"}
    gates = {str(row["candidate_id"]): dict(row) for row in gate_rows}
    if set(difficulty) != {str(row["candidate_id"]) for row in dev_eval}:
        raise RuntimeError("dev_difficulty_identity_mismatch")
    for row in dev_eval:
        cid = str(row["candidate_id"])
        if cid not in gates or gates[cid].get("gate_status") != "PROGRAM_PASS":
            raise RuntimeError(f"dev_gate_not_program_pass:{cid}")
        if str(row["candidate_pool_sha256"]) != str(gates[cid]["candidate_pool_sha256"]) or str(row["dialogue_sha256"]) != str(gates[cid]["dialogue_sha256"]):
            raise RuntimeError(f"dev_lineage_mismatch:{cid}")
    return dev_eval, difficulty, gates


def build_solver_plan(dev_eval: list[dict[str, Any]], *, solver_contract: Mapping[str, Any]) -> dict[str, Any]:
    counts = Counter(str(row["question_type"]) for row in dev_eval)
    by_type = {key: counts[key] * QTYPE_REQUESTS[key] for key in sorted(QTYPE_REQUESTS)}
    expected = sum(by_type.values())
    position_counts = {key: int(solver_contract["position_counts"][key]) for key in QTYPE_REQUESTS}
    formula = {
        key: {
            "candidate_count": counts[key], "positions": position_counts[key],
            "rollouts_per_position": ROLLOUTS, "context_modes": 2,
            "requests_per_candidate": QTYPE_REQUESTS[key], "request_count": by_type[key],
            "closed": QTYPE_REQUESTS[key] == position_counts[key] * ROLLOUTS * 2,
        }
        for key in sorted(QTYPE_REQUESTS)
    }
    if not all(row["closed"] for row in formula.values()) or expected != 12378:
        raise RuntimeError(f"dev_solver_plan_not_closed:{expected}:{formula}")
    return {
        "dev_solver_plan_version": f"{VERSION}_solver_plan_v1", "split": "dev",
        "candidate_count": len(dev_eval), "root_count": len({row["root_id"] for row in dev_eval}),
        "question_type_counts": dict(sorted(counts.items())), "request_count_by_question_type": by_type,
        "planned_request_count": expected, "planned_unique_prompt_count": sum(counts[key] * position_counts[key] * 2 for key in counts),
        "requests_per_candidate": QTYPE_REQUESTS, "position_counts": position_counts,
        "rollouts_per_position": ROLLOUTS, "context_modes": ["no_context", "full_context"], "formula": formula,
        "max_model_len": int(solver_contract["max_model_len"]), "max_tokens": int(solver_contract["generation_config"]["max_tokens"]),
        "temperature_by_question_type": solver_contract["temperatures"], "seed_schedule": solver_contract["seed_schedule"],
        "only_b0": True, "only_dev": True,
    }


def audit_m0_reuse(*, difficulty_run: Path, data_run: Path, dev_eval: list[dict[str, Any]], difficulty: Mapping[str, Mapping[str, Any]], gates: Mapping[str, Mapping[str, Any]], tokenizer: Any, solver_contract: Mapping[str, Any], contract_sha: str) -> dict[str, Any]:
    split_by_id = {str(row["candidate_id"]): str(row["split"]) for row in load_jsonl(difficulty_run / "difficulty_manifest.jsonl")}
    pools: dict[str, dict[str, Any]] = {}
    dialogues: dict[str, dict[str, Any]] = {}
    checked_results = 0
    checked_requests = 0
    reasons: list[str] = []
    result_hashes: dict[str, str] = {}
    for eval_row in dev_eval:
        cid, root = str(eval_row["candidate_id"]), str(eval_row["root_id"])
        digest = sha256_bytes(cid.encode("utf-8"))
        result_path = difficulty_run / "candidate_results" / f"{digest}.json"
        try:
            result_sha = verify_sha(result_path)
        except Exception as error:
            reasons.append(f"result_sha:{cid}:{error}")
            continue
        result_hashes[cid] = result_sha
        result = load_json(result_path)
        checked_results += 1
        if result.get("solver_contract_sha256") != contract_sha or result.get("candidate_pool_sha256") != gates[cid]["candidate_pool_sha256"] or result.get("dialogue_sha256") != gates[cid]["dialogue_sha256"]:
            reasons.append(f"result_identity:{cid}")
        if result.get("request_count") != QTYPE_REQUESTS[result["question_type"]] or result.get("technical_failure_count") != 0:
            reasons.append(f"result_count_or_technical:{cid}")
        if root not in pools:
            pool_path = Path(gates[cid]["candidate_pool_path"]); dialogue_path = Path(gates[cid]["dialogue_path"])
            if sha256_file(pool_path) != gates[cid]["candidate_pool_sha256"] or sha256_file(dialogue_path) != gates[cid]["dialogue_sha256"]:
                reasons.append(f"source_sha:{cid}"); continue
            pools[root] = {str(row["candidate_id"]): row for row in load_json(pool_path)}
            dialogues[root] = load_json(dialogue_path)
        candidate = pools[root][cid]
        dialogue = dialogue_rows(dialogues[root])
        variants = position_variants(candidate)
        expected = {(str(v["variant_id"]), mode): (build_prompt(tokenizer, candidate, v, dialogue, context_mode=mode), v) for v in variants for mode in ("no_context", "full_context")}
        for request in result.get("requests") or []:
            checked_requests += 1
            key = (str(request["variant_id"]), str(request["context_mode"]))
            if key not in expected:
                reasons.append(f"request_key:{cid}:{key}"); continue
            prompt, variant = expected[key]
            prompt_sha = sha256_bytes(prompt.encode("utf-8"))
            if request.get("prompt_sha256") != prompt_sha:
                reasons.append(f"prompt_sha:{cid}:{request.get('request_id')}")
            if request.get("parser_version") != ("boxed_letter_v1" if result["question_type"] == "single_choice" else "structured_qa_box_v1"):
                reasons.append(f"parser:{cid}:{request.get('request_id')}")
            meta = request.get("backend_metadata") or {}
            expected_temp = float(solver_contract["temperatures"][result["question_type"]])
            for field, expected_value in (("temperature", expected_temp), ("top_p", 0.9), ("max_tokens", 32)):
                if float(meta.get(field, -1)) != expected_value:
                    reasons.append(f"generation:{cid}:{request.get('request_id')}:{field}")
    return {
        "m0_reuse_audit_version": f"{VERSION}_m0_reuse_audit_v1", "status": "PASS" if checked_results == EXPECTED_DEV and not reasons else "FAIL",
        "reuse_allowed": checked_results == EXPECTED_DEV and not reasons, "m0_new_solver_generation": 0 if checked_results == EXPECTED_DEV and not reasons else None,
        "difficulty_run": str(difficulty_run), "difficulty_run_solver_contract_sha256": contract_sha,
        "dev_candidate_count": len(dev_eval), "checked_result_count": checked_results, "checked_request_count": checked_requests,
        "expected_request_count": 12378, "result_sha256": result_hashes, "mismatch_reasons": reasons[:100],
        "m0_source_result_scope": "difficulty_run_candidate_results_dev_only",
    }


def adapter_identity(*, name: str, adapter_path: Path | None, base_identity: Mapping[str, Any], formal_run: Path) -> dict[str, Any]:
    if adapter_path is None:
        return {"model_name": name, "adapter_name": "none", "adapter_enabled": False, "base_model_identity": base_identity}
    config = adapter_path / "adapter_config.json"; state = adapter_path / "adapter_model.pt"
    if not config.is_file() or not state.is_file():
        raise RuntimeError(f"adapter_files_missing:{adapter_path}")
    report = load_json(formal_run / ("clean_training_report.json" if "clean" in name else "inclusive_training_report.json"))
    if report.get("status") != "PASS" or not report.get("adapter_saved"):
        raise RuntimeError(f"formal_adapter_report_not_pass:{name}")
    config_sha, state_sha = sha256_file(config), sha256_file(state)
    if report.get("adapter_config_sha256") != config_sha or report.get("adapter_state_sha256") != state_sha:
        raise RuntimeError(f"adapter_sha_drift:{name}")
    cfg = load_json(config)
    if int(cfg.get("max_length")) != 8192 or int(cfg.get("lora_rank")) != 16 or int(cfg.get("lora_alpha")) != 32 or len(cfg.get("target_modules") or []) != 7:
        raise RuntimeError(f"adapter_config_invalid:{name}")
    return {
        "model_name": name, "adapter_name": name, "adapter_enabled": True,
        "adapter_path": str(adapter_path), "adapter_config_sha256": config_sha, "adapter_state_sha256": state_sha,
        "adapter_composite_sha256": composite_directory_sha(adapter_path), "base_model_identity": base_identity,
        "lora_config": cfg, "merged_linear_modules_expected": 252,
    }


def merge_adapter(*, adapter: Path, output: Path, log: Path) -> dict[str, Any]:
    if output.exists():
        if not (output / "config.json").is_file() or not any(output.glob("*.safetensors")):
            raise RuntimeError(f"partial_merged_model_exists:{output}")
        return {"output": str(output), "log": str(log), "reused": True, "model_identity": directory_identity(output)}
    command = [os.environ.get("PYTHON", "python"), "scripts/merge_custom_lora_adapter.py", "--model", str(BASE_MODEL), "--tokenizer", str(BASE_MODEL), "--adapter-dir", str(adapter), "--output-dir", str(output), "--dtype", "bfloat16"]
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"adapter_merge_failed:{adapter}:{process.returncode}")
    return {"output": str(output), "log": str(log), "reused": False, "model_identity": directory_identity(output)}


def raw_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(path, text)


def run_model(*, model_name: str, model_path: Path, run_root: Path, dev_eval: list[dict[str, Any]], gates: Mapping[str, Mapping[str, Any]], tokenizer: Any, contract_sha: str, contract: Mapping[str, Any], identity: Mapping[str, Any], split_name: str = "dev", expected_candidate_count: int | None = None) -> dict[str, Any]:
    expected_count = EXPECTED_DEV if expected_candidate_count is None else int(expected_candidate_count)
    scope = run_root / ("m0_" + split_name if model_name == "M0" else "clean_" + split_name if model_name == "M_clean_policy" else "inclusive_" + split_name)
    scope.mkdir(parents=True, exist_ok=True)
    for name in ("raw_prompts", "raw_responses", "request_ledger", "candidate_results"):
        (scope / name).mkdir(parents=True, exist_ok=True)
    backend = VLLMSolverBackend(model_path=model_path, tokenizer_path=Path(contract["tokenizer"]), max_model_len=MAX_MODEL_LEN, temperature=float(contract["temperatures"]["single_choice"]), top_p=0.9, max_tokens=MAX_TOKENS, gpu_memory_utilization=0.80)
    ledger = Round0RequestLedger(scope, solver_contract_sha256=contract_sha)
    sources: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    completed: list[dict[str, Any]] = []
    try:
        for eval_row in sorted(dev_eval, key=lambda row: str(row["candidate_id"])):
            cid, root = str(eval_row["candidate_id"]), str(eval_row["root_id"])
            gate = gates[cid]
            if root not in sources:
                pool_path, dialogue_path = Path(gate["candidate_pool_path"]), Path(gate["dialogue_path"])
                if sha256_file(pool_path) != gate["candidate_pool_sha256"] or sha256_file(dialogue_path) != gate["dialogue_sha256"]:
                    raise RuntimeError(f"eval_source_drift:{cid}")
                sources[root] = ({str(row["candidate_id"]): row for row in load_json(pool_path)}, load_json(dialogue_path))
            candidate, expanded = sources[root][0][cid], sources[root][1]
            result_path = scope / "candidate_results" / f"{sha256_bytes(cid.encode())}.json"
            if result_path.is_file():
                verify_sha(result_path)
                result = load_json(result_path)
            else:
                result = evaluate_candidate(candidate=candidate, expanded=expanded, gate_row=gate, tokenizer=tokenizer, backend=backend, ledger=ledger, solver_contract_sha256=contract_sha)
                write_once_json(result_path, result)
            for req in result.get("requests") or []:
                req_id = str(req["request_id"])
                digest = hashlib.sha256(req_id.encode()).hexdigest()
                variant = next(v for v in position_variants(candidate) if str(v["variant_id"]) == str(req["variant_id"]))
                prompt = build_prompt(tokenizer, candidate, variant, dialogue_rows(expanded), context_mode=str(req["context_mode"]))
                raw_write(scope / "raw_prompts" / f"{digest}.txt", prompt)
                raw_write(scope / "raw_responses" / f"{digest}.json", json.dumps({"request_id": req_id, "raw_output": req.get("raw_output"), "backend_metadata": req.get("backend_metadata"), "parse_status": req.get("parse_status"), "scored": req.get("scored"), "correct": req.get("correct")}, ensure_ascii=False, sort_keys=True) + "\n")
            completed.append(result)
    finally:
        backend.close()
    expected_requests = sum(QTYPE_REQUESTS[str(row["question_type"])] for row in dev_eval)
    observed_requests = sum(int(row.get("request_count", 0)) for row in completed)
    technical = sum(int(row.get("technical_failure_count", 0)) for row in completed)
    candidate_rows = []
    eval_metadata = {str(row["candidate_id"]): row for row in dev_eval}
    for result in completed:
        summary = summarize_candidate_result(result)
        gate = gates[str(result["candidate_id"])]
        metadata = eval_metadata[str(result["candidate_id"])]
        candidate_rows.append({**summary, "model_name": model_name, "candidate_id": result["candidate_id"], "root_id": result["root_id"], "question_type": result["question_type"], "generation_source": metadata.get("generation_source"), "position_mechanism": metadata.get("position_mechanism", "unknown"), "evidence_status": metadata.get("evidence_status", "unknown"), "request_count": result["request_count"], "technical_failure_count": result["technical_failure_count"], "candidate_pool_sha256": result["candidate_pool_sha256"], "dialogue_sha256": result["dialogue_sha256"], "solver_contract_sha256": result["solver_contract_sha256"]})
    write_jsonl(scope / "candidate_results.jsonl", completed)
    write_jsonl(scope / "difficulty_manifest.jsonl", candidate_rows)
    report = {
        "dev_model_report_version": f"{VERSION}_{model_name}_{split_name}_report_v1", "status": "PASS" if len(completed) == expected_count and observed_requests == expected_requests else "FAIL", "model_name": model_name, "identity": identity, "candidate_count": len(completed), "request_count": observed_requests, "expected_request_count": expected_requests, "technical_failure_count": technical, "candidate_results_path": str(scope / "candidate_results.jsonl"), "difficulty_manifest_path": str(scope / "difficulty_manifest.jsonl"), "scope": str(scope),
    }
    write_json(scope / "model_execution_report.json", report)
    return {"report": report, "candidate_rows": candidate_rows, "results": completed}


def aggregate_model(model_name: str, rows: list[dict[str, Any]], difficulty: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def weighted(context: str) -> float:
        correct = sum(int(row["position_diagnostics"][context]["correct_count"]) for row in rows)
        total = sum(int(row["position_diagnostics"][context]["total_count"]) for row in rows)
        return correct / total if total else None
    def root_equal(context: str) -> float:
        roots: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            value = row["position_diagnostics"][context]["accuracy"]
            if value is not None: roots[str(row["root_id"])].append(float(value))
        values = [sum(v) / len(v) for v in roots.values() if v]
        return sum(values) / len(values) if values else None
    def position_mean(context: str, which: str) -> float | None:
        vals=[]
        for row in rows:
            values=list((row["position_diagnostics"][context].get("position_accuracy") or {}).values())
            if values: vals.append(max(values) if which == "best" else min(values))
        return sum(vals)/len(vals) if vals else None
    return {
        "model_name": model_name, "candidate_count": len(rows),
        "candidate_weighted": {"full_context_accuracy": weighted("full_context"), "no_context_accuracy": weighted("no_context"), "context_gain": (weighted("full_context") - weighted("no_context")) if weighted("full_context") is not None and weighted("no_context") is not None else None},
        "root_equal": {"full_context_accuracy": root_equal("full_context"), "no_context_accuracy": root_equal("no_context"), "context_gain": (root_equal("full_context") - root_equal("no_context")) if root_equal("full_context") is not None and root_equal("no_context") is not None else None},
        "best_position_accuracy_mean": {"full_context": position_mean("full_context", "best"), "no_context": position_mean("no_context", "best")},
        "worst_position_accuracy_mean": {"full_context": position_mean("full_context", "worst"), "no_context": position_mean("no_context", "worst")},
        "canonical_flip_rate_mean": {context: sum(float(row["canonical_flip_rate"][context]) for row in rows) / len(rows) for context in ("full_context", "no_context")},
        "difficulty_bucket_counts": dict(sorted(Counter(str(row["difficulty_bucket"]) for row in rows).items())),
        "technical_failure_count": sum(int(row.get("technical_failure_count", 0)) for row in rows),
    }


def dimension_stats(rows: list[dict[str, Any]], dimension: str, difficulty: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if dimension == "context_token_bucket":
            key = str(row.get("context_token_bucket", "unknown"))
        else:
            key = str(row.get(dimension, "unknown"))
        groups[key].append(row)
    # `aggregate_model` keeps the difficulty mapping in its public contract for
    # provenance compatibility, although the aggregate metrics are derived from
    # the already-materialized candidate rows.
    difficulty = difficulty or {}
    return {key: aggregate_model(f"slice:{dimension}:{key}", values, difficulty) for key, values in sorted(groups.items())}


def hard_transitions(m0_rows: list[dict[str, Any]], model_rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    hard = {str(row["candidate_id"]): row for row in m0_rows if row.get("difficulty_bucket") == "hard_candidate"}
    output = {"hard_count": len(hard), "models": {}}
    for name, rows in model_rows.items():
        by_id = {str(row["candidate_id"]): row for row in rows}
        transitions = Counter(str(by_id[cid].get("difficulty_bucket", "missing")) for cid in hard if cid in by_id)
        output["models"][name] = {"transitions": dict(sorted(transitions.items())), "missing_count": sum(cid not in by_id for cid in hard), "hard_recovery_rate": (transitions["easy"] + transitions["full_context_resolved"]) / len(hard) if hard else None}
    return output


__all__ = ["MODEL_SPECS", "QTYPE_REQUESTS", "adapter_identity", "aggregate_model", "audit_m0_reuse", "build_solver_plan", "dimension_stats", "hard_transitions", "load_dev_inputs", "merge_adapter", "run_model", "source_snapshot", "write_json", "write_jsonl"]
