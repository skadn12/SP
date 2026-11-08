"""Run trained M_clean_policy and M_inclusive_policy on E1Q1 primary QA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_round0_b0_k3_difficulty_v1.contract import build_prompt
from ..h4_round0_b0_k3_difficulty_v1.gate import PROGRAM_PASS, build_solver_contract, gate_candidate
from ..h4_round0_b0_k3_difficulty_v1.io_utils import load_json, load_jsonl, write_once_json, write_once_jsonl
from ..h4_round0_b0_k3_difficulty_v1.run import assert_gpu_idle
from ..h4_round0_b0_k3_difficulty_v1 import MAX_MODEL_LEN, MAX_TOKENS, ROLLOUTS, SINGLE_TEMPERATURE, STRUCTURED_TEMPERATURE, TOP_P, candidate_request_count
from ..h4_round1_paired_blind_solver_eval_v1 import run as base
from ..h4_round1_paired_blind_solver_eval_v1.run import _identity_preflight, _run_model
from ..vllm_backends import VLLMSolverBackend
from . import BASE_MODEL, FILTER_RUN, GENERATION_RUN, IDENTITY_REPORT, MODELS, OUTPUT_RUN, PROBE_VERSION


def _write(path: Path, value: Any) -> None:
    if path.suffix == ".jsonl":
        write_once_jsonl(path, value)
    else:
        write_once_json(path, value)


def _verify(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing_source:{path}")
    sidecar = path.with_name(path.name + ".sha256")
    observed = sha256_file(path)
    if sidecar.is_file() and sidecar.read_text(encoding="utf-8").strip().split()[0] != observed:
        raise RuntimeError(f"source_sha_mismatch:{path}")
    return observed


def _inputs() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    source = FILTER_RUN / "e1q1_primary_solver_pool.jsonl"
    _verify(source)
    rows = load_jsonl(source)
    if not rows:
        raise RuntimeError("empty_e1q1_primary_pool")
    pools: dict[tuple[str, str], dict[str, Any]] = {}
    dialogues: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        root = str(row.get("root_id") or "")
        cid = str(row.get("candidate_id") or "")
        if not cid.startswith(root + ":R01-R02-feedback:E1Q1_FULL_QA_FEEDBACK_COMPOSITE:"):
            raise RuntimeError(f"candidate_scope_mismatch:{cid}")
        branch = "feedback"
        scoped_row = dict(row)
        scoped_row["branch"] = branch
        pools.setdefault((branch, root), {})[cid] = scoped_row
        dpath = GENERATION_RUN / root / "dialogue_source.json"
        _verify(dpath)
        dialogues[(branch, root)] = load_json(dpath)
    rows = [{**dict(row), "branch": "feedback"} for row in rows]
    return rows, pools, dialogues, {"pool_path": str(source.resolve()), "pool_sha256": sha256_file(source)}


def _build_inputs(rows, pools, dialogues, tokenizer):
    gates = {}
    candidates = []
    for row in rows:
        root, cid = str(row["root_id"]), str(row["candidate_id"])
        candidate = pools[("feedback", root)][cid]
        dialogue = dialogues[("feedback", root)]
        gate = gate_candidate(candidate, dialogue, tokenizer, root_id=root)
        scoped_path = GENERATION_RUN / root / "candidates_valid_scoped.json"
        gate.update({"candidate_pool_path": str(scoped_path.resolve()), "candidate_pool_sha256": sha256_file(scoped_path), "dialogue_path": str((GENERATION_RUN / root / "dialogue_source.json").resolve()), "dialogue_sha256": sha256_file(GENERATION_RUN / root / "dialogue_source.json"), "branch": "feedback"})
        gates[cid] = gate
        candidates.append(candidate)
    bad = [cid for cid, gate in gates.items() if gate.get("gate_status") != PROGRAM_PASS]
    if bad:
        raise RuntimeError(f"e1q1_program_gate_failed:{bad[:10]}")
    return candidates, gates


def _plan(rows, protocol):
    qcounts = {}
    for row in rows:
        q = str(row.get("question_type") or "")
        qcounts[q] = qcounts.get(q, 0) + 1
    request_count = sum(candidate_request_count(row) for row in rows)
    if any(qcounts.get(q, 0) == 0 for q in ("single_choice", "multi_select", "ordering")):
        raise RuntimeError(f"e1q1_solver_plan_missing_question_type:{qcounts}")
    return {"solver_plan_version": f"{PROBE_VERSION}_plan_v1", "input_pool": str(FILTER_RUN / "e1q1_primary_solver_pool.jsonl"), "candidate_count": len(rows), "root_count": len({str(x["root_id"]) for x in rows}), "branch": "feedback", "question_type_counts": dict(sorted(qcounts.items())), "per_model_request_count": request_count, "all_model_request_count": request_count * len(MODELS), "requests_per_candidate": "candidate_position_count * 2_context_modes * 3_rollouts", "position_counts": {"single_choice": {"fixed": 4}, "multi_select": {"min": 4, "max": 6, "source": "candidate"}, "ordering": {"min": 4, "max": 6, "source": "candidate"}}, "rollouts_per_position": ROLLOUTS, "context_modes": ["no_context", "full_context"], "temperatures": {"single_choice": SINGLE_TEMPERATURE, "multi_select": STRUCTURED_TEMPERATURE, "ordering": STRUCTURED_TEMPERATURE}, "max_model_len": MAX_MODEL_LEN, "max_tokens": MAX_TOKENS, "top_p": TOP_P, "protocol_sha256": sha256_bytes(canonical_json(protocol).encode()), "trained_models_only": True, "dev_test_accessed": False}


def preflight(run_root: Path = OUTPUT_RUN) -> dict[str, Any]:
    run_root = run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()) and not (run_root / "run_identity.json").is_file():
        raise RuntimeError(f"output_exists_or_nonempty:{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    _write(run_root / "run_identity.json", {"probe_version": PROBE_VERSION, "offline_preflight": True, "solver_calls": 0, "memorax_calls": 0, "qa_writer_calls": 0})
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True)
    rows, pools, dialogues, source = _inputs()
    identity, models = _identity_preflight()
    candidates, gates = _build_inputs(rows, pools, dialogues, tokenizer)
    protocol = build_solver_contract(tokenizer, model_path=BASE_MODEL, tokenizer_path=BASE_MODEL)
    plan = _plan(candidates, protocol)
    _write(run_root / "model_identity_report.json", identity)
    _write(run_root / "solver_protocol.json", protocol)
    _write(run_root / "solver_plan.json", plan)
    _write(run_root / "candidate_manifest.jsonl", candidates)
    _write(run_root / "program_gate_results.jsonl", list(gates.values()))
    report = {"status": "PASS", "probe_version": PROBE_VERSION, "input_pool": source, "candidate_count": len(candidates), "root_count": len({str(x["root_id"]) for x in candidates}), "models": list(models), "trained_models_only": True, "plan_sha256": sha256_file(run_root / "solver_plan.json"), "solver_protocol_sha256": sha256_file(run_root / "solver_protocol.json"), "program_gate_pass_count": sum(x.get("gate_status") == PROGRAM_PASS for x in gates.values()), "program_gate_reject_count": sum(x.get("gate_status") != PROGRAM_PASS for x in gates.values()), "dev_test_accessed": False, "solver_started": False}
    _write(run_root / "preflight_report.json", report)
    return report


def live(run_root: Path = OUTPUT_RUN, gpu_memory_utilization: float = .80) -> dict[str, Any]:
    run_root = run_root.resolve()
    pre = load_json(run_root / "preflight_report.json")
    if pre.get("status") != "PASS":
        raise RuntimeError("solver_preflight_required")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True)
    rows, pools, dialogues, _ = _inputs()
    identity, models = _identity_preflight()
    candidates, gates = _build_inputs(rows, pools, dialogues, tokenizer)
    protocol = load_json(run_root / "solver_protocol.json")
    # Reuse the frozen implementation but constrain it to the E1Q1 feedback branch.
    base.BRANCHES = ("feedback",)
    model_results = {}
    from ..h4_round0_b0_k3_difficulty_v1.io_utils import RunLock
    with RunLock(run_root):
        for model_name in MODELS:
            result = _run_model(model_name=model_name, model_identity=models[model_name], rows=rows, gates=gates, pools=pools, dialogues=dialogues, tokenizer=tokenizer, protocol=protocol, run_root=run_root, gpu_memory_utilization=gpu_memory_utilization)
            model_results[model_name] = result["model_summary"]
    comparison = {"candidate_count": len(rows), "branch": "feedback", "models": {name: summary["summary"] for name, summary in model_results.items()}, "trained_models_only": True, "input_pool": str(FILTER_RUN / "e1q1_primary_solver_pool.jsonl")}
    _write(run_root / "trained_solver_comparison.json", comparison)
    atomic_text(run_root / "trained_solver_comparison.md", "# E1Q1 trained Solver evaluation\n\n" + json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    atomic_text(run_root / "trained_solver_comparison.md.sha256", sha256_file(run_root / "trained_solver_comparison.md") + "  trained_solver_comparison.md\n")
    final = {"status": "E1Q1_TRAINED_SOLVER_COMPLETE", "candidate_count": len(rows), "models": list(MODELS), "model_summaries": model_results, "candidate_evaluations": sum(summary["summary"]["candidate_count"] for summary in model_results.values()), "request_count": sum(branch["request_count"] for summary in model_results.values() for branch in summary["branches"].values()), "trained_models_only": True, "base_model_used_as_solver": False, "dev_test_accessed": False, "memorax_calls": 0, "semantic_auditor_calls": 0, "next_step": "offline compare E1Q1 primary against the frozen E1Q0 Feedback baseline"}
    _write(run_root / "final_trained_solver_report.json", final)
    atomic_text(run_root / "final_trained_solver_report.md", "# E1Q1 trained Solver evaluation\n\n" + json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    atomic_text(run_root / "final_trained_solver_report.md.sha256", sha256_file(run_root / "final_trained_solver_report.md") + "  final_trained_solver_report.md\n")
    return final


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("preflight", "live", "all"), required=True); parser.add_argument("--run-root", type=Path, default=OUTPUT_RUN); parser.add_argument("--gpu-memory-utilization", type=float, default=.80); parser.add_argument("--live", action="store_true"); args = parser.parse_args()
    if args.phase in {"preflight", "all"}: print(json.dumps(preflight(args.run_root), ensure_ascii=False, indent=2), flush=True)
    if args.phase in {"live", "all"}:
        if not args.live: raise RuntimeError("solver_live_requires_explicit_live")
        print(json.dumps(live(args.run_root, args.gpu_memory_utilization), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
