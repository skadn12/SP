"""Replay only frozen hybrid multi-select candidates at temperature 0.3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..vllm_backends import VLLMSolverBackend
from .difficulty_filter import apply_hybrid_policy
from .solver_adapter import (
    MULTI_SELECT_SOLVER_TEMPERATURE,
    ORDERING_SOLVER_TEMPERATURE,
    evaluate_candidates,
)


DEFAULT_INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_178392_h4_qa_mining_hybrid_generation_v1_phase0_3"
)
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_178392_h4_qa_mining_hybrid_generation_v1_"
    "multi_select_temp03_replay_1"
)
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_SOLVER_MODEL = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "05_stage2_repair_topup_from_stage1/training/models/merged"
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_run.exists():
        raise RuntimeError(f"output_run_already_exists:{args.output_run}")
    source_manifest_path = args.input_run / "manifest.json"
    candidate_path = args.input_run / "candidate_pool.json"
    basic_path = args.input_run / "basic_validation.json"
    for path in (source_manifest_path, candidate_path, basic_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_manifest = _load(source_manifest_path)
    dialogue_path = Path(str(source_manifest.get("source_dialogue") or ""))
    if not dialogue_path.is_file():
        raise FileNotFoundError(dialogue_path)
    expanded = _load(dialogue_path)
    candidates = [row for row in _load(candidate_path)
                  if row.get("question_type") == "multi_select"]
    if not candidates:
        raise RuntimeError("no_multi_select_candidates")
    basic_all = _load(basic_path)
    basic_rows = {str(row["candidate_id"]): basic_all.get(str(row["candidate_id"]), {})
                  for row in candidates}
    manifest = {
        "manifest_version": "h4_hybrid_multi_select_temp03_replay_v1",
        "input_run": str(args.input_run),
        "input_manifest_sha256": sha256_file(source_manifest_path),
        "candidate_pool_sha256": sha256_file(candidate_path),
        "source_dialogue": str(dialogue_path),
        "source_dialogue_sha256": sha256_file(dialogue_path),
        "candidate_count": len(candidates),
        "question_type": "multi_select",
        "temperature": MULTI_SELECT_SOLVER_TEMPERATURE,
        "ordering_temperature_unchanged": ORDERING_SOLVER_TEMPERATURE,
        "position_variants": 5,
        "modes": ["no_context", "full_context"],
        "rollouts_per_position": 5,
        "expected_local_solver_requests": len(candidates) * 5 * 2 * 5,
        "challenger_api_calls": 0,
        "candidate_generation_calls": 0,
        "production_code_modified": False,
        "upstream_run_modified": False,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest).encode())
    args.output_run.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output_run / "manifest.json", manifest)
    atomic_json(args.output_run / "candidate_pool.json", candidates)
    ledger = RequestLedger(args.output_run, manifest["manifest_sha256"], "live")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    backend = None
    solver_error = None
    results: list[dict[str, Any]] = []
    try:
        backend = VLLMSolverBackend(
            model_path=args.solver_model, tokenizer_path=args.tokenizer,
            max_model_len=16_384, temperature=0.7, top_p=0.9,
            max_tokens=32, gpu_memory_utilization=0.80)
        results = evaluate_candidates(
            candidates=candidates, expanded=expanded, run_root=args.output_run,
            ledger=ledger, tokenizer=tokenizer, backend=backend)
    except Exception as error:
        solver_error = f"{type(error).__name__}:{error}"
    finally:
        if backend is not None:
            backend.close()
    atomic_json(args.output_run / "solver_results.json", {
        "solver_error": solver_error, "results": results,
    })
    difficulty = apply_hybrid_policy({"results": results}, candidates, basic_rows)
    atomic_json(args.output_run / "difficulty.json", difficulty)
    rows = difficulty.get("candidate_summaries") or []
    position_patterns = {
        str(row.get("candidate_id")): {
            mode: [
                item.get("modes", {}).get(mode, {}).get("accuracy")
                for item in row.get("variants") or []
            ] for mode in ("no_context", "full_context")
        } for row in rows
    }
    completed = ledger.completed()
    temperature_values = sorted({
        float((state.get("response_metadata") or {}).get("temperature"))
        for state in completed
        if (state.get("response_metadata") or {}).get("temperature") is not None
    })
    report = {
        "report_version": "h4_hybrid_multi_select_temp03_replay_report_v1",
        "status": "completed" if solver_error is None else "solver_failed",
        "input_run": str(args.input_run), "output_run": str(args.output_run),
        "candidate_count": len(candidates),
        "solver_request_count": len(completed), "expected_solver_request_count": 400,
        "temperature_requested": MULTI_SELECT_SOLVER_TEMPERATURE,
        "temperature_values_observed": temperature_values,
        "difficulty_bucket_counts": difficulty.get("bucket_counts"),
        "position_sensitive_count": sum(bool(row.get("position_sensitive")) for row in rows),
        "hard_candidate_count": sum(bool(row.get("hard_candidate")) for row in rows),
        "hard_and_position_stable_count": sum(
            bool(row.get("hard_candidate")) and not bool(row.get("position_sensitive"))
            for row in rows),
        "position_accuracy_patterns": position_patterns,
        "solver_error": solver_error,
        "challenger_api_call_count": 0,
        "upstream_run_modified": False,
    }
    atomic_json(args.output_run / "report.json", report)
    atomic_text(args.output_run / "summary.md", "\n".join([
        "# Multi-select temperature 0.3 replay", "",
        f"Status: **{report['status']}**",
        f"Frozen candidates: {len(candidates)}",
        f"Local Solver requests: {len(completed)}/400",
        f"Observed temperatures: `{temperature_values}`",
        f"Difficulty: `{json.dumps(report['difficulty_bucket_counts'], sort_keys=True)}`",
        f"Position-sensitive: {report['position_sensitive_count']}",
        f"Hard and position-stable: {report['hard_and_position_stable_count']}", "",
        "No Challenger API or candidate generation call was made.",
    ]) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-run", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-run", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))

