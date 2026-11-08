"""Run B0/B1/B2 over one frozen stability batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..vllm_backends import VLLMSolverBackend
from .solver_runner import (
    OUTPUT_MODE_CONTENT_HIDDEN,
    OUTPUT_MODE_CONTENT_VISIBLE,
    OUTPUT_MODE_LABEL,
    load_permutations,
    load_result_index,
    load_sample_candidates,
    run_requests,
    summarize,
)


TOTAL_CAP = 2_232
STAGES = (("B0", OUTPUT_MODE_LABEL), ("B1", OUTPUT_MODE_CONTENT_VISIBLE), ("B2", OUTPUT_MODE_CONTENT_HIDDEN))
BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
FORBIDDEN_MODEL = Path("/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/models/merged")
EXPECTED_BASE_HASHES = {
    "config_sha256": "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
    "model_index_sha256": "6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2",
    "tokenizer_json_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_identity(model_path: Path, tokenizer_path: Path) -> dict[str, Any]:
    """Fail closed unless the requested assets are the frozen raw base model."""
    model = model_path.resolve()
    tokenizer = tokenizer_path.resolve()
    if model != BASE_MODEL.resolve():
        raise RuntimeError(f"non_base_solver_model:{model}; expected:{BASE_MODEL.resolve()}")
    if model == FORBIDDEN_MODEL.resolve():
        raise RuntimeError("forbidden_archived_stage2_solver_model")
    if tokenizer != BASE_MODEL.resolve():
        raise RuntimeError(f"non_base_tokenizer:{tokenizer}; expected:{BASE_MODEL.resolve()}")
    required = (model / "config.json", model / "tokenizer.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"base_identity_files_missing:{missing}")
    model_index = model / "model.safetensors.index.json"
    if not model_index.is_file():
        raise FileNotFoundError(f"base_model_index_missing:{model_index}")
    identity = {
        "model_path": str(model),
        "tokenizer_path": str(tokenizer),
        "config_sha256": _sha256(model / "config.json"),
        "model_index_sha256": _sha256(model_index),
        "tokenizer_json_sha256": _sha256(model / "tokenizer.json"),
        "forbidden_solver_model": str(FORBIDDEN_MODEL.resolve()),
    }
    observed = {key: identity[key] for key in EXPECTED_BASE_HASHES}
    if observed != EXPECTED_BASE_HASHES:
        raise RuntimeError(f"base_model_hash_mismatch:{observed}")
    return identity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--solver-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--smoke", action="store_true", help="run one candidate per question type, resumable into the full batch")
    return parser


def _smoke_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for qtype in ("single_choice", "multi_select", "ordering"):
        candidate = next((row for row in candidates if str(row.get("question_type")) == qtype), None)
        if candidate is None:
            raise RuntimeError(f"smoke_candidate_missing:{qtype}")
        selected.append(candidate)
    return selected


def execute(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    for name in ("preflight_report.json", "sample_manifest.jsonl", "permutation_manifest.jsonl"):
        if not (run_root / name).is_file():
            raise FileNotFoundError(f"missing_stability_manifest:{run_root / name}")
    if not args.solver_model.exists() or not args.tokenizer.exists():
        raise FileNotFoundError("solver_model_or_tokenizer_not_found")
    base_identity = _base_identity(args.solver_model, args.tokenizer)
    candidates = load_sample_candidates(run_root)
    if args.smoke:
        candidates = _smoke_candidates(candidates)
    permutations = load_permutations(run_root)
    cyclic_count = sum(sum(bool(row.get("is_cyclic")) for row in permutations.get(str(candidate["candidate_id"]), [])) for candidate in candidates)
    expected_requests = cyclic_count * 2 * len(STAGES)
    if expected_requests > TOTAL_CAP:
        raise RuntimeError(f"generation_hard_cap_exceeded:{expected_requests}/{TOTAL_CAP}")
    manifest_sha = sha256_file(run_root / "permutation_manifest.jsonl")
    ledger = RequestLedger(run_root, manifest_sha, "local")
    result_index = load_result_index(run_root)
    resumed_request_count = len(result_index)
    backend = None
    stages: dict[str, Any] = {}
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        backend = VLLMSolverBackend(
            model_path=args.solver_model,
            tokenizer_path=args.tokenizer,
            max_model_len=16_384,
            temperature=0.7,
            top_p=0.9,
            max_tokens=32,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        for stage, output_mode in STAGES:
            rows = run_requests(
                run_root=run_root,
                candidates=candidates,
                permutations=permutations,
                backend=backend,
                tokenizer=tokenizer,
                ledger=ledger,
                output_mode=output_mode,
                include_broader=False,
                cyclic_only=True,
                result_index=result_index,
            )
            stages[stage] = summarize(rows)
            atomic_json(run_root / f"{stage.lower()}_results.json", {"stage": stage, "rows": rows})
            atomic_json(run_root / f"{stage.lower()}_summary.json", stages[stage])
    finally:
        if backend is not None:
            backend.close()
    accounting = ledger.summary()
    if accounting["request_count"] > TOTAL_CAP:
        raise RuntimeError(f"ledger_budget_postcondition_failed:{accounting['request_count']}/{TOTAL_CAP}")
    report = {
        "status": "smoke_completed" if args.smoke else "completed",
        "probe_version": "h4_position_permutation_ab_stability_batches_v1",
        "run_root": str(run_root),
        "manifest_sha256": manifest_sha,
        "candidate_count": len(candidates),
        "smoke": bool(args.smoke),
        "cyclic_permutation_count": cyclic_count,
        "expected_requests": expected_requests,
        "stages": stages,
        "request_accounting": accounting,
        "content_responder_prompt_version": "h4_position_ab_content_responder_v2",
        "solver_checkpoint": base_identity,
        "old_results_reused": False,
        "old_ledger_reused": False,
        "resumed_request_count_before_run": resumed_request_count,
    }
    atomic_json(run_root / "stability_solver_report.json", report)
    atomic_text(run_root / "stability_solver_summary.md", "# H4 stability B0/B1/B2 Solver run\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(execute(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
