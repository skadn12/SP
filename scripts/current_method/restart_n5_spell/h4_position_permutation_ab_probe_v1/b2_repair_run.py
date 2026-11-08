"""Run a repaired B1/B2 content Solver condition in isolation.

The source probe is treated as immutable.  This runner consumes a copied,
hash-checked sample/permutation manifest and writes all new ledger, prompts,
raw responses, and scored rows below its own run root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..vllm_backends import VLLMSolverBackend
from .solver_runner import (
    CONTENT_RESPONDER_PROMPT_VERSION,
    OUTPUT_MODE_CONTENT_HIDDEN,
    OUTPUT_MODE_CONTENT_VISIBLE,
    load_permutations,
    load_result_index,
    load_sample_candidates,
    run_requests,
    summarize,
)


TOTAL_CAP = 2_232
PROBE_VERSION = "h4_position_permutation_ab_probe_v1_{stage}_repair"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True,
                        help="immutable live_4 probe whose manifests were copied into run-root")
    parser.add_argument("--stage", choices=("b1", "b2"), default="b2",
                        help="content-visible B1 or content-hidden B2 (default: b2)")
    parser.add_argument("--solver-model", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--smoke", action="store_true",
                        help="run one cyclic multi-select and one cyclic ordering candidate")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _select_smoke(candidates: list[dict[str, Any]], permutations: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    selected: list[dict[str, Any]] = []
    reduced: dict[str, list[dict[str, Any]]] = {}
    for qtype in ("multi_select", "ordering"):
        candidate = next((row for row in candidates if str(row.get("question_type")) == qtype), None)
        if candidate is None:
            raise RuntimeError(f"smoke_candidate_missing:{qtype}")
        cid = str(candidate["candidate_id"])
        cyclic = next((row for row in permutations.get(cid, []) if row.get("is_cyclic")), None)
        if cyclic is None:
            raise RuntimeError(f"smoke_cyclic_permutation_missing:{cid}")
        selected.append(candidate)
        reduced[cid] = [cyclic]
    return selected, reduced


def _plan(candidates: list[dict[str, Any]], permutations: dict[str, list[dict[str, Any]]], *, stage: str) -> dict[str, Any]:
    cyclic_permutations = sum(sum(bool(row.get("is_cyclic")) for row in permutations.get(str(candidate["candidate_id"]), [])) for candidate in candidates)
    output_mode = OUTPUT_MODE_CONTENT_VISIBLE if stage == "b1" else OUTPUT_MODE_CONTENT_HIDDEN
    return {
        "probe_version": PROBE_VERSION.format(stage=stage),
        "stage": stage,
        "output_mode": output_mode,
        "candidate_count": len(candidates),
        "cyclic_permutation_count": cyclic_permutations,
        f"{stage.upper()}_content_{'visible' if stage == 'b1' else 'hidden'}_requests": cyclic_permutations * 2,
        "generation_hard_cap": TOTAL_CAP,
        "within_cap": cyclic_permutations * 2 <= TOTAL_CAP,
        "content_responder_prompt_version": CONTENT_RESPONDER_PROMPT_VERSION,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    source_run = args.source_run.resolve()
    stage = str(args.stage)
    output_mode = OUTPUT_MODE_CONTENT_VISIBLE if stage == "b1" else OUTPUT_MODE_CONTENT_HIDDEN
    probe_version = PROBE_VERSION.format(stage=stage)
    output_prefix = stage
    required = ("preflight_report.json", "sample_manifest.jsonl", "permutation_manifest.jsonl")
    for name in required:
        if not (run_root / name).is_file():
            raise FileNotFoundError(f"missing_run_manifest:{run_root / name}")
        if not (source_run / name).is_file():
            raise FileNotFoundError(f"missing_source_manifest:{source_run / name}")
    manifest_hashes = {name: sha256_file(run_root / name) for name in required}
    source_hashes = {name: sha256_file(source_run / name) for name in required}
    if manifest_hashes != source_hashes:
        raise RuntimeError(f"source_manifest_hash_mismatch:{manifest_hashes}:{source_hashes}")

    candidates = load_sample_candidates(run_root)
    permutations = load_permutations(run_root)
    if args.smoke:
        candidates, permutations = _select_smoke(candidates, permutations)
    if args.candidate_limit is not None:
        candidates = candidates[:args.candidate_limit]
    plan = _plan(candidates, permutations, stage=stage)
    if not plan["within_cap"]:
        request_count = next(value for key, value in plan.items() if key.endswith("_requests"))
        raise RuntimeError(f"generation_hard_cap_exceeded:{request_count}/{TOTAL_CAP}")
    repair_manifest = {
        "probe_version": probe_version,
        "source_run": str(source_run),
        "source_manifest_sha256": source_hashes,
        "copied_manifest_sha256": manifest_hashes,
        "output_mode": output_mode,
        "content_responder_prompt_version": CONTENT_RESPONDER_PROMPT_VERSION,
        "smoke": bool(args.smoke),
    }
    atomic_json(run_root / f"{output_prefix}_repair_manifest.json", repair_manifest)
    atomic_json(run_root / "solver_plan.json", plan)
    if args.dry_run:
        return {"status": "dry_run", "plan": plan, "repair_manifest": repair_manifest}
    if args.solver_model is None or args.tokenizer is None:
        raise ValueError("--solver-model and --tokenizer are required unless --dry-run is set")
    if not args.solver_model.exists() or not args.tokenizer.exists():
        raise FileNotFoundError("solver_model_or_tokenizer_not_found")

    permutation_sha = manifest_hashes["permutation_manifest.jsonl"]
    ledger = RequestLedger(run_root, permutation_sha, "local")
    result_index = load_result_index(run_root)
    backend = None
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
    finally:
        if backend is not None:
            backend.close()

    summary = summarize(rows)
    atomic_json(run_root / f"{output_prefix}_results.json", {"stage": f"{output_prefix}_repaired", "rows": rows})
    atomic_json(run_root / f"{output_prefix}_summary.json", summary)
    report = {
        "status": "completed",
        "probe_version": probe_version,
        "stage": stage,
        "source_run": str(source_run),
        "output_mode": output_mode,
        "content_responder_prompt_version": CONTENT_RESPONDER_PROMPT_VERSION,
        "smoke": bool(args.smoke),
        "plan": plan,
        "summary": summary,
        "request_accounting": ledger.summary(),
        "manifest_sha256": permutation_sha,
    }
    if report["request_accounting"]["request_count"] > TOTAL_CAP:
        raise RuntimeError(f"ledger_budget_postcondition_failed:{report['request_accounting']['request_count']}/{TOTAL_CAP}")
    atomic_json(run_root / f"{output_prefix}_solver_report.json", report)
    atomic_text(run_root / f"{output_prefix}_solver_summary.md", f"# Repaired H4 {stage.upper()} Solver probe\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    result = execute(_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
