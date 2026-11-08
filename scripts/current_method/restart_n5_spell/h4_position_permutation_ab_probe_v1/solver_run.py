"""Explicit Solver phases for the isolated H4 position A/B probe.

The module has no default live action.  ``--dry-run`` only reports the frozen
request plan.  A live run requires both a local model and tokenizer paths and
uses the probe-local RequestLedger for resumability.
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scope", choices=("a", "b", "all"), required=True,
                        help="A=all label permutations; B=cyclic B1/B2; all=A then B")
    parser.add_argument("--solver-model", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--smoke", action="store_true",
                        help="select exactly one multi-select and one ordering candidate, with one cyclic and one non-cyclic permutation each")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _plan(candidates: list[dict[str, Any]], permutations: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_count = sum(len(permutations[str(row["candidate_id"])]) for row in candidates)
    cyclic_count = sum(sum(bool(p["is_cyclic"]) for p in permutations[str(row["candidate_id"])]) for row in candidates)
    return {
        "candidate_count": len(candidates),
        "permutation_count": all_count,
        "cyclic_permutation_count": cyclic_count,
        "A_label_requests": all_count * 2,
        "B0_label_cyclic_requests": cyclic_count * 2,
        "B1_content_visible_requests": cyclic_count * 2,
        "B2_content_hidden_requests": cyclic_count * 2,
        "A_plus_B1_B2_requests": all_count * 2 + cyclic_count * 4,
        "generation_hard_cap": TOTAL_CAP,
        "within_cap": all_count * 2 + cyclic_count * 4 <= TOTAL_CAP,
    }


def _smoke_inputs(candidates: list[dict[str, Any]], permutations: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    selected: list[dict[str, Any]] = []
    reduced: dict[str, list[dict[str, Any]]] = {}
    for qtype in ("multi_select", "ordering"):
        matches = [row for row in candidates if str(row.get("question_type")) == qtype]
        if not matches:
            raise RuntimeError(f"smoke_candidate_missing:{qtype}")
        candidate = matches[0]
        cid = str(candidate["candidate_id"])
        rows = list(permutations.get(cid) or [])
        cyclic = next((row for row in rows if row.get("is_cyclic")), None)
        non_cyclic = next((row for row in rows if not row.get("is_cyclic")), None)
        if cyclic is None or non_cyclic is None:
            raise RuntimeError(f"smoke_permutation_missing:{cid}")
        selected.append(candidate)
        reduced[cid] = [cyclic, non_cyclic]
    return selected, reduced


def _write_stage(run_root: Path, name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = summarize(rows)
    atomic_json(run_root / f"{name}_results.json", {"stage": name, "rows": rows})
    atomic_json(run_root / f"{name}_summary.json", report)
    return report


def _cyclic_label_rows(result_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in result_index.values()
        if row.get("output_mode") == OUTPUT_MODE_LABEL and row.get("is_cyclic") is True
    ]


def execute(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    if not (run_root / "preflight_report.json").is_file():
        raise FileNotFoundError(f"missing_preflight:{run_root / 'preflight_report.json'}")
    if not (run_root / "sample_manifest.jsonl").is_file() or not (run_root / "permutation_manifest.jsonl").is_file():
        raise FileNotFoundError("missing_probe_manifests")
    candidates = load_sample_candidates(run_root)
    permutations = load_permutations(run_root)
    if args.smoke:
        if args.scope != "all":
            raise ValueError("--smoke_requires_scope_all")
        candidates, permutations = _smoke_inputs(candidates, permutations)
    plan = _plan(candidates, permutations)
    if not plan["within_cap"]:
        raise RuntimeError(f"generation_hard_cap_exceeded:{plan['A_plus_B1_B2_requests']}/{TOTAL_CAP}")
    atomic_json(run_root / "solver_plan.json", plan)
    if args.dry_run:
        return {"status": "dry_run", "smoke": bool(args.smoke), "plan": plan}
    if args.solver_model is None or args.tokenizer is None:
        raise ValueError("--solver-model and --tokenizer are required unless --dry-run is set")
    # The merged checkpoint is a directory while a tokenizer may be either a
    # directory or a single local file.  Existence, not ``is_file``, is the
    # correct precondition for both local assets.
    if not args.solver_model.exists() or not args.tokenizer.exists():
        raise FileNotFoundError("solver_model_or_tokenizer_not_found")
    manifest_sha = sha256_file(run_root / "permutation_manifest.jsonl")
    ledger = RequestLedger(run_root, manifest_sha, "local")
    tokenizer = None
    backend = None
    result_index = load_result_index(run_root)
    stages: dict[str, Any] = {}
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        backend = VLLMSolverBackend(
            model_path=args.solver_model, tokenizer_path=args.tokenizer,
            max_model_len=16_384, temperature=0.7, top_p=0.9, max_tokens=32,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        if args.scope in {"a", "all"}:
            rows = run_requests(
                run_root=run_root, candidates=candidates, permutations=permutations,
                backend=backend, tokenizer=tokenizer, ledger=ledger,
                output_mode=OUTPUT_MODE_LABEL, include_broader=True,
                candidate_limit=args.candidate_limit, result_index=result_index,
            )
            stages["A"] = _write_stage(run_root, "a", rows)
            # B0 is a report over A's cyclic rows, not another generation
            # condition.  Keep the source relation explicit for later paired
            # comparisons.
            stages["B0"] = _write_stage(run_root, "b0_cyclic", _cyclic_label_rows(result_index))
        if args.scope in {"b", "all"}:
            # B0 is the cyclic subset of A.  It is never regenerated as a
            # separate condition; this check prevents an unpaired B report.
            cyclic_ids = {
                f"{row['candidate_id']}__{perm['permutation_id']}__{mode}__{OUTPUT_MODE_LABEL}__s00"
                for row in candidates
                for perm in permutations[str(row["candidate_id"])] if perm["is_cyclic"]
                for mode in ("no_context", "full_context")
            }
            if not cyclic_ids.issubset(result_index):
                missing = len(cyclic_ids - set(result_index))
                raise RuntimeError(f"B0_cyclic_label_results_missing:{missing}; run --scope a first")
            stages.setdefault("B0", _write_stage(run_root, "b0_cyclic", _cyclic_label_rows(result_index)))
            for mode, name in ((OUTPUT_MODE_CONTENT_VISIBLE, "b1"), (OUTPUT_MODE_CONTENT_HIDDEN, "b2")):
                rows = run_requests(
                    run_root=run_root, candidates=candidates, permutations=permutations,
                    backend=backend, tokenizer=tokenizer, ledger=ledger,
                    output_mode=mode, include_broader=False, cyclic_only=True,
                    candidate_limit=args.candidate_limit, result_index=result_index,
                )
                stages[name.upper()] = _write_stage(run_root, name, rows)
    finally:
        if backend is not None:
            backend.close()
    accounting = ledger.summary()
    if accounting["request_count"] > TOTAL_CAP:
        raise RuntimeError(f"ledger_budget_postcondition_failed:{accounting['request_count']}/{TOTAL_CAP}")
    report = {"status": "completed", "scope": args.scope, "smoke": bool(args.smoke), "plan": plan,
              "stages": stages, "request_accounting": accounting,
              "manifest_sha256": manifest_sha}
    atomic_json(run_root / "solver_report.json", report)
    atomic_text(run_root / "solver_summary.md", "# H4 position permutation Solver probe\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = execute(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
