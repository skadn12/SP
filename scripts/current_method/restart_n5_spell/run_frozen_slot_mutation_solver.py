"""Zero-API evaluation of frozen slot-mutation MCQs with one local Solver."""

from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .core import atomic_json, sha256_file
from .input_adapter import read_json
from .run_canary import create_source_snapshot, project_diff_sha256, write_post_run_provenance
from .run_cross_dialogue_calibration import _base_manifest, _run_solver, _solver_config
from .run_frozen_mcq_base_solver_comparison import _solver_result_summary
from .run_rolling_outline_30k_canary import MODEL_CONTEXT_LIMIT, TOKENIZER, _usage
from .vllm_backends import VLLMSolverBackend


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_slot_mutation_distractor_canary_v1"
)
STAGE2 = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "accepted_models/solver_base_restart_stage2_merged"
)
BASE = Path("/root/autodl-tmp/Qwen3-4B")
EVALUATOR_VERSION = "frozen_slot_mutation_solver_evaluation_v1"


def attest_frozen_source(root: Path) -> dict[str, Any]:
    manifest_path = root / "config/run_manifest.json"
    context_path = root / "frozen/full_dialogue.json"
    catalog_path = root / "frozen/questioner_span_catalog.json"
    for path in (manifest_path, context_path, catalog_path):
        if not path.is_file():
            raise RuntimeError(f"source_file_missing:{path}")
    manifest = read_json(manifest_path)
    if (manifest.get("canary_version") not in {
            "slot_mutation_distractor_canary_v1",
            "balanced_factorial_distractor_canary_v1"} or
            manifest.get("budget", {}).get("physical_paid_api_requests") != 2):
        raise RuntimeError("source_manifest_contract_invalid")
    ledger_paths = sorted((root / "ledger/requests").glob("*.json"))
    if len(ledger_paths) != 2:
        raise RuntimeError("source_paid_request_count_invalid")
    ledgers = [read_json(path) for path in ledger_paths]
    if any(row.get("status") != "completed" or
           row.get("paid_generation_count") != 1 or
           not str(row.get("final_decision", "")).endswith("_accept")
           for row in ledgers):
        raise RuntimeError("source_paid_request_not_accepted")
    raw_paths = sorted((root / "raw/distractor_writer").glob("*.json"))
    candidate_paths = sorted((root / "frozen/materialized_mcqs").glob("*.json"))
    if len(raw_paths) != 2 or len(candidate_paths) != 2:
        raise RuntimeError("source_frozen_candidate_count_invalid")
    raw_records = [read_json(path) for path in raw_paths]
    if any(not str(row.get("decision", "")).endswith("_accept") or
           not str((row.get("validation") or {}).get(
               "decision", "")).endswith("_accept") for row in raw_records):
        raise RuntimeError("source_slot_mutation_validation_invalid")
    candidates = []
    for path in candidate_paths:
        wrapper = read_json(path)
        if ((wrapper.get("admission") or {}).get("solver_eligible") is not True or
                ((wrapper.get("admission") or {}).get("spell_style_minimal_v1") or
                 {}).get("decision") != "accept"):
            raise RuntimeError(f"source_candidate_not_admitted:{path.name}")
        candidates.append({"path": path, "sha256": sha256_file(path),
                           "wrapper": wrapper})
    return {
        "source_root": str(root.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_context_sha256": sha256_file(context_path),
        "source_catalog_sha256": sha256_file(catalog_path),
        "source_ledger_sha256": {path.name: sha256_file(path)
                                 for path in ledger_paths},
        "source_raw_sha256": {path.name: sha256_file(path) for path in raw_paths},
        "source_candidate_sha256": {entry["path"].name: entry["sha256"]
                                    for entry in candidates},
        "context": read_json(context_path), "catalog": read_json(catalog_path),
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--solver", choices=("stage2", "base"), required=True)
    args = parser.parse_args(argv)
    if args.run_root.exists():
        raise FileExistsError(f"run_root_exists:{args.run_root}")
    source = attest_frozen_source(args.source_root.resolve())
    model = STAGE2 if args.solver == "stage2" else BASE
    config = read_json(model / "config.json")
    if int(config.get("max_position_embeddings") or 0) < MODEL_CONTEXT_LIMIT:
        raise RuntimeError("solver_native_context_too_short")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)

    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    project_diff = project_diff_sha256(PROJECT)
    manifest = {
        **_base_manifest(run_id=args.run_root.name, mode=EVALUATOR_VERSION,
            snapshot=snapshot, project_diff=project_diff),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluator_version": EVALUATOR_VERSION, "solver_kind": args.solver,
        "solver_model": str(model), "solver_tokenizer": str(TOKENIZER),
        "solver_max_model_len": MODEL_CONTEXT_LIMIT,
        "source_root": source["source_root"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "source_context_sha256": source["source_context_sha256"],
        "source_catalog_sha256": source["source_catalog_sha256"],
        "source_ledger_sha256": source["source_ledger_sha256"],
        "source_raw_sha256": source["source_raw_sha256"],
        "source_candidate_sha256": source["source_candidate_sha256"],
        "runner_source_sha256": sha256_file(Path(__file__)),
        "vllm_backend_source_sha256": sha256_file(MODULE_ROOT / "vllm_backends.py"),
        "paid_api_requests": 0, "ordinary_questioner_generations": 0,
        "realizer_generations": 0, "verifier_generations": 0,
        "training_runs": 0, "semantic_repair_generations": 0,
    }
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    atomic_json(args.run_root / "config/resolved_evaluation_config.json", {
        "solver_kind": args.solver, "solver_model": str(model),
        "tokenizer": str(TOKENIZER), "max_model_len": MODEL_CONTEXT_LIMIT,
        "temperature": 0.7, "top_p": 0.9, "max_tokens": 32,
        "enable_thinking": False, "candidate_count": 2, "paid_api_requests": 0,
    })
    atomic_json(args.run_root / "frozen/source_attestation.json", {
        key: value for key, value in source.items()
        if key not in {"context", "catalog", "candidates"}
    })
    backend = VLLMSolverBackend(
        model_path=model, tokenizer_path=TOKENIZER,
        max_model_len=MODEL_CONTEXT_LIMIT, temperature=0.7,
        top_p=0.9, max_tokens=32, gpu_memory_utilization=0.90)
    rows = []
    for entry in source["candidates"]:
        wrapper = entry["wrapper"]
        qa = wrapper["materialized_qa"]
        qa_id = str(qa["qa_id"])
        solver_root = args.run_root / f"solver_runs/{qa_id}"
        solver_root.mkdir(parents=True)
        resolved = _solver_config(run_root=solver_root)
        resolved.update({"mode": EVALUATOR_VERSION,
                         "solver_effective_max_model_len": MODEL_CONTEXT_LIMIT,
                         "max_local_solver_generations": 80})
        child_manifest = {
            **_base_manifest(run_id=f"{args.run_root.name}-{qa_id}",
                mode=EVALUATOR_VERSION, snapshot=snapshot, project_diff=project_diff),
            "evaluator_version": EVALUATOR_VERSION, "qa_id": qa_id,
            "solver_kind": args.solver, "solver_model": str(model),
            "solver_max_model_len": MODEL_CONTEXT_LIMIT,
            "source_evaluation_manifest_sha256": sha256_file(
                args.run_root / "config/run_manifest.json"),
            "source_materialized_mcq_sha256": entry["sha256"],
        }
        atomic_json(solver_root / "config/resolved_canary_config.json", resolved)
        atomic_json(solver_root / "config/run_manifest.json", child_manifest)
        report = _run_solver(
            root=solver_root, manifest=child_manifest, config=resolved,
            raw_batch=[wrapper["raw_v8_qa"]], admission=wrapper["admission"],
            materialized=qa,
            artifacts={"context": source["context"], "catalog": source["catalog"],
                       "contracts": wrapper["reasoning_contracts"]},
            tokenizer=tokenizer, backend=backend)
        write_post_run_provenance(solver_root, PROJECT, child_manifest)
        rows.append({
            "qa_id": qa_id, "target_letter": qa["reference_answer_letter"],
            "source_materialized_mcq_sha256": entry["sha256"],
            "solver": _solver_result_summary(report),
            "solver_root": str(solver_root.relative_to(args.run_root)),
            "solver_report_sha256": sha256_file(
                solver_root / "reports/cross_dialogue_solver_audit.json"),
        })
    decisions = collections.Counter(row["solver"]["final_decision"] for row in rows)
    diagnostics = collections.Counter(row["solver"].get(
        "no_context_permutation_diagnostic") or "not_run" for row in rows)
    usage = _usage(args.run_root)
    report = {
        "evaluator_version": EVALUATOR_VERSION, "solver_kind": args.solver,
        "solver_model": str(model), "candidate_count": len(rows),
        "final_decisions": dict(sorted(decisions.items())),
        "grounding_diagnostics": dict(sorted(diagnostics.items())),
        "items": rows,
        "resource_usage": {**usage, "paid_api_requests": 0,
            "ordinary_questioner_generations": 0, "realizer_generations": 0,
            "verifier_generations": 0, "training_runs": 0,
            "semantic_repair_generations": 0},
    }
    atomic_json(args.run_root / "reports/frozen_slot_mutation_solver.json", report)
    provenance = write_post_run_provenance(args.run_root, PROJECT, manifest)
    checks = {"runner_source_sha_match": sha256_file(Path(__file__)) == manifest[
                  "runner_source_sha256"],
              "zero_paid_api_requests": usage["total_paid_api_requests"] == 0}
    provenance.update(checks)
    provenance["authoritative_release_provenance"] = (
        provenance.get("authoritative_release_provenance") is True and
        all(checks.values()))
    provenance["provenance_mixed"] = not provenance[
        "authoritative_release_provenance"]
    atomic_json(args.run_root / "reports/post_run_provenance.json", provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
