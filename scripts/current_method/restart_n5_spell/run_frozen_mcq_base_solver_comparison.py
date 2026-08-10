"""Evaluate frozen contrastive MCQs with the unmodified Qwen3-4B base model."""

from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .core import atomic_json, canonical_json, sha256_bytes, sha256_file
from .input_adapter import read_json
from .run_canary import (
    create_source_snapshot,
    git_value,
    project_diff_sha256,
    write_post_run_provenance,
)
from .run_cross_dialogue_calibration import _base_manifest, _run_solver, _solver_config
from .run_rolling_outline_30k_canary import MODEL_CONTEXT_LIMIT, _usage
from .vllm_backends import VLLMSolverBackend


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_SOURCE_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_contrastive_distractor_canary_v1"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_contrastive_distractor_base_qwen3_4b_comparison_v1"
)
COMPARISON_VERSION = "frozen_contrastive_mcq_base_solver_comparison_v1"


def _candidate_paths(source_root: Path) -> list[Path]:
    return sorted((source_root / "frozen/materialized_mcqs").glob("*.json"))


def _attest_source(source_root: Path, *, expected_candidate_count: int = 4,
                   source_report: str =
                   "reports/contrastive_distractor_canary.json") -> dict[str, Any]:
    manifest_path = source_root / "config/run_manifest.json"
    provenance_path = source_root / "reports/post_run_provenance.json"
    context_path = source_root / "frozen/full_dialogue.json"
    catalog_path = source_root / "frozen/questioner_span_catalog.json"
    main_report_path = source_root / source_report
    for path in (manifest_path, provenance_path, context_path, catalog_path,
                 main_report_path):
        if not path.is_file():
            raise RuntimeError(f"source_file_missing:{path}")
    provenance = read_json(provenance_path)
    if (provenance.get("authoritative_release_provenance") is not True or
            provenance.get("provenance_mixed") is not False or
            provenance.get("run_manifest_sha256") != sha256_file(manifest_path)):
        raise RuntimeError("source_provenance_not_authoritative")
    paths = _candidate_paths(source_root)
    if len(paths) != expected_candidate_count:
        raise RuntimeError("source_candidate_count_invalid")
    candidates = []
    for path in paths:
        wrapper = read_json(path)
        required = {"raw_v8_qa", "materialized_qa", "admission",
                    "reasoning_contracts"}
        if not required <= set(wrapper):
            raise RuntimeError(f"source_candidate_invalid:{path.name}")
        admission = wrapper["admission"]
        if ((admission.get("spell_style_minimal_v1") or {}).get("decision") !=
                "accept" or admission.get("solver_eligible") is not True):
            raise RuntimeError(f"source_candidate_not_admitted:{path.name}")
        candidates.append({"path": path, "sha256": sha256_file(path),
                           "wrapper": wrapper})
    main = read_json(main_report_path)
    stage2 = {(row["qa_id"], int(row["attempt_index"])): row
              for row in main.get("attempts") or []}
    if len(stage2) != expected_candidate_count:
        raise RuntimeError("source_stage2_result_count_invalid")
    return {
        "source_root": str(source_root.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_provenance_sha256": sha256_file(provenance_path),
        "source_context_sha256": sha256_file(context_path),
        "source_catalog_sha256": sha256_file(catalog_path),
        "source_main_report_sha256": sha256_file(main_report_path),
        "context": read_json(context_path),
        "catalog": read_json(catalog_path),
        "candidates": candidates,
        "stage2": stage2,
    }


def _attempt_index(path: Path) -> int:
    suffix = path.stem.rsplit("_attempt_", 1)
    if len(suffix) != 2 or suffix[1] not in {"0", "1"}:
        raise RuntimeError(f"candidate_attempt_name_invalid:{path.name}")
    return int(suffix[1])


def _solver_result_summary(report: dict[str, Any]) -> dict[str, Any]:
    result = report.get("result") or {}
    initial = result.get("initial_no_context") or {}
    no_context = result.get("no_context") or initial
    full_context = result.get("full_context") or {}
    diagnostic = report.get("no_context_permutation") or {}
    orders = (diagnostic.get("diagnostic") or {}).get("orders") or []
    return {
        "final_decision": report.get("final_decision"),
        "grounding_decision": result.get("grounding_decision"),
        "initial_no_context_raw_answers": [row.get("raw_response") for row in
            initial.get("rollouts") or []],
        "initial_no_context_valid_answers": [row.get("parsed_letter") for row in
            initial.get("rollouts") or [] if row.get("local_reward") is not None],
        "initial_no_context_rewards": initial.get("verified_rewards"),
        "adaptive_extension": result.get("adaptive_extension"),
        "no_context_rewards": no_context.get("verified_rewards"),
        "full_context_rewards": full_context.get("verified_rewards"),
        "no_context_permutation_diagnostic": report.get(
            "no_context_permutation_diagnostic"),
        "permutation_orders": orders,
        "full_context_ran": bool(full_context),
        "local_vllm_generations": int(report.get("request_count") or 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--expected-candidate-count", type=int, default=4)
    parser.add_argument("--source-report", default=
        "reports/contrastive_distractor_canary.json")
    args = parser.parse_args(argv)
    if args.run_root.exists():
        raise FileExistsError(f"run_root_exists:{args.run_root}")
    if args.expected_candidate_count < 1:
        raise ValueError("expected_candidate_count_must_be_positive")
    source = _attest_source(args.source_root.resolve(),
        expected_candidate_count=args.expected_candidate_count,
        source_report=args.source_report)
    config = read_json(BASE_MODEL / "config.json")
    if int(config.get("max_position_embeddings") or 0) < MODEL_CONTEXT_LIMIT:
        raise RuntimeError("base_model_native_context_too_short")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)

    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    project_diff = project_diff_sha256(PROJECT)
    manifest = {
        **_base_manifest(run_id=args.run_root.name, mode=COMPARISON_VERSION,
            snapshot=snapshot, project_diff=project_diff),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comparison_version": COMPARISON_VERSION,
        "source_root": source["source_root"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "source_provenance_sha256": source["source_provenance_sha256"],
        "source_context_sha256": source["source_context_sha256"],
        "source_catalog_sha256": source["source_catalog_sha256"],
        "source_main_report_sha256": source["source_main_report_sha256"],
        "source_candidate_sha256": {entry["path"].name: entry["sha256"]
                                    for entry in source["candidates"]},
        "base_model_path": str(BASE_MODEL),
        "base_model_config_sha256": sha256_file(BASE_MODEL / "config.json"),
        "base_model_index_sha256": sha256_file(
            BASE_MODEL / "model.safetensors.index.json"),
        "base_tokenizer_config_sha256": sha256_file(
            BASE_MODEL / "tokenizer_config.json"),
        "solver_max_model_len": MODEL_CONTEXT_LIMIT,
        "solver_temperature": 0.7,
        "solver_top_p": 0.9,
        "solver_max_tokens": 32,
        "solver_enable_thinking": False,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "vllm_backend_source_sha256": sha256_file(
            MODULE_ROOT / "vllm_backends.py"),
        "paid_api_requests": 0,
        "ordinary_questioner_generations": 0,
        "realizer_generations": 0,
        "verifier_generations": 0,
        "training_runs": 0,
        "semantic_repair_generations": 0,
    }
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    atomic_json(args.run_root / "config/resolved_comparison_config.json", {
        "comparison_version": COMPARISON_VERSION,
        "source_root": source["source_root"],
        "base_model_path": str(BASE_MODEL),
        "solver_max_model_len": MODEL_CONTEXT_LIMIT,
        "temperature": 0.7, "top_p": 0.9, "max_tokens": 32,
        "enable_thinking": False, "add_generation_prompt": True,
        "candidate_count": args.expected_candidate_count,
        "source_report": args.source_report, "paid_api_requests": 0,
    })
    atomic_json(args.run_root / "frozen/source_attestation.json", {
        key: value for key, value in source.items()
        if key not in {"context", "catalog", "candidates", "stage2"}
    })
    atomic_json(args.run_root / "frozen/full_dialogue.json", source["context"])
    atomic_json(args.run_root / "frozen/questioner_span_catalog.json",
                source["catalog"])
    for entry in source["candidates"]:
        atomic_json(args.run_root / f"frozen/mcqs/{entry['path'].name}",
                    entry["wrapper"])

    backend = VLLMSolverBackend(model_path=BASE_MODEL,
        tokenizer_path=BASE_MODEL, max_model_len=MODEL_CONTEXT_LIMIT,
        temperature=0.7, top_p=0.9, max_tokens=32,
        gpu_memory_utilization=0.90)
    rows = []
    for entry in source["candidates"]:
        wrapper = entry["wrapper"]
        qa = wrapper["materialized_qa"]
        attempt_index = _attempt_index(entry["path"])
        qa_id = str(qa["qa_id"])
        solver_root = args.run_root / f"solver_runs/{qa_id}/attempt_{attempt_index}"
        solver_root.mkdir(parents=True)
        solver_config = _solver_config(run_root=solver_root)
        solver_config.update({
            "mode": "local_base_solver_frozen_mcq_comparison",
            "solver_effective_max_model_len": MODEL_CONTEXT_LIMIT,
            "solver_tokenizer_path": str(BASE_MODEL),
            "max_local_solver_generations": 80,
        })
        solver_manifest = {
            **_base_manifest(run_id=(
                f"{args.run_root.name}-{qa_id}-d{attempt_index}"),
                mode="local_base_solver_frozen_mcq_comparison",
                snapshot=snapshot, project_diff=project_diff),
            "comparison_version": COMPARISON_VERSION,
            "qa_id": qa_id, "attempt_index": attempt_index,
            "solver_model": str(BASE_MODEL),
            "solver_tokenizer": str(BASE_MODEL),
            "solver_max_model_len": MODEL_CONTEXT_LIMIT,
            "source_comparison_manifest_sha256": sha256_file(
                args.run_root / "config/run_manifest.json"),
            "source_materialized_mcq_sha256": entry["sha256"],
            "vllm_backend_source_sha256": sha256_file(
                MODULE_ROOT / "vllm_backends.py"),
        }
        atomic_json(solver_root / "config/resolved_canary_config.json",
                    solver_config)
        atomic_json(solver_root / "config/run_manifest.json", solver_manifest)
        report = _run_solver(root=solver_root, manifest=solver_manifest,
            config=solver_config, raw_batch=[wrapper["raw_v8_qa"]],
            admission=wrapper["admission"], materialized=qa,
            artifacts={"context": source["context"],
                       "catalog": source["catalog"],
                       "contracts": wrapper["reasoning_contracts"]},
            tokenizer=tokenizer, backend=backend)
        write_post_run_provenance(solver_root, PROJECT, solver_manifest)
        base_summary = _solver_result_summary(report)
        stage2 = source["stage2"][(qa_id, attempt_index)]
        rows.append({
            "qa_id": qa_id, "attempt_index": attempt_index,
            "target_letter": qa["reference_answer_letter"],
            "source_materialized_mcq_sha256": entry["sha256"],
            "stage2_solver": {
                "final_decision": stage2.get("final_decision"),
                "no_context_rewards": (stage2.get("solver") or {}).get(
                    "no_context_rewards"),
                "no_context_permutation_diagnostic": stage2.get(
                    "grounding_diagnosis"),
            },
            "base_solver": base_summary,
            "solver_root": str(solver_root.relative_to(args.run_root)),
            "solver_report_sha256": sha256_file(
                solver_root / "reports/cross_dialogue_solver_audit.json"),
        })

    decisions = collections.Counter(row["base_solver"]["final_decision"]
                                    for row in rows)
    diagnostics = collections.Counter(row["base_solver"].get(
        "no_context_permutation_diagnostic") or "not_run" for row in rows)
    usage = _usage(args.run_root)
    report = {
        "comparison_version": COMPARISON_VERSION,
        "base_model": str(BASE_MODEL),
        "stage2_model": str(read_json(
            args.source_root / "config/run_manifest.json").get("solver_model")),
        "candidate_count": len(rows),
        "base_final_decisions": dict(sorted(decisions.items())),
        "base_grounding_diagnostics": dict(sorted(diagnostics.items())),
        "items": rows,
        "resource_usage": {
            **usage, "paid_api_requests": 0,
            "ordinary_questioner_generations": 0,
            "realizer_generations": 0, "verifier_generations": 0,
            "training_runs": 0, "semantic_repair_generations": 0,
        },
    }
    atomic_json(args.run_root / "reports/base_solver_comparison.json", report)
    provenance = write_post_run_provenance(args.run_root, PROJECT, manifest)
    extra_checks = {
        "runner_source_sha_match": sha256_file(Path(__file__)) == manifest[
            "runner_source_sha256"],
        "zero_paid_api_requests": usage["total_paid_api_requests"] == 0,
    }
    provenance.update(extra_checks)
    provenance["authoritative_release_provenance"] = (
        provenance.get("authoritative_release_provenance") is True and
        all(extra_checks.values()))
    provenance["provenance_mixed"] = not provenance[
        "authoritative_release_provenance"]
    atomic_json(args.run_root / "reports/post_run_provenance.json", provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
