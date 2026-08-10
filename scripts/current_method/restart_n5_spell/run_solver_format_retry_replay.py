from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .core import atomic_json, sha256_file
from .input_adapter import read_json
from .prompts import (
    SOLVER_PARSE_RETRY_VERSION,
    SOLVER_TRAILING_FORMAT_VERSION,
)
from .run_canary import create_source_snapshot, project_diff_sha256, write_post_run_provenance
from .run_cross_dialogue_calibration import _base_manifest, _run_solver, _solver_config
from .run_contrastive_distractor_canary import _solver_summary, _tree_sha256
from .run_single_item_qa_canary import DEFAULT_SOLVER, DEFAULT_TOKENIZER
from .vllm_backends import VLLMSolverBackend


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
REPLAY_VERSION = "solver_boxed_format_retry_replay_v1"
DEFAULT_SOURCE_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "spell_single_slot_distractor_search_v1")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "spell_solver_boxed_format_retry_replay_v1")
SOURCE_MCQ = Path(
    "frozen/materialized_mcqs/conv_41_single_item_001_B1.json")


def _attest_source(root: Path) -> dict[str, Any]:
    root = root.resolve()
    paths = {
        "manifest": root / "config/run_manifest.json",
        "provenance": root / "reports/post_run_provenance.json",
        "mcq": root / SOURCE_MCQ,
        "context": root / "frozen/contexts/conv-41.json",
        "catalog": root / "frozen/span_catalogs/conv-41.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise RuntimeError(f"format_replay_source_missing:{path}")
    provenance = read_json(paths["provenance"])
    if (provenance.get("authoritative_release_provenance") is not True or
            provenance.get("provenance_mixed") is not False):
        raise RuntimeError("format_replay_source_provenance_invalid")
    wrapper = read_json(paths["mcq"])
    if ((wrapper.get("admission") or {}).get(
            "spell_style_minimal_v1", {}).get("decision") != "accept"):
        raise RuntimeError("format_replay_source_mcq_not_admitted")
    return {
        "root": root,
        "tree_sha256_before": _tree_sha256(root),
        "paths": paths,
        "sha256": {key: sha256_file(path) for key, path in paths.items()},
        "wrapper": wrapper,
        "context": read_json(paths["context"]),
        "catalog": read_json(paths["catalog"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Zero-API replay of a frozen MCQ with format-aware retry")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.run_root.exists():
        raise FileExistsError(f"format_replay_root_exists:{args.run_root}")
    if not DEFAULT_SOLVER.is_dir() or not DEFAULT_TOKENIZER.is_dir():
        raise RuntimeError("format_replay_solver_or_tokenizer_missing")
    source = _attest_source(args.source_root)
    qa = source["wrapper"]["materialized_qa"]
    if args.dry_run:
        print(json.dumps({
            "decision": "dry_run_pass",
            "source_root": str(source["root"]),
            "source_mcq": str(source["paths"]["mcq"]),
            "qa_id": qa["qa_id"],
            "target_letter": qa["reference_answer_letter"],
            "paid_api_requests": 0,
        }, ensure_ascii=False, indent=2))
        return 0

    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    diff_sha = project_diff_sha256(PROJECT)
    manifest = {
        **_base_manifest(run_id=args.run_root.name, mode=REPLAY_VERSION,
            snapshot=snapshot, project_diff=diff_sha),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "replay_version": REPLAY_VERSION,
        "source_root": str(source["root"]),
        "source_tree_sha256_before": source["tree_sha256_before"],
        "source_sha256": source["sha256"],
        "source_mcq_relative_path": str(SOURCE_MCQ),
        "qa_id": qa["qa_id"],
        "solver_model": str(DEFAULT_SOLVER),
        "solver_tokenizer": str(DEFAULT_TOKENIZER),
        "solver_max_model_len": 16384,
        "solver_trailing_format_version": SOLVER_TRAILING_FORMAT_VERSION,
        "solver_parse_retry_version": SOLVER_PARSE_RETRY_VERSION,
        "orchestrator_source_sha256": sha256_file(
            MODULE_ROOT / "orchestrator.py"),
        "prompts_source_sha256": sha256_file(MODULE_ROOT / "prompts.py"),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "paid_api_requests": 0,
        "ordinary_questioner_generations": 0,
        "realizer_generations": 0,
        "verifier_generations": 0,
        "training_runs": 0,
    }
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    config = _solver_config(run_root=args.run_root)
    config.update({
        "mode": REPLAY_VERSION,
        "max_local_solver_generations": 80,
        "solver_trailing_format_version": SOLVER_TRAILING_FORMAT_VERSION,
        "solver_parse_retry_version": SOLVER_PARSE_RETRY_VERSION,
    })
    atomic_json(args.run_root / "config/resolved_canary_config.json", config)
    atomic_json(args.run_root / "frozen/source_attestation.json", {
        "source_root": str(source["root"]),
        "source_tree_sha256_before": source["tree_sha256_before"],
        "source_sha256": source["sha256"],
    })
    atomic_json(args.run_root / "frozen/source_mcq.json", source["wrapper"])
    atomic_json(args.run_root / "frozen/questioner_smoke_context.json",
                source["context"])
    atomic_json(args.run_root / "frozen/questioner_span_catalog.json",
                source["catalog"])

    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_TOKENIZER, local_files_only=True)
    backend = VLLMSolverBackend(
        model_path=DEFAULT_SOLVER, tokenizer_path=DEFAULT_TOKENIZER,
        max_model_len=16384, temperature=0.7, top_p=0.9,
        max_tokens=32, gpu_memory_utilization=0.80)
    report = _run_solver(
        root=args.run_root, manifest=manifest, config=config,
        raw_batch=[source["wrapper"]["raw_v8_qa"]],
        admission=source["wrapper"]["admission"], materialized=qa,
        artifacts={
            "context": source["context"],
            "catalog": source["catalog"],
            "contracts": source["wrapper"]["reasoning_contracts"],
        }, tokenizer=tokenizer, backend=backend)
    summary = {
        "replay_version": REPLAY_VERSION,
        "source_mcq_sha256": source["sha256"]["mcq"],
        "solver": _solver_summary(report),
        "local_vllm_generations": int(report.get("request_count") or 0),
        "paid_api_requests": 0,
    }
    atomic_json(args.run_root / "reports/format_retry_replay.json", summary)
    source_after = _tree_sha256(source["root"])
    provenance = write_post_run_provenance(args.run_root, PROJECT, manifest)
    checks = {
        "source_tree_unchanged": source_after == source["tree_sha256_before"],
        "orchestrator_source_sha_match": sha256_file(
            MODULE_ROOT / "orchestrator.py") ==
            manifest["orchestrator_source_sha256"],
        "prompts_source_sha_match": sha256_file(MODULE_ROOT / "prompts.py") ==
            manifest["prompts_source_sha256"],
        "runner_source_sha_match": sha256_file(Path(__file__)) ==
            manifest["runner_source_sha256"],
        "zero_paid_api_requests": True,
    }
    provenance.update({
        "source_tree_sha256_before": source["tree_sha256_before"],
        "source_tree_sha256_after": source_after, **checks})
    provenance["authoritative_release_provenance"] = (
        provenance.get("authoritative_release_provenance") is True and
        all(checks.values()))
    provenance["provenance_mixed"] = not provenance[
        "authoritative_release_provenance"]
    atomic_json(args.run_root / "reports/post_run_provenance.json", provenance)
    if provenance["provenance_mixed"]:
        raise RuntimeError("format_retry_replay_provenance_mixed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

