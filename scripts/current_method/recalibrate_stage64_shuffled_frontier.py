#!/usr/bin/env python3
"""Re-evaluate shuffled Stage64 samples and rebuild frontier splits.

This script fixes the common post-shuffle pitfall: option labels changed, so
old solver_eval/frontier_label fields are no longer valid. It reloads the
shuffled clean samples, removes stale eval labels, runs a fixed solver, selects
frontier from the new buckets, and writes grouped train/heldout splits.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.current_method import api_challenger_generation as generation  # noqa: E402
from scripts.current_method import diagnostics_context_ablation as diag  # noqa: E402
from scripts.current_method import frontier_selection  # noqa: E402


DEFAULT_INPUT = (
    ROOT
    / "base_restart_current_method_validation/69_stage64_api_selfplay500_ctx32768/"
    / "option_shuffled_seed20260710/clean_samples.jsonl"
)
DEFAULT_OUTPUT = ROOT / "base_restart_current_method_validation/71_stage64_shuffled_recalibrated_stage2"
DEFAULT_BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_STAGE2_SOLVER = ROOT / "base_restart_current_method_validation/accepted_models/solver_base_restart_stage2_merged"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_int(value: Any) -> int:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def strip_eval_fields(sample: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(sample)
    for key in (
        "solver_eval",
        "solver_eval_k1",
        "frontier_label",
        "frontier_pool_priority",
        "frontier_type",
        "stage64_recalibration",
    ):
        row.pop(key, None)
    return row


def exact_bucket(sample: dict[str, Any]) -> int | None:
    se = sample.get("solver_eval") if isinstance(sample.get("solver_eval"), dict) else {}
    value = se.get("exact_correct_count")
    if isinstance(value, (int, float)):
        return int(value)
    bucket = str(se.get("bucket") or "")
    if "/" in bucket:
        try:
            return int(bucket.split("/", 1)[0])
        except ValueError:
            return None
    return None


def bucket(sample: dict[str, Any]) -> str:
    se = sample.get("solver_eval") if isinstance(sample.get("solver_eval"), dict) else {}
    return str(se.get("bucket") or "unknown")


def qa_gold(sample: dict[str, Any]) -> str:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    gold = qa.get("gold")
    if isinstance(gold, list) and gold:
        return str(gold[0])
    return str(gold or "")


def group_key(sample: dict[str, Any]) -> str:
    meta = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    seed = sample.get("seed") if isinstance(sample.get("seed"), dict) else {}
    seed_source = sample.get("seed_source") if isinstance(sample.get("seed_source"), dict) else {}
    return str(
        meta.get("source_seed_id")
        or sample.get("source_id")
        or seed.get("sample_id")
        or seed_source.get("sample_id")
        or sample.get("sample_id")
        or ""
    )


def attack_skill(sample: dict[str, Any]) -> str:
    return frontier_selection.attack_skill(sample)


def question_type(sample: dict[str, Any]) -> str:
    return frontier_selection.question_type(sample)


def split_frontier_by_group(frontier: list[dict[str, Any]], *, seed: int, heldout_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in frontier:
        groups[group_key(sample)].append(sample)
    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)

    target = max(1, round(len(frontier) * heldout_ratio)) if frontier else 0
    heldout_groups: set[str] = set()
    heldout_count = 0
    # Prefer not to overshoot too badly, but keep whole groups intact.
    for key, rows in sorted(group_items, key=lambda item: (stable_int([seed, item[0]]), item[0])):
        if heldout_count >= target:
            break
        heldout_groups.add(key)
        heldout_count += len(rows)

    heldout = [sample for sample in frontier if group_key(sample) in heldout_groups]
    train = [sample for sample in frontier if group_key(sample) not in heldout_groups]
    summary = {
        "strategy": "grouped by source_seed_id after post-shuffle fixed-solver recalibration",
        "seed": seed,
        "heldout_ratio_target": heldout_ratio,
        "frontier_total": len(frontier),
        "frontier_train": len(train),
        "frontier_heldout": len(heldout),
        "train_groups": len({group_key(s) for s in train}),
        "heldout_groups": len({group_key(s) for s in heldout}),
        "frontier_train_by_skill": dict(Counter(attack_skill(s) for s in train).most_common()),
        "frontier_heldout_by_skill": dict(Counter(attack_skill(s) for s in heldout).most_common()),
        "frontier_train_by_bucket": dict(Counter(bucket(s) for s in train).most_common()),
        "frontier_heldout_by_bucket": dict(Counter(bucket(s) for s in heldout).most_common()),
        "frontier_train_gold": dict(Counter(qa_gold(s) for s in train).most_common()),
        "frontier_heldout_gold": dict(Counter(qa_gold(s) for s in heldout).most_common()),
        "heldout_group_keys": sorted(heldout_groups),
    }
    return train, heldout, summary


def write_sft(path: Path, samples: list[dict[str, Any]]) -> None:
    write_jsonl(path, generation.build_sft_records(samples))


def count_summary(samples: list[dict[str, Any]], frontier: list[dict[str, Any]], hard0: list[dict[str, Any]], easy5: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "clean_count": len(samples),
        "frontier_count": len(frontier),
        "hard0_count": len(hard0),
        "easy5_count": len(easy5),
        "bucket_distribution": dict(Counter(bucket(s) for s in samples).most_common()),
        "frontier_bucket_distribution": dict(Counter(bucket(s) for s in frontier).most_common()),
        "gold_distribution": dict(Counter(qa_gold(s) for s in samples).most_common()),
        "frontier_gold_distribution": dict(Counter(qa_gold(s) for s in frontier).most_common()),
        "frontier_by_question_type": dict(Counter(question_type(s) for s in frontier).most_common()),
        "frontier_by_attack_skill": dict(Counter(attack_skill(s) for s in frontier).most_common()),
        "prompt_token_mean": round(
            sum(int((s.get("solver_eval", {}).get("diagnostics") or {}).get("prompt_token_estimate") or 0) for s in samples)
            / len(samples),
            3,
        )
        if samples
        else 0,
        "prompt_token_max": max(
            [int((s.get("solver_eval", {}).get("diagnostics") or {}).get("prompt_token_estimate") or 0) for s in samples] or [0]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recalibrate shuffled Stage64 samples with a fixed solver.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_STAGE2_SOLVER)
    parser.add_argument("--solver-tokenizer", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--k-eval", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--heldout-ratio", type=float, default=0.20)
    parser.add_argument("--max-largest-question-type-share", type=float, default=0.35)
    parser.add_argument("--max-largest-attack-skill-share", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    report_dir = args.report_dir or (output_dir / "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    samples = [strip_eval_fields(row) for row in read_jsonl(args.input)]
    if not samples:
        raise RuntimeError(f"No samples found in {args.input}")

    solver = diag.SharedSolver(
        solver_model=args.solver_model,
        solver_tokenizer=args.solver_tokenizer,
        base_model=args.base_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    eval_rows = solver.evaluate(samples, k_eval=args.k_eval, debug_dir=report_dir / "stage2_recalibration_prompt_debug")
    write_jsonl(output_dir / "stage2_recalibration_eval_rows.jsonl", eval_rows)

    frontier, frontier_summary, failure_matrix = frontier_selection.select_frontier(
        samples=samples,
        output_dir=output_dir,
        report_dir=report_dir,
        frontier_splits={"train"},
        forbid_frontier_from_dev_heldout=True,
        max_question_type_share=args.max_largest_question_type_share,
        max_attack_skill_share=args.max_largest_attack_skill_share,
    )

    hard0 = [s for s in samples if exact_bucket(s) == 0]
    easy5 = [s for s in samples if exact_bucket(s) == 5]
    frontier_train, frontier_heldout, split_summary = split_frontier_by_group(
        frontier,
        seed=args.seed,
        heldout_ratio=args.heldout_ratio,
    )
    split_dir = output_dir / f"splits/frontier_group_holdout_seed{args.seed}"

    write_jsonl(output_dir / "clean_samples.jsonl", samples)
    write_jsonl(output_dir / "samples.jsonl", samples)
    write_jsonl(output_dir / "valid_samples.jsonl", [s for s in samples if generation.base.base_valid(s)])
    write_sft(output_dir / "sft_train.jsonl", samples)

    write_jsonl(split_dir / "frontier_train.jsonl", frontier_train)
    write_jsonl(split_dir / "frontier_heldout.jsonl", frontier_heldout)
    write_jsonl(split_dir / "diagnostic_hard0_samples.jsonl", hard0)
    write_jsonl(split_dir / "diagnostic_easy5_samples.jsonl", easy5)
    write_sft(split_dir / "frontier_train_sft.jsonl", frontier_train)
    write_sft(split_dir / "frontier_heldout_sft.jsonl", frontier_heldout)
    write_sft(split_dir / "diagnostic_hard0_sft.jsonl", hard0)
    write_sft(split_dir / "diagnostic_easy5_sft.jsonl", easy5)

    summary = {
        "input": str(args.input),
        "output_dir": str(output_dir),
        "fixed_solver": str(args.solver_model),
        "base_model": str(args.base_model),
        "k_eval": args.k_eval,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "heldout_ratio": args.heldout_ratio,
        **count_summary(samples, frontier, hard0, easy5),
        "frontier_summary": frontier_summary,
        "split_summary": split_summary,
        "failure_matrix": failure_matrix,
    }
    write_json(output_dir / "recalibration_summary.json", summary)
    write_json(report_dir / "recalibration_summary.json", summary)
    write_json(split_dir / "split_summary.json", split_summary)

    print(json.dumps({k: summary[k] for k in ("clean_count", "frontier_count", "hard0_count", "easy5_count")}, ensure_ascii=False))
    print(json.dumps(split_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
