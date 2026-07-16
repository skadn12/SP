#!/usr/bin/env python3
"""Deterministically balance Stage64 option labels without carrying stale eval.

Rows are balanced within (option arity, attack skill) strata so label position is
not confounded with one reasoning skill.  Exact/ambiguous duplicate options are
rejected before shuffling.  Solver evaluation fields are always removed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import api_challenger_generation as generation
from scripts.current_method import recalibrate_stage64_shuffled_frontier as recal


def stratum_key(sample: dict[str, Any]) -> tuple[int, str]:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    attack = sample.get("attack_plan") if isinstance(sample.get("attack_plan"), dict) else {}
    skill = str(attack.get("attack_skill") or sample.get("attack_skill") or "unknown")
    return len(options), skill


def shuffled_to_target(sample: dict[str, Any], *, target_gold: str, seed: int) -> dict[str, Any]:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    old_gold = recal.qa_gold(sample)
    old_ids = [str(option.get("id") or option.get("option_id") or recal.LETTERS[index]) for index, option in enumerate(options)]
    if old_gold not in old_ids or target_gold not in recal.LETTERS[: len(options)]:
        raise ValueError(f"Cannot move gold for {sample.get('sample_id')}: old={old_gold} target={target_gold}")
    old_index = old_ids.index(old_gold)
    target_index = recal.LETTERS.index(target_gold)
    shift = (old_index - target_index) % len(options)
    row = recal.cyclic_option_permutation(sample, shift=shift)
    row["sample_id"] = str(sample.get("sample_id") or "")
    permutation = row.pop("stage64_permutation")
    row["option_shuffle"] = {
        "seed": int(seed),
        "strategy": "balanced within option_arity_and_attack_skill",
        "original_gold": old_gold,
        "target_gold": target_gold,
        "labels": list(recal.LETTERS[: len(options)]),
        "original_to_new": permutation["old_to_new"],
        "api_used": False,
    }
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    metadata.update(
        {
            "option_shuffle_applied": True,
            "option_shuffle_seed": int(seed),
            "option_shuffle_original_gold": old_gold,
            "option_shuffle_target_gold": target_gold,
            "option_shuffle_note": "local deterministic shuffle; solver_eval removed and must be recomputed",
        }
    )
    row["metadata"] = metadata
    return row


def balance_options(samples: list[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_stratum: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_stratum[stratum_key(sample)].append(sample)
    output: list[dict[str, Any]] = []
    strata_summary: dict[str, Any] = {}
    for (arity, skill), rows in sorted(by_stratum.items()):
        labels = list(recal.LETTERS[:arity])
        ordered = list(rows)
        random.Random(recal.stable_int([seed, arity, skill])).shuffle(ordered)
        shuffled = [shuffled_to_target(row, target_gold=labels[index % arity], seed=seed) for index, row in enumerate(ordered)]
        output.extend(shuffled)
        strata_summary[f"{arity}|{skill}"] = {
            "count": len(shuffled),
            "gold_distribution": dict(Counter(recal.qa_gold(row) for row in shuffled).most_common()),
        }
    output.sort(key=lambda row: str(row.get("sample_id") or ""))
    summary = {
        "strategy": "deterministic cyclic option reorder balanced within option arity and attack skill",
        "seed": seed,
        "count": len(output),
        "gold_distribution": dict(Counter(recal.qa_gold(row) for row in output).most_common()),
        "strata": strata_summary,
        "stale_eval_removed": True,
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality-filter and balance Stage64 option positions.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument("--reject-near-duplicate-options", action="store_true")
    parser.add_argument(
        "--skip-semantic-quality-gate",
        action="store_true",
        help="Diagnostic-only: retain rows that fail Stage64 answer/operation/graph semantic checks.",
    )
    parser.add_argument("--write-clean-sft", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = [recal.strip_eval_fields(row) for row in recal.read_jsonl(args.input)]
    accepted, rejected, quality = recal.quality_filter_samples(
        raw,
        near_duplicate_threshold=args.near_duplicate_threshold,
        reject_near_duplicates=args.reject_near_duplicate_options,
        enforce_semantic_quality=not args.skip_semantic_quality_gate,
    )
    shuffled, shuffle_summary = balance_options(accepted, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recal.write_jsonl(args.output_dir / "clean_samples.jsonl", shuffled)
    recal.write_jsonl(args.output_dir / "quality_rejected_samples.jsonl", rejected)
    if args.write_clean_sft:
        recal.write_jsonl(args.output_dir / "clean_samples_sft.jsonl", generation.build_sft_records(shuffled))
    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "quality": quality,
        "shuffle": shuffle_summary,
    }
    recal.write_json(args.output_dir / "option_shuffle_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
