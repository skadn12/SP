#!/usr/bin/env python3
"""Build formal r4 train mix v1 from probe197 mining500.

This script is intentionally data-only:
  frontier_pool -> stratified train/heldout frontier
  candidate_pool 5/5 non-frontier -> clean replay anchors
  train frontier + clean replay -> raw train_mix + SFT chat records
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_v13_pilot as v13


DEFAULT_INPUT = (
    ROOT
    / "current_method_runs/current_method_r4_test_probe197_adaptive_smoke_hardening"
    / "current_method_r4_test_probe197_adaptive_mining500_frontier_mining_bonus_1"
)
DEFAULT_OUTPUT = ROOT / "current_method_runs/current_method_r4_from_probe197_mining500_train_v1"
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")

RUN_ID = "current_method_r4_from_probe197_mining500_train_v1"
DESIRED_CLEAN_REPLAY_QTYPE_SHARES = {
    "single_choice": 0.48,
    "multi_select": 0.26,
    "ordering": 0.26,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build formal r4 mining500 train mix v1.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--frontier-heldout", type=int, default=32)
    parser.add_argument("--clean-replay", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def qtype(row: dict[str, Any]) -> str:
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    return str(qa.get("question_type") or row.get("question_type") or "unknown")


def attack_skill(row: dict[str, Any]) -> str:
    plan = row.get("attack_plan") if isinstance(row.get("attack_plan"), dict) else {}
    return str(row.get("attack_skill") or plan.get("attack_skill") or "unknown")


def difficulty_subtype(row: dict[str, Any]) -> str:
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    if qtype(row) == "single_choice":
        return str(row.get("difficulty_subtype") or qa.get("difficulty_subtype") or "ordinary_single_choice")
    return str(row.get("difficulty_subtype") or qa.get("difficulty_subtype") or "ordinary")


def hard_template(row: dict[str, Any]) -> str:
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    value = row.get("hard_single_choice_template") or qa.get("hard_single_choice_template")
    return str(value or "none")


def solver_eval(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("solver_eval") if isinstance(row.get("solver_eval"), dict) else {}


def solver_bucket(row: dict[str, Any]) -> str:
    se = solver_eval(row)
    return str(se.get("bucket") or row.get("bucket") or "unknown")


def boundary_kind(row: dict[str, Any]) -> str:
    se = solver_eval(row)
    bucket = solver_bucket(row)
    exact = bool(se.get("exact_correct")) or int(se.get("exact_correct_count") or 0) > 0
    partial = float(se.get("partial_credit") or 0.0)
    if bucket in {"1/5", "2/5", "3/5", "4/5"}:
        return "boundary"
    if not exact and partial >= 0.70:
        return "near_miss"
    return "other"


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("expanded_graph_id") or "")


def stable_int(value: Any) -> int:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def allocation_largest_remainder(group_sizes: dict[tuple[str, ...], int], target: int) -> dict[tuple[str, ...], int]:
    total = sum(group_sizes.values())
    if target <= 0 or total <= 0:
        return {key: 0 for key in group_sizes}
    raw = {key: (target * size / total) for key, size in group_sizes.items()}
    counts = {key: min(group_sizes[key], int(math.floor(value))) for key, value in raw.items()}
    remaining = target - sum(counts.values())
    order = sorted(group_sizes, key=lambda key: (raw[key] - counts[key], group_sizes[key], key), reverse=True)
    while remaining > 0:
        changed = False
        for key in order:
            if remaining <= 0:
                break
            if counts[key] < group_sizes[key]:
                counts[key] += 1
                remaining -= 1
                changed = True
        if not changed:
            break
    return counts


def frontier_stratum_key(row: dict[str, Any]) -> tuple[str, str, str]:
    qt = qtype(row)
    template = hard_template(row) if qt == "single_choice" else "none"
    return (qt, boundary_kind(row), template)


def split_frontier(rows: list[dict[str, Any]], *, heldout_target: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[frontier_stratum_key(row)].append(row)
    group_sizes = {key: len(value) for key, value in groups.items()}
    heldout_counts = allocation_largest_remainder(group_sizes, heldout_target)
    heldout_ids: set[str] = set()
    per_group: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        rng = random.Random(seed + stable_int(key))
        shuffled = list(group)
        shuffled.sort(key=lambda row: sample_id(row))
        rng.shuffle(shuffled)
        take = heldout_counts[key]
        chosen = shuffled[:take]
        heldout_ids.update(sample_id(row) for row in chosen)
        per_group.append(
            {
                "stratum": {"question_type": key[0], "boundary_kind": key[1], "hard_single_choice_template": key[2]},
                "total": len(group),
                "heldout": take,
                "train": len(group) - take,
            }
        )
    train = [row for row in rows if sample_id(row) not in heldout_ids]
    heldout = [row for row in rows if sample_id(row) in heldout_ids]
    stats = {
        "heldout_target": heldout_target,
        "train_count": len(train),
        "heldout_count": len(heldout),
        "per_stratum": per_group,
    }
    return train, heldout, stats


def validation_clean(row: dict[str, Any]) -> bool:
    validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
    seed_source = row.get("seed_source") if isinstance(row.get("seed_source"), dict) else {}
    split = row.get("frontier_from_split") or seed_source.get("split") or row.get("split")
    return (
        row.get("graph_validation_pass") is True
        and seed_source.get("allowed_train_seed") is True
        and split == "train"
        and validation.get("option_valid") is True
        and validation.get("gold_derivable") is True
        and validation.get("graph_dialogue_consistent") is True
        and validation.get("anti_shortcut_pass") is True
        and not validation.get("warnings")
    )


def qtype_targets(total: int, available: dict[str, int]) -> dict[str, int]:
    raw_sizes = {
        key: int(round(DESIRED_CLEAN_REPLAY_QTYPE_SHARES.get(key, 0.0) * 1000000))
        for key in DESIRED_CLEAN_REPLAY_QTYPE_SHARES
        if available.get(key, 0) > 0
    }
    targets = allocation_largest_remainder({(key,): size for key, size in raw_sizes.items()}, total)
    out = {key[0]: min(available.get(key[0], 0), value) for key, value in targets.items()}
    deficit = total - sum(out.values())
    while deficit > 0:
        changed = False
        for key in sorted(available, key=lambda item: (available[item] - out.get(item, 0), item), reverse=True):
            if deficit <= 0:
                break
            if out.get(key, 0) < available[key]:
                out[key] = out.get(key, 0) + 1
                deficit -= 1
                changed = True
        if not changed:
            break
    return out


def clean_cell(row: dict[str, Any]) -> tuple[str, str, str, str]:
    template = hard_template(row) if qtype(row) == "single_choice" else "none"
    return (qtype(row), attack_skill(row), difficulty_subtype(row), template)


def select_by_cell(rows: list[dict[str, Any]], target: int, *, seed: int) -> list[dict[str, Any]]:
    if target <= 0:
        return []
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[clean_cell(row)].append(row)
    group_sizes = {key: len(value) for key, value in groups.items()}
    counts = allocation_largest_remainder(group_sizes, min(target, len(rows)))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for key, group in sorted(groups.items()):
        rng = random.Random(seed + stable_int(key))
        shuffled = list(group)
        shuffled.sort(key=lambda row: sample_id(row))
        rng.shuffle(shuffled)
        for row in shuffled[: counts[key]]:
            selected.append(row)
            selected_ids.add(sample_id(row))
    if len(selected) < target:
        leftovers = [row for row in rows if sample_id(row) not in selected_ids]
        leftovers.sort(key=lambda row: sample_id(row))
        random.Random(seed + 991).shuffle(leftovers)
        selected.extend(leftovers[: target - len(selected)])
    selected.sort(key=lambda row: (qtype(row), attack_skill(row), difficulty_subtype(row), hard_template(row), sample_id(row)))
    return selected[:target]


def select_clean_replay(
    candidates: list[dict[str, Any]],
    *,
    frontier_ids: set[str],
    target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [
        row
        for row in candidates
        if sample_id(row) not in frontier_ids and solver_bucket(row) == "5/5" and validation_clean(row)
    ]
    available = Counter(qtype(row) for row in eligible)
    targets = qtype_targets(target, dict(available))
    selected: list[dict[str, Any]] = []
    for qt in sorted(targets):
        pool = [row for row in eligible if qtype(row) == qt]
        selected.extend(select_by_cell(pool, targets[qt], seed=seed + stable_int(qt)))
    selected_ids = {sample_id(row) for row in selected}
    if len(selected) < target:
        leftovers = [row for row in eligible if sample_id(row) not in selected_ids]
        leftovers.sort(key=lambda row: sample_id(row))
        random.Random(seed + 307).shuffle(leftovers)
        selected.extend(leftovers[: target - len(selected)])
    selected = selected[:target]
    stats = {
        "target": target,
        "eligible_5of5_clean_nonfrontier": len(eligible),
        "eligible_by_qtype": dict(Counter(qtype(row) for row in eligible).most_common()),
        "eligible_by_cell": stringify_counter(Counter("|".join(clean_cell(row)) for row in eligible)),
        "selected": len(selected),
        "selected_by_qtype": dict(Counter(qtype(row) for row in selected).most_common()),
        "selected_by_cell": stringify_counter(Counter("|".join(clean_cell(row)) for row in selected)),
        "selected_source_bucket": dict(Counter(solver_bucket(row) for row in selected).most_common()),
        "qtype_targets": targets,
    }
    return selected, stats


def stringify_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common()}


def count_by(rows: list[dict[str, Any]], fn) -> dict[str, int]:
    return stringify_counter(Counter(fn(row) for row in rows))


def options_map(row: dict[str, Any]) -> dict[str, str]:
    options = row.get("options")
    if not isinstance(options, list):
        qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
        options = qa.get("options")
    out: dict[str, str] = {}
    if isinstance(options, list):
        for option in options:
            if not isinstance(option, dict):
                continue
            oid = str(option.get("id") or "").strip()
            if oid:
                out[oid] = str(option.get("text") or "")
    elif isinstance(options, dict):
        for key, value in options.items():
            out[str(key)] = str(value)
    return out


def gold_list(row: dict[str, Any]) -> list[str]:
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    for value in (qa.get("gold"), row.get("gold_options"), row.get("gold_answer"), row.get("gold_answers")):
        if isinstance(value, list) and value:
            return [str(item) for item in value]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    return []


def normalize_for_chat(row: dict[str, Any]) -> dict[str, Any]:
    sample = copy.deepcopy(row)
    qt = qtype(sample)
    opts = options_map(sample)
    gold = gold_list(sample)
    sample["question_type"] = qt
    sample["options"] = opts
    if qt == "single_choice":
        letter = gold[0] if gold else ""
        sample["gold_option"] = letter
        sample["gold_options"] = [letter] if letter else []
        sample["gold_answer"] = opts.get(letter, letter)
    elif qt == "ordering":
        sample["gold_order"] = gold
        sample["gold_options"] = gold
        sample["gold_answer"] = gold
    elif qt == "multi_select":
        sample["gold_options"] = gold
        sample["gold_answer"] = gold
    else:
        sample["gold_answer"] = str(gold[0]) if gold else str(sample.get("gold_answer") or "")
    if not sample.get("expanded_dialogue_text"):
        turns = sample.get("dialogue", {}).get("turns") if isinstance(sample.get("dialogue"), dict) else []
        sample["expanded_dialogue_text"] = "\n".join(
            f"[{turn.get('turn_id')}] {turn.get('speaker')}: {turn.get('text')}" for turn in turns if isinstance(turn, dict)
        )
    sample.setdefault("expanded_graph_id", sample_id(sample))
    sample.setdefault("expanded_dialogue_id", f"{sample_id(sample)}_dialogue")
    return sample


def stamp_raw(row: dict[str, Any], role: str, source: str) -> dict[str, Any]:
    out = copy.deepcopy(row)
    meta = dict(out.get("metadata") or {})
    meta.update(
        {
            "formal_r4_candidate_training": True,
            "r5_started": False,
            "train_mix_run_id": RUN_ID,
            "train_mix_version": "v1",
            "train_mix_role": role,
            "train_mix_source": source,
            "source_solver_eval_bucket": solver_bucket(row),
            "source_solver_eval_exact_count": solver_eval(row).get("exact_correct_count"),
            "boundary_kind": boundary_kind(row),
            "question_type": qtype(row),
            "attack_skill": attack_skill(row),
            "difficulty_subtype": difficulty_subtype(row),
            "hard_single_choice_template": hard_template(row),
        }
    )
    out["metadata"] = meta
    out["train_mix_role"] = role
    out["train_mix_source"] = source
    return out


def to_sft(row: dict[str, Any], role: str, source: str) -> dict[str, Any]:
    sft = v13.build_chat_record(normalize_for_chat(row))
    meta = dict(sft.get("metadata") or {})
    meta.update(
        {
            "formal_r4_candidate_training": True,
            "r5_started": False,
            "train_mix_run_id": RUN_ID,
            "train_mix_version": "v1",
            "train_mix_role": role,
            "train_mix_source": source,
            "source_sample_id": sample_id(row),
            "difficulty_cell": f"{qtype(row)}|{attack_skill(row)}",
            "source_solver_eval_bucket": solver_bucket(row),
            "source_solver_eval_exact_count": solver_eval(row).get("exact_correct_count"),
            "boundary_kind": boundary_kind(row),
            "difficulty_subtype": difficulty_subtype(row),
            "hard_single_choice_template": hard_template(row),
        }
    )
    sft["metadata"] = meta
    return sft


def token_length_summary(sft_rows: list[dict[str, Any]], *, tokenizer_path: str, max_length: int) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from train_qwen3_lora_sft import render_chat

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    lengths: list[int] = []
    over: list[dict[str, Any]] = []
    for row in sft_rows:
        text = render_chat(tokenizer, row["messages"], add_generation_prompt=False)
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if tokenizer.eos_token_id is not None and (not ids or ids[-1] != tokenizer.eos_token_id):
            ids.append(tokenizer.eos_token_id)
        length = len(ids)
        lengths.append(length)
        if length > max_length:
            over.append({"record_id": row.get("record_id"), "tokens": length})
    return {
        "tokenizer": tokenizer_path,
        "max_length": max_length,
        "count": len(lengths),
        "min": min(lengths) if lengths else None,
        "max": max(lengths) if lengths else None,
        "mean": round(sum(lengths) / len(lengths), 3) if lengths else None,
        "over_max_length": len(over),
        "over_max_length_preview": over[:10],
    }


def summarize_split(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": name,
        "count": len(rows),
        "by_qtype": count_by(rows, qtype),
        "by_boundary_kind": count_by(rows, boundary_kind),
        "by_qtype_boundary_kind": count_by(rows, lambda row: f"{qtype(row)}|{boundary_kind(row)}"),
        "by_bucket": count_by(rows, solver_bucket),
        "by_attack_skill": count_by(rows, attack_skill),
        "hard_single_choice_template": count_by(
            [row for row in rows if qtype(row) == "single_choice"],
            hard_template,
        ),
    }


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace: {output_dir}")
        shutil.rmtree(output_dir)
    train_mix_dir = output_dir / "train_mix"
    reports_dir = output_dir / "reports"
    models_dir = output_dir / "models"
    train_mix_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    frontier_path = input_dir / "frontier_pool.jsonl"
    candidate_path = input_dir / "candidate_pool.jsonl"
    adaptive_summary_path = input_dir / "adaptive_smoke_summary.json"
    frontier = read_jsonl(frontier_path)
    candidates = read_jsonl(candidate_path)
    adaptive_summary = read_json(adaptive_summary_path)

    frontier_train_raw, frontier_heldout_raw, split_stats = split_frontier(
        frontier,
        heldout_target=int(args.frontier_heldout),
        seed=int(args.seed),
    )
    frontier_ids = {sample_id(row) for row in frontier}
    clean_replay_raw, replay_stats = select_clean_replay(
        candidates,
        frontier_ids=frontier_ids,
        target=int(args.clean_replay),
        seed=int(args.seed) + 17,
    )

    frontier_train = [
        stamp_raw(row, "frontier_train", "mining500_frontier_pool_stratified_train")
        for row in frontier_train_raw
    ]
    frontier_heldout = [
        stamp_raw(row, "frontier_heldout", "mining500_frontier_pool_stratified_heldout")
        for row in frontier_heldout_raw
    ]
    clean_replay = [
        stamp_raw(row, "clean_replay", "mining500_candidate_pool_5of5_clean_replay_anchor")
        for row in clean_replay_raw
    ]
    train_mix = frontier_train + clean_replay
    sft_train = [
        to_sft(row, "frontier_train", "mining500_frontier_pool_stratified_train")
        for row in frontier_train_raw
    ] + [
        to_sft(row, "clean_replay", "mining500_candidate_pool_5of5_clean_replay_anchor")
        for row in clean_replay_raw
    ]

    write_jsonl(train_mix_dir / "frontier_train.jsonl", frontier_train)
    write_jsonl(train_mix_dir / "frontier_heldout.jsonl", frontier_heldout)
    write_jsonl(train_mix_dir / "clean_replay.jsonl", clean_replay)
    write_jsonl(train_mix_dir / "train_mix.jsonl", train_mix)
    write_jsonl(train_mix_dir / "sft_train.jsonl", sft_train)

    train_ids = {sample_id(row) for row in frontier_train}
    heldout_ids = {sample_id(row) for row in frontier_heldout}
    replay_ids = {sample_id(row) for row in clean_replay}
    mix_ids = [sample_id(row) for row in train_mix]
    summary = {
        "run_id": RUN_ID,
        "formal_r4_candidate_training": True,
        "probe_training": False,
        "r5_started": False,
        "training_status": "not_started",
        "input_dir": str(input_dir),
        "frontier_source": str(frontier_path),
        "candidate_source": str(candidate_path),
        "adaptive_smoke_summary": str(adaptive_summary_path),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "tokenizer": str(args.tokenizer),
        "max_length_required": int(args.max_length),
        "source_metrics": {
            "candidate_pool_count": len(candidates),
            "frontier_pool_count": len(frontier),
            "adaptive_valid_rate": adaptive_summary.get("valid_rate"),
            "adaptive_clean_rate": adaptive_summary.get("clean_rate"),
            "adaptive_boundary_frontier_pool_count": adaptive_summary.get("boundary_frontier_pool_count"),
            "adaptive_near_miss_pool_count": adaptive_summary.get("near_miss_pool_count"),
            "adaptive_qtype_distribution": adaptive_summary.get("qtype_distribution"),
            "adaptive_frontier_pool_qtype_distribution": adaptive_summary.get("frontier_pool_qtype_distribution"),
        },
        "split": split_stats,
        "frontier_train": summarize_split("frontier_train", frontier_train),
        "frontier_heldout": summarize_split("frontier_heldout", frontier_heldout),
        "clean_replay": {**summarize_split("clean_replay", clean_replay), **replay_stats},
        "train_mix": summarize_split("train_mix", train_mix),
        "counts": {
            "frontier_pool": len(frontier),
            "frontier_train": len(frontier_train),
            "frontier_heldout": len(frontier_heldout),
            "clean_replay": len(clean_replay),
            "train_mix": len(train_mix),
            "sft_train": len(sft_train),
        },
        "artifact_paths": {
            "frontier_train": str(train_mix_dir / "frontier_train.jsonl"),
            "frontier_heldout": str(train_mix_dir / "frontier_heldout.jsonl"),
            "clean_replay": str(train_mix_dir / "clean_replay.jsonl"),
            "train_mix": str(train_mix_dir / "train_mix.jsonl"),
            "sft_train": str(train_mix_dir / "sft_train.jsonl"),
            "adapter": str(models_dir / "adapter"),
            "merged": str(models_dir / "merged"),
        },
        "integrity_checks": {
            "frontier_train_heldout_overlap": len(train_ids & heldout_ids),
            "clean_replay_frontier_overlap": len(replay_ids & frontier_ids),
            "clean_replay_heldout_overlap": len(replay_ids & heldout_ids),
            "train_mix_duplicate_sample_ids": len(mix_ids) - len(set(mix_ids)),
            "heldout_frontier_in_train_mix": len(heldout_ids & set(mix_ids)),
            "candidate_pool_full_included": False,
            "candidate_pool_total": len(candidates),
        },
        "notes": [
            "Frontier pool is split stratified by question_type, boundary/near_miss, and hard_single_choice template.",
            "Clean replay anchors are selected only from candidate_pool 5/5 clean non-frontier rows.",
            "Heldout frontier rows are written for evaluation and are not included in train_mix or sft_train.",
            "This is formal r4 candidate training data, not probe training and not r5.",
        ],
    }
    try:
        summary["token_lengths"] = token_length_summary(
            sft_train,
            tokenizer_path=str(args.tokenizer),
            max_length=int(args.max_length),
        )
    except Exception as exc:  # Keep data construction usable if tokenization is unavailable.
        summary["token_lengths"] = {"error": repr(exc)}
    write_json(reports_dir / "train_mix_summary.json", summary)
    write_json(train_mix_dir / "train_mix_summary.json", summary)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["integrity_checks"], ensure_ascii=False, indent=2))
    print(json.dumps(summary.get("token_lengths"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
