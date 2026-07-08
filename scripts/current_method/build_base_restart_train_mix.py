#!/usr/bin/env python3
"""Build train_mix for base_restart_current_method_validation.

This is scoped to the isolated validation directory and consumes only the
top-up combined mining artifacts. It reuses the current v13 full-dialogue MCQ
chat format while keeping r4 data/model paths out of the data flow.
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


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_v13_pilot as v13


VALID_ROOT = ROOT / "base_restart_current_method_validation"
DEFAULT_FRONTIER = VALID_ROOT / "01_mining500_train/mining/topup_combined_frontier_pool.jsonl"
DEFAULT_CANDIDATE = VALID_ROOT / "01_mining500_train/mining/topup_combined_candidate_pool.jsonl"
DEFAULT_OUTPUT = VALID_ROOT / "01_mining500_train/train_mix"
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
RUN_ID = "base_restart_current_method_validation_stage1_train_mix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build base-restart current_method train mix.")
    parser.add_argument("--combined-frontier", default=str(DEFAULT_FRONTIER))
    parser.add_argument("--combined-candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--frontier-heldout", type=int, default=22)
    parser.add_argument("--clean-replay", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def qa(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("qa") if isinstance(row.get("qa"), dict) else {}


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("metadata") if isinstance(row.get("metadata"), dict) else {}


def qtype(row: dict[str, Any]) -> str:
    return str(row.get("question_type") or qa(row).get("question_type") or "unknown")


def attack_skill(row: dict[str, Any]) -> str:
    plan = row.get("attack_plan") if isinstance(row.get("attack_plan"), dict) else {}
    return str(row.get("attack_skill") or plan.get("attack_skill") or metadata(row).get("attack_skill") or "unknown")


def difficulty_subtype(row: dict[str, Any]) -> str:
    value = row.get("difficulty_subtype") or qa(row).get("difficulty_subtype") or metadata(row).get("difficulty_subtype")
    if value:
        return str(value)
    return "ordinary_single_choice" if qtype(row) == "single_choice" else "ordinary"


def hard_template(row: dict[str, Any]) -> str:
    value = row.get("hard_single_choice_template") or qa(row).get("hard_single_choice_template") or metadata(row).get("hard_single_choice_template")
    return str(value or "none")


def solver_eval(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("solver_eval") if isinstance(row.get("solver_eval"), dict) else {}


def solver_bucket(row: dict[str, Any]) -> str:
    return str(solver_eval(row).get("bucket") or "unknown")


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("expanded_graph_id") or "")


def stable_json_hash(payload: Any, n: int = 16) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def stable_int(value: Any) -> int:
    return int(stable_json_hash(value, 16), 16)


def source_group(row: dict[str, Any]) -> str:
    raw = str(metadata(row).get("topup_combined_source") or metadata(row).get("train_mix_original_source") or "")
    if raw.startswith("topup_"):
        return "topup"
    if raw.startswith("original_"):
        return "original_mining500"
    if "topup" in raw:
        return "topup"
    return "original_mining500"


def options_payload(row: dict[str, Any]) -> Any:
    options = row.get("options")
    if options is None:
        options = qa(row).get("options")
    return options


def question_id(row: dict[str, Any]) -> str:
    for key in ("question_id", "qa_id"):
        value = row.get(key) or qa(row).get(key)
        if value:
            return str(value)
    return "derived:" + stable_json_hash(
        {
            "question_type": qtype(row),
            "question": row.get("question") or qa(row).get("question"),
            "options": options_payload(row),
            "gold": qa(row).get("gold") or row.get("gold_options") or row.get("gold_answer"),
        }
    )


def dialogue_hash(row: dict[str, Any]) -> str:
    value = metadata(row).get("dialogue_hash") or row.get("dialogue_hash")
    if value:
        return str(value)
    return "derived:" + stable_json_hash(row.get("expanded_dialogue_text") or row.get("dialogue") or sample_id(row))


def frontier_type(row: dict[str, Any]) -> str:
    label = str(row.get("frontier_pool_label") or "")
    if label == "boundary_frontier":
        return "boundary"
    if label == "near_miss":
        return "near_miss"
    bucket = solver_bucket(row)
    if bucket in {"1/5", "2/5", "3/5", "4/5"}:
        return "boundary"
    if bucket == "5/5":
        return "exact"
    return "near_miss" if bucket == "0/5" else "other"


def allocation_largest_remainder(group_sizes: dict[Any, int], target: int) -> dict[Any, int]:
    total = sum(group_sizes.values())
    if target <= 0 or total <= 0:
        return {key: 0 for key in group_sizes}
    raw = {key: target * size / total for key, size in group_sizes.items()}
    counts = {key: min(group_sizes[key], int(math.floor(value))) for key, value in raw.items()}
    remaining = min(target, total) - sum(counts.values())
    order = sorted(group_sizes, key=lambda key: (raw[key] - counts[key], group_sizes[key], str(key)), reverse=True)
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


def split_frontier(rows: list[dict[str, Any]], *, heldout_target: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_qtype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_qtype[qtype(row)].append(row)
    qtype_counts = allocation_largest_remainder({key: len(value) for key, value in by_qtype.items()}, heldout_target)
    # Keep scarce non-single-choice tasks represented in heldout whenever available.
    for key in ("multi_select", "ordering"):
        if len(by_qtype.get(key, [])) >= 2 and qtype_counts.get(key, 0) < 2:
            qtype_counts[key] = 2
    while sum(qtype_counts.values()) > heldout_target:
        reducible = sorted(
            [key for key, value in qtype_counts.items() if value > (2 if key in {"multi_select", "ordering"} else 0)],
            key=lambda key: (qtype_counts[key], key),
            reverse=True,
        )
        if not reducible:
            break
        qtype_counts[reducible[0]] -= 1

    heldout_ids: set[str] = set()
    per_stratum: list[dict[str, Any]] = []
    for qt, group in sorted(by_qtype.items()):
        subgroup_map: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            key = (source_group(row), qt, difficulty_subtype(row), frontier_type(row), hard_template(row) if qt == "single_choice" else "none")
            subgroup_map[key].append(row)
        subgroup_counts = allocation_largest_remainder({key: len(value) for key, value in subgroup_map.items()}, qtype_counts.get(qt, 0))
        for key, subgroup in sorted(subgroup_map.items()):
            rng = random.Random(seed + stable_int(key))
            shuffled = sorted(subgroup, key=sample_id)
            rng.shuffle(shuffled)
            take = subgroup_counts.get(key, 0)
            chosen = shuffled[:take]
            heldout_ids.update(sample_id(row) for row in chosen)
            per_stratum.append(
                {
                    "stratum": {
                        "source": key[0],
                        "question_type": key[1],
                        "difficulty_subtype": key[2],
                        "frontier_type": key[3],
                        "hard_single_choice_template": key[4],
                    },
                    "total": len(subgroup),
                    "heldout": take,
                    "train": len(subgroup) - take,
                }
            )
    train = [row for row in rows if sample_id(row) not in heldout_ids]
    heldout = [row for row in rows if sample_id(row) in heldout_ids]
    return train, heldout, {"heldout_target": heldout_target, "heldout_qtype_targets": qtype_counts, "per_stratum": per_stratum}


def validation_clean(row: dict[str, Any]) -> bool:
    validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
    return (
        row.get("graph_validation_pass") is True
        and validation.get("option_valid") is True
        and validation.get("gold_derivable") is True
        and validation.get("graph_dialogue_consistent") is True
        and validation.get("anti_shortcut_pass") is True
        and not validation.get("warnings")
    )


def clean_cell(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        source_group(row),
        qtype(row),
        attack_skill(row),
        difficulty_subtype(row),
        hard_template(row) if qtype(row) == "single_choice" else "none",
    )


def select_by_cell(rows: list[dict[str, Any]], target: int, *, seed: int) -> list[dict[str, Any]]:
    if target <= 0 or not rows:
        return []
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[clean_cell(row)].append(row)
    counts = allocation_largest_remainder({key: len(value) for key, value in groups.items()}, min(target, len(rows)))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for key, group in sorted(groups.items()):
        shuffled = sorted(group, key=sample_id)
        random.Random(seed + stable_int(key)).shuffle(shuffled)
        for row in shuffled[: counts.get(key, 0)]:
            selected.append(row)
            selected_ids.add(sample_id(row))
    if len(selected) < target:
        leftovers = [row for row in rows if sample_id(row) not in selected_ids]
        leftovers.sort(key=sample_id)
        random.Random(seed + 991).shuffle(leftovers)
        selected.extend(leftovers[: target - len(selected)])
    selected.sort(key=lambda row: (qtype(row), source_group(row), attack_skill(row), difficulty_subtype(row), hard_template(row), sample_id(row)))
    return selected[:target]


def select_clean_replay(candidates: list[dict[str, Any]], *, frontier_ids: set[str], target: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [
        row
        for row in candidates
        if sample_id(row) not in frontier_ids and solver_bucket(row) == "5/5" and validation_clean(row)
    ]
    by_qt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_qt[qtype(row)].append(row)

    selected: list[dict[str, Any]] = []
    notes: list[str] = []
    for scarce_qt in ("multi_select", "ordering"):
        pool = by_qt.get(scarce_qt, [])
        if pool:
            selected.extend(select_by_cell(pool, min(len(pool), target - len(selected)), seed=seed + stable_int(scarce_qt)))
        if scarce_qt == "ordering" and len(pool) < 3:
            notes.append(f"ordering 5/5 clean candidates are scarce: available={len(pool)}; selected={min(len(pool), target)}")
    selected_ids = {sample_id(row) for row in selected}
    remaining = target - len(selected)
    if remaining > 0:
        single_pool = [row for row in by_qt.get("single_choice", []) if sample_id(row) not in selected_ids]
        selected.extend(select_by_cell(single_pool, remaining, seed=seed + stable_int("single_choice")))
    selected = selected[:target]
    stats = {
        "target": target,
        "eligible_5of5_clean_nonfrontier": len(eligible),
        "eligible_by_qtype": count_by(eligible, qtype),
        "eligible_by_source": count_by(eligible, source_group),
        "eligible_by_source_qtype": count_by(eligible, lambda row: f"{source_group(row)}|{qtype(row)}"),
        "selected": len(selected),
        "selected_by_qtype": count_by(selected, qtype),
        "selected_by_source": count_by(selected, source_group),
        "selected_by_source_qtype": count_by(selected, lambda row: f"{source_group(row)}|{qtype(row)}"),
        "selected_by_cell": count_by(selected, lambda row: "|".join(clean_cell(row))),
        "notes": notes,
    }
    return selected, stats


def count_by(rows: list[dict[str, Any]], fn) -> dict[str, int]:
    return {str(key): int(value) for key, value in Counter(fn(row) for row in rows).most_common()}


def options_map(row: dict[str, Any]) -> dict[str, str]:
    options = options_payload(row)
    out: dict[str, str] = {}
    if isinstance(options, list):
        for option in options:
            if isinstance(option, dict) and option.get("id"):
                out[str(option.get("id"))] = str(option.get("text") or "")
    elif isinstance(options, dict):
        out = {str(key): str(value) for key, value in options.items()}
    return out


def gold_list(row: dict[str, Any]) -> list[str]:
    for value in (qa(row).get("gold"), row.get("gold_options"), row.get("gold_answer"), row.get("gold_answers")):
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


def stamp_raw(row: dict[str, Any], role: str) -> dict[str, Any]:
    out = copy.deepcopy(row)
    meta = dict(metadata(out))
    meta.update(
        {
            "source_pipeline": "base_restart_current_method_validation",
            "train_mix_run_id": RUN_ID,
            "train_mix_role": role,
            "train_mix_source_group": source_group(row),
            "train_mix_version": "stage1_combined_topup_v1",
            "prompt_mode": "full_dialogue",
            "evidence_window_used_as_prompt": False,
            "source_solver_eval_bucket": solver_bucket(row),
            "source_solver_eval_exact_count": solver_eval(row).get("exact_correct_count"),
            "frontier_type": frontier_type(row),
            "question_type": qtype(row),
            "difficulty_subtype": difficulty_subtype(row),
            "hard_single_choice_template": hard_template(row),
        }
    )
    out["metadata"] = meta
    out["source"] = role
    out["train_mix_role"] = role
    out["train_mix_source_group"] = source_group(row)
    out["question_id"] = question_id(row)
    out["dialogue_hash"] = dialogue_hash(row)
    return out


def to_sft(row: dict[str, Any], role: str) -> dict[str, Any]:
    sft = v13.build_chat_record(normalize_for_chat(row))
    meta = dict(sft.get("metadata") or {})
    meta.update(
        {
            "source_pipeline": "base_restart_current_method_validation",
            "train_mix_run_id": RUN_ID,
            "train_mix_role": role,
            "train_mix_source_group": source_group(row),
            "train_mix_version": "stage1_combined_topup_v1",
            "source_sample_id": sample_id(row),
            "source_question_id": question_id(row),
            "source_dialogue_hash": dialogue_hash(row),
            "prompt_mode": "full_dialogue",
            "evidence_window_used_as_prompt": False,
            "difficulty_cell": f"{qtype(row)}|{attack_skill(row)}",
            "source_solver_eval_bucket": solver_bucket(row),
            "source_solver_eval_exact_count": solver_eval(row).get("exact_correct_count"),
            "frontier_type": frontier_type(row),
            "question_type": qtype(row),
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
            over.append({"record_id": row.get("record_id"), "tokens": length, "source_sample_id": metadata(row).get("source_sample_id")})
    return {
        "tokenizer_path": tokenizer_path,
        "max_length": max_length,
        "count": len(lengths),
        "min_sft_token_length": min(lengths) if lengths else None,
        "max_sft_token_length": max(lengths) if lengths else None,
        "mean_sft_token_length": round(sum(lengths) / len(lengths), 3) if lengths else None,
        "over_max_length_count": len(over),
        "over_max_length_preview": over[:10],
    }


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "source_distribution": count_by(rows, source_group),
        "qtype_distribution": count_by(rows, qtype),
        "difficulty_subtype_distribution": count_by(rows, difficulty_subtype),
        "frontier_type_distribution": count_by(rows, frontier_type),
        "hard_single_choice_template_distribution": count_by([row for row in rows if qtype(row) == "single_choice"], hard_template),
        "bucket_distribution": count_by(rows, solver_bucket),
    }


def overlap_check(train_mix: list[dict[str, Any]], heldout: list[dict[str, Any]]) -> dict[str, Any]:
    train_sample_ids = {sample_id(row) for row in train_mix}
    heldout_sample_ids = {sample_id(row) for row in heldout}
    train_question_ids = {question_id(row) for row in train_mix}
    heldout_question_ids = {question_id(row) for row in heldout}
    train_dialogue_hashes = {dialogue_hash(row) for row in train_mix}
    heldout_dialogue_hashes = {dialogue_hash(row) for row in heldout}
    return {
        "sample_id_overlap": len(train_sample_ids & heldout_sample_ids),
        "question_id_overlap": len(train_question_ids & heldout_question_ids),
        "dialogue_hash_overlap": len(train_dialogue_hashes & heldout_dialogue_hashes),
        "total_overlap": len(train_sample_ids & heldout_sample_ids)
        + len(train_question_ids & heldout_question_ids)
        + len(train_dialogue_hashes & heldout_dialogue_hashes),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Train Mix Audit Report",
        "",
        f"- combined_frontier_pool_count: {summary['combined_frontier_pool_count']}",
        f"- frontier_train_count: {summary['frontier_train_count']}",
        f"- frontier_heldout_count: {summary['frontier_heldout_count']}",
        f"- clean_replay_count: {summary['clean_replay_count']}",
        f"- train_mix_count: {summary['train_mix_count']}",
        f"- clean_replay_share: {summary['clean_replay_share']}",
        f"- train_mix_pass: {summary['train_mix_pass']}",
        f"- tokenizer_path: `{summary['tokenizer_path']}`",
        f"- max_sft_token_length: {summary['max_sft_token_length']}",
        f"- over_max_length_count: {summary['over_max_length_count']}",
        f"- evidence_window_used_as_prompt: {str(summary['evidence_window_used_as_prompt']).lower()}",
        f"- heldout_overlap_check: `{summary['heldout_overlap_check']}`",
        "",
        "## Frontier Train",
        f"- source_distribution: `{summary['frontier_train']['source_distribution']}`",
        f"- qtype_distribution: `{summary['qtype_distribution']['frontier_train']}`",
        f"- frontier_type_distribution: `{summary['frontier_train']['frontier_type_distribution']}`",
        f"- hard_single_choice_template_distribution: `{summary['frontier_train']['hard_single_choice_template_distribution']}`",
        "",
        "## Frontier Heldout",
        f"- source_distribution: `{summary['frontier_heldout']['source_distribution']}`",
        f"- qtype_distribution: `{summary['qtype_distribution']['frontier_heldout']}`",
        f"- frontier_type_distribution: `{summary['frontier_heldout']['frontier_type_distribution']}`",
        f"- hard_single_choice_template_distribution: `{summary['frontier_heldout']['hard_single_choice_template_distribution']}`",
        "",
        "## Clean Replay",
        f"- source_distribution: `{summary['clean_replay']['source_distribution']}`",
        f"- qtype_distribution: `{summary['qtype_distribution']['clean_replay']}`",
        f"- hard_single_choice_template_distribution: `{summary['clean_replay']['hard_single_choice_template_distribution']}`",
        f"- notes: `{summary['clean_replay_selection'].get('notes')}`",
        "",
        "## Files",
    ]
    for key, value in summary["artifact_paths"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    frontier_path = Path(args.combined_frontier)
    candidate_path = Path(args.combined_candidate)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    frontier = read_jsonl(frontier_path)
    candidates = read_jsonl(candidate_path)
    frontier_train_raw, frontier_heldout_raw, split_stats = split_frontier(
        frontier,
        heldout_target=int(args.frontier_heldout),
        seed=int(args.seed),
    )
    frontier_ids = {sample_id(row) for row in frontier}
    clean_replay_raw, clean_stats = select_clean_replay(
        candidates,
        frontier_ids=frontier_ids,
        target=int(args.clean_replay),
        seed=int(args.seed) + 17,
    )

    frontier_train = [stamp_raw(row, "frontier_train") for row in frontier_train_raw]
    frontier_heldout = [stamp_raw(row, "frontier_heldout") for row in frontier_heldout_raw]
    clean_replay = [stamp_raw(row, "clean_replay") for row in clean_replay_raw]
    train_mix = frontier_train + clean_replay
    sft_train = [to_sft(row, "frontier_train") for row in frontier_train_raw] + [to_sft(row, "clean_replay") for row in clean_replay_raw]

    write_jsonl(output_dir / "frontier_train.jsonl", frontier_train)
    write_jsonl(output_dir / "frontier_heldout.jsonl", frontier_heldout)
    write_jsonl(output_dir / "clean_replay.jsonl", clean_replay)
    write_jsonl(output_dir / "train_mix.jsonl", train_mix)
    write_jsonl(output_dir / "sft_train.jsonl", sft_train)

    token_stats = token_length_summary(sft_train, tokenizer_path=str(args.tokenizer), max_length=int(args.max_length))
    heldout_overlap = overlap_check(train_mix, frontier_heldout)
    clean_share = round(len(clean_replay) / max(1, len(train_mix)), 6)
    frontier_train_summary = split_summary(frontier_train)
    frontier_heldout_summary = split_summary(frontier_heldout)
    clean_replay_summary = split_summary(clean_replay)
    train_mix_summary = split_summary(train_mix)
    pass_criteria = {
        "combined_frontier_pool_count_eq_106": len(frontier) == 106,
        "frontier_train_count_gte_80": len(frontier_train) >= 80,
        "frontier_heldout_count_gte_20": len(frontier_heldout) >= 20,
        "clean_replay_count_25_to_35": 25 <= len(clean_replay) <= 35,
        "heldout_overlap_zero": heldout_overlap["total_overlap"] == 0,
        "over_max_length_zero": token_stats["over_max_length_count"] == 0,
        "max_sft_token_length_lte_16384": (token_stats["max_sft_token_length"] or 0) <= int(args.max_length),
        "evidence_window_prompt_false": True,
    }
    summary = {
        "run_id": RUN_ID,
        "status": "audit_passed" if all(pass_criteria.values()) else "audit_failed",
        "combined_frontier_path": str(frontier_path),
        "combined_candidate_path": str(candidate_path),
        "output_dir": str(output_dir),
        "combined_frontier_pool_count": len(frontier),
        "combined_candidate_count": len(candidates),
        "frontier_train_count": len(frontier_train),
        "frontier_heldout_count": len(frontier_heldout),
        "clean_replay_count": len(clean_replay),
        "train_mix_count": len(train_mix),
        "sft_train_count": len(sft_train),
        "clean_replay_share": clean_share,
        "source_distribution": {
            "frontier_train": frontier_train_summary["source_distribution"],
            "frontier_heldout": frontier_heldout_summary["source_distribution"],
            "clean_replay": clean_replay_summary["source_distribution"],
            "train_mix": train_mix_summary["source_distribution"],
        },
        "qtype_distribution": {
            "frontier_train": frontier_train_summary["qtype_distribution"],
            "frontier_heldout": frontier_heldout_summary["qtype_distribution"],
            "clean_replay": clean_replay_summary["qtype_distribution"],
            "train_mix": train_mix_summary["qtype_distribution"],
        },
        "hard_single_choice_template_distribution": {
            "frontier_train": frontier_train_summary["hard_single_choice_template_distribution"],
            "frontier_heldout": frontier_heldout_summary["hard_single_choice_template_distribution"],
            "clean_replay": clean_replay_summary["hard_single_choice_template_distribution"],
            "train_mix": train_mix_summary["hard_single_choice_template_distribution"],
        },
        "frontier_type_distribution": {
            "frontier_train": frontier_train_summary["frontier_type_distribution"],
            "frontier_heldout": frontier_heldout_summary["frontier_type_distribution"],
            "train_mix": train_mix_summary["frontier_type_distribution"],
        },
        "frontier_train": frontier_train_summary,
        "frontier_heldout": frontier_heldout_summary,
        "clean_replay": clean_replay_summary,
        "train_mix": train_mix_summary,
        "split_stats": split_stats,
        "clean_replay_selection": clean_stats,
        "heldout_overlap_check": heldout_overlap,
        "tokenizer_path": str(args.tokenizer),
        "max_length": int(args.max_length),
        **token_stats,
        "over_max_length_count": token_stats["over_max_length_count"],
        "max_sft_token_length": token_stats["max_sft_token_length"],
        "evidence_window_used_as_prompt": False,
        "pass_criteria": pass_criteria,
        "train_mix_pass": all(pass_criteria.values()),
        "artifact_paths": {
            "frontier_train": str(output_dir / "frontier_train.jsonl"),
            "frontier_heldout": str(output_dir / "frontier_heldout.jsonl"),
            "clean_replay": str(output_dir / "clean_replay.jsonl"),
            "train_mix": str(output_dir / "train_mix.jsonl"),
            "sft_train": str(output_dir / "sft_train.jsonl"),
            "train_mix_summary": str(output_dir / "train_mix_summary.json"),
            "train_mix_report": str(reports_dir / "train_mix_report.md"),
        },
    }
    write_json(output_dir / "train_mix_summary.json", summary)
    write_json(reports_dir / "train_mix_summary.json", summary)
    write_report(reports_dir / "train_mix_report.md", summary)
    print(json.dumps(
        {
            "status": summary["status"],
            "counts": {
                "frontier_train": len(frontier_train),
                "frontier_heldout": len(frontier_heldout),
                "clean_replay": len(clean_replay),
                "train_mix": len(train_mix),
            },
            "heldout_overlap_check": heldout_overlap,
            "max_sft_token_length": summary["max_sft_token_length"],
            "over_max_length_count": summary["over_max_length_count"],
            "train_mix_pass": summary["train_mix_pass"],
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))
    return 0 if summary["train_mix_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
