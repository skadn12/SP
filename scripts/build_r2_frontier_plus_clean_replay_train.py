#!/usr/bin/env python3
"""Build r2 frontier + clean_v1 replay SFT train data.

This script only builds data. It does not train a solver.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "eval_sets/graph_expansion_challenger_v1/r2_frontier_plus_clean_replay_train"
DEFAULT_FRONTIER = ROOT / "eval_reports/graph_expansion_challenger_v1/r2_auto_full_eval/solver_r1_frontier_candidates.jsonl"
DEFAULT_R2_SFT = ROOT / "eval_sets/graph_expansion_challenger_v1/r2_auto_full/sft_train.jsonl"

REPLAY_SOURCES = [
    {
        "source_tag": "clean_v1_old_solver_r1_train",
        "path": ROOT / "sft_data/clean_v1_solver_r1_train_chat.jsonl",
        "fallback_paths": [
            ROOT / "data/locomo_processed_datasets/_global_clean_v1_solver_r1_training_dataset/actual_sft/clean_v1_solver_r1_train_chat.jsonl",
            ROOT / "data/locomo_processed_datasets/_global_clean_v1_solver_r1/sft_data__clean_v1_solver_r1_train_chat.jsonl",
        ],
        "target": 40,
        "description": "original clean_v1 solver_r1 train replay",
    },
    {
        "source_tag": "clean_v1_stable_evidence_replay",
        "path": ROOT / "sft_data/clean_v1_train_evidence_train_chat.jsonl",
        "fallback_paths": [
            ROOT / "mainline/lineage/r3_selfplay/train_mix_ctx15360/r2_frontier_plus_clean_replay_train.jsonl",
            ROOT / "mainline/lineage/r2_selfplay/train_mix_with_s0_s2/r2_frontier_plus_clean_replay_train.jsonl",
        ],
        "source_tag_filter": "clean_v1_stable_evidence_replay",
        "target": 20,
        "description": "clean_v1 stable gold/evidence replay",
    },
    {
        "source_tag": "clean_v1_graph_aware_optioneval_replay",
        "path": ROOT / "sft_data/clean_v1_train_optioneval_k5_chat.jsonl",
        "fallback_paths": [
            ROOT / "mainline/lineage/r3_selfplay/train_mix_ctx15360/r2_frontier_plus_clean_replay_train.jsonl",
            ROOT / "mainline/lineage/r2_selfplay/train_mix_with_s0_s2/r2_frontier_plus_clean_replay_train.jsonl",
        ],
        "source_tag_filter": "clean_v1_graph_aware_optioneval_replay",
        "target": 20,
        "description": "clean_v1 graph-aware option-eval replay",
    },
]

FORBIDDEN_SOURCE_TERMS = (
    "train_v12_hard_combined",
    "heldout_v12_hard",
    "train_v13_pilot",
    "heldout_v13_pilot",
    "run_graph_expansion_challenger_v1.py",
)
HELDOUT_TERMS = ("sample6", "sample7", "sample8", "sample9", "conv-47", "conv-48", "conv-49", "conv-50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build r2 frontier plus clean_v1 replay SFT train data.")
    parser.add_argument("--frontier", action="append", default=None, help="Frontier JSONL. May be passed multiple times.")
    parser.add_argument("--r2-sft-train", action="append", default=None, help="Matching round SFT train JSONL. May be passed multiple times.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-replay", type=int, default=80)
    parser.add_argument("--min-replay", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()
    if args.frontier is None:
        args.frontier = [str(DEFAULT_FRONTIER)]
    if args.r2_sft_train is None:
        args.r2_sft_train = [str(DEFAULT_R2_SFT)]
    if len(args.frontier) != len(args.r2_sft_train):
        raise ValueError("--frontier and --r2-sft-train must have the same count")
    return args


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


def normalize(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def stable_hash(value: Any) -> str:
    return hashlib.sha1(normalize(value).encode("utf-8")).hexdigest()


def messages(record: dict[str, Any]) -> list[dict[str, str]]:
    value = record.get("messages")
    return value if isinstance(value, list) else []


def user_text(record: dict[str, Any]) -> str:
    return "\n".join(str(item.get("content") or "") for item in messages(record) if item.get("role") == "user")


def assistant_text(record: dict[str, Any]) -> str:
    return "\n".join(str(item.get("content") or "") for item in messages(record) if item.get("role") == "assistant")


def metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata")
    return value if isinstance(value, dict) else {}


def source_id(record: dict[str, Any]) -> str:
    meta = metadata(record)
    return str(
        meta.get("source_record_id")
        or meta.get("record_id")
        or record.get("source_record_id")
        or record.get("record_id")
        or record.get("sample_name")
        or ""
    )


def infer_question_type(record: dict[str, Any]) -> str:
    meta = metadata(record)
    explicit = meta.get("question_type") or record.get("question_type")
    if explicit:
        return str(explicit)
    system = " ".join(str(item.get("content") or "") for item in messages(record) if item.get("role") == "system").lower()
    assistant = assistant_text(record).strip()
    if "strict json" in system and "evidence-chain roles" in system:
        return "clean_v1_chain_role_json"
    if "strict json" in system and "cite the dialogue turn ids" in system:
        return "clean_v1_evidence_json"
    if re.match(r"^[A-F][\.\):]\s+", assistant, flags=re.IGNORECASE):
        return "single_choice"
    return "unknown"


def as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def numeric_stats(values: list[Any]) -> dict[str, Any]:
    nums: list[int] = []
    for value in values:
        parsed = as_int(value)
        if parsed is not None:
            nums.append(parsed)
    if not nums:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(nums),
        "min": min(nums),
        "max": max(nums),
        "mean": round(sum(nums) / len(nums), 6),
        "median": statistics.median(nums),
    }


def contamination_text(record: dict[str, Any], source_path: str = "") -> str:
    meta = metadata(record)
    fields = {
        "source_path": source_path,
        "record_id": record.get("record_id"),
        "sample_name": record.get("sample_name"),
        "source_record_id": record.get("source_record_id"),
        "metadata": {
            key: meta.get(key)
            for key in (
                "record_id",
                "source_record_id",
                "sample_name",
                "source_file",
                "source_path",
                "source_manifest_entry",
                "clean_split",
                "split",
                "source_split",
                "training_use",
                "parent_sample_id",
                "parent_pairing_key",
            )
        },
    }
    return normalize(fields)


def find_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    lower = text.lower()
    return [term for term in terms if term.lower() in lower]


def is_replay_allowed(record: dict[str, Any], source_path: str) -> tuple[bool, list[str]]:
    meta = metadata(record)
    reasons: list[str] = []
    text = contamination_text(record, source_path)
    for term in find_terms(text, FORBIDDEN_SOURCE_TERMS):
        reasons.append(f"forbidden_source:{term}")
    for term in find_terms(text, HELDOUT_TERMS):
        reasons.append(f"heldout_or_dev:{term}")
    training_use = normalize(meta.get("training_use"))
    if "holdout" in training_use or "review" in training_use:
        reasons.append(f"not_train_use:{training_use}")
    clean_split = normalize(meta.get("clean_split") or meta.get("source_split") or meta.get("split"))
    if clean_split and clean_split != "train":
        reasons.append(f"not_clean_train_split:{clean_split}")
    if not messages(record) or not assistant_text(record):
        reasons.append("missing_messages_or_target")
    return not reasons, reasons


def dedup_keys(record: dict[str, Any], source_tag: str = "") -> dict[str, str]:
    sid = source_id(record)
    return {
        "question_hash": stable_hash(user_text(record)),
        "gold_hash": stable_hash(assistant_text(record)),
        "source_id": stable_hash(f"{source_tag}:{sid}") if sid else "",
        "raw_source_id": sid,
    }


def add_mix_metadata(
    record: dict[str, Any],
    *,
    source_tag: str,
    source_path: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = copy.deepcopy(record)
    meta = dict(metadata(output))
    keys = dedup_keys(output, source_tag=source_tag)
    meta.update(
        {
            "source_tag": source_tag,
            "mix_source_path": source_path,
            "mix_question_hash": keys["question_hash"],
            "mix_gold_hash": keys["gold_hash"],
            "mix_source_id": keys["raw_source_id"],
            "question_type": meta.get("question_type") or infer_question_type(output),
        }
    )
    if extra:
        meta.update(extra)
    output["metadata"] = meta
    if "record_id" not in output and meta.get("record_id"):
        output["record_id"] = meta["record_id"]
    if "sample_name" not in output and meta.get("sample_name"):
        output["sample_name"] = meta["sample_name"]
    return output


def source_label_from_frontier(frontier_path: Path) -> str:
    parent = frontier_path.parent.name
    return parent or frontier_path.stem


def is_native_frontier_row(row: dict[str, Any]) -> bool:
    return bool(row.get("sample_id")) and not isinstance(row.get("reward"), dict) and not isinstance(row.get("eval"), dict)


def load_native_frontier(frontier_path: Path, r2_sft_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sft_by_record_id = {str(row.get("record_id") or metadata(row).get("record_id")): row for row in read_jsonl(r2_sft_path)}
    frontier_rows = read_jsonl(frontier_path)
    output: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    wrong = 0
    hard_correct = 0
    source_label = source_label_from_frontier(frontier_path)
    for row in frontier_rows:
        rid = str(row.get("sample_id") or "")
        if not rid:
            skipped["missing_native_sample_id"] += 1
            continue
        record = sft_by_record_id.get(rid)
        if not record:
            skipped["missing_native_sft_record"] += 1
            continue
        solver_eval = row.get("solver_eval") if isinstance(row.get("solver_eval"), dict) else {}
        correct_count = as_int(solver_eval.get("exact_correct_count"))
        if correct_count is None:
            correct_count = as_int(solver_eval.get("correct_count")) or 0
        successful_count = as_int(solver_eval.get("k")) or len(solver_eval.get("predictions") or []) or None
        first_correct = correct_count > 0
        source_tag = f"{source_label}_native_partial_frontier" if first_correct else f"{source_label}_native_zero_frontier"
        wrong += 0 if first_correct else 1
        hard_correct += 1 if first_correct else 0
        frontier_label = row.get("frontier_label") if isinstance(row.get("frontier_label"), dict) else {}
        output.append(
            add_mix_metadata(
                record,
                source_tag=source_tag,
                source_path=str(frontier_path),
                extra={
                    "mix_source": f"{source_label}_native_frontier",
                    "frontier_source_label": source_label,
                    "r2_frontier_record_id": rid,
                    "r2_frontier_bucket": solver_eval.get("bucket"),
                    "r2_frontier_reward": row.get("frontier_pool_priority"),
                    "r2_frontier_soft_accuracy": solver_eval.get("soft_accuracy"),
                    "r2_frontier_correct_count": correct_count,
                    "r2_frontier_successful_count": successful_count,
                    "r2_frontier_first_sample_correct": first_correct,
                    "r2_frontier_subset": "native_partial_frontier" if first_correct else "native_zero_frontier",
                    "native_frontier_category": frontier_label.get("category"),
                    "native_frontier_tier": frontier_label.get("frontier_tier"),
                    "qa_operation": row.get("qa_operation") or metadata(row).get("qa_operation"),
                    "qa_subtype": row.get("qa_subtype") or metadata(row).get("qa_subtype"),
                    "answer_type": row.get("answer_type") or metadata(row).get("answer_type"),
                    "answer_format": row.get("answer_format") or metadata(row).get("answer_format"),
                    "source_split": "train",
                },
            )
        )
    stats = {
        "input_frontier_rows": len(frontier_rows),
        "selected": len(output),
        "wrong_frontier": wrong,
        "hard_correct": hard_correct,
        "skipped": dict(skipped),
        "frontier_format": "native_current_method",
    }
    return output, stats


def load_r2_frontier(frontier_path: Path, r2_sft_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frontier_rows = read_jsonl(frontier_path)
    if frontier_rows and all(is_native_frontier_row(row) for row in frontier_rows):
        return load_native_frontier(frontier_path, r2_sft_path)
    sft_by_record_id = {str(row.get("record_id") or metadata(row).get("record_id")): row for row in read_jsonl(r2_sft_path)}
    output: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    wrong = 0
    hard_correct = 0
    source_label = source_label_from_frontier(frontier_path)
    for row in frontier_rows:
        reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
        eval_row = row.get("eval") if isinstance(row.get("eval"), dict) else {}
        rid = str(row.get("record_id") or reward.get("record_id") or eval_row.get("record_id") or "")
        if reward.get("split") != "train" or eval_row.get("eval_set") != "sft_train":
            skipped["non_train_frontier_row"] += 1
            continue
        if reward.get("hard_gate_valid") is not True:
            skipped["invalid_frontier_row"] += 1
            continue
        record = sft_by_record_id.get(rid)
        if not record:
            skipped["missing_r2_sft_record"] += 1
            continue
        first_correct = eval_row.get("correct") is True
        source_tag = f"{source_label}_hard_correct" if first_correct else f"{source_label}_wrong_frontier"
        wrong += 0 if first_correct else 1
        hard_correct += 1 if first_correct else 0
        output.append(
            add_mix_metadata(
                record,
                source_tag=source_tag,
                source_path=str(frontier_path),
                extra={
                    "mix_source": f"{source_label}_frontier",
                    "frontier_source_label": source_label,
                    "r2_frontier_record_id": rid,
                    "r2_frontier_bucket": reward.get("bucket"),
                    "r2_frontier_reward": reward.get("reward"),
                    "r2_frontier_soft_accuracy": reward.get("soft_accuracy"),
                    "r2_frontier_correct_count": reward.get("correct_count"),
                    "r2_frontier_successful_count": reward.get("successful_count"),
                    "r2_frontier_first_sample_correct": first_correct,
                    "r2_frontier_subset": "hard_correct" if first_correct else "wrong_frontier",
                    "source_split": "train",
                },
            )
        )
    stats = {
        "input_frontier_rows": len(frontier_rows),
        "selected": len(output),
        "wrong_frontier": wrong,
        "hard_correct": hard_correct,
        "skipped": dict(skipped),
    }
    return output, stats


def load_frontier_sources(frontier_paths: list[str], sft_paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    per_source: list[dict[str, Any]] = []
    duplicate_record_ids_removed = 0
    seen_record_ids: set[str] = set()
    skipped: Counter[str] = Counter()
    wrong = 0
    hard_correct = 0
    input_frontier_rows = 0
    for index, (frontier_raw, sft_raw) in enumerate(zip(frontier_paths, sft_paths)):
        frontier_path = Path(frontier_raw)
        sft_path = Path(sft_raw)
        source_records, source_stats = load_r2_frontier(frontier_path, sft_path)
        input_frontier_rows += int(source_stats.get("input_frontier_rows") or 0)
        source_selected = 0
        source_wrong = 0
        source_hard_correct = 0
        for record in source_records:
            meta = metadata(record)
            rid = str(meta.get("r2_frontier_record_id") or meta.get("record_id") or record.get("record_id") or "")
            dedup_key = rid if rid else f"{frontier_path}:{len(records)}"
            if dedup_key in seen_record_ids:
                duplicate_record_ids_removed += 1
                continue
            seen_record_ids.add(dedup_key)
            records.append(record)
            source_selected += 1
            if meta.get("r2_frontier_first_sample_correct") is True:
                hard_correct += 1
                source_hard_correct += 1
            else:
                wrong += 1
                source_wrong += 1
        skipped.update(source_stats.get("skipped") or {})
        per_source.append(
            {
                "source_index": index,
                "frontier_path": str(frontier_path),
                "r2_sft_path": str(sft_path),
                "source_label": source_label_from_frontier(frontier_path),
                "input_frontier_rows": source_stats.get("input_frontier_rows"),
                "selected": source_selected,
                "wrong_frontier": source_wrong,
                "hard_correct": source_hard_correct,
                "selected_after_cross_source_record_id_dedup": source_selected,
                "source_stats": source_stats,
            }
        )
    stats = {
        "input_frontier_rows": input_frontier_rows,
        "selected": len(records),
        "wrong_frontier": wrong,
        "hard_correct": hard_correct,
        "duplicate_record_ids_removed": duplicate_record_ids_removed,
        "skipped": dict(skipped),
        "per_source": per_source,
    }
    return records, stats


def replay_candidate_paths(source: dict[str, Any]) -> list[Path]:
    paths = [Path(source["path"])]
    paths.extend(Path(path) for path in source.get("fallback_paths", []))
    seen: set[str] = set()
    output: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            output.append(path)
    return output


def load_replay_rows(source: dict[str, Any]) -> tuple[list[dict[str, Any]], Path | None]:
    tag_filter = source.get("source_tag_filter")
    for path in replay_candidate_paths(source):
        if not path.exists():
            continue
        rows = read_jsonl(path)
        if tag_filter:
            rows = [row for row in rows if metadata(row).get("source_tag") == tag_filter]
        return rows, path
    return [], None


def select_replay(
    *,
    target_replay: int,
    seed: int,
    blocked_question_hashes: set[str],
    blocked_gold_hashes: set[str],
    blocked_source_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    selected_keys: list[dict[str, str]] = []
    duplicate_removed = Counter()
    rejected = Counter()
    source_attempts = Counter()
    source_selected = Counter()
    source_available = Counter()
    resolved_source_paths: dict[str, str | None] = {}

    def try_add(record: dict[str, Any], source_tag: str, source_path: Path) -> bool:
        source_attempts[source_tag] += 1
        allowed, reasons = is_replay_allowed(record, str(source_path))
        if not allowed:
            for reason in reasons:
                rejected[reason] += 1
            return False
        tagged = add_mix_metadata(
            record,
            source_tag=source_tag,
            source_path=str(source_path),
            extra={"mix_source": "clean_v1_replay", "source_split": "train"},
        )
        keys = dedup_keys(tagged, source_tag=source_tag)
        if keys["question_hash"] in blocked_question_hashes:
            duplicate_removed["question_hash"] += 1
            return False
        if keys["gold_hash"] in blocked_gold_hashes:
            duplicate_removed["gold_hash"] += 1
            return False
        if keys["source_id"] and keys["source_id"] in blocked_source_ids:
            duplicate_removed["source_id"] += 1
            return False
        selected.append(tagged)
        selected_keys.append(keys)
        blocked_question_hashes.add(keys["question_hash"])
        blocked_gold_hashes.add(keys["gold_hash"])
        if keys["source_id"]:
            blocked_source_ids.add(keys["source_id"])
        source_selected[source_tag] += 1
        return True

    all_valid_by_source: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    for source in REPLAY_SOURCES:
        source_tag = str(source["source_tag"])
        rows, resolved_path = load_replay_rows(source)
        resolved_source_paths[source_tag] = str(resolved_path) if resolved_path else None
        candidates: list[tuple[dict[str, Any], Path]] = []
        path = resolved_path or Path(source["path"])
        for record in rows:
            allowed, _reasons = is_replay_allowed(record, str(path))
            if allowed:
                candidates.append((record, path))
        rng.shuffle(candidates)
        all_valid_by_source[source_tag] = candidates
        source_available[source_tag] = len(candidates)

    for source in REPLAY_SOURCES:
        source_tag = str(source["source_tag"])
        target = min(int(source.get("target", 0)), max(0, target_replay - len(selected)))
        for record, path in all_valid_by_source.get(source_tag, []):
            if source_selected[source_tag] >= target or len(selected) >= target_replay:
                break
            try_add(record, source_tag, path)

    if len(selected) < target_replay:
        leftovers: list[tuple[str, dict[str, Any], Path]] = []
        for source_tag, rows in all_valid_by_source.items():
            leftovers.extend((source_tag, record, path) for record, path in rows)
        rng.shuffle(leftovers)
        for source_tag, record, path in leftovers:
            if len(selected) >= target_replay:
                break
            try_add(record, source_tag, path)

    stats = {
        "target_replay": target_replay,
        "selected": len(selected),
        "source_available_after_filter": dict(source_available),
        "source_attempts": dict(source_attempts),
        "source_selected": dict(source_selected),
        "resolved_source_paths": resolved_source_paths,
        "duplicate_removed_count": sum(duplicate_removed.values()),
        "duplicate_removed_by_reason": dict(duplicate_removed),
        "rejected_by_reason": dict(rejected),
    }
    return selected, stats


def count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        meta = metadata(record)
        value = meta.get(key) or record.get(key) or "unknown"
        counts[str(value)] += 1
    return dict(counts.most_common())


def contamination_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    heldout_hits: list[dict[str, Any]] = []
    forbidden_hits: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        text = contamination_text(record, str(metadata(record).get("mix_source_path") or ""))
        heldout_terms = find_terms(text, HELDOUT_TERMS)
        forbidden_terms = find_terms(text, FORBIDDEN_SOURCE_TERMS)
        if heldout_terms:
            heldout_hits.append({"index": index, "record_id": record.get("record_id"), "terms": heldout_terms})
        if forbidden_terms:
            forbidden_hits.append({"index": index, "record_id": record.get("record_id"), "terms": forbidden_terms})
    return {
        "heldout_contamination_bad": len(heldout_hits),
        "forbidden_source_contamination_bad": len(forbidden_hits),
        "heldout_hits_preview": heldout_hits[:10],
        "forbidden_hits_preview": forbidden_hits[:10],
    }


def build_summary(
    records: list[dict[str, Any]],
    *,
    r2_stats: dict[str, Any],
    replay_stats: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    source_counts = count_by(records, "source_tag")
    summary: dict[str, Any] = {
        "output_dir": str(output_dir),
        "total_sample_count": len(records),
        "r2_frontier_count": r2_stats["selected"],
        "r2_wrong_frontier_count": r2_stats["wrong_frontier"],
        "r2_hard_correct_count": r2_stats["hard_correct"],
        "clean_v1_replay_count": replay_stats["selected"],
        "composition": {
            "r2_frontier_ratio": round(r2_stats["selected"] / len(records), 6) if records else None,
            "clean_v1_replay_ratio": round(replay_stats["selected"] / len(records), 6) if records else None,
        },
        "question_type_distribution": count_by(records, "question_type"),
        "trap_type_distribution": count_by(records, "trap_type"),
        "source_tag_distribution": source_counts,
        "state_chain_depth_stats": numeric_stats([metadata(record).get("state_chain_depth") for record in records]),
        "added_turn_count_stats": numeric_stats([metadata(record).get("added_turn_count") for record in records]),
        "gold_origin_distribution": count_by(records, "gold_origin"),
        "duplicate_removal_count": replay_stats["duplicate_removed_count"],
        "duplicate_removed_by_reason": replay_stats["duplicate_removed_by_reason"],
        "replay_source_stats": replay_stats,
        "r2_source_stats": r2_stats,
        "contamination_checks": contamination_summary(records),
        "replay_sources": [
            {
                key: (
                    str(value)
                    if isinstance(value, Path)
                    else [str(item) if isinstance(item, Path) else item for item in value]
                    if isinstance(value, list)
                    else value
                )
                for key, value in source.items()
            }
            for source in REPLAY_SOURCES
        ],
        "notes": [
            "All frontier candidates are train split rows joined back to their matching round sft_train JSONL by record_id.",
            "Clean_v1 replay rows are copied from train chat JSONL files without changing messages or target answers.",
            "k=5 solver fields from r2 are metadata only; target answers are the original assistant messages.",
            "This script only builds train data; it does not train a solver.",
        ],
    }
    return summary


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# r2_frontier_plus_clean_replay_train",
        "",
        f"- total sample count: {summary['total_sample_count']}",
        f"- r2 frontier count: {summary['r2_frontier_count']}",
        f"- r2 wrong frontier count: {summary['r2_wrong_frontier_count']}",
        f"- r2 hard correct count: {summary['r2_hard_correct_count']}",
        f"- clean_v1 replay count: {summary['clean_v1_replay_count']}",
        f"- r2 frontier ratio: {summary['composition']['r2_frontier_ratio']}",
        f"- clean_v1 replay ratio: {summary['composition']['clean_v1_replay_ratio']}",
        f"- duplicate removal count: {summary['duplicate_removal_count']}",
        "",
        "## Contamination Checks",
        "",
    ]
    checks = summary["contamination_checks"]
    lines.append(f"- heldout contamination check: {checks['heldout_contamination_bad']}")
    lines.append(f"- forbidden source contamination check: {checks['forbidden_source_contamination_bad']}")
    lines.extend(["", "## Question Types", ""])
    for key, value in summary["question_type_distribution"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Trap Types", ""])
    for key, value in summary["trap_type_distribution"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Source Tags", ""])
    for key, value in summary["source_tag_distribution"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Numeric Stats", ""])
    for key in ("state_chain_depth_stats", "added_turn_count_stats"):
        lines.append(f"- {key}: `{json.dumps(summary[key], ensure_ascii=False)}`")
    lines.extend(["", "## Gold Origin", ""])
    for key, value in summary["gold_origin_distribution"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Notes", ""])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    r2_records, r2_stats = load_frontier_sources(args.frontier, args.r2_sft_train)
    if r2_stats["selected"] <= 0:
        raise RuntimeError(f"No train frontier rows selected ({r2_stats})")

    blocked_question_hashes = {dedup_keys(record, metadata(record).get("source_tag") or "")["question_hash"] for record in r2_records}
    blocked_gold_hashes = {dedup_keys(record, metadata(record).get("source_tag") or "")["gold_hash"] for record in r2_records}
    blocked_source_ids = {
        dedup_keys(record, metadata(record).get("source_tag") or "")["source_id"]
        for record in r2_records
        if dedup_keys(record, metadata(record).get("source_tag") or "")["source_id"]
    }
    replay_records, replay_stats = select_replay(
        target_replay=args.target_replay,
        seed=args.seed,
        blocked_question_hashes=blocked_question_hashes,
        blocked_gold_hashes=blocked_gold_hashes,
        blocked_source_ids=blocked_source_ids,
    )
    if replay_stats["selected"] < args.min_replay:
        raise RuntimeError(f"Only selected {replay_stats['selected']} replay rows; minimum is {args.min_replay}")

    records = r2_records + replay_records
    summary = build_summary(records, r2_stats=r2_stats, replay_stats=replay_stats, output_dir=output_dir)
    checks = summary["contamination_checks"]
    if checks["heldout_contamination_bad"] or checks["forbidden_source_contamination_bad"]:
        raise RuntimeError(f"Contamination check failed: {checks}")

    write_jsonl(output_dir / "r2_frontier_plus_clean_replay_train.jsonl", records)
    write_jsonl(output_dir / "r2_frontier_plus_clean_replay_sft_train.jsonl", records)
    write_json(output_dir / "train_summary.json", summary)
    (output_dir / "train_summary.md").write_text(format_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "total": summary["total_sample_count"],
        "r2_frontier": summary["r2_frontier_count"],
        "clean_v1_replay": summary["clean_v1_replay_count"],
        "duplicate_removed": summary["duplicate_removal_count"],
        "heldout_contamination_bad": checks["heldout_contamination_bad"],
        "forbidden_source_contamination_bad": checks["forbidden_source_contamination_bad"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
