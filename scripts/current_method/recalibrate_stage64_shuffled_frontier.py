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
import difflib
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
from scripts.current_method import stage3_operation_controller  # noqa: E402
from scripts.current_method import stage64_quality  # noqa: E402


DEFAULT_INPUT = (
    ROOT
    / "base_restart_current_method_validation/69_stage64_api_selfplay500_ctx32768/"
    / "option_shuffled_seed20260710/clean_samples.jsonl"
)
DEFAULT_OUTPUT = ROOT / "base_restart_current_method_validation/71_stage64_shuffled_recalibrated_stage2"
DEFAULT_BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_STAGE2_SOLVER = ROOT / "base_restart_current_method_validation/accepted_models/solver_base_restart_stage2_merged"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


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


def normalized_option_text(value: Any) -> str:
    return stage64_quality.normalized_option_text(value)


def option_quality_issues(
    sample: dict[str, Any],
    *,
    near_duplicate_threshold: float = 0.92,
    enforce_semantic_quality: bool = True,
) -> list[dict[str, Any]]:
    """Return hard option-quality failures before spending solver tokens."""

    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    issues: list[dict[str, Any]] = []
    if len(options) < 2 or not all(isinstance(item, dict) for item in options):
        return [{"reason": "invalid_option_container", "option_count": len(options)}]
    ids = [str(item.get("id") or "") for item in options]
    texts = [normalized_option_text(item.get("text")) for item in options]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        issues.append({"reason": "duplicate_or_empty_option_id", "ids": ids})
    if any(not text for text in texts):
        issues.append({"reason": "empty_option_text"})
    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            if not texts[left] or not texts[right]:
                continue
            if texts[left] == texts[right]:
                issues.append(
                    {
                        "reason": "exact_duplicate_option_text",
                        "labels": [ids[left], ids[right]],
                        "text": texts[left],
                    }
                )
                continue
            ratio = difflib.SequenceMatcher(None, texts[left], texts[right]).ratio()
            if near_duplicate_threshold > 0 and ratio >= near_duplicate_threshold:
                issues.append(
                    {
                        "reason": "near_duplicate_option_text",
                        "labels": [ids[left], ids[right]],
                        "similarity": round(ratio, 6),
                    }
                )
    gold = qa.get("gold")
    gold_items = [str(item) for item in gold] if isinstance(gold, list) else [str(gold or "")]
    if len(gold_items) != 1 or gold_items[0] not in set(ids):
        issues.append({"reason": "invalid_single_choice_gold", "gold": gold_items, "ids": ids})
    elif gold_items[0] in ids:
        gold_index = ids.index(gold_items[0])
        duplicate_labels = [ids[index] for index, text in enumerate(texts) if text == texts[gold_index]]
        if len(duplicate_labels) > 1:
            issues.append(
                {
                    "reason": "gold_text_maps_to_multiple_labels",
                    "labels": duplicate_labels,
                    "text": texts[gold_index],
                }
            )
    if enforce_semantic_quality:
        issues.extend(stage64_quality.semantic_quality_issues(sample))
    return issues


def quality_filter_samples(
    samples: list[dict[str, Any]],
    *,
    near_duplicate_threshold: float = 0.92,
    reject_near_duplicates: bool = False,
    enforce_semantic_quality: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejected_sample_reason_counts: Counter[str] = Counter()
    issue_occurrence_counts: Counter[str] = Counter()
    for sample in samples:
        issues = option_quality_issues(
            sample,
            near_duplicate_threshold=near_duplicate_threshold,
            enforce_semantic_quality=enforce_semantic_quality,
        )
        blocking = [
            issue
            for issue in issues
            if reject_near_duplicates or issue.get("reason") != "near_duplicate_option_text"
        ]
        if blocking:
            row = copy.deepcopy(sample)
            row["stage64_quality_rejection"] = {"issues": issues, "blocking_issues": blocking}
            rejected.append(row)
            reasons = [str(issue.get("reason") or "unknown") for issue in blocking]
            rejected_sample_reason_counts.update(set(reasons))
            issue_occurrence_counts.update(reasons)
        else:
            row = copy.deepcopy(sample)
            if issues:
                row["stage64_quality_warnings"] = issues
            accepted.append(row)
    summary = {
        "input_count": len(samples),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "reject_near_duplicates": reject_near_duplicates,
        "near_duplicate_threshold": near_duplicate_threshold,
        "semantic_quality_gate_enabled": enforce_semantic_quality,
        "rejected_sample_reasons": dict(rejected_sample_reason_counts.most_common()),
        "blocking_issue_occurrences": dict(issue_occurrence_counts.most_common()),
    }
    return accepted, rejected, summary


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


def _signature_rollout_outcome(
    sample: dict[str, Any],
) -> tuple[str, int, int]:
    audit = sample.get("stage64_permutation_eval")
    if isinstance(audit, dict):
        status = str(audit.get("status") or "unknown")
        k_eval = int(audit.get("k_eval_per_permutation") or 0)
        permutations = audit.get("permutations")
        if k_eval < 1 or not isinstance(permutations, list) or not permutations:
            raise ValueError(
                "Invalid stage64_permutation_eval for reasoning-signature metrics"
            )
        exact_rollouts = 0
        for permutation in permutations:
            if not isinstance(permutation, dict):
                raise ValueError(
                    "Invalid permutation row for reasoning-signature metrics"
                )
            value = permutation.get("exact_correct_count")
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    "Missing exact_correct_count for reasoning-signature metrics"
                )
            if not 0 <= value <= k_eval:
                raise ValueError(
                    f"Invalid exact_correct_count for reasoning-signature metrics: {value}"
                )
            exact_rollouts += value
        return status, exact_rollouts, len(permutations) * k_eval

    solver_eval = sample.get("solver_eval")
    solver_eval = solver_eval if isinstance(solver_eval, dict) else {}
    k_eval = int(solver_eval.get("k") or 0)
    exact_count = solver_eval.get("exact_correct_count")
    if (
        k_eval < 1
        or isinstance(exact_count, bool)
        or not isinstance(exact_count, int)
        or not 0 <= exact_count <= k_eval
    ):
        return "unknown", 0, 0
    if exact_count == 0:
        status = "robust_hard0"
    elif exact_count == k_eval:
        status = "robust_easy5"
    else:
        status = "robust_frontier"
    return status, exact_count, k_eval


def reasoning_signature_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize target-defined reasoning signatures without running a solver."""

    distribution: Counter[str] = Counter()
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    exact_rollouts: Counter[str] = Counter()
    evaluated_rollouts: Counter[str] = Counter()
    signatures: dict[str, dict[str, Any]] = {}
    legacy_untyped_count = 0
    for sample in samples:
        signature = stage3_operation_controller.sample_reasoning_signature(sample)
        if signature is None:
            legacy_untyped_count += 1
            distribution["legacy_untyped"] += 1
            continue
        key = stage3_operation_controller.reasoning_signature_key(signature)
        signatures[key] = signature
        distribution[key] += 1
        status, correct, total = _signature_rollout_outcome(sample)
        status_counts[key][status] += 1
        exact_rollouts[key] += correct
        evaluated_rollouts[key] += total

    metrics: dict[str, dict[str, Any]] = {}
    for key, signature in sorted(signatures.items()):
        total = int(distribution[key])
        by_status = {
            status: int(status_counts[key].get(status) or 0)
            for status in (*stage3_operation_controller.AUDIT_STATUSES, "unknown")
        }
        rollout_total = int(evaluated_rollouts[key])
        exact_total = int(exact_rollouts[key])
        metrics[key] = {
            "signature": dict(signature),
            "total": total,
            "status_counts": by_status,
            "exact_correct_rollout_count": exact_total,
            "evaluated_rollout_count": rollout_total,
            "weighted_rollout_exact_accuracy": (
                round(exact_total / rollout_total, 6) if rollout_total else None
            ),
            "frontier_yield": (
                round(by_status["robust_frontier"] / total, 6) if total else None
            ),
        }
    return {
        "typed_sample_count": len(samples) - legacy_untyped_count,
        "legacy_untyped_count": legacy_untyped_count,
        "distribution": dict(sorted(distribution.items())),
        "metrics": metrics,
    }


def qa_gold(sample: dict[str, Any]) -> str:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    gold = qa.get("gold")
    if isinstance(gold, list) and gold:
        return str(gold[0])
    return str(gold or "")


def group_key(sample: dict[str, Any], *, level: str = "parent_sample_id") -> str:
    meta = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    seed = sample.get("seed") if isinstance(sample.get("seed"), dict) else {}
    seed_source = sample.get("seed_source") if isinstance(sample.get("seed_source"), dict) else {}
    seed_nested = seed.get("source") if isinstance(seed.get("source"), dict) else {}
    if level == "lineage_group":
        candidates = (
            sample.get("lineage_group"),
            meta.get("lineage_group"),
            seed_source.get("lineage_group"),
            seed_nested.get("lineage_group"),
            seed_source.get("source_lineage_key"),
            seed_nested.get("source_lineage_key"),
        )
    elif level == "parent_sample_id":
        candidates = (
            seed_source.get("parent_sample_id"),
            seed_nested.get("parent_sample_id"),
            seed.get("parent_sample_id"),
            seed_source.get("parent_chunk_id"),
            seed_nested.get("parent_chunk_id"),
        )
    elif level == "parent_chunk_id":
        candidates = (
            seed_source.get("parent_chunk_id"),
            seed_nested.get("parent_chunk_id"),
            seed_source.get("parent_sample_id"),
            seed_nested.get("parent_sample_id"),
        )
    elif level == "source_seed_id":
        candidates = ()
    else:
        raise ValueError(f"Unsupported group level: {level}")
    fallback = (
        meta.get("source_seed_id"),
        sample.get("source_id"),
        seed.get("sample_id"),
        seed_source.get("sample_id"),
        sample.get("sample_id"),
    )
    return str(next((value for value in (*candidates, *fallback) if value), ""))


def _sync_top_level_qa_fields(sample: dict[str, Any]) -> None:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    gold = copy.deepcopy(qa.get("gold") or [])
    sample["question_type"] = qa.get("question_type")
    sample["question"] = qa.get("question")
    sample["options"] = copy.deepcopy(qa.get("options") or [])
    sample["gold_answer"] = gold
    sample["gold_answers"] = copy.deepcopy(gold)
    sample["gold_options"] = copy.deepcopy(gold)
    sample["gold_option"] = gold[0] if isinstance(gold, list) and gold else ""


def cyclic_option_permutation(sample: dict[str, Any], *, shift: int) -> dict[str, Any]:
    """Return one cyclic option permutation with every label-bearing field updated."""

    row = strip_eval_fields(sample)
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    if len(options) < 2:
        raise ValueError(f"Sample {sample.get('sample_id')} has fewer than two options")
    shift = int(shift) % len(options)
    ordered = options[shift:] + options[:shift]
    old_to_new: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []
    for index, option in enumerate(ordered):
        old_id = str(option.get("id") or option.get("option_id") or LETTERS[index])
        new_id = LETTERS[index]
        old_to_new[old_id] = new_id
        item = copy.deepcopy(option)
        item["id"] = new_id
        item["option_id"] = new_id
        normalized.append(item)
    old_gold = [str(item) for item in (qa.get("gold") or [])]
    new_gold = [old_to_new[item] for item in old_gold if item in old_to_new]
    if len(new_gold) != len(old_gold) or not new_gold:
        raise ValueError(f"Could not remap gold for sample {sample.get('sample_id')}: {old_gold}")
    for item in normalized:
        is_gold = item["id"] in new_gold
        item["is_gold"] = is_gold
        item["truth_value"] = is_gold
    qa["options"] = normalized
    qa["gold"] = new_gold
    row["qa"] = qa
    base_sample_id = str(sample.get("sample_id") or "")
    row["sample_id"] = f"{base_sample_id}::cyclic{shift}"
    row["stage64_permutation"] = {
        "base_sample_id": base_sample_id,
        "mode": "cyclic",
        "shift": shift,
        "option_count": len(options),
        "old_to_new": old_to_new,
        "gold": new_gold,
    }
    _sync_top_level_qa_fields(row)
    return row


def build_permutation_rows(samples: list[dict[str, Any]], *, mode: str) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        base_id = str(sample.get("sample_id") or "")
        qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
        options = qa.get("options") if isinstance(qa.get("options"), list) else []
        shifts = range(len(options)) if mode == "cyclic" else range(1)
        for shift in shifts:
            variant = cyclic_option_permutation(sample, shift=shift)
            if mode == "current":
                variant["stage64_permutation"]["mode"] = "current"
            rows.append(variant)
            grouped[base_id].append(variant)
    return rows, grouped


def aggregate_permutation_evaluations(
    samples: list[dict[str, Any]],
    grouped_variants: dict[str, list[dict[str, Any]]],
    *,
    k_eval: int,
    min_frontier_permutation_share: float,
) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    gold_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for sample in samples:
        base_id = str(sample.get("sample_id") or "")
        variants = grouped_variants.get(base_id) or []
        if not variants:
            raise RuntimeError(f"Missing evaluated permutations for {base_id}")
        counts: list[int] = []
        partials: list[float] = []
        per_permutation: list[dict[str, Any]] = []
        for variant in variants:
            se = variant.get("solver_eval") if isinstance(variant.get("solver_eval"), dict) else {}
            count = int(se.get("exact_correct_count") or 0)
            counts.append(count)
            partials.append(float(se.get("partial_credit") or 0.0))
            gold = qa_gold(variant)
            gold_stats[gold]["correct"] += count
            gold_stats[gold]["total"] += int(k_eval)
            per_permutation.append(
                {
                    "shift": variant.get("stage64_permutation", {}).get("shift"),
                    "gold": gold,
                    "exact_correct_count": count,
                    "bucket": se.get("bucket"),
                    "soft_accuracy": se.get("soft_accuracy"),
                    "partial_credit": se.get("partial_credit"),
                    "predictions": copy.deepcopy(se.get("predictions") or []),
                }
            )
        frontier_share = sum(1 for count in counts if 0 < count < k_eval) / len(counts)
        if all(count == 0 for count in counts):
            status = "robust_hard0"
            aggregate_count: int | None = 0
        elif all(count == k_eval for count in counts):
            status = "robust_easy5"
            aggregate_count = k_eval
        elif frontier_share >= min_frontier_permutation_share:
            status = "robust_frontier"
            aggregate_count = min(k_eval - 1, max(1, round(sum(counts) / len(counts))))
        else:
            status = "position_sensitive"
            aggregate_count = None
        status_counts[status] += 1
        aggregate_soft = sum(counts) / (len(counts) * k_eval)
        canonical_diag = copy.deepcopy((variants[0].get("solver_eval", {}).get("diagnostics") or {}))
        canonical_diag.update(
            {
                "permutation_mode": variants[0].get("stage64_permutation", {}).get("mode"),
                "permutation_count": len(variants),
                "frontier_permutation_share": round(frontier_share, 6),
                "permutation_status": status,
            }
        )
        sample["solver_eval"] = {
            "k": k_eval,
            "predictions": copy.deepcopy(variants[0].get("solver_eval", {}).get("predictions") or []),
            "exact_correct_count": aggregate_count,
            "soft_accuracy": round(aggregate_soft, 6),
            "partial_credit": round(sum(partials) / len(partials), 6),
            "bucket": f"{aggregate_count}/{k_eval}" if aggregate_count is not None else "position_sensitive",
            "diagnostics": canonical_diag,
        }
        sample["stage64_permutation_eval"] = {
            "status": status,
            "k_eval_per_permutation": k_eval,
            "permutation_count": len(variants),
            "frontier_permutation_share": round(frontier_share, 6),
            "required_frontier_permutation_share": min_frontier_permutation_share,
            "permutations": per_permutation,
        }
    by_gold = {
        gold: {
            **values,
            "accuracy": round(values["correct"] / values["total"], 6) if values["total"] else None,
        }
        for gold, values in sorted(gold_stats.items())
    }
    signature_summary = reasoning_signature_summary(samples)
    return {
        "status_counts": dict(status_counts.most_common()),
        "by_gold_label": by_gold,
        "reasoning_signature_typed_count": signature_summary["typed_sample_count"],
        "reasoning_signature_legacy_untyped_count": signature_summary[
            "legacy_untyped_count"
        ],
        "reasoning_signature_distribution": signature_summary["distribution"],
        "reasoning_signature_metrics": signature_summary["metrics"],
    }


def attack_skill(sample: dict[str, Any]) -> str:
    return frontier_selection.attack_skill(sample)


def question_type(sample: dict[str, Any]) -> str:
    return frontier_selection.question_type(sample)


def split_frontier_by_group(
    frontier: list[dict[str, Any]],
    *,
    seed: int,
    heldout_ratio: float,
    group_level: str = "parent_sample_id",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in frontier:
        groups[group_key(sample, level=group_level)].append(sample)
    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)

    target = max(1, round(len(frontier) * heldout_ratio)) if frontier and heldout_ratio > 0 else 0
    heldout_groups: set[str] = set()
    heldout_count = 0
    # Prefer not to overshoot too badly, but keep whole groups intact.
    for key, rows in sorted(group_items, key=lambda item: (stable_int([seed, item[0]]), item[0])):
        if heldout_count >= target:
            break
        heldout_groups.add(key)
        heldout_count += len(rows)

    heldout = [sample for sample in frontier if group_key(sample, level=group_level) in heldout_groups]
    train = [sample for sample in frontier if group_key(sample, level=group_level) not in heldout_groups]
    summary = {
        "strategy": f"grouped by {group_level} after post-shuffle fixed-solver recalibration",
        "group_level": group_level,
        "seed": seed,
        "heldout_ratio_target": heldout_ratio,
        "frontier_total": len(frontier),
        "frontier_train": len(train),
        "frontier_heldout": len(heldout),
        "train_groups": len({group_key(s, level=group_level) for s in train}),
        "heldout_groups": len({group_key(s, level=group_level) for s in heldout}),
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
    all_signatures = reasoning_signature_summary(samples)
    frontier_signatures = reasoning_signature_summary(frontier)
    hard0_signatures = reasoning_signature_summary(hard0)
    easy5_signatures = reasoning_signature_summary(easy5)
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
        "reasoning_signature_typed_count": all_signatures["typed_sample_count"],
        "reasoning_signature_legacy_untyped_count": all_signatures[
            "legacy_untyped_count"
        ],
        "reasoning_signature_distribution": all_signatures["distribution"],
        "reasoning_signature_metrics": all_signatures["metrics"],
        "frontier_reasoning_signature_distribution": frontier_signatures[
            "distribution"
        ],
        "frontier_reasoning_signature_metrics": frontier_signatures["metrics"],
        "hard0_reasoning_signature_distribution": hard0_signatures["distribution"],
        "easy5_reasoning_signature_distribution": easy5_signatures["distribution"],
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
    parser.add_argument(
        "--accepted-stage2-solver-model",
        type=Path,
        default=DEFAULT_STAGE2_SOLVER,
        help="Reference path defining the accepted base-restart Stage2 checkpoint.",
    )
    parser.add_argument("--k-eval", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=None,
        help="Optional challenger-generation seed recorded for provenance; does not affect solver evaluation.",
    )
    parser.add_argument("--heldout-ratio", type=float, default=0.20)
    parser.add_argument(
        "--group-level",
        choices=("lineage_group", "parent_sample_id", "parent_chunk_id", "source_seed_id"),
        default="parent_sample_id",
        help="Highest provenance unit kept intact across frontier train/heldout.",
    )
    parser.add_argument(
        "--permutation-mode",
        choices=("cyclic", "current"),
        default="cyclic",
        help="Evaluate every cyclic option position by default; 'current' is diagnostic-only legacy behavior.",
    )
    parser.add_argument(
        "--min-frontier-permutation-share",
        type=float,
        default=1.0,
        help="Fraction of option permutations that must independently remain in the 1/K..K-1/K band.",
    )
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument(
        "--reject-near-duplicate-options",
        action="store_true",
        help="Also reject lexical near-duplicates. Off by default because minimal-pair distractors can be valid.",
    )
    parser.add_argument(
        "--skip-semantic-quality-gate",
        action="store_true",
        help="Diagnostic-only: skip answer-type, operation-semantic, and graph-hardness checks.",
    )
    parser.add_argument("--quality-audit-only", action="store_true", help="Validate/filter input without loading a solver.")
    parser.add_argument(
        "--write-all-clean-sft",
        action="store_true",
        help="Also write all quality-passed rows as clean_samples_sft.jsonl. Never names them frontier train.",
    )
    parser.add_argument(
        "--allow-nonaccepted-solver",
        action="store_true",
        help="Allow a checkpoint other than the accepted base-restart Stage2 solver (diagnostic use only).",
    )
    parser.add_argument("--max-largest-question-type-share", type=float, default=0.35)
    parser.add_argument("--max-largest-attack-skill-share", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.k_eval != 5:
        raise ValueError("Stage64 frontier policy is calibrated for K=5; --k-eval must equal 5.")
    if not 0.0 < args.min_frontier_permutation_share <= 1.0:
        raise ValueError("--min-frontier-permutation-share must be in (0, 1].")
    if not 0.0 <= args.heldout_ratio < 1.0:
        raise ValueError("--heldout-ratio must be in [0, 1).")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")
    if args.temperature <= 0.0:
        raise ValueError("Stage64 K5 frontier mining requires --temperature > 0.")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1].")
    output_dir = args.output_dir
    report_dir = args.report_dir or (output_dir / "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    input_samples = [strip_eval_fields(row) for row in read_jsonl(args.input)]
    if not input_samples:
        raise RuntimeError(f"No samples found in {args.input}")
    sample_ids = [str(sample.get("sample_id") or "") for sample in input_samples]
    if any(not sample_id for sample_id in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Input sample_id values must be non-empty and unique.")

    samples, quality_rejected, quality_summary = quality_filter_samples(
        input_samples,
        near_duplicate_threshold=args.near_duplicate_threshold,
        reject_near_duplicates=args.reject_near_duplicate_options,
        enforce_semantic_quality=not args.skip_semantic_quality_gate,
    )
    write_jsonl(output_dir / "quality_rejected_samples.jsonl", quality_rejected)
    write_jsonl(output_dir / "quality_passed_unevaluated_samples.jsonl", samples)
    write_json(report_dir / "option_quality_summary.json", quality_summary)
    if args.quality_audit_only:
        audit_summary = {
            "input": str(args.input),
            "output_dir": str(output_dir),
            "mode": "quality_audit_only",
            "quality": quality_summary,
        }
        write_json(output_dir / "quality_audit_summary.json", audit_summary)
        write_json(report_dir / "quality_audit_summary.json", audit_summary)
        print(json.dumps(audit_summary, ensure_ascii=False, indent=2))
        return 0
    if not samples:
        raise RuntimeError("No samples remain after the Stage64 option-quality gate.")

    accepted_solver = args.accepted_stage2_solver_model.resolve()
    requested_solver = args.solver_model.resolve()
    if not args.allow_nonaccepted_solver and requested_solver != accepted_solver:
        raise RuntimeError(
            "Stage64 recalibration must use the accepted base-restart Stage2 solver. "
            f"requested={requested_solver} accepted={accepted_solver}; pass --allow-nonaccepted-solver only for diagnostics."
        )

    permutation_rows, grouped_variants = build_permutation_rows(samples, mode=args.permutation_mode)

    solver = diag.SharedSolver(
        solver_model=args.solver_model,
        solver_tokenizer=args.solver_tokenizer,
        base_model=args.base_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    loaded_solver = Path(solver.model_path).resolve()
    if not args.allow_nonaccepted_solver and loaded_solver != accepted_solver:
        raise RuntimeError(
            "Resolved vLLM model differs from the accepted base-restart Stage2 solver: "
            f"loaded={loaded_solver} accepted={accepted_solver}"
        )
    eval_rows = solver.evaluate(
        permutation_rows,
        k_eval=args.k_eval,
        debug_dir=report_dir / "stage2_recalibration_prompt_debug",
        strict_health=True,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    decoding_health = diag.assert_solver_eval_health(permutation_rows, k_eval=args.k_eval)
    permutation_summary = aggregate_permutation_evaluations(
        samples,
        grouped_variants,
        k_eval=args.k_eval,
        min_frontier_permutation_share=args.min_frontier_permutation_share,
    )
    write_jsonl(output_dir / "stage2_recalibration_permutation_eval_rows.jsonl", eval_rows)
    write_json(report_dir / "solver_decoding_health.json", decoding_health)

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
    position_sensitive = [
        s for s in samples if (s.get("stage64_permutation_eval") or {}).get("status") == "position_sensitive"
    ]
    frontier_train, frontier_heldout, split_summary = split_frontier_by_group(
        frontier,
        seed=args.seed,
        heldout_ratio=args.heldout_ratio,
        group_level=args.group_level,
    )
    split_dir = output_dir / f"splits/frontier_{args.group_level}_holdout_seed{args.seed}"

    write_jsonl(output_dir / "clean_samples.jsonl", samples)
    write_jsonl(output_dir / "samples.jsonl", samples)
    write_jsonl(output_dir / "valid_samples.jsonl", [s for s in samples if generation.base.base_valid(s)])
    if args.write_all_clean_sft:
        write_sft(output_dir / "clean_samples_sft.jsonl", samples)

    write_jsonl(split_dir / "frontier_train.jsonl", frontier_train)
    write_jsonl(split_dir / "frontier_heldout.jsonl", frontier_heldout)
    write_jsonl(split_dir / "diagnostic_hard0_samples.jsonl", hard0)
    write_jsonl(split_dir / "diagnostic_easy5_samples.jsonl", easy5)
    write_jsonl(split_dir / "diagnostic_position_sensitive_samples.jsonl", position_sensitive)
    write_sft(split_dir / "frontier_train_sft.jsonl", frontier_train)
    write_sft(split_dir / "frontier_heldout_sft.jsonl", frontier_heldout)
    write_sft(split_dir / "diagnostic_hard0_sft.jsonl", hard0)
    write_sft(split_dir / "diagnostic_easy5_sft.jsonl", easy5)
    write_sft(split_dir / "diagnostic_position_sensitive_sft.jsonl", position_sensitive)

    summary = {
        "input": str(args.input),
        "input_count": len(input_samples),
        "output_dir": str(output_dir),
        "fixed_solver": str(args.solver_model),
        "fixed_solver_resolved": str(requested_solver),
        "loaded_solver_model_resolved": str(loaded_solver),
        "loaded_solver_adapter": str(solver.adapter_path) if solver.adapter_path else None,
        "accepted_stage2_solver_resolved": str(accepted_solver),
        "base_model": str(args.base_model),
        "k_eval": args.k_eval,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "solver_eval_seed": args.seed,
        "generation_seed": args.generation_seed,
        "heldout_ratio": args.heldout_ratio,
        "group_level": args.group_level,
        "permutation_mode": args.permutation_mode,
        "min_frontier_permutation_share": args.min_frontier_permutation_share,
        "quality": quality_summary,
        "permutation_summary": permutation_summary,
        "decoding_health": decoding_health,
        "position_sensitive_count": len(position_sensitive),
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
