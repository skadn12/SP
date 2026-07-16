#!/usr/bin/env python3
"""Accumulate audited Stage64 frontiers and publish balanced Stage3 inputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import api_challenger_generation as generation  # noqa: E402
from scripts.current_method import recalibrate_stage64_shuffled_frontier as recal  # noqa: E402
from scripts.current_method import stage64_quality  # noqa: E402


TRAINING_READY_MANIFEST = "training_ready_manifest.json"
SOLVER_EVAL_CONFIG_FIELDS = (
    "k_eval",
    "max_model_len",
    "max_new_tokens",
    "temperature",
    "top_p",
    "seed",
)


def parent_sample_id(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    seed = sample.get("seed") if isinstance(sample.get("seed"), dict) else {}
    source = seed.get("source") if isinstance(seed.get("source"), dict) else {}
    seed_source = sample.get("seed_source") if isinstance(sample.get("seed_source"), dict) else {}
    for value in (
        sample.get("lineage_group"),
        metadata.get("lineage_group"),
        seed_source.get("lineage_group"),
        source.get("lineage_group"),
        seed_source.get("source_lineage_key"),
        source.get("source_lineage_key"),
        seed_source.get("parent_sample_id"),
        source.get("parent_sample_id"),
        seed.get("parent_sample_id"),
    ):
        if value:
            return str(value)
    return ""


def content_fingerprint(sample: dict[str, Any]) -> str:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    payload = {
        "parent_sample_id": parent_sample_id(sample),
        "question": stage64_quality.normalized_option_text(qa.get("question")),
        "options": sorted(
            stage64_quality.normalized_option_text(option.get("text"))
            for option in (qa.get("options") or [])
            if isinstance(option, dict)
        ),
        "gold_text": sorted(
            stage64_quality.normalized_option_text(option.get("text"))
            for option in (qa.get("options") or [])
            if isinstance(option, dict) and str(option.get("id")) in {str(item) for item in (qa.get("gold") or [])}
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fixed_solver_eval_config(summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Read fixed-solver sampling parameters without treating generation seed as eval seed.

    Current recalibration summaries keep the solver seed in top-level ``seed``.
    Newer producers may use an explicit solver-eval container or seed field.  The
    explicit forms win; ``seed`` is retained only as the legacy fallback.
    """

    nested: dict[str, Any] = {}
    nested_name = ""
    for key in ("solver_eval_config", "solver_eval_sampling"):
        value = summary.get(key)
        if isinstance(value, dict):
            nested = value
            nested_name = key
            break

    config: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for field in SOLVER_EVAL_CONFIG_FIELDS[:-1]:
        if field in nested:
            config[field] = nested.get(field)
            sources[field] = nested_name
        else:
            config[field] = summary.get(field)
            sources[field] = "top_level"

    for key in ("solver_eval_seed", "eval_seed", "seed"):
        if key in nested:
            config["seed"] = nested.get(key)
            sources["seed"] = f"{nested_name}.{key}"
            break
    else:
        for key in ("solver_eval_seed", "eval_seed", "seed"):
            if key in summary:
                config["seed"] = summary.get(key)
                sources["seed"] = f"top_level.{key}"
                break
        else:
            config["seed"] = None
            sources["seed"] = "missing"
    return config, sources


def option_arity(sample: dict[str, Any]) -> int:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    return len(options)


def _audit_gold(permutation: dict[str, Any]) -> str:
    gold = permutation.get("gold")
    if isinstance(gold, list):
        return str(gold[0]) if gold else ""
    return str(gold or "")


def _selected_solver_eval(
    permutation: dict[str, Any],
    *,
    k_eval: int,
    shift: int,
) -> dict[str, Any]:
    count = int(permutation["exact_correct_count"])
    diagnostics = {
        "selected_from_stage64_permutation_audit": True,
        "selected_cyclic_shift": shift,
        "no_solver_rerun_during_balance": True,
    }
    return {
        "k": k_eval,
        "predictions": copy.deepcopy(permutation.get("predictions") or []),
        "exact_correct_count": count,
        "soft_accuracy": permutation.get("soft_accuracy", round(count / k_eval, 6)),
        "partial_credit": permutation.get("partial_credit", round(count / k_eval, 6)),
        "bucket": str(permutation.get("bucket") or f"{count}/{k_eval}"),
        "diagnostics": diagnostics,
    }


def materialize_audited_permutation(
    sample: dict[str, Any],
    *,
    target_gold: str,
    partition: str,
    balance_seed: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Select one already-evaluated cyclic position; never invent or re-evaluate one."""

    arity = option_arity(sample)
    labels = list(recal.LETTERS[:arity])
    permutation_eval = (
        sample.get("stage64_permutation_eval")
        if isinstance(sample.get("stage64_permutation_eval"), dict)
        else {}
    )
    permutations = permutation_eval.get("permutations")
    permutations = permutations if isinstance(permutations, list) else []
    base_rejection = {
        "partition": partition,
        "target_gold": target_gold,
        "option_arity": arity,
        "available_gold": sorted({_audit_gold(item) for item in permutations if isinstance(item, dict)}),
    }
    if arity < 2 or target_gold not in labels:
        return None, {**base_rejection, "reason": "invalid_option_arity_or_target"}

    candidates = [
        item
        for item in permutations
        if isinstance(item, dict) and _audit_gold(item) == target_gold
    ]
    if len(candidates) != 1:
        return None, {
            **base_rejection,
            "reason": "missing_target_permutation" if not candidates else "duplicate_target_permutation",
            "matching_permutation_count": len(candidates),
        }
    selected = candidates[0]
    shift = selected.get("shift")
    if isinstance(shift, bool) or not isinstance(shift, int) or not 0 <= shift < arity:
        return None, {**base_rejection, "reason": "invalid_target_permutation_shift", "shift": shift}

    audit_shifts = [
        item.get("shift")
        for item in permutations
        if isinstance(item, dict)
        and isinstance(item.get("shift"), int)
        and not isinstance(item.get("shift"), bool)
    ]
    audit_golds = [_audit_gold(item) for item in permutations if isinstance(item, dict)]
    declared_permutation_count = permutation_eval.get("permutation_count", len(permutations))
    if (
        len(permutations) != arity
        or not all(isinstance(item, dict) for item in permutations)
        or declared_permutation_count != arity
        or sorted(audit_shifts) != list(range(arity))
        or sorted(audit_golds) != labels
    ):
        return None, {
            **base_rejection,
            "reason": "incomplete_cyclic_permutation_audit",
            "audited_shifts": sorted(audit_shifts),
            "declared_permutation_count": declared_permutation_count,
            "actual_permutation_count": len(permutations),
        }

    k_eval = permutation_eval.get("k_eval_per_permutation", 5)
    if isinstance(k_eval, bool) or not isinstance(k_eval, int) or k_eval < 2:
        return None, {**base_rejection, "reason": "invalid_permutation_audit_k", "k_eval": k_eval}
    for audit in permutations:
        audit_count = audit.get("exact_correct_count")
        if isinstance(audit_count, bool) or not isinstance(audit_count, int) or not 0 < audit_count < k_eval:
            return None, {
                **base_rejection,
                "reason": "permutation_audit_contains_nonfrontier_position",
                "shift": audit.get("shift"),
                "exact_correct_count": audit_count,
                "k_eval": k_eval,
            }
        audit_variant = recal.cyclic_option_permutation(sample, shift=int(audit["shift"]))
        if recal.qa_gold(audit_variant) != _audit_gold(audit):
            return None, {
                **base_rejection,
                "reason": "permutation_audit_mapping_mismatch",
                "shift": audit.get("shift"),
                "audited_gold": _audit_gold(audit),
                "reconstructed_gold": recal.qa_gold(audit_variant),
            }

    count = selected.get("exact_correct_count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 0 < count < k_eval
    ):
        return None, {
            **base_rejection,
            "reason": "target_permutation_not_frontier",
            "exact_correct_count": count,
            "k_eval": k_eval,
        }

    row = recal.cyclic_option_permutation(sample, shift=shift)
    reconstructed_gold = recal.qa_gold(row)
    if reconstructed_gold != target_gold:
        return None, {
            **base_rejection,
            "reason": "target_permutation_reconstruction_mismatch",
            "shift": shift,
            "reconstructed_gold": reconstructed_gold,
        }

    base_sample_id = str(sample.get("sample_id") or "")
    row["sample_id"] = base_sample_id
    row["solver_eval"] = _selected_solver_eval(selected, k_eval=k_eval, shift=shift)
    row["stage3_option_balance"] = {
        "schema_version": "stage3_post_frontier_option_balance_v1",
        "partition": partition,
        "seed": int(balance_seed),
        "option_arity": arity,
        "target_gold": target_gold,
        "selected_cyclic_shift": shift,
        "selected_permutation_eval": copy.deepcopy(selected),
        "source": "stage64_permutation_eval.permutations",
        "api_used": False,
        "solver_rerun": False,
    }
    return row, None


def balance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arity: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arity[option_arity(row)].append(row)
    arity_summary: dict[str, Any] = {}
    all_balanced = True
    for arity, grouped in sorted(by_arity.items()):
        labels = list(recal.LETTERS[:arity])
        counts = Counter(recal.qa_gold(row) for row in grouped)
        expected_counts = {label: counts.get(label, 0) for label in labels}
        unexpected = {label: count for label, count in counts.items() if label not in labels}
        spread = max(expected_counts.values(), default=0) - min(expected_counts.values(), default=0)
        balanced = not unexpected and spread <= 1
        all_balanced = all_balanced and balanced
        arity_summary[str(arity)] = {
            "count": len(grouped),
            "gold_distribution": expected_counts,
            "unexpected_gold_distribution": unexpected,
            "max_min_difference": spread,
            "balanced": balanced,
        }
    return {
        "count": len(rows),
        "gold_distribution": dict(Counter(recal.qa_gold(row) for row in rows).most_common()),
        "by_option_arity": arity_summary,
        "max_min_difference_limit": 1,
        "balanced": all_balanced,
    }


def materialize_balanced_partition(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    partition: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_arity: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arity[option_arity(row)].append(row)

    balanced: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for arity, grouped in sorted(by_arity.items()):
        labels = list(recal.LETTERS[:arity])
        ordered = sorted(
            grouped,
            key=lambda row: (
                recal.stable_int([seed, partition, arity, str(row.get("sample_id") or "")]),
                str(row.get("sample_id") or ""),
            ),
        )
        for index, row in enumerate(ordered):
            target = labels[index % arity] if labels else ""
            materialized, rejection = materialize_audited_permutation(
                row,
                target_gold=target,
                partition=partition,
                balance_seed=seed,
            )
            if materialized is not None:
                balanced.append(materialized)
                continue
            rejected_row = copy.deepcopy(row)
            rejected_row["stage3_option_balance_rejection"] = rejection
            rejected.append(rejected_row)

    balanced.sort(key=lambda row: str(row.get("sample_id") or ""))
    rejected.sort(key=lambda row: str(row.get("sample_id") or ""))
    summary = balance_summary(balanced)
    summary.update(
        {
            "partition": partition,
            "seed": seed,
            "input_count": len(rows),
            "rejected_count": len(rejected),
            "all_rows_materialized": len(balanced) == len(rows) and not rejected,
        }
    )
    return balanced, rejected, summary


def validate_batch(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = directory / "recalibration_summary.json"
    frontier_path = directory / "frontier.jsonl"
    if not summary_path.exists() or not frontier_path.exists():
        raise RuntimeError(f"Missing recalibration_summary.json or frontier.jsonl in {directory}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    solver_eval_config, _ = fixed_solver_eval_config(summary)
    required = {
        "k_eval_is_5": summary.get("k_eval") == 5,
        "cyclic_permutations": summary.get("permutation_mode") == "cyclic",
        "all_positions_required": float(summary.get("min_frontier_permutation_share") or 0.0) == 1.0,
        "semantic_gate": (summary.get("quality") or {}).get("semantic_quality_gate_enabled") is True,
        "decoding_health": (summary.get("decoding_health") or {}).get("healthy") is True,
        "accepted_solver": bool(summary.get("loaded_solver_model_resolved"))
        and summary.get("loaded_solver_model_resolved") == summary.get("accepted_stage2_solver_resolved"),
        "solver_eval_config_recorded": all(
            solver_eval_config.get(key) is not None for key in SOLVER_EVAL_CONFIG_FIELDS
        ),
    }
    if not all(required.values()):
        raise RuntimeError(f"Untrusted Stage64 batch {directory}: {required}")
    rows = recal.read_jsonl(frontier_path)
    if len(rows) != int(summary.get("frontier_count") or 0):
        raise RuntimeError(
            f"Frontier row count differs from summary in {directory}: {len(rows)} != {summary.get('frontier_count')}"
        )
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        status = (row.get("stage64_permutation_eval") or {}).get("status")
        permutations = (row.get("stage64_permutation_eval") or {}).get("permutations") or []
        counts = [item.get("exact_correct_count") for item in permutations if isinstance(item, dict)]
        checks = {
            "sample_id": bool(sample_id),
            "parent_sample_id": bool(parent_sample_id(row)),
            "semantic_quality": not stage64_quality.semantic_quality_issues(row),
            "robust_status": status == "robust_frontier",
            "all_permutations_frontier": bool(counts)
            and all(not isinstance(value, bool) and isinstance(value, int) and 0 < value < 5 for value in counts),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Invalid robust frontier row {sample_id!r} in {directory}: {checks}")
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Accumulate trusted Stage64 batches and gate balanced Stage3 training artifacts."
    )
    parser.add_argument("--recalibration-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-frontier", type=int, default=300)
    parser.add_argument("--heldout-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_frontier < 1:
        raise ValueError("--min-frontier must be positive")
    if not 0.0 < args.heldout_ratio < 1.0:
        raise ValueError("--heldout-ratio must be in (0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / TRAINING_READY_MANIFEST
    # A prior successful invocation must never leave a stale ready signal when
    # the current inputs fail validation or fall below the threshold.
    manifest_path.unlink(missing_ok=True)

    summaries: list[dict[str, Any]] = []
    unique_rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    fingerprints: set[str] = set()
    duplicate_content: list[dict[str, str]] = []
    accepted_solver: str | None = None
    solver_eval_config: dict[str, Any] | None = None
    solver_eval_config_sources: list[dict[str, str]] = []
    generation_seeds: list[Any] = []
    for directory in args.recalibration_dir:
        summary, rows = validate_batch(directory)
        solver = str(summary.get("accepted_stage2_solver_resolved") or "")
        if accepted_solver is None:
            accepted_solver = solver
        elif solver != accepted_solver:
            raise RuntimeError(f"Batch solver mismatch: {solver} != {accepted_solver}")
        current_eval_config, current_sources = fixed_solver_eval_config(summary)
        if solver_eval_config is None:
            solver_eval_config = current_eval_config
        elif current_eval_config != solver_eval_config:
            raise RuntimeError(
                "Batch fixed solver-eval configuration mismatch: "
                f"{current_eval_config} != {solver_eval_config}"
            )
        solver_eval_config_sources.append(current_sources)
        generation_seeds.append(summary.get("generation_seed"))
        summaries.append(summary)
        for row in rows:
            sample_id = str(row.get("sample_id"))
            if sample_id in sample_ids:
                raise RuntimeError(f"Duplicate sample_id across batches: {sample_id}")
            sample_ids.add(sample_id)
            fingerprint = content_fingerprint(row)
            if fingerprint in fingerprints:
                duplicate_content.append({"sample_id": sample_id, "fingerprint": fingerprint})
                continue
            fingerprints.add(fingerprint)
            unique_rows.append(row)

    recal.write_jsonl(args.output_dir / "frontier_pool.jsonl", unique_rows)
    threshold_reached = len(unique_rows) >= args.min_frontier
    canonical_train: list[dict[str, Any]] = []
    canonical_heldout: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    train_rejected: list[dict[str, Any]] = []
    heldout_rejected: list[dict[str, Any]] = []
    train_balance: dict[str, Any] | None = None
    heldout_balance: dict[str, Any] | None = None
    split_summary: dict[str, Any] | None = None
    split_dir: Path | None = None
    group_overlap: list[str] = []
    readiness_gates = {
        "minimum_reliable_frontier_reached": threshold_reached,
        "all_train_rows_materialized_from_audit": False,
        "all_heldout_rows_materialized_from_audit": False,
        "train_gold_balanced_within_each_arity": False,
        "heldout_gold_balanced_within_each_arity": False,
        "train_heldout_parent_group_overlap_zero": False,
        "train_heldout_lineage_group_overlap_zero": False,
        "train_partition_nonempty": False,
        "heldout_partition_nonempty": False,
    }
    if threshold_reached:
        canonical_train, canonical_heldout, split_summary = recal.split_frontier_by_group(
            unique_rows,
            seed=args.seed,
            heldout_ratio=args.heldout_ratio,
            group_level="lineage_group",
        )
        train_groups = {parent_sample_id(row) for row in canonical_train}
        heldout_groups = {parent_sample_id(row) for row in canonical_heldout}
        group_overlap = sorted(train_groups & heldout_groups)
        if group_overlap:
            raise RuntimeError("Parent group leakage detected after split")

        train, train_rejected, train_balance = materialize_balanced_partition(
            canonical_train,
            seed=args.seed,
            partition="train",
        )
        heldout, heldout_rejected, heldout_balance = materialize_balanced_partition(
            canonical_heldout,
            seed=args.seed,
            partition="heldout",
        )
        balanced_train_groups = {parent_sample_id(row) for row in train}
        balanced_heldout_groups = {parent_sample_id(row) for row in heldout}
        balanced_group_overlap = sorted(balanced_train_groups & balanced_heldout_groups)
        group_overlap = sorted(set(group_overlap) | set(balanced_group_overlap))

        readiness_gates.update(
            {
                "all_train_rows_materialized_from_audit": bool(train_balance["all_rows_materialized"]),
                "all_heldout_rows_materialized_from_audit": bool(heldout_balance["all_rows_materialized"]),
                "train_gold_balanced_within_each_arity": bool(train_balance["balanced"]),
                "heldout_gold_balanced_within_each_arity": bool(heldout_balance["balanced"]),
                "train_heldout_parent_group_overlap_zero": not group_overlap,
                "train_heldout_lineage_group_overlap_zero": not group_overlap,
                "train_partition_nonempty": bool(train),
                "heldout_partition_nonempty": bool(heldout),
            }
        )

        split_dir = args.output_dir / f"splits/frontier_lineage_group_holdout_seed{args.seed}"
        recal.write_jsonl(split_dir / "frontier_train_canonical.jsonl", canonical_train)
        recal.write_jsonl(split_dir / "frontier_heldout_canonical.jsonl", canonical_heldout)
        recal.write_jsonl(split_dir / "frontier_train.jsonl", train)
        recal.write_jsonl(split_dir / "frontier_heldout.jsonl", heldout)
        recal.write_jsonl(split_dir / "frontier_train_balance_rejected.jsonl", train_rejected)
        recal.write_jsonl(split_dir / "frontier_heldout_balance_rejected.jsonl", heldout_rejected)
        recal.write_jsonl(split_dir / "frontier_train_sft.jsonl", generation.build_sft_records(train))
        recal.write_jsonl(split_dir / "frontier_heldout_sft.jsonl", generation.build_sft_records(heldout))
        split_summary = {
            **split_summary,
            "canonical_frontier_train": len(canonical_train),
            "canonical_frontier_heldout": len(canonical_heldout),
            "balanced_frontier_train": len(train),
            "balanced_frontier_heldout": len(heldout),
            "train_option_balance": train_balance,
            "heldout_option_balance": heldout_balance,
            "parent_group_overlap": group_overlap,
            "readiness_gates": readiness_gates,
        }
        recal.write_json(split_dir / "split_summary.json", split_summary)

    training_ready = all(readiness_gates.values())
    if training_ready:
        assert split_dir is not None
        manifest = {
            "schema_version": "stage3_training_ready_manifest_v1",
            "status": "training_ready",
            "minimum_reliable_frontier": args.min_frontier,
            "reliable_frontier_count": len(unique_rows),
            "accepted_stage2_solver_resolved": accepted_solver,
            "fixed_solver_eval_config": solver_eval_config,
            "group_level": "lineage_group",
            "split_seed": args.seed,
            "readiness_gates": readiness_gates,
            "artifacts": {
                "canonical_frontier_pool": str(args.output_dir / "frontier_pool.jsonl"),
                "canonical_frontier_train": str(split_dir / "frontier_train_canonical.jsonl"),
                "canonical_frontier_heldout": str(split_dir / "frontier_heldout_canonical.jsonl"),
                "balanced_frontier_train": str(split_dir / "frontier_train.jsonl"),
                "balanced_frontier_heldout": str(split_dir / "frontier_heldout.jsonl"),
                "balanced_frontier_train_sft": str(split_dir / "frontier_train_sft.jsonl"),
                "balanced_frontier_heldout_sft": str(split_dir / "frontier_heldout_sft.jsonl"),
                "split_summary": str(split_dir / "split_summary.json"),
            },
        }
        recal.write_json(manifest_path, manifest)

    summary = {
        "method": "stage64_reliable_frontier_accumulation",
        "batch_dirs": [str(path) for path in args.recalibration_dir],
        "batch_count": len(summaries),
        "batch_frontier_total": sum(int(item.get("frontier_count") or 0) for item in summaries),
        "unique_frontier_count": len(unique_rows),
        "duplicate_content_count": len(duplicate_content),
        "duplicate_content": duplicate_content,
        "accepted_stage2_solver_resolved": accepted_solver,
        "fixed_solver_eval_config": solver_eval_config,
        "sampling_config": solver_eval_config,
        "sampling_config_role": "fixed_solver_eval_compatibility_alias",
        "solver_eval_config_sources": solver_eval_config_sources,
        "batch_generation_seeds": generation_seeds,
        "generation_seed_compatibility_required": False,
        "minimum_frontier": args.min_frontier,
        "threshold_reached": threshold_reached,
        "split_created": threshold_reached,
        "training_ready": training_ready,
        "training_ready_manifest": str(manifest_path) if training_ready else None,
        "readiness_gates": readiness_gates,
        "canonical_train_count": len(canonical_train),
        "canonical_heldout_count": len(canonical_heldout),
        "train_count": len(train),
        "heldout_count": len(heldout),
        "train_balance_rejected_count": len(train_rejected),
        "heldout_balance_rejected_count": len(heldout_rejected),
        "train_option_balance": train_balance,
        "heldout_option_balance": heldout_balance,
        "parent_group_overlap": group_overlap,
        "split_summary": split_summary,
        "gold_distribution": dict(Counter(recal.qa_gold(row) for row in unique_rows).most_common()),
    }
    recal.write_json(args.output_dir / "accumulation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
