from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from current_method import api_challenger_generation as generation
from current_method import frontier_selection


base = generation.base


def round_label(args: argparse.Namespace) -> str:
    if isinstance(args.round_number, int) and args.round_number > 0:
        return f"r{args.round_number}"
    return str(args.round_id)


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target = Path(path)
    if not target.exists():
        return rows
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def count_map(values: list[Any]) -> dict[str, int]:
    return dict(Counter(str(value if value is not None else "unknown") for value in values).most_common())


def qtype_of(sample: dict[str, Any]) -> str:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    return str(qa.get("question_type") or sample.get("question_type") or "unknown")


def difficulty_subtype_of(sample: dict[str, Any]) -> str:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    if qtype_of(sample) == "single_choice":
        return str(sample.get("difficulty_subtype") or qa.get("difficulty_subtype") or "ordinary_single_choice")
    return str(sample.get("difficulty_subtype") or qa.get("difficulty_subtype") or "ordinary")


def hard_single_choice_template_of(sample: dict[str, Any]) -> str:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    return str(sample.get("hard_single_choice_template") or qa.get("hard_single_choice_template") or "unknown")


def frontier_category_of(sample: dict[str, Any]) -> str:
    label = sample.get("frontier_label") if isinstance(sample.get("frontier_label"), dict) else {}
    return str(label.get("category") or "unknown")


def boundary_frontier_row(sample: dict[str, Any]) -> bool:
    se = sample.get("solver_eval") if isinstance(sample.get("solver_eval"), dict) else {}
    bucket = str(se.get("bucket") or "")
    return bucket in {"1/5", "2/5", "3/5", "4/5"}


def exact_frontier_row(sample: dict[str, Any]) -> bool:
    # Deprecated name retained for compatibility; this is a boundary bucket test.
    return boundary_frontier_row(sample)


def strict_frontier_row(sample: dict[str, Any]) -> bool:
    label = sample.get("frontier_label") if isinstance(sample.get("frontier_label"), dict) else {}
    return bool(label.get("is_frontier"))


def near_miss_frontier_row(sample: dict[str, Any]) -> bool:
    se = sample.get("solver_eval") if isinstance(sample.get("solver_eval"), dict) else {}
    exact = bool(se.get("exact_correct")) or int(se.get("exact_correct_count") or 0) > 0
    partial = float(se.get("partial_credit") or 0.0)
    return not exact and partial >= 0.70


def build_validation_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    valid = sum(1 for sample in samples if base.base_valid(sample))
    clean = sum(1 for sample in samples if base.is_clean(sample))
    return {
        "method": "current_method",
        "total": total,
        "valid": valid,
        "clean": clean,
        "validation_pass_rate": round(valid / total, 6) if total else 0.0,
        "clean_pass_rate": round(clean / total, 6) if total else 0.0,
        "option_validation_bad": sum(1 for sample in samples if sample.get("validation", {}).get("option_valid") is not True),
        "gold_not_derivable": sum(1 for sample in samples if sample.get("validation", {}).get("gold_derivable") is not True),
        "graph_dialogue_alignment_bad": sum(
            1 for sample in samples if sample.get("validation", {}).get("graph_dialogue_consistent") is not True
        ),
        "shortcut_issue_count": sum(1 for sample in samples if sample.get("validation", {}).get("anti_shortcut_pass") is not True),
        "warning_count": sum(1 for sample in samples if sample.get("validation", {}).get("warnings")),
        "all_checks_passed": total > 0 and valid == total,
    }


def split_single_choice_rows(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hard_rows = [sample for sample in samples if qtype_of(sample) == "single_choice" and difficulty_subtype_of(sample) == "hard_single_choice"]
    ordinary_rows = [sample for sample in samples if qtype_of(sample) == "single_choice" and difficulty_subtype_of(sample) == "ordinary_single_choice"]
    return hard_rows, ordinary_rows


def _eval_args(args: argparse.Namespace, k_eval: int, prompt_debug_dir: str | Path | None = None) -> argparse.Namespace:
    # k=1 (true accuracy) stays deterministic/low-temp; k=5 (bucket diversity for the learnability
    # signal) uses a higher temperature so a confident solver doesn't return the identical answer 5
    # times. Single_choice was fully degenerate at temp 0.7 (5/5 identical preds -> only 0/5 or 5/5
    # buckets, never a middle band). k5_eval_temperature lifts the per-sample bucket out of bimodal.
    if k_eval > 1:
        temperature = float(getattr(args, "k5_eval_temperature", 0.0) or 0.0) or float(getattr(args, "temperature", 0.7))
    else:
        temperature = float(getattr(args, "temperature", 0.7))
    return argparse.Namespace(
        k_eval=k_eval,
        seed=int(args.seed),
        eval_backend=getattr(args, "eval_backend", "auto"),
        base_model=str(getattr(args, "base_model", None) or "/root/autodl-tmp/Qwen3-4B"),
        max_model_len=int(args.max_model_len),
        max_new_tokens=int(args.max_new_tokens),
        temperature=temperature,
        top_p=float(getattr(args, "top_p", 0.9)),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        prompt_debug_dir=str(prompt_debug_dir) if prompt_debug_dir else None,
    )


def _solver_model_path(args: argparse.Namespace) -> tuple[str, str | None]:
    if getattr(args, "solver_model", None):
        return str(args.solver_model), str(args.solver_adapter) if getattr(args, "solver_adapter", None) else None
    return base.resolve_solver_checkpoint(args.solver_checkpoint)


def grouped_eval_summary(eval_rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        groups[str(row.get(key) or "unknown")].append(row)
    output: dict[str, Any] = {}
    for group, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        n = len(rows)
        soft = [float(row.get("solver_eval", {}).get("soft_accuracy") or 0.0) for row in rows]
        exact = [int(row.get("solver_eval", {}).get("exact_correct_count") or 0) for row in rows]
        k = max([int(row.get("solver_eval", {}).get("k") or 1) for row in rows] or [1])
        output[group] = {
            "n": n,
            "mean_soft_accuracy": round(sum(soft) / n, 6) if n else 0.0,
            "mean_exact_count": round(sum(exact) / n, 6) if n else 0.0,
            "k": k,
        }
    return output


def soft_buckets(samples: list[dict[str, Any]], k_eval: int) -> dict[str, int]:
    buckets = {f"{idx}/{k_eval}": 0 for idx in range(k_eval + 1)}
    for sample in samples:
        bucket = str(sample.get("solver_eval", {}).get("bucket") or f"0/{k_eval}")
        buckets[bucket] = buckets.get(bucket, 0) + 1
    return buckets


def solver_diagnostic_summary(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [
        row.get("solver_eval", {}).get("diagnostics")
        for row in eval_rows
        if isinstance(row.get("solver_eval", {}).get("diagnostics"), dict)
    ]
    if not diagnostics:
        return {
            "eval_context_overflow_count": 0,
            "truncated_count": 0,
            "answer_format_instruction_missing_count": 0,
        }
    token_estimates = [int(item.get("prompt_token_estimate") or 0) for item in diagnostics]
    char_lengths = [int(item.get("prompt_char_len") or 0) for item in diagnostics]
    max_new_tokens = [int(item.get("max_new_tokens") or 0) for item in diagnostics]
    return {
        "eval_context_overflow_count": sum(1 for item in diagnostics if item.get("eval_context_overflow") is True),
        "truncated_count": sum(1 for item in diagnostics if item.get("truncated") is True),
        "answer_format_instruction_missing_count": sum(
            1 for item in diagnostics if item.get("answer_format_instruction_present") is not True
        ),
        "prompt_token_estimate_mean": round(sum(token_estimates) / len(token_estimates), 3) if token_estimates else 0.0,
        "prompt_token_estimate_max": max(token_estimates) if token_estimates else 0,
        "prompt_char_len_mean": round(sum(char_lengths) / len(char_lengths), 3) if char_lengths else 0.0,
        "prompt_char_len_max": max(char_lengths) if char_lengths else 0,
        "max_new_tokens_mean": round(sum(max_new_tokens) / len(max_new_tokens), 3) if max_new_tokens else 0.0,
        "max_new_tokens_max": max(max_new_tokens) if max_new_tokens else 0,
    }


def run_solver_evaluations(
    *,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: str | Path,
    report_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    solver_model_path, adapter_path = _solver_model_path(args)
    if getattr(args, "skip_eval", False):
        write_jsonl(output_dir / "solver_k1_eval.jsonl", [])
        write_jsonl(output_dir / "solver_k5_eval.jsonl", [])
        summary = {
            "method": "current_method",
            "skipped": True,
            "fixed_solver": args.fixed_solver_name,
            "model": solver_model_path,
            "adapter_recorded": adapter_path,
        }
        write_json(report_dir / "solver_eval_summary.json", summary)
        return summary

    k1_debug_dir = report_dir
    k5_debug_dir = report_dir / "solver_prompt_debug_k5"
    k1_samples = copy.deepcopy(samples)
    k1_rows, k1_backend = base.run_solver_eval(k1_samples, _eval_args(args, 1, k1_debug_dir), solver_model_path)
    write_jsonl(output_dir / "solver_k1_eval.jsonl", k1_rows)

    if getattr(args, "run_k5", False):
        k5_rows, k5_backend = base.run_solver_eval(samples, _eval_args(args, 5, k5_debug_dir), solver_model_path)
    else:
        k5_rows = []
        k5_backend = "not_run"
        for sample, k1_sample in zip(samples, k1_samples):
            sample["solver_eval"] = k1_sample["solver_eval"]
    write_jsonl(output_dir / "solver_k5_eval.jsonl", k5_rows)

    first_correct = sum(1 for sample in k1_samples if sample.get("solver_eval", {}).get("exact_correct_count") == 1)
    k5_denominator = len(samples) or 1
    k5_mean = (
        sum(float(sample.get("solver_eval", {}).get("soft_accuracy") or 0.0) for sample in samples) / k5_denominator
        if getattr(args, "run_k5", False)
        else 0.0
    )
    k1_diagnostics = solver_diagnostic_summary(k1_rows)
    k5_diagnostics = solver_diagnostic_summary(k5_rows)
    summary = {
        "method": "current_method",
        "fixed_solver": args.fixed_solver_name,
        "model": solver_model_path,
        "adapter_recorded": adapter_path,
        "k1_backend": k1_backend,
        "k5_backend": k5_backend,
        "eval_context_overflow_count": k1_diagnostics.get("eval_context_overflow_count", 0)
        + k5_diagnostics.get("eval_context_overflow_count", 0),
        "prompt_debug_files": {
            "k1": sorted(str(path) for path in report_dir.glob("solver_prompt_debug_sample_*.txt")),
            "k5": sorted(str(path) for path in k5_debug_dir.glob("solver_prompt_debug_sample_*.txt")) if k5_debug_dir.exists() else [],
        },
        "solver_exact_accuracy_k1": round(first_correct / len(k1_samples), 6) if k1_samples else 0.0,
        "solver_soft_accuracy_k5": round(k5_mean, 6) if samples else 0.0,
        "k1": {
            "n": len(k1_samples),
            "exact_correct": first_correct,
            "accuracy": round(first_correct / len(k1_samples), 6) if k1_samples else 0.0,
            "diagnostics": k1_diagnostics,
            "by_question_type": grouped_eval_summary(k1_rows, "question_type"),
            "by_attack_skill": grouped_eval_summary(k1_rows, "attack_skill"),
        },
        "k5": {
            "n": len(samples) if getattr(args, "run_k5", False) else 0,
            "mean_soft_accuracy": round(k5_mean, 6) if getattr(args, "run_k5", False) and samples else 0.0,
            "soft_buckets": soft_buckets(samples, 5) if getattr(args, "run_k5", False) else {},
            "diagnostics": k5_diagnostics,
            "by_question_type": grouped_eval_summary(k5_rows, "question_type"),
            "by_attack_skill": grouped_eval_summary(k5_rows, "attack_skill"),
        },
    }
    write_json(report_dir / "solver_eval_summary.json", summary)
    return summary


def recommendation(report: dict[str, Any]) -> str:
    generated = int(report.get("generated_count") or 0)
    valid = int(report.get("valid_count") or 0)
    clean = int(report.get("clean_count") or 0)
    valid_rate = valid / generated if generated else 0.0
    clean_rate = clean / generated if generated else 0.0
    missing_refs = int(report.get("missing_refs") or report.get("missing_ref_total") or report.get("rows_with_missing_refs") or 0)
    if (
        valid_rate < 0.90
        or clean_rate < 0.90
        or missing_refs > 0
        or int(report.get("solver_visible_meta_leak_count") or 0) > 0
        or int(report.get("anchor_leakage_count") or 0) > 0
    ):
        return "do_not_expand_hardening_quality_failed"
    if int(report.get("rows_below_4000_delta") or 0) > 0:
        return "do_not_expand_hardening_length_failed"
    soft_buckets = report.get("soft_buckets") if isinstance(report.get("soft_buckets"), dict) else {}
    total_buckets = sum(int(v or 0) for v in soft_buckets.values())
    five_share = (int(soft_buckets.get("5/5") or 0) / total_buckets) if total_buckets else 0.0
    zero_share = (int(soft_buckets.get("0/5") or 0) / total_buckets) if total_buckets else 0.0
    if int(report.get("frontier_count") or 0) == 0 and five_share > 0.80:
        return "increase_difficulty"
    if int(report.get("frontier_count") or 0) == 0 and zero_share > 0.80:
        return "decrease_difficulty_or_fix_generation"
    success = (
        report["clean_count"] >= 800
        and report["frontier_count"] >= 100
        and report["frontier_yield"] >= 0.10
        and report["near_miss_count"] > report["true_too_hard_count"]
        and report["largest_type_share"] <= 0.35
        and report["largest_skill_share"] <= 0.25
        and report["dev_or_heldout_frontier_count"] == 0
        and report["policy_validation_pass"] is True
        and report["policy_fallback_used"] is False
        and report["seed_source_audit_pass"] is True
    )
    if success:
        return "proceed_to_current_method_r3_quality_audit"
    # Small-n stratified-frontier viability: a gate with enough candidate (primary+secondary)
    # samples and clean eval/leakage is a usable curriculum step even if the strict full-scale
    # frontier_count threshold isn't met. Do NOT dismiss it as do_not_train.
    candidate_count = int(report.get("frontier_candidate_count") or 0)
    leakage_clean = (
        report["dev_or_heldout_frontier_count"] == 0
        and report["policy_validation_pass"] is True
        and report["policy_fallback_used"] is False
        and report["seed_source_audit_pass"] is True
    )
    if candidate_count >= 4 and leakage_clean:
        return "viable_gate"
    if 30 <= report["frontier_count"] < 100:
        return "revise_challenger_policy_or_quota_and_rerun_current_method_r3_gate"
    return "do_not_train_current_method_r3_challenger_yield_too_low"


def build_report(
    *,
    args: argparse.Namespace,
    run_root: str | Path,
    samples: list[dict[str, Any]],
    seed_audit_report: dict[str, Any],
    generation_summary: dict[str, Any],
    validation_summary: dict[str, Any],
    solver_summary: dict[str, Any],
    frontier_summary: dict[str, Any],
    failure_matrix: dict[str, Any],
    compiled_policy: dict[str, Any],
    policy_meta: dict[str, Any],
) -> dict[str, Any]:
    frontier_by_question_type = frontier_summary.get("frontier_by_question_type", {})
    frontier_by_attack_skill = frontier_summary.get("frontier_by_attack_skill", {})
    hard_single_choice_rows, ordinary_single_choice_rows = split_single_choice_rows(samples)
    hard_single_choice_templates = [hard_single_choice_template_of(sample) for sample in hard_single_choice_rows]
    hard_template_counts = count_map(hard_single_choice_templates)
    strict_count = int(frontier_summary.get("frontier_count") or 0)
    boundary_count = sum(1 for sample in samples if boundary_frontier_row(sample))
    frontier_pool_count = len(
        {
            str(sample.get("sample_id") or sample.get("expanded_graph_id") or idx)
            for idx, sample in enumerate(samples)
            if boundary_frontier_row(sample) or near_miss_frontier_row(sample)
        }
    )
    qtype_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        qtype_groups[qtype_of(sample)].append(sample)
    qtype_zero_share = {
        qtype: round(
            sum(1 for row in rows if str(row.get("solver_eval", {}).get("bucket") or "") == "0/5") / len(rows),
            6,
        )
        for qtype, rows in sorted(qtype_groups.items())
        if rows
    }
    report = {
        "method": "current_method",
        "experiment": f"{args.run_name}_gate1000",
        "run_name": args.run_name,
        "run_root": str(Path(run_root).resolve()),
        "round": round_label(args),
        "parent_round": args.parent_round_id,
        "fixed_solver": args.fixed_solver_name,
        "seed_scope": "sample0_5_train_only",
        "generated_count": len(samples),
        "valid_count": validation_summary.get("valid", 0),
        "clean_count": validation_summary.get("clean", 0),
        "solver_exact_accuracy_k1": solver_summary.get("solver_exact_accuracy_k1", 0.0),
        "solver_soft_accuracy_k5": solver_summary.get("solver_soft_accuracy_k5", 0.0),
        "soft_buckets": solver_summary.get("k5", {}).get("soft_buckets", {}),
        "near_miss_count": frontier_summary.get("near_miss_count", 0),
        "hard_correct_count": frontier_summary.get("hard_correct_count", 0),
        "true_too_hard_count": frontier_summary.get("true_too_hard_count", 0),
        "bad_question_count": frontier_summary.get("bad_question_count", 0),
        "shortcut_issue_count": frontier_summary.get("shortcut_issue_count", 0),
        "frontier_count": frontier_summary.get("frontier_count", 0),
        "frontier_yield": frontier_summary.get("frontier_yield", 0.0),
        "strict_frontier_count": strict_count,
        "strict_frontier_yield": round(strict_count / len(samples), 6) if samples else 0.0,
        "boundary_frontier_count": boundary_count,
        "boundary_frontier_yield": round(boundary_count / len(samples), 6) if samples else 0.0,
        "frontier_pool_count": frontier_pool_count,
        "boundary_plus_near_miss_count": frontier_pool_count,
        # Deprecated alias; use boundary_frontier_count.
        "exact_frontier_count": boundary_count,
        "primary_frontier_count": frontier_summary.get("primary_frontier_count", 0),
        "secondary_frontier_count": frontier_summary.get("secondary_frontier_count", 0),
        "frontier_candidate_count": frontier_summary.get("frontier_candidate_count", 0),
        "weighted_frontier_candidate_score": frontier_summary.get("weighted_frontier_candidate_score", 0.0),
        "frontier_by_bucket": frontier_summary.get("frontier_by_bucket", {}),
        "excluded_0of5_count": frontier_summary.get("excluded_0of5_count", 0),
        "excluded_5of5_count": frontier_summary.get("excluded_5of5_count", 0),
        "frontier_by_question_type": frontier_by_question_type,
        "frontier_by_attack_skill": frontier_by_attack_skill,
        "hard_single_choice_count": len(hard_single_choice_rows),
        "ordinary_single_choice_count": len(ordinary_single_choice_rows),
        "hard_single_choice_bucket_distribution": count_map([str(row.get("solver_eval", {}).get("bucket") or "unknown") for row in hard_single_choice_rows]),
        "ordinary_single_choice_bucket_distribution": count_map([str(row.get("solver_eval", {}).get("bucket") or "unknown") for row in ordinary_single_choice_rows]),
        "hard_single_choice_boundary_count": sum(1 for row in hard_single_choice_rows if boundary_frontier_row(row)),
        "hard_single_choice_strict_frontier_count": sum(1 for row in hard_single_choice_rows if strict_frontier_row(row)),
        "hard_single_choice_near_miss_count": sum(1 for row in hard_single_choice_rows if near_miss_frontier_row(row)),
        "hard_single_choice_partial_credit_mean": round(
            sum(float(row.get("solver_eval", {}).get("partial_credit") or 0.0) for row in hard_single_choice_rows) / len(hard_single_choice_rows),
            6,
        ) if hard_single_choice_rows else 0.0,
        "hard_single_choice_five_of_five_share": round(
            sum(1 for row in hard_single_choice_rows if str(row.get("solver_eval", {}).get("bucket") or "") == "5/5") / len(hard_single_choice_rows),
            6,
        ) if hard_single_choice_rows else 0.0,
        "hard_single_choice_zero_of_five_share": round(
            sum(1 for row in hard_single_choice_rows if str(row.get("solver_eval", {}).get("bucket") or "") == "0/5") / len(hard_single_choice_rows),
            6,
        ) if hard_single_choice_rows else 0.0,
        "ordinary_single_choice_five_of_five_share": round(
            sum(1 for row in ordinary_single_choice_rows if str(row.get("solver_eval", {}).get("bucket") or "") == "5/5") / len(ordinary_single_choice_rows),
            6,
        ) if ordinary_single_choice_rows else 0.0,
        "ordinary_single_choice_zero_of_five_share": round(
            sum(1 for row in ordinary_single_choice_rows if str(row.get("solver_eval", {}).get("bucket") or "") == "0/5") / len(ordinary_single_choice_rows),
            6,
        ) if ordinary_single_choice_rows else 0.0,
        "qtype_zero_share": qtype_zero_share,
        "hard_single_choice_by_template": hard_template_counts,
        "hard_single_choice_by_template_bucket_distribution": {
            template: count_map([str(row.get("solver_eval", {}).get("bucket") or "unknown") for row in hard_single_choice_rows if hard_single_choice_template_of(row) == template])
            for template in sorted(hard_template_counts)
        },
        "hard_single_choice_by_template_frontier_count": {
            template: sum(1 for row in hard_single_choice_rows if hard_single_choice_template_of(row) == template and boundary_frontier_row(row))
            for template in sorted(hard_template_counts)
        },
        "hard_single_choice_by_template_boundary_count": {
            template: sum(1 for row in hard_single_choice_rows if hard_single_choice_template_of(row) == template and boundary_frontier_row(row))
            for template in sorted(hard_template_counts)
        },
        "hard_single_choice_by_template_strict_frontier_count": {
            template: sum(1 for row in hard_single_choice_rows if hard_single_choice_template_of(row) == template and strict_frontier_row(row))
            for template in sorted(hard_template_counts)
        },
        "hard_single_choice_by_template_near_miss_count": {
            template: sum(1 for row in hard_single_choice_rows if hard_single_choice_template_of(row) == template and near_miss_frontier_row(row))
            for template in sorted(hard_template_counts)
        },
        "largest_type_share": frontier_summary.get("largest_type_share", 0.0),
        "largest_skill_share": frontier_summary.get("largest_skill_share", 0.0),
        "dev_or_heldout_frontier_count": frontier_summary.get("dev_or_heldout_frontier_count", 0),
        "api_failure_count": generation_summary.get("api_failure_count", 0),
        "missing_refs": generation_summary.get("missing_refs", validation_summary.get("missing_refs", 0)),
        "rows_with_missing_refs": generation_summary.get("rows_with_missing_refs", 0),
        "missing_ref_total": generation_summary.get("missing_ref_total", 0),
        "anchor_leakage_count": generation_summary.get("anchor_leakage_count", 0),
        "solver_visible_meta_leak_count": generation_summary.get("solver_visible_meta_leak_count", 0),
        "qa_meta_leak_count": generation_summary.get("qa_meta_leak_count", 0),
        "dialogue_meta_leak_count": generation_summary.get("dialogue_meta_leak_count", 0),
        "length_guard_target_delta": generation_summary.get("length_guard_target_delta", generation_summary.get("min_length_delta_tokens", 0)),
        "length_guard_parent_baseline_tokens": generation_summary.get("length_guard_parent_baseline_tokens", 0),
        "length_guard_baseline_mode": generation_summary.get("length_guard_baseline_mode")
        or generation_summary.get("length_guard_baseline_mode_counts", {}),
        "parent_dialogue_tokens_mean": generation_summary.get("parent_dialogue_tokens_mean"),
        "final_dialogue_tokens_mean": generation_summary.get("final_dialogue_tokens_mean"),
        "length_delta_tokens_mean": generation_summary.get("length_delta_tokens_mean"),
        "length_guard_effective_parent_tokens_mean": generation_summary.get("length_guard_effective_parent_tokens_mean"),
        "length_guard_effective_delta_tokens_mean": generation_summary.get("length_guard_effective_delta_tokens_mean"),
        "length_guard_fallback_used": generation_summary.get("length_guard_fallback_used", False),
        "fallback_delta": generation_summary.get("fallback_delta", 0),
        "emergency_floor": generation_summary.get("emergency_floor", 0),
        "rows_below_4000_delta": generation_summary.get("rows_below_4000_delta", 0),
        "challenger_generation_sources": generation_summary.get("challenger_generation_sources", {}),
        "policy_created_by": compiled_policy.get("created_by") or policy_meta.get("policy_created_by"),
        "policy_source": compiled_policy.get("policy_source") or policy_meta.get("policy_source"),
        "policy_validation_pass": bool(compiled_policy.get("policy_validation_pass")),
        "policy_retry_count": int(compiled_policy.get("policy_retry_count") or policy_meta.get("policy_retry_count") or 0),
        "policy_fallback_used": bool(compiled_policy.get("policy_fallback_used")),
        "policy_attack_skill_weights": compiled_policy.get("attack_skill_weights", {}),
        "policy_question_type_weights": compiled_policy.get("question_type_weights", {}),
        "difficulty_plan_stage": compiled_policy.get("difficulty_plan_stage"),
        "difficulty_plan_manual_policy_correction": compiled_policy.get("difficulty_plan_manual_policy_correction"),
        "seed_source_audit_pass": bool(seed_audit_report.get("pass")),
        "failure_matrix": {
            "by_frontier_category": failure_matrix.get("by_frontier_category", {}),
            "validation_failures": failure_matrix.get("validation_failures", {}),
            "shortcut_failures": failure_matrix.get("shortcut_failures", {}),
        },
        "recommendation": "",
    }
    report["recommendation"] = recommendation(report)
    return report


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['run_name']} Report",
        "",
        f"- method: {report['method']}",
        f"- fixed_solver: {report['fixed_solver']}",
        f"- generated_count: {report['generated_count']}",
        f"- valid_count: {report['valid_count']}",
        f"- clean_count: {report['clean_count']}",
        f"- solver_exact_accuracy_k1: {report['solver_exact_accuracy_k1']}",
        f"- solver_soft_accuracy_k5: {report['solver_soft_accuracy_k5']}",
        f"- frontier_count: {report['frontier_count']}",
        f"- frontier_yield: {report['frontier_yield']}",
        f"- strict_frontier_count: {report.get('strict_frontier_count')}",
        f"- boundary_frontier_count: {report.get('boundary_frontier_count')}",
        f"- frontier_pool_count: {report.get('frontier_pool_count')}",
        f"- largest_type_share: {report['largest_type_share']}",
        f"- largest_skill_share: {report['largest_skill_share']}",
        f"- dev_or_heldout_frontier_count: {report['dev_or_heldout_frontier_count']}",
        f"- policy_created_by: {report['policy_created_by']}",
        f"- policy_validation_pass: {report['policy_validation_pass']}",
        f"- policy_fallback_used: {report['policy_fallback_used']}",
        f"- length_guard_target_delta: {report.get('length_guard_target_delta')}",
        f"- length_guard_parent_baseline_tokens: {report.get('length_guard_parent_baseline_tokens')}",
        f"- length_guard_baseline_mode: {report.get('length_guard_baseline_mode')}",
        f"- parent_dialogue_tokens_mean: {report.get('parent_dialogue_tokens_mean')}",
        f"- final_dialogue_tokens_mean: {report.get('final_dialogue_tokens_mean')}",
        f"- length_delta_tokens_mean: {report.get('length_delta_tokens_mean')}",
        f"- length_guard_effective_parent_tokens_mean: {report.get('length_guard_effective_parent_tokens_mean')}",
        f"- length_guard_effective_delta_tokens_mean: {report.get('length_guard_effective_delta_tokens_mean')}",
        f"- length_guard_fallback_used: {report.get('length_guard_fallback_used')}",
        f"- fallback_delta: {report.get('fallback_delta')}",
        f"- emergency_floor: {report.get('emergency_floor')}",
        f"- rows_below_4000_delta: {report.get('rows_below_4000_delta')}",
        f"- anchor_leakage_count: {report.get('anchor_leakage_count')}",
        f"- solver_visible_meta_leak_count: {report.get('solver_visible_meta_leak_count')}",
        f"- recommendation: {report['recommendation']}",
        "",
        "## Soft Buckets",
        "",
    ]
    for key, value in report.get("soft_buckets", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Frontier By Question Type", ""])
    for key, value in report.get("frontier_by_question_type", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Frontier By Attack Skill", ""])
    for key, value in report.get("frontier_by_attack_skill", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Single Choice Split", ""])
    lines.append(f"- hard_single_choice_count: {report.get('hard_single_choice_count')}")
    lines.append(f"- ordinary_single_choice_count: {report.get('ordinary_single_choice_count')}")
    lines.append(f"- hard_single_choice_bucket_distribution: {report.get('hard_single_choice_bucket_distribution')}")
    lines.append(f"- ordinary_single_choice_bucket_distribution: {report.get('ordinary_single_choice_bucket_distribution')}")
    lines.append(f"- hard_single_choice_boundary_count: {report.get('hard_single_choice_boundary_count')}")
    lines.append(f"- hard_single_choice_strict_frontier_count: {report.get('hard_single_choice_strict_frontier_count')}")
    lines.append(f"- hard_single_choice_near_miss_count: {report.get('hard_single_choice_near_miss_count')}")
    lines.append(f"- hard_single_choice_partial_credit_mean: {report.get('hard_single_choice_partial_credit_mean')}")
    lines.append(f"- hard_single_choice_five_of_five_share: {report.get('hard_single_choice_five_of_five_share')}")
    lines.append(f"- hard_single_choice_zero_of_five_share: {report.get('hard_single_choice_zero_of_five_share')}")
    lines.append(f"- ordinary_single_choice_five_of_five_share: {report.get('ordinary_single_choice_five_of_five_share')}")
    lines.append(f"- ordinary_single_choice_zero_of_five_share: {report.get('ordinary_single_choice_zero_of_five_share')}")
    lines.append("")
    return "\n".join(lines)


def write_reports(
    *,
    args: argparse.Namespace,
    run_root: str | Path,
    output_dir: str | Path,
    report_dir: str | Path,
    samples: list[dict[str, Any]],
    seed_audit_report: dict[str, Any],
    generation_summary: dict[str, Any],
    solver_summary: dict[str, Any],
    frontier_summary: dict[str, Any],
    failure_matrix: dict[str, Any],
    compiled_policy: dict[str, Any],
    policy_meta: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    validation_summary = build_validation_summary(samples)
    write_json(output_dir / "validation_summary.json", validation_summary)
    write_json(report_dir / "validation_summary.json", validation_summary)
    write_json(report_dir / "generation_summary.json", generation_summary)
    report = build_report(
        args=args,
        run_root=run_root,
        samples=samples,
        seed_audit_report=seed_audit_report,
        generation_summary=generation_summary,
        validation_summary=validation_summary,
        solver_summary=solver_summary,
        frontier_summary=frontier_summary,
        failure_matrix=failure_matrix,
        compiled_policy=compiled_policy,
        policy_meta=policy_meta,
    )
    write_json(report_dir / "report.json", report)
    (report_dir / "report.md").write_text(build_markdown_report(report), encoding="utf-8")
    return report
