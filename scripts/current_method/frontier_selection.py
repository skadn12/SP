from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from current_method import api_challenger_generation as generation


base = generation.base


FRONTIER_LABELS = (
    "clean_near_miss",
    "hard_correct",
    "true_too_hard",
    "bad_question",
    "shortcut_issue",
    "position_sensitive",
    "easy",
)


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


def question_type(sample: dict[str, Any]) -> str:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    return str(qa.get("question_type") or sample.get("question_type") or "unknown")


def attack_skill(sample: dict[str, Any]) -> str:
    attack_plan = sample.get("attack_plan") if isinstance(sample.get("attack_plan"), dict) else {}
    return str(attack_plan.get("attack_skill") or sample.get("attack_skill") or "unknown")


def difficulty_cell(sample: dict[str, Any]) -> str:
    cell = sample.get("difficulty_cell")
    if isinstance(cell, str) and cell:
        return cell
    return f"{question_type(sample)}|{attack_skill(sample)}"


def clean_validation(sample: dict[str, Any]) -> bool:
    return base.is_clean(sample)


def _exact_bucket_num(sample: dict[str, Any]) -> int | None:
    """k5 exact-correct count (0..5) from solver_eval, or None if unavailable."""
    se = sample.get("solver_eval") if isinstance(sample.get("solver_eval"), dict) else {}
    n = se.get("exact_correct_count")
    if isinstance(n, (int, float)):
        return int(n)
    bucket = str(se.get("bucket") or "")
    if "/" in bucket:
        try:
            return int(bucket.split("/", 1)[0])
        except ValueError:
            return None
    return None


def _is_leakage_clean(sample: dict[str, Any]) -> bool:
    """No meta/anchor leak, gold derivable, options valid, train-only — i.e. safe to train on.

    Independent of difficulty: a weak-frontier (1/5, 4/5) sample may still enter the train
    candidate pool only when it is leakage-clean and from an allowed train seed.
    """
    v = sample.get("validation") if isinstance(sample.get("validation"), dict) else {}
    seed_source = sample.get("seed_source") if isinstance(sample.get("seed_source"), dict) else {}
    split = sample.get("frontier_from_split") or seed_source.get("split") or sample.get("split")
    return (
        seed_source.get("allowed_train_seed") is True
        and split == "train"
        and v.get("option_valid") is True
        and v.get("gold_derivable") is True
        and v.get("graph_dialogue_consistent") is True
        and v.get("anti_shortcut_pass") is True
        and not v.get("warnings")
    )


# Stratified frontier tiers keyed on the k5 EXACT bucket (per user spec for small n=20 gates):
#   primary  : 2/5, 3/5  -> weight 1.0 (true frontier band)
#   secondary: 1/5, 4/5  -> weight 0.5 (weak frontier; only if leakage-clean train candidate)
#   excluded : 0/5 (too_hard_for_current_solver / future_hard_set), 5/5 (mastered / clean_replay)
PRIMARY_BUCKETS = {2, 3}
SECONDARY_BUCKETS = {1, 4}
TIER_WEIGHTS = {"primary": 1.0, "secondary": 0.5}

# Question types excluded from the train frontier regardless of bucket. r4 hardening may unlock
# multi_select and ordering via the qtype curriculum, so they must be eligible for frontier
# accounting when clean. evidence_selection remains metadata-like; state_reconstruction remains
# held back until its exact-text gold is deliberately revalidated.
NON_FRONTIER_QUESTION_TYPES = {"evidence_selection", "state_reconstruction"}


def frontier_tier(sample: dict[str, Any]) -> tuple[str, float]:
    """Return (tier, weight): primary | secondary | excluded_too_hard | excluded_mastered | invalid."""
    # invalid samples never enter the frontier regardless of bucket.
    cat = sample.get("frontier_label", {}).get("category")
    if cat in {"bad_question", "shortcut_issue"}:
        return "invalid", 0.0
    if question_type(sample) in NON_FRONTIER_QUESTION_TYPES:
        return "excluded_non_frontier_qtype", 0.0
    num = _exact_bucket_num(sample)
    if num is None:
        return "invalid", 0.0
    if num in PRIMARY_BUCKETS:
        return "primary", TIER_WEIGHTS["primary"]
    if num in SECONDARY_BUCKETS:
        # weak frontier only counts if it is a clean train candidate (no leakage).
        if _is_leakage_clean(sample):
            return "secondary", TIER_WEIGHTS["secondary"]
        return "excluded_weak_unclean", 0.0
    if num <= 0:
        return "excluded_too_hard", 0.0
    return "excluded_mastered", 0.0  # 5/5


def assign_frontier_labels(samples: list[dict[str, Any]]) -> None:
    for sample in samples:
        validation = sample.get("validation", {})
        solver_eval = sample.get("solver_eval") if isinstance(sample.get("solver_eval"), dict) else {}
        soft = float(solver_eval.get("soft_accuracy") or 0.0)
        partial = float(solver_eval.get("partial_credit") or 0.0)
        seed_source = sample.get("seed_source") if isinstance(sample.get("seed_source"), dict) else {}
        split = sample.get("frontier_from_split") or seed_source.get("split") or sample.get("split")

        if seed_source.get("allowed_train_seed") is not True or split != "train":
            category, reason = "bad_question", "seed source was not train-only sample0-5"
        elif validation.get("option_valid") is not True:
            category, reason = "bad_question", "option validation failed"
        elif validation.get("gold_derivable") is not True:
            category, reason = "bad_question", "gold was not derivable from graph/dialogue"
        elif validation.get("graph_dialogue_consistent") is not True:
            category, reason = "bad_question", "graph/dialogue evidence mismatch"
        elif validation.get("anti_shortcut_pass") is not True:
            category, reason = "shortcut_issue", "anti-shortcut validation failed"
        elif not clean_validation(sample):
            category, reason = "bad_question", "validation warnings present"
        elif (sample.get("stage64_permutation_eval") or {}).get("status") == "position_sensitive":
            category, reason = (
                "position_sensitive",
                "difficulty bucket changed across cyclic answer positions; excluded from reliable frontier",
            )
        elif 0.2 <= soft <= 0.6 and partial >= 0.4:
            category, reason = "clean_near_miss", "soft accuracy in frontier band with sufficient partial credit"
        elif abs(soft - 0.8) < 1e-9 and partial >= 0.4 and base.rare_mechanism(sample, samples):
            category, reason = "hard_correct", "soft accuracy 0.8 with rare reasoning mechanism"
        elif soft == 0.0:
            # Structural/gold/shortcut validation above decides whether a question is bad.
            # Solver failure alone is not evidence of an invalid item; for single-choice
            # tasks partial credit is necessarily zero whenever exact accuracy is zero.
            category, reason = "true_too_hard", "validated sample scored 0/5 with the fixed solver"
        elif soft > 0.6:
            category, reason = "easy", "solver soft accuracy above frontier band"
        else:
            category, reason = "true_too_hard", "below frontier band or insufficient partial credit"
        sample["frontier_label"] = {"is_frontier": False, "category": category, "reason": reason}
    # Stratified tier (primary/secondary/excluded) keyed on the k5 exact bucket. Stamped after
    # the category pass so invalid (bad_question/shortcut) samples are excluded from tiers.
    for sample in samples:
        tier, weight = frontier_tier(sample)
        label = sample.get("frontier_label") if isinstance(sample.get("frontier_label"), dict) else {}
        label["frontier_tier"] = tier
        label["frontier_weight"] = weight
        sample["frontier_label"] = label


def candidate_priority(sample: dict[str, Any]) -> tuple[float, float, float, str]:
    # Prefer primary tier (2/5,3/5) over secondary (1/5,4/5), then closeness to the band center
    # (bucket 2.5), then higher partial credit, then stable id ordering.
    tier = sample.get("frontier_label", {}).get("frontier_tier")
    tier_rank = 0.0 if tier == "primary" else 1.0
    num = _exact_bucket_num(sample)
    bucket_distance = abs((num if num is not None else 0) - 2.5)
    partial = float(sample.get("solver_eval", {}).get("partial_credit") or 0.0)
    return (tier_rank, bucket_distance, -partial, str(sample.get("sample_id") or ""))


def largest_share(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    return round(max(counts.values()) / total, 6) if total else 0.0


def balanced_select_frontier(
    samples: list[dict[str, Any]],
    *,
    max_question_type_share: float,
    max_attack_skill_share: float,
    allowed_splits: set[str],
    min_frontier_floor: int = 4,
) -> list[dict[str, Any]]:
    candidates = [
        sample
        for sample in samples
        if sample.get("frontier_label", {}).get("frontier_tier") in {"primary", "secondary"}
        and (sample.get("frontier_from_split") or sample.get("seed_source", {}).get("split")) in allowed_splits
    ]
    candidates.sort(key=candidate_priority)
    selected = list(candidates)
    changed = True
    while changed and selected:
        changed = False
        # Never balance-prune below the floor.
        if len(selected) <= max(1, min_frontier_floor):
            break
        type_counts = Counter(question_type(sample) for sample in selected)
        skill_counts = Counter(attack_skill(sample) for sample in selected)
        n = len(selected)
        largest_type, type_count = type_counts.most_common(1)[0]
        largest_skill, skill_count = skill_counts.most_common(1)[0]
        remove_group: tuple[str, str] | None = None
        # IMPORTANT: only enforce a dimension's balance cap when it is actually SATISFIABLE, i.e.
        # the dimension has at least ceil(1/cap) distinct categories among the candidates. The
        # curriculum deliberately concentrates the frontier (Phase 1 = single_choice only; just 2
        # attack skills). A 0.35 type cap needs >=3 types; a 0.25 skill cap needs >=4 skills. With
        # fewer categories the cap can NEVER be met, so enforcing it would prune a healthy frontier
        # down to the floor for no reason. We balance only what is genuinely balanceable.
        type_satisfiable = max_question_type_share > 0 and len(type_counts) >= math.ceil(1.0 / max_question_type_share)
        skill_satisfiable = max_attack_skill_share > 0 and len(skill_counts) >= math.ceil(1.0 / max_attack_skill_share)
        if n > 1 and type_satisfiable and type_count / n > max_question_type_share:
            remove_group = ("type", largest_type)
        elif n > 1 and skill_satisfiable and skill_count / n > max_attack_skill_share:
            remove_group = ("skill", largest_skill)
        if remove_group:
            group_kind, group_value = remove_group
            group_items = [
                sample
                for sample in selected
                if (question_type(sample) if group_kind == "type" else attack_skill(sample)) == group_value
            ]
            group_items.sort(key=candidate_priority, reverse=True)
            selected.remove(group_items[0])
            changed = True
    selected_ids = {str(sample.get("sample_id")) for sample in selected}
    for sample in samples:
        label = sample.get("frontier_label") if isinstance(sample.get("frontier_label"), dict) else {}
        if str(sample.get("sample_id")) in selected_ids:
            label["is_frontier"] = True
        elif label.get("frontier_tier") in {"primary", "secondary"}:
            label["reason"] = str(label.get("reason") or "") + "; excluded by balanced frontier cap"
        sample["frontier_label"] = label
    return selected


def build_failure_matrix(samples: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = base.build_failure_matrix(samples)
    matrix["method"] = "current_method"
    matrix["labels"] = list(FRONTIER_LABELS)
    matrix["by_seed_split"] = base.count_map([sample.get("frontier_from_split", "unknown") for sample in samples])
    matrix["by_seed_sample_name"] = base.count_map([sample.get("frontier_from_sample_name", "unknown") for sample in samples])
    matrix["frontier_by_seed_sample_name"] = base.count_map(
        [sample.get("frontier_from_sample_name", "unknown") for sample in samples if sample.get("frontier_label", {}).get("is_frontier")]
    )
    matrix["bad_question_reasons"] = base.count_map(
        [
            sample.get("frontier_label", {}).get("reason", "unknown")
            for sample in samples
            if sample.get("frontier_label", {}).get("category") == "bad_question"
        ]
    )
    matrix["shortcut_warnings"] = base.count_map(
        [
            warning
            for sample in samples
            for warning in (sample.get("validation", {}).get("warnings") or [])
            if "shortcut" in str(warning)
        ]
    )
    # difficulty_cell-level breakdown (foundation for the closed-loop difficulty controller).
    # Each cell aggregates count, soft-bucket distribution, mean soft accuracy, and frontier
    # category mix, so the controller can read per-cell mastery directly from this matrix.
    cell_stats: dict[str, dict[str, Any]] = {}
    for sample in samples:
        cell = difficulty_cell(sample)
        solver_eval = sample.get("solver_eval") if isinstance(sample.get("solver_eval"), dict) else {}
        bucket = str(solver_eval.get("bucket") or "unknown")
        soft = float(solver_eval.get("soft_accuracy") or 0.0)
        category = str(sample.get("frontier_label", {}).get("category") or "unknown")
        entry = cell_stats.setdefault(
            cell,
            {"n": 0, "soft_sum": 0.0, "soft_buckets": Counter(), "by_category": Counter(), "frontier_count": 0},
        )
        entry["n"] += 1
        entry["soft_sum"] += soft
        entry["soft_buckets"][bucket] += 1
        entry["by_category"][category] += 1
        if sample.get("frontier_label", {}).get("is_frontier"):
            entry["frontier_count"] += 1
    matrix["by_difficulty_cell"] = {
        cell: {
            "n": s["n"],
            "mean_soft_accuracy": round(s["soft_sum"] / s["n"], 6) if s["n"] else 0.0,
            "soft_buckets": dict(sorted(s["soft_buckets"].items())),
            "by_frontier_category": dict(sorted(s["by_category"].items())),
            "frontier_count": s["frontier_count"],
        }
        for cell, s in sorted(cell_stats.items())
    }
    matrix["difficulty_cell_counts"] = {cell: s["n"] for cell, s in sorted(cell_stats.items())}
    return matrix


def build_frontier_summary(samples: list[dict[str, Any]], frontier: list[dict[str, Any]]) -> dict[str, Any]:
    frontier_by_question_type = base.count_map([question_type(sample) for sample in frontier])
    frontier_by_attack_skill = base.count_map([attack_skill(sample) for sample in frontier])
    dev_or_heldout_frontier_count = sum(
        1 for sample in frontier if (sample.get("frontier_from_split") or sample.get("seed_source", {}).get("split")) != "train"
    )
    # Stratified frontier accounting (per-tier, bucket-keyed).
    def _tier(s: dict[str, Any]) -> str:
        return str(s.get("frontier_label", {}).get("frontier_tier") or "invalid")
    primary_count = sum(1 for s in samples if _tier(s) == "primary")
    secondary_count = sum(1 for s in samples if _tier(s) == "secondary")
    frontier_candidate_count = primary_count + secondary_count
    frontier_by_bucket = base.count_map([str(s.get("solver_eval", {}).get("bucket") or "unknown") for s in samples])
    excluded_0of5 = sum(1 for s in samples if _exact_bucket_num(s) == 0)
    excluded_5of5 = sum(1 for s in samples if _exact_bucket_num(s) == 5)
    weighted_candidate_score = round(
        sum(float(s.get("frontier_label", {}).get("frontier_weight") or 0.0) for s in samples if _tier(s) in {"primary", "secondary"}),
        4,
    )
    return {
        "method": "current_method",
        "frontier_count": len(frontier),
        "frontier_yield": round(len(frontier) / len(samples), 6) if samples else 0.0,
        # --- stratified frontier (small-n friendly) ---
        "primary_frontier_count": primary_count,
        "secondary_frontier_count": secondary_count,
        "frontier_candidate_count": frontier_candidate_count,
        "weighted_frontier_candidate_score": weighted_candidate_score,
        "frontier_by_bucket": frontier_by_bucket,
        "excluded_0of5_count": excluded_0of5,
        "excluded_5of5_count": excluded_5of5,
        "future_hard_set_count": excluded_0of5,
        "clean_replay_candidate_count": excluded_5of5,
        "near_miss_count": sum(1 for sample in samples if sample.get("frontier_label", {}).get("category") == "clean_near_miss"),
        "hard_correct_count": sum(1 for sample in samples if sample.get("frontier_label", {}).get("category") == "hard_correct"),
        "true_too_hard_count": sum(1 for sample in samples if sample.get("frontier_label", {}).get("category") == "true_too_hard"),
        "bad_question_count": sum(1 for sample in samples if sample.get("frontier_label", {}).get("category") == "bad_question"),
        "shortcut_issue_count": sum(1 for sample in samples if sample.get("frontier_label", {}).get("category") == "shortcut_issue"),
        "position_sensitive_count": sum(
            1 for sample in samples if sample.get("frontier_label", {}).get("category") == "position_sensitive"
        ),
        "easy_count": sum(1 for sample in samples if sample.get("frontier_label", {}).get("category") == "easy"),
        "label_counts": base.count_map([sample.get("frontier_label", {}).get("category") for sample in samples]),
        "frontier_by_question_type": frontier_by_question_type,
        "frontier_by_attack_skill": frontier_by_attack_skill,
        "largest_type_share": largest_share(frontier_by_question_type),
        "largest_skill_share": largest_share(frontier_by_attack_skill),
        "dev_or_heldout_frontier_count": dev_or_heldout_frontier_count,
        "selection_policy": (
            "stratified bucket frontier: primary {2/5,3/5} weight 1.0 + secondary {1/5,4/5} weight 0.5 "
            "(secondary requires leakage-clean train candidate); 0/5 -> future_hard_set, 5/5 -> clean_replay; "
            "balanced by question type and attack skill"
        ),
    }


def select_frontier(
    *,
    samples: list[dict[str, Any]],
    output_dir: str | Path,
    report_dir: str | Path,
    frontier_splits: set[str],
    forbid_frontier_from_dev_heldout: bool,
    max_question_type_share: float,
    max_attack_skill_share: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    assign_frontier_labels(samples)
    frontier = balanced_select_frontier(
        samples,
        max_question_type_share=max_question_type_share,
        max_attack_skill_share=max_attack_skill_share,
        allowed_splits=frontier_splits,
    )
    summary = build_frontier_summary(samples, frontier)
    if forbid_frontier_from_dev_heldout and summary["dev_or_heldout_frontier_count"] != 0:
        raise RuntimeError(f"Frontier contains dev/heldout rows: {summary['dev_or_heldout_frontier_count']}")
    failure_matrix = build_failure_matrix(samples)
    write_jsonl(output_dir / "frontier.jsonl", frontier)
    write_json(report_dir / "frontier_summary.json", summary)
    write_json(report_dir / "failure_matrix.json", failure_matrix)
    write_json(output_dir / "failure_matrix.json", failure_matrix)
    return frontier, summary, failure_matrix
