from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_old_improved_gate200 import normalize_text, score_prediction, turn_sort_value  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality audit for old_improved_gate200 frontier samples.")
    parser.add_argument("--frontier_jsonl", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--samples_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_frontier_audit", type=int, default=20)
    parser.add_argument("--num_zero_bucket_audit", type=int, default=10)
    parser.add_argument("--num_near_miss_audit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260615)
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def count_map(values: list[Any]) -> dict[str, int]:
    counts = Counter(str(value if value is not None else "unknown") for value in values)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def option_map(sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
    options = sample.get("qa", {}).get("options")
    if not isinstance(options, list):
        return {}
    return {str(option.get("id")): option for option in options if isinstance(option, dict) and option.get("id") is not None}


def dialogue_turn_ids(sample: dict[str, Any]) -> set[str]:
    turns = sample.get("dialogue", {}).get("turns")
    if not isinstance(turns, list):
        return set()
    return {str(turn.get("turn_id")) for turn in turns if isinstance(turn, dict) and turn.get("turn_id") is not None}


def graph_text(sample: dict[str, Any]) -> str:
    parts: list[str] = []
    graph_update = sample.get("graph_update") if isinstance(sample.get("graph_update"), dict) else {}
    for key in ("entities", "events", "state_changes", "visibility_edges", "invalidations", "delayed_activations", "cancellations", "restorations"):
        value = graph_update.get(key)
        if value:
            parts.append(json.dumps(value, ensure_ascii=False))
    return normalize_text("\n".join(parts))


def dialogue_text(sample: dict[str, Any]) -> str:
    turns = sample.get("dialogue", {}).get("turns")
    if not isinstance(turns, list):
        return ""
    return normalize_text("\n".join(str(turn.get("text") or "") for turn in turns if isinstance(turn, dict)))


def gold_derivable(sample: dict[str, Any]) -> tuple[bool, str]:
    qa = sample.get("qa", {})
    qtype = str(qa.get("question_type") or "")
    gold = [str(item) for item in qa.get("gold", [])]
    options = option_map(sample)
    turns = dialogue_turn_ids(sample)
    gtext = graph_text(sample)
    dtext = dialogue_text(sample)

    if not gold:
        return False, "missing_gold"
    if qtype in {"multi_select", "ordering"}:
        missing = [item for item in gold if item not in options]
        if missing:
            return False, f"gold_option_missing:{missing}"
        return True, "gold_option_ids_present"
    if qtype == "evidence_selection":
        missing = [item for item in gold if item not in turns]
        if missing:
            return False, f"gold_evidence_turn_missing:{missing}"
        return True, "gold_evidence_turns_present"
    if qtype == "invalidated_statement_detection":
        if not all(item in options for item in gold):
            return False, "gold_components_missing_from_options"
        invalidations = sample.get("graph_update", {}).get("invalidations", [])
        if not invalidations:
            return False, "missing_invalidation_graph_edge"
        return True, "gold_components_and_invalidation_present"
    if qtype == "state_reconstruction":
        gold_norm = normalize_text(gold[0])
        if gold_norm and (gold_norm in gtext or gold_norm in dtext):
            return True, "gold_text_in_graph_or_dialogue"
        return False, "state_gold_not_in_graph_or_dialogue"
    return False, f"unknown_question_type:{qtype}"


def options_unambiguous(sample: dict[str, Any]) -> tuple[bool, str]:
    qa = sample.get("qa", {})
    qtype = str(qa.get("question_type") or "")
    gold = [str(item) for item in qa.get("gold", [])]
    options = option_map(sample)
    if qtype == "state_reconstruction":
        return (bool(gold and str(gold[0]).strip()), "open_answer_gold_present")
    if not options:
        return False, "missing_options"
    ids = list(options)
    if len(ids) != len(set(ids)):
        return False, "duplicate_option_ids"
    texts = [normalize_text(option.get("text")) for option in options.values()]
    non_empty = [text for text in texts if text]
    if len(non_empty) != len(texts):
        return False, "empty_option_text"
    if len(set(texts)) != len(texts):
        return False, "duplicate_option_text"
    if not set(gold).issubset(set(ids)):
        return False, "gold_not_subset_of_options"
    if qtype == "multi_select" and len(gold) < 2:
        return False, "multi_select_needs_multiple_gold"
    if qtype == "ordering" and len(gold) != len(options):
        return False, "ordering_gold_must_include_all_options"
    if qtype == "invalidated_statement_detection" and len(gold) != 3:
        return False, "invalidated_gold_must_have_three_components"
    return True, "options_unique_and_gold_valid"


def distractors_effective(sample: dict[str, Any]) -> tuple[bool, str]:
    qa = sample.get("qa", {})
    qtype = str(qa.get("question_type") or "")
    if qtype in {"ordering", "state_reconstruction"}:
        return True, "not_distractor_based"
    gold = {str(item) for item in qa.get("gold", [])}
    options = option_map(sample)
    distractors = [option for oid, option in options.items() if oid not in gold]
    if len(distractors) < 2 and qtype != "invalidated_statement_detection":
        return False, "too_few_distractors"
    gold_texts = [normalize_text(options[item].get("text")) for item in gold if item in options]
    distractor_texts = [normalize_text(option.get("text")) for option in distractors]
    if any(text in gold_texts for text in distractor_texts):
        return False, "distractor_duplicates_gold"
    if qtype == "multi_select":
        gold_tokens = set().union(*(set(text.split()) for text in gold_texts)) if gold_texts else set()
        overlaps = []
        for text in distractor_texts:
            tokens = set(text.split())
            if gold_tokens and tokens:
                overlaps.append(len(gold_tokens & tokens) / max(1, len(gold_tokens)))
        if overlaps and max(overlaps) < 0.03:
            return False, "distractors_too_disjoint"
    return True, "distractors_valid"


def shortcut_clean(sample: dict[str, Any]) -> tuple[bool, str]:
    validation = sample.get("validation") if isinstance(sample.get("validation"), dict) else {}
    details = validation.get("details") if isinstance(validation.get("details"), dict) else {}
    shortcut_checks = details.get("anti_shortcut_checks") if isinstance(details.get("anti_shortcut_checks"), dict) else {}
    triggered = [key for key, value in shortcut_checks.items() if value]
    warnings = [str(item) for item in validation.get("warnings", []) if item]
    shortcut_warnings = [item for item in warnings if "shortcut" in item or "leakage" in item or "exposure" in item or "distractor_too_easy" in item]
    if validation.get("anti_shortcut_pass") is not True:
        return False, "anti_shortcut_pass_false"
    if triggered or shortcut_warnings:
        return False, f"shortcut_flags:{triggered + shortcut_warnings}"
    return True, "no_shortcut_flags"


def recompute_eval(sample: dict[str, Any]) -> tuple[bool, float, list[dict[str, Any]]]:
    predictions = sample.get("solver_eval", {}).get("predictions")
    if not isinstance(predictions, list):
        return False, 0.0, []
    rescored = []
    exact_count = 0
    partials = []
    for pred in predictions:
        if not isinstance(pred, dict):
            continue
        text = str(pred.get("prediction") or "")
        exact, partial, parsed = score_prediction(sample, text)
        exact_count += int(exact)
        partials.append(float(partial))
        rescored.append({"prediction": text, "exact": exact, "partial_credit": round(partial, 6), "parsed": parsed})
    soft = exact_count / len(predictions) if predictions else 0.0
    partial_mean = sum(partials) / len(predictions) if predictions else 0.0
    stored_soft = float(sample.get("solver_eval", {}).get("soft_accuracy") or 0.0)
    stored_partial = float(sample.get("solver_eval", {}).get("partial_credit") or 0.0)
    consistent = abs(soft - stored_soft) <= 1e-6 and abs(partial_mean - stored_partial) <= 1e-6
    return consistent, partial_mean, rescored


def eval_not_dummy(sample: dict[str, Any]) -> tuple[bool, str]:
    predictions = sample.get("solver_eval", {}).get("predictions")
    k = int(sample.get("solver_eval", {}).get("k") or 5)
    if not isinstance(predictions, list) or len(predictions) < k:
        return False, "missing_k_predictions"
    texts = [str(pred.get("prediction") or "").strip() for pred in predictions if isinstance(pred, dict)]
    if len(texts) < k:
        return False, "prediction_records_malformed"
    if any(not text for text in texts):
        return False, "empty_prediction"
    unique = set(texts)
    if len(unique) == 1:
        only = next(iter(unique)).lower()
        exact_count = int(sample.get("solver_eval", {}).get("exact_correct_count") or 0)
        if exact_count == 0 and (len(only) < 2 or only in {"a", "b", "c", "d", "i don't know", "unknown", "none"}):
            return False, "repeated_generic_wrong_dummy"
    return True, "k_predictions_present_and_non_dummy"


def classify_zero_bucket(sample: dict[str, Any], partial_mean: float, issues: list[str]) -> str:
    if sample.get("solver_eval", {}).get("bucket") != "0/5":
        return "not_zero_bucket"
    if issues:
        return "bad_question"
    if partial_mean >= 0.4:
        return "true_too_hard"
    return "bad_question"


def classify_near_miss(sample: dict[str, Any], partial_mean: float, issues: list[str]) -> str:
    bucket = str(sample.get("solver_eval", {}).get("bucket") or "")
    if bucket not in {"1/5", "2/5", "3/5"}:
        return "not_near_miss_bucket"
    if issues:
        return "not_clean"
    if partial_mean >= 0.4:
        return "clean_near_miss"
    return "partial_credit_too_low"


def audit_one(sample: dict[str, Any], group: str) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    issues: list[str] = []
    gold_ok, gold_reason = gold_derivable(sample)
    checks["gold_derivable"] = {"pass": gold_ok, "reason": gold_reason}
    if not gold_ok:
        issues.append("gold_issue")

    options_ok, options_reason = options_unambiguous(sample)
    checks["question_options_gold_unambiguous"] = {"pass": options_ok, "reason": options_reason}
    if not options_ok:
        issues.append("ambiguous_issue" if "ambiguous" in options_reason else "format_issue")

    distractors_ok, distractors_reason = distractors_effective(sample)
    checks["distractors_effective"] = {"pass": distractors_ok, "reason": distractors_reason}
    if not distractors_ok:
        issues.append("format_issue")

    shortcut_ok, shortcut_reason = shortcut_clean(sample)
    checks["anti_shortcut"] = {"pass": shortcut_ok, "reason": shortcut_reason}
    if not shortcut_ok:
        issues.append("shortcut_issue")

    eval_ok, eval_reason = eval_not_dummy(sample)
    checks["k5_eval_real"] = {"pass": eval_ok, "reason": eval_reason}
    if not eval_ok:
        issues.append("eval_dummy_suspect")

    partial_consistent, partial_mean, rescored = recompute_eval(sample)
    checks["partial_credit"] = {
        "pass": partial_consistent,
        "reason": "partial_credit_recomputed_matches" if partial_consistent else "partial_credit_mismatch",
        "recomputed_mean": round(partial_mean, 6),
        "stored_mean": sample.get("solver_eval", {}).get("partial_credit"),
    }
    if not partial_consistent:
        issues.append("partial_credit_issue")

    zero_class = classify_zero_bucket(sample, partial_mean, issues)
    near_class = classify_near_miss(sample, partial_mean, issues)
    checks["zero_bucket_classification"] = zero_class
    checks["near_miss_classification"] = near_class

    clean_pass = not issues
    return {
        "sample_id": sample.get("sample_id"),
        "audit_group": group,
        "question_type": sample.get("qa", {}).get("question_type"),
        "attack_skill": sample.get("attack_plan", {}).get("attack_skill"),
        "solver_bucket": sample.get("solver_eval", {}).get("bucket"),
        "soft_accuracy": sample.get("solver_eval", {}).get("soft_accuracy"),
        "partial_credit": sample.get("solver_eval", {}).get("partial_credit"),
        "frontier_category": sample.get("frontier_label", {}).get("category"),
        "is_frontier": sample.get("frontier_label", {}).get("is_frontier"),
        "audit_pass": clean_pass,
        "issues": sorted(set(issues)),
        "checks": checks,
        "rescored_predictions": rescored,
    }


def stratified_select(rows: list[dict[str, Any]], n: int, key_fn: Callable[[dict[str, Any]], tuple[Any, Any]]) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    buckets: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[key_fn(row)].append(row)
    for bucket_rows in buckets.values():
        bucket_rows.sort(key=lambda item: str(item.get("sample_id")))
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets, key=lambda key: (str(key[0]), str(key[1])))
    while len(selected) < n and any(buckets.values()):
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                if len(selected) >= n:
                    break
    return selected


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# old_improved_gate200_quality_audit",
        "",
        f"- frontier_audited: {report['frontier_audited']}",
        f"- zero_bucket_audited: {report['zero_bucket_audited']}",
        f"- near_miss_audited: {report['near_miss_audited']}",
        f"- gold_issue_count: {report['gold_issue_count']}",
        f"- ambiguous_issue_count: {report['ambiguous_issue_count']}",
        f"- format_issue_count: {report['format_issue_count']}",
        f"- shortcut_issue_count: {report['shortcut_issue_count']}",
        f"- eval_dummy_suspect_count: {report['eval_dummy_suspect_count']}",
        f"- partial_credit_issue_count: {report['partial_credit_issue_count']}",
        f"- bad_question_count: {report['bad_question_count']}",
        f"- clean_frontier_audit_pass_rate: {report['clean_frontier_audit_pass_rate']}",
        f"- zero_bucket_true_too_hard_rate: {report['zero_bucket_true_too_hard_rate']}",
        f"- near_miss_clean_rate: {report['near_miss_clean_rate']}",
        f"- recommendation: {report['recommendation']}",
        "",
        "## Coverage",
        "",
        f"- frontier_attack_skill_coverage: {report['frontier_attack_skill_coverage']}",
        f"- frontier_question_type_coverage: {report['frontier_question_type_coverage']}",
        "",
        "## Audited Frontier By Type",
        "",
    ]
    for key, value in report["audited_frontier_by_question_type"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Audited Frontier By Attack Skill", ""])
    for key, value in report["audited_frontier_by_attack_skill"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def recommendation(report: dict[str, Any]) -> str:
    passed = (
        report["clean_frontier_audit_pass_rate"] >= 0.85
        and report["near_miss_clean_rate"] >= 0.80
        and report["eval_dummy_suspect_count"] == 0
        and report["gold_issue_count"] <= 1
        and report["ambiguous_issue_count"] <= 2
        and report["shortcut_issue_count"] <= 2
    )
    return "proceed_to_old_improved_r5_train_small" if passed else "fix_gate_validation_before_training"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(args.samples_jsonl)
    frontier = load_jsonl(args.frontier_jsonl)
    eval_rows = load_jsonl(args.eval_jsonl)
    sample_by_id = {str(sample.get("sample_id")): sample for sample in samples}
    eval_by_id = {str(row.get("sample_id")): row for row in eval_rows}

    # Refresh sample eval from eval_jsonl when present so the audit catches missing external eval rows.
    for sample_id, sample in sample_by_id.items():
        eval_row = eval_by_id.get(sample_id)
        if eval_row and isinstance(eval_row.get("solver_eval"), dict):
            sample["solver_eval"] = eval_row["solver_eval"]

    frontier_samples = [sample_by_id[str(row.get("sample_id"))] for row in frontier if str(row.get("sample_id")) in sample_by_id]
    zero_bucket_samples = [sample for sample in samples if str(sample.get("solver_eval", {}).get("bucket")) == "0/5"]
    near_miss_samples = [
        sample
        for sample in samples
        if str(sample.get("solver_eval", {}).get("bucket")) in {"1/5", "2/5", "3/5"}
        or sample.get("frontier_label", {}).get("category") == "clean_near_miss"
    ]

    key_fn = lambda row: (row.get("qa", {}).get("question_type"), row.get("attack_plan", {}).get("attack_skill"))
    selected_frontier = stratified_select(frontier_samples, args.num_frontier_audit, key_fn)
    selected_zero = stratified_select(zero_bucket_samples, args.num_zero_bucket_audit, key_fn)
    selected_near = stratified_select(near_miss_samples, args.num_near_miss_audit, key_fn)

    audit_rows: list[dict[str, Any]] = []
    for group, rows in (
        ("frontier", selected_frontier),
        ("zero_bucket", selected_zero),
        ("near_miss", selected_near),
    ):
        for sample in rows:
            audit_rows.append(audit_one(sample, group))

    issue_counts = Counter(issue for row in audit_rows for issue in row["issues"])
    frontier_audit = [row for row in audit_rows if row["audit_group"] == "frontier"]
    zero_audit = [row for row in audit_rows if row["audit_group"] == "zero_bucket"]
    near_audit = [row for row in audit_rows if row["audit_group"] == "near_miss"]

    clean_frontier = sum(1 for row in frontier_audit if row["audit_pass"])
    zero_true_too_hard = sum(1 for row in zero_audit if row["checks"]["zero_bucket_classification"] == "true_too_hard")
    near_clean = sum(1 for row in near_audit if row["checks"]["near_miss_classification"] == "clean_near_miss" or row["audit_pass"])
    bad_question_count = sum(
        1
        for row in audit_rows
        if row["checks"].get("zero_bucket_classification") == "bad_question"
        or "gold_issue" in row["issues"]
        or "format_issue" in row["issues"]
        or "ambiguous_issue" in row["issues"]
    )

    report = {
        "audit_name": "old_improved_gate200_quality_audit",
        "frontier_audited": len(frontier_audit),
        "zero_bucket_audited": len(zero_audit),
        "near_miss_audited": len(near_audit),
        "gold_issue_count": int(issue_counts.get("gold_issue", 0)),
        "ambiguous_issue_count": int(issue_counts.get("ambiguous_issue", 0)),
        "format_issue_count": int(issue_counts.get("format_issue", 0)),
        "shortcut_issue_count": int(issue_counts.get("shortcut_issue", 0)),
        "eval_dummy_suspect_count": int(issue_counts.get("eval_dummy_suspect", 0)),
        "partial_credit_issue_count": int(issue_counts.get("partial_credit_issue", 0)),
        "bad_question_count": int(bad_question_count),
        "clean_frontier_audit_pass_rate": round(clean_frontier / len(frontier_audit), 6) if frontier_audit else 0.0,
        "zero_bucket_true_too_hard_rate": round(zero_true_too_hard / len(zero_audit), 6) if zero_audit else 0.0,
        "near_miss_clean_rate": round(near_clean / len(near_audit), 6) if near_audit else 0.0,
        "frontier_attack_skill_coverage": len({row.get("attack_skill") for row in frontier_audit}),
        "frontier_question_type_coverage": len({row.get("question_type") for row in frontier_audit}),
        "audited_frontier_by_question_type": count_map([row["question_type"] for row in frontier_audit]),
        "audited_frontier_by_attack_skill": count_map([row["attack_skill"] for row in frontier_audit]),
        "eval_rows_found": len(eval_by_id),
        "sample_rows_found": len(sample_by_id),
        "frontier_rows_found": len(frontier_samples),
        "recommendation": "",
    }
    report["recommendation"] = recommendation(report)

    write_jsonl(output_dir / "audit_samples.jsonl", audit_rows)
    write_json(output_dir / "audit_report.json", report)
    (output_dir / "audit_report.md").write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved audit outputs to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
