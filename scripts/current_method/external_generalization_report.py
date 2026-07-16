#!/usr/bin/env python3
"""Build an auditable external-eval report with paired and label-stratified stats."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable


DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/29a_external_generalization_base_r1_r2_eval"
)
BUCKET_ORDER = ("0-8k", "8k-16k", "16k-24k", "24k-32k")
COMPARABILITY_FIELDS = (
    "backend",
    "input",
    "tokenizer",
    "max_model_len",
    "max_new_tokens",
    "prompt_budget",
    "temperature",
    "top_p",
    "dtype",
)
ROW_ALIGNMENT_FIELDS = ("dataset", "prompt_bucket", "gold_label")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def comparable_config_value(field: str, value: Any) -> Any:
    if field in {"input", "tokenizer"} and value:
        return str(Path(str(value)).resolve())
    return value


def mcnemar_exact_p(left_only: int, right_only: int) -> float:
    discordant = int(left_only) + int(right_only)
    if discordant <= 0:
        return 1.0
    lower = min(int(left_only), int(right_only))
    numerator = sum(math.comb(discordant, index) for index in range(lower + 1))
    return min(1.0, 2.0 * numerator / (2**discordant))


def scored_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in payload.get("results") or []:
        if not isinstance(item, dict) or item.get("scored") is not True:
            continue
        record_id = str(item.get("record_id") or "")
        if not record_id or record_id in rows:
            raise ValueError(f"Missing or duplicate scored record_id: {record_id!r}")
        if not isinstance(item.get("correct"), bool):
            raise ValueError(f"Scored record {record_id!r} has non-boolean correct={item.get('correct')!r}")
        rows[record_id] = item
    return rows


def paired_stats(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    left = scored_by_id(baseline)
    right = scored_by_id(candidate)
    if set(left) != set(right):
        raise ValueError("External eval record_id sets are not aligned")
    for record_id in left:
        for field in ROW_ALIGNMENT_FIELDS:
            if left[record_id].get(field) != right[record_id].get(field):
                raise ValueError(
                    f"External eval metadata differs for record_id={record_id!r}, field={field!r}: "
                    f"{left[record_id].get(field)!r} != {right[record_id].get(field)!r}"
                )
    ids = [record_id for record_id in sorted(left) if predicate is None or predicate(left[record_id])]
    both = sum(bool(left[rid].get("correct")) and bool(right[rid].get("correct")) for rid in ids)
    baseline_only = sum(bool(left[rid].get("correct")) and not bool(right[rid].get("correct")) for rid in ids)
    candidate_only = sum(not bool(left[rid].get("correct")) and bool(right[rid].get("correct")) for rid in ids)
    neither = len(ids) - both - baseline_only - candidate_only
    baseline_correct = both + baseline_only
    candidate_correct = both + candidate_only
    return {
        "n": len(ids),
        "both_correct": both,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "neither_correct": neither,
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "baseline_accuracy": round(baseline_correct / len(ids), 6) if ids else None,
        "candidate_accuracy": round(candidate_correct / len(ids), 6) if ids else None,
        "delta": round((candidate_correct - baseline_correct) / len(ids), 6) if ids else None,
        "mcnemar_exact_p": mcnemar_exact_p(baseline_only, candidate_only),
    }


def label_stats(payload: dict[str, Any]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    predicted: Counter[str] = Counter()
    for item in scored_by_id(payload).values():
        gold = str(item.get("gold_label") or "")
        pred = str(item.get("pred_label") or "")
        totals[gold] += 1
        predicted[pred] += 1
        if item.get("correct") is True:
            correct[gold] += 1
    return {
        "by_gold": {
            gold: {
                "correct": correct[gold],
                "total": total,
                "accuracy": round(correct[gold] / total, 6) if total else None,
            }
            for gold, total in sorted(totals.items())
        },
        "prediction_distribution": dict(predicted.most_common()),
    }


def validate_eval_compatibility(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    reference_ids: set[str] | None = None
    reference_name: str | None = None
    reference_payload: dict[str, Any] | None = None
    reference_rows: dict[str, dict[str, Any]] | None = None
    for name, payload in payloads.items():
        if float(payload.get("temperature") or 0.0) != 0.0:
            issues.append(f"{name}: temperature is not greedy zero")
        if float(payload.get("top_p") or 1.0) != 1.0:
            issues.append(f"{name}: top_p is not 1.0")
        rows = scored_by_id(payload)
        ids = set(rows)
        if reference_ids is None:
            reference_ids = ids
            reference_name = name
            reference_payload = payload
            reference_rows = rows
        elif ids != reference_ids:
            issues.append(f"{name}: scored record_id set differs")
        if reference_payload is not None and name != reference_name:
            for field in COMPARABILITY_FIELDS:
                actual = comparable_config_value(field, payload.get(field))
                expected = comparable_config_value(field, reference_payload.get(field))
                if actual != expected:
                    issues.append(
                        f"{name}: {field} differs from {reference_name}: "
                        f"{actual!r} != {expected!r}"
                    )
            if ids == reference_ids and reference_rows is not None:
                for record_id in sorted(ids):
                    for field in ROW_ALIGNMENT_FIELDS:
                        if rows[record_id].get(field) != reference_rows[record_id].get(field):
                            issues.append(
                                f"{name}: record_id={record_id!r} field={field!r} differs from {reference_name}"
                            )
                            break
                    if issues and issues[-1].startswith(f"{name}: record_id="):
                        break
    return {"healthy": not issues, "issues": issues, "scored_record_count": len(reference_ids or set())}


def build_summary(payloads: dict[str, dict[str, Any]], *, baseline: str) -> dict[str, Any]:
    if baseline not in payloads:
        raise ValueError(f"Baseline {baseline!r} not found in {sorted(payloads)}")
    health = validate_eval_compatibility(payloads)
    if not health["healthy"]:
        raise RuntimeError(f"External eval compatibility gate failed: {health['issues']}")
    comparisons: dict[str, Any] = {}
    for name, payload in payloads.items():
        if name == baseline:
            continue
        comparisons[name] = {
            "overall": paired_stats(payloads[baseline], payload),
            "by_dataset": {
                dataset: paired_stats(payloads[baseline], payload, predicate=lambda row, dataset=dataset: row.get("dataset") == dataset)
                for dataset in sorted({str(row.get("dataset")) for row in scored_by_id(payload).values()})
            },
            "by_prompt_bucket": {
                bucket: paired_stats(payloads[baseline], payload, predicate=lambda row, bucket=bucket: row.get("prompt_bucket") == bucket)
                for bucket in BUCKET_ORDER
            },
        }
    return {
        "baseline": baseline,
        "health": health,
        "models": {
            name: {
                "model": payload.get("model"),
                "summary": payload.get("summary"),
                "label_stats": label_stats(payload),
            }
            for name, payload in payloads.items()
        },
        "paired_comparisons": comparisons,
    }


def markdown(summary: dict[str, Any]) -> str:
    baseline = summary["baseline"]
    baseline_summary = summary["models"][baseline]["summary"] or {}
    by_dataset = baseline_summary.get("by_dataset") or {}
    by_bucket = baseline_summary.get("by_prompt_bucket") or {}
    lines = [
        "# External Generalization Eval — Audited",
        "",
        "All accuracy comparisons below are aligned by `record_id` and use paired exact McNemar tests. Positive subgroup deltas are descriptive unless the paired test supports them.",
        "",
        "## Scored scope",
        "",
        f"- total records: {baseline_summary.get('total', 'n/a')}",
        f"- scored records: {baseline_summary.get('scored_total', 'n/a')}",
        f"- unscored records: {baseline_summary.get('unscored_total', 'n/a')}",
        "- by dataset: " + ", ".join(
            f"{name}={item.get('scored_total', 0)} scored/{item.get('total', 0)} total" for name, item in sorted(by_dataset.items())
        ),
        "- by prompt bucket: " + ", ".join(
            f"{name}={item.get('scored_total', 0)}" for name, item in sorted(by_bucket.items())
        ),
        "",
        "## Overall",
        "",
        "| candidate vs baseline | baseline | candidate | delta | paired n | baseline-only | candidate-only | p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate, comparison in summary["paired_comparisons"].items():
        stats = comparison["overall"]
        lines.append(
            f"| {candidate} vs {baseline} | {pct(stats['baseline_accuracy'])} | {pct(stats['candidate_accuracy'])} | "
            f"{(stats['delta'] or 0.0) * 100:+.2f} pp | {stats['n']} | {stats['baseline_only']} | "
            f"{stats['candidate_only']} | {stats['mcnemar_exact_p']:.4g} |"
        )
    lines.extend(["", "## Paired subgroups", ""])
    for candidate, comparison in summary["paired_comparisons"].items():
        lines.extend(
            [
                f"### {candidate} vs {baseline}",
                "",
                "| subgroup | baseline | candidate | delta | n | p |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for kind in ("by_dataset", "by_prompt_bucket"):
            for label, stats in comparison[kind].items():
                if not stats["n"]:
                    continue
                lines.append(
                    f"| {kind}:{label} | {pct(stats['baseline_accuracy'])} | {pct(stats['candidate_accuracy'])} | "
                    f"{(stats['delta'] or 0.0) * 100:+.2f} pp | {stats['n']} | {stats['mcnemar_exact_p']:.4g} |"
                )
        lines.append("")
    lines.extend(["## Accuracy by gold label", ""])
    for name, model in summary["models"].items():
        values = model["label_stats"]["by_gold"]
        rendered = ", ".join(f"{label}={item['correct']}/{item['total']} ({pct(item['accuracy'])})" for label, item in values.items())
        lines.append(f"- {name}: {rendered}")
    lines.extend(
        [
            "",
            "## Interpretation guard",
            "",
            "This report does not infer a long-context, graph-anchor, or question-diversity capability gain from a small positive bucket alone. Such a claim requires a sufficiently powered, gold-scored, independently designed test set.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired external-generalization report.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--baseline", default="r2")
    parser.add_argument("--models", nargs="*", default=None, help="Model suffixes from eval/full_<name>.json; default discovers all.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_dir = args.run_root / "eval"
    names = args.models or sorted(path.stem.removeprefix("full_") for path in eval_dir.glob("full_*.json"))
    payloads = {name: read_json(eval_dir / f"full_{name}.json") for name in names}
    summary = build_summary(payloads, baseline=args.baseline)
    output_json = args.output_json or (args.run_root / "external_generalization_audited_summary.json")
    output_md = args.output_md or (args.run_root / "reports/external_generalization_report_audited.md")
    write_json(output_json, summary)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown(summary) + "\n", encoding="utf-8")
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
