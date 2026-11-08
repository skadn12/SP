"""Compare B0, repaired B1, and repaired B2 on canonical paired answers."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text, sha256_file
from .analyze_results import canonical_gold, canonical_prediction


CONDITIONS = ("B0", "B1", "B2")
EXPECTED_ROWS = 264
CONTENT_PROMPT_VERSION = "h4_position_ab_content_responder_v2"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    value = _load(path)
    return list(value.get("rows") or value) if isinstance(value, dict) else list(value)


def _permutations(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[(str(row["candidate_id"]), str(row["permutation_id"]))] = row
    return result


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["candidate_id"]), str(row["permutation_id"]), str(row["context_mode"]))


def _summary(rows: list[dict[str, Any]], permutations: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("scored")]
    correct = sum(canonical_prediction(row, permutations) == canonical_gold(row) for row in scored)
    result: dict[str, Any] = {
        "count": len(rows),
        "scored_count": len(scored),
        "correct_count": correct,
        "accuracy": correct / len(scored) if scored else None,
        "technical_failure_count": len(rows) - len(scored),
    }
    for context in ("no_context", "full_context"):
        result[context] = _summary_subset([row for row in rows if row.get("context_mode") == context], permutations)
    for qtype in ("single_choice", "multi_select", "ordering"):
        part = [row for row in rows if row.get("question_type") == qtype]
        if part:
            result[qtype] = _summary_subset(part, permutations)
    return result


def _summary_subset(rows: list[dict[str, Any]], permutations: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("scored")]
    correct = sum(canonical_prediction(row, permutations) == canonical_gold(row) for row in scored)
    return {
        "count": len(rows),
        "scored_count": len(scored),
        "correct_count": correct,
        "accuracy": correct / len(scored) if scored else None,
        "technical_failure_count": len(rows) - len(scored),
    }


def _agreement(rows: list[dict[str, Any]], permutations: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[tuple[str, ...]]] = defaultdict(list)
    for row in rows:
        prediction = canonical_prediction(row, permutations)
        if prediction is not None:
            groups[(str(row["candidate_id"]), str(row["context_mode"]))].append(prediction)
    records = []
    for (candidate_id, context), values in sorted(groups.items()):
        counts = Counter(values)
        majority = max(counts.values()) / len(values)
        records.append({
            "candidate_id": candidate_id,
            "context_mode": context,
            "permutation_count": len(values),
            "distinct_canonical_answers": len(counts),
            "majority_share": majority,
            "canonical_flip_rate": 1.0 - majority,
        })
    return {
        "group_count": len(records),
        "mean_majority_share": sum(row["majority_share"] for row in records) / len(records) if records else None,
        "mean_canonical_flip_rate": sum(row["canonical_flip_rate"] for row in records) / len(records) if records else None,
        "groups_with_multiple_canonical_answers": sum(row["distinct_canonical_answers"] > 1 for row in records),
    }


def _pairwise(rows_by_condition: dict[str, dict[tuple[str, str, str], dict[str, Any]]], permutations: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    common = set.intersection(*(set(rows_by_condition[name]) for name in CONDITIONS))
    predictions: dict[str, dict[tuple[str, str, str], tuple[str, ...] | None]] = {
        name: {key: canonical_prediction(rows_by_condition[name][key], permutations) for key in common}
        for name in CONDITIONS
    }
    gold = {key: canonical_gold(rows_by_condition["B0"][key]) for key in common}
    result: dict[str, Any] = {"paired_count": len(common)}
    for left, right in (("B0", "B1"), ("B1", "B2"), ("B0", "B2")):
        changed = sum(predictions[left][key] != predictions[right][key] for key in common)
        transitions = Counter()
        by_context: dict[str, dict[str, int]] = {}
        for key in common:
            left_correct = predictions[left][key] == gold[key]
            right_correct = predictions[right][key] == gold[key]
            transitions[f"{int(left_correct)}->{int(right_correct)}"] += 1
            context = key[2]
            bucket = by_context.setdefault(context, {"paired_count": 0, "changed_count": 0})
            bucket["paired_count"] += 1
            bucket["changed_count"] += int(predictions[left][key] != predictions[right][key])
        result[f"{left}_vs_{right}"] = {
            "changed_canonical_count": changed,
            "canonical_agreement": 1.0 - changed / len(common) if common else None,
            "correctness_transition": dict(sorted(transitions.items())),
            "by_context": by_context,
        }
    return result


def _condition_rows(source_run: Path, b1_run: Path, b2_run: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "B0": _rows(source_run / "b0_cyclic_results.json"),
        "B1": _rows(b1_run / "b1_results.json"),
        "B2": _rows(b2_run / "b2_results.json"),
    }


def analyze(*, source_run: Path, b1_run: Path, b2_run: Path, output_root: Path) -> dict[str, Any]:
    manifest_path = source_run / "permutation_manifest.jsonl"
    permutations = _permutations(manifest_path)
    condition_rows = _condition_rows(source_run, b1_run, b2_run)
    rows_by_condition = {name: {_key(row): row for row in rows} for name, rows in condition_rows.items()}
    key_sets = {name: set(rows) for name, rows in rows_by_condition.items()}
    common = set.intersection(*(key_sets[name] for name in CONDITIONS))
    duplicate_counts = {name: len(condition_rows[name]) - len(rows_by_condition[name]) for name in CONDITIONS}
    manifest_hashes = {
        "source": sha256_file(source_run / "permutation_manifest.jsonl"),
        "b1": sha256_file(b1_run / "permutation_manifest.jsonl"),
        "b2": sha256_file(b2_run / "permutation_manifest.jsonl"),
    }
    seed_mismatches = sum(len({int(rows_by_condition[name][key].get("seed")) for name in CONDITIONS}) != 1 for key in common)
    gold_mismatches = sum(
        len({tuple(str(value) for value in rows_by_condition[name][key].get("canonical_gold") or []) for name in CONDITIONS}) != 1
        for key in common
    )
    reserve_mismatches = sum(
        rows_by_condition["B1"][key].get("output_reserve") != rows_by_condition["B2"][key].get("output_reserve")
        for key in common
    )
    prompt_versions = {
        name: sorted({row.get("content_responder_prompt_version") for row in condition_rows[name]})
        for name in CONDITIONS
    }
    report = {
        "analysis_version": "h4_position_ab_repaired_b_comparison_v1",
        "source_run": str(source_run.resolve()),
        "condition_runs": {"B0": str(source_run.resolve()), "B1": str(b1_run.resolve()), "B2": str(b2_run.resolve())},
        "permutation_manifest_count": len(permutations),
        "manifest_sha256": manifest_hashes,
        "condition_summaries": {name: _summary(condition_rows[name], permutations) for name in CONDITIONS},
        "canonical_agreement": {name: _agreement(condition_rows[name], permutations) for name in CONDITIONS},
        "paired_comparison": _pairwise(rows_by_condition, permutations),
        "alignment": {
            "expected_rows_per_condition": EXPECTED_ROWS,
            "observed_rows": {name: len(condition_rows[name]) for name in CONDITIONS},
            "duplicate_key_count": duplicate_counts,
            "key_counts": {name: len(key_sets[name]) for name in CONDITIONS},
            "common_key_count": len(common),
            "key_sets_identical": all(key_sets[name] == key_sets["B0"] for name in CONDITIONS),
            "seed_mismatch_count": seed_mismatches,
            "canonical_gold_mismatch_count": gold_mismatches,
            "b1_b2_output_reserve_mismatch_count": reserve_mismatches,
            "manifest_hashes_identical": len(set(manifest_hashes.values())) == 1,
            "content_prompt_versions": prompt_versions,
            "repaired_content_prompt_version": CONTENT_PROMPT_VERSION,
        },
        "interpretation_scope": "B0 from immutable live_4 plus repaired B1/B2 on the same 30-candidate cyclic sample",
    }
    atomic_json(output_root / "b_comparison_report.json", report)
    atomic_text(output_root / "b_comparison_report.md", "# Repaired B experiment comparison\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--b1-run", type=Path, required=True)
    parser.add_argument("--b2-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(
        source_run=args.source_run.resolve(),
        b1_run=args.b1_run.resolve(),
        b2_run=args.b2_run.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
