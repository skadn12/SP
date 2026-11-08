"""Offline canonical analysis for a completed position A/B probe run.

This never calls a model.  It rebinds label-mode predictions through the
immutable permutation manifest, so ordering slot permutations are scored with
the correct displayed Gold while raw result files remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(run_root: Path) -> list[dict[str, Any]]:
    return [_load(path) for path in sorted((run_root / "results").glob("*.json"))]


def _permutations(run_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for line in (run_root / "permutation_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[(str(row["candidate_id"]), str(row["permutation_id"]))] = row
    return result


def canonical_prediction(row: dict[str, Any], permutations: dict[tuple[str, str], dict[str, Any]]) -> tuple[str, ...] | None:
    parsed = list(row.get("parsed_answer") or [])
    if not parsed or not row.get("scored"):
        return None
    if row.get("output_mode") == "label":
        mapping = permutations[(str(row["candidate_id"]), str(row["permutation_id"]))]["display_to_canonical_mapping"]
        parsed = [str(mapping[value]) for value in parsed if value in mapping]
    else:
        parsed = list((row.get("parse") or {}).get("parsed_canonical") or parsed)
    if row.get("question_type") == "multi_select":
        return tuple(sorted(set(parsed)))
    return tuple(parsed)


def canonical_gold(row: dict[str, Any]) -> tuple[str, ...]:
    gold = [str(value) for value in row.get("canonical_gold") or []]
    return tuple(sorted(set(gold))) if row.get("question_type") == "multi_select" else tuple(gold)


def _entropy(values: list[tuple[str, ...]]) -> float | None:
    if not values:
        return None
    counts = Counter(values)
    n = len(values)
    return -sum((count / n) * math.log(count / n) for count in counts.values())


def _subset_summary(rows: list[dict[str, Any]], permutations: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("scored")]
    correct = [row for row in scored if canonical_prediction(row, permutations) == canonical_gold(row)]
    out: dict[str, Any] = {
        "count": len(rows),
        "scored_count": len(scored),
        "correct_count": len(correct),
        "accuracy": len(correct) / len(scored) if scored else None,
        "technical_failure_count": len(rows) - len(scored),
    }
    for mode in ("no_context", "full_context"):
        part = [row for row in rows if row.get("context_mode") == mode]
        out[mode] = _subset_summary_basic(part, permutations)
    for qtype in ("single_choice", "multi_select", "ordering"):
        part = [row for row in rows if row.get("question_type") == qtype]
        if part:
            out[qtype] = _subset_summary_basic(part, permutations)
    return out


def _subset_summary_basic(rows: list[dict[str, Any]], permutations: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("scored")]
    correct = sum(canonical_prediction(row, permutations) == canonical_gold(row) for row in scored)
    return {"count": len(rows), "scored_count": len(scored),
            "correct_count": correct, "accuracy": correct / len(scored) if scored else None,
            "technical_failure_count": len(rows) - len(scored)}


def _agreement(rows: list[dict[str, Any]], permutations: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[tuple[str, ...]]] = defaultdict(list)
    for row in rows:
        prediction = canonical_prediction(row, permutations)
        if prediction is not None:
            groups[(str(row["candidate_id"]), str(row["context_mode"]))].append(prediction)
    records = []
    for (candidate_id, mode), values in sorted(groups.items()):
        counts = Counter(values)
        majority = max(counts.values()) / len(values)
        records.append({"candidate_id": candidate_id, "context_mode": mode,
                        "permutation_count": len(values), "distinct_canonical_answers": len(counts),
                        "majority_share": majority, "canonical_flip_rate": 1.0 - majority,
                        "canonical_entropy": _entropy(values)})
    return {
        "group_count": len(records),
        "mean_majority_share": sum(x["majority_share"] for x in records) / len(records) if records else None,
        "mean_canonical_flip_rate": sum(x["canonical_flip_rate"] for x in records) / len(records) if records else None,
        "groups_with_multiple_canonical_answers": sum(x["distinct_canonical_answers"] > 1 for x in records),
        "groups": records,
    }


def _paired(rows_by_key: dict[tuple[str, str, str], dict[str, dict[str, Any]]], permutations: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    pairs = [value for value in rows_by_key.values() if all(name in value for name in ("B0", "B1", "B2"))]
    def changed(left: str, right: str) -> int:
        return sum(canonical_prediction(pair[left], permutations) != canonical_prediction(pair[right], permutations) for pair in pairs)
    return {
        "paired_complete_count": len(pairs),
        "b0_b1_canonical_changed_count": changed("B0", "B1"),
        "b1_b2_canonical_changed_count": changed("B1", "B2"),
        "b0_b2_canonical_changed_count": changed("B0", "B2"),
        "b0_b1_canonical_agreement": 1.0 - changed("B0", "B1") / len(pairs) if pairs else None,
        "b1_b2_canonical_agreement": 1.0 - changed("B1", "B2") / len(pairs) if pairs else None,
        "b0_b2_canonical_agreement": 1.0 - changed("B0", "B2") / len(pairs) if pairs else None,
    }


def analyze(run_root: Path) -> dict[str, Any]:
    permutations = _permutations(run_root)
    rows = _rows(run_root)
    stages = {
        "A_broader_all": [row for row in rows if row.get("output_mode") == "label"],
        "A_cyclic": [row for row in rows if row.get("output_mode") == "label" and row.get("is_cyclic") is True],
        "A_non_cyclic_exclusive": [row for row in rows if row.get("output_mode") == "label" and not row.get("is_cyclic")],
        "B0": [row for row in rows if row.get("output_mode") == "label" and row.get("is_cyclic") is True],
        "B1": [row for row in rows if row.get("output_mode") == "content_visible"],
        "B2": [row for row in rows if row.get("output_mode") == "content_hidden"],
    }
    summaries = {name: _subset_summary(part, permutations) for name, part in stages.items()}
    agreement = {name: _agreement(part, permutations) for name, part in stages.items()}
    by_key: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for name in ("B0", "B1", "B2"):
        for row in stages[name]:
            key = (str(row["candidate_id"]), str(row["permutation_id"]), str(row["context_mode"]))
            by_key[key][name] = row
    report = {
        "analysis_version": "h4_position_ab_canonical_analysis_v1",
        "run_root": str(run_root.resolve()),
        "row_count": len(rows), "permutation_manifest_count": len(permutations),
        "raw_technical_failure_count": sum(not row.get("scored") for row in rows),
        "ordering_gold_rebinding": "corrected_offline_from_display_to_canonical_mapping",
        "stage_summaries": summaries,
        "canonical_agreement": agreement,
        "paired_B0_B1_B2": _paired(by_key, permutations),
        "interpretation_scope": "current 30-candidate sample and its 792 manifest permutations only",
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.run_root.resolve())
    atomic_json(args.run_root / "corrected_analysis.json", report)
    atomic_text(args.run_root / "corrected_analysis.md", "# H4 position A/B canonical analysis\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
