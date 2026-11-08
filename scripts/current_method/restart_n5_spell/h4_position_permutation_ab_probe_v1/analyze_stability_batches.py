"""Analyze B0/B1/B2 stability across frozen H4 batches."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text
from .analyze_results import canonical_gold, canonical_prediction


CONDITIONS = ("B0", "B1", "B2")
BOOTSTRAP_REPLICATES = 2000


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
        part = [row for row in rows if row.get("context_mode") == context]
        result[context] = _summary_basic(part, permutations)
    for qtype in ("single_choice", "multi_select", "ordering"):
        part = [row for row in rows if row.get("question_type") == qtype]
        if part:
            result[qtype] = _summary_basic(part, permutations)
    return result


def _summary_basic(rows: list[dict[str, Any]], permutations: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("scored")]
    correct = sum(canonical_prediction(row, permutations) == canonical_gold(row) for row in scored)
    return {"count": len(rows), "scored_count": len(scored), "correct_count": correct,
            "accuracy": correct / len(scored) if scored else None,
            "technical_failure_count": len(rows) - len(scored)}


def _bootstrap_root_accuracy(rows: list[dict[str, Any]], permutations: dict[tuple[str, str], dict[str, Any]], *, seed: int, cluster_prefix: str = "") -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[f"{cluster_prefix}{row.get('root_id')}"] .append(row)
    names = sorted(clusters)
    if not names:
        return {"cluster_count": 0, "replicates": 0, "accuracy": None, "ci95": [None, None]}
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [clusters[name] for name in (rng.choice(names) for _ in names)]
        flat = [row for cluster in sampled for row in cluster]
        values.append(sum(canonical_prediction(row, permutations) == canonical_gold(row) for row in flat) / len(flat))
    values.sort()
    lower = values[int(0.025 * (len(values) - 1))]
    upper = values[int(0.975 * (len(values) - 1))]
    scored = [row for row in rows if row.get("scored")]
    correct = sum(canonical_prediction(row, permutations) == canonical_gold(row) for row in scored)
    return {"cluster_count": len(names), "replicates": BOOTSTRAP_REPLICATES,
            "accuracy": correct / len(scored) if scored else None, "ci95": [lower, upper]}


def _pairwise(rows: dict[str, list[dict[str, Any]]], permutations: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    indexed = {name: {_key(row): row for row in rows[name]} for name in CONDITIONS}
    common = set.intersection(*(set(indexed[name]) for name in CONDITIONS))
    result: dict[str, Any] = {"paired_count": len(common)}
    for left, right in (("B0", "B1"), ("B1", "B2"), ("B0", "B2")):
        changed = 0
        transitions = Counter()
        for key in common:
            left_row, right_row = indexed[left][key], indexed[right][key]
            left_pred = canonical_prediction(left_row, permutations)
            right_pred = canonical_prediction(right_row, permutations)
            gold = canonical_gold(indexed["B0"][key])
            changed += int(left_pred != right_pred)
            transitions[f"{int(left_pred == gold)}->{int(right_pred == gold)}"] += 1
        result[f"{left}_vs_{right}"] = {
            "changed_canonical_count": changed,
            "canonical_agreement": 1.0 - changed / len(common) if common else None,
            "correctness_transition": dict(sorted(transitions.items())),
        }
    return result


def _load_batch(name: str, *, manifest: Path, b0: Path, b1: Path, b2: Path) -> dict[str, Any]:
    permutations = _permutations(manifest)
    rows = {"B0": _rows(b0), "B1": _rows(b1), "B2": _rows(b2)}
    indexed = {condition: {_key(row): row for row in rows[condition]} for condition in CONDITIONS}
    key_sets = {condition: set(indexed[condition]) for condition in CONDITIONS}
    common = set.intersection(*(key_sets[condition] for condition in CONDITIONS))
    return {
        "name": name,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": __import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
        "permutations": permutations,
        "rows": rows,
        "indexed": indexed,
        "alignment": {
            "row_counts": {condition: len(rows[condition]) for condition in CONDITIONS},
            "unique_key_counts": {condition: len(key_sets[condition]) for condition in CONDITIONS},
            "common_key_count": len(common),
            "key_sets_identical": all(key_sets[condition] == key_sets["B0"] for condition in CONDITIONS),
            "seed_mismatch_count": sum(len({int(indexed[condition][key].get("seed")) for condition in CONDITIONS}) != 1 for key in common),
        },
    }


def analyze(*, batches: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    for batch in batches:
        batch["summary"] = {condition: _summary(batch["rows"][condition], batch["permutations"]) for condition in CONDITIONS}
        batch["pairwise"] = _pairwise(batch["rows"], batch["permutations"])
        batch["root_bootstrap"] = {
            condition: _bootstrap_root_accuracy(batch["rows"][condition], batch["permutations"], seed=20260831 + index * 100 + ord(condition[-1]))
            for index, condition in enumerate(CONDITIONS)
        }

    pooled: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    same_source: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for batch in batches:
        for condition in CONDITIONS:
            pooled[condition].extend(batch["rows"][condition])
            if batch["name"] in {"current_30", "same_source_holdout_41"}:
                same_source[condition].extend(batch["rows"][condition])
    pooled_permutations = {}
    # Predictions are evaluated with each batch's own manifest before pooling.
    pooled_summaries = {}
    same_source_summaries = {}
    for label, rows_by_condition in (("all_105", pooled), ("same_source_71", same_source)):
        pooled_summaries[label] = {}
        for condition in CONDITIONS:
            rows = []
            for batch in batches:
                if label == "same_source_71" and batch["name"] not in {"current_30", "same_source_holdout_41"}:
                    continue
                rows.extend((row, batch["permutations"]) for row in batch["rows"][condition])
            scored = [(row, perms) for row, perms in rows if row.get("scored")]
            correct = sum(canonical_prediction(row, perms) == canonical_gold(row) for row, perms in scored)
            pooled_summaries[label][condition] = {"count": len(rows), "scored_count": len(scored), "correct_count": correct,
                                                  "accuracy": correct / len(scored) if scored else None,
                                                  "technical_failure_count": len(rows) - len(scored)}

    report = {
        "analysis_version": "h4_position_ab_stability_comparison_v1",
        "batch_count": len(batches),
        "batches": [{key: value for key, value in batch.items() if key not in {"permutations", "rows", "indexed"}} for batch in batches],
        "pooled_summaries": pooled_summaries,
        "interpretation": {
            "independent_observation_unit": "candidate clustered by root; permutations are repeated measurements",
            "same_source_batch_names": ["current_30", "same_source_holdout_41"],
            "independent_batch_name": "independent_semantic_34",
        },
    }
    atomic_json(output_root / "stability_comparison_report.json", report)
    atomic_text(output_root / "stability_comparison_report.md", "# H4 B0/B1/B2 stability comparison\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-manifest", type=Path, required=True)
    parser.add_argument("--current-b0", type=Path, required=True)
    parser.add_argument("--current-b1", type=Path, required=True)
    parser.add_argument("--current-b2", type=Path, required=True)
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--independent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    holdout = args.holdout_root.resolve()
    independent = args.independent_root.resolve()
    batches = [
        _load_batch("current_30", manifest=args.current_manifest.resolve(), b0=args.current_b0.resolve(), b1=args.current_b1.resolve(), b2=args.current_b2.resolve()),
        _load_batch("same_source_holdout_41", manifest=holdout / "permutation_manifest.jsonl", b0=holdout / "b0_results.json", b1=holdout / "b1_results.json", b2=holdout / "b2_results.json"),
        _load_batch("independent_semantic_34", manifest=independent / "permutation_manifest.jsonl", b0=independent / "b0_results.json", b1=independent / "b1_results.json", b2=independent / "b2_results.json"),
    ]
    report = analyze(batches=batches, output_root=args.output_root.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
