"""Audit and summarize independent base-Qwen3-4B B0/B1/B2 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text


CONDITIONS = ("B0", "B1", "B2")
EXPECTED_HASHES = {
    "config_sha256": "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
    "model_index_sha256": "6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2",
    "tokenizer_json_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    value = _load(path)
    rows = value.get("rows") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError(f"rows_missing:{path}")
    return [dict(row) for row in rows]


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["candidate_id"]), str(row["permutation_id"]), str(row["context_mode"]))


def _wilson(correct: int, total: int) -> list[float | None]:
    if not total:
        return [None, None]
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("scored")]
    correct = sum(bool(row.get("correct")) for row in scored)
    result: dict[str, Any] = {
        "count": len(rows), "scored_count": len(scored), "correct_count": correct,
        "accuracy": correct / len(scored) if scored else None,
        "wilson95": _wilson(correct, len(scored)),
        "technical_failure_count": len(rows) - len(scored),
    }
    for field, values in (("context_mode", ("no_context", "full_context")),
                          ("question_type", ("single_choice", "multi_select", "ordering"))):
        result[field] = {}
        for value in values:
            part = [row for row in rows if row.get(field) == value]
            if part:
                result[field][value] = _summary_basic(part)
    return result


def _summary_basic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("scored")]
    correct = sum(bool(row.get("correct")) for row in scored)
    return {"count": len(rows), "scored_count": len(scored), "correct_count": correct,
            "accuracy": correct / len(scored) if scored else None,
            "wilson95": _wilson(correct, len(scored)),
            "technical_failure_count": len(rows) - len(scored)}


def _load_run(name: str, run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    report = _load(run_root / "stability_solver_report.json")
    if report.get("status") != "completed":
        raise RuntimeError(f"run_not_completed:{name}:{report.get('status')}")
    preflight = _load(run_root / "preflight_report.json") if (run_root / "preflight_report.json").is_file() else {}
    checkpoint = dict(report.get("solver_checkpoint") or report.get("base_model_identity")
                     or preflight.get("base_model_identity") or {})
    if "model_path" not in checkpoint and report.get("solver_model"):
        checkpoint["model_path"] = report.get("solver_model")
    if "tokenizer_path" not in checkpoint and report.get("tokenizer"):
        checkpoint["tokenizer_path"] = report.get("tokenizer")
    if "model_path" not in checkpoint and preflight.get("solver_model"):
        checkpoint["model_path"] = preflight.get("solver_model")
    if "tokenizer_path" not in checkpoint and preflight.get("tokenizer"):
        checkpoint["tokenizer_path"] = preflight.get("tokenizer")
    if checkpoint.get("model_path") != "/root/autodl-tmp/Qwen3-4B":
        raise RuntimeError(f"non_base_model:{name}:{checkpoint.get('model_path')}")
    if {key: checkpoint.get(key) for key in EXPECTED_HASHES} != EXPECTED_HASHES:
        raise RuntimeError(f"base_hash_mismatch:{name}")
    sample_rows = [json.loads(line) for line in (run_root / "sample_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    roots = {str(row["root_id"]) for row in sample_rows}
    ids = {str(row["candidate_id"]) for row in sample_rows}
    if len(sample_rows) != 30 or len(ids) != 30:
        raise RuntimeError(f"sample_count_or_duplicate:{name}")
    rows = {condition: _rows(run_root / f"{condition.lower()}_results.json") for condition in CONDITIONS}
    indexes = {condition: {_key(row): row for row in rows[condition]} for condition in CONDITIONS}
    key_sets = {condition: set(indexes[condition]) for condition in CONDITIONS}
    if any(len(rows[condition]) != 264 for condition in CONDITIONS) or len(set.intersection(*key_sets.values())) != 264:
        raise RuntimeError(f"condition_alignment_failed:{name}")
    if any(not row.get("scored") for condition in CONDITIONS for row in rows[condition]):
        raise RuntimeError(f"unscored_result:{name}")
    seed_mismatch = sum(len({indexes[condition][key].get("seed") for condition in CONDITIONS}) != 1 for key in key_sets["B0"])
    accounting = report.get("request_accounting") or {}
    if accounting.get("request_count") != 792 or accounting.get("completed_request_count") != 792 or accounting.get("ambiguous_request_count") != 0:
        raise RuntimeError(f"request_accounting_failed:{name}:{accounting}")
    return {"name": name, "run_root": str(run_root), "report": report, "sample_rows": sample_rows,
            "roots": roots, "ids": ids, "rows": rows, "indexes": indexes,
            "manifest_sha256": _sha(run_root / "permutation_manifest.jsonl"),
            "seed_mismatch_count": seed_mismatch}


def _paired(batch: dict[str, Any], left: str, right: str) -> dict[str, Any]:
    common = set(batch["indexes"][left]) & set(batch["indexes"][right])
    left_only = right_only = ties = changed = 0
    for key in common:
        a = bool(batch["indexes"][left][key].get("correct"))
        b = bool(batch["indexes"][right][key].get("correct"))
        left_only += int(a and not b); right_only += int(b and not a)
        ties += int(a == b); changed += int(a != b)
    return {"paired_count": len(common), "left_correct_only": left_only,
            "right_correct_only": right_only, "ties": ties,
            "changed_canonical_or_correctness_count": changed,
            "accuracy_difference_pp": (left_only - right_only) / len(common) * 100 if common else None}


def _pooled(batches: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [row for batch in batches for row in batch["rows"][condition]]
        result[condition] = _summary(rows)
    pairwise: dict[str, Any] = {}
    for left, right in (("B0", "B1"), ("B0", "B2"), ("B2", "B1")):
        values = [_paired(batch, left, right) for batch in batches]
        pairwise[f"{left}_vs_{right}"] = {
            "batches": values,
            "paired_count": sum(value["paired_count"] for value in values),
            "left_correct_only": sum(value["left_correct_only"] for value in values),
            "right_correct_only": sum(value["right_correct_only"] for value in values),
            "ties": sum(value["ties"] for value in values),
            "accuracy_difference_pp": sum(value["left_correct_only"] - value["right_correct_only"] for value in values) / sum(value["paired_count"] for value in values) * 100,
        }
    result["pairwise"] = pairwise
    return result


def _root_bootstrap(batches: list[dict[str, Any]], left: str, right: str, seed: int = 20260906, replicates: int = 10000) -> dict[str, Any]:
    clusters: list[dict[str, tuple[int, int]]] = []
    for batch in batches:
        by_root: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {condition: [] for condition in CONDITIONS})
        for condition in (left, right):
            for row in batch["rows"][condition]:
                by_root[str(row["root_id"])][condition].append(row)
        for values in by_root.values():
            clusters.append({condition: (sum(bool(row.get("correct")) for row in values[condition]), len(values[condition])) for condition in (left, right)})
    def difference(sample: list[dict[str, tuple[int, int]]]) -> float:
        left_correct = sum(item[left][0] for item in sample); left_total = sum(item[left][1] for item in sample)
        right_correct = sum(item[right][0] for item in sample); right_total = sum(item[right][1] for item in sample)
        return left_correct / left_total - right_correct / right_total
    observed = difference(clusters)
    rng = random.Random(seed)
    values = sorted(difference([rng.choice(clusters) for _ in clusters]) for _ in range(replicates))
    return {"root_cluster_count": len(clusters), "replicates": replicates, "observed_difference_pp": observed * 100,
            "ci95_difference_pp": [values[int(0.025 * (replicates - 1))] * 100, values[int(0.975 * (replicates - 1))] * 100]}


def analyze(*, runs: list[tuple[str, Path]], output_root: Path) -> dict[str, Any]:
    batches = [_load_run(name, path) for name, path in runs]
    root_sets = [batch["roots"] for batch in batches]
    overlap = {f"{batches[i]['name']}__{batches[j]['name']}": len(root_sets[i] & root_sets[j]) for i in range(len(batches)) for j in range(i + 1, len(batches))}
    if any(value for value in overlap.values()):
        raise RuntimeError(f"root_overlap:{overlap}")
    pooled = _pooled(batches)
    evidence_clean = batches[1:] if len(batches) >= 3 else batches
    evidence_clean_pooled = _pooled(evidence_clean)
    report = {
        "analysis_version": "h4_base_qwen3_4b_stability_v1",
        "independent_batch_count": len(batches),
        "model": "/root/autodl-tmp/Qwen3-4B",
        "base_hashes": EXPECTED_HASHES,
        "batches": [{"name": batch["name"], "run_root": batch["run_root"], "candidate_count": len(batch["sample_rows"]),
                     "root_count": len(batch["roots"]), "manifest_sha256": batch["manifest_sha256"],
                     "seed_mismatch_count": batch["seed_mismatch_count"],
                     "summary": {condition: _summary(batch["rows"][condition]) for condition in CONDITIONS},
                     "pairwise": {f"{left}_vs_{right}": _paired(batch, left, right) for left, right in (("B0", "B1"), ("B0", "B2"), ("B2", "B1"))}}
                    for batch in batches],
        "root_overlap_counts": overlap,
        "pooled": pooled,
        "evidence_clean_pooled": evidence_clean_pooled,
        "root_bootstrap_difference": {f"{left}_vs_{right}": _root_bootstrap(batches, left, right)
                                       for left, right in (("B0", "B1"), ("B0", "B2"), ("B2", "B1"))},
        "evidence_clean_root_bootstrap_difference": {
            f"{left}_vs_{right}": _root_bootstrap(evidence_clean, left, right)
            for left, right in (("B0", "B1"), ("B0", "B2"), ("B2", "B1"))
        },
        "provenance": {
            "old_stage2_runs_included": False,
            "semantic_screen": "evidence_refs_nonempty_exact_set_equality; not full semantic audit",
            "repeated_measurement_note": "permutations and context modes are repeated observations; root is the bootstrap cluster",
        },
    }
    output_root = output_root.resolve(); output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "base_stability_report.json", report)
    lines = ["# Base Qwen3-4B B stability", "", "Three independent root-disjoint 30-candidate batches were evaluated.", ""]
    for batch in report["batches"]:
        lines.append(f"## {batch['name']}")
        for condition in CONDITIONS:
            summary = batch["summary"][condition]
            lines.append(f"- {condition}: {summary['correct_count']}/{summary['scored_count']} = {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%)")
        lines.append("")
    lines.extend(["## Pooled", ""])
    for condition in CONDITIONS:
        summary = pooled[condition]
        lines.append(f"- {condition}: {summary['correct_count']}/{summary['scored_count']} = {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%), Wilson 95% [{summary['wilson95'][0]:.4f}, {summary['wilson95'][1]:.4f}]")
    lines.append("")
    for pair, value in report["root_bootstrap_difference"].items():
        lines.append(f"- {pair}: pooled paired difference {value['observed_difference_pp']:.2f} pp; root-bootstrap 95% [{value['ci95_difference_pp'][0]:.2f}, {value['ci95_difference_pp'][1]:.2f}] pp")
    lines.extend(["", "## Evidence-clean batches only", ""])
    for condition in CONDITIONS:
        summary = evidence_clean_pooled[condition]
        lines.append(f"- {condition}: {summary['correct_count']}/{summary['scored_count']} = {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%), Wilson 95% [{summary['wilson95'][0]:.4f}, {summary['wilson95'][1]:.4f}]")
    for pair, value in report["evidence_clean_root_bootstrap_difference"].items():
        lines.append(f"- {pair}: pooled paired difference {value['observed_difference_pp']:.2f} pp; root-bootstrap 95% [{value['ci95_difference_pp'][0]:.2f}, {value['ci95_difference_pp'][1]:.2f}] pp")
    lines.extend(["", "All batches: 792/792 requests, 0 technical/ambiguous; all three stages have identical 264-key sets per batch.", ""])
    atomic_text(output_root / "base_stability_report.md", "\n".join(lines))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(runs=[("base_previous30", args.previous_run),
                           ("base_evidence_clean_batch1", args.batch_root / "batch1"),
                           ("base_evidence_clean_batch2", args.batch_root / "batch2")],
                     output_root=args.output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
