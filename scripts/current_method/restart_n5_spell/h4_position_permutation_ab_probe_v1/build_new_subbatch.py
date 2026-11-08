"""Freeze a standalone 30-item subbatch from the new 60-item run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, sha256_file
from .core import build_permutations, stable_score


TYPE_QUOTAS = {"single_choice": 6, "multi_select": 12, "ordering": 12}
SOURCE_RUN_ID = "new122_generation_30_from_60_v1"
SELECTION_ALGORITHM = "new_generation_subbatch_30_balanced_type_root_v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _score(row: Mapping[str, Any]) -> tuple[int, str]:
    return (stable_score(SOURCE_RUN_ID, row["root_id"], row["candidate_id"], SELECTION_ALGORITHM), str(row["candidate_id"]))


def _select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    root_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    # One item per root per type keeps this standalone subbatch broad and
    # prevents any root from dominating the 30-item result.
    for qtype, quota in TYPE_QUOTAS.items():
        options = [row for row in rows if row["question_type"] == qtype]
        roots = sorted({row["root_id"] for row in options}, key=lambda root: stable_score(SOURCE_RUN_ID, qtype, root, SELECTION_ALGORITHM))
        by_root = {root: sorted((row for row in options if row["root_id"] == root), key=_score) for root in roots}
        for root in roots:
            if type_counts[qtype] >= quota:
                break
            candidates = [row for row in by_root[root] if row["candidate_id"] not in selected_ids and root_counts[root] < 2]
            if not candidates:
                continue
            chosen = candidates[0]
            selected.append(chosen)
            selected_ids.add(chosen["candidate_id"])
            root_counts[root] += 1
            type_counts[qtype] += 1
    if dict(type_counts) != TYPE_QUOTAS:
        raise RuntimeError(f"sample_quota_unmet:{dict(type_counts)}")
    if len(selected) != 30 or len(root_counts) < 20 or max(root_counts.values()) > 2:
        raise RuntimeError(f"sample_postcondition_failed:{len(selected)}:{len(root_counts)}:{dict(root_counts)}")
    return sorted(selected, key=lambda row: (tuple(TYPE_QUOTAS).index(row["question_type"]), row["root_id"], row["candidate_id"]))


def build(source_run: Path, run_root: Path) -> dict[str, Any]:
    source_run, run_root = source_run.resolve(), run_root.resolve()
    if run_root.exists():
        raise RuntimeError(f"run_root_already_exists:{run_root}")
    source_manifest = source_run / "sample_manifest.jsonl"
    source_preflight = source_run / "preflight_report.json"
    if not source_manifest.is_file() or not source_preflight.is_file():
        raise FileNotFoundError("new_60_manifest_missing")
    rows = [_load_line(line) for line in source_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [dict(row) for row in rows]
    selected = _select(rows)
    sample_rows = []
    permutation_rows = []
    for row in selected:
        sample_rows.append({
            **row,
            "selection_algorithm": SELECTION_ALGORITHM,
            "candidate_source": "new_122_generation_only_subbatch_from_new60",
            "parent_run": str(source_run),
            "stable_hash_score": _score(row)[0],
        })
        # build_permutations only needs candidate content; reconstruct it from
        # the immutable lineage pool referenced by the parent manifest.
        pool = _load(Path(str(row["lineage"]["candidate_pool_path"])))
        matches = [candidate for candidate in pool if str(candidate.get("candidate_id")) == str(row["candidate_id"])]
        if len(matches) != 1:
            raise RuntimeError(f"candidate_lineage_not_unique:{row['candidate_id']}:{len(matches)}")
        permutation_rows.extend(build_permutations({
            "candidate": matches[0],
            "candidate_id": row["candidate_id"],
            "root_id": row["root_id"],
        }, source_run_id=SOURCE_RUN_ID))
    parent_preflight = _load(source_preflight)
    cyclic_count = sum(bool(row["is_cyclic"]) for row in permutation_rows)
    preflight = {
        "probe_version": "h4_position_permutation_ab_new_generation_subbatch_v1",
        "source_run": str(source_run),
        "parent_run": str(source_run),
        "parent_manifest_sha256": sha256_file(source_manifest),
        "parent_preflight": {
            "source_run": parent_preflight.get("source_run"),
            "provenance": parent_preflight.get("provenance"),
            "screening_note": parent_preflight.get("provenance", {}).get("screening_note"),
        },
        "sample_count": len(sample_rows),
        "root_count": len({row["root_id"] for row in sample_rows}),
        "permutation_count": len(permutation_rows),
        "cyclic_permutation_count": cyclic_count,
        "B0_B1_B2_requests": cyclic_count * 6,
        "solver_generation_calls": 0,
        "api_calls": 0,
        "generation_hard_cap": 2232,
        "mainline_untouched": True,
        "historical_runs_untouched": True,
        "question_type": dict(sorted(Counter(row["question_type"] for row in sample_rows).items())),
        "generation_source": dict(sorted(Counter(row["generation_source"] for row in sample_rows).items())),
        "root": dict(sorted(Counter(row["root_id"] for row in sample_rows).items())),
    }
    run_root.mkdir(parents=True, exist_ok=False)
    for directory in ("raw_prompts", "raw_responses", "results", "logs"):
        (run_root / directory).mkdir()
    _write_jsonl(run_root / "sample_manifest.jsonl", sample_rows)
    _write_jsonl(run_root / "permutation_manifest.jsonl", permutation_rows)
    atomic_json(run_root / "preflight_report.json", preflight)
    atomic_json(run_root / "stability_batch_manifest.json", {
        "probe_version": preflight["probe_version"],
        "batch": "new_122_generation_only_30_standalone",
        "parent_run": str(source_run),
        "candidate_ids": [row["candidate_id"] for row in sample_rows],
        "root_ids": sorted({row["root_id"] for row in sample_rows}),
        "manifest_sha256": sha256_file(run_root / "sample_manifest.jsonl"),
    })
    atomic_json(run_root / "sample_distribution.json", {
        "sample_count": len(sample_rows),
        "question_type": preflight["question_type"],
        "generation_source": preflight["generation_source"],
        "root": preflight["root"],
    })
    atomic_json(run_root / "permutation_distribution.json", {
        "permutation_count": len(permutation_rows),
        "cyclic_count": cyclic_count,
        "non_cyclic_count": len(permutation_rows) - cyclic_count,
        "question_type": dict(sorted(Counter(row["question_type"] for row in permutation_rows).items())),
    })
    for name in ("sample_manifest.jsonl", "permutation_manifest.jsonl", "preflight_report.json", "stability_batch_manifest.json"):
        atomic_text(run_root / f"{name}.sha256", sha256_file(run_root / name) + "\n")
    atomic_text(run_root / "summary.md", "# H4 standalone B experiment: new generation-only 30\n\nFrozen only from the new 60-item parent run; no other data mixed.\n")
    return {"status": "preflight_passed", "run_root": str(run_root), **preflight}


def _load_line(line: str) -> dict[str, Any]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("manifest_row_not_object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.source_run, args.run_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
