"""Freeze fresh, root-disjoint B0/B1/B2 batches from the Round0 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, canonical_json, sha256_file
from .core import build_permutations, stable_score


SOURCE_RUN = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_k3_difficulty_v1_live_1")
OLD_BASE_RUN = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_position_permutation_ab_new122_30_base_qwen3_4b_final")
TYPE_QUOTAS = {"single_choice": 6, "multi_select": 12, "ordering": 12}
KNOWN_BAD = {
    "cc-train-episode-162896:I001-C-Q02",
    "cc-train-episode-138343:I003-C-Q03",
    "cc-train-episode-239287:I003-C-Q03",
    "cc-train-episode-232414:I002-C-Q04",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _candidate_sha(candidate: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(candidate).encode("utf-8")).hexdigest()


def _old_exclusions() -> tuple[set[str], set[str]]:
    path = OLD_BASE_RUN / "sample_manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ({str(row["candidate_id"]) for row in rows}, {str(row["root_id"]) for row in rows})


def _collect(source_run: Path, excluded_ids: set[str], excluded_roots: set[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats: Counter[str] = Counter()
    for line in (source_run / "difficulty_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = _load_line(line)
        cid = str(row.get("candidate_id") or "")
        root_id = str(row.get("root_id") or "")
        if not cid or cid in excluded_ids or cid in KNOWN_BAD or root_id in excluded_roots:
            stats["historical_or_known_bad_excluded"] += 1
            continue
        if row.get("program_gate_status") != "PROGRAM_PASS":
            stats["non_program_pass_excluded"] += 1
            continue
        if cid in seen:
            raise RuntimeError(f"duplicate_candidate_id:{cid}")
        candidate = dict(row.get("candidate") or {})
        root_id = str(row.get("root_id") or candidate.get("root_id") or "")
        pool_path = Path(str(row.get("candidate_pool_path") or ""))
        dialogue_path = Path(str(row.get("dialogue_path") or ""))
        if not root_id or not pool_path.is_file() or not dialogue_path.is_file():
            raise RuntimeError(f"lineage_missing:{cid}")
        if str(candidate.get("candidate_id") or "") != cid:
            raise RuntimeError(f"candidate_payload_mismatch:{cid}")
        flat_refs = {str(value) for value in candidate.get("evidence_refs") or []}
        grouped_refs = {
            str(value)
            for values in (candidate.get("evidence_groups") or {}).values()
            for value in (values or [])
        }
        if not flat_refs or flat_refs != grouped_refs:
            stats["evidence_alignment_rejected"] += 1
            continue
        seen.add(cid)
        rows.append({
            "candidate": candidate,
            "candidate_id": cid,
            "root_id": root_id,
            "lineage": {
                "candidate_pool_path": str(pool_path),
                "candidate_pool_sha256": str(row.get("candidate_pool_sha256") or sha256_file(pool_path)),
                "dialogue_path": str(dialogue_path),
                "dialogue_sha256": str(row.get("dialogue_sha256") or sha256_file(dialogue_path)),
            },
            "candidate_sha256": _candidate_sha(candidate),
        })
        stats["eligible"] += 1
    return rows, dict(stats)


def _load_line(line: str) -> dict[str, Any]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("difficulty_row_not_object")
    return value


def _select(rows: list[dict[str, Any]], batch: str, used_roots: set[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    chosen_roots = set(used_roots)
    for qtype, quota in TYPE_QUOTAS.items():
        candidates = [row for row in rows if str(row["candidate"].get("question_type")) == qtype and row["root_id"] not in chosen_roots]
        by_root: dict[str, list[dict[str, Any]]] = {}
        for row in candidates:
            by_root.setdefault(row["root_id"], []).append(row)
        representatives = []
        for root_id, root_rows in by_root.items():
            root_rows.sort(key=lambda row: (stable_score("base_b_stability", batch, qtype, root_id, row["candidate_id"]), row["candidate_id"]))
            representatives.append(root_rows[0])
        representatives.sort(key=lambda row: (stable_score("base_b_stability", batch, qtype, row["root_id"]), row["root_id"]))
        if len(representatives) < quota:
            raise RuntimeError(f"type_quota_unmet:{batch}:{qtype}:{len(representatives)}/{quota}")
        selected.extend(representatives[:quota])
        chosen_roots.update(row["root_id"] for row in representatives[:quota])
    if len(selected) != 30 or len({row["candidate_id"] for row in selected}) != 30 or len({row["root_id"] for row in selected}) != 30:
        raise RuntimeError(f"selection_postcondition_failed:{batch}")
    return selected


def _build_batch(rows: list[dict[str, Any]], batch: str, run_root: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    sample_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    for row in rows:
        candidate = row["candidate"]
        sample_rows.append({
            "candidate_id": row["candidate_id"], "root_id": row["root_id"],
            "question_type": candidate.get("question_type"),
            "generation_source": candidate.get("generation_source"),
            "hard_mechanism": "generation_only_program_pass",
            "selection_algorithm": "base_b_stability_root_disjoint_v1",
            "candidate_source": "round0_program_pass_pool",
            "lineage": row["lineage"], "candidate_sha256": row["candidate_sha256"],
        })
        permutation_rows.extend(build_permutations(row, source_run_id=f"base_b_stability_{batch}_v1"))
    cyclic_count = sum(bool(row["is_cyclic"]) for row in permutation_rows)
    preflight = {
        "probe_version": "h4_position_permutation_ab_base_stability_v1",
        "batch": batch, "source_run": str(SOURCE_RUN.resolve()), "provenance": dict(provenance),
        "sample_count": len(sample_rows), "root_count": len({row["root_id"] for row in sample_rows}),
        "permutation_count": len(permutation_rows), "cyclic_permutation_count": cyclic_count,
        "B0_B1_B2_requests": cyclic_count * 6, "solver_generation_calls": 0, "api_calls": 0,
        "question_type": dict(sorted(Counter(str(row["question_type"]) for row in sample_rows).items())),
        "root": dict(sorted(Counter(str(row["root_id"]) for row in sample_rows).items())),
        "semantic_screen_applied": False, "known_bad_ids_excluded": sorted(KNOWN_BAD),
        "mainline_untouched": True, "historical_runs_untouched": True,
    }
    run_root.mkdir(parents=True, exist_ok=False)
    for directory in ("raw_prompts", "raw_responses", "results", "logs"):
        (run_root / directory).mkdir()
    _write_jsonl(run_root / "sample_manifest.jsonl", sample_rows)
    _write_jsonl(run_root / "permutation_manifest.jsonl", permutation_rows)
    atomic_json(run_root / "preflight_report.json", preflight)
    atomic_json(run_root / "stability_batch_manifest.json", {
        "probe_version": preflight["probe_version"], "batch": batch,
        "candidate_ids": [row["candidate_id"] for row in sample_rows],
        "root_ids": sorted({row["root_id"] for row in sample_rows}),
        "manifest_sha256": sha256_file(run_root / "sample_manifest.jsonl"),
        "permutation_manifest_sha256": sha256_file(run_root / "permutation_manifest.jsonl"),
    })
    atomic_json(run_root / "sample_distribution.json", {"sample_count": len(sample_rows), "question_type": preflight["question_type"], "root": preflight["root"]})
    atomic_json(run_root / "permutation_distribution.json", {"permutation_count": len(permutation_rows), "cyclic_count": cyclic_count, "non_cyclic_count": len(permutation_rows) - cyclic_count})
    for name in ("sample_manifest.jsonl", "permutation_manifest.jsonl", "preflight_report.json", "stability_batch_manifest.json"):
        atomic_text(run_root / f"{name}.sha256", sha256_file(run_root / name) + "\n")
    atomic_text(run_root / "summary.md", f"# Base B stability batch {batch}\n\nFrozen Round0 PROGRAM_PASS input; no Solver calls in preflight.\n")
    return preflight


def build(source_run: Path, output_root: Path) -> dict[str, Any]:
    source_run = source_run.resolve(); output_root = output_root.resolve()
    if output_root.exists():
        raise RuntimeError(f"output_root_already_exists:{output_root}")
    excluded, excluded_roots = _old_exclusions()
    rows, collection_stats = _collect(source_run, excluded, excluded_roots)
    if len(rows) < 60:
        raise RuntimeError(f"candidate_pool_under_60:{len(rows)}")
    used_roots: set[str] = set()
    reports: dict[str, Any] = {}
    for batch in ("batch1", "batch2"):
        selected = _select(rows, batch, used_roots)
        reports[batch] = _build_batch(selected, batch, output_root / batch, {
            "candidate_pool_count_after_exclusions": len(rows),
            "excluded_old_base_candidate_count": len(excluded),
            "excluded_old_base_root_count": len(excluded_roots),
            "excluded_known_bad_count": len(KNOWN_BAD),
            "collection_stats": collection_stats,
            "evidence_alignment_gate": "nonempty_exact_set_equality",
            "root_disjoint_from_previous_and_other_batch": True,
        })
        used_roots.update(row["root_id"] for row in selected)
    atomic_json(output_root / "batch_manifest.json", {"status": "preflight_passed", "source_run": str(source_run), "batches": reports, "used_root_count": len(used_roots)})
    return {"status": "preflight_passed", "output_root": str(output_root), "batches": reports}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.source_run, args.output_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
