"""Offline preflight for the isolated H4 position permutation probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes
from .core import (
    PERMUTATION_ALGORITHM, PROBE_VERSION, SAMPLE_ALGORITHM, SOURCE_RUN,
    build_permutations, load_hard_pool, profile_hard, stratified_sample,
)


DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_position_permutation_ab_probe_v1_live_1"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight",), default="preflight")
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args(argv)
    args._argv = list(argv or [])
    return args


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_root.exists():
        raise RuntimeError(f"run_root_already_exists:{args.run_root}")
    rows, source_report = load_hard_pool(args.source_run)
    if not source_report["passed"]:
        args.run_root.mkdir(parents=True, exist_ok=False)
        atomic_json(args.run_root / "preflight_blocker_report.json", source_report)
        print(json.dumps({"status": "blocked", "report": str(args.run_root / "preflight_blocker_report.json"), "errors": source_report["errors"]}, ensure_ascii=False, indent=2))
        return 2
    profiles = {row["candidate_id"]: profile_hard(row) for row in rows}
    sample = stratified_sample(rows, profiles, source_run_id=args.source_run.name)
    permutations = [perm for row in sample for perm in build_permutations(row, source_run_id=args.source_run.name)]
    args.run_root.mkdir(parents=True, exist_ok=False)
    args.run_root.joinpath("raw_prompts").mkdir()
    args.run_root.joinpath("raw_responses").mkdir()
    args.run_root.joinpath("logs").mkdir()
    atomic_json(args.run_root / "preflight_report.json", {
        "probe_version": PROBE_VERSION,
        "source_run": str(args.source_run.resolve()),
        "source_report": source_report,
        "hard_pool_pass": True,
        "sample_count": len(sample),
        "permutation_count": len(permutations),
        "solver_generation_calls": 0,
        "api_calls": 0,
        "model_len": 16_384,
        "generation_hard_cap": 2_232,
        "mainline_untouched": True,
        "historical_runs_untouched": True,
        "argv": args._argv,
    })
    hard_manifest_rows = [
        {"manifest_version": "h4_position_hard_source_manifest_v1",
         "probe_version": PROBE_VERSION,
         "source_run": str(args.source_run.resolve()),
         "source_run_report": source_report if index == 0 else None,
         "candidate_id": row["candidate_id"], "root_id": row["root_id"],
         "question_type": row["candidate"].get("question_type"),
         "generation_source": row["candidate"].get("generation_source"),
         "lineage": row["lineage"], "candidate_sha256": row["candidate_sha256"]}
        for index, row in enumerate(rows)
    ]
    _write_jsonl(args.run_root / "hard_source_manifest.jsonl", hard_manifest_rows)
    profile_rows = [profiles[row["candidate_id"]] for row in rows]
    _write_jsonl(args.run_root / "hard_mechanism_profile.jsonl", profile_rows)
    atomic_json(args.run_root / "hard_mechanism_summary.json", {
        "profile_count": len(profile_rows),
        "by_mechanism": dict(sorted(Counter(row["hard_mechanism"] for row in profile_rows).items())),
        "by_question_type": {key: dict(sorted(Counter(row["hard_mechanism"] for row in profile_rows if row["question_type"] == key).items())) for key in ("single_choice", "multi_select", "ordering")},
        "by_generation_source": {key: dict(sorted(Counter(row["hard_mechanism"] for row in profile_rows if row["generation_source"] == key).items())) for key in ("contract_guided", "free_exploration")},
        "by_root": {key: dict(sorted(Counter(row["hard_mechanism"] for row in profile_rows if row["root_id"] == key).items())) for key in sorted({row["root_id"] for row in profile_rows})},
        "variance_tolerance": 1e-9,
    })
    sample_rows = []
    for row in sample:
        profile = profiles[row["candidate_id"]]
        sample_rows.append({
            "candidate_id": row["candidate_id"], "root_id": row["root_id"],
            "question_type": row["candidate"].get("question_type"),
            "generation_source": row["candidate"].get("generation_source"),
            "hard_mechanism": profile["hard_mechanism"],
            "historical_full_accuracy": profile["historical_full_accuracy"],
            "historical_no_accuracy": profile["historical_no_accuracy"],
            "stable_hash_score": row["stable_hash_score"],
            "selection_algorithm": SAMPLE_ALGORITHM,
            "lineage": row["lineage"],
            "candidate_sha256": row["candidate_sha256"],
        })
    _write_jsonl(args.run_root / "sample_manifest.jsonl", sample_rows)
    atomic_json(args.run_root / "sample_distribution.json", {
        "sample_count": len(sample_rows),
        "question_type": dict(sorted(Counter(row["question_type"] for row in sample_rows).items())),
        "generation_source": dict(sorted(Counter(row["generation_source"] for row in sample_rows).items())),
        "root": dict(sorted(Counter(row["root_id"] for row in sample_rows).items())),
        "hard_mechanism": dict(sorted(Counter(row["hard_mechanism"] for row in sample_rows).items())),
    })
    _write_jsonl(args.run_root / "permutation_manifest.jsonl", permutations)
    atomic_json(args.run_root / "permutation_distribution.json", {
        "permutation_count": len(permutations),
        "by_question_type": dict(sorted(Counter(row["question_type"] for row in permutations).items())),
        "cyclic_count": sum(bool(row["is_cyclic"]) for row in permutations),
        "non_cyclic_count": sum(not bool(row["is_cyclic"]) for row in permutations),
    })
    for filename in ("hard_source_manifest.jsonl", "hard_mechanism_profile.jsonl", "hard_mechanism_summary.json", "sample_manifest.jsonl", "sample_distribution.json", "permutation_manifest.jsonl", "permutation_distribution.json"):
        path = args.run_root / filename
        atomic_text(args.run_root / f"{filename}.sha256", hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    atomic_text(args.run_root / "summary.md", "# H4 position permutation A/B probe\n\nPreflight passed. No Solver or API calls were made.\n")
    print(json.dumps({"status": "preflight_passed", "run_root": str(args.run_root.resolve()), "hard_count": len(rows), "sample_count": len(sample_rows), "permutation_count": len(permutations), "solver_generation_calls": 0, "api_calls": 0}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
