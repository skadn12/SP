"""Build immutable stability-test manifests for additional H4 B batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, canonical_json, sha256_file
from .core import build_permutations, load_hard_pool, profile_hard


PROBE_VERSION = "h4_position_permutation_ab_stability_batches_v1"
CURRENT_SOURCE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_batch30_memory_event_k3_probe_v1_live_1")
CURRENT_SAMPLE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_position_permutation_ab_probe_v1_live_4")
INDEPENDENT_AUDIT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_qa_multiroot_k3_t08_semantic_audit_v1_live_3")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _candidate_sha(candidate: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(candidate).encode("utf-8")).hexdigest()


def _lineage(root_dir: Path) -> dict[str, str]:
    pool = root_dir / "candidate_pool.json"
    dialogue = root_dir / "expanded_dialogue_clean.json"
    if not pool.is_file() or not dialogue.is_file():
        raise FileNotFoundError(f"lineage_files_missing:{root_dir}")
    return {
        "candidate_pool_path": str(pool),
        "candidate_pool_sha256": sha256_file(pool),
        "dialogue_path": str(dialogue),
        "dialogue_sha256": sha256_file(dialogue),
    }


def _current_holdout(source_run: Path, reference_probe: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hard, source_report = load_hard_pool(source_run)
    if not source_report["passed"]:
        raise RuntimeError(f"current_source_preflight_failed:{source_report['errors']}")
    sampled_ids = {
        json.loads(line)["candidate_id"]
        for line in (reference_probe / "sample_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    records: list[dict[str, Any]] = []
    for row in hard:
        if row["candidate_id"] in sampled_ids:
            continue
        profile = profile_hard(row)
        records.append({
            "candidate": row["candidate"],
            "candidate_id": row["candidate_id"],
            "root_id": row["root_id"],
            "lineage": row["lineage"],
            "candidate_sha256": row["candidate_sha256"],
            "profile": profile,
            "hard_mechanism": profile["hard_mechanism"],
        })
    if len(records) != len(hard) - len(sampled_ids & {row["candidate_id"] for row in hard}):
        raise RuntimeError("holdout_count_mismatch")
    return sorted(records, key=lambda row: str(row["candidate_id"])), {
        "source_kind": "same_source_holdout",
        "source_run": str(source_run.resolve()),
        "source_report": source_report,
        "excluded_probe": str(reference_probe.resolve()),
        "excluded_candidate_count": len(sampled_ids),
    }


def _independent_batch(audit_run: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_path = audit_run / "semantically_valid_hard_candidates.json"
    if not valid_path.is_file():
        raise FileNotFoundError(f"missing_semantically_valid_candidates:{valid_path}")
    valid = _load(valid_path)
    valid_rows = list(valid.get("candidates") or valid.get("rows") or []) if isinstance(valid, dict) else list(valid)
    audit_manifest = _load(audit_run / "manifest.json")
    input_run = Path(str(audit_manifest.get("input_run") or "")).resolve()
    if not input_run.is_dir():
        raise FileNotFoundError(f"independent_input_run_missing:{input_run}")
    records: list[dict[str, Any]] = []
    for audit_row in valid_rows:
        cid = str(audit_row.get("candidate_id") or "")
        root_id = str(audit_row.get("root_id") or cid.split(":", 1)[0])
        root_dir = input_run / "roots" / root_id
        lineage = _lineage(root_dir)
        pool = _load(Path(lineage["candidate_pool_path"]))
        matches = [row for row in pool if str(row.get("candidate_id")) == cid]
        if len(matches) != 1:
            raise RuntimeError(f"independent_candidate_lineage_not_unique:{cid}:{len(matches)}")
        candidate = dict(matches[0])
        if str(candidate.get("root_id") or root_id) != root_id:
            raise RuntimeError(f"independent_root_mismatch:{cid}")
        difficulty = dict(audit_row.get("difficulty") or candidate.get("difficulty") or {})
        records.append({
            "candidate": candidate,
            "candidate_id": cid,
            "root_id": root_id,
            "lineage": lineage,
            "candidate_sha256": _candidate_sha(candidate),
            "profile": difficulty,
            "hard_mechanism": "external_semantic_hard",
        })
    return sorted(records, key=lambda row: (str(row["root_id"]), str(row["candidate_id"]))), {
        "source_kind": "independent_semantic_hard",
        "source_run": str(audit_run.resolve()),
        "input_run": str(input_run),
        "source_manifest_sha256": sha256_file(audit_run / "manifest.json"),
        "semantic_candidate_file": str(valid_path.resolve()),
        "semantic_candidate_file_sha256": sha256_file(valid_path),
    }


def build(*, batch: str, run_root: Path, source_run: Path, reference_probe: Path, independent_audit: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    if run_root.exists():
        raise RuntimeError(f"run_root_already_exists:{run_root}")
    records, provenance = (
        _current_holdout(source_run, reference_probe)
        if batch == "holdout"
        else _independent_batch(independent_audit)
    )
    if not records:
        raise RuntimeError("empty_stability_batch")
    sample_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    for record in records:
        candidate = record["candidate"]
        profile = record["profile"]
        sample_rows.append({
            "candidate_id": record["candidate_id"],
            "root_id": record["root_id"],
            "question_type": candidate.get("question_type"),
            "generation_source": candidate.get("generation_source"),
            "hard_mechanism": record["hard_mechanism"],
            "historical_full_accuracy": profile.get("historical_full_accuracy", profile.get("full_context_accuracy")),
            "historical_no_accuracy": profile.get("historical_no_accuracy", profile.get("no_context_accuracy")),
            "selection_algorithm": "stability_batch_frozen_v1",
            "lineage": record["lineage"],
            "candidate_sha256": record["candidate_sha256"],
        })
        permutation_rows.extend(build_permutations(record, source_run_id=f"{batch}_stability_v1"))
    source_ids = {row["candidate_id"] for row in records}
    source_roots = {row["root_id"] for row in records}
    if len(source_ids) != len(records):
        raise RuntimeError("duplicate_candidate_ids_in_stability_batch")
    cyclic_count = sum(bool(row["is_cyclic"]) for row in permutation_rows)
    preflight = {
        "probe_version": PROBE_VERSION,
        "batch": batch,
        "provenance": provenance,
        "sample_count": len(sample_rows),
        "root_count": len(source_roots),
        "permutation_count": len(permutation_rows),
        "cyclic_permutation_count": cyclic_count,
        "B0_B1_B2_requests": cyclic_count * 6,
        "solver_generation_calls": 0,
        "api_calls": 0,
        "mainline_untouched": True,
        "historical_runs_untouched": True,
        "question_type": dict(sorted(Counter(str(row["question_type"]) for row in sample_rows).items())),
        "root": dict(sorted(Counter(str(row["root_id"]) for row in sample_rows).items())),
    }
    run_root.mkdir(parents=True, exist_ok=False)
    _write_jsonl(run_root / "sample_manifest.jsonl", sample_rows)
    _write_jsonl(run_root / "permutation_manifest.jsonl", permutation_rows)
    atomic_json(run_root / "preflight_report.json", preflight)
    atomic_json(run_root / "stability_batch_manifest.json", {
        "probe_version": PROBE_VERSION,
        "batch": batch,
        "provenance": provenance,
        "candidate_ids": sorted(source_ids),
        "root_ids": sorted(source_roots),
        "manifest_sha256": sha256_file(run_root / "sample_manifest.jsonl"),
    })
    atomic_json(run_root / "sample_distribution.json", {
        "sample_count": len(sample_rows),
        "question_type": preflight["question_type"],
        "root": preflight["root"],
    })
    atomic_json(run_root / "permutation_distribution.json", {
        "permutation_count": len(permutation_rows),
        "cyclic_count": cyclic_count,
        "non_cyclic_count": len(permutation_rows) - cyclic_count,
        "question_type": dict(sorted(Counter(str(row["question_type"]) for row in permutation_rows).items())),
    })
    for name in ("sample_manifest.jsonl", "permutation_manifest.jsonl", "preflight_report.json", "stability_batch_manifest.json"):
        atomic_text(run_root / f"{name}.sha256", sha256_file(run_root / name) + "\n")
    atomic_text(run_root / "summary.md", f"# H4 stability batch {batch}\n\nFrozen input manifest; no Solver calls made.\n")
    return {"status": "preflight_passed", **preflight, "run_root": str(run_root)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", choices=("holdout", "independent"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=CURRENT_SOURCE)
    parser.add_argument("--reference-probe", type=Path, default=CURRENT_SAMPLE)
    parser.add_argument("--independent-audit", type=Path, default=INDEPENDENT_AUDIT)
    args = parser.parse_args(argv)
    result = build(
        batch=args.batch,
        run_root=args.run_root,
        source_run=args.source_run,
        reference_probe=args.reference_probe,
        independent_audit=args.independent_audit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
