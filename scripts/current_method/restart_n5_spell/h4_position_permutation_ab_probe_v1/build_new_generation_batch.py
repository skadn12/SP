"""Freeze a deterministic 60-candidate B experiment from the new QA batch.

This builder deliberately does not apply the historical difficulty or semantic
admission filters.  The resulting run is therefore a direct test of the new
generation-only batch and is kept in a separate namespace from the historical
30-candidate probe and its stability batches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, canonical_json, sha256_file
from .core import PROBE_VERSION, build_permutations, stable_score


DEFAULT_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_unexpanded122_qa_generation_only_v1"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_position_permutation_ab_new122_60_v1"
)
SAMPLE_ALGORITHM = "new_generation_batch_60_balanced_type_root_v1"
SOURCE_RUN_ID = "new122_generation_60_v1"
TYPE_QUOTAS = {"single_choice": 12, "multi_select": 24, "ordering": 24}
QUESTION_TYPES = tuple(TYPE_QUOTAS)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _candidate_sha(candidate: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(candidate).encode("utf-8")).hexdigest()


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _validate_candidate(candidate: Mapping[str, Any], root_dir: Path, root_id: str) -> list[str]:
    errors: list[str] = []
    cid = str(candidate.get("candidate_id") or "")
    if not cid:
        return ["candidate_id_missing"]
    if str(candidate.get("root_id") or root_id) != root_id:
        errors.append("root_id_mismatch")
    if not candidate.get("schema_valid") or candidate.get("schema_errors"):
        errors.append("schema_not_valid")
    qtype = str(candidate.get("question_type") or "")
    if qtype not in TYPE_QUOTAS:
        errors.append("unsupported_question_type")
        return errors
    if qtype == "single_choice":
        values = candidate.get("options") or {}
        if set(values) != {"A", "B", "C", "D"} or str(candidate.get("correct_option") or "") not in values:
            errors.append("single_choice_contract_invalid")
        texts = [values.get(key) for key in ("A", "B", "C", "D")]
    elif qtype == "multi_select":
        values = candidate.get("options") or {}
        gold = candidate.get("gold") or []
        if set(values) != {"A", "B", "C", "D", "E"} or not gold or any(str(key) not in values for key in gold):
            errors.append("multi_select_contract_invalid")
        texts = [values.get(key) for key in ("A", "B", "C", "D", "E")]
    else:
        values = candidate.get("items") or {}
        gold = candidate.get("gold") or []
        if set(values) != {"I1", "I2", "I3", "I4"} or set(map(str, gold)) != {"I1", "I2", "I3", "I4"}:
            errors.append("ordering_contract_invalid")
        texts = [values.get(key) for key in ("I1", "I2", "I3", "I4")]
    normalized = [_norm(text) for text in texts]
    if any(not text for text in normalized) or len(set(normalized)) != len(normalized):
        errors.append("display_texts_not_unique")
    if not (root_dir / "expanded_dialogue_clean.json").is_file():
        errors.append("dialogue_missing")
    return sorted(set(errors))


def _collect(source_run: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    root_dirs = sorted(source_run.glob("shards/*/roots/*"))
    for root_dir in root_dirs:
        pool_path = root_dir / "candidate_pool.json"
        dialogue_path = root_dir / "expanded_dialogue_clean.json"
        if not pool_path.is_file():
            continue
        root_id = root_dir.name
        try:
            pool = _load(pool_path)
        except Exception as exc:
            errors.append(f"{root_id}:pool_load:{type(exc).__name__}:{exc}")
            continue
        if not isinstance(pool, list):
            errors.append(f"{root_id}:pool_not_list")
            continue
        for index, candidate in enumerate(pool):
            if not isinstance(candidate, Mapping):
                errors.append(f"{root_id}:candidate_{index}:not_object")
                continue
            candidate = dict(candidate)
            cid = str(candidate.get("candidate_id") or "")
            validation = _validate_candidate(candidate, root_dir, root_id)
            if validation:
                errors.extend(f"{cid or root_id}:" + error for error in validation)
                continue
            if cid in seen:
                errors.append(f"duplicate_candidate_id:{cid}")
                continue
            seen.add(cid)
            rows.append({
                "candidate": candidate,
                "candidate_id": cid,
                "root_id": root_id,
                "lineage": {
                    "candidate_pool_path": str(pool_path),
                    "candidate_pool_sha256": sha256_file(pool_path),
                    "dialogue_path": str(dialogue_path),
                    "dialogue_sha256": sha256_file(dialogue_path),
                    "source_root_dir": str(root_dir),
                },
                "candidate_sha256": _candidate_sha(candidate),
            })
    observed = {
        "root_count": len({row["root_id"] for row in rows}),
        "candidate_count": len(rows),
        "question_type": dict(sorted(Counter(str(row["candidate"].get("question_type")) for row in rows).items())),
        "generation_source": dict(sorted(Counter(str(row["candidate"].get("generation_source")) for row in rows).items())),
        "validation_errors": sorted(set(errors)),
    }
    return rows, observed


def _score(row: Mapping[str, Any]) -> tuple[int, str]:
    return (stable_score(SOURCE_RUN_ID, row["root_id"], row["candidate_id"], SAMPLE_ALGORITHM), str(row["candidate_id"]))


def _select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type_root: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_type_root[str(row["candidate"].get("question_type"))][row["root_id"]].append(row)
    for qtype in QUESTION_TYPES:
        for root in by_type_root[qtype]:
            by_type_root[qtype][root].sort(key=_score)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    root_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()

    # Take at most one item per root in each type first.  With the new batch's
    # 101 roots this gives broad root coverage while preserving exact quotas.
    for qtype in QUESTION_TYPES:
        roots = sorted(by_type_root[qtype], key=lambda root: stable_score(SOURCE_RUN_ID, qtype, root, SAMPLE_ALGORITHM))
        for root in roots:
            if type_counts[qtype] >= TYPE_QUOTAS[qtype]:
                break
            options = [row for row in by_type_root[qtype][root] if row["candidate_id"] not in selected_ids and root_counts[root] < 3]
            if not options:
                continue
            chosen = options[0]
            selected.append(chosen)
            selected_ids.add(chosen["candidate_id"])
            root_counts[root] += 1
            type_counts[qtype] += 1

    # Fill any quota left by the diversity pass, still enforcing the root cap.
    while any(type_counts[qtype] < TYPE_QUOTAS[qtype] for qtype in QUESTION_TYPES):
        options = [
            row for row in rows
            if row["candidate_id"] not in selected_ids
            and type_counts[str(row["candidate"].get("question_type"))] < TYPE_QUOTAS[str(row["candidate"].get("question_type"))]
            and root_counts[row["root_id"]] < 3
        ]
        if not options:
            raise RuntimeError(f"sample_quota_unmet:{dict(type_counts)}")
        options.sort(key=_score)
        chosen = options[0]
        qtype = str(chosen["candidate"].get("question_type"))
        selected.append(chosen)
        selected_ids.add(chosen["candidate_id"])
        root_counts[chosen["root_id"]] += 1
        type_counts[qtype] += 1
    if dict(type_counts) != TYPE_QUOTAS or len(selected) != 60:
        raise RuntimeError(f"sample_postcondition_failed:{dict(type_counts)}:{len(selected)}")
    if len(root_counts) < 20 or max(root_counts.values()) > 3:
        raise RuntimeError(f"root_diversity_postcondition_failed:{len(root_counts)}:{dict(root_counts)}")
    return sorted(selected, key=lambda row: (QUESTION_TYPES.index(str(row["candidate"].get("question_type"))), row["root_id"], row["candidate_id"]))


def build(source_run: Path, run_root: Path) -> dict[str, Any]:
    source_run = source_run.resolve()
    run_root = run_root.resolve()
    if run_root.exists():
        raise RuntimeError(f"run_root_already_exists:{run_root}")
    rows, observed = _collect(source_run)
    if observed["validation_errors"]:
        raise RuntimeError(f"input_validation_failed:{observed['validation_errors'][:8]}")
    sample = _select(rows)
    sample_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    for row in sample:
        candidate = row["candidate"]
        sample_rows.append({
            "candidate_id": row["candidate_id"],
            "root_id": row["root_id"],
            "question_type": candidate.get("question_type"),
            "generation_source": candidate.get("generation_source"),
            "hard_mechanism": "generation_only_unclassified",
            "selection_algorithm": SAMPLE_ALGORITHM,
            "candidate_source": "new_122_generation_only",
            "lineage": row["lineage"],
            "candidate_sha256": row["candidate_sha256"],
            "stable_hash_score": _score(row)[0],
        })
        permutation_rows.extend(build_permutations(row, source_run_id=SOURCE_RUN_ID))

    generation_manifest = source_run / "manifest.json"
    generation_report = source_run / "generation_only_report.json"
    report = _load(generation_report)
    provenance = {
        "source_run": str(source_run),
        "source_manifest_sha256": sha256_file(generation_manifest),
        "generation_only_report_sha256": sha256_file(generation_report),
        "qa_probe_version": report.get("probe_version"),
        "qa_status": report.get("status"),
        "qa_root_count": report.get("root_count"),
        "qa_candidate_count": report.get("valid_candidate_count", report.get("candidate_count")),
        "qa_challenger_generation_calls": report.get("challenger_generation_calls"),
        "qa_solver_calls": report.get("solver_calls"),
        "semantic_admission_applied": report.get("semantic_admission_applied"),
        "difficulty_filter_applied": report.get("difficulty_filter_applied"),
        "screening_note": "structural generation-only pool; no semantic or difficulty admission",
    }
    cyclic_count = sum(bool(row["is_cyclic"]) for row in permutation_rows)
    run_root.mkdir(parents=True, exist_ok=False)
    for name in ("raw_prompts", "raw_responses", "results", "logs"):
        (run_root / name).mkdir()
    preflight = {
        "probe_version": "h4_position_permutation_ab_new_generation_batch_v1",
        "source_run": str(source_run),
        "provenance": provenance,
        "observed_input": observed,
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
        "question_type": dict(sorted(Counter(str(row["question_type"]) for row in sample_rows).items())),
        "generation_source": dict(sorted(Counter(str(row["generation_source"]) for row in sample_rows).items())),
        "root": dict(sorted(Counter(str(row["root_id"]) for row in sample_rows).items())),
    }
    _write_jsonl(run_root / "sample_manifest.jsonl", sample_rows)
    _write_jsonl(run_root / "permutation_manifest.jsonl", permutation_rows)
    atomic_json(run_root / "preflight_report.json", preflight)
    atomic_json(run_root / "stability_batch_manifest.json", {
        "probe_version": preflight["probe_version"],
        "batch": "new_122_generation_only_60",
        "provenance": provenance,
        "candidate_ids": [row["candidate_id"] for row in sample_rows],
        "root_ids": sorted({row["root_id"] for row in sample_rows}),
        "manifest_sha256": sha256_file(run_root / "sample_manifest.jsonl"),
    })
    atomic_json(run_root / "sample_distribution.json", {
        "sample_count": len(sample_rows),
        "question_type": preflight["question_type"],
        "generation_source": preflight["generation_source"],
        "root": preflight["root"],
        "hard_mechanism": {"generation_only_unclassified": len(sample_rows)},
    })
    atomic_json(run_root / "permutation_distribution.json", {
        "permutation_count": len(permutation_rows),
        "cyclic_count": cyclic_count,
        "non_cyclic_count": len(permutation_rows) - cyclic_count,
        "question_type": dict(sorted(Counter(str(row["question_type"]) for row in permutation_rows).items())),
    })
    for name in ("sample_manifest.jsonl", "permutation_manifest.jsonl", "preflight_report.json", "stability_batch_manifest.json"):
        atomic_text(run_root / f"{name}.sha256", sha256_file(run_root / name) + "\n")
    atomic_text(run_root / "summary.md", "# H4 B experiment: new 122 generation-only batch\n\nFrozen 60-candidate input; no Solver calls made during preflight.\n")
    return {"status": "preflight_passed", "run_root": str(run_root), **preflight}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.source_run, args.run_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
