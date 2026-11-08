"""Integer-exact difficulty buckets, position diagnostics, and final manifests."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core import sha256_file
from . import PROBE_VERSION, QUESTION_KEYS, ROLLOUTS
from .gate import CONTEXT_OVERFLOW, PROGRAM_PASS, PROGRAM_REJECT
from .io_utils import (
    load_json,
    load_jsonl,
    verify_sha_sidecar,
    write_once_json,
    write_once_jsonl,
    write_once_text,
)
from .solver import candidate_result_path


DIFFICULTY_BUCKETS = (
    "technical_excluded",
    "full_zero_review",
    "full_context_worse",
    "hard_candidate",
    "full_context_resolved",
    "easy",
)
POSITION_MECHANISMS = ("none", "sampling_only", "position_only", "mixed", "unknown")


def classify_difficulty_counts(
    *,
    full_correct: int,
    full_total: int,
    no_correct: int,
    no_total: int,
    technical: bool = False,
) -> str:
    """Apply the mutually exclusive priority using only integer arithmetic."""
    if technical or full_total <= 0 or no_total <= 0:
        return "technical_excluded"
    if full_correct == 0:
        return "full_zero_review"
    # Compare full_correct/full_total < no_correct/no_total exactly.
    if full_correct * no_total < no_correct * full_total:
        return "full_context_worse"
    if (
        0 < full_correct < full_total
        and no_correct < no_total
        and full_correct * no_total >= no_correct * full_total
    ):
        return "hard_candidate"
    if full_correct == full_total and no_correct < no_total:
        return "full_context_resolved"
    if full_correct == full_total and no_correct == no_total:
        return "easy"
    raise RuntimeError(
        "non_exhaustive_difficulty_state:"
        f"{full_correct}/{full_total}:{no_correct}/{no_total}"
    )


def _prediction_key(value: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in value)


def position_diagnostics(
    requests: Iterable[Mapping[str, Any]],
    *,
    question_type: str,
    context_mode: str,
    position_count: int | None = None,
) -> dict[str, Any]:
    selected = [row for row in requests if row.get("context_mode") == context_mode]
    expected_positions = int(position_count or len(QUESTION_KEYS[question_type]))
    by_position: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_position[str(row["variant_id"])].append(row)
    expected_variants = [f"shift_{index}" for index in range(expected_positions)]
    complete = (
        set(by_position) == set(expected_variants)
        and all(len(by_position[key]) == ROLLOUTS for key in expected_variants)
        and all(
            {int(row["slot_id"]) for row in by_position[key]} == set(range(ROLLOUTS))
            for key in expected_variants
        )
        and all(bool(row.get("scored")) for row in selected)
    )
    if not complete:
        partial_predictions = {
            variant_id: [
                list(row.get("canonical_prediction") or [])
                for row in sorted(rows, key=lambda row: int(row["slot_id"]))
                if row.get("scored")
            ]
            for variant_id, rows in sorted(by_position.items())
        }
        return {
            "context_mode": context_mode,
            "technical_complete": False,
            "correct_count": sum(row.get("correct") is True for row in selected),
            "total_count": expected_positions * ROLLOUTS,
            "accuracy": None,
            "position_accuracy": {},
            "best_positions": [],
            "worst_positions": [],
            "canonical_predictions": partial_predictions,
            "canonical_flip_rate": None,
            "within_position_uncertainty": None,
            "between_position_sensitivity": None,
            "position_mechanism": "unknown",
        }
    position_correct: dict[str, int] = {}
    predictions: dict[str, list[list[str]]] = {}
    within = False
    for variant_id in expected_variants:
        rows = sorted(by_position[variant_id], key=lambda row: int(row["slot_id"]))
        position_correct[variant_id] = sum(row.get("correct") is True for row in rows)
        predictions[variant_id] = [list(row.get("canonical_prediction") or []) for row in rows]
        within = within or len({_prediction_key(value) for value in predictions[variant_id]}) > 1
    best_score = max(position_correct.values())
    worst_score = min(position_correct.values())
    between = len(set(position_correct.values())) > 1
    mechanism = (
        "mixed" if between and within
        else "position_only" if between
        else "sampling_only" if within
        else "none"
    )
    baseline = predictions["shift_0"]
    comparisons = flips = 0
    for variant_id in expected_variants[1:]:
        for slot_id in range(ROLLOUTS):
            comparisons += 1
            flips += _prediction_key(predictions[variant_id][slot_id]) != _prediction_key(
                baseline[slot_id]
            )
    correct = sum(position_correct.values())
    total = expected_positions * ROLLOUTS
    return {
        "context_mode": context_mode,
        "technical_complete": True,
        "correct_count": correct,
        "total_count": total,
        "accuracy": correct / total,
        "position_correct_counts": position_correct,
        "position_accuracy": {
            key: value / ROLLOUTS for key, value in position_correct.items()
        },
        "best_positions": sorted(
            key for key, value in position_correct.items() if value == best_score
        ),
        "worst_positions": sorted(
            key for key, value in position_correct.items() if value == worst_score
        ),
        "canonical_predictions": predictions,
        "canonical_flip_count": flips,
        "canonical_flip_comparison_count": comparisons,
        "canonical_flip_rate": flips / comparisons if comparisons else 0.0,
        "within_position_uncertainty": within,
        "between_position_sensitivity": between,
        "position_mechanism": mechanism,
    }


def summarize_candidate_result(result: Mapping[str, Any]) -> dict[str, Any]:
    qtype = str(result["question_type"])
    requests = list(result.get("requests") or [])
    no_diag = position_diagnostics(
        requests, question_type=qtype, context_mode="no_context",
        position_count=int(result.get("position_count") or 0) or None,
    )
    full_diag = position_diagnostics(
        requests, question_type=qtype, context_mode="full_context",
        position_count=int(result.get("position_count") or 0) or None,
    )
    technical = not (no_diag["technical_complete"] and full_diag["technical_complete"])
    bucket = classify_difficulty_counts(
        full_correct=int(full_diag["correct_count"]),
        full_total=int(full_diag["total_count"]),
        no_correct=int(no_diag["correct_count"]),
        no_total=int(no_diag["total_count"]),
        technical=technical,
    )
    context_helpful = (
        not technical
        and int(full_diag["correct_count"]) * int(no_diag["total_count"])
        > int(no_diag["correct_count"]) * int(full_diag["total_count"])
    )
    return {
        "difficulty_bucket": bucket,
        "technical_exclusion_reason": "solver_request_incomplete_or_unscored" if technical else None,
        "full_context_correct_count": full_diag["correct_count"],
        "full_context_total_count": full_diag["total_count"],
        "full_context_accuracy": full_diag["accuracy"],
        "no_context_correct_count": no_diag["correct_count"],
        "no_context_total_count": no_diag["total_count"],
        "no_context_accuracy": no_diag["accuracy"],
        "context_helpful": context_helpful,
        "position_diagnostics": {
            "no_context": no_diag,
            "full_context": full_diag,
        },
        "position_accuracy": {
            "no_context": no_diag["position_accuracy"],
            "full_context": full_diag["position_accuracy"],
        },
        "best_positions": {
            "no_context": no_diag["best_positions"],
            "full_context": full_diag["best_positions"],
        },
        "worst_positions": {
            "no_context": no_diag["worst_positions"],
            "full_context": full_diag["worst_positions"],
        },
        "canonical_predictions": {
            "no_context": no_diag["canonical_predictions"],
            "full_context": full_diag["canonical_predictions"],
        },
        "canonical_flip_rate": {
            "no_context": no_diag["canonical_flip_rate"],
            "full_context": full_diag["canonical_flip_rate"],
        },
        "within_position_uncertainty": {
            "no_context": no_diag["within_position_uncertainty"],
            "full_context": full_diag["within_position_uncertainty"],
        },
        "between_position_sensitivity": {
            "no_context": no_diag["between_position_sensitivity"],
            "full_context": full_diag["between_position_sensitivity"],
        },
        "position_mechanism": full_diag["position_mechanism"],
        "position_mechanism_by_context": {
            "no_context": no_diag["position_mechanism"],
            "full_context": full_diag["position_mechanism"],
        },
    }


def _technical_gate_summary(gate_row: Mapping[str, Any]) -> dict[str, Any]:
    status = str(gate_row["gate_status"])
    reason = (
        "program_reject" if status == PROGRAM_REJECT
        else "context_overflow" if status == CONTEXT_OVERFLOW
        else "k3_incomplete"
    )
    return {
        "difficulty_bucket": "technical_excluded",
        "technical_exclusion_reason": reason,
        "full_context_correct_count": None,
        "full_context_total_count": None,
        "full_context_accuracy": None,
        "no_context_correct_count": None,
        "no_context_total_count": None,
        "no_context_accuracy": None,
        "context_helpful": False,
        "position_accuracy": {"no_context": {}, "full_context": {}},
        "best_positions": {"no_context": [], "full_context": []},
        "worst_positions": {"no_context": [], "full_context": []},
        "canonical_predictions": {"no_context": {}, "full_context": {}},
        "canonical_flip_rate": {"no_context": None, "full_context": None},
        "within_position_uncertainty": {"no_context": None, "full_context": None},
        "between_position_sensitivity": {"no_context": None, "full_context": None},
        "position_mechanism": "unknown",
        "position_mechanism_by_context": {
            "no_context": "unknown", "full_context": "unknown",
        },
    }


def _group_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    buckets = Counter(str(row["difficulty_bucket"]) for row in values)
    mechanisms = Counter(str(row["position_mechanism"]) for row in values)
    return {
        "candidate_count": len(values),
        "difficulty_buckets": {
            bucket: buckets[bucket] for bucket in DIFFICULTY_BUCKETS
        },
        "hard_candidate_count": sum(
            row["difficulty_bucket"] == "hard_candidate" for row in values
        ),
        "position_mechanisms": {
            mechanism: mechanisms[mechanism] for mechanism in POSITION_MECHANISMS
        },
    }


def build_difficulty_rows(run_root: Path) -> list[dict[str, Any]]:
    gate_path = run_root / "program_gate_results.jsonl"
    split_path = run_root / "root_split_manifest.jsonl"
    verify_sha_sidecar(gate_path)
    verify_sha_sidecar(split_path)
    gate_rows = load_jsonl(gate_path)
    split_rows = load_jsonl(split_path)
    split_by_root = {str(row["root_id"]): str(row["split"]) for row in split_rows}
    pools: dict[str, dict[str, dict[str, Any]]] = {}
    output: list[dict[str, Any]] = []
    for gate_row in gate_rows:
        root_id = str(gate_row["root_id"])
        candidate_id = str(gate_row["candidate_id"])
        if root_id not in pools:
            pool_path = Path(str(gate_row["candidate_pool_path"]))
            if sha256_file(pool_path) != gate_row["candidate_pool_sha256"]:
                raise RuntimeError(f"candidate_pool_drift_during_finalize:{root_id}")
            pools[root_id] = {
                str(row["candidate_id"]): row for row in load_json(pool_path)
            }
        candidate = pools[root_id][candidate_id]
        if gate_row["gate_status"] == PROGRAM_PASS:
            result_path = candidate_result_path(run_root, candidate_id)
            if not result_path.is_file():
                summary = _technical_gate_summary({**gate_row, "gate_status": "K3_INCOMPLETE"})
            else:
                verify_sha_sidecar(result_path)
                summary = summarize_candidate_result(load_json(result_path))
        else:
            summary = _technical_gate_summary(gate_row)
        output.append({
            "difficulty_manifest_version": "h4_round0_b0_k3_difficulty_v1",
            "probe_version": PROBE_VERSION,
            "candidate_id": candidate_id,
            "root_id": root_id,
            "split": split_by_root[root_id],
            "question_type": str(gate_row["question_type"]),
            "generation_source": candidate.get("generation_source"),
            "program_gate_status": gate_row["gate_status"],
            "program_errors": list(gate_row.get("program_errors") or []),
            "candidate_pool_path": gate_row["candidate_pool_path"],
            "candidate_pool_sha256": gate_row["candidate_pool_sha256"],
            "dialogue_path": gate_row["dialogue_path"],
            "dialogue_sha256": gate_row["dialogue_sha256"],
            "candidate": candidate,
            **summary,
        })
    return sorted(output, key=lambda row: (row["root_id"], row["candidate_id"]))


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Round-0 B0 K=3 Difficulty Report",
        "",
        f"- Source roots: `{report['source_root_count']}`",
        f"- QA-ready roots: `{report['candidate_bearing_root_count']}`",
        f"- Candidates: `{report['candidate_count']}`",
        f"- Program pass/reject/overflow: `{report['program_gate_counts']}`",
        f"- Actual Solver requests: `{report['actual_solver_request_count']}`",
        f"- Difficulty buckets: `{report['difficulty_buckets']}`",
        f"- Hard-contributing roots: `{report['hard_contributing_root_count']}`",
        f"- Position mechanisms: `{report['position_mechanisms']}`",
        "",
        "## Result Paths",
        "",
    ]
    lines.extend(f"- `{path}`" for path in report["major_result_paths"])
    lines.extend([
        "",
        "This phase stops at B0 K=3 difficulty and hard manifests. It does not run "
        "semantic audit, Strict/Broad classification, training, B1/B2, or fresh mining.",
        "",
    ])
    return "\n".join(lines)


def finalize(run_root: Path) -> dict[str, Any]:
    inventory_path = run_root / "source_inventory.json"
    preflight_path = run_root / "preflight_report.json"
    batch_path = run_root / "batch_report.json"
    for path in (inventory_path, preflight_path, batch_path):
        verify_sha_sidecar(path)
    inventory = load_json(inventory_path)
    preflight = load_json(preflight_path)
    batch = load_json(batch_path)
    if batch.get("status") != "PASS":
        raise RuntimeError("passing_batch_required_before_finalize")
    rows = build_difficulty_rows(run_root)
    write_once_jsonl(run_root / "difficulty_manifest.jsonl", rows)
    summary = _group_summary(rows)
    by_split = {
        split: _group_summary(row for row in rows if row["split"] == split)
        for split in ("train", "dev", "test")
    }
    by_type = {
        qtype: _group_summary(row for row in rows if row["question_type"] == qtype)
        for qtype in QUESTION_KEYS
    }
    by_root = {
        root_id: _group_summary(row for row in rows if row["root_id"] == root_id)
        for root_id in sorted({str(row["root_id"]) for row in rows})
    }
    write_once_json(run_root / "difficulty_summary.json", summary)
    write_once_json(run_root / "difficulty_by_split.json", by_split)
    write_once_json(run_root / "difficulty_by_question_type.json", by_type)
    write_once_json(run_root / "difficulty_by_root.json", by_root)
    for split in ("train", "dev", "test"):
        hard = [
            row for row in rows
            if row["split"] == split and row["difficulty_bucket"] == "hard_candidate"
        ]
        write_once_jsonl(run_root / f"{split}_hard_manifest.jsonl", hard)
    hard_rows = [row for row in rows if row["difficulty_bucket"] == "hard_candidate"]
    paths = [
        str(run_root / name) for name in (
            "source_inventory.json", "preflight_report.json", "program_gate_results.jsonl",
            "analysis_exposed_roots.json", "root_split_manifest.jsonl",
            "root_split_summary.json", "solver_contract.json", "solver_plan.json",
            "difficulty_manifest.jsonl", "difficulty_summary.json",
            "difficulty_by_split.json", "difficulty_by_question_type.json",
            "difficulty_by_root.json", "train_hard_manifest.jsonl",
            "dev_hard_manifest.jsonl", "test_hard_manifest.jsonl",
            "request_ledger", "raw_prompts", "raw_responses",
            "final_phase_report.json", "final_phase_report.md",
        )
    ]
    report = {
        "final_phase_report_version": "h4_round0_b0_k3_final_v1",
        "probe_version": PROBE_VERSION,
        "status": "COMPLETE",
        "source_root_count": inventory["source_root_count"],
        "candidate_bearing_root_count": inventory["candidate_bearing_root_count"],
        "incomplete_root_id": inventory["incomplete_root_id"],
        "incomplete_root_status": "QA_INCOMPLETE_EXCLUDED",
        "candidate_count": len(rows),
        "program_gate_counts": preflight["gate_counts"],
        "actual_solver_request_count": batch["ledger"]["request_count"],
        "actual_solver_generation_count": batch["ledger"]["actual_solver_generation_count"],
        "difficulty_buckets": summary["difficulty_buckets"],
        "difficulty_by_split": by_split,
        "hard_by_question_type": {
            qtype: by_type[qtype]["hard_candidate_count"] for qtype in QUESTION_KEYS
        },
        "hard_contributing_root_count": len({row["root_id"] for row in hard_rows}),
        "position_mechanisms": summary["position_mechanisms"],
        "major_result_paths": paths,
        "semantic_audit_applied": False,
        "strict_broad_classification_applied": False,
        "training_manifest_created": False,
        "b1_b2_executed": False,
        "fresh_mining_executed": False,
    }
    write_once_json(run_root / "final_phase_report.json", report)
    write_once_text(run_root / "final_phase_report.md", _markdown_report(report))
    return report


__all__ = [
    "DIFFICULTY_BUCKETS",
    "POSITION_MECHANISMS",
    "build_difficulty_rows",
    "classify_difficulty_counts",
    "finalize",
    "position_diagnostics",
    "summarize_candidate_result",
]
