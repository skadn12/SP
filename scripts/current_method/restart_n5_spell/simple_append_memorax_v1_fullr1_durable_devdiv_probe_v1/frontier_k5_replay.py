"""Incrementally re-evaluate a frozen QA batch with the SPELL K=5 frontier.

The source batch is read-only. Existing four-position no-context results and
the first three full-context rollouts are reused; only full-context slots 3 and
4 are generated for questions that passed the grounding gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..simple_append_v1.qa import classify_full_context_frontier
from ..simple_append_v1.qa_live import (
    DEFAULT_SOLVER_MODEL,
    DEFAULT_TOKENIZER,
    LocalSolverRunner,
    build_local_solver_runtime,
    build_solver_source,
)
from .qa_admission import classify_qa_admission_tier


REPLAY_VERSION = "simple_append_memorax_spell_frontier_k5_replay_v1"
DEFAULT_SOURCE_BATCH = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "simple_append_memorax_v1_batch_remaining8_pilot_8")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "simple_append_memorax_v1_batch_remaining8_pilot_8_"
    "spell_frontier_k5_replay_1")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*")
                       if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _root_inputs(root: Path) -> dict[str, Path]:
    return {
        "expanded": root / "expansion/expanded_dialogue.json",
        "batch": root / "qa/qa/final_mcq_batch.json",
        "report": root / "qa/reports/MEMORAX_QA_LIVE_REPORT.json",
    }


def merge_incremental_k5_result(
    existing: Mapping[str, Any], new_rollouts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge two new rollouts without weakening the frozen grounding result."""
    row = json.loads(json.dumps(existing))
    grounding = str((row.get("grounding") or {}).get("decision") or "")
    if grounding != "pass_grounding":
        if new_rollouts:
            raise ValueError("new_full_context_rollout_for_grounding_rejection")
        row.update({
            "admitted": False,
            "training_ready": False,
            "frontier_replay_decision": grounding or "grounding_result_missing",
            "new_full_context_request_count": 0,
        })
        return row
    historical = list(row.get("full_context_rollouts") or [])
    if (len(historical) != 3 or len(new_rollouts) != 2 or
            any(item.get("local_reward") not in (0, 1)
                for item in historical + list(new_rollouts))):
        raise ValueError("incremental_k5_rollout_contract_invalid")
    combined = [
        {**item, "response_source": "imported_historical_pilot8"}
        for item in historical
    ] + [
        {**dict(item), "response_source": "new_local_solver"}
        for item in new_rollouts
    ]
    rewards = [item["local_reward"] for item in combined]
    frontier = classify_full_context_frontier(rewards)
    row.update({
        "full_context_rollouts": combined,
        "full_context_correct": sum(rewards),
        "full_context_total": len(combined),
        "full_context_frontier": frontier,
        "full_context_bucket": frontier["bucket"],
        "frontier_replay_decision": frontier["decision"],
        "decision": ("admit_frontier_k5" if frontier["admitted"] else
                     frontier["decision"]),
        "admitted": frontier["admitted"],
        "new_full_context_request_count": 2,
        "imported_full_context_response_count": 3,
        "evidence_mismatch": not bool(row.get("evidence_ready", True)),
    })
    row.update(classify_qa_admission_tier(row))
    return row


def preflight(
    *, source_batch: Path, run_root: Path, tokenizer_path: Path,
    solver_model: Path,
) -> dict[str, Any]:
    root_dirs = sorted((source_batch / "roots").glob("*"))
    checks: dict[str, bool] = {
        "source_batch_exists": source_batch.is_dir(),
        "run_root_absent": not run_root.exists(),
        "tokenizer_exists": tokenizer_path.is_dir(),
        "solver_model_exists": solver_model.is_dir(),
        "root_directories_present": bool(root_dirs),
        "network_api_calls_zero": True,
    }
    question_count = solver_ready_count = grounding_pass_count = 0
    imported_no_context = imported_full_context = 0
    roots: list[dict[str, Any]] = []
    if all(checks.values()):
        try:
            for root in root_dirs:
                paths = _root_inputs(root)
                if not all(path.is_file() for path in paths.values()):
                    raise ValueError(f"source_root_artifact_missing:{root.name}")
                expanded = _load(paths["expanded"])
                batch = _load(paths["batch"])
                report = _load(paths["report"])
                qa_by_id = {str(item["qa_id"]): item
                            for item in batch.get("items") or []}
                local_grounding = 0
                for item in report.get("items") or []:
                    question_count += 1
                    qa_id = str(item.get("qa_id") or "")
                    if not qa_id:
                        continue
                    solver_ready_count += 1
                    if qa_id not in qa_by_id:
                        raise ValueError(f"qa_materialization_missing:{qa_id}")
                    orders = list(item.get("no_context_orders") or [])
                    if len(orders) != 4:
                        raise ValueError(f"no_context_four_positions_missing:{qa_id}")
                    imported_no_context += sum(
                        len(order.get("rollouts") or []) for order in orders)
                    grounding = str(
                        (item.get("grounding") or {}).get("decision") or "")
                    if grounding == "pass_grounding":
                        if len(item.get("full_context_rollouts") or []) != 3:
                            raise ValueError(f"historical_full_k3_missing:{qa_id}")
                        grounding_pass_count += 1
                        local_grounding += 1
                        imported_full_context += 3
                roots.append({
                    "root_id": root.name,
                    "expanded_dialogue_sha256": sha256_file(paths["expanded"]),
                    "materialized_question_count": len(qa_by_id),
                    "grounding_pass_count": local_grounding,
                })
        except Exception as error:
            checks["source_contract_valid"] = False
            roots.append({"error": f"{type(error).__name__}:{error}"})
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "preflight_version": REPLAY_VERSION,
        "decision": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "source_batch": str(source_batch),
        "source_batch_tree_sha256": (
            _hash_tree(source_batch) if source_batch.is_dir() else None),
        "root_count": len(root_dirs),
        "question_count": question_count,
        "solver_ready_count": solver_ready_count,
        "grounding_pass_count": grounding_pass_count,
        "imported_no_context_response_count": imported_no_context,
        "imported_full_context_response_count": imported_full_context,
        "new_local_solver_request_count": grounding_pass_count * 2,
        "roots": roots,
        "api_calls": 0,
    }


def run(
    *, source_batch: Path, run_root: Path, tokenizer_path: Path,
    solver_model: Path,
) -> dict[str, Any]:
    frozen = preflight(
        source_batch=source_batch, run_root=run_root,
        tokenizer_path=tokenizer_path, solver_model=solver_model)
    if frozen["decision"] != "pass":
        raise RuntimeError("SPELL_K5_REPLAY_PREFLIGHT_FAILED:" +
                           ",".join(frozen["failed_checks"]))

    # Model loading is a hard gate and happens before the fresh run root exists.
    tokenizer, backend = build_local_solver_runtime(
        tokenizer_path=tokenizer_path, solver_model=solver_model)
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "reports").mkdir(parents=True)
    freeze = {
        **frozen,
        "freeze_version": REPLAY_VERSION,
        "frozen_at": _now(),
        "fresh": True,
        "no_resume": True,
        "solver_model": str(solver_model),
        "tokenizer": str(tokenizer_path),
        "policy": {
            "no_context": "balanced_four_position_k1_all_correct_rejected",
            "full_context": "spell_frontier_k5_accept_1_to_4",
            "evidence_mismatch": "mark_only",
        },
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    manifest_sha = sha256_file(run_root / "run_freeze_manifest.json")
    all_results: list[dict[str, Any]] = []
    all_training: list[dict[str, Any]] = []
    per_root: list[dict[str, Any]] = []
    try:
        for source_root in sorted((source_batch / "roots").glob("*")):
            output_root = run_root / "roots" / source_root.name
            for relative in ("filter", "qa", "reports", "raw/prompts"):
                (output_root / relative).mkdir(parents=True, exist_ok=True)
            paths = _root_inputs(source_root)
            expanded = _load(paths["expanded"])
            source = build_solver_source(expanded)
            batch = _load(paths["batch"])
            report = _load(paths["report"])
            qa_by_id = {str(item["qa_id"]): item
                        for item in batch.get("items") or []}
            ledger = RequestLedger(output_root, manifest_sha, "local")
            local_pass = sum(
                str((item.get("grounding") or {}).get("decision") or "") ==
                "pass_grounding" for item in report.get("items") or [])
            runner = LocalSolverRunner(
                run_root=output_root, ledger=ledger, tokenizer=tokenizer,
                backend=backend, dialogue=source["combined_dialogue"],
                physical_request_budget=max(4, local_pass * 4))
            results: list[dict[str, Any]] = []
            training: list[dict[str, Any]] = []
            for existing in report.get("items") or []:
                qa_id = str(existing.get("qa_id") or "")
                new_rollouts: list[dict[str, Any]] = []
                if str((existing.get("grounding") or {}).get("decision") or "") == \
                        "pass_grounding":
                    qa = qa_by_id[qa_id]
                    for slot in (3, 4):
                        new_rollouts.append(runner.one(
                            qa=qa, options=qa["options"], mode="full_context",
                            order_id="original", slot=slot))
                result = merge_incremental_k5_result(existing, new_rollouts)
                results.append(result)
                all_results.append({"root_id": source_root.name, **result})
                if qa_id:
                    atomic_json(output_root / f"filter/{qa_id}.json", result)
                if result.get("training_ready") is True:
                    qa = {"root_id": source_root.name, **qa_by_id[qa_id],
                          "solver_filter": {
                              "full_context_bucket": result["full_context_bucket"],
                              "grounding_decision": (
                                  result.get("grounding") or {}).get("decision"),
                              "evidence_semantic_status": result.get(
                                  "evidence_semantic_status"),
                          }}
                    training.append(qa)
                    all_training.append(qa)
            buckets = {f"{index}/5": sum(
                item.get("full_context_bucket") == f"{index}/5"
                for item in results) for index in range(6)}
            root_report = {
                "root_id": source_root.name,
                "question_count": len(results),
                "grounding_pass_count": local_pass,
                "full_context_buckets": buckets,
                "training_frontier_count": len(training),
                "items": results,
            }
            atomic_json(output_root / "qa/training_frontier_mcq_batch.json", {
                "version": REPLAY_VERSION, "items": training})
            atomic_json(output_root / "reports/SPELL_FRONTIER_K5_REPORT.json",
                        root_report)
            per_root.append({key: value for key, value in root_report.items()
                             if key != "items"})
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    source_unchanged = _hash_tree(source_batch) == frozen[
        "source_batch_tree_sha256"]
    buckets = {f"{index}/5": sum(
        item.get("full_context_bucket") == f"{index}/5"
        for item in all_results) for index in range(6)}
    grounding_counts: dict[str, int] = {}
    for item in all_results:
        decision = str((item.get("grounding") or {}).get("decision") or
                       item.get("decision") or "unknown")
        grounding_counts[decision] = grounding_counts.get(decision, 0) + 1
    summary = {
        "report_version": REPLAY_VERSION,
        "status": "completed",
        "source_batch": str(source_batch),
        "run_root": str(run_root),
        "question_count": len(all_results),
        "grounding_decisions": grounding_counts,
        "full_context_buckets": buckets,
        "training_frontier_count": len(all_training),
        "new_local_solver_request_count": sum(
            int(item.get("new_full_context_request_count") or 0)
            for item in all_results),
        "imported_no_context_response_count": frozen[
            "imported_no_context_response_count"],
        "imported_full_context_response_count": frozen[
            "imported_full_context_response_count"],
        "paid_api_calls": 0,
        "writer_calls": 0,
        "planner_calls": 0,
        "realizer_calls": 0,
        "memorax_calls": 0,
        "source_batch_unchanged": source_unchanged,
        "per_root": per_root,
        "training_ready": bool(all_training),
        "completed_at": _now(),
    }
    atomic_json(run_root / "qa/training_frontier_mcq_batch.json", {
        "version": REPLAY_VERSION, "items": all_training})
    atomic_json(run_root / "reports/SPELL_FRONTIER_K5_REEVALUATION.json",
                summary)
    lines = [
        "# SPELL Frontier K5 Re-evaluation", "",
        f"- Questions: `{summary['question_count']}`",
        f"- Training frontier: `{summary['training_frontier_count']}`",
        f"- New local Solver requests: `{summary['new_local_solver_request_count']}`",
        "- Paid API calls: `0`", "", "## Full-context Buckets", "",
    ]
    lines.extend(f"- {bucket}: `{count}`" for bucket, count in buckets.items())
    atomic_text(run_root / "reports/SPELL_FRONTIER_K5_REEVALUATION.md",
                "\n".join(lines))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, default=DEFAULT_SOURCE_BATCH)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    kwargs = {
        "source_batch": args.source_batch.resolve(),
        "run_root": args.run_root.resolve(),
        "tokenizer_path": args.tokenizer.resolve(),
        "solver_model": args.solver_model.resolve(),
    }
    report = preflight(**kwargs) if args.preflight_only else run(**kwargs)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("decision", report.get("status")) in {
        "pass", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
