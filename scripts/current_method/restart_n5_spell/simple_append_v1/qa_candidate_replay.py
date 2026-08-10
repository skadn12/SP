"""Local-only candidate-level replay of a frozen simple-append MCQ response."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..vllm_backends import VLLMSolverBackend
from .qa import (
    MCQ_BATCH_SCHEMA_VERSION, classify_full_context_frontier,
    materialize_solver_candidate,
    partition_mcq_candidates,
)
from .qa_live import (
    DEFAULT_SOLVER_MODEL, DEFAULT_SOURCE_RUN, DEFAULT_TOKENIZER,
    SOLVER_GPU_MEMORY_UTILIZATION, SOLVER_MAX_MODEL_LEN, SOLVER_MAX_TOKENS,
    LocalSolverRunner, _evaluate_grounding, build_solver_source,
)


RUNNER_VERSION = "simple_append_candidate_replay_v3_spell_frontier_k5"
DEFAULT_WRITER_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_simple_append_v1_qa_only_pilot_4")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_simple_append_v1_"
    "qa_only_pilot_4_local_replay_1")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _migrate_frozen_writer(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project the historical fixed-slot response into the current loose schema."""
    migrated = json.loads(json.dumps(value))
    migrated["mcq_batch_schema_version"] = MCQ_BATCH_SCHEMA_VERSION
    for row in migrated.get("questions") or []:
        if isinstance(row, dict):
            row.pop("candidate_slot_id", None)
    return migrated


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*")
                       if candidate.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(relative + b"\0" + str(len(payload)).encode() + b"\0" +
                      hashlib.sha256(payload).hexdigest().encode() + b"\0")
    return digest.hexdigest()


def preflight(
    *, source_run: Path, writer_run: Path, run_root: Path,
    tokenizer_path: Path, solver_model: Path,
    minimum_count: int = 8, maximum_count: int = 12,
) -> dict[str, Any]:
    expanded_path = source_run / "expanded_dialogue.json"
    writer_path = writer_run / "qa/writer_output.json"
    checks = {
        "run_root_absent": not run_root.exists(),
        "source_run_exists": source_run.is_dir(),
        "expanded_dialogue_exists": expanded_path.is_file(),
        "writer_run_exists": writer_run.is_dir(),
        "writer_output_exists": writer_path.is_file(),
        "tokenizer_exists": tokenizer_path.is_dir(),
        "solver_model_exists": solver_model.is_dir(),
        "live_api_calls_zero": True,
    }
    details: dict[str, Any] = {}
    if all(checks.values()):
        try:
            expanded = _load(expanded_path)
            writer = _migrate_frozen_writer(_load(writer_path))
            source = build_solver_source(expanded)
            partition = partition_mcq_candidates(
                writer, expanded_dialogue=expanded,
                minimum_count=minimum_count, maximum_count=maximum_count)
            checks.update({
                "candidate_envelope_valid": partition["envelope_valid"],
                "solver_ready_candidates_present":
                    partition["solver_ready_count"] > 0,
                "source_has_parent_and_child": (
                    "original_parent" in source["partition_by_turn"].values() and
                    "promoted_child" in source["partition_by_turn"].values()),
            })
            details = {
                "source_run_tree_sha256": _hash_tree(source_run),
                "writer_run_tree_sha256": _hash_tree(writer_run),
                "expanded_dialogue_sha256": sha256_file(expanded_path),
                "writer_output_sha256": sha256_file(writer_path),
                "source_view_sha256": source["source_view_sha256"],
                "candidate_partition": partition,
                "replay_count_range": [minimum_count, maximum_count],
            }
        except Exception as error:
            checks["candidate_replay_input_valid"] = False
            details["preflight_error"] = f"{type(error).__name__}:{error}"
    failed = sorted(key for key, value in checks.items() if not value)
    return {"preflight_version": RUNNER_VERSION,
            "decision": "pass" if not failed else "fail",
            "checks": checks, "failed_checks": failed, "details": details,
            "api_calls": 0, "run_root_created": run_root.exists()}


def run(
    *, source_run: Path, writer_run: Path, run_root: Path,
    tokenizer_path: Path, solver_model: Path, preflight_only: bool = False,
    minimum_count: int = 8, maximum_count: int = 12,
) -> dict[str, Any]:
    frozen = preflight(
        source_run=source_run, writer_run=writer_run, run_root=run_root,
        tokenizer_path=tokenizer_path, solver_model=solver_model,
        minimum_count=minimum_count, maximum_count=maximum_count)
    if preflight_only:
        return frozen
    if frozen["decision"] != "pass":
        raise RuntimeError("PRE_REPLAY_PREFLIGHT_FAILED:" +
                           ",".join(frozen["failed_checks"]))
    expanded = _load(source_run / "expanded_dialogue.json")
    writer = _migrate_frozen_writer(
        _load(writer_run / "qa/writer_output.json"))
    source = build_solver_source(expanded)
    partition = partition_mcq_candidates(
        writer, expanded_dialogue=expanded,
        minimum_count=minimum_count, maximum_count=maximum_count)
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("frozen", "qa", "filter", "reports", "raw/prompts"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    freeze = {
        "freeze_version": RUNNER_VERSION,
        "created_at": _now(),
        "source_run": str(source_run),
        "writer_run": str(writer_run),
        "source_run_tree_sha256": frozen["details"]["source_run_tree_sha256"],
        "writer_run_tree_sha256": frozen["details"]["writer_run_tree_sha256"],
        "expanded_dialogue_sha256": frozen["details"]["expanded_dialogue_sha256"],
        "writer_output_sha256": frozen["details"]["writer_output_sha256"],
        "imported_writer_response_count": 1,
        "paid_api_request_budget": 0,
        "replay_count_range": [minimum_count, maximum_count],
        "local_solver_policy": {
            "four_answer_positions_k1": True,
            "boundary_extension_k2": True,
            "full_context_runs_after_grounding": 5,
            "full_context_acceptance_rule": "spell_frontier_1_to_4_of_5",
        },
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "frozen/expanded_dialogue.json", expanded)
    atomic_json(run_root / "frozen/writer_output.json", writer)
    atomic_json(run_root / "qa/candidate_partition.json", partition)
    manifest_sha = sha256_file(run_root / "run_freeze_manifest.json")
    ledger = RequestLedger(run_root, manifest_sha, "local")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    solver = VLLMSolverBackend(
        model_path=solver_model, tokenizer_path=tokenizer_path,
        max_model_len=SOLVER_MAX_MODEL_LEN, temperature=0.7, top_p=0.9,
        max_tokens=SOLVER_MAX_TOKENS,
        gpu_memory_utilization=SOLVER_GPU_MEMORY_UTILIZATION)
    runner = LocalSolverRunner(
        run_root=run_root, ledger=ledger, tokenizer=tokenizer,
        backend=solver, dialogue=source["combined_dialogue"])

    materialized: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for row, status in zip(writer["questions"], partition["items"]):
        if not status["solver_ready"]:
            results.append({
                "candidate_id": status["candidate_id"],
                "solver_ready": False,
                "structural_errors": status["structural_errors"],
                "evidence_issues": status["evidence_issues"],
                "qa_content_ready": False,
                "evidence_ready": False,
                "decision": "candidate_structural_quarantine",
            })
            continue
        qa = materialize_solver_candidate(
            row, index=int(status["question_index"]),
            expanded_dialogue=expanded, span_catalog=source["span_catalog"],
            candidate_status=status)
        materialized.append(qa)
        grounding, no_context = _evaluate_grounding(runner, qa)
        full_context: list[dict[str, Any]] = []
        if grounding["decision"] == "pass_grounding":
            for slot in range(5):
                full_context.append(runner.one(
                    qa=qa, options=qa["options"], mode="full_context",
                    order_id="original", slot=slot))
        full_rewards = [item.get("local_reward") for item in full_context]
        frontier = classify_full_context_frontier(full_rewards)
        full_correct = sum(int(value) for value in full_rewards
                           if value in (0, 1))
        content_ready = (grounding["decision"] == "pass_grounding" and
                         frontier["admitted"] is True)
        decision = ("qa_content_ready" if content_ready else
                    frontier["decision"] if full_context else
                    grounding["decision"])
        result = {
            "qa_id": qa["qa_id"],
            "candidate_id": qa["candidate_id"],
            "question": qa["question"],
            "solver_ready": True,
            "evidence_ready": qa["evidence_ready"],
            "evidence_issues": qa["evidence_issues"],
            "invalid_evidence_turn_ids": qa["invalid_evidence_turn_ids"],
            "actual_source_scope": qa["source_scope"],
            "grounding": grounding,
            "no_context_orders": no_context,
            "full_context_rollouts": full_context,
            "full_context_correct": full_correct,
            "full_context_total": len(full_context),
            "full_context_frontier": frontier,
            "full_context_bucket": frontier["bucket"],
            "qa_content_ready": content_ready,
            "decision": decision,
        }
        results.append(result)
        atomic_json(run_root / f"filter/{qa['qa_id']}.json", result)

    atomic_json(run_root / "qa/materialized_solver_candidates.json", {
        "version": RUNNER_VERSION, "items": materialized})
    admitted_ids = {row["qa_id"] for row in results
                    if row.get("qa_content_ready")}
    admitted = [row for row in materialized if row["qa_id"] in admitted_ids]
    atomic_json(run_root / "qa/content_admitted_mcq_batch.json", {
        "version": RUNNER_VERSION, "items": admitted})
    request_states = list((run_root / "ledger/requests").glob("*.json"))
    source_unchanged = _hash_tree(source_run) == freeze["source_run_tree_sha256"]
    writer_unchanged = _hash_tree(writer_run) == freeze["writer_run_tree_sha256"]
    report = {
        "report_version": RUNNER_VERSION,
        "status": "completed",
        "source_run": str(source_run),
        "writer_run": str(writer_run),
        "run_root": str(run_root),
        "candidate_count": len(partition["items"]),
        "solver_ready_count": partition["solver_ready_count"],
        "evidence_ready_count": partition["evidence_ready_count"],
        "evidence_issue_count": partition["evidence_issue_count"],
        "qa_content_ready_count": len(admitted_ids),
        "items": results,
        "api_statistics": {
            "paid_api_requests": 0,
            "imported_writer_responses": 1,
            "local_solver_generations": len(request_states),
            "planner_calls": 0, "realizer_calls": 0,
            "memory_calls": 0, "semantic_auditor_calls": 0,
        },
        "source_unchanged": source_unchanged,
        "writer_run_unchanged": writer_unchanged,
        "qa_pipeline_ready": bool(admitted_ids),
        "evidence_pipeline_ready": any(
            row.get("qa_content_ready") and row.get("evidence_ready")
            for row in results),
        "training_ready": False,
        "completed_at": _now(),
    }
    atomic_json(run_root / "reports/SIMPLE_APPEND_CANDIDATE_REPLAY_REPORT.json",
                report)
    lines = ["# Simple Append Candidate Replay", "",
             f"- Candidates: `{report['candidate_count']}`",
             f"- Solver-ready: `{report['solver_ready_count']}`",
             f"- Evidence-ready: `{report['evidence_ready_count']}`",
             f"- QA content-ready: `{report['qa_content_ready_count']}`",
             f"- Paid API requests: `0`", ""]
    for row in results:
        lines.extend([f"## {row.get('candidate_id')}", "",
                      f"- Decision: `{row['decision']}`",
                      f"- Evidence ready: `{row['evidence_ready']}`", ""])
    atomic_text(run_root / "reports/SIMPLE_APPEND_CANDIDATE_REPLAY_REPORT.md",
                "\n".join(lines))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local-only replay of frozen simple-append MCQ candidates")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--writer-run", type=Path, default=DEFAULT_WRITER_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--minimum-count", type=int, default=8)
    parser.add_argument("--maximum-count", type=int, default=12)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    report = run(
        source_run=args.source_run.resolve(),
        writer_run=args.writer_run.resolve(), run_root=args.run_root.resolve(),
        tokenizer_path=args.tokenizer.resolve(),
        solver_model=args.solver_model.resolve(),
        preflight_only=args.preflight_only,
        minimum_count=args.minimum_count,
        maximum_count=args.maximum_count)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (report.get("decision") == "pass" if args.preflight_only else
                 report.get("status") == "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
