"""QA Writer + local Solver run over recovered Memory-augmented dialogue."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..simple_append_v1.qa import (
    build_evidence_alias_registry,
    retain_focus_writer_overflow,
    evaluate_focus_round_profile,
    evaluate_qa_quota_status,
    normalize_focus_writer_batch,
    partition_mcq_candidates,
)
from ..simple_append_v1.qa_live import (
    CONTEXT_LIMIT,
    DEFAULT_API_CONFIG,
    DEFAULT_SOLVER_MODEL,
    DEFAULT_TOKENIZER,
    MIN_CHILD_SUPPORTED,
    MIN_JOINT_PARENT_CHILD,
    MAX_QUESTION_COUNT,
    QUESTION_COUNT,
    WRITER_MAX_TOKENS,
    _evaluate_solver_candidates,
    build_local_solver_runtime,
    _writer_call,
    build_solver_source,
)
from .memory_context import MEMORY_OVERVIEW_VERSION
from .context import select_fresh_r1_qa_context
from .prompts import build_memory_augmented_mcq_prompt
from .qa_admission import classify_qa_admission_tier
from .qa_sources import resolve_qa_artifacts


RUNNER_VERSION = "simple_append_memorax_fullr1_devdiv_qa_live_v1"
DEFAULT_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_"
    "simple_append_memorax_fullr1_devdiv_probe_v1_live_1"
)
DEFAULT_MEMORY_RECOVERY = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_"
    "simple_append_memorax_fullr1_devdiv_probe_v1_live_1"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_"
    "simple_append_memorax_fullr1_devdiv_probe_v1_qa_live_1"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(row for row in root.rglob("*") if row.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _writer_request_summary(ledger: RequestLedger) -> dict[str, int]:
    """Report Writer attempts without treating an ambiguous send as zero calls."""
    states = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in ledger.state_dir.glob("*.json")
    ]
    writer_states = [
        row for row in states if row.get("budget_class") == "mcq_writer"
    ]
    return {
        "request_attempt_count": len(writer_states),
        "completed_response_count": sum(
            row.get("status") == "completed" for row in writer_states),
        "ambiguous_request_count": sum(
            row.get("status") != "completed" for row in writer_states),
        "paid_generation_count": sum(
            int(row.get("paid_generation_count", 0)) for row in writer_states),
    }


def preflight(
    *, source_run: Path, memory_recovery: Path, run_root: Path,
    api_config: Path, tokenizer_path: Path, solver_model: Path,
    solver_runtime: tuple[Any, Any] | None = None,
    writer_output_replay: Path | None = None,
) -> dict[str, Any]:
    expanded_path = source_run / "expanded_dialogue.json"
    overview_path = memory_recovery / "memory/post_generation_overview.json"
    saved_prompt_manifest_path = memory_recovery / "qa/mcq_prompt_manifest.json"
    legacy_prompt_path = memory_recovery / "qa/mcq_prompt.json"
    checks = {
        "run_root_absent": not run_root.exists(),
        "source_run_exists": source_run.is_dir(),
        "memory_recovery_exists": memory_recovery.is_dir(),
        "expanded_dialogue_exists": expanded_path.is_file(),
        "memory_overview_exists": overview_path.is_file(),
        "saved_prompt_artifact_exists": (
            saved_prompt_manifest_path.is_file() or legacy_prompt_path.is_file()),
        "api_config_exists": api_config.is_file(),
        "tokenizer_exists": tokenizer_path.is_dir(),
        "solver_model_exists": solver_model.is_dir(),
        "writer_replay_exists_if_requested": (
            writer_output_replay is None or writer_output_replay.is_file()),
        "api_calls_zero": True,
    }
    details: dict[str, Any] = {}
    if all(checks.values()):
        try:
            handoff = resolve_qa_artifacts(
                source_run=source_run, memory_artifacts=memory_recovery)
            expanded = _load(expanded_path)
            overview = _load(overview_path)
            saved_prompt = _load(
                saved_prompt_manifest_path if saved_prompt_manifest_path.is_file()
                else legacy_prompt_path)
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=True)
            token_counter = lambda text: len(tokenizer.encode(
                text, add_special_tokens=False))
            qa_context = select_fresh_r1_qa_context(
                expanded, token_counter=token_counter)
            prompt = build_memory_augmented_mcq_prompt(
                qa_context, overview, target_questions=QUESTION_COUNT,
                maximum_questions=MAX_QUESTION_COUNT,
                focus_round_index=1)
            selection = qa_context["qa_context_selection"]
            checks.update({
                # Complete Parent+Child is the QA fact source.  Memorax write
                # verification is reported, but deliberately not a QA gate.
                "memory_write_status_recorded": (
                    "memory_write_verified" in handoff["readiness"]),
                "memory_context_status_recorded": (
                    "memory_context_degraded" in handoff["readiness"]),
                "upstream_qa_prompt_ready": (
                    handoff["readiness"]["qa_prompt_ready"] is True),
                "memory_overview_version": (
                    overview.get("memory_overview_version") == MEMORY_OVERVIEW_VERSION),
                "root_identity_matches": (
                    str(expanded.get("root_id") or "") ==
                    str(overview.get("root_id") or "")),
                "complete_dialogue_is_qa_fact_source": (
                    expanded.get("qa_fact_source") == "complete_parent_plus_child" and
                    expanded.get("memory_is_qa_fact_source") is False),
                "memory_not_qa_fact_source": (
                    overview.get("memory_is_qa_fact_source") is False),
                "qa_raw_dialogue_budget_frozen": (
                    selection["raw_dialogue_context_budget_tokens"] == 4096),
                "qa_context_selection_consistent": (
                    selection["full_parent_child_included"] is
                    (selection["full_parent_child_raw_dialogue_tokens"] <= 4096)),
                "qa_complete_sessions_only": (
                    selection["complete_sessions_only"] is True),
                "saved_prompt_memory_matches": (
                    saved_prompt.get("memory_overview_sha256") ==
                    overview.get("memory_overview_sha256")),
                "prompt_fits_context": (
                    prompt["token_estimate"] + WRITER_MAX_TOKENS <= CONTEXT_LIMIT),
            })
            details = {
                "source_run_tree_sha256": _hash_tree(source_run),
                "memory_recovery_tree_sha256": _hash_tree(memory_recovery),
                "expanded_dialogue_sha256": sha256_file(expanded_path),
                "memory_overview_sha256": overview["memory_overview_sha256"],
                "writer_prompt_sha256": prompt["prompt_sha256"],
                "upstream_writer_prompt_sha256": saved_prompt.get(
                    "prompt_sha256"),
                "upstream_prompt_contract_changed": (
                    prompt["prompt_sha256"] != saved_prompt.get("prompt_sha256")),
                "writer_prompt_token_estimate": prompt["token_estimate"],
                "root_id": expanded["root_id"],
                "turn_count": expanded["turn_count"],
                "qa_context_selection": selection,
                "qa_source_kind": handoff["source_kind"],
                "qa_readiness_report": str(handoff["readiness_report_path"]),
                "memory_write_verified": bool(
                    handoff["readiness"].get("memory_write_verified")),
                "memory_context_degraded": bool(
                    handoff["readiness"].get("memory_context_degraded")),
                "request_budget": {
                    "mcq_writer": 0 if writer_output_replay is not None else 1,
                    "local_solver_physical_max": 264,
                    "memorax": 0,
                    "planner": 0,
                    "realizer": 0,
                },
                "writer_output_replay": (
                    str(writer_output_replay.resolve())
                    if writer_output_replay is not None else None),
                "writer_output_replay_sha256": (
                    sha256_file(writer_output_replay)
                    if writer_output_replay is not None else None),
            }
        except Exception as error:
            checks["inputs_and_prompt_valid"] = False
            details["error"] = f"{type(error).__name__}:{error}"
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "preflight_version": RUNNER_VERSION,
        "decision": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "details": details,
        "source_run": str(source_run.resolve()),
        "memory_recovery": str(memory_recovery.resolve()),
        "run_root": str(run_root.resolve()),
        "api_calls": 0,
    }


def run(
    *, source_run: Path, memory_recovery: Path, run_root: Path,
    api_config: Path, tokenizer_path: Path, solver_model: Path,
    solver_runtime: tuple[Any, Any] | None = None,
    writer_output_replay: Path | None = None,
) -> dict[str, Any]:
    frozen = preflight(
        source_run=source_run, memory_recovery=memory_recovery,
        run_root=run_root, api_config=api_config,
        tokenizer_path=tokenizer_path, solver_model=solver_model,
        writer_output_replay=writer_output_replay)
    if frozen["decision"] != "pass":
        raise RuntimeError("MEMORAX_QA_PRE_API_PREFLIGHT_FAILED:" +
                           ",".join(frozen["failed_checks"]))
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("frozen", "qa", "filter", "reports", "raw/prompts"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    expanded = _load(source_run / "expanded_dialogue.json")
    overview = _load(
        memory_recovery / "memory/post_generation_overview.json")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True)
    token_counter = lambda text: len(tokenizer.encode(
        text, add_special_tokens=False))
    qa_context = select_fresh_r1_qa_context(
        expanded, token_counter=token_counter)
    prompt = build_memory_augmented_mcq_prompt(
        qa_context, overview, target_questions=QUESTION_COUNT,
        maximum_questions=MAX_QUESTION_COUNT,
        focus_round_index=1)
    freeze = {
        **frozen,
        "freeze_version": RUNNER_VERSION,
        "frozen_at": _now(),
        "fresh": True,
        "no_resume": True,
        "writer_prompt_sha256": prompt["prompt_sha256"],
        "memory_is_qa_fact_source": False,
        "writer_response_replay": writer_output_replay is not None,
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "frozen/expanded_dialogue.json", expanded)
    atomic_json(run_root / "frozen/qa_context_view.json", qa_context)
    atomic_json(run_root / "frozen/memory_overview.json", overview)
    atomic_text(run_root / "qa/mcq_prompt.txt", str(prompt["prompt"]))
    atomic_json(run_root / "qa/mcq_prompt_manifest.json", {
        key: value for key, value in prompt.items() if key != "prompt"})

    source = build_solver_source(expanded)
    aliases = build_evidence_alias_registry(qa_context)
    atomic_json(run_root / "frozen/evidence_alias_registry.json", aliases)
    atomic_json(run_root / "frozen/questioner_span_catalog.json",
                source["span_catalog"])
    manifest_sha = sha256_file(run_root / "run_freeze_manifest.json")
    ledger = RequestLedger(run_root, manifest_sha, "live")
    failure: str | None = None
    writer_meta: dict[str, Any] = {}
    partition: dict[str, Any] = {}
    items: list[dict[str, Any]] = []
    materialized: list[dict[str, Any]] = []
    try:
        if writer_output_replay is None:
            raw, writer_meta = _writer_call(
                run_root=run_root, ledger=ledger, prompt=prompt,
                api_config_path=api_config)
        else:
            raw = _load(writer_output_replay)
            writer_meta = {
                "replay": True,
                "source_path": str(writer_output_replay.resolve()),
                "source_sha256": sha256_file(writer_output_replay),
                "usage": {},
            }
        atomic_json(run_root / "qa/writer_output.json", raw)
        if raw is None:
            raise RuntimeError("memorax_mcq_writer_parse_failed")
        overflow = retain_focus_writer_overflow(
            raw, target_count=QUESTION_COUNT,
            maximum_count=MAX_QUESTION_COUNT)
        atomic_json(run_root / "qa/writer_overflow_audit.json", overflow)
        envelope = normalize_focus_writer_batch(overflow["value"])
        atomic_json(run_root / "qa/writer_focus_envelope.json", envelope)
        if not envelope["valid"]:
            raise RuntimeError("memorax_mcq_writer_focus_envelope_failed")
        normalized = envelope["batch"]
        atomic_json(run_root / "qa/writer_output_normalized.json", normalized)
        partition = partition_mcq_candidates(
            normalized, expanded_dialogue=qa_context,
            minimum_count=QUESTION_COUNT,
            maximum_count=None,
            min_child_supported=MIN_CHILD_SUPPORTED,
            min_joint_parent_child=MIN_JOINT_PARENT_CHILD)
        focus = evaluate_focus_round_profile(
            partition, expanded_dialogue=qa_context, focus_round_index=1,
            pool_assignments=envelope["pool_assignments"])
        partition["focus_round"] = focus
        partition["focus_round_profile_ready"] = focus[
            "focus_round_profile_ready"]
        partition.update(evaluate_qa_quota_status(partition, focus))
        partition["writer_admission_ready"] = bool(
            partition["envelope_valid"] and partition["solver_ready_count"])
        atomic_json(run_root / "qa/writer_validation.json", partition)
        if not partition["writer_admission_ready"]:
            raise RuntimeError("memorax_mcq_writer_no_solver_ready_candidate")
        owned_solver_runtime = False
        if solver_runtime is None:
            solver_runtime = build_local_solver_runtime(
                tokenizer_path=tokenizer_path, solver_model=solver_model)
            owned_solver_runtime = True
        try:
            items, materialized = _evaluate_solver_candidates(
                raw=normalized, partition=partition, expanded=expanded,
                source=source, run_root=run_root, manifest_sha=manifest_sha,
                tokenizer_path=tokenizer_path, solver_model=solver_model,
                solver_runtime=solver_runtime)
        finally:
            if owned_solver_runtime:
                close = getattr(solver_runtime[1], "close", None)
                if callable(close):
                    close()
        for item in items:
            item.update(classify_qa_admission_tier(item))
            if item.get("qa_id"):
                atomic_json(run_root / f"filter/{item['qa_id']}.json", item)
                atomic_json(
                    run_root / f"reports/{item['qa_id']}_evaluation.json", item)
    except Exception as error:
        failure = f"{type(error).__name__}:{error}"

    admitted_ids = [str(row["qa_id"]) for row in items if row.get("admitted")]
    if materialized:
        accepted = [row for row in materialized
                    if str(row.get("qa_id")) in admitted_ids]
        atomic_json(run_root / "qa/admitted_mcq_batch.json", {
            "batch_version": RUNNER_VERSION, "items": accepted})
        training_ids = {
            str(row.get("qa_id")) for row in items
            if row.get("training_ready") is True
        }
        atomic_json(run_root / "qa/training_ready_mcq_batch.json", {
            "batch_version": RUNNER_VERSION,
            "items": [row for row in materialized
                      if str(row.get("qa_id")) in training_ids],
        })
    focus = partition.get("focus_round") or {}
    source_unchanged = _hash_tree(source_run) == frozen["details"][
        "source_run_tree_sha256"]
    memory_unchanged = _hash_tree(memory_recovery) == frozen["details"][
        "memory_recovery_tree_sha256"]
    grounding_counts: dict[str, int] = {}
    for row in items:
        decision = str((row.get("grounding") or {}).get("decision") or
                       row.get("decision") or "unknown")
        grounding_counts[decision] = grounding_counts.get(decision, 0) + 1
    writer_requests = _writer_request_summary(ledger)
    writer_transport_ambiguous = bool(
        failure is not None and
        writer_requests["ambiguous_request_count"] and
        not writer_requests["completed_response_count"])
    failure_stage = None
    if writer_transport_ambiguous:
        failure_stage = "qa_writer_transport_ambiguous"
    elif failure is not None:
        failure_stage = "qa_writer_or_solver"
    report = {
        "report_version": RUNNER_VERSION,
        "status": (
            "completed" if failure is None else
            "pending_recovery" if writer_transport_ambiguous else "failed"),
        "failure": failure,
        "failure_stage": failure_stage,
        "source_run": str(source_run),
        "memory_recovery": str(memory_recovery),
        "run_root": str(run_root),
        "question_count": len(partition.get("items") or []),
        "solver_ready_count": int(partition.get("solver_ready_count") or 0),
        "solver_evaluated_count": len(items),
        "admitted_count": len(admitted_ids),
        "admitted_qa_ids": admitted_ids,
        "observation_ready_count": sum(
            bool(row.get("observation_ready")) for row in items),
        "training_ready_count": sum(
            bool(row.get("training_ready")) for row in items),
        "semantic_evidence_audited_count": sum(
            bool(row.get("semantic_evidence_audit_executed")) for row in items),
        "grounding_decisions": grounding_counts,
        "focus_round_supported_count": int(
            focus.get("focus_round_supported_count") or 0),
        "focus_round_joint_with_history_count": int(
            focus.get("focus_round_joint_with_history_count") or 0),
        "writer_admission_ready": bool(partition.get("writer_admission_ready")),
        "writer_usage": writer_meta.get("usage") or {},
        "qa_context_selection": qa_context.get("qa_context_selection"),
        "items": items,
        "source_run_unchanged": source_unchanged,
        "memory_recovery_unchanged": memory_unchanged,
        "memorax_calls": 0,
        "planner_calls": 0,
        "realizer_calls": 0,
        "qa_writer_calls": writer_requests["request_attempt_count"],
        "qa_writer_completed_responses": writer_requests[
            "completed_response_count"],
        "qa_writer_ambiguous_requests": writer_requests[
            "ambiguous_request_count"],
        "qa_writer_paid_generations_confirmed": writer_requests[
            "paid_generation_count"],
        "writer_response_replay_count": int(writer_output_replay is not None),
        "generation_ready": True,
        "memory_write_verified": bool(
            frozen["details"].get("memory_write_verified", True)),
        "memory_context_degraded": bool(
            frozen["details"].get("memory_context_degraded", False)),
        "qa_pipeline_ready": bool(admitted_ids),
        "training_ready": False,
        "completed_at": _now(),
    }
    atomic_json(run_root / "reports/MEMORAX_QA_LIVE_REPORT.json", report)
    lines = [
        "# Memorax QA Writer + Solver Live Run", "",
        f"- Status: `{report['status']}`",
        f"- Questions: `{report['question_count']}`",
        f"- Solver ready: `{report['solver_ready_count']}`",
        f"- Admitted: `{report['admitted_count']}`",
        f"- QA pipeline ready: `{str(report['qa_pipeline_ready']).lower()}`",
        "- Memorax, Planner, and Realizer calls: `0`", "",
    ]
    for row in items:
        lines.extend([
            f"## {row.get('qa_id') or row.get('candidate_id')}", "",
            f"- Decision: `{row.get('decision')}`",
            f"- Question: {row.get('question')}", "",
        ])
    atomic_text(run_root / "reports/MEMORAX_QA_LIVE_REPORT.md", "\n".join(lines))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument(
        "--memory-artifacts", "--memory-recovery", dest="memory_recovery",
        type=Path, default=DEFAULT_MEMORY_RECOVERY)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--writer-output-replay", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        report = preflight(
            source_run=args.source_run.resolve(),
            memory_recovery=args.memory_recovery.resolve(),
            run_root=args.run_root.resolve(),
            api_config=args.api_config.resolve(),
            tokenizer_path=args.tokenizer.resolve(),
            solver_model=args.solver_model.resolve(),
            writer_output_replay=(args.writer_output_replay.resolve()
                                  if args.writer_output_replay else None))
    else:
        report = run(
            source_run=args.source_run.resolve(),
            memory_recovery=args.memory_recovery.resolve(),
            run_root=args.run_root.resolve(),
            api_config=args.api_config.resolve(),
            tokenizer_path=args.tokenizer.resolve(),
            solver_model=args.solver_model.resolve(),
            writer_output_replay=(args.writer_output_replay.resolve()
                                  if args.writer_output_replay else None))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (
        report.get("decision") == "pass" if args.preflight_only
        else report.get("status") == "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
