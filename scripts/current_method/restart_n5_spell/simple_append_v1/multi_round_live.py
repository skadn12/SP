"""Three-round simple-append expansion and per-round MCQ/Solver comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient
from ..questioner_contract import span_catalog_sha256
from ..vllm_backends import VLLMSolverBackend
from .assembly import append_expansion_round, validate_appended_round
from .context import build_tail_context_from_expanded_dialogue
from .contracts import (
    compile_round_plan, normalize_round_realizer_output,
    validate_planner_output,
)
from .live_canary import (
    CONTEXT_LIMIT, PLANNER_MAX_TOKENS, ROUND_REALIZER_MAX_TOKENS,
    _effective_api_config, _source_manifest,
)
from .live_runtime import SimpleAppendLiveCaller, aggregate_usage
from .prompts import (
    build_mcq_batch_prompt, build_planner_prompt, build_round_realizer_prompt,
)
from .qa import (
    MIN_FOCUS_ROUND_JOINT_WITH_HISTORY,
    MIN_FOCUS_ROUND_SUPPORTED,
    classify_full_context_frontier,
    evaluate_focus_round_profile, evaluate_qa_quota_status,
    materialize_solver_candidate,
    normalize_focus_writer_batch,
    partition_mcq_candidates,
)
from .qa_live import (
    DEFAULT_API_CONFIG, DEFAULT_SOLVER_MODEL, DEFAULT_TOKENIZER,
    MIN_CHILD_SUPPORTED, MIN_JOINT_PARENT_CHILD, QUESTION_COUNT,
    SOLVER_GPU_MEMORY_UTILIZATION, SOLVER_MAX_MODEL_LEN, SOLVER_MAX_TOKENS,
    LocalSolverRunner, _evaluate_grounding, _writer_call, build_solver_source,
)


RUNNER_VERSION = "simple_append_multi_round_live_v4_spell_frontier_k5"
DEFAULT_BASELINE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_simple_append_v1_live_canary_4")
DEFAULT_BASELINE_QA_REPLAY = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_simple_append_v1_"
    "qa_only_pilot_6_local_solver_replay_1")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_simple_append_v1_multiround_live_1")
DEFAULT_TOTAL_ROUNDS = 3


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(row for row in root.rglob("*") if row.is_file()):
        payload = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(payload).hexdigest().encode() + b"\0")
    return digest.hexdigest()


def _round_focus_stats(
    partition: Mapping[str, Any], *, expanded: Mapping[str, Any],
    round_index: int,
    pool_assignments: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return evaluate_focus_round_profile(
        partition, expanded_dialogue=expanded,
        focus_round_index=round_index,
        pool_assignments=pool_assignments)


def _baseline_metrics(
    *, expanded: Mapping[str, Any], qa_replay: Mapping[str, Any],
    baseline_run: Path = DEFAULT_BASELINE_RUN,
    baseline_qa_replay: Path = DEFAULT_BASELINE_QA_REPLAY,
) -> dict[str, Any]:
    materialized_path = baseline_qa_replay / \
        "qa/materialized_solver_candidates.json"
    materialized = (_load(materialized_path).get("items")
                    if materialized_path.is_file() else [])
    evidence_by_id = {str(row.get("qa_id")): list(
        row.get("evidence_turn_ids") or []) for row in materialized or []}
    items = list(qa_replay.get("items") or [])
    rates = [float((row.get("grounding") or {}).get(
        "semantic_correct_rate") or 0.0) for row in items]
    ready = [row for row in items if row.get("qa_content_ready")]
    current_ready = [row for row in ready if any(
        str(ref).startswith("R01:")
        for ref in evidence_by_id.get(str(row.get("qa_id")), []))]
    return {
        "round_index": 1,
        "turn_count": int(expanded.get("turn_count") or 0),
        "added_turn_count": int(expanded.get("child_turn_count") or 0),
        "candidate_count": len(items),
        "qa_content_ready_count": len(ready),
        "current_round_content_ready_count": len(current_ready),
        "mean_no_context_correct_rate": sum(rates) / len(rates) if rates else None,
        "grounding_decisions": _counts(
            str((row.get("grounding") or {}).get("decision")) for row in items),
        "historical_baseline": True,
        "expansion_run": str(baseline_run),
        "qa_run": str(baseline_qa_replay),
    }


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return result


def preflight(
    *, baseline_run: Path, baseline_qa_replay: Path, run_root: Path,
    api_config: Path, tokenizer_path: Path, solver_model: Path,
    total_rounds: int,
) -> dict[str, Any]:
    baseline_expanded_path = baseline_run / "expanded_dialogue.json"
    baseline_qa_path = baseline_qa_replay / \
        "reports/SIMPLE_APPEND_CANDIDATE_REPLAY_REPORT.json"
    checks = {
        "run_root_absent": not run_root.exists(),
        "baseline_expanded_exists": baseline_expanded_path.is_file(),
        "baseline_qa_exists": baseline_qa_path.is_file(),
        "api_config_exists": api_config.is_file(),
        "tokenizer_exists": tokenizer_path.is_dir(),
        "solver_model_exists": solver_model.is_dir(),
        "total_rounds_is_three": total_rounds == 3,
        "api_calls_zero": True,
    }
    details: dict[str, Any] = {}
    if all(checks.values()):
        try:
            expanded = _load(baseline_expanded_path)
            tail = build_tail_context_from_expanded_dialogue(expanded, session_count=2)
            planner = build_planner_prompt(tail)
            qa = build_mcq_batch_prompt(expanded, focus_round_index=1)
            checks.update({
                "baseline_has_round1": any(str(row.get("scene_ref") or "").startswith(
                    "R01:") for row in expanded.get("turns") or []),
                "planner_prompt_fits": planner["token_estimate"] +
                    PLANNER_MAX_TOKENS <= CONTEXT_LIMIT,
                "qa_prompt_fits": qa["token_estimate"] + 6144 <= CONTEXT_LIMIT,
                "baseline_qa_completed": _load(baseline_qa_path).get(
                    "status") == "completed",
            })
            details = {
                "baseline_run_tree_sha256": _hash_tree(baseline_run),
                "baseline_qa_tree_sha256": _hash_tree(baseline_qa_replay),
                "baseline_expanded_sha256": sha256_file(baseline_expanded_path),
                "baseline_turn_count": expanded.get("turn_count"),
                "source_manifest": _source_manifest(),
                "request_budget": {"round_2": 3, "round_3": 3,
                                   "total_paid_api_max": 6},
                "local_solver_reuses_one_loaded_model": True,
            }
        except Exception as error:
            checks["inputs_and_prompts_valid"] = False
            details["error"] = f"{type(error).__name__}:{error}"
    failed = sorted(key for key, value in checks.items() if not value)
    return {"preflight_version": RUNNER_VERSION,
            "decision": "pass" if not failed else "fail",
            "checks": checks, "failed_checks": failed,
            "details": details, "api_calls": 0,
            "run_root_created": run_root.exists()}


def _evaluate_round_qa(
    *, round_root: Path, expanded: Mapping[str, Any], round_index: int,
    api_config: Path, tokenizer: Any, solver_backend: VLLMSolverBackend,
    manifest_sha: str,
) -> dict[str, Any]:
    qa_root = round_root / "qa_evaluation"
    for relative in ("raw/prompts", "qa", "filter", "reports"):
        (qa_root / relative).mkdir(parents=True, exist_ok=True)
    prompt = build_mcq_batch_prompt(expanded, focus_round_index=round_index)
    writer_ledger = RequestLedger(qa_root, manifest_sha, "live")
    raw, writer_meta = _writer_call(
        run_root=qa_root, ledger=writer_ledger, prompt=prompt,
        api_config_path=api_config)
    atomic_json(qa_root / "qa/writer_output.json", raw)
    if raw is None:
        raise RuntimeError("multi_round_writer_parse_failed")
    writer_envelope = normalize_focus_writer_batch(raw)
    atomic_json(qa_root / "qa/writer_focus_envelope.json", writer_envelope)
    if not writer_envelope["valid"]:
        raise RuntimeError("multi_round_writer_focus_envelope_failed")
    normalized_raw = writer_envelope["batch"]
    atomic_json(qa_root / "qa/writer_output_normalized.json", normalized_raw)
    partition = partition_mcq_candidates(
        normalized_raw, expanded_dialogue=expanded, minimum_count=QUESTION_COUNT,
        maximum_count=None, min_child_supported=MIN_CHILD_SUPPORTED,
        min_joint_parent_child=MIN_JOINT_PARENT_CHILD)
    focus = _round_focus_stats(
        partition, expanded=expanded, round_index=round_index,
        pool_assignments=writer_envelope["pool_assignments"])
    partition["focus_round"] = focus
    partition["focus_round_profile_ready"] = focus[
        "focus_round_profile_ready"]
    quota = evaluate_qa_quota_status(partition, focus)
    partition.update(quota)
    partition["writer_admission_ready"] = bool(
        partition["envelope_valid"] and partition["solver_ready_count"])
    atomic_json(qa_root / "qa/writer_validation.json", partition)
    if not partition["envelope_valid"] or not partition["solver_ready_count"]:
        raise RuntimeError("multi_round_writer_envelope_failed")
    source = build_solver_source(expanded)
    materialized: list[dict[str, Any]] = []
    for row, status in zip(normalized_raw["questions"], partition["items"]):
        if status["solver_ready"]:
            materialized.append(materialize_solver_candidate(
                row, index=int(status["question_index"]),
                expanded_dialogue=expanded,
                span_catalog=source["span_catalog"],
                candidate_status=status))
    atomic_json(qa_root / "qa/materialized_solver_candidates.json", {
        "version": RUNNER_VERSION, "items": materialized})
    local_ledger = RequestLedger(qa_root, manifest_sha, "local")
    runner = LocalSolverRunner(
        run_root=qa_root, ledger=local_ledger, tokenizer=tokenizer,
        backend=solver_backend, dialogue=source["combined_dialogue"],
        physical_request_budget=max(264, len(materialized) * 22))
    focus_by_candidate = {str(row["candidate_id"]): row
                          for row in focus["items"]}
    results: list[dict[str, Any]] = []
    for qa in materialized:
        grounding, no_context = _evaluate_grounding(runner, qa)
        full_context: list[dict[str, Any]] = []
        if grounding["decision"] == "pass_grounding":
            for slot in range(5):
                full_context.append(runner.one(
                    qa=qa, options=qa["options"], mode="full_context",
                    order_id="original", slot=slot))
        full_rewards = [row.get("local_reward") for row in full_context]
        frontier = classify_full_context_frontier(full_rewards)
        full_correct = sum(int(value) for value in full_rewards
                           if value in (0, 1))
        ready = (grounding["decision"] == "pass_grounding" and
                 frontier["admitted"] is True)
        focus_row = focus_by_candidate.get(str(qa["candidate_id"]), {})
        results.append({
            "qa_id": qa["qa_id"], "candidate_id": qa["candidate_id"],
            "question": qa["question"], "source_scope": qa["source_scope"],
            "evidence_ready": qa["evidence_ready"],
            "evidence_issues": qa["evidence_issues"],
            **focus_row, "grounding": grounding,
            "no_context_orders": no_context,
            "full_context_correct": full_correct,
            "full_context_total": len(full_context),
            "full_context_frontier": frontier,
            "full_context_bucket": frontier["bucket"],
            "qa_content_ready": ready,
            "decision": "qa_content_ready" if ready else (
                frontier["decision"] if full_context else
                grounding["decision"]),
        })
    states = list((qa_root / "ledger/requests").glob("*.json"))
    rates = [float((row.get("grounding") or {}).get(
        "semantic_correct_rate") or 0.0) for row in results]
    ready = [row for row in results if row["qa_content_ready"]]
    current_ready = [row for row in ready if row.get("focus_round_supported")]
    joint_ready = [row for row in ready
                   if row.get("focus_round_joint_with_history")]
    child_ready = [row for row in ready if row.get("source_scope") in {
        "child_only", "parent_child"}]
    parent_child_ready = [row for row in ready
                          if row.get("source_scope") == "parent_child"]
    post_solver_quota_deficits: list[str] = []
    if len(child_ready) < MIN_CHILD_SUPPORTED:
        post_solver_quota_deficits.append("admitted_child_supported_below_minimum")
    if len(parent_child_ready) < MIN_JOINT_PARENT_CHILD:
        post_solver_quota_deficits.append("admitted_parent_child_below_minimum")
    if len(current_ready) < MIN_FOCUS_ROUND_SUPPORTED:
        post_solver_quota_deficits.append("admitted_current_round_below_minimum")
    if len(joint_ready) < MIN_FOCUS_ROUND_JOINT_WITH_HISTORY:
        post_solver_quota_deficits.append(
            "admitted_current_round_joint_history_below_minimum")
    top_up_required = bool(post_solver_quota_deficits)
    report = {
        "report_version": RUNNER_VERSION,
        "round_index": round_index,
        "candidate_count": len(results),
        "evidence_ready_count": partition["evidence_ready_count"],
        "evidence_issue_count": partition["evidence_issue_count"],
        "batch_profile_ready": partition["batch_profile_ready"],
        "focus_round_supported_count": focus["focus_round_supported_count"],
        "focus_round_joint_with_history_count":
            focus["focus_round_joint_with_history_count"],
        "focus_round_profile_ready": (
            focus["focus_round_supported_count"] >=
                MIN_FOCUS_ROUND_SUPPORTED and
            focus["focus_round_joint_with_history_count"] >=
                MIN_FOCUS_ROUND_JOINT_WITH_HISTORY),
        "focus_round_profile_errors": focus["focus_round_profile_errors"],
        "coverage_role_warnings": focus["coverage_role_warnings"],
        "writer_admission_ready": True,
        "pre_solver_quota_profile_ready": quota["quota_profile_ready"],
        "pre_solver_quota_deficits": quota["quota_deficits"],
        "qa_content_ready_count": len(ready),
        "current_round_content_ready_count": len(current_ready),
        "mean_no_context_correct_rate": sum(rates) / len(rates) if rates else None,
        "grounding_decisions": _counts(
            str(row["grounding"]["decision"]) for row in results),
        "items": results,
        "writer_usage": writer_meta.get("usage") or {},
        "local_solver_generations": len(states),
        "paid_api_requests": 1,
        "semantic_auditor_calls": 0,
        "top_up_required": top_up_required,
        "post_solver_quota_deficits": post_solver_quota_deficits,
        "existing_valid_qa_preserved_when_quota_incomplete": True,
        "qa_stage_status": (
            "completed_top_up_required" if top_up_required else "completed"),
        "qa_pipeline_ready": bool(ready) and not top_up_required,
        "current_round_qa_ready": bool(current_ready),
        "span_catalog_sha256": span_catalog_sha256(source["span_catalog"]),
    }
    atomic_json(qa_root / "reports/ROUND_QA_SOLVER_REPORT.json", report)
    return report


def run(
    *, baseline_run: Path, baseline_qa_replay: Path, run_root: Path,
    api_config: Path, tokenizer_path: Path, solver_model: Path,
    total_rounds: int = DEFAULT_TOTAL_ROUNDS,
    preflight_only: bool = False,
) -> dict[str, Any]:
    frozen = preflight(
        baseline_run=baseline_run, baseline_qa_replay=baseline_qa_replay,
        run_root=run_root, api_config=api_config,
        tokenizer_path=tokenizer_path, solver_model=solver_model,
        total_rounds=total_rounds)
    if preflight_only:
        return frozen
    if frozen["decision"] != "pass":
        raise RuntimeError("MULTI_ROUND_PRE_API_PREFLIGHT_FAILED:" +
                           ",".join(frozen["failed_checks"]))
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "reports").mkdir(parents=True, exist_ok=True)
    atomic_json(run_root / "run_freeze_manifest.json", frozen)
    current = _load(baseline_run / "expanded_dialogue.json")
    baseline_qa = _load(baseline_qa_replay /
                        "reports/SIMPLE_APPEND_CANDIDATE_REPLAY_REPORT.json")
    round_metrics = [_baseline_metrics(
        expanded=current, qa_replay=baseline_qa,
        baseline_run=baseline_run, baseline_qa_replay=baseline_qa_replay)]
    source_manifest_sha = frozen["details"]["source_manifest"]["manifest_sha256"]
    baseline_run_sha = frozen["details"]["baseline_run_tree_sha256"]
    baseline_qa_sha = frozen["details"]["baseline_qa_tree_sha256"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    solver_backend = VLLMSolverBackend(
        model_path=solver_model, tokenizer_path=tokenizer_path,
        max_model_len=SOLVER_MAX_MODEL_LEN, temperature=0.7, top_p=0.9,
        max_tokens=SOLVER_MAX_TOKENS,
        gpu_memory_utilization=SOLVER_GPU_MEMORY_UTILIZATION)
    failure: str | None = None
    paid_requests = 0
    for round_index in range(2, total_rounds + 1):
        round_root = run_root / f"round_{round_index:02d}"
        for relative in ("planner", "realizer", "attempts", "raw/prompts",
                         "reports"):
            (round_root / relative).mkdir(parents=True, exist_ok=True)
        before_turn_count = len(current["turns"])
        prior_sha = str(current["complete_dialogue_sha256"])

        def before_request(logical_role: str, role_version: str) -> None:
            if _source_manifest()["manifest_sha256"] != source_manifest_sha:
                raise RuntimeError("multi_round_source_manifest_drift")
            if _hash_tree(baseline_run) != baseline_run_sha or \
                    _hash_tree(baseline_qa_replay) != baseline_qa_sha:
                raise RuntimeError("multi_round_historical_input_drift")

        try:
            tail = build_tail_context_from_expanded_dialogue(
                current, session_count=2)
            atomic_json(round_root / "tail_context.json", tail)
            config = _effective_api_config(api_config)
            ledger = RequestLedger(round_root, source_manifest_sha, "live")
            caller = SimpleAppendLiveCaller(
                run_root=round_root, ledger=ledger,
                client=OpenAICompatibleChallengerClient(config),
                token_counter=lambda text: len(tokenizer.encode(
                    text, add_special_tokens=False)),
                root_id=str(current["root_id"]), context_limit=CONTEXT_LIMIT,
                before_request=before_request)
            planner_prompt = build_planner_prompt(tail)
            plan, _, plan_parse = caller.call_json(
                request_id=f"round_{round_index:02d}_planner_attempt_01",
                role="simple_append_planner_v1", prompt=planner_prompt["prompt"],
                prompt_version=planner_prompt["prompt_version"],
                max_tokens=PLANNER_MAX_TOKENS)
            paid_requests += 1
            validation = validate_planner_output(
                plan, parent_speakers=tail["parent_speakers"])
            validation["parse_errors"] = plan_parse
            atomic_json(round_root / "planner/planner_output.json", plan)
            atomic_json(round_root / "planner/validation.json", validation)
            if plan_parse or not validation["valid"]:
                raise RuntimeError("multi_round_planner_contract_failed")
            compiled = compile_round_plan(
                plan, tail_context=tail, round_index=round_index)
            atomic_json(round_root / "planner/compiled_round_plan.json", compiled)
            realizer_prompt = build_round_realizer_prompt(
                compiled_plan=compiled, tail_context=tail)
            realized, _, realized_parse = caller.call_json(
                request_id=f"round_{round_index:02d}_realizer_attempt_01",
                role="simple_append_round_realizer_v1",
                prompt=realizer_prompt["prompt"],
                prompt_version=realizer_prompt["prompt_version"],
                max_tokens=ROUND_REALIZER_MAX_TOKENS)
            paid_requests += 1
            normalized = normalize_round_realizer_output(
                realized, compiled_plan=compiled,
                parent_utterances=[row["text"] for row in current["turns"]],
                parent_turn_refs=[row["turn_ref"] for row in current["turns"]])
            normalized["parse_errors"] = realized_parse
            atomic_json(round_root / "realizer/round_realizer_output.json", realized)
            atomic_json(round_root / "realizer/validation.json", normalized)
            if realized_parse or not normalized["valid"]:
                raise RuntimeError("multi_round_realizer_contract_failed")
            next_expanded = append_expansion_round(
                prior_expanded=current, child=normalized["child"])
            appended = validate_appended_round(
                next_expanded, prior_expanded=current)
            if not appended["valid"]:
                raise RuntimeError("multi_round_append_failed")
            atomic_json(round_root / "child_dialogue.json", normalized["child"])
            atomic_json(round_root / "expanded_dialogue.json", next_expanded)
            qa_report = _evaluate_round_qa(
                round_root=round_root, expanded=next_expanded,
                round_index=round_index, api_config=api_config,
                tokenizer=tokenizer, solver_backend=solver_backend,
                manifest_sha=source_manifest_sha)
            paid_requests += 1
            usage = aggregate_usage(caller.records)
            metric = {
                "round_index": round_index,
                "prior_complete_dialogue_sha256": prior_sha,
                "complete_dialogue_sha256": next_expanded[
                    "complete_dialogue_sha256"],
                "turn_count": next_expanded["turn_count"],
                "added_turn_count": len(next_expanded["turns"]) - before_turn_count,
                "planner_scene_count": len(compiled["scenes"]),
                "expansion_api_usage": usage,
                **{key: qa_report[key] for key in (
                    "candidate_count", "qa_content_ready_count",
                    "current_round_content_ready_count",
                    "mean_no_context_correct_rate", "grounding_decisions",
                    "focus_round_supported_count",
                    "focus_round_joint_with_history_count",
                    "focus_round_profile_ready", "writer_admission_ready",
                    "qa_stage_status", "qa_pipeline_ready", "writer_usage",
                    "local_solver_generations")},
                "historical_baseline": False,
                "round_root": str(round_root),
            }
            round_metrics.append(metric)
            atomic_json(round_root / "reports/ROUND_SUMMARY.json", metric)
            current = next_expanded
        except Exception as error:
            failure = f"round_{round_index:02d}:{type(error).__name__}:{error}"
            atomic_json(round_root / "reports/ROUND_FAILURE.json", {
                "failure": failure, "paid_requests_so_far": paid_requests})
            break

    measured = [row for row in round_metrics
                if row.get("mean_no_context_correct_rate") is not None]
    hardest = (min(measured, key=lambda row: float(
        row["mean_no_context_correct_rate"])) if measured else None)
    paid_state_files = list(run_root.glob("round_*/ledger/requests/*.json")) + \
        list(run_root.glob("round_*/qa_evaluation/ledger/requests/*.json"))
    paid_requests = sum(
        _load(path).get("budget_class") in
        {"planner", "round_realizer", "mcq_writer"}
        for path in paid_state_files)
    report = {
        "report_version": RUNNER_VERSION,
        "status": "completed" if failure is None and
            len(round_metrics) == total_rounds else "failed",
        "failure": failure,
        "total_rounds_requested": total_rounds,
        "rounds_completed": len(round_metrics),
        "round_metrics": round_metrics,
        "hardest_round_by_lowest_no_context_correct_rate": (
            hardest.get("round_index") if hardest else None),
        "hardest_round_metric": hardest,
        "paid_api_requests": paid_requests,
        "baseline_runs_unchanged": (
            _hash_tree(baseline_run) == baseline_run_sha and
            _hash_tree(baseline_qa_replay) == baseline_qa_sha),
        "round_2_or_3_qa_calls_only_after_successful_expansion": True,
        "memory_calls": 0, "semantic_auditor_calls": 0,
        "qa_pipeline_ready": any(int(row.get(
            "current_round_content_ready_count") or 0) > 0
            for row in round_metrics),
        "training_ready": False,
    }
    atomic_json(run_root / "reports/MULTI_ROUND_DIFFICULTY_REPORT.json", report)
    lines = ["# Simple Append Multi-round Difficulty", "",
             f"Status: `{report['status']}`", "",
             f"Hardest round by no-context accuracy: "
             f"`{report['hardest_round_by_lowest_no_context_correct_rate']}`", ""]
    for row in round_metrics:
        lines.extend([
            f"## Round {row['round_index']}", "",
            f"- Turns: `{row['turn_count']}`",
            f"- Added turns: `{row['added_turn_count']}`",
            f"- Candidates: `{row['candidate_count']}`",
            f"- Content ready: `{row['qa_content_ready_count']}`",
            f"- Current-round content ready: "
            f"`{row['current_round_content_ready_count']}`",
            f"- Mean no-context correct rate: "
            f"`{row['mean_no_context_correct_rate']}`", ""])
    atomic_text(run_root / "reports/MULTI_ROUND_DIFFICULTY_REPORT.md",
                "\n".join(lines))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simple append three-round expansion/QA comparison")
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--baseline-qa-replay", type=Path,
                        default=DEFAULT_BASELINE_QA_REPLAY)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--total-rounds", type=int, default=DEFAULT_TOTAL_ROUNDS)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    report = run(
        baseline_run=args.baseline_run.resolve(),
        baseline_qa_replay=args.baseline_qa_replay.resolve(),
        run_root=args.run_root.resolve(), api_config=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve(),
        solver_model=args.solver_model.resolve(),
        total_rounds=args.total_rounds,
        preflight_only=args.preflight_only)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (report.get("decision") == "pass" if args.preflight_only else
                 report.get("status") == "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
