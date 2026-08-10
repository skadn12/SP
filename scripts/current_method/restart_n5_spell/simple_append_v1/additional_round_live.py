"""Append exactly one new round to an existing simple-append dialogue."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from ..core import atomic_json, sha256_file
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient
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
from .multi_round_live import _evaluate_round_qa, _hash_tree, _load
from .prompts import build_mcq_batch_prompt, build_planner_prompt, build_round_realizer_prompt
from .qa_live import (
    DEFAULT_API_CONFIG, DEFAULT_SOLVER_MODEL, DEFAULT_TOKENIZER,
    SOLVER_GPU_MEMORY_UTILIZATION, SOLVER_MAX_MODEL_LEN, SOLVER_MAX_TOKENS,
)


RUNNER_VERSION = "simple_append_additional_round_live_v1"
DEFAULT_PRIOR_ROUND_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_simple_append_v1_"
    "multiround_live_2/round_03")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_simple_append_v1_round_04_live_1")
DEFAULT_ROUND_INDEX = 4


def _highest_round_index(expanded: Mapping[str, Any]) -> int:
    values = []
    for row in expanded.get("turns") or []:
        match = re.fullmatch(r"R(\d{2}):S\d{2}", str(row.get("scene_ref") or ""))
        if match:
            values.append(int(match.group(1)))
    return max(values, default=0)


def preflight(
    *, prior_round_root: Path, run_root: Path, round_index: int,
    api_config: Path, tokenizer_path: Path, solver_model: Path,
) -> dict[str, Any]:
    expanded_path = prior_round_root / "expanded_dialogue.json"
    prior_qa_path = prior_round_root / "qa_evaluation/reports/ROUND_QA_SOLVER_REPORT.json"
    checks = {
        "run_root_absent": not run_root.exists(),
        "prior_expanded_exists": expanded_path.is_file(),
        "prior_qa_exists": prior_qa_path.is_file(),
        "api_config_exists": api_config.is_file(),
        "tokenizer_exists": tokenizer_path.is_dir(),
        "solver_model_exists": solver_model.is_dir(),
        "api_calls_zero": True,
    }
    details: dict[str, Any] = {}
    if all(checks.values()):
        try:
            expanded = _load(expanded_path)
            highest = _highest_round_index(expanded)
            tail = build_tail_context_from_expanded_dialogue(expanded, session_count=2)
            planner_prompt = build_planner_prompt(tail)
            qa_prompt = build_mcq_batch_prompt(expanded, focus_round_index=highest)
            checks.update({
                "round_index_is_next": round_index == highest + 1,
                "prior_tail_matches_latest_round": str(
                    tail["parent_tail_turn_id"]).startswith(f"R{highest:02d}:"),
                "planner_prompt_fits": planner_prompt["token_estimate"] +
                    PLANNER_MAX_TOKENS <= CONTEXT_LIMIT,
                "projected_qa_prompt_fits": qa_prompt["token_estimate"] +
                    6144 + 4096 <= CONTEXT_LIMIT,
            })
            details = {
                "prior_round_root": str(prior_round_root),
                "prior_round_tree_sha256": _hash_tree(prior_round_root),
                "prior_expanded_sha256": sha256_file(expanded_path),
                "prior_qa_sha256": sha256_file(prior_qa_path),
                "prior_turn_count": expanded.get("turn_count"),
                "highest_existing_round": highest,
                "requested_round_index": round_index,
                "source_manifest": _source_manifest(),
                "request_budget": {
                    "planner": 1, "round_realizer": 1,
                    "mcq_writer": 1, "total_paid_api_max": 3,
                },
                "local_solver_reuses_one_loaded_model": True,
            }
        except Exception as error:
            checks["inputs_and_prompts_valid"] = False
            details["error"] = f"{type(error).__name__}:{error}"
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "preflight_version": RUNNER_VERSION,
        "decision": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "details": details,
        "api_calls": 0,
        "run_root_created": run_root.exists(),
    }


def run(
    *, prior_round_root: Path, run_root: Path, round_index: int,
    api_config: Path, tokenizer_path: Path, solver_model: Path,
    preflight_only: bool = False,
) -> dict[str, Any]:
    frozen = preflight(
        prior_round_root=prior_round_root, run_root=run_root,
        round_index=round_index, api_config=api_config,
        tokenizer_path=tokenizer_path, solver_model=solver_model)
    if preflight_only:
        return frozen
    if frozen["decision"] != "pass":
        raise RuntimeError("ADDITIONAL_ROUND_PRE_API_PREFLIGHT_FAILED:" +
                           ",".join(frozen["failed_checks"]))

    run_root.mkdir(parents=True, exist_ok=False)
    round_root = run_root / f"round_{round_index:02d}"
    for relative in ("planner", "realizer", "attempts", "raw/prompts", "reports"):
        (round_root / relative).mkdir(parents=True, exist_ok=True)
    (run_root / "reports").mkdir(parents=True, exist_ok=True)
    atomic_json(run_root / "run_freeze_manifest.json", frozen)

    expanded_path = prior_round_root / "expanded_dialogue.json"
    prior_qa_path = prior_round_root / "qa_evaluation/reports/ROUND_QA_SOLVER_REPORT.json"
    current = _load(expanded_path)
    source_manifest_sha = frozen["details"]["source_manifest"]["manifest_sha256"]
    prior_tree_sha = frozen["details"]["prior_round_tree_sha256"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    solver_backend = VLLMSolverBackend(
        model_path=solver_model, tokenizer_path=tokenizer_path,
        max_model_len=SOLVER_MAX_MODEL_LEN, temperature=0.7, top_p=0.9,
        max_tokens=SOLVER_MAX_TOKENS,
        gpu_memory_utilization=SOLVER_GPU_MEMORY_UTILIZATION)

    def before_request(logical_role: str, role_version: str) -> None:
        del logical_role, role_version
        if _source_manifest()["manifest_sha256"] != source_manifest_sha:
            raise RuntimeError("additional_round_source_manifest_drift")
        if _hash_tree(prior_round_root) != prior_tree_sha:
            raise RuntimeError("additional_round_prior_input_drift")

    paid_requests = 0
    try:
        tail = build_tail_context_from_expanded_dialogue(current, session_count=2)
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
            raise RuntimeError("additional_round_planner_contract_failed")
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
            raise RuntimeError("additional_round_realizer_contract_failed")

        next_expanded = append_expansion_round(
            prior_expanded=current, child=normalized["child"])
        appended = validate_appended_round(next_expanded, prior_expanded=current)
        if not appended["valid"]:
            raise RuntimeError("additional_round_append_failed")
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
            "prior_complete_dialogue_sha256": current["complete_dialogue_sha256"],
            "complete_dialogue_sha256": next_expanded["complete_dialogue_sha256"],
            "turn_count": next_expanded["turn_count"],
            "added_turn_count": next_expanded["turn_count"] - current["turn_count"],
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
        }
        report = {
            "report_version": RUNNER_VERSION,
            "status": "completed",
            "failure": None,
            "round_metric": metric,
            "paid_api_requests": paid_requests,
            "prior_round_unchanged": _hash_tree(prior_round_root) == prior_tree_sha,
            "memory_calls": 0,
            "semantic_auditor_calls": 0,
            "qa_pipeline_ready": bool(qa_report["qa_pipeline_ready"]),
            "current_round_qa_ready": bool(
                qa_report["current_round_qa_ready"]),
            "training_ready": False,
        }
        atomic_json(run_root / "reports/ADDITIONAL_ROUND_REPORT.json", report)
        atomic_json(round_root / "reports/ROUND_SUMMARY.json", metric)
        return report
    except Exception as error:
        report = {
            "report_version": RUNNER_VERSION,
            "status": "failed",
            "failure": f"{type(error).__name__}:{error}",
            "round_index": round_index,
            "paid_api_requests": paid_requests,
            "prior_round_unchanged": _hash_tree(prior_round_root) == prior_tree_sha,
            "memory_calls": 0,
            "semantic_auditor_calls": 0,
            "qa_pipeline_ready": False,
            "current_round_qa_ready": False,
            "training_ready": False,
        }
        atomic_json(round_root / "reports/ROUND_FAILURE.json", report)
        atomic_json(run_root / "reports/ADDITIONAL_ROUND_REPORT.json", report)
        return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append and evaluate exactly one round")
    parser.add_argument("--prior-round-root", type=Path,
                        default=DEFAULT_PRIOR_ROUND_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--round-index", type=int, default=DEFAULT_ROUND_INDEX)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    report = run(
        prior_round_root=args.prior_round_root.resolve(),
        run_root=args.run_root.resolve(), round_index=args.round_index,
        api_config=args.api_config.resolve(), tokenizer_path=args.tokenizer.resolve(),
        solver_model=args.solver_model.resolve(),
        preflight_only=args.preflight_only)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (report.get("decision") == "pass" if args.preflight_only else
                 report.get("status") == "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
