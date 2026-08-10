"""Generate and evaluate a diagnostic batch of MCQs from frozen v11 core QAs."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..contrastive_distractor_canary import (
    build_contrastive_distractor_prompt, materialize_contrastive_candidate,
)
from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..lineage_distractor_writer import validate_distractor_object
from ..lineage_three_stage_canary import ApiBudget, run_single_object_task
from ..live_backends import (LiveSemanticRepairBackend,
                             OpenAICompatibleChallengerClient,
                             load_challenger_config)
from ..orchestrator import CanaryOrchestrator
from ..run_cross_dialogue_calibration import _solver_config
from ..vllm_backends import VLLMSolverBackend
from .live_mcq_canary_1 import DEFAULT_API_CONFIG, DEFAULT_TOKENIZER
from .mcq_option_diagnostics import diagnose_option_shape
from .provenance import git_value, hash_tree, project_diff_sha256
from .run_mcq_spell_filter import (DEFAULT_SOLVER_MODEL,
                                   SOLVER_GPU_MEMORY_UTILIZATION,
                                   SOLVER_MAX_MODEL_LEN,
                                   SOLVER_MAX_NEW_TOKENS)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNS_ROOT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_SOURCE_ROOTS = tuple(
    RUNS_ROOT /
    f"conversation_chronicles_episode_61555_slim_v11_canary34_qa_canary_{index}"
    for index in (2, 3, 4, 5))
DEFAULT_RUN_ROOT = RUNS_ROOT / (
    "conversation_chronicles_episode_61555_slim_v11_canary34_"
    "mcq_batch_diagnostic_1")
BATCH_REPLICATES_PER_QA = 2
BATCH_SIZE = len(DEFAULT_SOURCE_ROOTS) * BATCH_REPLICATES_PER_QA
DISTRACTOR_MAX_TOKENS = 2048
BATCH_VERSION = "slim_v11_mcq_option_shape_batch_diagnostic_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _inputs(source_roots: tuple[Path, ...]) -> dict[str, Any]:
    candidates = []
    for source_index, root in enumerate(source_roots, 1):
        qa = _load(root / "qa/core_qa.json")
        summary = _load(root / "reports/QA_CANARY_SUMMARY.json")
        candidates.append({"source_index":source_index, "source_root":root,
                           "qa":qa, "summary":summary})
    source_view = _load(source_roots[-1] / "frozen/qa_source_view.json")
    catalog = _load(source_roots[-1] / "frozen/span_catalog.json")
    lineage_view = {"view_sha256":source_view["view_sha256"],
                    "combined_dialogue":source_view["combined_dialogue"],
                    "span_catalog":catalog}
    return {"candidates":candidates, "source_view":source_view,
            "span_catalog":catalog, "lineage_view":lineage_view}


def preflight(
    *, source_roots: tuple[Path, ...], run_root: Path, api_config: Path,
    tokenizer_path: Path, solver_model: Path,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "run_root_absent": not run_root.exists(),
        "four_source_roots": len(source_roots) == 4,
        "api_config_exists": api_config.is_file(),
        "tokenizer_exists": tokenizer_path.is_dir(),
        "solver_model_exists": solver_model.is_dir(),
    }
    for index, root in enumerate(source_roots, 1):
        checks[f"source_{index}_exists"] = root.is_dir()
        checks[f"source_{index}_qa_exists"] = (root / "qa/core_qa.json").is_file()
        checks[f"source_{index}_summary_exists"] = (
            root / "reports/QA_CANARY_SUMMARY.json").is_file()
    checks["source_view_exists"] = (
        source_roots[-1] / "frozen/qa_source_view.json").is_file()
    checks["span_catalog_exists"] = (
        source_roots[-1] / "frozen/span_catalog.json").is_file()
    details: dict[str, Any] = {}
    if all(checks.values()):
        values = _inputs(source_roots)
        catalog_ids = {row["span_id"] for row in
                       values["span_catalog"].get("spans") or []}
        questions = []
        for index, candidate in enumerate(values["candidates"], 1):
            qa = candidate["qa"]
            support = qa.get("answer_support_span_ids") or []
            checks[f"source_{index}_mechanically_ready"] = (
                candidate["summary"].get("status") == "completed" and
                (candidate["summary"].get("readiness") or {}).get(
                    "core_qa_ready") is True)
            checks[f"source_{index}_support_exists"] = (
                bool(support) and all(span_id in catalog_ids for span_id in support))
            questions.append(qa.get("question"))
        checks["questions_are_distinct"] = len(set(questions)) == 4
        checks["complete_dialogue_92_turns"] = len(
            values["source_view"]["combined_dialogue"]) == 92
        details = {"batch_size":BATCH_SIZE,
            "replicates_per_qa":BATCH_REPLICATES_PER_QA,
            "questions":questions,
            "source_trees":[hash_tree(root) for root in source_roots],
            "combined_dialogue_turn_count":len(
                values["source_view"]["combined_dialogue"]),
            "span_count":len(values["span_catalog"].get("spans") or [])}
    failed = sorted(key for key, value in checks.items() if not value)
    return {"preflight_version":BATCH_VERSION, "checked_at":_now(),
            "decision":"pass" if not failed else "fail",
            "checks":checks, "failed_checks":failed, "details":details,
            "api_calls":0, "run_root_created":run_root.exists()}


def _report_markdown(report: dict[str, Any]) -> str:
    lines = ["# MCQ Option-shape Batch Diagnostic", "",
        f"- Batch size: `{report['batch_size']}`",
        f"- MCQs materialized: `{report['materialized_count']}`",
        f"- Single-mutation stars: `{report['single_mutation_star_count']}`",
        f"- Correct unique medoids: `{report['correct_unique_medoid_count']}`",
        f"- Any answer-shape leakage risk: `{report['answer_shape_leakage_risk_count']}`",
        f"- Paid distractor requests: `{report['api_statistics']['paid_distractor_requests']}`",
        f"- Local Solver generations: `{report['api_statistics']['local_solver_generations']}`",
        "", "## Items", ""]
    for item in report["items"]:
        solver = item.get("balanced_solver") or {}
        lines.extend([
            f"### {item['item_id']}", "",
            f"- Question: {item.get('question')}",
            f"- Gold letter: `{item.get('reference_answer_letter')}`",
            f"- Single-mutation star: `{item.get('option_shape', {}).get('single_mutation_star')}`",
            f"- Correct unique medoid: `{item.get('option_shape', {}).get('correct_unique_medoid')}`",
            f"- Balanced grounding: `{solver.get('grounding_decision')}`",
            f"- No-context correct: `{(solver.get('balanced_grounding') or {}).get('semantic_correct_count')}/12`",
            ""])
    return "\n".join(lines)


def run(
    *, source_roots: tuple[Path, ...], run_root: Path, api_config_path: Path,
    tokenizer_path: Path, solver_model: Path, preflight_only: bool = False,
    no_api: bool = False,
) -> dict[str, Any]:
    frozen = preflight(source_roots=source_roots, run_root=run_root,
                       api_config=api_config_path,
                       tokenizer_path=tokenizer_path,
                       solver_model=solver_model)
    if preflight_only or no_api:
        return frozen
    if frozen["decision"] != "pass":
        raise RuntimeError("PRE_API_PREFLIGHT_FAILED:" + ",".join(
            frozen["failed_checks"]))
    values = _inputs(source_roots)
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("config", "frozen", "qa", "reports", "raw/prompts"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    api = replace(load_challenger_config(api_config_path), timeout_seconds=600,
                  temperature=0.25, enable_thinking=False)
    freeze = {"freeze_version":BATCH_VERSION, "created_at":_now(),
        "run_root":str(run_root), "batch_size":BATCH_SIZE,
        "replicates_per_qa":BATCH_REPLICATES_PER_QA,
        "source_roots":[str(root) for root in source_roots],
        "source_tree_sha256":[row["tree_sha256"]
                              for row in frozen["details"]["source_trees"]],
        "model_configuration":{"model":api.model, "base_url":api.base_url,
            "temperature":api.temperature, "timeout_seconds":api.timeout_seconds,
            "max_tokens":DISTRACTOR_MAX_TOKENS, "api_key_recorded":False},
        "request_budget":{"distractor_normal":BATCH_SIZE,
            "format_retry":BATCH_SIZE, "physical_maximum":BATCH_SIZE * 2,
            "qa_writer":0, "full_context_api_solver":0},
        "project":{"git_head":git_value(PROJECT_ROOT, "rev-parse", "HEAD"),
                   "diff_sha256":project_diff_sha256(PROJECT_ROOT)},
        "preflight":frozen}
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "run_status.json", {
        "status":"initialized", "created_at":_now(), "api_calls":0})
    atomic_json(run_root / "frozen/questioner_smoke_context.json", {
        "combined_dialogue":values["source_view"]["combined_dialogue"]})
    atomic_json(run_root / "frozen/questioner_span_catalog.json",
                values["span_catalog"])
    ledger = RequestLedger(run_root, sha256_file(
        run_root / "run_freeze_manifest.json"), "live")
    client = OpenAICompatibleChallengerClient(api)
    budget = ApiBudget(normal_caps={"distractor_writer":BATCH_SIZE},
                       format_retry_cap=BATCH_SIZE,
                       physical_cap=BATCH_SIZE * 2)
    items: list[dict[str, Any]] = []
    placements = "ABCDABCD"
    batch_index = 0
    for candidate in values["candidates"]:
        for replicate in range(BATCH_REPLICATES_PER_QA):
            batch_index += 1
            item_id = f"batch_item_{batch_index:02d}"
            base_qa = dict(candidate["qa"])
            base_qa["qa_id"] = (
                f"cc_train_episode_61555_round_1_{item_id}")
            prompt = build_contrastive_distractor_prompt(
                frozen_dialogue=values["source_view"]["combined_dialogue"],
                span_catalog=values["span_catalog"], base_qa=base_qa,
                attempt_index=0, structured_feedback=None)
            result = run_single_object_task(
                run_root=run_root, ledger=ledger,
                backend=LiveSemanticRepairBackend(client, DISTRACTOR_MAX_TOKENS),
                budget=budget, stage="distractor_writer",
                logical_request_id=f"{item_id}-distractors",
                semantic_prompt=prompt, lineage_view=values["lineage_view"],
                validator=lambda raw, answer=base_qa["correct_answer_text"]:
                    validate_distractor_object(raw, correct_answer_text=answer),
                schema_retry_decisions=("distractor_schema_invalid",),
                ledger_budget_limit=BATCH_SIZE * 2,
                ledger_total_budget_limit=BATCH_SIZE * 2,
                json_invalid_decision="distractor_json_invalid")
            item: dict[str, Any] = {"item_id":item_id,
                "source_root":str(candidate["source_root"]),
                "source_candidate_index":candidate["source_index"],
                "replicate_index":replicate,
                "question":base_qa["question"],
                "distractor_result":result, "materialized":False}
            if result["accepted"]:
                compiled = materialize_contrastive_candidate(
                    base_qa=base_qa,
                    distractors=result["accepted_validation"]["distractors"],
                    target_letter=placements[batch_index - 1],
                    span_catalog=values["span_catalog"])
                mcq = compiled["materialized_qa"]
                if mcq is not None:
                    shape = diagnose_option_shape(
                        options=mcq["options"],
                        reference_answer_letter=mcq["reference_answer_letter"])
                    item.update({"materialized":True,
                        "reference_answer_letter":mcq["reference_answer_letter"],
                        "option_shape":shape, "mcq_path":f"qa/{item_id}.json"})
                    atomic_json(run_root / f"qa/{item_id}.json", mcq)
            items.append(item)
            atomic_json(run_root / f"reports/{item_id}_generation.json", item)

    materialized = [item for item in items if item["materialized"]]
    config = _solver_config(run_root=run_root)
    config.update({"solver_position_neutral_prompt":True,
                   "balanced_grounding_protocol":"balanced_four_position_k3_v2",
                   "max_local_solver_generations":512})
    manifest = {"run_id":run_root.name,
        "mode":"local_solver_mcq_batch_diagnostic",
        "solver_max_model_len":SOLVER_MAX_MODEL_LEN,
        "source_freeze_manifest_sha256":sha256_file(
            run_root / "run_freeze_manifest.json"),
        "solver_model":str(solver_model.resolve()),
        "solver_tokenizer":str(tokenizer_path.resolve()),
        "grounding_protocol":"balanced_four_position_k3_v2",
        "position_neutral_prompt":True,
        "paid_api_solver_calls":0}
    atomic_json(run_root / "config/resolved_canary_config.json", config)
    atomic_json(run_root / "config/run_manifest.json", manifest)
    if materialized:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True)
        backend = VLLMSolverBackend(
            model_path=solver_model, tokenizer_path=tokenizer_path,
            max_model_len=SOLVER_MAX_MODEL_LEN, temperature=0.7, top_p=0.9,
            max_tokens=SOLVER_MAX_NEW_TOKENS,
            gpu_memory_utilization=SOLVER_GPU_MEMORY_UTILIZATION)
        orchestrator = CanaryOrchestrator(
            run_root=run_root, manifest=manifest, config=config, data={},
            tokenizer=tokenizer, realizer=None, questioner=None,
            solver=backend, backend_mode="local")
        for item in materialized:
            mcq = _load(run_root / item["mcq_path"])
            solver_result = orchestrator.audit_minimal_candidate_balanced_grounding(mcq)
            item["balanced_solver"] = solver_result
            atomic_json(run_root / f"reports/{item['item_id']}_evaluation.json",
                        solver_result)

    completed = ledger.completed()
    paid_states = [row for row in completed if row.get("budget_class") == "paid_api"]
    all_states = list((run_root / "ledger/requests").glob("*.json"))
    local_count = len(all_states) - len(paid_states)
    source_after = [hash_tree(root) for root in source_roots]
    report = {"report_version":BATCH_VERSION, "status":"completed",
        "run_root":str(run_root), "batch_size":BATCH_SIZE,
        "materialized_count":len(materialized),
        "single_mutation_star_count":sum(
            bool((item.get("option_shape") or {}).get("single_mutation_star"))
            for item in materialized),
        "correct_unique_medoid_count":sum(
            bool((item.get("option_shape") or {}).get("correct_unique_medoid"))
            for item in materialized),
        "answer_shape_leakage_risk_count":sum(
            bool((item.get("option_shape") or {}).get("answer_shape_leakage_risk"))
            for item in materialized),
        "grounding_decision_counts":{}, "items":items,
        "api_statistics":{"paid_distractor_requests":len(paid_states),
            "local_solver_generations":local_count,
            "questioner_requests":0, "full_context_api_solver_requests":0},
        "source_integrity":{"before":[row["tree_sha256"] for row in
                                             frozen["details"]["source_trees"]],
                            "after":[row["tree_sha256"] for row in source_after]},
        "training_ready":False}
    for item in materialized:
        decision = (item.get("balanced_solver") or {}).get("grounding_decision")
        report["grounding_decision_counts"][decision] = (
            report["grounding_decision_counts"].get(decision, 0) + 1)
    report["source_integrity"]["unchanged"] = (
        report["source_integrity"]["before"] ==
        report["source_integrity"]["after"])
    atomic_json(run_root / "reports/MCQ_BATCH_DIAGNOSTIC.json", report)
    atomic_text(run_root / "reports/MCQ_BATCH_DIAGNOSTIC.md",
                _report_markdown(report))
    atomic_json(run_root / "run_status.json", {
        "status":"completed", "completed_at":_now(),
        "paid_api_calls":len(paid_states),
        "local_solver_generations":local_count})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-api", action="store_true")
    args = parser.parse_args(argv)
    report = run(source_roots=DEFAULT_SOURCE_ROOTS,
        run_root=args.run_root, api_config_path=args.api_config,
        tokenizer_path=args.tokenizer, solver_model=args.solver_model,
        preflight_only=args.preflight_only, no_api=args.no_api)
    print(json.dumps({key:report.get(key) for key in (
        "decision", "status", "batch_size", "materialized_count",
        "single_mutation_star_count", "correct_unique_medoid_count",
        "answer_shape_leakage_risk_count", "grounding_decision_counts",
        "api_statistics")}, ensure_ascii=False, indent=2))
    return 0 if report.get("decision", "pass") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
