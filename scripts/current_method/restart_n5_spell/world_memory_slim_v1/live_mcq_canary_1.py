"""Materialize and full-context-solve one verified expanded-dialogue MCQ."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..contrastive_distractor_canary import build_contrastive_distractor_prompt
from ..core import atomic_json, atomic_text, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..lineage_distractor_writer import validate_distractor_object
from ..lineage_three_stage_canary import ApiBudget, run_single_object_task
from ..live_backends import (LiveSemanticRepairBackend,
                             OpenAICompatibleChallengerClient,
                             load_challenger_config)
from ..single_item_qa_canary import (deterministic_answer_placement,
                                     materialize_single_item_mcq)
from .continuous_expansion_mcq import (
    MCQ_SOLVER_MAX_TOKENS,
    MCQ_SOLVER_PROMPT_VERSION,
    build_full_context_mcq_solver_prompt,
    validate_mcq_solver_output,
)
from .provenance import git_value, hash_tree, project_diff_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_GOLD_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_v11_canary34_qa_semantic_audit_2")
DEFAULT_QA_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_v11_canary34_qa_canary_5")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_v11_canary34_mcq_canary_1")
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DISTRACTOR_MAX_TOKENS = 2048
MODEL_CONTEXT_LIMIT = 40960


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inputs(gold_run: Path, qa_source_run: Path) -> dict[str, Any]:
    gold = _load(gold_run / "qa/admitted_gold.json")
    audit_summary = _load(gold_run / "reports/QA_SEMANTIC_AUDIT_SUMMARY.json")
    source_view = _load(qa_source_run / "frozen/qa_source_view.json")
    span_catalog = _load(qa_source_run / "frozen/span_catalog.json")
    lineage_view = {"combined_dialogue": source_view["combined_dialogue"],
                    "span_catalog": span_catalog,
                    "view_sha256": source_view["view_sha256"]}
    return {"gold": gold, "audit_summary": audit_summary,
            "source_view": source_view, "span_catalog": span_catalog,
            "lineage_view": lineage_view}


def preflight(*, gold_run: Path, qa_source_run: Path, run_root: Path,
              api_config: Path, tokenizer_path: Path) -> dict[str, Any]:
    checks = {"run_root_absent": not run_root.exists(),
              "gold_run_exists": gold_run.is_dir(),
              "qa_source_run_exists": qa_source_run.is_dir(),
              "admitted_gold_exists": (gold_run / "qa/admitted_gold.json").is_file(),
              "audit_summary_exists": (gold_run / "reports/QA_SEMANTIC_AUDIT_SUMMARY.json").is_file(),
              "source_view_exists": (qa_source_run / "frozen/qa_source_view.json").is_file(),
              "span_catalog_exists": (qa_source_run / "frozen/span_catalog.json").is_file(),
              "api_config_exists": api_config.is_file(),
              "tokenizer_exists": tokenizer_path.exists()}
    values = None
    distractor_prompt = solver_fixture_prompt = ""
    if all(checks.values()):
        values = _inputs(gold_run, qa_source_run)
        ready = values["audit_summary"].get("readiness") or {}
        checks.update({
            "semantic_gold_verified": ready.get(
                "semantic_entailment_independently_verified") is True,
            "open_answer_gold_ready": ready.get("open_answer_gold_ready") is True,
            "gold_has_no_options": "options" not in values["gold"],
            "complete_dialogue_turn_count_92": len(
                values["source_view"]["combined_dialogue"]) == 92,
        })
        distractor_prompt = build_contrastive_distractor_prompt(
            frozen_dialogue=values["source_view"]["combined_dialogue"],
            span_catalog=values["span_catalog"], base_qa=values["gold"],
            attempt_index=0, structured_feedback=None)
        fixture = {"question": values["gold"]["question"], "options": {
            "A": "fixture-a", "B": "fixture-b", "C": "fixture-c",
            "D": "fixture-d"}}
        solver_fixture_prompt = build_full_context_mcq_solver_prompt(
            frozen_dialogue=values["source_view"]["combined_dialogue"],
            materialized_qa=fixture)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True)
        distractor_tokens = len(tokenizer.encode(
            distractor_prompt, add_special_tokens=False))
        solver_tokens = len(tokenizer.encode(
            solver_fixture_prompt, add_special_tokens=False))
        checks["distractor_prompt_within_context"] = (
            distractor_tokens + DISTRACTOR_MAX_TOKENS <= MODEL_CONTEXT_LIMIT)
        checks["solver_prompt_within_context"] = (
            solver_tokens + MCQ_SOLVER_MAX_TOKENS <= MODEL_CONTEXT_LIMIT)
    else:
        distractor_tokens = solver_tokens = 0
    errors = [name for name, passed in checks.items() if not passed]
    return {"preflight_version": "slim_v11_verified_gold_mcq_preflight_v1",
            "valid": not errors, "errors": errors, "checks": checks,
            "gold_run": str(gold_run), "qa_source_run": str(qa_source_run),
            "gold_tree": hash_tree(gold_run) if gold_run.is_dir() else None,
            "qa_source_tree": hash_tree(qa_source_run)
                if qa_source_run.is_dir() else None,
            "gold_sha256": sha256_file(gold_run / "qa/admitted_gold.json")
                if (gold_run / "qa/admitted_gold.json").is_file() else None,
            "distractor_prompt": distractor_prompt,
            "distractor_prompt_sha256": sha256_bytes(
                distractor_prompt.encode("utf-8")) if distractor_prompt else None,
            "distractor_prompt_tokens": distractor_tokens,
            "solver_fixture_prompt_tokens": solver_tokens,
            "values": values}


def run(*, gold_run: Path, qa_source_run: Path, run_root: Path,
        api_config_path: Path, tokenizer_path: Path,
        preflight_only: bool = False, no_api: bool = False) -> dict[str, Any]:
    frozen = preflight(gold_run=gold_run, qa_source_run=qa_source_run,
                       run_root=run_root, api_config=api_config_path,
                       tokenizer_path=tokenizer_path)
    public = {key: value for key, value in frozen.items()
              if key not in {"distractor_prompt", "values"}}
    if preflight_only or no_api:
        return public
    if not frozen["valid"]:
        raise RuntimeError(f"PRE_API_PREFLIGHT_FAILED:{frozen['errors']}")
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("frozen", "raw/prompts", "qa", "reports"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    api = replace(load_challenger_config(api_config_path),
                  timeout_seconds=600, temperature=0.25,
                  enable_thinking=False)
    code_before = {"git_head": git_value(PROJECT_ROOT, "rev-parse", "HEAD"),
                   "project_diff_sha256": project_diff_sha256(PROJECT_ROOT),
                   "runner_sha256": sha256_file(Path(__file__))}
    freeze = {"freeze_version": "slim_v11_verified_gold_mcq_freeze_v1",
              "created_at": _now(), "run_root": str(run_root),
              "gold_run": str(gold_run), "qa_source_run": str(qa_source_run),
              "gold_tree_sha256": frozen["gold_tree"]["tree_sha256"],
              "qa_source_tree_sha256": frozen["qa_source_tree"]["tree_sha256"],
              "gold_sha256": frozen["gold_sha256"],
              "distractor_prompt_sha256": frozen["distractor_prompt_sha256"],
              "model_configuration": {"model": api.model,
                  "base_url": api.base_url, "temperature": api.temperature,
                  "timeout_seconds": api.timeout_seconds,
                  "enable_thinking": api.enable_thinking,
                  "api_key_recorded": False},
              "request_budget": {"distractor_writer_normal": 1,
                  "full_context_solver_normal": 1,
                  "format_only_retry_each": 1, "physical_maximum": 4},
              "code_manifest": code_before, "preflight": public}
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "run_status.json", {
        "status": "initialized", "created_at": _now(), "api_calls": 0})
    atomic_json(run_root / "frozen/admitted_core_qa.json",
                frozen["values"]["gold"])
    atomic_json(run_root / "frozen/span_catalog.json",
                frozen["values"]["span_catalog"])
    atomic_text(run_root / "frozen/distractor_prompt.txt",
                frozen["distractor_prompt"])
    ledger = RequestLedger(
        run_root, sha256_file(run_root / "run_freeze_manifest.json"), "live")
    client = OpenAICompatibleChallengerClient(api)
    budget = ApiBudget(normal_caps={"distractor_writer": 1, "mcq_solver": 1},
                       format_retry_cap=2, physical_cap=4)
    distractor_result = run_single_object_task(
        run_root=run_root, ledger=ledger,
        backend=LiveSemanticRepairBackend(client, DISTRACTOR_MAX_TOKENS),
        budget=budget, stage="distractor_writer",
        logical_request_id="verified-gold-distractors-001",
        semantic_prompt=frozen["distractor_prompt"],
        lineage_view=frozen["values"]["lineage_view"],
        validator=lambda raw: validate_distractor_object(
            raw, correct_answer_text=frozen["values"]["gold"][
                "correct_answer_text"]),
        schema_retry_decisions=("distractor_schema_invalid",),
        ledger_budget_limit=4, ledger_total_budget_limit=4,
        json_invalid_decision="distractor_json_invalid")
    materialized = mcq = solver_result = None
    solver_correct = False
    if distractor_result["accepted"]:
        placement = deterministic_answer_placement(
            "cc-train-episode-61555-round-1", 0)
        materialized = materialize_single_item_mcq(
            core_qa=frozen["values"]["gold"],
            distractors=distractor_result["accepted_validation"]["distractors"],
            placement=placement,
            span_catalog=frozen["values"]["span_catalog"])
        mcq = materialized["materialized_qa"]
        atomic_json(run_root / "qa/materialized_mcq.json", mcq)
        atomic_json(run_root / "qa/materialization_record.json", materialized)
        solver_prompt = build_full_context_mcq_solver_prompt(
            frozen_dialogue=frozen["values"]["source_view"]["combined_dialogue"],
            materialized_qa=mcq)
        atomic_text(run_root / "frozen/full_context_solver_prompt.txt",
                    solver_prompt)
        solver_result = run_single_object_task(
            run_root=run_root, ledger=ledger,
            backend=LiveSemanticRepairBackend(client, MCQ_SOLVER_MAX_TOKENS),
            budget=budget, stage="mcq_solver",
            logical_request_id="verified-gold-full-context-solver-001",
            semantic_prompt=solver_prompt,
            lineage_view=frozen["values"]["lineage_view"],
            validator=validate_mcq_solver_output,
            schema_retry_decisions=("mcq_solver_schema_invalid",),
            ledger_budget_limit=4, ledger_total_budget_limit=4,
            json_invalid_decision="mcq_solver_json_invalid")
        solver_letter = ((solver_result.get("accepted_validation") or {}).get(
            "answer_letter") if solver_result.get("accepted") else None)
        solver_correct = solver_letter == mcq["reference_answer_letter"]
        atomic_json(run_root / "reports/full_context_solver_result.json", {
            "solver_result": solver_result,
            "predicted_answer_letter": solver_letter,
            "reference_answer_letter": mcq["reference_answer_letter"],
            "correct": solver_correct})
    mcq_ready = bool(mcq and solver_correct)
    if mcq_ready:
        atomic_json(run_root / "qa/final_mcq.json", mcq)
    completed = ledger.completed()
    usage = [row.get("response_metadata", {}).get("usage") or {}
             for row in completed]
    gold_after, source_after = hash_tree(gold_run), hash_tree(qa_source_run)
    summary = {"report_version": "slim_v11_verified_gold_mcq_report_v1",
        "status": "completed" if mcq_ready else "failed",
        "failure_stage": None if mcq_ready else (
            "full_context_solver" if mcq else "distractor_writer"),
        "final_decision": "A_MCQ_READY" if mcq_ready else "B_MCQ_REJECTED",
        "run_root": str(run_root), "gold_run": str(gold_run),
        "question": mcq.get("question") if mcq else None,
        "options": mcq.get("options") if mcq else None,
        "reference_answer_letter": mcq.get("reference_answer_letter")
            if mcq else None,
        "evidence_span_ids": mcq.get("evidence_span_ids") if mcq else None,
        "distractor_result": distractor_result,
        "solver_result": solver_result,
        "api_statistics": {"distractor_writer_physical_requests":
                distractor_result["physical_request_count"],
            "solver_physical_requests": solver_result["physical_request_count"]
                if solver_result else 0,
            "total_physical_requests": len(completed),
            "prompt_tokens": sum(int(x.get("prompt_tokens") or 0) for x in usage),
            "completion_tokens": sum(int(x.get("completion_tokens") or 0) for x in usage),
            "total_tokens": sum(int(x.get("total_tokens") or 0) for x in usage)},
        "integrity": {"gold_run_unchanged": frozen["gold_tree"]["tree_sha256"] ==
                gold_after["tree_sha256"],
            "qa_source_run_unchanged": frozen["qa_source_tree"]["tree_sha256"] ==
                source_after["tree_sha256"]},
        "readiness": {"semantic_core_qa_ready": True,
            "options_ready": mcq is not None,
            "full_context_solver_correct": solver_correct,
            "mcq_ready": mcq_ready, "qa_pipeline_ready": mcq_ready,
            "training_ready": False}}
    atomic_json(run_root / "reports/MCQ_CANARY_SUMMARY.json", summary)
    lines = ["# Expanded-dialogue MCQ Canary", "",
             f"- Decision: `{summary['final_decision']}`", "",
             f"**Question:** {summary.get('question') or ''}", ""]
    for letter, option in (summary.get("options") or {}).items():
        lines.append(f"- **{letter}.** {option}")
    lines.extend(["", f"Reference answer: `{summary.get('reference_answer_letter')}`",
                  f"Full-context Solver correct: `{str(solver_correct).lower()}`", ""])
    atomic_text(run_root / "reports/MCQ_CANARY_SUMMARY.md", "\n".join(lines))
    atomic_json(run_root / "run_status.json", {"status": summary["status"],
        "completed_at": _now(), "api_calls": len(completed),
        "final_decision": summary["final_decision"]})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-run", type=Path, default=DEFAULT_GOLD_RUN)
    parser.add_argument("--qa-source-run", type=Path, default=DEFAULT_QA_SOURCE_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-api", action="store_true")
    args = parser.parse_args(argv)
    result = run(gold_run=args.gold_run, qa_source_run=args.qa_source_run,
                 run_root=args.run_root, api_config_path=args.api_config,
                 tokenizer_path=args.tokenizer,
                 preflight_only=args.preflight_only, no_api=args.no_api)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("valid", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
