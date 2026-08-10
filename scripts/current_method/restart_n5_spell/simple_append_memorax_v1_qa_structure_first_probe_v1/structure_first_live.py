"""One-call structure/evidence-first QA probe over one frozen dialogue."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..live_backends import (
    OpenAICompatibleChallengerClient, load_challenger_config,
)
from ..simple_append_v1.live_runtime import load_json_object_strict
from ..simple_append_v1.qa_live import (
    CONTEXT_LIMIT,
    DEFAULT_API_CONFIG,
    DEFAULT_SOLVER_MODEL,
    DEFAULT_TOKENIZER,
    _evaluate_solver_candidates,
    build_solver_source,
)
from .integrity import initial_git_status, protected_manifests
from .qa_admission import classify_qa_admission_tier
from .structure_first_contract import (
    QUESTION_CAPACITY_MAX,
    STRUCTURE_FIRST_SCHEMA_VERSION,
    solver_partition,
    validate_structure_first_writer_output,
)
from .structure_first_prompts import build_structure_first_prompt


RUNNER_VERSION = "structure_evidence_first_qa_probe_live_v1"
# This branch owns the Writer completion budget.  The shared QA runtime's
# budget remains unchanged for every other caller.
WRITER_COMPLETION_TOKENS = 16_384
EXPECTED_EXPANDED_SHA256 = (
    "97cf7b5aef3b9e6b7a04e9323f015d7d3cbef6f82fc985da97b89cbc70f0f7f5")
EXPECTED_MEMORY_OVERVIEW_SHA256 = (
    "0a494b2495b77cf1e4d54fd006cef45dee5e905d8debe637f786902fecdbb962")
INPUT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_structural_"
    "v19_2_budget16384_probe_v1_live_1")
DEFAULT_EXPANDED = INPUT_RUN_ROOT / "frozen/expanded_dialogue.json"
DEFAULT_MEMORY_OVERVIEW = INPUT_RUN_ROOT / "frozen/memory_overview.json"
EXPERIMENT_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_"
    "simple_append_memorax_fullr1_devdiv_probe_v1_experiment_1")
DEFAULT_BASELINE_REPORT = EXPERIMENT_ROOT / "qa/reports/MEMORAX_QA_LIVE_REPORT.json"
DEFAULT_BASELINE_BATCH = EXPERIMENT_ROOT / "qa/qa/final_mcq_batch.json"
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_"
    "qa_structure_first_probe_v1_live_1")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _writer_call(
    *, run_root: Path, ledger: RequestLedger, prompt: Mapping[str, Any],
    api_config_path: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    api = replace(load_challenger_config(api_config_path),
                  timeout_seconds=600, temperature=0.25,
                  enable_thinking=False)
    client = OpenAICompatibleChallengerClient(api)
    request_id = "structure-first-qa-writer-v1-a00"
    atomic_text(run_root / "raw/prompts" / f"{request_id}.txt",
                str(prompt["prompt"]))
    response = ledger.invoke(
        request_id=request_id, intent="structure_evidence_first_qa_writer_v1",
        attempt=1,
        prompt_sha256=str(prompt["prompt_sha256"]),
        call=lambda: client.complete(
            prompt=str(prompt["prompt"]), max_tokens=WRITER_COMPLETION_TOKENS,
            json_object=True),
        budget_class="mcq_writer", budget_limit=1, total_budget_limit=1,
        request_metadata={
            "role": "structure_evidence_first_qa_writer_v1",
            "question_capacity_max": QUESTION_CAPACITY_MAX,
            "retry_allowed": False,
        },
        backend_identity={
            "transport": "openai_compatible", "model": api.model,
            "base_url": api.base_url, "temperature": api.temperature,
        },
    )
    try:
        parsed = load_json_object_strict(response.raw_text)
        parse_error = None
    except (json.JSONDecodeError, ValueError) as error:
        parsed = None
        parse_error = f"{type(error).__name__}:{error}"
    return parsed, {
        "request_id": request_id,
        "parse_error": parse_error,
        "model": response.model_identifier,
        "usage": ((response.metadata or {}).get("usage") or {}),
        "response_metadata": response.metadata,
    }


def preflight(
    *, expanded_path: Path = DEFAULT_EXPANDED,
    memory_overview_path: Path = DEFAULT_MEMORY_OVERVIEW,
    baseline_report_path: Path = DEFAULT_BASELINE_REPORT,
    baseline_batch_path: Path = DEFAULT_BASELINE_BATCH,
    run_root: Path = DEFAULT_RUN_ROOT,
    api_config_path: Path = DEFAULT_API_CONFIG,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    solver_model: Path = DEFAULT_SOLVER_MODEL,
) -> dict[str, Any]:
    repo = _repo_root()
    protected = protected_manifests(repo)
    checks = {
        "run_root_absent": not run_root.exists(),
        "expanded_dialogue_exists": expanded_path.is_file(),
        "memory_overview_exists": memory_overview_path.is_file(),
        "baseline_report_exists": baseline_report_path.is_file(),
        "baseline_batch_exists": baseline_batch_path.is_file(),
        "api_config_exists": api_config_path.is_file(),
        "tokenizer_exists": tokenizer_path.is_dir(),
        "solver_model_exists": solver_model.is_dir(),
        "protected_modules_unchanged": protected["unchanged"],
        "api_calls_zero": True,
        "expansion_calls_zero": True,
        "memorax_calls_zero": True,
    }
    details: dict[str, Any] = {"protected": protected}
    if all(checks.values()):
        try:
            expanded = _load(expanded_path)
            overview = _load(memory_overview_path)
            baseline = _load(baseline_report_path)
            prompt = build_structure_first_prompt(expanded, overview)
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=True)
            prompt_tokens = len(tokenizer.encode(
                str(prompt["prompt"]), add_special_tokens=False))
            checks.update({
                "expanded_sha_frozen": (
                    sha256_file(expanded_path) == EXPECTED_EXPANDED_SHA256),
                "memory_overview_sha_frozen": (
                    sha256_file(memory_overview_path) ==
                    EXPECTED_MEMORY_OVERVIEW_SHA256),
                "root_identity": (
                    expanded.get("root_id") == "cc-train-episode-146700" and
                    overview.get("root_id") == expanded.get("root_id")),
                "complete_dialogue_frozen": (
                    int(expanded.get("turn_count") or 0) == 110),
                "baseline_is_expected_14_zero": (
                    int(baseline.get("question_count") or 0) == 14 and
                    int(baseline.get("admitted_count") or 0) == 0),
                "writer_prompt_fits": (
                    prompt_tokens + WRITER_COMPLETION_TOKENS <= CONTEXT_LIMIT),
                "writer_request_budget_one": True,
                "writer_retry_budget_zero": True,
                "structure_first_schema_active": (
                    STRUCTURE_FIRST_SCHEMA_VERSION in str(prompt["prompt"])),
                "evidence_first_order_active": (
                    str(prompt["prompt"]).find("Lock 2-6 minimal sufficient") <
                    str(prompt["prompt"]).find("Only then materialize")),
                "solver_contract_frozen": True,
            })
            api = load_challenger_config(api_config_path)
            details.update({
                "expanded_dialogue_path": str(expanded_path.resolve()),
                "expanded_dialogue_sha256": sha256_file(expanded_path),
                "memory_overview_path": str(memory_overview_path.resolve()),
                "memory_overview_sha256": sha256_file(memory_overview_path),
                "baseline_report_sha256": sha256_file(baseline_report_path),
                "baseline_batch_sha256": sha256_file(baseline_batch_path),
                "writer_prompt_sha256": prompt["prompt_sha256"],
                "writer_prompt_tokens": prompt_tokens,
                "writer_max_output_tokens": WRITER_COMPLETION_TOKENS,
                "context_limit": CONTEXT_LIMIT,
                "model": api.model,
                "backend": api.base_url,
                "temperature": 0.25,
                "solver_model": str(solver_model.resolve()),
                "tokenizer": str(tokenizer_path.resolve()),
                "run_root": str(run_root.resolve()),
                "request_budget": {
                    "qa_writer": 1, "planner": 0, "realizer": 0,
                    "memorax": 0, "semantic_auditor": 0,
                },
            })
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
        "api_calls": 0,
    }


def _write_preflight_artifacts(report: Mapping[str, Any]) -> None:
    module_reports = Path(__file__).resolve().parent / "reports"
    module_reports.mkdir(parents=True, exist_ok=True)
    atomic_json(module_reports / "STRUCTURE_FIRST_QA_PREFLIGHT.json", report)
    atomic_json(module_reports / "PROTECTED_BASELINE_MANIFEST.json",
                report.get("details", {}).get("protected", {}))
    atomic_text(module_reports / "INITIAL_GIT_STATUS.txt",
                initial_git_status(_repo_root()))


def _question_audit(
    raw: Mapping[str, Any] | None,
    validation: Mapping[str, Any],
    evaluated: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join Writer discovery fields, structural outcome, and frozen Solver result."""
    raw_questions = list((raw or {}).get("questions") or [])
    diagnostics = list(validation.get("diagnostics") or [])
    mapping = {
        str(row.get("raw_candidate_id") or ""):
        str(row.get("materialized_candidate_id") or "")
        for row in validation.get("materialization_map") or []
    }
    evaluations = {str(row.get("qa_id") or ""): row for row in evaluated}
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(raw_questions):
        candidate = candidate if isinstance(candidate, Mapping) else {}
        plan = candidate.get("structure_plan")
        plan = plan if isinstance(plan, Mapping) else {}
        uniqueness = plan.get("gold_uniqueness")
        uniqueness = uniqueness if isinstance(uniqueness, Mapping) else {}
        diagnostic = diagnostics[index] if index < len(diagnostics) else {}
        candidate_id = str(candidate.get("candidate_id") or "")
        qa_id = mapping.get(candidate_id)
        evaluation = evaluations.get(str(qa_id or ""), {})
        rows.append({
            "candidate_id": candidate_id,
            "materialized_qa_id": qa_id,
            "question": candidate.get("question"),
            "question_type": candidate.get("question_type"),
            "structure_type": plan.get("structure_type"),
            "discriminative_variable": plan.get("discriminative_variable"),
            "evidence_count": len(plan.get("local_evidence_turn_ids") or []),
            "gold_uniqueness": uniqueness,
            "distractor_sources": plan.get("distractor_sources") or [],
            "structural_valid": bool(diagnostic.get("valid")),
            "structural_errors": diagnostic.get("structural_errors") or [],
            "structural_warnings": diagnostic.get("warnings") or [],
            "no_context": evaluation.get("no_context"),
            "grounding": evaluation.get("grounding"),
            "full_context_correct_count": evaluation.get(
                "full_context_correct_count"),
            "full_context_bucket": evaluation.get("full_context_bucket"),
            "admitted": bool(evaluation.get("admitted")),
        })
    return rows


def run(
    *, expanded_path: Path = DEFAULT_EXPANDED,
    memory_overview_path: Path = DEFAULT_MEMORY_OVERVIEW,
    baseline_report_path: Path = DEFAULT_BASELINE_REPORT,
    baseline_batch_path: Path = DEFAULT_BASELINE_BATCH,
    run_root: Path = DEFAULT_RUN_ROOT,
    api_config_path: Path = DEFAULT_API_CONFIG,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    solver_model: Path = DEFAULT_SOLVER_MODEL,
) -> dict[str, Any]:
    git_status_before = initial_git_status(_repo_root())
    frozen = preflight(
        expanded_path=expanded_path,
        memory_overview_path=memory_overview_path,
        baseline_report_path=baseline_report_path,
        baseline_batch_path=baseline_batch_path,
        run_root=run_root, api_config_path=api_config_path,
        tokenizer_path=tokenizer_path, solver_model=solver_model)
    if frozen["decision"] != "pass":
        raise RuntimeError("STRUCTURE_FIRST_QA_PRE_API_PREFLIGHT_FAILED:" +
                           ",".join(frozen["failed_checks"]))

    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("frozen", "qa", "filter", "reports", "raw/prompts"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    expanded = _load(expanded_path)
    overview = _load(memory_overview_path)
    baseline = _load(baseline_report_path)
    prompt = build_structure_first_prompt(expanded, overview)
    freeze = {
        **frozen,
        "freeze_version": RUNNER_VERSION,
        "frozen_at": _now(),
        "fresh": True,
        "no_resume": True,
        "writer_prompt_sha256": prompt["prompt_sha256"],
        "source_input_read_only": True,
        "solver_contract_changed": False,
        "writer_calls_max": 1,
        "writer_retries": 0,
        "git_status_porcelain_before": git_status_before,
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "frozen/expanded_dialogue.json", expanded)
    atomic_json(run_root / "frozen/memory_overview.json", overview)
    atomic_json(run_root / "frozen/protected_before.json",
                frozen["details"]["protected"])
    atomic_json(run_root / "frozen/baseline_qa_report.json", baseline)
    atomic_text(run_root / "qa/writer_prompt.txt", str(prompt["prompt"]))
    atomic_json(run_root / "qa/writer_prompt_manifest.json",
                {key: value for key, value in prompt.items() if key != "prompt"})
    atomic_json(run_root / "run_status.json", {
        "status": "initialized", "api_calls": 0,
        "created_at": _now(),
    })

    manifest_sha = sha256_file(run_root / "run_freeze_manifest.json")
    ledger = RequestLedger(run_root, manifest_sha, "live")
    failure: str | None = None
    raw_payload: dict[str, Any] | None = None
    writer_meta: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    evaluated: list[dict[str, Any]] = []
    materialized: list[dict[str, Any]] = []
    try:
        raw_payload, writer_meta = _writer_call(
            run_root=run_root, ledger=ledger, prompt=prompt,
            api_config_path=api_config_path)
        atomic_json(run_root / "qa/writer_response_parsed.json", raw_payload)
        if raw_payload is None:
            raise RuntimeError("structure_first_writer_parse_failed")
        validation = validate_structure_first_writer_output(
            raw_payload, expanded_dialogue=expanded)
        atomic_json(run_root / "qa/writer_structure_first_validation.json", validation)
        atomic_json(run_root / "qa/writer_structure_first_diagnostics.json", {
            "schema_version": RUNNER_VERSION,
            "items": validation.get("diagnostics") or [],
            "abstention_summary": raw_payload.get("abstention_summary"),
        })
        if not validation.get("envelope_valid"):
            raise RuntimeError("structural_qa_writer_envelope_invalid")
        normalized = validation["normalized_batch"]
        atomic_json(run_root / "qa/writer_output_normalized.json", normalized)
        if validation["structural_valid_count"]:
            source = build_solver_source(expanded)
            partition = solver_partition(validation)
            evaluated, materialized = _evaluate_solver_candidates(
            raw=normalized, partition=partition, expanded=expanded,
                source=source, run_root=run_root,
                manifest_sha=manifest_sha, tokenizer_path=tokenizer_path,
                solver_model=solver_model)
            for item in evaluated:
                item.update(classify_qa_admission_tier(item))
                if item.get("qa_id"):
                    atomic_json(run_root / f"filter/{item['qa_id']}.json", item)
                    atomic_json(
                        run_root / f"reports/{item['qa_id']}_evaluation.json", item)
        else:
            atomic_json(run_root / "qa/final_mcq_batch.json", {
                "batch_version": RUNNER_VERSION,
                "root_id": expanded["root_id"], "items": [],
            })
    except Exception as error:
        failure = f"{type(error).__name__}:{error}"

    admitted_ids = [str(row["qa_id"]) for row in evaluated
                    if row.get("admitted")]
    atomic_json(run_root / "qa/admitted_mcq_batch.json", {
        "batch_version": RUNNER_VERSION,
        "items": [row for row in materialized
                  if str(row.get("qa_id")) in admitted_ids],
    })
    before = frozen["details"]["protected"]
    after = protected_manifests(_repo_root())
    protected_same_by_path = {
        key: (before["manifests"][key]["tree_sha256"] ==
              after["manifests"][key]["tree_sha256"] ==
              after["expected"][key])
        for key in before.get("manifests") or {}}
    protected_same = (
        before.get("unchanged") is True and after.get("unchanged") is True and
        all(protected_same_by_path.values()))
    git_status_after = initial_git_status(_repo_root())
    source_same = sha256_file(expanded_path) == EXPECTED_EXPANDED_SHA256
    memory_same = (
        sha256_file(memory_overview_path) == EXPECTED_MEMORY_OVERVIEW_SHA256)
    atomic_json(run_root / "reports/protected_after.json", after)

    request_states = [
        _load(path) for path in (run_root / "ledger/requests").glob("*.json")]
    writer_requests = [row for row in request_states
                       if row.get("budget_class") == "mcq_writer"]
    solver_requests = [row for row in request_states
                       if row.get("budget_class") == "local_solver"]
    grounding_counts: dict[str, int] = {}
    full_buckets: dict[str, int] = {}
    for row in evaluated:
        ground = str((row.get("grounding") or {}).get("decision") or "unknown")
        grounding_counts[ground] = grounding_counts.get(ground, 0) + 1
        bucket = row.get("full_context_bucket")
        if bucket:
            full_buckets[str(bucket)] = full_buckets.get(str(bucket), 0) + 1
    old_evidence_counts = [
        len(row.get("focus_round_evidence_turn_refs") or []) +
        len(row.get("earlier_evidence_turn_refs") or [])
        for row in baseline.get("items") or []]
    evidence = validation.get("evidence_statistics") or {}
    per_question = _question_audit(raw_payload, validation, evaluated)
    response_metadata = writer_meta.get("response_metadata") or {}
    report = {
        "report_version": RUNNER_VERSION,
        "status": "completed" if failure is None else "failed",
        "failure": failure,
        "root_id": expanded.get("root_id"),
        "run_root": str(run_root),
        "writer_raw_candidates": int(validation.get("question_count") or 0),
        "writer_estimated_skipped": (
            ((raw_payload or {}).get("abstention_summary") or {}).get(
                "estimated_skipped_count")),
        "structural_valid_candidates": int(
            validation.get("structural_valid_count") or 0),
        "structural_invalid_candidates": int(
            validation.get("structural_invalid_count") or 0),
        "materialization_map": validation.get("materialization_map") or [],
        "exact_duplicate_count": sum(
            any(str(error).startswith("exact_duplicate_overflow:")
                for error in row.get("structural_errors") or [])
            for row in validation.get("items") or []),
        "evidence_statistics": evidence,
        "baseline_evidence_statistics": {
            "question_count": len(old_evidence_counts),
            "min": min(old_evidence_counts) if old_evidence_counts else None,
            "max": max(old_evidence_counts) if old_evidence_counts else None,
            "mean": (sum(old_evidence_counts) / len(old_evidence_counts)
                     if old_evidence_counts else None),
            "evidence_too_many_count": sum(
                "evidence_too_many" in (row.get("evidence_issues") or [])
                for row in baseline.get("items") or []),
        },
        "question_type_distribution": validation.get(
            "question_type_distribution") or {},
        "discriminative_variable_distribution": validation.get(
            "discriminative_variable_distribution") or {},
        "child_supported_count": int(
            validation.get("child_supported_count") or 0),
        "nonempty_batch_has_child_support": validation.get(
            "nonempty_batch_has_child_support"),
        "solver_evaluated_count": len(evaluated),
        "grounding_decisions": grounding_counts,
        "full_context_buckets": full_buckets,
        "admitted_count": len(admitted_ids),
        "admitted_qa_ids": admitted_ids,
        "per_question_audit": per_question,
        "writer_finish_reason": response_metadata.get("finish_reason"),
        "writer_parse_success": raw_payload is not None,
        "writer_usage": writer_meta.get("usage") or {},
        "api_statistics": {
            "qa_writer_calls": len(writer_requests),
            "local_solver_generations": len(solver_requests),
            "expansion_calls": 0, "planner_calls": 0,
            "realizer_calls": 0, "memorax_calls": 0,
            "semantic_auditor_calls": 0,
        },
        "baseline": {
            "writer_questions": 14,
            "no_context_semantic_leakage": 9,
            "passed_grounding_full_context_5_of_5": 5,
            "admitted": 0,
        },
        "safety": {
            "mainline_modified": not protected_same_by_path.get(
                "scripts/current_method/restart_n5_spell/"
                "simple_append_memorax_v1", False),
            "existing_expansion_probe_modified": not protected_same_by_path.get(
                "scripts/current_method/restart_n5_spell/"
                "simple_append_memorax_v1_fullr1_devdiv_probe_v1", False),
            "frozen_expanded_dialogue_modified": not source_same,
            "old_runs_modified": False,
            "old_memory_modified": not memory_same,
            "expansion_api_calls": 0,
            "memory_write_calls": 0,
            "qa_writer_api_calls": len(writer_requests),
            "semantic_auditor_calls": 0,
            "r2_r3_executed": False,
            "training_executed": False,
            "git_commit_push": False,
            "git_status_porcelain_unchanged_during_live": (
                git_status_before == git_status_after),
        },
        "manual_semantic_audit_required": True,
        "writer_calls_max": 1,
        "writer_retries": 0,
        "qa_pipeline_ready": bool(admitted_ids),
        "completed_at": _now(),
    }
    atomic_json(run_root / "reports/STRUCTURE_FIRST_PER_QUESTION_AUDIT.json", {
        "audit_version": RUNNER_VERSION,
        "items": per_question,
    })
    type_binding_errors = sum(
        "structure_type_question_type_mismatch" in
        (row.get("structural_errors") or []) for row in per_question)
    gold_uniqueness_errors = sum(
        "gold_not_uniquely_supported" in
        (row.get("structural_errors") or []) for row in per_question)
    weak_distractor_errors = sum(
        "weak_invented_distractor" in
        (row.get("structural_errors") or []) for row in per_question)
    frontier_count = sum(
        row.get("full_context_correct_count") in {1, 2, 3, 4}
        for row in per_question)
    atomic_json(run_root / "reports/STRUCTURE_FIRST_QA_PROBE_REPORT.json", report)
    lines = [
        "# Structure/Evidence-First QA Probe", "",
        f"- Status: `{report['status']}`",
        f"- Raw / structurally valid / invalid: "
        f"`{report['writer_raw_candidates']} / "
        f"{report['structural_valid_candidates']} / "
        f"{report['structural_invalid_candidates']}`",
        f"- Solver evaluated: `{report['solver_evaluated_count']}`",
        f"- Grounding decisions: `{json.dumps(grounding_counts, sort_keys=True)}`",
        f"- Full-context buckets: `{json.dumps(full_buckets, sort_keys=True)}`",
        f"- Admitted: `{report['admitted_count']}`",
        f"- QA Writer calls: `{len(writer_requests)}`",
        f"- Writer finish reason: `{report['writer_finish_reason']}`",
        f"- Local Solver generations: `{len(solver_requests)}`",
        f"- Validity rate: `{validation.get('validity_rate')}`",
        f"- Evidence statistics: `{json.dumps(evidence, sort_keys=True)}`",
        f"- Structure types: "
        f"`{json.dumps(validation.get('question_type_distribution') or {}, sort_keys=True)}`",
        f"- Type-binding errors: `{type_binding_errors}`",
        f"- Gold non-unique errors: `{gold_uniqueness_errors}`",
        f"- Weak distractor errors: `{weak_distractor_errors}`",
        f"- Full-context frontier (1-4/5): `{frontier_count}`",
        f"- Protected modules unchanged: `{str(protected_same).lower()}`",
        "", "A separate manual audit must classify commonsense leakage, option-level",
        "leakage, direct recall, genuine contrastive reasoning, and unsupported",
        "reasoning structures before drawing a semantic conclusion.", "",
    ]
    atomic_text(run_root / "reports/STRUCTURE_FIRST_QA_PROBE_REPORT.md",
                "\n".join(lines))
    atomic_json(run_root / "run_status.json", {
        "status": report["status"], "failure": failure,
        "qa_writer_calls": len(writer_requests),
        "qa_pipeline_ready": report["qa_pipeline_ready"],
        "completed_at": report["completed_at"],
    })
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-dialogue", type=Path, default=DEFAULT_EXPANDED)
    parser.add_argument("--memory-overview", type=Path,
                        default=DEFAULT_MEMORY_OVERVIEW)
    parser.add_argument("--baseline-report", type=Path,
                        default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--baseline-batch", type=Path,
                        default=DEFAULT_BASELINE_BATCH)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    kwargs = {
        "expanded_path": args.expanded_dialogue.resolve(),
        "memory_overview_path": args.memory_overview.resolve(),
        "baseline_report_path": args.baseline_report.resolve(),
        "baseline_batch_path": args.baseline_batch.resolve(),
        "run_root": args.run_root.resolve(),
        "api_config_path": args.api_config.resolve(),
        "tokenizer_path": args.tokenizer.resolve(),
        "solver_model": args.solver_model.resolve(),
    }
    if args.live:
        report = run(**kwargs)
        ok = report.get("status") == "completed"
    else:
        report = preflight(**kwargs)
        _write_preflight_artifacts(report)
        ok = report.get("decision") == "pass"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
