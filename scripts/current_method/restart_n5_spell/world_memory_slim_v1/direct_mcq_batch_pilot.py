"""Direct MCQ generation and local SPELL filtering for one expanded dialogue.

This pilot intentionally bypasses the per-item QA semantic-auditor stage.  One
writer request owns the question, gold text, distractors, and candidate evidence;
the program owns only reference validation and answer-letter placement.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..lineage_three_stage_canary import ApiBudget, run_single_object_task
from ..live_backends import (
    LiveSemanticRepairBackend,
    OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from ..minimal_admission import (
    SPAN_IDS_ONLY_SCHEMA_PROFILE,
    materialize_spell_style_minimal,
    normalize_minimal_text_v1,
)
from ..orchestrator import CanaryOrchestrator
from ..questioner_contract import span_catalog_sha256
from ..run_cross_dialogue_calibration import _solver_config
from ..vllm_backends import VLLMSolverBackend
from .continuous_expansion_qa import build_qa_source_view, classify_evidence_scope
from .mcq_option_diagnostics import diagnose_option_shape
from .provenance import git_value, hash_tree, project_diff_sha256
from .run_mcq_spell_filter import (
    DEFAULT_SOLVER_MODEL,
    SOLVER_GPU_MEMORY_UTILIZATION,
    SOLVER_MAX_MODEL_LEN,
    SOLVER_MAX_NEW_TOKENS,
    classify_spell_admission,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNS_ROOT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_SOURCE_RUN = RUNS_ROOT / (
    "slim_v11_soft_evidence_scope_two_root_end_to_end_pilot_1/roots/"
    "cc-train-episode-41625/01_expansion")
DEFAULT_RUN_ROOT = RUNS_ROOT / "slim_v11_direct_mcq_ground_filter_live_pilot_1"
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")

PILOT_VERSION = "slim_v11_direct_mcq_ground_filter_pilot_v1"
PROMPT_VERSION = "direct_mcq_writer_child_involved_v1"
MCQ_COUNT = 3
WRITER_MAX_TOKENS = 6144
WRITER_MAX_ATTEMPTS = 2
EVIDENCE_MISMATCH_POLICY = "mark_only"
MODEL_ITEM_FIELDS = frozenset({
    "question", "correct_answer_text", "distractors",
    "answer_support_span_ids",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _imports_semantic_auditor() -> bool:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(str(node.module or ""))
            imported.extend(alias.name for alias in node.names)
    return any("live_qa_semantic_audit" in name for name in imported)


def _answer_letters(root_id: str, count: int = MCQ_COUNT) -> tuple[str, ...]:
    offset = int(sha256_bytes(root_id.encode("utf-8"))[:8], 16) % 4
    return tuple("ABCD"[(offset + index) % 4] for index in range(count))


def build_direct_mcq_batch_prompt(source_view: Mapping[str, Any]) -> str:
    """Render one non-duplicated dialogue/evidence view for the MCQ writer."""
    partition = source_view["partition_by_turn"]
    rows = []
    for span in source_view["span_catalog"]["spans"]:
        rows.append({
            "span_id": span["span_id"],
            "source_partition": partition[span["turn_id"]],
            "session_id": span["session_id"],
            "speaker": span["speaker"],
            "text": span["exact_text"],
        })
    schema = {
        "mcq_candidates": [{
            "question": "string",
            "correct_answer_text": "string",
            "distractors": ["string", "string", "string"],
            "answer_support_span_ids": ["exact span_id", "exact span_id"],
        } for _ in range(MCQ_COUNT)]
    }
    rules = [
        f"Return exactly {MCQ_COUNT} distinct multiple-choice candidates in one JSON object.",
        "Every question must involve the generated Child: its evidence must contain at least one promoted_child span. parent_only questions are forbidden.",
        "Valid scopes are child_only, multi_child, and parent_child. The newest Child turn is not required.",
        "Use 2 to 6 exact span IDs that minimally support the correct answer. Never invent or alter an ID.",
        "The correct answer must be uniquely true in the complete dialogue. Each distractor must be plausible but false or unsupported by that complete dialogue.",
        "Write the three distractors on the same semantic axis and in the same grammatical form as the correct answer. Keep option lengths comparable.",
        "Do not use all/none of the above, meta-commentary, trick wording, or an option that merely restates the question more closely than the others.",
        "Do not reveal an answer letter. The program assigns A/B/C/D positions after validation.",
        "Do not output qa_id, options, answer letters, explanations, reasoning, scope labels, or any fields outside the template.",
        "Use only the frozen complete dialogue below; do not use World Memory or outside knowledge.",
    ]
    return "\n".join([
        "DIRECT MCQ WRITER", f"PROMPT VERSION: {PROMPT_VERSION}", "",
        "RULES", *[f"- {rule}" for rule in rules], "",
        "FROZEN COMPLETE DIALOGUE AND EVIDENCE CATALOG",
        canonical_json(rows), "", "OUTPUT ONLY THIS JSON SHAPE",
        json.dumps(schema, ensure_ascii=False, indent=2),
    ])


def _retry_suffix(_: Any, validation: Mapping[str, Any]) -> str:
    reasons = list(validation.get("reasons") or ["invalid_batch"])
    return "\n".join([
        "DIRECT MCQ CONTRACT RETRY.",
        "Regenerate the complete three-item object for the unchanged dialogue.",
        "Previous compact errors: " + ", ".join(str(row) for row in reasons) + ".",
        "Every item must use 2-6 exact evidence span IDs including at least one promoted_child span.",
        "Return only mcq_candidates. Do not return answer letters, qa_id, options, or explanations.",
    ])


def validate_direct_mcq_batch(
    raw: Any, *, source_view: Mapping[str, Any], root_id: str,
) -> dict[str, Any]:
    """Apply mechanical schema/reference gates, including the Child hard gate."""
    reasons: list[str] = []
    if not isinstance(raw, dict) or set(raw) != {"mcq_candidates"}:
        return {"decision": "direct_mcq_batch_schema_invalid",
                "reasons": ["top_level_fields_invalid"]}
    candidates = raw.get("mcq_candidates")
    if not isinstance(candidates, list) or len(candidates) != MCQ_COUNT:
        return {"decision": "direct_mcq_batch_schema_invalid",
                "reasons": ["candidate_count_invalid"]}
    letters = _answer_letters(root_id)
    accepted: list[dict[str, Any]] = []
    seen_questions: dict[str, int] = {}
    for index, candidate in enumerate(candidates, 1):
        prefix = f"item_{index:02d}"
        if not isinstance(candidate, dict) or set(candidate) != MODEL_ITEM_FIELDS:
            reasons.append(f"{prefix}:item_fields_invalid")
            continue
        question = candidate.get("question")
        answer = candidate.get("correct_answer_text")
        distractors = candidate.get("distractors")
        support = candidate.get("answer_support_span_ids")
        if not isinstance(question, str) or not question.strip():
            reasons.append(f"{prefix}:question_invalid")
        if not isinstance(answer, str) or not answer.strip():
            reasons.append(f"{prefix}:correct_answer_invalid")
        if (not isinstance(distractors, list) or len(distractors) != 3 or
                any(not isinstance(value, str) or not value.strip()
                    for value in distractors)):
            reasons.append(f"{prefix}:distractors_invalid")
        if (not isinstance(support, list) or not 2 <= len(support) <= 6 or
                len(support) != len(set(support)) or
                any(not isinstance(value, str) or not value.strip()
                    for value in support)):
            reasons.append(f"{prefix}:evidence_count_or_shape_invalid")
            support = []
        normalized_question = normalize_minimal_text_v1(question)
        if normalized_question:
            if normalized_question in seen_questions:
                reasons.append(
                    f"{prefix}:duplicate_question:item_{seen_questions[normalized_question]:02d}")
            else:
                seen_questions[normalized_question] = index
        local_item = {
            "qa_id": f"{root_id}-direct-mcq-{index:02d}",
            "question": question,
            "correct_answer_text": answer,
            "distractors": distractors,
            "answer_support_span_ids": support,
        }
        materialized, materialize_errors, diagnostics = materialize_spell_style_minimal(
            local_item, letters[index - 1],
            span_catalog=dict(source_view["span_catalog"]),
            schema_profile=SPAN_IDS_ONLY_SCHEMA_PROFILE)
        reasons.extend(f"{prefix}:{error}" for error in materialize_errors)
        scope: dict[str, Any] | None = None
        if not materialize_errors:
            scope = classify_evidence_scope(
                support, span_catalog=source_view["span_catalog"],
                partition_by_turn=source_view["partition_by_turn"])
            if not scope["child_involved"]:
                reasons.append(f"{prefix}:parent_only_forbidden")
        accepted.append({
            "item_index": index,
            "raw_item": local_item,
            "target_letter": letters[index - 1],
            "materialized_qa": materialized,
            "materialize_errors": materialize_errors,
            "materialize_diagnostics": diagnostics,
            "evidence_scope": scope,
        })
    if reasons:
        return {"decision": "direct_mcq_batch_contract_reject",
                "reasons": sorted(set(reasons)), "items": accepted}
    return {"decision": "direct_mcq_batch_accept", "reasons": [],
            "items": accepted}


def _source_inputs(source_run: Path) -> dict[str, Any]:
    expanded_path = source_run / "expanded_dialogue_round_1.json"
    report_path = source_run / "reports/SLIM_CONTINUOUS_EXPANSION_V11_LIVE_CANARY_REPORT.json"
    expanded = _load_object(expanded_path)
    report = _load_object(report_path)
    return {"expanded_path": expanded_path, "report_path": report_path,
            "expanded": expanded, "report": report,
            "source_view": build_qa_source_view(expanded)}


def preflight(
    *, source_run: Path, run_root: Path, api_config_path: Path,
    tokenizer_path: Path, solver_model: Path,
) -> dict[str, Any]:
    paths = {
        "expanded_dialogue": source_run / "expanded_dialogue_round_1.json",
        "expansion_report": source_run /
            "reports/SLIM_CONTINUOUS_EXPANSION_V11_LIVE_CANARY_REPORT.json",
    }
    checks = {
        "run_root_absent": not run_root.exists(),
        "source_run_exists": source_run.is_dir(),
        "api_config_exists": api_config_path.is_file(),
        "tokenizer_exists": tokenizer_path.is_dir(),
        "solver_model_exists": solver_model.is_dir(),
        **{f"{key}_exists": path.is_file() for key, path in paths.items()},
    }
    details: dict[str, Any] = {}
    if all(checks.values()):
        values = _source_inputs(source_run)
        expanded = values["expanded"]
        view = values["source_view"]
        partitions = list(view["partition_by_turn"].values())
        turn_ids = [row["turn_id"] for row in view["combined_dialogue"]]
        checks.update({
            "canonical_expanded_dialogue":
                expanded.get("artifact_role") == "canonical_expanded_dialogue",
            "qa_source_candidate": expanded.get("qa_source_candidate") is True,
            "expansion_generation_ready":
                values["report"].get("generation_ready") is True,
            "parent_turns_present": "original_parent" in partitions,
            "promoted_child_turns_present": "promoted_child" in partitions,
            "turn_ids_unique": len(turn_ids) == len(set(turn_ids)) and all(turn_ids),
            "span_count_matches_turn_count": len(
                view["span_catalog"]["spans"]) == len(turn_ids),
            "semantic_auditor_not_imported": not _imports_semantic_auditor(),
            "writer_budget_is_bounded": WRITER_MAX_ATTEMPTS == 2,
            "parent_only_is_hard_rejected": True,
        })
        details = {
            "root_id": expanded.get("root_id"),
            "turn_count": len(turn_ids),
            "parent_turn_count": partitions.count("original_parent"),
            "child_turn_count": partitions.count("promoted_child"),
            "span_count": len(view["span_catalog"]["spans"]),
            "source_tree": hash_tree(source_run),
            "expanded_dialogue_sha256": sha256_file(paths["expanded_dialogue"]),
            "source_view_sha256": view["view_sha256"],
        }
    failed = sorted(key for key, value in checks.items() if not value)
    return {"preflight_version": PILOT_VERSION, "checked_at": _now(),
            "decision": "pass" if not failed else "fail", "checks": checks,
            "failed_checks": failed, "details": details, "api_calls": 0,
            "run_root_created": run_root.exists()}


def _solver_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    balanced = result.get("balanced_grounding") or {}
    full = result.get("full_context") or {}
    evidence = result.get("evidence_audit") or {}
    option = result.get("option_audit") or {}
    return {
        "grounding_decision": result.get("grounding_decision"),
        "no_context_correct": balanced.get("semantic_correct_count"),
        "no_context_total": balanced.get("total_valid_rollouts"),
        "full_context_correct": sum(full.get("verified_rewards") or []),
        "full_context_total": len(full.get("verified_rewards") or []),
        "full_context_mean": result.get("verified_reward_mean"),
        "evidence_decision": evidence.get("decision"),
        "option_decision": option.get("decision"),
        "raw_final_decision": result.get("final_decision"),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    usage = report.get("api_statistics") or {}
    lines = ["# Direct MCQ + Local SPELL Pilot", "",
        f"- Status: `{report.get('status')}`",
        f"- Source dialogue turns: `{report.get('source_turn_count')}`",
        f"- MCQ Writer requests: `{usage.get('mcq_writer_requests')}`",
        f"- Local Solver generations: `{usage.get('local_solver_generations')}`",
        f"- Materialized MCQs: `{report.get('materialized_count')}`",
        f"- Admitted: `{report.get('admitted_count')}`",
        f"- Semantic Auditor calls: `0`", "", "## Items", ""]
    for item in report.get("items") or []:
        summary = item.get("solver_summary") or {}
        lines.extend([
            f"### {item.get('qa_id')}", "",
            f"- Scope: `{item.get('evidence_scope')}`",
            f"- Question: {item.get('question')}",
            f"- Grounding: `{summary.get('grounding_decision')}`",
            f"- Full context: `{summary.get('full_context_correct')}/{summary.get('full_context_total')}`",
            f"- Evidence: `{summary.get('evidence_decision')}`",
            f"- Option audit: `{summary.get('option_decision')}`",
            f"- Admission: `{(item.get('admission') or {}).get('final_decision')}`",
            ""])
    return "\n".join(lines)


def run(
    *, source_run: Path, run_root: Path, api_config_path: Path,
    tokenizer_path: Path, solver_model: Path, preflight_only: bool = False,
    no_api: bool = False,
) -> dict[str, Any]:
    frozen = preflight(source_run=source_run, run_root=run_root,
                       api_config_path=api_config_path,
                       tokenizer_path=tokenizer_path, solver_model=solver_model)
    if preflight_only or no_api:
        return frozen
    if frozen["decision"] != "pass":
        raise RuntimeError("PRE_API_PREFLIGHT_FAILED:" + ",".join(
            frozen["failed_checks"]))

    values = _source_inputs(source_run)
    expanded = values["expanded"]
    source_view = values["source_view"]
    root_id = str(expanded["root_id"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True)
    prompt = build_direct_mcq_batch_prompt(source_view)
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("config", "frozen", "qa", "filter", "reports",
                     "raw/prompts"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    api = replace(load_challenger_config(api_config_path), timeout_seconds=600,
                  temperature=0.25, enable_thinking=False)
    freeze = {
        "freeze_version": PILOT_VERSION, "created_at": _now(),
        "run_root": str(run_root), "source_run": str(source_run),
        "source_tree_sha256": frozen["details"]["source_tree"]["tree_sha256"],
        "expanded_dialogue_sha256": frozen["details"]["expanded_dialogue_sha256"],
        "source_view_sha256": source_view["view_sha256"],
        "span_catalog_sha256": span_catalog_sha256(source_view["span_catalog"]),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "prompt_tokens": prompt_tokens,
        "model_configuration": {"model": api.model, "base_url": api.base_url,
            "temperature": api.temperature, "timeout_seconds": api.timeout_seconds,
            "max_tokens": WRITER_MAX_TOKENS, "api_key_recorded": False},
        "request_budget": {"mcq_writer_max": WRITER_MAX_ATTEMPTS,
            "semantic_auditor": 0, "paid_solver": 0},
        "principle": {"every_question_must_involve_child": True,
            "parent_only_allowed": False,
            "allowed_scopes": ["child_only", "multi_child", "parent_child"],
            "latest_child_required": False},
        "project": {"git_head": git_value(PROJECT_ROOT, "rev-parse", "HEAD"),
            "diff_sha256": project_diff_sha256(PROJECT_ROOT)},
        "preflight": frozen,
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "run_status.json", {
        "status": "initialized", "created_at": _now(), "api_calls": 0})
    atomic_json(run_root / "frozen/qa_source_view.json", source_view)
    atomic_json(run_root / "frozen/questioner_smoke_context.json", {
        "combined_dialogue": source_view["combined_dialogue"]})
    atomic_json(run_root / "frozen/questioner_span_catalog.json",
                source_view["span_catalog"])
    atomic_text(run_root / "frozen/direct_mcq_writer_prompt.txt", prompt)

    manifest_sha = sha256_file(run_root / "run_freeze_manifest.json")
    ledger = RequestLedger(run_root, manifest_sha, "live")
    budget = ApiBudget(normal_caps={"direct_mcq_writer": 1},
                       format_retry_cap=1, physical_cap=WRITER_MAX_ATTEMPTS)
    writer_result: dict[str, Any]
    failure_error: str | None = None
    try:
        writer_result = run_single_object_task(
            run_root=run_root, ledger=ledger,
            backend=LiveSemanticRepairBackend(
                OpenAICompatibleChallengerClient(api), WRITER_MAX_TOKENS),
            budget=budget, stage="direct_mcq_writer",
            logical_request_id=f"{root_id}-direct-mcq-batch",
            semantic_prompt=prompt, lineage_view=source_view,
            validator=lambda raw: validate_direct_mcq_batch(
                raw, source_view=source_view, root_id=root_id),
            schema_retry_decisions=("direct_mcq_batch_schema_invalid",
                                    "direct_mcq_batch_contract_reject"),
            retry_prompt_suffix_factory=_retry_suffix,
            max_attempts=WRITER_MAX_ATTEMPTS,
            ledger_budget_limit=WRITER_MAX_ATTEMPTS,
            ledger_total_budget_limit=WRITER_MAX_ATTEMPTS,
            json_invalid_decision="direct_mcq_batch_json_invalid")
    except Exception as exc:  # preserve a transport/schema failure run for audit
        failure_error = f"{type(exc).__name__}:{exc}"
        writer_result = {"accepted": False, "physical_request_count": 0,
                         "final_decision": "direct_mcq_writer_runtime_failure",
                         "attempts": []}
    atomic_json(run_root / "reports/direct_mcq_writer_result.json", writer_result)

    items: list[dict[str, Any]] = []
    if writer_result.get("accepted"):
        validated = writer_result["accepted_validation"]
        materialized_batch = []
        for row in validated["items"]:
            qa = row["materialized_qa"]
            scope = row["evidence_scope"]
            shape = diagnose_option_shape(
                options=qa["options"],
                reference_answer_letter=qa["reference_answer_letter"],
                question=qa["question"])
            materialized_batch.append(qa)
            atomic_json(run_root / f"qa/{qa['qa_id']}.json", qa)
            items.append({"qa_id": qa["qa_id"], "question": qa["question"],
                "evidence_scope": scope["evidence_scope"],
                "child_support_count": scope["child_support_count"],
                "parent_support_count": scope["parent_support_count"],
                "target_letter": row["target_letter"],
                "option_shape": shape, "solver_result": None,
                "solver_summary": None, "admission": None})
        atomic_json(run_root / "qa/final_mcq_batch.json", {
            "batch_version": PILOT_VERSION, "root_id": root_id,
            "items": materialized_batch})

        config = _solver_config(run_root=run_root)
        config.update({"solver_position_neutral_prompt": True,
            "balanced_grounding_protocol": "balanced_four_position_k3_v2",
            "max_local_solver_generations": 256})
        manifest = {"run_id": run_root.name,
            "mode": "direct_mcq_local_spell_filter",
            "solver_max_model_len": SOLVER_MAX_MODEL_LEN,
            "source_freeze_manifest_sha256": manifest_sha,
            "solver_model": str(solver_model.resolve()),
            "solver_tokenizer": str(tokenizer_path.resolve()),
            "grounding_protocol": "balanced_four_position_k3_v2",
            "position_neutral_prompt": True, "paid_api_solver_calls": 0}
        atomic_json(run_root / "config/resolved_canary_config.json", config)
        atomic_json(run_root / "config/run_manifest.json", manifest)
        try:
            solver = VLLMSolverBackend(
                model_path=solver_model, tokenizer_path=tokenizer_path,
                max_model_len=SOLVER_MAX_MODEL_LEN, temperature=0.7, top_p=0.9,
                max_tokens=SOLVER_MAX_NEW_TOKENS,
                gpu_memory_utilization=SOLVER_GPU_MEMORY_UTILIZATION)
            orchestrator = CanaryOrchestrator(
                run_root=run_root, manifest=manifest, config=config, data={},
                tokenizer=tokenizer, realizer=None, questioner=None,
                solver=solver, backend_mode="local")
            for index, item in enumerate(items):
                qa = materialized_batch[index]
                result = orchestrator.audit_minimal_candidate_balanced_grounding(qa)
                evidence_decision = str(
                    (result.get("evidence_audit") or {}).get("decision") or "")
                evidence_marked = evidence_decision in {
                    "bad_evidence_relevance", "evidence_audit_inconclusive"}
                admission = classify_spell_admission(
                    result,
                    evidence_alignment={
                        "evidence_mismatch_marked": evidence_marked},
                    evidence_mismatch_policy=EVIDENCE_MISMATCH_POLICY)
                item.update({"solver_result": result,
                    "solver_summary": _solver_summary(result),
                    "admission": admission,
                    "evidence_mismatch_marked": evidence_marked})
                atomic_json(run_root / f"reports/{qa['qa_id']}_evaluation.json",
                            item)
        except Exception as exc:
            failure_error = f"{type(exc).__name__}:{exc}"

    request_states = list((run_root / "ledger/requests").glob("*.json"))
    paid_states = []
    local_states = []
    usage = []
    for path in request_states:
        state = _load_object(path)
        if state.get("budget_class") == "paid_api":
            paid_states.append(state)
            usage.append((state.get("response_metadata") or {}).get("usage") or {})
        elif state.get("budget_class") == "local_solver":
            local_states.append(state)
    source_after = hash_tree(source_run)
    admitted = [item for item in items
                if (item.get("admission") or {}).get("spell_filter_passed")]
    report = {
        "report_version": PILOT_VERSION,
        "status": "completed" if writer_result.get("accepted") and
            not failure_error else "failed",
        "failure_error": failure_error,
        "run_root": str(run_root), "source_run": str(source_run),
        "source_turn_count": len(source_view["combined_dialogue"]),
        "materialized_count": len(items), "admitted_count": len(admitted),
        "quarantined_count": len(items) - len(admitted), "items": items,
        "writer_result": writer_result,
        "api_statistics": {"mcq_writer_requests": len(paid_states),
            "mcq_writer_prompt_tokens": sum(int(row.get("prompt_tokens") or 0)
                                             for row in usage),
            "mcq_writer_completion_tokens": sum(
                int(row.get("completion_tokens") or 0) for row in usage),
            "mcq_writer_total_tokens": sum(int(row.get("total_tokens") or 0)
                                            for row in usage),
            "local_solver_generations": len(local_states),
            "semantic_auditor_requests": 0,
            "paid_solver_requests": 0},
        "scope_principle": {"parent_only_rejected": True,
            "every_materialized_question_child_involved": all(
                item["child_support_count"] >= 1 for item in items),
            "latest_child_required": False},
        "source_integrity": {
            "before": frozen["details"]["source_tree"]["tree_sha256"],
            "after": source_after["tree_sha256"],
            "unchanged": frozen["details"]["source_tree"]["tree_sha256"] ==
                source_after["tree_sha256"]},
        "generation_ready": bool(items),
        "qa_pipeline_ready": bool(admitted),
        "training_ready": False,
    }
    atomic_json(run_root / "reports/DIRECT_MCQ_GROUND_FILTER_PILOT.json", report)
    atomic_text(run_root / "reports/DIRECT_MCQ_GROUND_FILTER_PILOT.md",
                _markdown(report))
    atomic_json(run_root / "run_status.json", {
        "status": report["status"], "completed_at": _now(),
        "paid_api_calls": len(paid_states),
        "local_solver_generations": len(local_states),
        "qa_pipeline_ready": report["qa_pipeline_ready"]})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-api", action="store_true")
    args = parser.parse_args(argv)
    report = run(source_run=args.source_run, run_root=args.run_root,
        api_config_path=args.api_config, tokenizer_path=args.tokenizer,
        solver_model=args.solver_model, preflight_only=args.preflight_only,
        no_api=args.no_api)
    print(json.dumps({key: report.get(key) for key in (
        "decision", "status", "materialized_count", "admitted_count",
        "quarantined_count", "api_statistics", "qa_pipeline_ready")},
        ensure_ascii=False, indent=2))
    return 0 if report.get("decision", "pass") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
