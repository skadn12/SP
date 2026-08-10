"""One fresh semantic audit of a frozen generated core QA candidate."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from transformers import AutoTokenizer

from ..core import atomic_json, atomic_text, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..lineage_three_stage_canary import ApiBudget, run_single_object_task
from ..live_backends import (LiveSemanticRepairBackend,
                             OpenAICompatibleChallengerClient,
                             load_challenger_config)
from .provenance import git_value, hash_tree, project_diff_sha256
from .qa_semantic_audit import (
    QA_SEMANTIC_AUDIT_MAX_TOKENS,
    QA_SEMANTIC_AUDIT_PROMPT_VERSION,
    build_qa_semantic_audit_prompt,
    qa_semantic_audit_protocol_sha256,
    validate_qa_semantic_audit_output,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE_QA_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_v11_canary34_qa_canary_4"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_v11_canary34_qa_semantic_audit_1"
)
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
MODEL_CONTEXT_LIMIT = 40960
EVIDENCE_MISMATCH_POLICIES = frozenset({"strict", "mark_only"})


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_semantic_audit_admission(
    *, qa: dict[str, Any], audit: dict[str, Any] | None,
    evidence_mismatch_policy: str,
) -> dict[str, Any]:
    if evidence_mismatch_policy not in EVIDENCE_MISMATCH_POLICIES:
        raise ValueError("evidence_mismatch_policy_invalid")
    answer_verified = bool(
        audit and audit.get("audit_decision") == "accept")
    submitted_evidence_sufficient = bool(
        audit and audit.get("support_set_sufficient") is True)
    evidence_mismatch = bool(
        qa.get("evidence_mismatch") or
        (answer_verified and not submitted_evidence_sufficient))
    strictly_verified = bool(
        answer_verified and submitted_evidence_sufficient and
        not qa.get("evidence_mismatch"))
    mismatch_marked = bool(
        answer_verified and evidence_mismatch_policy == "mark_only" and
        evidence_mismatch)
    existing_reasons = list(qa.get("evidence_mismatch_reasons") or [])
    audit_reasons: list[str] = (
        ["semantic_audit_submitted_evidence_insufficient"]
        if answer_verified and not submitted_evidence_sufficient else [])
    mismatch_reasons = list(dict.fromkeys(existing_reasons + audit_reasons))
    admitted = bool(strictly_verified or mismatch_marked)
    return {
        "admitted": admitted,
        "downstream_observation_ready": admitted,
        "training_ready": strictly_verified,
        "answer_semantically_verified": answer_verified,
        "submitted_evidence_sufficient": submitted_evidence_sufficient,
        "strictly_verified": strictly_verified,
        "evidence_mismatch": evidence_mismatch,
        "evidence_mismatch_marked": mismatch_marked,
        "evidence_mismatch_reasons": mismatch_reasons,
        "evidence_mismatch_policy": evidence_mismatch_policy,
    }


def preflight(*, source_qa_run: Path, run_root: Path,
              api_config: Path, tokenizer_path: Path) -> dict[str, Any]:
    qa_path = source_qa_run / "qa/core_qa.json"
    summary_path = source_qa_run / "reports/QA_CANARY_SUMMARY.json"
    source_view_path = source_qa_run / "frozen/qa_source_view.json"
    checks = {
        "run_root_absent": not run_root.exists(),
        "source_qa_run_exists": source_qa_run.is_dir(),
        "core_qa_exists": qa_path.is_file(),
        "source_summary_exists": summary_path.is_file(),
        "complete_dialogue_view_exists": source_view_path.is_file(),
        "api_config_exists": api_config.is_file(),
        "tokenizer_exists": tokenizer_path.exists(),
    }
    prompt = ""
    qa: dict[str, Any] | None = None
    requires_cross_boundary = False
    complete_dialogue: list[dict[str, Any]] = []
    if qa_path.is_file() and summary_path.is_file() and source_view_path.is_file():
        qa = _load(qa_path)
        summary = _load(summary_path)
        source_view = _load(source_view_path)
        partition_by_turn = source_view.get("partition_by_turn") or {}
        complete_dialogue = [
            {**row, "source_partition": partition_by_turn.get(row.get("turn_id"))}
            for row in source_view.get("combined_dialogue") or []
            if isinstance(row, dict)
        ]
        checks.update({
            "source_core_qa_ready": bool(
                (summary.get("readiness") or {}).get("core_qa_ready")),
            "source_complete_dialogue_qa_ready": bool(
                (summary.get("readiness") or {}).get(
                    "complete_dialogue_qa_ready",
                    (summary.get("readiness") or {}).get(
                        "cross_boundary_qa_ready"))),
            "source_not_semantically_verified": not bool(
                (summary.get("readiness") or {}).get(
                    "semantic_entailment_independently_verified")),
            "source_not_pipeline_admitted": not bool(
                (summary.get("readiness") or {}).get("qa_pipeline_ready")),
            "selected_evidence_present": bool(qa.get("selected_evidence")),
            "complete_dialogue_nonempty": bool(complete_dialogue),
        })
        if (checks["selected_evidence_present"] and
                checks["complete_dialogue_nonempty"]):
            requires_cross_boundary = {
                str(row.get("source_partition") or "")
                for row in qa.get("selected_evidence") or []
            } == {"original_parent", "promoted_child"}
            prompt = build_qa_semantic_audit_prompt(
                core_qa={key: qa[key] for key in
                         ("question", "correct_answer_text",
                          "answer_support_span_ids")},
                selected_evidence=qa["selected_evidence"],
                complete_dialogue=complete_dialogue)
    tokens = 0
    if prompt and tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True)
        tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        checks["prompt_within_context"] = (
            tokens + QA_SEMANTIC_AUDIT_MAX_TOKENS <= MODEL_CONTEXT_LIMIT)
    errors = [name for name, passed in checks.items() if not passed]
    return {
        "preflight_version": "slim_core_qa_semantic_audit_preflight_v4",
        "valid": not errors, "errors": errors, "checks": checks,
        "source_qa_run": str(source_qa_run),
        "source_tree": hash_tree(source_qa_run)
            if source_qa_run.is_dir() else None,
        "source_core_qa_sha256": sha256_file(qa_path)
            if qa_path.is_file() else None,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8"))
            if prompt else None,
        "prompt_tokens": tokens,
        "qa": qa,
        "complete_dialogue": complete_dialogue,
        "requires_cross_boundary": requires_cross_boundary,
    }


def run(*, source_qa_run: Path, run_root: Path, api_config_path: Path,
        tokenizer_path: Path, preflight_only: bool = False,
        no_api: bool = False,
        evidence_mismatch_policy: str = "strict",
        before_paid_request: Callable[
            [str, bool, int], None] | None = None) -> dict[str, Any]:
    if evidence_mismatch_policy not in EVIDENCE_MISMATCH_POLICIES:
        raise ValueError("evidence_mismatch_policy_invalid")
    frozen = preflight(source_qa_run=source_qa_run, run_root=run_root,
                       api_config=api_config_path,
                       tokenizer_path=tokenizer_path)
    public = {key: value for key, value in frozen.items()
              if key not in {"prompt", "qa", "complete_dialogue"}}
    if preflight_only or no_api:
        return public
    if not frozen["valid"]:
        raise RuntimeError(f"PRE_API_PREFLIGHT_FAILED:{frozen['errors']}")
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("frozen", "raw/prompts", "reports", "qa"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    api = replace(load_challenger_config(api_config_path),
                  timeout_seconds=600,
                  questioner_max_tokens=QA_SEMANTIC_AUDIT_MAX_TOKENS,
                  temperature=0.0, enable_thinking=False)
    code_before = {
        "git_head": git_value(PROJECT_ROOT, "rev-parse", "HEAD"),
        "project_diff_sha256": project_diff_sha256(PROJECT_ROOT),
        "protocol_sha256": qa_semantic_audit_protocol_sha256(),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    freeze = {
        "freeze_version": "slim_core_qa_semantic_audit_freeze_v4",
        "created_at": _now(), "run_root": str(run_root),
        "source_qa_run": str(source_qa_run),
        "source_tree_sha256": frozen["source_tree"]["tree_sha256"],
        "source_core_qa_sha256": frozen["source_core_qa_sha256"],
        "prompt_version": QA_SEMANTIC_AUDIT_PROMPT_VERSION,
        "prompt_sha256": frozen["prompt_sha256"],
        "prompt_tokens": frozen["prompt_tokens"],
        "protocol_sha256": qa_semantic_audit_protocol_sha256(),
        "model_configuration": {
            "model": api.model, "base_url": api.base_url,
            "temperature": api.temperature,
            "timeout_seconds": api.timeout_seconds,
            "enable_thinking": api.enable_thinking,
            "max_completion_tokens": QA_SEMANTIC_AUDIT_MAX_TOKENS,
            "api_key_recorded": False},
        "request_budget": {"semantic_auditor_normal": 1,
                           "format_only_retry": 1,
                           "physical_maximum": 2},
        "code_manifest": code_before,
        "auditor_may_modify_qa": False,
        "auditor_may_change_evidence": False,
        "evidence_mismatch_policy": evidence_mismatch_policy,
        "preflight": public,
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "run_status.json", {
        "status": "initialized", "created_at": _now(), "api_calls": 0})
    atomic_text(run_root / "frozen/semantic_audit_prompt.txt", frozen["prompt"])
    atomic_json(run_root / "frozen/submitted_core_qa.json", frozen["qa"])
    ledger = RequestLedger(
        run_root, sha256_file(run_root / "run_freeze_manifest.json"), "live")
    backend = LiveSemanticRepairBackend(
        OpenAICompatibleChallengerClient(api), QA_SEMANTIC_AUDIT_MAX_TOKENS)
    budget = ApiBudget(normal_caps={"qa_auditor": 1},
                       format_retry_cap=1, physical_cap=2)
    evidence_ids = list(frozen["qa"]["answer_support_span_ids"])
    lineage_view = {
        "combined_dialogue": [{
            "turn_id": row["turn_id"], "session_id": row["session_id"],
            "speaker": row["speaker"], "text": row["exact_text"],
        } for row in frozen["qa"]["selected_evidence"]],
        "span_catalog": {"span_catalog_version": "dialogue_span_catalog_v1",
                         "spans": [{key: row[key] for key in
                                    ("span_id", "turn_id", "session_id",
                                     "speaker", "exact_text")}
                                   for row in frozen["qa"]["selected_evidence"]]},
        "view_sha256": sha256_bytes(json.dumps(
            frozen["qa"]["selected_evidence"], ensure_ascii=False,
            sort_keys=True).encode("utf-8")),
    }
    result = run_single_object_task(
        run_root=run_root, ledger=ledger, backend=backend, budget=budget,
        stage="qa_auditor", logical_request_id="qa-semantic-audit-001",
        semantic_prompt=frozen["prompt"], lineage_view=lineage_view,
        validator=lambda raw: validate_qa_semantic_audit_output(
            raw, expected_span_ids=evidence_ids,
            submitted_extractive_answer=frozen["qa"]["correct_answer_text"],
            requires_cross_boundary=frozen["requires_cross_boundary"]),
        schema_retry_decisions=("qa_semantic_audit_schema_invalid",),
        ledger_budget_limit=2, ledger_total_budget_limit=2,
        before_paid_request=before_paid_request,
        json_invalid_decision="qa_semantic_audit_json_invalid")
    audit = ((result.get("accepted_validation") or {}).get("audit")
             if result.get("accepted") else None)
    admission = classify_semantic_audit_admission(
        qa=frozen["qa"], audit=audit,
        evidence_mismatch_policy=evidence_mismatch_policy)
    verified = admission["strictly_verified"]
    answer_verified = admission["answer_semantically_verified"]
    admitted = admission["admitted"]
    atomic_json(run_root / "reports/semantic_audit.json", {
        "contract_result": result, "audit": audit,
        "semantic_entailment_verified": verified})
    if admitted:
        atomic_json(run_root / "qa/admitted_gold.json", {
            **frozen["qa"],
            "correct_answer_text": audit["grounded_answer"],
            "source_questioner_answer_text": frozen["qa"]["correct_answer_text"],
            "source_extractive_answer_text": frozen["qa"]["correct_answer_text"],
            "semantic_entailment_independently_verified": verified,
            "answer_semantically_verified_against_complete_dialogue":
                answer_verified,
            "submitted_evidence_sufficient":
                admission["submitted_evidence_sufficient"],
            "training_ready": admission["training_ready"],
            "evidence_mismatch": admission["evidence_mismatch"],
            "evidence_mismatch_marked": admission["evidence_mismatch_marked"],
            "evidence_mismatch_reasons": admission["evidence_mismatch_reasons"],
            "evidence_gaps": list((audit or {}).get("evidence_gaps") or []),
            "evidence_mismatch_policy": evidence_mismatch_policy,
            "semantic_audit_decision": audit.get("audit_decision") if audit else None,
            "semantic_audit_run": str(run_root)})
    else:
        atomic_json(run_root / "qa/quarantined_candidate.json", {
            **frozen["qa"], "semantic_entailment_independently_verified": False,
            "semantic_audit": audit})
    completed = ledger.completed()
    usage = [row.get("response_metadata", {}).get("usage") or {}
             for row in completed]
    source_after = hash_tree(source_qa_run)
    code_after = {
        "git_head": git_value(PROJECT_ROOT, "rev-parse", "HEAD"),
        "project_diff_sha256": project_diff_sha256(PROJECT_ROOT),
        "protocol_sha256": qa_semantic_audit_protocol_sha256(),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    summary = {
        "report_version": "slim_core_qa_semantic_audit_report_v4",
        "status": "completed", "failure_stage": None,
        "final_decision": (
            "C_QA_CONTINUES_WITH_EVIDENCE_MISMATCH"
            if admitted and admission["evidence_mismatch"] else
            "A_QA_GOLD_ADMITTED" if verified else
            "B_QA_CANDIDATE_QUARANTINED"),
        "source_qa_run": str(source_qa_run), "run_root": str(run_root),
        "audit": audit,
        "api_statistics": {
            "semantic_auditor_physical_requests": len(completed),
            "format_retries": int(result["format_retry_used"]),
            "questioner_requests": 0, "distractor_requests": 0,
            "solver_requests": 0,
            "prompt_tokens": sum(int(x.get("prompt_tokens") or 0) for x in usage),
            "completion_tokens": sum(int(x.get("completion_tokens") or 0) for x in usage),
            "total_tokens": sum(int(x.get("total_tokens") or 0) for x in usage)},
        "integrity": {
            "source_tree_before": frozen["source_tree"]["tree_sha256"],
            "source_tree_after": source_after["tree_sha256"],
            "source_qa_run_unchanged": frozen["source_tree"]["tree_sha256"] ==
                source_after["tree_sha256"],
            "code_manifest_before": code_before,
            "code_manifest_after": code_after,
            "code_unchanged_during_run": code_before == code_after},
        "readiness": {
            "core_qa_ready": True,
            "answer_semantically_verified": answer_verified,
            "semantic_entailment_independently_verified": verified,
            "submitted_evidence_sufficient":
                admission["submitted_evidence_sufficient"],
            "evidence_mismatch_marked": admission["evidence_mismatch_marked"],
            "open_answer_gold_ready": admitted,
            "downstream_observation_ready": admitted,
            "training_ready": admission["training_ready"],
            "qa_pipeline_ready": admitted,
            "mcq_ready": False, "solver_validated": False},
    }
    atomic_json(run_root / "reports/QA_SEMANTIC_AUDIT_SUMMARY.json", summary)
    atomic_text(run_root / "reports/QA_SEMANTIC_AUDIT_SUMMARY.md", "\n".join([
        "# QA Semantic Audit", "",
        f"- Decision: `{summary['final_decision']}`",
        f"- Semantic entailment verified: `{str(verified).lower()}`",
        f"- Audit reason: {audit.get('reason') if audit else 'contract failure'}",
        f"- Unsupported claims: `{json.dumps((audit or {}).get('unsupported_claims') or [], ensure_ascii=False)}`",
        f"- Evidence gaps: `{json.dumps((audit or {}).get('evidence_gaps') or [], ensure_ascii=False)}`",
        f"- Source QA run unchanged: `{str(summary['integrity']['source_qa_run_unchanged']).lower()}`",
        ""]))
    atomic_json(run_root / "run_status.json", {
        "status": "completed", "completed_at": _now(),
        "api_calls": len(completed), "final_decision": summary["final_decision"]})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-qa-run", type=Path,
                        default=DEFAULT_SOURCE_QA_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--evidence-mismatch-policy",
                        choices=sorted(EVIDENCE_MISMATCH_POLICIES),
                        default="strict")
    args = parser.parse_args(argv)
    result = run(source_qa_run=args.source_qa_run, run_root=args.run_root,
                 api_config_path=args.api_config, tokenizer_path=args.tokenizer,
                 preflight_only=args.preflight_only, no_api=args.no_api,
                 evidence_mismatch_policy=args.evidence_mismatch_policy)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("valid", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
