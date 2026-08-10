"""Run the unchanged SPELL local-Solver filter on one frozen v11 MCQ.

This module is an adapter only.  Question generation and distractor generation are
out of scope; the authoritative filtering semantics remain in
``CanaryOrchestrator.audit_minimal_candidate``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..minimal_admission import SPELL_STYLE_MINIMAL_MODE, build_answer_evidence_contexts
from ..orchestrator import CanaryOrchestrator
from ..questioner_contract import span_catalog_sha256
from ..run_canary import git_value, project_diff_sha256
from ..run_cross_dialogue_calibration import _solver_config
from ..vllm_backends import VLLMSolverBackend
from .provenance import hash_tree


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_MCQ_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_v11_canary34_mcq_canary_1")
DEFAULT_QA_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_v11_canary34_qa_canary_5")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_v11_canary34_mcq_spell_filter_balanced_2")
DEFAULT_SOLVER_MODEL = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "accepted_models/solver_base_restart_stage2_merged")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
SOLVER_MAX_MODEL_LEN = 16384
SOLVER_MAX_NEW_TOKENS = 32
SOLVER_GPU_MEMORY_UTILIZATION = 0.80
FILTER_ADAPTER_VERSION = (
    "slim_v11_mcq_spell_filter_balanced_adapter_v3_evidence_mark_only")
EVIDENCE_MISMATCH_POLICIES = frozenset({"strict", "mark_only"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def classify_spell_admission(
    result: dict[str, Any], *, evidence_alignment: dict[str, Any],
    evidence_mismatch_policy: str,
) -> dict[str, Any]:
    """Apply the batch-only evidence exception without weakening other gates."""
    if evidence_mismatch_policy not in EVIDENCE_MISMATCH_POLICIES:
        raise ValueError("invalid_evidence_mismatch_policy")
    raw_final = str(result.get("final_decision") or "")
    evidence_decision = str(
        (result.get("evidence_audit") or {}).get("decision") or "")
    option_decision = str(
        (result.get("option_audit") or {}).get("decision") or "")
    marked = bool(evidence_alignment.get("evidence_mismatch_marked"))
    evidence_only_failure = (
        raw_final == "bad_evidence_relevance" or
        (raw_final == "frontier_audit_inconclusive" and
         evidence_decision == "evidence_audit_inconclusive" and
         option_decision not in {"bad_option_bias", "option_audit_inconclusive"})
    )
    override = (
        evidence_mismatch_policy == "mark_only" and marked and
        evidence_only_failure)
    return {
        "solver_raw_final_decision": raw_final,
        "final_decision": (
            "retain_frontier_evidence_mismatch_marked" if override else raw_final),
        "evidence_mismatch_override_applied": override,
        "spell_filter_passed": raw_final == "retain_frontier" or override,
    }


def _source_paths(mcq_run: Path, qa_source_run: Path) -> dict[str, Path]:
    return {
        "mcq": mcq_run / "qa/final_mcq.json",
        "mcq_summary": mcq_run / "reports/MCQ_CANARY_SUMMARY.json",
        "source_view": qa_source_run / "frozen/qa_source_view.json",
        "span_catalog": qa_source_run / "frozen/span_catalog.json",
    }


def preflight(
    *, mcq_run: Path, qa_source_run: Path, run_root: Path,
    solver_model: Path, tokenizer_path: Path,
) -> dict[str, Any]:
    """Validate all immutable inputs without creating the filter run root."""
    paths = _source_paths(mcq_run, qa_source_run)
    checks: dict[str, bool] = {
        "run_root_absent": not run_root.exists(),
        "mcq_run_exists": mcq_run.is_dir(),
        "qa_source_run_exists": qa_source_run.is_dir(),
        "solver_model_exists": solver_model.is_dir(),
        "tokenizer_exists": tokenizer_path.is_dir(),
        **{f"{name}_exists": path.is_file() for name, path in paths.items()},
    }
    details: dict[str, Any] = {}
    if all(checks.values()):
        qa = _load_object(paths["mcq"])
        summary = _load_object(paths["mcq_summary"])
        source_view = _load_object(paths["source_view"])
        catalog = _load_object(paths["span_catalog"])
        turns = source_view.get("combined_dialogue")
        options = qa.get("options")
        evidence_contexts = None
        try:
            evidence_contexts = build_answer_evidence_contexts(
                qa, span_catalog=catalog, frozen_dialogue=turns or [])
            evidence_contexts_valid = True
        except (KeyError, TypeError, ValueError):
            evidence_contexts_valid = False
        turn_ids = ([str(turn.get("turn_id") or "") for turn in turns]
                    if isinstance(turns, list) else [])
        reference_letter = qa.get("reference_answer_letter")
        checks.update({
            "mcq_summary_completed": summary.get("status") == "completed",
            "mcq_source_ready": (summary.get("readiness") or {}).get("mcq_ready") is True,
            "spell_minimal_admission": qa.get("admission_mode") == SPELL_STYLE_MINIMAL_MODE,
            "option_schema_abcd": (isinstance(options, dict) and
                                   tuple(options) == ("A", "B", "C", "D") and
                                   all(isinstance(options[key], str) and options[key].strip()
                                       for key in ("A", "B", "C", "D"))),
            "reference_binding_valid": (reference_letter in ("A", "B", "C", "D") and
                                        isinstance(options, dict) and
                                        options.get(reference_letter) ==
                                        qa.get("canonical_answer_text") ==
                                        qa.get("reference_answer_text")),
            "combined_dialogue_nonempty": isinstance(turns, list) and bool(turns),
            "combined_dialogue_turn_ids_unique": (bool(turn_ids) and
                                                   all(turn_ids) and
                                                   len(turn_ids) == len(set(turn_ids))),
            "span_catalog_sha_matches_mcq": (
                qa.get("span_catalog_sha256") == span_catalog_sha256(catalog)),
            "evidence_contexts_valid": evidence_contexts_valid,
            "evidence_crosses_parent_child": (
                any(not str(value).startswith("R01:S")
                    for value in qa.get("evidence_turn_ids") or []) and
                any(str(value).startswith("R01:S")
                    for value in qa.get("evidence_turn_ids") or [])),
        })
        details = {
            "qa_id": qa.get("qa_id"),
            "admission_mode": qa.get("admission_mode"),
            "combined_dialogue_turn_count": len(turns or []),
            "span_count": len(catalog.get("spans") or []),
            "evidence_turn_ids": list(qa.get("evidence_turn_ids") or []),
            "evidence_only_turn_ids": ((evidence_contexts or {}).get(
                "answer_turn_ids") or []),
            "mcq_source_tree": hash_tree(mcq_run),
            "qa_source_tree": hash_tree(qa_source_run),
            "source_file_sha256": {
                name: sha256_file(path) for name, path in paths.items()
            },
        }
    failed = sorted(key for key, passed in checks.items() if not passed)
    return {
        "preflight_version": FILTER_ADAPTER_VERSION,
        "checked_at": _now(),
        "decision": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "details": details,
        "network_api_calls": 0,
        "run_root_created": run_root.exists(),
    }


def _manifest(
    *, run_root: Path, mcq_run: Path, qa_source_run: Path,
    solver_model: Path, tokenizer_path: Path, preflight_report: dict[str, Any],
) -> dict[str, Any]:
    paths = _source_paths(mcq_run, qa_source_run)
    return {
        "run_id": run_root.name,
        "mode": "local_solver_spell_filter",
        "adapter_version": FILTER_ADAPTER_VERSION,
        "created_at": _now(),
        "project_git_commit": git_value(PROJECT_ROOT, "rev-parse", "HEAD"),
        "project_diff_sha256": project_diff_sha256(PROJECT_ROOT),
        "source_mcq_run": str(mcq_run.resolve()),
        "source_qa_run": str(qa_source_run.resolve()),
        "source_mcq_tree_sha256": preflight_report["details"]["mcq_source_tree"][
            "tree_sha256"],
        "source_qa_tree_sha256": preflight_report["details"]["qa_source_tree"][
            "tree_sha256"],
        "source_file_sha256": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "qa_id": preflight_report["details"]["qa_id"],
        "admission_mode": SPELL_STYLE_MINIMAL_MODE,
        "solver_model": str(solver_model.resolve()),
        "solver_tokenizer": str(tokenizer_path.resolve()),
        "solver_max_model_len": SOLVER_MAX_MODEL_LEN,
        "solver_protocol": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": SOLVER_MAX_NEW_TOKENS,
            "enable_thinking": False,
            "add_generation_prompt": True,
            "parser": "boxed_letter_v1",
            "max_parse_retries_per_target": 1,
            "filter_entry": (
                "CanaryOrchestrator.audit_minimal_candidate_balanced_grounding"),
            "grounding_protocol": "balanced_four_position_k3_v2",
            "position_neutral_prompt": True,
        },
        "allowed_role": "local_spell_solver",
        "questioner_calls": 0,
        "distractor_calls": 0,
        "realizer_calls": 0,
        "evidence_committer_calls": 0,
        "paid_api_calls": 0,
        "orchestrator_sha256": sha256_file(MODULE_ROOT.parent / "orchestrator.py"),
        "vllm_backend_sha256": sha256_file(MODULE_ROOT.parent / "vllm_backends.py"),
        "adapter_sha256": sha256_file(Path(__file__)),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    result = report.get("result") or {}
    ledger = report.get("ledger_summary") or {}
    lines = [
        "# SPELL Solver Filter Summary",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Final decision: `{report.get('final_decision')}`",
        f"- QA ID: `{report.get('qa_id')}`",
        f"- Grounding: `{result.get('grounding_decision')}`",
        f"- Frontier: `{result.get('frontier_decision')}`",
        f"- Evidence audit: `{(result.get('evidence_audit') or {}).get('decision')}`",
        f"- Evidence alignment: `{(report.get('evidence_alignment') or {}).get('evidence_alignment_status')}`",
        f"- Evidence mismatch override: `{report.get('evidence_mismatch_override_applied')}`",
        f"- Option audit: `{(result.get('option_audit') or {}).get('decision')}`",
        f"- Local Solver generations: `{ledger.get('request_count', 0)}`",
        "- Paid/API generations: `0`",
        f"- Full SPELL filter passed: `{report.get('spell_filter_passed')}`",
        f"- QA pipeline ready: `{report.get('qa_pipeline_ready')}`",
        "",
        "## Source integrity",
        "",
        f"- MCQ source unchanged: `{report.get('source_integrity', {}).get('mcq_source_unchanged')}`",
        f"- QA source unchanged: `{report.get('source_integrity', {}).get('qa_source_unchanged')}`",
        "",
        "This report uses balanced four-position no-context grounding followed by the "
        "unchanged SPELL full-context, evidence-only, evidence-masked, and "
        "option-permutation filter chain.",
        "",
    ]
    return "\n".join(lines)


def run_filter(
    *, mcq_run: Path, qa_source_run: Path, run_root: Path,
    solver_model: Path, tokenizer_path: Path,
    evidence_mismatch_policy: str = "strict",
) -> dict[str, Any]:
    if evidence_mismatch_policy not in EVIDENCE_MISMATCH_POLICIES:
        raise ValueError("invalid_evidence_mismatch_policy")
    report = preflight(mcq_run=mcq_run, qa_source_run=qa_source_run,
                       run_root=run_root, solver_model=solver_model,
                       tokenizer_path=tokenizer_path)
    if report["decision"] != "pass":
        raise RuntimeError("spell_filter_preflight_failed:" + ",".join(
            report["failed_checks"]))

    run_root.mkdir(parents=True, exist_ok=False)
    qa = _load_object(mcq_run / "qa/final_mcq.json")
    mcq_summary = _load_object(mcq_run / "reports/MCQ_CANARY_SUMMARY.json")
    evidence_alignment = mcq_summary.get("evidence_alignment") or {
        "evidence_mismatch_policy": "strict",
        "evidence_alignment_status": "aligned",
        "evidence_mismatch_marked": False,
        "evidence_mismatch_reasons": [],
    }
    source_view = _load_object(qa_source_run / "frozen/qa_source_view.json")
    catalog = _load_object(qa_source_run / "frozen/span_catalog.json")
    manifest = _manifest(run_root=run_root, mcq_run=mcq_run,
                         qa_source_run=qa_source_run,
                         solver_model=solver_model, tokenizer_path=tokenizer_path,
                         preflight_report=report)
    manifest["evidence_mismatch_policy"] = evidence_mismatch_policy
    manifest["evidence_alignment"] = evidence_alignment
    config = _solver_config(run_root=run_root)
    config["solver_position_neutral_prompt"] = True
    config["balanced_grounding_protocol"] = "balanced_four_position_k3_v2"
    atomic_json(run_root / "preflight.json", report)
    atomic_json(run_root / "config/run_manifest.json", manifest)
    atomic_json(run_root / "config/resolved_canary_config.json", config)
    atomic_json(run_root / "frozen/questioner_smoke_context.json", {
        "combined_dialogue": source_view["combined_dialogue"]})
    atomic_json(run_root / "frozen/questioner_span_catalog.json", catalog)
    atomic_json(run_root / "frozen/solver_qa.json", qa)
    atomic_json(run_root / "run_status.json", {
        "status": "initialized", "initialized_at": _now(),
        "paid_api_calls": 0, "local_solver_started": False})

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True)
        backend = VLLMSolverBackend(
            model_path=solver_model, tokenizer_path=tokenizer_path,
            max_model_len=SOLVER_MAX_MODEL_LEN, temperature=0.7, top_p=0.9,
            max_tokens=SOLVER_MAX_NEW_TOKENS,
            gpu_memory_utilization=SOLVER_GPU_MEMORY_UTILIZATION)
        atomic_json(run_root / "run_status.json", {
            "status": "running", "started_at": _now(),
            "paid_api_calls": 0, "local_solver_started": True})
        orchestrator = CanaryOrchestrator(
            run_root=run_root, manifest=manifest, config=config, data={},
            tokenizer=tokenizer, realizer=None, questioner=None,
            solver=backend, backend_mode="local")
        result = orchestrator.audit_minimal_candidate_balanced_grounding(qa)
        diagnostic = result.get("balanced_grounding")
        ledger_summary = orchestrator.ledger.summary()
        mcq_after = hash_tree(mcq_run)
        qa_after = hash_tree(qa_source_run)
        source_integrity = {
            "mcq_source_before": report["details"]["mcq_source_tree"],
            "mcq_source_after": mcq_after,
            "mcq_source_unchanged": (mcq_after["tree_sha256"] ==
                                     report["details"]["mcq_source_tree"]["tree_sha256"]),
            "qa_source_before": report["details"]["qa_source_tree"],
            "qa_source_after": qa_after,
            "qa_source_unchanged": (qa_after["tree_sha256"] ==
                                    report["details"]["qa_source_tree"]["tree_sha256"]),
        }
        admission = classify_spell_admission(
            result, evidence_alignment=evidence_alignment,
            evidence_mismatch_policy=evidence_mismatch_policy)
        final_decision = admission["final_decision"]
        passed = admission["spell_filter_passed"]
        final_report = {
            "report_version": FILTER_ADAPTER_VERSION,
            "status": "completed",
            "completed_at": _now(),
            "run_root": str(run_root.resolve()),
            "qa_id": qa["qa_id"],
            "result": result,
            "solver_raw_final_decision": admission["solver_raw_final_decision"],
            "no_context_permutation_diagnostic": diagnostic,
            "final_decision": final_decision,
            "evidence_mismatch_policy": evidence_mismatch_policy,
            "evidence_alignment": evidence_alignment,
            "evidence_mismatch_override_applied": admission[
                "evidence_mismatch_override_applied"],
            "ledger_summary": ledger_summary,
            "local_solver_generation_count": ledger_summary["request_count"],
            "paid_api_calls": 0,
            "questioner_calls": 0,
            "distractor_calls": 0,
            "realizer_calls": 0,
            "evidence_committer_calls": 0,
            "source_integrity": source_integrity,
            "spell_filter_passed": passed,
            "full_ground_ready": passed,
            "qa_pipeline_ready": passed,
            "training_ready": False,
        }
        atomic_json(run_root / "filter_results.json", [result])
        atomic_json(run_root / "reports/SPELL_FILTER_SUMMARY.json", final_report)
        atomic_text(run_root / "reports/SPELL_FILTER_SUMMARY.md",
                    _render_markdown(final_report))
        atomic_json(run_root / "run_status.json", {
            "status": "completed", "completed_at": _now(),
            "final_decision": final_decision,
            "local_solver_generation_count": ledger_summary["request_count"],
            "paid_api_calls": 0})
        return final_report
    except Exception as exc:
        failure = {
            "status": "failed", "failed_at": _now(),
            "failure_stage": "local_spell_solver_filter",
            "error_type": type(exc).__name__, "error": str(exc),
            "paid_api_calls": 0,
        }
        atomic_json(run_root / "run_status.json", failure)
        atomic_json(run_root / "reports/SPELL_FILTER_FAILURE.json", failure)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the original SPELL local Solver filter on one v11 MCQ")
    parser.add_argument("--mcq-run", type=Path, default=DEFAULT_MCQ_RUN)
    parser.add_argument("--qa-source-run", type=Path, default=DEFAULT_QA_SOURCE_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--evidence-mismatch-policy", choices=sorted(
        EVIDENCE_MISMATCH_POLICIES), default="strict")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        report = preflight(
            mcq_run=args.mcq_run, qa_source_run=args.qa_source_run,
            run_root=args.run_root, solver_model=args.solver_model,
            tokenizer_path=args.tokenizer)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["decision"] == "pass" else 2
    result = run_filter(
        mcq_run=args.mcq_run, qa_source_run=args.qa_source_run,
        run_root=args.run_root, solver_model=args.solver_model,
        tokenizer_path=args.tokenizer,
        evidence_mismatch_policy=args.evidence_mismatch_policy)
    print(json.dumps({
        "status": result["status"],
        "final_decision": result["final_decision"],
        "local_solver_generation_count": result["local_solver_generation_count"],
        "paid_api_calls": 0,
        "qa_pipeline_ready": result["qa_pipeline_ready"],
        "run_root": result["run_root"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
