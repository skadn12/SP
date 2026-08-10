"""Fresh multi-root pilot for expansion -> QA -> MCQ -> local SPELL audit.

The batch layer is intentionally thin: every root owns isolated stage roots and
the existing semantic components keep their own retry and request budgets.
Failures are quarantined per root and never resumed inside the same batch.
"""

from __future__ import annotations

import argparse
import compileall
import json
import random
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..live_backends import load_challenger_config
from ..unified_dialogue_root import read_unified_parent, validate_unified_dialogue_root
from . import live_canary_v11 as expansion_runtime
from . import live_preflight_b2_v11 as frozen_preflight
from . import live_qa_canary_1 as qa_runtime
from . import live_qa_semantic_audit_1 as qa_audit_runtime
from . import run_mcq_spell_filter as spell_runtime
from . import single_root_mcq_materializer as mcq_runtime
from .batch_cost_control_v11 import (
    BatchRequestBudget, PaidRequestBudgetExhausted,
    candidate_terminal_status,
)
from .live_contracts_v11 import prompt_schema_versions_v11
from .live_role_runtime_v11 import ALLOWED_LIVE_ROLES_V11
from .planner_transport_v11 import planner_transport_configuration_v11
from .provenance import git_value, hash_tree, project_diff_sha256
from .round_validator_transport_v11 import round_validator_transport_configuration_v11


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
ROOTS = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots")
DEFAULT_BATCH_ROOT = RUNS / "slim_continuous_expansion_v11_end_to_end_batch_pilot_1"
DEFAULT_ROOT_IDS = ("cc-train-episode-216742", "cc-train-episode-41625")
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_SOLVER_MODEL = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "accepted_models/solver_base_restart_stage2_merged")
BATCH_VERSION = (
    "slim_v11_end_to_end_batch_v23_soft_child_evidence_coverage")
DURABLE_UPDATE_POLICY = "dialogue_only"
EVIDENCE_MISMATCH_POLICY = "mark_only"
QA_CANDIDATE_LIMIT = 3
STOP_ON_FIRST_ADMITTED = True
PLANNER_ATTEMPT_LIMIT = 2
REALIZER_SCENE_ATTEMPT_LIMIT = 2
ROUND_VALIDATOR_ATTEMPT_LIMIT = 2
QUESTIONER_ATTEMPT_LIMIT = 2
DISTRACTOR_ATTEMPT_LIMIT = 2
SOLVER_GUIDED_ATTEMPT_LIMIT = 1
PER_ROOT_PAID_REQUEST_LIMIT = 24
CHILD_INVOLVED_QA_TARGET_RATIO = 0.50


def _remember_questioner_attempts(
    qa_report: Mapping[str, Any], *, excluded_questions: list[str],
    excluded_support_pairs: list[tuple[str, ...]],
) -> None:
    for attempt in ((qa_report.get("questioner_result") or {}).get(
            "attempts") or []):
        validation = attempt.get("validation") or {}
        question = str(validation.get("question") or "").strip()
        if question and question not in excluded_questions:
            excluded_questions.append(question)
        support = tuple(str(value) for value in
                        validation.get("answer_support_span_ids") or [])
        if (2 <= len(support) <= 6 and
                support not in excluded_support_pairs):
            excluded_support_pairs.append(support)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _root_seed(root_id: str) -> int:
    suffix = root_id.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isdigit() else int(
        sha256_bytes(root_id.encode("utf-8"))[:8], 16)


def _root_paths(batch_root: Path, root_id: str) -> dict[str, Path]:
    base = batch_root / "roots" / root_id
    first = _candidate_paths(base, 1)
    return {
        "base": base,
        "expansion": base / "01_expansion",
        "candidates": base / "qa_candidates",
        # Compatibility aliases point to candidate 1. Runtime rows record every
        # candidate's actual isolated paths separately.
        "qa": first["qa"],
        "qa_audit": first["qa_audit"],
        "mcq": first["mcq"],
        "spell": first["spell"],
    }


def _candidate_paths(root_base: Path, candidate_index: int) -> dict[str, Path]:
    if candidate_index < 1:
        raise ValueError("qa_candidate_index_invalid")
    base = root_base / "qa_candidates" / f"candidate_{candidate_index:02d}"
    return {
        "base": base,
        "qa": base / "02_questioner",
        "qa_audit": base / "03_semantic_audit",
        "mcq": base / "04_mcq_materializer",
        "spell": base / "05_spell_filter",
        "mcq_solver_guided": base / "06_mcq_materializer_solver_guided",
        "spell_solver_guided": base / "07_spell_filter_solver_guided",
    }


def solver_guided_retry_required(spell: Mapping[str, Any]) -> bool:
    """Permit one distractor-only retry for a strict grounding rejection."""
    if spell.get("qa_pipeline_ready"):
        return False
    return mcq_runtime.solver_guided_retry_eligible(
        str(spell.get("final_decision") or ""))


def candidate_generation_should_stop(candidate: Mapping[str, Any]) -> bool:
    return STOP_ON_FIRST_ADMITTED and candidate.get("status") == "admitted"


def summarize_evidence_scope_coverage(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report Child use as a batch objective, never as item admission."""
    scopes = [
        str((candidate.get("qa") or {}).get("evidence_scope") or "unknown")
        for row in rows for candidate in row.get("qa_candidates") or []
        if (candidate.get("qa") or {}).get("readiness", {}).get(
            "core_qa_ready")
    ]
    counts = {
        scope: scopes.count(scope)
        for scope in ("parent_only", "child_only", "parent_child",
                      "multi_child", "unknown")
    }
    child_involved = sum(
        scope in {"child_only", "parent_child", "multi_child"}
        for scope in scopes)
    ratio = child_involved / len(scopes) if scopes else 0.0
    return {
        "evidence_scope_counts": counts,
        "child_involved_qa_candidate_count": child_involved,
        "child_evidence_coverage_ratio": ratio,
        "child_evidence_coverage_target": CHILD_INVOLVED_QA_TARGET_RATIO,
        "child_evidence_coverage_low": (
            bool(scopes) and ratio < CHILD_INVOLVED_QA_TARGET_RATIO),
        "child_evidence_coverage_is_admission_gate": False,
    }


def _git_diff_check() -> bool:
    return subprocess.run(
        ["git", "diff", "--check"], cwd=PROJECT_ROOT,
        capture_output=True).returncode == 0


def _b1_readiness() -> tuple[dict[str, Any], bool]:
    path = frozen_preflight.PHASE_B1 / "readiness.json"
    value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    readiness = value.get("readiness") or {}
    required = (
        "v11_runner_contract_ready", "local_scene_gate_ready",
        "review_world_view_ready", "round_expansion_validator_contract_ready",
        "round_validator_output_validation_ready", "scene_decision_closure_ready",
        "promotion_v2_ready", "durable_update_compiler_ready",
        "binder_transaction_compatibility_ready", "hybrid_request_budget_ready",
        "mock_hybrid_runtime_ready", "live_v11_preflight_ready",
    )
    return readiness, all(readiness.get(key) is True for key in required)


def preflight_root_expansion(
    *, root_id: str, input_path: Path, run_root: Path,
    api_config_path: Path, tokenizer_path: Path,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "run_root_absent": not run_root.exists(),
        "input_exists": input_path.is_file(),
        "input_under_authoritative_roots": input_path.parent.parent == ROOTS,
        "input_directory_matches_root": input_path.parent.name == root_id,
        "api_config_exists": api_config_path.is_file(),
        "tokenizer_exists": tokenizer_path.exists(),
        "phase_a_present": frozen_preflight.PHASE_A.is_dir(),
        "phase_b1_present": frozen_preflight.PHASE_B1.is_dir(),
    }
    root: dict[str, Any] = {}
    parent: dict[str, Any] = {}
    validation = {"valid": False, "errors": ["input_missing"]}
    if input_path.is_file():
        root = json.loads(input_path.read_text(encoding="utf-8"))
        validation = validate_unified_dialogue_root(root)
        if validation["valid"]:
            parent = read_unified_parent(input_path)
    dialogue = parent.get("dialogue") or {}
    turns = dialogue.get("turns") or []
    turn_refs = [str(row.get("turn_id") or "") for row in turns]
    parent_tail_refs = turn_refs[-7:]
    checks.update({
        "authoritative_input_valid": bool(validation["valid"]),
        "root_identity_valid": root.get("root_id") == root_id,
        "parent_nonempty": bool(turns),
        "parent_turn_refs_unique": bool(turn_refs) and
            len(turn_refs) == len(set(turn_refs)) and all(turn_refs),
        "dynamic_parent_tail_seven": len(parent_tail_refs) == 7,
        "parent_session_count_consistent": int(
            (root.get("dialogue") or {}).get("session_count") or 0) ==
            len({str(row.get("session_id") or "") for row in turns}),
        "parent_turn_count_consistent": int(
            (root.get("dialogue") or {}).get("turn_count") or 0) == len(turns),
    })
    _, b1_ready = _b1_readiness()
    checks["phase_b1_required_readiness"] = b1_ready
    implementation = frozen_preflight.implementation_manifest_v11()
    import_audit = frozen_preflight._recursive_import_audit(
        Path(expansion_runtime.__file__).resolve())
    checks.update({
        "implementation_manifest_valid": implementation["valid"],
        "recursive_import_graph_clean": import_audit["valid"],
        "allowed_live_roles_exact": ALLOWED_LIVE_ROLES_V11 == {
            "planner_v11", "realizer_v11", "round_expansion_validator_v1"},
        "compileall_passed": compileall.compile_dir(
            str(frozen_preflight.PACKAGE_ROOT), quiet=1),
        "git_diff_check_passed": _git_diff_check(),
    })
    api_configuration = None
    try:
        api = expansion_runtime.live_api_configuration_v11(api_config_path)
        api_configuration = {
            "model": api.model, "base_url": api.base_url,
            "temperature": api.temperature,
            "timeout_seconds": api.timeout_seconds,
            "enable_thinking": api.enable_thinking,
            "api_key_recorded": False,
        }
        AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        checks["api_configuration_valid"] = True
        checks["local_tokenizer_valid"] = True
    except Exception:
        checks["api_configuration_valid"] = False
        checks["local_tokenizer_valid"] = False
    planner_transport = planner_transport_configuration_v11()
    validator_transport = round_validator_transport_configuration_v11()
    checks.update({
        "planner_read_timeout_600": planner_transport["read_timeout_seconds"] == 600,
        "planner_streaming": planner_transport["streaming"] is True,
        "planner_output_10240": planner_transport["max_output_tokens"] == 10_240,
        "validator_read_timeout_600": validator_transport[
            "read_timeout_seconds"] == 600,
        "validator_streaming": validator_transport["streaming"] is True,
        "validator_output_12288": validator_transport[
            "max_output_tokens"] == 12_288,
    })
    prompt_versions = prompt_schema_versions_v11()
    prompt_sha = sha256_bytes(canonical_json(prompt_versions).encode("utf-8"))
    errors = sorted(key for key, value in checks.items() if not value)
    input_sha = sha256_file(input_path) if input_path.is_file() else None
    parent_sha = (sha256_bytes(canonical_json([dialogue]).encode("utf-8"))
                  if dialogue else None)
    return {
        "preflight_version": "slim_v11_generic_root_preflight_v1",
        "valid": not errors,
        "errors": errors,
        "checks": checks,
        "root_id": root_id,
        "runner_path": str(Path(expansion_runtime.__file__).resolve()),
        "run_root": str(run_root),
        "input_path": str(input_path),
        "input_sha256": input_sha,
        "parent_dialogue_sha256": parent_sha,
        "parent_turn_count": len(turns),
        "parent_session_count": len({str(row.get("session_id") or "")
                                     for row in turns}),
        "parent_tail_required": parent_tail_refs,
        "orchestration_seed": _root_seed(root_id),
        "api_config_path": str(api_config_path),
        "api_config_sha256": sha256_file(api_config_path)
            if api_config_path.is_file() else None,
        "api_configuration": api_configuration,
        "tokenizer_path": str(tokenizer_path),
        "implementation_manifest": implementation,
        "project_working_tree_manifest_sha256": project_diff_sha256(PROJECT_ROOT),
        "git_head": git_value(PROJECT_ROOT, "rev-parse", "HEAD"),
        "historical_manifests": frozen_preflight.historical_manifests_v11(),
        "prompt_schema_versions": prompt_versions,
        "prompt_schema_contract_sha256": prompt_sha,
        "planner_transport_configuration": planner_transport,
        "round_validator_transport_configuration": validator_transport,
        "realizer_transport_configuration": {
            "read_timeout_seconds": api_configuration["timeout_seconds"]
                if api_configuration else None,
            "streaming": False,
            "automatic_retry_limit": 0,
        },
        "recursive_import_audit": import_audit,
        "request_budget": {
            "batch_cost_control": True,
            "planner_max": PLANNER_ATTEMPT_LIMIT,
            "realizer_per_scene_max": REALIZER_SCENE_ATTEMPT_LIMIT,
            "round_validator_max": ROUND_VALIDATOR_ATTEMPT_LIMIT,
            "round_validator_terminal_repair_max": 0,
            "scene_min": 5, "scene_max": 7,
            "worst_case": (
                PLANNER_ATTEMPT_LIMIT +
                7 * REALIZER_SCENE_ATTEMPT_LIMIT +
                ROUND_VALIDATOR_ATTEMPT_LIMIT),
            "evidence_committer": 0,
            "initial_structure_extractor": 0,
        },
        "growth_profile": None,
        "dialogue_first_participants": True,
        "durable_update_policy": DURABLE_UPDATE_POLICY,
        "evidence_mismatch_policy": EVIDENCE_MISMATCH_POLICY,
        "api_calls": 0,
        "network_attempts": 0,
    }


def preflight_batch(
    *, batch_root: Path, root_ids: Sequence[str], api_config_path: Path,
    tokenizer_path: Path, solver_model: Path,
) -> dict[str, Any]:
    unique_roots = list(dict.fromkeys(root_ids))
    checks = {
        "batch_root_absent": not batch_root.exists(),
        "at_least_two_distinct_roots": len(unique_roots) >= 2,
        "solver_model_exists": solver_model.is_dir(),
        "api_config_exists": api_config_path.is_file(),
        "tokenizer_exists": tokenizer_path.exists(),
    }
    roots = []
    for root_id in unique_roots:
        paths = _root_paths(batch_root, root_id)
        input_path = ROOTS / root_id / "unified_root.json"
        report = preflight_root_expansion(
            root_id=root_id, input_path=input_path,
            run_root=paths["expansion"], api_config_path=api_config_path,
            tokenizer_path=tokenizer_path)
        roots.append({"root_id": root_id, "input_path": str(input_path),
                      "paths": {key: str(value) for key, value in paths.items()},
                      "expansion_preflight": report})
        checks[f"root_preflight:{root_id}"] = report["valid"]
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "batch_preflight_version": BATCH_VERSION,
        "valid": not failed,
        "failed_checks": failed,
        "checks": checks,
        "batch_root": str(batch_root),
        "root_ids": unique_roots,
        "roots": roots,
        "request_budget": {
            "per_root_expansion_max": (
                PLANNER_ATTEMPT_LIMIT +
                7 * REALIZER_SCENE_ATTEMPT_LIMIT +
                ROUND_VALIDATOR_ATTEMPT_LIMIT),
            "qa_candidate_limit": QA_CANDIDATE_LIMIT,
            "stop_on_first_admitted": STOP_ON_FIRST_ADMITTED,
            "per_candidate_questioner_max": QUESTIONER_ATTEMPT_LIMIT,
            "per_candidate_semantic_audit_max": 2,
            "per_candidate_distractor_max": DISTRACTOR_ATTEMPT_LIMIT,
            "per_candidate_solver_guided_distractor_retry_max": (
                SOLVER_GUIDED_ATTEMPT_LIMIT),
            "solver_guided_retry_per_candidate_max": 1,
            "per_root_paid_max": PER_ROOT_PAID_REQUEST_LIMIT,
            "batch_paid_max": PER_ROOT_PAID_REQUEST_LIMIT * len(unique_roots),
            "local_solver_unmetered_paid_api": True,
        },
        "api_calls": 0,
        "network_attempts": 0,
    }


def _usage(summary: Mapping[str, Any] | None) -> tuple[int, int]:
    if not summary:
        return 0, 0
    api = summary.get("api_usage") or summary.get("api_statistics") or {}
    # Expansion records a request before its transport freeze gate.  Count
    # completed role requests here so a locally rejected envelope is not
    # mislabeled as a paid API call in future batch summaries.
    roles = api.get("roles") or {}
    calls = (sum(int((row or {}).get("completed_requests") or 0)
                 for row in roles.values()) if roles else
             int(api.get("total_physical_requests") or
                 api.get("physical_requests") or
                 api.get("questioner_physical_requests") or
                 api.get("semantic_auditor_physical_requests") or 0))
    return calls, int(api.get("total_tokens") or 0)


def expansion_qa_admission(expansion: Mapping[str, Any]) -> dict[str, Any]:
    """Admit reviewed complete dialogue without requiring a Memory transaction."""
    dialogue_ready = bool(expansion.get("dialogue_ready") or
                          expansion.get("generation_ready"))
    qa_source_ready = bool(expansion.get("qa_source_ready"))
    memory_growth_ready = bool(expansion.get("memory_growth_ready") or
                               expansion.get("fresh_e2e_ready"))
    return {
        "admission_version": "dialogue_first_qa_admission_v1",
        "valid": dialogue_ready and qa_source_ready,
        "dialogue_ready": dialogue_ready,
        "qa_source_ready": qa_source_ready,
        "memory_growth_ready": memory_growth_ready,
        "memory_growth_required": False,
    }


def run_batch(
    *, batch_root: Path, root_ids: Sequence[str], api_config_path: Path,
    tokenizer_path: Path, solver_model: Path, preflight_only: bool = False,
    no_api: bool = False,
) -> dict[str, Any]:
    preflight = preflight_batch(
        batch_root=batch_root, root_ids=root_ids,
        api_config_path=api_config_path, tokenizer_path=tokenizer_path,
        solver_model=solver_model)
    if preflight_only or no_api:
        return preflight
    if not preflight["valid"]:
        raise RuntimeError(
            "PRE_API_PREFLIGHT_FAILED:" + ",".join(preflight["failed_checks"]))
    batch_root.mkdir(parents=True, exist_ok=False)
    (batch_root / "reports").mkdir(parents=True, exist_ok=True)
    freeze = {
        "freeze_version": BATCH_VERSION,
        "created_at": _now(),
        "batch_root": str(batch_root),
        "fresh": True, "no_resume": True,
        "root_ids": list(preflight["root_ids"]),
        "api_config_sha256": sha256_file(api_config_path),
        "tokenizer_path": str(tokenizer_path),
        "solver_model": str(solver_model),
        "request_budget": preflight["request_budget"],
        "evidence_mismatch_policy": EVIDENCE_MISMATCH_POLICY,
        "git_head": git_value(PROJECT_ROOT, "rev-parse", "HEAD"),
        "project_diff_sha256": project_diff_sha256(PROJECT_ROOT),
        "runner_sha256": sha256_file(Path(__file__)),
        "preflight": preflight,
    }
    atomic_json(batch_root / "batch_freeze_manifest.json", freeze)
    atomic_json(batch_root / "run_status.json", {
        "status": "initialized", "created_at": _now(), "api_calls": 0})
    rows: list[dict[str, Any]] = []
    request_budget = BatchRequestBudget(
        root_ids=preflight["root_ids"],
        per_root_paid_limit=PER_ROOT_PAID_REQUEST_LIMIT,
        batch_paid_limit=PER_ROOT_PAID_REQUEST_LIMIT * len(preflight["root_ids"]),
    )

    def paid_observer(root_id: str, item_ref: str):
        def observe(stage: str, is_retry: bool, attempt_index: int) -> None:
            request_budget.reserve(
                root_id=root_id, stage=stage, item_ref=item_ref,
                is_retry=is_retry)
        return observe

    # Candidate generation and local Solver filtering are interleaved so the
    # first strict admission stops all later paid candidate work for this root.
    for root_row in preflight["roots"]:
        root_id = root_row["root_id"]
        paths = {key: Path(value) for key, value in root_row["paths"].items()}
        input_path = Path(root_row["input_path"])
        row: dict[str, Any] = {"root_id": root_id, "paths": {
            key: str(value) for key, value in paths.items()},
            "status": "running", "failure_stage": None,
            "qa_candidates": [], "selected_candidate_index": None}
        rows.append(row)
        try:
            expansion_preflight = deepcopy(root_row["expansion_preflight"])
            random.seed(expansion_preflight["orchestration_seed"])
            expansion = expansion_runtime.execute_live_canary(
                preflight=expansion_preflight, input_path=input_path,
                run_root=paths["expansion"], api_config_path=api_config_path,
                tokenizer_path=tokenizer_path, growth_profile=None,
                dialogue_first_participants=True, root_id=root_id,
                durable_update_policy=DURABLE_UPDATE_POLICY,
                orchestration_seed=expansion_preflight["orchestration_seed"],
                run_id=f"{root_id}-slim-v11-batch-pilot-1",
                parent_tail_refs=expansion_preflight["parent_tail_required"],
                before_paid_request=paid_observer(root_id, "expansion"))
            row["expansion"] = expansion
            row["expansion_qa_admission"] = expansion_qa_admission(expansion)
            if not row["expansion_qa_admission"]["valid"]:
                row.update(status="quarantined", failure_stage="expansion")
                continue
            source_tree = hash_tree(paths["expansion"])["tree_sha256"]
            expanded_path = paths["expansion"] / "expanded_dialogue_round_1.json"
            expanded = json.loads(expanded_path.read_text(encoding="utf-8"))
            excluded_questions: list[str] = []
            excluded_support_pairs: list[tuple[str, ...]] = []
            for candidate_index in range(1, QA_CANDIDATE_LIMIT + 1):
                candidate_paths = _candidate_paths(paths["base"], candidate_index)
                candidate: dict[str, Any] = {
                    "candidate_index": candidate_index,
                    "paths": {key: str(value)
                              for key, value in candidate_paths.items()},
                    "status": "running", "failure_stage": None,
                }
                row["qa_candidates"].append(candidate)
                try:
                    qa = qa_runtime.run(
                        source_run=paths["expansion"],
                        run_root=candidate_paths["qa"],
                        api_config_path=api_config_path,
                        tokenizer_path=tokenizer_path,
                        expected_root_id=root_id,
                        expected_source_tree_sha256=source_tree,
                        expected_expanded_file_sha256=sha256_file(expanded_path),
                        expected_parent_turn_count=int(
                            expanded["parent_turn_count"]),
                        expected_child_turn_count=int(
                            expanded["child_turn_count"]),
                        evidence_mismatch_policy=EVIDENCE_MISMATCH_POLICY,
                        candidate_index=candidate_index,
                        excluded_questions=tuple(excluded_questions),
                        excluded_support_pairs=tuple(excluded_support_pairs),
                        maximum_attempts=QUESTIONER_ATTEMPT_LIMIT,
                        before_paid_request=paid_observer(
                            root_id, f"candidate_{candidate_index:02d}:questioner"))
                    candidate["qa"] = qa
                    _remember_questioner_attempts(
                        qa, excluded_questions=excluded_questions,
                        excluded_support_pairs=excluded_support_pairs)
                    question = str((qa.get("qa") or {}).get("question") or "")
                    if question and question not in excluded_questions:
                        excluded_questions.append(question)
                    support = tuple(str(value) for value in
                                    (qa.get("qa") or {}).get(
                                        "answer_support_span_ids") or [])
                    if (2 <= len(support) <= 6 and
                            support not in excluded_support_pairs):
                        excluded_support_pairs.append(support)
                    if not (qa.get("readiness") or {}).get("core_qa_ready"):
                        candidate.update(
                            status="quarantined", failure_stage="questioner",
                            terminal_status=candidate_terminal_status(
                                stage="questioner",
                                final_decision=qa.get("final_decision")))
                        continue
                    qa_audit = qa_audit_runtime.run(
                        source_qa_run=candidate_paths["qa"],
                        run_root=candidate_paths["qa_audit"],
                        api_config_path=api_config_path,
                        tokenizer_path=tokenizer_path,
                        evidence_mismatch_policy=EVIDENCE_MISMATCH_POLICY,
                        before_paid_request=paid_observer(
                            root_id, f"candidate_{candidate_index:02d}:semantic_audit"))
                    candidate["qa_audit"] = qa_audit
                    if not (qa_audit.get("readiness") or {}).get(
                            "open_answer_gold_ready"):
                        candidate.update(
                            status="quarantined", failure_stage="semantic_audit",
                            terminal_status=candidate_terminal_status(
                                stage="semantic_audit",
                                final_decision=qa_audit.get("final_decision")))
                        continue
                    mcq = mcq_runtime.run(
                        semantic_audit_run=candidate_paths["qa_audit"],
                        qa_source_run=candidate_paths["qa"],
                        run_root=candidate_paths["mcq"],
                        api_config_path=api_config_path,
                        tokenizer_path=tokenizer_path, root_id=root_id,
                        candidate_index=candidate_index,
                        maximum_attempts=DISTRACTOR_ATTEMPT_LIMIT,
                        before_paid_request=paid_observer(
                            root_id, f"candidate_{candidate_index:02d}:distractor"))
                    candidate["mcq"] = mcq
                    if not (mcq.get("readiness") or {}).get("mcq_ready"):
                        candidate.update(
                            status="quarantined", failure_stage="distractor",
                            terminal_status=candidate_terminal_status(
                                stage="distractor",
                                final_decision=mcq.get("final_decision")))
                        continue
                    candidate["status"] = "mcq_materialized"
                    spell = spell_runtime.run_filter(
                        mcq_run=candidate_paths["mcq"],
                        qa_source_run=candidate_paths["qa"],
                        run_root=candidate_paths["spell"], solver_model=solver_model,
                        tokenizer_path=tokenizer_path,
                        evidence_mismatch_policy=EVIDENCE_MISMATCH_POLICY)
                    candidate["spell"] = spell
                    if spell.get("qa_pipeline_ready"):
                        candidate.update(
                            status="admitted", failure_stage=None,
                            terminal_status="admitted")
                        row.update(
                            status="admitted", failure_stage=None,
                            selected_candidate_index=candidate_index,
                            selected_candidate_paths=candidate["paths"])
                        for key in ("qa", "qa_audit", "mcq", "spell"):
                            row[key] = candidate.get(key)
                        if candidate_generation_should_stop(candidate):
                            break
                    if solver_guided_retry_required(spell):
                        guided_mcq = mcq_runtime.run(
                            semantic_audit_run=candidate_paths["qa_audit"],
                            qa_source_run=candidate_paths["qa"],
                            run_root=candidate_paths["mcq_solver_guided"],
                            api_config_path=api_config_path,
                            tokenizer_path=tokenizer_path, root_id=root_id,
                            candidate_index=candidate_index,
                            prior_mcq_run=candidate_paths["mcq"],
                            prior_spell_run=candidate_paths["spell"],
                            maximum_attempts=SOLVER_GUIDED_ATTEMPT_LIMIT,
                            before_paid_request=paid_observer(
                                root_id,
                                f"candidate_{candidate_index:02d}:solver_guided"))
                        candidate["mcq_solver_guided"] = guided_mcq
                        guided_attempt: dict[str, Any] = {
                            "attempted": True,
                            "source_final_decision": spell.get("final_decision"),
                            "mcq": guided_mcq,
                            "mcq_path": str(candidate_paths["mcq_solver_guided"]),
                            "spell_path": str(candidate_paths["spell_solver_guided"]),
                        }
                        candidate["solver_guided_retry"] = guided_attempt
                        if (guided_mcq.get("readiness") or {}).get("mcq_ready"):
                            guided_spell = spell_runtime.run_filter(
                                mcq_run=candidate_paths["mcq_solver_guided"],
                                qa_source_run=candidate_paths["qa"],
                                run_root=candidate_paths["spell_solver_guided"],
                                solver_model=solver_model,
                                tokenizer_path=tokenizer_path,
                                evidence_mismatch_policy=EVIDENCE_MISMATCH_POLICY)
                            candidate["spell_solver_guided"] = guided_spell
                            guided_attempt["spell"] = guided_spell
                            guided_attempt["final_decision"] = guided_spell.get(
                                "final_decision")
                            guided_attempt["admitted"] = bool(
                                guided_spell.get("qa_pipeline_ready"))
                            if guided_spell.get("qa_pipeline_ready"):
                                candidate.update(
                                    status="admitted", failure_stage=None,
                                    terminal_status="admitted")
                                row.update(
                                    status="admitted", failure_stage=None,
                                    selected_candidate_index=candidate_index,
                                    selected_candidate_variant="solver_guided_retry")
                                row["qa"] = qa
                                row["qa_audit"] = qa_audit
                                row["mcq"] = guided_mcq
                                row["spell"] = guided_spell
                                row["selected_candidate_paths"] = {
                                    **candidate["paths"],
                                    "selected_mcq": str(
                                        candidate_paths["mcq_solver_guided"]),
                                    "selected_spell": str(
                                        candidate_paths["spell_solver_guided"]),
                                }
                                if candidate_generation_should_stop(candidate):
                                    break
                        else:
                            guided_attempt["admitted"] = False
                    if candidate["status"] != "admitted":
                        final_spell = (candidate.get("spell_solver_guided") or spell)
                        candidate.update(
                            status="quarantined", failure_stage="spell_filter",
                            terminal_status=candidate_terminal_status(
                                stage="spell_filter",
                                final_decision=final_spell.get("final_decision")))
                except PaidRequestBudgetExhausted as error:
                    candidate.update(
                        status="quarantined",
                        failure_stage=str(error).split(":", 1)[0],
                        terminal_status="budget_exhausted",
                        error=f"{type(error).__name__}:{error}")
                    row.update(
                        status="quarantined",
                        failure_stage=str(error).split(":", 1)[0])
                    break
                except Exception as error:
                    candidate.update(
                        status="failed",
                        failure_stage=candidate.get("failure_stage") or "runtime",
                        error=f"{type(error).__name__}:{error}")
            if row["status"] != "admitted":
                terminal = [candidate for candidate in row["qa_candidates"]
                            if candidate.get("qa") or candidate.get("error")]
                if terminal:
                    last = terminal[-1]
                    for key in ("qa", "qa_audit", "mcq", "spell"):
                        if last.get(key) is not None:
                            row[key] = last[key]
                if not str(row.get("failure_stage") or "").startswith(
                        "budget_exhausted_before_"):
                    row.update(
                        status="quarantined", failure_stage="spell_candidate_pool")
        except PaidRequestBudgetExhausted as error:
            row.update(
                status="quarantined",
                failure_stage=str(error).split(":", 1)[0],
                error=f"{type(error).__name__}:{error}")
        except Exception as error:
            row.update(status="failed", failure_stage=row.get("failure_stage") or
                       "runtime", error=f"{type(error).__name__}:{error}")
    paid_calls = paid_tokens = 0
    token_count_by_stage: dict[str, int] = {}

    def add_usage(stage: str, value: Mapping[str, Any] | None) -> None:
        nonlocal paid_calls, paid_tokens
        calls, tokens = _usage(value)
        paid_calls += calls
        paid_tokens += tokens
        token_count_by_stage[stage] = token_count_by_stage.get(stage, 0) + tokens

    for row in rows:
        add_usage("expansion", row.get("expansion"))
        for candidate in row.get("qa_candidates") or []:
            for key, stage in (
                    ("qa", "questioner"),
                    ("qa_audit", "semantic_audit"),
                    ("mcq", "distractor"),
                    ("mcq_solver_guided", "solver_guided_distractor")):
                add_usage(stage, candidate.get(key))
    def record_attempt_errors(
        *, stage: str, item_ref: str, attempts: Sequence[Mapping[str, Any]],
    ) -> None:
        for attempt in attempts:
            validation = attempt.get("validation") or {}
            errors = (attempt.get("errors") or validation.get("errors") or
                      validation.get("reasons") or [])
            if errors:
                request_budget.record_errors(
                    stage=stage, item_ref=item_ref, errors=errors)

    for row in rows:
        root_id = row["root_id"]
        expansion = row.get("expansion") or {}
        record_attempt_errors(
            stage="planner", item_ref=f"{root_id}:round_1",
            attempts=expansion.get("planner_attempts") or [])
        for scene_ref, attempts in (
                expansion.get("realizer_attempts") or {}).items():
            record_attempt_errors(
                stage="realizer", item_ref=f"{root_id}:{scene_ref}",
                attempts=attempts or [])
        record_attempt_errors(
            stage="round_validator", item_ref=f"{root_id}:round_1",
            attempts=((expansion.get("validator_run") or {}).get("attempts") or []))
        for candidate in row.get("qa_candidates") or []:
            item_ref = f"{root_id}:candidate_{candidate['candidate_index']:02d}"
            record_attempt_errors(
                stage="questioner", item_ref=item_ref,
                attempts=((candidate.get("qa") or {}).get(
                    "questioner_result") or {}).get("attempts") or [])
            record_attempt_errors(
                stage="semantic_audit", item_ref=item_ref,
                attempts=(((candidate.get("qa_audit") or {}).get(
                    "contract_result") or {}).get("attempts") or []))
            for key, stage in (("mcq", "distractor"),
                               ("mcq_solver_guided", "solver_guided_distractor")):
                record_attempt_errors(
                    stage=stage, item_ref=item_ref,
                    attempts=(((candidate.get(key) or {}).get(
                        "distractor_result") or {}).get("attempts") or []))
    budget_audit = request_budget.snapshot()
    admitted_count = sum(row["status"] == "admitted" for row in rows)
    evidence_scope_coverage = summarize_evidence_scope_coverage(rows)
    summary = {
        "report_version": BATCH_VERSION,
        "status": "completed",
        "completed_at": _now(),
        "batch_root": str(batch_root),
        "root_count": len(rows),
        "expanded_count": sum(bool(row.get("expansion", {}).get(
            "qa_source_ready")) for row in rows),
        "dialogue_ready_count": sum(bool(row.get("expansion", {}).get(
            "dialogue_ready")) for row in rows),
        "memory_growth_ready_count": sum(bool(row.get("expansion", {}).get(
            "memory_growth_ready")) for row in rows),
        "qa_candidate_count": sum(len(row.get("qa_candidates") or [])
                                  for row in rows),
        "core_qa_candidate_count": sum(bool((candidate.get("qa") or {}).get(
            "readiness", {}).get("core_qa_ready")) for row in rows
            for candidate in row.get("qa_candidates") or []),
        "core_qa_count": sum(any(bool((candidate.get("qa") or {}).get(
            "readiness", {}).get("core_qa_ready"))
            for candidate in row.get("qa_candidates") or []) for row in rows),
        **evidence_scope_coverage,
        "semantic_gold_count": sum(bool(
            (candidate.get("qa_audit") or {}).get("readiness", {}).get(
                "semantic_entailment_independently_verified") and
            not (candidate.get("qa_audit") or {}).get("readiness", {}).get(
                "evidence_mismatch_marked")) for row in rows
            for candidate in row.get("qa_candidates") or []),
        "evidence_mismatch_marked_count": sum(bool(
            (candidate.get("qa_audit") or {}).get("readiness", {}).get(
                "evidence_mismatch_marked")) for row in rows
            for candidate in row.get("qa_candidates") or []),
        "audited_qa_admitted_count": sum(bool(
            (candidate.get("qa_audit") or {}).get("readiness", {}).get(
                "open_answer_gold_ready")) for row in rows
            for candidate in row.get("qa_candidates") or []),
        "mcq_candidate_count": sum(bool((candidate.get("mcq") or {}).get(
            "readiness", {}).get("mcq_ready")) for row in rows
            for candidate in row.get("qa_candidates") or []),
        "mcq_materialized_count": sum(any(bool((candidate.get("mcq") or {}).get(
            "readiness", {}).get("mcq_ready"))
            for candidate in row.get("qa_candidates") or []) for row in rows),
        "spell_admitted_count": sum(row["status"] == "admitted" for row in rows),
        "solver_guided_retry_count": sum(
            bool(candidate.get("solver_guided_retry")) for row in rows
            for candidate in row.get("qa_candidates") or []),
        "solver_guided_admitted_count": sum(
            bool((candidate.get("solver_guided_retry") or {}).get("admitted"))
            for row in rows for candidate in row.get("qa_candidates") or []),
        "quarantined_count": sum(row["status"] == "quarantined" for row in rows),
        "failed_count": sum(row["status"] == "failed" for row in rows),
        "paid_api_calls": paid_calls,
        "paid_api_tokens": paid_tokens,
        "physical_request_count": budget_audit["paid_request_count"],
        "paid_request_count": paid_calls,
        "cached_request_count": budget_audit["cached_request_count"],
        "retry_request_count": budget_audit["retry_request_count"],
        "request_count_by_stage": budget_audit["request_count_by_stage"],
        "token_count_by_stage": dict(sorted(token_count_by_stage.items())),
        "request_budget_audit": budget_audit,
        "requests_per_admitted_sample": (
            paid_calls / admitted_count if admitted_count else None),
        "tokens_per_admitted_sample": (
            paid_tokens / admitted_count if admitted_count else None),
        "cost_per_admitted_sample": None,
        "cost_per_admitted_sample_unavailable_reason": (
            "provider_price_not_present_in_frozen_configuration"),
        "candidate_index_of_first_admission": {
            row["root_id"]: row.get("selected_candidate_index")
            for row in rows if row.get("selected_candidate_index") is not None
        },
        "wasted_requests_by_error_signature": budget_audit[
            "wasted_requests_by_error_signature"],
        "rows": rows,
        "training_ready": False,
        "batch_production_ready": (
            any(row["status"] == "admitted" for row in rows) and
            not any(row["status"] == "failed" for row in rows)),
    }
    atomic_json(batch_root / "reports/END_TO_END_BATCH_PILOT_SUMMARY.json", summary)
    lines = ["# Slim v11 End-to-end Batch Pilot", "",
             f"- Roots: `{len(rows)}`",
             f"- Expanded: `{summary['expanded_count']}`",
             f"- Core QA: `{summary['core_qa_count']}`",
             f"- Evidence scopes: `{summary['evidence_scope_counts']}`",
             f"- Child-involved QA coverage: `{summary['child_evidence_coverage_ratio']:.3f}` (soft target `{CHILD_INVOLVED_QA_TARGET_RATIO:.2f}`)",
             f"- Semantic gold: `{summary['semantic_gold_count']}`",
             f"- Evidence mismatch marked: `{summary['evidence_mismatch_marked_count']}`",
             f"- Audited QA admitted: `{summary['audited_qa_admitted_count']}`",
             f"- MCQ materialized: `{summary['mcq_materialized_count']}`",
             f"- SPELL admitted: `{summary['spell_admitted_count']}`",
             f"- Quarantined / failed: `{summary['quarantined_count']} / {summary['failed_count']}`",
             f"- Paid calls / tokens: `{paid_calls} / {paid_tokens}`", "",
             f"- Retry calls: `{summary['retry_request_count']}`",
             f"- QA candidate limit / early stop: `{QA_CANDIDATE_LIMIT} / {STOP_ON_FIRST_ADMITTED}`",
             "",
             "## Roots", ""]
    for row in rows:
        lines.append(
            f"- `{row['root_id']}` — `{row['status']}`"
            f" ({row.get('failure_stage') or 'complete'})")
    lines.extend(["", "No Round 2 or training was run.", ""])
    atomic_text(batch_root / "reports/END_TO_END_BATCH_PILOT_SUMMARY.md",
                "\n".join(lines))
    atomic_json(batch_root / "run_status.json", {
        "status": "completed", "completed_at": _now(),
        "paid_api_calls": paid_calls,
        "spell_admitted_count": summary["spell_admitted_count"]})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a fresh multi-root v11 expansion-to-MCQ pilot")
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--root-id", action="append", dest="root_ids")
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-api", action="store_true")
    args = parser.parse_args(argv)
    result = run_batch(
        batch_root=args.batch_root.resolve(),
        root_ids=tuple(args.root_ids or DEFAULT_ROOT_IDS),
        api_config_path=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve(),
        solver_model=args.solver_model.resolve(),
        preflight_only=args.preflight_only, no_api=args.no_api)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("valid", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BATCH_VERSION", "expansion_qa_admission", "preflight_batch",
           "preflight_root_expansion", "run_batch",
           "solver_guided_retry_required"]
