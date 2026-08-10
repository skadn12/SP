"""One fresh cross-boundary core-QA canary for a v11 expanded dialogue."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from transformers import AutoTokenizer

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..lineage_three_stage_canary import ApiBudget, run_single_object_task
from ..live_backends import (LiveSemanticRepairBackend,
                             OpenAICompatibleChallengerClient,
                             load_challenger_config)
from ..questioner_contract import span_catalog_sha256
from ..single_item_qa_canary import freeze_core_qa, frozen_core_qa_metadata
from .continuous_expansion_qa import (
    CONTINUOUS_EXPANSION_QA_PROMPT_VERSION,
    CONTINUOUS_EXPANSION_QA_VERSION,
    MAX_COMPLETION_TOKENS,
    MODEL_CONTEXT_LIMIT,
    build_cross_boundary_questioner_prompt,
    build_qa_source_view,
    protocol_sha256,
    selected_evidence_rows,
    validate_cross_boundary_core_object,
)
from .provenance import git_value, hash_tree, project_diff_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_34"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_v11_canary34_qa_canary_1"
)
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
EXPECTED_SOURCE_TREE_SHA256 = (
    "beee9e4a36afe1f3cc662c5374b15ec8f253ebf99502d2a787d826260641f61d"
)
EXPECTED_EXPANDED_FILE_SHA256 = (
    "8a84db61cdc8e73896318afb8ace69d2d9ff886f59abd8958d9f46ec85cbeb21"
)
QUESTIONER_RETRY_CONTRACT_VERSION = (
    "cross_boundary_questioner_retry_v6_questioner_owned_evidence")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def build_questioner_retry_suffix(
    raw: Any, validation: dict[str, Any],
    lineage_pair_registry: dict[str, Any] | None = None,
    frozen_support_pair: dict[str, Any] | None = None,
) -> str:
    if frozen_support_pair is None and not (
            (lineage_pair_registry or {}).get("rows")):
        return " ".join([
            "QUESTION AND EVIDENCE CONTRACT RETRY.",
            "The previous candidate failed:",
            ", ".join(validation.get("reasons") or
                      ["unknown_contract_error"]) + ".",
            "Return exactly question, correct_answer_text, and "
            "answer_support_span_ids.",
            "Select 2 to 6 exact span IDs from the unchanged complete dialogue "
            "that actually support the question. Prefer an under-covered "
            "promoted_child development, but do not force unrelated evidence.",
            "Choose the minimal sufficient evidence yourself. Do not invent an "
            "ID, and do not rely on World Memory or unstated facts.",
            "The answer must directly answer the question; the program will not "
            "replace or synthesize it.",
            "Do not repeat an excluded question or an excluded support set.",
        ])
    lineage_pair_registry = lineage_pair_registry or {"rows": []}
    allowed_pairs = {
        (str(pair.get("parent_span_id") or ""),
         str(pair.get("child_span_id") or "")): str(pair.get("speaker") or "")
        for row in lineage_pair_registry.get("rows") or []
        for pair in row.get("eligible_same_speaker_pairs") or []
    }
    submitted = (raw.get("answer_support_span_ids")
                 if isinstance(raw, dict) else None)
    pair = tuple(str(value) for value in submitted) if (
        isinstance(submitted, list) and len(submitted) == 2) else ()
    if frozen_support_pair is not None:
        frozen_speaker = str(frozen_support_pair["speaker"])
        pair_rule = (
            "The program has already frozen and will bind the support pair. Its "
            "exact speaker is "
            f"{json.dumps(frozen_speaker, ensure_ascii=False)}. "
            "Rewrite only the question and do not return any support ID. "
            f"The only dialogue speaker label allowed in the question is "
            f"{json.dumps(frozen_speaker, ensure_ascii=False)}. Do not name the "
            "person they spoke with or any other dialogue participant.")
    elif pair in allowed_pairs:
        pair_rule = (
            "Reuse this exact frozen support pair without changing either ID: "
            f"{json.dumps(list(pair), ensure_ascii=False)}. Its exact speaker is "
            f"{json.dumps(allowed_pairs[pair], ensure_ascii=False)}. Rewrite only "
            "the question so it is supported by those two turns.")
    else:
        pair_rule = (
            "Select one exact pair from ALLOWED LINEAGE PAIRS. Both spans must "
            "have the same exact speaker label.")
    return " ".join([
        "QUESTION SELECTION CONTRACT RETRY.",
        "The previous candidate failed:",
        ", ".join(validation.get("reasons") or ["unknown_contract_error"]) + ".",
        pair_rule,
        "Keep the frozen dialogue and ALLOWED LINEAGE PAIRS unchanged.",
        "Do not repeat an excluded question or excluded support pair.",
        "Every named dialogue speaker must speak in one selected span or be "
        "explicitly named by one selected span; remove any unsupported named speaker.",
        "Keep the question abstract: do not copy two or more distinctive "
        "one-sided content words from either selected turn, and do not state the "
        "concrete later outcome that the answer must supply.",
        "If the error says question_repeats_support_details, remove the concrete "
        "parent object/event and child project/action. Retain one abstract shared "
        "theme such as approach, responsibility, planning, support, or confidence.",
        "Do not use an evolve/change-from-X-to-Y stem that already states both "
        "transition endpoints.",
        "Do not ask why, what prompted/caused/motivated the change, or what "
        "happened in response to something. Ask a descriptive comparison.",
        "Return only the question field.",
    ])


def preflight(*, source_run: Path, run_root: Path, api_config: Path,
              tokenizer_path: Path,
              expected_root_id: str = "cc-train-episode-61555",
              expected_source_tree_sha256: str | None = EXPECTED_SOURCE_TREE_SHA256,
              expected_expanded_file_sha256: str | None = EXPECTED_EXPANDED_FILE_SHA256,
              expected_parent_turn_count: int | None = 60,
              expected_child_turn_count: int | None = 32,
              candidate_index: int = 1,
              excluded_questions: tuple[str, ...] = (),
              excluded_support_pairs: tuple[tuple[str, str], ...] = (),
              ) -> dict[str, Any]:
    if candidate_index < 1:
        raise ValueError("qa_candidate_index_invalid")
    expanded_path = source_run / "expanded_dialogue_round_1.json"
    manifest_path = source_run / "complete_dialogue_manifest.json"
    summary_path = source_run / "reports/SLIM_CONTINUOUS_EXPANSION_V11_LIVE_CANARY_REPORT.json"
    checks: dict[str, bool] = {
        "run_root_absent": not run_root.exists(),
        "source_run_exists": source_run.is_dir(),
        "expanded_dialogue_exists": expanded_path.is_file(),
        "complete_dialogue_manifest_exists": manifest_path.is_file(),
        "source_summary_exists": summary_path.is_file(),
        "api_config_exists": api_config.is_file(),
        "tokenizer_exists": tokenizer_path.exists(),
    }
    errors: list[str] = []
    source_tree = hash_tree(source_run) if source_run.is_dir() else None
    if source_tree:
        checks["source_tree_sha_matches"] = (
            expected_source_tree_sha256 is None or
            source_tree["tree_sha256"] == expected_source_tree_sha256)
    if expanded_path.is_file():
        checks["expanded_file_sha_matches"] = (
            expected_expanded_file_sha256 is None or
            sha256_file(expanded_path) == expected_expanded_file_sha256)
        expanded = _load_json(expanded_path)
        try:
            view = build_qa_source_view(expanded)
        except ValueError as error:
            view = None
            errors.append(str(error))
        if view:
            frozen_support_pair = None
            view["view_sha256"] = sha256_bytes(canonical_json({
                "combined_dialogue": view["combined_dialogue"],
                "span_catalog": view["span_catalog"],
                "partition_by_turn": view["partition_by_turn"],
            }).encode("utf-8"))
            partition_values = list(view["partition_by_turn"].values())
            parent_count = partition_values.count("original_parent")
            child_count = partition_values.count("promoted_child")
            checks.update({
                "root_identity": expanded.get("root_id") == expected_root_id,
                "complete_turn_count_consistent": len(view["combined_dialogue"]) ==
                    int(expanded.get("turn_count") or 0),
                "parent_turn_count_expected": expected_parent_turn_count is None or
                    parent_count == expected_parent_turn_count,
                "child_turn_count_expected": expected_child_turn_count is None or
                    child_count == expected_child_turn_count,
                "parent_child_counts_consistent": (
                    parent_count == int(expanded.get("parent_turn_count") or 0) and
                    child_count == int(expanded.get("child_turn_count") or 0)),
                "stable_unique_turn_refs": len(view["partition_by_turn"]) ==
                    len(view["combined_dialogue"]),
                "qa_source_candidate": expanded.get("qa_source_candidate") is True,
                "source_qa_not_previously_generated": expanded.get("qa_generated") is False,
                "questioner_evidence_selection_enabled": True,
                "program_frozen_support_pair_disabled": frozen_support_pair is None,
            })
            prompt = build_cross_boundary_questioner_prompt(
                root_id=expanded["root_id"], span_catalog=view["span_catalog"],
                partition_by_turn=view["partition_by_turn"],
                excluded_questions=excluded_questions,
                excluded_support_pairs=excluded_support_pairs)
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=True)
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            checks["prompt_within_context"] = (
                prompt_tokens + MAX_COMPLETION_TOKENS <= MODEL_CONTEXT_LIMIT)
        else:
            prompt = ""
            prompt_tokens = 0
            frozen_support_pair = None
    else:
        expanded = None
        view = None
        prompt = ""
        prompt_tokens = 0
        frozen_support_pair = None
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        checks["manifest_qa_source_ready"] = bool(
            (manifest.get("readiness") or {}).get("qa_source_ready"))
        checks["manifest_uses_dialogue_not_memory"] = (
            manifest.get("memory_is_qa_text_source") is False and
            manifest.get("qa_gold_source_role") == "canonical_expanded_dialogue")
    if summary_path.is_file():
        summary = _load_json(summary_path)
        checks["source_generation_completed"] = (
            summary.get("status") == "completed" and
            str(summary.get("conclusion") or "").startswith("A.") and
            summary.get("qa_source_ready") is True)
    for name, passed in checks.items():
        if not passed:
            errors.append(name)
    return {
        "preflight_version": "slim_v11_cross_boundary_qa_preflight_v1",
        "valid": not errors,
        "errors": errors,
        "checks": checks,
        "source_run": str(source_run),
        "source_tree": source_tree,
        "expanded_dialogue_path": str(expanded_path),
        "expanded_dialogue_file_sha256": sha256_file(expanded_path)
            if expanded_path.is_file() else None,
        "source_view_sha256": view["view_sha256"] if view else None,
        "lineage_pair_registry_sha256": None,
        "lineage_pair_row_count": 0,
        "span_catalog_sha256": span_catalog_sha256(view["span_catalog"])
            if view else None,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")) if prompt else None,
        "prompt_tokens": prompt_tokens,
        "candidate_index": candidate_index,
        "excluded_question_count": len(excluded_questions),
        "excluded_support_pair_count": len(excluded_support_pairs),
        "frozen_support_pair": frozen_support_pair,
        "evidence_selection_owner": "questioner",
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "context_limit": MODEL_CONTEXT_LIMIT,
        "expanded": expanded,
        "view": view,
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    qa = summary.get("qa") or {}
    evidence = qa.get("selected_evidence") or []
    lines = [
        "# Slim v11 Canary34 Cross-boundary QA Canary 1", "",
        f"- Status: `{summary['status']}`",
        f"- Final decision: `{summary['final_decision']}`",
        f"- Physical Questioner requests: `{summary['api_statistics']['questioner_physical_requests']}`",
        f"- Prompt/completion/total tokens: `{summary['api_statistics']['prompt_tokens']} / {summary['api_statistics']['completion_tokens']} / {summary['api_statistics']['total_tokens']}`",
        f"- Core QA ready: `{str(summary['readiness']['core_qa_ready']).lower()}`",
        f"- Evidence scope: `{summary.get('evidence_scope')}`",
        f"- Cross-boundary coverage: `{str(summary['readiness']['cross_boundary_qa_ready']).lower()}`",
        f"- Full MCQ/Solver pipeline ready: `{str(summary['readiness']['qa_pipeline_ready']).lower()}`",
        "", "## Generated QA", "",
        f"**Question:** {qa.get('question', '')}", "",
        f"**Answer:** {qa.get('correct_answer_text', '')}", "",
        "## Evidence", "",
    ]
    for row in evidence:
        lines.append(
            f"- `{row['span_id']}` ({row['source_partition']}) — "
            f"**{row['speaker']}**: {row['exact_text']}")
    lines.extend(["", "## Scope", "",
        "This canary generated one open-answer core QA from the 92-turn canonical expanded dialogue. It did not generate distractors or run a Solver. Mechanical grounding and the parent/child evidence boundary were checked locally; semantic entailment was not independently model-audited.", ""])
    return "\n".join(lines)


def run(*, source_run: Path, run_root: Path, api_config_path: Path,
        tokenizer_path: Path, preflight_only: bool = False,
        no_api: bool = False,
        expected_root_id: str = "cc-train-episode-61555",
        expected_source_tree_sha256: str | None = EXPECTED_SOURCE_TREE_SHA256,
        expected_expanded_file_sha256: str | None = EXPECTED_EXPANDED_FILE_SHA256,
        expected_parent_turn_count: int | None = 60,
        expected_child_turn_count: int | None = 32,
        evidence_mismatch_policy: str = "strict",
        candidate_index: int = 1,
        excluded_questions: tuple[str, ...] = (),
        excluded_support_pairs: tuple[tuple[str, ...], ...] = (),
        maximum_attempts: int = 3,
        before_paid_request: Callable[[str, bool, int], None] | None = None,
        ) -> dict[str, Any]:
    if evidence_mismatch_policy not in {"strict", "mark_only"}:
        raise ValueError("evidence_mismatch_policy_invalid")
    if maximum_attempts not in {1, 2, 3}:
        raise ValueError("questioner_maximum_attempts_invalid")
    frozen = preflight(source_run=source_run, run_root=run_root,
                       api_config=api_config_path,
                       tokenizer_path=tokenizer_path,
                       expected_root_id=expected_root_id,
                       expected_source_tree_sha256=expected_source_tree_sha256,
                       expected_expanded_file_sha256=expected_expanded_file_sha256,
                       expected_parent_turn_count=expected_parent_turn_count,
                       expected_child_turn_count=expected_child_turn_count,
                       candidate_index=candidate_index,
                       excluded_questions=excluded_questions,
                       excluded_support_pairs=excluded_support_pairs)
    public_preflight = {key: value for key, value in frozen.items()
                        if key not in {"prompt", "expanded", "view"}}
    if preflight_only or no_api:
        return public_preflight
    if not frozen["valid"]:
        raise RuntimeError(f"PRE_API_PREFLIGHT_FAILED:{frozen['errors']}")
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("config", "frozen", "qa", "reports", "raw/prompts"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    api = replace(load_challenger_config(api_config_path),
                  timeout_seconds=600,
                  questioner_max_tokens=MAX_COMPLETION_TOKENS,
                  temperature=0.25,
                  enable_thinking=False)
    source_tree_before = frozen["source_tree"]
    code_manifest_before = {
        "git_head": git_value(PROJECT_ROOT, "rev-parse", "HEAD"),
        "project_diff_sha256": project_diff_sha256(PROJECT_ROOT),
        "protocol_sha256": protocol_sha256(),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    freeze_manifest = {
        "freeze_version": "slim_v11_cross_boundary_qa_freeze_v1",
        "created_at": _utc_now(),
        "run_root": str(run_root),
        "source_run": str(source_run),
        "source_run_tree_sha256": source_tree_before["tree_sha256"],
        "expanded_dialogue_file_sha256": frozen["expanded_dialogue_file_sha256"],
        "source_view_sha256": frozen["source_view_sha256"],
        "span_catalog_sha256": frozen["span_catalog_sha256"],
        "prompt_version": CONTINUOUS_EXPANSION_QA_PROMPT_VERSION,
        "retry_contract_version": QUESTIONER_RETRY_CONTRACT_VERSION,
        "prompt_sha256": frozen["prompt_sha256"],
        "prompt_tokens": frozen["prompt_tokens"],
        "protocol_sha256": protocol_sha256(),
        "model_configuration": {
            "model": api.model, "base_url": api.base_url,
            "temperature": api.temperature,
            "timeout_seconds": api.timeout_seconds,
            "enable_thinking": api.enable_thinking,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "api_key_recorded": False,
        },
        "request_budget": {"questioner_normal": 1,
                           "format_only_retry": maximum_attempts - 1,
                           "physical_maximum": maximum_attempts,
                           "distractor": 0, "solver": 0, "training": 0},
        "code_manifest": code_manifest_before,
        "preflight": public_preflight,
        "world_memory_read": False,
        "source_fact_role": "canonical_expanded_dialogue",
        "evidence_mismatch_policy": evidence_mismatch_policy,
        "candidate_index": candidate_index,
        "excluded_questions": list(excluded_questions),
        "excluded_support_pairs": [list(pair) for pair in excluded_support_pairs],
        "evidence_selection_owner": "questioner",
        "program_frozen_support_pair": None,
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze_manifest)
    atomic_json(run_root / "run_status.json", {
        "status": "initialized", "created_at": _utc_now(),
        "api_calls": 0, "source_run_modified": False})
    atomic_json(run_root / "frozen/qa_source_view.json", {
        "view_sha256": frozen["view"]["view_sha256"],
        "combined_dialogue": frozen["view"]["combined_dialogue"],
        "partition_by_turn": frozen["view"]["partition_by_turn"],
        "evidence_selection_owner": "questioner",
        "program_frozen_support_pair": None,
    })
    atomic_json(run_root / "frozen/span_catalog.json",
                frozen["view"]["span_catalog"])
    atomic_text(run_root / "frozen/questioner_prompt.txt", frozen["prompt"])
    manifest_sha = sha256_file(run_root / "run_freeze_manifest.json")
    ledger = RequestLedger(run_root, manifest_sha, "live")
    backend = LiveSemanticRepairBackend(
        OpenAICompatibleChallengerClient(api), MAX_COMPLETION_TOKENS)
    budget = ApiBudget(normal_caps={"qa_writer": 1},
                       format_retry_cap=maximum_attempts - 1,
                       physical_cap=maximum_attempts)
    result = run_single_object_task(
        run_root=run_root, ledger=ledger, backend=backend, budget=budget,
        stage="qa_writer", logical_request_id=(
            f"{expected_root_id}-round1-cross-boundary-qa-{candidate_index:02d}"),
        semantic_prompt=frozen["prompt"], lineage_view=frozen["view"],
        validator=lambda raw: validate_cross_boundary_core_object(
            raw, span_catalog=frozen["view"]["span_catalog"],
            partition_by_turn=frozen["view"]["partition_by_turn"],
            evidence_mismatch_policy=evidence_mismatch_policy,
            excluded_questions=excluded_questions,
            excluded_support_pairs=excluded_support_pairs),
        schema_retry_decisions=("cross_boundary_qa_schema_invalid",
                                "cross_boundary_qa_contract_reject"),
        retry_prompt_suffix_factory=lambda raw, validation:
            build_questioner_retry_suffix(raw, validation),
        max_attempts=maximum_attempts,
        ledger_budget_limit=maximum_attempts,
        ledger_total_budget_limit=maximum_attempts,
        before_paid_request=before_paid_request,
        json_invalid_decision="cross_boundary_qa_json_invalid")
    qa_payload: dict[str, Any] = {}
    admission: dict[str, Any] | None = None
    if result["accepted"]:
        # Freeze the Questioner's complete three-field object.  The program owns
        # ID validation only; it neither selects evidence nor rewrites the answer.
        accepted_validation = result.get("accepted_validation") or {}
        accepted_core = accepted_validation.get("core_qa")
        frozen_qa = freeze_core_qa(
            accepted_core,
            lineage_id=f"{expected_root_id}-round-1",
            sample_index=candidate_index - 1,
            span_catalog=frozen["view"]["span_catalog"])
        admission = {
            "decision": "cross_boundary_reference_accept",
            "reasons": [],
            "admission_scope": "reference_integrity_only_not_entailment",
            "questioner_selected_evidence": True,
            "program_selected_evidence": False,
            "selected_span_count": len(frozen_qa["answer_support_span_ids"]),
            "selected_span_ids": list(frozen_qa["answer_support_span_ids"]),
            "reference_integrity_verified": True,
            "semantic_entailment_verified": False,
        }
        evidence = selected_evidence_rows(
            frozen_qa, span_catalog=frozen["view"]["span_catalog"],
            partition_by_turn=frozen["view"]["partition_by_turn"])
        qa_payload = {**frozen_qa,
            "source_dialogue_path": str(source_run / "expanded_dialogue_round_1.json"),
            "source_dialogue_sha256": frozen["expanded_dialogue_file_sha256"],
            "selected_evidence": evidence,
            "source_partitions": list(dict.fromkeys(
                row["source_partition"] for row in evidence)),
            "evidence_scope": accepted_validation.get("evidence_scope"),
            "child_involved": bool(
                accepted_validation.get("child_involved")),
            "cross_boundary": {"required": False,
                "present": accepted_validation.get("evidence_scope") ==
                    "parent_child",
                "parent_evidence_count": sum(
                    row["source_partition"] == "original_parent" for row in evidence),
                "child_evidence_count": sum(
                    row["source_partition"] == "promoted_child" for row in evidence)},
            "semantic_entailment_independently_verified": False,
            "evidence_mismatch_policy": evidence_mismatch_policy,
            "evidence_mismatch": bool(
                accepted_validation.get("evidence_mismatch")),
            "evidence_alignment_status": "awaiting_semantic_audit",
            "evidence_mismatch_reasons": list(
                accepted_validation.get("evidence_mismatch_reasons") or []),
            "distractors_generated": False,
            "solver_executed": False,
        }
        atomic_json(run_root / "qa/core_qa.json", qa_payload)
        atomic_json(run_root / "qa/core_qa_metadata.json",
            frozen_core_qa_metadata(
                frozen_core_qa=frozen_qa,
                lineage_id=f"{expected_root_id}-round-1",
                sample_index=candidate_index - 1,
                span_catalog=frozen["view"]["span_catalog"]))
        atomic_json(run_root / "qa/mechanical_admission.json", admission)
    completed = ledger.completed()
    usage = [state.get("response_metadata", {}).get("usage") or {}
             for state in completed]
    api_statistics = {
        "questioner_physical_requests": len(completed),
        "questioner_format_retries": int(result["format_retry_used"]),
        "distractor_requests": 0, "solver_requests": 0,
        "qa_replay_requests": 0, "training_requests": 0,
        "total_physical_requests": len(completed),
        "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in usage),
        "completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in usage),
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in usage),
        "cache_reuse": 0, "imported_response_reuse": 0,
    }
    source_tree_after = hash_tree(source_run)
    code_manifest_after = {
        "git_head": git_value(PROJECT_ROOT, "rev-parse", "HEAD"),
        "project_diff_sha256": project_diff_sha256(PROJECT_ROOT),
        "protocol_sha256": protocol_sha256(),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    mechanical_ready = bool(
        admission and admission.get("decision") ==
        "cross_boundary_reference_accept")
    evidence_scope = str(
        (result.get("accepted_validation") or {}).get("evidence_scope") or
        "unknown")
    complete_ready = bool(result["accepted"] and qa_payload)
    cross_ready = evidence_scope == "parent_child"
    summary = {
        "report_version": "slim_v11_cross_boundary_qa_canary_report_v1",
        "run_root": str(run_root),
        "source_run": str(source_run),
        "candidate_index": candidate_index,
        "status": "completed" if mechanical_ready and complete_ready else "failed",
        "failure_stage": None if mechanical_ready and complete_ready else (
            "mechanical_admission" if result["accepted"] else "questioner_contract"),
        "final_decision": ("A_CORE_QA_ACCEPTED" if mechanical_ready and complete_ready
                           else "B_CORE_QA_REJECTED"),
        "evidence_scope": evidence_scope,
        "questioner_result": result,
        "mechanical_admission": admission,
        "qa": qa_payload,
        "api_statistics": api_statistics,
        "integrity": {
            "source_tree_sha_before": source_tree_before["tree_sha256"],
            "source_tree_sha_after": source_tree_after["tree_sha256"],
            "source_run_unchanged": source_tree_before["tree_sha256"] ==
                source_tree_after["tree_sha256"],
            "code_manifest_before": code_manifest_before,
            "code_manifest_after": code_manifest_after,
            "code_unchanged_during_run": code_manifest_before == code_manifest_after,
            "world_memory_read": False,
        },
        "readiness": {
            "qa_source_ready": True,
            "core_qa_generated": bool(qa_payload),
            "core_qa_ready": mechanical_ready,
            "complete_dialogue_qa_ready": complete_ready,
            "cross_boundary_qa_ready": cross_ready,
            "semantic_entailment_independently_verified": False,
            "evidence_mismatch_marked": bool(
                qa_payload.get("evidence_mismatch")),
            "qa_pipeline_ready": False,
            "mcq_ready": False,
            "solver_validated": False,
        },
        "not_run": ["Distractor", "Solver", "QA Replay", "Training"],
    }
    atomic_json(run_root / "reports/QA_CANARY_SUMMARY.json", summary)
    atomic_text(run_root / "reports/QA_CANARY_SUMMARY.md",
                _summary_markdown(summary))
    atomic_json(run_root / "reports/post_run_provenance.json", {
        "source_tree_before": source_tree_before,
        "source_tree_after": source_tree_after,
        "source_run_unchanged": summary["integrity"]["source_run_unchanged"],
        "code_manifest_before": code_manifest_before,
        "code_manifest_after": code_manifest_after,
        "code_unchanged_during_run": summary["integrity"]["code_unchanged_during_run"],
        "expanded_dialogue_sha256": frozen["expanded_dialogue_file_sha256"],
        "source_fact_role": "canonical_expanded_dialogue",
        "world_memory_read": False,
    })
    atomic_json(run_root / "run_status.json", {
        "status": summary["status"], "completed_at": _utc_now(),
        "api_calls": len(completed), "source_run_modified": False,
        "final_decision": summary["final_decision"]})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--evidence-mismatch-policy",
                        choices=("strict", "mark_only"), default="strict")
    args = parser.parse_args(argv)
    result = run(source_run=args.source_run, run_root=args.run_root,
                 api_config_path=args.api_config, tokenizer_path=args.tokenizer,
                 preflight_only=args.preflight_only, no_api=args.no_api,
                 evidence_mismatch_policy=args.evidence_mismatch_policy)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("valid", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
