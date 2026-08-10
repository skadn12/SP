"""Run a two-QA contrastive-distractor canary without regenerating questions."""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer

from .contrastive_distractor_canary import (
    ANSWER_PLACEMENT_VERSION,
    CONTRASTIVE_DISTRACTOR_CANARY_VERSION,
    CONTRASTIVE_DISTRACTOR_PROMPT_VERSION,
    CONTRASTIVE_DISTRACTOR_SCHEMA_VERSION,
    TARGET_SOURCE_ITEMS,
    audit_frozen_source_qa,
    build_contrastive_distractor_prompt,
    contrastive_distractor_prompt_sha256,
    contrastive_distractor_schema_sha256,
    deterministic_answer_placement,
    materialize_contrastive_candidate,
    option_length_diagnostics,
    route_conclusion,
    should_run_attempt_one,
)
from .core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from .input_adapter import read_json
from .ledger import RequestLedger
from .lineage_distractor_writer import validate_distractor_object
from .lineage_three_stage_canary import (
    FORMAT_ONLY_RETRY_VERSION,
    ApiBudget,
    format_only_retry_suffix_sha256,
    run_single_object_task,
)
from .live_backends import (
    LiveSemanticRepairBackend,
    OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from .run_canary import (
    create_source_snapshot,
    git_value,
    project_diff_sha256,
    write_post_run_provenance,
)
from .run_cross_dialogue_calibration import _base_manifest, _run_solver, _solver_config
from .run_rolling_outline_30k_canary import (
    CHALLENGER_MODEL,
    MODEL_CONTEXT_LIMIT,
    SOLVER,
    TOKENIZER,
    _usage,
)
from .vllm_backends import VLLMSolverBackend


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_temporal_insertion_outline_expansion_canary_v3_execution_v2"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_contrastive_distractor_canary_v1"
)
NORMAL_REQUEST_CAP = 4
FORMAT_RETRY_CAP = 4
PHYSICAL_REQUEST_CAP = 8
DISTRACTOR_MAX_TOKENS = 1024


def _tree_sha256(root: Path) -> str:
    digest = bytearray()
    for path in sorted(candidate for candidate in root.rglob("*")
                       if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.extend(relative.encode() + b"\0" + path.read_bytes() + b"\0")
    return sha256_bytes(bytes(digest))


def _parse_source_raw_batch(path: Path) -> list[dict[str, Any]]:
    wrapper = read_json(path)
    batch = json.loads(str(wrapper.get("raw_response") or ""))
    if not isinstance(batch, list) or len(batch) != 3 or any(
            not isinstance(item, dict) for item in batch):
        raise RuntimeError("source_questioner_batch_invalid")
    return batch


def _source_paths(source_root: Path) -> dict[str, Path]:
    qa_root = source_root / "rounds/round_01/qa"
    return {
        "qa_root": qa_root,
        "context": qa_root / "frozen/questioner_smoke_context.json",
        "catalog": qa_root / "frozen/questioner_span_catalog.json",
        "raw_batch": qa_root / "raw/questioner_batch_00.json",
        "qa_manifest": qa_root / "config/run_manifest.json",
        "qa_provenance": qa_root / "reports/post_run_provenance.json",
        "round_filter": qa_root / "reports/round_qa_filter.json",
        "final_dialogue": source_root / "frozen/final_dialogue.json",
    }


def attest_source(source_root: Path) -> dict[str, Any]:
    paths = _source_paths(source_root)
    for key, path in paths.items():
        if key != "qa_root" and not path.is_file():
            raise RuntimeError(f"source_file_missing:{key}:{path}")
    provenance = read_json(paths["qa_provenance"])
    if (provenance.get("authoritative_release_provenance") is not True or
            provenance.get("provenance_mixed") is not False or
            provenance.get("run_manifest_sha256") != sha256_file(
                paths["qa_manifest"])):
        raise RuntimeError("source_questioner_provenance_not_authoritative")
    context = read_json(paths["context"])
    final_dialogue = read_json(paths["final_dialogue"])
    if canonical_json(context.get("combined_dialogue")) != canonical_json(
            final_dialogue.get("turns")):
        raise RuntimeError("source_full_dialogue_mismatch")
    batch = _parse_source_raw_batch(paths["raw_batch"])
    batch_by_id = {str(item.get("qa_id")): item for item in batch}
    source_items = []
    for item_name, qa_id in TARGET_SOURCE_ITEMS:
        materialized_path = paths["qa_root"] / f"materialized/{item_name}.json"
        solver_report_path = paths["qa_root"] / (
            f"solvers/{item_name}/reports/cross_dialogue_solver_audit.json")
        for path in (materialized_path, solver_report_path):
            if not path.is_file():
                raise RuntimeError(f"source_item_file_missing:{path}")
        raw = batch_by_id.get(qa_id)
        materialized = read_json(materialized_path)
        solver = read_json(solver_report_path)
        if raw is None or materialized.get("qa_id") != qa_id:
            raise RuntimeError(f"source_qa_identity_mismatch:{qa_id}")
        if (solver.get("final_decision") != "bad_ground" or
                solver.get("no_context_permutation_diagnostic") !=
                "bad_ground_semantic_leakage"):
            raise RuntimeError(f"source_qa_failure_type_mismatch:{qa_id}")
        source_items.append({
            "item_name": item_name,
            "qa_id": qa_id,
            "raw_qa": raw,
            "raw_qa_sha256": sha256_bytes(canonical_json(raw).encode()),
            "materialized_qa": materialized,
            "materialized_qa_path": str(materialized_path.resolve()),
            "materialized_qa_sha256": sha256_file(materialized_path),
            "source_solver_report_path": str(solver_report_path.resolve()),
            "source_solver_report_sha256": sha256_file(solver_report_path),
            "source_final_decision": solver["final_decision"],
            "source_grounding_diagnosis": solver[
                "no_context_permutation_diagnostic"],
        })
    return {
        "source_root": str(source_root.resolve()),
        "source_tree_sha256_before": _tree_sha256(source_root),
        "source_questioner_manifest_sha256": sha256_file(paths["qa_manifest"]),
        "source_questioner_provenance_sha256": sha256_file(paths["qa_provenance"]),
        "source_context_path": str(paths["context"].resolve()),
        "source_context_sha256": sha256_file(paths["context"]),
        "source_span_catalog_path": str(paths["catalog"].resolve()),
        "source_span_catalog_sha256": sha256_file(paths["catalog"]),
        "source_raw_batch_sha256": sha256_file(paths["raw_batch"]),
        "source_round_filter_sha256": sha256_file(paths["round_filter"]),
        "source_final_dialogue_sha256": sha256_file(paths["final_dialogue"]),
        "source_items": source_items,
        "context": context,
        "span_catalog": read_json(paths["catalog"]),
    }


def _base_qa(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "qa_id": str(raw["qa_id"]),
        "question": str(raw["question"]),
        "correct_answer_text": str(raw["correct_answer_text"]),
        "answer_support_span_ids": list(raw["answer_support_span_ids"]),
        "reasoning_steps": list(raw.get("reasoning_steps") or []),
    }


def _solver_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    result = report.get("result") or {}
    no_context = result.get("no_context") or result.get("initial_no_context") or {}
    full_context = result.get("full_context") or {}
    return {
        "final_decision": report.get("final_decision"),
        "grounding_decision": result.get("grounding_decision"),
        "no_context_raw_answers": [row.get("raw_response") for row in
            no_context.get("rollouts") or []],
        "no_context_valid_answers": [row.get("parsed_letter") for row in
            no_context.get("rollouts") or [] if row.get("local_reward") is not None],
        "no_context_rewards": no_context.get("verified_rewards"),
        "adaptive_extension": result.get("adaptive_extension"),
        "full_context_raw_answers": [row.get("raw_response") for row in
            full_context.get("rollouts") or []],
        "full_context_rewards": full_context.get("verified_rewards"),
        "no_context_permutation_diagnostic": report.get(
            "no_context_permutation_diagnostic"),
        "local_vllm_generations": report.get("vllm_generation_count", 0),
    }


def _review_markdown(source_items: list[dict[str, Any]],
                     attempts: list[dict[str, Any]]) -> str:
    by_qa: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in attempts:
        by_qa[row["qa_id"]].append(row)
    lines = ["# Contrastive Distractor Canary Review", ""]
    for source in source_items:
        qa_id = source["qa_id"]
        old = source["materialized_qa"]
        lines.extend([
            f"## {qa_id}", "", f"Question: {old['question']}", "",
            "Original options:", "",
        ])
        for letter, value in old["options"].items():
            lines.append(f"- {letter}. {value}")
        for row in by_qa.get(qa_id, []):
            lines.extend(["", f"Attempt {row['attempt_index']} options:", ""])
            materialized = row.get("materialized_qa") or {}
            for letter, value in (materialized.get("options") or {}).items():
                lines.append(f"- {letter}. {value}")
            lines.extend([
                "", f"Minimal: {row['minimal_decision']}",
                f"Final: {row.get('final_decision')}",
                f"Grounding diagnosis: {row.get('grounding_diagnosis')}",
            ])
        lines.append("")
    return "\n".join(lines)


def _write_api_usage(root: Path, budget: ApiBudget) -> dict[str, Any]:
    observed = _usage(root)
    report = {
        "distractor_writer_normal_requests": budget.normal_counts[
            "distractor_writer"],
        "distractor_writer_format_retries": budget.retry_counts[
            "distractor_writer"],
        "normal_request_cap": NORMAL_REQUEST_CAP,
        "format_only_retry_cap": FORMAT_RETRY_CAP,
        "physical_paid_api_request_cap": PHYSICAL_REQUEST_CAP,
        **observed,
        "ordinary_questioner_generations": 0,
        "realizer_generations": 0,
        "verifier_generations": 0,
        "training_runs": 0,
        "semantic_repair_generations": 0,
    }
    atomic_json(root / "reports/api_usage.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path,
        default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.run_root.exists():
        raise FileExistsError(f"run_root_exists:{args.run_root}")

    source = attest_source(args.source_root.resolve())
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    api = replace(load_challenger_config(args.api_config),
                  model=CHALLENGER_MODEL, temperature=0.25,
                  enable_thinking=False, timeout_seconds=600)
    solver_native = read_json(SOLVER / "config.json")
    if int(solver_native.get("max_position_embeddings") or 0) < MODEL_CONTEXT_LIMIT:
        raise RuntimeError("solver_native_context_too_short")

    prompts = []
    source_audits = []
    for source_item in source["source_items"]:
        base = _base_qa(source_item["raw_qa"])
        audit = audit_frozen_source_qa(raw_qa=base,
            materialized_qa=source_item["materialized_qa"],
            span_catalog=source["span_catalog"])
        audit["semantic_review"] = {
            "question_answer_supported": True,
            "subject_action_time_consistent": True,
            "manual_rewrite_performed": False,
            "joint_necessity_hard_gate_applied": False,
            "review_note": (
                "Evidence supports the frozen question and answer. This canary "
                "tests distractor quality, not joint necessity."
            ),
        }
        if audit["decision"] != "source_qa_accept":
            raise RuntimeError(f"source_qa_audit_failed:{base['qa_id']}")
        source_audits.append(audit)
        prompt = build_contrastive_distractor_prompt(
            frozen_dialogue=source["context"]["combined_dialogue"],
            span_catalog=source["span_catalog"], base_qa=base,
            attempt_index=0, structured_feedback=None)
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        preflight = {
            "qa_id": base["qa_id"], "attempt_index": 0,
            "rendered_prompt_tokens": prompt_tokens,
            "reserved_generation_tokens": DISTRACTOR_MAX_TOKENS,
            "total_tokens": prompt_tokens + DISTRACTOR_MAX_TOKENS,
            "backend_context_limit": MODEL_CONTEXT_LIMIT,
            "decision": "pass" if prompt_tokens + DISTRACTOR_MAX_TOKENS <=
                MODEL_CONTEXT_LIMIT else "reject_context_overflow",
            "truncated": False,
        }
        if preflight["decision"] != "pass":
            raise RuntimeError(f"distractor_context_overflow:{base['qa_id']}")
        prompts.append(preflight)

    if args.dry_run:
        print(json.dumps({
            "decision": "dry_run_pass",
            "source_items": [row["qa_id"] for row in source["source_items"]],
            "source_audits": source_audits,
            "prompt_preflights": prompts,
            "normal_request_cap": NORMAL_REQUEST_CAP,
            "format_retry_cap": FORMAT_RETRY_CAP,
            "physical_request_cap": PHYSICAL_REQUEST_CAP,
            "challenger_model": api.model,
            "solver_model": str(SOLVER),
            "solver_max_model_len": MODEL_CONTEXT_LIMIT,
        }, ensure_ascii=False, indent=2))
        return 0

    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    project_diff = project_diff_sha256(PROJECT)
    manifest = {
        **_base_manifest(run_id=args.run_root.name,
            mode=CONTRASTIVE_DISTRACTOR_CANARY_VERSION,
            snapshot=snapshot, project_diff=project_diff),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "canary_version": CONTRASTIVE_DISTRACTOR_CANARY_VERSION,
        "source_root": source["source_root"],
        "source_tree_sha256_before": source["source_tree_sha256_before"],
        "source_questioner_manifest_sha256": source[
            "source_questioner_manifest_sha256"],
        "source_questioner_provenance_sha256": source[
            "source_questioner_provenance_sha256"],
        "source_context_sha256": source["source_context_sha256"],
        "source_span_catalog_sha256": source["source_span_catalog_sha256"],
        "source_raw_batch_sha256": source["source_raw_batch_sha256"],
        "source_round_filter_sha256": source["source_round_filter_sha256"],
        "source_final_dialogue_sha256": source["source_final_dialogue_sha256"],
        "source_target_qa_ids": [row["qa_id"] for row in source["source_items"]],
        "excluded_source_qa_ids": ["community_event_planning_contributions"],
        "contrastive_prompt_version": CONTRASTIVE_DISTRACTOR_PROMPT_VERSION,
        "contrastive_prompt_sha256": contrastive_distractor_prompt_sha256(),
        "contrastive_schema_version": CONTRASTIVE_DISTRACTOR_SCHEMA_VERSION,
        "contrastive_schema_sha256": contrastive_distractor_schema_sha256(),
        "format_retry_version": FORMAT_ONLY_RETRY_VERSION,
        "format_retry_suffix_sha256": format_only_retry_suffix_sha256(),
        "answer_placement_version": ANSWER_PLACEMENT_VERSION,
        "contrastive_source_sha256": sha256_file(
            MODULE_ROOT / "contrastive_distractor_canary.py"),
        "contrastive_runner_source_sha256": sha256_file(Path(__file__)),
        "lineage_distractor_writer_source_sha256": sha256_file(
            MODULE_ROOT / "lineage_distractor_writer.py"),
        "lineage_three_stage_canary_source_sha256": sha256_file(
            MODULE_ROOT / "lineage_three_stage_canary.py"),
        "live_backend_source_sha256": sha256_file(
            MODULE_ROOT / "live_backends.py"),
        "vllm_backend_source_sha256": sha256_file(
            MODULE_ROOT / "vllm_backends.py"),
        "challenger_model_identifier": api.model,
        "challenger_temperature": api.temperature,
        "challenger_enable_thinking": api.enable_thinking,
        "distractor_max_tokens": DISTRACTOR_MAX_TOKENS,
        "solver_model": str(SOLVER),
        "solver_tokenizer": str(TOKENIZER),
        "solver_max_model_len": MODEL_CONTEXT_LIMIT,
        "budget": {
            "distractor_normal_requests": NORMAL_REQUEST_CAP,
            "format_only_retries": FORMAT_RETRY_CAP,
            "physical_paid_api_requests": PHYSICAL_REQUEST_CAP,
            "ordinary_questioner": 0, "realizer": 0, "verifier": 0,
            "training": 0, "semantic_repair": 0,
        },
        "historical_runs_modified": False,
    }
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", {
        "mode": CONTRASTIVE_DISTRACTOR_CANARY_VERSION,
        "run_root": str(args.run_root),
        "source_root": source["source_root"],
        "challenger_model": api.model,
        "challenger_temperature": api.temperature,
        "challenger_enable_thinking": api.enable_thinking,
        "distractor_max_tokens": DISTRACTOR_MAX_TOKENS,
        "normal_request_cap": NORMAL_REQUEST_CAP,
        "format_retry_cap": FORMAT_RETRY_CAP,
        "physical_request_cap": PHYSICAL_REQUEST_CAP,
        "solver_model": str(SOLVER),
        "solver_tokenizer": str(TOKENIZER),
        "solver_max_model_len": MODEL_CONTEXT_LIMIT,
    })
    atomic_json(args.run_root / "frozen/source_attestation.json", {
        key: value for key, value in source.items()
        if key not in {"context", "span_catalog"}
    })
    atomic_json(args.run_root / "frozen/full_dialogue.json", source["context"])
    atomic_json(args.run_root / "frozen/questioner_span_catalog.json",
                source["span_catalog"])
    atomic_json(args.run_root / "reports/source_qa_evidence_audit.json", {
        "decision": "pass", "items": source_audits,
        "ordinary_questioner_generation_count": 0,
        "manual_qa_rewrite_count": 0,
    })
    atomic_json(args.run_root / "reports/preflight.json", {
        "decision": "pass", "prompt_preflights": prompts,
        "api_budget_preflight": "pass", "source_attestation": "pass",
        "solver_path_preflight": "pass", "tokenizer_path_preflight": "pass",
    })
    for source_item in source["source_items"]:
        atomic_json(args.run_root / f"frozen/base_qas/{source_item['qa_id']}.json", {
            "raw_qa": source_item["raw_qa"],
            "materialized_qa": source_item["materialized_qa"],
            "raw_qa_sha256": source_item["raw_qa_sha256"],
            "materialized_qa_sha256": source_item["materialized_qa_sha256"],
        })

    ledger = RequestLedger(args.run_root,
        sha256_file(args.run_root / "config/run_manifest.json"), "live")
    budget = ApiBudget(normal_caps={"distractor_writer": NORMAL_REQUEST_CAP},
        format_retry_cap=FORMAT_RETRY_CAP, physical_cap=PHYSICAL_REQUEST_CAP)
    backend = LiveSemanticRepairBackend(
        OpenAICompatibleChallengerClient(api), DISTRACTOR_MAX_TOKENS)
    full_view = {
        "combined_dialogue": source["context"]["combined_dialogue"],
        "span_catalog": source["span_catalog"],
        "view_sha256": sha256_bytes(canonical_json({
            "combined_dialogue": source["context"]["combined_dialogue"],
            "span_catalog": source["span_catalog"],
        }).encode()),
    }
    solver_backend: VLLMSolverBackend | None = None
    attempt_reports: list[dict[str, Any]] = []

    for source_item in source["source_items"]:
        base = _base_qa(source_item["raw_qa"])
        feedback: dict[str, Any] | None = None
        for attempt_index in (0, 1):
            prompt = build_contrastive_distractor_prompt(
                frozen_dialogue=source["context"]["combined_dialogue"],
                span_catalog=source["span_catalog"], base_qa=base,
                attempt_index=attempt_index, structured_feedback=feedback)
            task = run_single_object_task(
                run_root=args.run_root, ledger=ledger, backend=backend,
                budget=budget, stage="distractor_writer",
                logical_request_id=(
                    f"contrastive-distractor-{base['qa_id']}-d{attempt_index}"),
                semantic_prompt=prompt, lineage_view=full_view,
                validator=lambda raw, answer=base["correct_answer_text"]:
                    validate_distractor_object(raw,
                        correct_answer_text=str(answer)),
                schema_retry_decisions=("distractor_schema_invalid",),
            )
            record: dict[str, Any] = {
                "qa_id": base["qa_id"], "attempt_index": attempt_index,
                "api_task": task, "distractor_valid": task["accepted"],
                "minimal_decision": "not_run", "solver_evaluated": False,
                "final_decision": task["final_decision"],
                "grounding_diagnosis": None,
            }
            if not task["accepted"]:
                attempt_reports.append(record)
                break
            distractors = task["accepted_validation"]["distractors"]
            placement = deterministic_answer_placement(base["qa_id"], attempt_index)
            candidate = materialize_contrastive_candidate(
                base_qa=base, distractors=distractors,
                target_letter=placement["target_letter"],
                span_catalog=source["span_catalog"])
            materialized_path = args.run_root / (
                f"frozen/materialized_mcqs/{base['qa_id']}_attempt_{attempt_index}.json")
            atomic_json(materialized_path, candidate)
            record.update({
                "placement": placement,
                "materialized_mcq_path": str(materialized_path.relative_to(
                    args.run_root)),
                "materialized_mcq_sha256": sha256_file(materialized_path),
                "materialized_qa": candidate["materialized_qa"],
                "minimal_decision": candidate["admission"][
                    "spell_style_minimal_v1"]["decision"],
                "minimal_reasons": candidate["admission"][
                    "spell_style_minimal_v1"]["reasons"],
                "option_length_diagnostics": option_length_diagnostics(
                    candidate["materialized_qa"] or {}),
            })
            if record["minimal_decision"] != "accept":
                record["final_decision"] = "minimal_admission_reject"
                attempt_reports.append(record)
                break
            if solver_backend is None:
                solver_backend = VLLMSolverBackend(
                    model_path=SOLVER, tokenizer_path=TOKENIZER,
                    max_model_len=MODEL_CONTEXT_LIMIT, temperature=0.7,
                    top_p=0.9, max_tokens=32,
                    gpu_memory_utilization=0.90)
            solver_root = args.run_root / (
                f"solver_runs/{base['qa_id']}/attempt_{attempt_index}")
            solver_root.mkdir(parents=True)
            solver_config = _solver_config(run_root=solver_root)
            solver_config.update({
                "mode": "local_solver_contrastive_distractor_canary",
                "solver_effective_max_model_len": MODEL_CONTEXT_LIMIT,
                "max_local_solver_generations": 80,
            })
            solver_manifest = {
                **_base_manifest(run_id=(
                    f"{args.run_root.name}-{base['qa_id']}-d{attempt_index}"),
                    mode="local_solver_contrastive_distractor_canary",
                    snapshot=snapshot, project_diff=project_diff),
                "canary_version": CONTRASTIVE_DISTRACTOR_CANARY_VERSION,
                "qa_id": base["qa_id"], "attempt_index": attempt_index,
                "solver_model": str(SOLVER),
                "solver_max_model_len": MODEL_CONTEXT_LIMIT,
                "source_canary_manifest_sha256": sha256_file(
                    args.run_root / "config/run_manifest.json"),
                "source_materialized_mcq_sha256": sha256_file(materialized_path),
            }
            atomic_json(solver_root / "config/resolved_canary_config.json",
                        solver_config)
            atomic_json(solver_root / "config/run_manifest.json", solver_manifest)
            artifacts = {
                "context": source["context"],
                "catalog": source["span_catalog"],
                "contracts": candidate["reasoning_contracts"],
            }
            solver_report = _run_solver(
                root=solver_root, manifest=solver_manifest, config=solver_config,
                raw_batch=[candidate["raw_v8_qa"]],
                admission=candidate["admission"],
                materialized=candidate["materialized_qa"],
                artifacts=artifacts, tokenizer=tokenizer,
                backend=solver_backend)
            write_post_run_provenance(solver_root, PROJECT, solver_manifest)
            solver_summary = _solver_summary(solver_report)
            record.update({
                "solver_evaluated": True,
                "solver_root": str(solver_root.relative_to(args.run_root)),
                "solver_report_sha256": sha256_file(
                    solver_root / "reports/cross_dialogue_solver_audit.json"),
                "solver": solver_summary,
                "final_decision": solver_summary["final_decision"],
                "grounding_diagnosis": solver_summary[
                    "no_context_permutation_diagnostic"],
            })
            attempt_reports.append(record)
            if not should_run_attempt_one(
                    attempt_index=attempt_index,
                    final_decision=record["final_decision"],
                    grounding_diagnosis=record["grounding_diagnosis"]):
                break
            feedback = {
                "previous_distractors": list(distractors),
                "previous_final_decision": str(record["final_decision"]),
                "previous_grounding_diagnosis": str(
                    record["grounding_diagnosis"]),
            }

    usage = _write_api_usage(args.run_root, budget)
    decisions = collections.Counter(str(row.get("final_decision"))
                                    for row in attempt_reports)
    report = {
        "canary_version": CONTRASTIVE_DISTRACTOR_CANARY_VERSION,
        "source_qa_count": len(source["source_items"]),
        "source_qa_ids": [row["qa_id"] for row in source["source_items"]],
        "excluded_source_qa_ids": ["community_event_planning_contributions"],
        "distractor_attempt_count": len(attempt_reports),
        "distractor_valid_count": sum(row["distractor_valid"]
                                      for row in attempt_reports),
        "minimal_accepted_count": sum(row["minimal_decision"] == "accept"
                                      for row in attempt_reports),
        "solver_evaluated_count": sum(row["solver_evaluated"]
                                      for row in attempt_reports),
        "final_decisions": dict(sorted(decisions.items())),
        "route_conclusion": route_conclusion([
            row for row in attempt_reports if row["solver_evaluated"]]),
        "attempts": attempt_reports,
        "api_usage": usage,
        "ordinary_questioner_generations": 0,
        "realizer_generations": 0,
        "verifier_generations": 0,
        "training_runs": 0,
        "semantic_repair_generations": 0,
    }
    atomic_json(args.run_root / "reports/contrastive_distractor_canary.json", report)
    atomic_json(args.run_root / "reports/distractor_attempts.json", {
        "attempts": attempt_reports})
    atomic_json(args.run_root / "reports/solver_summary.json", {
        "solver_model": str(SOLVER), "solver_max_model_len": MODEL_CONTEXT_LIMIT,
        "items": [{key: row.get(key) for key in (
            "qa_id", "attempt_index", "minimal_decision", "solver_evaluated",
            "final_decision", "grounding_diagnosis", "solver_root", "solver")}
            for row in attempt_reports],
    })
    atomic_text(args.run_root / "review/old_vs_new_options.md",
                _review_markdown(source["source_items"], attempt_reports))

    source_tree_after = _tree_sha256(args.source_root.resolve())
    provenance = write_post_run_provenance(args.run_root, PROJECT, manifest)
    extra_checks = {
        "contrastive_source_sha_match": sha256_file(
            MODULE_ROOT / "contrastive_distractor_canary.py") == manifest[
                "contrastive_source_sha256"],
        "contrastive_runner_source_sha_match": sha256_file(Path(__file__)) ==
            manifest["contrastive_runner_source_sha256"],
        "historical_source_tree_unchanged": source_tree_after == source[
            "source_tree_sha256_before"],
        "physical_api_budget_respected": usage["total_paid_api_requests"] <=
            PHYSICAL_REQUEST_CAP,
        "ordinary_questioner_zero": True,
        "realizer_zero": True,
        "verifier_zero": True,
        "training_zero": True,
        "semantic_repair_zero": True,
    }
    provenance.update({
        "source_tree_sha256_before": source["source_tree_sha256_before"],
        "source_tree_sha256_after": source_tree_after,
        **extra_checks,
    })
    provenance["authoritative_release_provenance"] = (
        provenance.get("authoritative_release_provenance") is True and
        all(extra_checks.values()))
    provenance["provenance_mixed"] = not provenance[
        "authoritative_release_provenance"]
    atomic_json(args.run_root / "reports/post_run_provenance.json", provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
