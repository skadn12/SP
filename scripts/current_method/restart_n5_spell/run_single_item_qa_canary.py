from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer

from .contrastive_distractor_canary import build_contrastive_distractor_prompt
from .core import atomic_json, canonical_json, sha256_bytes, sha256_file
from .core_qa_admission import (
    CORE_QA_ADMISSION_VERSION,
    admit_core_qa,
    core_qa_schema_sha256,
    core_qa_validator_sha256,
)
from .input_adapter import load_lineage
from .ledger import RequestLedger
from .lineage_distractor_writer import (
    DISTRACTOR_WRITER_PROMPT_VERSION,
    distractor_prompt_template_sha256,
    distractor_schema_sha256,
    validate_distractor_object,
)
from .lineage_three_stage_canary import (
    FORMAT_ONLY_RETRY_VERSION,
    ApiBudget,
    artifact_sha256_manifest,
    format_only_retry_suffix_sha256,
    run_single_object_task,
)
from .live_backends import (
    LiveSemanticRepairBackend,
    OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from .questioner_contract import build_span_catalog, span_catalog_sha256
from .run_canary import (
    create_source_snapshot,
    git_value,
    project_diff_sha256,
    write_post_run_provenance,
)
from .run_cross_dialogue_calibration import (
    _base_manifest,
    _run_solver,
    _solver_config,
)
from .run_full_graph_qa_pilot import attest_graph_source
from .single_item_qa_canary import (
    SINGLE_ITEM_QA_CANARY_VERSION,
    SINGLE_ITEM_QA_MAX_COMPLETION_TOKENS,
    build_single_item_questioner_prompt,
    classify_candidate_disposition,
    deterministic_answer_placement,
    freeze_core_qa,
    frozen_core_qa_metadata,
    materialize_single_item_mcq,
    single_item_qa_protocol_metadata,
    validate_single_item_core_object,
)
from .vllm_backends import VLLMSolverBackend


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5")
DEFAULT_EXPANSION_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "spell_four_dialogue_graph_qa_retry_minimal_fix_v1/expansions"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "spell_single_item_qa_format_canary_v1"
)
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_LEGACY_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation"
)
DEFAULT_SOLVER = DEFAULT_LEGACY_ROOT / "accepted_models/solver_base_restart_stage2_merged"
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
LINEAGES = ("conv-26", "conv-41", "conv-44")
CHALLENGER_MODEL = "Qwen/Qwen3-32B"
MODEL_CONTEXT_LIMIT = 40960
DISTRACTOR_MAX_COMPLETION_TOKENS = 1024
FULL_PIPELINE_PAID_PHYSICAL_CAP = 12
FULL_PIPELINE_FORMAT_RETRY_CAP = 6
FORMAT_CANARY_PAID_PHYSICAL_CAP = 6
FORMAT_CANARY_FORMAT_RETRY_CAP = 3
PRE_IMPLEMENTATION_PROJECT_DIFF_SHA256 = (
    "9db7bd593fc68f3d6d91e1704a5408fb2623c99f424f0af6da71267740146fc0"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tree_sha256(root: Path) -> str:
    material = bytearray()
    for path in sorted(candidate for candidate in root.rglob("*")
                       if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        material.extend(relative.encode("utf-8") + b"\0" +
                        path.read_bytes() + b"\0")
    return sha256_bytes(bytes(material))


def _usage(ledger: RequestLedger) -> dict[str, int]:
    prompt = completion = total = 0
    for state in ledger.completed():
        usage = (state.get("response_metadata") or {}).get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        total += int(usage.get("total_tokens") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total}


def _lineage_view(*, lineage: str, dialogue: list[dict[str, Any]],
                  catalog: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": "single_item_full_dialogue_view_v1",
        "lineage": lineage,
        "combined_dialogue": [dict(turn) for turn in dialogue],
        "span_catalog": catalog,
    }
    return {**body, "view_sha256":
            sha256_bytes(canonical_json(body).encode("utf-8"))}


def _preflight_prompt(tokenizer: Any, prompt: str, max_tokens: int) -> dict[str, Any]:
    rendered = len(tokenizer.encode(prompt, add_special_tokens=False))
    total = rendered + int(max_tokens)
    return {
        "rendered_prompt_tokens": rendered,
        "max_completion_tokens": int(max_tokens),
        "total_tokens": total,
        "backend_context_limit": MODEL_CONTEXT_LIMIT,
        "decision": "pass" if total <= MODEL_CONTEXT_LIMIT else
            "reject_context_overflow",
        "truncated": False,
    }


def _source_bundle(*, input_root: Path, expansion_root: Path, lineage: str,
                   tokenizer: Any) -> dict[str, Any]:
    source_root = (expansion_root / lineage).resolve()
    tree_before = _tree_sha256(source_root)
    data = load_lineage(input_root, lineage, DEFAULT_LEGACY_ROOT)
    source = attest_graph_source(source_root, data, tokenizer)
    dialogue = [*source["parent_turns"], *source["child_turns"]]
    if not dialogue or len(source["sessions"]) != 10:
        raise RuntimeError(f"single_item_source_dialogue_invalid:{lineage}")
    catalog = build_span_catalog(dialogue)
    view = _lineage_view(lineage=lineage, dialogue=dialogue, catalog=catalog)
    prompt = build_single_item_questioner_prompt(
        span_catalog=catalog, lineage_id=lineage)
    preflight = _preflight_prompt(
        tokenizer, prompt, SINGLE_ITEM_QA_MAX_COMPLETION_TOKENS)
    if preflight["decision"] != "pass":
        raise RuntimeError(f"single_item_questioner_preflight_failed:{lineage}")
    public = {key: value for key, value in source.items()
              if key not in {"sessions", "parent_turns", "child_turns"}}
    public.update({
        "source_tree_sha256_before": tree_before,
        "parent_turn_count": len(source["parent_turns"]),
        "child_turn_count": len(source["child_turns"]),
        "combined_turn_count": len(dialogue),
        "combined_dialogue_sha256": sha256_bytes(
            canonical_json(dialogue).encode("utf-8")),
        "span_catalog_sha256": span_catalog_sha256(catalog),
        "lineage_view_sha256": view["view_sha256"],
    })
    context = {
        "schema_version": "single_item_questioner_context_v1",
        "lineage": lineage,
        "parent_turns": source["parent_turns"],
        "accepted_child_sessions": source["sessions"],
        "combined_dialogue": dialogue,
        "source_graph_root": str(source_root),
    }
    return {
        "lineage": lineage,
        "source_root": source_root,
        "source": public,
        "context": context,
        "catalog": catalog,
        "view": view,
        "prompt": prompt,
        "preflight": preflight,
    }


def _source_code_sha256() -> dict[str, str]:
    names = (
        "core.py", "core_qa_admission.py", "ledger.py", "live_backends.py",
        "lineage_three_stage_canary.py", "lineage_distractor_writer.py",
        "minimal_admission.py", "orchestrator.py", "questioner_contract.py",
        "run_canary.py", "run_cross_dialogue_calibration.py",
        "run_full_graph_qa_pilot.py", "single_item_qa_canary.py",
        "vllm_backends.py",
    )
    return {name: sha256_file(MODULE_ROOT / name) for name in names}


def budget_limits(execute_downstream: bool) -> dict[str, Any]:
    if execute_downstream:
        return {
            "normal_caps": {"single_item_questioner": 3,
                            "distractor_writer": 3},
            "format_retry_cap": FULL_PIPELINE_FORMAT_RETRY_CAP,
            "physical_cap": FULL_PIPELINE_PAID_PHYSICAL_CAP,
        }
    return {
        "normal_caps": {"single_item_questioner": 3,
                        "distractor_writer": 0},
        "format_retry_cap": FORMAT_CANARY_FORMAT_RETRY_CAP,
        "physical_cap": FORMAT_CANARY_PAID_PHYSICAL_CAP,
    }


def _api_usage(*, budget: ApiBudget, ledger: RequestLedger) -> dict[str, Any]:
    return {
        "single_item_questioner_normal_requests":
            budget.normal_counts.get("single_item_questioner", 0),
        "single_item_questioner_format_retries":
            budget.retry_counts.get("single_item_questioner", 0),
        "distractor_writer_normal_requests":
            budget.normal_counts.get("distractor_writer", 0),
        "distractor_writer_format_retries":
            budget.retry_counts.get("distractor_writer", 0),
        "total_paid_api_requests": ledger.summary()["paid_api_generation_count"],
        **_usage(ledger),
        "normal_request_cap": sum(budget.normal_caps.values()),
        "format_only_retry_cap": budget.format_retry_cap,
        "physical_paid_api_request_cap": budget.physical_cap,
        "ordinary_batch_questioner_generations": 0,
        "realizer_generations": 0,
        "verifier_generations": 0,
        "training_runs": 0,
        "semantic_repair_generations": 0,
    }


def summarize_records(records: list[Mapping[str, Any]], *,
                      execute_downstream: bool) -> dict[str, Any]:
    final_counts: dict[str, int] = {}
    for record in records:
        decision = str(record.get("final_decision") or "not_evaluated")
        final_counts[decision] = final_counts.get(decision, 0) + 1
    strict = sum(record.get("strict_json_valid") is True for record in records)
    schema = sum(record.get("questioner_schema_valid") is True for record in records)
    return {
        "lineages_attempted": len(records),
        "strict_json_valid": strict,
        "questioner_schema_valid": schema,
        "format_canary_target": "3/3 strict JSON and schema valid",
        "format_canary_target_met": strict == schema == len(LINEAGES) == 3,
        "core_qa_accepted": sum(record.get("core_qa_accepted") is True
                                for record in records),
        "distractor_writer_attempted": sum(
            record.get("distractor_writer_attempted") is True
            for record in records),
        "minimal_accepted": sum(record.get("minimal_accepted") is True
                                for record in records),
        "solver_evaluated": sum(record.get("solver_evaluated") is True
                                for record in records),
        "final_decisions": final_counts,
        "execute_downstream": execute_downstream,
        "no_dialogue_expansion": True,
    }


def _solver_manifest(*, root: Path, snapshot: dict[str, Any],
                     project_diff: str, lineage: str, qa: Mapping[str, Any],
                     bundle: Mapping[str, Any], materialized_sha: str
                     ) -> dict[str, Any]:
    base = _base_manifest(
        run_id=f"{root.parent.parent.name}-{qa['qa_id']}",
        mode="local_solver_single_item_qa_canary",
        snapshot=snapshot,
        project_diff=project_diff,
    )
    return {
        **base,
        "single_item_qa_canary_version": SINGLE_ITEM_QA_CANARY_VERSION,
        "lineage": lineage,
        "qa_id": qa["qa_id"],
        "materialized_qa_sha256": materialized_sha,
        "frozen_dialogue_sha256": sha256_bytes(canonical_json(
            bundle["context"]).encode("utf-8")),
        "span_catalog_sha256": span_catalog_sha256(bundle["catalog"]),
        "solver_max_model_len": 16384,
        "vllm_backend_source_sha256": sha256_file(
            MODULE_ROOT / "vllm_backends.py"),
        "solver_protocol": {
            "enable_thinking": False,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 32,
            "parser": "boxed_letter_v1",
            "max_parse_retries_per_target": 1,
        },
        "semantic_repair_enabled": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-item Questioner format/core canary")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--source-expansion-root", type=Path,
                        default=DEFAULT_EXPANSION_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--execute-downstream", action="store_true",
        help=("After the requested 3-item format/core canary, also call the "
              "Distractor Writer and local Solver chain."))
    args = parser.parse_args(argv)

    if args.run_root.exists():
        raise FileExistsError(f"refusing to overwrite canary root: {args.run_root}")
    if not DEFAULT_SOLVER.is_dir() or not DEFAULT_TOKENIZER.is_dir():
        raise RuntimeError("single_item_solver_or_tokenizer_missing")

    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_TOKENIZER, local_files_only=True)
    api = replace(
        load_challenger_config(args.api_config),
        model=CHALLENGER_MODEL,
        temperature=0.25,
        enable_thinking=False,
        timeout_seconds=600,
        questioner_max_tokens=SINGLE_ITEM_QA_MAX_COMPLETION_TOKENS,
    )

    # All source and context checks finish before the first paid request.
    bundles = [_source_bundle(
        input_root=args.input_root.resolve(),
        expansion_root=args.source_expansion_root.resolve(),
        lineage=lineage,
        tokenizer=tokenizer,
    ) for lineage in LINEAGES]

    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    project_diff = project_diff_sha256(PROJECT)
    source_code = _source_code_sha256()
    limits = budget_limits(args.execute_downstream)
    resolved = {
        "schema_version": "single_item_qa_canary_resolved_config_v1",
        "run_root": str(args.run_root.resolve()),
        "input_root": str(args.input_root.resolve()),
        "source_expansion_root": str(args.source_expansion_root.resolve()),
        "lineages": list(LINEAGES),
        "execute_downstream": args.execute_downstream,
        "questioner": {
            "model": api.model,
            "base_url": api.base_url,
            "temperature": api.temperature,
            "enable_thinking": api.enable_thinking,
            "timeout_seconds": api.timeout_seconds,
            "max_tokens": SINGLE_ITEM_QA_MAX_COMPLETION_TOKENS,
            "response_format": "json_object",
        },
        "distractor_writer": {
            "max_tokens": DISTRACTOR_MAX_COMPLETION_TOKENS,
            "semantic_attempt_cap_per_qa": 1,
            "format_retry_cap_per_logical_request": 1,
        },
        "solver": {
            "model": str(DEFAULT_SOLVER),
            "tokenizer": str(DEFAULT_TOKENIZER),
            "max_model_len": 16384,
            "enable_thinking": False,
        },
        "budgets": {
            "single_item_questioner_normal": 3,
            "distractor_writer_normal": limits["normal_caps"][
                "distractor_writer"],
            "global_format_only_retries": limits["format_retry_cap"],
            "physical_paid_api_requests": limits["physical_cap"],
        },
    }
    atomic_json(args.run_root / "config/resolved_canary_config.json", resolved)
    manifest = {
        "run_id": args.run_root.name,
        "mode": SINGLE_ITEM_QA_CANARY_VERSION,
        "created_at": _utc_now(),
        "project_git_commit": git_value(PROJECT, "rev-parse", "HEAD"),
        "pre_implementation_project_diff_sha256":
            PRE_IMPLEMENTATION_PROJECT_DIFF_SHA256,
        "project_diff_sha256": project_diff,
        "source_snapshot_manifest_path": snapshot[
            "source_snapshot_manifest_path"],
        "source_snapshot_manifest_sha256": snapshot[
            "source_snapshot_manifest_sha256"],
        "resolved_config_sha256": sha256_file(
            args.run_root / "config/resolved_canary_config.json"),
        "protocol": single_item_qa_protocol_metadata(),
        "core_qa_admission_version": CORE_QA_ADMISSION_VERSION,
        "core_qa_schema_sha256": core_qa_schema_sha256(),
        "core_qa_validator_sha256": core_qa_validator_sha256(),
        "distractor_writer_prompt_version": DISTRACTOR_WRITER_PROMPT_VERSION,
        "distractor_writer_prompt_sha256": distractor_prompt_template_sha256(),
        "distractor_writer_schema_sha256": distractor_schema_sha256(),
        "format_only_retry_version": FORMAT_ONLY_RETRY_VERSION,
        "format_only_retry_suffix_sha256": format_only_retry_suffix_sha256(),
        "source_code_sha256": source_code,
        "source_attestations": [bundle["source"] for bundle in bundles],
        "no_dialogue_expansion": True,
        "ordinary_batch_questioner_generations": 0,
        "realizer_generations": 0,
        "verifier_generations": 0,
        "training_runs": 0,
        "semantic_repair_generations": 0,
    }
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    manifest_sha = sha256_file(args.run_root / "config/run_manifest.json")

    for bundle in bundles:
        lineage = bundle["lineage"]
        atomic_json(args.run_root / f"frozen/source_attestations/{lineage}.json",
                    bundle["source"])
        atomic_json(args.run_root / f"frozen/contexts/{lineage}.json",
                    bundle["context"])
        atomic_json(args.run_root / f"frozen/span_catalogs/{lineage}.json",
                    bundle["catalog"])
        atomic_json(args.run_root / f"frozen/lineage_views/{lineage}.json",
                    bundle["view"])
        atomic_json(args.run_root / f"reports/preflight/{lineage}.json",
                    bundle["preflight"])

    ledger = RequestLedger(args.run_root, manifest_sha, "live")
    budget = ApiBudget(
        normal_caps=limits["normal_caps"],
        format_retry_cap=limits["format_retry_cap"],
        physical_cap=limits["physical_cap"],
    )
    client = OpenAICompatibleChallengerClient(api)
    questioner_backend = LiveSemanticRepairBackend(
        client, SINGLE_ITEM_QA_MAX_COMPLETION_TOKENS)
    distractor_backend = LiveSemanticRepairBackend(
        client, DISTRACTOR_MAX_COMPLETION_TOKENS)
    solver_backend: VLLMSolverBackend | None = None
    records: list[dict[str, Any]] = []

    for bundle in bundles:
        lineage = bundle["lineage"]
        result = run_single_object_task(
            run_root=args.run_root,
            ledger=ledger,
            backend=questioner_backend,
            budget=budget,
            stage="single_item_questioner",
            logical_request_id=f"single-item-questioner-{lineage}-q001",
            semantic_prompt=bundle["prompt"],
            lineage_view=bundle["view"],
            validator=lambda raw, catalog=bundle["catalog"]:
                validate_single_item_core_object(raw, span_catalog=catalog),
            schema_retry_decisions=("single_item_questioner_schema_invalid",),
            ledger_budget_limit=limits["physical_cap"],
            ledger_total_budget_limit=limits["physical_cap"],
            json_invalid_decision="single_item_questioner_json_invalid",
        )
        final_attempt = result["attempts"][-1]
        strict_json_valid = final_attempt["parse_status"] == "json_parsed"
        record: dict[str, Any] = {
            "lineage": lineage,
            "sample_index": 0,
            "source_root": str(bundle["source_root"]),
            "questioner": result,
            "strict_json_valid": strict_json_valid,
            "questioner_schema_valid": result["accepted"],
            "core_qa_accepted": False,
            "distractor_writer_attempted": False,
            "minimal_accepted": False,
            "solver_evaluated": False,
            "final_decision": "questioner_format_or_schema_reject",
        }
        if not result["accepted"]:
            records.append(record)
            continue

        model_core = dict(result["accepted_validation"]["core_qa"])
        atomic_json(args.run_root /
                    f"frozen/questioner_objects/{lineage}_q001.json", model_core)
        frozen_core = freeze_core_qa(
            model_core, lineage_id=lineage, sample_index=0,
            span_catalog=bundle["catalog"])
        core_admission = admit_core_qa(
            model_core,
            span_catalog=bundle["catalog"],
            qa_id=frozen_core["qa_id"],
            known_speakers=sorted({turn["speaker"] for turn in
                                   bundle["context"]["combined_dialogue"]}),
        )
        core_path = args.run_root / f"reports/core_qa/{lineage}_q001.json"
        atomic_json(core_path, core_admission)
        record.update({
            "qa_id": frozen_core["qa_id"],
            "core_qa_admission": core_admission,
            "core_qa_accepted": core_admission["decision"] == "core_qa_accept",
            "final_decision": core_admission["decision"],
        })
        if core_admission["decision"] != "core_qa_accept":
            records.append(record)
            continue

        atomic_json(args.run_root / f"frozen/core_qas/{frozen_core['qa_id']}.json",
                    frozen_core)
        atomic_json(args.run_root /
                    f"frozen/core_qa_metadata/{frozen_core['qa_id']}.json",
                    frozen_core_qa_metadata(
                        frozen_core_qa=frozen_core,
                        lineage_id=lineage,
                        sample_index=0,
                        span_catalog=bundle["catalog"],
                    ))
        if not args.execute_downstream:
            record.update({
                "final_decision": "core_qa_accept_format_canary_stop",
                "downstream_not_run_reason":
                    "requested_single_item_format_canary_stop",
            })
            records.append(record)
            continue

        distractor_prompt = build_contrastive_distractor_prompt(
            frozen_dialogue=bundle["context"]["combined_dialogue"],
            span_catalog=bundle["catalog"],
            base_qa=frozen_core,
            attempt_index=0,
            structured_feedback=None,
        )
        distractor_preflight = _preflight_prompt(
            tokenizer, distractor_prompt, DISTRACTOR_MAX_COMPLETION_TOKENS)
        atomic_json(args.run_root /
                    f"reports/preflight/{lineage}_distractor_q001.json",
                    distractor_preflight)
        if distractor_preflight["decision"] != "pass":
            record["final_decision"] = "distractor_context_overflow"
            records.append(record)
            continue
        record["distractor_writer_attempted"] = True
        distractor_result = run_single_object_task(
            run_root=args.run_root,
            ledger=ledger,
            backend=distractor_backend,
            budget=budget,
            stage="distractor_writer",
            logical_request_id=f"distractor-writer-{lineage}-q001",
            semantic_prompt=distractor_prompt,
            lineage_view=bundle["view"],
            validator=lambda raw, answer=frozen_core["correct_answer_text"]:
                validate_distractor_object(raw, correct_answer_text=answer),
            schema_retry_decisions=("distractor_schema_invalid",),
            ledger_budget_limit=limits["physical_cap"],
            ledger_total_budget_limit=limits["physical_cap"],
            json_invalid_decision="distractor_json_invalid",
        )
        record["distractor_writer"] = distractor_result
        if not distractor_result["accepted"]:
            record["final_decision"] = "distractor_format_or_schema_reject"
            records.append(record)
            continue

        distractors = distractor_result["accepted_validation"]["distractors"]
        placement = deterministic_answer_placement(lineage, 0)
        materialized = materialize_single_item_mcq(
            core_qa=frozen_core,
            distractors=distractors,
            placement=placement,
            span_catalog=bundle["catalog"],
        )
        materialized_path = (
            args.run_root /
            f"frozen/materialized_mcqs/{frozen_core['qa_id']}.json"
        )
        atomic_json(materialized_path, materialized)
        minimal = materialized["admission"]["spell_style_minimal_v1"]
        record.update({
            "placement": placement,
            "materialized_mcq_path": str(materialized_path.relative_to(
                args.run_root)),
            "materialized_mcq_sha256": sha256_file(materialized_path),
            "minimal_admission": minimal,
            "minimal_accepted": minimal["decision"] == "accept",
            "final_decision": "minimal_admission_reject",
        })
        if minimal["decision"] != "accept":
            records.append(record)
            continue

        if solver_backend is None:
            solver_backend = VLLMSolverBackend(
                model_path=DEFAULT_SOLVER,
                tokenizer_path=DEFAULT_TOKENIZER,
                max_model_len=16384,
                temperature=0.7,
                top_p=0.9,
                max_tokens=32,
                gpu_memory_utilization=0.80,
            )
        solver_root = args.run_root / f"solver_runs/{lineage}_q001"
        solver_root.mkdir(parents=True)
        solver_config = _solver_config(run_root=solver_root)
        solver_manifest = _solver_manifest(
            root=solver_root,
            snapshot=snapshot,
            project_diff=project_diff,
            lineage=lineage,
            qa=materialized["materialized_qa"],
            bundle=bundle,
            materialized_sha=sha256_bytes(canonical_json(
                materialized["materialized_qa"]).encode("utf-8")),
        )
        atomic_json(solver_root / "config/resolved_canary_config.json",
                    solver_config)
        atomic_json(solver_root / "config/run_manifest.json", solver_manifest)
        solver_report = _run_solver(
            root=solver_root,
            manifest=solver_manifest,
            config=solver_config,
            raw_batch=[materialized["raw_v8_qa"]],
            admission=materialized["admission"],
            materialized=materialized["materialized_qa"],
            artifacts={"context": bundle["context"],
                       "catalog": bundle["catalog"],
                       "contracts": materialized["reasoning_contracts"]},
            tokenizer=tokenizer,
            backend=solver_backend,
        )
        write_post_run_provenance(
            solver_root, PROJECT, solver_manifest)
        disposition = classify_candidate_disposition(
            solver_report["final_decision"])
        if disposition["resample_required"]:
            disposition["resample_deferred_by_format_canary"] = True
            disposition["resample_deferred_reason"] = (
                "this canary is capped at one fresh QA per lineage")
        record.update({
            "solver_evaluated": True,
            "solver_root": str(solver_root),
            "solver_report_sha256": sha256_file(
                solver_root / "reports/cross_dialogue_solver_audit.json"),
            "final_decision": solver_report["final_decision"],
            "no_context_permutation_diagnostic": solver_report[
                "no_context_permutation_diagnostic"],
            "candidate_disposition": disposition,
        })
        records.append(record)

    api_usage = _api_usage(budget=budget, ledger=ledger)
    summary = summarize_records(
        records, execute_downstream=args.execute_downstream)
    report = {
        "canary_version": SINGLE_ITEM_QA_CANARY_VERSION,
        "summary": summary,
        "items": records,
        "api_usage": api_usage,
        "format_only_retry_version": FORMAT_ONLY_RETRY_VERSION,
        "format_only_retry_suffix_sha256": format_only_retry_suffix_sha256(),
        "ordinary_batch_questioner_generations": 0,
        "realizer_generations": 0,
        "verifier_generations": 0,
        "training_runs": 0,
        "semantic_repair_generations": 0,
        "historical_runs_modified": False,
        "dialogue_expansion_generations": 0,
    }
    atomic_json(args.run_root / "reports/single_item_qa_canary.json", report)
    atomic_json(args.run_root / "reports/core_qa_admission_summary.json", {
        "admission_version": CORE_QA_ADMISSION_VERSION,
        "items": [{"lineage": row["lineage"],
                   "qa_id": row.get("qa_id"),
                   "decision": (row.get("core_qa_admission") or {}).get(
                       "decision", "not_run")}
                  for row in records],
    })
    atomic_json(args.run_root / "reports/distractor_summary.json", {
        "execute_downstream": args.execute_downstream,
        "attempted": summary["distractor_writer_attempted"],
        "same_qa_semantic_retry_count": 0,
    })
    atomic_json(args.run_root / "reports/solver_summary.json", {
        "execute_downstream": args.execute_downstream,
        "evaluated": summary["solver_evaluated"],
        "final_decisions": summary["final_decisions"],
    })
    atomic_json(args.run_root / "reports/api_usage.json", api_usage)

    source_after = {bundle["lineage"]: _tree_sha256(bundle["source_root"])
                    for bundle in bundles}
    source_before = {bundle["lineage"]:
                     bundle["source"]["source_tree_sha256_before"]
                     for bundle in bundles}
    checks = {
        "git_head_unchanged": git_value(PROJECT, "rev-parse", "HEAD") ==
            manifest["project_git_commit"],
        "project_diff_unchanged": project_diff_sha256(PROJECT) ==
            manifest["project_diff_sha256"],
        "source_snapshot_manifest_match": sha256_file(Path(
            manifest["source_snapshot_manifest_path"])) ==
            manifest["source_snapshot_manifest_sha256"],
        "source_code_sha_match": _source_code_sha256() == source_code,
        "historical_source_trees_unchanged": source_after == source_before,
        "no_dialogue_expansion": True,
        "ordinary_batch_questioner_zero": True,
        "realizer_zero": True,
        "verifier_zero": True,
        "training_zero": True,
        "semantic_repair_zero": True,
    }
    provenance = {
        "checks": checks,
        "source_tree_sha256_before": source_before,
        "source_tree_sha256_after": source_after,
        "project_git_commit": manifest["project_git_commit"],
        "project_diff_sha256": manifest["project_diff_sha256"],
        "source_snapshot_manifest_sha256": manifest[
            "source_snapshot_manifest_sha256"],
        "authoritative_release_provenance": all(checks.values()),
        "provenance_mixed": not all(checks.values()),
    }
    atomic_json(args.run_root / "reports/artifact_sha256.json",
                artifact_sha256_manifest(args.run_root))
    atomic_json(args.run_root / "reports/post_run_provenance.json", provenance)
    if provenance["provenance_mixed"]:
        raise RuntimeError("single_item_qa_canary_provenance_mixed")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
