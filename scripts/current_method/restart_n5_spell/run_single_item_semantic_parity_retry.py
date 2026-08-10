from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .balanced_factorial_distractor_canary import (
    BALANCED_FACTORIAL_PLACEMENT_VERSION,
    deterministic_balanced_factorial_placement,
)
from .contrastive_distractor_canary import (
    _evidence_rows,
    audit_frozen_source_qa,
    materialize_contrastive_candidate,
    option_length_diagnostics,
)
from .core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from .input_adapter import read_json
from .ledger import RequestLedger
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
from .run_canary import create_source_snapshot, project_diff_sha256, write_post_run_provenance
from .run_contrastive_distractor_canary import _base_qa, _solver_summary, _tree_sha256
from .run_cross_dialogue_calibration import _base_manifest, _run_solver, _solver_config
from .run_single_item_qa_canary import (
    CHALLENGER_MODEL,
    DEFAULT_SOLVER,
    DEFAULT_TOKENIZER,
    MODEL_CONTEXT_LIMIT,
    _solver_manifest,
)
from .semantic_parity_distractor_retry import (
    SEMANTIC_PARITY_PROMPT_VERSION,
    SEMANTIC_PARITY_RETRY_VERSION,
    SEMANTIC_PARITY_SCHEMA_VERSION,
    build_semantic_parity_prompt,
    semantic_parity_prompt_sha256,
    semantic_parity_schema_sha256,
    validate_semantic_parity_bundle,
)
from .vllm_backends import VLLMSolverBackend


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "spell_single_item_qa_full_canary_v1"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "spell_single_item_semantic_parity_retry_v1"
)
TARGET_LINEAGES = ("conv-41", "conv-44")
MAX_TOKENS = 1024
NORMAL_CAP = 2
FORMAT_RETRY_CAP = 2
PHYSICAL_CAP = 4


def retry_eligible(final_decision: str) -> bool:
    return final_decision == "bad_ground"


def _usage(ledger: RequestLedger) -> dict[str, int]:
    prompt = completion = total = 0
    for state in ledger.completed():
        usage = (state.get("response_metadata") or {}).get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        total += int(usage.get("total_tokens") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total}


def _attest_source(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "config/run_manifest.json"
    provenance_path = root / "reports/post_run_provenance.json"
    report_path = root / "reports/single_item_qa_canary.json"
    for path in (manifest_path, provenance_path, report_path):
        if not path.is_file():
            raise RuntimeError(f"semantic_parity_source_missing:{path}")
    provenance = read_json(provenance_path)
    if (provenance.get("authoritative_release_provenance") is not True or
            provenance.get("provenance_mixed") is not False):
        raise RuntimeError("semantic_parity_source_provenance_invalid")
    report = read_json(report_path)
    by_lineage = {row["lineage"]: row for row in report.get("items") or []}
    items = []
    for lineage in TARGET_LINEAGES:
        source_row = by_lineage.get(lineage)
        if not isinstance(source_row, dict) or not retry_eligible(
                str(source_row.get("final_decision"))):
            raise RuntimeError(f"semantic_parity_source_not_bad_ground:{lineage}")
        context_path = root / f"frozen/contexts/{lineage}.json"
        catalog_path = root / f"frozen/span_catalogs/{lineage}.json"
        mcq_path = root / (
            f"frozen/materialized_mcqs/{lineage.replace('-', '_')}_"
            "single_item_001.json")
        solver_report_path = root / (
            f"solver_runs/{lineage}_q001/reports/"
            "cross_dialogue_solver_audit.json")
        for path in (context_path, catalog_path, mcq_path, solver_report_path):
            if not path.is_file():
                raise RuntimeError(f"semantic_parity_source_item_missing:{path}")
        wrapper = read_json(mcq_path)
        solver_report = read_json(solver_report_path)
        if solver_report.get("final_decision") != "bad_ground":
            raise RuntimeError(f"semantic_parity_solver_source_drift:{lineage}")
        raw_qa = wrapper["raw_v8_qa"]
        base = _base_qa(raw_qa)
        catalog = read_json(catalog_path)
        source_audit = audit_frozen_source_qa(
            raw_qa=base,
            materialized_qa=wrapper["materialized_qa"],
            span_catalog=catalog)
        if source_audit["decision"] != "source_qa_accept":
            raise RuntimeError(f"semantic_parity_source_qa_invalid:{lineage}")
        items.append({
            "lineage": lineage,
            "qa_id": base["qa_id"],
            "base_qa": base,
            "previous_distractors": list(raw_qa["distractors"]),
            "previous_final_decision": solver_report["final_decision"],
            "previous_grounding_diagnosis": (
                solver_report.get("no_context_permutation_diagnostic") or
                "bad_ground_permutation_inconclusive"),
            "previous_solver_summary": _solver_summary(solver_report),
            "context": read_json(context_path),
            "catalog": catalog,
            "evidence": _evidence_rows(base_qa=base, span_catalog=catalog),
            "source_audit": source_audit,
            "source_mcq_path": str(mcq_path),
            "source_mcq_sha256": sha256_file(mcq_path),
            "source_solver_report_path": str(solver_report_path),
            "source_solver_report_sha256": sha256_file(solver_report_path),
            "source_context_sha256": sha256_file(context_path),
            "source_catalog_sha256": sha256_file(catalog_path),
        })
    return {
        "source_root": str(root),
        "source_tree_sha256_before": _tree_sha256(root),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_provenance_sha256": sha256_file(provenance_path),
        "source_report_sha256": sha256_file(report_path),
        "items": items,
    }


def _minimal_request_view(item: dict[str, Any]) -> dict[str, Any]:
    turns = [{
        "turn_id": row["turn_id"], "session_id": row["session_id"],
        "speaker": row["speaker"], "text": row["exact_text"],
    } for row in item["evidence"]]
    body = {"combined_dialogue": turns, "span_catalog": item["catalog"]}
    return {**body, "view_sha256": sha256_bytes(
        canonical_json(body).encode())}


def _review_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Single-item semantic-parity distractor retry", ""]
    for row in rows:
        lines.extend([
            f"## {row['lineage']} / {row['qa_id']}", "",
            f"Question: {row['question']}", "",
            f"Correct answer: {row['correct_answer_text']}", "",
            "Previous options:", "",
        ])
        lines.extend(f"- {value}" for value in row["previous_distractors"])
        validation = (row.get("api_task") or {}).get(
            "accepted_validation") or {}
        lines.extend(["", f"Frame: {validation.get('frame')}", "", "Slots:", ""])
        for slot_id, values in (validation.get("slots") or {}).items():
            lines.append(
                f"- {slot_id}: `{values['gold_value']}` / "
                f"`{values['contrast_value']}`")
        lines.extend(["", "New options:", ""])
        for letter, value in (row.get("options") or {}).items():
            lines.append(f"- {letter}. {value}")
        lines.extend([
            "", f"Minimal: {row.get('minimal_decision')}",
            f"Solver final: {row.get('final_decision')}",
            f"Grounding: {row.get('grounding_decision')}", "",
        ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One semantic-parity 2x2 distractor retry for bad-ground QAs")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path,
        default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.run_root.exists():
        raise FileExistsError(f"semantic_parity_run_root_exists:{args.run_root}")

    source = _attest_source(args.source_root)
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_TOKENIZER, local_files_only=True)
    api = replace(load_challenger_config(args.api_config),
                  model=CHALLENGER_MODEL, temperature=0.25,
                  enable_thinking=False, timeout_seconds=600,
                  questioner_max_tokens=MAX_TOKENS)
    prepared = []
    for item in source["items"]:
        prompt = build_semantic_parity_prompt(
            base_qa=item["base_qa"], evidence_rows=item["evidence"],
            previous_distractors=item["previous_distractors"],
            previous_grounding_diagnosis=item[
                "previous_grounding_diagnosis"])
        tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        preflight = {
            "lineage": item["lineage"], "qa_id": item["qa_id"],
            "rendered_prompt_tokens": tokens,
            "reserved_generation_tokens": MAX_TOKENS,
            "total_tokens": tokens + MAX_TOKENS,
            "backend_context_limit": MODEL_CONTEXT_LIMIT,
            "decision": "pass" if tokens + MAX_TOKENS <=
                MODEL_CONTEXT_LIMIT else "reject_context_overflow",
            "truncated": False,
        }
        if preflight["decision"] != "pass":
            raise RuntimeError(f"semantic_parity_preflight_failed:{item['lineage']}")
        prepared.append({**item, "prompt": prompt, "preflight": preflight})
    if args.dry_run:
        print(json.dumps({
            "decision": "dry_run_pass",
            "source_root": source["source_root"],
            "target_lineages": list(TARGET_LINEAGES),
            "preflights": [row["preflight"] for row in prepared],
            "normal_request_cap": NORMAL_CAP,
            "format_retry_cap": FORMAT_RETRY_CAP,
            "physical_request_cap": PHYSICAL_CAP,
        }, ensure_ascii=False, indent=2))
        return 0

    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    diff_sha = project_diff_sha256(PROJECT)
    manifest = {
        **_base_manifest(run_id=args.run_root.name,
            mode=SEMANTIC_PARITY_RETRY_VERSION,
            snapshot=snapshot, project_diff=diff_sha),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "retry_version": SEMANTIC_PARITY_RETRY_VERSION,
        "source_root": source["source_root"],
        "source_tree_sha256_before": source["source_tree_sha256_before"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "source_provenance_sha256": source["source_provenance_sha256"],
        "source_report_sha256": source["source_report_sha256"],
        "target_lineages": list(TARGET_LINEAGES),
        "prompt_version": SEMANTIC_PARITY_PROMPT_VERSION,
        "prompt_sha256": semantic_parity_prompt_sha256(),
        "schema_version": SEMANTIC_PARITY_SCHEMA_VERSION,
        "schema_sha256": semantic_parity_schema_sha256(),
        "placement_version": BALANCED_FACTORIAL_PLACEMENT_VERSION,
        "format_retry_version": FORMAT_ONLY_RETRY_VERSION,
        "format_retry_suffix_sha256": format_only_retry_suffix_sha256(),
        "semantic_parity_source_sha256": sha256_file(
            MODULE_ROOT / "semantic_parity_distractor_retry.py"),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "challenger_model_identifier": api.model,
        "budget": {"normal_requests": NORMAL_CAP,
                   "format_only_retries": FORMAT_RETRY_CAP,
                   "physical_paid_api_requests": PHYSICAL_CAP,
                   "ordinary_questioner": 0, "realizer": 0,
                   "verifier": 0, "training": 0},
        "question_answer_evidence_frozen": True,
        "same_qa_distractor_semantic_retry_count": 1,
        "historical_runs_modified": False,
    }
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", {
        "mode": SEMANTIC_PARITY_RETRY_VERSION,
        "source_root": source["source_root"],
        "model": api.model, "temperature": api.temperature,
        "max_tokens": MAX_TOKENS, "target_lineages": list(TARGET_LINEAGES),
        "normal_request_cap": NORMAL_CAP,
        "format_retry_cap": FORMAT_RETRY_CAP,
        "physical_request_cap": PHYSICAL_CAP,
        "solver_model": str(DEFAULT_SOLVER),
        "solver_tokenizer": str(DEFAULT_TOKENIZER),
        "solver_max_model_len": 16384,
    })
    for item in prepared:
        lineage = item["lineage"]
        atomic_json(args.run_root / f"frozen/source/{lineage}.json", {
            key: value for key, value in item.items()
            if key not in {"context", "catalog", "prompt"}
        })
        atomic_json(args.run_root / f"frozen/contexts/{lineage}.json",
                    item["context"])
        atomic_json(args.run_root / f"frozen/span_catalogs/{lineage}.json",
                    item["catalog"])
        atomic_json(args.run_root / f"reports/preflight/{lineage}.json",
                    item["preflight"])

    ledger = RequestLedger(args.run_root,
        sha256_file(args.run_root / "config/run_manifest.json"), "live")
    budget = ApiBudget(
        normal_caps={"distractor_writer": NORMAL_CAP},
        format_retry_cap=FORMAT_RETRY_CAP, physical_cap=PHYSICAL_CAP)
    backend = LiveSemanticRepairBackend(
        OpenAICompatibleChallengerClient(api), MAX_TOKENS)
    solver_backend: VLLMSolverBackend | None = None
    rows = []
    for item in prepared:
        base = item["base_qa"]
        task = run_single_object_task(
            run_root=args.run_root, ledger=ledger, backend=backend,
            budget=budget, stage="distractor_writer",
            logical_request_id=f"semantic-parity-{item['lineage']}-q001",
            semantic_prompt=item["prompt"],
            lineage_view=_minimal_request_view(item),
            validator=lambda raw, answer=base["correct_answer_text"]:
                validate_semantic_parity_bundle(
                    raw, correct_answer_text=answer),
            schema_retry_decisions=("semantic_parity_schema_invalid",),
            ledger_budget_limit=PHYSICAL_CAP,
            ledger_total_budget_limit=PHYSICAL_CAP,
            json_invalid_decision="semantic_parity_json_invalid")
        row: dict[str, Any] = {
            "lineage": item["lineage"], "qa_id": item["qa_id"],
            "question": base["question"],
            "correct_answer_text": base["correct_answer_text"],
            "previous_distractors": item["previous_distractors"],
            "previous_solver_summary": item["previous_solver_summary"],
            "api_task": task, "minimal_decision": "not_run",
            "solver_evaluated": False,
            "final_decision": task["final_decision"],
        }
        if not task["accepted"]:
            rows.append(row)
            continue
        validation = task["accepted_validation"]
        placement = deterministic_balanced_factorial_placement(base["qa_id"])
        candidate = materialize_contrastive_candidate(
            base_qa=base, distractors=validation["distractors"],
            target_letter=placement["target_letter"],
            span_catalog=item["catalog"])
        candidate_path = (
            args.run_root /
            f"frozen/materialized_mcqs/{base['qa_id']}.json"
        )
        atomic_json(candidate_path, candidate)
        minimal = candidate["admission"]["spell_style_minimal_v1"]
        row.update({
            "placement": placement,
            "options": (candidate["materialized_qa"] or {}).get("options"),
            "materialized_mcq_path": str(candidate_path.relative_to(args.run_root)),
            "materialized_mcq_sha256": sha256_file(candidate_path),
            "minimal_decision": minimal["decision"],
            "minimal_reasons": minimal["reasons"],
            "option_length_diagnostics": option_length_diagnostics(
                candidate["materialized_qa"] or {}),
            "final_decision": "minimal_admission_reject",
        })
        if minimal["decision"] != "accept":
            rows.append(row)
            continue
        if solver_backend is None:
            solver_backend = VLLMSolverBackend(
                model_path=DEFAULT_SOLVER, tokenizer_path=DEFAULT_TOKENIZER,
                max_model_len=16384, temperature=0.7, top_p=0.9,
                max_tokens=32, gpu_memory_utilization=0.80)
        solver_root = args.run_root / f"solver_runs/{item['lineage']}_q001"
        solver_root.mkdir(parents=True)
        solver_config = _solver_config(run_root=solver_root)
        solver_manifest = _solver_manifest(
            root=solver_root, snapshot=snapshot, project_diff=diff_sha,
            lineage=item["lineage"], qa=candidate["materialized_qa"],
            bundle=item, materialized_sha=sha256_bytes(canonical_json(
                candidate["materialized_qa"]).encode()))
        solver_manifest.update({
            "semantic_parity_retry_version": SEMANTIC_PARITY_RETRY_VERSION,
            "source_bad_ground_solver_report_sha256":
                item["source_solver_report_sha256"],
        })
        atomic_json(solver_root / "config/resolved_canary_config.json",
                    solver_config)
        atomic_json(solver_root / "config/run_manifest.json", solver_manifest)
        solver_report = _run_solver(
            root=solver_root, manifest=solver_manifest, config=solver_config,
            raw_batch=[candidate["raw_v8_qa"]],
            admission=candidate["admission"],
            materialized=candidate["materialized_qa"],
            artifacts={"context": item["context"],
                       "catalog": item["catalog"],
                       "contracts": candidate["reasoning_contracts"]},
            tokenizer=tokenizer, backend=solver_backend)
        write_post_run_provenance(solver_root, PROJECT, solver_manifest)
        summary = _solver_summary(solver_report)
        row.update({
            "solver_evaluated": True,
            "solver_root": str(solver_root),
            "solver_report_sha256": sha256_file(
                solver_root / "reports/cross_dialogue_solver_audit.json"),
            "new_solver_summary": summary,
            "grounding_decision": summary["grounding_decision"],
            "final_decision": summary["final_decision"],
            "grounding_improved": summary["grounding_decision"] != "bad_ground",
        })
        rows.append(row)

    usage = {
        "distractor_writer_normal_requests":
            budget.normal_counts.get("distractor_writer", 0),
        "distractor_writer_format_retries":
            budget.retry_counts.get("distractor_writer", 0),
        "total_paid_api_requests": ledger.summary()["paid_api_generation_count"],
        **_usage(ledger), "normal_request_cap": NORMAL_CAP,
        "format_only_retry_cap": FORMAT_RETRY_CAP,
        "physical_paid_api_request_cap": PHYSICAL_CAP,
        "ordinary_questioner_generations": 0,
        "realizer_generations": 0, "verifier_generations": 0,
        "training_runs": 0, "semantic_repair_generations": 0,
    }
    report = {
        "retry_version": SEMANTIC_PARITY_RETRY_VERSION,
        "source_bad_ground_count": len(rows),
        "semantic_parity_valid": sum(row["api_task"]["accepted"] for row in rows),
        "minimal_accepted": sum(row["minimal_decision"] == "accept" for row in rows),
        "solver_evaluated": sum(row["solver_evaluated"] for row in rows),
        "grounding_improved": sum(row.get("grounding_improved") is True for row in rows),
        "still_bad_ground": sum(row.get("grounding_decision") == "bad_ground" for row in rows),
        "items": rows, "api_usage": usage,
        "question_answer_evidence_modified": False,
        "same_qa_distractor_semantic_retry_count": 1,
        "ordinary_questioner_generations": 0,
        "realizer_generations": 0, "verifier_generations": 0,
        "training_runs": 0,
    }
    atomic_json(args.run_root / "reports/semantic_parity_retry.json", report)
    atomic_json(args.run_root / "reports/api_usage.json", usage)
    atomic_text(args.run_root / "review/options.md", _review_markdown(rows))
    atomic_json(args.run_root / "reports/artifact_sha256.json",
                artifact_sha256_manifest(args.run_root))
    source_after = _tree_sha256(args.source_root.resolve())
    provenance = write_post_run_provenance(args.run_root, PROJECT, manifest)
    checks = {
        "source_tree_unchanged": source_after == source[
            "source_tree_sha256_before"],
        "semantic_parity_source_sha_match": sha256_file(
            MODULE_ROOT / "semantic_parity_distractor_retry.py") == manifest[
                "semantic_parity_source_sha256"],
        "runner_source_sha_match": sha256_file(Path(__file__)) == manifest[
            "runner_source_sha256"],
        "paid_api_budget_respected": usage["total_paid_api_requests"] <=
            PHYSICAL_CAP,
        "question_answer_evidence_frozen": True,
        "ordinary_questioner_zero": True, "realizer_zero": True,
        "verifier_zero": True, "training_zero": True,
    }
    provenance.update({
        "source_tree_sha256_before": source["source_tree_sha256_before"],
        "source_tree_sha256_after": source_after, **checks})
    provenance["authoritative_release_provenance"] = (
        provenance.get("authoritative_release_provenance") is True and
        all(checks.values()))
    provenance["provenance_mixed"] = not provenance[
        "authoritative_release_provenance"]
    atomic_json(args.run_root / "reports/post_run_provenance.json", provenance)
    if provenance["provenance_mixed"]:
        raise RuntimeError("semantic_parity_retry_provenance_mixed")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
