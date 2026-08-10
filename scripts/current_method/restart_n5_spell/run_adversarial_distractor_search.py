from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .adversarial_distractor_search import (
    ADVERSARIAL_DISTRACTOR_PROMPT_VERSION,
    ADVERSARIAL_DISTRACTOR_SCHEMA_VERSION,
    ADVERSARIAL_DISTRACTOR_SEARCH_VERSION,
    BUNDLE_IDS,
    adversarial_distractor_prompt_sha256,
    adversarial_distractor_schema_sha256,
    build_adversarial_distractor_prompt,
    deterministic_bundle_target_letter,
    is_grounding_search_success,
    validate_adversarial_bundles,
)
from .contrastive_distractor_canary import (
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
from .run_cross_dialogue_calibration import _base_manifest, _run_solver, _solver_config
from .run_contrastive_distractor_canary import _solver_summary, _tree_sha256
from .run_single_item_qa_canary import (
    CHALLENGER_MODEL,
    DEFAULT_SOLVER,
    DEFAULT_TOKENIZER,
    MODEL_CONTEXT_LIMIT,
    _solver_manifest,
)
from .run_single_item_semantic_parity_retry import (
    _attest_source,
    _minimal_request_view,
    _usage,
)
from .vllm_backends import VLLMSolverBackend


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "spell_single_item_qa_full_canary_v1"
)
DEFAULT_PRIOR_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "spell_single_item_semantic_parity_retry_v1"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "spell_adversarial_distractor_search_v1"
)
MAX_TOKENS = 2048
NORMAL_CAP = 2
FORMAT_RETRY_CAP = 2
PHYSICAL_CAP = 4


def _prior_failed_sets(prior_root: Path) -> dict[str, list[list[str]]]:
    result: dict[str, list[list[str]]] = {}
    report_path = prior_root / "reports/semantic_parity_retry.json"
    if not report_path.is_file():
        return result
    report = read_json(report_path)
    for row in report.get("items") or []:
        sets = [list(row.get("previous_distractors") or [])]
        options = row.get("options") or {}
        correct = str(row.get("correct_answer_text") or "").casefold()
        generated = [str(value) for value in options.values()
                     if str(value).casefold() != correct]
        if len(generated) == 3:
            sets.append(generated)
        result[str(row["lineage"])] = sets
    return result


def _review_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Adversarial distractor bundle search", ""]
    for row in rows:
        lines.extend([
            f"## {row['lineage']} / {row['qa_id']}", "",
            f"Question: {row['question']}", "",
            f"Correct answer: {row['correct_answer_text']}", "",
        ])
        if not row["api_task"]["accepted"]:
            lines.extend([
                f"Generation: {row['api_task']['final_decision']}", ""])
            continue
        for bundle in row.get("bundle_results") or []:
            lines.extend([
                f"### {bundle['bundle_id']}", "",
                f"Target letter: {bundle['placement']['target_letter']}", "",
            ])
            for letter, value in (bundle.get("options") or {}).items():
                lines.append(f"- {letter}. {value}")
            solver = bundle.get("solver_summary") or {}
            lines.extend([
                "",
                f"Minimal: {bundle.get('minimal_decision')}",
                f"No-context answers: {solver.get('no_context_valid_answers')}",
                f"No-context rewards: {solver.get('no_context_rewards')}",
                f"Full-context rewards: {solver.get('full_context_rewards')}",
                f"Final decision: {bundle.get('final_decision')}",
                "",
            ])
        lines.extend([
            f"Selected bundle: {row.get('selected_bundle_id')}",
            f"Search decision: {row.get('search_decision')}", "",
        ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate four distractor bundles and search with Solver")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path,
                        default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.run_root.exists():
        raise FileExistsError(f"adversarial_search_root_exists:{args.run_root}")
    if not DEFAULT_SOLVER.is_dir() or not DEFAULT_TOKENIZER.is_dir():
        raise RuntimeError("adversarial_search_solver_or_tokenizer_missing")

    source = _attest_source(args.source_root)
    source_tree_before = source["source_tree_sha256_before"]
    prior_tree_before = (_tree_sha256(args.prior_root.resolve())
                         if args.prior_root.is_dir() else None)
    prior_sets = _prior_failed_sets(args.prior_root)
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_TOKENIZER, local_files_only=True)
    api = replace(load_challenger_config(args.api_config),
                  model=CHALLENGER_MODEL, temperature=0.65,
                  enable_thinking=False, timeout_seconds=600,
                  questioner_max_tokens=MAX_TOKENS)

    prepared = []
    for item in source["items"]:
        sets = prior_sets.get(item["lineage"], [item["previous_distractors"]])
        prompt = build_adversarial_distractor_prompt(
            base_qa=item["base_qa"], evidence_rows=item["evidence"],
            prior_failed_sets=sets)
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        preflight = {
            "lineage": item["lineage"], "qa_id": item["qa_id"],
            "rendered_prompt_tokens": prompt_tokens,
            "reserved_generation_tokens": MAX_TOKENS,
            "total_tokens": prompt_tokens + MAX_TOKENS,
            "backend_context_limit": MODEL_CONTEXT_LIMIT,
            "decision": "pass" if prompt_tokens + MAX_TOKENS <=
                MODEL_CONTEXT_LIMIT else "reject_context_overflow",
            "truncated": False,
        }
        if preflight["decision"] != "pass":
            raise RuntimeError(
                f"adversarial_search_preflight_failed:{item['lineage']}")
        prepared.append({**item, "prompt": prompt, "preflight": preflight,
                         "prior_failed_sets": sets})
    if args.dry_run:
        print(json.dumps({
            "decision": "dry_run_pass",
            "source_root": source["source_root"],
            "preflights": [item["preflight"] for item in prepared],
            "bundle_count_per_qa": len(BUNDLE_IDS),
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
            mode=ADVERSARIAL_DISTRACTOR_SEARCH_VERSION,
            snapshot=snapshot, project_diff=diff_sha),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": source["source_root"],
        "source_tree_sha256_before": source_tree_before,
        "prior_failed_root": str(args.prior_root.resolve()),
        "prior_failed_tree_sha256_before": prior_tree_before,
        "prompt_version": ADVERSARIAL_DISTRACTOR_PROMPT_VERSION,
        "prompt_sha256": adversarial_distractor_prompt_sha256(),
        "schema_version": ADVERSARIAL_DISTRACTOR_SCHEMA_VERSION,
        "schema_sha256": adversarial_distractor_schema_sha256(),
        "format_retry_version": FORMAT_ONLY_RETRY_VERSION,
        "format_retry_suffix_sha256": format_only_retry_suffix_sha256(),
        "challenger_model_identifier": api.model,
        "challenger_temperature": api.temperature,
        "question_answer_evidence_frozen": True,
        "bundle_search_order": list(BUNDLE_IDS),
        "bundle_stop_decisions": ["reject_easy", "retain_frontier"],
        "source_code_sha256": {
            "protocol": sha256_file(
                MODULE_ROOT / "adversarial_distractor_search.py"),
            "runner": sha256_file(Path(__file__)),
        },
        "budget": {
            "normal_requests": NORMAL_CAP,
            "format_only_retries": FORMAT_RETRY_CAP,
            "physical_paid_api_requests": PHYSICAL_CAP,
            "ordinary_questioner": 0, "realizer": 0,
            "verifier": 0, "training": 0,
        },
        "historical_runs_modified": False,
    }
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", {
        "mode": ADVERSARIAL_DISTRACTOR_SEARCH_VERSION,
        "source_root": source["source_root"],
        "prior_root": str(args.prior_root.resolve()),
        "model": api.model, "temperature": api.temperature,
        "max_tokens": MAX_TOKENS,
        "bundle_count_per_qa": len(BUNDLE_IDS),
        "solver_model": str(DEFAULT_SOLVER),
        "solver_tokenizer": str(DEFAULT_TOKENIZER),
        "solver_max_model_len": 16384,
        "normal_request_cap": NORMAL_CAP,
        "format_retry_cap": FORMAT_RETRY_CAP,
        "physical_request_cap": PHYSICAL_CAP,
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
        atomic_text(args.run_root / f"prompts/{lineage}.txt", item["prompt"])

    ledger = RequestLedger(
        args.run_root, sha256_file(args.run_root / "config/run_manifest.json"),
        "live")
    budget = ApiBudget(
        normal_caps={"distractor_writer": NORMAL_CAP},
        format_retry_cap=FORMAT_RETRY_CAP, physical_cap=PHYSICAL_CAP)
    backend = LiveSemanticRepairBackend(
        OpenAICompatibleChallengerClient(api), MAX_TOKENS)
    solver_backend: VLLMSolverBackend | None = None
    rows: list[dict[str, Any]] = []
    local_vllm_generations = 0

    for item in prepared:
        base = item["base_qa"]
        task = run_single_object_task(
            run_root=args.run_root, ledger=ledger, backend=backend,
            budget=budget, stage="distractor_writer",
            logical_request_id=f"adversarial-bundles-{item['lineage']}-q001",
            semantic_prompt=item["prompt"],
            lineage_view=_minimal_request_view(item),
            validator=lambda raw, answer=base["correct_answer_text"]:
                validate_adversarial_bundles(
                    raw, correct_answer_text=answer),
            schema_retry_decisions=("adversarial_bundles_schema_invalid",),
            ledger_budget_limit=PHYSICAL_CAP,
            ledger_total_budget_limit=PHYSICAL_CAP,
            json_invalid_decision="adversarial_bundles_json_invalid")
        row: dict[str, Any] = {
            "lineage": item["lineage"], "qa_id": item["qa_id"],
            "question": base["question"],
            "correct_answer_text": base["correct_answer_text"],
            "answer_support_span_ids": list(base["answer_support_span_ids"]),
            "prior_failed_sets": item["prior_failed_sets"],
            "api_task": task, "bundle_results": [],
            "selected_bundle_id": None,
            "search_decision": task["final_decision"],
        }
        if not task["accepted"]:
            rows.append(row)
            continue

        for bundle_index, bundle in enumerate(
                task["accepted_validation"]["bundles"]):
            placement = deterministic_bundle_target_letter(
                base["qa_id"], bundle_index)
            candidate = materialize_contrastive_candidate(
                base_qa=base, distractors=bundle["distractors"],
                target_letter=placement["target_letter"],
                span_catalog=item["catalog"])
            candidate_path = args.run_root / (
                f"frozen/materialized_mcqs/{base['qa_id']}_"
                f"{bundle['bundle_id']}.json")
            atomic_json(candidate_path, candidate)
            minimal = candidate["admission"]["spell_style_minimal_v1"]
            bundle_row: dict[str, Any] = {
                "bundle_id": bundle["bundle_id"],
                "distractors": bundle["distractors"],
                "generation_diagnostics": bundle["diagnostics"],
                "placement": placement,
                "options": (candidate["materialized_qa"] or {}).get("options"),
                "materialized_mcq_path": str(
                    candidate_path.relative_to(args.run_root)),
                "materialized_mcq_sha256": sha256_file(candidate_path),
                "minimal_decision": minimal["decision"],
                "minimal_reasons": minimal["reasons"],
                "option_length_diagnostics": option_length_diagnostics(
                    candidate["materialized_qa"] or {}),
                "solver_evaluated": False,
                "final_decision": "minimal_admission_reject",
            }
            if minimal["decision"] != "accept":
                row["bundle_results"].append(bundle_row)
                continue
            if solver_backend is None:
                solver_backend = VLLMSolverBackend(
                    model_path=DEFAULT_SOLVER, tokenizer_path=DEFAULT_TOKENIZER,
                    max_model_len=16384, temperature=0.7, top_p=0.9,
                    max_tokens=32, gpu_memory_utilization=0.80)
            solver_root = args.run_root / (
                f"solver_runs/{item['lineage']}_q001/{bundle['bundle_id']}")
            solver_root.mkdir(parents=True)
            solver_config = _solver_config(run_root=solver_root)
            solver_manifest = _solver_manifest(
                root=solver_root, snapshot=snapshot, project_diff=diff_sha,
                lineage=item["lineage"], qa=candidate["materialized_qa"],
                bundle=item, materialized_sha=sha256_bytes(canonical_json(
                    candidate["materialized_qa"]).encode()))
            solver_manifest.update({
                "adversarial_distractor_search_version":
                    ADVERSARIAL_DISTRACTOR_SEARCH_VERSION,
                "bundle_id": bundle["bundle_id"],
                "bundle_order_index": bundle_index,
            })
            atomic_json(solver_root / "config/resolved_canary_config.json",
                        solver_config)
            atomic_json(solver_root / "config/run_manifest.json",
                        solver_manifest)
            solver_report = _run_solver(
                root=solver_root, manifest=solver_manifest,
                config=solver_config,
                raw_batch=[candidate["raw_v8_qa"]],
                admission=candidate["admission"],
                materialized=candidate["materialized_qa"],
                artifacts={"context": item["context"],
                           "catalog": item["catalog"],
                           "contracts": candidate["reasoning_contracts"]},
                tokenizer=tokenizer, backend=solver_backend)
            write_post_run_provenance(
                solver_root, PROJECT, solver_manifest)
            summary = _solver_summary(solver_report)
            local_vllm_generations += int(
                solver_report.get("request_count") or 0)
            bundle_row.update({
                "solver_evaluated": True,
                "solver_root": str(solver_root),
                "solver_report_sha256": sha256_file(
                    solver_root / "reports/cross_dialogue_solver_audit.json"),
                "solver_summary": summary,
                "grounding_decision": summary["grounding_decision"],
                "final_decision": summary["final_decision"],
                "grounding_search_success": is_grounding_search_success(
                    summary["final_decision"]),
            })
            row["bundle_results"].append(bundle_row)
            if bundle_row["grounding_search_success"]:
                row["selected_bundle_id"] = bundle["bundle_id"]
                row["search_decision"] = "grounded_option_bundle_found"
                break
        if row["selected_bundle_id"] is None:
            row["search_decision"] = "no_grounded_option_bundle_found"
        rows.append(row)

    usage = {
        "distractor_writer_normal_requests":
            budget.normal_counts.get("distractor_writer", 0),
        "distractor_writer_format_retries":
            budget.retry_counts.get("distractor_writer", 0),
        "total_paid_api_requests": ledger.summary()["paid_api_generation_count"],
        **_usage(ledger),
        "local_vllm_generations": local_vllm_generations,
        "ordinary_questioner_generations": 0,
        "realizer_generations": 0, "verifier_generations": 0,
        "training_runs": 0, "semantic_repair_generations": 0,
    }
    report = {
        "search_version": ADVERSARIAL_DISTRACTOR_SEARCH_VERSION,
        "qa_items": len(rows),
        "generation_valid": sum(row["api_task"]["accepted"] for row in rows),
        "bundles_evaluated": sum(len(row["bundle_results"]) for row in rows),
        "grounded_option_bundles_found": sum(
            row["selected_bundle_id"] is not None for row in rows),
        "items": rows,
        "api_usage": usage,
        "question_answer_evidence_modified": False,
        "historical_runs_modified": False,
    }
    atomic_json(args.run_root / "reports/adversarial_distractor_search.json",
                report)
    atomic_json(args.run_root / "reports/api_usage.json", usage)
    atomic_text(args.run_root / "review/options.md", _review_markdown(rows))
    atomic_json(args.run_root / "reports/artifact_sha256.json",
                artifact_sha256_manifest(args.run_root))

    source_tree_after = _tree_sha256(args.source_root.resolve())
    prior_tree_after = (_tree_sha256(args.prior_root.resolve())
                        if args.prior_root.is_dir() else None)
    provenance = write_post_run_provenance(args.run_root, PROJECT, manifest)
    checks = {
        "source_tree_unchanged": source_tree_after == source_tree_before,
        "prior_tree_unchanged": prior_tree_after == prior_tree_before,
        "protocol_source_sha_match": sha256_file(
            MODULE_ROOT / "adversarial_distractor_search.py") ==
            manifest["source_code_sha256"]["protocol"],
        "runner_source_sha_match": sha256_file(Path(__file__)) ==
            manifest["source_code_sha256"]["runner"],
        "paid_api_budget_respected": usage["total_paid_api_requests"] <=
            PHYSICAL_CAP,
        "question_answer_evidence_frozen": True,
        "ordinary_questioner_zero": True,
        "realizer_zero": True, "verifier_zero": True,
        "training_zero": True,
    }
    provenance.update({
        "source_tree_sha256_before": source_tree_before,
        "source_tree_sha256_after": source_tree_after,
        "prior_tree_sha256_before": prior_tree_before,
        "prior_tree_sha256_after": prior_tree_after,
        **checks,
    })
    provenance["authoritative_release_provenance"] = (
        provenance.get("authoritative_release_provenance") is True and
        all(checks.values()))
    provenance["provenance_mixed"] = not provenance[
        "authoritative_release_provenance"]
    atomic_json(args.run_root / "reports/post_run_provenance.json", provenance)
    if provenance["provenance_mixed"]:
        raise RuntimeError("adversarial_distractor_search_provenance_mixed")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
