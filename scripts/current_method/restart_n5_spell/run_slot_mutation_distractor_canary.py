"""Generate deterministic single-slot distractors for two frozen QAs and evaluate them."""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .contrastive_distractor_canary import (
    _evidence_rows,
    audit_frozen_source_qa,
    materialize_contrastive_candidate,
    option_length_diagnostics,
)
from .core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from .input_adapter import read_json
from .ledger import RequestLedger
from .live_backends import (
    LiveSemanticRepairBackend,
    OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from .run_canary import create_source_snapshot, project_diff_sha256, write_post_run_provenance
from .run_contrastive_distractor_canary import _base_qa, _solver_summary, _tree_sha256
from .run_cross_dialogue_calibration import _base_manifest, _run_solver, _solver_config
from .run_rolling_outline_30k_canary import (
    CHALLENGER_MODEL, MODEL_CONTEXT_LIMIT, SOLVER, TOKENIZER, _usage,
)
from .run_semantic_neighbor_distractor_canary import (
    DEFAULT_SOURCE_ROOT,
    _attest_source,
    _run_single_paid_request,
)
from .semantic_neighbor_distractor_canary import deterministic_semantic_neighbor_placement
from .slot_mutation_distractor_canary import (
    SLOT_MUTATION_CANARY_VERSION,
    SLOT_MUTATION_PROMPT_VERSION,
    SLOT_MUTATION_SCHEMA_VERSION,
    build_slot_mutation_prompt,
    slot_mutation_prompt_sha256,
    slot_mutation_schema_sha256,
    validate_and_render_slot_mutations,
)
from .vllm_backends import VLLMSolverBackend


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_slot_mutation_distractor_canary_v1"
)
NORMAL_REQUEST_CAP = 2
PHYSICAL_REQUEST_CAP = 2
FORMAT_RETRY_CAP = 0
MAX_TOKENS = 1024


def _minimal_view(evidence: list[dict[str, str]]) -> dict[str, Any]:
    turns = [{
        "turn_id": row["turn_id"], "session_id": row["session_id"],
        "speaker": row["speaker"], "text": row["exact_text"],
    } for row in evidence]
    catalog = {"spans": evidence}
    return {
        "combined_dialogue": turns, "span_catalog": catalog,
        "view_sha256": sha256_bytes(canonical_json({
            "combined_dialogue": turns, "span_catalog": catalog,
        }).encode()),
    }


def _review(source: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    by_id = {row["qa_id"]: row for row in rows}
    lines = ["# Slot-Mutation Distractor Canary", ""]
    for item in source["source_items"]:
        row = by_id[item["qa_id"]]
        validation = (row.get("api_task") or {}).get("accepted_validation") or {}
        lines.extend([
            f"## {item['qa_id']}", "",
            f"Question: {item['raw_qa']['question']}", "",
            f"Correct answer: {item['raw_qa']['correct_answer_text']}", "",
            f"Frame: {validation.get('frame')}", "",
            "Mutations:", "",
        ])
        for mutation in validation.get("mutations") or []:
            lines.append(
                f"- {mutation['changed_slot']}: `{mutation['gold_value']}` -> "
                f"`{mutation['replacement_value']}`")
        lines.extend(["", "Rendered options:", ""])
        for letter, value in ((row.get("materialized_qa") or {}).get("options") or {}).items():
            lines.append(f"- {letter}. {value}")
        lines.extend([
            "", f"Minimal: {row.get('minimal_decision')}",
            f"Stage2 final: {row.get('final_decision')}",
            f"Grounding diagnosis: {row.get('grounding_diagnosis')}", "",
        ])
    return "\n".join(lines)


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
    source = _attest_source(args.source_root.resolve())
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    api = replace(load_challenger_config(args.api_config), model=CHALLENGER_MODEL,
                  temperature=0.25, enable_thinking=False, timeout_seconds=600)
    solver_config_native = read_json(SOLVER / "config.json")
    if int(solver_config_native.get("max_position_embeddings") or 0) < MODEL_CONTEXT_LIMIT:
        raise RuntimeError("solver_native_context_too_short")

    prepared: list[dict[str, Any]] = []
    source_audits = []
    for item in source["source_items"]:
        base = _base_qa(item["raw_qa"])
        audit = audit_frozen_source_qa(raw_qa=base,
            materialized_qa=item["materialized_qa"], span_catalog=source["catalog"])
        if audit["decision"] != "source_qa_accept":
            raise RuntimeError(f"source_qa_audit_failed:{base['qa_id']}")
        source_audits.append(audit)
        evidence = _evidence_rows(base_qa=base, span_catalog=source["catalog"])
        prompt = build_slot_mutation_prompt(base_qa=base, evidence_rows=evidence)
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        preflight = {
            "qa_id": base["qa_id"], "rendered_prompt_tokens": prompt_tokens,
            "reserved_generation_tokens": MAX_TOKENS,
            "total_tokens": prompt_tokens + MAX_TOKENS,
            "backend_context_limit": MODEL_CONTEXT_LIMIT,
            "decision": "pass" if prompt_tokens + MAX_TOKENS <=
                MODEL_CONTEXT_LIMIT else "reject_context_overflow",
            "truncated": False,
        }
        if preflight["decision"] != "pass":
            raise RuntimeError(f"prompt_context_overflow:{base['qa_id']}")
        prepared.append({"base": base, "evidence": evidence, "prompt": prompt,
                         "preflight": preflight})
    if args.dry_run:
        print(json.dumps({
            "decision": "dry_run_pass",
            "source_audits": source_audits,
            "prompt_preflights": [row["preflight"] for row in prepared],
            "normal_request_cap": NORMAL_REQUEST_CAP,
            "format_retry_cap": FORMAT_RETRY_CAP,
            "physical_request_cap": PHYSICAL_REQUEST_CAP,
            "challenger_model": api.model, "solver_model": str(SOLVER),
        }, ensure_ascii=False, indent=2))
        return 0

    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    project_diff = project_diff_sha256(PROJECT)
    manifest = {
        **_base_manifest(run_id=args.run_root.name,
            mode=SLOT_MUTATION_CANARY_VERSION, snapshot=snapshot,
            project_diff=project_diff),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "canary_version": SLOT_MUTATION_CANARY_VERSION,
        "source_root": source["source_root"],
        "source_tree_sha256_before": source["source_tree_sha256_before"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "source_provenance_sha256": source["source_provenance_sha256"],
        "source_context_sha256": source["source_context_sha256"],
        "source_catalog_sha256": source["source_catalog_sha256"],
        "slot_mutation_prompt_version": SLOT_MUTATION_PROMPT_VERSION,
        "slot_mutation_prompt_sha256": slot_mutation_prompt_sha256(),
        "slot_mutation_schema_version": SLOT_MUTATION_SCHEMA_VERSION,
        "slot_mutation_schema_sha256": slot_mutation_schema_sha256(),
        "slot_mutation_source_sha256": sha256_file(
            MODULE_ROOT / "slot_mutation_distractor_canary.py"),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "challenger_model_identifier": api.model,
        "challenger_temperature": api.temperature,
        "challenger_enable_thinking": api.enable_thinking,
        "challenger_input_scope": "frozen_qa_plus_minimal_answer_evidence_only",
        "full_dialogue_in_challenger_prompt": False,
        "previous_distractors_in_challenger_prompt": False,
        "solver_model": str(SOLVER), "solver_tokenizer": str(TOKENIZER),
        "solver_max_model_len": MODEL_CONTEXT_LIMIT,
        "budget": {"normal_requests": 2, "format_only_retries": 0,
                   "physical_paid_api_requests": 2,
                   "ordinary_questioner": 0, "realizer": 0, "verifier": 0,
                   "training": 0, "semantic_repair": 0},
        "historical_runs_modified": False,
    }
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", {
        "mode": SLOT_MUTATION_CANARY_VERSION, "source_root": source["source_root"],
        "challenger_model": api.model, "temperature": api.temperature,
        "enable_thinking": api.enable_thinking, "normal_request_cap": 2,
        "format_retry_cap": 0, "physical_request_cap": 2,
        "solver_model": str(SOLVER), "solver_max_model_len": MODEL_CONTEXT_LIMIT,
    })
    atomic_json(args.run_root / "frozen/source_attestation.json", {
        key: value for key, value in source.items()
        if key not in {"context", "catalog", "source_items"}
    })
    atomic_json(args.run_root / "frozen/full_dialogue.json", source["context"])
    atomic_json(args.run_root / "frozen/questioner_span_catalog.json", source["catalog"])
    atomic_json(args.run_root / "reports/preflight.json", {
        "decision": "pass", "source_audits": source_audits,
        "prompt_preflights": [row["preflight"] for row in prepared],
        "api_budget_preflight": "pass", "solver_path_preflight": "pass",
    })

    ledger = RequestLedger(args.run_root,
        sha256_file(args.run_root / "config/run_manifest.json"), "live")
    backend = LiveSemanticRepairBackend(OpenAICompatibleChallengerClient(api), MAX_TOKENS)
    rows: list[dict[str, Any]] = []
    for prepared_item in prepared:
        base = prepared_item["base"]
        view = _minimal_view(prepared_item["evidence"])
        task = _run_single_paid_request(
            run_root=args.run_root, ledger=ledger, backend=backend,
            prompt=prepared_item["prompt"], qa_id=base["qa_id"], view=view,
            answer=base["correct_answer_text"],
            validator=lambda raw, answer=base["correct_answer_text"]:
                validate_and_render_slot_mutations(raw,
                    correct_answer_text=str(answer)))
        row: dict[str, Any] = {
            "qa_id": base["qa_id"], "attempt_index": 0, "api_task": task,
            "slot_mutation_valid": task["accepted"],
            "minimal_decision": "not_run", "solver_evaluated": False,
            "final_decision": task["final_decision"], "grounding_diagnosis": None,
        }
        if task["accepted"]:
            placement = deterministic_semantic_neighbor_placement(base["qa_id"])
            candidate = materialize_contrastive_candidate(
                base_qa=base, distractors=task["accepted_validation"]["distractors"],
                target_letter=placement["target_letter"], span_catalog=source["catalog"])
            path = args.run_root / f"frozen/materialized_mcqs/{base['qa_id']}_attempt_0.json"
            atomic_json(path, candidate)
            row.update({
                "placement": placement,
                "materialized_mcq_path": str(path.relative_to(args.run_root)),
                "materialized_mcq_sha256": sha256_file(path),
                "materialized_qa": candidate["materialized_qa"],
                "minimal_decision": candidate["admission"]["spell_style_minimal_v1"]["decision"],
                "minimal_reasons": candidate["admission"]["spell_style_minimal_v1"]["reasons"],
                "option_length_diagnostics": option_length_diagnostics(
                    candidate["materialized_qa"] or {}),
                "candidate": candidate,
            })
        rows.append(row)

    solver_backend: VLLMSolverBackend | None = None
    for row in rows:
        candidate = row.pop("candidate", None)
        if candidate is None or row["minimal_decision"] != "accept":
            if candidate is not None:
                row["final_decision"] = "minimal_admission_reject"
            continue
        if solver_backend is None:
            solver_backend = VLLMSolverBackend(
                model_path=SOLVER, tokenizer_path=TOKENIZER,
                max_model_len=MODEL_CONTEXT_LIMIT, temperature=0.7,
                top_p=0.9, max_tokens=32, gpu_memory_utilization=0.90)
        qa_id = row["qa_id"]
        solver_root = args.run_root / f"solver_runs/{qa_id}/attempt_0"
        solver_root.mkdir(parents=True)
        resolved = _solver_config(run_root=solver_root)
        resolved.update({"mode": "local_stage2_solver_slot_mutation_canary",
                         "solver_effective_max_model_len": MODEL_CONTEXT_LIMIT,
                         "max_local_solver_generations": 80})
        solver_manifest = {
            **_base_manifest(run_id=f"{args.run_root.name}-{qa_id}",
                mode="local_stage2_solver_slot_mutation_canary",
                snapshot=snapshot, project_diff=project_diff),
            "qa_id": qa_id, "attempt_index": 0, "solver_model": str(SOLVER),
            "solver_max_model_len": MODEL_CONTEXT_LIMIT,
            "source_canary_manifest_sha256": sha256_file(
                args.run_root / "config/run_manifest.json"),
            "source_materialized_mcq_sha256": row["materialized_mcq_sha256"],
        }
        atomic_json(solver_root / "config/resolved_canary_config.json", resolved)
        atomic_json(solver_root / "config/run_manifest.json", solver_manifest)
        solver_report = _run_solver(
            root=solver_root, manifest=solver_manifest, config=resolved,
            raw_batch=[candidate["raw_v8_qa"]], admission=candidate["admission"],
            materialized=candidate["materialized_qa"],
            artifacts={"context": source["context"], "catalog": source["catalog"],
                       "contracts": candidate["reasoning_contracts"]},
            tokenizer=tokenizer, backend=solver_backend)
        write_post_run_provenance(solver_root, PROJECT, solver_manifest)
        summary = _solver_summary(solver_report)
        row.update({"solver_evaluated": True,
                    "solver_root": str(solver_root.relative_to(args.run_root)),
                    "solver_report_sha256": sha256_file(
                        solver_root / "reports/cross_dialogue_solver_audit.json"),
                    "solver": summary, "final_decision": summary["final_decision"],
                    "grounding_diagnosis": summary["no_context_permutation_diagnostic"]})

    usage = {**_usage(args.run_root), "normal_request_cap": 2,
             "format_only_retry_cap": 0, "physical_paid_api_request_cap": 2,
             "ordinary_questioner_generations": 0, "realizer_generations": 0,
             "verifier_generations": 0, "training_runs": 0,
             "semantic_repair_generations": 0}
    atomic_json(args.run_root / "reports/api_usage.json", usage)
    report = {
        "canary_version": SLOT_MUTATION_CANARY_VERSION,
        "source_qa_count": len(rows), "slot_mutation_valid_count": sum(
            row["slot_mutation_valid"] for row in rows),
        "minimal_accepted_count": sum(row["minimal_decision"] == "accept" for row in rows),
        "solver_evaluated_count": sum(row["solver_evaluated"] for row in rows),
        "final_decisions": dict(sorted(collections.Counter(
            str(row["final_decision"]) for row in rows).items())),
        "attempts": rows, "api_usage": usage,
        "ordinary_questioner_generations": 0, "realizer_generations": 0,
        "verifier_generations": 0, "training_runs": 0,
        "semantic_repair_generations": 0,
    }
    atomic_json(args.run_root / "reports/slot_mutation_distractor_canary.json", report)
    atomic_text(args.run_root / "review/options_and_results.md", _review(source, rows))
    source_tree_after = _tree_sha256(args.source_root.resolve())
    provenance = write_post_run_provenance(args.run_root, PROJECT, manifest)
    checks = {
        "slot_mutation_source_sha_match": sha256_file(
            MODULE_ROOT / "slot_mutation_distractor_canary.py") == manifest[
                "slot_mutation_source_sha256"],
        "runner_source_sha_match": sha256_file(Path(__file__)) == manifest[
            "runner_source_sha256"],
        "historical_source_tree_unchanged": source_tree_after == source[
            "source_tree_sha256_before"],
        "physical_api_budget_respected": usage["total_paid_api_requests"] <= 2,
        "ordinary_questioner_zero": True, "realizer_zero": True,
        "verifier_zero": True, "training_zero": True, "semantic_repair_zero": True,
    }
    provenance.update({"source_tree_sha256_before": source["source_tree_sha256_before"],
                       "source_tree_sha256_after": source_tree_after, **checks})
    provenance["authoritative_release_provenance"] = (
        provenance.get("authoritative_release_provenance") is True and all(checks.values()))
    provenance["provenance_mixed"] = not provenance["authoritative_release_provenance"]
    atomic_json(args.run_root / "reports/post_run_provenance.json", provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
