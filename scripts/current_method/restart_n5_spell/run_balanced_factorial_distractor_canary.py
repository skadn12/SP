"""Generate two balanced 2x2 distractor sets without running a Solver."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .balanced_factorial_distractor_canary import (
    BALANCED_FACTORIAL_CANARY_VERSION,
    BALANCED_FACTORIAL_PROMPT_VERSION,
    BALANCED_FACTORIAL_SCHEMA_VERSION,
    balanced_factorial_prompt_sha256,
    balanced_factorial_schema_sha256,
    build_balanced_factorial_prompt,
    deterministic_balanced_factorial_placement,
    validate_and_render_balanced_factorial,
)
from .contrastive_distractor_canary import (
    _evidence_rows, audit_frozen_source_qa, materialize_contrastive_candidate,
    option_length_diagnostics,
)
from .core import atomic_json, atomic_text, sha256_file
from .ledger import RequestLedger
from .live_backends import (
    LiveSemanticRepairBackend, OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from .run_canary import create_source_snapshot, project_diff_sha256, write_post_run_provenance
from .run_contrastive_distractor_canary import _base_qa, _tree_sha256
from .run_cross_dialogue_calibration import _base_manifest
from .run_rolling_outline_30k_canary import (
    CHALLENGER_MODEL, MODEL_CONTEXT_LIMIT, SOLVER, TOKENIZER, _usage,
)
from .run_semantic_neighbor_distractor_canary import (
    DEFAULT_SOURCE_ROOT, _attest_source, _run_single_paid_request,
)
from .run_slot_mutation_distractor_canary import _minimal_view


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_balanced_factorial_distractor_canary_v1"
)
MAX_TOKENS = 1024


def _review(source: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    by_id = {row["qa_id"]: row for row in rows}
    lines = ["# Balanced 2x2 Distractor Canary", ""]
    for item in source["source_items"]:
        row = by_id[item["qa_id"]]
        validation = (row["api_task"].get("accepted_validation") or {})
        lines.extend([
            f"## {item['qa_id']}", "",
            f"Question: {item['raw_qa']['question']}", "",
            f"Frame: {validation.get('frame')}", "", "Slots:", "",
        ])
        for slot_id, values in (validation.get("slots") or {}).items():
            lines.append(
                f"- {slot_id}: `{values['gold_value']}` / "
                f"`{values['contrast_value']}`")
        lines.extend(["", "Balanced options:", ""])
        for letter, value in (row["materialized_qa"].get("options") or {}).items():
            lines.append(f"- {letter}. {value}")
        lines.extend(["", "Each S1 and S2 value occurs exactly twice.", ""])
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
    prepared = []
    source_audits = []
    for item in source["source_items"]:
        base = _base_qa(item["raw_qa"])
        audit = audit_frozen_source_qa(raw_qa=base,
            materialized_qa=item["materialized_qa"], span_catalog=source["catalog"])
        if audit["decision"] != "source_qa_accept":
            raise RuntimeError(f"source_qa_audit_failed:{base['qa_id']}")
        source_audits.append(audit)
        evidence = _evidence_rows(base_qa=base, span_catalog=source["catalog"])
        prompt = build_balanced_factorial_prompt(
            base_qa=base, evidence_rows=evidence)
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
        prepared.append({"base": base, "evidence": evidence,
                         "prompt": prompt, "preflight": preflight})
    if args.dry_run:
        print(json.dumps({
            "decision": "dry_run_pass", "source_audits": source_audits,
            "prompt_preflights": [row["preflight"] for row in prepared],
            "normal_request_cap": 2, "format_retry_cap": 0,
            "physical_request_cap": 2, "challenger_model": api.model,
        }, ensure_ascii=False, indent=2))
        return 0

    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    project_diff = project_diff_sha256(PROJECT)
    manifest = {
        **_base_manifest(run_id=args.run_root.name,
            mode=BALANCED_FACTORIAL_CANARY_VERSION, snapshot=snapshot,
            project_diff=project_diff),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "canary_version": BALANCED_FACTORIAL_CANARY_VERSION,
        "source_root": source["source_root"],
        "source_tree_sha256_before": source["source_tree_sha256_before"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "source_provenance_sha256": source["source_provenance_sha256"],
        "source_context_sha256": source["source_context_sha256"],
        "source_catalog_sha256": source["source_catalog_sha256"],
        "balanced_factorial_prompt_version": BALANCED_FACTORIAL_PROMPT_VERSION,
        "balanced_factorial_prompt_sha256": balanced_factorial_prompt_sha256(),
        "balanced_factorial_schema_version": BALANCED_FACTORIAL_SCHEMA_VERSION,
        "balanced_factorial_schema_sha256": balanced_factorial_schema_sha256(),
        "balanced_factorial_source_sha256": sha256_file(
            MODULE_ROOT / "balanced_factorial_distractor_canary.py"),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "challenger_model_identifier": api.model,
        "challenger_temperature": api.temperature,
        "challenger_enable_thinking": api.enable_thinking,
        "challenger_input_scope": "frozen_qa_plus_minimal_answer_evidence_only",
        "full_dialogue_in_challenger_prompt": False,
        "previous_distractors_in_challenger_prompt": False,
        "solver_model_for_later_evaluation": str(SOLVER),
        "budget": {"normal_requests": 2, "format_only_retries": 0,
                   "physical_paid_api_requests": 2,
                   "ordinary_questioner": 0, "realizer": 0, "verifier": 0,
                   "training": 0, "semantic_repair": 0},
        "historical_runs_modified": False,
    }
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", {
        "mode": BALANCED_FACTORIAL_CANARY_VERSION,
        "source_root": source["source_root"], "challenger_model": api.model,
        "temperature": api.temperature, "enable_thinking": api.enable_thinking,
        "normal_request_cap": 2, "format_retry_cap": 0,
        "physical_request_cap": 2, "solver_run_in_this_root": False,
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
        "api_budget_preflight": "pass",
    })
    ledger = RequestLedger(args.run_root,
        sha256_file(args.run_root / "config/run_manifest.json"), "live")
    backend = LiveSemanticRepairBackend(OpenAICompatibleChallengerClient(api), MAX_TOKENS)
    rows = []
    for prepared_item in prepared:
        base = prepared_item["base"]
        task = _run_single_paid_request(
            run_root=args.run_root, ledger=ledger, backend=backend,
            prompt=prepared_item["prompt"], qa_id=base["qa_id"],
            view=_minimal_view(prepared_item["evidence"]),
            answer=base["correct_answer_text"],
            validator=lambda raw, answer=base["correct_answer_text"]:
                validate_and_render_balanced_factorial(
                    raw, correct_answer_text=str(answer)))
        row: dict[str, Any] = {
            "qa_id": base["qa_id"], "attempt_index": 0,
            "api_task": task, "balanced_factorial_valid": task["accepted"],
            "minimal_decision": "not_run", "final_decision": task["final_decision"],
        }
        if task["accepted"]:
            placement = deterministic_balanced_factorial_placement(base["qa_id"])
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
            })
        rows.append(row)
    usage = {**_usage(args.run_root), "normal_request_cap": 2,
             "format_only_retry_cap": 0, "physical_paid_api_request_cap": 2,
             "ordinary_questioner_generations": 0, "realizer_generations": 0,
             "verifier_generations": 0, "training_runs": 0,
             "semantic_repair_generations": 0}
    atomic_json(args.run_root / "reports/api_usage.json", usage)
    report = {
        "canary_version": BALANCED_FACTORIAL_CANARY_VERSION,
        "source_qa_count": len(rows),
        "balanced_factorial_valid_count": sum(
            row["balanced_factorial_valid"] for row in rows),
        "minimal_accepted_count": sum(row["minimal_decision"] == "accept" for row in rows),
        "attempts": rows, "api_usage": usage,
        "solver_evaluated_in_this_root": False,
        "ordinary_questioner_generations": 0, "realizer_generations": 0,
        "verifier_generations": 0, "training_runs": 0,
        "semantic_repair_generations": 0,
    }
    atomic_json(args.run_root / "reports/balanced_factorial_distractor_canary.json", report)
    atomic_text(args.run_root / "review/options.md", _review(source, rows))
    source_tree_after = _tree_sha256(args.source_root.resolve())
    provenance = write_post_run_provenance(args.run_root, PROJECT, manifest)
    checks = {
        "balanced_factorial_source_sha_match": sha256_file(
            MODULE_ROOT / "balanced_factorial_distractor_canary.py") == manifest[
                "balanced_factorial_source_sha256"],
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
