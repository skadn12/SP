"""Regenerate semantic-neighbor distractors for two frozen QAs and evaluate Stage2."""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from transformers import AutoTokenizer

from .backends import QuestionerRequest
from .contrastive_distractor_canary import audit_frozen_source_qa
from .core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from .input_adapter import read_json
from .ledger import RequestLedger
from .lineage_distractor_writer import validate_distractor_object
from .live_backends import (
    LiveSemanticRepairBackend,
    OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from .run_canary import (
    create_source_snapshot,
    project_diff_sha256,
    write_post_run_provenance,
)
from .run_contrastive_distractor_canary import (
    _base_qa,
    _solver_summary,
    _tree_sha256,
)
from .run_cross_dialogue_calibration import _base_manifest, _run_solver, _solver_config
from .run_rolling_outline_30k_canary import (
    CHALLENGER_MODEL,
    MODEL_CONTEXT_LIMIT,
    SOLVER,
    TOKENIZER,
    _usage,
)
from .semantic_neighbor_distractor_canary import (
    SEMANTIC_NEIGHBOR_CANARY_VERSION,
    SEMANTIC_NEIGHBOR_PLACEMENT_VERSION,
    SEMANTIC_NEIGHBOR_PROMPT_VERSION,
    SEMANTIC_NEIGHBOR_SCHEMA_VERSION,
    build_semantic_neighbor_prompt,
    deterministic_semantic_neighbor_placement,
    materialize_contrastive_candidate,
    option_length_diagnostics,
    semantic_neighbor_prompt_sha256,
    semantic_neighbor_schema_sha256,
)
from .vllm_backends import VLLMSolverBackend


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_contrastive_distractor_canary_v1"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_semantic_neighbor_distractor_canary_v1"
)
TARGET_QA_IDS = (
    "johns_motivation_for_politics",
    "marias_volunteering_inspiration",
)
NORMAL_REQUEST_CAP = 2
PHYSICAL_REQUEST_CAP = 2
FORMAT_RETRY_CAP = 0
DISTRACTOR_MAX_TOKENS = 768


def _attest_source(root: Path) -> dict[str, Any]:
    paths = {
        "manifest": root / "config/run_manifest.json",
        "provenance": root / "reports/post_run_provenance.json",
        "report": root / "reports/contrastive_distractor_canary.json",
        "context": root / "frozen/full_dialogue.json",
        "catalog": root / "frozen/questioner_span_catalog.json",
    }
    for key, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"source_file_missing:{key}:{path}")
    provenance = read_json(paths["provenance"])
    if (provenance.get("authoritative_release_provenance") is not True or
            provenance.get("provenance_mixed") is not False or
            provenance.get("run_manifest_sha256") != sha256_file(paths["manifest"])):
        raise RuntimeError("source_provenance_not_authoritative")
    report = read_json(paths["report"])
    attempts = report.get("attempts") or []
    source_items = []
    for qa_id in TARGET_QA_IDS:
        base_path = root / f"frozen/base_qas/{qa_id}.json"
        if not base_path.is_file():
            raise RuntimeError(f"source_base_qa_missing:{qa_id}")
        base_wrapper = read_json(base_path)
        rows = [row for row in attempts if row.get("qa_id") == qa_id]
        if len(rows) != 2 or [row.get("attempt_index") for row in rows] != [0, 1]:
            raise RuntimeError(f"source_attempt_schedule_invalid:{qa_id}")
        previous = rows[-1]
        if (previous.get("final_decision") != "bad_ground" or
                previous.get("grounding_diagnosis") !=
                "bad_ground_semantic_leakage"):
            raise RuntimeError(f"source_diagnosis_invalid:{qa_id}")
        old_distractors = ((previous.get("api_task") or {}).get(
            "accepted_validation") or {}).get("distractors") or []
        if len(old_distractors) != 3:
            raise RuntimeError(f"source_previous_distractors_invalid:{qa_id}")
        source_items.append({
            "qa_id": qa_id,
            "base_path": str(base_path.resolve()),
            "base_sha256": sha256_file(base_path),
            "raw_qa": base_wrapper["raw_qa"],
            "materialized_qa": base_wrapper["materialized_qa"],
            "previous_distractors": list(old_distractors),
            "previous_materialized_mcq_sha256": previous.get(
                "materialized_mcq_sha256"),
            "previous_solver_report_sha256": previous.get("solver_report_sha256"),
            "previous_grounding_diagnosis": previous.get("grounding_diagnosis"),
        })
    return {
        "source_root": str(root.resolve()),
        "source_tree_sha256_before": _tree_sha256(root),
        "source_manifest_sha256": sha256_file(paths["manifest"]),
        "source_provenance_sha256": sha256_file(paths["provenance"]),
        "source_report_sha256": sha256_file(paths["report"]),
        "source_context_sha256": sha256_file(paths["context"]),
        "source_catalog_sha256": sha256_file(paths["catalog"]),
        "context": read_json(paths["context"]),
        "catalog": read_json(paths["catalog"]),
        "source_items": source_items,
    }


def _run_single_paid_request(
    *, run_root: Path, ledger: RequestLedger, backend: LiveSemanticRepairBackend,
    prompt: str, qa_id: str, view: Mapping[str, Any], answer: str,
    validator: Callable[[Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    request_id = f"semantic-neighbor-distractor-{qa_id}-a00"
    prompt_sha = sha256_bytes(prompt.encode())
    atomic_json(run_root / f"raw/prompts/{request_id}.json", {
        "request_id": request_id,
        "attempt_index": 0,
        "semantic_prompt_sha256": prompt_sha,
        "format_retry_allowed": False,
        "prompt": prompt,
    })
    request = QuestionerRequest(
        request_id, 0, ("A", "B", "C"), prompt, prompt_sha,
        tuple(turn["turn_id"] for turn in view["combined_dialogue"]),
        tuple(sorted({turn["session_id"] for turn in view["combined_dialogue"]})),
        dict(view["span_catalog"]), tuple(),
    )
    response = ledger.invoke(
        request_id=request_id, intent="semantic_neighbor_distractor_writer",
        attempt=1, prompt_sha256=prompt_sha,
        call=lambda: backend.generate_batch(request), budget_class="paid_api",
        budget_limit=PHYSICAL_REQUEST_CAP, total_budget_limit=PHYSICAL_REQUEST_CAP,
        request_metadata={
            "stage": "semantic_neighbor_distractor_writer",
            "qa_id": qa_id, "attempt_index": 0,
            "format_retry_allowed": False,
            "source_view_sha256": view["view_sha256"],
        },
        backend_identity={
            "backend": "openai_compatible_single_json_object",
            "model_identifier": backend.client.config.model,
            "base_url": backend.client.config.base_url,
            "temperature": backend.client.config.temperature,
            "max_tokens": backend.max_tokens,
            "response_format": "json_object",
        },
    )
    try:
        parsed = json.loads(response.raw_text)
        validation = (validator(parsed) if validator is not None else
            validate_distractor_object(parsed, correct_answer_text=answer))
        parse_status = "json_parsed"
    except json.JSONDecodeError:
        parsed = None
        parse_status = "json_invalid"
        validation = {
            "decision": "distractor_json_invalid",
            "reasons": ["distractor_json_invalid"],
        }
    accepted = str(validation.get("decision", "")).endswith("_accept")
    decision = str(validation.get("decision"))
    ledger.record_decision(request_id, decision)
    state = ledger.read(request_id) or {}
    raw_record = {
        "request_id": request_id, "qa_id": qa_id, "attempt_index": 0,
        "semantic_prompt_sha256": prompt_sha, "parse_status": parse_status,
        "validation": validation, "decision": decision,
        "raw_response": response.raw_text,
        "raw_response_sha256": state.get("raw_response_sha256"),
        "response_metadata": dict(response.metadata),
        "format_retry_used": False,
    }
    raw_path = run_root / f"raw/distractor_writer/{request_id}.json"
    atomic_json(raw_path, raw_record)
    return {
        "request_id": request_id,
        "semantic_prompt_sha256": prompt_sha,
        "physical_request_count": 1,
        "format_retry_used": False,
        "accepted": accepted,
        "final_decision": decision,
        "accepted_validation": validation if accepted else None,
        "raw_record_path": str(raw_path.relative_to(run_root)),
        "raw_record_sha256": sha256_file(raw_path),
        "response_metadata": dict(response.metadata),
    }


def _review_markdown(source: Mapping[str, Any], rows: list[dict[str, Any]]) -> str:
    by_id = {row["qa_id"]: row for row in rows}
    lines = ["# Semantic-Neighbor Distractor Canary", ""]
    for item in source["source_items"]:
        qa_id = item["qa_id"]
        row = by_id[qa_id]
        lines.extend([
            f"## {qa_id}", "", f"Question: {item['raw_qa']['question']}", "",
            f"Correct answer: {item['raw_qa']['correct_answer_text']}", "",
            "Previous failed distractors:", "",
        ])
        lines.extend(f"- {value}" for value in item["previous_distractors"])
        lines.extend(["", "Semantic-neighbor options:", ""])
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
    api = replace(load_challenger_config(args.api_config),
                  model=CHALLENGER_MODEL, temperature=0.25,
                  enable_thinking=False, timeout_seconds=600)
    solver_native = read_json(SOLVER / "config.json")
    if int(solver_native.get("max_position_embeddings") or 0) < MODEL_CONTEXT_LIMIT:
        raise RuntimeError("solver_native_context_too_short")

    prompt_preflights = []
    source_audits = []
    rendered_prompts: dict[str, str] = {}
    for item in source["source_items"]:
        base = _base_qa(item["raw_qa"])
        audit = audit_frozen_source_qa(raw_qa=base,
            materialized_qa=item["materialized_qa"], span_catalog=source["catalog"])
        if audit["decision"] != "source_qa_accept":
            raise RuntimeError(f"source_qa_audit_failed:{base['qa_id']}")
        source_audits.append(audit)
        prompt = build_semantic_neighbor_prompt(
            frozen_dialogue=source["context"]["combined_dialogue"],
            base_qa=base, previous_distractors=item["previous_distractors"])
        rendered_prompts[base["qa_id"]] = prompt
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        prompt_preflights.append({
            "qa_id": base["qa_id"], "rendered_prompt_tokens": prompt_tokens,
            "reserved_generation_tokens": DISTRACTOR_MAX_TOKENS,
            "total_tokens": prompt_tokens + DISTRACTOR_MAX_TOKENS,
            "backend_context_limit": MODEL_CONTEXT_LIMIT,
            "decision": "pass" if prompt_tokens + DISTRACTOR_MAX_TOKENS <=
                MODEL_CONTEXT_LIMIT else "reject_context_overflow",
            "truncated": False,
        })
    if any(row["decision"] != "pass" for row in prompt_preflights):
        raise RuntimeError("semantic_neighbor_prompt_context_overflow")
    if args.dry_run:
        print(json.dumps({
            "decision": "dry_run_pass", "source_audits": source_audits,
            "prompt_preflights": prompt_preflights,
            "normal_request_cap": NORMAL_REQUEST_CAP,
            "physical_request_cap": PHYSICAL_REQUEST_CAP,
            "format_retry_cap": FORMAT_RETRY_CAP,
            "challenger_model": api.model, "solver_model": str(SOLVER),
        }, ensure_ascii=False, indent=2))
        return 0

    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    project_diff = project_diff_sha256(PROJECT)
    manifest = {
        **_base_manifest(run_id=args.run_root.name,
            mode=SEMANTIC_NEIGHBOR_CANARY_VERSION,
            snapshot=snapshot, project_diff=project_diff),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "canary_version": SEMANTIC_NEIGHBOR_CANARY_VERSION,
        "source_root": source["source_root"],
        "source_tree_sha256_before": source["source_tree_sha256_before"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "source_provenance_sha256": source["source_provenance_sha256"],
        "source_report_sha256": source["source_report_sha256"],
        "source_context_sha256": source["source_context_sha256"],
        "source_catalog_sha256": source["source_catalog_sha256"],
        "source_target_qa_ids": list(TARGET_QA_IDS),
        "semantic_neighbor_prompt_version": SEMANTIC_NEIGHBOR_PROMPT_VERSION,
        "semantic_neighbor_prompt_sha256": semantic_neighbor_prompt_sha256(),
        "semantic_neighbor_schema_version": SEMANTIC_NEIGHBOR_SCHEMA_VERSION,
        "semantic_neighbor_schema_sha256": semantic_neighbor_schema_sha256(),
        "answer_placement_version": SEMANTIC_NEIGHBOR_PLACEMENT_VERSION,
        "semantic_neighbor_source_sha256": sha256_file(
            MODULE_ROOT / "semantic_neighbor_distractor_canary.py"),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "live_backend_source_sha256": sha256_file(MODULE_ROOT / "live_backends.py"),
        "vllm_backend_source_sha256": sha256_file(MODULE_ROOT / "vllm_backends.py"),
        "challenger_model_identifier": api.model,
        "challenger_temperature": api.temperature,
        "challenger_enable_thinking": api.enable_thinking,
        "distractor_max_tokens": DISTRACTOR_MAX_TOKENS,
        "solver_model": str(SOLVER), "solver_tokenizer": str(TOKENIZER),
        "solver_max_model_len": MODEL_CONTEXT_LIMIT,
        "budget": {
            "normal_requests": NORMAL_REQUEST_CAP,
            "format_only_retries": FORMAT_RETRY_CAP,
            "physical_paid_api_requests": PHYSICAL_REQUEST_CAP,
            "ordinary_questioner": 0, "realizer": 0, "verifier": 0,
            "training": 0, "semantic_repair": 0,
        },
        "historical_runs_modified": False,
    }
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    atomic_json(args.run_root / "config/resolved_canary_config.json", {
        "mode": SEMANTIC_NEIGHBOR_CANARY_VERSION,
        "source_root": source["source_root"], "challenger_model": api.model,
        "temperature": api.temperature, "enable_thinking": api.enable_thinking,
        "normal_request_cap": NORMAL_REQUEST_CAP,
        "format_retry_cap": FORMAT_RETRY_CAP,
        "physical_request_cap": PHYSICAL_REQUEST_CAP,
        "solver_model": str(SOLVER), "solver_max_model_len": MODEL_CONTEXT_LIMIT,
    })
    atomic_json(args.run_root / "frozen/source_attestation.json", {
        key: value for key, value in source.items()
        if key not in {"context", "catalog", "source_items"}
    })
    atomic_json(args.run_root / "frozen/full_dialogue.json", source["context"])
    atomic_json(args.run_root / "frozen/questioner_span_catalog.json", source["catalog"])
    atomic_json(args.run_root / "reports/source_qa_evidence_audit.json", {
        "decision": "pass", "items": source_audits,
        "ordinary_questioner_generation_count": 0, "manual_qa_rewrite_count": 0,
    })
    atomic_json(args.run_root / "reports/preflight.json", {
        "decision": "pass", "prompt_preflights": prompt_preflights,
        "api_budget_preflight": "pass", "source_attestation": "pass",
        "solver_path_preflight": "pass", "tokenizer_path_preflight": "pass",
    })
    for item in source["source_items"]:
        atomic_json(args.run_root / f"frozen/base_qas/{item['qa_id']}.json", item)

    ledger = RequestLedger(args.run_root,
        sha256_file(args.run_root / "config/run_manifest.json"), "live")
    backend = LiveSemanticRepairBackend(
        OpenAICompatibleChallengerClient(api), DISTRACTOR_MAX_TOKENS)
    view = {
        "combined_dialogue": source["context"]["combined_dialogue"],
        "span_catalog": source["catalog"],
        "view_sha256": sha256_bytes(canonical_json({
            "combined_dialogue": source["context"]["combined_dialogue"],
            "span_catalog": source["catalog"],
        }).encode()),
    }
    rows: list[dict[str, Any]] = []
    for item in source["source_items"]:
        base = _base_qa(item["raw_qa"])
        task = _run_single_paid_request(
            run_root=args.run_root, ledger=ledger, backend=backend,
            prompt=rendered_prompts[base["qa_id"]], qa_id=base["qa_id"],
            view=view, answer=base["correct_answer_text"])
        row: dict[str, Any] = {
            "qa_id": base["qa_id"], "attempt_index": 0,
            "api_task": task, "distractor_valid": task["accepted"],
            "minimal_decision": "not_run", "solver_evaluated": False,
            "final_decision": task["final_decision"],
            "grounding_diagnosis": None,
        }
        if task["accepted"]:
            placement = deterministic_semantic_neighbor_placement(base["qa_id"])
            candidate = materialize_contrastive_candidate(
                base_qa=base,
                distractors=task["accepted_validation"]["distractors"],
                target_letter=placement["target_letter"],
                span_catalog=source["catalog"])
            path = args.run_root / (
                f"frozen/materialized_mcqs/{base['qa_id']}_attempt_0.json")
            atomic_json(path, candidate)
            row.update({
                "placement": placement,
                "materialized_mcq_path": str(path.relative_to(args.run_root)),
                "materialized_mcq_sha256": sha256_file(path),
                "materialized_qa": candidate["materialized_qa"],
                "minimal_decision": candidate["admission"][
                    "spell_style_minimal_v1"]["decision"],
                "minimal_reasons": candidate["admission"][
                    "spell_style_minimal_v1"]["reasons"],
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
        solver_config = _solver_config(run_root=solver_root)
        solver_config.update({
            "mode": "local_stage2_solver_semantic_neighbor_canary",
            "solver_effective_max_model_len": MODEL_CONTEXT_LIMIT,
            "max_local_solver_generations": 80,
        })
        solver_manifest = {
            **_base_manifest(run_id=f"{args.run_root.name}-{qa_id}",
                mode="local_stage2_solver_semantic_neighbor_canary",
                snapshot=snapshot, project_diff=project_diff),
            "canary_version": SEMANTIC_NEIGHBOR_CANARY_VERSION,
            "qa_id": qa_id, "attempt_index": 0,
            "solver_model": str(SOLVER),
            "solver_max_model_len": MODEL_CONTEXT_LIMIT,
            "source_canary_manifest_sha256": sha256_file(
                args.run_root / "config/run_manifest.json"),
            "source_materialized_mcq_sha256": row["materialized_mcq_sha256"],
        }
        atomic_json(solver_root / "config/resolved_canary_config.json", solver_config)
        atomic_json(solver_root / "config/run_manifest.json", solver_manifest)
        solver_report = _run_solver(
            root=solver_root, manifest=solver_manifest, config=solver_config,
            raw_batch=[candidate["raw_v8_qa"]], admission=candidate["admission"],
            materialized=candidate["materialized_qa"],
            artifacts={"context": source["context"], "catalog": source["catalog"],
                       "contracts": candidate["reasoning_contracts"]},
            tokenizer=tokenizer, backend=solver_backend)
        write_post_run_provenance(solver_root, PROJECT, solver_manifest)
        summary = _solver_summary(solver_report)
        row.update({
            "solver_evaluated": True,
            "solver_root": str(solver_root.relative_to(args.run_root)),
            "solver_report_sha256": sha256_file(
                solver_root / "reports/cross_dialogue_solver_audit.json"),
            "solver": summary,
            "final_decision": summary["final_decision"],
            "grounding_diagnosis": summary["no_context_permutation_diagnostic"],
        })

    usage = {
        **_usage(args.run_root),
        "normal_request_cap": NORMAL_REQUEST_CAP,
        "format_only_retry_cap": FORMAT_RETRY_CAP,
        "physical_paid_api_request_cap": PHYSICAL_REQUEST_CAP,
        "ordinary_questioner_generations": 0, "realizer_generations": 0,
        "verifier_generations": 0, "training_runs": 0,
        "semantic_repair_generations": 0,
    }
    atomic_json(args.run_root / "reports/api_usage.json", usage)
    decisions = collections.Counter(str(row.get("final_decision")) for row in rows)
    report = {
        "canary_version": SEMANTIC_NEIGHBOR_CANARY_VERSION,
        "source_qa_count": len(source["source_items"]),
        "source_qa_ids": list(TARGET_QA_IDS),
        "distractor_attempt_count": len(rows),
        "distractor_valid_count": sum(row["distractor_valid"] for row in rows),
        "minimal_accepted_count": sum(row["minimal_decision"] == "accept" for row in rows),
        "solver_evaluated_count": sum(row["solver_evaluated"] for row in rows),
        "final_decisions": dict(sorted(decisions.items())),
        "attempts": rows, "api_usage": usage,
        "ordinary_questioner_generations": 0, "realizer_generations": 0,
        "verifier_generations": 0, "training_runs": 0,
        "semantic_repair_generations": 0,
    }
    atomic_json(args.run_root / "reports/semantic_neighbor_distractor_canary.json", report)
    atomic_json(args.run_root / "reports/solver_summary.json", {
        "solver_model": str(SOLVER), "items": rows})
    atomic_text(args.run_root / "review/options_and_results.md",
                _review_markdown(source, rows))

    source_tree_after = _tree_sha256(args.source_root.resolve())
    provenance = write_post_run_provenance(args.run_root, PROJECT, manifest)
    checks = {
        "semantic_neighbor_source_sha_match": sha256_file(
            MODULE_ROOT / "semantic_neighbor_distractor_canary.py") == manifest[
                "semantic_neighbor_source_sha256"],
        "runner_source_sha_match": sha256_file(Path(__file__)) == manifest[
            "runner_source_sha256"],
        "historical_source_tree_unchanged": source_tree_after == source[
            "source_tree_sha256_before"],
        "physical_api_budget_respected": usage["total_paid_api_requests"] <= 2,
        "format_retry_zero": usage.get("total_paid_api_requests") == len(rows),
        "ordinary_questioner_zero": True, "realizer_zero": True,
        "verifier_zero": True, "training_zero": True,
        "semantic_repair_zero": True,
    }
    provenance.update({
        "source_tree_sha256_before": source["source_tree_sha256_before"],
        "source_tree_sha256_after": source_tree_after, **checks,
    })
    provenance["authoritative_release_provenance"] = (
        provenance.get("authoritative_release_provenance") is True and
        all(checks.values()))
    provenance["provenance_mixed"] = not provenance[
        "authoritative_release_provenance"]
    atomic_json(args.run_root / "reports/post_run_provenance.json", provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
