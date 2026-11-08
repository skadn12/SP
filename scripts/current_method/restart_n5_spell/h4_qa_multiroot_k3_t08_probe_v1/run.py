"""Runner for the isolated H4 multi-root K=3/T=0.8 probe.

The command is intentionally staged:

``inventory`` freezes eligible frozen expansion roots without API calls;
``smoke`` generates fresh QA for one root and evaluates a small type-balanced
subset; ``batch`` resumes that same run and evaluates every valid candidate on
the selected roots.  No historical candidate pool is imported.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from ..vllm_backends import VLLMSolverBackend
from .checkpoint import read_checkpoint, write_checkpoint
from .diagnostics import MAIN_CATEGORIES
from .generation import GENERATOR_VERSION, generate_root_qa
from .reporting import summarize_batch, summarize_root
from .root_inventory import clean_named_view, inventory_sources, load_expanded
from .solver_adapter import (
    ROLLOUTS, SINGLE_TEMPERATURE, SOLVER_MAX_MODEL_LEN,
    SOLVER_MAX_TOKENS, STRUCTURED_TEMPERATURE, evaluate_candidates,
)


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_RUN_ROOT = RUN_BASE / "h4_qa_multiroot_k3_t08_probe_v1_live_1"
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_SOLVER_MODEL = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "05_stage2_repair_topup_from_stage1/training/models/merged"
)
PROBE_VERSION = "h4_qa_multiroot_k3_t08_probe_v1"


def _load(path: Path) -> Any:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value


def _load_object(path: Path) -> dict[str, Any]:
    value = _load(path)
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _write(path: Path, value: Any) -> None:
    atomic_json(path, value)


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text.strip("._") or "root"


def _candidate_file(root_dir: Path, candidate_id: str) -> Path:
    digest = sha256_bytes(candidate_id.encode("utf-8"))[:20]
    return root_dir / "solver_results" / f"{digest}.json"


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(dict(payload)).encode("utf-8"))


def inventory_phase(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_root.exists():
        raise RuntimeError(f"run_root_already_exists:{args.run_root}")
    rows = inventory_sources(tuple(args.root_source) if args.root_source else None)
    eligible = [row for row in rows if row.get("eligible")]
    if args.max_roots <= 0:
        raise ValueError("max_roots_must_be_positive")
    selected = eligible[: int(args.max_roots)]
    if not selected:
        raise RuntimeError("no_eligible_roots")
    payload = {
        "manifest_version": "h4_k3_batch_manifest_v1",
        "probe_version": PROBE_VERSION,
        "generation_version": GENERATOR_VERSION,
        "run_root": str(args.run_root),
        "selected_root_ids": [str(row["root_id"]) for row in selected],
        "root_count": len(selected),
        "roots": selected,
        "solver": {
            "rollouts_per_variant": ROLLOUTS,
            "single_temperature": SINGLE_TEMPERATURE,
            "structured_temperature": STRUCTURED_TEMPERATURE,
            "structured_question_types": ["multi_select", "ordering"],
            "max_tokens": SOLVER_MAX_TOKENS,
            "max_model_len": SOLVER_MAX_MODEL_LEN,
            "context_modes": ["no_context", "full_context"],
            "position_variants": {
                "single_choice": 4,
                "multi_select": {"min": 4, "max": 6, "source": "candidate"},
                "ordering": {"min": 4, "max": 6, "source": "candidate"},
            },
        },
        "seed_schedule": {
            "version": "h4_k3_paired_seed_v1",
            "inputs": ["root_id", "candidate_id", "slot_id"],
            "excludes": ["context_mode", "display_position"],
        },
        "fresh_qa_policy": {
            "candidate_source": "fresh_generated",
            "historical_candidate_pool_imported": False,
            "one_generation_then_immediate_screen": True,
        },
        "semantic_grounding_judgment": False,
        "mainline_untouched": True,
        "argv": list(getattr(args, "_argv", [])),
    }
    manifest_sha = _manifest_hash(payload)
    manifest = {**payload, "manifest_sha256": manifest_sha}
    args.run_root.mkdir(parents=True, exist_ok=False)
    _write(args.run_root / "manifest.json", manifest)
    _write(args.run_root / "root_inventory.json", {
        "inventory_version": "h4_k3_root_inventory_v1",
        "all_rows": rows,
        "selected_rows": selected,
        "selected_root_ids": manifest["selected_root_ids"],
        "api_calls": 0,
        "solver_calls": 0,
    })
    atomic_text(args.run_root / "summary.md", "# H4 K=3/T=0.8 Probe\n\nInventory frozen; no API or Solver calls were made.\n")
    return {
        "status": "inventory_ready",
        "run_root": str(args.run_root),
        "root_count": len(selected),
        "selected_root_ids": manifest["selected_root_ids"],
        "api_calls": 0,
        "solver_calls": 0,
    }


def _manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_object(args.run_root / "manifest.json")
    if manifest.get("probe_version") != PROBE_VERSION:
        raise RuntimeError("probe_manifest_version_mismatch")
    return manifest


def _selected_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in manifest.get("roots") or [] if row.get("eligible")]
    selected_ids = [str(x) for x in manifest.get("selected_root_ids") or []]
    by_id = {str(row.get("root_id")): row for row in rows}
    missing = [root_id for root_id in selected_ids if root_id not in by_id]
    if missing:
        raise RuntimeError(f"manifest_root_missing:{missing}")
    return [by_id[root_id] for root_id in selected_ids]


def _root_dir(run_root: Path, root_id: str) -> Path:
    return run_root / "roots" / _safe_name(root_id)


def _prepare_root(run_root: Path, entry: Mapping[str, Any], *, manifest_sha: str) -> tuple[Path, dict[str, Any]]:
    root_id = str(entry["root_id"])
    root_dir = _root_dir(run_root, root_id)
    root_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(str(entry["source_path"]))
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha256 = sha256_file(source_path)
    expected_source_sha256 = str(entry.get("source_sha256") or "")
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        raise RuntimeError(f"source_hash_drift:{root_id}")
    source = load_expanded(source_path)
    if str(source.get("root_id") or "") != root_id:
        raise RuntimeError(f"root_id_drift:{root_id}")
    cleaned, mapping = clean_named_view(source)
    source_metadata = {
        "root_id": root_id,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "source_sha256_from_inventory": expected_source_sha256 or None,
        "cleaned_dialogue_sha256": entry.get("cleaned_dialogue_sha256"),
        "speaker_name_mapping": mapping,
        "turn_count": len(cleaned.get("turns") or []),
        "source_is_frozen": True,
        "cleaning_version": "h4_named_speaker_view_v1",
    }
    _write(root_dir / "source_metadata.json", source_metadata)
    _write(root_dir / "expanded_dialogue_clean.json", cleaned)
    checkpoint = root_dir / "checkpoint.json"
    if read_checkpoint(checkpoint) is None:
        write_checkpoint(checkpoint, root_id=root_id, stage="ROOT_STARTED",
                         manifest_sha256=manifest_sha)
        write_checkpoint(checkpoint, root_id=root_id, stage="EXPANSION_READY",
                         manifest_sha256=manifest_sha,
                         cleaned_dialogue_sha256=source_metadata["cleaned_dialogue_sha256"])
    return root_dir, cleaned


def _client(args: argparse.Namespace) -> OpenAICompatibleChallengerClient:
    config = load_challenger_config(args.api_config)
    return OpenAICompatibleChallengerClient(replace(
        config, temperature=0.1, enable_thinking=False,
        timeout_seconds=max(180, int(config.timeout_seconds))))


def _generation_coverage(root_dir: Path) -> dict[str, Any]:
    records_path = root_dir / "candidate_generation.json"
    contracts_path = root_dir / "reasoning_contracts.json"
    records = _load(records_path) if records_path.is_file() else []
    contracts = _load(contracts_path) if contracts_path.is_file() else []
    final: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, Mapping):
            continue
        key = (str(record.get("intent_id") or ""),
               str(record.get("generation_source") or ""))
        final[key] = record
    expected = len(contracts) * 2 if isinstance(contracts, list) else 0
    accepted: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for (intent_id, source), record in sorted(final.items()):
        minimum = ((record.get("validation") or {}).get("minimum_acceptance") or {})
        row = {
            "intent_id": intent_id,
            "generation_source": source,
            "request_id": record.get("request_id"),
            "repair_of": record.get("repair_of"),
            "valid_counts": minimum.get("valid_counts") or {},
            "minimum_counts": minimum.get("minimum_counts") or {},
            "missing_minimums": minimum.get("missing_minimums") or {},
        }
        (accepted if minimum.get("accepted") else failed).append(row)
    return {
        "expected_intent_source_batches": expected,
        "observed_intent_source_batches": len(final),
        "accepted_intent_source_batches": len(accepted),
        "failed_intent_source_batches": len(failed),
        "minimum_coverage_rate": len(accepted) / expected if expected else None,
        "minimum_coverage_complete": bool(expected and len(accepted) == expected),
        "accepted_batches": accepted,
        "failed_batches": failed,
        "policy": "at_least_2_single_choice_1_multi_select_1_ordering_per_intent_source",
    }


def _ensure_qa(*, root_dir: Path, expanded: Mapping[str, Any], root_id: str,
               args: argparse.Namespace, manifest_sha: str,
               client: OpenAICompatibleChallengerClient) -> dict[str, Any]:
    candidates_path = root_dir / "candidates_valid.json"
    checkpoint_path = root_dir / "checkpoint.json"
    existing = read_checkpoint(checkpoint_path) or {}
    existing_candidates = _load(candidates_path) if candidates_path.is_file() else []
    current_policy_ready = (
        isinstance(existing_candidates, list)
        and bool(existing_candidates)
        and any(str(row.get("generation_source")) == "free_exploration"
                for row in existing_candidates if isinstance(row, Mapping))
    )
    if candidates_path.is_file() and current_policy_ready and existing.get("stage") in {
        "QA_READY", "QA_VALIDATED", "NO_CONTEXT_STARTED", "NO_CONTEXT_COMPLETED",
        "FULL_CONTEXT_STARTED", "FULL_CONTEXT_COMPLETED", "DIAGNOSTICS_COMPLETED",
        "REPORT_COMPLETED",
    }:
        candidates = existing_candidates
        return {
            "status": "ready_reused_within_current_run",
            "root_id": root_id,
            "valid_candidate_count": len(candidates) if isinstance(candidates, list) else 0,
            "candidate_count": len(_load(root_dir / "candidate_pool.json"))
            if (root_dir / "candidate_pool.json").is_file() else None,
            "generation_source": "fresh_generated_current_run",
            "generation_coverage": _generation_coverage(root_dir),
        }
    result = generate_root_qa(
        expanded=expanded, root_dir=root_dir, root_id=root_id,
        api_config=args.api_config, ledger=RequestLedger(
            root_dir, manifest_sha, "live" if args.live else "mock"), client=client,
    )
    result["generation_coverage"] = _generation_coverage(root_dir)
    _write(root_dir / "qa_generation_report.json", result)
    checkpoint_extra = {key: value for key, value in result.items()
                        if key not in {"root_id", "manifest_sha256"}}
    if result.get("status") == "ready":
        write_checkpoint(checkpoint_path, root_id=root_id, stage="QA_READY",
                         manifest_sha256=manifest_sha, **checkpoint_extra)
        write_checkpoint(checkpoint_path, root_id=root_id, stage="QA_VALIDATED",
                         manifest_sha256=manifest_sha, **checkpoint_extra)
    else:
        write_checkpoint(checkpoint_path, root_id=root_id, stage="QA_VALIDATED",
                         manifest_sha256=manifest_sha, **checkpoint_extra)
    return result


def _read_candidates(root_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = _load(root_dir / "candidates_valid.json")
    validity = _load_object(root_dir / "validity_audit.json")
    if not isinstance(candidates, list):
        raise ValueError("candidates_valid_must_be_list")
    return [dict(row) for row in candidates], validity


def _smoke_selection(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit < 3:
        raise ValueError("smoke_candidate_limit_must_cover_three_question_types")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for qtype in ("single_choice", "multi_select", "ordering"):
        row = next((row for row in candidates if row.get("question_type") == qtype), None)
        if row is not None:
            selected.append(row)
            seen.add(str(row.get("candidate_id")))
    if len(selected) < 3:
        raise RuntimeError("smoke_missing_question_type")
    for row in candidates:
        if len(selected) >= limit:
            break
        if str(row.get("candidate_id")) not in seen:
            selected.append(row)
            seen.add(str(row.get("candidate_id")))
    return selected


def _existing_results(root_dir: Path) -> list[dict[str, Any]]:
    directory = root_dir / "solver_results"
    if not directory.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            value = _load(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("candidate_id"):
            rows.append(value)
    return rows


def _ledger_accounting(ledger: RequestLedger) -> dict[str, Any]:
    states = ledger.completed()
    by_class: dict[str, int] = {}
    for state in states:
        name = str(state.get("budget_class") or "unknown")
        by_class[name] = by_class.get(name, 0) + 1
    return {
        "completed_requests_by_budget_class": dict(sorted(by_class.items())),
        "challenger_api_generation_count": by_class.get("fresh_qa_generation", 0),
        "local_solver_generation_count": by_class.get("k3_solver", 0),
        "ambiguous_request_count": ledger.summary().get("ambiguous_request_count", 0),
        "note": "RequestLedger's generic live counter includes local Solver calls; use the budget-class counts above.",
    }


def _evaluate_root(*, root_dir: Path, expanded: Mapping[str, Any], root_id: str,
                   candidates: list[dict[str, Any]], validity: Mapping[str, Any],
                   args: argparse.Namespace, ledger: RequestLedger, tokenizer: Any,
                   backend: VLLMSolverBackend, candidate_limit: int | None = None) -> dict[str, Any]:
    selected = _smoke_selection(candidates, candidate_limit) if candidate_limit is not None else candidates
    existing = {str(row.get("candidate_id")): row for row in _existing_results(root_dir)}
    write_checkpoint(root_dir / "checkpoint.json", root_id=root_id, stage="NO_CONTEXT_STARTED",
                     manifest_sha256=args.manifest_sha)
    errors: list[str] = []
    for candidate in selected:
        candidate_id = str(candidate.get("candidate_id"))
        if candidate_id in existing:
            continue
        try:
            result_rows = evaluate_candidates(
                candidates=[candidate], expanded=expanded, root_id=root_id,
                run_root=root_dir, ledger=ledger, tokenizer=tokenizer,
                backend=backend,
            )
            if not result_rows:
                raise RuntimeError("solver_returned_no_candidate_result")
            result = result_rows[0]
            _write(_candidate_file(root_dir, candidate_id), result)
            existing[candidate_id] = result
        except Exception as error:
            errors.append(f"{candidate_id}:{type(error).__name__}:{error}")
    write_checkpoint(root_dir / "checkpoint.json", root_id=root_id, stage="NO_CONTEXT_COMPLETED",
                     manifest_sha256=args.manifest_sha, evaluated_candidate_count=len(existing))
    write_checkpoint(root_dir / "checkpoint.json", root_id=root_id, stage="FULL_CONTEXT_COMPLETED",
                     manifest_sha256=args.manifest_sha, evaluated_candidate_count=len(existing))
    result_list = [existing[key] for key in sorted(existing)]
    validity_rows = list(validity.get("rows") or [])
    root_report = summarize_root(
        root_id=root_id, candidates=candidates, validity_rows=validity_rows,
        solver_results=result_list,
    )
    _write(root_dir / "solver_results.json", {
        "probe_version": PROBE_VERSION,
        "root_id": root_id,
        "rollouts_per_variant": ROLLOUTS,
        "results": result_list,
        "errors": errors,
        "evaluated_candidate_count": len(result_list),
    })
    _write(root_dir / "difficulty.json", root_report)
    write_checkpoint(root_dir / "checkpoint.json", root_id=root_id, stage="DIAGNOSTICS_COMPLETED",
                     manifest_sha256=args.manifest_sha, evaluated_candidate_count=len(result_list),
                     solver_errors=errors)
    return {"report": root_report, "results": result_list, "errors": errors, "selected_count": len(selected)}


def _smoke_report(root_dir: Path, generation: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    results = list(evaluation.get("results") or [])
    qtypes = {str(row.get("question_type")) for row in results}
    complete = all(
        bool(((row.get("position_diagnostics") or {}).get("no_context") or {}).get("complete"))
        and bool(((row.get("position_diagnostics") or {}).get("full_context") or {}).get("complete"))
        for row in results
    ) and bool(results)
    rebinding = all(bool((row.get("gold_rebinding_audit") or {}).get("pass")) for row in results)
    passed = (
        generation.get("status") in {"ready", "ready_reused_within_current_run"}
        and qtypes == {"single_choice", "multi_select", "ordering"}
        and complete and rebinding and not evaluation.get("errors")
    )
    return {
        "probe_version": PROBE_VERSION,
        "root_id": generation.get("root_id"),
        "status": "pass" if passed else "fail",
        "smoke_gate_pass": passed,
        "candidate_count_evaluated": len(results),
        "question_types_evaluated": sorted(qtypes),
        "all_positions_complete": complete,
        "gold_rebinding_pass": rebinding,
        "errors": list(evaluation.get("errors") or []),
        "reason": None if passed else "smoke_gate_conditions_not_met",
        "note": "K=3 smoke is an execution gate, not a claim of stable difficulty.",
    }


def smoke_phase(args: argparse.Namespace) -> dict[str, Any]:
    if not args.live:
        raise RuntimeError("smoke_requires_live_to_generate_and_evaluate")
    if not args.run_root.is_dir():
        raise FileNotFoundError(args.run_root)
    manifest = _manifest(args)
    entries = _selected_entries(manifest)
    if not entries:
        raise RuntimeError("no_selected_roots")
    args.manifest_sha = str(manifest["manifest_sha256"])
    entry = entries[0]
    root_id = str(entry["root_id"])
    root_dir, expanded = _prepare_root(args.run_root, entry, manifest_sha=args.manifest_sha)
    client = _client(args)
    generation = _ensure_qa(
        root_dir=root_dir, expanded=expanded, root_id=root_id,
        args=args, manifest_sha=args.manifest_sha, client=client,
    )
    if generation.get("status") not in {"ready", "ready_reused_within_current_run"}:
        report = _smoke_report(root_dir, generation, {"results": [], "errors": ["qa_generation_incomplete"]})
        _write(root_dir / "smoke_report.json", report)
        _write(args.run_root / "smoke_report.json", report)
        return report
    candidates, validity = _read_candidates(root_dir)
    ledger = RequestLedger(root_dir, args.manifest_sha, "live" if args.live else "mock")
    tokenizer = None
    backend = None
    evaluation: dict[str, Any]
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        backend = VLLMSolverBackend(
            model_path=args.solver_model, tokenizer_path=args.tokenizer,
            max_model_len=SOLVER_MAX_MODEL_LEN, temperature=SINGLE_TEMPERATURE,
            top_p=0.9, max_tokens=SOLVER_MAX_TOKENS,
            gpu_memory_utilization=0.80,
        )
        evaluation = _evaluate_root(
            root_dir=root_dir, expanded=expanded, root_id=root_id,
            candidates=candidates, validity=validity, args=args, ledger=ledger,
            tokenizer=tokenizer, backend=backend,
            candidate_limit=args.smoke_candidate_limit,
        )
    except Exception as error:
        evaluation = {"results": _existing_results(root_dir), "errors": [f"{type(error).__name__}:{error}"]}
    finally:
        if backend is not None:
            backend.close()
    report = _smoke_report(root_dir, generation, evaluation)
    report["request_accounting"] = _ledger_accounting(ledger)
    _write(root_dir / "smoke_report.json", report)
    _write(args.run_root / "smoke_report.json", report)
    atomic_text(root_dir / "summary.md", "# H4 K=3/T=0.8 Smoke\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def batch_phase(args: argparse.Namespace) -> dict[str, Any]:
    if not args.live:
        raise RuntimeError("batch_requires_live_to_generate_and_evaluate")
    if not args.run_root.is_dir():
        raise FileNotFoundError(args.run_root)
    manifest = _manifest(args)
    smoke = _load_object(args.run_root / "smoke_report.json")
    if smoke.get("status") != "pass":
        raise RuntimeError("smoke_gate_not_passed")
    entries = _selected_entries(manifest)
    args.manifest_sha = str(manifest["manifest_sha256"])
    smoke_root_id = str(smoke.get("root_id") or "")
    smoke_root_dir = _root_dir(args.run_root, smoke_root_id)
    if smoke_root_id and smoke_root_dir.is_dir():
        smoke.pop("ledger_summary", None)
        smoke["request_accounting"] = _ledger_accounting(
            RequestLedger(smoke_root_dir, args.manifest_sha, "live"))
        _write(args.run_root / "smoke_report.json", smoke)
        _write(smoke_root_dir / "smoke_report.json", smoke)
    client = _client(args)
    tokenizer = None
    backend = None
    root_reports: list[dict[str, Any]] = []
    root_errors: list[dict[str, Any]] = []
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        backend = VLLMSolverBackend(
            model_path=args.solver_model, tokenizer_path=args.tokenizer,
            max_model_len=SOLVER_MAX_MODEL_LEN, temperature=SINGLE_TEMPERATURE,
            top_p=0.9, max_tokens=SOLVER_MAX_TOKENS,
            gpu_memory_utilization=0.80,
        )
        for entry in entries:
            root_id = str(entry["root_id"])
            root_dir, expanded = _prepare_root(args.run_root, entry, manifest_sha=args.manifest_sha)
            try:
                generation = _ensure_qa(
                    root_dir=root_dir, expanded=expanded, root_id=root_id,
                    args=args, manifest_sha=args.manifest_sha, client=client,
                )
                if generation.get("status") not in {"ready", "ready_reused_within_current_run"}:
                    root_errors.append({"root_id": root_id, "stage": "qa_generation", "generation": generation})
                    continue
                candidates, validity = _read_candidates(root_dir)
                ledger = RequestLedger(root_dir, args.manifest_sha, "live" if args.live else "mock")
                evaluation = _evaluate_root(
                    root_dir=root_dir, expanded=expanded, root_id=root_id,
                    candidates=candidates, validity=validity, args=args, ledger=ledger,
                    tokenizer=tokenizer, backend=backend,
                )
                report = dict(evaluation["report"])
                report.update({
                    "generation": generation,
                    "request_accounting": _ledger_accounting(ledger),
                    "solver_errors": evaluation.get("errors") or [],
                })
                _write(root_dir / "report.json", report)
                root_reports.append(report)
                checkpoint_report = {key: value for key, value in report.items()
                                     if key not in {"root_id", "manifest_sha256"}}
                write_checkpoint(root_dir / "checkpoint.json", root_id=root_id, stage="REPORT_COMPLETED",
                                 manifest_sha256=args.manifest_sha, **checkpoint_report)
            except Exception as error:
                root_errors.append({"root_id": root_id, "stage": "root", "error": f"{type(error).__name__}:{error}"})
    finally:
        if backend is not None:
            backend.close()
    batch = summarize_batch(root_reports)
    generation_gaps = [
        {
            "root_id": report.get("root_id"),
            "coverage": (report.get("generation") or {}).get("generation_coverage"),
        }
        for report in root_reports
        if not bool(((report.get("generation") or {}).get("generation_coverage") or {}).get(
            "minimum_coverage_complete"))
    ]
    base_complete = not root_errors and len(root_reports) == len(entries)
    result = {
        "probe_version": PROBE_VERSION,
        "status": (
            "completed_with_generation_gaps" if base_complete and generation_gaps
            else "completed" if base_complete else "partial"
        ),
        "run_root": str(args.run_root),
        "root_reports": root_reports,
        "root_errors": root_errors,
        "generation_gaps": generation_gaps,
        "batch_summary": batch,
        "smoke_report": smoke,
        "main_categories": list(MAIN_CATEGORIES),
        "semantic_grounding_judgment": False,
        "k3_interpretation": "exploratory diagnostic; not a K=5 equivalence claim",
    }
    _write(args.run_root / "batch_report.json", result)
    _write(args.run_root / "report.json", result)
    lines = [
        "# H4 Multi-root K=3/T=0.8 Probe",
        "",
        f"Status: **{result['status']}**",
        f"Roots completed: {len(root_reports)}/{len(entries)}",
        f"Micro summary: `{json.dumps(batch['micro'], ensure_ascii=False, sort_keys=True)}`",
        f"Macro summary: `{json.dumps(batch['macro'], ensure_ascii=False, sort_keys=True)}`",
        f"Root errors: `{json.dumps(root_errors, ensure_ascii=False, sort_keys=True)}`",
        f"Generation gaps: `{json.dumps(generation_gaps, ensure_ascii=False, sort_keys=True)}`",
        "",
        "Each root generated fresh QA candidates and screened them immediately.",
        "No production QA Writer, Planner, Realizer, Memorax, Solver, or historical run was modified.",
        "Evidence refs were checked structurally only; semantic grounding remains for critic/manual review.",
    ]
    atomic_text(args.run_root / "summary.md", "\n".join(lines) + "\n")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("inventory", "smoke", "batch"), required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--max-roots", type=int, default=5)
    parser.add_argument("--root-source", type=Path, action="append", default=[])
    parser.add_argument("--smoke-candidate-limit", type=int, default=3)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    args._argv = list(argv or [])
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase == "inventory":
        result = inventory_phase(args)
    elif args.phase == "smoke":
        result = smoke_phase(args)
    else:
        result = batch_phase(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
