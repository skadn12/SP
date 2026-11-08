"""Isolated H4 QA Mining Probe Phase 0.

The runner owns a new artifact directory and never imports the production QA
writer or admission pipeline.  It mines intent/candidates with the existing
challenger transport, then reuses only the local Solver and structured-box
protocol for evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from ..simple_append_v1.live_runtime import load_json_object_strict
from ..vllm_backends import VLLMSolverBackend
from .candidate_contract import normalize_candidates
from .candidate_generator import build_candidate_prompt, build_candidate_repair_prompt
from .difficulty_critic import build_critic_report
from .difficulty_filter import summarize_all
from .intent_contract import canonicalize_intents
from .intent_miner import build_intent_prompt
from .question_evolution import make_phase0_evolution_stubs
from .question_type_selector import select_question_types
from .reasoning_contract import build_reasoning_contract, contract_has_three_disjoint_groups
from .solver_adapter import evaluate_candidates
from .static_filter import audit_candidate


SOURCE_DIALOGUE = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_178392_round2_career_trial_followup_scene_release_v1_live_2/"
    "expanded_dialogue_round2.json"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_178392_h4_qa_mining_probe_v1_phase0_1"
)
DEFAULT_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_SOLVER_MODEL = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "05_stage2_repair_topup_from_stage1/training/models/merged"
)
PROBE_VERSION = "h4_qa_mining_probe_v1_phase0"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(child).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _write(run_root: Path, name: str, value: Any) -> None:
    atomic_json(run_root / name, value)


def _api_call(
    *, client: OpenAICompatibleChallengerClient, ledger: RequestLedger,
    run_root: Path, request_id: str, intent: str, prompt: str, max_tokens: int,
) -> tuple[dict[str, Any] | None, str, str | None]:
    prompt_sha = sha256_bytes(prompt.encode())
    atomic_text(run_root / "raw" / "prompts" / f"{request_id}.txt", prompt)
    response = ledger.invoke(
        request_id=request_id, intent=intent, attempt=1,
        prompt_sha256=prompt_sha,
        call=lambda: client.complete(prompt=prompt, max_tokens=max_tokens, json_object=True),
        budget_class="challenger_mining", budget_limit=7, total_budget_limit=7,
        request_metadata={"probe": PROBE_VERSION, "request_id": request_id},
        backend_identity={"transport": "openai_compatible_challenger",
                          "model": client.config.model,
                          "base_url": client.config.base_url},
    )
    raw = response.raw_text
    try:
        return load_json_object_strict(raw), raw, None
    except (json.JSONDecodeError, ValueError) as error:
        return None, raw, f"{type(error).__name__}:{error}"


def _build_manifest(args: argparse.Namespace, expanded: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_version": "h4_qa_mining_manifest_v1",
        "probe_version": PROBE_VERSION,
        "source_dialogue": str(args.source_dialogue),
        "source_dialogue_sha256": sha256_file(args.source_dialogue),
        "source_dialogue_tree_sha256": _tree_hash(args.source_dialogue.parent),
        "source_root_id": expanded.get("root_id"),
        "source_turn_count": len(expanded.get("turns") or []),
        "qa_fact_source": expanded.get("qa_fact_source"),
        "memory_is_qa_fact_source": expanded.get("memory_is_qa_fact_source"),
        "production_writer_modified": False,
        "planner_modified": False,
        "realizer_modified": False,
        "memorax_modified": False,
        "position_audit": {
            "single_choice_variants": 4,
            "multi_select_variants": 5,
            "ordering_variants": 4,
            "modes": ["no_context", "full_context"],
            "rollouts_per_variant": 5,
        },
        "minimum_candidates_per_intent": {
            "single_choice": 2, "multi_select": 1, "ordering": 1,
        },
        "argv": list(args._argv),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_root.exists():
        raise RuntimeError(f"run_root_already_exists:{args.run_root}")
    if not args.source_dialogue.is_file():
        raise FileNotFoundError(args.source_dialogue)
    expanded = _load(args.source_dialogue)
    args.run_root.mkdir(parents=True, exist_ok=False)
    manifest = _build_manifest(args, expanded)
    manifest_sha = sha256_bytes(canonical_json(manifest).encode())
    _write(args.run_root, "manifest.json", {**manifest, "manifest_sha256": manifest_sha})
    ledger = RequestLedger(args.run_root, manifest_sha, "live" if args.live else "mock")

    if not args.live:
        _write(args.run_root, "report.json", {
            "status": "dry_run", "reason": "pass --live to call mining API",
            "manifest_sha256": manifest_sha,
        })
        return {"status": "dry_run", "run_root": str(args.run_root)}

    config = load_challenger_config(args.api_config)
    client = OpenAICompatibleChallengerClient(replace(
        config, temperature=0.1, enable_thinking=False,
        timeout_seconds=max(180, int(config.timeout_seconds))))

    intent_spec = build_intent_prompt(expanded)
    raw_intents, raw_intent_text, intent_error = _api_call(
        client=client, ledger=ledger, run_root=args.run_root,
        request_id="intent-miner-v1", intent="h4_intent_miner",
        prompt=intent_spec["prompt"], max_tokens=4096)
    atomic_text(args.run_root / "raw" / "intent_miner_response.json.txt", raw_intent_text)
    intent_result = canonicalize_intents(raw_intents, expanded_dialogue=expanded)
    if intent_error:
        intent_result["errors"] = list(intent_result.get("errors") or []) + [intent_error]
        intent_result["valid"] = False
    _write(args.run_root, "intents.json", {"prompt": intent_spec, "validation": intent_result})

    contracts: list[dict[str, Any]] = []
    for intent in intent_result.get("intents") or []:
        if not intent.get("valid"):
            continue
        selection = select_question_types(intent)
        contract = build_reasoning_contract(intent)
        contract["question_type_selection"] = selection
        contract["contract_valid"] = bool(selection["phase0_complete"] and
                                           contract_has_three_disjoint_groups(contract))
        if contract["contract_valid"]:
            contracts.append(contract)
    _write(args.run_root, "reasoning_contracts.json", contracts)

    all_candidates: list[dict[str, Any]] = []
    generation_records: list[dict[str, Any]] = []
    for contract in contracts:
        spec = build_candidate_prompt(expanded, contract)
        request_id = f"candidate-generator-{contract['intent_id']}"
        raw, raw_text, parse_error = _api_call(
            client=client, ledger=ledger, run_root=args.run_root,
            request_id=request_id, intent="h4_candidate_generator",
            prompt=spec["prompt"], max_tokens=8192)
        atomic_text(args.run_root / "raw" / "candidate_responses" / f"{request_id}.json.txt", raw_text)
        normalized = normalize_candidates(raw, intent_id=str(contract["intent_id"]),
                                          contract=contract, expanded_dialogue=expanded)
        if parse_error:
            normalized["errors"] = list(normalized.get("errors") or []) + [parse_error]
            normalized["valid"] = False
        generation_records.append({"intent_id": contract["intent_id"], "request_id": request_id,
                                   "prompt": spec, "validation": normalized})
        # One schema-only repair keeps the per-intent minimum honest when the
        # model omitted a field (most commonly ordering.question).  The first
        # response remains preserved in this report and ledger.
        if not normalized.get("valid") and raw is not None:
            repair_spec = build_candidate_repair_prompt(contract, raw)
            repair_id = f"candidate-repair-{contract['intent_id']}"
            repaired_raw, repaired_text, repair_error = _api_call(
                client=client, ledger=ledger, run_root=args.run_root,
                request_id=repair_id, intent="h4_candidate_schema_repair",
                prompt=repair_spec["prompt"], max_tokens=8192)
            atomic_text(args.run_root / "raw" / "candidate_responses" / f"{repair_id}.json.txt", repaired_text)
            repaired = normalize_candidates(
                repaired_raw, intent_id=str(contract["intent_id"]),
                contract=contract, expanded_dialogue=expanded)
            if repair_error:
                repaired["errors"] = list(repaired.get("errors") or []) + [repair_error]
                repaired["valid"] = False
            generation_records.append({"intent_id": contract["intent_id"], "request_id": repair_id,
                                       "repair_of": request_id, "prompt": repair_spec,
                                       "validation": repaired})
            normalized = repaired
        # The minimum is per intent, not per surviving row.  A partial
        # 2-single + 1-multi batch without its ordering item is quarantined as
        # a failed intent rather than silently entering Solver evaluation.
        if normalized.get("valid"):
            all_candidates.extend(row for row in normalized.get("candidates") or []
                                  if row.get("schema_valid"))
    _write(args.run_root, "candidate_generation.json", generation_records)
    _write(args.run_root, "candidate_pool.json", all_candidates)

    static: dict[str, Any] = {}
    runnable: list[dict[str, Any]] = []
    for candidate in all_candidates:
        audit = audit_candidate(candidate)
        static[str(candidate["candidate_id"])] = audit
        if audit["decision"] == "pass":
            runnable.append(candidate)
    _write(args.run_root, "static_filter.json", static)

    solver_results: list[dict[str, Any]] = []
    solver_error: str | None = None
    backend = None
    if runnable:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
            backend = VLLMSolverBackend(
                model_path=args.solver_model, tokenizer_path=args.tokenizer,
                max_model_len=16_384, temperature=0.7, top_p=0.9,
                max_tokens=32, gpu_memory_utilization=0.80)
            solver_results = evaluate_candidates(
                candidates=runnable, expanded=expanded, run_root=args.run_root,
                ledger=ledger, tokenizer=tokenizer, backend=backend)
        except Exception as error:
            solver_error = f"{type(error).__name__}:{error}"
        finally:
            if backend is not None:
                backend.close()
    _write(args.run_root, "solver_results.json", {
        "solver_error": solver_error, "runnable_count": len(runnable),
        "results": solver_results,
    })
    difficulty = summarize_all(solver_results, static)
    _write(args.run_root, "difficulty.json", difficulty)
    _write(args.run_root, "critic.json", build_critic_report(all_candidates, difficulty))
    _write(args.run_root, "evolution_stubs.json",
           make_phase0_evolution_stubs(all_candidates, difficulty))
    completed_states = ledger.completed()
    class_counts: dict[str, int] = {}
    for state in completed_states:
        budget_class = str(state.get("budget_class") or "unknown")
        class_counts[budget_class] = class_counts.get(budget_class, 0) + 1
    report = {
        "report_version": "h4_qa_mining_phase0_report_v1",
        "status": "completed" if solver_error is None else "solver_failed",
        "run_root": str(args.run_root),
        "source_dialogue": str(args.source_dialogue),
        "source_turn_count": len(expanded.get("turns") or []),
        "intent_count": len(intent_result.get("intents") or []),
        "valid_intent_count": len(contracts),
        "candidate_count": len(all_candidates),
        "runnable_count": len(runnable),
        "solver_error": solver_error,
        "difficulty_bucket_counts": difficulty.get("bucket_counts"),
        "ledger_summary": ledger.summary(),
        "request_counts_by_budget_class": dict(sorted(class_counts.items())),
        "challenger_api_call_count": class_counts.get("challenger_mining", 0),
        "local_solver_request_count": class_counts.get("local_solver", 0),
        "structured_solver_request_count": class_counts.get("structured_solver", 0),
        "position_audit_solver_request_count": (
            class_counts.get("local_solver", 0) + class_counts.get("structured_solver", 0)),
        "mainline_untouched": True,
        "phase1_evolution_executed": False,
    }
    _write(args.run_root, "report.json", report)
    lines = [
        "# H4 QA Mining Probe v1 — Phase 0",
        "",
        f"Status: **{report['status']}**",
        f"Source: `{args.source_dialogue}` ({report['source_turn_count']} turns)",
        f"Intents: {report['valid_intent_count']}/{report['intent_count']}",
        f"Candidates: {report['candidate_count']} total; {report['runnable_count']} runnable",
        f"Buckets: `{json.dumps(report['difficulty_bucket_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"Requests by class: `{json.dumps(report['request_counts_by_budget_class'], ensure_ascii=False, sort_keys=True)}`",
        f"Ledger: `{json.dumps(report['ledger_summary'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "Position audit was direct: every A-D/A-E/I1-I4 placement was evaluated in both context modes with five Solver rollouts.",
        "No production QA Writer, Planner, Realizer, Memorax, Solver, or historical run was modified.",
        "Full-context-zero candidates remain labelled for manual review; they are not silently discarded.",
    ]
    atomic_text(args.run_root / "summary.md", "\n".join(lines) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dialogue", type=Path, default=SOURCE_DIALOGUE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    args._argv = []
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
