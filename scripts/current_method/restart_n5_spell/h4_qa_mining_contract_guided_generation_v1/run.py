"""Isolated contract-guided candidate generation and Phase-0 replay."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_qa_mining_probe_v1.difficulty_filter import summarize_all
from ..h4_qa_mining_probe_v1.difficulty_critic import build_critic_report
from ..h4_qa_mining_probe_v1.solver_adapter import evaluate_candidates
from ..h4_qa_mining_probe_v1.static_filter import audit_candidate
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from ..simple_append_v1.live_runtime import load_json_object_strict
from ..vllm_backends import VLLMSolverBackend
from .contract_prompt import build_prompt, build_ordering_repair_prompt, build_repair_prompt
from .difficulty_policy import apply_policy
from .strict_contract import validate_guided_candidates


DEFAULT_INPUT_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_178392_h4_qa_mining_probe_v1_phase0_4"
)
DEFAULT_OUTPUT_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_178392_h4_qa_mining_contract_guided_generation_v1_phase0_1"
)
DEFAULT_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_SOLVER_MODEL = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "05_stage2_repair_topup_from_stage1/training/models/merged"
)
VERSION = "h4_qa_mining_contract_guided_generation_v1_phase0"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(root: Path, name: str, value: Any) -> None:
    atomic_json(root / name, value)


def _api_call(
    *, client: OpenAICompatibleChallengerClient, ledger: RequestLedger,
    root: Path, request_id: str, prompt: str, max_tokens: int,
) -> tuple[dict[str, Any] | None, str, str | None]:
    prompt_sha = sha256_bytes(prompt.encode())
    atomic_text(root / "raw" / "prompts" / f"{request_id}.txt", prompt)
    response = ledger.invoke(
        request_id=request_id, intent="h4_contract_guided_candidate_generation", attempt=1,
        prompt_sha256=prompt_sha,
        call=lambda: client.complete(prompt=prompt, max_tokens=max_tokens, json_object=True),
        budget_class="challenger_guided_generation", budget_limit=7, total_budget_limit=7,
        request_metadata={"probe_version": VERSION, "request_id": request_id},
        backend_identity={"transport": "openai_compatible_challenger",
                          "model": client.config.model, "base_url": client.config.base_url},
    )
    raw = response.raw_text
    try:
        return load_json_object_strict(raw), raw, None
    except (json.JSONDecodeError, ValueError) as error:
        return None, raw, f"{type(error).__name__}:{error}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_run.exists():
        raise RuntimeError(f"output_run_already_exists:{args.output_run}")
    required = [args.input_run / name for name in (
        "manifest.json", "reasoning_contracts.json", "candidate_pool.json")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = _load(args.input_run / "manifest.json")
    contracts = _load(args.input_run / "reasoning_contracts.json")
    source_path = Path(str(manifest.get("source_dialogue") or ""))
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    expanded = _load(source_path)
    source_hash = sha256_file(source_path)
    if source_hash != manifest.get("source_dialogue_sha256"):
        raise RuntimeError("source_dialogue_hash_mismatch")
    input_hashes = {path.name: sha256_file(path) for path in required}
    out_manifest = {
        "manifest_version": "h4_contract_guided_generation_manifest_v1",
        "probe_version": VERSION,
        "input_run": str(args.input_run),
        "input_hashes": input_hashes,
        "source_dialogue": str(source_path),
        "source_dialogue_sha256": source_hash,
        "contract_count": len(contracts),
        "production_writer_modified": False,
        "planner_modified": False, "realizer_modified": False,
        "memorax_modified": False, "solver_code_modified": False,
        "candidate_generation_only_variable": True,
        "strict_contract_gate": True,
        "argv": list(getattr(args, "_argv", [])),
    }
    out_manifest["manifest_sha256"] = sha256_bytes(canonical_json(out_manifest).encode())
    args.output_run.mkdir(parents=True, exist_ok=False)
    _write(args.output_run, "manifest.json", out_manifest)
    ledger = RequestLedger(args.output_run, out_manifest["manifest_sha256"],
                           "live" if args.live else "mock")
    if not args.live:
        _write(args.output_run, "report.json", {
            "status": "dry_run", "offline_generation": True,
            "reason": "pass --live to call the isolated candidate generator",
        })
        return {"status": "dry_run", "output_run": str(args.output_run)}

    config = load_challenger_config(args.api_config)
    client = OpenAICompatibleChallengerClient(replace(
        config, temperature=0.1, enable_thinking=False,
        timeout_seconds=max(180, int(config.timeout_seconds))))
    records: list[dict[str, Any]] = []
    guided_candidates: list[dict[str, Any]] = []
    for contract in contracts:
        intent_id = str(contract.get("intent_id"))
        prompt_spec = build_prompt(expanded, contract)
        request_id = f"guided-generator-{intent_id}"
        raw, raw_text, parse_error = _api_call(
            client=client, ledger=ledger, root=args.output_run,
            request_id=request_id, prompt=prompt_spec["prompt"], max_tokens=8192)
        atomic_text(args.output_run / "raw" / "candidate_responses" / f"{request_id}.json.txt", raw_text)
        validation = validate_guided_candidates(raw, contract=contract, expanded_dialogue=expanded)
        if parse_error:
            validation["errors"] = list(validation.get("errors") or []) + [parse_error]
            validation["valid"] = False
        record = {"intent_id": intent_id, "request_id": request_id,
                  "prompt": prompt_spec, "validation": validation}
        records.append(record)
        if not validation.get("valid") and raw_text:
            # A malformed JSON response (for example duplicate ``gold`` keys)
            # still contains useful candidate content.  Pass the raw text to
            # the isolated schema repair instead of silently losing the intent.
            repair_input = raw if raw is not None else {"raw_response_text": raw_text}
            repair_spec = build_repair_prompt(expanded, contract, repair_input)
            repair_id = f"guided-repair-{intent_id}"
            repaired_raw, repaired_text, repair_error = _api_call(
                client=client, ledger=ledger, root=args.output_run,
                request_id=repair_id, prompt=repair_spec["prompt"], max_tokens=8192)
            atomic_text(args.output_run / "raw" / "candidate_responses" / f"{repair_id}.json.txt", repaired_text)
            repaired = validate_guided_candidates(
                repaired_raw, contract=contract, expanded_dialogue=expanded)
            if repair_error:
                repaired["errors"] = list(repaired.get("errors") or []) + [repair_error]
                repaired["valid"] = False
            records.append({"intent_id": intent_id, "request_id": repair_id,
                            "repair_of": request_id, "prompt": repair_spec,
                            "validation": repaired})
            validation = repaired
        # If a batch has the required two singles and multi-select but its
        # ordering row still violates the fixed I1-I4 protocol, run one
        # ordering-only repair. This keeps the failure attributable to the
        # contract/generator rather than silently lowering the minimum.
        if (not validation.get("valid") and
                any("ordering" in str(error) for error in validation.get("errors") or [])):
            invalid_ordering = next(
                (row for row in validation.get("candidates") or []
                 if row.get("question_type") == "ordering"), {})
            order_spec = build_ordering_repair_prompt(expanded, contract, invalid_ordering)
            order_id = f"guided-ordering-repair-{intent_id}"
            order_raw, order_text, order_error = _api_call(
                client=client, ledger=ledger, root=args.output_run,
                request_id=order_id, prompt=order_spec["prompt"], max_tokens=4096)
            atomic_text(args.output_run / "raw" / "candidate_responses" / f"{order_id}.json.txt", order_text)
            order_object = order_raw
            if isinstance(order_raw, dict) and isinstance(order_raw.get("candidate"), dict):
                order_object = {"candidates": [order_raw["candidate"]]}
            existing_rows = [row for row in validation.get("candidates") or []
                             if row.get("guided_contract_valid")]
            combined_raw = {"candidates": existing_rows + list((order_object or {}).get("candidates") or [])}
            repaired = validate_guided_candidates(
                combined_raw, contract=contract, expanded_dialogue=expanded)
            if order_error:
                repaired["errors"] = list(repaired.get("errors") or []) + [order_error]
                repaired["valid"] = False
            records.append({"intent_id": intent_id, "request_id": order_id,
                            "repair_of": "ordering", "prompt": order_spec,
                            "validation": repaired})
            validation = repaired
        if validation.get("valid"):
            guided_candidates.extend(
                row for row in validation.get("candidates") or []
                if row.get("guided_contract_valid"))
    _write(args.output_run, "candidate_generation.json", records)
    _write(args.output_run, "candidate_pool.json", guided_candidates)

    static_rows: dict[str, Any] = {}
    runnable: list[dict[str, Any]] = []
    for candidate in guided_candidates:
        audit = audit_candidate(candidate)
        static_rows[str(candidate["candidate_id"])] = audit
        if audit["decision"] == "pass":
            runnable.append(candidate)
    _write(args.output_run, "static_filter.json", static_rows)

    solver_results: list[dict[str, Any]] = []
    solver_error = None
    backend = None
    if runnable:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
            backend = VLLMSolverBackend(
                model_path=args.solver_model, tokenizer_path=args.tokenizer,
                max_model_len=16_384, temperature=0.7, top_p=0.9,
                max_tokens=32, gpu_memory_utilization=0.80)
            solver_results = evaluate_candidates(
                candidates=runnable, expanded=expanded, run_root=args.output_run,
                ledger=ledger, tokenizer=tokenizer, backend=backend)
        except Exception as error:
            solver_error = f"{type(error).__name__}:{error}"
        finally:
            if backend is not None:
                backend.close()
    _write(args.output_run, "solver_results.json", {
        "solver_error": solver_error, "runnable_count": len(runnable),
        "results": solver_results,
    })
    difficulty = apply_policy(summarize_all(solver_results, static_rows))
    _write(args.output_run, "difficulty.json", difficulty)
    _write(args.output_run, "critic.json", build_critic_report(guided_candidates, difficulty))

    completed = ledger.completed()
    class_counts: dict[str, int] = {}
    for state in completed:
        key = str(state.get("budget_class") or "unknown")
        class_counts[key] = class_counts.get(key, 0) + 1
    baseline_alignment = None
    baseline_alignment_path = args.input_run.parent / (
        args.input_run.name.replace(
            "h4_qa_mining_probe_v1", "h4_qa_alignment_audit_v1")) / "report.json"
    if baseline_alignment_path.is_file():
        try:
            baseline_alignment = _load(baseline_alignment_path)
        except (OSError, json.JSONDecodeError):
            baseline_alignment = None
    baseline_report = _load(args.input_run / "report.json") if (args.input_run / "report.json").is_file() else {}
    report = {
        "report_version": "h4_contract_guided_generation_report_v1",
        "status": "completed" if solver_error is None else "solver_failed",
        "output_run": str(args.output_run), "input_run": str(args.input_run),
        "candidate_count": len(guided_candidates), "runnable_count": len(runnable),
        "contract_count": len(contracts),
        "guided_generation_valid_intent_count": len({
            str(row.get("intent_id")) for row in records
            if row["validation"].get("valid")
        }),
        "difficulty_bucket_counts": difficulty.get("bucket_counts"),
        "solver_error": solver_error,
        "request_counts_by_budget_class": dict(sorted(class_counts.items())),
        "challenger_api_call_count": class_counts.get("challenger_guided_generation", 0),
        "local_solver_request_count": class_counts.get("local_solver", 0),
        "structured_solver_request_count": class_counts.get("structured_solver", 0),
        "comparison_to_phase0_baseline": {
            "baseline_run": str(args.input_run),
            "baseline_difficulty_bucket_counts": baseline_report.get("difficulty_bucket_counts"),
            "baseline_alignment_bucket_counts": (
                baseline_alignment.get("alignment_bucket_counts")
                if baseline_alignment else None),
        },
        "mainline_untouched": True,
        "phase0_input_modified": False,
    }
    _write(args.output_run, "report.json", report)
    lines = [
        "# H4 QA Mining Contract-Guided Generation v1",
        "",
        f"Status: **{report['status']}**",
        f"Input: `{args.input_run}`",
        f"Contracts: {len(contracts)}; candidates admitted by strict gate: {len(guided_candidates)}",
        f"Difficulty buckets: `{json.dumps(report['difficulty_bucket_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"Requests by class: `{json.dumps(report['request_counts_by_budget_class'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "Evidence metadata was admitted only when it exactly matched the frozen reasoning contract.",
        "No production QA, Planner, Realizer, Memorax, Solver code, or Phase-0 input was modified.",
    ]
    atomic_text(args.output_run / "summary.md", "\n".join(lines) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-run", type=Path, default=DEFAULT_INPUT_RUN)
    parser.add_argument("--output-run", type=Path, default=DEFAULT_OUTPUT_RUN)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    args._argv = []
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
