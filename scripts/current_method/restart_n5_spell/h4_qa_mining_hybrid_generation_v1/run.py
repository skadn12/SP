from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_qa_alignment_audit_v1.audit import audit_bundle
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from ..simple_append_v1.live_runtime import load_json_object_strict
from ..vllm_backends import VLLMSolverBackend
from .basic_validator import validate_candidate
from .candidate_contract import normalize_batch
from .contract_guided_generator import build_prompt as build_guided_prompt
from .contract_guided_generator import build_repair_prompt as build_guided_repair
from .difficulty_filter import apply_hybrid_policy
from .free_exploration_generator import build_prompt as build_free_prompt
from .free_exploration_generator import build_repair_prompt as build_free_repair
from .hybrid_contract import expected_counts
from .solver_adapter import evaluate_candidates


BASELINE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_178392_h4_qa_mining_probe_v1_phase0_4"
)
CONTRACT_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_178392_h4_qa_mining_contract_guided_generation_v1_phase0_3"
)
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_178392_h4_qa_mining_hybrid_generation_v1_phase0_1"
)
DEFAULT_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_SOLVER_MODEL = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "05_stage2_repair_topup_from_stage1/training/models/merged"
)
VERSION = "h4_qa_mining_hybrid_generation_v1_phase0"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(root: Path, name: str, value: Any) -> None:
    atomic_json(root / name, value)


def _contracts(contract_run: Path) -> tuple[list[dict[str, Any]], Path, str]:
    direct = contract_run / "reasoning_contracts.json"
    provenance = str(contract_run)
    source_run = contract_run
    if direct.is_file():
        return _load(direct), direct, provenance
    manifest = _load(contract_run / "manifest.json")
    source_run = Path(str(manifest.get("input_run") or ""))
    path = source_run / "reasoning_contracts.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _load(path), path, provenance + " (contract file resolved via manifest.input_run)"


def _source_dialogue(baseline_run: Path) -> tuple[dict[str, Any], Path]:
    manifest = _load(baseline_run / "manifest.json")
    path = Path(str(manifest.get("source_dialogue") or ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != manifest.get("source_dialogue_sha256"):
        raise RuntimeError("source_dialogue_hash_mismatch")
    return _load(path), path


def _api_call(*, client: OpenAICompatibleChallengerClient, ledger: RequestLedger,
              root: Path, request_id: str, prompt: str, source: str) -> tuple[Any, str, str | None]:
    prompt_sha = sha256_bytes(prompt.encode())
    atomic_text(root / "raw" / "prompts" / f"{request_id}.txt", prompt)
    response = ledger.invoke(
        request_id=request_id, intent="h4_hybrid_candidate_generation", attempt=1,
        prompt_sha256=prompt_sha,
        call=lambda: client.complete(prompt=prompt, max_tokens=8192, json_object=True),
        budget_class=f"challenger_{source}", budget_limit=12, total_budget_limit=20,
        request_metadata={"probe_version": VERSION, "source": source, "request_id": request_id},
        backend_identity={"transport": "openai_compatible_challenger",
                          "model": client.config.model, "base_url": client.config.base_url},
    )
    raw_text = response.raw_text
    atomic_text(root / "raw" / "candidate_responses" / f"{request_id}.json.txt", raw_text)
    try:
        return load_json_object_strict(raw_text), raw_text, None
    except (json.JSONDecodeError, ValueError) as error:
        return None, raw_text, f"{type(error).__name__}:{error}"


def _validate_source_batch(raw: Any, *, intent: Mapping[str, Any], source: str,
                           expanded: Mapping[str, Any]) -> dict[str, Any]:
    return normalize_batch(raw, intent_id=str(intent.get("intent_id")), source=source,
                           contract=intent, expanded=expanded)


def _counts(candidates: list[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in candidates:
        qtype = str(row.get("question_type") or "unknown")
        out[qtype] = out.get(qtype, 0) + 1
    return dict(sorted(out.items()))


def _comparison(run: Path, *, alignment_name: str = "alignment_audit.json") -> dict[str, Any]:
    result: dict[str, Any] = {"run": str(run)}
    for name in ("report.json", "difficulty.json", alignment_name):
        path = run / name
        if not path.is_file():
            continue
        data = _load(path)
        if name == "report.json":
            result["report"] = data
        elif name == "difficulty.json":
            result["difficulty_bucket_counts"] = data.get("bucket_counts")
            result["difficulty_by_source"] = data.get("by_source")
        else:
            result["alignment_bucket_counts"] = data.get("alignment_bucket_counts")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_run.exists():
        raise RuntimeError(f"output_run_already_exists:{args.output_run}")
    contracts, contract_path, contract_provenance = _contracts(args.contract_run)
    expanded, source_path = _source_dialogue(args.input_run)
    if len(contracts) < 3:
        raise RuntimeError(f"expected_at_least_three_intents:{len(contracts)}")
    contracts = contracts[:3]
    input_hashes = {
        "dialogue": sha256_file(source_path),
        "contract_file": sha256_file(contract_path),
        "baseline_manifest": sha256_file(args.input_run / "manifest.json"),
        "contract_run_manifest": sha256_file(args.contract_run / "manifest.json"),
    }
    manifest = {
        "manifest_version": "h4_hybrid_generation_manifest_v1",
        "probe_version": VERSION,
        "input_run": str(args.input_run),
        "contract_run": str(args.contract_run),
        "contract_file": str(contract_path),
        "contract_provenance": contract_provenance,
        "source_dialogue": str(source_path),
        "source_dialogue_sha256": input_hashes["dialogue"],
        "input_hashes": input_hashes,
        "intent_count": len(contracts),
        "candidate_counts_per_intent": {
            "contract_guided": expected_counts("contract_guided"),
            "free_exploration": expected_counts("free_exploration"),
        },
        "position_audit": {"single_choice_variants": 4, "multi_select_variants": 5,
                           "ordering_variants": 4, "modes": ["no_context", "full_context"],
                           "rollouts_per_variant": 5},
        "solver_temperatures": {"single_choice": 0.7, "multi_select": 0.3,
                                 "ordering": 0.0},
        "production_qa_modified": False, "planner_modified": False,
        "realizer_modified": False, "memorax_modified": False,
        "solver_code_modified": False, "upstream_runs_modified": False,
        "contract_is_guidance_not_gate": True,
        "semantic_grounding_judgment": False,
        "argv": list(getattr(args, "_argv", [])),
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest).encode())
    args.output_run.mkdir(parents=True, exist_ok=False)
    _write(args.output_run, "manifest.json", manifest)
    ledger = RequestLedger(args.output_run, manifest["manifest_sha256"], "live" if args.live else "mock")

    if not args.live:
        report = {"status": "dry_run", "offline_generation": True,
                  "expected_candidate_count": 36,
                  "expected_challenger_calls": 6,
                  "reason": "pass --live to generate and run the isolated hybrid probe"}
        _write(args.output_run, "report.json", report)
        return report

    config = load_challenger_config(args.api_config)
    client = OpenAICompatibleChallengerClient(replace(
        config, temperature=0.1, enable_thinking=False,
        timeout_seconds=max(180, int(config.timeout_seconds))))
    generation_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    basic_rows: dict[str, Any] = {}
    valid_refs = {str(x.get("turn_ref")) for x in expanded.get("turns") or []}

    for intent in contracts:
        intent_id = str(intent.get("intent_id"))
        for source in ("contract_guided", "free_exploration"):
            prompt_spec = (build_guided_prompt(expanded, intent) if source == "contract_guided"
                           else build_free_prompt(expanded, intent))
            request_id = f"{source.replace('_', '-')}-generator-{intent_id}"
            raw, raw_text, parse_error = _api_call(
                client=client, ledger=ledger, root=args.output_run,
                request_id=request_id, prompt=prompt_spec["prompt"], source=source)
            validation = _validate_source_batch(raw, intent=intent, source=source, expanded=expanded)
            if parse_error:
                validation["errors"] = sorted(set(list(validation.get("errors") or []) + [parse_error]))
                validation["valid"] = False
            generation_records.append({"intent_id": intent_id, "generation_source": source,
                                       "request_id": request_id, "prompt": prompt_spec,
                                       "validation": validation})
            if not validation.get("valid"):
                repair_spec = (build_guided_repair(expanded, intent, raw, source) if source == "contract_guided"
                               else build_free_repair(expanded, intent, raw))
                repair_id = f"{source.replace('_', '-')}-repair-{intent_id}"
                repaired_raw, _, repair_error = _api_call(
                    client=client, ledger=ledger, root=args.output_run,
                    request_id=repair_id, prompt=repair_spec["prompt"], source=source)
                repaired = _validate_source_batch(repaired_raw, intent=intent, source=source, expanded=expanded)
                if repair_error:
                    repaired["errors"] = sorted(set(list(repaired.get("errors") or []) + [repair_error]))
                    repaired["valid"] = False
                generation_records.append({"intent_id": intent_id, "generation_source": source,
                                           "request_id": repair_id, "repair_of": request_id,
                                           "prompt": repair_spec, "validation": repaired})
                validation = repaired
            rows = list(validation.get("candidates") or [])
            if validation.get("valid"):
                candidates.extend(rows)
            else:
                # Preserve only structurally valid rows for diagnostic Solver replay;
                # the report records the incomplete batch instead of silently filling it.
                candidates.extend(row for row in rows if row.get("schema_valid"))

    _write(args.output_run, "candidate_generation.json", generation_records)
    _write(args.output_run, "candidate_pool.json", candidates)
    for candidate in candidates:
        basic_rows[str(candidate["candidate_id"])] = validate_candidate(candidate, valid_turn_refs=valid_refs)
    _write(args.output_run, "basic_validation.json", basic_rows)
    runnable = [x for x in candidates if basic_rows[str(x["candidate_id"])] ["decision"] == "pass"]
    _write(args.output_run, "structurally_grounded_candidates.json", runnable)

    solver_error = None
    solver_results: list[dict[str, Any]] = []
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
            if backend is not None: backend.close()
    _write(args.output_run, "solver_results.json", {"solver_error": solver_error,
                                                        "runnable_count": len(runnable),
                                                        "results": solver_results})
    difficulty = apply_hybrid_policy({"results": solver_results}, runnable, basic_rows)
    _write(args.output_run, "difficulty.json", difficulty)
    difficulty_rows = difficulty.get("candidate_summaries") or []
    turn_text = {str(x.get("turn_ref")): str(x.get("text") or "") for x in expanded.get("turns") or []}
    alignment = audit_bundle(candidates=runnable, contracts=contracts, turn_text=turn_text,
                             static_rows=basic_rows, difficulty_rows=difficulty_rows)
    _write(args.output_run, "alignment_audit.json", alignment)

    completed = ledger.completed()
    class_counts: dict[str, int] = {}
    for state in completed:
        key = str(state.get("budget_class") or "unknown")
        class_counts[key] = class_counts.get(key, 0) + 1
    report = {
        "report_version": "h4_hybrid_generation_report_v1",
        "status": "completed" if solver_error is None else "solver_failed",
        "output_run": str(args.output_run), "input_run": str(args.input_run),
        "contract_run": str(args.contract_run), "candidate_count": len(candidates),
        "runnable_count": len(runnable), "candidate_counts": _counts(candidates),
        "contract_generated": sum(x.get("generation_source") == "contract_guided" for x in candidates),
        "free_generated": sum(x.get("generation_source") == "free_exploration" for x in candidates),
        "structurally_grounded_count": len(runnable),
        "difficulty_bucket_counts": difficulty.get("bucket_counts"),
        "hard_candidate_count": difficulty.get("bucket_counts", {}).get("hard_candidate", 0),
        "context_helpful_count": sum(bool(x.get("context_helpful")) for x in difficulty_rows),
        "invalid_or_ambiguous_count": difficulty.get("bucket_counts", {}).get("invalid_or_ambiguous", 0),
        "alignment_bucket_counts": alignment.get("alignment_bucket_counts"),
        "alignment_contract_pass_count": alignment.get("contract_alignment_pass_count"),
        "difficulty_by_source": difficulty.get("by_source"),
        "solver_error": solver_error,
        "request_counts_by_budget_class": dict(sorted(class_counts.items())),
        "challenger_api_call_count": sum(value for key, value in class_counts.items() if key.startswith("challenger_")),
        "local_solver_request_count": class_counts.get("local_solver", 0),
        "structured_solver_request_count": class_counts.get("structured_solver", 0),
        "comparison": {
            "phase0_baseline": _comparison(args.input_run),
            "contract_guided": _comparison(args.contract_run),
        },
        "mainline_untouched": True, "upstream_runs_modified": False,
    }
    _write(args.output_run, "report.json", report)
    lines = [
        "# H4 QA Mining Hybrid Generation Probe v1", "",
        f"Status: **{report['status']}**", f"Candidates: {len(candidates)} (guided={report['contract_generated']}, free={report['free_generated']})",
        f"Structurally grounded: {len(runnable)}; semantic grounding was not judged.",
        f"Difficulty buckets: `{json.dumps(report['difficulty_bucket_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"By source: `{json.dumps(report['difficulty_by_source'], ensure_ascii=False, sort_keys=True)}`",
        f"Alignment: `{json.dumps(report['alignment_bucket_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"Request counts: `{json.dumps(report['request_counts_by_budget_class'], ensure_ascii=False, sort_keys=True)}`", "",
        "Hard means `0 < full_context_accuracy < 1` and `no_context_accuracy < 1`.",
        "context_helpful is an auxiliary label (`full_context_accuracy > no_context_accuracy`), not a separate bucket.",
        "Position rebinding is deterministic and performed before each Solver replay; semantic evidence validity remains for critic/manual review.",
        "No production QA, Planner, Realizer, Memorax, Solver code, or upstream run was modified.",
    ]
    atomic_text(args.output_run / "summary.md", "\n".join(lines) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-run", type=Path, default=BASELINE_RUN)
    parser.add_argument("--contract-run", type=Path, default=CONTRACT_RUN)
    parser.add_argument("--output-run", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    args._argv = []
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
