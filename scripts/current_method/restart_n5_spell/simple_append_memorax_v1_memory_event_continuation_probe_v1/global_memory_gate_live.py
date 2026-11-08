"""One-call Gate-only live probe using the frozen R4 global Memory view."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..live_backends import (
    OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from .contracts import build_semantic_gate_context, validate_semantic_gate_output
from .incremental_round import load_incremental_round_input
from .live import CONTEXT_LIMIT, LIVE_TIMEOUT_SECONDS, SEMANTIC_GATE_MAX_TOKENS, _manifest
from .live_runtime import SimpleAppendLiveCaller, aggregate_usage
from .memory_context import compile_global_memory_state_view
from .prompts import build_semantic_continuation_gate_prompt


RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
R3 = RUNS / "conversation_chronicles_episode_41625_memory_only_incremental_r3_live_2"
R4 = RUNS / "conversation_chronicles_episode_41625_memory_only_incremental_r4_live_2"
RUN_ROOT = RUNS / "conversation_chronicles_episode_41625_global_memory_candidate_gate_probe_2"
CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    if RUN_ROOT.exists():
        raise FileExistsError(f"global_memory_gate_run_exists:{RUN_ROOT}")
    source_paths = {
        "prior_expanded": R3 / "promotion/promotable_expanded_dialogue.json",
        "memory_snapshot": R4 / "memory/recovery_pre_generation_bucket_snapshot.json",
        "event_candidates": R4 / "memory/event_candidates.json",
        "compiled_plan": R4 / "planner/compiled_plan.json",
        "config": CONFIG,
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("global_memory_gate_sources_missing:" +
                                ",".join(missing))
    parent, _, input_validation = load_incremental_round_input(
        source_paths["prior_expanded"], round_index=4)
    snapshot = _read(source_paths["memory_snapshot"])
    candidates = _read(source_paths["event_candidates"])["candidates"]
    plan = _read(source_paths["compiled_plan"])
    view = compile_global_memory_state_view(snapshot)
    if view["root_id"] != parent["root_id"]:
        raise ValueError("global_memory_gate_root_mismatch")
    context = build_semantic_gate_context(
        parent=parent, candidates=candidates, plan=plan,
        memory_only_fast=True, global_memory_state_view=view)
    parent_text_present = any(
        str(row.get("text") or "") and len(str(row.get("text") or "")) > 20
        and str(row.get("text") or "") in str(context)
        for row in parent["dialogue"]["turns"])
    if parent_text_present:
        raise ValueError("global_memory_gate_parent_text_leak")
    prompt = build_semantic_continuation_gate_prompt(gate_context=context)

    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    for part in ("input", "semantic_gate", "reports", "raw/prompts",
                 "attempts"):
        (RUN_ROOT / part).mkdir(parents=True, exist_ok=True)
    source_hashes = {name: sha256_file(path)
                     for name, path in source_paths.items()}
    freeze = {
        "freeze_version": "global_memory_candidate_gate_probe_v1",
        "frozen_at": _now(),
        "run_root": str(RUN_ROOT),
        "root_id": parent["root_id"],
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_sha256": source_hashes,
        "source_manifest": _manifest(),
        "input_validation": input_validation,
        "global_memory_state_view_sha256": view["view_sha256"],
        "expected_candidate_refs": context["expected_candidate_refs"],
        "parent_transcript_text_in_gate_context": False,
        "semantic_gate_calls_max": 1,
        "retry": 0,
        "planner_calls": 0,
        "realizer_calls": 0,
        "memorax_calls": 0,
        "qa_calls": 0,
    }
    atomic_json(RUN_ROOT / "run_freeze_manifest.json", freeze)
    atomic_json(RUN_ROOT / "input/global_memory_state_view.json", view)
    atomic_json(RUN_ROOT / "input/gate_context.json", context)
    atomic_json(RUN_ROOT / "input/candidates.json", {"candidates": candidates})
    atomic_json(RUN_ROOT / "input/compiled_plan.json", plan)
    atomic_text(RUN_ROOT / "semantic_gate/prompt.txt", prompt["prompt"])
    atomic_json(RUN_ROOT / "semantic_gate/prompt_manifest.json",
                {key: value for key, value in prompt.items() if key != "prompt"})

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER,
                                               local_files_only=True)
    counter = lambda text: len(tokenizer.encode(
        text, add_special_tokens=False))
    config = replace(load_challenger_config(CONFIG),
                     timeout_seconds=LIVE_TIMEOUT_SECONDS)
    ledger = RequestLedger(
        RUN_ROOT, manifest_sha256=freeze["source_manifest"]["manifest_sha256"],
        backend_mode="live")

    def before_request(logical_role: str, role_version: str) -> None:
        if logical_role != "semantic_continuation_gate":
            raise RuntimeError("global_memory_gate_forbidden_role")
        if _manifest()["manifest_sha256"] != freeze["source_manifest"][
                "manifest_sha256"]:
            raise RuntimeError("global_memory_gate_source_drift")
        for name, path in source_paths.items():
            if sha256_file(path) != source_hashes[name]:
                raise RuntimeError(f"global_memory_gate_input_drift:{name}")
        atomic_json(RUN_ROOT / "run_status.json", {
            "status": "live_request_started", "started_at": _now(),
            "logical_role": logical_role, "role_version": role_version,
        })

    caller = SimpleAppendLiveCaller(
        run_root=RUN_ROOT, ledger=ledger,
        client=OpenAICompatibleChallengerClient(config), token_counter=counter,
        root_id=parent["root_id"], context_limit=CONTEXT_LIMIT,
        before_request=before_request)
    parsed = None
    raw = ""
    parse_errors: list[str] = []
    runtime_error = None
    try:
        parsed, raw, parse_errors = caller.call_json(
            request_id="semantic_continuation_gate_attempt_01",
            role="memory_event_realization_aware_semantic_gate_v1",
            prompt=prompt["prompt"], prompt_version=prompt["prompt_version"],
            max_tokens=SEMANTIC_GATE_MAX_TOKENS)
    except Exception as error:
        runtime_error = f"{type(error).__name__}:{error}"
    atomic_text(RUN_ROOT / "semantic_gate/raw_output.txt", raw)
    if parsed is not None:
        atomic_json(RUN_ROOT / "semantic_gate/output.json", parsed)
    validation = (validate_semantic_gate_output(
        parsed, plan=plan, parent=parent, memory_only_fast=True,
        global_memory_state_view=view)
        if parsed is not None else
        {"valid": False, "errors": parse_errors or [runtime_error or
                                                       "no_response"]})
    atomic_json(RUN_ROOT / "semantic_gate/validation.json", validation)
    usage = aggregate_usage(caller.records)
    record = caller.records[0] if caller.records else {}
    report = {
        "probe_version": "global_memory_candidate_gate_probe_v2_memory_refs",
        "status": "completed" if validation.get("valid") else "failed",
        "runtime_error": runtime_error,
        "parse_errors": parse_errors,
        "finish_reason": (record.get("response_metadata") or {}).get(
            "finish_reason"),
        "candidate_count": context["expected_candidate_count"],
        "global_memory_bucket_counts": view["bucket_counts"],
        "global_memory_view_tokens_estimate": view["token_estimate"],
        "parent_transcript_text_in_gate_context": False,
        "validation": validation,
        "usage": usage,
        "semantic_gate_calls": len(caller.records),
        "retry": 0,
        "planner_calls": 0,
        "realizer_calls": 0,
        "memorax_calls": 0,
        "qa_calls": 0,
    }
    atomic_json(RUN_ROOT / "reports/GLOBAL_MEMORY_GATE_LIVE_REPORT.json",
                report)
    rows = validation.get("normalized_candidate_reviews") or []
    lines = [
        "# Global Memory Gate Live Probe",
        "",
        f"- Status: `{report['status']}`",
        f"- Gate calls / retries: `{report['semantic_gate_calls']}/0`",
        f"- Finish reason: `{report['finish_reason']}`",
        f"- Candidates: `{report['candidate_count']}`",
        f"- Parent transcript in Gate: `false`",
        f"- Usage: `{usage['prompt_tokens']} + {usage['completion_tokens']} = {usage['total_tokens']}`",
        "",
        "## Decisions",
        "",
    ]
    for row in rows:
        lines.extend([
            f"- `{row['candidate_ref']}`: `{row['decision']}` / "
            f"`{row['parent_relation']}` - {row['reason']}",
        ])
    atomic_text(RUN_ROOT / "reports/GLOBAL_MEMORY_GATE_LIVE_REPORT.md",
                "\n".join(lines) + "\n")
    atomic_json(RUN_ROOT / "run_status.json", {
        "status": report["status"], "finished_at": _now(),
        "semantic_gate_calls": report["semantic_gate_calls"], "retry": 0,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
