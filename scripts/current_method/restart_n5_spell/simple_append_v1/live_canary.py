"""Single-use live canary for the isolated simple append branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..live_backends import (
    OpenAICompatibleChallengerClient, load_challenger_config,
)
from ..unified_dialogue_root import validate_unified_dialogue_root
from .assembly import (
    assemble_expanded_dialogue, expanded_dialogue_markdown,
    validate_expanded_dialogue,
)
from .context import build_tail_context
from .contracts import (
    compile_round_plan, normalize_round_realizer_output,
    validate_planner_output,
)
from .live_runtime import SimpleAppendLiveCaller, aggregate_usage
from .prompts import (
    build_mcq_batch_prompt, build_planner_prompt, build_round_realizer_prompt,
)


RUNNER_VERSION = "simple_append_live_canary_v1"
DEFAULT_INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots/cc-train-episode-61555/unified_root.json")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_simple_append_v1_live_canary_1")
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
PLANNER_MAX_TOKENS = 4096
ROUND_REALIZER_MAX_TOKENS = 8192
CONTEXT_LIMIT = 40_960
LIVE_TIMEOUT_SECONDS = 600


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    files = sorted(path for path in root.glob("*.py") if path.is_file())
    rows = [{"path": str(path), "sha256": sha256_file(path)} for path in files]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"files": rows, "manifest_sha256": hashlib.sha256(payload).hexdigest()}


def _forbidden_imports() -> list[str]:
    findings = []
    for path in Path(__file__).resolve().parent.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in ("world_memory_slim_v1", "live_role_runtime_v11"):
            if (f"import {token}" in text or f"from {token}" in text or
                    f"from ..{token}" in text):
                findings.append(f"{path.name}:{token}")
    return findings


def _effective_api_config(api_config_path: Path):
    """Freeze branch-local transport settings without editing shared config."""
    return replace(load_challenger_config(api_config_path),
                   timeout_seconds=LIVE_TIMEOUT_SECONDS)


def run_preflight(
    *, input_path: Path, run_root: Path, api_config_path: Path,
    tokenizer_path: Path,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "run_root_absent": not run_root.exists(),
        "input_exists": input_path.is_file(),
        "api_config_exists": api_config_path.is_file(),
        "tokenizer_exists": tokenizer_path.exists(),
    }
    root: dict[str, Any] = {}
    tail: dict[str, Any] = {}
    planner_prompt: dict[str, Any] = {}
    config = None
    try:
        root = json.loads(input_path.read_text(encoding="utf-8"))
        validation = validate_unified_dialogue_root(root)
        checks["authoritative_parent_valid"] = validation["valid"]
        tail = build_tail_context(root, session_count=2)
        checks["last_two_complete_sessions"] = tail["selected_session_count"] == 2
        checks["parent_true_tail_selected"] = (
            tail["parent_tail_turn_id"] == root["dialogue"]["turns"][-1]["turn_id"])
        planner_prompt = build_planner_prompt(tail)
    except Exception:
        checks.update({
            "authoritative_parent_valid": False,
            "last_two_complete_sessions": False,
            "parent_true_tail_selected": False,
        })
    try:
        config = _effective_api_config(api_config_path)
        checks["api_timeout_at_least_600"] = (
            config.timeout_seconds == LIVE_TIMEOUT_SECONDS)
        checks["model_configured"] = bool(config.model)
    except Exception:
        checks["api_timeout_at_least_600"] = False
        checks["model_configured"] = False
    checks["recursive_world_memory_imports_zero"] = not _forbidden_imports()
    checks["planner_prompt_under_context_limit"] = bool(planner_prompt) and (
        int(planner_prompt.get("token_estimate") or CONTEXT_LIMIT) +
        PLANNER_MAX_TOKENS <= CONTEXT_LIMIT)
    manifest = _source_manifest()
    return {
        "preflight_version": "simple_append_live_preflight_v1",
        "valid": all(checks.values()),
        "failed_checks": [key for key, value in checks.items() if not value],
        "checks": checks,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path) if input_path.is_file() else None,
        "parent_dialogue_sha256": ((root.get("dialogue") or {}).get("dialogue_hash")
                                     if root else None),
        "root_id": root.get("root_id") if root else None,
        "run_root": str(run_root.resolve()),
        "api_config_path": str(api_config_path.resolve()),
        "api_config_sha256": (sha256_file(api_config_path)
                              if api_config_path.is_file() else None),
        "api_configuration": ({
            "model": config.model,
            "base_url": config.base_url,
            "temperature": config.temperature,
            "timeout_seconds": config.timeout_seconds,
            "api_key_recorded": False,
        } if config is not None else None),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "source_manifest": manifest,
        "planner_prompt_sha256": planner_prompt.get("prompt_sha256"),
        "planner_prompt_token_estimate": planner_prompt.get("token_estimate"),
        "request_budget": {"planner": 1, "round_realizer": 1, "total": 2},
        "max_completion_tokens": {
            "planner": PLANNER_MAX_TOKENS,
            "round_realizer": ROUND_REALIZER_MAX_TOKENS,
        },
        "cache_reuse": 0,
        "imported_response_reuse": 0,
        "world_memory_calls": 0,
        "qa_calls": 0,
    }


class FreezeGuard:
    def __init__(self, *, run_root: Path, freeze: Mapping[str, Any]):
        self.run_root = run_root
        self.freeze = dict(freeze)
        self.first_api_at: str | None = None

    def before_request(self, logical_role: str, role_version: str) -> None:
        if logical_role not in {"planner", "round_realizer"}:
            raise RuntimeError(f"simple_append_forbidden_role:{logical_role}:{role_version}")
        if _source_manifest()["manifest_sha256"] != (
                self.freeze["source_manifest"]["manifest_sha256"]):
            raise RuntimeError("simple_append_source_drift_after_freeze")
        if sha256_file(Path(self.freeze["input_path"])) != self.freeze["input_sha256"]:
            raise RuntimeError("simple_append_input_drift_after_freeze")
        if self.first_api_at is None:
            self.first_api_at = _now()
            atomic_json(self.run_root / "run_status.json", {
                "status": "live_requests_started",
                "first_api_request_at": self.first_api_at,
                "code_prompt_input_configuration_frozen": True,
            })


def execute_live_canary(
    *, preflight: Mapping[str, Any], input_path: Path, run_root: Path,
    api_config_path: Path, tokenizer_path: Path,
) -> dict[str, Any]:
    if not preflight.get("valid"):
        raise RuntimeError("simple_append_preflight_not_valid")
    if run_root.exists():
        raise FileExistsError(f"simple_append_run_root_exists:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    freeze = {**dict(preflight),
              "freeze_version": "simple_append_live_freeze_v1",
              "frozen_at": _now(), "fresh": True, "no_resume": True}
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "run_status.json", {
        "status": "initialized", "created_at": _now(), "api_calls": 0})

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    token_counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    client = OpenAICompatibleChallengerClient(_effective_api_config(api_config_path))
    ledger = RequestLedger(
        run_root, manifest_sha256=freeze["source_manifest"]["manifest_sha256"],
        backend_mode="live")
    guard = FreezeGuard(run_root=run_root, freeze=freeze)
    caller = SimpleAppendLiveCaller(
        run_root=run_root, ledger=ledger, client=client,
        token_counter=token_counter, root_id=str(freeze["root_id"]),
        context_limit=CONTEXT_LIMIT, before_request=guard.before_request)
    failure_stage: str | None = None
    result: dict[str, Any] = {}
    planner_scene_count = 0
    try:
        parent = json.loads(input_path.read_text(encoding="utf-8"))
        tail = build_tail_context(parent, session_count=2)
        atomic_json(run_root / "tail_context.json", tail)
        planner_prompt = build_planner_prompt(tail)
        atomic_json(run_root / "planner/prompt_manifest.json", {
            key: planner_prompt[key] for key in planner_prompt if key != "prompt"})
        parsed_plan, _, parse_errors = caller.call_json(
            request_id="planner_attempt_01", role="simple_append_planner_v1",
            prompt=planner_prompt["prompt"],
            prompt_version=planner_prompt["prompt_version"],
            max_tokens=PLANNER_MAX_TOKENS)
        planner_validation = validate_planner_output(
            parsed_plan, parent_speakers=tail["parent_speakers"])
        planner_validation["parse_errors"] = parse_errors
        atomic_json(run_root / "planner/validation.json", planner_validation)
        atomic_json(run_root / "planner/planner_output.json", parsed_plan)
        if parse_errors or not planner_validation["valid"]:
            failure_stage = "Planner contract"
            raise RuntimeError("simple_append_planner_contract_failed")
        compiled = compile_round_plan(parsed_plan, tail_context=tail)
        planner_scene_count = len(compiled["scenes"])
        atomic_json(run_root / "planner/compiled_round_plan.json", compiled)

        realizer_prompt = build_round_realizer_prompt(
            compiled_plan=compiled, tail_context=tail)
        atomic_json(run_root / "realizer/prompt_manifest.json", {
            key: realizer_prompt[key] for key in realizer_prompt if key != "prompt"})
        parsed_realizer, _, realizer_parse_errors = caller.call_json(
            request_id="round_realizer_attempt_01",
            role="simple_append_round_realizer_v1",
            prompt=realizer_prompt["prompt"],
            prompt_version=realizer_prompt["prompt_version"],
            max_tokens=ROUND_REALIZER_MAX_TOKENS)
        normalized = normalize_round_realizer_output(
            parsed_realizer, compiled_plan=compiled,
            parent_utterances=[row["text"] for row in parent["dialogue"]["turns"]],
            parent_turn_refs=[row["turn_id"] for row in parent["dialogue"]["turns"]])
        normalized["parse_errors"] = realizer_parse_errors
        atomic_json(run_root / "realizer/validation.json", normalized)
        atomic_json(run_root / "realizer/round_realizer_output.json", parsed_realizer)
        if realizer_parse_errors or not normalized["valid"]:
            failure_stage = "Round Realizer contract"
            raise RuntimeError("simple_append_round_realizer_contract_failed")

        child = normalized["child"]
        atomic_json(run_root / "child_dialogue.json", child)
        expanded = assemble_expanded_dialogue(parent=parent, child=child)
        expanded_validation = validate_expanded_dialogue(expanded, parent=parent)
        if not expanded_validation["valid"]:
            failure_stage = "Parent Child assembly"
            raise RuntimeError("simple_append_assembly_failed")
        atomic_json(run_root / "expanded_dialogue.json", expanded)
        atomic_text(run_root / "expanded_dialogue.md",
                    expanded_dialogue_markdown(expanded))
        mcq_prompt = build_mcq_batch_prompt(expanded)
        atomic_text(run_root / "qa/mcq_batch_prompt.txt", mcq_prompt["prompt"])
        atomic_json(run_root / "qa/mcq_batch_prompt_manifest.json", {
            key: mcq_prompt[key] for key in mcq_prompt if key != "prompt"})
        result = {
            "planner_validation": planner_validation,
            "compiled_plan": compiled,
            "realizer_validation": normalized,
            "expanded_validation": expanded_validation,
            "expanded_dialogue": expanded,
            "mcq_prompt_manifest": {
                key: mcq_prompt[key] for key in mcq_prompt if key != "prompt"},
        }
    except Exception as error:
        result["runtime_error"] = f"{type(error).__name__}:{error}"

    usage = aggregate_usage(caller.records)
    success = (failure_stage is None and
               result.get("expanded_validation", {}).get("valid") is True)
    summary = {
        "summary_version": "simple_append_live_canary_summary_v1",
        "status": "completed" if success else "failed",
        "failure_stage": failure_stage,
        "run_root": str(run_root),
        "root_id": freeze["root_id"],
        "input_sha256": freeze["input_sha256"],
        "source_manifest_sha256": freeze["source_manifest"]["manifest_sha256"],
        "api_usage": usage,
        "planner_scene_count": planner_scene_count,
        "child_turn_count": (result.get("expanded_validation") or {}).get(
            "child_turn_count", 0),
        "long_scene_count": (result.get("realizer_validation") or {}).get(
            "long_scene_count", 0),
        "generation_ready": success,
        "qa_source_ready": success,
        "qa_pipeline_ready": False,
        "memory_required": False,
        "cache_reuse": 0,
        "imported_response_reuse": 0,
        "runtime_error": result.get("runtime_error"),
        "completed_at": _now(),
    }
    atomic_json(run_root / "reports/simple_append_live_canary_summary.json", summary)
    atomic_text(run_root / "reports/simple_append_live_canary_summary.md", "\n".join([
        "# Simple Append v1 Live Canary",
        "",
        f"Status: `{summary['status']}`",
        f"Failure stage: `{summary['failure_stage']}`",
        f"Planner scenes: {summary['planner_scene_count']}",
        f"Child turns: {summary['child_turn_count']}",
        f"Long scenes: {summary['long_scene_count']}",
        f"API requests: {usage['total_requests']}",
        f"Total tokens: {usage['total_tokens']}",
        f"Generation ready: `{str(success).lower()}`",
        "QA was not executed; only the complete-dialogue batch prompt was materialized.",
        "",
    ]))
    atomic_json(run_root / "run_status.json", {
        "status": summary["status"], "failure_stage": failure_stage,
        "generation_ready": success, "qa_source_ready": success,
        "qa_pipeline_ready": False, "api_usage": usage,
        "completed_at": summary["completed_at"],
    })
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Single-use simple append live canary")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    preflight = run_preflight(
        input_path=args.input.resolve(), run_root=args.run_root.resolve(),
        api_config_path=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve())
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if preflight["valid"] else 2
    if not preflight["valid"]:
        print(json.dumps({"status": "PRE_API_PREFLIGHT_FAILED", **preflight},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    summary = execute_live_canary(
        preflight=preflight, input_path=args.input.resolve(),
        run_root=args.run_root.resolve(), api_config_path=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
