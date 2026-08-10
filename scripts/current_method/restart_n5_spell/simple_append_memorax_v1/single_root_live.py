"""Fresh single-root live canary for the optional Memorax context layer.

This entry point is intentionally separate from ``simple_append_v1``.  It
performs one Planner call and one round Realizer call, but only materializes
the Memory-augmented MCQ prompt; QA Writer, Solver, and training are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient
from ..simple_append_v1.assembly import expanded_dialogue_markdown
from ..simple_append_v1.context import build_tail_context
from ..simple_append_v1.live_canary import (
    CONTEXT_LIMIT,
    DEFAULT_API_CONFIG,
    DEFAULT_INPUT,
    DEFAULT_TOKENIZER,
    PLANNER_MAX_TOKENS,
    ROUND_REALIZER_MAX_TOKENS,
    _effective_api_config,
)
from ..simple_append_v1.live_runtime import SimpleAppendLiveCaller, aggregate_usage
from .client import MemoraxClient, load_memorax_config
from .memory_context import compile_memory_overview, normalize_bucket_snapshot
from .payloads import build_bucket_search_payload, memorax_user_id
from .prompts import build_memory_augmented_planner_prompt
from .workflow import MemoryWorkflowError, execute_single_round


RUNNER_VERSION = "simple_append_memorax_single_root_live_v2"
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_simple_append_memorax_v1_live_canary_2"
)
DEFAULT_MEMORAX_CREDENTIALS = Path("/root/.config/memorax/credentials.env")
DEFAULT_PREFLIGHT_REPORT_ROOT = Path(__file__).with_name("reports")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_manifest() -> dict[str, Any]:
    spell_root = Path(__file__).resolve().parent.parent
    roots = [spell_root / "simple_append_v1", spell_root / "simple_append_memorax_v1"]
    files = sorted(
        path for root in roots for path in root.glob("*.py") if path.is_file())
    rows = [{"path": str(path), "sha256": sha256_file(path)} for path in files]
    payload = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "files": rows,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _empty_bucket_response() -> dict[str, Any]:
    return {
        "success": True,
        "data": {"buckets": {
            "person": {}, "location": {}, "event": {},
            "item": {}, "dreaming": {},
        }},
    }


def run_preflight(
    *, input_path: Path, run_root: Path, api_config_path: Path,
    tokenizer_path: Path, memorax_credentials_path: Path,
    allow_insecure_memorax_http: bool,
    reuse_existing_parent_memory: bool = False,
    wait_for_memory_completion: bool = True,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "run_root_absent": not run_root.exists(),
        "input_exists": input_path.is_file(),
        "api_config_exists": api_config_path.is_file(),
        "tokenizer_exists": tokenizer_path.exists(),
        "memorax_credentials_exist": memorax_credentials_path.is_file(),
        "memorax_platform_memory": True,
        "memorax_async_add_enabled": True,
        "api_calls_zero": True,
    }
    details: dict[str, Any] = {}
    parent: dict[str, Any] = {}
    memorax_config = None
    llm_config = None
    planner_prompt: dict[str, Any] = {}
    try:
        parent = json.loads(input_path.read_text(encoding="utf-8"))
        tail = build_tail_context(parent, session_count=2)
        root_id = str(parent["root_id"])
        query = build_bucket_search_payload(root_id=root_id)
        snapshot = normalize_bucket_snapshot(
            _empty_bucket_response(), root_id=root_id,
            user_id=memorax_user_id(root_id))
        overview = compile_memory_overview(
            snapshot,
            recent_dialogue_text="\n".join(
                str(row["text"]) for row in tail["selected_turns"]),
        )
        planner_prompt = build_memory_augmented_planner_prompt(tail, overview)
        checks["authoritative_parent_valid"] = True
        checks["last_two_complete_sessions"] = tail["selected_session_count"] == 2
        checks["true_parent_tail_selected"] = (
            tail["parent_tail_turn_id"] == parent["dialogue"]["turns"][-1]["turn_id"])
        checks["root_isolated_user_id"] = query["payload"]["user_id"] == (
            f"gxy-selfplay:{root_id}")
        checks["memory_planner_prompt_fits"] = (
            planner_prompt["token_estimate"] + PLANNER_MAX_TOKENS <= CONTEXT_LIMIT)
    except Exception as error:
        checks["authoritative_parent_valid"] = False
        checks["last_two_complete_sessions"] = False
        checks["true_parent_tail_selected"] = False
        checks["root_isolated_user_id"] = False
        checks["memory_planner_prompt_fits"] = False
        details["input_error"] = f"{type(error).__name__}:{error}"
    try:
        memorax_config = load_memorax_config(
            credentials_path=memorax_credentials_path,
            allow_insecure_http=allow_insecure_memorax_http,
        )
        checks["memorax_platform_memory"] = memorax_config.platform == "memory"
        checks["memorax_transport_authorized"] = (
            memorax_config.normalized_base_url.startswith("https://") or
            allow_insecure_memorax_http)
    except Exception as error:
        checks["memorax_transport_authorized"] = False
        details["memorax_configuration_error"] = f"{type(error).__name__}:{error}"
    try:
        llm_config = _effective_api_config(api_config_path)
        checks["llm_model_configured"] = bool(llm_config.model)
        checks["llm_timeout_frozen"] = llm_config.timeout_seconds == 600
    except Exception as error:
        checks["llm_model_configured"] = False
        checks["llm_timeout_frozen"] = False
        details["llm_configuration_error"] = f"{type(error).__name__}:{error}"
    manifest = _source_manifest()
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "preflight_version": RUNNER_VERSION,
        "decision": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "details": details,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path) if input_path.is_file() else None,
        "root_id": parent.get("root_id") if parent else None,
        "run_root": str(run_root.resolve()),
        "api_config_path": str(api_config_path.resolve()),
        "api_config_sha256": (
            sha256_file(api_config_path) if api_config_path.is_file() else None),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "memorax_credentials_path": str(memorax_credentials_path.resolve()),
        "memorax_configuration": (
            memorax_config.redacted() if memorax_config is not None else None),
        "llm_configuration": ({
            "model": llm_config.model,
            "base_url": llm_config.base_url,
            "temperature": llm_config.temperature,
            "timeout_seconds": llm_config.timeout_seconds,
            "api_key_recorded": False,
        } if llm_config is not None else None),
        "source_manifest": manifest,
        "planner_prompt_sha256": planner_prompt.get("prompt_sha256"),
        "planner_prompt_token_estimate": planner_prompt.get("token_estimate"),
        "request_budget": {
            "planner": 1,
            "round_realizer": 1,
            "memorax_add": 1 if reuse_existing_parent_memory else 2,
            "memorax_search": 2,
            "memorax_add_status_max": (
                (18 if reuse_existing_parent_memory else 36)
                if wait_for_memory_completion else 0),
            "qa_writer": 0,
            "solver": 0,
        },
        "parent_memory_mode": (
            "reuse_existing" if reuse_existing_parent_memory else (
                "initialize_async_verified" if wait_for_memory_completion
                else "initialize_async_pending")),
        "wait_for_memory_completion": bool(wait_for_memory_completion),
        "max_completion_tokens": {
            "planner": PLANNER_MAX_TOKENS,
            "round_realizer": ROUND_REALIZER_MAX_TOKENS,
        },
        "cache_reuse": 0,
        "imported_response_reuse": 0,
        "api_calls": 0,
    }


def _preflight_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Simple Append Memorax Single-root Live Preflight",
        "",
        f"- Decision: `{report.get('decision')}`",
        f"- API calls: `{report.get('api_calls')}`",
        f"- Run root: `{report.get('run_root')}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- {key}: `{str(value).lower()}`"
        for key, value in (report.get("checks") or {}).items())
    lines.extend([
        "",
        "No run root or outbound request is created by preflight-only mode.",
        "",
    ])
    return "\n".join(lines)


class FreezeGuard:
    def __init__(
        self, *, run_root: Path, freeze: Mapping[str, Any],
        input_path: Path, api_config_path: Path,
    ) -> None:
        self.run_root = run_root
        self.freeze = dict(freeze)
        self.input_path = input_path
        self.api_config_path = api_config_path
        self.first_api_at: str | None = None

    def before_request(self, logical_role: str, role_version: str) -> None:
        allowed = {
            "planner", "round_realizer", "parent_add", "child_add",
            "pre_generation_search", "post_generation_search", "add_status",
        }
        if logical_role not in allowed:
            raise RuntimeError(f"memorax_single_root_forbidden_role:{logical_role}")
        if _source_manifest()["manifest_sha256"] != (
                self.freeze["source_manifest"]["manifest_sha256"]):
            raise RuntimeError("memorax_single_root_source_drift")
        if sha256_file(self.input_path) != self.freeze["input_sha256"]:
            raise RuntimeError("memorax_single_root_input_drift")
        if sha256_file(self.api_config_path) != self.freeze["api_config_sha256"]:
            raise RuntimeError("memorax_single_root_llm_config_drift")
        if self.first_api_at is None:
            self.first_api_at = _now()
            atomic_json(self.run_root / "run_status.json", {
                "status": "live_requests_started",
                "first_api_request_at": self.first_api_at,
                "code_prompt_input_configuration_frozen": True,
            })


def _artifact_sink(run_root: Path):
    def write(name: str, value: Mapping[str, Any]) -> None:
        target = run_root / name
        if (name.endswith("/prompt") or name.endswith("_prompt")) and \
                isinstance(value.get("prompt"), str):
            atomic_text(target.with_suffix(".txt"), str(value["prompt"]))
            atomic_json(target.with_name(target.name + "_manifest.json"), {
                key: row for key, row in value.items() if key != "prompt"
            })
            return
        if name == "workflow_result":
            atomic_json(run_root / "reports/workflow_result_summary.json", {
                "workflow_version": value.get("workflow_version"),
                "root_id": value.get("root_id"),
                "round_index": value.get("round_index"),
                "memory_call_count": value.get("memory_call_count"),
                "generation_ready": value.get("generation_ready"),
                "qa_prompt_ready": value.get("qa_prompt_ready"),
                "qa_pipeline_ready": value.get("qa_pipeline_ready"),
                "memory_is_qa_fact_source": value.get("memory_is_qa_fact_source"),
            })
            return
        atomic_json(target.with_suffix(".json"), value)
    return write


def execute_live(
    *, preflight: Mapping[str, Any], input_path: Path, run_root: Path,
    api_config_path: Path, tokenizer_path: Path,
    memorax_credentials_path: Path, allow_insecure_memorax_http: bool,
    reuse_existing_parent_memory: bool = False,
    wait_for_memory_completion: bool = True,
) -> dict[str, Any]:
    if preflight.get("decision") != "pass":
        raise RuntimeError("MEMORAX_SINGLE_ROOT_PRE_API_PREFLIGHT_FAILED")
    if run_root.exists():
        raise FileExistsError(f"memorax_single_root_run_exists:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("planner", "realizer", "memory", "qa", "reports"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    freeze = {
        **dict(preflight),
        "freeze_version": RUNNER_VERSION,
        "frozen_at": _now(),
        "fresh": True,
        "no_resume": True,
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "run_status.json", {
        "status": "initialized", "api_calls": 0})

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    token_counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    ledger = RequestLedger(
        run_root,
        manifest_sha256=freeze["source_manifest"]["manifest_sha256"],
        backend_mode="live",
    )
    guard = FreezeGuard(
        run_root=run_root, freeze=freeze, input_path=input_path,
        api_config_path=api_config_path)
    caller = SimpleAppendLiveCaller(
        run_root=run_root,
        ledger=ledger,
        client=OpenAICompatibleChallengerClient(
            _effective_api_config(api_config_path)),
        token_counter=token_counter,
        root_id=str(freeze["root_id"]),
        context_limit=CONTEXT_LIMIT,
        before_request=guard.before_request,
    )
    memorax = MemoraxClient(load_memorax_config(
        credentials_path=memorax_credentials_path,
        allow_insecure_http=allow_insecure_memorax_http,
    ))

    def planner_call(prompt: Mapping[str, Any]) -> Mapping[str, Any]:
        parsed, _, errors = caller.call_json(
            request_id="planner_attempt_01",
            role="simple_append_planner_v1",
            prompt=str(prompt["prompt"]),
            prompt_version=str(prompt["prompt_version"]),
            max_tokens=PLANNER_MAX_TOKENS,
        )
        if errors:
            raise MemoryWorkflowError("planner_parse", errors)
        return parsed or {}

    def realizer_call(prompt: Mapping[str, Any]) -> Mapping[str, Any]:
        parsed, _, errors = caller.call_json(
            request_id="round_realizer_attempt_01",
            role="simple_append_round_realizer_v1",
            prompt=str(prompt["prompt"]),
            prompt_version=str(prompt["prompt_version"]),
            max_tokens=ROUND_REALIZER_MAX_TOKENS,
        )
        if errors:
            raise MemoryWorkflowError("realizer_parse", errors)
        return parsed or {}

    failure_stage: str | None = None
    runtime_error: str | None = None
    result: dict[str, Any] = {}
    try:
        parent = json.loads(input_path.read_text(encoding="utf-8"))
        result = execute_single_round(
            parent=parent,
            memorax=memorax,
            planner_call=planner_call,
            realizer_call=realizer_call,
            initialize_parent_memory=not reuse_existing_parent_memory,
            artifact_sink=_artifact_sink(run_root),
            before_memory_request=lambda stage, request_id: guard.before_request(
                stage, request_id),
            async_add=True,
            wait_for_memory_completion=wait_for_memory_completion,
        )
        atomic_text(
            run_root / "expanded_dialogue.md",
            expanded_dialogue_markdown(result["expanded_dialogue"]),
        )
    except MemoryWorkflowError as error:
        failure_stage = error.stage
        runtime_error = str(error)
    except Exception as error:
        failure_stage = "runtime"
        runtime_error = f"{type(error).__name__}:{error}"

    usage = aggregate_usage(caller.records)
    generation_ready = bool(result.get("generation_ready"))
    integrity = {
        "input_unchanged": sha256_file(input_path) == freeze["input_sha256"],
        "source_manifest_unchanged": _source_manifest()["manifest_sha256"] == (
            freeze["source_manifest"]["manifest_sha256"]),
        "llm_config_unchanged": sha256_file(api_config_path) == (
            freeze["api_config_sha256"]),
    }
    summary = {
        "summary_version": RUNNER_VERSION,
        "status": "completed" if generation_ready else "failed",
        "failure_stage": failure_stage,
        "runtime_error": runtime_error,
        "run_root": str(run_root),
        "root_id": freeze["root_id"],
        "llm_api_usage": usage,
        "memorax_calls": len(memorax.calls),
        "memorax_call_records": memorax.calls,
        "child_turn_count": int(
            (result.get("expanded_dialogue_validation") or {}).get(
                "child_turn_count") or 0),
        "planner_scene_count": len(
            (result.get("compiled_plan") or {}).get("scenes") or []),
        "generation_ready": generation_ready,
        "qa_prompt_ready": bool(result.get("qa_prompt_ready")),
        "qa_pipeline_ready": False,
        "memory_is_qa_fact_source": False,
        "memory_write_verified": bool(result.get("memory_write_verified")),
        "memory_context_degraded": bool(result.get("memory_context_degraded")),
        "pending_memory_writes": list(result.get("pending_memory_writes") or []),
        "memory_write_states": list(result.get("memory_write_states") or []),
        "memory_visibility_verified": bool(
            result.get("memory_visibility_verified")),
        "parent_memory_mode": (
            "reuse_existing" if reuse_existing_parent_memory else (
                "initialize_async_verified" if wait_for_memory_completion
                else "initialize_async_pending")),
        "wait_for_memory_completion": bool(wait_for_memory_completion),
        "qa_writer_calls": 0,
        "solver_calls": 0,
        "cache_reuse": 0,
        "imported_response_reuse": 0,
        "integrity": integrity,
        "completed_at": _now(),
    }
    atomic_json(run_root / "reports/SIMPLE_APPEND_MEMORAX_LIVE_REPORT.json", summary)
    atomic_text(run_root / "reports/SIMPLE_APPEND_MEMORAX_LIVE_REPORT.md", "\n".join([
        "# Simple Append Memorax Single-root Live Canary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Failure stage: `{summary['failure_stage']}`",
        f"- Planner scenes: `{summary['planner_scene_count']}`",
        f"- Child turns: `{summary['child_turn_count']}`",
        f"- LLM requests: `{usage['total_requests']}`",
        f"- Memorax calls: `{summary['memorax_calls']}`",
        f"- Generation ready: `{str(generation_ready).lower()}`",
        f"- QA prompt ready: `{str(summary['qa_prompt_ready']).lower()}`",
        "- QA Writer and Solver were not called.",
        "- Complete Parent+Child remains the only QA fact source.",
        "",
    ]))
    atomic_json(run_root / "run_status.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--memorax-credentials", type=Path,
        default=DEFAULT_MEMORAX_CREDENTIALS)
    parser.add_argument(
        "--preflight-report-root", type=Path,
        default=DEFAULT_PREFLIGHT_REPORT_ROOT)
    parser.add_argument("--allow-insecure-memorax-http", action="store_true")
    parser.add_argument("--reuse-existing-parent-memory", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    preflight = run_preflight(
        input_path=args.input.resolve(),
        run_root=args.run_root.resolve(),
        api_config_path=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve(),
        memorax_credentials_path=args.memorax_credentials.resolve(),
        allow_insecure_memorax_http=args.allow_insecure_memorax_http,
        reuse_existing_parent_memory=args.reuse_existing_parent_memory,
    )
    if args.preflight_only:
        atomic_json(
            args.preflight_report_root / "MEMORAX_SINGLE_ROOT_LIVE_PREFLIGHT.json",
            preflight)
        atomic_text(
            args.preflight_report_root / "MEMORAX_SINGLE_ROOT_LIVE_PREFLIGHT.md",
            _preflight_markdown(preflight))
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if preflight["decision"] == "pass" else 2
    if preflight["decision"] != "pass":
        print(json.dumps(
            {"status": "PRE_API_PREFLIGHT_FAILED", **preflight},
            ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    summary = execute_live(
        preflight=preflight,
        input_path=args.input.resolve(),
        run_root=args.run_root.resolve(),
        api_config_path=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve(),
        memorax_credentials_path=args.memorax_credentials.resolve(),
        allow_insecure_memorax_http=args.allow_insecure_memorax_http,
        reuse_existing_parent_memory=args.reuse_existing_parent_memory,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
