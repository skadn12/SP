"""Single-use Chain Initializer v2 probe; never enters Planner or Fresh R1."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..live_backends import load_challenger_config
from ..simple_append_memorax_v1_fullr1_durable_devdiv_probe_v1.context import (
    build_fresh_r1_context,
)
from .contracts import (
    audit_episode_146700_initialization, materialize_event_chains,
    validate_initializer_output,
)
from .live_runtime import (
    IncrementalEventChainLiveCaller, NoRetryOpenAIClient, aggregate_usage,
)
from .preflight import _markdown as base_preflight_markdown
from .preflight import run_preflight
from .prompts import build_chain_initializer_prompt
from .settings import (
    CONTEXT_LIMIT, DEFAULT_API_CONFIG, DEFAULT_INITIALIZER_RUN_ROOT,
    DEFAULT_INPUT, DEFAULT_OVERVIEW, DEFAULT_REPORT_ROOT, DEFAULT_TOKENIZER,
    INITIALIZER_MAX_TOKENS, LIVE_TIMEOUT_SECONDS,
    protected_integrity_snapshot, source_manifest,
)


RUNNER_VERSION = "simple_append_local_event_chain_initializer_live_v2"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_object:{path}")
    return value


def run_initializer_only_preflight(
    *, input_path: Path = DEFAULT_INPUT, overview_path: Path = DEFAULT_OVERVIEW,
    run_root: Path = DEFAULT_INITIALIZER_RUN_ROOT,
    api_config_path: Path = DEFAULT_API_CONFIG,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
) -> dict[str, Any]:
    report = run_preflight(
        input_path=input_path, overview_path=overview_path, run_root=run_root,
        api_config_path=api_config_path, tokenizer_path=tokenizer_path)
    checks = dict(report["checks"])
    checks["initializer_only_live_role"] = True
    checks["planner_live_calls_frozen_zero"] = True
    checks["realizer_live_calls_frozen_zero"] = True
    checks["delta_updater_live_calls_frozen_zero"] = True
    checks["initializer_live_budget_one"] = True
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        **report,
        "preflight_version": "simple_append_local_event_chain_initializer_only_preflight_v2",
        "decision": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "request_budget": {
            "chain_initializer": 1, "planner": 0, "round_realizer": 0,
            "delta_updater": 0, "total": 1, "retries": 0,
            "memorax": 0, "qa": 0, "solver": 0,
        },
        "max_completion_tokens": {"chain_initializer": INITIALIZER_MAX_TOKENS},
        "api_calls": 0,
    }


class InitializerFreezeGuard:
    def __init__(self, *, run_root: Path, freeze: Mapping[str, Any]) -> None:
        self.run_root = run_root
        self.freeze = dict(freeze)
        self.first_api_at: str | None = None

    def before_request(self, logical_role: str, role_version: str) -> None:
        if logical_role != "chain_initializer":
            raise RuntimeError(f"initializer_only_forbidden_role:{logical_role}:{role_version}")
        if source_manifest()["manifest_sha256"] != (
                self.freeze["source_manifest"]["manifest_sha256"]):
            raise RuntimeError("initializer_only_source_drift_after_freeze")
        for path_key, sha_key in (
            ("input_path", "input_sha256"),
            ("overview_path", "overview_sha256"),
            ("api_config_path", "api_config_sha256"),
        ):
            if sha256_file(Path(self.freeze[path_key])) != self.freeze[sha_key]:
                raise RuntimeError(f"initializer_only_{path_key}_drift_after_freeze")
        if self.first_api_at is None:
            self.first_api_at = _now()
            atomic_json(self.run_root / "run_status.json", {
                "status": "initializer_request_started",
                "first_api_request_at": self.first_api_at,
                "code_prompt_input_configuration_frozen": True,
            })


def execute_initializer_only(
    *, preflight: Mapping[str, Any], input_path: Path, overview_path: Path,
    run_root: Path, api_config_path: Path, tokenizer_path: Path,
) -> dict[str, Any]:
    if preflight.get("decision") != "pass":
        raise RuntimeError("INITIALIZER_ONLY_PRE_API_PREFLIGHT_FAILED")
    if run_root.exists():
        raise FileExistsError(f"initializer_only_run_root_exists:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in (
        "initializer", "reports", "attempts", "raw/prompts",
        "raw/backend_responses", "ledger",
    ):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    freeze = {
        **dict(preflight), "freeze_version": RUNNER_VERSION,
        "frozen_at": _now(), "fresh": True, "no_resume": True,
        "only_live_roles": ["chain_initializer"],
        "planner_realizer_delta_disabled": True,
        "memorax_mode": "frozen_pre_generation_overview_only",
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "run_status.json", {
        "status": "initialized", "created_at": _now(), "api_calls": 0,
    })
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    config = replace(
        load_challenger_config(api_config_path), timeout_seconds=LIVE_TIMEOUT_SECONDS)
    ledger = RequestLedger(
        run_root, manifest_sha256=freeze["source_manifest"]["manifest_sha256"],
        backend_mode="live")
    caller = IncrementalEventChainLiveCaller(
        run_root=run_root, ledger=ledger, client=NoRetryOpenAIClient(config),
        token_counter=counter, root_id=str(freeze["root_id"]),
        context_limit=CONTEXT_LIMIT,
        before_request=InitializerFreezeGuard(
            run_root=run_root, freeze=freeze).before_request,
        total_budget_limit=1)
    failure_stage: str | None = None
    runtime_error: str | None = None
    validation: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    chain_set: dict[str, Any] = {}
    try:
        parent = _load(input_path)
        overview = _load(overview_path)
        tail = build_fresh_r1_context(parent, token_counter=counter)
        atomic_json(run_root / "selected_parent_context.json", tail)
        atomic_json(run_root / "memory_pre_generation_overview.json", overview)
        prompt = build_chain_initializer_prompt(
            tail_context=tail, memory_overview=overview)
        atomic_text(run_root / "initializer/prompt.txt", str(prompt["prompt"]))
        atomic_json(run_root / "initializer/prompt_manifest.json", {
            key: value for key, value in prompt.items() if key != "prompt"
        })
        output, _, parse_errors = caller.call_json(
            request_id="chain_initializer_attempt_01",
            role="event_chain_initializer_v2", prompt=str(prompt["prompt"]),
            prompt_version=str(prompt["prompt_version"]),
            max_tokens=INITIALIZER_MAX_TOKENS)
        atomic_json(run_root / "initializer/output.json", output or {})
        validation = validate_initializer_output(
            output, parent_turns=parent["dialogue"]["turns"])
        validation["parse_errors"] = parse_errors
        atomic_json(run_root / "initializer/validation.json", validation)
        if parse_errors or not validation["valid"]:
            failure_stage = "chain_initialization_contract"
            raise RuntimeError("initializer_only_contract_failed")
        chain_set = materialize_event_chains(
            output or {}, parent_turns=parent["dialogue"]["turns"],
            root_id=parent["root_id"],
            parent_dialogue_sha256=parent["dialogue"]["dialogue_hash"])
        atomic_json(run_root / "initializer/initial_event_chains.json", chain_set)
        audit = audit_episode_146700_initialization(chain_set)
        atomic_json(run_root / "initializer/canary_semantic_gate.json", audit)
        if not audit["valid"]:
            failure_stage = "chain_initialization_semantic"
            raise RuntimeError("initializer_only_semantic_checkpoint_failed")
    except Exception as error:
        runtime_error = f"{type(error).__name__}:{error}"
        if failure_stage is None:
            failure_stage = "runtime"
    usage = aggregate_usage(caller.records)
    integrity_after = protected_integrity_snapshot()
    integrity = {
        "input_unchanged": sha256_file(input_path) == freeze["input_sha256"],
        "overview_unchanged": sha256_file(overview_path) == freeze["overview_sha256"],
        "api_config_unchanged": sha256_file(api_config_path) == freeze["api_config_sha256"],
        "source_manifest_unchanged": source_manifest()["manifest_sha256"] == (
            freeze["source_manifest"]["manifest_sha256"]),
        "protected_integrity_unchanged": integrity_after == freeze["protected_integrity"],
        "after": integrity_after,
    }
    success = failure_stage is None and audit.get("valid") is True
    summary = {
        "summary_version": RUNNER_VERSION,
        "status": "completed" if success else "failed",
        "failure_stage": failure_stage,
        "runtime_error": runtime_error,
        "run_root": str(run_root),
        "root_id": freeze["root_id"],
        "api_usage": usage,
        "initializer_calls": usage["roles"]["chain_initializer"]["requests"],
        "planner_calls": 0, "realizer_calls": 0, "delta_updater_calls": 0,
        "retries": 0, "memorax_calls": 0, "qa_calls": 0, "solver_calls": 0,
        "candidate_count": validation.get("candidate_count", 0),
        "continuation_link_count": validation.get("link_count", 0),
        "initial_chain_count": len(chain_set.get("chains") or []),
        "initialization_supported": success,
        "semantic_gate": audit,
        "integrity": integrity,
        "completed_at": _now(),
    }
    atomic_json(run_root / "reports/INITIALIZER_ONLY_LIVE_REPORT.json", summary)
    atomic_text(run_root / "reports/INITIALIZER_ONLY_LIVE_REPORT.md", "\n".join([
        "# Local Event Chain Initializer-only Live Report", "",
        f"- Status: `{summary['status']}`",
        f"- Failure stage: `{summary['failure_stage']}`",
        f"- Initializer calls / retries: `{summary['initializer_calls']} / 0`",
        f"- Planner / Realizer / Delta calls: `0 / 0 / 0`",
        f"- Candidates / links / chains: `{summary['candidate_count']} / "
        f"{summary['continuation_link_count']} / {summary['initial_chain_count']}`",
        f"- Total tokens: `{usage['total_tokens']}`",
        f"- Initialization supported: `{str(success).lower()}`",
        "- Memorax, R1, QA, Solver, Repair, R2/R3, and training calls: `0`", "",
    ]))
    atomic_json(run_root / "run_status.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--overview", type=Path, default=DEFAULT_OVERVIEW)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_INITIALIZER_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    preflight = run_initializer_only_preflight(
        input_path=args.input.resolve(), overview_path=args.overview.resolve(),
        run_root=args.run_root.resolve(), api_config_path=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve())
    if args.preflight_only:
        atomic_json(args.report_root / "INITIALIZER_V2_ONLY_PREFLIGHT.json", preflight)
        atomic_text(
            args.report_root / "INITIALIZER_V2_ONLY_PREFLIGHT.md",
            base_preflight_markdown(preflight))
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if preflight["decision"] == "pass" else 2
    if preflight["decision"] != "pass":
        print(json.dumps(
            {"status": "PRE_API_PREFLIGHT_FAILED", **preflight},
            ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    summary = execute_initializer_only(
        preflight=preflight, input_path=args.input.resolve(),
        overview_path=args.overview.resolve(), run_root=args.run_root.resolve(),
        api_config_path=args.api_config.resolve(), tokenizer_path=args.tokenizer.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "execute_initializer_only", "run_initializer_only_preflight",
]
