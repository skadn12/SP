"""Five-call Stage A: pure Parent-session local event extraction, then stop."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..live_backends import load_challenger_config
from .live_runtime import NoRetryOpenAIClient, load_json_object_strict
from .local_event_contract import (
    LOCAL_EVENT_SCHEMA_VERSION, audit_episode_146700_local_candidates,
    compile_parent_session_windows, materialize_local_candidates,
    validate_local_event_output,
)
from .local_event_prompts import build_local_event_prompt
from .settings import (
    CONTEXT_LIMIT, DEFAULT_API_CONFIG, DEFAULT_INPUT,
    DEFAULT_LOCAL_EXTRACTION_RUN_ROOT, DEFAULT_OVERVIEW, DEFAULT_REPORT_ROOT,
    DEFAULT_TOKENIZER, FROZEN_INPUT_SHA256, FROZEN_OVERVIEW_SHA256,
    LIVE_TIMEOUT_SECONDS, LOCAL_EVENT_MAX_TOKENS,
    protected_integrity_snapshot, source_manifest,
)


RUNNER_VERSION = "simple_append_session_local_event_stage_a_live_v1"
PREFLIGHT_VERSION = "simple_append_session_local_event_stage_a_preflight_v1"
WINDOW_CALLS = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_object:{path}")
    return value


def _parent_speakers(parent: Mapping[str, Any]) -> list[str]:
    return list(dict.fromkeys(
        str(row["speaker"]) for row in parent["dialogue"]["turns"]))


def _fixture_outputs(windows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    root = "cc-train-episode-146700"
    speakers = ["Co-workers A", "Co-workers B"]
    events = [
        [{
            "event_summary": "B proposed renewed soccer practice and A accepted",
            "event_state": "A accepted B's suggestion to resume regular soccer practice.",
            "participants": speakers,
            "evidence_turn_refs": [f"{root}:P01:T006", f"{root}:P01:T007"],
        }],
        [{
            "event_summary": "A told B that the store has a new cashier",
            "event_state": "A knows there is a new cashier but has not worked with her.",
            "participants": speakers,
            "evidence_turn_refs": [f"{root}:P02:T001", f"{root}:P02:T003"],
        }],
        [{
            "event_summary": "A apologized to a friend after being rude",
            "event_state": "A apologized and felt relief afterward.",
            "participants": speakers,
            "evidence_turn_refs": [f"{root}:P03:T003", f"{root}:P03:T005"],
        }],
        [{
            "event_summary": "A arranged a doctor visit and B accepted shift coverage",
            "event_state": "A plans to see a doctor and B accepted work coverage.",
            "participants": speakers,
            "evidence_turn_refs": [f"{root}:P04:T005", f"{root}:P04:T006"],
        }],
        [{
            "event_summary": "A placed an urgent call about a report mistake",
            "event_state": "A is waiting for the boss to answer and intends to disclose the report mistake and take responsibility.",
            "participants": speakers,
            "evidence_turn_refs": [
                f"{root}:P05:T001", f"{root}:P05:T003", f"{root}:P05:T009",
            ],
        }],
    ]
    if len(windows) != len(events):
        raise ValueError("stage_a_fixture_window_count")
    return [{"local_event_schema_version": LOCAL_EVENT_SCHEMA_VERSION, "events": rows}
            for rows in events]


def run_stage_a_preflight(
    *, input_path: Path = DEFAULT_INPUT, overview_path: Path = DEFAULT_OVERVIEW,
    run_root: Path = DEFAULT_LOCAL_EXTRACTION_RUN_ROOT,
    api_config_path: Path = DEFAULT_API_CONFIG,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "run_root_absent": not run_root.exists(),
        "input_sha_frozen": input_path.is_file() and sha256_file(input_path) == FROZEN_INPUT_SHA256,
        "overview_sha_frozen_but_not_prompted": (
            overview_path.is_file() and sha256_file(overview_path) == FROZEN_OVERVIEW_SHA256),
        "api_config_exists": api_config_path.is_file(),
        "tokenizer_exists": tokenizer_path.exists(),
        "api_calls_zero": True, "memorax_calls_zero": True,
        "planner_realizer_delta_calls_zero": True,
        "linking_disabled": True,
    }
    details: dict[str, Any] = {}
    prompts: dict[str, dict[str, Any]] = {}
    parent: dict[str, Any] = {}
    try:
        parent = _load(input_path)
        windows = compile_parent_session_windows(parent)
        checks["five_natural_parent_sessions"] = (
            len(windows) == 5 and [row["window_id"] for row in windows] ==
            ["P01", "P02", "P03", "P04", "P05"])
        checks["all_parent_turns_preserved"] = (
            sum(row["turn_count"] for row in windows) == parent["dialogue"]["turn_count"])
        fixtures = _fixture_outputs(windows)
        validations = [validate_local_event_output(
            output, window=window,
            parent_speakers=_parent_speakers(parent))
            for window, output in zip(windows, fixtures)]
        checks["all_local_fixtures_valid"] = all(row["valid"] for row in validations)
        candidates = materialize_local_candidates(list(zip(windows, fixtures)))
        checks["deterministic_candidate_refs"] = [
            row["candidate_ref"] for row in candidates
        ] == ["P01:E01", "P02:E01", "P03:E01", "P04:E01", "P05:E01"]
        audit = audit_episode_146700_local_candidates(candidates)
        checks["episode_local_event_audit_passes"] = audit["valid"]
        checks["cashier_independent_fixture_present"] = (
            audit["expected_event_matches"]["new_cashier"] == "P02:E01")
        checks["coverage_word_form_preserved"] = (
            "health_accepted_coverage_missing" not in audit["errors"])
        invalid = json.loads(json.dumps(fixtures[0]))
        invalid["events"][0]["evidence_turn_refs"] = [
            "cc-train-episode-146700:P02:T001"]
        checks["cross_session_evidence_rejected"] = not validate_local_event_output(
            invalid, window=windows[0],
            parent_speakers=_parent_speakers(parent))["valid"]
        empty = {"local_event_schema_version": LOCAL_EVENT_SCHEMA_VERSION, "events": []}
        checks["empty_window_output_allowed"] = validate_local_event_output(
            empty, window=windows[0],
            parent_speakers=_parent_speakers(parent))["valid"]
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        for window in windows:
            prompt = build_local_event_prompt(window)
            prompts[window["window_id"]] = prompt
            checks[f"{window['window_id']}_prompt_fits"] = (
                len(tokenizer.encode(prompt["prompt"], add_special_tokens=False)) +
                LOCAL_EVENT_MAX_TOKENS <= CONTEXT_LIMIT)
            checks[f"{window['window_id']}_prompt_has_no_memorax"] = (
                "memorax" not in prompt["prompt"].casefold())
            checks[f"{window['window_id']}_prompt_has_no_link_output"] = (
                "continuation_links\"" not in prompt["prompt"])
    except Exception as error:
        details["preflight_error"] = f"{type(error).__name__}:{error}"
        for name in (
            "five_natural_parent_sessions", "all_parent_turns_preserved",
            "all_local_fixtures_valid", "deterministic_candidate_refs",
            "episode_local_event_audit_passes", "cashier_independent_fixture_present",
            "coverage_word_form_preserved", "cross_session_evidence_rejected",
            "empty_window_output_allowed",
        ):
            checks.setdefault(name, False)
    config = None
    try:
        config = replace(
            load_challenger_config(api_config_path), timeout_seconds=LIVE_TIMEOUT_SECONDS,
            temperature=0.1, enable_thinking=False)
        checks["model_configured"] = bool(config.model)
        checks["stage_a_temperature_0_1"] = config.temperature == 0.1
        checks["stage_a_thinking_disabled"] = config.enable_thinking is False
    except Exception as error:
        checks["model_configured"] = False
        checks["stage_a_temperature_0_1"] = False
        checks["stage_a_thinking_disabled"] = False
        details["api_config_error"] = f"{type(error).__name__}:{error}"
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "preflight_version": PREFLIGHT_VERSION,
        "decision": "pass" if not failed else "fail",
        "checks": checks, "failed_checks": failed, "details": details,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path) if input_path.is_file() else None,
        "overview_path": str(overview_path.resolve()),
        "overview_sha256": sha256_file(overview_path) if overview_path.is_file() else None,
        "overview_in_prompt": False,
        "run_root": str(run_root.resolve()),
        "root_id": parent.get("root_id") if parent else None,
        "api_config_path": str(api_config_path.resolve()),
        "api_config_sha256": sha256_file(api_config_path) if api_config_path.is_file() else None,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "api_configuration": ({
            "model": config.model, "base_url": config.base_url,
            "temperature": config.temperature,
            "enable_thinking": config.enable_thinking,
            "timeout_seconds": config.timeout_seconds, "api_key_recorded": False,
        } if config is not None else None),
        "source_manifest": source_manifest(),
        "protected_integrity": protected_integrity_snapshot(),
        "prompt_manifests": {
            key: {field: value for field, value in prompt.items() if field != "prompt"}
            for key, prompt in prompts.items()
        },
        "request_budget": {
            "local_event_extractor": WINDOW_CALLS, "retries": 0,
            "linker": 0, "planner": 0, "realizer": 0, "delta_updater": 0,
            "memorax": 0, "qa": 0, "solver": 0,
        },
        "max_completion_tokens_per_window": LOCAL_EVENT_MAX_TOKENS,
        "api_calls": 0,
    }


def _assert_frozen(freeze: Mapping[str, Any]) -> None:
    if source_manifest()["manifest_sha256"] != freeze["source_manifest"]["manifest_sha256"]:
        raise RuntimeError("stage_a_source_drift_after_freeze")
    for path_key, sha_key in (
        ("input_path", "input_sha256"), ("overview_path", "overview_sha256"),
        ("api_config_path", "api_config_sha256"),
    ):
        if sha256_file(Path(freeze[path_key])) != freeze[sha_key]:
            raise RuntimeError(f"stage_a_{path_key}_drift_after_freeze")


def execute_stage_a(
    *, preflight: Mapping[str, Any], input_path: Path, overview_path: Path,
    run_root: Path, api_config_path: Path,
) -> dict[str, Any]:
    if preflight.get("decision") != "pass":
        raise RuntimeError("STAGE_A_PRE_API_PREFLIGHT_FAILED")
    if run_root.exists():
        raise FileExistsError(f"stage_a_run_root_exists:{run_root}")
    parent = _load(input_path)
    windows = compile_parent_session_windows(parent)
    prompts = {row["window_id"]: build_local_event_prompt(row) for row in windows}
    config = replace(
        load_challenger_config(api_config_path), timeout_seconds=LIVE_TIMEOUT_SECONDS,
        temperature=0.1, enable_thinking=False)
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("frozen", "raw/prompts", "raw/backend_responses", "windows", "reports"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    freeze = {
        **dict(preflight), "runner_version": RUNNER_VERSION,
        "frozen_at": _now(), "fresh": True, "no_resume": True,
        "only_live_role": "local_event_extractor",
        "linking_enabled": False, "planner_realizer_delta_enabled": False,
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "frozen/parent.json", parent)
    atomic_json(run_root / "frozen/session_windows.json", {"windows": windows})
    atomic_json(run_root / "run_status.json", {
        "status": "initialized", "api_calls": 0, "created_at": _now(),
    })
    ledger = RequestLedger(
        run_root, manifest_sha256=freeze["source_manifest"]["manifest_sha256"],
        backend_mode="live")
    client = NoRetryOpenAIClient(config)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    window_reports: list[dict[str, Any]] = []
    valid_outputs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for window in windows:
        _assert_frozen(freeze)
        window_id = str(window["window_id"])
        prompt = prompts[window_id]
        request_id = f"local-event-{window_id.lower()}-attempt-01"
        atomic_text(run_root / f"raw/prompts/{request_id}.txt", prompt["prompt"])
        base = {
            "window_id": window_id, "session_ref": window["session_ref"],
            "attempts": 1, "retries": 0,
            "prompt_sha256": prompt["prompt_sha256"],
        }
        try:
            response = ledger.invoke(
                request_id=request_id, intent="session_local_event_extraction_v1",
                attempt=1, prompt_sha256=prompt["prompt_sha256"],
                call=lambda prompt_text=prompt["prompt"]: client.complete(
                    prompt=prompt_text, max_tokens=LOCAL_EVENT_MAX_TOKENS),
                budget_class="local_event_extractor", budget_limit=WINDOW_CALLS,
                total_budget_limit=WINDOW_CALLS,
                request_metadata={
                    "window_id": window_id, "retry_allowed": False,
                    "linking_calls": 0, "planner_calls": 0,
                    "realizer_calls": 0, "delta_updater_calls": 0,
                    "memorax_calls": 0,
                },
                backend_identity={
                    "transport": "openai_compatible_no_retry",
                    "model": config.model, "base_url": config.base_url,
                    "temperature": config.temperature,
                },
            )
        except Exception as error:
            window_reports.append({
                **base, "status": "transport_error",
                "error": f"{type(error).__name__}:{error}",
            })
            continue
        raw = response.raw_text
        atomic_text(run_root / f"raw/backend_responses/{request_id}.txt", raw)
        metadata = response.metadata or {}
        provider_usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else {}
        for key in usage:
            usage[key] += int(provider_usage.get(key) or 0)
        finish_reason = metadata.get("finish_reason")
        if finish_reason != "stop":
            window_reports.append({
                **base, "status": "finish_reason_invalid",
                "finish_reason": finish_reason, "usage": provider_usage,
            })
            continue
        try:
            parsed = load_json_object_strict(raw)
        except Exception as error:
            window_reports.append({
                **base, "status": "json_parse_failed",
                "finish_reason": finish_reason, "usage": provider_usage,
                "error": f"{type(error).__name__}:{error}",
            })
            continue
        window_dir = run_root / "windows" / window_id
        window_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(window_dir / "response_parsed.json", parsed)
        validation = validate_local_event_output(
            parsed, window=window, parent_speakers=_parent_speakers(parent))
        atomic_json(window_dir / "validation.json", validation)
        if validation["valid"]:
            valid_outputs.append((window, parsed))
        window_reports.append({
            **base, "status": "completed" if validation["valid"] else "contract_failed",
            "finish_reason": finish_reason, "usage": provider_usage,
            "event_count": validation["event_count"],
            "validation_errors": validation["errors"],
        })
    candidates = materialize_local_candidates(valid_outputs)
    audit = audit_episode_146700_local_candidates(candidates)
    atomic_json(run_root / "local_event_candidates.json", {
        "candidate_schema_version": "simple_append_materialized_local_events_v1",
        "candidates": candidates,
    })
    atomic_json(run_root / "reports/STAGE_A_LOCAL_EVENT_AUDIT.json", audit)
    atomic_json(run_root / "reports/window_reports.json", {"windows": window_reports})
    integrity_after = protected_integrity_snapshot()
    integrity = {
        "source_manifest_unchanged": source_manifest()["manifest_sha256"] ==
        freeze["source_manifest"]["manifest_sha256"],
        "protected_integrity_unchanged": integrity_after == freeze["protected_integrity"],
        "after": integrity_after,
    }
    successful = sum(row["status"] == "completed" for row in window_reports)
    success = successful == WINDOW_CALLS and audit["valid"]
    summary = {
        "runner_version": RUNNER_VERSION,
        "status": "completed" if success else "failed",
        "failure_stage": None if success else "local_event_extraction",
        "run_root": str(run_root), "root_id": parent["root_id"],
        "local_event_extractor_calls": ledger.summary()["request_count"],
        "successful_window_count": successful,
        "failed_window_count": WINDOW_CALLS - successful,
        "retries": 0, "linker_calls": 0, "planner_calls": 0,
        "realizer_calls": 0, "delta_updater_calls": 0,
        "memorax_calls": 0, "qa_calls": 0, "solver_calls": 0,
        "usage": usage, "candidate_count": len(candidates),
        "window_reports": window_reports, "semantic_audit": audit,
        "stage_a_supported": success, "integrity": integrity,
        "completed_at": _now(),
    }
    atomic_json(run_root / "reports/STAGE_A_LOCAL_EVENT_LIVE_REPORT.json", summary)
    atomic_text(run_root / "reports/STAGE_A_LOCAL_EVENT_LIVE_REPORT.md", "\n".join([
        "# Stage A Session-local Event Extraction Live Report", "",
        f"- Status: `{summary['status']}`",
        f"- Calls / retries: `{summary['local_event_extractor_calls']} / 0`",
        f"- Successful windows: `{successful}/5`",
        f"- Materialized candidates: `{len(candidates)}`",
        f"- Tokens: `{usage['prompt_tokens']} + {usage['completion_tokens']} = "
        f"{usage['total_tokens']}`",
        f"- Stage A supported: `{str(success).lower()}`",
        "- Linker, Planner, Realizer, Delta, Memorax, QA, and Solver calls: `0`", "",
    ]))
    atomic_json(run_root / "run_status.json", summary)
    return summary


def _preflight_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage A Session-local Event Extraction Preflight", "",
        f"- Decision: `{report['decision']}`", f"- API calls: `{report['api_calls']}`",
        f"- Run root: `{report['run_root']}`", "", "## Checks", "",
    ]
    lines.extend(f"- {key}: `{str(value).lower()}`" for key, value in report["checks"].items())
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--overview", type=Path, default=DEFAULT_OVERVIEW)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_LOCAL_EXTRACTION_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    preflight = run_stage_a_preflight(
        input_path=args.input.resolve(), overview_path=args.overview.resolve(),
        run_root=args.run_root.resolve(), api_config_path=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve())
    if args.preflight_only:
        atomic_json(args.report_root / "STAGE_A_LOCAL_EVENT_PREFLIGHT.json", preflight)
        atomic_text(
            args.report_root / "STAGE_A_LOCAL_EVENT_PREFLIGHT.md",
            _preflight_markdown(preflight))
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if preflight["decision"] == "pass" else 2
    if preflight["decision"] != "pass":
        print(json.dumps(
            {"status": "PRE_API_PREFLIGHT_FAILED", **preflight},
            ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    summary = execute_stage_a(
        preflight=preflight, input_path=args.input.resolve(),
        overview_path=args.overview.resolve(), run_root=args.run_root.resolve(),
        api_config_path=args.api_config.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["execute_stage_a", "run_stage_a_preflight"]
