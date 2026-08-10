"""One-shot-per-window live Local Fact Extractor runner."""

from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..backends import BackendResponse
from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..live_backends import load_challenger_config
from ..simple_append_v1.live_runtime import load_json_object_strict
from ..simple_append_v1.qa_live import DEFAULT_API_CONFIG
from .fact_bundler import bundle_facts
from .fact_contract import materialize_facts, validate_fact_extractor_output
from .fact_prompts import build_fact_prompt
from .integrity import tree_manifest
from .inputs import (
    DEFAULT_EXPANDED, DEFAULT_MEMORY_OVERVIEW, EXPECTED_EXPANDED_SHA256,
    EXPECTED_MEMORY_OVERVIEW_SHA256, load_object,
)
from .preflight import TARGET_RUN_ROOT, frozen_integrity
from .windows import compile_natural_windows


RUNNER_VERSION = "qa_fact_first_live_v1"
FACT_COMPLETION_TOKENS = 2048
WINDOW_CALLS = 8
RETRIES = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _complete_once(config: Any, *, prompt: str, max_tokens: int) -> BackendResponse:
    """One HTTP attempt, deliberately without retry or semantic fallback."""
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if config.enable_thinking is not None:
        payload["enable_thinking"] = config.enable_thinking
    request = urllib.request.Request(
        config.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    choice = body["choices"][0]
    return BackendResponse(
        str(choice["message"]["content"]), config.model,
        {
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage") if isinstance(body.get("usage"), dict) else {},
            "transport_attempt_count": 1,
            "transport_retry_count": 0,
        },
    )


def _add_usage(total: dict[str, int], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if isinstance(usage.get(key), int):
            total[key] += usage[key]


def run_fact_first(
    *,
    run_root: Path = TARGET_RUN_ROOT,
    expanded_path: Path = DEFAULT_EXPANDED,
    overview_path: Path = DEFAULT_MEMORY_OVERVIEW,
    api_config_path: Path = DEFAULT_API_CONFIG,
) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(f"fresh_run_root_required:{run_root}")
    if sha256_file(expanded_path) != EXPECTED_EXPANDED_SHA256:
        raise RuntimeError("expanded_dialogue_sha_mismatch")
    if sha256_file(overview_path) != EXPECTED_MEMORY_OVERVIEW_SHA256:
        raise RuntimeError("memory_overview_sha_mismatch")
    module_root = Path(__file__).resolve().parent
    preflight_path = module_root / "reports/FACT_FIRST_PREFLIGHT.json"
    if not preflight_path.exists():
        raise RuntimeError("fact_first_preflight_report_missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    integrity = frozen_integrity()
    if preflight.get("decision") != "pass":
        raise RuntimeError("fact_first_preflight_not_passed")
    required = (
        integrity["expanded_unchanged"] and
        integrity["memory_overview_unchanged"] and
        integrity["old_position_module_unchanged"] and
        integrity["historical_runs_unchanged"] and
        integrity["target_run_absent"]
    )
    if not required:
        raise RuntimeError("fact_first_integrity_preflight_failed")

    expanded = load_object(expanded_path)
    overview = load_object(overview_path)
    compilation = compile_natural_windows(expanded)
    windows = compilation["windows"]
    if not compilation["valid"] or len(windows) != WINDOW_CALLS:
        raise RuntimeError("natural_window_compilation_failed")
    prompts = {
        window["window_id"]: build_fact_prompt(window) for window in windows
    }
    api = replace(
        load_challenger_config(api_config_path),
        timeout_seconds=600,
        temperature=0.1,
        enable_thinking=False,
    )

    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("frozen", "raw/prompts", "qa/windows", "reports"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    freeze = {
        "runner_version": RUNNER_VERSION,
        "created_at": _now(),
        "fresh": True,
        "no_resume": True,
        "input_episode": "episode-146700",
        "window_count": len(windows),
        "fact_extractor_calls_max": WINDOW_CALLS,
        "fact_extractor_retries": RETRIES,
        "mcq_writer_calls": 0,
        "solver_calls": 0,
        "expansion_calls": 0,
        "memorax_calls": 0,
        "expanded_dialogue_sha256": sha256_file(expanded_path),
        "memory_overview_sha256": sha256_file(overview_path),
        "window_compiler_version": compilation["window_compiler_version"],
        "prompt_version": next(iter(prompts.values()))["prompt_version"],
        "prompt_sha256_by_window": {
            window_id: prompt["prompt_sha256"]
            for window_id, prompt in prompts.items()
        },
        "module_tree_sha256": tree_manifest(module_root)["tree_sha256"],
        "model": api.model,
        "backend": api.base_url,
        "temperature": api.temperature,
        "enable_thinking": api.enable_thinking,
        "timeout_seconds": api.timeout_seconds,
        "completion_tokens": FACT_COMPLETION_TOKENS,
        "integrity": integrity,
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "frozen/expanded_dialogue.json", expanded)
    # Kept as a frozen audit input; deliberately omitted from the local prompt.
    atomic_json(run_root / "frozen/memory_overview.json", overview)
    atomic_json(run_root / "frozen/natural_windows.json", compilation)
    manifest_sha = sha256_file(run_root / "run_freeze_manifest.json")
    ledger = RequestLedger(run_root, manifest_sha, "live")

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    validations: list[dict[str, Any]] = []
    window_reports: list[dict[str, Any]] = []
    for window in windows:
        window_id = window["window_id"]
        prompt = prompts[window_id]
        request_id = f"fact-extractor-{window_id.lower()}-a00"
        atomic_text(run_root / f"raw/prompts/{request_id}.txt", prompt["prompt"])
        base = {
            "window_id": window_id,
            "window_kind": window["window_kind"],
            "boundary_ref": window["boundary_ref"],
            "global_position_start": window["global_position_start"],
            "global_position_end": window["global_position_end"],
            "turn_count": window["turn_count"],
            "prompt_sha256": prompt["prompt_sha256"],
            "attempts": 1,
            "retries": 0,
        }
        try:
            response = ledger.invoke(
                request_id=request_id,
                intent="local_atomic_fact_extraction_v1",
                attempt=1,
                prompt_sha256=prompt["prompt_sha256"],
                call=lambda prompt_text=prompt["prompt"]: _complete_once(
                    api, prompt=prompt_text, max_tokens=FACT_COMPLETION_TOKENS,
                ),
                budget_class="fact_extractor",
                budget_limit=WINDOW_CALLS,
                total_budget_limit=WINDOW_CALLS,
                request_metadata={
                    "window_id": window_id,
                    "retry_allowed": False,
                    "mcq_writer_calls": 0,
                    "solver_calls": 0,
                    "expansion_calls": 0,
                    "memorax_calls": 0,
                },
                backend_identity={
                    "transport": "one_shot_openai_compatible",
                    "model": api.model,
                    "base_url": api.base_url,
                    "temperature": api.temperature,
                },
            )
        except Exception as error:
            window_reports.append({
                **base,
                "status": "transport_error",
                "finish_reason": None,
                "error_type": type(error).__name__,
                "error": str(error),
                "usage": {},
            })
            continue
        metadata = response.metadata or {}
        finish_reason = metadata.get("finish_reason")
        window_usage = metadata.get("usage") or {}
        _add_usage(usage, window_usage)
        if finish_reason == "length":
            window_reports.append({
                **base,
                "status": "finish_reason_length",
                "finish_reason": finish_reason,
                "usage": window_usage,
            })
            continue
        try:
            parsed = load_json_object_strict(response.raw_text)
        except Exception as error:
            window_reports.append({
                **base,
                "status": "json_parse_failed",
                "finish_reason": finish_reason,
                "usage": window_usage,
                "error_type": type(error).__name__,
                "error": str(error),
            })
            continue
        window_dir = run_root / "qa/windows" / window_id
        window_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(window_dir / "response_parsed.json", parsed)
        validation = validate_fact_extractor_output(
            parsed, expanded_dialogue=expanded, window=window,
        )
        atomic_json(window_dir / "validation.json", validation)
        validations.append(validation)
        window_reports.append({
            **base,
            "status": "completed",
            "finish_reason": finish_reason,
            "usage": window_usage,
            "raw_fact_count": validation["raw_fact_count"],
            "mechanically_valid_count": validation["mechanically_valid_count"],
            "quarantine_count": validation["quarantine_count"],
        })

    windows_by_id = {window["window_id"]: window for window in windows}
    facts = materialize_facts(validations, windows_by_id=windows_by_id)
    bundles = bundle_facts(facts)
    atomic_json(run_root / "qa/materialized_facts.json", {
        "fact_schema_version": "local_atomic_fact_v1",
        "facts": facts,
    })
    atomic_json(run_root / "qa/candidate_bundles.json", bundles)
    atomic_json(run_root / "qa/window_reports.json", {"windows": window_reports})
    ledger_summary = ledger.summary()
    raw_count = sum(row.get("raw_fact_count", 0) for row in window_reports)
    valid_count = sum(row.get("mechanically_valid_count", 0) for row in window_reports)
    quarantine_count = sum(row.get("quarantine_count", 0) for row in window_reports)
    report = {
        "runner_version": RUNNER_VERSION,
        "status": "completed_with_window_failures" if len(validations) < len(windows) else "completed",
        "window_count": len(windows),
        "successful_window_count": len(validations),
        "failed_window_count": len(windows) - len(validations),
        "fact_extractor_calls": ledger_summary["request_count"],
        "fact_extractor_retries": 0,
        "mcq_writer_calls": 0,
        "solver_calls": 0,
        "expansion_calls": 0,
        "memorax_calls": 0,
        "raw_fact_count": raw_count,
        "mechanically_valid_fact_count": valid_count,
        "mechanical_quarantine_count": quarantine_count,
        "materialized_fact_count": len(facts),
        "bundle_count": bundles["bundle_count"],
        "usage": usage,
        "ledger_summary": ledger_summary,
        "window_reports": window_reports,
        "bundler_limits": bundles["limits"],
        "mcq_generated": False,
        "next_stage_allowed": "reverse_question_writer_only_after_fact_audit",
    }
    atomic_json(run_root / "reports/FACT_FIRST_LIVE_REPORT.json", report)
    atomic_text(run_root / "reports/FACT_FIRST_LIVE_REPORT.md", "\n".join([
        "# Fact-First Live Report", "",
        f"- Status: `{report['status']}`",
        f"- Windows: `{len(validations)}/{len(windows)}` successful",
        f"- Fact calls / retries: `{report['fact_extractor_calls']} / 0`",
        f"- Raw / valid / quarantine facts: `{raw_count} / {valid_count} / {quarantine_count}`",
        f"- Materialized facts: `{len(facts)}`",
        f"- Cross-window candidate bundles: `{bundles['bundle_count']}`",
        f"- Tokens: `{json.dumps(usage, sort_keys=True)}`",
        "- MCQ / Solver / Expansion / Memorax calls: `0 / 0 / 0 / 0`",
        "- MCQ generation: `false`",
        "",
    ]))
    integrity_after = frozen_integrity()
    post_integrity = {
        "module_tree_sha256_before_first_api": freeze["module_tree_sha256"],
        "module_tree_sha256_after_live": tree_manifest(module_root)["tree_sha256"],
        "module_unchanged_after_first_api": (
            freeze["module_tree_sha256"] == tree_manifest(module_root)["tree_sha256"]
        ),
        "expanded_dialogue_sha256_before": freeze["expanded_dialogue_sha256"],
        "expanded_dialogue_sha256_after": integrity_after["expanded_sha256"],
        "expanded_dialogue_unchanged": integrity_after["expanded_unchanged"],
        "memory_overview_sha256_before": freeze["memory_overview_sha256"],
        "memory_overview_sha256_after": integrity_after["memory_overview_sha256"],
        "memory_overview_unchanged": integrity_after["memory_overview_unchanged"],
        "old_position_module_unchanged": integrity_after["old_position_module_unchanged"],
        "historical_runs_unchanged": integrity_after["historical_runs_unchanged"],
        "calls": {
            "fact_extractor": ledger_summary["request_count"],
            "retries": 0,
            "mcq_writer": 0,
            "solver": 0,
            "expansion": 0,
            "memorax": 0,
        },
        "r2_r3": False,
        "training": False,
        "git_commit_push": False,
    }
    atomic_json(run_root / "reports/POST_RUN_INTEGRITY.json", post_integrity)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--run-root", type=Path, default=TARGET_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    args = parser.parse_args(argv)
    if not args.live:
        print(json.dumps({
            "runner_version": RUNNER_VERSION,
            "live_executed": False,
            "message": "Run fact-first preflight before --live.",
        }, indent=2))
        return 0
    report = run_fact_first(
        run_root=args.run_root.resolve(),
        api_config_path=args.api_config.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"].startswith("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
