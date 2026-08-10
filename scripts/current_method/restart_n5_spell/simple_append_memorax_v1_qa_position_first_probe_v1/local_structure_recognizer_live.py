"""Fresh local-window Structure Recognizer live runner (one call per window)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..live_backends import load_challenger_config
from ..simple_append_v1.live_runtime import load_json_object_strict
from ..simple_append_v1.qa_live import DEFAULT_API_CONFIG
from .integrity import tree_manifest
from .local_structure_recognizer_contract import validate_local_recognizer_output
from .local_structure_recognizer_preflight import TARGET_RUN_ROOT, frozen_integrity
from .local_structure_recognizer_prompts import build_local_recognizer_prompt
from .local_structure_windows import compile_natural_windows
from .position_inputs import DEFAULT_EXPANDED, EXPECTED_EXPANDED_SHA256, load_object
from .structure_miner_live import _complete_once


RUNNER_VERSION = "local_structure_recognizer_live_v1"
PER_WINDOW_COMPLETION_TOKENS = 2048
EXPECTED_WINDOW_COUNT = 8
MAX_TOTAL_CALLS = 8
RETRIES = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_add(total: dict[str, int], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            total[key] += value


def _window_report_base(window: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_id": window["window_id"],
        "window_kind": window["window_kind"],
        "boundary_ref": window["boundary_ref"],
        "global_position_start": window["global_position_start"],
        "global_position_end": window["global_position_end"],
        "turn_count": window["turn_count"],
        "prompt_version": prompt["prompt_version"],
        "prompt_sha256": prompt["prompt_sha256"],
        "prompt_token_estimate": prompt["token_estimate"],
        "request_attempts": 1,
        "retries": 0,
    }


def run_local_recognizer(
    *, run_root: Path = TARGET_RUN_ROOT,
    expanded_path: Path = DEFAULT_EXPANDED,
    api_config_path: Path = DEFAULT_API_CONFIG,
) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(f"fresh_run_root_required:{run_root}")
    if sha256_file(expanded_path) != EXPECTED_EXPANDED_SHA256:
        raise RuntimeError("expanded_dialogue_sha_mismatch")
    module_root = Path(__file__).resolve().parent
    preflight_path = module_root / "reports/LOCAL_STRUCTURE_RECOGNIZER_PREFLIGHT.json"
    if not preflight_path.exists():
        raise RuntimeError("local_recognizer_preflight_report_missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    integrity = frozen_integrity()
    required_integrity = (
        integrity["expanded_dialogue_unchanged"] and
        integrity["protected_sources"]["unchanged"] and
        integrity["frozen_files_unchanged"] and
        integrity["historical_full_miner_run_unchanged"] and
        integrity["target_run_absent"]
    )
    if preflight.get("decision") != "pass" or not required_integrity:
        raise RuntimeError("local_recognizer_integrity_preflight_failed")

    expanded = load_object(expanded_path)
    compilation = compile_natural_windows(expanded)
    windows = compilation["windows"]
    if not compilation["valid"] or len(windows) != EXPECTED_WINDOW_COUNT:
        raise RuntimeError("natural_window_compilation_failed")
    prompts = {
        window["window_id"]: build_local_recognizer_prompt(window)
        for window in windows
    }
    api = replace(
        load_challenger_config(api_config_path),
        timeout_seconds=600, temperature=0.1, enable_thinking=False,
    )

    run_root.mkdir(parents=True, exist_ok=False)
    for relative in (
        "frozen", "raw/prompts", "qa/windows", "reports",
    ):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    freeze = {
        "runner_version": RUNNER_VERSION,
        "created_at": _now(),
        "fresh": True,
        "no_resume": True,
        "window_count": len(windows),
        "recognizer_calls_max": MAX_TOTAL_CALLS,
        "recognizer_retries": RETRIES,
        "writer_calls": 0,
        "full_miner_calls": 0,
        "solver_calls": 0,
        "expanded_dialogue_sha256": sha256_file(expanded_path),
        "window_compiler_version": compilation["window_compiler_version"],
        "prompt_version": next(iter(prompts.values()))["prompt_version"],
        "prompt_sha256_by_window": {
            key: value["prompt_sha256"] for key, value in prompts.items()
        },
        "module_tree_sha256": tree_manifest(module_root)["tree_sha256"],
        "model": api.model,
        "backend": api.base_url,
        "temperature": api.temperature,
        "enable_thinking": api.enable_thinking,
        "timeout_seconds": api.timeout_seconds,
        "per_window_completion_tokens": PER_WINDOW_COMPLETION_TOKENS,
        "integrity": integrity,
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "frozen/expanded_dialogue.json", expanded)
    atomic_json(run_root / "frozen/natural_windows.json", compilation)
    manifest_sha = sha256_file(run_root / "run_freeze_manifest.json")
    ledger = RequestLedger(run_root, manifest_sha, "live")

    window_reports: list[dict[str, Any]] = []
    aggregate_items: list[dict[str, Any]] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for window in windows:
        window_id = window["window_id"]
        prompt = prompts[window_id]
        request_id = f"local-structure-{window_id.lower()}-a00"
        atomic_text(run_root / f"raw/prompts/{request_id}.txt", prompt["prompt"])
        base = _window_report_base(window, prompt)
        try:
            response = ledger.invoke(
                request_id=request_id,
                intent="local_structure_recognizer_v1",
                attempt=1,
                prompt_sha256=prompt["prompt_sha256"],
                call=lambda prompt_text=prompt["prompt"]: _complete_once(
                    api, prompt=prompt_text, max_tokens=PER_WINDOW_COMPLETION_TOKENS,
                ),
                budget_class="local_structure_recognizer",
                budget_limit=MAX_TOTAL_CALLS,
                total_budget_limit=MAX_TOTAL_CALLS,
                request_metadata={
                    "window_id": window_id, "retry_allowed": False,
                    "writer_calls": 0, "solver_calls": 0, "full_miner_calls": 0,
                },
                backend_identity={
                    "transport": "one_shot_openai_compatible",
                    "model": api.model, "base_url": api.base_url,
                    "temperature": api.temperature,
                },
            )
        except Exception as error:
            window_reports.append({
                **base, "status": "transport_error", "finish_reason": None,
                "error_type": type(error).__name__, "error": str(error), "usage": {},
            })
            continue

        metadata = response.metadata or {}
        finish_reason = metadata.get("finish_reason")
        usage = metadata.get("usage") or {}
        _usage_add(usage_total, usage)
        if finish_reason == "length":
            window_reports.append({
                **base, "status": "finish_reason_length",
                "finish_reason": finish_reason, "usage": usage,
            })
            continue
        try:
            parsed = load_json_object_strict(response.raw_text)
        except Exception as error:
            window_reports.append({
                **base, "status": "json_parse_failed", "finish_reason": finish_reason,
                "usage": usage, "error_type": type(error).__name__, "error": str(error),
            })
            continue

        window_dir = run_root / "qa/windows" / window_id
        window_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(window_dir / "response_parsed.json", parsed)
        validation = validate_local_recognizer_output(
            parsed, expanded_dialogue=expanded, window=window,
        )
        atomic_json(window_dir / "validation.json", validation)
        aggregate_items.extend(validation["items"])
        window_reports.append({
            **base, "status": "completed", "finish_reason": finish_reason,
            "usage": usage,
            "raw_structure_count": validation["raw_structure_count"],
            "mechanically_valid_count": validation["mechanically_valid_count"],
            "quarantine_count": validation["quarantine_count"],
            "structure_type_distribution": validation["structure_type_distribution"],
        })

    atomic_json(run_root / "qa/window_reports.json", {"windows": window_reports})
    atomic_json(run_root / "qa/all_structure_items.json", {"items": aggregate_items})
    completed = [row for row in window_reports if row["status"] == "completed"]
    valid_items = [row for row in aggregate_items if row["valid"]]
    quarantine = [row for row in aggregate_items if not row["valid"]]
    ledger_summary = ledger.summary()
    report = {
        "runner_version": RUNNER_VERSION,
        "status": "completed_with_window_failures" if len(completed) < len(windows) else "completed",
        "window_count": len(windows),
        "successful_window_count": len(completed),
        "failed_window_count": len(windows) - len(completed),
        "recognizer_calls": ledger_summary["request_count"],
        "recognizer_retries": 0,
        "writer_calls": 0,
        "full_miner_calls": 0,
        "solver_calls": 0,
        "expansion_calls": 0,
        "memorax_calls": 0,
        "usage": usage_total,
        "raw_structure_count": sum(row.get("raw_structure_count", 0) for row in completed),
        "mechanically_valid_count": len(valid_items),
        "quarantine_count": len(quarantine),
        "structure_type_distribution": dict(Counter(
            row["structure_type"] for row in valid_items
        )),
        "window_reports": window_reports,
        "ledger_summary": ledger_summary,
    }
    atomic_json(run_root / "reports/LOCAL_STRUCTURE_RECOGNIZER_LIVE_REPORT.json", report)
    atomic_text(run_root / "reports/LOCAL_STRUCTURE_RECOGNIZER_LIVE_REPORT.md", "\n".join([
        "# Local Structure Recognizer Live Report", "",
        f"- Status: `{report['status']}`",
        f"- Windows successful: `{len(completed)}/{len(windows)}`",
        f"- Calls / retries: `{report['recognizer_calls']} / 0`",
        f"- Raw / mechanically valid / quarantine: "
        f"`{report['raw_structure_count']} / {len(valid_items)} / {len(quarantine)}`",
        f"- Tokens: `{json.dumps(usage_total, sort_keys=True)}`",
        "- Writer / full Miner / Solver / Expansion / Memorax calls: `0`",
        "",
    ]))
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
            "message": "Run local_structure_recognizer_preflight before --live.",
        }, indent=2))
        return 0
    report = run_local_recognizer(
        run_root=args.run_root.resolve(), api_config_path=args.api_config.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"].startswith("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
