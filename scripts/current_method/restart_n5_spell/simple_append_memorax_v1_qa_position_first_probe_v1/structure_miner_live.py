"""Fresh one-request live runner for the minimal Structure Miner."""

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
from ..live_backends import ChallengerAPIConfig, load_challenger_config
from ..simple_append_v1.live_runtime import load_json_object_strict
from ..simple_append_v1.qa_live import DEFAULT_API_CONFIG
from .integrity import tree_manifest
from .position_inputs import DEFAULT_EXPANDED, EXPECTED_EXPANDED_SHA256, load_object
from .structure_miner_contract import validate_structure_miner_output
from .structure_miner_preflight import TARGET_RUN_ROOT, frozen_integrity
from .structure_miner_prompts import build_structure_miner_prompt


RUNNER_VERSION = "structure_miner_only_live_v1"
MINER_COMPLETION_TOKENS = 4096
MINER_MAX_CALLS = 1
MINER_RETRIES = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _complete_once(
    config: ChallengerAPIConfig, *, prompt: str, max_tokens: int,
) -> BackendResponse:
    """Make exactly one HTTP attempt; this Miner owns no retry mechanism."""
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
        str(choice["message"]["content"]),
        config.model,
        {
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage") if isinstance(body.get("usage"), dict) else None,
            "transport_attempt_count": 1,
            "transport_retry_count": 0,
            "rate_limit_retry_count": 0,
        },
    )


def _write_report(run_root: Path, report: dict[str, Any]) -> None:
    atomic_json(run_root / "reports/STRUCTURE_MINER_ONLY_REPORT.json", report)


def run_miner_only(
    *, run_root: Path = TARGET_RUN_ROOT,
    expanded_path: Path = DEFAULT_EXPANDED,
    api_config_path: Path = DEFAULT_API_CONFIG,
) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(f"fresh_run_root_required:{run_root}")
    if sha256_file(expanded_path) != EXPECTED_EXPANDED_SHA256:
        raise RuntimeError("expanded_dialogue_sha_mismatch")
    integrity = frozen_integrity()
    required_integrity = (
        integrity["expanded_dialogue_unchanged"] and
        integrity["protected_sources"]["unchanged"] and
        integrity["position_files_unchanged"] and
        integrity["historical_position_run_unchanged"] and
        integrity["target_run_absent"]
    )
    if not required_integrity:
        raise RuntimeError("structure_miner_integrity_preflight_failed")
    module_root = Path(__file__).resolve().parent
    preflight_path = module_root / "reports/STRUCTURE_MINER_PREFLIGHT.json"
    if not preflight_path.exists():
        raise RuntimeError("structure_miner_preflight_report_missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("decision") != "pass":
        raise RuntimeError("structure_miner_preflight_not_passed")

    expanded = load_object(expanded_path)
    prompt = build_structure_miner_prompt(expanded)
    api = replace(
        load_challenger_config(api_config_path),
        timeout_seconds=600,
        temperature=0.1,
        enable_thinking=False,
    )
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("frozen", "raw/prompts", "qa", "reports"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    freeze = {
        "runner_version": RUNNER_VERSION,
        "created_at": _now(),
        "fresh": True,
        "no_resume": True,
        "miner_calls_max": MINER_MAX_CALLS,
        "miner_retries": MINER_RETRIES,
        "writer_calls": 0,
        "solver_calls": 0,
        "expanded_dialogue_sha256": sha256_file(expanded_path),
        "prompt_version": prompt["prompt_version"],
        "prompt_sha256": prompt["prompt_sha256"],
        "module_tree_sha256": tree_manifest(module_root)["tree_sha256"],
        "model": api.model,
        "backend": api.base_url,
        "temperature": api.temperature,
        "enable_thinking": api.enable_thinking,
        "timeout_seconds": api.timeout_seconds,
        "completion_tokens": MINER_COMPLETION_TOKENS,
        "integrity": integrity,
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "frozen/expanded_dialogue.json", expanded)
    atomic_text(run_root / "qa/miner_prompt.txt", prompt["prompt"])
    atomic_text(run_root / "raw/prompts/structure-miner-v1-a00.txt", prompt["prompt"])
    ledger = RequestLedger(
        run_root, sha256_file(run_root / "run_freeze_manifest.json"), "live"
    )
    request_id = "structure-miner-v1-a00"
    base_report = {
        "runner_version": RUNNER_VERSION,
        "miner_calls": 1,
        "miner_retries": 0,
        "writer_calls": 0,
        "solver_calls": 0,
        "expansion_calls": 0,
        "memorax_calls": 0,
    }
    try:
        response = ledger.invoke(
            request_id=request_id,
            intent="structure_miner_only_v1",
            attempt=1,
            prompt_sha256=prompt["prompt_sha256"],
            call=lambda: _complete_once(
                api, prompt=prompt["prompt"], max_tokens=MINER_COMPLETION_TOKENS
            ),
            budget_class="structure_miner",
            budget_limit=1,
            total_budget_limit=1,
            request_metadata={"retry_allowed": False, "writer_calls": 0, "solver_calls": 0},
            backend_identity={
                "transport": "one_shot_openai_compatible",
                "model": api.model,
                "base_url": api.base_url,
                "temperature": api.temperature,
            },
        )
    except Exception as error:
        report = {
            **base_report,
            "status": "transport_error",
            "error_type": type(error).__name__,
            "error": str(error),
            "finish_reason": None,
        }
        _write_report(run_root, report)
        return report

    metadata = response.metadata or {}
    finish_reason = metadata.get("finish_reason")
    if finish_reason == "length":
        report = {
            **base_report,
            "status": "finish_reason_length",
            "finish_reason": finish_reason,
            "usage": metadata.get("usage") or {},
        }
        _write_report(run_root, report)
        return report
    try:
        parsed = load_json_object_strict(response.raw_text)
    except Exception as error:
        report = {
            **base_report,
            "status": "json_parse_failed",
            "finish_reason": finish_reason,
            "usage": metadata.get("usage") or {},
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_report(run_root, report)
        return report

    atomic_json(run_root / "qa/miner_response_parsed.json", parsed)
    validation = validate_structure_miner_output(
        parsed, expanded_dialogue=expanded
    )
    atomic_json(run_root / "qa/structure_validation.json", validation)
    report = {
        **base_report,
        "status": "completed",
        "finish_reason": finish_reason,
        "usage": metadata.get("usage") or {},
        "raw_structure_count": validation["raw_structure_count"],
        "mechanically_valid_count": validation["mechanically_valid_count"],
        "quarantine_count": validation["quarantine_count"],
        "transport_attempt_count": metadata.get("transport_attempt_count"),
        "transport_retry_count": metadata.get("transport_retry_count"),
    }
    _write_report(run_root, report)
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
            "message": "Run structure_miner_preflight before separately authorized --live.",
        }, indent=2))
        return 0
    report = run_miner_only(
        run_root=args.run_root.resolve(),
        api_config_path=args.api_config.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
