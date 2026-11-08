"""One-call live probe for reference-only reverse question selection."""

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
from .fact_bundle_writer_contract import materialize_batch, validate_writer_output
from .fact_bundle_writer_prompts import build_fact_bundle_writer_prompt
from .preflight import DEFAULT_FACTS, DEFAULT_BUNDLES


RUNNER_VERSION = "fact_bundle_writer_live_v1"
WRITER_COMPLETION_TOKENS = 16_384
WRITER_CALLS = 1
WRITER_RETRIES = 0
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_fact_bundle_writer_probe_v1_live_1"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _complete_once(config: Any, *, prompt: str, max_tokens: int) -> BackendResponse:
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
        headers={"Authorization": f"Bearer {config.api_key}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    choice = body["choices"][0]
    return BackendResponse(
        str(choice["message"]["content"]), config.model,
        {"finish_reason": choice.get("finish_reason"),
         "usage": body.get("usage") if isinstance(body.get("usage"), dict) else {},
         "transport_attempt_count": 1, "transport_retry_count": 0},
    )


def run_writer_only(
    *, run_root: Path = DEFAULT_RUN_ROOT,
    facts_path: Path = DEFAULT_FACTS,
    bundles_path: Path = DEFAULT_BUNDLES,
    api_config_path: Path = DEFAULT_API_CONFIG,
) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(f"fresh_run_root_required:{run_root}")
    facts_value = _load(facts_path)
    bundles_value = _load(bundles_path)
    facts = list(facts_value.get("facts") or [])
    bundles = list(bundles_value.get("bundles") or [])
    prompt = build_fact_bundle_writer_prompt(facts, bundles)
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("frozen", "raw/prompts", "qa", "reports"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    api = replace(load_challenger_config(api_config_path), timeout_seconds=600,
                  temperature=0.2, enable_thinking=False)
    freeze = {
        "runner_version": RUNNER_VERSION, "created_at": _now(),
        "fresh": True, "no_resume": True, "input_episode": "episode-146700",
        "writer_calls_max": WRITER_CALLS, "writer_retries": WRITER_RETRIES,
        "solver_calls": 0, "expansion_calls": 0, "memorax_calls": 0,
        "facts_sha256": sha256_file(facts_path),
        "bundles_sha256": sha256_file(bundles_path),
        "prompt_sha256": prompt["prompt_sha256"],
        "prompt_version": prompt["prompt_version"],
        "model": api.model, "backend": api.base_url,
        "temperature": api.temperature, "completion_tokens": WRITER_COMPLETION_TOKENS,
        "fact_count": len(facts), "bundle_count": len(bundles),
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "frozen/facts.json", facts_value)
    atomic_json(run_root / "frozen/bundles.json", bundles_value)
    atomic_text(run_root / "qa/writer_prompt.txt", prompt["prompt"])
    manifest_sha = sha256_file(run_root / "run_freeze_manifest.json")
    ledger = RequestLedger(run_root, manifest_sha, "live")
    request_id = "fact-bundle-writer-v1-a00"
    atomic_text(run_root / f"raw/prompts/{request_id}.txt", prompt["prompt"])
    response = ledger.invoke(
        request_id=request_id, intent="fact_bundle_writer_v1", attempt=1,
        prompt_sha256=prompt["prompt_sha256"],
        call=lambda: _complete_once(api, prompt=prompt["prompt"],
                                    max_tokens=WRITER_COMPLETION_TOKENS),
        budget_class="mcq_writer", budget_limit=1, total_budget_limit=1,
        request_metadata={"retry_allowed": False, "writer_calls": 1,
                          "writer_retries": 0, "solver_calls": 0,
                          "expansion_calls": 0, "memorax_calls": 0},
        backend_identity={"transport": "one_shot_openai_compatible",
                          "model": api.model, "base_url": api.base_url,
                          "temperature": api.temperature},
    )
    metadata = response.metadata or {}
    atomic_text(run_root / "raw/writer_response.txt", response.raw_text)
    report: dict[str, Any] = {
        "runner_version": RUNNER_VERSION, "status": "completed",
        "writer_calls": 1, "writer_retries": 0, "solver_calls": 0,
        "finish_reason": metadata.get("finish_reason"),
        "usage": metadata.get("usage") or {}, "json_parse_success": False,
    }
    try:
        parsed = load_json_object_strict(response.raw_text)
    except Exception as error:
        report.update({"status": "json_parse_failed", "error_type": type(error).__name__,
                       "error": str(error)})
        atomic_json(run_root / "reports/FACT_BUNDLE_WRITER_LIVE_REPORT.json", report)
        return report
    atomic_json(run_root / "qa/writer_response_parsed.json", parsed)
    report["json_parse_success"] = True
    validation = validate_writer_output(parsed, facts=facts, bundles=bundles)
    materialized = materialize_batch(parsed, facts=facts, bundles=bundles)
    atomic_json(run_root / "qa/validation.json", validation)
    atomic_json(run_root / "qa/materialized_mcq.json", materialized)
    report.update({
        "raw_questions": validation["raw_question_count"],
        "reference_valid": validation["valid_question_count"],
        "quarantine": validation["quarantine_count"],
        "materialized_valid": len(materialized["materialized_questions"]),
        "materialization_errors": materialized["materialization_errors"],
        "status": "completed",
        "solver_executed": False,
    })
    atomic_json(run_root / "reports/FACT_BUNDLE_WRITER_LIVE_REPORT.json", report)
    atomic_text(run_root / "reports/FACT_BUNDLE_WRITER_LIVE_REPORT.md", "\n".join([
        "# Fact-Bundle Writer Live Probe", "",
        f"- Finish reason: `{report['finish_reason']}`",
        f"- JSON parse: `{report['json_parse_success']}`",
        f"- Raw / valid / quarantine: `{report.get('raw_questions', 0)} / "
        f"{report.get('reference_valid', 0)} / {report.get('quarantine', 0)}`",
        f"- Materialized MCQ: `{report.get('materialized_valid', 0)}`",
        f"- Tokens: `{json.dumps(report['usage'], sort_keys=True)}`",
        "- Writer calls / retries / Solver calls: `1 / 0 / 0`", "",
        "No Solver or downstream admission was run in this Writer-only probe.",
    ]) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    parser.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLES)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    args = parser.parse_args(argv)
    if not args.live:
        print(json.dumps({"live_executed": False, "message": "Use --live only for the authorized one-call probe."}, indent=2))
        return 0
    report = run_writer_only(run_root=args.run_root.resolve(), facts_path=args.facts.resolve(),
                             bundles_path=args.bundles.resolve(), api_config_path=args.api_config.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
