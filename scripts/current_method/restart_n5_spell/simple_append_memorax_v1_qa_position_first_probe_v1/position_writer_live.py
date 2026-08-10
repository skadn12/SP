"""Fresh one-call Writer-only runner for a future authorized live probe."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from ..simple_append_v1.live_runtime import load_json_object_strict
from ..simple_append_v1.qa_live import DEFAULT_API_CONFIG
from .integrity import protected_manifests
from .position_contract import materialize_position_writer_output
from .position_inputs import (
    DEFAULT_EXPANDED, DEFAULT_MEMORY_OVERVIEW, EXPECTED_EXPANDED_SHA256,
    EXPECTED_MEMORY_OVERVIEW_SHA256, load_object, repo_root,
)
from .position_prompts import build_position_writer_prompt


RUNNER_VERSION = "position_distractor_provenance_writer_only_live_v2"
WRITER_COMPLETION_TOKENS = 16_384
WRITER_MAX_CALLS = 1
WRITER_RETRIES = 0
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_qa_position_semantic_probe_v1_live_1")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_writer_only(
    *, run_root: Path = DEFAULT_RUN_ROOT,
    expanded_path: Path = DEFAULT_EXPANDED,
    overview_path: Path = DEFAULT_MEMORY_OVERVIEW,
    api_config_path: Path = DEFAULT_API_CONFIG,
) -> dict[str, Any]:
    """Execute only after a separate user authorization; there is no retry path."""
    if run_root.exists():
        raise FileExistsError(f"fresh_run_root_required:{run_root}")
    if sha256_file(expanded_path) != EXPECTED_EXPANDED_SHA256:
        raise RuntimeError("expanded_dialogue_sha_mismatch")
    if sha256_file(overview_path) != EXPECTED_MEMORY_OVERVIEW_SHA256:
        raise RuntimeError("memory_overview_sha_mismatch")
    protected = protected_manifests(repo_root())
    if not protected["unchanged"]:
        raise RuntimeError("protected_source_manifest_mismatch")
    expanded = load_object(expanded_path)
    overview = load_object(overview_path)
    prompt = build_position_writer_prompt(expanded, overview)
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("frozen", "raw/prompts", "qa", "reports"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    freeze = {
        "runner_version": RUNNER_VERSION,
        "created_at": _now(),
        "fresh": True,
        "no_resume": True,
        "writer_calls_max": WRITER_MAX_CALLS,
        "writer_retries": WRITER_RETRIES,
        "solver_calls": 0,
        "expanded_dialogue_sha256": sha256_file(expanded_path),
        "memory_overview_sha256": sha256_file(overview_path),
        "prompt_sha256": prompt["prompt_sha256"],
        "protected_before": protected,
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "frozen/expanded_dialogue.json", expanded)
    atomic_json(run_root / "frozen/memory_overview.json", overview)
    atomic_text(run_root / "qa/writer_prompt.txt", prompt["prompt"])
    manifest_sha = sha256_file(run_root / "run_freeze_manifest.json")
    ledger = RequestLedger(run_root, manifest_sha, "live")
    api = replace(load_challenger_config(api_config_path), timeout_seconds=600,
                  temperature=0.25, enable_thinking=False)
    client = OpenAICompatibleChallengerClient(api)
    request_id = "position-semantic-writer-v1-a00"
    atomic_text(run_root / f"raw/prompts/{request_id}.txt", prompt["prompt"])
    response = ledger.invoke(
        request_id=request_id,
        intent="position_semantic_writer_v1",
        attempt=1,
        prompt_sha256=prompt["prompt_sha256"],
        call=lambda: client.complete(
            prompt=prompt["prompt"], max_tokens=WRITER_COMPLETION_TOKENS,
            json_object=True),
        budget_class="mcq_writer", budget_limit=1, total_budget_limit=1,
        request_metadata={"retry_allowed": False, "solver_calls": 0},
        backend_identity={
            "transport": "openai_compatible", "model": api.model,
            "base_url": api.base_url, "temperature": api.temperature,
        },
    )
    parsed = load_json_object_strict(response.raw_text)
    atomic_json(run_root / "qa/writer_response_parsed.json", parsed)
    materialized = materialize_position_writer_output(
        parsed, expanded_dialogue=expanded)
    atomic_json(run_root / "qa/materialization.json", materialized)
    report = {
        "runner_version": RUNNER_VERSION,
        "status": "completed",
        "writer_calls": 1,
        "writer_retries": 0,
        "solver_calls": 0,
        "question_count": materialized["question_count"],
        "structural_valid_count": materialized["structural_valid_count"],
        "usage": (response.metadata or {}).get("usage") or {},
        "finish_reason": (response.metadata or {}).get("finish_reason"),
    }
    atomic_json(run_root / "reports/POSITION_SEMANTIC_WRITER_ONLY_REPORT.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    args = parser.parse_args(argv)
    if not args.live:
        print(json.dumps({
            "runner_version": RUNNER_VERSION,
            "live_executed": False,
            "message": "Use position_preflight in zero-API turns; --live requires explicit authorization.",
        }, indent=2))
        return 0
    report = run_writer_only(
        run_root=args.run_root.resolve(), api_config_path=args.api_config.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
