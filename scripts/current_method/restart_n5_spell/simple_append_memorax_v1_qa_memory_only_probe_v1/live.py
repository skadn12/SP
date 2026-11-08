"""One-shot live Writer probe for the isolated Memory-only QA branch.

The runner intentionally stops after parse/materialization.  It never calls
Solver, Expansion, Planner, Realizer, or Memorax.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from .memory_only_contract import memory_source_refs, materialize_solver_rows, validate_writer_output
from .memory_only_prompts import build_memory_only_writer_prompt
from .preflight import DEFAULT_SOURCE_RUN, _load, _tree_sha

RUNNER_VERSION = "qa_memory_only_probe_v1_live"
WRITER_MAX_TOKENS = 16_384
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_41625_qa_memory_only_probe_v1_live_1")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_object(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as error:
        return None, f"{type(error).__name__}:{error}"
    if not isinstance(value, dict):
        return None, "top_level_object_required"
    return value, None


def run(*, source_run: Path = DEFAULT_SOURCE_RUN,
        run_root: Path = DEFAULT_RUN_ROOT,
        api_config: Path = DEFAULT_API_CONFIG) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(f"fresh_run_root_already_exists:{run_root}")
    memory_path = source_run / "memory" / "post_generation_overview.json"
    dialogue_path = source_run / "expanded_dialogue.json"
    overview = _load(memory_path)
    dialogue = _load(dialogue_path)
    prompt = build_memory_only_writer_prompt(overview)
    source_tree_sha = _tree_sha(source_run)
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("frozen", "raw/prompts", "raw/backend_responses", "qa", "reports"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    shutil.copy2(memory_path, run_root / "frozen/memory_overview.json")
    shutil.copy2(dialogue_path, run_root / "frozen/expanded_dialogue.json")
    freeze = {
        "freeze_version": RUNNER_VERSION, "created_at": _now(), "fresh": True,
        "no_resume": True, "source_run": str(source_run),
        "source_run_tree_sha256": source_tree_sha,
        "memory_overview_sha256": sha256_file(memory_path),
        "expanded_dialogue_sha256": sha256_file(dialogue_path),
        "root_id": overview.get("root_id"),
        "writer_input_mode": "memory_only_no_transcript",
        "writer_prompt_sha256": prompt["prompt_sha256"],
        "writer_max_tokens": WRITER_MAX_TOKENS,
        "writer_calls_max": 1, "writer_retries": 0,
        "solver_calls": 0, "expansion_calls": 0, "memorax_calls": 0,
        "transcript_reserved_for": "post_generation_truth_audit_only",
        "memory_is_qa_fact_source": False,
        "method_contract_changed": False,
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "raw/prompts/qa_writer_prompt_manifest.json", prompt)
    atomic_text(run_root / "raw/prompts/qa_writer_prompt.txt", prompt["prompt"])
    ledger = RequestLedger(run_root, hashlib.sha256(
        json.dumps(freeze, ensure_ascii=False, sort_keys=True).encode()).hexdigest(), "live")
    api = replace(load_challenger_config(api_config), timeout_seconds=600,
                  temperature=0.25, enable_thinking=False)
    client = OpenAICompatibleChallengerClient(api)
    request_id = "memory-only-qa-writer-a00"
    metadata = {"writer_input_mode": "memory_only_no_transcript",
                "source_root_id": overview.get("root_id"),
                "memory_overview_sha256": sha256_file(memory_path),
                "expanded_dialogue_not_in_prompt": True,
                "question_capacity": 12, "retry_allowed": False}
    response_meta: dict[str, Any] = {}
    raw = ""
    parsed: dict[str, Any] | None = None
    parse_error: str | None = None
    transport_error: str | None = None
    try:
        response = ledger.invoke(
            request_id=request_id, intent="memory_only_qa_writer", attempt=1,
            prompt_sha256=prompt["prompt_sha256"],
            call=lambda: client.complete(prompt=prompt["prompt"],
                                         max_tokens=WRITER_MAX_TOKENS,
                                         json_object=True),
            budget_class="memory_only_qa_writer", budget_limit=1,
            total_budget_limit=1, request_metadata=metadata,
            backend_identity={"transport": "openai_compatible", "model": api.model,
                              "base_url": api.base_url, "temperature": api.temperature,
                              "max_tokens": WRITER_MAX_TOKENS})
        raw = response.raw_text
        response_meta = dict(response.metadata or {})
        atomic_json(run_root / "raw/response_metadata.json", {
            "model_identifier": response.model_identifier, **response_meta})
        parsed, parse_error = _parse_object(raw)
    except Exception as error:
        transport_error = f"{type(error).__name__}:{error}"
    atomic_text(run_root / "raw/backend_responses/writer_response.txt", raw)
    atomic_json(run_root / "qa/writer_parse.json", {
        "parse_success": parsed is not None, "parse_error": parse_error,
        "transport_error": transport_error,
        "finish_reason": response_meta.get("finish_reason"),
        "usage": response_meta.get("usage"),
    })
    validation: dict[str, Any]
    materialized: list[dict[str, Any]] = []
    if parsed is None:
        validation = {"valid": False, "errors": [parse_error or transport_error or "writer_no_response"],
                      "raw_question_count": 0, "valid_question_count": 0,
                      "quarantine_count": 0, "questions": [], "quarantine": []}
    else:
        validation = validate_writer_output(parsed, source_refs=memory_source_refs(overview))
        materialized = materialize_solver_rows(validation, root_id=str(overview.get("root_id") or "unknown"))
    atomic_json(run_root / "qa/validation.json", validation)
    atomic_json(run_root / "qa/materialized_mcq.json", {
        "schema_version": "memory_only_qa_materialized_v1",
        "root_id": overview.get("root_id"), "items": materialized,
        "solver_executed": False, "transcript_verified": False,
    })
    source_unchanged = (_tree_sha(source_run) == source_tree_sha)
    report = {
        "report_version": RUNNER_VERSION, "status": "completed" if not transport_error else "failed",
        "source_run": str(source_run), "run_root": str(run_root),
        "writer_calls": 1 if not transport_error or (run_root / "ledger/requests/memory-only-qa-writer-a00.json").exists() else 0,
        "writer_retries": 0, "solver_calls": 0, "expansion_calls": 0,
        "memorax_calls": 0, "finish_reason": response_meta.get("finish_reason"),
        "usage": response_meta.get("usage"), "transport_error": transport_error,
        "json_parse_success": parsed is not None, "parse_error": parse_error,
        "raw_questions": validation.get("raw_question_count", 0),
        "valid_questions": validation.get("valid_question_count", 0),
        "quarantine_questions": validation.get("quarantine_count", 0),
        "memory_source_ref_count": len(memory_source_refs(overview)),
        "transcript_verified": False, "solver_executed": False,
        "source_run_unchanged": source_unchanged,
        "input_memory_sha256": sha256_file(memory_path),
        "input_dialogue_sha256": sha256_file(dialogue_path),
        "next_phase": "transcript_truth_audit_before_any_admission",
    }
    atomic_json(run_root / "reports/MEMORY_ONLY_QA_WRITER_LIVE_REPORT.json", report)
    atomic_text(run_root / "reports/MEMORY_ONLY_QA_WRITER_LIVE_REPORT.md", "\n".join([
        "# Memory-only QA Writer Live Probe v1", "",
        f"- Status: `{report['status']}`", f"- Finish reason: `{report['finish_reason']}`",
        f"- JSON parse: `{report['json_parse_success']}`",
        f"- Raw / valid / quarantine: `{report['raw_questions']} / {report['valid_questions']} / {report['quarantine_questions']}`",
        f"- Tokens: `{report['usage']}`", "- Writer calls / retries: `1 / 0`",
        "- Solver / Expansion / Memorax calls: `0 / 0 / 0`",
        "- Transcript verification: `not run`", "",
        "This output is diagnostic Memory-only generation. It is not admitted training data.", ""
    ]))
    atomic_json(run_root / "integrity.json", {"source_run_tree_sha256_before": source_tree_sha,
        "source_run_tree_sha256_after": _tree_sha(source_run), "source_run_unchanged": source_unchanged})
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    args = parser.parse_args(argv)
    if not args.live:
        print("Use --live for the single authorized Writer request.")
        return 2
    result = run(source_run=args.source_run, run_root=args.run_root,
                 api_config=args.api_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
