"""QA-only local Fact Extractor live runner for frozen episode-91716."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..live_backends import load_challenger_config
from ..simple_append_v1.live_runtime import load_json_object_strict
from ..simple_append_v1.qa_live import DEFAULT_API_CONFIG
from ..simple_append_memorax_v1_qa_fact_first_probe_v1.fact_bundler import bundle_facts
from .subject_binding_contract import (
    known_entity_refs,
    materialize_subject_bound_facts,
    validate_subject_binding_output,
)
from ..simple_append_memorax_v1_qa_fact_first_probe_v1.fact_live import _complete_once
from .subject_binding_prompts import build_subject_binding_prompt
from ..simple_append_memorax_v1_qa_fact_first_probe_v1.integrity import tree_manifest
from ..simple_append_memorax_v1_qa_fact_first_probe_v1.windows import compile_natural_windows
from .inputs import (
    DEFAULT_EXPANDED,
    DEFAULT_MEMORY_OVERVIEW,
    SUBJECT_BINDING_TARGET_RUN_ROOT,
    frozen_input_manifest,
)


RUNNER_VERSION = "qa_fact_first_episode91716_subject_binding_live_v1"
FACT_COMPLETION_TOKENS = 2048
WINDOW_CALLS = 8
RETRIES = 0
MODULE_ROOT = Path(__file__).resolve().parent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_usage(total: dict[str, int], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    for key in total:
        if isinstance(usage.get(key), int):
            total[key] += usage[key]


def run_live(
    *, run_root: Path = SUBJECT_BINDING_TARGET_RUN_ROOT,
    api_config_path: Path = DEFAULT_API_CONFIG,
) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(f"fresh_run_root_required:{run_root}")
    preflight_path = MODULE_ROOT / "reports/SUBJECT_BINDING_PREFLIGHT.json"
    if not preflight_path.is_file():
        raise RuntimeError("episode91716_fact_first_preflight_missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    inputs = frozen_input_manifest()
    if preflight.get("decision") != "pass":
        raise RuntimeError("episode91716_fact_first_preflight_failed")
    if not inputs["expanded_unchanged"] or not inputs["subject_binding_target_run_absent"]:
        raise RuntimeError("episode91716_fact_first_input_or_run_drift")

    expanded = json.loads(DEFAULT_EXPANDED.read_text(encoding="utf-8"))
    overview = json.loads(DEFAULT_MEMORY_OVERVIEW.read_text(encoding="utf-8"))
    compilation = compile_natural_windows(expanded)
    windows = compilation["windows"]
    if not compilation["valid"] or len(windows) != WINDOW_CALLS:
        raise RuntimeError("episode91716_natural_window_compilation_failed")
    entity_refs = known_entity_refs(expanded)
    prompts = {
        window["window_id"]: build_subject_binding_prompt(
            window, entity_refs=entity_refs,
        )
        for window in windows
    }
    api = replace(
        load_challenger_config(api_config_path),
        timeout_seconds=600,
        temperature=0.1,
        enable_thinking=False,
    )
    module_manifest = tree_manifest(MODULE_ROOT)

    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("frozen", "raw/prompts", "qa/windows", "reports"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    freeze = {
        "runner_version": RUNNER_VERSION,
        "created_at": _now(),
        "fresh": True,
        "no_resume": True,
        "root_id": expanded.get("root_id"),
        "turn_count": len(expanded.get("turns") or []),
        "window_count": len(windows),
        "fact_extractor_calls_max": WINDOW_CALLS,
        "fact_extractor_retries": RETRIES,
        "writer_calls": 0,
        "solver_calls": 0,
        "expansion_calls": 0,
        "memorax_calls": 0,
        "expanded_dialogue_sha256": sha256_file(DEFAULT_EXPANDED),
        "memory_overview_sha256": sha256_file(DEFAULT_MEMORY_OVERVIEW),
        "window_compiler_version": compilation["window_compiler_version"],
        "prompt_version": next(iter(prompts.values()))["prompt_version"],
        "known_entity_refs": entity_refs,
        "prompt_sha256_by_window": {
            key: value["prompt_sha256"] for key, value in prompts.items()
        },
        "module_tree_sha256": module_manifest["tree_sha256"],
        "model": api.model,
        "backend": api.base_url,
        "temperature": api.temperature,
        "enable_thinking": api.enable_thinking,
        "timeout_seconds": api.timeout_seconds,
        "completion_tokens_per_window": FACT_COMPLETION_TOKENS,
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "frozen/expanded_dialogue.json", expanded)
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
        request_id = f"fact-extractor-{window_id.lower()}-attempt-01"
        atomic_text(run_root / f"raw/prompts/{request_id}.txt", prompt["prompt"])
        base = {
            "window_id": window_id,
            "window_kind": window["window_kind"],
            "boundary_ref": window["boundary_ref"],
            "turn_count": window["turn_count"],
            "position_start": window["global_position_start"],
            "position_end": window["global_position_end"],
            "prompt_sha256": prompt["prompt_sha256"],
            "attempts": 1,
            "retries": 0,
        }
        if tree_manifest(MODULE_ROOT)["tree_sha256"] != freeze["module_tree_sha256"]:
            raise RuntimeError("episode91716_fact_first_source_drift_after_freeze")
        try:
            response = ledger.invoke(
                request_id=request_id,
                intent="local_atomic_fact_extraction_episode91716_subject_binding_v1",
                attempt=1,
                prompt_sha256=prompt["prompt_sha256"],
                call=lambda prompt_text=prompt["prompt"]: _complete_once(
                    api, prompt=prompt_text, max_tokens=FACT_COMPLETION_TOKENS,
                ),
                budget_class="fact_extractor_episode91716",
                budget_limit=WINDOW_CALLS,
                total_budget_limit=WINDOW_CALLS,
                request_metadata={
                    "window_id": window_id,
                    "retry_allowed": False,
                    "writer_calls": 0,
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
        except Exception as exc:
            window_reports.append({
                **base, "status": "transport_error", "finish_reason": None,
                "error": f"{type(exc).__name__}:{exc}", "usage": {},
            })
            continue
        metadata = response.metadata or {}
        finish_reason = metadata.get("finish_reason")
        window_usage = metadata.get("usage") or {}
        _add_usage(usage, window_usage)
        if finish_reason == "length":
            window_reports.append({
                **base, "status": "finish_reason_length",
                "finish_reason": finish_reason, "usage": window_usage,
            })
            continue
        try:
            parsed = load_json_object_strict(response.raw_text)
        except Exception as exc:
            window_reports.append({
                **base, "status": "json_parse_failed",
                "finish_reason": finish_reason, "usage": window_usage,
                "error": f"{type(exc).__name__}:{exc}",
            })
            continue
        window_dir = run_root / "qa/windows" / window_id
        window_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(window_dir / "response_parsed.json", parsed)
        validation = validate_subject_binding_output(
            parsed, expanded_dialogue=expanded, window=window,
        )
        atomic_json(window_dir / "validation.json", validation)
        validations.append(validation)
        window_reports.append({
            **base, "status": "completed", "finish_reason": finish_reason,
            "usage": window_usage,
            "raw_fact_count": validation["raw_fact_count"],
            "mechanically_valid_count": validation["mechanically_valid_count"],
            "quarantine_count": validation["quarantine_count"],
        })

    facts = materialize_subject_bound_facts(
        validations,
        windows_by_id={window["window_id"]: window for window in windows},
    )
    bundles = bundle_facts(facts)
    atomic_json(run_root / "qa/materialized_facts.json", {
        "fact_schema_version": "local_atomic_fact_subject_binding_v1", "facts": facts,
    })
    atomic_json(run_root / "qa/candidate_bundles.json", bundles)
    atomic_json(run_root / "qa/window_reports.json", {"windows": window_reports})
    ledger_summary = ledger.summary()
    successful = sum(row["status"] == "completed" for row in window_reports)
    report = {
        "runner_version": RUNNER_VERSION,
        "status": "completed" if successful == len(windows) else "completed_with_window_failures",
        "run_root": str(run_root),
        "root_id": expanded.get("root_id"),
        "turn_count": len(expanded.get("turns") or []),
        "window_count": len(windows),
        "successful_window_count": successful,
        "failed_window_count": len(windows) - successful,
        "fact_extractor_calls": ledger_summary["request_count"],
        "fact_extractor_retries": 0,
        "raw_fact_count": sum(row.get("raw_fact_count", 0) for row in window_reports),
        "mechanically_valid_fact_count": sum(row.get("mechanically_valid_count", 0) for row in window_reports),
        "mechanical_quarantine_count": sum(row.get("quarantine_count", 0) for row in window_reports),
        "materialized_fact_count": len(facts),
        "bundle_count": bundles["bundle_count"],
        "bundle_class_counts": bundles.get("materialized_class_counts") or {},
        "usage": usage,
        "window_reports": window_reports,
        "writer_calls": 0,
        "solver_calls": 0,
        "expansion_calls": 0,
        "memorax_calls": 0,
        "mcq_generated": False,
        "next_stage_allowed": "offline_fact_audit_before_reverse_writer",
    }
    atomic_json(run_root / "reports/SUBJECT_BINDING_EPISODE91716_LIVE_REPORT.json", report)
    atomic_text(run_root / "reports/SUBJECT_BINDING_EPISODE91716_LIVE_REPORT.md", "\n".join([
        "# Episode 91716 Subject-Binding Fact Live Report", "",
        f"- Status: `{report['status']}`",
        f"- Windows: `{successful}/{len(windows)}`",
        f"- Fact calls / retries: `{report['fact_extractor_calls']} / 0`",
        f"- Raw / valid / quarantine facts: `{report['raw_fact_count']} / {report['mechanically_valid_fact_count']} / {report['mechanical_quarantine_count']}`",
        f"- Materialized facts: `{len(facts)}`",
        f"- Bundles: `{bundles['bundle_count']}`",
        f"- Tokens: `{json.dumps(usage, sort_keys=True)}`",
        "- Writer / Solver / Expansion / Memorax: `0 / 0 / 0 / 0`",
        "",
    ]))
    atomic_json(run_root / "reports/POST_RUN_INTEGRITY.json", {
        "module_tree_sha256_before": freeze["module_tree_sha256"],
        "module_tree_sha256_after": tree_manifest(MODULE_ROOT)["tree_sha256"],
        "module_unchanged_after_first_api": freeze["module_tree_sha256"] == tree_manifest(MODULE_ROOT)["tree_sha256"],
        "expanded_sha256_before": freeze["expanded_dialogue_sha256"],
        "expanded_sha256_after": sha256_file(DEFAULT_EXPANDED),
        "expanded_unchanged": freeze["expanded_dialogue_sha256"] == sha256_file(DEFAULT_EXPANDED),
        "calls": {"fact_extractor": ledger_summary["request_count"], "retries": 0, "writer": 0, "solver": 0, "expansion": 0, "memorax": 0},
        "r2_r3": False,
        "training": False,
        "git_commit_push": False,
    })
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--run-root", type=Path, default=SUBJECT_BINDING_TARGET_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    args = parser.parse_args(argv)
    if not args.live:
        print(json.dumps({"live_executed": False, "run_preflight_first": True}, indent=2))
        return 0
    report = run_live(run_root=args.run_root.resolve(), api_config_path=args.api_config.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"].startswith("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
