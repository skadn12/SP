"""Isolated Fresh R1 full-context/development-diversity experiment runner."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from ..simple_append_v1.qa_live import build_local_solver_runtime
from .qa_live import DEFAULT_SOLVER_MODEL, run as run_qa
from .single_root_live import (
    DEFAULT_API_CONFIG,
    DEFAULT_INPUT,
    DEFAULT_MEMORAX_CREDENTIALS,
    DEFAULT_TOKENIZER,
    execute_live as execute_expansion,
    run_preflight as run_expansion_preflight,
)


RUNNER_VERSION = "simple_append_memorax_fullr1_devdiv_probe_v1"
DEFAULT_EXPERIMENT_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_"
    "simple_append_memorax_fullr1_devdiv_probe_v1_experiment_1"
)
MODULE_ROOT = Path(__file__).resolve().parent
MAINLINE_MANIFEST = MODULE_ROOT / "MAINLINE_BASELINE_SHA256.txt"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _verify_mainline_manifest() -> dict[str, Any]:
    errors: list[str] = []
    checked = 0
    for line in MAINLINE_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, path_text = line.split(maxsplit=1)
        path = Path(path_text.strip())
        checked += 1
        if not path.is_file():
            errors.append(f"missing:{path}")
        elif sha256_file(path) != expected:
            errors.append(f"sha256_mismatch:{path}")
    return {
        "checked_file_count": checked,
        "errors": errors,
        "unchanged": not errors,
    }


def preflight(
    *, experiment_root: Path, input_path: Path, api_config: Path,
    tokenizer_path: Path, solver_model: Path, memorax_credentials: Path,
    allow_insecure_http: bool,
) -> dict[str, Any]:
    expansion_root = experiment_root / "expansion"
    qa_root = experiment_root / "qa"
    mainline = _verify_mainline_manifest()
    expansion = run_expansion_preflight(
        input_path=input_path,
        run_root=expansion_root,
        api_config_path=api_config,
        tokenizer_path=tokenizer_path,
        memorax_credentials_path=memorax_credentials,
        allow_insecure_memorax_http=allow_insecure_http,
        reuse_existing_parent_memory=False,
        wait_for_memory_completion=True,
    )
    checks = {
        "experiment_root_absent": not experiment_root.exists(),
        "qa_root_absent": not qa_root.exists(),
        "protected_mainline_unchanged": mainline["unchanged"],
        "fresh_non_61555_root": expansion.get("root_id") not in {None, "cc-train-episode-61555"},
        "expected_probe_root": expansion.get("root_id") == "cc-train-episode-146700",
        "expansion_preflight_passed": expansion.get("decision") == "pass",
        "full_parent_selected": bool(
            (expansion.get("selected_parent_context") or {}).get("full_parent")),
        "raw_dialogue_budget_4096": bool(
            (expansion.get("checks") or {}).get("raw_dialogue_budget_frozen")),
        "solver_model_exists": solver_model.is_dir(),
        "api_calls_zero": True,
    }
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "preflight_version": RUNNER_VERSION,
        "decision": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "experiment_root": str(experiment_root.resolve()),
        "expansion_root": str(expansion_root.resolve()),
        "qa_root": str(qa_root.resolve()),
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path) if input_path.is_file() else None,
        "mainline_integrity": mainline,
        "expansion_preflight": expansion,
        "api_calls": 0,
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Fresh R1 Full-Context Development-Diversity Probe",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Root: `{report.get('root_id')}`",
        f"- Expansion ready: `{str(bool(report.get('generation_ready'))).lower()}`",
        f"- QA pipeline ready: `{str(bool(report.get('qa_pipeline_ready'))).lower()}`",
        f"- Mainline unchanged: `{str(bool(report.get('original_mainline_modified') is False)).lower()}`",
        f"- R2/R3 executed: `false`",
        f"- Training executed: `false`",
        "",
        "The development-diversity semantic decision must be completed by inspecting",
        "the saved Planner scene fields and generated dialogue; it is not replaced by",
        "a local keyword gate.",
        "",
    ]
    return "\n".join(lines)


def execute(
    *, frozen: Mapping[str, Any], experiment_root: Path, input_path: Path,
    api_config: Path, tokenizer_path: Path, solver_model: Path,
    memorax_credentials: Path, allow_insecure_http: bool,
) -> dict[str, Any]:
    if frozen.get("decision") != "pass":
        raise RuntimeError("FULLR1_DEVDIV_PRE_API_PREFLIGHT_FAILED")
    if experiment_root.exists():
        raise FileExistsError(f"experiment_root_exists:{experiment_root}")

    # Solver loading is a pre-paid-request gate. It must succeed before the
    # experiment root or any Planner/Realizer request is created.
    solver_runtime = build_local_solver_runtime(
        tokenizer_path=tokenizer_path, solver_model=solver_model)
    experiment_root.mkdir(parents=True, exist_ok=False)
    (experiment_root / "reports").mkdir(parents=True, exist_ok=True)
    atomic_json(experiment_root / "run_freeze_manifest.json", {
        **dict(frozen), "frozen_at": _now(), "fresh": True, "no_resume": True,
    })
    expansion_root = experiment_root / "expansion"
    qa_root = experiment_root / "qa"
    expansion: dict[str, Any] = {}
    qa: dict[str, Any] = {}
    failure: str | None = None
    try:
        expansion = execute_expansion(
            preflight=frozen["expansion_preflight"],
            input_path=input_path,
            run_root=expansion_root,
            api_config_path=api_config,
            tokenizer_path=tokenizer_path,
            memorax_credentials_path=memorax_credentials,
            allow_insecure_memorax_http=allow_insecure_http,
            reuse_existing_parent_memory=False,
            wait_for_memory_completion=True,
        )
        if expansion.get("status") != "completed":
            raise RuntimeError("probe_expansion_failed")
        qa = run_qa(
            source_run=expansion_root,
            memory_recovery=expansion_root,
            run_root=qa_root,
            api_config=api_config,
            tokenizer_path=tokenizer_path,
            solver_model=solver_model,
            solver_runtime=solver_runtime,
        )
    except Exception as error:
        failure = f"{type(error).__name__}:{error}"
    finally:
        close = getattr(solver_runtime[1], "close", None)
        if callable(close):
            close()

    mainline_after = _verify_mainline_manifest()
    context = expansion.get("selected_parent_context") or {}
    qa_context = expansion.get("qa_context_selection") or {}
    report = {
        "report_version": RUNNER_VERSION,
        "status": "completed" if failure is None else "failed",
        "failure": failure,
        "root_id": frozen["expansion_preflight"].get("root_id"),
        "experiment_root": str(experiment_root),
        "expansion": expansion,
        "qa": qa,
        "context": {
            "parent_sessions": context.get("parent_session_count"),
            "parent_turns": context.get("parent_turn_count"),
            "parent_raw_tokens": context.get("full_parent_raw_dialogue_tokens"),
            "planner_selected_sessions": context.get("selected_session_count"),
            "planner_selected_turns": context.get("selected_turn_count"),
            "planner_selected_raw_tokens": context.get("selected_raw_dialogue_tokens"),
            "planner_full_parent": context.get("full_parent_included"),
            "realizer_full_parent": context.get("full_parent_included"),
            "qa_context_selection": qa_context,
        },
        "generation_ready": bool(expansion.get("generation_ready")),
        "qa_pipeline_ready": bool(qa.get("qa_pipeline_ready")),
        "original_mainline_modified": not mainline_after["unchanged"],
        "mainline_integrity_after": mainline_after,
        "old_runs_modified": False,
        "old_memory_modified": False,
        "r2_r3_executed": False,
        "training_executed": False,
        "git_commit_push": False,
        "development_diversity_manual_review_required": True,
        "completed_at": _now(),
    }
    atomic_json(experiment_root / "reports/FULLR1_DEVDIV_PROBE_REPORT.json", report)
    atomic_text(
        experiment_root / "reports/FULLR1_DEVDIV_PROBE_REPORT.md",
        _report_markdown(report))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument(
        "--memorax-credentials", type=Path,
        default=DEFAULT_MEMORAX_CREDENTIALS)
    parser.add_argument("--allow-insecure-memorax-http", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    frozen = preflight(
        experiment_root=args.run_root.resolve(),
        input_path=args.input.resolve(),
        api_config=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve(),
        solver_model=args.solver_model.resolve(),
        memorax_credentials=args.memorax_credentials.resolve(),
        allow_insecure_http=args.allow_insecure_memorax_http,
    )
    if not args.live:
        atomic_json(MODULE_ROOT / "reports/FULLR1_DEVDIV_PROBE_PREFLIGHT.json", frozen)
        print(json.dumps(frozen, ensure_ascii=False, indent=2))
        return 0 if frozen["decision"] == "pass" else 2
    if frozen["decision"] != "pass":
        print(json.dumps(frozen, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    report = execute(
        frozen=frozen,
        experiment_root=args.run_root.resolve(),
        input_path=args.input.resolve(),
        api_config=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve(),
        solver_model=args.solver_model.resolve(),
        memorax_credentials=args.memorax_credentials.resolve(),
        allow_insecure_http=args.allow_insecure_memorax_http,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
