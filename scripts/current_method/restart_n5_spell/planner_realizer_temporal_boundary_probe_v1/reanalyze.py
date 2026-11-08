from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty,
    read_json,
    sha256_file,
    write_json,
)


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_SOURCE = RUN_BASE / "planner_realizer_temporal_boundary_probe_v1_20260822T170000Z"


def run(*, source_run: Path, run_root: Path) -> dict:
    prepare_empty(run_root)
    metrics = read_json(source_run / "metrics.json")
    b = metrics.get("condition_b") or {}
    error = str(b.get("transport_error") or "")
    b_completed = int(b.get("realizer_calls") or 0) > 0
    if error and not b_completed:
        conclusion = "TEMPORAL_BOUNDARY_CONTROL_UNDECIDED_TIMEOUT"
        reason = "Condition B timed out without a Realizer response; effectiveness is not identified and no retry is permitted."
    elif metrics.get("comparison", {}).get("temporal_boundary_recovered"):
        conclusion = "TEMPORAL_BOUNDARY_CONTROL_RECOVERED"
        reason = "Condition B passed strict temporal-boundary audit while frozen A failed."
    elif not b_completed:
        conclusion = "TEMPORAL_BOUNDARY_CONTROL_UNDECIDED_NO_RESPONSE"
        reason = "Condition B did not produce an auditable Realizer response."
    else:
        conclusion = "TEMPORAL_BOUNDARY_CONTROL_NOT_RECOVERED"
        reason = "Condition B produced an auditable response but did not pass strict temporal-boundary audit."
    out = {
        "probe_version": "planner_realizer_temporal_boundary_probe_reanalysis_v1",
        "source_run": str(source_run.resolve()),
        "source_metrics_sha256": sha256_file(source_run / "metrics.json"),
        "source_b_raw_response_sha256": sha256_file(source_run / "condition_b_raw_response.txt"),
        "condition_a_temporal_audit_pass": bool((metrics.get("condition_a") or {}).get("temporal_audit_pass")),
        "condition_b_mechanical_contract_valid": bool(b.get("mechanical_contract_valid")),
        "condition_b_temporal_audit_pass": bool(b.get("temporal_audit_pass")),
        "condition_b_realizer_calls_completed": int(b.get("realizer_calls") or 0),
        "condition_b_transport_error": error,
        "analysis_execution_api_calls": 0,
        "conclusion": conclusion,
        "reason": reason,
        "retry_budget_used": 0,
        "planner_calls": 0,
        "qa_writer_calls": 0,
        "solver_calls": 0,
        "writer_calls": 0,
        "auditor_calls": 0,
        "memorax_search_calls": 0,
        "memorax_add_calls": 0,
    }
    write_json(run_root / "metrics.json", out)
    write_json(run_root / "manifest.json", {
        "probe_version": out["probe_version"], "isolated": True,
        "source_run_read_only": str(source_run.resolve()),
        "analysis_execution_api_calls": 0, "retry_budget_used": 0,
        "planner_calls": 0, "qa_writer_calls": 0, "solver_calls": 0,
        "writer_calls": 0, "auditor_calls": 0,
        "memorax_search_calls": 0, "memorax_add_calls": 0,
    })
    (run_root / "report.md").write_text(
        "# Planner Realizer Temporal Boundary Probe Reanalysis v1\n\n"
        f"Verdict: **{conclusion}**\n\n{reason}\n\n"
        "This reanalysis made zero API calls and does not modify the timed-out source run.\n",
        encoding="utf-8",
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="zero-API temporal-boundary timeout reanalysis")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"planner_realizer_temporal_boundary_reanalysis_v1_{stamp}").resolve()
    print(json.dumps({"run_root": str(run_root), "metrics": run(source_run=args.source_run, run_root=run_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
