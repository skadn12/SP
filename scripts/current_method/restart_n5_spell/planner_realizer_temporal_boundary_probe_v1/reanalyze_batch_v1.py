from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty,
    sha256_file,
    write_json,
)
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.batch_probe_v1 import (
    _audit,
)


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_SOURCE = RUN_BASE / "planner_realizer_temporal_boundary_batch_v1_20260822T190000Z"


def run(*, source_run: Path, run_root: Path) -> dict:
    prepare_empty(run_root)
    source_metrics = json.loads((source_run / "metrics.json").read_text())
    rows = []
    for case_id in source_metrics["case_ids"]:
        case_root = source_run / case_id
        child_path = case_root / "child_dialogue.json"
        mechanical = json.loads((case_root / "mechanical_audit.json").read_text())
        child = json.loads(child_path.read_text()) if child_path.exists() else {}
        semantic = _audit(child, case_id) if mechanical.get("valid") else {"overall_pass": False, "checks": {}}
        row = {
            "case_id": case_id,
            "root_id": next(item["root_id"] for item in source_metrics["cases"] if item["case_id"] == case_id),
            "source_child_sha256": sha256_file(child_path) if child_path.exists() else None,
            "mechanical_contract_valid": bool(mechanical.get("valid")),
            "temporal_audit_pass": bool(semantic.get("overall_pass")),
            "source_transport_error": next(item["transport_error"] for item in source_metrics["cases"] if item["case_id"] == case_id),
            "semantic_audit": semantic,
        }
        write_json(run_root / f"{case_id}_semantic_audit.json", semantic)
        rows.append(row)
    passed = sum(row["temporal_audit_pass"] and row["mechanical_contract_valid"] for row in rows)
    completed = sum(next(item["realizer_calls_completed"] for item in source_metrics["cases"] if item["case_id"] == row["case_id"]) > 0 for row in rows)
    metrics = {
        "probe_version": "planner_realizer_temporal_boundary_batch_reanalysis_v1",
        "source_run": str(source_run.resolve()),
        "source_metrics_sha256": sha256_file(source_run / "metrics.json"),
        "source_case_child_sha256": {row["case_id"]: row["source_child_sha256"] for row in rows},
        "cases": rows,
        "passed_cases": passed,
        "completed_cases": completed,
        "total_cases": len(rows),
        "conclusion": "TEMPORAL_BOUNDARY_BATCH_REANALYSIS_PASS" if passed == len(rows) else "TEMPORAL_BOUNDARY_BATCH_REANALYSIS_MIXED",
        "analysis_execution_api_calls": 0,
        "planner_calls": 0,
        "qa_writer_calls": 0,
        "solver_calls": 0,
        "writer_calls": 0,
        "auditor_calls": 0,
        "memorax_search_calls": 0,
        "memorax_add_calls": 0,
    }
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {"probe_version": metrics["probe_version"], "isolated": True, "source_run_read_only": str(source_run.resolve()), "analysis_execution_api_calls": 0, "api_calls": 0})
    lines = [
        "# Planner Realizer Temporal Boundary Batch Reanalysis v1",
        "",
        f"Verdict: **{metrics['conclusion']}**",
        "",
        "This is a zero-API reanalysis of the already generated child dialogues. It does not rerun Realizer and does not modify the source batch.",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['case_id']}` (`{row['root_id']}`): mechanical={row['mechanical_contract_valid']}, temporal={row['temporal_audit_pass']}, source_error={row['source_transport_error'] or 'none'}")
    (run_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="zero-API temporal-boundary batch reanalysis")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"planner_realizer_temporal_boundary_batch_reanalysis_v1_{stamp}").resolve()
    print(json.dumps({"run_root": str(run_root), "metrics": run(source_run=args.source_run, run_root=run_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
