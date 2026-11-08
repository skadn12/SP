from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty,
    read_json,
    sha256_file,
    write_json,
)

from .canary import PROBE_VERSION, _report, semantic_audit


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_SOURCE = RUN_BASE / "planner_realizer_adherence_canary_v1_20260822T160000Z"


def run(*, source_run: Path, run_root: Path) -> dict[str, Any]:
    prepare_empty(run_root)
    child_path = source_run / "child_dialogue.json"
    mechanical_path = source_run / "mechanical_audit.json"
    metrics_path = source_run / "metrics.json"
    if not child_path.exists() or not mechanical_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"reanalysis_source_missing:{source_run}")
    child = read_json(child_path)
    mechanical = read_json(mechanical_path)
    source_metrics = read_json(metrics_path)
    semantic = semantic_audit(child)
    passed = bool(mechanical.get("valid") and semantic.get("overall_pass"))
    metrics: dict[str, Any] = {
        "probe_version": "planner_realizer_adherence_canary_reanalysis_v2",
        "source_probe_version": source_metrics.get("probe_version"),
        "source_run": str(source_run.resolve()),
        "source_child_sha256": sha256_file(child_path),
        "source_mechanical_audit_sha256": sha256_file(mechanical_path),
        "source_generation_realizer_calls": source_metrics.get("realizer_calls"),
        "analysis_execution_api_calls": 0,
        "mechanical_contract_valid": bool(mechanical.get("valid")),
        "semantic_evidence_pass": bool(semantic.get("overall_pass")),
        "conclusion": "REALIZER_ADHERENCE_CANARY_PASS" if passed else "REALIZER_ADHERENCE_CANARY_FAIL_PREMATURE_CLOSURE",
        "realizer_calls": 0,
        "planner_calls": 0,
        "qa_writer_calls": 0,
        "solver_calls": 0,
        "writer_calls": 0,
        "auditor_calls": 0,
        "memorax_search_calls": 0,
        "memorax_add_calls": 0,
    }
    report_metrics = {**metrics, "realizer_calls": source_metrics.get("realizer_calls")}
    write_json(run_root / "semantic_adherence_audit.json", semantic)
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {
        "probe_version": metrics["probe_version"], "isolated": True,
        "source_run_read_only": str(source_run.resolve()),
        "analysis_execution_api_calls": 0, "realizer_calls": 0,
        "planner_calls": 0, "qa_writer_calls": 0, "solver_calls": 0,
        "writer_calls": 0, "auditor_calls": 0,
        "memorax_search_calls": 0, "memorax_add_calls": 0,
    })
    (run_root / "report.md").write_text(
        _report(report_metrics, mechanical, semantic), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="zero-API Realizer canary reanalysis")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"planner_realizer_adherence_canary_reanalysis_v2_{stamp}").resolve()
    print(json.dumps({"run_root": str(run_root), "metrics": run(
        source_run=args.source_run, run_root=run_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
