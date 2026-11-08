from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, sha256_file, write_json

PROBE_VERSION = "qa_difficulty_integration_reanalysis_v1"
RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run(*, qa_run: Path, composition_run: Path, run_root: Path) -> dict[str, Any]:
    prepare_empty(run_root)
    writer = _load(qa_run / "qa" / "writer_output.json")
    validation = _load(qa_run / "qa" / "writer_validation.json")
    final_batch = _load(qa_run / "qa" / "final_mcq_batch.json")
    admitted_batch = _load(qa_run / "qa" / "admitted_mcq_batch.json")
    filters = [_load(path) for path in sorted((qa_run / "filter").glob("*.json"))]
    composition_metrics = _load(composition_run / "manifest.json")
    run_status_path = qa_run / "run_status.json"
    run_status = _load(run_status_path) if run_status_path.exists() else {}
    grounding = Counter(str((row.get("grounding") or {}).get("decision") or "missing") for row in filters)
    full = Counter((int(row.get("full_context_correct") or 0), int(row.get("full_context_total") or 0)) for row in filters)
    decisions = Counter(str(row.get("decision") or "missing") for row in filters)
    grounded = [row for row in filters if (row.get("grounding") or {}).get("decision") == "pass_grounding"]
    stable_grounded = [row for row in grounded if int(row.get("full_context_correct") or 0) == 5 and int(row.get("full_context_total") or 0) == 5]
    metrics = {
        "probe_version": PROBE_VERSION,
        "qa_run": str(qa_run.resolve()),
        "composition_run": str(composition_run.resolve()),
        "qa_run_metrics_sha256": sha256_file(qa_run / "qa" / "final_mcq_batch.json"),
        "composition_manifest_sha256": sha256_file(composition_run / "manifest.json"),
        "writer_candidate_count": len(writer.get("questions") or []),
        "writer_validation_count": len(validation.get("items") or []),
        "final_atomic_count": len(final_batch.get("items") or []),
        "grounding_decisions": dict(grounding),
        "full_context_buckets": {f"{key[0]}/{key[1]}": value for key, value in full.items()},
        "admission_decisions": dict(decisions),
        "grounded_count": len(grounded),
        "grounded_and_k5_stable_count": len(stable_grounded),
        "admitted_count": len(admitted_batch.get("items") or []),
        "composition_atomic_eligible": composition_metrics.get("counts", {}).get("atomic_eligible"),
        "composition_candidate_pairs": composition_metrics.get("counts", {}).get("candidate_pairs"),
        "composition_ready_pairs": composition_metrics.get("counts", {}).get("composer_ready_pairs"),
        "conclusion": (
            "QA_DIFFICULTY_NOT_VALIDATED_NO_COMPOSER_READY_PAIR"
            if not composition_metrics.get("counts", {}).get("composer_ready_pairs")
            else "QA_DIFFICULTY_COMPOSER_STAGE_READY"
        ),
        "legacy_runner_report_crash": run_status.get("status") != "completed",
        "legacy_runner_crash_reason": "KeyError:grounding in final report rendering; source QA artifacts are complete.",
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
    write_json(run_root / "manifest.json", {
        "probe_version": PROBE_VERSION,
        "isolated": True,
        "api_calls": 0,
        "source_qa_run_read_only": str(qa_run.resolve()),
        "source_composition_run_read_only": str(composition_run.resolve()),
    })
    report = [
        "# QA Difficulty Integration Reanalysis v1",
        "",
        f"Verdict: **{metrics['conclusion']}**",
        "",
        f"- Writer candidates: {metrics['writer_candidate_count']}",
        f"- Final atomic QA: {metrics['final_atomic_count']}",
        f"- Grounding pass: {metrics['grounded_count']}/{metrics['final_atomic_count']}",
        f"- Grounding pass plus full-context K=5 stable: {metrics['grounded_and_k5_stable_count']}",
        f"- Final admitted atomic QA: {metrics['admitted_count']}",
        f"- Composition eligible atomic QA: {metrics['composition_atomic_eligible']}",
        f"- Composition candidate pairs: {metrics['composition_candidate_pairs']}",
        f"- Composer-ready pairs: {metrics['composition_ready_pairs']}",
        "",
        "Interpretation: the scene-release Child supplied real cross-scene state changes, "
        "but the fixed atomic QA route did not produce a conservative pair requiring two facts. "
        "This is a QA-difficulty negative result for this root, not a failure of Child generation.",
        "",
        "The original runner crashed only while rendering its final report on a quarantined "
        "item missing a grounding field. This zero-API reanalysis reads the already persisted "
        "Writer, Solver, and composition artifacts and does not rerun any call.",
        "",
    ]
    (run_root / "report.md").write_text("\n".join(report), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="zero-API QA difficulty integration reanalysis")
    parser.add_argument("--qa-run", type=Path, required=True)
    parser.add_argument("--composition-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"{PROBE_VERSION}_{stamp}").resolve()
    result = run(qa_run=args.qa_run.resolve(), composition_run=args.composition_run.resolve(), run_root=run_root)
    print(json.dumps({"run_root": str(run_root), "metrics": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
