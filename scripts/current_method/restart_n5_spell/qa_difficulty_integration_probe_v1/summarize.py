from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, sha256_file, write_json

PROBE_VERSION = "qa_difficulty_integration_summary_v1"
RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
SOURCES = [
    {
        "case_id": "POB03",
        "reanalysis": RUN_BASE / "qa_difficulty_integration_reanalysis_v1_POB03_20260823T061000Z",
    },
    {
        "case_id": "POS07",
        "reanalysis": RUN_BASE / "qa_difficulty_integration_reanalysis_v1_POS07_20260823T070000Z",
    },
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(*, run_root: Path) -> dict[str, Any]:
    prepare_empty(run_root)
    rows: list[dict[str, Any]] = []
    for spec in SOURCES:
        metrics = _load(spec["reanalysis"] / "metrics.json")
        rows.append({
            "case_id": spec["case_id"],
            "reanalysis_run": str(spec["reanalysis"].resolve()),
            "metrics_sha256": sha256_file(spec["reanalysis"] / "metrics.json"),
            "writer_candidates": metrics["writer_candidate_count"],
            "final_atomic": metrics["final_atomic_count"],
            "grounded": metrics["grounded_count"],
            "k5_stable": metrics["grounded_and_k5_stable_count"],
            "admitted_atomic": metrics["admitted_count"],
            "composition_eligible": metrics["composition_atomic_eligible"],
            "candidate_pairs": metrics["composition_candidate_pairs"],
            "composer_ready_pairs": metrics["composition_ready_pairs"],
            "conclusion": metrics["conclusion"],
        })
    totals = {
        key: sum(int(row[key] or 0) for row in rows)
        for key in ("writer_candidates", "final_atomic", "grounded", "k5_stable", "admitted_atomic", "composition_eligible", "candidate_pairs", "composer_ready_pairs")
    }
    decision = "HOLD_EXPANSION_DESIGN_REVISION" if totals["composer_ready_pairs"] == 0 else "PROCEED_TO_COMPOSER_LIVE_PROBE"
    metrics = {
        "probe_version": PROBE_VERSION,
        "isolated": True,
        "cases": rows,
        "totals": totals,
        "decision": decision,
        "interpretation": {
            "state_chain_signal": "Both roots had independently strict scene-release state-transition passes before QA.",
            "qa_endpoint_signal": "Neither root produced an eligible composite relation; all grounded atomic questions were reject_easy and no atomic QA was admitted.",
            "bottleneck": "The current expansion plus fixed atomic QA route does not expose two conservative, distinct, jointly necessary facts to the composition compiler.",
            "next_experiment": "Add a QA-targeted relation contract to the expansion objective: at least two linked state slots or a cross-scene actor/time/decision dependency, then rerun the same Writer, grounding, K=5, and composition gates.",
        },
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
    write_json(run_root / "manifest.json", {"probe_version": PROBE_VERSION, "isolated": True, "api_calls": 0, "source_runs_read_only": [row["reanalysis_run"] for row in rows]})
    lines = [
        "# QA Difficulty Integration Summary v1",
        "",
        f"Decision: **{decision}**",
        "",
        "## Cross-root results",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['case_id']}: Writer {row['writer_candidates']}; final atomic {row['final_atomic']}; "
            f"grounded {row['grounded']}; K=5 stable {row['k5_stable']}; admitted {row['admitted_atomic']}; "
            f"candidate pairs {row['candidate_pairs']}; Composer-ready {row['composer_ready_pairs']}"
        )
    lines.extend([
        "",
        "## Conclusion",
        "",
        "Scene-release solved a Realizer state-ordering problem in these roots, "
        "but it did not solve the end-to-end QA difficulty objective.",
        "The next expansion probe must encode a relation that the QA compiler can observe, "
        "not merely add another locally coherent state transition.",
        "",
        "This synthesis used zero API calls and did not modify any source run or production module.",
        "",
    ])
    (run_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="zero-API cross-root QA difficulty synthesis")
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"{PROBE_VERSION}_{stamp}").resolve()
    print(json.dumps(run(run_root=run_root), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
