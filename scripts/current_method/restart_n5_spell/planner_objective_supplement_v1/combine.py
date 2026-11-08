from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_BATCH = RUN_BASE / "planner_objective_probe_v1_batch_20260822T110000Z"
DEFAULT_SUPPLEMENT = RUN_BASE / "planner_objective_supplement_v1_20260822T150000Z"
PROBE_VERSION = "planner_objective_combined_stability_v1"


def run(*, batch_run: Path, supplement_run: Path, run_root: Path) -> dict[str, Any]:
    prepare_empty(run_root)
    batch_records = read_jsonl(batch_run / "planner_cases.jsonl")
    supplement_records = read_jsonl(supplement_run / "planner_cases.jsonl")
    batch_metrics = read_json(batch_run / "metrics.json")
    supplement_metrics = read_json(supplement_run / "metrics.json")
    if not supplement_metrics.get("paired_complete"):
        raise ValueError("supplement_pair_incomplete")
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in [*batch_records, *supplement_records]:
        by_case.setdefault(str(row.get("case_id") or ""), {})[str(row.get("objective") or "")] = row
    complete = []
    for case_id, rows in sorted(by_case.items()):
        if all(rows.get(key, {}).get("contract", {}).get("valid") for key in ("O0", "O1")):
            o0 = bool(rows["O0"].get("analysis", {}).get("complex_chain"))
            o1 = bool(rows["O1"].get("analysis", {}).get("complex_chain"))
            complete.append({
                "case_id": case_id,
                "root_id": rows["O0"].get("root_id"),
                "event_key": rows["O0"].get("event_key"),
                "O0_complex": o0,
                "O1_complex": o1,
                "objective_gain": int(o1) - int(o0),
            })
    roots = [str(row.get("root_id") or "") for row in complete]
    if len(roots) != len(set(roots)):
        raise ValueError("complete_pair_roots_not_distinct")
    positive = sum(row["objective_gain"] > 0 for row in complete)
    negative = sum(row["objective_gain"] < 0 for row in complete)
    count = len(complete)
    label = (
        "OBJECTIVE_CONTROL_COMBINED_SIGNAL"
        if count >= 3 and positive >= 2 and negative == 0
        else "COMBINED_SIGNAL_NOT_ESTABLISHED"
    )
    metrics = {
        "probe_version": PROBE_VERSION,
        "source_runs": {
            "batch": str(batch_run.resolve()),
            "supplement": str(supplement_run.resolve()),
            "batch_planner_cases_sha256": sha256_file(batch_run / "planner_cases.jsonl"),
            "supplement_planner_cases_sha256": sha256_file(supplement_run / "planner_cases.jsonl"),
        },
        "paired_complete_cases": count,
        "distinct_complete_roots": len(set(roots)),
        "pair_outcomes": complete,
        "O0_complex_chain_rate": sum(row["O0_complex"] for row in complete) / count if count else None,
        "O1_complex_chain_rate": sum(row["O1_complex"] for row in complete) / count if count else None,
        "mean_objective_gain": sum(row["objective_gain"] for row in complete) / count if count else None,
        "positive_pairs": positive,
        "unchanged_pairs": count - positive - negative,
        "negative_pairs": negative,
        "conclusion": {
            "label": label,
            "reason": f"Across {count} distinct-root complete pairs, O1 improved {positive}, left {count - positive - negative} unchanged, and reduced {negative}.",
        },
        "source_generation_planner_calls": int(batch_metrics.get("planner_calls") or 0) + int(supplement_metrics.get("planner_calls") or 0),
        "analysis_execution_api_calls": 0,
        "realizer_calls": 0,
        "qa_writer_calls": 0,
        "solver_calls": 0,
        "writer_calls": 0,
        "memorax_search_calls": 0,
        "memorax_add_calls": 0,
        "limitations": [
            "Three complete roots remain a small directional stability sample.",
            "Generation used temperature=0 but the provider exposes no explicit sampling seed.",
            "Planner structure is measured; Realizer adherence is not yet tested.",
        ],
    }
    lines = [
        "# Planner Objective Combined Stability v1", "",
        f"Verdict: **{label}**", "",
        metrics["conclusion"]["reason"], "",
        f"- Complete distinct-root pairs: `{count}`",
        f"- O0 complex-chain rate: `{metrics['O0_complex_chain_rate']}`",
        f"- O1 complex-chain rate: `{metrics['O1_complex_chain_rate']}`",
        f"- Mean objective gain: `{metrics['mean_objective_gain']}`", "",
        "## Pair outcomes", "",
    ]
    for row in complete:
        lines.append(f"- `{row['case_id']}` / `{row['root_id']}` / `{row['event_key']}`: O0={row['O0_complex']}, O1={row['O1_complex']}, gain={row['objective_gain']:+d}")
    lines.extend(["", "## Next gate", "",
                  "This result permits one isolated Realizer-adherence canary on an accepted O1 plan. It does not permit a mainline change or QA run.", ""])
    write_jsonl(run_root / "paired_outcomes.jsonl", complete)
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {
        "probe_version": PROBE_VERSION, "isolated": True,
        "source_runs_read_only": [str(batch_run.resolve()), str(supplement_run.resolve())],
        "analysis_execution_api_calls": 0, "realizer_calls": 0,
        "qa_writer_calls": 0, "solver_calls": 0, "writer_calls": 0,
        "memorax_search_calls": 0, "memorax_add_calls": 0,
    })
    (run_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="combine frozen Planner objective pair results")
    parser.add_argument("--batch-run", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--supplement-run", type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"planner_objective_combined_stability_v1_{stamp}").resolve()
    print(json.dumps({"run_root": str(run_root), "metrics": run(
        batch_run=args.batch_run, supplement_run=args.supplement_run, run_root=run_root,
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
