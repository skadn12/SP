from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, sha256_file, write_json

RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PROBE_VERSION = "planner_realizer_temporal_scene_release_summary_v1"

CASES = [
    {
        "case_id": "POB03",
        "probe_run": "planner_realizer_temporal_scene_release_probe_v4_POB03_20260823T020000Z",
        "audit_run": "planner_realizer_temporal_scene_release_reanalysis_v4_POB03_20260823T023000Z",
        "protocol": "v4",
    },
    {
        "case_id": "POB04",
        "probe_run": "planner_realizer_temporal_scene_release_probe_v2_POB04_20260823T013000Z",
        "audit_run": "planner_realizer_temporal_scene_release_reanalysis_v2_POB04_20260823T020000Z",
        "protocol": "v2",
    },
    {
        "case_id": "POS07",
        "probe_run": "planner_realizer_temporal_scene_release_probe_v4_POS07_20260823T030000Z",
        "audit_run": "planner_realizer_temporal_scene_release_reanalysis_v4_POS07_20260823T034500Z",
        "protocol": "v4",
    },
]


def _usage(record: dict[str, Any]) -> dict[str, Any]:
    return (record.get("response_meta") or {}).get("usage") or {}


def _row(base: Path, spec: dict[str, str]) -> dict[str, Any]:
    probe = base / spec["probe_run"]
    audit = base / spec["audit_run"]
    metrics = json.loads((probe / "metrics.json").read_text(encoding="utf-8"))
    audit_metrics = json.loads((audit / "metrics.json").read_text(encoding="utf-8"))
    records = metrics.get("scene_records") or []
    prompt_tokens = [int(_usage(x).get("prompt_tokens") or 0) for x in records]
    completion_tokens = [int(_usage(x).get("completion_tokens") or 0) for x in records]
    reasoning_tokens = [int((_usage(x).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0) for x in records]
    return {
        "case_id": spec["case_id"],
        "protocol": spec["protocol"],
        "root_id": metrics.get("root_id"),
        "probe_run": str(probe.resolve()),
        "audit_run": str(audit.resolve()),
        "probe_metrics_sha256": sha256_file(probe / "metrics.json"),
        "child_sha256": sha256_file(probe / "child_dialogue.json"),
        "scenes_completed": metrics.get("scenes_completed"),
        "realizer_calls": metrics.get("realizer_calls_completed"),
        "all_scene_mechanical_valid": all(bool(x.get("mechanical_contract_valid")) for x in records) and len(records) == 3,
        "transport_error": metrics.get("transport_error"),
        "finish_reasons": [(_usage(x), (x.get("response_meta") or {}).get("finish_reason"))[-1] for x in records],
        "prompt_tokens_by_scene": prompt_tokens,
        "completion_tokens_by_scene": completion_tokens,
        "reasoning_tokens_by_scene": reasoning_tokens,
        "strict_stage_audit_pass": audit_metrics.get("strict_stage_audit_pass"),
        "strict_stage_violations": audit_metrics.get("strict_stage_violations", audit_metrics.get("violations", [])),
        "conclusion": audit_metrics.get("conclusion"),
    }


def run(*, base: Path, run_root: Path) -> dict[str, Any]:
    prepare_empty(run_root)
    rows = [_row(base, spec) for spec in CASES]
    strict_passes = sum(bool(row["strict_stage_audit_pass"]) for row in rows)
    complete = sum(bool(row["all_scene_mechanical_valid"]) and not row["transport_error"] for row in rows)
    same_protocol_passes = sum(bool(row["strict_stage_audit_pass"]) for row in rows if row["protocol"] == "v4")
    decision = "READY_FOR_ISOLATED_INTEGRATION_PREFLIGHT" if strict_passes == 3 and complete == 3 else "HOLD_AND_COLLECT_MORE_PROBES"
    metrics = {
        "probe_version": PROBE_VERSION,
        "isolated": True,
        "source_runs_read_only": [row["probe_run"] for row in rows] + [row["audit_run"] for row in rows],
        "cases": rows,
        "strict_stage_pass_count": strict_passes,
        "complete_transport_clean_count": complete,
        "v4_same_protocol_pass_count": same_protocol_passes,
        "decision": decision,
        "interpretation": {
            "positive_signal": "All three selected roots complete under scene-local release and pass strict future-trigger ordering.",
            "residual_risk": "Two roots use v4 max_tokens=4096; POB04 uses prior v2 budget. Historical POB03 v3 timeout remains an independent transport/serialization risk.",
            "scope": "Probe evidence only; no production prompt, QA, Planner, Solver, Writer, Auditor, or Memorax changes.",
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
    write_json(run_root / "manifest.json", {"probe_version": PROBE_VERSION, "isolated": True, "api_calls": 0, "source_runs_read_only": metrics["source_runs_read_only"]})
    lines = ["# Realizer Temporal Scene-Release Summary v1", "", f"Decision: **{decision}**", "", f"- Strict stage pass: {strict_passes}/3", f"- Complete and transport-clean: {complete}/3", f"- v4 same-protocol pass: {same_protocol_passes}/2", "", "## Case results", ""]
    for row in rows:
        lines.append(f"- {row['case_id']} ({row['protocol']}): {row['conclusion']}; scenes={row['scenes_completed']}/3; calls={row['realizer_calls']}; strict={row['strict_stage_audit_pass']}; violations={row['strict_stage_violations'] or 'none'}")
    lines.extend(["", "## Decision boundary", "", "This is a zero-API read-only synthesis. It authorizes at most an isolated integration preflight; it does not authorize production or QA changes.", "", "## Risks", "", "Historical POB03 v3 timeout and POB03 v2 envelope failure remain preserved as separate prior evidence.", ""])
    (run_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="zero-API scene-release cross-root summary")
    parser.add_argument("--base", type=Path, default=RUN_BASE)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"{PROBE_VERSION}_{stamp}").resolve()
    result = run(base=args.base, run_root=run_root)
    print(json.dumps({"run_root": str(run_root), "metrics": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
