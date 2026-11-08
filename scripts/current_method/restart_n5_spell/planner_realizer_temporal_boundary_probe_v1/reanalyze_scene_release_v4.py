from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, sha256_file, write_json
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.stage_transition_reanalysis_v2 import strict_stage_audit

RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PROBE_VERSION = "planner_realizer_temporal_scene_release_reanalysis_v4"


def run(*, source_run: Path, run_root: Path, case_id: str = "POB03") -> dict:
    prepare_empty(run_root)
    metrics = json.loads((source_run / "metrics.json").read_text(encoding="utf-8"))
    child_path = source_run / "child_dialogue.json"
    child = json.loads(child_path.read_text(encoding="utf-8"))
    audit = strict_stage_audit(child, case_id)
    write_json(run_root / "strict_stage_audit.json", audit)
    transport_error = str(metrics.get("transport_error") or "")
    complete = int(metrics.get("scenes_completed") or 0) == 3
    if transport_error:
        conclusion = "SCENE_RELEASE_V4_UNDECIDED_TIMEOUT"
        reason = "The source run contains a transport error; strict content checks are not treated as a completed chain."
    elif complete and audit.get("overall_pass"):
        conclusion = "SCENE_RELEASE_V4_STRICT_PASS"
        reason = "All three scene-local requests completed, passed the mechanical/final-action audit, and passed the strict stage-transition audit."
    else:
        conclusion = "SCENE_RELEASE_V4_FAIL"
        reason = "The source run completed without a transport error but did not pass the strict stage-transition audit."
    out = {
        "probe_version": PROBE_VERSION,
        "case_id": case_id,
        "source_run": str(source_run.resolve()),
        "source_metrics_sha256": sha256_file(source_run / "metrics.json"),
        "source_child_sha256": sha256_file(child_path),
        "source_generation_budget_max_tokens": metrics.get("generation_budget_max_tokens"),
        "source_scenes_completed": int(metrics.get("scenes_completed") or 0),
        "source_realizer_calls_completed": int(metrics.get("realizer_calls_completed") or 0),
        "source_transport_error": transport_error or None,
        "strict_stage_audit_pass": bool(audit.get("overall_pass")),
        "strict_stage_violations": audit.get("violations", []),
        "conclusion": conclusion,
        "reason": reason,
        "analysis_execution_api_calls": 0,
        "planner_calls": 0,
        "qa_writer_calls": 0,
        "solver_calls": 0,
        "writer_calls": 0,
        "auditor_calls": 0,
        "memorax_search_calls": 0,
        "memorax_add_calls": 0,
    }
    write_json(run_root / "metrics.json", out)
    write_json(
        run_root / "manifest.json",
        {
            "probe_version": PROBE_VERSION,
            "isolated": True,
            "source_run_read_only": str(source_run.resolve()),
            "api_calls": 0,
            "retry_budget": 0,
            "audit": "strict_stage_transition_v2",
        },
    )
    (run_root / "report.md").write_text(
        "# Realizer Temporal Scene Release Reanalysis v4\n\n"
        f"Verdict: **{conclusion}**\n\n{reason}\n\n"
        f"Source scenes completed: {out['source_scenes_completed']}/3; "
        f"source Realizer calls: {out['source_realizer_calls_completed']}; "
        f"max_tokens: {out['source_generation_budget_max_tokens']}\n"
        f"Strict stage audit: {out['strict_stage_audit_pass']}; "
        f"violations: {out['strict_stage_violations'] or 'none'}\n\n"
        "This reanalysis made zero API calls and does not modify the source run or production code.\n",
        encoding="utf-8",
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="zero-API strict reanalysis of scene-release v4")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--case-id", default="POB03")
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"{PROBE_VERSION}_{stamp}").resolve()
    result = run(source_run=args.source_run, run_root=run_root, case_id=args.case_id)
    print(json.dumps({"run_root": str(run_root), "metrics": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
