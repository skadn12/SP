from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, sha256_file, write_json

RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")


def run(*, source_run: Path, run_root: Path) -> dict:
    prepare_empty(run_root)
    metrics = json.loads((source_run / "metrics.json").read_text())
    error = str(metrics.get("transport_error") or "")
    completed = int(metrics.get("realizer_calls_completed") or 0)
    if error and "Timeout" in error:
        conclusion = "SCENE_RELEASE_V3_UNDECIDED_TIMEOUT"
        reason = "Scene-release generation timed out before all scenes completed; no temporal conclusion is identified and no retry is permitted."
    elif int(metrics.get("scenes_completed") or 0) == 3 and metrics.get("final_action_audit_pass"):
        conclusion = "SCENE_RELEASE_V3_STRICT_PASS"
        reason = "All three scene-release outputs completed and passed the strict audit."
    else:
        conclusion = "SCENE_RELEASE_V3_FAIL"
        reason = "The scene-release run completed without a timeout but did not pass the strict audit."
    out = {
        "probe_version": "planner_realizer_temporal_scene_release_reanalysis_v3",
        "source_run": str(source_run.resolve()),
        "source_metrics_sha256": sha256_file(source_run / "metrics.json"),
        "source_child_sha256": sha256_file(source_run / "child_dialogue.json"),
        "source_realizer_calls_completed": completed,
        "source_scenes_completed": int(metrics.get("scenes_completed") or 0),
        "source_transport_error": error,
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
    write_json(run_root / "manifest.json", {"probe_version": out["probe_version"], "isolated": True, "source_run_read_only": str(source_run.resolve()), "api_calls": 0, "retry_budget": 0})
    (run_root / "report.md").write_text(
        "# Realizer Temporal Scene Release Reanalysis v3\n\n"
        f"Verdict: **{conclusion}**\n\n{reason}\n\n"
        "This reanalysis made zero API calls and does not modify the source run.\n",
        encoding="utf-8",
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="zero-API scene-release timeout reanalysis")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"planner_realizer_temporal_scene_release_reanalysis_v3_{stamp}").resolve()
    print(json.dumps({"run_root": str(run_root), "metrics": run(source_run=args.source_run, run_root=run_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
