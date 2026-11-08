from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty,
    sha256_file,
    write_json,
)
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.batch_probe_v1 import (
    _profile_audit,
)

RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_BATCH = RUN_BASE / "planner_realizer_temporal_boundary_batch_v1_20260822T190000Z"
DEFAULT_INTEGRATION = RUN_BASE / "planner_realizer_temporal_boundary_integration_preflight_v1_20260822T201000Z"
PROBE_VERSION = "planner_realizer_temporal_stage_transition_reanalysis_v2"


def _scene_texts(child: Mapping[str, Any]) -> list[str]:
    scenes = child.get("scenes") if isinstance(child.get("scenes"), list) else []
    return [
        " ".join(str(turn.get("text") or "").casefold() for turn in scene.get("dialogue_turns") or [])
        for scene in scenes if isinstance(scene, Mapping)
    ]


def _contains_any(text: str, patterns: list[str]) -> list[str]:
    return [pattern for pattern in patterns if pattern.casefold() in text]


def strict_stage_audit(child: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    texts = _scene_texts(child)
    base = _profile_audit(child, case_id)
    if len(texts) != 3:
        return {"audit_version": PROBE_VERSION, "case_id": case_id, "overall_pass": False, "checks": {"three_scenes": False}, "base_boundary_audit": base, "violations": ["scene_count_not_three"]}
    s1, s2, s3 = texts
    violations: list[str] = []
    evidence: dict[str, Any] = {}
    if case_id == "POB03":
        trigger = ["too thick", "overly thick", "thicker than"]
        early, owned = _contains_any(s1, trigger), _contains_any(s2, trigger)
        evidence = {"scene2_trigger_hits_in_scene1": early, "scene2_trigger_hits_in_scene2": owned}
        if early: violations.append("scene2_thickness_trigger_leaked_in_scene1")
        if not owned: violations.append("scene2_thickness_trigger_missing_in_scene2")
    elif case_id == "POB04":
        trigger = ["only available on weekends", "only free on weekends", "only available this weekend", "available this weekend", "available this saturday"]
        resolution = ["venue is only available on weekends", "venue was only available on weekends", "venue confirmed they’re available", "venue confirmed they're available", "venue confirmed its availability"]
        early, owned = _contains_any(s1, trigger), _contains_any(s2, trigger)
        resolution_early = _contains_any(" ".join([s1, s2]), resolution)
        resolution_owned = _contains_any(s3, resolution)
        if "venue" in s3 and "only available on weekends" in s3:
            resolution_owned.append("venue + only available on weekends")
        evidence = {"scene2_group_weekend_trigger_in_scene1": early, "scene2_group_weekend_trigger_in_scene2": owned, "scene3_venue_resolution_in_scenes1_2": resolution_early, "scene3_venue_resolution_in_scene3": resolution_owned}
        if early: violations.append("scene2_group_weekend_trigger_leaked_in_scene1")
        if not owned: violations.append("scene2_group_weekend_trigger_missing_in_scene2")
        if resolution_early: violations.append("scene3_venue_resolution_leaked_before_scene3")
        if not resolution_owned: violations.append("scene3_venue_resolution_missing_in_scene3")
    elif case_id == "POS07":
        trigger = ["family heirloom", "heirloom", "grandmother's vase", "grandmother’s vase", "grandfather's", "grandfather’s", "dad's old watch", "dad’s old watch"]
        resolution = ["therapist said", "therapist recommended", "therapist recommends", "therapist suggested", "therapist suggests", "therapist just recommended", "my therapist just recommended", "my therapist recommended", "recommended cognitive-behavioral", "recommended cognitive behavioral", "therapist recommended cognitive"]
        early, owned = _contains_any(s1, trigger), _contains_any(s2, trigger)
        resolution_early, resolution_owned = _contains_any(" ".join([s1, s2]), resolution), _contains_any(s3, resolution)
        evidence = {"scene2_heirloom_severity_in_scene1": early, "scene2_heirloom_severity_in_scene2": owned, "scene3_therapist_recommendation_in_scenes1_2": resolution_early, "scene3_therapist_recommendation_in_scene3": resolution_owned}
        if early: violations.append("scene2_heirloom_severity_leaked_in_scene1")
        if not owned: violations.append("scene2_heirloom_severity_missing_in_scene2")
        if resolution_early: violations.append("scene3_therapist_recommendation_leaked_before_scene3")
        if not resolution_owned: violations.append("scene3_therapist_recommendation_missing_in_scene3")
    else:
        raise ValueError(f"unsupported_case_id:{case_id}")
    checks = {"base_final_action_boundary_pass": bool(base.get("overall_pass")), "future_trigger_order_pass": not violations}
    return {"audit_version": PROBE_VERSION, "case_id": case_id, "overall_pass": bool(all(checks.values())), "checks": checks, "trigger_evidence": evidence, "violations": violations, "base_boundary_audit": base, "limitations": ["Case-specific deterministic trigger paraphrase lists; not general semantic entailment."]}


def run(*, batch_run: Path, integration_run: Path, run_root: Path) -> dict[str, Any]:
    prepare_empty(run_root)
    rows = []
    for case_id in ["POB03", "POB04", "POS07"]:
        child_path = batch_run / case_id / "child_dialogue.json"
        audit = strict_stage_audit(json.loads(child_path.read_text()), case_id)
        write_json(run_root / f"batch_{case_id}_strict_audit.json", audit)
        rows.append({"case_id": case_id, "child_sha256": sha256_file(child_path), "strict_stage_audit_pass": bool(audit["overall_pass"]), "violations": audit["violations"]})
    integration_child = integration_run / "live_child_dialogue.json"
    integration_audit = strict_stage_audit(json.loads(integration_child.read_text()), "POB04")
    write_json(run_root / "integration_POB04_strict_audit.json", integration_audit)
    batch_passed = sum(row["strict_stage_audit_pass"] for row in rows)
    integration_passed = bool(integration_audit["overall_pass"])
    metrics = {"probe_version": PROBE_VERSION, "source_batch_run": str(batch_run.resolve()), "source_batch_metrics_sha256": sha256_file(batch_run / "metrics.json"), "source_integration_run": str(integration_run.resolve()), "source_integration_metrics_sha256": sha256_file(integration_run / "live_metrics.json"), "batch_cases": rows, "batch_passed_cases": batch_passed, "batch_total_cases": len(rows), "integration_case_id": "POB04", "integration_child_sha256": sha256_file(integration_child), "integration_strict_stage_audit_pass": integration_passed, "integration_violations": integration_audit["violations"], "conclusion": "TEMPORAL_BOUNDARY_SIGNAL_PARTIAL_NOT_INTEGRATION_READY" if batch_passed < len(rows) or not integration_passed else "TEMPORAL_BOUNDARY_STRICT_STAGE_PASS", "analysis_execution_api_calls": 0, "planner_calls": 0, "qa_writer_calls": 0, "solver_calls": 0, "writer_calls": 0, "auditor_calls": 0, "memorax_search_calls": 0, "memorax_add_calls": 0}
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {"probe_version": PROBE_VERSION, "isolated": True, "source_runs_read_only": [str(batch_run.resolve()), str(integration_run.resolve())], "api_calls": 0, "supersedes_final_action_only_batch_reanalysis": True})
    lines = ["# Realizer Temporal Stage-Transition Reanalysis v2", "", f"Verdict: **{metrics['conclusion']}**", "", f"- Frozen batch strict pass: {batch_passed}/{len(rows)}", f"- Integration POB04 strict pass: {integration_passed}", "- API calls: 0", "", "This audit adds future-trigger ordering to the prior final-action-only profile.", "It does not modify or rerun any source generation.", ""]
    for row in rows:
        lines.append(f"- Batch {row['case_id']}: pass={row['strict_stage_audit_pass']}; violations={row['violations']}")
    lines.append(f"- Integration POB04: pass={integration_passed}; violations={integration_audit['violations']}")
    (run_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="zero-API strict stage-transition reanalysis")
    parser.add_argument("--batch-run", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--integration-run", type=Path, default=DEFAULT_INTEGRATION)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"{PROBE_VERSION}_{stamp}").resolve()
    result = run(batch_run=args.batch_run, integration_run=args.integration_run, run_root=run_root)
    print(json.dumps({"run_root": str(run_root), "metrics": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
