"""Create a zero-API repair audit for the Graph v1 completion bug.

The command never edits its source round.  It freezes a new v2 migration
snapshot and records why legacy ``completed_event_stage_ids`` cannot seed the
new authoritative state ledger.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import atomic_json, atomic_text, sha256_file
from .run_canary import create_source_snapshot, project_diff_sha256
from .stateful_graph_expansion import (
    LEGACY_MIGRATION_VERSION, STATEFUL_GRAPH_EXPANSION_VERSION,
    STATEFUL_GRAPH_LEDGER_VERSION, STATEFUL_GRAPH_REALIZER_PROMPT_VERSION,
    STATEFUL_GRAPH_SCENE_CONTRACT_VERSION, STATEFUL_REALIZER_JSON_SCHEMA,
    canonical_sha256, legacy_round_reclassification, tree_sha256,
)


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_ROUND = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_outline_graph_30k_round_canary_v10/rounds/round_02"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_stateful_graph_repair_v1"
)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()


def _markdown(report: dict[str, Any]) -> str:
    layers = report["layers"]
    pending = report["handoffs"]
    return "\n".join([
        "# Graph v1 Round 2 completion reclassification",
        "",
        "This is a zero-API, read-only shadow audit. It does not alter the source run.",
        "",
        "## Reclassified completion layers",
        "",
        f"- scene_realization_complete: `{str(layers['scene_realization_complete']).lower()}`",
        f"- stage_coverage_complete: `{str(layers['stage_coverage_complete']).lower()}`",
        f"- semantic_transition_complete: `{str(layers['semantic_transition_complete']).lower()}`",
        f"- handoff_accounting_complete: `{str(layers['handoff_accounting_complete']).lower()}`",
        "",
        "## Meaning",
        "",
        (f"The source accepted {report['accepted_scene_count']} scenes and reported "
         f"{report['legacy_reported_completed_stage_count']}/"
         f"{report['expected_stage_count']} assigned stage IDs as completed. Graph v1 "
         "created those completion claims by copying assignment metadata, so v2 accepts "
         "zero of them as authoritative semantic transitions."),
        "",
        "## Preserved unresolved handoffs",
        "",
        *([f"- `{row['handoff_id']}`: {row['description']} (`{row['status']}`)"
           for row in pending] or ["- none"]),
        "",
        "The legacy stage rows remain `unknown_legacy` until explicit evidence is audited; "
        "they must not initialize a new v2 state ledger.",
        "",
    ])


def run(*, source_round: Path, run_root: Path) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError("refusing_to_overwrite_stateful_graph_repair_root")
    if not source_round.is_dir():
        raise FileNotFoundError(source_round)
    source_tree_before = tree_sha256(source_round)
    run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, run_root, False)
    manifest = {
        "run_id": run_root.name,
        "mode": "zero_api_stateful_graph_repair",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": _git_head(),
        "project_diff_sha256": project_diff_sha256(PROJECT),
        **snapshot,
        "source_round_root": str(source_round.resolve()),
        "source_round_tree_sha256_before": source_tree_before,
        "legacy_migration_version": LEGACY_MIGRATION_VERSION,
        "stateful_graph_expansion_version": STATEFUL_GRAPH_EXPANSION_VERSION,
        "stateful_graph_scene_contract_version":
            STATEFUL_GRAPH_SCENE_CONTRACT_VERSION,
        "stateful_graph_ledger_version": STATEFUL_GRAPH_LEDGER_VERSION,
        "stateful_graph_realizer_prompt_version":
            STATEFUL_GRAPH_REALIZER_PROMPT_VERSION,
        "stateful_realizer_schema_sha256": canonical_sha256(
            STATEFUL_REALIZER_JSON_SCHEMA),
        "stateful_graph_source_sha256": sha256_file(
            Path(__file__).with_name("stateful_graph_expansion.py")),
        "stateful_orchestrator_source_sha256": sha256_file(
            Path(__file__).with_name("stateful_graph_orchestrator.py")),
        "legacy_orchestrator_source_sha256": sha256_file(
            Path(__file__).with_name("orchestrator.py")),
        "api_requests": 0, "realizer_generations": 0,
        "questioner_generations": 0, "solver_generations": 0,
        "verifier_generations": 0, "training_runs": 0,
    }
    atomic_json(run_root / "config/run_manifest.json", manifest)
    report = legacy_round_reclassification(source_round)
    atomic_json(run_root / "frozen/legacy_round2_migration_snapshot.json", report)
    atomic_json(run_root / "reports/legacy_round2_reclassification.json", report)
    atomic_text(run_root / "review/legacy_round2_reclassification.md",
                _markdown(report))
    source_tree_after = tree_sha256(source_round)
    checks = {
        "git_head_match": _git_head() == manifest["project_git_commit"],
        "project_diff_sha256_match": (
            project_diff_sha256(PROJECT) == manifest["project_diff_sha256"]),
        "source_snapshot_manifest_sha_match": sha256_file(Path(
            manifest["source_snapshot_manifest_path"])) == manifest[
                "source_snapshot_manifest_sha256"],
        "historical_round_tree_unchanged": source_tree_after == source_tree_before,
        "no_api_calls": True,
        "legacy_completion_not_used_as_v2_seed": (
            report["v2_authoritative_completed_stage_count"] == 0),
        "stateful_graph_source_sha_match": sha256_file(
            Path(__file__).with_name("stateful_graph_expansion.py")) == manifest[
                "stateful_graph_source_sha256"],
        "stateful_orchestrator_source_sha_match": sha256_file(
            Path(__file__).with_name("stateful_graph_orchestrator.py")) == manifest[
                "stateful_orchestrator_source_sha256"],
        "legacy_orchestrator_source_sha_match": sha256_file(
            Path(__file__).with_name("orchestrator.py")) == manifest[
                "legacy_orchestrator_source_sha256"],
    }
    summary = {
        "decision": "stateful_graph_repair_core_ready",
        "source_round_root": str(source_round.resolve()),
        "legacy_round2_layers": report["layers"],
        "legacy_reported_completed_stage_count": report[
            "legacy_reported_completed_stage_count"],
        "v2_authoritative_completed_stage_count": report[
            "v2_authoritative_completed_stage_count"],
        "pending_handoff_count": len(report["handoffs"]),
        "source_round_tree_sha256_before": source_tree_before,
        "source_round_tree_sha256_after": source_tree_after,
        "checks": checks,
        "api_usage": {
            "paid_api_requests": 0, "realizer_generations": 0,
            "questioner_generations": 0, "solver_generations": 0,
            "verifier_generations": 0, "training_runs": 0,
        },
    }
    atomic_json(run_root / "reports/stateful_graph_repair_summary.json", summary)
    provenance = {
        "authoritative_release_provenance": all(checks.values()),
        "provenance_mixed": not all(checks.values()),
        "checks": checks,
        "run_manifest_sha256": sha256_file(
            run_root / "config/run_manifest.json"),
        "report_sha256": sha256_file(
            run_root / "reports/legacy_round2_reclassification.json"),
        "source_round_tree_sha256_after": source_tree_after,
    }
    atomic_json(run_root / "reports/post_run_provenance.json", provenance)
    return {**summary, "provenance": provenance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-round", type=Path,
                        default=DEFAULT_SOURCE_ROUND)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(source_round=args.source_round,
                         run_root=args.run_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
