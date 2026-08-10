#!/usr/bin/env python3
"""Create the zero-API E1-E4 dependency feasibility audit package."""

from __future__ import annotations

import argparse, json, subprocess
from datetime import datetime, timezone
from pathlib import Path

from .core import atomic_json, atomic_text, sha256_file
from .outline_arc_dependency_feasibility import build_audit, render_review
from .run_canary import create_source_snapshot, project_diff_sha256
from .run_outline_dependency_mainline import PROJECT, _source

DEFAULT_ROOT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conv-41_outline_arc_dependency_feasibility_audit_v1")
HISTORICAL = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conv-41_outline_dependency_constrained_questioner_canary_v1")

def _tree(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv); root = args.run_root.resolve()
    if root.exists(): raise FileExistsError(f"run_root_exists:{root}")
    source = _source(); before = _tree(HISTORICAL); root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, root, False)
    audit = build_audit(source["outline"], source["anchors"], source["parent_dialogue"])
    atomic_json(root / "frozen/feasible_arc_operator_candidates.json", audit["items"])
    atomic_json(root / "reports/outline_arc_dependency_feasibility_audit.json", audit)
    atomic_text(root / "review/outline_arc_dependency_feasibility_review.md", render_review(audit))
    manifest = {"run_id": root.name, "mode": audit["audit_version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "project_diff_sha256": project_diff_sha256(PROJECT), **snapshot,
        "source_outline_path": str(source["input_root"] / "expansion_outline.json"),
        "source_outline_sha256": source["outline_sha256"],
        "source_parent_anchor_path": str(source["input_root"] / "parent_anchor_snapshot.json"),
        "source_parent_anchor_sha256": source["anchor_sha256"],
        "source_parent_dialogue_content_sha256": source["parent"]["source_dialogue_content_sha256"],
        "source_parent_dialogue_frozen_sha256": source["parent"]["parent_dialogue_sha256"],
        "parent_turn_count": 237, "child_turn_count": 0, "api_calls": 0,
        "questioner_changes": 0, "distractor": 0, "solver": 0, "training": 0,
        "historical_tree_before": before}
    atomic_json(root / "config/run_manifest.json", manifest)
    after = _tree(HISTORICAL); checks = {"git_head_match": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip() == manifest["git_head"],
        "project_diff_match": project_diff_sha256(PROJECT) == manifest["project_diff_sha256"],
        "outline_sha_match": sha256_file(Path(manifest["source_outline_path"])) == manifest["source_outline_sha256"],
        "anchor_sha_match": sha256_file(Path(manifest["source_parent_anchor_path"])) == manifest["source_parent_anchor_sha256"],
        "historical_run_unchanged": before == after}
    atomic_json(root / "reports/post_run_provenance.json", {"checks": checks,
        "authoritative_release_provenance": all(checks.values()),
        "provenance_mixed": not all(checks.values()), "historical_tree_after": after})
    print(json.dumps(audit, ensure_ascii=False, indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
