"""Zero-API shadow replay of the v3 cross-arc stage conflict.

The historical artifact is never modified or silently migrated.  A copied
outline is explicitly annotated with v4-only identity fields solely to test
the deterministic conflict gate against the already observed debt duplicate.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import atomic_json, sha256_file
from .stateful_graph_expansion import (
    STATEFUL_CROSS_ARC_CONFLICT_GATE_VERSION,
    canonical_sha256, cross_arc_stage_conflict_gate,
)


DEFAULT_SOURCE = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_fresh_e2e_canary_v3/"
    "planner/raw_stateful_outline.json")
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v3_"
    "cross_arc_conflict_shadow_audit_v1")


def build_shadow(source: dict[str, Any]) -> dict[str, Any]:
    shadow = json.loads(json.dumps(source))
    for arc_position, arc in enumerate(shadow["event_arcs"], 1):
        for stage_position, stage in enumerate(arc["stages"], 1):
            stage["event_identity_key"] = (
                f"shadow_e{arc_position}_stage_{stage_position:02d}")
            stage["stage_role"] = "base_fact_completion"
            stage["predecessor_stage_ref"] = None
    # Explicit audit annotation from the two frozen debt stages.  This is not
    # a production migration or a semantic heuristic.
    for arc_position, stage_position in ((1, 4), (2, 2)):
        shadow["event_arcs"][arc_position - 1]["stages"][stage_position - 1][
            "event_identity_key"] = "friend_c_debt_100"
    return shadow


def run(*, source_path: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"shadow_audit_root_exists:{output_root}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    shadow = build_shadow(source)
    gate = cross_arc_stage_conflict_gate(shadow["event_arcs"])
    report = {
        "schema_version": "cross_arc_conflict_shadow_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_calls": 0,
        "historical_source_modified": False,
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "source_canonical_sha256": canonical_sha256(source),
        "shadow_annotation_policy": "explicit_test_only_no_semantic_inference",
        "explicit_identity_annotations": [
            {"stage_ref": "E1:stage_04",
             "event_identity_key": "friend_c_debt_100"},
            {"stage_ref": "E2:stage_02",
             "event_identity_key": "friend_c_debt_100"},
        ],
        "gate_version": STATEFUL_CROSS_ARC_CONFLICT_GATE_VERSION,
        "gate": gate,
        "expected_error": "duplicate_completion_obligation",
        "expected_conflicting_stage": "E2:stage_02",
        "expected_conflict_detected": (
            "duplicate_completion_obligation" in gate["errors"] and
            "E2:stage_02" in gate["conflicting_stage_refs"]),
        "shadow_sha256": canonical_sha256(shadow),
    }
    output_root.mkdir(parents=True)
    atomic_json(output_root / "frozen/explicitly_annotated_shadow_outline.json",
                shadow)
    atomic_json(output_root / "reports/cross_arc_conflict_shadow_audit.json",
                report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run(source_path=args.source, output_root=args.output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["expected_conflict_detected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
