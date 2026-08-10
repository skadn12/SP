"""Zero-API materializer for the accepted canary14 Round-1 dialogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ..core import atomic_json, atomic_text, sha256_file
from .expanded_dialogue_assembly import (
    assemble_expanded_dialogue, complete_dialogue_manifest,
    expanded_dialogue_markdown, validate_expanded_dialogue,
)
from .provenance import hash_tree


PARENT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots/cc-train-episode-61555/unified_root.json"
)
CANARY_14 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_14"
)
EXPECTED_CANARY_14_TREE_SHA = (
    "f950428152ab5937d5ba41c0774d35e60e3be63a81be450767ebc1ae6a2596b7"
)
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-derived/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_"
    "round1_complete_v1"
)


def run(*, parent_path: Path, canary_root: Path, output_root: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(f"complete_dialogue_output_exists:{output_root}")
    parent_path, canary_root = parent_path.resolve(), canary_root.resolve()
    parent_sha_before = sha256_file(parent_path)
    canary_before = hash_tree(canary_root)
    if (canary_root == CANARY_14.resolve() and
            canary_before["tree_sha256"] != EXPECTED_CANARY_14_TREE_SHA):
        raise RuntimeError("canary14_history_baseline_mismatch")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    child_path = canary_root / "child_dialogue_promotable.json"
    child = json.loads(child_path.read_text(encoding="utf-8"))
    narrative = json.loads((canary_root / "narrative_order_manifest.json").read_text(
        encoding="utf-8"))
    source = json.loads((canary_root / "source_initialization.json").read_text(
        encoding="utf-8"))
    expanded = assemble_expanded_dialogue(
        original_parent=parent, promotable_child=child,
        narrative_order_manifest=narrative,
        speaker_to_entity_ref=source["speaker_to_entity_ref"],
        parent_path=parent_path, parent_sha256=parent_sha_before,
        child_path=child_path, child_sha256=sha256_file(child_path))
    validation = validate_expanded_dialogue(
        expanded, original_parent=parent, promotable_child=child)
    if not validation["valid"]:
        raise RuntimeError("expanded_dialogue_validation_failed:" +
                           ",".join(validation["errors"]))
    markdown = expanded_dialogue_markdown(expanded)

    output_root.mkdir(parents=True, exist_ok=False)
    json_path = output_root / "expanded_dialogue_round_1.json"
    markdown_path = output_root / "expanded_dialogue_round_1.md"
    atomic_json(json_path, expanded)
    atomic_text(markdown_path, markdown)
    manifest = complete_dialogue_manifest(
        expanded_dialogue=expanded, validation=validation,
        expanded_json_path=json_path, expanded_json_sha256=sha256_file(json_path),
        expanded_markdown_path=markdown_path,
        expanded_markdown_sha256=sha256_file(markdown_path))
    atomic_json(output_root / "complete_dialogue_manifest.json", manifest)

    parent_sha_after = sha256_file(parent_path)
    canary_after = hash_tree(canary_root)
    integrity = {
        "parent_sha_before": parent_sha_before, "parent_sha_after": parent_sha_after,
        "parent_unchanged": parent_sha_before == parent_sha_after,
        "canary_tree_sha_before": canary_before["tree_sha256"],
        "canary_tree_sha_after": canary_after["tree_sha256"],
        "canary_unchanged": canary_before["tree_sha256"] == canary_after["tree_sha256"],
    }
    if not integrity["parent_unchanged"] or not integrity["canary_unchanged"]:
        raise RuntimeError("complete_dialogue_source_integrity_changed")
    report = {
        "report_version": "round1_complete_dialogue_materialization_v1",
        "status": "completed", "output_root": str(output_root.resolve()),
        "validation": validation, "manifest": manifest, "source_integrity": integrity,
        "api_calls": 0, "qa_calls": 0, "round_2_calls": 0,
        "qa_pipeline_ready": False,
    }
    atomic_json(output_root / "reports/ROUND1_COMPLETE_DIALOGUE_MATERIALIZATION_REPORT.json",
                report)
    atomic_text(output_root / "reports/ROUND1_COMPLETE_DIALOGUE_MATERIALIZATION_REPORT.md",
                _report_markdown(report))
    return report


def _report_markdown(report: dict) -> str:
    validation = report["validation"]
    return f"""# Round 1 Complete Dialogue Materialization

- Status: `{report['status']}`
- Parent turns: {validation['parent_turn_count']}
- Child turns: {validation['child_turn_count']}
- Total turns: {validation['turn_count']}
- Unique refs: {validation['unique_turn_ref_count']}
- Complete dialogue materialized: `{str(validation['complete_dialogue_materialized']).lower()}`
- QA source ready: `{str(validation['qa_source_ready']).lower()}`
- QA pipeline executed: `false`
- API calls: `0`
- Parent unchanged: `{str(report['source_integrity']['parent_unchanged']).lower()}`
- Canary14 unchanged: `{str(report['source_integrity']['canary_unchanged']).lower()}`
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--canary-root", type=Path, default=CANARY_14)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = run(parent_path=args.parent, canary_root=args.canary_root,
                     output_root=args.output_root)
    except Exception as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}:{error}"},
                         ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
