"""Run a frozen, zero-API Realizer-adherence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, sha256_file
from .realizer_adherence import (
    build_adherence_auditor_input,
    build_adherence_auditor_prompt,
    load_json,
    validate_and_materialize_adherence_audit,
)


DEFAULT_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_34406_memory_event_continuation_"
    "contract_v2_live_1")
DEFAULT_JUDGMENT = Path(__file__).resolve().parent / "fixtures" / (
    "episode_34406_realizer_adherence_semantic_judgment.json")
DEFAULT_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_34406_realizer_adherence_offline_audit_1")


def _directory_manifest(root: Path) -> str:
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        rows.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path)})
    body = json.dumps(
        rows, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _report(*, source_run: Path, audit: Mapping[str, Any],
            validation: Mapping[str, Any]) -> str:
    lines = [
        "# Realizer Adherence Offline Report",
        "",
        f"- Source run: `{source_run}`",
        "- API calls: `0`",
        "- Semantic judgment source: `frozen human-audited fixture`",
        f"- Contract valid: `{validation['valid']}`",
        "",
    ]
    for scene in audit["scenes"]:
        lines.extend([
            f"## {scene['scene_ref']}",
            "",
            f"- Selected development: {scene['selected_new_development']}",
            f"- Selected durable outcome: {scene['selected_durable_outcome']}",
            f"- Participant adherence: `{scene['participant_adherence']}`",
            "- Participant evidence: `" + "`, `".join(
                scene["participant_evidence_turn_refs"]) + "`",
            f"- Development realization: `{scene['development_status']}`",
            "- Development evidence: `" + "`, `".join(
                scene["development_evidence_turn_refs"]) + "`",
            f"- Durable outcome realization: `{scene['durable_outcome_status']}`",
            "- Durable outcome evidence: `" + "`, `".join(
                scene["durable_outcome_evidence_turn_refs"]) + "`",
            f"- Final verdict: **{scene['scene_verdict']}**",
            f"- Reason: {scene['reason']}",
            "",
        ])
    summary = audit["summary"]
    lines.extend([
        "## Summary",
        "",
        f"- Scene count: `{summary['scene_count']}`",
        f"- Scene pass/fail: `{summary['scene_pass_count']}` / "
        f"`{summary['scene_fail_count']}`",
        f"- Participant drift: `{summary['participant_drift_count']}`",
        "- Development partial/missing: "
        f"`{summary['development_partial_or_missing_count']}`",
        "- Durable outcome partial/missing: "
        f"`{summary['durable_outcome_partial_or_missing_count']}`",
        "",
        "The semantic statuses are frozen human judgments for this zero-API "
        "probe. The reusable program only validates the ordered judgment "
        "contract, maps scene-local positions to canonical Child turn refs, "
        "and computes deterministic verdicts.",
        "",
    ])
    return "\n".join(lines)


def run(*, source_run: Path, judgment_path: Path,
        output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"offline_audit_output_exists:{output_root}")
    plan_path = source_run / "planner" / "materialized_plan.json"
    child_path = source_run / "child_dialogue.json"
    for path in (plan_path, child_path, judgment_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    before = _directory_manifest(source_run)
    plan = load_json(plan_path)
    child = load_json(child_path)
    judgment = load_json(judgment_path)
    auditor_input = build_adherence_auditor_input(plan=plan, child=child)
    prompt = build_adherence_auditor_prompt(auditor_input=auditor_input)
    result = validate_and_materialize_adherence_audit(
        judgment, plan=plan, child=child)
    after = _directory_manifest(source_run)
    validation = {
        "validation_version": "realizer_adherence_validation_v1",
        "valid": bool(result["valid"] and before == after),
        "errors": list(result["errors"]),
        "source_run_manifest_before": before,
        "source_run_manifest_after": after,
        "source_run_unchanged": before == after,
        "semantic_judgment_source": "frozen_human_audited_fixture",
        "api_calls": 0,
        "memorax_calls": 0,
        "solver_calls": 0,
    }
    if before != after:
        validation["errors"].append("source_run_changed")
    if not validation["valid"]:
        raise ValueError("realizer_adherence_validation_failed:" + ",".join(
            validation["errors"]))
    output_root.mkdir(parents=True)
    atomic_json(output_root / "realizer_adherence" / "auditor_input.json",
                auditor_input)
    atomic_text(output_root / "realizer_adherence" / "auditor_prompt.txt",
                str(prompt["prompt"]))
    atomic_json(output_root / "realizer_adherence" / "semantic_judgment.json",
                judgment)
    atomic_json(output_root / "realizer_adherence" / "audit.json",
                result["audit"])
    atomic_json(output_root / "realizer_adherence" / "validation.json",
                validation)
    atomic_text(output_root / "reports" / "REALIZER_ADHERENCE_OFFLINE_REPORT.md",
                _report(source_run=source_run, audit=result["audit"],
                        validation=validation))
    return {"audit": result["audit"], "validation": validation,
            "output_root": str(output_root)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--semantic-judgment", type=Path,
                        default=DEFAULT_JUDGMENT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    result = run(source_run=args.source_run,
                 judgment_path=args.semantic_judgment,
                 output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
