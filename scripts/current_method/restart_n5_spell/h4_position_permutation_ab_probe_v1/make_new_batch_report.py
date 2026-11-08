"""Write a compact, correctly scoped report for an isolated B run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_summary(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_count": stage.get("row_count"),
        "scored_count": stage.get("scored_count"),
        "correct_count": stage.get("correct_count"),
        "accuracy": stage.get("accuracy"),
        "technical_failure_count": stage.get("technical_failure_count"),
        "by_context": stage.get("by_context"),
        "by_question_type": stage.get("by_question_type"),
    }


def build(run_root: Path, old_run: Path | None = None) -> dict[str, Any]:
    run_root = run_root.resolve()
    preflight = _load(run_root / "preflight_report.json")
    solver = _load(run_root / "stability_solver_report.json")
    audit = _load(run_root / "output_audit.json")
    canonical = _load(run_root / "corrected_analysis.json")
    report: dict[str, Any] = {
        "report_version": "h4_position_ab_new_generation_batch_report_v1",
        "run_root": str(run_root),
        "input": {
            "source_run": preflight.get("source_run"),
            "provenance": preflight.get("provenance") or preflight.get("parent_preflight"),
            "sample_count": preflight.get("sample_count"),
            "root_count": preflight.get("root_count"),
            "question_type": preflight.get("question_type"),
            "generation_source": preflight.get("generation_source"),
            "permutation_count": preflight.get("permutation_count"),
            "cyclic_permutation_count": preflight.get("cyclic_permutation_count"),
        },
        "new_results": {
            "expected_requests": solver.get("expected_requests"),
            "request_accounting": solver.get("request_accounting"),
            "content_responder_prompt_version": solver.get("content_responder_prompt_version"),
            "stages": {name: _stage_summary(value) for name, value in (solver.get("stages") or {}).items()},
        },
        "output_audit": {
            "row_count": audit.get("row_count"),
            "model_output_clean": audit.get("model_output_clean"),
            "model_output_issues": audit.get("model_output_issues"),
            "metadata_or_scoring_issues": audit.get("metadata_or_scoring_issues"),
            "parser_status": audit.get("parser_status"),
        },
        "canonical_pairing": canonical.get("paired_B0_B1_B2"),
        "scope_note": (
            f"Standalone {preflight.get('sample_count')}-candidate sample from the 122 expansion's "
            "generation-only QA pool; no semantic admission or difficulty filter was applied, "
            "and no other batch is included."
        ),
    }
    if old_run is not None:
        old_run = old_run.resolve()
        old_report = _load(old_run / "b_comparison_report.json")
        report["repaired_original_30_comparison"] = {
            "source_run": str(old_run),
            "scope": old_report.get("interpretation_scope"),
            "condition_summaries": old_report.get("condition_summaries"),
        }
    atomic_json(run_root / "new_batch_report.json", report)
    atomic_text(run_root / "new_batch_report.md", f"# H4 B experiment: standalone {preflight.get('sample_count')}-candidate batch\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--old-run", type=Path)
    args = parser.parse_args(argv)
    report = build(args.run_root, args.old_run)
    print(json.dumps({
        "status": "written",
        "report": str((args.run_root.resolve() / "new_batch_report.json")),
        "new_accuracy": {name: value.get("accuracy") for name, value in report["new_results"]["stages"].items()},
        "audit_clean": report["output_audit"]["model_output_clean"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
