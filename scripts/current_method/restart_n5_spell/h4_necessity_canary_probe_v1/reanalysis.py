"""Zero-API reanalysis for a completed necessity canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, write_json

from .probe import _audit


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(*, source_run: Path, output: Path) -> dict:
    prepare_empty(output)
    child = _load(source_run / "child_dialogue.json")
    contract = _load(source_run / "relation_contract.json")
    audit = _audit(child, contract)
    conclusion = "NECESSITY_CANARY_SOURCE_READY" if audit["necessity_source_ready"] else "NECESSITY_CANARY_LEAKAGE_OR_REALIZATION_FAIL"
    metrics = {
        "probe_version": "h4_necessity_canary_reanalysis_v1",
        "analysis_execution_api_calls": 0,
        "source_run": str(source_run.resolve()),
        "relation_present": audit["relation_present"],
        "temporal_leakage_free": audit["temporal_leakage_free"],
        "necessity_source_ready": audit["necessity_source_ready"],
        "resolution_leakage_terms": audit["resolution_leakage_terms"],
        "relation_source_turns": audit["relation_source_turns"],
        "required_role_refs_disjoint": audit["required_role_refs_disjoint"],
        "conclusion": conclusion,
    }
    write_json(output / "relation_audit.json", audit)
    write_json(output / "metrics.json", metrics)
    write_json(output / "run_freeze_manifest.json", {
        "freeze_version": "h4_necessity_canary_reanalysis_v1",
        "source_run": str(source_run.resolve()),
        "source_run_read_only": True,
        "analysis_execution_api_calls": 0,
        "planner_calls": 0,
        "realizer_calls": 0,
        "qa_writer_calls": 0,
        "composer_calls": 0,
        "solver_calls": 0,
        "memorax_calls": 0,
    })
    write_json(output / "request_ledger.json", {
        "probe_version": "h4_necessity_canary_reanalysis_v1",
        "execution_mode": "zero_api_read_only",
        "requests": [],
    })
    write_json(output / "POST_RUN_INTEGRITY.json", {
        "probe_version": "h4_necessity_canary_reanalysis_v1",
        "status": "complete",
        "analysis_execution_api_calls": 0,
        "source_run_mutated": False,
        "mainline_files_mutated": False,
        "necessity_source_ready": audit["necessity_source_ready"],
    })
    (output / "report.md").write_text("\n".join([
        "# H4 Necessity Canary Reanalysis v1", "", f"Verdict: **{conclusion}**", "",
        f"- Source run: `{source_run}`",
        f"- Relation present: `{audit['relation_present']}`",
        f"- Temporal leakage free: `{audit['temporal_leakage_free']}`",
        f"- Resolution leakage terms: `{audit['resolution_leakage_terms']}`",
        f"- Necessity source ready: `{audit['necessity_source_ready']}`", "",
        "This is zero-API reanalysis with word-boundary matching; the source canary was not regenerated or modified.", "",
    ]), encoding="utf-8")
    (output / "SUMMARY_REPORT.md").write_text("\n".join([
        "# H4 Necessity Canary Reanalysis Summary", "",
        "The completed three-scene canary is re-audited with whole-word forbidden-term matching. No Realizer or QA request was made during this reanalysis.", "",
        f"Verdict: **{conclusion}**", "",
    ]), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(source_run=args.source_run.resolve(), output=args.output.resolve()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
