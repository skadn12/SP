"""Zero-API authoritative reanalysis for relation-contract probe artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, write_json

from .audit import audit_relation


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PROBE_VERSION = "h4_relation_contract_refinement_reanalysis_v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run(*, probe_run: Path, run_root: Path) -> dict[str, Any]:
    prepare_empty(run_root)
    child = _load(probe_run / "child_dialogue.json")
    contract = _load(probe_run / "relation_contract.json")
    audit = audit_relation(child, contract)
    metrics = {
        "probe_version": PROBE_VERSION,
        "source_probe_run": str(probe_run.resolve()),
        "source_probe_metrics": str((probe_run / "metrics.json").resolve()),
        "analysis_execution_api_calls": 0,
        "relation_type": contract.get("relation_type"),
        "relation_present": audit["relation_present"],
        "relation_audit_pass": audit["relation_audit_pass"],
        "trigger_supported": audit["trigger_supported"],
        "state_chain_valid": audit["state_chain_valid"],
        "temporal_leakage_free": audit["temporal_leakage_free"],
        "early_relation_leakage": audit["early_relation_leakage"],
        "relation_source_turns": audit["relation_source_turns"],
        "conclusion": (
            "RELATION_CONTRACT_REANALYSIS_PASS"
            if audit["relation_audit_pass"] else
            "RELATION_PRESENT_BUT_TEMPORAL_AUDIT_FAIL"
            if audit["relation_present"] else
            "RELATION_NOT_PRESENT"
        ),
    }
    write_json(run_root / "relation_audit.json", audit)
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {
        "probe_version": PROBE_VERSION,
        "isolated": True,
        "api_calls": 0,
        "source_probe_read_only": str(probe_run.resolve()),
        "audit_code": "h4_relation_contract_audit_v1",
    })
    report = [
        "# H4 Relation-Contract Refinement Reanalysis v1", "",
        f"Verdict: **{metrics['conclusion']}**", "",
        f"- Source probe: `{probe_run}`",
        f"- Relation type: `{metrics['relation_type']}`",
        f"- relation_present: `{metrics['relation_present']}`",
        f"- relation_audit_pass: `{metrics['relation_audit_pass']}`",
        f"- temporal_leakage_free: `{metrics['temporal_leakage_free']}`",
        f"- early_relation_leakage: `{metrics['early_relation_leakage']}`", "",
        "This report is zero-API and does not modify the source probe.", "",
    ]
    (run_root / "report.md").write_text("\n".join(report), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="zero-API H4 relation-contract reanalysis")
    parser.add_argument("--probe-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"{PROBE_VERSION}_{stamp}").resolve()
    result = run(probe_run=args.probe_run.resolve(), run_root=run_root)
    print(json.dumps({"run_root": str(run_root), "metrics": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
