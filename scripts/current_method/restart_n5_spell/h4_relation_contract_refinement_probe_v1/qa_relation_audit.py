"""Zero-API attribution of relation data to unchanged QA artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, write_json


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PROBE_VERSION = "h4_relation_contract_qa_relation_audit_v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _refs(row: Mapping[str, Any]) -> set[str]:
    values = row.get("evidence_turn_refs")
    if values is None:
        values = row.get("evidence_turn_ids")
    return {str(value) for value in values} if isinstance(values, list) else set()


def _groups(relation_audit: Mapping[str, Any]) -> dict[str, set[str]]:
    payload = relation_audit.get("relation_source_turns") or {}
    return {key: {str(ref) for ref in values} for key, values in payload.items() if isinstance(values, list)}


def _hit_groups(refs: set[str], groups: Mapping[str, set[str]]) -> list[str]:
    return sorted(key for key, values in groups.items() if refs & values)


def run(*, probe_run: Path, qa_run: Path, composition_run: Path, run_root: Path, relation_audit_path: Path | None = None) -> dict[str, Any]:
    prepare_empty(run_root)
    relation_audit = _load(relation_audit_path or (probe_run / "relation_audit.json"))
    final_batch = _load(qa_run / "qa" / "final_mcq_batch.json")
    filter_rows = [_load(path) for path in sorted((qa_run / "filter").glob("*.json"))]
    candidates = _jsonl(composition_run / "pair_candidates.jsonl")
    composer_requests = _jsonl(composition_run / "composer_requests.jsonl")
    groups = _groups(relation_audit)
    atomic_rows: list[dict[str, Any]] = []
    for item, evaluation in zip(final_batch.get("items") or [], filter_rows):
        refs = _refs(item)
        hits = _hit_groups(refs, groups)
        grounded = (evaluation.get("grounding") or {}).get("decision") == "pass_grounding"
        k5 = int(evaluation.get("full_context_correct") or 0) == 5 and int(evaluation.get("full_context_total") or 0) == 5
        stable = bool(grounded and k5)
        atomic_rows.append({
            "qa_id": str(item.get("qa_id") or ""),
            "evidence_turn_refs": sorted(refs),
            "relation_evidence_groups_hit": hits,
            "relation_touched": bool(hits),
            "relation_joint_atomic": len(set(hits) & {"fact_A", "trigger", "fact_B"}) >= 2,
            "relation_joint_atomic_k5_stable": bool(stable and len(set(hits) & {"fact_A", "trigger", "fact_B"}) >= 2),
            "relation_joint_atomic_not_easy": bool(stable and evaluation.get("decision") not in {"reject_easy", "admit_frontier_k5"} and len(set(hits) & {"fact_A", "trigger", "fact_B"}) >= 2),
            "grounding_pass": grounded,
            "full_context_k5_stable": k5,
            "easy_rejected": evaluation.get("decision") == "reject_easy",
        })
    pair_rows: list[dict[str, Any]] = []
    atomic_by_id = {row["qa_id"]: row for row in atomic_rows}
    for pair in candidates:
        qa_ids = [str(value) for value in pair.get("qa_ids") or []]
        left, right = (atomic_by_id.get(qa_ids[0], {}), atomic_by_id.get(qa_ids[1], {})) if len(qa_ids) == 2 else ({}, {})
        union = sorted(set(left.get("relation_evidence_groups_hit") or []) | set(right.get("relation_evidence_groups_hit") or []))
        pair_rows.append({
            "pair_id": pair.get("pair_id"),
            "qa_ids": qa_ids,
            "candidate_type": pair.get("candidate_type"),
            "relation_evidence_groups_union": union,
            "relation_joint_pair": len(set(union) & {"fact_A", "trigger", "fact_B"}) >= 3,
            "composer_ready": bool(pair.get("composer_ready")),
        })
    relation_joint_pairs = [row for row in pair_rows if row["relation_joint_pair"]]
    metrics = {
        "probe_version": PROBE_VERSION,
        "probe_run": str(probe_run.resolve()),
        "qa_run": str(qa_run.resolve()),
        "composition_run": str(composition_run.resolve()),
        "analysis_execution_api_calls": 0,
        "relation_present": bool(relation_audit.get("relation_present")),
        "relation_audit_pass": bool(relation_audit.get("relation_audit_pass")),
        "relation_type": relation_audit.get("relation_type"),
        "atomic_generated": len(atomic_rows),
        "atomic_grounding_pass": sum(row["grounding_pass"] for row in atomic_rows),
        "atomic_full_context_k5_stable": sum(row["full_context_k5_stable"] for row in atomic_rows if row["grounding_pass"]),
        "atomic_easy_rejected": sum(row["easy_rejected"] for row in atomic_rows),
        "relation_touched_atomic": sum(row["relation_touched"] for row in atomic_rows),
        "relation_joint_atomic": sum(row["relation_joint_atomic"] for row in atomic_rows),
        "relation_joint_atomic_k5_stable": sum(row["relation_joint_atomic_k5_stable"] for row in atomic_rows),
        "relation_joint_atomic_not_easy": sum(row["relation_joint_atomic_not_easy"] for row in atomic_rows),
        "pair_candidate": len(candidates),
        "relation_joint_pair_candidate": len(relation_joint_pairs),
        "composer_ready": len(composer_requests),
        "admitted": 0,
        "ablation": {
            "status": "not_run_no_composer_ready_pair",
            "full_context_accuracy": None,
            "minus_fact_A_accuracy": None,
            "minus_fact_B_accuracy": None,
            "minus_trigger_accuracy": None,
            "necessity_proven": False,
        },
        "qa_utilized_relation": bool(relation_joint_pairs and composer_requests),
        "qa_utilization_conclusion": (
            "QA_RELATION_UTILIZED" if relation_joint_pairs else
            "DATA_RELATION_PRESENT_QA_UTILIZATION_NOT_OBSERVED"),
        "atomic_rows": atomic_rows,
        "pair_rows": pair_rows,
    }
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {
        "probe_version": PROBE_VERSION,
        "isolated": True,
        "api_calls": 0,
        "source_probe_read_only": str(probe_run.resolve()),
        "source_qa_read_only": str(qa_run.resolve()),
        "source_composition_read_only": str(composition_run.resolve()),
    })
    lines = [
        "# H4 Relation-Contract QA Relation Audit v1", "",
        f"- Relation data present: `{metrics['relation_present']}`",
        f"- Relation audit pass: `{metrics['relation_audit_pass']}`",
        f"- Relation type: `{metrics['relation_type']}`", "",
        "## Data vs QA", "",
        f"- Atomic generated: `{metrics['atomic_generated']}`",
        f"- Grounding pass: `{metrics['atomic_grounding_pass']}`",
        f"- Grounded + K=5 stable: `{metrics['atomic_full_context_k5_stable']}`",
        f"- Easy rejected: `{metrics['atomic_easy_rejected']}`",
        f"- Atomic touching relation evidence: `{metrics['relation_touched_atomic']}`",
        f"- Atomic covering multiple relation sides (any): `{metrics['relation_joint_atomic']}`",
        f"- Atomic covering multiple relation sides (grounded + K=5): `{metrics['relation_joint_atomic_k5_stable']}`",
        f"- Atomic covering multiple relation sides (grounded + K=5 + not easy): `{metrics['relation_joint_atomic_not_easy']}`",
        f"- Pair candidates: `{metrics['pair_candidate']}`",
        f"- Relation-covered pair candidates: `{metrics['relation_joint_pair_candidate']}`",
        f"- Composer-ready: `{metrics['composer_ready']}`",
        f"- Admitted: `{metrics['admitted']}`", "",
        "## Necessity Ablation", "",
        "`full_context_accuracy`, `minus_fact_A_accuracy`, `minus_fact_B_accuracy`, "
        "and `minus_trigger_accuracy` are not run because no Composer-ready pair "
        "was produced. This is an explicit QA-utilization failure, not a claim "
        "that the data relation is absent.", "",
        f"Conclusion: **{metrics['qa_utilization_conclusion']}**", "",
        "This is a zero-API attribution report. It does not alter the Writer, "
        "Solver, admission, or any source run.", "",
    ]
    (run_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="zero-API relation-to-QA attribution")
    parser.add_argument("--probe-run", type=Path, required=True)
    parser.add_argument("--relation-audit", type=Path)
    parser.add_argument("--qa-run", type=Path, required=True)
    parser.add_argument("--composition-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"{PROBE_VERSION}_{stamp}").resolve()
    result = run(probe_run=args.probe_run.resolve(), qa_run=args.qa_run.resolve(), composition_run=args.composition_run.resolve(), run_root=run_root, relation_audit_path=args.relation_audit.resolve() if args.relation_audit else None)
    print(json.dumps({"run_root": str(run_root), "metrics": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
