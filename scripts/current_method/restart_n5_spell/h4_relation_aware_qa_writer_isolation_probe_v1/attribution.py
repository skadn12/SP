"""Zero-API A/B attribution and relation-binding audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _refs(row: Mapping[str, Any]) -> set[str]:
    refs = row.get("evidence_turn_refs") or row.get("evidence_turn_ids")
    return {str(ref) for ref in refs} if isinstance(refs, list) else set()


def _group_hits(refs: set[str], groups: Mapping[str, Any]) -> list[str]:
    return sorted(key for key, values in groups.items() if refs & {str(value) for value in values})


def summarize_condition(*, qa_run: Path, composition_run: Path, relation_audit: Mapping[str, Any], condition: str) -> dict[str, Any]:
    final = _load(qa_run / "qa" / "final_mcq_batch.json")
    filters = [_load(path) for path in sorted((qa_run / "filter").glob("*.json"))]
    candidates = _jsonl(composition_run / "pair_candidates.jsonl")
    requests = _jsonl(composition_run / "composer_requests.jsonl")
    groups = relation_audit.get("relation_source_turns") or {}
    rows: list[dict[str, Any]] = []
    for item, evaluation in zip(final.get("items") or [], filters):
        refs = _refs(item)
        hits = _group_hits(refs, groups)
        hit_set = set(hits)
        grounded = (evaluation.get("grounding") or {}).get("decision") == "pass_grounding"
        k5 = int(evaluation.get("full_context_correct") or 0) == 5 and int(evaluation.get("full_context_total") or 0) == 5
        rows.append({
            "qa_id": item.get("qa_id"),
            "relation_groups_hit": hits,
            "used_fact_A": "fact_A" in hit_set,
            "used_trigger": "trigger" in hit_set,
            "used_fact_B": "fact_B" in hit_set,
            "used_resolution": "resolution" in hit_set,
            "relation_targeted": len(hit_set & {"fact_A", "trigger", "fact_B"}) >= 2,
            "relation_complete_chain": {"fact_A", "trigger", "fact_B"}.issubset(hit_set),
            "grounding_pass": grounded,
            "k5_stable": k5,
            "decision": evaluation.get("decision"),
        })
    targeted = [row for row in rows if row["relation_targeted"]]
    complete_chain = [row for row in rows if row["relation_complete_chain"]]
    targeted_stable = [row for row in targeted if row["grounding_pass"] and row["k5_stable"]]
    complete_chain_stable = [row for row in complete_chain if row["grounding_pass"] and row["k5_stable"]]
    targeted_easy = [row for row in targeted_stable if row["decision"] == "reject_easy"]
    targeted_non_easy = [row for row in targeted_stable if row["decision"] != "reject_easy"]
    return {
        "condition": condition,
        "atomic_generated": len(rows),
        "relation_targeted_questions": len(targeted),
        "relation_complete_chain_questions": len(complete_chain),
        "relation_targeted_grounding_k5": len(targeted_stable),
        "relation_complete_chain_grounding_k5": len(complete_chain_stable),
        "relation_targeted_easy_rejected": len(targeted_easy),
        "relation_targeted_non_easy": len(targeted_non_easy),
        "grounding_pass": sum(row["grounding_pass"] for row in rows),
        "k5_stable": sum(row["grounding_pass"] and row["k5_stable"] for row in rows),
        "easy_rejected": sum(row["decision"] == "reject_easy" for row in rows),
        "pair_candidates": len(candidates),
        "composer_ready": len(requests),
        "admitted": 0,
        # This is evidence utilization, not proof of a difficult QA. Difficulty
        # requires a relation-targeted item to survive the easy-rejection gate.
        "relation_used_by_QA": bool(targeted),
        "relation_utilization_rate": (len(targeted_stable) / len(rows)) if rows else 0.0,
        "relation_evidence_coverage_rate": (len(targeted) / len(rows)) if rows else 0.0,
        "relation_used_by_difficult_QA": bool(targeted_non_easy),
        "difficulty_qualified_relation_utilization_rate": (len(targeted_non_easy) / len(rows)) if rows else 0.0,
        "fact_A_binding_questions": sum(row["used_fact_A"] for row in rows),
        "trigger_binding_questions": sum(row["used_trigger"] for row in rows),
        "fact_B_binding_questions": sum(row["used_fact_B"] for row in rows),
        "rows": rows,
    }


def compare(*, baseline: Mapping[str, Any], relation_aware: Mapping[str, Any]) -> dict[str, Any]:
    b_ready = int(baseline.get("composer_ready") or 0)
    r_ready = int(relation_aware.get("composer_ready") or 0)
    b_necessity = False
    r_necessity = False
    if b_ready == 0 and r_ready > 0 and r_necessity:
        conclusion = "WRITER_BOTTLENECK_CONFIRMED"
    elif b_ready == 0 and r_ready == 0:
        conclusion = "RELATION_AWARE_WRITER_NO_COMPOSER_READY"
    elif r_ready > 0 and not r_necessity:
        conclusion = "WRITER_RELATION_FOUND_BUT_NECESSITY_UNPROVEN"
    else:
        conclusion = "INCONCLUSIVE"
    return {
        "baseline": dict(baseline),
        "relation_aware": dict(relation_aware),
        "necessity": {
            "baseline": {"full_context_accuracy": None, "minus_A_accuracy": None, "minus_B_accuracy": None, "necessity_pass": b_necessity, "status": "not_run_no_composer_response"},
            "relation_aware": {"full_context_accuracy": None, "minus_A_accuracy": None, "minus_B_accuracy": None, "necessity_pass": r_necessity, "status": "not_run_no_composer_response"},
        },
        "conclusion": conclusion,
    }
