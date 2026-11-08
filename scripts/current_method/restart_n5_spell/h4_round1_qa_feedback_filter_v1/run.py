"""Offline exact/near-duplicate and feedback-alignment filter for E1Q1."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..core import atomic_json, atomic_text, canonical_json, sha256_file
from ..h4_round1_qa_feedback_compiler_v2.compiler import duplicate_relation
from . import DEFAULT_RUN, E1Q0_RUN, GENERATION_RUN


def rows(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def write(path: Path, value):
    if path.suffix == ".jsonl":
        text = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in value)
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    atomic_text(path, text)
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "  " + path.name + "\n")


def main(run: Path = DEFAULT_RUN) -> dict:
    run = run.resolve()
    if run.exists() and any(run.iterdir()):
        raise RuntimeError(f"output_exists_or_nonempty:{run}")
    run.mkdir(parents=True, exist_ok=False)
    e1q1 = []
    alignment_by_id = {}
    for root_dir in sorted(GENERATION_RUN.glob("cc-train-episode-*")):
        e1q1.extend(rows(root_dir / "candidates_valid_scoped.json"))
        audit_path = root_dir / "feedback_alignment_audit.jsonl"
        if audit_path.is_file():
            alignment_by_id.update({str(x.get("candidate_id")): x for x in rows(audit_path)})
    e1q0 = rows(E1Q0_RUN / "all_candidates_valid.jsonl")
    by_root = {}
    for row in e1q0:
        by_root.setdefault(str(row.get("root_id")), []).append(row)
    # First route against the existing baseline, then enforce uniqueness within E1Q1.
    audit = []
    exact_ids = set(); near_ids = set()
    for row in e1q1:
        root = str(row.get("root_id") or "")
        alignment = alignment_by_id.get(str(row.get("candidate_id")), {})
        baseline_relations = [duplicate_relation(row, old) for old in by_root.get(root, [])]
        exact = any(x["exact_duplicate"] for x in baseline_relations)
        near = any(x["near_duplicate"] for x in baseline_relations)
        audit.append({"candidate_id": row.get("candidate_id"), "root_id": root,
                      "baseline_exact_duplicate": exact, "baseline_near_duplicate": near,
                      "baseline_relation_count": len(baseline_relations),
                      "feedback_alignment_status": alignment.get("feedback_alignment_status"),
                      "feedback_analysis_route": alignment.get("feedback_analysis_route")})
        if exact: exact_ids.add(str(row.get("candidate_id")))
        elif near: near_ids.add(str(row.get("candidate_id")))
    # E1Q1 self-duplicates are compared on normalized question/alternatives/Gold.
    normalized_seen = {}
    self_exact = set()
    for row in e1q1:
        key = canonical_json({"question": row.get("question"), "options": row.get("options") or row.get("items"), "gold": row.get("gold") or []})
        previous = normalized_seen.get(key)
        if previous:
            self_exact.add(str(row.get("candidate_id"))); self_exact.add(previous)
        else:
            normalized_seen[key] = str(row.get("candidate_id"))
    exact_ids |= self_exact
    exact_by_id = {str(x.get("candidate_id")): x for x in audit}
    for x in audit:
        cid = str(x.get("candidate_id")); x["self_exact_duplicate"] = cid in self_exact
        x["exact_duplicate"] = cid in exact_ids
        x["near_duplicate"] = cid in near_ids and cid not in exact_ids
        align = str(x.get("feedback_analysis_route") or "")
        if x["exact_duplicate"]: x["final_route"] = "EXACT_DUPLICATE_REJECT"
        elif x["near_duplicate"]: x["final_route"] = "NEAR_DUPLICATE_SECONDARY"
        elif align.startswith("PRIMARY"): x["final_route"] = "PRIMARY"
        else: x["final_route"] = "SECONDARY_OFF_TARGET"
    audit_by_id = {str(x["candidate_id"]): x for x in audit}
    primary = [x for x in e1q1 if audit_by_id[str(x.get("candidate_id"))]["final_route"] == "PRIMARY"]
    secondary = [x for x in e1q1 if audit_by_id[str(x.get("candidate_id"))]["final_route"] != "PRIMARY"]
    write(run / "e1q1_duplicate_feedback_audit.jsonl", audit)
    write(run / "e1q1_primary_solver_pool.jsonl", primary)
    write(run / "e1q1_secondary_contamination_pool.jsonl", secondary)
    summary = {"status": "PASS", "candidate_count": len(e1q1), "primary_solver_candidate_count": len(primary), "secondary_candidate_count": len(secondary), "exact_duplicate_count": sum(x["exact_duplicate"] for x in audit), "near_duplicate_count": sum(x["near_duplicate"] for x in audit), "baseline_e1q0_candidate_count": len(e1q0), "route_counts": dict(Counter(x["final_route"] for x in audit)), "feedback_alignment_counts": dict(Counter(str(x.get("feedback_alignment_status") or "") for x in audit)), "source_generation_run": str(GENERATION_RUN), "baseline_run": str(E1Q0_RUN), "solver_calls": 0, "memorax_calls": 0, "semantic_auditor_calls": 0}
    write(run / "final_qa_feedback_filter_report.json", summary)
    write(run / "final_qa_feedback_filter_report.md", "# E1Q1 QA Feedback Filter v1\n\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
