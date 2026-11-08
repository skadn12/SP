"""Freeze a deterministic, policy-independent Gold audit sample after selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


VERSION = "h4_crossroot_multiselect_v28_raw_gold_sample_v1"
ROOTS = {
    "cc-train-episode-231899",
    "cc-train-episode-194259",
    "cc-train-episode-210046",
    "cc-train-episode-96544",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def rank(root, qid):
    return digest((VERSION + "\n" + root + "\n" + qid).encode("utf-8"))


def build_manifest(results, selected, result_sha, selected_sha):
    counts = results.get("counts", {})
    provenance = results.get("generator_provenance", {})
    if (results.get("report_version") != "h4_crossroot_multiselect_solver_selection_v2"
            or results.get("evaluation_data_status") != "VALID"
            or provenance.get("semantic_review_performed") is not False
            or provenance.get("codex_question_edits") is not False
            or provenance.get("mechanical_gate_version") != "raw_source_and_gold_v1"):
        raise ValueError("COMPLETE_RAW_SOLVER_REPORT_REQUIRED")
    questions = results.get("questions")
    if not isinstance(questions, dict) or len(questions) != 16:
        raise ValueError("V28_SIXTEEN_QUESTIONS_REQUIRED")
    by_root = {root: [] for root in ROOTS}
    selected_questions = {}
    for qid, row in questions.items():
        question = row.get("canonical_question", {})
        root = row.get("root_id")
        decision = row.get("selected_by_policy")
        if (question.get("candidate_id") != qid or question.get("root_id") != root
                or root not in ROOTS or type(decision) is not bool):
            raise ValueError("QUESTION_ID_ROOT_OR_DECISION_INVALID:" + qid)
        by_root[root].append(qid)
        if decision:
            selected_questions[qid] = question
    if any(len(qids) != 4 for qids in by_root.values()):
        raise ValueError("V28_FOUR_PER_ROOT_REQUIRED")
    listed = selected.get("questions")
    if not isinstance(listed, list) or selected.get("count") != len(listed):
        raise ValueError("SELECTED_LIST_INVALID")
    listed_by_id = {q.get("candidate_id"): q for q in listed if isinstance(q, dict)}
    if (len(listed_by_id) != len(listed) or listed_by_id != selected_questions
            or counts.get("canonical_questions") != 16
            or counts.get("selected_questions") != len(selected_questions)
            or counts.get("unresolved_questions") != 0):
        raise ValueError("SELECTED_LIST_OR_COUNTS_MISMATCH")

    rows = [{"candidate_id": qid, "root_id": q["root_id"],
             "selection_hash": rank(q["root_id"], qid)}
            for qid, q in selected_questions.items()]
    ordered = sorted(rows, key=lambda row: (row["selection_hash"], row["candidate_id"]))
    target = len(ordered) if len(ordered) <= 4 else max(4, (len(ordered) + 2) // 3)
    sample_ids = set()
    for root in sorted(ROOTS):
        root_rows = [row for row in ordered if row["root_id"] == root]
        if root_rows:
            sample_ids.add(root_rows[0]["candidate_id"])
    for row in ordered:
        if len(sample_ids) >= target:
            break
        sample_ids.add(row["candidate_id"])
    sampled = [row for row in ordered if row["candidate_id"] in sample_ids]
    return {
        "version": VERSION,
        "status": "AUDIT_NOT_STARTED",
        "input_sha256": {"RESULTS.json": result_sha, "SELECTED_QUESTIONS.json": selected_sha},
        "sampling_rule": "All if N<=4; otherwise one lowest-hash question per represented root, then lowest remaining hashes to max(4, ceil(N/3)).",
        "hash_input": "UTF-8: version + newline + root_id + newline + candidate_id",
        "selection_uses_semantics_or_solver_scores": False,
        "selection_policy_result_modified": False,
        "candidate_count": 16,
        "selected_count": len(ordered),
        "target_sample_count": target,
        "sample_count": len(sampled),
        "selected_root_counts": {root: len([row for row in rows if row["root_id"] == root])
                                 for root in sorted(ROOTS)},
        "selected_ids_and_hashes": ordered,
        "sampled_ids_and_hashes": sampled,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--selected", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result_data, selected_data = args.results.read_bytes(), args.selected.read_bytes()
    manifest = build_manifest(json.loads(result_data), json.loads(selected_data),
                              digest(result_data), digest(selected_data))
    data = encoded(manifest)
    path = args.output / ("SAMPLE_" + digest(data) + ".json")
    args.output.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("CONTENT_ADDRESSED_SAMPLE_CHANGED")
    else:
        with path.open("xb") as handle:
            handle.write(data)
    print(json.dumps({"sample_manifest": str(path), "selected": manifest["selected_count"],
                      "sampled": manifest["sample_count"]}))


if __name__ == "__main__":
    main()
