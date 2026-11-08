#!/usr/bin/env python3
"""Compare frozen blind reviews with Gold without changing either artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RUN = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("batch")
args = parser.parse_args()
batch = (RUN / "review_batches" / args.batch).resolve()
assert batch.parent == (RUN / "review_batches").resolve()
items = read(batch / "ITEMS_PRIVATE.json")
mechanical = {row["candidate_id"]: row for row in read(batch / "MECHANICAL_REVIEW.json")["items"]}
adjudication_path = batch / "metadata_adjudication" / "DECISIONS.json"
adjudications = {row["candidate_id"]: row for row in read(adjudication_path)["items"]} if adjudication_path.exists() else {}
rows = []
for item in items:
    rid = item["review_id"]
    review_path = batch / "blind_results" / f"{rid}.json"
    assert review_path.is_file(), f"MISSING_BLIND_REVIEW: {rid}"
    review = read(review_path)
    errors = []
    if review.get("review_id") != rid or review.get("reviewer_mode") != "same_model_blind":
        errors.append("review_identity")
    if review.get("full_source_read") is not True or review.get("source_sha256") != item["source_sha256"] or review.get("source_turn_count") != item["source_turn_count"]:
        errors.append("review_source_binding")
    labels = {option["id"] for option in item["options"]}
    assessments = review.get("per_option", [])
    if {row.get("label") for row in assessments} != labels or len(assessments) != len(labels):
        errors.append("per_option_coverage")
    for row in assessments:
        if row.get("judgment") not in {"supported", "contradicted", "not_supported", "ambiguous"}:
            errors.append("per_option_judgment")
        if not str(row.get("reason", "")).strip():
            errors.append("per_option_reason")
    checks = review.get("quality_checks", {})
    for name in ("source_binding", "question_clarity", "evidence_sufficiency", "option_quality", "naturalness", "shortcut_assessment", "joint_evidence_character"):
        if not checks.get(name):
            errors.append("missing_quality_check:" + name)
    if not str(review.get("answer_reason", "")).strip():
        errors.append("missing_answer_reason")
    kind = item["question_type"]
    if kind == "multi_select":
        answer = review.get("independent_answer", [])
        agreement = len(answer) == len(set(answer)) and set(answer) == set(item["gold_labels"])
    elif kind == "ordering":
        answer = review.get("independent_answer", [])
        agreement = answer == item["gold_labels"]
    else:
        answer = review.get("independent_pairs", [])
        try:
            pairs = [tuple(pair) for pair in answer]
            agreement = len(pairs) == len(set(pairs)) and set(pairs) == {tuple(pair) for pair in item["gold_labels"]}
        except (TypeError, ValueError):
            agreement = False
    holds = list(errors)
    metadata_decision = adjudications.get(item["candidate_id"])
    mechanical_pass = metadata_decision["mechanical_pass"] if metadata_decision else mechanical[item["candidate_id"]]["mechanical_pass"]
    if not mechanical_pass:
        holds.append("mechanical_review_pending")
    if not agreement:
        holds.append("blind_gold_disagreement")
    if review.get("answer_unique") is not True:
        holds.append("answer_not_unique_or_unconfirmed")
    if review.get("hard_issues"):
        holds.append("blind_hard_issue")
    rows.append({"candidate_id": item["candidate_id"], "root_id": item["root_id"], "review_id": rid,
                 "content_pass": not holds, "status": "PASS" if not holds else "HOLD", "hold_reasons": holds,
                 "blind_gold_agreement": agreement, "review_path": str(review_path), "review_sha256": sha(review_path),
                 "source_sha256": item["source_sha256"], "hard_issues": review.get("hard_issues", []),
                 "soft_issues": review.get("soft_issues", []), "metadata_adjudication": metadata_decision,
                 "formal_training_approval_created": False})

with (batch / "QUALITY_FINAL.json").open("x", encoding="utf-8") as stream:
    json.dump({"items": rows}, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"batch": args.batch, "total": len(rows), "pass": sum(row["content_pass"] for row in rows),
                  "holds": [{"candidate_id": row["candidate_id"], "reasons": row["hold_reasons"]} for row in rows if not row["content_pass"]]}))
