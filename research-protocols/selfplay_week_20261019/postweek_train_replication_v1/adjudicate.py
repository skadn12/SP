"""Merge source-semantic reviews with fixed mechanical scores after raw freeze."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, obj):
    with Path(path).open("x") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def compute():
    mechanical = read(HERE / "MECHANICAL_SCORES.json")
    requests = {r["request_id"]: r for r in read(HERE / "REQUESTS.json")["requests"]}
    reviews, review_files = {}, []
    for path in sorted((HERE / "reviews").glob("citation_*.json")):
        review_files.append(binding(path))
        for review in read(path)["requests"]:
            rid = review["request_id"]
            assert rid not in reviews, "DUPLICATE_CITATION_REVIEW"
            reviews[rid] = review
    assert len(requests) == len(reviews) == 12 and set(requests) == set(reviews)
    categories = Counter()
    result_rows = []
    for row in mechanical["requests"]:
        req = requests[row["request_id"]]
        review = reviews[row["request_id"]]
        assert all(review[key] == req[key] for key in ("candidate_id", "root_id", "variant_id", "source_sha256"))
        assert set(review["items"]) == set(req["labels"])
        answer = json.loads(read(row["response_binding"]["path"])["raw_output"])
        for label, detail in review["items"].items():
            assert detail["selected_citation_refs"] == answer["item_evidence"][label]["turn_refs"], "REVIEW_NOT_ACTUAL_SELECTED_REFS"
            assert isinstance(detail["judgment_correct"], bool)
            assert isinstance(detail["citation_sufficient_for_first_full_report"], bool)
            if not detail["citation_sufficient_for_first_full_report"]:
                categories[detail["category"]] += 1
        judgments = sum(x["judgment_correct"] for x in review["items"].values())
        sufficient = sum(x["citation_sufficient_for_first_full_report"] for x in review["items"].values())
        assert review["all_six_judgments_correct"] == (judgments == 6)
        assert review["all_six_citation_sets_sufficient"] == (sufficient == 6)
        qualifies = bool(row["mechanically_valid"] and judgments == 6 and sufficient == 6 and not row["exact_correct"])
        result_rows.append({
            "request_id": row["request_id"], "candidate_id": row["candidate_id"],
            "root_id": row["root_id"], "variant_id": row["variant_id"],
            "fact_family_id": req["fact_family_id"],
            "mechanically_valid": row["mechanically_valid"],
            "exact_correct": row["exact_correct"],
            "canonical_prediction": row["canonical_prediction"], "canonical_gold": row["canonical_gold"],
            "judgments_correct": judgments, "citation_sets_sufficient": sufficient,
            "all_six_judgments_correct": judgments == 6,
            "all_six_citations_sufficient": sufficient == 6,
            "qualifies_source_sufficient_wrong_order": qualifies,
        })
    roots = []
    for root in sorted({r["root_id"] for r in result_rows}):
        rows = [r for r in result_rows if r["root_id"] == root]
        assert len(rows) == 2 and {r["variant_id"] for r in rows} == {"perm_00", "perm_01"}
        replicated = all(r["qualifies_source_sufficient_wrong_order"] for r in rows)
        roots.append({
            "root_id": root, "candidate_id": rows[0]["candidate_id"],
            "fact_family_id": rows[0]["fact_family_id"],
            "training_exposure": "OWN_ARM_TRAINED_QUESTION",
            "correct_mappings": sum(r["exact_correct"] for r in rows),
            "all_six_sufficient_mappings": sum(r["all_six_citations_sufficient"] for r in rows),
            "qualifying_mappings": sum(r["qualifies_source_sufficient_wrong_order"] for r in rows),
            "replicated_source_sufficient_wrong_order": replicated,
            "same_canonical_order_both_mappings": rows[0]["canonical_prediction"] == rows[1]["canonical_prediction"],
            "mappings": [{key: r[key] for key in (
                "variant_id", "exact_correct", "judgments_correct", "citation_sets_sufficient",
                "all_six_citations_sufficient", "qualifies_source_sufficient_wrong_order",
                "canonical_prediction", "canonical_gold")} for r in rows],
        })
    replicated_roots = sum(r["replicated_source_sufficient_wrong_order"] for r in roots)
    replicated_families = len({r["fact_family_id"] for r in roots if r["replicated_source_sufficient_wrong_order"]})
    protocol = read(HERE / "PROTOCOL.json")
    gate = protocol["diagnostic_readout_gate"]
    sufficient = (replicated_roots >= gate["minimum_new_replicated_roots"] and
                  replicated_families >= gate["minimum_new_replicated_fact_families"])
    summary = dict(mechanical["summary"])
    summary.update({
        "semantic_evidence_gate": "COMPLETE_SOURCE_AUDITED",
        "correct_item_judgments": sum(r["judgments_correct"] for r in result_rows),
        "sufficient_citation_sets": sum(r["citation_sets_sufficient"] for r in result_rows),
        "insufficient_citation_sets": 72 - sum(r["citation_sets_sufficient"] for r in result_rows),
        "all_six_citations_sufficient_requests": sum(r["all_six_citations_sufficient"] for r in result_rows),
        "source_sufficient_wrong_order_requests": sum(r["qualifies_source_sufficient_wrong_order"] for r in result_rows),
        "new_replicated_roots": replicated_roots,
        "new_replicated_fact_families": replicated_families,
        "required_new_roots": gate["minimum_new_replicated_roots"],
        "required_new_fact_families": gate["minimum_new_replicated_fact_families"],
        "diagnostic_replication_sufficient": sufficient,
        "non_sufficient_citation_categories": dict(categories),
        "old_root130221_pooled": False,
    })
    return {
        "status": "COMPLETED_SEMANTICALLY_AUDITED_TRAIN_ONLY_REPLICATION",
        "summary": summary, "requests": result_rows, "roots": roots,
        "bindings": {"mechanical_scores": binding(HERE / "MECHANICAL_SCORES.json"),
                     "requests": binding(HERE / "REQUESTS.json"),
                     "raw_freeze": binding(HERE / "runs/B1/RAW_RESPONSE_FREEZE.json"),
                     "protocol": binding(HERE / "PROTOCOL.json"),
                     "citation_reviews": review_files,
                     "adjudication_code": binding(Path(__file__))},
        "limitations": [
            "Six own-arm trained questions, not heldout generalization.",
            "One sample per mapping; permutation and stochastic sampling changes are confounded.",
            "No A1 comparator, no new training, and no causal feedback/memory efficacy test.",
            "Visible citations cannot establish the model's internal reasoning mechanism.",
            "The 179186/109164 citation adjudicator had prior access to source anchors and aggregate mechanical statistics; this is disclosed in the review. The 135466/162896 and 104814/122711 reviewers were blind to final order. An independent post-audit rechecked all conclusions from raw records and source context.",
            "The 104814/122711 citation reviewer briefly saw extra citation-only packet rows for other roots before filtering to the assigned four requests; no Gold or final prediction was exposed, and the scope deviation is recorded in the review provenance.",
        ],
    }


def main():
    result = compute()
    save(HERE / "FINAL_SCORES.json", result)
    sufficient = result["summary"]["diagnostic_replication_sufficient"]
    decision = {
        "schema_version": "postweek_train_replication_decision_v1",
        "status": "DIAGNOSTIC_REPLICATION_SUFFICIENT_CONTROLS_STILL_NOT_RELEASED" if sufficient else "DIAGNOSTIC_REPLICATION_INSUFFICIENT_NO_CONTROL_RELEASE",
        "new_requests_planned": 12, "new_requests_submitted": 12,
        "new_requests_returned": result["summary"]["returned_requests"],
        "automatic_retries": 0, "additional_calls_authorized": False,
        "new_roots_qualifying_both_mappings": result["summary"]["new_replicated_roots"],
        "new_families_qualifying_both_mappings": result["summary"]["new_replicated_fact_families"],
        "required_new_roots_and_families": 3,
        "qualifying_new_root_ids": [r["root_id"] for r in result["roots"] if r["replicated_source_sufficient_wrong_order"]],
        "historical_roots_pooled": False,
        "old_day6_control_budget": {"planned_reserved": 24, "released": False, "executed": 0},
        "general_probe": "RETIRED_NOT_REOPENED",
        "formal_B_feedback_released": False, "training_launched": False,
        "dev_test_item_feedback_used": False,
        "causal_feedback_benefit_established": False,
        "next_action": "Archive this fixed replication. Design a separately frozen Train-only source-state/citation diagnostic control; do not expand or retry to reach3/3, use TEST feedback, or release current narrow B feedback.",
        "scores_binding": binding(HERE / "FINAL_SCORES.json"),
    }
    save(HERE / "DECISION.json", decision)
    print(json.dumps({"status": decision["status"], "summary": result["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
