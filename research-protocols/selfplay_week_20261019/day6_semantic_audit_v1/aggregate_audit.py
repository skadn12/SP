"""Validate manual source reviews and summarize their fixed denominators."""
from __future__ import annotations

from collections import Counter
import copy
import itertools
import json
from pathlib import Path

from prepare_audit import HERE, bind, check, read, write_new


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    outputs = ("SEMANTIC_AUDIT.json", "SUMMARY.json", "B1_FEEDBACK_CANDIDATES.json")
    require(not any((HERE / name).exists() for name in outputs), "NO_OVERWRITE_AUDIT_RESULTS")
    check(read(HERE / "INPUT_MANIFEST.json")["files"])
    packet = read(HERE / "AUDIT_PACKET.json")
    expected = {q["candidate_id"]: q for q in packet["questions"]}
    reviewed = {}
    for path in sorted((HERE / "reviews").glob("roots_*.json")):
        document = read(path)
        for question in document["questions"]:
            cid = question["candidate_id"]
            require(cid in expected and cid not in reviewed, "UNEXPECTED_OR_DUPLICATE_REVIEW:" + cid)
            reviewed[cid] = {"question": question, "reviewer": document["reviewer"], "file": bind(path)}
    require(set(reviewed) == set(expected), "INCOMPLETE_12_QUESTION_REVIEW")
    adjudications = read(HERE / "ADJUDICATIONS.json")["adjudications"]
    overrides = {(a["candidate_id"], a["role"], a["label"], a["field"]): a for a in adjudications}
    used_overrides = set()
    ledger = []
    all_responses = []
    for cid, q in expected.items():
        record = reviewed[cid]
        review = record["question"]
        labels = set(q["display"])
        kind = q["question_type"]
        require(review["root_id"] == q["root_id"] and review["question_type"] == kind,
                "REVIEW_IDENTITY_DRIFT")
        require(review["source_gold_match"] is True, "SOURCE_GOLD_REQUIRES_RESOLUTION")
        source_answer = review["source_expected_answer"]
        require(len(source_answer) == len(set(source_answer)) and set(source_answer) <= labels,
                "INVALID_SOURCE_ANSWER")
        source_gold_match = (set(source_answer) == set(q["display_gold"]) if kind == "multi_select"
                             else source_answer == q["display_gold"])
        require(source_gold_match, "SOURCE_EXPECTED_VS_DISPLAY_GOLD_MISMATCH")
        require(set(review["source_basis"]) == labels, "INCOMPLETE_SOURCE_BASIS")
        turns = read(q["source_binding"]["path"])["turns"]
        by_ref = {t["turn_ref"]: t for t in turns}
        first_refs, ordinals = {}, {}
        for label, basis in review["source_basis"].items():
            support = basis["expected_binary_support"]
            require(type(support) is bool, "NON_BOOLEAN_SOURCE_SUPPORT")
            expected_support = kind == "ordering" or label in source_answer
            require(support == expected_support, "SOURCE_BASIS_VS_SOURCE_ANSWER")
            require(basis["anchor_turn_refs"] and all(ref in by_ref for ref in basis["anchor_turn_refs"]),
                    "SOURCE_ANCHOR_MISSING")
            if kind == "ordering":
                ordinal = basis["first_relevant_source_ordinal"]
                require(type(ordinal) is int and 1 <= ordinal <= len(turns), "INVALID_FIRST_ORDINAL")
                first_ref = turns[ordinal - 1]["turn_ref"]
                require(first_ref in basis["anchor_turn_refs"], "FIRST_ORDINAL_NOT_IN_REVIEWED_ANCHORS")
                first_refs[label], ordinals[label] = first_ref, ordinal
        if kind == "ordering":
            require(len(set(ordinals.values())) == len(labels), "UNRESOLVED_FIRST_OCCURRENCE_TIE")
            require(sorted(labels, key=ordinals.get) == source_answer, "SOURCE_ORDINAL_ORDER_MISMATCH")
        question_row = {
            "candidate_id": cid, "root_id": q["root_id"], "question_type": kind,
            "fact_family_id": q["annotation"]["fact_family_id"],
            "source_arm": q["annotation"]["source_arm"], "question_text": q["question_text"],
            "display": q["display"], "source_expected_answer": source_answer,
            "display_gold": q["display_gold"], "source_gold_match": source_gold_match,
            "source_issue": review["source_issue"], "source_basis": review["source_basis"],
            "source_binding": q["source_binding"], "review_file": record["file"],
            "reviewer": record["reviewer"], "responses": {},
        }
        require(set(review["responses"]) == {"A1", "B1"}, "MISSING_MODEL_REVIEW")
        for role in ("A1", "B1"):
            rr = review["responses"][role]
            raw = q["models"][role]["parsed"]
            items = rr["items"]
            require(isinstance(items, dict) and set(items) == labels, "INCOMPLETE_ITEM_REVIEW")
            require(rr["final_answer"] == raw["final_answer"], "REVIEW_MUTATED_RAW_FINAL")
            answer = raw["final_answer"]
            exact = set(answer) == set(source_answer) if kind == "multi_select" else answer == source_answer
            require(rr["exact_source_correct"] == exact, "MANUAL_EXACT_COUNT_DISAGREEMENT")
            item_rows = {}
            for label in q["display"]:
                item = copy.deepcopy(items[label])
                original = raw["item_evidence"][label]
                require(item["model_judgment"] == original["judgment"] and
                        item["model_turn_refs"] == original["turn_refs"], "REVIEW_MUTATED_RAW_ITEM")
                binary = (original["judgment"] == "supported") == review["source_basis"][label]["expected_binary_support"]
                require(item["judgment_semantically_correct"] == binary, "BINARY_JUDGMENT_REVIEW_DISAGREEMENT")
                require(type(item["selected_refs_sufficient"]) is bool, "UNRESOLVED_CITATION_JUDGMENT")
                item["reviewer_selected_refs_sufficient"] = item["selected_refs_sufficient"]
                key = cid, role, label, "selected_refs_sufficient"
                if key in overrides:
                    adj = overrides[key]
                    require(item["selected_refs_sufficient"] == adj["reviewer_original_value"], "ADJUDICATION_ORIGINAL_DRIFT")
                    item["selected_refs_sufficient"] = adj["adjudicated_value"]
                    item["primary_adjudication"] = adj
                    used_overrides.add(key)
                if item["selected_refs_sufficient"]:
                    require(bool(original["turn_refs"]), "EMPTY_REFERENCE_CANNOT_BE_SUFFICIENT")
                item["binary_judgment_correct"] = binary
                item["actual_cited_turns"] = q["models"][role]["actual_cited_turns"][label]
                item_rows[label] = item
            response_row = {
                "role": role, "candidate_id": cid, "root_id": q["root_id"],
                "fact_family_id": q["annotation"]["fact_family_id"], "question_type": kind,
                "raw_binding": q["models"][role]["raw_binding"],
                "training_exposure": q["annotation"]["model_training_exposure"][role],
                "final_answer": answer, "exact_source_correct": exact, "items": item_rows,
                "binary_judgments_correct": sum(x["binary_judgment_correct"] for x in item_rows.values()),
                "selected_reference_sets_sufficient": sum(x["selected_refs_sufficient"] for x in item_rows.values()),
                "source_qualified_fixed_citation_ordering_failure": False,
            }
            if kind == "multi_select":
                supported = {label for label, item in raw["item_evidence"].items() if item["judgment"] == "supported"}
                consistency = supported == set(answer)
                require(rr["judgment_final_consistent"] == consistency, "MC_CONSISTENCY_DISAGREEMENT")
                response_row.update(judgment_final_consistent=consistency,
                    false_positive_labels=sorted(set(answer) - set(source_answer)),
                    false_negative_labels=sorted(set(source_answer) - set(answer)),
                    jaccard=len(set(answer) & set(source_answer)) / len(set(answer) | set(source_answer)))
            else:
                require(rr["judgment_final_consistent"] is None, "ORDER_EXISTENCE_IS_NOT_ORDER_CONSISTENCY")
                positions = {label: i for i, label in enumerate(answer)}
                inverted = []
                for left, right in itertools.combinations(source_answer, 2):
                    if positions[left] > positions[right]:
                        scene_left = first_refs[left].rsplit(":T", 1)[0]
                        scene_right = first_refs[right].rsplit(":T", 1)[0]
                        inverted.append({"earlier": left, "later": right,
                            "earlier_first_ref": first_refs[left], "later_first_ref": first_refs[right],
                            "cross_scene": scene_left != scene_right})
                all_first_anchors_selected = all(first_refs[label] in raw["item_evidence"][label]["turn_refs"] for label in labels)
                all_citations_qualified = all(x["selected_refs_sufficient"] and x["binary_judgment_correct"] for x in item_rows.values()) and all_first_anchors_selected
                qualified_failure = all_citations_qualified and not exact and any(pair["cross_scene"] for pair in inverted)
                response_row.update(judgment_final_consistent=None,
                    event_membership_consistent=set(answer) == labels,
                    source_first_refs=first_refs, source_first_ordinals=ordinals,
                    source_inverted_pairs=inverted, source_pair_correct=15 - len(inverted), source_pair_total=15,
                    all_model_references_qualified_for_first_report=all_citations_qualified,
                    source_qualified_fixed_citation_ordering_failure=qualified_failure)
            question_row["responses"][role] = response_row
            all_responses.append(response_row)
        ledger.append(question_row)
    require(used_overrides == set(overrides), "UNUSED_ADJUDICATION")
    require(len(all_responses) == 24 and sum(len(r["items"]) for r in all_responses) == 144, "DENOMINATOR_DRIFT")
    summary = {
        "status": "SOURCE_SEMANTIC_AUDIT_COMPLETE",
        "scope": "TRAIN_DIAGNOSTIC_OUTPUTS_ONLY_NOT_DIRECT_ACCURACY_OR_CAUSAL_BENEFIT",
        "roots": 6, "questions": 12, "responses": 24, "item_judgments": 144,
        "source_gold_matches": sum(q["source_gold_match"] for q in ledger),
        "new_solver_requests": 0, "new_training_runs": 0, "new_questions": 0,
        "unresolved_items": 0, "primary_adjudications": len(adjudications), "models": {},
        "independent_human_annotation_study": False,
        "four_way_judgment_taxonomy_accuracy": "NOT_SCORED_NO_PREDECLARED_COMPLETE_TAXONOMY_GOLD",
    }
    for role in ("A1", "B1"):
        responses = [r for r in all_responses if r["role"] == role]
        items = [item for r in responses for item in r["items"].values()]
        ordering = [r for r in responses if r["question_type"] == "ordering"]
        multi = [r for r in responses if r["question_type"] == "multi_select"]
        qualified = [r for r in ordering if r["source_qualified_fixed_citation_ordering_failure"]]
        summary["models"][role] = {
            "responses": len(responses), "exact_answers": sum(r["exact_source_correct"] for r in responses),
            "multi_select_questions": len(multi), "multi_select_exact": sum(r["exact_source_correct"] for r in multi),
            "ordering_questions": len(ordering), "ordering_exact": sum(r["exact_source_correct"] for r in ordering),
            "item_judgments": len(items), "binary_judgments_correct": sum(i["binary_judgment_correct"] for i in items),
            "selected_reference_sets_sufficient": sum(i["selected_refs_sufficient"] for i in items),
            "selected_reference_sets_insufficient": sum(not i["selected_refs_sufficient"] for i in items),
            "reviewer_original_reference_sets_sufficient": sum(i["reviewer_selected_refs_sufficient"] for i in items),
            "empty_reference_sets": sum(not i["model_turn_refs"] for i in items),
            "multi_select_judgment_final_consistent": sum(r["judgment_final_consistent"] for r in multi),
            "multi_select_false_positive_labels": sum(len(r["false_positive_labels"]) for r in multi),
            "multi_select_false_negative_labels": sum(len(r["false_negative_labels"]) for r in multi),
            "ordering_pair_correct": sum(r["source_pair_correct"] for r in ordering),
            "ordering_pair_total": sum(r["source_pair_total"] for r in ordering),
            "ordering_cross_scene_inverted_pairs": sum(p["cross_scene"] for r in ordering for p in r["source_inverted_pairs"]),
            "ordering_within_scene_inverted_pairs": sum(not p["cross_scene"] for r in ordering for p in r["source_inverted_pairs"]),
            "ordering_failure_roots": sorted({r["root_id"] for r in ordering if not r["exact_source_correct"]}),
            "qualified_fixed_citation_failure_candidates": [r["candidate_id"] for r in qualified],
            "qualified_fixed_citation_failure_roots": sorted({r["root_id"] for r in qualified}),
            "qualified_fixed_citation_failure_fact_families": sorted({r["fact_family_id"] for r in qualified}),
        }
    b1 = summary["models"]["B1"]
    snapshot = {
        "status": "PRIVATE_REVIEWED_HYPOTHESES_NOT_FORMAL_REPRODUCED",
        "solver_model": "B1", "released_to_challenger": False,
        "new_solver_requests": 0, "control_requests": 0, "causal_feedback_benefit_established": False,
        "source_audit_protocol": bind(HERE / "PROTOCOL_ZH.md"),
        "candidates": [
            {"card_id": "D6-B1-LOCATED-CITATIONS-ORDER", "status": "hypothesis",
             "observable_claim": "A source-qualified first-report citation chain can coexist with an incorrect final order.",
             "supporting_candidates": b1["qualified_fixed_citation_failure_candidates"],
             "roots": b1["qualified_fixed_citation_failure_roots"],
             "fact_families": b1["qualified_fixed_citation_failure_fact_families"],
             "formal_reproduced": False, "control_support": "NOT_EXECUTED",
             "limits": "One sampled diagnostic output per item; new interface; trained/shared sources; no S0 status inheritance."},
            {"card_id": "D6-B1-SOURCE-JUDGMENT-REJECTION", "status": "hypothesis",
             "supporting_candidates": [r["candidate_id"] for r in all_responses if r["role"] == "B1" and any(not i["binary_judgment_correct"] for i in r["items"].values())],
             "observable_claim": "An explicitly source-supported statement is marked contradicted and omitted from the final multi-select answer.",
             "formal_reproduced": False, "control_support": "NOT_EXECUTED"},
            {"card_id": "D6-B1-CITATION-SEMANTICS", "status": "hypothesis",
             "supporting_item_count": b1["selected_reference_sets_insufficient"],
             "observable_claim": "Valid reference IDs can point to plans, questions, other events or incomplete claims; correct final answers do not repair this evidence gap.",
             "scope": "Diagnostic evidence quality; not an internal mechanism or training-utility claim.",
             "formal_reproduced": False, "control_support": "NOT_EXECUTED"},
        ],
        "second_wave_formal_B_GO": False,
        "A1_diagnosis_released_to_A_challenger": False,
    }
    write_new(HERE / "SEMANTIC_AUDIT.json", {
        "status": summary["status"], "scope": summary["scope"], "questions": ledger,
        "input_manifest": bind(HERE / "INPUT_MANIFEST.json"),
        "adjudications": bind(HERE / "ADJUDICATIONS.json"), "aggregator": bind(Path(__file__)),
    })
    write_new(HERE / "SUMMARY.json", summary)
    write_new(HERE / "B1_FEEDBACK_CANDIDATES.json", snapshot)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
