from .compiler import classify_priority_v2, duplicate_relation, ordering_audit


def _m(full, no, bucket="hard_candidate"):
    return {"full_context_accuracy": full, "no_context_accuracy": no, "difficulty_bucket": bucket}


def test_multi_select_p4_is_not_blocked_by_non_ordering_occurrence():
    row = {"question_type": "multi_select", "ordering_occurrence_status": "NOT_APPLICABLE",
           "models": {"M0": _m(0, 0, "full_zero_review"), "M_clean_policy": _m(.2, 0), "M_inclusive_policy": _m(.2, 0)}}
    assert classify_priority_v2(row, {"shared_error_set": ["B"], "shared_overlap": 1.0}) == "P4"


def test_duplicate_and_near_duplicate():
    a = {"question": "Which event happened first?", "options": {"A": "one", "B": "two"}, "gold": ["A"], "evidence_refs": ["R01:T1"]}
    assert duplicate_relation(a, dict(a))["exact_duplicate"]
    b = {**a, "question": "Which event happened first in the dialogue?"}
    assert duplicate_relation(a, b)["near_duplicate"]


def test_ordering_without_item_refs_is_unknown():
    assert ordering_audit({"question_type": "ordering", "evidence_groups": {"before": ["R01:T1"]}})["status"] == "UNKNOWN_OCCURRENCE"

