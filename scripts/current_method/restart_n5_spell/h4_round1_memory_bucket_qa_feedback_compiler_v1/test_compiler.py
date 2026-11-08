from .compiler import classify_priority, duplicate_relation, ordering_audit


def _model(full, no, bucket="hard_candidate"):
    return {"full_context_accuracy": full, "no_context_accuracy": no, "difficulty_bucket": bucket}


def _row(m0, clean, inclusive, qtype="single_choice"):
    return {"question_type": qtype, "models": {"M0": m0, "M_clean_policy": clean, "M_inclusive_policy": inclusive}}


def test_p1_is_frontier_both():
    row = _row(_model(0.0, 0.0, "full_zero_review"), _model(0.5, 0.25), _model(0.5, 0.25))
    assert classify_priority(row, {"shared_error_set": set()}) == "P1"


def test_duplicate_and_near_duplicate_routes():
    a = {"question": "Which event happened first?", "options": {"A": "one", "B": "two"}, "gold": ["A"], "evidence_refs": ["R01:T1"]}
    b = dict(a)
    assert duplicate_relation(a, b)["exact_duplicate"]
    c = {"question": "Which event happened first in the dialogue?", "options": {"A": "one", "B": "two"}, "gold": ["A"], "evidence_refs": ["R01:T1"]}
    assert duplicate_relation(a, c)["near_duplicate"]


def test_ordering_without_item_refs_is_unknown():
    assert ordering_audit({"question_type": "ordering", "evidence_groups": {"before": ["R01:T1"]}})["status"] == "UNKNOWN_OCCURRENCE"

