import sys
from pathlib import Path

HERE = Path(__file__).parents[1]
sys.path.insert(0, str(HERE))

from protocol import make_variants
from scoring import score_response
from selection import audit_complete_call_set, select
from fractions import Fraction
from copy import deepcopy


def ms():
    return {"candidate_id": "q", "root_id": "r", "question_type": "multi_select",
            "question": "q", "options": {c: c * 2 for c in "ABCDEF"}, "gold": ["A", "C"]}


def ordering():
    return {"candidate_id": "o", "root_id": "r", "question_type": "ordering",
            "question": "q", "items": {f"I{i}": str(i) for i in range(1, 7)}, "gold": [f"I{i}" for i in range(1, 7)]}


def test_random_variants_are_reproducible_and_not_cyclic():
    q = ms(); variants = make_variants(q, seed=3, count=6)
    assert variants == make_variants(q, seed=3, count=6)
    assert len({tuple(v["display_order_content_ids"]) for v in variants}) == 6
    assert tuple(variants[0]["canonical_order_content_ids"]) != tuple(variants[0]["content_to_text"])


def test_random_variants_remain_unique_when_collisions_are_unavoidable():
    q = ordering()
    variants = make_variants(q, seed=3, count=720)
    orders = [tuple(row["display_order_content_ids"]) for row in variants]
    assert len(set(orders)) == 720
    q["gold"] = list(reversed(q["gold"]))
    assert variants == make_variants(q, seed=3, count=720)
    for count in (0, 721, True):
        try:
            make_variants(q, seed=3, count=count)
        except ValueError as error:
            assert str(error) == "VARIANT_COUNT_OUT_OF_RANGE"
        else:
            raise AssertionError("invalid permutation count accepted")


def _full_call_fixture():
    planned, records = [], []
    for variant in range(6):
        for slot in range(3):
            rid = f"q__perm_{variant:02d}__{slot}"
            identity = {"request_id": rid, "candidate_id": "q", "slot": slot}
            planned.append({**identity, "mode": "full_context", "variant": {"variant_id": f"perm_{variant:02d}"}})
            records.append({**identity, "variant_id": f"perm_{variant:02d}", "evaluation_valid": True})
    return {"requests": planned}, records


def test_missing_one_of_three_rollouts_cannot_admit_question():
    manifest, records = _full_call_fixture()
    assert audit_complete_call_set(manifest, records, "q") == []
    # Every permutation still has answers; a nonempty-permutation check misses this.
    missing = records.pop(1)
    errors = audit_complete_call_set(manifest, records, "q")
    assert errors == ["MISSING_RESPONSE:" + missing["request_id"]]
    scores = {f"perm_{index:02d}": [Fraction(1, 2) for row in records
                                  if row["variant_id"] == f"perm_{index:02d}"] for index in range(6)}
    result = select(scores, list(scores), data_errors=errors)
    assert result["selected_by_policy"] is None
    assert result["selection_reason"] == "EVALUATION_INCOMPLETE_OR_INVALID"


def test_response_count_cannot_hide_duplicate_or_wrong_slot():
    manifest, records = _full_call_fixture()
    original_second = records[1]["request_id"]
    records[1] = deepcopy(records[0])
    errors = audit_complete_call_set(manifest, records)
    assert "DUPLICATE_RESPONSE:" + records[0]["request_id"] in errors
    assert "MISSING_RESPONSE:" + original_second in errors
    manifest, records = _full_call_fixture()
    records[0]["slot"] = 2
    records[1]["candidate_id"] = "wrong_candidate"
    records[2]["evaluation_valid"] = False
    errors = audit_complete_call_set(manifest, records, "q")
    assert "RESPONSE_PLAN_MISMATCH:" + records[0]["request_id"] + ":slot" in errors
    assert "RESPONSE_PLAN_MISMATCH:" + records[1]["request_id"] + ":candidate_id" in errors
    assert "INVALID_RESPONSE:" + records[2]["request_id"] in errors


def test_unplanned_response_and_duplicate_plan_are_invalid():
    manifest, records = _full_call_fixture()
    records.append({**records[0], "request_id": "extra"})
    assert "UNPLANNED_RESPONSE:extra" in audit_complete_call_set(manifest, records, "q")
    manifest["requests"].append({**manifest["requests"][0], "request_id": "extra"})
    assert "DUPLICATE_PLANNED_SLOT:q:perm_00:0" in audit_complete_call_set(manifest, records, "q")


def test_gold_rebind_and_multiselect_six_true_score():
    q = ms(); v = make_variants(q, seed=9, count=1)[0]
    text_to_content = {text: cid for cid, text in v["content_to_text"].items()}
    q["gold_content_ids"] = [text_to_content[q["options"][label]] for label in q["gold"]]
    chosen = [label for label, cid in v["display_to_content"].items() if cid in q["gold_content_ids"]]
    out = score_response(q, {"variant": v}, f"<box>{','.join(chosen)}</box>")
    assert out["scored"] and out["exact_match"]


def test_multiselect_all_six_gold_uses_tp_over_six():
    q = ms(); q["gold"] = list("ABCDEF")
    v = make_variants(q, seed=4, count=1)[0]
    q["gold_content_ids"] = list(v["content_to_text"])
    out = score_response(q, {"variant": v}, "<box>A,B,C,D,E,F</box>")
    assert out["partial_score"] == "1"


def test_selection_has_no_ground_gate_and_invalid_is_pending():
    result = select({"a": [Fraction(1, 3)]}, ["a"])
    assert result["selected_by_policy"] is True
    result = select({}, ["a"])
    assert result["selected_by_policy"] is None
