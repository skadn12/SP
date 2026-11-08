import sys
from pathlib import Path

HERE = Path(__file__).parents[1]
sys.path.insert(0, str(HERE))

from protocol import make_variants
from scoring import score_response
from selection import select
from fractions import Fraction


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
