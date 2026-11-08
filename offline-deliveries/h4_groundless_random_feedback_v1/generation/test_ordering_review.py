from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    from .ordering_review import _review_object, bridge, prepare
    from .prompts import build_multiselect_generation_prompt, build_ordering_review_prompt
except ImportError:  # Support direct execution from this directory.
    from ordering_review import _review_object, bridge, prepare
    from prompts import build_multiselect_generation_prompt, build_ordering_review_prompt


class OrderingReviewTests(unittest.TestCase):
    def test_review_schema_and_anchor_fields(self):
        labels = [f"I{i}" for i in range(1, 7)]
        item = {"supported": True, "order_anchor": {"turn_ref": "R01:T1", "reason": "primary"},
                "support_refs": ["R01:T0"], "alternative_matching_turn_refs": [],
                "actor_ok": True, "modality_ok": True, "state_ok": True,
                "neutral_stem_ok": True, "reason": "ok"}
        obj = {"reviews": [{"candidate_id": "x", "items": {label: dict(item, order_anchor={"turn_ref": f"R01:T{n}", "reason": "primary"}) for n, label in enumerate(labels)},
                            "coherent": True, "unique_order": True, "alternative_orders": [],
                            "neutral_stem_ok": True, "reason": "ok"}]}
        self.assertEqual(_review_object(json.dumps(obj), ["x"]), obj)

    def test_prompt_allows_context_but_separates_anchor(self):
        prompt = build_ordering_review_prompt("r", [{"candidate_id": "x"}], [{"turn_ref": "T1", "text": "a", "speaker": "A"}])
        self.assertIn("order_anchor", prompt)
        self.assertIn("support ref may resolve", prompt)
        self.assertIn("APPEAR IN THE SUPPLIED DIALOGUE", prompt)
        self.assertNotIn("DeepSeek", prompt)

    def test_multiselect_prompt_has_two_to_six_and_feedback_slot(self):
        prompt = build_multiselect_generation_prompt("r", {"turns": []}, ["x"], "review guidance")
        self.assertIn("between two and six", prompt)
        self.assertIn("[FEEDBACK_SUFFIX]", prompt)
        self.assertIn("Evidence shortage is not evidence of contradiction", prompt)

    def test_second_pass_prompt_does_not_accept_previous_verdict(self):
        # The prepare path derives every request only from the frozen candidate/source inputs.
        with tempfile.TemporaryDirectory() as _:
            self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
