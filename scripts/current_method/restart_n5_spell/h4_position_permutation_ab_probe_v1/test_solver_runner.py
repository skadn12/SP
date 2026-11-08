import unittest

from .seed_schedule import stable_seed
from .solver_runner import (
    OUTPUT_MODE_CONTENT_HIDDEN,
    OUTPUT_MODE_CONTENT_VISIBLE,
    OUTPUT_MODE_LABEL,
    SOLVER_B2_CONTENT_RESPONDER_PROMPT,
    _content_parse,
    _content_choices,
    _content_prompt,
    _content_reserve,
    _display,
    _label_parse,
)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(str(text))

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"{row['role']}: {row['content']}" for row in messages)


class SolverRunnerTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "candidate_id": "r:C-1", "root_id": "r",
            "question_type": "single_choice", "question": "Which?",
            "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            "correct_option": "B",
        }
        self.permutation = {
            "permutation_id": "perm_000", "is_cyclic": True,
            "display_order": ["B", "C", "D", "A"],
            "display_to_canonical_mapping": {"A": "B", "B": "C", "C": "D", "D": "A"},
        }

    def test_seed_is_paired_across_conditions(self):
        self.assertEqual(stable_seed("r", "r:C-1", 0), stable_seed("r", "r:C-1", 0))
        self.assertNotEqual(stable_seed("r", "r:C-1", 0), stable_seed("r", "r:C-1", 1))

    def test_display_gold_rebinds_and_label_parser_scores(self):
        display, displayed_gold, mapping = _display(self.candidate, self.permutation, "label")
        self.assertEqual(displayed_gold, ["A"])
        parsed, audit = _label_parse("<box>A</box>", self.candidate, display, displayed_gold)
        self.assertEqual(parsed, ["A"])
        self.assertTrue(audit["correct"])

    def test_content_modes_have_same_reserve_and_parse_to_canonical(self):
        display, displayed_gold, mapping = _display(self.candidate, self.permutation, OUTPUT_MODE_CONTENT_VISIBLE)
        tokenizer = _Tokenizer()
        self.assertEqual(
            _content_reserve(tokenizer, self.candidate, display),
            _content_reserve(tokenizer, self.candidate, display),
        )
        parsed, audit = _content_parse("<box>beta</box>", self.candidate, display, mapping, ["B"])
        self.assertEqual(parsed, ["B"])
        self.assertTrue(audit["correct"])

    def test_hidden_content_prompt_uses_complete_text_contract(self):
        display, _, _ = _display(self.candidate, self.permutation, OUTPUT_MODE_CONTENT_HIDDEN)
        prompt = _content_prompt(_Tokenizer(), self.candidate, display, [], visible_labels=False)
        self.assertIn(SOLVER_B2_CONTENT_RESPONDER_PROMPT, prompt)
        self.assertIn("alpha", prompt)
        self.assertIn("beta", prompt)
        self.assertNotIn("A. beta", prompt)
        self.assertIn("No labels or item IDs are shown", prompt)

    def test_visible_content_prompt_uses_same_complete_text_contract(self):
        display, _, _ = _display(self.candidate, self.permutation, OUTPUT_MODE_CONTENT_VISIBLE)
        prompt = _content_prompt(_Tokenizer(), self.candidate, display, [], visible_labels=True)
        self.assertIn(SOLVER_B2_CONTENT_RESPONDER_PROMPT, prompt)
        self.assertIn("A. beta", prompt)
        self.assertIn("Do not return labels", prompt)
        self.assertNotIn("Return exactly one uppercase option letter", prompt)

    def test_multi_content_order_is_set_scored(self):
        candidate = {
            "candidate_id": "r:M-1", "root_id": "r", "question_type": "multi_select",
            "question": "Which?", "options": {"A": "one", "B": "two", "C": "three", "D": "four", "E": "five"},
            "gold": ["A", "C"],
        }
        permutation = {"permutation_id": "perm_000", "is_cyclic": True,
                       "display_order": ["A", "B", "C", "D", "E"],
                       "display_to_canonical_mapping": {key: key for key in "ABCDE"}}
        display, _, mapping = _display(candidate, permutation, OUTPUT_MODE_CONTENT_HIDDEN)
        parsed, audit = _content_parse("<box>three || one</box>", candidate, display, mapping, ["A", "C"])
        self.assertEqual(set(parsed), {"A", "C"})
        self.assertTrue(audit["correct"])

    def test_ordering_content_choices_are_complete_permutations(self):
        candidate = {
            "question_type": "ordering",
            "items": {"I1": "one", "I2": "two", "I3": "three", "I4": "four"},
        }
        display = {"I1": "two", "I2": "three", "I3": "four", "I4": "one"}
        mapping = {"I1": "I2", "I2": "I3", "I3": "I4", "I4": "I1"}
        choices = _content_choices(candidate, display, mapping)
        self.assertEqual(len(choices), 24)
        self.assertEqual(len(set(choices)), 24)

    def test_ordering_label_gold_rebinds_to_display_slots(self):
        candidate = {
            "candidate_id": "r:O-1", "root_id": "r", "question_type": "ordering",
            "question": "Order?", "items": {"I1": "one", "I2": "two", "I3": "three", "I4": "four"},
            "gold": ["I1", "I2", "I3", "I4"],
        }
        permutation = {"permutation_id": "perm_001", "is_cyclic": False,
                       "display_order": ["I1", "I2", "I4", "I3"],
                       "display_to_canonical_mapping": {"I1": "I1", "I2": "I2", "I3": "I4", "I4": "I3"}}
        _, displayed_gold, _ = _display(candidate, permutation, OUTPUT_MODE_LABEL)
        self.assertEqual(displayed_gold, ["I1", "I2", "I4", "I3"])


if __name__ == "__main__":
    unittest.main()
