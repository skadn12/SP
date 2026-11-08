import unittest

from .protocol import build_prompt, validate_output


class ProtocolTests(unittest.TestCase):
    def test_prompt_contains_complete_dialogue_and_type_rules(self):
        candidate = {
            "candidate_id": "r:I-Q",
            "question_type": "single_choice",
            "question": "Why?",
            "options": {"A": "x", "B": "y", "C": "z", "D": "w"},
            "correct_option": "A",
            "evidence_refs": ["r:T1"],
        }
        prompt = build_prompt(
            candidate=candidate,
            turns=[{"turn_ref": "r:T1", "speaker": "Mia", "text": "x"}],
            evidence_turns=[{"turn_ref": "r:T1", "speaker": "Mia", "text": "x"}],
        )
        self.assertIn("COMPLETE_DIALOGUE", prompt)
        self.assertIn("single_choice", prompt)
        self.assertIn("OUTPUT_SCHEMA", prompt)

    def test_valid_accept(self):
        raw = {
            "answer_decision": "correct",
            "question_answerable": True,
            "evidence_sufficient": "sufficient",
            "question_answer_leakage": "clean",
            "option_quality": "unique",
            "reasoning_type": "causal",
            "grounded_answer": "Mia changed plans after receiving the new information.",
            "unsupported_claims": [],
            "evidence_gaps": [],
            "checked_evidence_refs": ["r:T1"],
            "reason": "The dialogue explicitly supports the answer and cited turn.",
        }
        result = validate_output(raw, expected_evidence_refs=["r:T1"])
        self.assertEqual(result["decision"], "protocol_valid")
        self.assertEqual(result["semantic_status"], "accept")
        self.assertTrue(result["training_ready"])

    def test_evidence_gap_is_not_answer_rejection(self):
        raw = {
            "answer_decision": "correct",
            "question_answerable": True,
            "evidence_sufficient": "insufficient",
            "question_answer_leakage": "clean",
            "option_quality": "unique",
            "reasoning_type": "multi_fact",
            "grounded_answer": "Mia changed plans after receiving the new information.",
            "unsupported_claims": [],
            "evidence_gaps": ["The antecedent turn is missing."],
            "checked_evidence_refs": ["r:T1"],
            "reason": "The full dialogue supports it, but the submitted refs omit a needed turn.",
        }
        result = validate_output(raw, expected_evidence_refs=["r:T1"])
        self.assertEqual(result["semantic_status"], "accept_with_evidence_gap")
        self.assertFalse(result["training_ready"])

    def test_changed_refs_fail_closed(self):
        raw = {
            "answer_decision": "incorrect",
            "question_answerable": False,
            "evidence_sufficient": "uncertain",
            "question_answer_leakage": "clean",
            "option_quality": "unique",
            "reasoning_type": "uncertain",
            "grounded_answer": "",
            "unsupported_claims": ["The gold is unsupported."],
            "evidence_gaps": ["The cited turn is insufficient."],
            "checked_evidence_refs": ["r:T2"],
            "reason": "The cited evidence does not support the gold.",
        }
        result = validate_output(raw, expected_evidence_refs=["r:T1"])
        self.assertEqual(result["decision"], "protocol_invalid")
        self.assertIn("evidence_refs_changed", result["errors"])

    def test_short_or_empty_grounded_answer_is_warning_only(self):
        raw = {
            "answer_decision": "correct",
            "question_answerable": True,
            "evidence_sufficient": "sufficient",
            "question_answer_leakage": "clean",
            "option_quality": "unique",
            "reasoning_type": "ordering",
            "grounded_answer": "",
            "unsupported_claims": [],
            "evidence_gaps": [],
            "checked_evidence_refs": ["r:T1"],
            "reason": "The complete dialogue fixes the order.",
        }
        result = validate_output(raw, expected_evidence_refs=["r:T1"])
        self.assertEqual(result["decision"], "protocol_valid")
        self.assertIn("grounded_answer_empty", result["warnings"])

    def test_optional_extra_and_missing_reason_are_warnings(self):
        raw = {
            "answer_decision": "correct",
            "question_answerable": True,
            "evidence_sufficient": "sufficient",
            "question_answer_leakage": "clean",
            "option_quality": "unique",
            "reasoning_type": "multi_fact",
            "grounded_answer": "The two supported selections are A and C.",
            "unsupported_claims": [],
            "evidence_gaps": [],
            "checked_evidence_refs": ["r:T1"],
            "evidence_groups": {"evidence": ["r:T1"]},
        }
        result = validate_output(raw, expected_evidence_refs=["r:T1"])
        self.assertEqual(result["decision"], "protocol_valid")
        self.assertIn("ignored_model_extra:evidence_groups", result["warnings"])
        self.assertIn("reason_omitted", result["warnings"])

    def test_ambiguous_supported_gold_is_not_semantically_valid(self):
        raw = {
            "answer_decision": "correct",
            "question_answerable": True,
            "evidence_sufficient": "sufficient",
            "question_answer_leakage": "clean",
            "option_quality": "ambiguous",
            "reasoning_type": "multi_fact",
            "grounded_answer": "The submitted answer is supported.",
            "unsupported_claims": [],
            "evidence_gaps": [],
            "checked_evidence_refs": ["r:T1"],
            "reason": "Another option is also supported.",
        }
        result = validate_output(raw, expected_evidence_refs=["r:T1"])
        self.assertTrue(result["answer_supported"])
        self.assertFalse(result["answer_semantically_valid"])
        self.assertEqual(result["semantic_status"], "uncertain")


if __name__ == "__main__":
    unittest.main()
