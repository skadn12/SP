import copy
import unittest

from scripts.current_method.restart_n5_spell.core_qa_admission import (
    CORE_QA_JSON_SCHEMA,
    admit_core_qa,
    core_qa_schema_sha256,
    core_qa_validator_sha256,
    validate_core_qa_schema,
    validate_core_qa_structure,
)
from scripts.current_method.restart_n5_spell.questioner_contract import (
    build_span_catalog,
)


class CoreQaAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.turns = [
            {"turn_id": "parent:turn_001", "session_id": "parent",
             "speaker": "Alice Chen",
             "text": "On January 5, Alice Chen promised to submit the budget."},
            {"turn_id": "child:turn_001", "session_id": "child",
             "speaker": "Bob Reed",
             "text": "Bob Reed approved the budget on January 8."},
            {"turn_id": "child:turn_002", "session_id": "child",
             "speaker": "Carol Singh",
             "text": "Carol Singh discussed transportation logistics."},
        ]
        self.catalog = build_span_catalog(self.turns)
        self.s1 = "parent:turn_001#S0001"
        self.s2 = "child:turn_001#S0001"
        self.s3 = "child:turn_002#S0001"

    def valid(self):
        return {"question": "What did Alice Chen promise to submit?",
                "correct_answer_text": "the budget",
                "answer_support_span_ids": [self.s1]}

    def test_exact_three_field_schema_accepts(self):
        result = validate_core_qa_structure(self.valid(),
                                            span_catalog=self.catalog)
        self.assertEqual(result["decision"], "core_qa_schema_accept")
        self.assertEqual(set(CORE_QA_JSON_SCHEMA["required"]), set(self.valid()))
        runner_result = validate_core_qa_schema(self.valid(),
                                                span_catalog=self.catalog)
        self.assertEqual(runner_result["core_qa"], self.valid())

    def test_model_owned_fields_are_rejected(self):
        for field, value in (("qa_id", "model-id"), ("distractors", ["x"] * 3),
                             ("reasoning_steps", [])):
            with self.subTest(field=field):
                raw = {**self.valid(), field: value}
                result = validate_core_qa_structure(raw,
                                                    span_catalog=self.catalog)
                self.assertEqual(result["decision"], "core_qa_schema_invalid")
                self.assertIn("core_qa_field_set_invalid", result["reasons"])

    def test_support_must_have_one_to_six_unique_existing_ids(self):
        cases = (
            ([], "core_qa_support_count_invalid"),
            ([[self.s1]], "core_qa_support_invalid"),
            ([self.s1] * 2, "core_qa_duplicate_support"),
            ([f"missing:{index}" for index in range(7)],
             "core_qa_support_count_invalid"),
            (["missing#S0001"], "core_qa_unknown_support_span"),
        )
        for support, reason in cases:
            with self.subTest(reason=reason):
                raw = {**self.valid(), "answer_support_span_ids": support}
                result = validate_core_qa_structure(raw,
                                                    span_catalog=self.catalog)
                self.assertEqual(result["decision"], "core_qa_schema_invalid")
                self.assertIn(reason, result["reasons"])

    def test_support_is_canonicalized_to_catalog_order(self):
        raw = {"question": "What happened after Alice Chen made her promise?",
               "correct_answer_text": "Bob Reed approved the budget",
               "answer_support_span_ids": [self.s2, self.s1]}
        result = validate_core_qa_structure(raw, span_catalog=self.catalog)
        self.assertEqual(result["core_qa_candidate"]["answer_support_span_ids"],
                         [self.s1, self.s2])

    def test_empty_and_artifact_text_are_schema_failures(self):
        for field, value in (("question", "  "),
                             ("correct_answer_text", "```json\nanswer\n```")):
            with self.subTest(field=field):
                raw = {**self.valid(), field: value}
                self.assertEqual(validate_core_qa_structure(
                    raw, span_catalog=self.catalog)["decision"],
                    "core_qa_schema_invalid")

    def test_program_assigns_qa_id_and_accepts_grounded_item(self):
        raw = self.valid()
        original = copy.deepcopy(raw)
        result = admit_core_qa(raw, span_catalog=self.catalog,
                               qa_id="conv26_item_001")
        self.assertEqual(result["decision"], "core_qa_accept")
        self.assertEqual(result["core_qa"]["qa_id"], "conv26_item_001")
        self.assertEqual(raw, original)
        self.assertFalse(result["mechanical_checks"]
                         ["semantic_entailment_verified"])
        self.assertEqual(result["admission_scope"],
                         "mechanical_only_not_entailment")

    def test_runner_may_attach_program_qa_id_after_admission(self):
        result = admit_core_qa(self.valid(), span_catalog=self.catalog)
        self.assertEqual(result["decision"], "core_qa_accept")
        self.assertEqual(set(result["core_qa"]),
                         {"question", "correct_answer_text",
                          "answer_support_span_ids"})

    def test_explicit_speaker_must_be_in_selected_evidence(self):
        raw = {"question": "What did Alice Chen approve?",
               "correct_answer_text": "the budget",
               "answer_support_span_ids": [self.s2]}
        result = admit_core_qa(raw, span_catalog=self.catalog, qa_id="q1")
        self.assertEqual(result["decision"], "core_qa_reject")
        self.assertIn("core_qa_subject_not_grounded", result["reasons"])

    def test_speaker_can_be_grounded_by_evidence_text(self):
        raw = {"question": "What did Alice Chen promise?",
               "correct_answer_text": "Alice Chen promised the budget",
               "answer_support_span_ids": [self.s1]}
        result = admit_core_qa(raw, span_catalog=self.catalog, qa_id="q1")
        self.assertEqual(result["decision"], "core_qa_accept")

    def test_explicit_number_and_time_must_be_grounded(self):
        raw = {"question": "What was approved on January 9?",
               "correct_answer_text": "the budget",
               "answer_support_span_ids": [self.s2]}
        result = admit_core_qa(raw, span_catalog=self.catalog, qa_id="q1")
        self.assertEqual(result["decision"], "core_qa_reject")
        self.assertIn("core_qa_number_or_time_not_grounded", result["reasons"])
        self.assertIn("number:9", result["mechanical_checks"]
                      ["ungrounded_number_or_absolute_time_literals"])

    def test_number_words_and_digits_use_same_mechanical_key(self):
        turns = [{"turn_id": "t1", "session_id": "s", "speaker": "Dana",
                  "text": "Dana filed 3 reports."}]
        catalog = build_span_catalog(turns)
        raw = {"question": "How many reports did Dana file?",
               "correct_answer_text": "three reports",
               "answer_support_span_ids": ["t1#S0001"]}
        result = admit_core_qa(raw, span_catalog=catalog, qa_id="q1")
        self.assertEqual(result["decision"], "core_qa_accept")

    def test_relative_temporal_question_requires_two_turns(self):
        raw = {"question": "What happened after Alice Chen promised the budget?",
               "correct_answer_text": "Bob Reed approved the budget",
               "answer_support_span_ids": [self.s2]}
        result = admit_core_qa(raw, span_catalog=self.catalog, qa_id="q1")
        self.assertIn("core_qa_relative_time_requires_multiple_turns",
                      result["reasons"])
        raw["answer_support_span_ids"] = [self.s2, self.s1]
        result = admit_core_qa(raw, span_catalog=self.catalog, qa_id="q1")
        self.assertNotIn("core_qa_relative_time_requires_multiple_turns",
                         result["reasons"])

    def test_answer_requires_meaningful_lexical_evidence_anchor(self):
        raw = {"question": "What did Carol Singh discuss?",
               "correct_answer_text": "a medical diagnosis",
               "answer_support_span_ids": [self.s3]}
        result = admit_core_qa(raw, span_catalog=self.catalog, qa_id="q1")
        self.assertEqual(result["decision"], "core_qa_reject")
        self.assertIn("core_qa_answer_lexical_anchor_missing", result["reasons"])

    def test_schema_and_validator_hashes_are_stable(self):
        self.assertEqual(core_qa_schema_sha256(), core_qa_schema_sha256())
        self.assertEqual(core_qa_validator_sha256(), core_qa_validator_sha256())
        self.assertEqual(len(core_qa_schema_sha256()), 64)
        self.assertEqual(len(core_qa_validator_sha256()), 64)


if __name__ == "__main__":
    unittest.main()
