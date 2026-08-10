from __future__ import annotations

import unittest

from scripts.current_method.restart_n5_spell.contrastive_distractor_canary import (
    audit_frozen_source_qa,
    build_contrastive_distractor_prompt,
    deterministic_answer_placement,
    materialize_contrastive_candidate,
    option_length_diagnostics,
    should_run_attempt_one,
)
from scripts.current_method.restart_n5_spell.lineage_distractor_writer import (
    validate_distractor_object,
)


def _catalog() -> dict:
    return {"span_catalog_version": "dialogue_span_catalog_v1", "spans": [
        {"span_id": "T1#S0001", "turn_id": "T1", "session_id": "S1",
         "speaker": "John", "exact_text": "John attended community meetings."},
        {"span_id": "T2#S0001", "turn_id": "T2", "session_id": "S2",
         "speaker": "John", "exact_text": "The meetings reinforced his political commitment."},
    ]}


def _raw() -> dict:
    return {"qa_id": "johns_motivation_for_politics",
        "question": "What reinforced John's political commitment?",
        "correct_answer_text": "His participation in community meetings.",
        "distractors": ["One", "Two", "Three"],
        "answer_support_span_ids": ["T1#S0001", "T2#S0001"]}


class ContrastiveDistractorCanaryTests(unittest.TestCase):
    def test_distractors_require_exactly_three_unique_nonanswer(self) -> None:
        self.assertEqual(validate_distractor_object(
            {"distractors": ["a", "b", "c"]},
            correct_answer_text="gold")["decision"], "distractor_accept")
        for values in (["a", "b"], ["a", "b", "c", "d"],
                       ["a", "a", "c"], ["a", "gold", "c"]):
            self.assertEqual(validate_distractor_object(
                {"distractors": values}, correct_answer_text="gold")["decision"],
                "distractor_schema_invalid")

    def test_prompt_freezes_qa_and_attempt_feedback(self) -> None:
        prompt = build_contrastive_distractor_prompt(
            frozen_dialogue=[{"turn_id": "T1", "session_id": "S1",
                              "speaker": "John", "text": "text"}],
            span_catalog=_catalog(), base_qa=_raw(), attempt_index=0,
            structured_feedback=None)
        self.assertIn("What reinforced John's political commitment?", prompt)
        self.assertIn("full_frozen_dialogue", prompt)
        self.assertNotIn("previous_distractors\": [", prompt)
        feedback = {"previous_distractors": ["a", "b", "c"],
            "previous_final_decision": "bad_ground",
            "previous_grounding_diagnosis": "bad_ground_semantic_leakage"}
        retry = build_contrastive_distractor_prompt(
            frozen_dialogue=[], span_catalog=_catalog(), base_qa=_raw(),
            attempt_index=1, structured_feedback=feedback)
        self.assertIn("bad_ground_semantic_leakage", retry)
        with self.assertRaises(ValueError):
            build_contrastive_distractor_prompt(
                frozen_dialogue=[], span_catalog=_catalog(), base_qa=_raw(),
                attempt_index=1, structured_feedback={**feedback, "notes": "free"})

    def test_source_audit_rejects_question_or_support_drift(self) -> None:
        materialized = {
            "qa_id": _raw()["qa_id"], "question": _raw()["question"],
            "canonical_answer_text": _raw()["correct_answer_text"],
            "evidence_span_ids": _raw()["answer_support_span_ids"],
            "admission_mode": "spell_style_minimal_v1",
            "local_option_binding": True,
        }
        accepted = audit_frozen_source_qa(raw_qa=_raw(),
            materialized_qa=materialized, span_catalog=_catalog())
        self.assertEqual(accepted["decision"], "source_qa_accept")
        drifted = dict(materialized, question="changed")
        self.assertEqual(audit_frozen_source_qa(raw_qa=_raw(),
            materialized_qa=drifted, span_catalog=_catalog())["decision"],
            "source_qa_reject")

    def test_materialization_preserves_frozen_fields(self) -> None:
        result = materialize_contrastive_candidate(base_qa=_raw(),
            distractors=["His sports success.", "His library job.",
                         "His writing mentorship."], target_letter="B",
            span_catalog=_catalog())
        qa = result["materialized_qa"]
        self.assertEqual(result["admission"]["spell_style_minimal_v1"]["decision"],
                         "accept")
        self.assertEqual(qa["question"], _raw()["question"])
        self.assertEqual(qa["canonical_answer_text"],
                         _raw()["correct_answer_text"])
        self.assertEqual(set(qa["evidence_span_ids"]),
                         set(_raw()["answer_support_span_ids"]))

    def test_answer_placement_is_fixed_and_content_independent(self) -> None:
        expected = {
            ("johns_motivation_for_politics", 0): "B",
            ("johns_motivation_for_politics", 1): "D",
            ("marias_volunteering_inspiration", 0): "C",
            ("marias_volunteering_inspiration", 1): "A",
        }
        for key, letter in expected.items():
            first = deterministic_answer_placement(*key)
            second = deterministic_answer_placement(*key)
            self.assertEqual(first, second)
            self.assertEqual(first["target_letter"], letter)
            self.assertFalse(first["content_dependent"])

    def test_attempt_one_only_for_semantic_leakage(self) -> None:
        self.assertTrue(should_run_attempt_one(attempt_index=0,
            final_decision="bad_ground",
            grounding_diagnosis="bad_ground_semantic_leakage"))
        for decision, diagnosis in (
            ("bad_ground", "bad_ground_position_bias"),
            ("bad_ground", "bad_ground_mixed_bias"),
            ("reject_easy", "bad_ground_semantic_leakage"),
        ):
            self.assertFalse(should_run_attempt_one(attempt_index=0,
                final_decision=decision, grounding_diagnosis=diagnosis))
        self.assertFalse(should_run_attempt_one(attempt_index=1,
            final_decision="bad_ground",
            grounding_diagnosis="bad_ground_semantic_leakage"))

    def test_length_metrics_are_diagnostic_only(self) -> None:
        result = materialize_contrastive_candidate(base_qa=_raw(),
            distractors=["Short.", "A much longer but still valid distractor.",
                         "Another one."], target_letter="B",
            span_catalog=_catalog())
        diagnostics = option_length_diagnostics(result["materialized_qa"])
        self.assertTrue(diagnostics["diagnostic_only"])
        self.assertIn("correct_is_unique_longest", diagnostics)


if __name__ == "__main__":
    unittest.main()
