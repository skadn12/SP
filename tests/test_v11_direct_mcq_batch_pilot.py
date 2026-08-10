from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_expansion_qa import (
    build_qa_source_view,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.direct_mcq_batch_pilot import (
    MCQ_COUNT,
    _answer_letters,
    build_direct_mcq_batch_prompt,
    validate_direct_mcq_batch,
)


class V11DirectMCQBatchPilotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.expanded = {
            "artifact_role": "canonical_expanded_dialogue",
            "qa_source_candidate": True,
            "root_id": "root-test",
            "turns": [
                {"turn_ref": "P:T1", "session_ref": "parent",
                 "speaker_label": "A", "text": "A planned a picnic.",
                 "source_partition": "original_parent"},
                {"turn_ref": "P:T2", "session_ref": "parent",
                 "speaker_label": "B", "text": "B packed lunch.",
                 "source_partition": "original_parent"},
                {"turn_ref": "C:S1:T1", "session_ref": "child-1",
                 "speaker_label": "A", "text": "A moved it indoors because of rain.",
                 "source_partition": "promoted_child"},
                {"turn_ref": "C:S2:T1", "session_ref": "child-2",
                 "speaker_label": "B", "text": "B booked the community hall.",
                 "source_partition": "promoted_child"},
            ],
        }
        self.expanded["canonical_turn_refs"] = [
            row["turn_ref"] for row in self.expanded["turns"]]
        self.view = build_qa_source_view(self.expanded)
        self.ids = [row["span_id"] for row in self.view["span_catalog"]["spans"]]

    def _candidate(self, index: int, support: list[str] | None = None) -> dict:
        return {"question": f"Question {index}?",
                "correct_answer_text": f"Correct {index}",
                "distractors": [f"Wrong {index}a", f"Wrong {index}b",
                                f"Wrong {index}c"],
                "answer_support_span_ids": support or self.ids[2:]}

    def _batch(self) -> dict:
        return {"mcq_candidates": [self._candidate(index)
                                    for index in range(1, MCQ_COUNT + 1)]}

    def test_prompt_keeps_child_involvement_as_hard_rule(self) -> None:
        prompt = build_direct_mcq_batch_prompt(self.view)
        self.assertIn("parent_only questions are forbidden", prompt)
        self.assertIn("at least one promoted_child", prompt)
        self.assertIn("The newest Child turn is not required", prompt)
        self.assertNotIn("World Memory", prompt.split(
            "FROZEN COMPLETE DIALOGUE AND EVIDENCE CATALOG", 1)[1])

    def test_child_only_and_parent_child_are_accepted(self) -> None:
        batch = self._batch()
        batch["mcq_candidates"][0]["answer_support_span_ids"] = self.ids[2:]
        batch["mcq_candidates"][1]["answer_support_span_ids"] = [
            self.ids[0], self.ids[2]]
        result = validate_direct_mcq_batch(
            batch, source_view=self.view, root_id="root-test")
        self.assertEqual("direct_mcq_batch_accept", result["decision"])
        self.assertEqual(["multi_child", "parent_child", "multi_child"],
            [row["evidence_scope"]["evidence_scope"] for row in result["items"]])

    def test_parent_only_is_rejected(self) -> None:
        batch = self._batch()
        batch["mcq_candidates"][0]["answer_support_span_ids"] = self.ids[:2]
        result = validate_direct_mcq_batch(
            batch, source_view=self.view, root_id="root-test")
        self.assertEqual("direct_mcq_batch_contract_reject", result["decision"])
        self.assertTrue(any("parent_only_forbidden" in reason
                            for reason in result["reasons"]))
        batch["mcq_candidates"][0]["answer_support_span_ids"] = [self.ids[0],
                                                                  self.ids[0] + "x"]
        result = validate_direct_mcq_batch(
            batch, source_view=self.view, root_id="root-test")
        self.assertTrue(any("bad_span_id" in reason
                            for reason in result["reasons"]))

    def test_true_parent_only_with_two_parent_spans_is_rejected(self) -> None:
        batch = self._batch()
        batch["mcq_candidates"][0]["answer_support_span_ids"] = self.ids[:2]
        result = validate_direct_mcq_batch(
            batch, source_view=self.view, root_id="root-test")
        self.assertIn("item_01:parent_only_forbidden", result["reasons"])

    def test_unknown_evidence_duplicate_question_and_reserved_field_reject(self) -> None:
        batch = self._batch()
        batch["mcq_candidates"][0]["answer_support_span_ids"] = [
            self.ids[2], "missing#S0001"]
        batch["mcq_candidates"][1]["question"] = " question 1? "
        batch["mcq_candidates"][2]["reference_answer_letter"] = "A"
        result = validate_direct_mcq_batch(
            batch, source_view=self.view, root_id="root-test")
        self.assertEqual("direct_mcq_batch_contract_reject", result["decision"])
        joined = "|".join(result["reasons"])
        self.assertIn("bad_span_id", joined)
        self.assertIn("duplicate_question", joined)
        self.assertIn("item_fields_invalid", joined)

    def test_duplicate_options_reject(self) -> None:
        batch = self._batch()
        batch["mcq_candidates"][0]["distractors"][0] = "Correct 1"
        result = validate_direct_mcq_batch(
            batch, source_view=self.view, root_id="root-test")
        self.assertTrue(any("bad_unlabeled_options" in reason
                            for reason in result["reasons"]))

    def test_answer_letters_are_deterministic_and_varied(self) -> None:
        first = _answer_letters("root-test")
        self.assertEqual(first, _answer_letters("root-test"))
        self.assertEqual(MCQ_COUNT, len(first))
        self.assertEqual(MCQ_COUNT, len(set(first)))

    def test_validator_materializes_without_model_answer_letters(self) -> None:
        result = validate_direct_mcq_batch(
            self._batch(), source_view=self.view, root_id="root-test")
        for row in result["items"]:
            qa = row["materialized_qa"]
            self.assertEqual(row["target_letter"], qa["reference_answer_letter"])
            self.assertEqual("spell_style_minimal_v1", qa["admission_mode"])
            self.assertGreaterEqual(row["evidence_scope"]["child_support_count"], 1)


if __name__ == "__main__":
    unittest.main()
