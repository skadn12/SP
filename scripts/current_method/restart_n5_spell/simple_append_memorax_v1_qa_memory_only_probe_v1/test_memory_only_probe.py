from __future__ import annotations

import unittest

from .memory_only_contract import (
    QUESTION_TYPES,
    materialize_solver_rows,
    memory_source_refs,
    validate_writer_output,
)
from .memory_only_prompts import build_memory_only_writer_prompt


def overview() -> dict:
    return {
        "root_id": "root-1", "memory_overview_sha256": "a" * 64,
        "memory_is_qa_fact_source": False,
        "people": [{"canonical_key": "a", "display_key": "A",
                    "content": ["A planned the event."]}],
        "events": [{"canonical_key": "event", "display_key": "Event",
                    "content": ["A later completed the event."]}],
        "locations": [], "items": [], "themes": [],
    }


def question(ref: str) -> dict:
    return {"question_type": "state_change", "question": "What happened?",
            "options": {"A": "It was completed.", "B": "It was proposed.",
                        "C": "It was delayed.", "D": "It was cancelled."},
            "gold_text": "It was completed.", "memory_source_refs": [ref]}


class MemoryOnlyProbeTests(unittest.TestCase):
    def test_prompt_contains_memory_but_no_transcript_contract(self) -> None:
        prompt = build_memory_only_writer_prompt(overview())["prompt"]
        self.assertIn("A later completed the event.", prompt)
        self.assertIn("No transcript", prompt)
        self.assertNotIn("D001", prompt)

    def test_prompt_uses_validator_question_type_enum(self) -> None:
        prompt = build_memory_only_writer_prompt(overview())["prompt"]
        for value in sorted(QUESTION_TYPES):
            self.assertIn(value, prompt)

    def test_variable_batch_and_gold_binding(self) -> None:
        refs = memory_source_refs(overview())
        ref = next(iter(refs))
        for count in (0, 1, 3, 12):
            result = validate_writer_output(
                {"questions": [question(ref) for _ in range(count)]},
                source_refs=refs)
            self.assertEqual(count, result["valid_question_count"])
        result = validate_writer_output(
            {"questions": [question(ref) for _ in range(13)]},
            source_refs=refs)
        self.assertFalse(result["valid"])

    def test_unknown_source_and_ambiguous_gold_quarantine(self) -> None:
        refs = memory_source_refs(overview())
        invalid = question("unknown")
        result = validate_writer_output({"questions": [invalid]}, source_refs=refs)
        self.assertEqual(0, result["valid_question_count"])
        ambiguous = question(next(iter(refs)))
        ambiguous["options"]["B"] = ambiguous["options"]["A"]
        result = validate_writer_output({"questions": [ambiguous]}, source_refs=refs)
        self.assertEqual(0, result["valid_question_count"])

    def test_solver_shape_is_deferred_not_admitted(self) -> None:
        refs = memory_source_refs(overview())
        normalized = validate_writer_output(
            {"questions": [question(next(iter(refs)))]}, source_refs=refs)
        row = materialize_solver_rows(normalized, root_id="root-1")[0]
        self.assertEqual("A", row["correct_option"])
        self.assertTrue(row["memory_grounded"])
        self.assertFalse(row["transcript_verified"])
        self.assertEqual([], row["evidence_turn_refs"])


if __name__ == "__main__":
    unittest.main()
