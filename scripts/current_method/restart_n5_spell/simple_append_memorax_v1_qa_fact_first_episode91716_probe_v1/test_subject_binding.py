"""Zero-API tests for constrained subject binding."""

from __future__ import annotations

import unittest

from ..simple_append_memorax_v1_qa_fact_first_probe_v1.inputs import load_object
from ..simple_append_memorax_v1_qa_fact_first_probe_v1.windows import compile_natural_windows
from .inputs import DEFAULT_EXPANDED
from .subject_binding_contract import (
    materialize_subject_bound_facts,
    validate_subject_binding_output,
)
from .subject_binding_prompts import build_subject_binding_prompt


class SubjectBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expanded = load_object(DEFAULT_EXPANDED)
        cls.compiled = compile_natural_windows(cls.expanded)
        cls.windows = cls.compiled["windows"]
        cls.window = cls.windows[5]  # C01 includes the known legacy attribution case.

    def validate(self, rows, window=None):
        return validate_subject_binding_output(
            {"facts": rows}, expanded_dialogue=self.expanded,
            window=window or self.window,
        )

    def test_speaker_binding_resolves_actual_speaker(self):
        result = self.validate([{
            "fact_text": "The speaker recalls the squirrel incident.",
            "source_positions": [59],
            "subject_binding": {"mode": "speaker_at_position", "position": 59},
        }])
        self.assertEqual(1, result["mechanically_valid_count"])
        self.assertEqual("Neighbors B", result["valid_facts"][0]["subject"])
        self.assertEqual("Neighbors B", result["valid_facts"][0]["asserting_speaker"])

    def test_known_entity_binding_supports_other_person(self):
        result = self.validate([{
            "fact_text": "Neighbors A remembers the squirrel incident.",
            "source_positions": [59],
            "subject_binding": {"mode": "known_entity", "entity_ref": "Neighbors A"},
        }])
        self.assertEqual(1, result["mechanically_valid_count"])
        self.assertEqual("Neighbors A", result["valid_facts"][0]["subject"])
        self.assertEqual("Neighbors B", result["valid_facts"][0]["asserting_speaker"])

    def test_named_entity_conflict_is_quarantined(self):
        result = self.validate([{
            "fact_text": "Neighbors A remembers the squirrel incident.",
            "source_positions": [59],
            "subject_binding": {"mode": "speaker_at_position", "position": 59},
        }])
        self.assertIn(
            "named_entity_requires_known_entity_binding",
            result["items"][0]["mechanical_errors"],
        )

    def test_legacy_free_subject_is_rejected(self):
        result = self.validate([{
            "fact_text": "Neighbors A remembers the squirrel incident.",
            "source_positions": [59],
            "subject": "Neighbors A",
        }])
        self.assertEqual(0, result["mechanically_valid_count"])
        self.assertIn("fact_candidate_schema", result["items"][0]["mechanical_errors"])
        self.assertIn("free_subject_forbidden", result["items"][0]["mechanical_errors"])

    def test_binding_position_must_be_evidence(self):
        result = self.validate([{
            "fact_text": "The speaker recalls the squirrel incident.",
            "source_positions": [59],
            "subject_binding": {"mode": "speaker_at_position", "position": 60},
        }])
        self.assertIn("speaker_binding_position_not_in_evidence", result["items"][0]["mechanical_errors"])

    def test_unknown_entity_is_rejected(self):
        result = self.validate([{
            "fact_text": "An unknown person did something.",
            "source_positions": [59],
            "subject_binding": {"mode": "known_entity", "entity_ref": "Unknown"},
        }])
        self.assertIn("entity_binding_entity_ref_unknown", result["items"][0]["mechanical_errors"])

    def test_materialization_preserves_program_resolved_binding(self):
        validation = self.validate([{
            "fact_text": "Neighbors A remembers the squirrel incident.",
            "source_positions": [59],
            "subject_binding": {"mode": "known_entity", "entity_ref": "Neighbors A"},
        }])
        rows = materialize_subject_bound_facts(
            [validation], windows_by_id={x["window_id"]: x for x in self.windows},
        )
        self.assertEqual("F0001", rows[0]["fact_id"])
        self.assertEqual("Neighbors A", rows[0]["subject"])
        self.assertEqual("Neighbors B", rows[0]["asserting_speaker"])
        self.assertEqual("known_entity", rows[0]["binding_mode"])

    def test_prompt_forbids_free_subject(self):
        prompt = build_subject_binding_prompt(
            self.window, entity_refs=["Neighbors A", "Neighbors B", "Elena"],
        )
        self.assertEqual(
            ["fact_text", "source_positions", "subject_binding"],
            prompt["output_fields"],
        )
        self.assertIn("Do not output a free subject field", prompt["prompt"])
        self.assertIn("known_entity", prompt["prompt"])


if __name__ == "__main__":
    unittest.main()
