"""Offline tests for position-only semantic Writer materialization."""

from __future__ import annotations

import unittest

from ..simple_append_v1.qa import resolve_evidence_turn_refs
from .position_contract import materialize_position_writer_output
from .position_fixtures import (
    cross_story_distractor_source, duplicate_gold, envelope,
    invented_distractor_without_source, missing_distractor_provenance,
    out_of_range_distractor_source, temporal,
)
from .position_inputs import DEFAULT_EXPANDED, load_object


class PositionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expanded = load_object(DEFAULT_EXPANDED)

    def materialize(self, value):
        return materialize_position_writer_output(
            value, expanded_dialogue=self.expanded)

    def test_position_maps_to_canonical_turn_and_alias(self) -> None:
        result = self.materialize(envelope(1, [1, 2, 3]))
        row = result["materialized_candidates"][0]
        self.assertEqual([
            "cc-train-episode-146700:P01:T001",
            "cc-train-episode-146700:P01:T002",
            "cc-train-episode-146700:P01:T003",
        ], row["canonical_turn_ids"])
        self.assertEqual(["D001", "D002", "D003"], row["evidence_turn_ids"])
        self.assertEqual(row["evidence_turn_ids"], row["supporting_turn_ids"])
        self.assertEqual("parent_only", row["source_scope"])

    def test_two_through_six_evidence_materialize(self) -> None:
        for count in range(2, 7):
            with self.subTest(count=count):
                result = self.materialize(envelope(1, list(range(1, count + 1))))
                self.assertEqual(1, result["structural_valid_count"])

    def test_over_eight_evidence_rejected(self) -> None:
        result = self.materialize(envelope(1, list(range(1, 10))))
        self.assertIn("evidence_over_hard_maximum", result["items"][0]["structural_errors"])

    def test_out_of_range_evidence_position_rejected(self) -> None:
        result = self.materialize(envelope(1, [1, 111]))
        self.assertIn("evidence_position_out_of_range", result["items"][0]["structural_errors"])

    def test_three_distractor_provenance_positions_are_required(self) -> None:
        result = self.materialize(envelope(1))
        item = result["items"][0]
        self.assertTrue(item["valid"])
        self.assertEqual({"A", "C", "D"}, set(item["distractor_sources"]))
        row = result["materialized_candidates"][0]
        self.assertEqual(
            ["cc-train-episode-146700:P01:T001"],
            row["distractor_source_turn_ids"]["A"],
        )

    def test_distractor_source_out_of_range_rejected(self) -> None:
        result = self.materialize(out_of_range_distractor_source())
        self.assertIn(
            "distractor_source_position_out_of_range:D",
            result["items"][0]["structural_errors"],
        )

    def test_missing_distractor_provenance_rejected(self) -> None:
        result = self.materialize(missing_distractor_provenance())
        self.assertIn("distractor_sources_keys", result["items"][0]["structural_errors"])

    def test_invented_distractor_without_source_rejected(self) -> None:
        result = self.materialize(invented_distractor_without_source())
        self.assertIn("distractor_source_schema:D", result["items"][0]["structural_errors"])

    def test_cross_story_source_is_audit_warning_not_semantic_judgment(self) -> None:
        result = self.materialize(cross_story_distractor_source())
        item = result["items"][0]
        self.assertTrue(item["valid"])
        self.assertIn("cross_story_arc_provenance:C", item["warnings"])

    def test_duplicate_gold_rejected(self) -> None:
        result = self.materialize(duplicate_gold())
        self.assertIn("gold_not_exactly_one_option", result["items"][0]["structural_errors"])

    def test_semantic_structure_is_not_a_mechanical_hard_gate(self) -> None:
        value = envelope(1)
        value["questions"][0]["structure_type"] = "information_revision"
        result = self.materialize(value)
        self.assertEqual(1, result["structural_valid_count"])

    def test_temporal_sequence_materializes(self) -> None:
        result = self.materialize(temporal())
        self.assertEqual(1, result["structural_valid_count"])

    def test_zero_one_three_twelve_batches(self) -> None:
        zero = self.materialize(envelope(0))
        self.assertTrue(zero["zero_question_batch_valid"])
        for count in (1, 3, 12):
            with self.subTest(count=count):
                result = self.materialize(envelope(count))
                self.assertEqual(count, result["structural_valid_count"])

    def test_final_mcq_is_solver_compatible(self) -> None:
        row = self.materialize(envelope(1))["normalized_batch"]["questions"][0]
        self.assertEqual(
            {"question_type", "question", "options", "correct_option", "evidence_turn_refs"},
            set(row),
        )
        resolution = resolve_evidence_turn_refs(
            row["evidence_turn_refs"], expanded_dialogue=self.expanded)
        self.assertFalse(resolution["invalid_evidence_refs"])
        self.assertEqual(3, len(resolution["resolved_turn_refs"]))

    def test_writer_mechanical_fields_are_rejected(self) -> None:
        value = envelope(1)
        value["questions"][0]["source_scope"] = "parent_only"
        result = self.materialize(value)
        self.assertIn("question_schema", result["items"][0]["structural_errors"])

    def test_structure_field_is_rejected(self) -> None:
        value = envelope(1)
        value["questions"][0]["structure"] = {"old_state": "x"}
        result = self.materialize(value)
        self.assertIn("question_schema", result["items"][0]["structural_errors"])


if __name__ == "__main__":
    unittest.main()
