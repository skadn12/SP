"""Zero-API tests for participant-connected fact bundling."""

from __future__ import annotations

import unittest

from .inputs import DEFAULT_EXPANDED, load_object
from .participant_bundler import (
    bundle_participant_connected_facts,
    known_dialogue_participants,
    materialize_fact_participants,
    validate_participant_bundle,
)


def row(fact_id, subject, window_id, participants=None, order=None):
    return {
        "fact_id": fact_id,
        "fact_text": fact_id,
        "subject": subject,
        "window_id": window_id,
        "window_order": order if order is not None else (1 if window_id.startswith("P") else 6),
        "window_kind": "parent_session" if window_id.startswith("P") else "child_scene",
        **({"participants": participants} if participants is not None else {}),
    }


class ParticipantRebundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expanded = load_object(DEFAULT_EXPANDED)

    def test_known_participants_come_from_dialogue_speakers(self):
        self.assertEqual(
            ["Co-workers A", "Co-workers B", "Specialist D", "Boss"],
            known_dialogue_participants(self.expanded),
        )

    def test_subject_is_forced_and_explicit_names_are_added(self):
        facts = [{
            "fact_id": "F1", "subject": "Co-workers A",
            "fact_text": "Boss agrees to help Co-workers A.",
            "source_positions": [95],
        }]
        result = materialize_fact_participants(facts, expanded=self.expanded)
        self.assertEqual({"Co-workers A", "Boss"}, set(result[0]["participants"]))

    def test_pronouns_are_not_resolved(self):
        facts = [{
            "fact_id": "F1", "subject": "Co-workers A",
            "fact_text": "They helped them.", "source_positions": [1],
        }]
        result = materialize_fact_participants(facts, expanded=self.expanded)
        self.assertEqual(["Co-workers A"], result[0]["participants"])

    def test_two_fact_explicit_bridge_is_valid(self):
        facts = [
            row("F1", "Co-workers A", "P01", ["Co-workers A"]),
            row("F2", "Boss", "C03", ["Co-workers A", "Boss"]),
        ]
        self.assertEqual([], validate_participant_bundle(facts))
        result = bundle_participant_connected_facts(facts)
        self.assertEqual(1, result["bundle_count"])
        self.assertEqual("cross_subject", result["bundles"][0]["connection_pattern"])

    def test_transitive_bridge_is_valid(self):
        facts = [
            row("F1", "Co-workers A", "P01", ["Co-workers A"]),
            row("F2", "Specialist D", "P05", ["Co-workers A", "Specialist D"]),
            row("F3", "Specialist D", "C02", ["Specialist D"]),
        ]
        self.assertEqual([], validate_participant_bundle(facts))

    def test_disconnected_pair_is_rejected(self):
        facts = [
            row("F1", "Co-workers A", "P01", ["Co-workers A"]),
            row("F2", "Boss", "C03", ["Boss"]),
        ]
        self.assertIn("participant_graph_disconnected", validate_participant_bundle(facts))
        self.assertEqual(0, bundle_participant_connected_facts(facts)["bundle_count"])

    def test_disconnected_third_fact_is_rejected(self):
        facts = [
            row("F1", "Co-workers A", "P01", ["Co-workers A"]),
            row("F2", "Boss", "P05", ["Co-workers A", "Boss"]),
            row("F3", "Specialist D", "C02", ["Specialist D"]),
        ]
        self.assertIn("participant_graph_disconnected", validate_participant_bundle(facts))

    def test_same_subject_remains_valid(self):
        facts = [
            row("F1", "Co-workers A", "P01", ["Co-workers A"]),
            row("F2", "Co-workers A", "C01", ["Co-workers A"]),
        ]
        result = bundle_participant_connected_facts(facts)
        self.assertEqual(1, result["bundle_count"])
        self.assertEqual("same_subject", result["bundles"][0]["connection_pattern"])

    def test_window_and_size_limits_remain(self):
        same_window = [
            row("F1", "A", "P01", ["A"]), row("F2", "B", "P01", ["A", "B"]),
        ]
        self.assertIn("bundle_requires_two_windows", validate_participant_bundle(same_window))
        four = [
            row("F1", "A", "P01", ["A"]),
            row("F2", "A", "P01", ["A"]),
            row("F3", "B", "C01", ["A", "B"]),
            row("F4", "B", "C01", ["B"]),
        ]
        self.assertEqual([], validate_participant_bundle(four))
        five = four + [row("F5", "B", "C02", ["B"])]
        self.assertIn("bundle_size_above_maximum", validate_participant_bundle(five))

    def test_output_is_deterministic_and_keeps_late_window(self):
        facts = [
            row("F1", "A", "P01", ["A"]),
            row("F2", "B", "P01", ["A", "B"]),
            row("F3", "B", "C01", ["B"]),
            row("F4", "Boss", "C03", ["B", "Boss"]),
            row("F5", "Boss", "C03", ["Boss"]),
        ]
        first = bundle_participant_connected_facts(facts)
        second = bundle_participant_connected_facts(list(reversed(facts)))
        self.assertEqual(first, second)
        self.assertTrue(any("C03" in bundle["window_ids"] for bundle in first["bundles"]))


if __name__ == "__main__":
    unittest.main()
