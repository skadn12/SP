"""Zero-API tests for fact extraction, materialization, and bundling."""

from __future__ import annotations

import copy
import unittest

from .fact_bundler import MAX_BUNDLES_EPISODE, bundle_facts, validate_bundle
from .fact_contract import materialize_facts, validate_fact_extractor_output
from .fact_prompts import build_fact_prompt
from .inputs import DEFAULT_EXPANDED, load_object
from .windows import compile_natural_windows


def fact(positions, text="A planned a concrete action.", subject="A"):
    return {"fact_text": text, "source_positions": list(positions), "subject": subject}


class FactFirstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expanded = load_object(DEFAULT_EXPANDED)
        cls.compiled = compile_natural_windows(cls.expanded)
        cls.windows = cls.compiled["windows"]
        cls.window = cls.windows[0]

    def validate(self, facts, window=None):
        return validate_fact_extractor_output(
            {"facts": facts}, expanded_dialogue=self.expanded,
            window=window or self.window,
        )

    def test_natural_boundaries_and_global_positions(self):
        self.assertTrue(self.compiled["valid"])
        self.assertEqual(5, self.compiled["parent_session_count"])
        self.assertEqual(3, self.compiled["child_scene_count"])
        self.assertEqual(110, self.compiled["assigned_turn_count"])
        self.assertEqual("P01", self.windows[0]["window_id"])
        self.assertEqual("C01", self.windows[5]["window_id"])
        self.assertEqual(63, self.windows[5]["global_position_start"])

    def test_one_and_three_position_facts_valid(self):
        self.assertEqual(1, self.validate([fact([1])])["mechanically_valid_count"])
        self.assertEqual(1, self.validate([fact([1, 2, 3])])["mechanically_valid_count"])

    def test_zero_and_four_positions_rejected(self):
        zero = self.validate([fact([])])["items"][0]
        four = self.validate([fact([1, 2, 3, 4])])["items"][0]
        self.assertIn("source_positions_below_minimum", zero["mechanical_errors"])
        self.assertIn("source_positions_above_maximum", four["mechanical_errors"])

    def test_duplicate_and_outside_positions_rejected(self):
        duplicate = self.validate([fact([1, 1])])["items"][0]
        outside = self.validate([fact([1, 9])])["items"][0]
        self.assertIn("source_positions_duplicate", duplicate["mechanical_errors"])
        self.assertIn("source_position_outside_window", outside["mechanical_errors"])

    def test_empty_text_and_subject_rejected(self):
        empty_text = self.validate([fact([1], text="")])["items"][0]
        empty_subject = self.validate([fact([1], subject="")])["items"][0]
        self.assertIn("fact_text_empty", empty_text["mechanical_errors"])
        self.assertIn("subject_empty", empty_subject["mechanical_errors"])

    def test_empty_fact_list_valid(self):
        self.assertTrue(self.validate([])["empty_facts_valid"])

    def test_more_than_eight_facts_rejected(self):
        result = self.validate([fact([1]) for _ in range(9)])
        self.assertIn("fact_capacity_exceeded", result["envelope_errors"])
        self.assertEqual(0, result["mechanically_valid_count"])

    def test_materialization_is_stable_and_has_metadata(self):
        validation = self.validate([fact([1], "A planned a concrete action.")])
        rows = materialize_facts(
            [validation], windows_by_id={self.window["window_id"]: self.window},
        )
        again = materialize_facts(
            [validation], windows_by_id={self.window["window_id"]: self.window},
        )
        self.assertEqual(rows, again)
        self.assertEqual("F0001", rows[0]["fact_id"])
        self.assertEqual("cc-train-episode-146700:P01:T001", rows[0]["source_turn_ids"][0])
        self.assertEqual("P01", rows[0]["window_id"])
        self.assertEqual("parent_session", rows[0]["window_kind"])

    def test_prompt_has_only_fact_fields_and_no_reasoning_taxonomy(self):
        prompt = build_fact_prompt(self.window)
        self.assertEqual(["fact_text", "source_positions", "subject"], prompt["output_fields"])
        self.assertNotIn("structure_type", prompt["prompt"])
        self.assertIn("Do not output IDs", prompt["prompt"])
        self.assertIn("questions, answers, options, or distractors", prompt["prompt"])

    def test_bundler_same_subject_different_windows(self):
        facts = [
            {"fact_id": "F0001", "fact_text": "x", "subject": "A", "window_id": "P01", "window_order": 1},
            {"fact_id": "F0002", "fact_text": "y", "subject": "A", "window_id": "P03", "window_order": 3},
        ]
        result = bundle_facts(facts)
        self.assertEqual(1, result["bundle_count"])
        self.assertEqual(["F0001", "F0002"], result["bundles"][0]["fact_ids"])

    def test_bundler_different_subjects_do_not_combine(self):
        facts = [
            {"fact_id": "F0001", "fact_text": "x", "subject": "A", "window_id": "P01", "window_order": 1},
            {"fact_id": "F0002", "fact_text": "y", "subject": "B", "window_id": "P03", "window_order": 3},
        ]
        self.assertEqual(0, bundle_facts(facts)["bundle_count"])

    def test_bundler_same_window_does_not_combine(self):
        facts = [
            {"fact_id": "F0001", "fact_text": "x", "subject": "A", "window_id": "P01", "window_order": 1},
            {"fact_id": "F0002", "fact_text": "y", "subject": "A", "window_id": "P01", "window_order": 1},
        ]
        self.assertEqual(0, bundle_facts(facts)["bundle_count"])

    def test_bundler_size_and_window_limits(self):
        facts = [
            {"fact_id": f"F{i:04d}", "fact_text": str(i), "subject": "A",
             "window_id": f"P{i:02d}", "window_order": i}
            for i in range(1, 13)
        ]
        result = bundle_facts(facts)
        self.assertLessEqual(result["bundle_count"], MAX_BUNDLES_EPISODE)
        self.assertTrue(all(2 <= x["bundle_size"] <= 4 for x in result["bundles"]))
        self.assertTrue(all(len(x["window_ids"]) >= 2 for x in result["bundles"]))

    def test_bundler_does_not_truncate_late_child_facts(self):
        facts = [
            {"fact_id": f"F{i:04d}", "fact_text": str(i), "subject": "A",
             "window_id": "P01" if i <= 12 else "C03", "window_order": 1 if i <= 12 else 8,
             "window_kind": "parent_session" if i <= 12 else "child_scene"}
            for i in range(1, 15)
        ]
        result = bundle_facts(facts)
        selected_ids = {fact_id for row in result["bundles"] for fact_id in row["fact_ids"]}
        self.assertEqual(14, result["subject_stats"][0]["considered_fact_count"])
        self.assertIn("F0013", selected_ids)
        self.assertIn("F0014", selected_ids)
        self.assertGreater(result["materialized_class_counts"].get("parent_child", 0), 0)

    def test_bundle_validator_rejects_more_than_four_and_three_per_window(self):
        too_many = [
            {"fact_id": f"F{i:04d}", "subject": "A", "window_id": f"P{i:02d}"}
            for i in range(1, 6)
        ]
        self.assertIn("bundle_size_above_maximum", validate_bundle(too_many))
        crowded = [
            {"fact_id": "F0001", "subject": "A", "window_id": "P01"},
            {"fact_id": "F0002", "subject": "A", "window_id": "P01"},
            {"fact_id": "F0003", "subject": "A", "window_id": "P01"},
            {"fact_id": "F0004", "subject": "A", "window_id": "C01"},
        ]
        self.assertIn("bundle_window_fact_limit_exceeded", validate_bundle(crowded))

    def test_round_robin_covers_subjects_and_window_pairs(self):
        facts = []
        counter = 1
        for subject in ("A", "B"):
            for window_id, order, kind in (
                ("P01", 1, "parent_session"),
                ("P05", 5, "parent_session"),
                ("C01", 6, "child_scene"),
                ("C02", 7, "child_scene"),
                ("C03", 8, "child_scene"),
            ):
                facts.append({
                    "fact_id": f"F{counter:04d}", "fact_text": str(counter),
                    "subject": subject, "window_id": window_id,
                    "window_order": order, "window_kind": kind,
                })
                counter += 1
        result = bundle_facts(facts)
        self.assertEqual({"a", "b"}, {
            row["subject"] for row in result["subject_stats"] if row["emitted_bundle_count"]
        })
        pairs = {tuple(row["coverage_window_pair"]) for row in result["bundles"]}
        self.assertGreater(len(pairs), 1)
        self.assertGreater(result["materialized_class_counts"].get("parent_child", 0), 0)

    def test_episode_hard_max_is_sixty(self):
        facts = [
            {"fact_id": f"F{i:04d}", "fact_text": str(i), "subject": "A",
             "window_id": f"P{((i - 1) % 5) + 1:02d}" if i <= 20 else f"C{((i - 21) % 3) + 1:02d}",
             "window_order": ((i - 1) % 5) + 1 if i <= 20 else ((i - 21) % 3) + 6,
             "window_kind": "parent_session" if i <= 20 else "child_scene"}
            for i in range(1, 31)
        ]
        result = bundle_facts(facts)
        self.assertEqual(60, result["bundle_count"])
        self.assertEqual(60, result["limits"]["max_bundles_episode"])

    def test_bundler_is_deterministic(self):
        facts = [
            {"fact_id": "F0002", "fact_text": "y", "subject": "A", "window_id": "P03", "window_order": 3},
            {"fact_id": "F0001", "fact_text": "x", "subject": "A", "window_id": "P01", "window_order": 1},
            {"fact_id": "F0003", "fact_text": "z", "subject": "A", "window_id": "C01", "window_order": 6},
        ]
        self.assertEqual(bundle_facts(facts), bundle_facts(list(reversed(facts))))


if __name__ == "__main__":
    unittest.main()
