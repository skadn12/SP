"""Focused checks for source-bound citation formatting repair."""

import unittest

from citation_normalization import normalize_event, normalize_ref


class CitationNormalizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.valid_refs = {"R02:S01:T006", "R03:S01:T001"}

    def test_only_single_bracketed_existing_refs_change(self) -> None:
        self.assertEqual(
            normalize_ref("[R02:S01:T006]", self.valid_refs),
            ("R02:S01:T006", True),
        )
        for value in (
            "[R02:S01:T999]", "[[R02:S01:T006]]", "[ R02:S01:T006 ]",
            "R02:S01:T006", "[R02:S01:T006", None,
        ):
            with self.subTest(value=value):
                self.assertEqual(normalize_ref(value, self.valid_refs), (value, False))

    def test_event_preserves_raw_data_and_records_exact_changes(self) -> None:
        original = {
            "proposal_ref": "[R02:S01:T006]",
            "acceptance_ref": "[R02:S01:T999]",
            "plan_refs": ["[R02:S01:T006]", "[[R03:S01:T001]]"],
            "outcome_refs": ["[R03:S01:T001]"],
            "proposer": "David",
        }
        normalized, changes = normalize_event(original, self.valid_refs, 3)
        self.assertEqual(original["proposal_ref"], "[R02:S01:T006]")
        self.assertEqual(original["plan_refs"][0], "[R02:S01:T006]")
        self.assertEqual(normalized["acceptance_ref"], "[R02:S01:T999]")
        self.assertEqual(normalized["plan_refs"][1], "[[R03:S01:T001]]")
        self.assertEqual(normalized["proposer"], "David")
        self.assertEqual(changes, [
            {"event_index": 3, "field": "proposal_ref", "original": "[R02:S01:T006]", "normalized": "R02:S01:T006"},
            {"event_index": 3, "field": "plan_refs[0]", "original": "[R02:S01:T006]", "normalized": "R02:S01:T006"},
            {"event_index": 3, "field": "outcome_refs[0]", "original": "[R03:S01:T001]", "normalized": "R03:S01:T001"},
        ])


if __name__ == "__main__":
    unittest.main()
