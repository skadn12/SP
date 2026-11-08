"""Offline tests of strict option-shape conversion and frozen v5 bindings."""

from __future__ import annotations

import unittest

from normalize import ROOTS, normalize_options, normalize_root, v5_module


class NormalizeTests(unittest.TestCase):
    def test_ordered_explicit_prefixes_only(self):
        options = [f"{label}. statement {label} " for label in "ABCDEF"]
        normalized, status = normalize_options(options)
        self.assertEqual(status, "ORDERED_PREFIX_ARRAY_NORMALIZED")
        self.assertEqual(normalized, {label: f"statement {label} " for label in "ABCDEF"})
        for invalid in (
            list("ABCDEF"),
            ["B. second", "A. first"] + options[2:],
            ["A: first"] + options[1:],
            ["A. "] + options[1:],
            ["A.  extra space"] + options[1:],
            options[:5],
        ):
            self.assertIsNone(normalize_options(invalid)[0])

    def test_v5_responses_are_bound_and_revalidated(self):
        validator = v5_module()
        results = [normalize_root(root, validator) for root in ROOTS]
        self.assertEqual([row["mechanically_normalized_count"] for row in results], [0, 1, 1])
        self.assertEqual(results[0]["dispositions"][0]["reason"],
                         "OPTION_A_MISSING_OR_AMBIGUOUS_PREFIX")
        for row in results[1:]:
            self.assertFalse(row["content_audited"])
            self.assertEqual(len(row["original_candidates"]), 1)
            self.assertEqual(len(row["normalized_candidates"]), 1)
            self.assertEqual(len(row["derived_candidates"]), 1)
            mapping = row["relabel_mappings"][0]["source_to_derived_label"]
            original = row["normalized_candidates"][0]
            derived = row["derived_candidates"][0]
            for old, new in mapping.items():
                self.assertEqual(original["options"][old], derived["options"][new])
                self.assertEqual(original["option_evidence"][old],
                                 derived["option_evidence"][new])


if __name__ == "__main__":
    unittest.main()
