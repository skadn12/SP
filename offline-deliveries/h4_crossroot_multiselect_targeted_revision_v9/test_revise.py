"""Offline checks for the exact three-field revision contract."""

from __future__ import annotations

from copy import deepcopy
import unittest

from revise import TARGETS, check_revision, dependency, prompt_for, source_inputs


class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.runtime = dependency()

    def test_only_three_target_strings_may_change(self):
        for root_id, target in TARGETS.items():
            _, _, source, original = source_inputs(root_id)
            label = target["label"]
            revised = deepcopy(original)
            revised["options"][label] += " Revised."
            revised["option_evidence"][label]["reason"] += " Revised."
            revised["answer_reason"] += " Revised."
            self.assertEqual(check_revision(revised, original, root_id,
                                            source["turns"], self.runtime.check_candidate), [])
            revised["gold"] = list(reversed(original["gold"]))
            self.assertTrue(any("changed_fields_not_exactly_allowed" in error
                                for error in check_revision(revised, original, root_id,
                                                            source["turns"], self.runtime.check_candidate)))

    def test_missing_edit_or_changed_quote_is_rejected(self):
        for root_id, target in TARGETS.items():
            _, _, source, original = source_inputs(root_id)
            label = target["label"]
            revised = deepcopy(original)
            revised["options"][label] += " Revised."
            revised["option_evidence"][label]["reason"] += " Revised."
            errors = check_revision(revised, original, root_id, source["turns"],
                                    self.runtime.check_candidate)
            self.assertTrue(any("changed_fields_not_exactly_allowed" in error
                                for error in errors))
            revised["answer_reason"] += " Revised."
            revised["option_evidence"][label]["quote"] += " Changed."
            errors = check_revision(revised, original, root_id, source["turns"],
                                    self.runtime.check_candidate)
            self.assertTrue(any("changed_fields_not_exactly_allowed" in error
                                for error in errors))
            self.assertIn(f"{label}:quote_not_verbatim", errors)

    def test_prompt_contains_original_and_no_suggested_rewrite(self):
        for root_id in TARGETS:
            _, _, source, original = source_inputs(root_id)
            prompt = prompt_for(root_id, original, source)
            self.assertIn(original["options"][TARGETS[root_id]["label"]], prompt)
            self.assertIn("COMPLETE R02 TRANSCRIPT", prompt)
            self.assertIn('"revised_question":null', prompt)


if __name__ == "__main__":
    unittest.main()
