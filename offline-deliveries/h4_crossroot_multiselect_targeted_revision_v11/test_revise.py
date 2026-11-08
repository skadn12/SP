"""Offline checks for the v11 target-only revision contract."""

from copy import deepcopy
import unittest

from revise import check_revision, dependency, prompt_for


class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v9, cls.runtime, cls.source, cls.original = dependency()

    def test_only_c_change_is_required(self):
        revised = deepcopy(self.original)
        revised["options"]["C"] = "Emily said she scheduled her first session for next Tuesday at 3 PM."
        self.assertEqual(check_revision(revised, self.original, self.source["turns"],
                                        self.runtime.check_candidate, self.v9), [])
        revised["options"]["E"] += " Altered."
        self.assertTrue(check_revision(revised, self.original, self.source["turns"],
                                       self.runtime.check_candidate, self.v9))

    def test_therapy_word_and_unchanged_c_are_rejected(self):
        self.assertTrue(check_revision(self.original, self.original, self.source["turns"],
                                       self.runtime.check_candidate, self.v9))
        revised = deepcopy(self.original)
        revised["options"]["C"] += " Therapy."
        self.assertIn("unsupported_therapy_claim_remains",
                      check_revision(revised, self.original, self.source["turns"],
                                     self.runtime.check_candidate, self.v9))

    def test_prompt_uses_frozen_v10_question(self):
        prompt = prompt_for(self.original, self.source)
        self.assertIn(self.original["options"]["E"], prompt)
        self.assertIn(self.original["options"]["C"], prompt)
        self.assertIn("COMPLETE R02 TRANSCRIPT", prompt)


if __name__ == "__main__":
    unittest.main()
