import unittest

from check import chunks, evidence, resolve, turns


class CitationChecks(unittest.TestCase):
    def setUp(self):
        self.rows = turns({"first_session_dialogue": ["alpha", "beta"],
                           "first_session_speakers": ["a", "b"],
                           "second_session_dialogue": ["gamma", "delta"],
                           "second_session_speakers": ["a", "b"]})

    def test_actual_reference_forms(self):
        self.assertEqual(set(resolve("S01:T001", self.rows)), {1})
        self.assertEqual(set(resolve("S01:T1", self.rows)), {1})
        self.assertEqual(set(resolve("S01:T001-T002", self.rows)), {1, 2})
        self.assertEqual(set(resolve("second_session_dialogue:T003", self.rows)), {3})
        self.assertEqual(set(resolve("second_session:T1", self.rows)), {3})
        self.assertFalse(resolve("S01:T001-T003", self.rows))

    def test_origin_and_unique_quote(self):
        good = {"turn_ref": "S02:T003", "quote": "gamma", "origin": "SOURCE.json"}
        self.assertEqual(evidence(good, self.rows)["issues"], [])
        self.assertTrue(evidence({**good, "origin": "elsewhere"}, self.rows)["issues"])
        self.assertTrue(evidence({**good, "quote": "missing"}, self.rows)["issues"])
        duplicated = turns({"first_session_dialogue": ["same", "same"],
                            "first_session_speakers": ["a", "b"]})
        self.assertTrue(evidence({"turn_ref": "S01:T001-T002", "quote": "same",
                                  "origin": "SOURCE.json"}, duplicated)["issues"])

    def test_chunks(self):
        self.assertEqual(sum(map(len, chunks(self.rows))), len(self.rows))
        huge = [{**self.rows[0], "text": "x" * 12001}]
        with self.assertRaises(ValueError):
            chunks(huge)


if __name__ == "__main__":
    unittest.main()
