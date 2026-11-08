import copy
import unittest
from review_attestation_resume_v2 import validate_correction


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.original = {"read_attestation": {"full_source_read": True, "all_continuation_constraints_read": True},
                         "statements": {"S1": {"relation": "supports", "quote": "one"}}, "findings": []}
        self.corrected = copy.deepcopy(self.original)
        self.corrected["read_attestation"]["private_answers_read_before_save"] = False

    def test_exact_addition(self):
        validate_correction(self.original, self.corrected)

    def test_semantic_edit_rejected(self):
        self.corrected["statements"]["S1"]["relation"] = "contradicts"
        with self.assertRaisesRegex(ValueError, "ONLY_SOURCE_FIRST"):
            validate_correction(self.original, self.corrected)

    def test_quote_edit_rejected(self):
        self.corrected["statements"]["S1"]["quote"] = "two"
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_true_value_rejected(self):
        self.corrected["read_attestation"]["private_answers_read_before_save"] = True
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_zero_is_not_false(self):
        self.corrected["read_attestation"]["private_answers_read_before_save"] = 0
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_numeric_replacement_is_not_boolean(self):
        self.corrected["read_attestation"]["full_source_read"] = 1
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_existing_false_rejected(self):
        self.original["read_attestation"]["private_answers_read_before_save"] = False
        with self.assertRaisesRegex(ValueError, "ONLY_MISSING"):
            validate_correction(self.original, self.corrected)

    def test_other_attestation_change_rejected(self):
        self.corrected["read_attestation"]["full_source_read"] = False
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_extra_field_rejected(self):
        self.corrected["new"] = 1
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_missing_semantic_field_rejected(self):
        del self.corrected["findings"]
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)


if __name__ == "__main__":
    unittest.main()
