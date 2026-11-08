import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from review_public_reread_resume_v1 import validate_correction


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.original = {'read_attestation': {'full_source_read': True,
                                             'private_answers_read_before_save': False},
                         'questions': {'q': {'scope_valid': True, 'reason': 'Exact source statement.'}},
                         'findings': []}
        self.corrected = copy.deepcopy(self.original)
        self.corrected['read_attestation']['final_public_questions_actually_reread'] = True

    def test_exact_addition(self):
        validate_correction(self.original, self.corrected)

    def test_false_to_true_rejected(self):
        self.original['read_attestation']['final_public_questions_actually_reread'] = False
        with self.assertRaisesRegex(ValueError, 'ONLY_MISSING'):
            validate_correction(self.original, self.corrected)

    def test_semantic_edit_rejected(self):
        self.corrected['questions']['q']['scope_valid'] = False
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_reason_edit_rejected(self):
        self.corrected['questions']['q']['reason'] = 'Different judgment.'
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_private_read_change_rejected(self):
        self.original['read_attestation']['private_answers_read_before_save'] = True
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_numeric_true_rejected(self):
        self.corrected['read_attestation']['final_public_questions_actually_reread'] = 1
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_numeric_other_boolean_rejected(self):
        self.corrected['read_attestation']['private_answers_read_before_save'] = 0
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_extra_field_rejected(self):
        self.corrected['extra'] = True
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)


if __name__ == '__main__':
    unittest.main()
