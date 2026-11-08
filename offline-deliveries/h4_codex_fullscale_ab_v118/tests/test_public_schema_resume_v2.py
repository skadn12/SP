import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from review_public_schema_resume_v2 import validate_correction


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.original = {'read_attestation': {'full_source_read': True,
                                             'private_answers_read_before_save': False},
                         'questions': [{'candidate_id': 'q', 'reason': 'Distinct independent options.',
                                        'options': {label: {'relation': 'supports', 'quote': 'Evidence.'} for label in 'ABCDEF'}}],
                         'findings': []}
        self.corrected = copy.deepcopy(self.original)
        self.corrected['read_attestation']['final_public_questions_actually_reread'] = True
        self.corrected['questions'] = {q['candidate_id']: q for q in self.corrected['questions']}
        self.corrected['questions']['q'].update(fact_distinct=True, no_entangled_options=True)

    def test_exact_rekey_and_additions(self):
        validate_correction(self.original, self.corrected)

    def test_relation_change_rejected(self):
        self.corrected['questions']['q']['options']['A']['relation'] = 'contradicts'
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_quote_change_rejected(self):
        self.corrected['questions']['q']['options']['A']['quote'] = 'Different.'
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_reread_false_to_true_rejected(self):
        self.original['read_attestation']['final_public_questions_actually_reread'] = False
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_existing_entanglement_false_cannot_change(self):
        self.original['questions'][0]['no_entangled_options'] = False
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_existing_distinct_false_cannot_change(self):
        self.original['questions'][0]['fact_distinct'] = False
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_duplicate_candidate_ids_rejected(self):
        self.original['questions'].append(copy.deepcopy(self.original['questions'][0]))
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_numeric_true_rejected(self):
        self.corrected['questions']['q']['fact_distinct'] = 1
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)

    def test_reason_change_rejected(self):
        self.corrected['questions']['q']['reason'] = 'Edited.'
        with self.assertRaises(ValueError):
            validate_correction(self.original, self.corrected)


if __name__ == '__main__':
    unittest.main()
