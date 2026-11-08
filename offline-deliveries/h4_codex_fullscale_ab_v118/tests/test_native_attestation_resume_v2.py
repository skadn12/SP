import copy
import importlib.util
from pathlib import Path
import unittest

P = Path(__file__).resolve().parents[1] / 'tools/review_scene_native_attestation_resume_v2.py'
spec = importlib.util.spec_from_file_location('native_resume_v2', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class CorrectionBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.original = {'read_attestation': {'full_source_read': True, 'all_continuation_constraints_read': True}, 'decision': 'READY', 'checks': {'time': {'status':'PASS','reason':'Future inquiry'}}}
        self.corrected = copy.deepcopy(self.original)
        self.corrected['read_attestation']['native_event_full_read'] = True

    def test_missing_field_addition_allowed(self):
        m.validate_correction(self.original, self.corrected)

    def test_explicit_false_cannot_be_changed_to_true(self):
        self.original['read_attestation']['native_event_full_read'] = False
        with self.assertRaisesRegex(ValueError, 'ONLY_MISSING_NATIVE_FIELD_ALLOWED'):
            m.validate_correction(self.original, self.corrected)

    def test_semantic_change_rejected(self):
        self.corrected['decision'] = 'HOLD'
        with self.assertRaisesRegex(ValueError, 'ONLY_NATIVE_ATTESTATION_ADDITION_ALLOWED'):
            m.validate_correction(self.original, self.corrected)

    def test_reason_change_rejected(self):
        self.corrected['checks']['time']['reason'] = 'Different semantic account'
        with self.assertRaisesRegex(ValueError, 'ONLY_NATIVE_ATTESTATION_ADDITION_ALLOWED'):
            m.validate_correction(self.original, self.corrected)

if __name__ == '__main__':
    unittest.main()
