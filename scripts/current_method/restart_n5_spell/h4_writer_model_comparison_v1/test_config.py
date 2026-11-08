"""The user's narrow service-documentation exception must not relax other guards."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .config import _strong_profile, read
from .constants import ARMS, STRONG_CONFIG
from .prepare import _capacity_sendable, _capacity_with_authorization


class ServiceExceptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.raw = read(STRONG_CONFIG)
        source = STRONG_CONFIG.parent / self.raw['user_exception_record']
        self.exception = read(source)
        self.raw['user_exception_record'] = str(self.folder/'exception.json')

    def profile(self):
        (self.folder/'exception.json').write_text(json.dumps(self.exception))
        (self.folder/'config.json').write_text(json.dumps(self.raw))
        return _strong_profile(self.folder/'config.json')

    def capacity(self, profile, output=8192):
        return _capacity_with_authorization(profile,[{'role':'user','content':'full unchanged input'}],output,'offline')

    def test_explicit_exception_retains_unknowns_without_asserting_service_support(self):
        profile = self.profile()
        self.assertFalse(profile['unresolved'])
        self.assertIn('parameter_support.max_tokens',profile['unknown_configuration'])
        self.assertEqual(profile['reasoning_policy']['verified_status'],'UNKNOWN')
        self.assertEqual(profile['requested_parameter_adoption']['max_tokens'],'UNKNOWN')
        self.assertFalse(profile['parameter_support']['supported'])
        result = self.capacity(profile)
        self.assertEqual(result['status'],'UNKNOWN')
        self.assertFalse(result['capacity_verified'])
        self.assertTrue(_capacity_sendable(result))

    def test_no_exception_keeps_unknown_service_blocked(self):
        self.raw.pop('user_exception_record')
        profile = self.profile()
        self.assertIn('service_limits.context_window_tokens',profile['unresolved'])
        self.assertFalse(_capacity_sendable(self.capacity(profile)))

    def test_exception_cannot_change_budget_model_endpoint_or_parameters(self):
        original = copy.deepcopy(self.raw)
        mutations = [('model','another-model'),('endpoint','https://other.invalid/chat/completions')]
        for key,value in mutations:
            with self.subTest(key=key):
                self.raw = copy.deepcopy(original)
                self.raw['api'][key] = value
                with self.assertRaisesRegex(ValueError,'ENDPOINT_OR_MODEL_MISMATCH'):
                    self.profile()
        self.raw = copy.deepcopy(original)
        self.raw['generation']['temperature'] = 0.2
        with self.assertRaisesRegex(ValueError,'EXACT_REQUEST_AND_DEFAULT_REASONING'):
            self.profile()
        self.raw = copy.deepcopy(original)
        self.exception['budgets']['maximum_http_attempts'] = 25
        with self.assertRaisesRegex(ValueError,'SCOPE_OR_POLICY_MISMATCH'):
            self.profile()

    def test_exception_does_not_waive_qwen_known_capacity_failure_or_different_output(self):
        profile = self.profile()
        self.assertFalse(_capacity_sendable(self.capacity(profile,4096)))
        profile['arm_id'] = ARMS[0]
        self.assertFalse(_capacity_sendable(self.capacity(profile)))
        profile['arm_id'] = ARMS[1]
        with patch('scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.prepare._capacity',
                   return_value={'status':'BLOCKED','arm_id':ARMS[1]}):
            self.assertFalse(_capacity_sendable(self.capacity(profile)))

    def test_exception_does_not_hide_unrelated_missing_api_configuration(self):
        self.raw['api']['timeout_seconds'] = None
        self.assertIn('api.timeout_seconds',self.profile()['unresolved'])


if __name__ == '__main__':
    unittest.main()
