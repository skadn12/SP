"""Zero-network token guard tests using explicit synthetic context boundaries."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .token_preflight import GuardrailStop, TokenPreflight, measure_request


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert messages[0]['role'] == 'user'
        assert add_generation_prompt is True and enable_thinking is False and tokenize is False
        return '<user>' + messages[0]['content'] + '</user><assistant><think></think>'

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text)


class TokenPreflightTests(unittest.TestCase):
    def measure(self, limit, **kwargs):
        return measure_request(CharacterTokenizer(), prompt='abc', max_output_tokens=4096,
            request_id='root-intent-miner', context_limit=limit, supported_max_output=8192, **kwargs)

    def test_wrapper_output_and_reserve_are_all_required_at_boundary(self):
        base = self.measure(20000)
        total = base['rendered_message_tokens'] + 4096 + 512
        self.assertEqual(base['required_context_tokens'], total)
        self.assertGreater(base['rendered_message_tokens'], 3)
        self.assertEqual(self.measure(total)['status'], 'PASS')
        self.assertEqual(self.measure(total - 1)['status'], 'BLOCKED')

    def test_component_overlap_is_diagnostic_and_not_added_to_total(self):
        plain = self.measure(20000)
        duplicated = self.measure(20000, components={name: 'abc' * 500 for name in
            ('transcript', 'feedback', 'binding', 'brief', 'template')})
        self.assertEqual(plain['required_context_tokens'], duplicated['required_context_tokens'])
        self.assertTrue(duplicated['components_overlap'])
        self.assertEqual(set(duplicated['component_tokens']),
            {'transcript', 'feedback', 'binding', 'brief', 'template'})

    def test_output_policy_cannot_grow_and_guardrail_is_not_swallowed(self):
        for output in (4097, 8193, 16384):
            with self.subTest(output=output), self.assertRaises(GuardrailStop):
                measure_request(CharacterTokenizer(), prompt='x', max_output_tokens=output,
                    request_id='id', context_limit=131072, supported_max_output=8192)
        self.assertFalse(issubclass(GuardrailStop, Exception))

    def make_preflight(self, run):
        obj = TokenPreflight.__new__(TokenPreflight)
        obj.run = run
        obj.report_path = run / 'audit/token_report.json'
        obj.identity = {'context_limit': 20000, 'verified_supported_max_output_tokens': 8192}
        obj.tokenizer = CharacterTokenizer()
        return obj

    def test_replay_and_current_usage_drift_are_guarded_before_future_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            obj = self.make_preflight(Path(temp))
            record = obj.check('abc', 4096, 'id')
            obj.check('abc', 4096, 'id')
            self.assertEqual(json.loads(obj.report_path.read_text())['request_count'], 1)
            with self.assertRaises(GuardrailStop):
                obj.check('changed', 4096, 'id')
            obj.verify_usage('id', {'prompt_tokens': record['rendered_message_tokens']})
            self.assertEqual(obj.verify_usage('id', None)['service_usage_count_status'], 'UNKNOWN')
            with self.assertRaises(GuardrailStop):
                obj.verify_usage('id', {'prompt_tokens': record['rendered_message_tokens'] + 1})
            self.assertEqual(json.loads(obj.report_path.read_text())['status'], 'BLOCKED')

    def test_exceeded_prompt_is_saved_without_truncation_or_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            obj = self.make_preflight(Path(temp))
            obj.identity['context_limit'] = 100
            with self.assertRaises(GuardrailStop):
                obj.check('abc', 4096, 'id')
            report = json.loads(obj.report_path.read_text())
            self.assertEqual(report['requests'][0]['prompt_content_tokens'], 3)
            self.assertEqual(report['request_count'], 1)
            self.assertEqual(report['status'], 'BLOCKED')


if __name__ == '__main__':
    unittest.main()
