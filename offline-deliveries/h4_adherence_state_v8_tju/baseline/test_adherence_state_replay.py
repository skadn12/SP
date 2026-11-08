"""Offline checks of the frozen adherence replay and its two HTTP ceiling."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from ..core import atomic_json, atomic_text
from ..h4_writer_model_comparison_v1.transport import ProfileClient, build_payload
from . import adherence_state_replay as replay
from .development_runtime import DevelopmentRuntimeError
from .realizer_adherence import build_adherence_auditor_input, build_adherence_auditor_prompt
from .test_development_runtime import TemplateTokenizer
from .test_realizer_adherence import _fixture


class AdherenceStateReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source, self.run = self.directory / 'source', self.directory / 'run'
        self.tokenizer = TemplateTokenizer()
        self.profile = {
            'model': 'Qwen/Qwen3-14B', 'base_url': 'https://api.siliconflow.cn/v1',
            'endpoint': 'https://api.siliconflow.cn/v1/chat/completions',
            'timeout_seconds': 600, 'api_key_env': 'MOCK_UNUSED_CREDENTIAL',
            'request_parameters': {'temperature': 0.25, 'max_tokens': 4096,
                                   'response_format': {'type': 'json_object'}},
            'output_limit': 4096, 'http_retry_limit': 0, 'redirects': False,
            'ssl_fallback': False, 'parameter_fallback': False,
            'limits': {'context_window_tokens': 40960, 'max_output_tokens': 8192,
                       'reasoning_budget_tokens': 0, 'reasoning_in_output_budget': True,
                       'reserve_tokens': 1024},
            'token_accounting': {'chat_template_kwargs': {'add_generation_prompt': True,
                                                        'enable_thinking': False}},
        }
        self.plan, self.child, self.judgment = _fixture()
        self.judgment['scene_audits'][0]['participant_adherence'] = 'pass'
        self.opener = MagicMock()
        response = self.opener.open.return_value.__enter__.return_value
        response.getcode.return_value = 200
        self.set_response(self.judgment)
        self.make_source()

    def set_response(self, value, finish='stop', usage=True):
        body = {'model': 'returned-version', 'choices': [{'finish_reason': finish,
                'message': {'content': json.dumps(value)}}]}
        if usage:
            body['usage'] = {'prompt_tokens': 100, 'completion_tokens': 25, 'total_tokens': 125}
        self.opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(body).encode()

    def make_source(self):
        tokenizer = self.source / 'tokenizer'
        tokenizer.mkdir(parents=True)
        atomic_json(tokenizer / 'tokenizer_config.json', {})
        atomic_json(self.source / 'freeze_manifest.json', {
            'public_profile': self.profile, 'tokenizer_path': str(tokenizer),
            'config_path': str(self.source / 'do_not_read_config.json')})
        atomic_json(self.source / 'frozen/provider_profile.json', self.profile)
        auditor_input = build_adherence_auditor_input(plan=self.plan, child=self.child)
        prompt = build_adherence_auditor_prompt(auditor_input=auditor_input)['prompt']
        for case in replay.CASES:
            directory = self.source / 'cases' / case
            atomic_json(directory / 'planner/materialized_plan.json', self.plan)
            atomic_json(directory / 'child_dialogue.json', self.child)
            atomic_json(directory / 'realizer_adherence/input.json', auditor_input)
            atomic_text(directory / 'realizer_adherence/prompt.txt', prompt)
            atomic_json(self.source / 'execution/calls' / case / 'adherence/request.json',
                        {'endpoint': self.profile['endpoint'], 'payload': build_payload(self.profile, prompt)})

    def factory(self, profile, *, on_http_attempt):
        with patch('urllib.request.build_opener', return_value=self.opener):
            return ProfileClient(profile, credential='MOCK-NOT-A-REAL-KEY', on_http_attempt=on_http_attempt)

    def prepare(self):
        with (patch.object(replay, 'load_challenger_config', side_effect=AssertionError('secret read')),
              patch.object(replay, 'public_profile', side_effect=AssertionError('secret read')),
              patch.object(replay, 'ProfileClient', side_effect=AssertionError('HTTP client created'))):
            return replay.prepare(self.run, source_run=self.source, tokenizer=self.tokenizer)

    def execute(self):
        return replay.execute(self.run, http_budget=2, client_factory=self.factory, tokenizer=self.tokenizer)

    def test_prepare_never_reads_credentials_and_freezes_full_inputs(self):
        result = self.prepare()
        self.assertEqual(result['external_calls'], 0)
        self.assertEqual(result['case_order'], list(replay.CASES))
        self.assertEqual(result['maximum_output_tokens_total'], 8192)
        for case in replay.CASES:
            frozen = self.run / 'frozen' / case
            self.assertTrue(replay.read(frozen / 'identity.json')['auditor_input_bytes_identical'])
            self.assertEqual((frozen / 'child.json').read_bytes(),
                (self.source / 'cases' / case / 'child_dialogue.json').read_bytes())
        self.opener.open.assert_not_called()

    def test_two_calls_only_and_continuation_is_local(self):
        self.prepare()
        result = self.execute()
        self.assertEqual(result['status'], 'FINISHED')
        self.assertEqual(result['cost']['http_attempts'], 2)
        self.assertEqual(result['cost']['total_tokens'], 250)
        self.assertEqual(result['cost']['reasoning_tokens'], 'UNKNOWN')
        self.assertEqual(result['continuation_calls'], 0)
        self.assertEqual([o['eligible_continuation_scene_count'] for o in result['outcomes']], [1, 1])
        self.assertEqual(self.opener.open.call_count, 2)
        ledger = replay.read(self.run / 'execution/development_call_ledger.json')
        self.assertEqual([r['stage'] for r in ledger['records']], ['adherence', 'adherence'])
        self.assertEqual([r['root_id'] for r in ledger['records']], list(replay.CASES))
        for call, case in zip(self.opener.open.call_args_list, replay.CASES):
            self.assertEqual(json.loads(call.args[0].data), replay.read(self.run / 'frozen' / case / 'payload.json'))

    def test_timeout_keeps_used_attempt_and_stops_unsent_case(self):
        self.prepare()
        self.opener.open.side_effect = TimeoutError()
        result = self.execute()
        self.assertEqual(result['status'], 'STOPPED_TECHNICAL')
        self.assertEqual(result['cost']['http_attempts'], 1)
        self.assertEqual(result['cost']['total_tokens'], 'UNKNOWN')
        self.assertEqual(result['unsent_cases'], [replay.CASES[1]])
        self.assertEqual(self.opener.open.call_count, 1)
        with self.assertRaisesRegex(FileExistsError, 'no_automatic_resume'):
            self.execute()
        self.assertEqual(self.opener.open.call_count, 1)

    def test_nonstop_finish_blocks_remaining_case(self):
        self.prepare()
        self.set_response(self.judgment, finish='length')
        result = self.execute()
        self.assertEqual(result['status'], 'STOPPED_TECHNICAL')
        self.assertEqual(result['cost']['http_attempts'], 1)
        self.assertEqual(result['cost']['total_tokens'], 125)

    def test_presend_client_failure_leaves_both_cases_unsent(self):
        self.prepare()
        def failed_factory(profile, *, on_http_attempt):
            raise ValueError('mock presend failure')
        result = replay.execute(self.run, http_budget=2,
            client_factory=failed_factory, tokenizer=self.tokenizer)
        self.assertEqual(result['status'], 'STOPPED_TECHNICAL')
        self.assertEqual(result['cost']['logical_calls'], 1)
        self.assertEqual(result['cost']['http_attempts'], 0)
        self.assertEqual(result['unsent_cases'], list(replay.CASES))
        self.opener.open.assert_not_called()

    def test_contract_failure_is_retained_without_retry(self):
        self.prepare()
        self.set_response({'scene_audits': []}, usage=False)
        result = self.execute()
        self.assertEqual(result['cost']['http_attempts'], 2)
        self.assertEqual(result['cost']['total_tokens'], 'UNKNOWN')
        self.assertEqual([o['status'] for o in result['outcomes']], ['CONTENT_OR_CONTRACT_FAILURE'] * 2)
        self.assertFalse(any((self.run / 'cases').rglob('continuation_eligibility_only.json')))

    def test_changed_source_blocks_before_execution(self):
        self.prepare()
        atomic_json(self.source / 'cases' / replay.CASES[0] / 'child_dialogue.json', {})
        with self.assertRaisesRegex(ValueError, 'frozen_input_or_implementation_changed'):
            self.execute()
        self.opener.open.assert_not_called()

    def test_wrong_budget_rejected_before_creating_execution(self):
        self.prepare()
        for budget in (None, 1, 3, True):
            with self.assertRaisesRegex(ValueError, 'explicit_http_budget_two_required'):
                replay.execute(self.run, http_budget=budget, client_factory=self.factory, tokenizer=self.tokenizer)
        self.assertFalse((self.run / 'execution').exists())
        self.opener.open.assert_not_called()

    def test_caller_forbids_other_stages_and_cases(self):
        for i, (case, stage) in enumerate(((replay.CASES[0], 'continuation'),
                (replay.CASES[0], 'realizer'), ('cc-train-episode-198480__HARD', 'adherence'))):
            with replay.AdherenceStateCaller(self.directory / f'caller{i}', self.profile, self.tokenizer,
                    client_factory=self.factory) as caller:
                with self.assertRaises(DevelopmentRuntimeError):
                    caller.call(case, stage, {'prompt': 'never send'}, 4096)
                self.assertEqual(caller.state['http_attempt_count'], 0)
        self.opener.open.assert_not_called()

    def test_source_messages_mismatch_or_parameters_drift_rejected(self):
        case = replay.CASES[0]
        path = self.source / 'execution/calls' / case / 'adherence/request.json'
        original = replay.read(path)
        for field in ('messages', 'temperature'):
            changed = deepcopy(original)
            if field == 'messages':
                changed['payload'][field][0]['role'] = 'system'
            else:
                changed['payload'][field] = 0.1
            atomic_json(path, changed)
            with self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.run.exists())
        self.opener.open.assert_not_called()

    def test_changed_input_embedded_in_new_prompt_rejected(self):
        original = replay.build_adherence_auditor_prompt
        def altered(**kwargs):
            result = original(**kwargs)
            result['prompt'] = result['prompt'].replace('I might go.', 'I already went.')
            return result
        with patch.object(replay, 'build_adherence_auditor_prompt', side_effect=altered):
            with self.assertRaisesRegex(ValueError, 'frozen_auditor_input_changed'):
                self.prepare()
        self.opener.open.assert_not_called()


if __name__ == '__main__':
    unittest.main()
