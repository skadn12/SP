"""Offline adapter contract tests. All credentials and HTTP responses are synthetic."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v123_api_under_test', HERE / 'evaluate_tju.py')
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
REAL_BATCH = api.B
FAKE_KEY = 'FAKE_ONLY_DO_NOT_USE_123456789'


def fixture(qtype='multi_select', rid='request_1'):
    labels = list('ABCDEF') if qtype == 'multi_select' else [f'I{i}' for i in range(1, 7)]
    field = 'options' if qtype == 'multi_select' else 'items'
    mapping = {label: f'content_{index}' for index, label in enumerate(labels)}
    question = {'candidate_id': 'candidate_fixture', 'root_id': 'root_fixture',
                'question_type': qtype, 'question': 'Which facts follow?'}
    row = {'request_id': rid, 'candidate_id': question['candidate_id'], 'root_id': question['root_id'],
           'question_type': qtype, 'variant': {'variant_id': 'perm_00',
           field: {label: f'Public item {index}' for index, label in enumerate(labels)},
           'display_to_content': mapping}, 'mode': 'full_context', 'slot': 0, 'seed': 1234}
    gold = [mapping['A'], mapping['C']] if qtype == 'multi_select' else list(mapping.values())
    source = {'turns': [{'turn_ref': 'T001', 'speaker': 'Avery', 'text': 'A public fact.'}]}
    return question, row, gold, source


def outcome(text='<box>A,C</box>', finish='stop', status='HTTP_OK', http=200):
    value = {'transport_status': status, 'http_status': http, 'elapsed_seconds': 0.01}
    if status == 'HTTP_OK':
        value['response_json'] = {'id': 'fake_response', 'model': 'fake_service_alias',
            'usage': {'prompt_tokens': 10, 'completion_tokens': 8},
            'choices': [{'message': {'content': text}, 'finish_reason': finish}]}
    else:
        value['raw_body'] = 'Synthetic failure'
    return value


class OfflineCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v123_adapter_test_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for context in (patch.dict(api.os.environ, {}, clear=True),
                        patch.object(api, 'KEY_FILE', self.root / 'never_real_credential'),
                        patch.object(api.urllib.request, 'build_opener',
                                     side_effect=AssertionError('REAL_NETWORK_FORBIDDEN'))):
            context.start()
            self.addCleanup(context.stop)


class PublicInputTests(OfflineCase):
    def test_public_messages_do_not_include_question_or_request_metadata(self):
        q, row, _, source = fixture()
        expected = api.payload_for(api.messages_for(q, row, source), row['seed'])
        for key in ('gold', 'gold_content_ids', 'option_evidence', 'private_design', 'old_score'):
            q[key] = 'PRIVATE_SENTINEL_' + key
        for key in ('gold_content_ids', 'display_to_content', 'content_to_text'):
            row['variant'][key] = {'PRIVATE_SENTINEL_' + key: 'DO_NOT_SEND'}
        source['private_evidence'] = 'PRIVATE_SENTINEL_SOURCE'
        actual = api.payload_for(api.messages_for(q, row, source), row['seed'])
        self.assertEqual(expected, actual)
        self.assertNotIn('PRIVATE_SENTINEL', json.dumps(actual))
        self.assertEqual(set(actual), {'model', 'messages', 'temperature', 'top_p', 'max_tokens', 'stream', 'seed'})
        self.assertEqual(actual['messages'][0]['role'], 'system')
        self.assertEqual(actual['messages'][1]['role'], 'user')

    def test_native_messages_have_one_template_layer(self):
        q, row, _, source = fixture()
        messages = api.messages_for(q, row, source)
        self.assertTrue(api.native_chat_prompt(messages).startswith('<|im_start|>system\n'))
        self.assertTrue(api.native_chat_prompt(messages).endswith('<think>\n\n</think>\n\n'))
        self.assertNotIn('<|im_start|>', json.dumps(api.payload_for(messages, 1)))

    def test_actual_prepared_42_questions_684_requests_match_reference(self):
        prepared = api.read(REAL_BATCH / 'prepared/MANIFEST.json')
        reference = api.read(api.REFERENCE / 'manifest.json')
        self.assertEqual(prepared['question_count'], 42)
        self.assertEqual(prepared['request_count'], 684)
        self.assertEqual(prepared['requests'], reference['requests'])
        rows = prepared['requests']
        self.assertEqual(len({r['request_id'] for r in rows}), 684)
        questions = api.read(REAL_BATCH / 'prepared/QUESTIONS.json')['candidates']
        qmap = {q['candidate_id']: q for q in questions}
        self.assertEqual(len(qmap), 42)
        self.assertEqual(sum(q['question_type'] == 'multi_select' for q in questions), 30)
        self.assertEqual(sum(q['question_type'] == 'ordering' for q in questions), 12)
        sources = {root: api.read(REAL_BATCH / 'prepared/sources' / (root + '.json'))
                   for root in {q['root_id'] for q in questions}}
        for row in rows:
            with self.subTest(request_id=row['request_id']):
                messages = api.messages_for(qmap[row['candidate_id']], row, sources[row['root_id']])
                native_path = api.REFERENCE / 'prompts' / (row['request_id'] + '.txt')
                self.assertEqual(api.native_chat_prompt(messages), native_path.read_text())
                self.assertEqual(api.sha(native_path), row['prompt_sha256'])
                payload = api.read(REAL_BATCH / 'prepared/payloads' / (row['request_id'] + '.json'))
                self.assertEqual(payload, api.payload_for(messages, row['seed']))
                self.assertNotIn('<|im_start|>', json.dumps(payload))
        api.verify(prepared['input_bindings'])


class ScoringTests(OfflineCase):
    def record(self, text, finish='stop', qtype='multi_select'):
        q, row, gold, _ = fixture(qtype)
        return api.record_for(row, q, gold, outcome(text, finish))

    def test_valid_answer_uses_native_scoring_and_content_mapping(self):
        r = self.record('<box>A,C</box>')
        self.assertTrue(r['evaluation_valid'])
        self.assertTrue(r['score']['exact_match'])
        self.assertEqual(r['score']['partial_score'], '1')
        self.assertEqual(r['score']['content_prediction'], ['content_0', 'content_2'])

    def test_native_partial_score_penalizes_overselection(self):
        r = self.record('<box>A,B,C</box>')
        self.assertTrue(r['evaluation_valid'])
        self.assertFalse(r['score']['exact_match'])
        self.assertEqual(r['score']['partial_score'], '3/4')

    def test_reasoning_or_multiple_boxes_fail_strict_parsing(self):
        for text in ('The answer is <box>A,C</box>', '<box>A,C</box> extra',
                     '<box>A</box><box>C</box>', 'A,C'):
            with self.subTest(text=text):
                r = self.record(text)
                self.assertFalse(r['evaluation_valid'])
                self.assertEqual(r['score']['scoring_status'], 'PARSE_FAILURE')

    def test_unknown_id_is_not_scored(self):
        r = self.record('<box>A,Z</box>')
        self.assertFalse(r['evaluation_valid'])
        self.assertEqual(r['score']['scoring_status'], 'INVALID_ID')

    def test_duplicate_id_is_not_scored(self):
        r = self.record('<box>A,A</box>')
        self.assertFalse(r['evaluation_valid'])
        self.assertEqual(r['score']['parse_reason'], 'duplicate_id')

    def test_incomplete_ordering_is_not_scored(self):
        r = self.record('<box>I1,I2,I3,I4,I5</box>', qtype='ordering')
        self.assertFalse(r['evaluation_valid'])
        self.assertEqual(r['score']['scoring_status'], 'INCOMPLETE_ORDER')

    def test_ordering_uses_native_pairwise_score(self):
        r = self.record('<box>I2,I1,I3,I4,I5,I6</box>', qtype='ordering')
        self.assertTrue(r['evaluation_valid'])
        self.assertEqual(r['score']['partial_score'], '13/15')

    def test_length_finish_invalidates_even_parseable_answer(self):
        r = self.record('<box>A,C</box>', finish='length')
        self.assertFalse(r['evaluation_valid'])
        self.assertIn('INCOMPLETE_FINISH_REASON:length', r['data_errors'])

    def test_multiple_choices_or_missing_content_is_invalid(self):
        q, row, gold, _ = fixture()
        for value in ({}, {'choices': []}, {'choices': [{}, {}]},
                      {'choices': [{'message': {'content': ['not_text']}, 'finish_reason': 'stop'}]}):
            r = api.record_for(row, q, gold, {**outcome(), 'response_json': value})
            self.assertFalse(r['evaluation_valid'])


class TransportTests(OfflineCase):
    def call_with(self, effect):
        opener = MagicMock()
        opener.open.side_effect = effect
        with patch.object(api.urllib.request, 'build_opener', return_value=opener) as factory:
            result = api.http_once({'model': 'tju-llm', 'messages': []}, FAKE_KEY)
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(factory.call_count, 1)
        self.assertIsInstance(factory.call_args.args[0], api.NoRedirect)
        self.assertNotIn(FAKE_KEY, json.dumps(result))
        return result

    def test_http401_once_and_redacted(self):
        error = urllib.error.HTTPError(api.ENDPOINT, 401, 'Denied ' + FAKE_KEY, {},
                                       io.BytesIO(('Denied ' + FAKE_KEY).encode()))
        result = self.call_with(error)
        self.assertEqual(result['transport_status'], 'HTTP_ERROR')
        self.assertEqual(result['http_status'], 401)
        self.assertIn('[REDACTED]', result['raw_body'])

    def test_network_exception_once_without_message_or_secret(self):
        result = self.call_with(urllib.error.URLError('request secret: ' + FAKE_KEY))
        self.assertEqual(result['transport_status'], 'TRANSPORT_ERROR')
        self.assertNotIn('request secret', json.dumps(result))
        self.assertEqual(result['error_type'], 'URLError')

    def test_http200_response_echo_redacted(self):
        response = MagicMock()
        response.status = 200
        response.read.return_value = json.dumps({'echo': FAKE_KEY, 'choices': []}).encode()
        response.__enter__.return_value = response
        result = self.call_with([response])
        self.assertEqual(result['response_json']['echo'], '[REDACTED]')

    def test_non_json_response_remains_redacted(self):
        response = MagicMock()
        response.status = 200
        response.read.return_value = ('not JSON ' + FAKE_KEY).encode()
        response.__enter__.return_value = response
        result = self.call_with([response])
        self.assertEqual(result['transport_status'], 'NON_JSON_RESPONSE')
        self.assertNotIn(FAKE_KEY, result['raw_body'])

    def test_redirect_is_not_followed(self):
        self.assertIsNone(api.NoRedirect().redirect_request(None, None, 307, '', {}, 'https://other.invalid'))

    def test_missing_credential_does_not_send_or_create_run(self):
        with patch.object(api, 'B', self.root), patch.object(api, 'frozen', return_value={}), \
             patch.object(api, 'http_once') as http:
            for function in (api.probe, api.live):
                with self.subTest(function=function.__name__), self.assertRaisesRegex(ValueError, 'CREDENTIAL_FILE_MISSING'):
                    function()
            http.assert_not_called()
        self.assertFalse((self.root / 'probes').exists())
        self.assertFalse((self.root / 'run').exists())

    def test_credential_environment_and_private_file_only(self):
        with patch.dict(api.os.environ, {'TJU_API_KEY': FAKE_KEY}):
            self.assertEqual(api.credential(), FAKE_KEY)
        api.KEY_FILE.write_text(FAKE_KEY)
        api.KEY_FILE.chmod(0o600)
        self.assertEqual(api.credential(), FAKE_KEY)
        api.KEY_FILE.chmod(0o644)
        with self.assertRaisesRegex(ValueError, 'PRIVATE_0600'):
            api.credential()


class ResumeTests(OfflineCase):
    def make_run(self, count=2):
        q, _, gold, source = fixture()
        rows = [fixture(rid=f'request_{i + 1}')[1] for i in range(count)]
        api.save(self.root / 'TOOLING_FREEZE.json', {'files': []})
        api.save(self.root / 'prepared/QUESTIONS.json', {'candidates': [q]})
        api.save(self.root / 'prepared/GOLD.json', {q['candidate_id']: gold})
        for row in rows:
            api.save(self.root / 'prepared/payloads' / (row['request_id'] + '.json'),
                     api.payload_for(api.messages_for(q, row, source), row['seed']))
        probe = self.root / 'probes/probe_0001'
        api.save(probe / 'REQUEST.json', {'synthetic': True})
        api.save(probe / 'RAW_RESPONSE.json', {'synthetic': True})
        api.save(probe / 'RESULT.json', {'status': 'PROBE_PASS',
            'tooling_freeze_binding': api.bind(self.root / 'TOOLING_FREEZE.json'),
            'request_binding': api.bind(probe / 'REQUEST.json'),
            'raw_binding': api.bind(probe / 'RAW_RESPONSE.json')})
        return {'requests': rows}

    def live_context(self, manifest, outcomes):
        contexts = [patch.object(api, 'B', self.root),
                    patch.object(api, 'frozen', return_value=manifest),
                    patch.object(api, 'credential', return_value=FAKE_KEY),
                    patch.object(api, 'http_once', side_effect=outcomes),
                    patch.object(api.time, 'sleep'), redirect_stdout(io.StringIO())]
        return contexts

    def test_http_failure_resume_skips_failure_preserves_receipts_and_no_rerun(self):
        manifest = self.make_run()
        contexts = self.live_context(manifest, [outcome(status='HTTP_ERROR', http=401), outcome()])
        from contextlib import ExitStack
        with ExitStack() as stack:
            entered = [stack.enter_context(c) for c in contexts]
            http = entered[3]
            with self.assertRaisesRegex(RuntimeError, 'HTTP_OR_TRANSPORT_FAILURE'):
                api.live()
            self.assertEqual(http.call_count, 1)
            first_files = [self.root / f'run/{kind}/request_1.json' for kind in ('responses', 'attempts', 'raw')]
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in first_files}
            api.live()
            self.assertEqual(http.call_count, 2)
            for p, expected in before.items():
                self.assertEqual((p.read_bytes(), p.stat().st_mtime_ns), expected)
            complete = api.read(self.root / 'run/COMPLETE.json')
            self.assertEqual(complete['actual_call_count'], 2)
            self.assertEqual(complete['valid_call_count'], 1)
            self.assertEqual(complete['invalid_call_count'], 1)
            with self.assertRaisesRegex(ValueError, 'EVALUATION_ALREADY_COMPLETE'):
                api.live()
            self.assertEqual(http.call_count, 2)
        saved = '\n'.join(p.read_text() for p in (self.root / 'run').rglob('*.json'))
        self.assertNotIn(FAKE_KEY, saved)

    def test_ambiguous_attempt_with_or_without_raw_never_resends(self):
        from contextlib import ExitStack
        manifest = self.make_run()
        payload = self.root / 'prepared/payloads/request_1.json'
        attempt = self.root / 'run/attempts/request_1.json'
        api.save(attempt, {'payload_binding': api.bind(payload)})
        before = attempt.read_bytes()
        for with_raw in (False, True):
            if with_raw:
                api.save(self.root / 'run/raw/request_1.json', outcome())
            with self.subTest(with_raw=with_raw), ExitStack() as stack:
                entered = [stack.enter_context(c) for c in self.live_context(manifest, [])]
                with self.assertRaisesRegex(ValueError, 'AMBIGUOUS_PRIOR_ATTEMPT_NO_AUTOMATIC_RESEND'):
                    api.live()
                entered[3].assert_not_called()
                self.assertEqual(attempt.read_bytes(), before)

    def test_resume_rejects_prior_response_identity_or_score_tampering(self):
        from contextlib import ExitStack
        manifest = self.make_run()
        with ExitStack() as stack:
            entered = [stack.enter_context(c) for c in self.live_context(
                manifest, [outcome(status='HTTP_ERROR', http=401)])]
            with self.assertRaisesRegex(RuntimeError, 'HTTP_OR_TRANSPORT_FAILURE'):
                api.live()
            http = entered[3]
            self.assertEqual(http.call_count, 1)
            path = self.root / 'run/responses/request_1.json'
            original = api.read(path)
            mutations = {
                'root_id': 'wrong_root', 'question_type': 'ordering',
                'variant_id': 'perm_wrong', 'slot': 99, 'seed': 9999,
                'score': {**original['score'], 'scored': True, 'exact_match': True, 'partial_score': '1'},
            }
            for key, value in mutations.items():
                with self.subTest(field=key):
                    path.write_text(json.dumps({**original, key: value}))
                    with self.assertRaises(ValueError):
                        api.live()
                    self.assertEqual(http.call_count, 1)
            path.write_text(json.dumps(original))

    def test_resume_rejects_prior_attempt_bound_to_other_payload_path(self):
        from contextlib import ExitStack
        manifest = self.make_run()
        with ExitStack() as stack:
            entered = [stack.enter_context(c) for c in self.live_context(
                manifest, [outcome(status='HTTP_ERROR', http=401)])]
            with self.assertRaisesRegex(RuntimeError, 'HTTP_OR_TRANSPORT_FAILURE'):
                api.live()
            path = self.root / 'run/attempts/request_1.json'
            attempt = api.read(path)
            other = self.root / 'other_payload.json'
            other.write_bytes((self.root / 'prepared/payloads/request_1.json').read_bytes())
            path.write_text(json.dumps({**attempt, 'payload_binding': api.bind(other)}))
            with self.assertRaises(ValueError):
                api.live()
            self.assertEqual(entered[3].call_count, 1)

    def test_resume_rejects_changed_attempt_request_identity(self):
        from contextlib import ExitStack
        manifest = self.make_run()
        with ExitStack() as stack:
            entered = [stack.enter_context(c) for c in self.live_context(
                manifest, [outcome(status='HTTP_ERROR', http=401)])]
            with self.assertRaisesRegex(RuntimeError, 'HTTP_OR_TRANSPORT_FAILURE'):
                api.live()
            path = self.root / 'run/attempts/request_1.json'
            attempt = api.read(path)
            path.write_text(json.dumps({**attempt, 'request_id': 'wrong_request_id'}))
            with self.assertRaises(ValueError):
                api.live()
            self.assertEqual(entered[3].call_count, 1)

    def test_completed_run_wrapper_never_probes_or_calls_again(self):
        api.save(self.root / 'run/COMPLETE.json', {'status': 'COMPLETE'})
        api.save(self.root / 'delivery/SCHOOL_SOLVER_SUMMARY_V1.json', {'status': 'EXISTING'})
        with patch.object(api, 'B', self.root), patch.object(api, 'frozen', return_value={}), \
             patch.object(api, 'probe') as probe, patch.object(api, 'live') as live, \
             patch.object(api, 'credential') as credential, redirect_stdout(io.StringIO()):
            api.run()
            probe.assert_not_called()
            live.assert_not_called()
            credential.assert_not_called()

    def test_write_once_does_not_overwrite(self):
        path = self.root / 'proof.json'
        api.save(path, {'first': True})
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            api.save(path, {'replacement': True})
        self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
