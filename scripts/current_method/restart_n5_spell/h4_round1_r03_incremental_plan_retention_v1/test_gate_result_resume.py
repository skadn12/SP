"""Approved Gate-view resume through the real CLI with copied state and mock HTTP."""
from contextlib import ExitStack
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import resume, run
from .gate_result_resume import build_gate_reuse_manifest, validate_gate_reuse_source
from .fixtures import FixtureBackend
from .planning import sha
from .task_selection import VERSION as SELECTION_VERSION
from .test_readiness import Response

RUNS = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs')
SOURCE = RUNS / 'h4_round1_r03_same_task_selection_resume_v1_live_1'
DELIVERY = RUNS / 'h4_round1_r03_gate_contract_alignment_v1_offline_preflight_live_1'


def read(path):
    return json.loads(Path(path).read_text())


class GateResumeHarness:
    def __enter__(self):
        self.stack = ExitStack()
        self.stack.enter_context(run.offline_network_guard())
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix='h4-gate-reuse-MOCK-')))
        self.source = self.root / 'source'
        shutil.copytree(SOURCE / 'execution', self.source / 'execution')
        for name in ('selection_sidecar.json', 'selected_execution_plan.json', 'resume_execution_authorization.json'):
            shutil.copy2(SOURCE / name, self.source / name)
        report_path = self.source / 'execution/final_report.json'
        report = read(report_path)
        self.real_ledger = Path(report['persistent_ledger_path'])
        self.real_before = self.real_ledger.read_bytes()
        self.registry = self.root / 'private_ledger'
        self.registry.mkdir()
        self.ledger = self.registry / self.real_ledger.name
        # Copy the spent checkpoint verbatim, never manufacture new quota.
        self.ledger.write_bytes(self.real_before)
        report['persistent_ledger_path'] = str(self.ledger)
        report_path.write_text(json.dumps(report))
        self.bundle = read(self.source / 'execution/input_freeze.json')
        self.model = read(self.source / 'execution/model_config.json')
        self.budget = read(self.source / 'execution/execution_budget.json')
        self.fixture = FixtureBackend(self.bundle)
        self.manifest = build_gate_reuse_manifest(DELIVERY, self.source)
        self.manifest_path = self.root / 'reuse_manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest))
        authorization = read(SOURCE / 'resume_execution_authorization.json')
        authorization.pop('credential_source', None)
        authorization.update(implementation_manifest=run.implementation_manifest(),
            resume_source_manifest_sha256=sha(self.manifest), gate_result_reuse_explicitly_authorized=True,
            gate_reuse_manifest_sha256=sha(self.manifest), normalization_version=SELECTION_VERSION)
        self.authorization_path = self.root / 'authorization.json'
        self.authorization_path.write_text(json.dumps(authorization))
        self.requests, self.http_payloads = [], []
        self.fail_role = None
        self.stack.enter_context(patch.dict('os.environ', {self.model['api_key_env']: 'MOCK-IN-PROCESS-ONLY'}))
        self.stack.enter_context(patch.object(resume, 'experiment_registry', return_value=self.registry))
        self.stack.enter_context(patch('urllib.request.build_opener', return_value=Mock(open=self.http_open)))
        original = run.SingleShotTransport.request
        def observed(transport, request):
            self.requests.append(deepcopy(request))
            return original(transport, request)
        self.stack.enter_context(patch.object(run.SingleShotTransport, 'request', observed))
        self.stack.enter_context(patch('sys.stdout', new=io.StringIO()))
        self.stack.enter_context(patch('sys.stderr', new=io.StringIO()))
        return self

    def __exit__(self, *args):
        try:
            assert self.real_ledger.read_bytes() == self.real_before
        finally:
            self.stack.close()

    def http_open(self, request, **kwargs):
        actual = self.requests[-1]
        assert actual['role'] in {'realizer', 'adherence'}
        payload = json.loads(request.data)
        assert payload['messages'] == actual['messages']
        self.http_payloads.append(payload)
        if actual['role'] == self.fail_role:
            raise TimeoutError('MOCK ambiguous request; never resend')
        content = self.fixture.request(actual)
        return Response(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(content)}}]}).encode())

    def args(self, name='MOCK_resume'):
        return ['--gate-reuse-manifest', str(self.manifest_path), '--run-dir', str(self.root / name),
                '--authorization', str(self.authorization_path), '--experiment-id', self.manifest['execution_scope']['experiment_id'],
                '--normalization-version', SELECTION_VERSION, '--execute']


def exercise_gate_result_resume(*, fail_role=None):
    with GateResumeHarness() as harness:
        harness.fail_role = fail_role
        result = resume.main(harness.args())
        path = harness.root / 'MOCK_resume'
        return {'notice': 'MOCK ONLY; no production writes, grants or credentials exported',
                'result_MOCK': result, 'requests_MOCK': harness.requests, 'http_payloads_MOCK': harness.http_payloads,
                'formal_reuse_records_MOCK': read(path / 'formal_gate_reuse_records.json'),
                'ledger_MOCK': read(path / 'ledger_snapshot.json'), 'actual_ledger_unchanged': True,
                'new_real_calls': 0}


class GateResultResumeTests(unittest.TestCase):
    def test_actual_frozen_loader_has_eight_matching_results(self):
        with run.offline_network_guard():
            manifest = build_gate_reuse_manifest(DELIVERY, SOURCE)
            before = Path(manifest['ledger_path']).read_bytes()
            loaded = validate_gate_reuse_source(manifest)
            self.assertEqual(len(loaded['gate_reuse']['by_phase']), 8)
            self.assertEqual(loaded['bundle']['realizer_mode'], 'text_free')
            self.assertNotIn('phase:new-event_2', loaded['gate_reuse']['by_phase'])
            self.assertEqual(Path(manifest['ledger_path']).read_bytes(), before)

    def test_formal_cli_loads_views_and_only_sends_realizer_adherence(self):
        result = exercise_gate_result_resume()
        report = result['result_MOCK']
        self.assertEqual(report['status'], 'EXECUTION_ENDED_GOALS_RECORDED_SEPARATELY')
        self.assertEqual(report['attempted_request_counts'], {'planner': 0, 'supplement': 0, 'gate': 0,
                         'realizer': 3, 'adherence': 3, 'closure': 0})
        self.assertEqual(report['experiment_total_attempted_counts'], {'planner': 0, 'supplement': 1, 'gate': 8,
                         'realizer': 3, 'adherence': 3, 'closure': 0})
        self.assertEqual(report['retained_scene_count'], 6)
        self.assertEqual(len(result['formal_reuse_records_MOCK']), 8)
        self.assertTrue(any(row.get('status') == 'DEFERRED_GATE_RESULT_UNAVAILABLE' for row in report['scene_decisions']))
        for request in result['requests_MOCK']:
            text = json.dumps(request['messages'])
            self.assertNotIn('ND09', text)
            self.assertNotIn('phase:new-event_2', text)
            if request['role'] == 'realizer':
                data = json.loads(request['messages'][0]['content'].rsplit('\n\nINPUT DATA\n', 1)[1])
                self.assertLessEqual(len(data['planned_phases_not_facts']), 2)
                self.assertNotIn('authoritative_history', data)
                self.assertEqual(request['prompt_version'], 'incremental_retention_v1.realizer.successful_prompt_alignment_v3.text_free')
        self.assertEqual([r['call_id'] for r in result['requests_MOCK']], [10, 11, 12, 13, 14, 15])

    def test_same_manifest_restart_or_new_directory_never_replays(self):
        with GateResumeHarness() as harness:
            resume.main(harness.args())
            before = harness.ledger.read_bytes()
            calls = len(harness.requests)
            with self.assertRaises(ValueError):
                resume.main(harness.args('new_output_directory'))
            self.assertEqual(len(harness.requests), calls)
            self.assertEqual(harness.ledger.read_bytes(), before)

    def test_sent_failure_consumes_once_and_does_not_resume(self):
        for role in ('realizer', 'adherence'):
            with self.subTest(role=role), GateResumeHarness() as harness:
                harness.fail_role = role
                report = resume.main(harness.args())
                self.assertEqual(report['status'], 'STAGE_FAILURE_STOP_NO_RETRY')
                self.assertEqual(report['attempted_request_counts']['realizer'], 1)
                self.assertEqual(report['attempted_request_counts']['adherence'], int(role == 'adherence'))
                count = len(harness.requests)
                with self.assertRaises(ValueError):
                    resume.main(harness.args('no_replay'))
                self.assertEqual(len(harness.requests), count)

    def test_missing_authorization_or_credentials_blocks_before_claim(self):
        for mode in ('authorization', 'credentials'):
            with self.subTest(mode=mode), GateResumeHarness() as harness:
                before = harness.ledger.read_bytes()
                if mode == 'authorization':
                    auth = read(harness.authorization_path)
                    auth.pop('gate_result_reuse_explicitly_authorized')
                    harness.authorization_path.write_text(json.dumps(auth))
                else:
                    harness.stack.enter_context(patch.dict('os.environ', {harness.model['api_key_env']: ''}))
                with self.assertRaises(ValueError):
                    resume.main(harness.args())
                self.assertEqual(harness.requests, [])
                self.assertEqual(harness.ledger.read_bytes(), before)

    def test_tampered_file_or_identity_rejected_before_sending(self):
        with GateResumeHarness() as harness:
            before = harness.ledger.read_bytes()
            original = deepcopy(harness.manifest)
            for mutation in (lambda m: m.update(selected_queue_sha256='0' * 64),
                             lambda m: m.update(identity={'root_id': 'wrong'}),
                             lambda m: m.update(ledger_sha256='0' * 64)):
                changed = deepcopy(original)
                mutation(changed)
                harness.manifest_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    resume.main(harness.args())
            self.assertEqual(harness.ledger.read_bytes(), before)
            self.assertEqual(harness.requests, [])


if __name__ == '__main__':
    unittest.main()
