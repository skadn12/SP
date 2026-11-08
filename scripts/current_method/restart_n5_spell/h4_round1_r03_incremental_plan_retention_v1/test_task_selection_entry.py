"""Actual saved patch and formal resume routing with private state and mocks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import resume, run
from .planning import sha
from .task_selection import VERSION
from .test_root_entry import RootCliHarness
from .test_saved_patch_resume import ACTUAL, freeze_mock_resume
from .test_readiness import Response


class ActualPatchHarness(RootCliHarness):
    def http_open(self, request, **kwargs):
        actual = self.requests[-1]
        if actual['role'] == 'supplement':
            self.http_payloads.append(json.loads(request.data))
            content = resume.read(ACTUAL / 'experiment/execution/increment_patch_and_diff.json')['raw_patch']
        else:
            self.http_payloads.append(json.loads(request.data))
            content = self.fixture.request(actual)
            if actual['role'] == 'gate' and self.case == 'reject_preferred':
                for row in content['candidate_reviews']:
                    if row['candidate_ref'] == 'ND02:C01':
                        row.update(decision='fail', reason='MOCK preferred rejected; no variant retry allowed.')
        return Response(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(content)}}]}).encode())


def select_args(harness):
    manifest, args = freeze_mock_resume(harness)
    manifest['normalization_version'] = VERSION
    path = Path(args[args.index('--source-manifest') + 1])
    path.write_text(json.dumps(manifest))
    auth_path = Path(args[args.index('--authorization') + 1])
    auth = resume.read(auth_path)
    auth.update(normalization_version=VERSION, resume_source_manifest_sha256=sha(manifest))
    auth_path.write_text(json.dumps(auth))
    args[args.index('--normalization-version') + 1] = VERSION
    return manifest, args


class TaskSelectionEntryTests(unittest.TestCase):
    def test_actual_saved_source_is_offline_ready_without_call_or_consumption(self):
        with run.offline_network_guard(), tempfile.TemporaryDirectory(prefix='h4-task-select-readonly-') as temp:
            manifest = resume.build_resume_manifest(ACTUAL / 'experiment/execution',
                source_authorization=ACTUAL / 'experiment/inputs/authorization.json', normalization_version=VERSION)
            manifest_path = Path(temp) / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            ledger_before = Path(manifest['ledger_path']).read_bytes()
            with patch.object(resume, 'credential_environment', side_effect=AssertionError('no credentials')), \
                    patch.object(resume, 'SingleShotTransport', side_effect=AssertionError('no transport')):
                result = resume.main(['--source-manifest', str(manifest_path), '--run-dir', str(Path(temp) / 'out'),
                    '--experiment-id', manifest['execution_scope']['experiment_id'], '--normalization-version', VERSION])
            self.assertEqual(result['status'], 'SAVED_PATCH_RESUME_OFFLINE_READY')
            self.assertEqual(result['external_call_count'], 0)
            self.assertEqual(Path(manifest['ledger_path']).read_bytes(), ledger_before)

    def test_real_patch_full_mock_resume_keeps_nd02_never_reintroduces_nd09(self):
        for case in ('ordinary', 'reject_preferred'):
            with self.subTest(case=case), ActualPatchHarness() as harness:
                manifest, args = select_args(harness)
                harness.case = case
                initial_bytes = Path(manifest['ledger_path']).read_bytes()
                result = resume.main(args)
                self.assertEqual(result['attempted_request_counts']['supplement'], 0)
                self.assertEqual(result['experiment_total_attempted_counts']['supplement'], 1)
                self.assertTrue(Path(manifest['ledger_path']).read_bytes().startswith(initial_bytes))
                normalized = resume.read(harness.root / 'MOCK_resume/plan_normalization.json')
                self.assertEqual(len(normalized['queue']), 12)
                self.assertEqual(normalized['aliases'], [])
                rejected = normalized['not_selected'][0]
                self.assertEqual(rejected['status'], 'NOT_SELECTED_SAME_TASK_VARIANT')
                self.assertNotIn('ND09:C01', normalized.get('candidate_aliases', {}))
                gate_refs, realizer_phases = [], []
                for request in harness.requests[1:]:
                    self.assertNotIn(request['role'], {'planner', 'supplement'})
                    self.assertEqual(request['execution_scope'], harness.scope)
                    text = json.dumps(request['messages'])
                    self.assertNotIn('ND09', text)
                    self.assertNotIn('phase:new-event_2', text)
                    if request['role'] == 'gate':
                        for ref in ('ND02:C01', 'ND03:C01', 'ND04:C01'):
                            if ref in text:
                                gate_refs.append(ref)
                    if request['role'] == 'realizer':
                        data = json.loads(request['messages'][0]['content'].rsplit('\n\nINPUT DATA\n', 1)[1])
                        self.assertNotIn('authoritative_history', data)
                        realizer_phases.extend(s['phase_id'] for s in data['planned_phases_not_facts'])
                self.assertIn('ND02:C01', gate_refs)
                self.assertIn('ND03:C01', gate_refs)
                if case == 'reject_preferred':
                    self.assertNotIn('initial:ND02:C01', realizer_phases)
                self.assertLessEqual(sum(result['experiment_total_attempted_counts'].values()), 17)
                self.assertGreater(result['retained_scene_count'], 0)
                self.assertFalse(result['all_goals_complete'])

    def test_manifest_cli_version_mismatch_cannot_silently_select_new_rules(self):
        with ActualPatchHarness() as harness:
            _, args = freeze_mock_resume(harness)
            args[args.index('--normalization-version') + 1] = VERSION
            with self.assertRaisesRegex(ValueError, 'selected_version_mismatch'):
                resume.main(args)
            self.assertEqual(len(harness.requests), 1)
