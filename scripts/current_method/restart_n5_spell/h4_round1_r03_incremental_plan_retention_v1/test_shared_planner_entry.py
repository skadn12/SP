"""Shared Planner contracts through frozen bundles and mocked HTTP execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from .bundle import build_failed_bundle
from .plan_normalization import merged_queue
from .planning import sha
from .root_scope import prepare_bundle, merge_for_bundle
from .run import offline_network_guard
from .shared_planner_contract import ROOT_SHARED_PROMPT_VERSION, output_schema
from .shared_planning import compile_initial_output, source_continuation_rows
from .test_readiness import Response
from .test_root_entry import RootCliHarness
from .test_root_planning import independent_new_patch, existing_phase_patch, root_fixture_bundle
from .test_shared_planning import shared_initial, dependency
from .test_supplement_alignment import read_json
from .task_selection import VERSION as SELECTION_VERSION, select_same_task
from .workflow import prepare_saved_resume


def shared_additions(bundle):
    prepared = prepare_bundle(bundle)
    row = next(row for row in source_continuation_rows(prepared)
               if row['source_candidate_ref'] == 'ME03:C01')
    phase = existing_phase_patch(bundle, ('ME03:C01',))['phases'][0]
    for key in ('required_transition', 'interaction_beats'):
        row[key] = deepcopy(phase[key])
    row['dependencies'] = [dependency(source='ME03:C01')]
    new = independent_new_patch(bundle)['new_developments'][0]['candidate']
    new.update(source_candidate_ref=None, dependencies=[])
    return {'round_goal': bundle['original_plan']['round_goal'],
            'selected_continuations': [row], 'new_development_candidates': [new]}


class SharedCliHarness(RootCliHarness):
    def __enter__(self):
        super().__enter__()
        self.bundle['supplement_prompt_version'] = ROOT_SHARED_PROMPT_VERSION
        self.pin_fixture_approval()
        return self

    def http_open(self, request, **kwargs):
        actual = self.requests[-1]
        if actual['role'] != 'supplement':
            return super().http_open(request, **kwargs)
        payload = json.loads(request.data)
        self.http_payloads.append(payload)
        assert payload['messages'] == actual['messages']
        content = shared_additions(self.bundle)
        if self.case == 'empty_shared':
            content.update(selected_continuations=[], new_development_candidates=[])
        elif self.case == 'changed_goal':
            content['selected_continuations'][0]['goal'] += ' changed'
        elif self.case == 'initial_forward_dependency':
            raise AssertionError('Invalid frozen relations must fail before HTTP')
        content = json.dumps(content)
        if self.case == 'duplicate_key':
            content = '{"round_goal":"silently overwritten",' + content[1:]
        return Response(json.dumps({'choices': [{'finish_reason': 'stop',
            'message': {'content': content}}]}).encode())


class SharedPlannerEntryTests(unittest.TestCase):
    def test_shared_additions_reach_gate_realizer_and_retention(self):
        with SharedCliHarness() as harness:
            original = deepcopy(harness.bundle['original_plan'])
            report = harness.invoke(case='shared',
                extra_args=('--fresh-normalization-version', SELECTION_VERSION),
                authorization_mutation=lambda value: value.update(fresh_normalization_version=SELECTION_VERSION))
            roles = [request['role'] for request in harness.requests]
            self.assertEqual(roles.count('supplement'), 1)
            for role in ('gate', 'realizer', 'adherence'):
                self.assertIn(role, roles, report)
            output = harness.root / 'MOCK_run'
            merged = read_json(output / 'increment_patch_and_diff.json')
            self.assertTrue(merged['valid'], merged)
            self.assertEqual(set(merged['raw_patch']), set(output_schema()['properties']))
            self.assertEqual(merged['merged']['original_plan'], original)
            self.assertEqual(harness.bundle['original_plan'], original)
            self.assertEqual(merged['merged']['original_sha256'], sha(original))
            self.assertGreater(report['retained_scene_count'], 0, report)
            identity = harness.requests[0]['supplement_contract_identity']
            self.assertEqual(identity['schema_sha256'], sha(output_schema()))
            self.assertEqual(identity['execution_mode'], 'append_only')
            self.assertEqual(report['fresh_normalization_version'], SELECTION_VERSION)
            resumed = prepare_saved_resume(harness.bundle, merged['raw_patch'],
                                           normalization_version=SELECTION_VERSION)
            self.assertTrue(resumed['valid'], resumed.get('status'))
            self.assertEqual(resumed['merged']['raw_patch'], merged['raw_patch'])

    def test_changed_source_goal_stops_before_downstream(self):
        with SharedCliHarness() as harness:
            report = harness.invoke(case='changed_goal')
            self.assertEqual(report['status'], 'SUPPLEMENT_INVALID_STOP')
            self.assertEqual([row['role'] for row in harness.requests], ['supplement'])

    def test_empty_shared_arrays_continue_original_plan(self):
        with SharedCliHarness() as harness:
            report = harness.invoke(case='empty_shared')
            self.assertGreater(report['retained_scene_count'], 0, report)
            merged = read_json(harness.root / 'MOCK_run/increment_patch_and_diff.json')
            self.assertEqual(merged['merged']['added_phases'], [])
            self.assertEqual(merged['compiled_patch']['status'], 'NO_SUPPLEMENT')

    def test_duplicate_json_key_stops_before_downstream(self):
        with SharedCliHarness() as harness:
            report = harness.invoke(case='duplicate_key')
            self.assertEqual(report['status'], 'SUPPLEMENT_FAILED_STOP_NO_RETRY')
            self.assertEqual([row['role'] for row in harness.requests], ['supplement'])

    def test_frozen_initial_relations_are_replayed_and_preserved_in_queue(self):
        with offline_network_guard(), tempfile.TemporaryDirectory(prefix='shared-initial-MOCK-') as tmp:
            original = root_fixture_bundle()
            raw = shared_initial(original['original_plan'], original['parent'])
            raw['selected_continuations'][1]['dependencies'] = [dependency(output='/selected_continuations/0')]
            selected = compile_initial_output(raw, parent=original['parent'],
                candidates=original['candidates'], verified_tracks=original['verified_tracks'])
            self.assertTrue(selected['valid'], selected['errors'])
            source = Path(original['frozen_sources']['plan']['path']).parent.parent
            target = Path(tmp) / 'initial'
            for relative in ('run_freeze_manifest.json', 'memory/event_candidates.json',
                             'event_verifier/materialized_tracks.json',
                             'planner/actual_request/request_manifest.json'):
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, path)
            manifest = read_json(target / 'planner/actual_request/request_manifest.json')
            relative = Path('planner/actual_request') / manifest['messages'][0]['content_path']
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target / relative)
            selection_path = target / 'planner/variant_selection.json'
            selection_path.write_text(json.dumps(selected), encoding='utf-8')
            bundle = build_failed_bundle(failed_run=target, branch_id='MOCK-SHARED-INITIAL')
            self.assertEqual(bundle['supplement_prompt_version'], ROOT_SHARED_PROMPT_VERSION)
            prepared = prepare_bundle(bundle)
            self.assertIn('source_scope', prepared)
            empty = {'round_goal': raw['round_goal'], 'selected_continuations': [],
                     'new_development_candidates': []}
            merged = merge_for_bundle(bundle, prepared, empty)
            self.assertTrue(merged['valid'], merged)
            queue = merged_queue(prepared, merged)
            scene = next(row['scene'] for row in queue if row['scene']['source_candidate_id'] == 'ME07:C01')
            self.assertEqual(scene['dependencies'][0]['phase_id'], 'initial:ME08:C01')
            normalized = select_same_task(prepared, merged)
            self.assertTrue(normalized['valid'], normalized['errors'])
            normalized_scene = next(row['scene'] for row in normalized['queue']
                                    if row['scene']['source_candidate_id'] == 'ME07:C01')
            self.assertEqual(normalized_scene['dependencies'], scene['dependencies'])
            tampered = deepcopy(bundle)
            tampered['initial_planner_output']['selected_continuations'][1]['dependencies'] = []
            with self.assertRaisesRegex(ValueError, 'shared_initial_not_frozen_selection'):
                prepare_bundle(tampered)
            selected['shared_output']['round_goal'] += ' rewritten'
            selection_path.write_text(json.dumps(selected), encoding='utf-8')
            tampered = deepcopy(bundle)
            tampered['frozen_sources']['plan']['sha256'] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
            tampered['initial_planner_output'] = selected['shared_output']
            with self.assertRaisesRegex(ValueError, 'canonical_mismatch'):
                prepare_bundle(tampered)


if __name__ == '__main__':
    unittest.main()
