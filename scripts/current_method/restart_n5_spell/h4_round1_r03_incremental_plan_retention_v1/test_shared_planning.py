"""Offline shared Planner compilation, identity and dependency regressions."""
from copy import deepcopy
import unittest

from .planning import prepare_original, sha
from .root_planning import prepare_root_original
from .run import offline_network_guard
from .shared_planner_contract import VERSION, validate_output
from .shared_planning import (apply_initial_relations, compile_initial_output,
                              merge_shared_output, prepare_shared_original, source_continuation_rows)
from .test_root_planning import root_fixture_bundle, independent_new_patch
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import build_speaker_registry


def shared_initial(plan, parent):
    raw = deepcopy(plan)
    refs = {row['canonical_name']: row['speaker_ref'] for row in build_speaker_registry(parent)['speakers']}
    for collection in ('selected_continuations', 'new_development_candidates'):
        for row in raw[collection]:
            row['speaker_refs'] = [refs[name] for name in row.pop('participants') if name in refs]
            row['actor_bindings'] = [{'actor_ref': refs[binding['actor']], 'binding': binding['binding']}
                                     for binding in row['actor_bindings']]
            row.update(source_candidate_ref=None, dependencies=[])
    return raw


def dependency(*, source=None, output=None):
    return {'source_candidate_ref': source, 'output_ref': output,
            'condition': 'The earlier retained scene establishes the concrete arrangement.'}


class SharedPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guard = offline_network_guard()
        guard.__enter__()
        cls.addClassCleanup(guard.__exit__, None, None, None)
        cls.bundle = root_fixture_bundle()
        cls.prepared = prepare_root_original(cls.bundle)

    def initial(self):
        return shared_initial(self.prepared['original'], self.bundle['parent'])

    def compile(self, raw):
        return compile_initial_output(raw, parent=self.bundle['parent'],
            candidates=self.bundle['candidates'], verified_tracks=self.bundle['verified_tracks'])

    def append(self, prepared=None, source='ME03:C01'):
        prepared = prepared or self.prepared
        row = next(row for row in source_continuation_rows(prepared) if row['source_candidate_ref'] == source)
        row['dependencies'] = [dependency(source=source)]
        row['required_transition'] = 'Emily records a specific new workshop arrangement on a card.'
        row['interaction_beats'] = ['Emily chooses a specific date.', 'James checks the proposed date.',
                                    'Emily records the agreed date on the card.']
        return {'round_goal': prepared['original']['round_goal'],
                'selected_continuations': [row], 'new_development_candidates': []}

    def test_initial_reuses_legacy_target_waivers_and_canonical_plan(self):
        raw = self.initial()
        before = sha(raw)
        result = self.compile(raw)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['value'], self.prepared['original'])
        self.assertEqual(result['shared_output'], raw)
        self.assertEqual(result['planner_contract_version'], VERSION)
        self.assertEqual(sha(raw), before)
        self.assertTrue(result['prepared']['goal_diagnostics'])

    def test_initial_dependencies_keep_original_indices_and_plan_immutable(self):
        raw = self.initial()
        raw['selected_continuations'][1]['dependencies'] = [dependency(output='/selected_continuations/0')]
        raw['new_development_candidates'][0]['dependencies'] = [dependency(output='/selected_continuations/1')]
        result = self.compile(raw)
        self.assertTrue(result['valid'], result['errors'])
        prepared = result['prepared']
        mapping = prepared['initial_output_phase_map']
        self.assertEqual(mapping['/selected_continuations/0'], 'initial:ME08:C01')
        self.assertEqual(mapping['/selected_continuations/1'], 'initial:ME07:C01')
        self.assertEqual(mapping['/new_development_candidates/0'], 'initial:ND01:C01')
        self.assertEqual(prepared['candidate_registry']['ME07:C01']['scene']['dependencies'][0]['phase_id'],
                         'initial:ME08:C01')
        self.assertEqual(prepared['candidate_registry']['ND01:C01']['scene']['dependencies'][0]['phase_id'],
                         'initial:ME07:C01')
        self.assertEqual(prepared['original'], self.prepared['original'])
        replay = apply_initial_relations(self.prepared, raw, parent=self.bundle['parent'],
            candidates=self.bundle['candidates'], verified_tracks=self.bundle['verified_tracks'])
        self.assertTrue(replay['valid'], replay['errors'])
        self.assertEqual(replay['prepared']['source_scope']['candidate_registry_sha256'],
                         sha(replay['prepared']['candidate_registry']))
        self.assertNotIn('dependencies', self.prepared['candidate_registry']['ND01:C01']['scene'])

    def test_prepare_shared_original_rejects_unfrozen_relation_metadata(self):
        bundle = deepcopy(self.bundle)
        bundle['initial_planner_output'] = self.initial()
        with self.assertRaisesRegex(ValueError, 'unfrozen_initial_relations'):
            prepare_shared_original(bundle)

    def test_initial_source_track_limits_and_backward_only_dependencies(self):
        changes = (
            lambda raw: raw['selected_continuations'][0].update(source_candidate_ref='ME08:C01'),
            lambda raw: raw['new_development_candidates'][0].update(source_candidate_ref='ND01:C01'),
            lambda raw: raw['selected_continuations'][0].update(verified_track_ref=None),
            lambda raw: raw['selected_continuations'][0].update(verified_track_ref='VT999'),
            lambda raw: raw['selected_continuations'][1].update(verified_track_ref=raw['selected_continuations'][0]['verified_track_ref']),
            lambda raw: raw['selected_continuations'].append(deepcopy(raw['selected_continuations'][0])),
            lambda raw: raw['selected_continuations'][0].update(dependencies=[dependency(source='ME08:C01')]),
            lambda raw: raw['selected_continuations'][0].update(dependencies=[dependency(output='/selected_continuations/0')]),
            lambda raw: raw['selected_continuations'][0].update(dependencies=[dependency(output='/new_development_candidates/0')]),
            lambda raw: raw['new_development_candidates'][0].update(dependencies=[dependency(output='/selected_continuations/99')]),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                raw = self.initial()
                change(raw)
                self.assertFalse(self.compile(raw)['valid'])

    def test_initial_dependency_cannot_point_to_quarantined_candidate(self):
        raw = self.initial()
        raw['new_development_candidates'][0]['required_transition'] = 'Emily discusses familiar progress.'
        raw['new_development_candidates'][1]['dependencies'] = [dependency(output='/new_development_candidates/0')]
        result = self.compile(raw)
        self.assertFalse(result['valid'])
        self.assertTrue(any('dependency_target_unavailable' in error for error in result['errors']), result['errors'])

    def test_append_preserves_full_source_and_actual_raw_output(self):
        raw = self.append()
        before = sha(self.prepared), sha(raw)
        result = merge_shared_output(self.prepared, raw)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['raw_patch'], raw)
        self.assertEqual(result['merged']['original_plan'], self.prepared['original'])
        self.assertEqual(result['merged']['original_sha256'], self.prepared['original_sha256'])
        self.assertEqual(result['merged_sha256'], sha(result['merged']))
        self.assertEqual(result['compiled_patch']['phases'][0]['phase_id'], 'phase:append-continuation-1')
        self.assertEqual((sha(self.prepared), sha(raw)), before)

    def test_append_cannot_change_goal_target_actor_speakers_or_source(self):
        changes = (
            lambda row: row.update(goal=row['goal'] + ' altered'),
            lambda row: row.update(after_state=row['after_state'] + ' altered'),
            lambda row: row.update(transition_from_previous=row['transition_from_previous'] + ' altered'),
            lambda row: row['actor_bindings'][0].update(binding='Different action owner.'),
            lambda row: row['actor_bindings'][0].update(actor_ref='SP999'),
            lambda row: row.update(speaker_refs=list(reversed(row['speaker_refs']))),
            lambda row: row.update(new_people=[{'name': 'Someone', 'relationship_basis': 'New person.'}]),
            lambda row: row.update(verified_track_ref='VT08'),
            lambda row: row.update(source_candidate_ref='ME999:C01'),
            lambda row: row.update(source_candidate_ref=None),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                raw = self.append()
                change(raw['selected_continuations'][0])
                self.assertFalse(merge_shared_output(self.prepared, raw)['valid'])
        raw = self.append()
        raw['round_goal'] += ' altered'
        self.assertIn('shared_round_goal_changed', merge_shared_output(self.prepared, raw)['errors'])

    def test_append_dependency_sources_self_forward_unknown_and_empty_reject(self):
        for dependencies in ([], [dependency(source='ND999:C01')],
                [dependency(output='/selected_continuations/0')],
                [dependency(output='/new_development_candidates/0')],
                [dependency(output='/selected_continuations/4')],
                [dependency(source='initial:ME03:C01')]):
            with self.subTest(dependencies=dependencies):
                raw = self.append()
                raw['selected_continuations'][0]['dependencies'] = dependencies
                self.assertFalse(merge_shared_output(self.prepared, raw)['valid'])

    def test_independent_additions_and_output_dependencies_get_stable_backend_ids(self):
        raw = self.append()
        second = deepcopy(raw['selected_continuations'][0])
        second['dependencies'] = [dependency(output='/selected_continuations/0')]
        raw['selected_continuations'].append(second)
        new = independent_new_patch(self.bundle)['new_developments'][0]['candidate']
        new.update(source_candidate_ref=None, dependencies=[dependency(output='/selected_continuations/1')])
        raw['new_development_candidates'].append(new)
        result = merge_shared_output(self.prepared, raw)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result, merge_shared_output(self.prepared, raw))
        scenes = [entry['scene'] for entry in result['merged']['added_phases']]
        self.assertEqual([scene['phase_id'] for scene in scenes],
                         ['phase:append-continuation-1', 'phase:append-continuation-2', 'phase:append-development-1'])
        self.assertEqual(scenes[-1]['dependencies'][0]['phase_id'], 'phase:append-continuation-2')
        raw['new_development_candidates'][0]['source_candidate_ref'] = 'ND01:C01'
        self.assertFalse(merge_shared_output(self.prepared, raw)['valid'])

    def test_old_new_development_can_continue_with_null_track_and_existing_people(self):
        original = deepcopy(self.prepared['original'])
        original['new_development_candidates'][0]['new_people'] = [
            {'name': 'MockVisitor', 'relationship_basis': 'A person introduced at the original activity.'}]
        original['new_development_candidates'][0]['participants'].append('MockVisitor')
        prepared = prepare_original(original, parent=self.bundle['parent'],
            candidates=self.bundle['candidates'], verified_tracks=self.bundle['verified_tracks'])
        self.assertTrue(prepared['valid_for_new_policy'], prepared['hard_errors'])
        prepared.update(eligible_source_candidate_ids=list(prepared['candidate_registry']),
            source_scope={**deepcopy(self.prepared['source_scope']),
                'original_plan_sha256': prepared['original_sha256'],
                'candidate_registry_sha256': sha(prepared['candidate_registry'])},
            _root_materialization_inputs=deepcopy(self.prepared['_root_materialization_inputs']))
        raw = self.append(prepared, source='ND01:C01')
        self.assertIsNone(raw['selected_continuations'][0]['verified_track_ref'])
        self.assertEqual(raw['selected_continuations'][0]['new_people'][0]['name'], 'MockVisitor')
        result = merge_shared_output(prepared, raw)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['merged']['added_phases'][0]['scene']['new_people'],
                         prepared['candidate_registry']['ND01:C01']['scene']['new_people'])

    def test_empty_append_is_no_supplement_and_legacy_shape_rejected(self):
        raw = {'round_goal': self.prepared['original']['round_goal'],
               'selected_continuations': [], 'new_development_candidates': []}
        self.assertTrue(validate_output(raw)['valid'])
        initial = self.compile(raw)
        self.assertTrue(initial['valid'], initial['errors'])
        self.assertEqual(initial['prepared']['candidate_registry'], {})
        result = merge_shared_output(self.prepared, raw)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['compiled_patch']['status'], 'NO_SUPPLEMENT')
        self.assertEqual(result['merged']['added_phases'], [])
        legacy = {'status': 'NO_SUPPLEMENT', 'original_plan_sha256': self.prepared['original_sha256'],
                  'phases': [], 'new_developments': [], 'reason': 'Old format.'}
        self.assertFalse(merge_shared_output(self.prepared, legacy)['valid'])
        self.assertFalse(self.compile(self.prepared['original'])['valid'])


if __name__ == '__main__':
    unittest.main()
