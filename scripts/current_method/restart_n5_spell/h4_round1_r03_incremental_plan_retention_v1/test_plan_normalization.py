"""Pure deterministic checks; no workflow, credentials or network requests."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from .plan_normalization import VERSION, merged_queue, normalize_queue
from .retention import dependencies_satisfied
from .root_planning import merge_root_patch
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    validate_planner_outline_distinctness,
)

ACTUAL = Path(__file__).resolve().parents[5] / 'restart-n5-runs/h4_round1_r03_authorization_registry_v1_execution_live_1/experiment/execution'


def entry(candidate, *, phase=None, goal='Review the book.', participants=None, dependencies=None):
    scene = {'phase_id': phase or 'initial:' + candidate, 'source_candidate_id': candidate,
             'memory_event_id': candidate.split(':')[0], 'selected_candidate_ref': candidate,
             'goal': goal, 'required_transition': goal, 'after_state': goal,
             'participants': participants or ['Emily', 'James'],
             'required_participants': participants or ['Emily', 'James'],
             'before_state': 'They have the book.', 'transition_from_previous': 'The book arrived.',
             'actor_bindings': [{'actor': 'Emily', 'binding': 'starts review'}],
             'interaction_beats': ['Open the book.', 'Review chapter one.'],
             'new_people': [], 'dependencies': dependencies or []}
    gate = {key: deepcopy(value) for key, value in scene.items()
            if key not in {'phase_id', 'source_candidate_id', 'dependencies', 'selected_candidate_ref'}}
    return {'scene': scene, 'gate_scene': gate}


def dep(phase, condition='book review actually started'):
    return {'phase_id': phase, 'condition': condition}


class PlanNormalizationTests(unittest.TestCase):
    def test_actual_old_failure_and_valid_patch_are_unchanged(self):
        read = lambda name: json.loads((ACTUAL / name).read_text())
        prepared = read('original_plan_new_policy_diagnostics.json')
        stored = read('increment_patch_and_diff.json')
        self.assertEqual(merge_root_patch(prepared, stored['raw_patch']), stored)
        queue = merged_queue(prepared, stored)
        before = deepcopy(queue)
        old = validate_planner_outline_distinctness({'scenes': [e['gate_scene'] for e in queue]})
        self.assertEqual(old['errors'], ['planner_outline_duplicate:S05:S13'])
        result = normalize_queue(queue)
        self.assertEqual(queue, before)
        self.assertFalse(result['valid'])
        self.assertEqual(result['before_count'], 13)
        self.assertEqual(result['after_count'], 13)
        self.assertEqual(result['aliases'], [])
        collision = result['unresolved_collisions'][0]
        self.assertEqual((collision['left_candidate_id'], collision['right_candidate_id']),
                         ('ND02:C01', 'ND09:C01'))
        fields = {d['path'] for d in collision['differences']}
        self.assertIn('/scene/required_participants', fields)
        self.assertIn('/scene/before_state', fields)
        self.assertIn('/scene/interaction_beats', fields)

    def test_exact_duplicate_keeps_original_and_does_not_merge_beats(self):
        original = entry('ND02:C01')
        alias = entry('ND09:C01', phase='phase:new')
        alias['scene']['increment_type'] = 'independent_new_development'
        queue = [original, alias]
        frozen = deepcopy(queue)
        result = normalize_queue(queue)
        self.assertTrue(result['valid'])
        self.assertEqual(result['queue'], [original])
        self.assertEqual(result['aliases'][0]['duplicate_of'], 'ND02:C01')
        self.assertEqual(result['candidate_aliases'], {'ND09:C01': 'ND02:C01'})
        self.assertEqual(queue, frozen)

    def test_group_stability_and_idempotence(self):
        queue = [entry('ND01:C01'), entry('ND02:C01'),
                 entry('ND03:C01', phase='phase:third')]
        result = normalize_queue(queue)
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['queue']), 1)
        self.assertEqual(len(result['aliases']), 2)
        self.assertEqual(normalize_queue(queue), result)
        again = normalize_queue(result['queue'])
        self.assertEqual(again['queue'], result['queue'])
        self.assertEqual(again['execution_queue_sha256'], result['execution_queue_sha256'])

    def test_original_precedence_and_stable_order_within_origin(self):
        result = normalize_queue([entry('ND09:C01', phase='phase:new'), entry('ND02:C01')])
        self.assertEqual(result['queue'][0]['scene']['source_candidate_id'], 'ND02:C01')
        result = normalize_queue([entry('ND09:C01', phase='phase:first'),
                                  entry('ND10:C01', phase='phase:second')])
        self.assertEqual(result['aliases'][0]['duplicate_of'], 'ND09:C01')

    def test_people_time_location_precondition_and_unique_action_do_not_merge(self):
        for field, value in [('participants', ['Emily', 'Sarah']),
                             ('required_participants', ['Emily']), ('explicit_time', 'tomorrow'),
                             ('explicit_location', 'library'), ('before_state', 'Only after approval.'),
                             ('interaction_beats', ['Open the book.', 'Sign a new contract.']),
                             ('actor_bindings', [{'actor': 'James', 'binding': 'starts review'}]),
                             ('goal', 'Review a different book.')]:
            with self.subTest(field=field):
                original, alias = entry('ND02:C01'), entry('ND09:C01', phase='phase:new')
                alias['scene'][field] = value
                alias['gate_scene'][field] = value
                result = normalize_queue([original, alias])
                self.assertEqual(result['aliases'], [])
                self.assertEqual(len(result['queue']), 2)
                self.assertFalse(result['valid'])

    def test_core_text_uses_existing_casefold_whitespace_only(self):
        original, alias = entry('ND01:C01'), entry('ND02:C01')
        for layer in ('scene', 'gate_scene'):
            for key in ('goal', 'required_transition', 'after_state'):
                alias[layer][key] = '  REVIEW   THE\nBOOK.  '
        result = normalize_queue([original, alias])
        self.assertTrue(result['valid'])
        self.assertEqual(result['queue'], [original])

    def test_dependency_alias_rewrites_only_structured_reference(self):
        a = entry('ND01:C01')
        b = entry('ND02:C01', phase='phase:alias')
        c = entry('ND03:C01', phase='phase:next', goal='Close the book.',
                  dependencies=[dep('phase:alias')])
        c['scene']['note'] = 'phase:alias is only a literal mention'
        result = normalize_queue([a, b, c])
        self.assertTrue(result['valid'])
        updated = result['queue'][1]['scene']
        self.assertEqual(updated['dependencies'], [dep('initial:ND01:C01')])
        self.assertEqual(updated['note'], c['scene']['note'])
        self.assertFalse(dependencies_satisfied(updated, {}))
        self.assertFalse(dependencies_satisfied(updated, {'initial:ND01:C01': {'retained': True}}))
        state = {'initial:ND01:C01': {'retained': True, 'established_conditions': [
            {'condition': dep('x')['condition'], 'turn_refs': ['R03:S01:T001']} ]}}
        self.assertTrue(dependencies_satisfied(updated, state))

    def test_exact_redundant_relation_collapses_but_different_conditions_remain(self):
        a, b = entry('ND01:C01'), entry('ND02:C01', phase='phase:alias')
        c = entry('ND03:C01', phase='phase:next', goal='Close book.', dependencies=[
            dep('initial:ND01:C01'), dep('phase:alias'), dep('phase:alias', 'approval actually granted')])
        result = normalize_queue([a, b, c])
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['queue'][-1]['scene']['dependencies']), 2)
        self.assertTrue(any(change['removed_exact_redundant_relation'] for change in result['dependency_changes']))

    def test_dangling_self_cycle_forward_refs_and_duplicate_phase_fail(self):
        for queue, expected in [
            ([entry('ND01:C01', dependencies=[dep('phase:missing')])], 'dangling_dependency'),
            ([entry('ND01:C01', dependencies=[dep('initial:ND01:C01')])], 'self_dependency'),
            ([entry('ND01:C01', dependencies=[dep('initial:ND02:C01')]),
              entry('ND02:C01', dependencies=[dep('initial:ND01:C01')])], 'dependency_cycle'),
            ([entry('ND01:C01'), entry('ND01:C01')], 'duplicate_execution_phase_id')]:
            with self.subTest(expected=expected):
                result = normalize_queue(queue)
                self.assertFalse(result['valid'])
                self.assertTrue(any(error.startswith(expected) for error in result['errors']))

    def test_dependency_conditions_are_not_casefolded_or_ignored(self):
        prior = entry('ND00:C01', goal='Get approval.')
        a = entry('ND01:C01', phase='phase:a', dependencies=[dep('initial:ND00:C01', 'APPROVED')])
        b = entry('ND02:C01', phase='phase:b', dependencies=[dep('initial:ND00:C01', 'approved')])
        result = normalize_queue([prior, a, b])
        self.assertFalse(result['valid'])
        self.assertEqual(result['aliases'], [])

    def test_later_phase_of_duplicate_candidate_uses_canonical_structural_ids(self):
        original, alias = entry('ND01:C01'), entry('ND02:C01')
        followup = entry('ND02:C01', phase='phase:followup', goal='Review second chapter.',
                         dependencies=[dep('initial:ND02:C01')])
        result = normalize_queue([original, alias, followup])
        self.assertTrue(result['valid'])
        self.assertEqual(result['candidate_aliases'], {'ND02:C01': 'ND01:C01'})
        scene = result['queue'][1]['scene']
        self.assertEqual(scene['phase_id'], 'phase:followup')
        self.assertEqual(scene['source_candidate_id'], 'ND01:C01')
        self.assertEqual(scene['dependencies'], [dep('initial:ND01:C01')])
        self.assertEqual(result['queue'][1]['gate_scene']['memory_event_id'], 'ND01')
        self.assertEqual(followup['scene']['source_candidate_id'], 'ND02:C01')

    def test_version_explicit(self):
        with self.assertRaisesRegex(ValueError, 'unknown_plan_normalization_version'):
            normalize_queue([], version='unknown')
        self.assertEqual(normalize_queue([])['version'], VERSION)
