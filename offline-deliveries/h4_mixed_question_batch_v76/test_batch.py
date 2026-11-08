"""Hermetic batch contract tests: no writer, native gate, Solver, or network.

The two lane engines and saved-review gates are mocks. Temporary files model
their boundary artifacts so batch policy, scheduling, writes, and SHA checks
still run through the production implementation.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import batch
import common
import ordering_review


CONFIG = common.read(Path(__file__).with_name('CONFIG.json'))
VALIDATE_REVIEW_SOURCES = batch.validate_review_sources
VERIFY_PLAN = batch.verify_plan


def replace(path, value):
    """Only mutate test-owned artifacts to simulate a changed frozen file."""
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def timestamp(offset=-10):
    return (datetime.now(timezone.utc) + timedelta(seconds=offset)).isoformat()


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v76-batch-tests-')
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name).resolve()
        self.home = root / 'batch'
        self.engine = root / 'frozen-v75'
        self.value = deepcopy(CONFIG)
        self.value['multiselect_engine'] = str(self.engine)
        common.write(self.engine / 'CONFIG.json', self.value['multiselect_config'])
        common.write(self.home / 'CONFIG.json', self.value)

        def forbidden(*args, **kwargs):
            raise AssertionError('Unexpected external generation, native, process, or network call')

        self.forbidden = forbidden
        self.start_patch(patch.object(batch, 'V75', self.engine))
        self.start_patch(patch.object(batch, 'HERE', self.home))
        self.plan_gate = self.start_patch(patch.object(batch, 'verify_plan', return_value={'status': 'PREPARED'}))
        self.source_gate = self.start_patch(patch.object(batch, 'validate_review_sources', return_value=None))
        self.multi_loader = self.start_patch(patch.object(batch, 'load_multi', side_effect=forbidden))
        self.ordering_generator = self.start_patch(patch.object(batch.ordering_generation, 'generate', side_effect=forbidden))
        for module, name in ((common, 'load_multi'), (common, 'load_ordering_gate'),
                             (batch.ordering_generation, 'load_multi'),
                             (batch.ordering_generation, 'load_ordering_gate'),
                             (batch.ordering_generation, 'prepare'),
                             (batch.ordering_generation, 'verify'),
                             (ordering_review, 'load_ordering_gate')):
            self.start_patch(patch.object(module, name, side_effect=forbidden))
        self.ordering_review_gate = self.start_patch(patch.object(ordering_review, 'verify_review', side_effect=forbidden))
        self.ordering_source_gate = self.start_patch(patch.object(ordering_review, 'validate_sources', side_effect=forbidden))
        self.start_patch(patch.object(socket, 'create_connection', side_effect=forbidden))
        self.start_patch(patch.object(socket.socket, 'connect', side_effect=forbidden))
        self.start_patch(patch.object(subprocess, 'Popen', side_effect=forbidden))

    def start_patch(self, item):
        result = item.start()
        self.addCleanup(item.stop)
        return result

    def save_config(self):
        replace(self.home / 'CONFIG.json', self.value)

    def assert_no_freeze(self):
        self.assertFalse((self.home / 'EVALUATION_POOL.json').exists())
        self.assertFalse((self.home / 'BATCH_REVIEW.COMPLETE.json').exists())

    def install_generation(self, *, multi_error=None, ordering_error=None):
        events = []

        def multi_generate(home):
            self.assertEqual(home, self.home / 'multiselect')
            events.append('multi_select')
            common.write(home / 'assets/family/rounds/initial/requests/001/HTTP_ATTEMPT.json', {'synthetic': True})
            if multi_error is not None:
                raise multi_error
            common.write(home / 'INITIAL_GENERATION.COMPLETE.json', {'synthetic': True})

        def ordering_generate(home):
            self.assertEqual(home, self.home)
            events.append('ordering')
            common.write(home / 'ordering/root/requests/001/HTTP_ATTEMPT.json', {'synthetic': True})
            if ordering_error is not None:
                raise ordering_error
            common.write(home / 'ordering/GENERATION.COMPLETE.json', {'synthetic': True})

        driver = Mock()
        driver.generate.side_effect = multi_generate

        def load(name):
            self.assertEqual(name, 'generation_driver')
            return driver

        self.multi_loader.side_effect = load
        self.ordering_generator.side_effect = ordering_generate
        return driver, events

    def install_reviewed_artifacts(self):
        """Supply synthetic lane outcomes at the already-reviewed boundary."""
        common.write(self.home / 'BATCH_PREPARATION.COMPLETE.json', {
            'status': 'PREPARED', 'completed_at_utc': timestamp(-30)})
        child_generations = [self.home / 'multiselect/INITIAL_GENERATION.COMPLETE.json',
                             self.home / 'ordering/GENERATION.COMPLETE.json']
        for path in child_generations:
            common.write(path, {'synthetic': True})
        common.write(self.home / 'BATCH_GENERATION.COMPLETE.json', {
            'version': batch.VERSION, 'completed_at_utc': timestamp(-20),
            'status': 'GENERATED', 'completed_types': ['multi_select', 'ordering'],
            'errors': [], 'http_attempts': 0, 'training_allowed': False, 'automatic_retry': False,
            'files': [common.bind(path) for path in child_generations],
        })
        self.multi_review_path = self.home / 'multiselect/FINAL_CONTENT_REVIEW.COMPLETE.json'
        self.ordering_review_path = self.home / 'ordering/CONTENT_REVIEW.COMPLETE.json'
        for path in (self.multi_review_path, self.ordering_review_path):
            common.write(path, {'status': 'REVIEWED', 'completed_at_utc': timestamp()})
        self.multi_pool_path = self.home / 'multiselect/EVALUATION_POOL.json'
        self.ordering_pool_path = self.home / 'ordering/EVALUATION_POOL.json'
        common.write(self.multi_pool_path, {'questions': [
            {'candidate_id': 'synthetic-multi-1', 'root_id': 'synthetic-root',
             'question_type': 'multi_select', 'question': 'Synthetic reviewed multi-select', 'gold': ['A']},
        ]})
        common.write(self.ordering_pool_path, {'questions': [
            {'candidate_id': 'synthetic-ordering-1', 'root_id': 'synthetic-root',
             'question_type': 'ordering', 'question': 'Synthetic reviewed ordering', 'gold': ['I1', 'I2']},
        ]})
        self.review_events = []

        def verify_review(home, path, lane):
            self.assertEqual(home, self.home / 'multiselect' if lane == 'multi_select' else self.home)
            self.review_events.append(lane)
            marker = common.read(path)
            if marker['status'] != 'REVIEWED':
                raise ValueError('synthetic_review_incomplete')
            return marker

        self.multi_review_gate = Mock(side_effect=lambda home: verify_review(home, self.multi_review_path, 'multi_select'))
        self.multi_content = Mock(side_effect=lambda home: (
            {}, common.read(self.multi_pool_path)['questions'], [], {}))
        evaluate = Mock(verify_review=self.multi_review_gate, reviewed_content=self.multi_content)

        def load(name):
            self.assertEqual(name, 'evaluate')
            return evaluate

        self.multi_loader.side_effect = load
        self.ordering_review_gate.side_effect = lambda home: verify_review(home, self.ordering_review_path, 'ordering')

    def rebind_frozen_file(self, path):
        marker_path = self.home / 'BATCH_REVIEW.COMPLETE.json'
        frozen = common.read(marker_path)
        self.assertIn(str(path.resolve()), [row['path'] for row in frozen['files']])
        frozen['files'] = [common.bind(path) if row['path'] == str(path.resolve()) else row
                           for row in frozen['files']]
        replace(marker_path, frozen)

    def test_frozen_config_accepts_exact_current_contract(self):
        self.assertEqual(batch.config(self.home), self.value)
        self.assertEqual((self.value['max_generation_http'], self.value['max_candidates'],
                          self.value['max_solver_calls_per_model']), (12, 28, 456))

    def test_config_rejects_generation_candidate_and_solver_budget_changes(self):
        original = deepcopy(self.value)
        for key in ('max_generation_http', 'max_candidates', 'max_solver_calls_per_model'):
            for delta in (-1, 1):
                with self.subTest(key=key, delta=delta):
                    self.value = deepcopy(original)
                    self.value[key] += delta
                    self.save_config()
                    with self.assertRaisesRegex(ValueError, '^typed_batch_budget_changed$'):
                        batch.config(self.home)

    def test_config_rejects_question_type_changes(self):
        for types in (['ordering', 'multi_select'], ['multi_select'], ['ordering'],
                      ['multi_select', 'ordering', 'other'], ['multi_select', 'multi_select']):
            with self.subTest(types=types):
                self.value['question_types'] = types
                self.save_config()
                with self.assertRaisesRegex(ValueError, '^mixed_batch_policy_changed$'):
                    batch.config(self.home)

    def test_config_rejects_missing_duplicate_and_changed_roots(self):
        original = deepcopy(self.value)
        mutations = {
            'missing': lambda slots: slots.pop(),
            'duplicate_root': lambda slots: slots[1].update(root_id=slots[0]['root_id']),
            'unknown_root': lambda slots: slots[0].update(root_id='new-root'),
            'duplicate_slot': lambda slots: slots[1].update(slot_id=slots[0]['slot_id']),
            'changed_count': lambda slots: slots[0].update(candidate_count=1),
        }
        for label, mutate in mutations.items():
            with self.subTest(change=label):
                self.value = deepcopy(original)
                mutate(self.value['ordering_slots'])
                self.save_config()
                with self.assertRaisesRegex(ValueError, '^both_types_require_same_distinct_roots$'):
                    batch.config(self.home)

    def test_config_rejects_source_history_and_prior_pool_changes(self):
        original = deepcopy(self.value)
        for key in ('source_path', 'source_sha256', 'history_path', 'history_sha256', 'prior_pool_path'):
            with self.subTest(key=key):
                self.value = deepcopy(original)
                self.value['ordering_slots'][0][key] += '-changed'
                self.save_config()
                with self.assertRaisesRegex(ValueError, '^ordering_full_source_contract_changed$'):
                    batch.config(self.home)

    def test_config_rejects_changed_multiselect_engine_or_strategy(self):
        original = deepcopy(self.value)
        for change in ('engine', 'strategy'):
            with self.subTest(change=change):
                self.value = deepcopy(original)
                if change == 'engine':
                    self.value['multiselect_engine'] = str(self.home)
                else:
                    self.value['multiselect_config']['max_http'] += 1
                self.save_config()
                with self.assertRaisesRegex(ValueError, '^multiselect_strategy_changed$'):
                    batch.config(self.home)

    def test_config_rejects_retry_training_and_unreviewed_ordering_policies(self):
        original = deepcopy(self.value)
        for key in ('training_allowed', 'automatic_followup', 'transport_retries',
                    'semantic_retries', 'mechanical_repairs', 'ordering_review_required'):
            with self.subTest(key=key):
                self.value = deepcopy(original)
                self.value[key] = False if key == 'ordering_review_required' else True
                self.save_config()
                reason = 'typed_batch_budget_changed' if key == 'ordering_review_required' else 'mixed_batch_policy_changed'
                with self.assertRaisesRegex(ValueError, '^' + reason + '$'):
                    batch.config(self.home)

    def test_config_rejects_numeric_coercions_and_boolean_budgets(self):
        original = deepcopy(self.value)
        fields = ('transport_retries', 'semantic_retries', 'mechanical_repairs',
                  'max_generation_http', 'max_candidates', 'max_solver_calls_per_model',
                  'ordering_output_max_tokens')
        for key in fields:
            for value in (float(original[key]), str(original[key]), False):
                with self.subTest(key=key, value=value):
                    self.value = deepcopy(original)
                    self.value[key] = value
                    self.save_config()
                    with self.assertRaises(ValueError):
                        batch.config(self.home)
        for value in (2.0, '2', True):
            with self.subTest(candidate_count=value):
                self.value = deepcopy(original)
                self.value['ordering_slots'][0]['candidate_count'] = value
                self.save_config()
                with self.assertRaisesRegex(ValueError, '^both_types_require_same_distinct_roots$'):
                    batch.config(self.home)

    def test_config_rejects_unsafe_batch_and_ordering_slot_identifiers(self):
        original = deepcopy(self.value)
        for value in ('../escape', 'contains/slash', '', '_', 'has space', None):
            with self.subTest(batch_id=value):
                self.value = deepcopy(original)
                self.value['batch_id'] = value
                self.save_config()
                with self.assertRaisesRegex(ValueError, '^mixed_batch_field_types_or_source_mode$'):
                    batch.config(self.home)
        self.value = deepcopy(original)
        self.value['ordering_slots'][0]['slot_id'] = '../escape'
        self.save_config()
        with self.assertRaisesRegex(ValueError, '^both_types_require_same_distinct_roots$'):
            batch.config(self.home)

    def test_first_generation_lane_exception_still_schedules_ordering_and_never_completes(self):
        driver, events = self.install_generation(multi_error=RuntimeError('synthetic multiselect failure'))
        with self.assertRaisesRegex(ValueError, '^batch_generation_incomplete$'):
            batch.generate(self.home)
        self.assertEqual(events, ['multi_select', 'ordering'])
        driver.generate.assert_called_once_with(self.home / 'multiselect')
        self.ordering_generator.assert_called_once_with(self.home)
        result = common.read(self.home / 'BATCH_GENERATION.ERROR.json')
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(result['completed_types'], ['ordering'])
        self.assertEqual(result['errors'], [{'question_type': 'multi_select', 'error': 'synthetic multiselect failure'}])
        self.assertEqual(result['http_attempts'], 2)
        self.assertIs(result['automatic_retry'], False)
        self.assertTrue((self.home / 'BATCH_GENERATION.STARTED.json').exists())
        self.assertTrue((self.home / 'ordering/GENERATION.COMPLETE.json').exists())
        self.assertFalse((self.home / 'BATCH_GENERATION.COMPLETE.json').exists())

    def test_ordering_generation_failure_preserves_successful_multiselect_without_batch_complete(self):
        _, events = self.install_generation(ordering_error=ValueError('synthetic ordering failure'))
        with self.assertRaisesRegex(ValueError, '^batch_generation_incomplete$'):
            batch.generate(self.home)
        self.assertEqual(events, ['multi_select', 'ordering'])
        result = common.read(self.home / 'BATCH_GENERATION.ERROR.json')
        self.assertEqual(result['completed_types'], ['multi_select'])
        self.assertEqual(result['errors'], [{'question_type': 'ordering', 'error': 'synthetic ordering failure'}])
        self.assertFalse((self.home / 'BATCH_GENERATION.COMPLETE.json').exists())

    def test_both_generation_errors_are_saved_in_schedule_order(self):
        _, events = self.install_generation(multi_error=RuntimeError('first'), ordering_error=RuntimeError('second'))
        with self.assertRaisesRegex(ValueError, '^batch_generation_incomplete$'):
            batch.generate(self.home)
        self.assertEqual(events, ['multi_select', 'ordering'])
        result = common.read(self.home / 'BATCH_GENERATION.ERROR.json')
        self.assertEqual(result['completed_types'], [])
        self.assertEqual(result['errors'], [{'question_type': 'multi_select', 'error': 'first'},
                                           {'question_type': 'ordering', 'error': 'second'}])
        self.assertFalse((self.home / 'BATCH_GENERATION.COMPLETE.json').exists())

    def test_successful_generation_binds_both_lane_completions(self):
        _, events = self.install_generation()
        result = batch.generate(self.home)
        self.assertEqual(events, ['multi_select', 'ordering'])
        self.assertEqual(result['status'], 'GENERATED')
        self.assertEqual(result['completed_types'], ['multi_select', 'ordering'])
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['http_attempts'], 2)
        self.assertEqual(result['files'], [common.bind(self.home / 'multiselect/INITIAL_GENERATION.COMPLETE.json'),
                                           common.bind(self.home / 'ordering/GENERATION.COMPLETE.json')])
        self.assertEqual(common.read(self.home / 'BATCH_GENERATION.COMPLETE.json'), result)

    def test_generation_cannot_replay_started_batch(self):
        driver, events = self.install_generation()
        common.write(self.home / 'BATCH_GENERATION.STARTED.json', {'synthetic': True})
        with self.assertRaisesRegex(ValueError, '^batch_generation_no_replay$'):
            batch.generate(self.home)
        self.assertEqual(events, [])
        driver.generate.assert_not_called()
        self.ordering_generator.assert_not_called()
        self.assertFalse((self.home / 'BATCH_GENERATION.COMPLETE.json').exists())

    def test_missing_multiselect_review_stops_before_mixed_pool_write(self):
        self.install_reviewed_artifacts()
        self.multi_review_path.unlink()
        with patch.object(batch, 'write', wraps=common.write) as writes:
            with self.assertRaises(FileNotFoundError):
                batch.freeze(self.home)
        writes.assert_not_called()
        self.assertEqual(self.review_events, ['multi_select'])
        self.multi_content.assert_not_called()
        self.ordering_review_gate.assert_not_called()
        self.assert_no_freeze()

    def test_missing_ordering_review_stops_before_mixed_pool_write(self):
        self.install_reviewed_artifacts()
        self.ordering_review_path.unlink()
        with patch.object(batch, 'write', wraps=common.write) as writes:
            with self.assertRaises(FileNotFoundError):
                batch.freeze(self.home)
        writes.assert_not_called()
        self.assertEqual(self.review_events, ['multi_select', 'ordering'])
        self.assert_no_freeze()

    def test_freeze_requires_completed_generation_for_both_types(self):
        self.install_reviewed_artifacts()
        path = self.home / 'BATCH_GENERATION.COMPLETE.json'
        original = common.read(path)
        for field, value in (('status', 'ERROR'), ('completed_types', ['multi_select']),
                             ('completed_types', ['ordering', 'multi_select'])):
            with self.subTest(field=field, value=value):
                result = deepcopy(original)
                result[field] = value
                replace(path, result)
                with self.assertRaisesRegex(ValueError, '^both_generated_types_required$'):
                    batch.freeze(self.home)
                self.assert_no_freeze()
        self.multi_review_gate.assert_not_called()
        self.ordering_review_gate.assert_not_called()

    def test_generation_verification_rejects_error_marker_and_missing_lane_binding(self):
        self.install_reviewed_artifacts()
        batch.verify_generation(self.home)
        path = self.home / 'BATCH_GENERATION.COMPLETE.json'
        generation = common.read(path)
        common.write(self.home / 'BATCH_GENERATION.ERROR.json', {'synthetic': True})
        with self.assertRaisesRegex(ValueError, '^both_generated_types_required$'):
            batch.verify_generation(self.home)
        (self.home / 'BATCH_GENERATION.ERROR.json').unlink()
        for lane in ('multiselect', 'ordering'):
            with self.subTest(lane=lane):
                changed = deepcopy(generation)
                changed['files'] = [row for row in changed['files'] if Path(row['path']).parent.name != lane]
                replace(path, changed)
                with self.assertRaisesRegex(ValueError, '^frozen_binding_coverage$'):
                    batch.verify_generation(self.home)

    def test_generation_verification_requires_actual_http_receipts(self):
        self.install_reviewed_artifacts()
        path = self.home / 'BATCH_GENERATION.COMPLETE.json'
        original = common.read(path)
        for count in (1, 0.0, False):
            with self.subTest(http_attempts=count):
                changed = deepcopy(original)
                changed['http_attempts'] = count
                replace(path, changed)
                with self.assertRaisesRegex(ValueError, '^mixed_http_budget_or_receipt_changed$'):
                    batch.verify_generation(self.home)

    def test_invalid_sources_in_either_lane_block_freeze_and_frozen_verification_before_private_access(self):
        self.install_reviewed_artifacts()
        batch.freeze(self.home)
        for action in (batch.freeze, batch.verify_frozen):
            for invalid_lane in ('multi_select', 'ordering'):
                with self.subTest(action=action.__name__, invalid_lane=invalid_lane):
                    events = []
                    self.plan_gate.reset_mock()
                    self.multi_review_gate.reset_mock()
                    self.multi_content.reset_mock()
                    self.ordering_review_gate.reset_mock()

                    def validate_multi(home):
                        self.assertEqual(home, self.home / 'multiselect')
                        events.append('multi_select_sources')
                        if invalid_lane == 'multi_select':
                            raise ValueError('synthetic_invalid_multi_select_sources')

                    def validate_ordering(home):
                        self.assertEqual(home, self.home)
                        events.append('ordering_sources')
                        if invalid_lane == 'ordering':
                            raise ValueError('synthetic_invalid_ordering_sources')

                    schema = Mock(validate_final_sources=Mock(side_effect=validate_multi))

                    def load(name):
                        self.assertEqual(name, 'review_schema', 'Invalid source review loaded a private replay engine')
                        return schema

                    self.multi_loader.side_effect = load
                    self.source_gate.side_effect = VALIDATE_REVIEW_SOURCES
                    self.ordering_source_gate.side_effect = validate_ordering
                    with patch.object(batch, 'verify_generation', side_effect=self.forbidden) as generation, \
                         patch.object(batch, 'reviewed_pools', side_effect=self.forbidden) as pools, \
                         patch.object(batch, 'read', side_effect=self.forbidden) as private_read, \
                         patch.object(batch, 'write', side_effect=self.forbidden) as writes:
                        with self.assertRaisesRegex(ValueError, '^synthetic_invalid_' + invalid_lane + '_sources$'):
                            action(self.home)
                    self.assertEqual(events, ['multi_select_sources'] +
                                     (['ordering_sources'] if invalid_lane == 'ordering' else []))
                    self.plan_gate.assert_not_called()
                    generation.assert_not_called()
                    pools.assert_not_called()
                    private_read.assert_not_called()
                    writes.assert_not_called()
                    self.multi_review_gate.assert_not_called()
                    self.multi_content.assert_not_called()
                    self.ordering_review_gate.assert_not_called()

    def test_both_public_source_checks_precede_generation_and_private_pool_replay(self):
        self.install_reviewed_artifacts()
        evaluate = self.multi_loader('evaluate')
        for action in (batch.freeze, batch.verify_frozen):
            with self.subTest(action=action.__name__):
                events = []
                schema = Mock(validate_final_sources=Mock(side_effect=lambda home: events.append('multi_select_sources')))

                def load(name):
                    return schema if name == 'review_schema' else evaluate

                self.multi_loader.side_effect = load
                self.source_gate.side_effect = VALIDATE_REVIEW_SOURCES
                self.ordering_source_gate.side_effect = lambda home: events.append('ordering_sources')
                self.plan_gate.side_effect = lambda home: events.append('plan_private_replay')
                original_generation = batch.verify_generation
                original_pools = batch.reviewed_pools

                def generation(home):
                    events.append('generation_private_bindings')
                    return original_generation(home)

                def pools(home):
                    events.append('reviewed_private_pools')
                    return original_pools(home)

                with patch.object(batch, 'verify_generation', side_effect=generation), \
                     patch.object(batch, 'reviewed_pools', side_effect=pools):
                    action(self.home)
                self.assertEqual(events[:5], ['multi_select_sources', 'ordering_sources',
                                            'plan_private_replay', 'generation_private_bindings',
                                            'reviewed_private_pools'])

    def test_successful_freeze_binds_both_reviews_and_typed_pool(self):
        self.install_reviewed_artifacts()
        result = batch.freeze(self.home)
        self.assertEqual(self.review_events, ['multi_select', 'ordering'])
        self.assertEqual(result['by_type'], {'multi_select': 1, 'ordering': 1})
        self.assertIs(result['before_any_solver'], True)
        self.assertIs(result['solver_scores_used'], False)
        self.assertIs(result['training_allowed'], False)
        pool = common.read(self.home / 'EVALUATION_POOL.json')
        self.assertEqual(pool['questions'], common.read(self.multi_pool_path)['questions']
                         + common.read(self.ordering_pool_path)['questions'])
        paths = {record['path'] for record in result['files']}
        for path in (self.multi_review_path, self.ordering_review_path,
                     self.home / 'EVALUATION_POOL.json', self.home / 'CONFIG.json',
                     self.home / 'BATCH_PREPARATION.COMPLETE.json',
                     self.home / 'BATCH_GENERATION.COMPLETE.json'):
            self.assertIn(str(path.resolve()), paths)
        common.verify_bindings(result['files'])
        self.assertEqual(batch.verify_frozen(self.home), result)

    def test_freeze_rejects_wrong_question_type_in_either_reviewed_lane(self):
        self.install_reviewed_artifacts()
        for path, wrong_type in ((self.multi_pool_path, 'ordering'), (self.ordering_pool_path, 'multi_select')):
            with self.subTest(lane=path.parent.name):
                original = common.read(path)
                changed = deepcopy(original)
                changed['questions'][0]['question_type'] = wrong_type
                replace(path, changed)
                with self.assertRaisesRegex(ValueError, '^lane_question_type_changed$'):
                    batch.freeze(self.home)
                replace(path, original)
                self.assert_no_freeze()

    def test_freeze_rejects_duplicate_candidate_ids_across_types(self):
        self.install_reviewed_artifacts()
        ordered = common.read(self.ordering_pool_path)
        ordered['questions'][0]['candidate_id'] = common.read(self.multi_pool_path)['questions'][0]['candidate_id']
        replace(self.ordering_pool_path, ordered)
        with self.assertRaisesRegex(ValueError, '^mixed_candidate_ids_or_budget$'):
            batch.freeze(self.home)
        self.assert_no_freeze()

    def test_freeze_cannot_start_after_any_solver_started(self):
        self.install_reviewed_artifacts()
        common.write(self.home / 'solver/trained/STARTED.json', {'synthetic': True})
        with self.assertRaisesRegex(ValueError, '^cannot_freeze_after_solver$'):
            batch.freeze(self.home)
        self.multi_review_gate.assert_not_called()
        self.ordering_review_gate.assert_not_called()
        self.assert_no_freeze()

    def test_verify_frozen_rejects_changed_typed_pool_even_after_hash_rebinding(self):
        self.install_reviewed_artifacts()
        batch.freeze(self.home)
        pool_path = self.home / 'EVALUATION_POOL.json'
        pool = common.read(pool_path)
        pool['questions'][0]['question_type'] = 'ordering'
        replace(pool_path, pool)
        self.rebind_frozen_file(pool_path)
        with self.assertRaisesRegex(ValueError, '^mixed_frozen_pool_changed$'):
            batch.verify_frozen(self.home)

    def test_verify_frozen_rejects_mismatched_type_counts(self):
        self.install_reviewed_artifacts()
        batch.freeze(self.home)
        path = self.home / 'BATCH_REVIEW.COMPLETE.json'
        frozen = common.read(path)
        frozen['by_type'] = {'multi_select': 2}
        replace(path, frozen)
        with self.assertRaisesRegex(ValueError, '^mixed_frozen_pool_changed$'):
            batch.verify_frozen(self.home)

    def test_verify_frozen_rejects_changed_pool_binding_before_review_access(self):
        self.install_reviewed_artifacts()
        batch.freeze(self.home)
        self.multi_review_gate.reset_mock()
        self.ordering_review_gate.reset_mock()
        pool_path = self.home / 'EVALUATION_POOL.json'
        pool = common.read(pool_path)
        pool['questions'][0]['question'] += ' changed'
        replace(pool_path, pool)
        with self.assertRaisesRegex(ValueError, '^frozen_file_changed:'):
            batch.verify_frozen(self.home)
        self.multi_review_gate.assert_not_called()
        self.ordering_review_gate.assert_not_called()

    def test_verify_frozen_rejects_changed_child_review_binding(self):
        self.install_reviewed_artifacts()
        batch.freeze(self.home)
        marker = common.read(self.ordering_review_path)
        marker['reviewer'] = 'different-synthetic-reviewer'
        replace(self.ordering_review_path, marker)
        with self.assertRaisesRegex(ValueError, '^frozen_file_changed:'):
            batch.verify_frozen(self.home)

    def test_verify_frozen_rejects_missing_required_binding(self):
        self.install_reviewed_artifacts()
        batch.freeze(self.home)
        path = self.home / 'BATCH_REVIEW.COMPLETE.json'
        original = common.read(path)
        for removed in ('EVALUATION_POOL.json', 'CONFIG.json', 'BATCH_PREPARATION.COMPLETE.json', 'BATCH_GENERATION.COMPLETE.json',
                        'FINAL_CONTENT_REVIEW.COMPLETE.json', 'CONTENT_REVIEW.COMPLETE.json'):
            with self.subTest(removed=removed):
                frozen = deepcopy(original)
                frozen['files'] = [row for row in frozen['files'] if Path(row['path']).name != removed]
                replace(path, frozen)
                with self.assertRaises(ValueError):
                    batch.verify_frozen(self.home)


if __name__ == '__main__':
    unittest.main()
