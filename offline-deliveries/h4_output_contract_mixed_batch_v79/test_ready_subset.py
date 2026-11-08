"""Offline checks for the narrow v77 READY subset and its pre-writer gate."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import batch
import source_subset as subset
from common import HERE, read, write


class ReadySubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Real frozen review/pin replay once; no HTTP or execution is involved.
        cls.summary = subset.verified_summary()
        cls.production = read(HERE / 'CONFIG.json')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.home = Path(self.temp.name).resolve()
        write(self.home / 'CONFIG.json', self.production)
        self.network_patch = patch('socket.socket.connect', side_effect=AssertionError('Real network prohibited'))
        self.process_patch = patch('subprocess.run', side_effect=AssertionError('Real subprocess prohibited'))
        self.network_patch.start(); self.process = self.process_patch.start()

    def tearDown(self):
        self.process_patch.stop(); self.network_patch.stop(); self.temp.cleanup()

    def save(self, value):
        (self.home / 'CONFIG.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    @contextmanager
    def returned_source_review(self, summary):
        loader = SimpleNamespace(exec_module=Mock())
        module = SimpleNamespace(verify_frozen=Mock(return_value=summary))
        with (patch.object(subset.importlib.util, 'spec_from_file_location', return_value=SimpleNamespace(loader=loader)),
              patch.object(subset.importlib.util, 'module_from_spec', return_value=module)):
            yield module

    def assert_prepare_blocked(self):
        with (patch.object(Path, 'mkdir', side_effect=AssertionError('Invalid subset must not mkdir')) as mkdir,
              patch.object(batch, 'load_multi') as multi,
              patch.object(batch.ordering_generation, 'prepare') as ordering):
            with self.assertRaises(ValueError):
                batch.prepare(self.home)
            mkdir.assert_not_called(); multi.assert_not_called(); ordering.assert_not_called()
        self.assertFalse((self.home / 'BATCH_PREPARATION.STARTED.json').exists())
        self.assertFalse((self.home / 'multiselect').exists()); self.assertFalse((self.home / 'ordering').exists())
        self.process.assert_not_called()

    def test_real_production_config_accepts_only_two_ready_roots_and_typed_budgets(self):
        value = batch.config(HERE)
        selected = ['cc-train-episode-194259', 'cc-train-episode-130221']
        self.assertEqual([s['root_id'] for s in value['multiselect_config']['slots']], selected)
        self.assertEqual([s['root_id'] for s in value['ordering_slots']], selected)
        self.assertEqual(value['source_consistency']['selected_roots'], selected)
        self.assertEqual(set(value['source_consistency']['excluded_held_roots']),
                         {'cc-train-episode-231899', 'cc-train-episode-138343'})
        self.assertEqual((value['max_generation_http'], value['max_candidates'], value['max_solver_calls_per_model']), (6, 14, 228))
        m = value['multiselect_config']
        self.assertEqual((m['max_http'], m['max_questions'], m['max_calls_per_model']), (4, 10, 180))
        self.assertEqual(m['new_child_target_distribution'], {'2': 2, '3': 2, '4': 2, '5': 2})
        self.assertEqual(m['control_target_distribution'], {'6': 2})
        self.assertIs(value['source_consistency']['source_review_is_question_answer_certification'], False)

    def test_selected_root_hold_is_rejected_before_any_preparation(self):
        for root in subset.ROOTS:
            with self.subTest(root=root):
                summary = deepcopy(self.summary)
                next(r for r in summary['roots'] if r['root_id'] == root)['decision'] = 'HOLD'
                with self.returned_source_review(summary):
                    with self.assertRaisesRegex(ValueError, 'selected_source_not_ready'):
                        subset.verified_summary()
                    self.assert_prepare_blocked()

    def test_selected_source_binding_must_match_frozen_full_source(self):
        for field, wrong in (('sha256', '0' * 64), ('path', '/tmp/shortened-source.json')):
            with self.subTest(field=field):
                summary = deepcopy(self.summary)
                next(r for r in summary['roots'] if r['root_id'] == subset.ROOTS[0])['source'][field] = wrong
                with patch.object(subset, 'verified_summary', return_value=summary):
                    with self.assertRaisesRegex(ValueError, 'selected_source_binding_changed'):
                        subset.validate(self.production)
                    self.assert_prepare_blocked()

    def test_v77_code_or_completion_pin_change_blocks_before_source_gate_import(self):
        original_sha = subset.sha
        for pinned in subset.PINS:
            if 'h4_source_consistency_minfix_v77' not in pinned:
                continue
            with self.subTest(path=pinned):
                def changed_sha(path):
                    return '0' * 64 if str(path) == pinned else original_sha(path)
                with (patch.object(subset, 'sha', side_effect=changed_sha),
                      patch.object(subset.importlib.util, 'spec_from_file_location') as importer):
                    with self.assertRaisesRegex(ValueError, 'upstream_binding_changed'):
                        subset.verified_summary()
                    self.assert_prepare_blocked()
                    importer.assert_not_called()

    def test_cannot_insert_held_root_remove_ready_root_or_reorder_selection(self):
        changes = [
            lambda c: c['multiselect_config']['slots'][0].update(root_id='cc-train-episode-231899'),
            lambda c: c['ordering_slots'][0].update(root_id='cc-train-episode-138343'),
            lambda c: c['ordering_slots'].pop(),
            lambda c: c['multiselect_config']['slots'].reverse(),
            lambda c: c['source_consistency']['selected_roots'].append('cc-train-episode-231899'),
            lambda c: c['source_consistency'].update(excluded_held_roots=[]),
        ]
        with patch.object(subset, 'verified_summary', return_value=self.summary):
            for change in changes:
                with self.subTest(change=change):
                    value = deepcopy(self.production); change(value); self.save(value)
                    self.assert_prepare_blocked()

    def test_gold_distribution_control_or_raw_gold_injection_cannot_change(self):
        changes = [
            lambda c: c['multiselect_config']['slots'][0].update(target_counts=[2, 3, 4, 6]),
            lambda c: c['multiselect_config']['slots'][0].update(control_count=5),
            lambda c: c['multiselect_config'].update(control_gold_count=5),
            lambda c: c['multiselect_config'].update(control_is_new_candidate=True),
            lambda c: c['multiselect_config']['target_distribution'].update({'2': 1, '3': 3}),
            lambda c: c['multiselect_config']['slots'][0].update(gold=['A', 'B']),
        ]
        with patch.object(subset, 'verified_summary', return_value=self.summary):
            for change in changes:
                with self.subTest(change=change):
                    value = deepcopy(self.production); change(value); self.save(value)
                    self.assert_prepare_blocked()

    def test_budget_even_coordinated_change_is_rejected(self):
        changes = [lambda c: c.update(max_generation_http=7), lambda c: c.update(max_candidates=15),
            lambda c: c.update(max_solver_calls_per_model=229),
            lambda c: c['multiselect_config'].update(max_http=5),
            lambda c: c['multiselect_config'].update(max_questions=11),
            lambda c: c['ordering_slots'][0].update(candidate_count=3)]
        def coordinated(c):
            c['max_generation_http'] = 8; c['multiselect_config']['max_http'] = 6
        changes.append(coordinated)
        with patch.object(subset, 'verified_summary', return_value=self.summary):
            for change in changes:
                with self.subTest(change=change):
                    value = deepcopy(self.production); change(value); self.save(value)
                    self.assert_prepare_blocked()

    def test_unrelated_generation_review_and_selection_policies_are_immutable(self):
        changes = [lambda c: c.update(training_allowed=True), lambda c: c.update(transport_retries=1),
            lambda c: c.update(ordering_review_required=False),
            lambda c: c.update(source_mode='trimmed_sources'),
            lambda c: c['multiselect_config'].update(selection_rule='all_questions'),
            lambda c: c['multiselect_config'].update(program_fixed_citations=False),
            lambda c: c['multiselect_config'].update(review_schema_version='legacy'),
            lambda c: c['multiselect_config']['slots'][0].update(source_sha256='0' * 64),
            lambda c: c['ordering_slots'][0].update(history_path='/tmp/alternate-history.json'),
            lambda c: c['source_consistency']['upstream_complete'].update(sha256='0' * 64)]
        with patch.object(subset, 'verified_summary', return_value=self.summary):
            for change in changes:
                with self.subTest(change=change):
                    value = deepcopy(self.production); change(value); self.save(value)
                    self.assert_prepare_blocked()

    def test_ready_prepare_uses_derived_config_in_both_lanes_with_mock_writers(self):
        def prepare_multi(home):
            value = read(home / 'CONFIG.json')
            self.assertEqual([s['root_id'] for s in value['slots']], list(subset.ROOTS))
            self.assertEqual(value['max_questions'], 10)
            write(home / 'EXECUTION_PLAN.json', {'synthetic': True, 'http_attempts': 0})
        def prepare_ordering(home):
            value = batch.config(home)
            self.assertEqual([s['root_id'] for s in value['ordering_slots']], list(subset.ROOTS))
            write(home / 'ordering/PREPARATION.COMPLETE.json', {'synthetic': True, 'http_attempts': 0})
        multi = SimpleNamespace(prepare=Mock(side_effect=prepare_multi))
        with (patch.object(subset, 'verified_summary', return_value=self.summary),
              patch.object(batch, 'load_multi', return_value=multi) as loader,
              patch.object(batch.ordering_generation, 'prepare', side_effect=prepare_ordering) as ordering):
            result = batch.prepare(self.home)
        self.assertEqual(result['status'], 'PREPARED')
        self.assertEqual(result['completed_types'], ['multi_select', 'ordering'])
        self.assertIs(result['training_allowed'], False)
        multi.prepare.assert_called_once_with(self.home / 'multiselect')
        ordering.assert_called_once_with(self.home); loader.assert_called_once_with('generation_driver')
        self.assertEqual(read(self.home / 'BATCH_PREPARATION.STARTED.json')['real_http_calls'], 0)
        self.process.assert_not_called()


if __name__ == '__main__':
    unittest.main()
