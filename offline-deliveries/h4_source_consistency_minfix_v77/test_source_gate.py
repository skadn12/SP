"""Offline source-review and guarded-launch tests using temporary artifacts.

No semantic judgment is inferred by these tests. Synthetic review decisions
exercise coverage, disclosures, exact citations, immutability and launch gates.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import guarded_batch
import source_gate as gate


def replace(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def fixture(root):
    home = root / 'audit'; engine = root / 'engine'
    source_path = root / 'SOURCE.json'; pool_path = root / 'PRIVATE_POOL.json'
    source = {'root_id': 'synthetic-root', 'turns': [
        {'turn_ref': 'P1:T1', 'session_ref': 'parent-session', 'scene_ref': None,
         'speaker': 'Ada', 'text': 'I plan to book the north hall.', 'origin': 'source_grounded'},
        {'turn_ref': 'P1:T2', 'session_ref': 'parent-session', 'scene_ref': None,
         'speaker': 'Ben', 'text': 'Please confirm the room before announcing it.', 'origin': 'source_grounded'},
        {'turn_ref': 'S1:T1', 'session_ref': 'extension-session', 'scene_ref': 'extension-scene',
         'speaker': 'Ada', 'text': 'I booked the north hall today.', 'origin': 'generated_extension'},
        {'turn_ref': 'S1:T2', 'session_ref': 'extension-session', 'scene_ref': 'extension-scene',
         'speaker': 'Ben', 'text': 'Thank you for confirming the room.', 'origin': 'generated_extension'},
    ]}
    gate.write(source_path, source)
    gate.write(pool_path, {'questions': [{
        'candidate_id': 'synthetic-question', 'root_id': 'synthetic-root', 'question_type': 'multi_select',
        'source_sha256': gate.sha(source_path), 'gold': ['A', 'C'],
        'option_evidence': {'A': {'turn_ref': 'P1:T1', 'quote': 'north hall'}}}]})
    gate.write(engine / 'CONFIG.json', {'baseline_pool': str(pool_path), 'max_candidates': 7,
        'ordering_slots': [{'root_id': 'synthetic-root', 'source_path': str(source_path),
                            'source_sha256': gate.sha(source_path)}],
        'multiselect_config': {'slots': [{'root_id': 'synthetic-root'}]}})
    (engine / 'batch.py').write_text('# Synthetic upstream: subprocess execution is mocked.\n')
    review_path = home / 'reviews' / 'root_REVIEW.json'
    review = {'schema_version': 'dialogue_consistency_v1', 'root_id': 'synthetic-root',
        'reviewer': 'Synthetic test reviewer', 'created_at_utc': gate.now(), 'source': gate.bind(source_path),
        'full_source_read': True, 'current_question_gold_read': False, 'current_solver_scores_read': False,
        'historical_question_gold_known': True, 'fresh_blind_review': False,
        'review_scope': 'source_consistency_only', 'turn_refs_read': [t['turn_ref'] for t in source['turns']],
        'checks_completed': {key: True for key in gate.CHECKS},
        'units': [{'unit_id': uid, 'origin': rows[0]['origin'],
                   'turn_refs': [t['turn_ref'] for t in rows], 'decision': 'PASS',
                   'reason': 'Synthetic coherent plan-to-confirmation sequence.', 'finding_ids': []}
                  for uid, rows in gate.units_for(source).items()],
        'findings': [], 'decision': 'READY', 'limitations': 'Synthetic review; not question answer certification.'}
    gate.write(review_path, review)
    config = {'schema_version': 'source_consistency_gate_v1',
        'root_policy': 'hold_entire_full_source_if_any_blocking_unit',
        'training_allowed': False, 'automatic_retry': False, 'source_edits_allowed': False,
        'source_config': gate.bind(engine / 'CONFIG.json'), 'upstream_engine': str(engine),
        'baseline_pool': gate.bind(pool_path), 'roots': [{'root_id': 'synthetic-root',
            'source': gate.bind(source_path), 'review_path': str(review_path)}]}
    gate.write(home / 'CONFIG.json', config)
    return home, engine, source_path, pool_path, review_path, review


class SourceGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        (self.home, self.engine, self.source_path, self.pool_path,
         self.review_path, self.review) = fixture(self.root)
        self.network_patch = patch('socket.socket.connect', side_effect=AssertionError('Real network prohibited'))
        self.process_patch = patch.object(guarded_batch.subprocess, 'run', side_effect=AssertionError('Real subprocess prohibited'))
        self.popen_patch = patch.object(guarded_batch.subprocess, 'Popen', side_effect=AssertionError('Real Popen prohibited'))
        self.network_patch.start(); self.process = self.process_patch.start(); self.popen = self.popen_patch.start()

    def tearDown(self):
        self.popen_patch.stop(); self.process_patch.stop(); self.network_patch.stop(); self.temp.cleanup()

    def save(self):
        replace(self.review_path, self.review)

    def add_finding(self, severity='WARNING', both_units=False):
        units = ['parent-session', 'extension-scene'] if both_units else ['parent-session']
        evidence = [{'turn_ref': 'P1:T1', 'quote': 'I plan to book the north hall.'}]
        if both_units:
            evidence.append({'turn_ref': 'S1:T1', 'quote': 'I booked the north hall today.'})
        self.review['findings'] = [{'finding_id': 'SYNTHETIC-01', 'severity': severity,
            'kind': 'cross_scene_continuity', 'unit_ids': units, 'evidence': evidence,
            'reason': 'Synthetic author judgment used only to test gate enforcement.',
            'proposed_action': 'Hold for independent review.' if severity == 'BLOCKING' else 'Keep this caveat.'}]
        for unit in self.review['units']:
            if unit['unit_id'] in units:
                unit['decision'] = 'HOLD' if severity == 'BLOCKING' else 'WARN'
                unit['finding_ids'] = ['SYNTHETIC-01']
        self.review['decision'] = 'HOLD' if severity == 'BLOCKING' else 'READY'
        self.save()

    def assert_rejects_before_private(self, action):
        original_read, original_sha = gate.read, gate.sha
        touches = []
        def read(path):
            if Path(path).resolve() == self.pool_path:
                touches.append('read')
                self.fail('Invalid/stale review opened the baseline question pool')
            return original_read(path)
        def sha(path):
            if Path(path).resolve() == self.pool_path:
                touches.append('hash')
                self.fail('Invalid/stale review hashed the baseline question pool')
            return original_sha(path)
        with patch.object(gate, 'read', read), patch.object(gate, 'sha', sha):
            with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
                action()
        self.assertEqual(touches, [])

    def test_complete_ready_review_remains_distinct_from_question_certification(self):
        before = gate.bind(self.pool_path)
        summary = gate.verify_reviews(self.home)
        self.assertEqual(summary['ready_roots'], ['synthetic-root']); self.assertEqual(summary['held_roots'], [])
        self.assertEqual(summary['roots'][0]['turn_count'], 4)
        self.assertEqual(summary['roots'][0]['unit_decisions'], {'PASS': 2})
        self.assertIs(summary['ready_is_question_answer_certification'], False)
        self.assertEqual(summary['question_or_gold_edits'], [])
        self.assertEqual(gate.bind(self.pool_path), before)

    def test_strict_disclosures_reject_missing_boolean_integer_or_wrong_value(self):
        original = deepcopy(self.review)
        fields = {'full_source_read': True, 'current_question_gold_read': False,
            'current_solver_scores_read': False, 'historical_question_gold_known': True,
            'fresh_blind_review': False}
        for key, expected in fields.items():
            for value in ('MISSING', int(expected), not expected):
                with self.subTest(key=key, value=value):
                    self.review = deepcopy(original)
                    if value == 'MISSING': self.review.pop(key)
                    else: self.review[key] = value
                    self.save(); self.assert_rejects_before_private(lambda: gate.finalize(self.home))

    def test_full_turn_order_unit_coverage_and_checks_cannot_be_partial(self):
        original = deepcopy(self.review)
        changes = [lambda r: r['turn_refs_read'].pop(), lambda r: r['turn_refs_read'].reverse(),
            lambda r: r['units'].pop(), lambda r: r['units'].reverse(),
            lambda r: r['units'][0]['turn_refs'].pop(),
            lambda r: r['units'][0].update(origin='generated_extension'),
            lambda r: r['checks_completed'].pop('time'),
            lambda r: r['checks_completed'].update(location=1),
            lambda r: r['checks_completed'].update(actor_identity=False)]
        for change in changes:
            with self.subTest(change=change):
                self.review = deepcopy(original); change(self.review); self.save()
                self.assert_rejects_before_private(lambda: gate.finalize(self.home))

    def test_findings_require_exact_quotes_and_evidence_for_every_involved_unit(self):
        self.add_finding(both_units=True); original = deepcopy(self.review)
        self.assertEqual(gate.verify_reviews(self.home)['roots'][0]['unit_decisions'], {'WARN': 2})
        changes = [lambda f: f['evidence'][0].update(quote='I booked the south hall.'),
            lambda f: f['evidence'][0].update(turn_ref='unknown'),
            lambda f: f['evidence'].pop(), lambda f: f['unit_ids'].pop(),
            lambda f: f.update(evidence=[]), lambda f: f.update(proposed_action=''),
            lambda f: f.update(unit_ids=['parent-session', 'unknown'])]
        for change in changes:
            with self.subTest(change=change):
                self.review = deepcopy(original); change(self.review['findings'][0]); self.save()
                self.assert_rejects_before_private(lambda: gate.finalize(self.home))

    def test_blocking_finding_cannot_be_downgraded_by_root_or_unit_decision(self):
        self.add_finding('BLOCKING', both_units=True); original = deepcopy(self.review)
        summary = gate.verify_reviews(self.home)
        self.assertEqual(summary['held_roots'], ['synthetic-root'])
        self.assertEqual(summary['roots'][0]['held_units'], ['parent-session', 'extension-scene'])
        changes = [lambda r: r.update(decision='READY'),
            lambda r: r['units'][0].update(decision='WARN'),
            lambda r: r['units'][0].update(decision='PASS'),
            lambda r: r['units'][0].update(finding_ids=[])]
        for change in changes:
            with self.subTest(change=change):
                self.review = deepcopy(original); change(self.review); self.save()
                self.assert_rejects_before_private(lambda: gate.finalize(self.home))

    def test_warning_requires_warn_but_keeps_root_ready(self):
        self.add_finding(); self.assertEqual(gate.verify_reviews(self.home)['ready_roots'], ['synthetic-root'])
        self.review['units'][0]['decision'] = 'PASS'; self.save()
        self.assert_rejects_before_private(lambda: gate.finalize(self.home))

    def test_unknown_or_missing_review_files_reject_before_private_access(self):
        extra = self.home / 'reviews' / 'unregistered.json'; gate.write(extra, self.review)
        self.assert_rejects_before_private(lambda: gate.finalize(self.home))
        extra.unlink(); self.review_path.unlink()
        self.assert_rejects_before_private(lambda: gate.finalize(self.home))

    def test_review_time_requires_timezone_and_no_future_timestamp(self):
        for value in (None, 'not a timestamp', datetime.now().isoformat(),
                      (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()):
            with self.subTest(value=value):
                self.review['created_at_utc'] = value; self.save()
                self.assert_rejects_before_private(lambda: gate.finalize(self.home))

    def test_source_sha_and_review_source_binding_are_enforced(self):
        original = deepcopy(self.review)
        self.review['source']['sha256'] = '0' * 64; self.save()
        self.assert_rejects_before_private(lambda: gate.finalize(self.home))
        self.review = original; self.save()
        source = gate.read(self.source_path); source['turns'][0]['text'] = 'Changed source.'
        replace(self.source_path, source)
        self.assert_rejects_before_private(lambda: gate.finalize(self.home))

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers(self):
        for raw in ('{"full_source_read":true,"full_source_read":false}', '{"value":NaN}'):
            with self.subTest(raw=raw):
                self.review_path.write_text(raw)
                self.assert_rejects_before_private(lambda: gate.finalize(self.home))

    def test_finalize_freezes_input_coverage_and_is_append_only(self):
        before = gate.bind(self.pool_path); summary = gate.finalize(self.home)
        self.assertEqual(gate.verify_frozen(self.home), summary)
        self.assertEqual(gate.require_ready(self.home), summary)
        impact = gate.read(self.home / 'QUESTION_IMPACT.json')
        self.assertEqual(impact['questions'][0]['classification'], 'NO_SOURCE_GATE_HOLD')
        self.assertIs(impact['questions'][0]['question_answer_recertified'], False)
        self.assertIs(impact['existing_pool_modified'], False)
        self.assertEqual(gate.bind(self.pool_path), before)
        with self.assertRaisesRegex(ValueError, 'already_finalized'):
            gate.finalize(self.home)

    def test_held_source_impact_marks_dependency_without_editing_gold(self):
        self.add_finding('BLOCKING'); before = gate.bind(self.pool_path)
        gate.finalize(self.home); gate.verify_frozen(self.home)
        impact = gate.read(self.home / 'QUESTION_IMPACT.json')
        self.assertEqual(impact['questions'][0]['classification'], 'DIRECT_HELD_UNIT_CITATION')
        self.assertEqual(impact['questions'][0]['matching_held_turn_refs'], ['P1:T1'])
        self.assertIs(impact['questions'][0]['question_answer_recertified'], False)
        self.assertEqual(gate.bind(self.pool_path), before)
        with self.assertRaisesRegex(ValueError, 'held_sources'):
            gate.require_ready(self.home)

    def test_missing_frozen_review_binding_rejects_before_private_read(self):
        gate.finalize(self.home); path = self.home / 'CONSISTENCY_REVIEW.COMPLETE.json'
        frozen = gate.read(path)
        frozen['files'] = [row for row in frozen['files'] if row['path'] != str(self.review_path)]
        replace(path, frozen)
        self.assert_rejects_before_private(lambda: gate.verify_frozen(self.home))

    def test_stale_valid_review_binding_rejects_before_private_read(self):
        gate.finalize(self.home)
        self.review['units'][0]['reason'] = 'Changed but still structurally valid review.'; self.save()
        self.assertEqual(gate.verify_reviews(self.home)['ready_roots'], ['synthetic-root'])
        self.assert_rejects_before_private(lambda: gate.verify_frozen(self.home))

    def test_freeze_time_cannot_precede_saved_review(self):
        gate.finalize(self.home); path = self.home / 'CONSISTENCY_REVIEW.COMPLETE.json'
        frozen = gate.read(path)
        frozen['impact_replay_started_at_utc'] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        replace(path, frozen)
        self.assert_rejects_before_private(lambda: gate.verify_frozen(self.home))

    def test_hold_launch_rejects_before_mkdir_popen_subprocess_or_started(self):
        self.add_finding('BLOCKING'); gate.finalize(self.home)
        run = self.root / 'never-created-run'
        with patch.object(Path, 'mkdir', side_effect=AssertionError('HOLD must not create a directory')) as mkdir:
            for action in ('prepare', 'generate', 'live-trained'):
                with self.subTest(action=action), self.assertRaisesRegex(ValueError, 'held_sources'):
                    guarded_batch.launch(action, self.home, run)
            mkdir.assert_not_called()
        self.assertFalse(run.exists()); self.assertEqual(list(self.home.rglob('*STARTED*')), [])
        self.process.assert_not_called(); self.popen.assert_not_called()
        self.assertEqual(guarded_batch.preflight(self.home)['status'], 'BLOCKED_SOURCE_CONSISTENCY')

    def test_ready_prepare_copies_exact_config_and_lock_then_runs_only_mock(self):
        gate.finalize(self.home); run = self.root / 'ready-run'
        with patch.object(guarded_batch.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as process:
            result = guarded_batch.launch('prepare', self.home, run, slot='synthetic-slot', round_id='supplement')
            self.assertEqual(result['status'], 'COMPLETE')
            process.assert_called_once_with([str(guarded_batch.PYTHON), '-B', str(self.engine / 'batch.py'),
                'prepare', '--home', str(run), '--slot', 'synthetic-slot', '--round', 'supplement'], cwd=run)
        self.assertEqual((run / 'CONFIG.json').read_bytes(), (self.engine / 'CONFIG.json').read_bytes())
        self.assertEqual(gate.read(run / 'SOURCE_CONSISTENCY_LOCK.json'), {
            'review': gate.bind(self.home / 'CONSISTENCY_REVIEW.COMPLETE.json'),
            'source_config': gate.bind(self.engine / 'CONFIG.json')})
        self.assertEqual(guarded_batch.preflight(self.home)['status'], 'SOURCE_GATE_READY')
        self.popen.assert_not_called()

    def test_existing_run_with_stale_config_cannot_launch(self):
        gate.finalize(self.home); run = self.root / 'ready-run'
        with patch.object(guarded_batch.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
            guarded_batch.launch('prepare', self.home, run)
        config = gate.read(run / 'CONFIG.json'); config['max_candidates'] += 1
        replace(run / 'CONFIG.json', config)
        with self.assertRaisesRegex(ValueError, 'batch_config_changed'):
            guarded_batch.launch('generate', self.home, run)
        self.process.assert_not_called(); self.popen.assert_not_called()
        self.assertEqual(list(run.glob('*STARTED*')), [])

    def test_existing_run_with_stale_review_lock_cannot_launch(self):
        gate.finalize(self.home); run = self.root / 'ready-run'
        with patch.object(guarded_batch.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
            guarded_batch.launch('prepare', self.home, run)
        lock_path = run / 'SOURCE_CONSISTENCY_LOCK.json'; lock = gate.read(lock_path)
        lock['review']['sha256'] = '0' * 64; replace(lock_path, lock)
        with self.assertRaisesRegex(ValueError, 'batch_source_lock_changed'):
            guarded_batch.launch('generate', self.home, run)
        self.process.assert_not_called(); self.popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
