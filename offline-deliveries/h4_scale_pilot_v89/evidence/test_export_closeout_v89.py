"""Offline adversarial fixtures for independent export and immutable pilot seal.

All mutations are in TemporaryDirectory. No actual shard is exported/sealed and
no network, model, historical-pool mutation or Solver invocation occurs.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


writer = load('export_closeout_v89'); auditor = load('audit_export_v89')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def question(qid, kind='multi_select', gold=None, root='cc-train-episode-1'):
    labels = list('ABCDEF') if kind == 'multi_select' else ['I' + str(i) for i in range(1, 7)]
    field = 'options' if kind == 'multi_select' else 'items'
    return {'candidate_id': qid, 'root_id': root, 'question_type': kind,
            field: {label: qid + ' statement ' + str(i) for i, label in enumerate(labels)},
            'gold': gold or labels[:2] if kind == 'multi_select' else labels,
            'training_allowed': False}


def responses(shard, questions):
    summaries = []; requests = []
    for q in questions:
        ordering = q['question_type'] == 'ordering'; field = 'items' if ordering else 'options'
        labels = list(q[field]); mapping = {label: 'content_' + hashlib.sha256(q[field][label].strip().encode()).hexdigest()[:16] for label in labels}
        control = len(q['gold']) == 6 and not ordering
        raw = '<box>I2,I1,I3,I4,I5,I6</box>' if ordering else ('<box>A,B,C,D,E,F</box>' if control else '<box>A</box>')
        partial = '13/15' if ordering else ('1' if control else '1/2')
        selected = not control
        summaries.append({'candidate_id': q['candidate_id'], 'full_partial_mean': partial, 'full_exact_rate': '1' if control else '0', 'selected_by_policy': selected})
        for i in range(12 if ordering else 6):
            for slot in range(1 if ordering else 3):
                rid = q['candidate_id'] + f'__perm_{i:02d}__s{slot:02d}'
                plan = {'request_id': rid, 'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
                        'question_type': q['question_type'], 'slot': slot, 'seed': i * 3 + slot,
                        'variant': {'variant_id': f'perm_{i:02d}', 'display_to_content': mapping, field: q[field]}}
                requests.append(plan)
                for side in ('trained', 'base'):
                    row = {k: plan[k] for k in ('request_id', 'candidate_id', 'root_id', 'question_type', 'slot', 'seed')}
                    row.update(variant_id=plan['variant']['variant_id'], raw_output=raw,
                               score={'partial_score_fraction': partial, 'exact_match': control})
                    put(shard / 'solver' / side / 'responses' / (rid + '.json'), row)
    for side in ('trained', 'base'):
        put(shard / 'solver' / side / 'manifest.json', {'requests': requests})
    return summaries


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory(prefix='v89-closeout-offline-') as directory:
        delivery = Path(directory); home = delivery / 'h4_scale_pilot_v89'; home.mkdir()
        (home / 'evidence').mkdir(); shard = home / 'shards/shard_ok'; shard.mkdir(parents=True)
        baseline = delivery / 'h4_anchor_records_mixed_batch_v88/CUMULATIVE_AUDITED_NEW_CANDIDATES.json'
        old = question('baseline'); prior = writer.pool([old], 'DEDUPLICATED_CANDIDATE_INVENTORY'); put(baseline, prior)
        put(home / 'PRESERVATION_BEFORE.json', {'files': {str(baseline): writer.sha(baseline)}, 'protected_count': 1})
        questions = [question('new_multi'), question('new_order', 'ordering'), question('control', gold=list('ABCDEF'))]
        put(shard / 'EVALUATION_POOL.json', {'questions': questions})
        summaries = responses(shard, questions); selected = ['new_multi', 'new_order']
        put(shard / 'SELECTED_CANDIDATES.json', {'questions': questions[:2]})
        auditor_files = [writer.bind(HERE / name) for name in
                         ('audit_shard.py', 'audit_contract_v89.py', 'audit_results_core_v89.py', 'audit_generation_core_v89.py')]
        input_audit = shard / 'evidence/INPUT_AUDIT.json'; put(input_audit, {'status': 'PASS', 'phase': 'inputs_only', 'auditor_files': auditor_files})
        result_path = shard / 'evidence/RESULT_AUDIT.json'
        put(result_path, {'home': str(shard), 'status': 'PASS', 'phase': 'paired_results', 'checks': [{'status': 'PASS'}],
                          'failed_checks': [], 'files': [], 'auditor_files': auditor_files,
                          'passing_input_audit': writer.bind(input_audit), 'completed_at_utc': '2026-10-14T00:00:00+00:00',
                          'models': {side: {'per_question': summaries, 'selected_ids': selected} for side in ('trained', 'base')},
                          'controls': ['control'], 'selected_new_ids': selected, 'selected_new_count': 2})
        candidates = []; slots = []
        gate = load('source_gate_disclosure_v89')
        adapter = home / 'evidence/source_gate_disclosure_v89.py'; adapter.write_bytes((HERE / adapter.name).read_bytes())
        put(home / 'evidence/SOURCE_GATE_DISCLOSURE_ADAPTATION.json', {'upstream': writer.bind(HERE / 'source_gate_disclosure_v89.py'),
            'adapted': writer.bind(adapter), 'source_decision_rules_changed': False, 'upstream_modified': False})
        for i in range(1, 21):
            root = 'cc-train-episode-' + str(i); source_path = home / 'inventory' / ('source' + str(i) + '.json')
            turn = {'turn_ref': root + ':T001', 'speaker': 'Person', 'text': 'A complete turn.', 'origin': 'source_grounded', 'session_ref': 'S01'}
            put(source_path, {'root_id': root, 'turns': [turn]}); source = writer.bind(source_path)
            candidates.append({'root_id': root, 'source': source, 'candidate_status': 'AUDITED_SIX_FACT_PARENT' if i <= 2 else 'NO_AUDITED_SIX_FACT_PARENT'})
            if i <= 2:
                slots.append({'root_id': root, 'source_path': str(source_path), 'source_sha256': source['sha256']})
            hold = i == 20
            finding = {'finding_id': 'F1', 'severity': 'BLOCKING', 'kind': 'naturalness', 'reason': 'Fixture blocking judgment.',
                       'proposed_action': 'Hold source.', 'unit_ids': ['S01'], 'evidence': [{'turn_ref': turn['turn_ref'], 'quote': turn['text']}]}
            review = {'schema_version': 'dialogue_consistency_v1', 'root_id': root, 'source': source, 'review_scope': 'source_consistency_only',
                'reviewer': 'offline_fixture', 'limitations': 'Synthetic metadata validation only.', 'full_source_read': True,
                'current_question_gold_read': False, 'current_solver_scores_read': False, 'historical_question_gold_known': False,
                'fresh_blind_review': True, 'created_at_utc': '2026-10-14T00:00:00+00:00', 'turn_refs_read': [turn['turn_ref']],
                'checks_completed': {key: True for key in gate.CHECKS}, 'findings': [finding] if hold else [],
                'units': [{'unit_id': 'S01', 'origin': turn['origin'], 'turn_refs': [turn['turn_ref']], 'reason': 'Synthetic source.',
                           'decision': 'HOLD' if hold else 'PASS', 'finding_ids': ['F1'] if hold else []}], 'decision': 'HOLD' if hold else 'READY'}
            put(home / 'source_reviews' / ('root' + str(i) + '_REVIEW.json'), review)
        put(home / 'inventory/CANDIDATE_ROOTS_20.json', {'candidate_count': 20, 'candidates': candidates})
        put(shard / 'CONFIG.json', {'ordering_slots': slots})
        for side in ('TRAINED', 'BASE'):
            put(shard / (side + '_EVALUATION.STARTED.json'), {'time': '2026-10-14T00:00:00+00:00'})
            put(shard / (side + '_EVALUATION.COMPLETE.json'), {'time': '2026-10-14T00:01:00+00:00'})
        put(shard / 'MIXED_EVALUATION_REPORT.json', {'calls_per_model': 48, 'by_type': {kind: {
            'mechanically_assembled': 2 if kind == 'multi_select' else 1, 'content_usable': 2 if kind == 'multi_select' else 1,
            'content_excluded': 0, 'solver_selected': 1, 'models': {}} for kind in ('multi_select', 'ordering')}})
        failed = home / 'shards/shard_fail'; failed.mkdir()
        put(failed / 'BATCH_PREPARATION.ERROR.json', {'status': 'ERROR', 'automatic_retry': False, 'training_allowed': False})
        put(home / 'evidence/FAILED_SHARD_FAIL_SNAPSHOT.json', {'files': {str(p): writer.sha(p) for p in failed.iterdir()},
            'generation_http_calls': 0, 'solver_calls': 0})
        yield home, baseline, shard, failed


class ExportCloseoutTests(unittest.TestCase):
    def cli(self, home, shard=None, failed=None, *args):
        command = [sys.executable, '-B', str(HERE / 'export_closeout_v89.py'), '--home', str(home)]
        if shard:
            command += ['--shard', str(shard)]
        if failed:
            command += ['--failed-shard', str(failed)]
        return subprocess.run(command + list(args), text=True, capture_output=True)

    def test_typed_export_raw_replay_controls_and_six_item_ordering(self):
        with fixture() as (home, baseline, shard, _):
            entries = writer.selected_shards(home, [shard]); new, union = writer.build_exports(home, baseline, entries)
            result = auditor.audit_exports(home, baseline, entries, new, union)
            self.assertEqual(result['new_count'], 2); self.assertEqual(result['cumulative_count'], 3)
            self.assertEqual(new['by_type'], {'multi_select': 1, 'ordering': 1})
            self.assertEqual(len(new['questions'][1]['gold']), 6)
            self.assertNotIn('control', [q['candidate_id'] for q in new['questions']])

    def test_raw_tamper_missing_coverage_and_audit_id_mismatch_rejected(self):
        for failure in ('raw', 'missing', 'selected'):
            with self.subTest(failure=failure), fixture() as (home, _, shard, _):
                path = next((shard / 'solver/trained/responses').glob('new_multi*.json'))
                if failure == 'raw':
                    row = writer.read(path); row['raw_output'] = '<box>A,B</box>'; put(path, row)
                elif failure == 'missing':
                    path.unlink()
                else:
                    path = shard / 'evidence/RESULT_AUDIT.json'; row = writer.read(path); row['selected_new_ids'] = ['new_order']; put(path, row)
                with self.assertRaises(ValueError):
                    auditor.shard_replay(shard, writer.latest_pass(shard))

    def test_all_six_set_dedup_ignores_label_order_whitespace_and_case(self):
        q = question('original'); relabelled = deepcopy(q); relabelled['candidate_id'] = 'alternate'
        relabelled['options'] = {label: '  ' + value.upper().replace(' ', '   ') + ' ' for label, value in zip('FEDCBA', q['options'].values())}
        other_root = deepcopy(q); other_root['candidate_id'] = 'different_root'; other_root['root_id'] = 'cc-train-episode-2'
        kept, removed = writer.deduplicate([q, relabelled, other_root])
        self.assertEqual(len(kept), 2); self.assertEqual(removed[0]['retained_candidate_id'], 'original')
        self.assertEqual(auditor.statement_key(q), auditor.statement_key(relabelled))

    def test_history_exact_inventory_and_failed_zero_call_preservation(self):
        with fixture() as (home, baseline, shard, failed):
            self.assertEqual(writer.historical_preservation(home)['file_count'], 1)
            writer.failed_shards(home, [failed], [shard])
            (baseline.parent / 'unrecorded').write_text('extra')
            with self.assertRaises(ValueError):
                writer.historical_preservation(home)
            (baseline.parent / 'unrecorded').unlink()
            put(failed / 'HTTP_ATTEMPT.json', {})
            with self.assertRaises(ValueError):
                writer.failed_shards(home, [failed], [shard])
            with self.assertRaises(ValueError):
                writer.failed_shards(home, [], [shard])

    def test_all_twenty_reviews_and_hold_missing_parent_are_distinct(self):
        with fixture() as (home, _, shard, _):
            result = writer.source_screening(home, [shard])
            self.assertEqual(result['source_decision_counts'], {'READY': 19, 'HOLD': 1})
            self.assertEqual(result['production_admission_counts']['SOURCE_READY_MISSING_AUDITED_PARENT'], 17)
            self.assertEqual(result['production_admission_counts']['SOURCE_HOLD'], 1)
            path = home / 'source_reviews/root3_REVIEW.json'; row = writer.read(path); row['turn_refs_read'] = []; put(path, row)
            with self.assertRaises(ValueError):
                writer.source_screening(home, [shard])

    def test_check_only_full_export_closeout_seal_and_after_seal_rejections(self):
        with fixture() as (home, _, shard, failed):
            before = {p: writer.sha(p) for p in writer.files_under(home)}
            check = self.cli(home, shard, None, '--check-only'); self.assertEqual(check.returncode, 0, check.stderr)
            self.assertEqual(before, {p: writer.sha(p) for p in writer.files_under(home)})
            result = self.cli(home, shard); self.assertEqual(result.returncode, 0, result.stderr)
            before = {p: writer.sha(p) for p in writer.files_under(home)}
            check = self.cli(home, shard, failed, '--action', 'closeout', '--check-only'); self.assertEqual(check.returncode, 0, check.stderr)
            self.assertEqual(before, {p: writer.sha(p) for p in writer.files_under(home)})
            result = self.cli(home, shard, failed, '--action', 'closeout'); self.assertEqual(result.returncode, 0, result.stderr)
            result = self.cli(home, None, None, '--verify-complete'); self.assertEqual(result.returncode, 0, result.stderr)
            seal = writer.read(home / 'PILOT_COMPLETE.json'); self.assertNotIn(str(home / 'PILOT_COMPLETE.json'), seal['files'])
            blocked = self.cli(home, shard); self.assertNotEqual(blocked.returncode, 0)
            (home / 'extra_after_seal').write_text('extra')
            blocked = self.cli(home, None, None, '--verify-complete'); self.assertNotEqual(blocked.returncode, 0)
            (home / 'extra_after_seal').unlink(); (home / 'REPORT.md').write_text('changed')
            blocked = self.cli(home, None, None, '--verify-complete'); self.assertNotEqual(blocked.returncode, 0)

    def test_strict_json_and_global_training_policy(self):
        with fixture() as (home, baseline, shard, _):
            entries = writer.selected_shards(home, [shard]); new, union = writer.build_exports(home, baseline, entries)
            union['training_allowed'] = True
            with self.assertRaises(ValueError):
                auditor.audit_exports(home, baseline, entries, new, union)
            path = home / 'bad.json'; path.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):
                auditor.read(path)
            path.write_text('{"a":NaN}')
            with self.assertRaises(ValueError):
                writer.read(path)

    def test_actual_python_guard_blocks_external_calls_and_nonoutput_mutations(self):
        with fixture() as (home, baseline, shard, _):
            probes = {
                'socket': 'import socket; socket.socket()',
                'subprocess': "import subprocess; subprocess.Popen(['true'])",
                'historical_write': 'open(' + repr(str(baseline)) + ", 'w').write('bad')",
                'historical_delete': 'import os; os.unlink(' + repr(str(baseline)) + ')',
            }
            for label, probe in probes.items():
                with self.subTest(probe=label):
                    source = ('import importlib.util, sys\n'
                              's=importlib.util.spec_from_file_location("guard_tool",' + repr(str(HERE / 'export_closeout_v89.py')) + ')\n'
                              'm=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n'
                              'sys.argv=["tool","--home",' + repr(str(home)) + ',"--shard",' + repr(str(shard)) + ',"--check-only"]\n'
                              'def probe(*args):\n    exec(' + repr(probe) + ')\n'
                              'm.module=probe\nm.main()\n')
                    before = baseline.read_bytes()
                    result = subprocess.run([sys.executable, '-B', '-c', source], text=True, capture_output=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertTrue('forbids_' in result.stderr or 'write_forbidden:' in result.stderr, result.stderr)
                    self.assertEqual(baseline.read_bytes(), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
