"""Mixed evaluation tests with real permutation/scoring/selection/audit helpers.

Preflight and live model execution are replaced by local artifact builders.
All sources, review markers and response records live in temporary directories;
every socket connection and subprocess is forbidden by default.
"""
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from common import NATIVE, bind, now, read, sha, write
import mixed_evaluation as evaluation


N = evaluation.native()


def fixture_questions():
    stem = 'Choose the supported statements.'
    options = {label: 'Claim ' + label + '.' for label in 'ABCDEF'}
    return [
        {'candidate_id': 'multi-two', 'root_id': 'root', 'question_type': 'multi_select',
         'question': stem, 'options': options, 'gold': ['A', 'C']},
        {'candidate_id': 'multi-six-control', 'root_id': 'root', 'question_type': 'multi_select',
         'question': stem, 'options': {label: 'Control ' + label + '.' for label in 'ABCDEF'},
         'gold': list('ABCDEF')},
        {'candidate_id': 'ordering-new', 'root_id': 'root', 'question_type': 'ordering',
         'chronology_kind': 'conversation_order',
         'question': 'Order these statements as they APPEAR IN THE SUPPLIED DIALOGUE.',
         'items': {'I' + str(i): 'Event ' + str(i) + '.' for i in range(1, 7)},
         'gold': ['I3', 'I1', 'I6', 'I2', 'I5', 'I4']},
    ]


def request_plan(questions):
    requests = []; gold = {}; prompts = {}
    for question in questions:
        variants, _ = N._variants_for(question)
        field = 'options' if question['question_type'] == 'multi_select' else 'items'
        text_to_id = {text: cid for cid, text in variants[0]['content_to_text'].items()}
        gold[question['candidate_id']] = [text_to_id[question[field][label]] for label in question['gold']]
        for variant in variants:
            for slot in range(3 if field == 'options' else 1):
                rid = question['candidate_id'] + '__' + variant['variant_id'] + f'__full_context__s{slot:02d}'
                prompt = 'Synthetic full source.\n' + question['question'] + '\n' + json.dumps(variant[field], sort_keys=True)
                requests.append({'request_id': rid, 'candidate_id': question['candidate_id'],
                    'root_id': question['root_id'], 'question_type': question['question_type'],
                    'variant': variant, 'mode': 'full_context', 'slot': slot,
                    'seed': N.SEED + len(requests), 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                    'prompt_tokens': 128})
                prompts[rid] = prompt
    return {'candidate_count': len(questions), 'request_count': len(requests), 'requests': requests}, gold, prompts


def basic_request(kind):
    labels = list('ABCDEF') if kind == 'multi_select' else ['I' + str(i) for i in range(1, 7)]
    field = 'options' if kind == 'multi_select' else 'items'
    return {'question_type': kind, 'variant': {'display_to_content': {
        label: 'content-' + str(i) for i, label in enumerate(labels, 1)},
        field: {label: 'Text ' + str(i) for i, label in enumerate(labels, 1)}}}


class OfflineNativeExecution:
    """Replace only model-bearing commands; report uses the real native code."""
    def __init__(self, home):
        self.home = home; self.calls = []

    def execute(self, home, args, log_name):
        self.calls.append((list(args), log_name))
        run = Path(args[args.index('--run') + 1]); action = args[0]
        if action == 'preflight':
            model = args[args.index('--solver-model') + 1]
            self.preflight(run, model)
        elif action == 'live':
            self.live(run)
        elif action == 'report':
            with redirect_stdout(io.StringIO()):
                N.report(run)
        else:
            raise AssertionError('Unexpected offline action: ' + action)

    def preflight(self, run, model):
        questions = read(self.home / 'EVALUATION_POOL.json')['questions']
        manifest, golden, prompts = request_plan(questions)
        public = [{k:v for k,v in q.items() if k not in ('gold', 'gold_content_ids')} for q in questions]
        identity = {'model_path': str(N.MODEL_PATHS[model]), 'model_identifier': model,
                    'tokenizer_path': str(N.TOKENIZER), 'synthetic': True}
        decoder = {'temperature': 0.8, 'top_p': 0.9, 'max_tokens': 32, 'enable_thinking': False,
                   'structured_output_version': 'finite_choice_set_v1', 'structured_output_api': 'synthetic_finite_choice'}
        write(run / 'questions.json', {'candidates': public}); write(run / 'gold.json', golden)
        write(run / 'solver_identity.json', identity)
        write(run / 'canonical_bindings.json', {'synthetic': True})
        write(run / 'sources/root.json', read(self.home / 'SOURCE.json'))
        for rid, text in prompts.items():
            path = run / 'prompts' / (rid + '.txt'); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        inputs = {'pool': sha(self.home / 'EVALUATION_POOL.json')}
        for name in ('questions.json', 'gold.json', 'canonical_bindings.json', 'sources/root.json'):
            inputs[name] = sha(run / name)
        manifest.update(status='PREFLIGHT_PASS', prepared_at_utc=now(), model=str(N.MODEL_PATHS[model]),
            solver_model=model, solver_identity=identity, tokenizer=str(N.TOKENIZER), policy_version=N.POLICY_VERSION,
            execution='full_only_no_ground_v1', training_started=False, training_promotion=False,
            pool_path=str(self.home / 'EVALUATION_POOL.json'), pool_sha256=inputs['pool'], input_sha256=inputs,
            code_sha256={str(NATIVE / name): sha(NATIVE / name)
                for name in ('run_solver.py', 'protocol.py', 'scoring.py', 'selection.py')},
            random_display={'version': 'independent_seeded_permutation_v2', 'seed': N.SEED,
                'gold_blind': True, 'cyclic_rebinding': False, 'shape_filter': False}, decoder_contract=decoder)
        write(run / 'manifest.json', manifest)

    def live(self, run):
        manifest = read(run / 'manifest.json'); golden = read(run / 'gold.json')
        public = {q['candidate_id']: q for q in read(run / 'questions.json')['candidates']}
        write(run / 'STARTED.json', {'manifest_sha256': sha(run / 'manifest.json'), 'started_at_utc': now()})
        records = []
        for request in manifest['requests']:
            qid = request['candidate_id']; gold = golden[qid]; prediction = list(gold)
            if manifest['solver_model'] == evaluation.MODELS['trained']:
                if request['question_type'] == 'multi_select':
                    prediction = prediction[:len(gold) // 2]
                else:
                    prediction[0], prediction[1] = prediction[1], prediction[0]
            inverse = {cid: label for label, cid in request['variant']['display_to_content'].items()}
            raw = '<box>' + ','.join(inverse[cid] for cid in prediction) + '</box>'
            score = N.score_response({**public[qid], 'gold_content_ids': gold}, request, raw)
            metadata = {key: manifest['decoder_contract'][key] for key in
                ('temperature', 'top_p', 'max_tokens', 'structured_output_api', 'structured_output_version')}
            metadata.update(finish_reason='stop', sampling_seed=request['seed'], prompt_token_count=request['prompt_tokens'])
            record = {'request_id': request['request_id'], 'candidate_id': qid, 'root_id': request['root_id'],
                'question_type': request['question_type'], 'variant_id': request['variant']['variant_id'],
                'slot': request['slot'], 'seed': request['seed'], 'evaluation_valid': True,
                'solver_model': manifest['solver_model'], 'model_path': manifest['model'],
                'raw_output': raw, 'score': score, 'data_errors': [], 'metadata': metadata}
            write(run / 'responses' / (request['request_id'] + '.json'), record); records.append(record)
        if N.audit_complete_call_set(manifest, records):
            raise AssertionError('Synthetic response call set invalid')
        write(run / 'COMPLETE.json', {'actual_call_count': len(records), 'synthetic': True})


class MixedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.home = Path(self.tmp.name).resolve()
        self.questions = fixture_questions()
        self.frozen = {'completed_at_utc': (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()}
        self.config = {'max_solver_calls_per_model': 456, 'multiselect_config': {'max_questions': 20},
                       'ordering_slots': [{'candidate_count': 2}]}
        write(self.home / 'CONFIG.json', self.config)
        write(self.home / 'SOURCE.json', {'root_id': 'root', 'turns': [
            {'turn_ref': 'S1:T1', 'speaker': 'Ada', 'text': 'Complete synthetic source.'}]})
        write(self.home / 'EVALUATION_POOL.json', {'questions': self.questions})
        write(self.home / 'BATCH_REVIEW.COMPLETE.json', self.frozen)
        write(self.home / 'multiselect/FINAL_CONTENT_REVIEW.COMPLETE.json', {'candidate_count': 3})
        write(self.home / 'ordering/CONTENT_REVIEW.COMPLETE.json', {'status': 'CONTENT_REVIEW_FROZEN'})
        write(self.home / 'ordering/CONTENT_REVIEW.json', {'accepted_count': 1, 'rejected_count': 1})
        self.frozen_mock = patch.object(evaluation.batch, 'verify_frozen', return_value=self.frozen)
        self.config_mock = patch.object(evaluation.batch, 'config', return_value=self.config)
        self.native_mock = patch.object(evaluation, 'native', return_value=N)
        self.network_mock = patch('socket.socket.connect', side_effect=AssertionError('Real network prohibited'))
        self.subprocess_mock = patch.object(evaluation.subprocess, 'run', side_effect=AssertionError('Real subprocess prohibited'))
        self.frozen_mock.start(); self.config_mock.start(); self.native_mock.start(); self.network_mock.start()
        self.subprocess = self.subprocess_mock.start()
        self.offline = OfflineNativeExecution(self.home)

    def tearDown(self):
        self.subprocess_mock.stop(); self.network_mock.stop(); self.native_mock.stop()
        self.config_mock.stop(); self.frozen_mock.stop(); self.tmp.cleanup()

    def run_preparation(self):
        with patch.object(evaluation, 'execute', side_effect=self.offline.execute):
            return evaluation.prepare(self.home)

    def run_live(self, side):
        with patch.object(evaluation, 'execute', side_effect=self.offline.execute):
            return evaluation.live(self.home, side)

    def test_multiselect_two_gold_reward_penalty_and_native_agreement(self):
        request = basic_request('multi_select'); gold = ['content-1', 'content-3']
        raw = '<box>A,B</box>'
        actual = evaluation.independent_score(raw, request, gold)
        self.assertEqual(actual, {'exact': False, 'partial': '1/4', 'tp': 1, 'fp': 1,
            'fn': 1, 'selected_count': 2, 'gold_count': 2})
        native = N.score_response({'question_type': 'multi_select', 'gold_content_ids': gold}, request, raw)
        self.assertEqual(actual['partial'], native['partial_score_fraction'])
        exact = evaluation.independent_score(' <box> c, a </box> ', request, gold)
        self.assertEqual(exact['partial'], '1'); self.assertIs(exact['exact'], True)

    def test_multiselect_six_gold_is_supported_control_not_ordering(self):
        request = basic_request('multi_select'); gold = ['content-' + str(i) for i in range(1, 7)]
        score = evaluation.independent_score('<box>A,B,C</box>', request, gold)
        self.assertEqual(score['partial'], '1/2'); self.assertEqual(score['fn'], 3)
        self.assertEqual(score['fp'], 0); self.assertIs(score['exact'], False)
        self.assertIs(evaluation.independent_score('<box>F,E,D,C,B,A</box>', request, gold)['exact'], True)

    def test_ordering_correct_inverse_and_single_inversion_scores(self):
        request = basic_request('ordering'); gold = ['content-' + str(i) for i in [3, 1, 6, 2, 5, 4]]
        for labels, partial, tp, fp, exact in [
            (['I3', 'I1', 'I6', 'I2', 'I5', 'I4'], '1', 15, 0, True),
            (['I4', 'I5', 'I2', 'I6', 'I1', 'I3'], '-1', 0, 15, False),
            (['I1', 'I3', 'I6', 'I2', 'I5', 'I4'], '13/15', 14, 1, False),
        ]:
            with self.subTest(labels=labels):
                raw = '<box>' + ','.join(labels) + '</box>'
                score = evaluation.independent_score(raw, request, gold)
                self.assertEqual((score['partial'], score['tp'], score['fp'], score['exact']), (partial, tp, fp, exact))
                native = N.score_response({'question_type': 'ordering', 'gold_content_ids': gold}, request, raw)
                self.assertEqual(score['partial'], native['partial_score_fraction'])

    def test_incomplete_duplicate_unknown_and_unboxed_answers_reject(self):
        request = basic_request('ordering'); gold = list(request['variant']['display_to_content'].values())
        for raw in ('<box>I1,I2,I3,I4,I5</box>', '<box>I1,I1,I3,I4,I5,I6</box>',
                    '<box>I1,I2,I3,I4,I5,I7</box>', 'I1,I2,I3,I4,I5,I6',
                    'Reason. <box>I1,I2,I3,I4,I5,I6</box>', '<box></box>'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                evaluation.independent_score(raw, request, gold)

    def test_typed_plan_has_eighteen_multi_calls_and_twelve_ordering_calls(self):
        questions = [self.questions[0], self.questions[2]]
        manifest, golden, _ = request_plan(questions)
        self.assertEqual(evaluation.verify_request_plan(manifest, questions, N), golden)
        self.assertEqual(Counter(r['question_type'] for r in manifest['requests']), {'multi_select': 18, 'ordering': 12})
        self.assertEqual(evaluation.typed_counts(questions), {
            'multi_select': {'candidates': 1, 'calls': 18}, 'ordering': {'candidates': 1, 'calls': 12}})
        self.assertEqual([r['seed'] for r in manifest['requests']], list(range(N.SEED, N.SEED + 30)))

    def test_wrong_slot_seed_variant_and_prompt_capacity_reject(self):
        questions = [self.questions[0], self.questions[2]]; original, _, _ = request_plan(questions)
        mutations = [lambda m: m['requests'][-1].update(slot=1),
            lambda m: m['requests'][0].update(seed=N.SEED + 1),
            lambda m: m['requests'][0].update(seed=True),
            lambda m: m['requests'][0]['variant'].update(variant_id='perm_99'),
            lambda m: m['requests'][0].update(prompt_tokens=0),
            lambda m: m['requests'][0].update(prompt_tokens=16353),
            lambda m: m.update(request_count=29), lambda m: m.update(candidate_count=1)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                manifest = deepcopy(original); mutation(manifest)
                with self.assertRaises(ValueError):
                    evaluation.verify_request_plan(manifest, questions, N)

    def test_native_variants_and_typed_plan_are_gold_blind(self):
        manifest, _, _ = request_plan(self.questions)
        changed = deepcopy(self.questions)
        changed[0]['gold'] = ['B', 'D', 'E']; changed[2]['gold'].reverse()
        modified, _, _ = request_plan(changed)
        self.assertEqual(manifest['requests'], modified['requests'])

    def test_prepare_gate_failure_starts_no_execution_or_stage(self):
        with (patch.object(evaluation.batch, 'verify_frozen', side_effect=ValueError('invalid content gate')),
              patch.object(evaluation, 'execute') as execute):
            with self.assertRaisesRegex(ValueError, 'invalid content gate'):
                evaluation.prepare(self.home)
            execute.assert_not_called()
        self.subprocess.assert_not_called()
        self.assertFalse((self.home / 'EVALUATION_PREPARATION.STARTED.json').exists())

    def test_live_gate_failure_starts_no_execution_or_stage(self):
        for side in evaluation.MODELS:
            with (self.subTest(side=side), patch.object(evaluation.batch, 'verify_frozen', side_effect=ValueError('invalid content gate')),
                  patch.object(evaluation, 'execute') as execute):
                with self.assertRaisesRegex(ValueError, 'invalid content gate'):
                    evaluation.live(self.home, side)
                execute.assert_not_called()
                self.assertFalse((self.home / (side.upper() + '_EVALUATION.STARTED.json')).exists())
        self.subprocess.assert_not_called()

    def test_execute_rechecks_gate_before_log_or_subprocess(self):
        with patch.object(evaluation.batch, 'verify_frozen', side_effect=ValueError('invalid content gate')):
            with self.assertRaisesRegex(ValueError, 'invalid content gate'):
                evaluation.execute(self.home, ['preflight'], 'blocked.log')
        self.subprocess.assert_not_called(); self.assertFalse((self.home / 'logs/blocked.log').exists())

    def test_base_requires_completed_trained_stage_before_start(self):
        self.run_preparation()
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.run_live('base')
        self.assertFalse((self.home / 'BASE_EVALUATION.STARTED.json').exists())
        self.assertEqual([args[0] for args, _ in self.offline.calls], ['preflight', 'preflight'])

    def test_two_model_prepare_live_report_uses_native_replay_and_type_statistics(self):
        prepared = self.run_preparation()
        self.assertEqual(prepared['calls_per_model'], 48)
        self.assertEqual(prepared['paired_inputs']['by_type'], {
            'multi_select': {'candidates': 2, 'calls': 36}, 'ordering': {'candidates': 1, 'calls': 12}})
        trained = self.run_live('trained'); base = self.run_live('base')
        self.assertEqual((trained['calls'], base['calls']), (48, 48))
        result = evaluation.report(self.home)
        self.assertEqual(result['selected_candidate_ids'], ['multi-two', 'multi-six-control', 'ordering-new'])
        self.assertEqual(result['selected_new_candidate_count'], 2)
        multi = result['by_type']['multi_select']; ordering = result['by_type']['ordering']
        self.assertEqual((multi['mechanically_assembled'], multi['content_usable'], multi['content_excluded'], multi['solver_selected']), (3, 2, 1, 2))
        self.assertEqual((ordering['mechanically_assembled'], ordering['content_usable'], ordering['content_excluded'], ordering['solver_selected']), (2, 1, 1, 1))
        self.assertEqual(multi['models']['trained'], {'calls': 36, 'exact_correct': 0, 'exact_accuracy': '0', 'partial_mean': '1/2'})
        self.assertEqual(ordering['models']['trained'], {'calls': 12, 'exact_correct': 0, 'exact_accuracy': '0', 'partial_mean': '13/15'})
        self.assertEqual(multi['models']['base']['exact_accuracy'], '1')
        self.assertEqual(ordering['models']['base']['exact_accuracy'], '1')
        control = multi['by_gold_count']['6']
        self.assertEqual((control['content_usable'], control['solver_selected'], control['role']), (1, 1, 'historical_six_supported_control'))
        self.assertNotIn('by_gold_count', ordering)
        pairs = read(self.home / 'PAIRED_CALLS.json')['calls']; self.assertEqual(len(pairs), 48)
        self.assertEqual(Counter(p['question_type'] for p in pairs), {'multi_select': 36, 'ordering': 12})
        self.assertIs(result['training_started'], False); self.assertIs(result['formal_training_admission'], False)
        selected = read(self.home / 'SELECTED_CANDIDATES.json')
        self.assertIs(selected['provisional'], True); self.assertIs(selected['training_allowed'], False)
        self.assertEqual([args[0] for args, _ in self.offline.calls], ['preflight', 'preflight', 'live', 'report', 'live', 'report'])
        self.subprocess.assert_not_called()

    def test_native_call_audit_rejects_changed_response_seed(self):
        self.run_preparation(); self.run_live('trained')
        path = next((evaluation.solver_dir(self.home, 'trained') / 'responses').glob('*.json'))
        record = read(path); record['seed'] += 1; path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'native_call_set_invalid'):
            evaluation.replay_side(self.home, 'trained')

    def test_native_score_replay_rejects_modified_saved_score(self):
        self.run_preparation(); self.run_live('trained')
        path = next((evaluation.solver_dir(self.home, 'trained') / 'responses').glob('*.json'))
        record = read(path); record['score']['partial_score_fraction'] = '1'; path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'native_score_replay_mismatch'):
            evaluation.replay_side(self.home, 'trained')

    def test_empty_pool_skips_both_models_and_keeps_type_statistics(self):
        path = self.home / 'EVALUATION_POOL.json'; path.write_text(json.dumps({'questions': []}))
        prepared = self.run_preparation(); self.assertEqual(prepared['calls_per_model'], 0)
        self.run_live('trained'); self.run_live('base'); result = evaluation.report(self.home)
        self.assertEqual(result['evaluation_status'], 'SKIPPED_EMPTY_CONTENT_USABLE_POOL')
        self.assertEqual(self.offline.calls, [])
        self.assertEqual(result['selected_new_candidate_count'], 0)
        for kind in evaluation.CALLS:
            self.assertEqual(result['by_type'][kind]['content_usable'], 0)
            self.assertEqual(result['by_type'][kind]['models']['trained']['calls'], 0)
            self.assertIsNone(result['by_type'][kind]['models']['trained']['partial_mean'])


if __name__ == '__main__':
    unittest.main()
