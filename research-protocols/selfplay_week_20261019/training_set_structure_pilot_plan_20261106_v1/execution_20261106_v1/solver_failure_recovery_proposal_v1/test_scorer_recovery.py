"""Synthetic CPU scorer regression tests; no actual experimental inputs are read."""
from contextlib import ExitStack
import ast
import builtins
import copy
import hashlib
import itertools
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

PROPOSAL = Path(__file__).resolve().parent
FROZEN = PROPOSAL.parent / 'solver/adapter.py'
FORBIDDEN_IMPORT_ATTEMPTS = []
REAL_IMPORT = builtins.__import__


def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'vllm', 'transformers', 'torch', 'openai'}:
        FORBIDDEN_IMPORT_ATTEMPTS.append(name)
        raise AssertionError('MODEL_OR_API_IMPORT_FORBIDDEN_IN_SCORER_TEST:' + name)
    return REAL_IMPORT(name, *args, **kwargs)


def load_source(name, path):
    module = types.ModuleType(name)
    # Preserve the inherited relative code dependency paths, without writes.
    module.__file__ = str(FROZEN)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


with patch('builtins.__import__', guarded_import):
    a = load_source('offline_proposed_adapter', PROPOSAL / 'adapter.py.proposed')
    frozen = load_source('offline_frozen_adapter', FROZEN)


def request(qtype, rid='synthetic', mode='no_context', gold_count=2):
    k = 3 if qtype == 'partial_order' else 4
    labels = ['I' + str(i + 1) for i in range(k)]
    mapping = {label: 'synthetic_' + str(i) for i, label in enumerate(labels)}
    ids = list(mapping.values())
    gold = ids[:gold_count] if qtype == 'multi_select' else ids if qtype == 'ordering' else []
    pairs = [[ids[0], ids[1]], [ids[1], ids[2]], [ids[0], ids[2]]] if qtype == 'partial_order' else []
    inverse = {value: key for key, value in mapping.items()}
    result = {'request_id': rid, 'candidate_id': 'synthetic_' + qtype, 'root_id': 'synthetic_root',
              'group': 'OLD_F', 'question_type': qtype, 'context_mode': mode, 'variant_id': 'synthetic_v0',
              'repeat_index': 0, 'sampling_seed': 12345, 'protocol_markers': [], 'clean_contrast_eligible': True,
              'structure_result': {'independent': True, 'status': 'SYNTHETIC'},
              'overlap_result': {'independent': True, 'status': 'SYNTHETIC'},
              'display_to_option_id': mapping, 'options': [{'id': label, 'text': 'Synthetic item ' + label} for label in labels],
              'gold_option_ids': gold, 'gold_labels': [[inverse[x], inverse[y]] for x, y in pairs] if pairs else [inverse[x] for x in gold],
              'g': len(gold) if qtype == 'multi_select' else len(pairs) if qtype == 'partial_order' else k * (k - 1) // 2,
              'n': k if qtype == 'multi_select' else k * (k - 1)}
    if qtype == 'partial_order':
        result.update(gold_pairs=pairs, query_pairs=[[x, y] for x in ids for y in ids if x != y])
    return result


def answer_body(req):
    return ';'.join('<'.join(pair) for pair in req['gold_labels']) if req['question_type'] == 'partial_order' else ' '.join(req['gold_labels'])


def returned(req, body=None, malformed_aux=False, finish_reason='stop'):
    obj = {'item_evidence': {label: {'judgment': 'not_supported', 'turn_refs': [], 'reason': 'Synthetic fixture only.'}
                             for label in req['display_to_option_id']},
           'answer_reason': 'Synthetic offline scorer fixture.',
           'final_answer': '<box>' + (answer_body(req) if body is None else body) + '</box>'}
    if malformed_aux:
        del obj['item_evidence']
    return {'request_id': req['request_id'], 'status': 'RETURNED', 'finish_reason': finish_reason,
            'raw_prediction': json.dumps(obj)}


class ScorerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('builtins.__import__', guarded_import))
        for name in ('prepare', 'run', 'check_inputs', 'configure_legacy_run'):
            self.stack.enter_context(patch.object(a, name, side_effect=AssertionError('EXPERIMENT_ENTRY_FORBIDDEN:' + name)))

    def sandbox(self, cases, bad_binding=False):
        root = PROPOSAL / 'synthetic_fixtures' / self._testMethodName
        root.mkdir(parents=True, exist_ok=False)
        output = root / 'evaluation'
        output.mkdir()
        self.stack.enter_context(patch.object(a, 'HERE', root))
        self.stack.enter_context(patch.object(a, 'OUTPUT', output))
        a.write_new(root / 'RUN_ATTEMPT.json', {'synthetic_fixture_only': True, 'model_calls': 0, 'automatic_retry': False})
        requests = []
        for req, raw in cases:
            req = copy.deepcopy(req)
            req['prompt_path'] = 'prompts/' + req['request_id'] + '.txt'
            req['schema_path'] = 'schemas/' + req['request_id'] + '.json'
            a.write_new(output / req['prompt_path'], b'Synthetic offline scorer fixture. Never submitted.\n')
            a.write_new(output / req['schema_path'], a.schema_for(list(req['display_to_option_id']), [], req['question_type']))
            for kind in ('prompt', 'schema'):
                req[kind + '_sha256'] = a.bind(output / req[kind + '_path'])['sha256']
            requests.append(req)
            if raw is not None:
                a.write_new(output / 'raw' / (req['request_id'] + '.json'), raw)
        a.write_new(output / 'REQUESTS.json', requests)
        a.write_new(root / 'SYNTHETIC_DEPENDENCY.txt', b'Synthetic binding, never actual source or question.\n')
        dependency = a.bind(root / 'SYNTHETIC_DEPENDENCY.txt')
        if bad_binding:
            dependency['sha256'] = '0' * 64
        a.write_new(output / 'INPUT_BINDINGS.json', {'files': [dependency]})
        a.write_new(output / 'PLAN_FREEZE.json', {'synthetic_fixture_only': True, 'files': [a.bind(output / name) for name in ('REQUESTS.json', 'INPUT_BINDINGS.json')]})
        return output

    def test_original_failure_and_historical_invalid_record(self):
        for qtype in ('multi_select', 'ordering'):
            with self.assertRaises(ZeroDivisionError):
                frozen.invalid(qtype, 'INVALID_FINAL_ANSWER')
            req = request(qtype)
            expected = a.load_history().old_parse({'status': 'INVALID'}, req)
            expected['parse_reason'] = 'INVALID_FINAL_ANSWER'
            self.assertEqual(a.invalid(req, 'INVALID_FINAL_ANSWER'), expected)
            self.assertEqual(expected['item_completeness'], 0)
        self.assertEqual(a.invalid(request('partial_order'), 'INVALID_FINAL_ANSWER'), a.load_history().partial_invalid('INVALID_FINAL_ANSWER'))

    def test_valid_answers_never_evaluate_fallback(self):
        cases = [(req, returned(req)) for req in (request(q, 'valid_' + q) for q in ('multi_select', 'ordering', 'partial_order'))]
        output = self.sandbox(cases)
        with patch.object(a, 'invalid', side_effect=AssertionError('EAGER_FALLBACK')) as fallback:
            a.score()
        fallback.assert_not_called()
        rows = a.read(output / 'SCORES.json')['requests']
        self.assertTrue(all(row['valid'] and row['exact_correct'] for row in rows))

    def test_all_missing_infrastructure_has_no_model_denominator(self):
        cases = [(request(q, q + '_' + mode, mode), None) for q in ('multi_select', 'ordering', 'partial_order') for mode in ('full_context', 'no_context')]
        output = self.sandbox(cases)
        a.score()
        result = a.read(output / 'SCORES.json')
        for mode in ('full_context', 'no_context'):
            summary = result['by_context'][mode]
            self.assertEqual((summary['planned_calls'], summary['executed_returned_calls'], summary['missing_calls']), (3, 0, 3))
            self.assertEqual(summary['invalid_executed_calls'], 0)
            self.assertIsNone(summary['exact_accuracy_executed'])
            self.assertIsNone(summary['planned_accuracy'])
            self.assertEqual(set(summary['metrics_by_type']), {'multi_select', 'ordering', 'partial_order'})
            self.assertNotIn('mean_J_executed', summary)
            for typed in summary['metrics_by_type'].values():
                self.assertTrue(all(value is None for key, value in typed.items() if key.startswith('mean_')))
        self.assertTrue(all(row['execution_status'] == 'MISSING' for row in result['requests']))
        with self.assertRaisesRegex(AssertionError, 'NO_SCORE_OVERWRITE'):
            a.score()

    def test_mixed_missing_backend_invalid_and_returned_denominators(self):
        cases = []
        for qtype in ('multi_select', 'ordering', 'partial_order'):
            for status in ('valid', 'invalid', 'truncated', 'missing', 'backend'):
                req = request(qtype, qtype + '_' + status)
                raw = returned(req, malformed_aux=True) if status == 'valid' else returned(req, 'UNKNOWN') if status == 'invalid' else returned(req, finish_reason='length') if status == 'truncated' else None if status == 'missing' else {'request_id': req['request_id'], 'status': 'BACKEND_INVALID', 'reason': 'Synthetic transport failure; no model answer.'}
                cases.append((req, raw))
        output = self.sandbox(cases)
        a.score()
        result = a.read(output / 'SCORES.json')
        summary = result['by_context']['no_context']
        self.assertEqual((summary['planned_calls'], summary['executed_returned_calls'], summary['missing_calls']), (15, 9, 6))
        self.assertEqual((summary['valid_calls'], summary['invalid_executed_calls'], summary['exact_correct_calls']), (3, 6, 3))
        self.assertEqual(summary['exact_accuracy_executed'], 1 / 3)
        self.assertIsNone(summary['planned_accuracy'])
        for qtype, typed in summary['metrics_by_type'].items():
            metric = 'mean_directed_pair_f1_executed' if qtype == 'partial_order' else 'mean_J_executed'
            self.assertEqual(typed[metric], 1 / 3)
        valid = [row for row in result['requests'] if row['valid']]
        self.assertTrue(all(not row['auxiliary_protocol_valid'] and row['exact_correct'] for row in valid))

    def test_invalid_responses_score_zero_for_all_types(self):
        cases = []
        for qtype in ('multi_select', 'ordering', 'partial_order'):
            req0 = request(qtype)
            bodies = ['UNKNOWN', 'I1<I1', 'I1<I2;I1<I2'] if qtype == 'partial_order' else ['UNKNOWN', 'I1 I1']
            if qtype == 'ordering':
                bodies += ['I1 I2', '']
            for index, body in enumerate(bodies):
                req = request(qtype, qtype + '_bad_' + str(index))
                cases.append((req, returned(req, body)))
            req = request(qtype, qtype + '_bad_json')
            raw = returned(req)
            raw['raw_prediction'] = '{"final_answer":"<box></box>","final_answer":"<box></box>"}'
            cases.append((req, raw))
        output = self.sandbox(cases)
        a.score()
        result = a.read(output / 'SCORES.json')
        summary = result['by_context']['no_context']
        self.assertEqual(summary['executed_returned_calls'], len(cases))
        self.assertEqual(summary['invalid_executed_calls'], len(cases))
        self.assertEqual(summary['exact_accuracy_executed'], 0)
        self.assertEqual(summary['planned_accuracy'], 0)
        for row in result['requests']:
            self.assertFalse(row['valid'])
            self.assertEqual(row.get('J', row.get('partial_score')), 0)

    def test_multiselect_exact_historical_J_equivalence(self):
        old = a.load_history()
        for gold_count in (1, 2, 3, 4):
            req = request('multi_select', gold_count=gold_count)
            labels = list(req['display_to_option_id'])
            for mask in range(1, 1 << len(labels)):
                values = [label for index, label in enumerate(labels) if mask & (1 << index)]
                body = ' '.join(values)
                raw = returned(req, body)
                expected = old.parse_and_score({**raw, 'raw_prediction': ','.join(values)}, req)
                self.assertEqual(a.parse_answer(body, raw, req), expected)
            empty = a.parse_answer('', returned(req, ''), req)
            self.assertTrue(empty['valid'])
            self.assertEqual((empty['J'], empty['J_fraction'], empty['exact_correct'], empty['fn']), (0, '0', False, gold_count))
        req = request('multi_select')
        self.assertEqual(a.parse_answer('I1 I3', returned(req), req)['J_fraction'], '0')
        self.assertEqual(a.parse_answer('I3 I4', returned(req), req)['J'], -1)
        self.assertEqual(a.parse_answer('I1', returned(req), request('multi_select', gold_count=4))['J_fraction'], '1/4')

    def test_ordering_exact_historical_J_equivalence(self):
        req = request('ordering')
        old = a.load_history()
        for values in itertools.permutations(req['display_to_option_id']):
            body = ' '.join(values)
            raw = returned(req, body)
            expected = old.parse_and_score({**raw, 'raw_prediction': '>'.join(values)}, req)
            self.assertEqual(a.parse_answer(body, raw, req), expected)
        self.assertEqual(a.parse_answer('I4 I3 I2 I1', returned(req), req)['J'], -1)
        self.assertEqual(a.parse_answer('I2 I1 I3 I4', returned(req), req)['J_fraction'], '2/3')

    def test_partial_F1_exact_equivalence_and_no_closure_repair(self):
        req = request('partial_order')
        old = a.load_history()
        all_pairs = list(itertools.permutations(req['display_to_option_id'], 2))
        for mask in range(1 << len(all_pairs)):
            body = ';'.join('<'.join(pair) for index, pair in enumerate(all_pairs) if mask & (1 << index))
            raw = returned(req, body)
            expected = old.parse_and_score({**raw, 'raw_prediction': body}, req)
            self.assertEqual(a.parse_answer(body, raw, req), expected)
        incomplete = a.parse_answer('I1<I2;I2<I3', returned(req), req)
        self.assertEqual(incomplete['partial_score'], .8)
        self.assertEqual(len(incomplete['predicted_option_ids']), 2)
        self.assertFalse(incomplete['exact_correct'])
        self.assertNotIn('J', incomplete)
        req['gold_pairs'], req['gold_labels'], req['g'] = [], [], 0
        empty = a.parse_answer('', returned(req, ''), req)
        self.assertTrue(empty['exact_correct'])
        self.assertEqual((empty['precision'], empty['recall'], empty['partial_score']), (0, 0, 0))

    def test_partial_domain_outside_prediction_is_invalid(self):
        req = request('partial_order')
        req['query_pairs'] = [req['gold_pairs'][0]]
        schema = a.schema_for(list(req['display_to_option_id']), [], 'partial_order')
        checked = a.validate(returned(req, 'I2<I3'), req, schema)
        self.assertFalse(checked['final_answer_valid'])
        self.assertIn('ANSWER_OUTSIDE_QUERY_DOMAIN', checked['answer_errors'])

    def test_synthetic_frozen_binding_is_enforced(self):
        req = request('multi_select')
        output = self.sandbox([(req, returned(req))], bad_binding=True)
        with self.assertRaisesRegex(AssertionError, 'FROZEN_INPUT_CHANGED'):
            a.score()
        self.assertFalse((output / 'SCORES.json').exists())
        self.assertFalse((output / 'RAW_ANSWERS.json').exists())

    def test_only_two_scorer_functions_change_and_entry_is_disabled(self):
        before = ast.parse(FROZEN.read_text())
        after = ast.parse((PROPOSAL / 'adapter.py.proposed').read_text())
        old_functions = {node.name: ast.dump(node, include_attributes=False) for node in before.body if isinstance(node, ast.FunctionDef)}
        new_functions = {node.name: ast.dump(node, include_attributes=False) for node in after.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(set(old_functions), set(new_functions))
        self.assertEqual({name for name in old_functions if old_functions[name] != new_functions[name]}, {'invalid', 'score'})
        self.assertEqual(ast.dump(ast.Module(body=before.body[:-1], type_ignores=[]), include_attributes=False).replace(old_functions['invalid'], new_functions['invalid']).replace(old_functions['score'], new_functions['score']), ast.dump(ast.Module(body=after.body[:-1], type_ignores=[]), include_attributes=False))
        self.assertIsInstance(after.body[-1], ast.If)
        self.assertIsInstance(after.body[-1].body[0], ast.Raise)
        with self.assertRaisesRegex(SystemExit, 'OFFLINE_PROPOSAL_ONLY'):
            exec(compile((PROPOSAL / 'adapter.py.proposed').read_bytes(), '<disabled-proposal-entry>', 'exec'), {'__name__': '__main__', '__file__': str(FROZEN)})


if __name__ == '__main__':
    raise SystemExit('Use offline_validate.py; standalone test execution does not produce the binding audit.')
