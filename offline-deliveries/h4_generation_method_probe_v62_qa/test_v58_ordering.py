"""Offline coverage for restoring the existing ordering path on a single type."""
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate
import evaluate_and_pool as pipeline
import qa_transport
import test_generate_adapter as fixtures


class OrderingRestoreTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CandidateGenerationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_actual_ordering_payload_restores_review_budget_and_preserves_context(self):
        run = self.fixture.prepare()
        manifest = generate.verify(run)
        root = self.fixture.root
        payload, receipt = generate.make_payload(run, root, 'ordering',
            generate.initial_context(run, root, 'ordering'))
        context = fixtures.input_json(payload, '\nINPUT_JSON\n')
        self.assertEqual(manifest['question_types'], ['ordering'])
        self.assertEqual(manifest['budget'], {'candidate_slots': 2,
            'initial_generation_requests': 1, 'mechanical_repair_requests_max': 1,
            'ordering_review_requests_max': 2, 'http_attempts_max': 4,
            'solver_full_calls_max': 24})
        self.assertTrue(manifest['ordering_independent_qwen_review'])
        self.assertEqual(manifest['full_slots'], {'ordering': '12 random permutations x 1 full rollout'})
        self.assertEqual(payload['max_tokens'], 4096)
        self.assertEqual(context['source_turns'], self.fixture.turns)
        self.assertEqual(len(context['candidate_ids']), 2)
        self.assertTrue(all('-merged-order-' in key for key in context['candidate_ids']))
        self.assertNotIn('DO NOT LEAK', json.dumps(context))
        self.assertNotIn('gold', json.dumps(context))
        suffix = (run / 'feedback' / root / 'writer__M_clean_policy_feedback.txt').read_text()
        generate.prompt_layout.verify_request(payload, receipt, suffix, root, generate.TARGET,
            generate.base.digest((run / 'feedback/source_lock.json').read_bytes()), expected_kind='ordering')
        self.assertIn(generate.prompt_rules.ORDERING_STEM, payload['messages'][0]['content'])

    def test_wrong_type_budget_or_disabled_review_is_rejected(self):
        run = self.fixture.prepare()
        original = generate.read(run / 'MANIFEST.json')
        for field, value in [('question_types', ['multi_select']),
                             ('ordering_independent_qwen_review', False),
                             ('full_slots', {'ordering': '6 permutations'}),
                             ('budget', {**original['budget'], 'http_attempts_max': 2})]:
            with self.subTest(field=field):
                changed = {**original, field: value}
                (run / 'MANIFEST.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    generate.verify(run)
        (run / 'MANIFEST.json').write_text(json.dumps(original))
        for roots, slots in ((1, 1), (1, 4), (2, 2)):
            with self.assertRaises(ValueError):
                generate.generation_budget(roots, slots)

    def test_mocked_generation_runs_two_independent_reviews_and_preserves_public_content(self):
        run = self.fixture.prepare()
        root, turns = self.fixture.root, self.fixture.turns
        source = generate.read(Path(generate.verify(run)['source_rows'][0]['source_path']))
        seen, emitted = [], {}

        class MockCaller:
            def __init__(self, path): self.run, self.counter = path, 0

            def call(self, label, payload, proof):
                self.counter += 1
                directory = self.run / 'requests' / f'{self.counter:03d}-{label}'
                generate.write(directory / 'REQUEST.json', payload)
                generate.write(directory / 'PROOF.json', {**proof, 'payload_sha256': generate.base.digest(payload)})
                generate.write(directory / 'HTTP_ATTEMPT.json', {'synthetic_test_only': True})
                seen.append(deepcopy(proof))
                if proof['role'] == 'question_generator':
                    assert proof['question_type'] == 'ordering'
                    context = fixtures.input_json(payload, '\nINPUT_JSON\n')
                    assert context['source_turns'] == turns
                    questions = []
                    for index, qid in enumerate(context['candidate_ids']):
                        chosen = turns[1:] if index == 0 else turns[:6]
                        row = {'candidate_id': qid, 'root_id': root, 'question_type': 'ordering',
                            'chronology_kind': 'conversation_order', 'question': generate.prompt_rules.ORDERING_STEM,
                            'item_evidence': {f'I{i}': {'turn_ref': turn['turn_ref'], 'speaker': turn['speaker'],
                                'quote': turn['text'], 'status': 'statement', 'reason': 'Exact synthetic anchor.'}
                                for i, turn in enumerate(chosen, 1)},
                            'items': {f'I{i}': turn['text'] for i, turn in enumerate(chosen, 1)},
                            'gold': [f'I{i}' for i in range(6, 0, -1)]}
                        emitted[qid] = deepcopy(row)
                        questions.append(row)
                    value = {'root_id': root, 'questions': questions, 'abstentions': []}
                else:
                    assert proof['role'] == 'ordering_reviewer'
                    context = fixtures.input_json(payload, '\nDATA:\n')
                    assert context['source_turns'] == turns
                    row = context['candidates'][0]
                    assert 'gold' not in row and 'item_evidence' not in row
                    items = {}
                    for key, text in row['items'].items():
                        turn = next(t for t in turns if t['text'] == text)
                        items[key] = {'supported': True, 'order_anchor': {'turn_ref': turn['turn_ref'],
                            'quote': text, 'reason': 'Synthetic exact clause.'}, 'support_refs': [],
                            'alternative_matching_anchors': [], 'actor_ok': True, 'modality_ok': True,
                            'state_ok': True, 'neutral_stem_ok': True, 'reason': 'Synthetic supported item.'}
                    value = {'reviews': [{'candidate_id': row['candidate_id'], 'items': items,
                        'coherent': True, 'unique_order': True, 'alternative_orders': [],
                        'neutral_stem_ok': True, 'reason': 'Synthetic independent review.'}]}
                raw = json.dumps(value)
                metadata = {'requested_model': generate.MODEL, 'returned_model': generate.MODEL,
                    'finish_reason': 'stop', 'http_status': 200, 'synthetic_test_only': True}
                generate.write(directory / 'RESPONSE.json', {'text': raw, 'metadata': metadata})
                return raw, directory, metadata

        with patch.object(qa_transport, 'Caller', MockCaller), \
             patch.object(qa_transport, 'shared_lock', return_value=nullcontext()), redirect_stdout(io.StringIO()):
            report = generate.generate(run)
        self.assertEqual(report['candidate_slots'], 2)
        self.assertEqual(report['http_attempts'], 3)
        self.assertEqual(report['mechanical_pass'], 2)
        self.assertEqual(report['ready_for_solver'], 2)
        self.assertEqual([row['role'] for row in seen], ['question_generator', 'ordering_reviewer', 'ordering_reviewer'])
        questions = generate.read(run / 'SOLVER_POOL.json')['questions']
        for question in questions:
            raw = emitted[question['candidate_id']]
            self.assertEqual(question['items'], raw['items'])
            self.assertEqual(question['original_gold'], raw['gold'])
            self.assertEqual(question['gold'], [f'I{i}' for i in range(1, 7)])
            variants, _ = pipeline.evaluator._variants_for(question)
            self.assertEqual(len(variants), 12)
            self.assertEqual(len({tuple(v['items'].values()) for v in variants}), 12)
            self.assertTrue(all(set(v['items'].values()) == set(raw['items'].values()) for v in variants))
            changed_gold = {**question, 'gold': list(reversed(question['gold']))}
            self.assertEqual(pipeline.evaluator._variants_for(changed_gold)[0], variants)
        with patch.object(qa_transport, 'shared_lock', return_value=nullcontext()):
            with self.assertRaisesRegex(ValueError, 'generation_already_started_no_replay'):
                generate.generate(run)


class OrderingOnlyDeliveryTests(unittest.TestCase):
    def exercise(self, candidate, selected):
        with tempfile.TemporaryDirectory(prefix='v58-ordering-delivery-') as temp:
            home = Path(temp)
            def put(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            gen, solver, output = [home / name for name in ('generation', 'solver', 'delivery')]
            question = {'candidate_id': 'order', 'root_id': 'fixture', 'question_type': 'ordering',
                'question': generate.prompt_rules.ORDERING_STEM,
                'items': {f'I{i}': f'Fact {i}.' for i in range(1, 7)},
                'gold': [f'I{i}' for i in range(1, 7)], 'source_path': '/synthetic/source.json', 'source_sha256': 'a' * 64}
            questions = [question] if candidate else []
            calls = 12 if candidate else 0
            put(gen / 'REPORT.json', {'ready_for_solver': len(questions), 'candidate_slots': 2,
                'feedback_in_every_request': True, 'http_attempts': 1,
                'roots': [{'root_id': 'fixture', 'question_type': 'ordering', 'candidate_slots': 2,
                    'mechanical_pass': len(questions), 'ready_for_solver': len(questions),
                    'abstentions': [], 'mechanical_rejected': [], 'review_rejected': []}]})
            put(gen / 'SOLVER_POOL.json', {'questions': questions,
                'provenance': {'requested_model': 'synthetic', 'ordering_review_owner': 'synthetic'}})
            put(solver / 'manifest.json', {'pool_sha256': pipeline.delivery.sha(gen / 'SOLVER_POOL.json'), 'request_count': calls})
            put(solver / 'COMPLETE.json', {'actual_call_count': calls})
            put(solver / 'solver_identity.json', {'synthetic_test_only': True})
            put(solver / 'RESULTS.json', {'call_set_errors': [], 'solver_model': 'synthetic',
                'model_path': '/synthetic/model', 'policy_version': 'synthetic',
                'questions': [{'candidate_id': 'order', 'selected_by_policy': selected,
                    'actual_full_calls': calls, 'expected_full_calls': calls,
                    'full_partial_mean': '1/2' if selected else '1'}] if candidate else []})
            report = pipeline.delivery.deliver(gen, solver, output)
            self.assertEqual(report['selected_multi_select'], 0)
            self.assertEqual(report['selected_ordering'], int(candidate and selected))
            self.assertEqual(report['full_solver_calls'], calls)
            self.assertIn('| 0 / 0 |', (output / 'REPORT.md').read_text())
            delivered = pipeline.delivery.read(output / 'SELECTED_QUESTIONS.json')['questions']
            if selected:
                self.assertEqual(delivered[0]['items'], question['items'])
                self.assertEqual(delivered[0]['gold'], question['gold'])
            else:
                self.assertEqual(delivered, [])

    def test_selected_ordering_only(self): self.exercise(True, True)
    def test_all_rejected_ordering_only(self): self.exercise(True, False)
    def test_no_ready_ordering(self): self.exercise(False, False)


if __name__ == '__main__':
    unittest.main()
