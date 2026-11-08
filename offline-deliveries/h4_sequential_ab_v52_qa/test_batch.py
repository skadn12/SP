"""Offline real five-chunk pipeline, with HTTP and model execution mocked."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import batch
import evaluate_and_pool as pipeline
import qa_transport


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix='v52-bulk-SYNTHETIC-')))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch('socket.socket', side_effect=AssertionError('No network in batch tests')))
        self.adapter = pipeline.generation
        self.run = self.temp / 'batch'
        self.root_id = 'cc-train-episode-138343'
        self.source_path = self.temp / 'collection/runs/fixture/CANDIDATE_HISTORY.json'
        self.source_path.parent.parent.parent.mkdir(parents=True)
        (self.source_path.parent.parent.parent / 'candidate_runner.py').write_text('# SYNTHETIC audit implementation\n')
        scenes = [f'CANDIDATE:{route}:S{number:02d}' for route in ('A','B') for number in range(1,4)]
        turns = [{'turn_ref': 'P:T001', 'speaker': 'Alex', 'text': 'SYNTHETIC original baseline.'}]
        turns += [{'turn_ref': scenes[i // 10] + ':T' + f'{i % 10 + 1:03d}',
                  'scene_ref': scenes[i // 10], 'speaker': 'Alex',
                  'text': f'SYNTHETIC observed fixture event number {i+1:03d}.'} for i in range(60)]
        for turn in turns[1:]:
            turn.update(origin='provisional_candidate', training_pool_eligible=False)
        self.source = {'root_id': self.root_id, 'new_admitted_scenes': 0,
            'retained_candidate_scenes': 2, 'training_pool_eligible': False,
            'formal_baseline_turns': 1,
            'history_policy': 'formal_prefix_then_provisional_retained_dialogue', 'turns': turns}
        self.adapter.write(self.source_path, self.source)
        expansion = self.source_path.parent
        self.adapter.write(expansion / 'FORMAL_BASELINE.json', {'turns': turns[:1]})
        self.adapter.write(expansion / 'CANDIDATE_RECORDS.json', [
            {'candidate_id': 'candidate-' + str(i), 'disposition': 'retained_pending_review',
             'realization': {'turns': turns[1+i*10:1+(i+1)*10]}} for i in range(6)])
        self.adapter.write(expansion / 'COLLECTION_BINDINGS.json', {'files': self.adapter.base.bindings([
            self.source_path, expansion / 'FORMAL_BASELINE.json', expansion / 'CANDIDATE_RECORDS.json'])})
        self.adapter.write(expansion / 'MANIFEST.json', {'version': self.adapter.EXPANSION_VERSION,
            'scope': 'fresh_collection', 'root_id': self.root_id, 'model': self.adapter.MODEL,
            'training_pool_eligible': False})
        self.prior = self.temp / 'prior.jsonl'
        self.prior.write_text('')
        self.old_bytes = {p: p.read_bytes() for p in expansion.iterdir() if p.is_file()}
        self.old_bytes[self.prior] = self.prior.read_bytes()
        audit = {'status': 'ARTIFACT_REPLAY_VERIFIED', 'semantic_certified': False,
                 'training_pool_eligible': False, 'synthetic_test_only': True}
        self.stack.enter_context(patch.object(self.adapter.collection_source, 'replay_audit', return_value=audit))
        self.stack.enter_context(patch.object(self.adapter, '_dependency_paths', return_value=({}, {Path(self.adapter.__file__)})))
        self.stack.enter_context(patch.object(self.adapter, 'DEFAULT_PRIOR', self.prior))
        self.stack.enter_context(patch.object(self.adapter.v15, 'prior_candidates', side_effect=lambda root: ([], [])))
        self.stack.enter_context(patch.object(qa_transport, 'shared_lock', contextlib.nullcontext))
        self.contexts, self.callers, self.eval_actions = [], [], []
        self.raw_gold = {}
        self.first_mc = None
        self.omissions = False
        test = self

        class MockCaller:
            def __init__(self, run):
                self.run, self.counter = run, 0
                test.callers.append(self)
                self.chunk_index = len(test.callers)

            def call(self, label, payload, proof):
                self.counter += 1
                adapter = test.adapter
                directory = self.run / 'requests' / f'{self.counter:03d}-{label}'
                adapter.write(directory / 'REQUEST.json', payload)
                adapter.write(directory / 'PROOF.json', {**proof, 'payload_sha256': adapter.base.digest(payload)})
                adapter.write(directory / 'HTTP_ATTEMPT.json', {'synthetic_test_only': True,
                    'actual_payload_equals_frozen': True, 'payload_sha256': adapter.base.digest(payload)})
                if proof['role'] == 'ordering_reviewer':
                    context = pipeline._input_json(payload, '\nDATA:\n')
                    q = context['candidates'][0]
                    items = {}
                    for key, text in q['items'].items():
                        turn = next(t for t in test.source['turns'] if t['text'] == text)
                        items[key] = {'supported': True, 'order_anchor': {'turn_ref': turn['turn_ref'],
                            'quote': text, 'reason': 'SYNTHETIC exact anchor.'}, 'support_refs': [],
                            'alternative_matching_anchors': [], 'actor_ok': True, 'modality_ok': True,
                            'state_ok': True, 'neutral_stem_ok': True, 'reason': 'SYNTHETIC support.'}
                    value = {'reviews': [{'candidate_id': q['candidate_id'], 'items': items,
                        'coherent': True, 'unique_order': True, 'alternative_orders': [],
                        'neutral_stem_ok': True, 'reason': 'SYNTHETIC review.'}]}
                else:
                    context = pipeline._input_json(payload, '\nINPUT_JSON\n')
                    test.contexts.append((self.chunk_index, proof['question_type'], deepcopy(context)))
                    questions, abstentions = [], []
                    for slot, qid in enumerate(context['candidate_ids']):
                        if test.omissions and self.chunk_index == 5 and proof['question_type'] == 'ordering' and slot == 1:
                            abstentions.append({'candidate_id': qid, 'reason': 'SYNTHETIC abstention; no replacement.'})
                            continue
                        if proof['question_type'] == 'multi_select':
                            q = {'candidate_id': qid, 'topic': 'SYNTHETIC test',
                                 'question': 'Fixture facts? Select all that apply.', 'options': [
                                {'turn_ref': t['turn_ref'], 'speaker': t['speaker'], 'quote': t['text'],
                                 'proposition': f'SYNTHETIC {qid} claim {i}.',
                                 'relation': 'supports' if i < 3 else 'contradicts', 'rationale': 'SYNTHETIC relation.'}
                                for i, t in enumerate(test.source['turns'][:6])]}
                            if self.chunk_index == 1 and slot == 0:
                                test.first_mc = deepcopy(q)
                            if test.omissions and self.chunk_index == 3 and slot == 0:
                                q = {**deepcopy(test.first_mc), 'candidate_id': qid}
                        else:
                            offset = (self.chunk_index - 1) * 12 + slot * 6
                            selected_turns = test.source['turns'][offset:offset + 6]
                            q = {'candidate_id': qid, 'root_id': test.root_id, 'question_type': 'ordering',
                                'topic': 'SYNTHETIC test', 'chronology_kind': 'conversation_order',
                                'question': 'Order statements as they APPEAR IN THE SUPPLIED DIALOGUE.',
                                'item_evidence': {f'I{i}': {'turn_ref': t['turn_ref'], 'speaker': t['speaker'],
                                    'quote': t['text'], 'status': 'statement', 'reason': 'SYNTHETIC evidence.'}
                                    for i, t in enumerate(selected_turns, 1)},
                                'items': {f'I{i}': t['text'] for i, t in enumerate(selected_turns, 1)},
                                'gold': [f'I{i}' for i in range(6, 0, -1)], 'order_reason': 'SYNTHETIC writer order.'}
                        test.raw_gold[qid] = q.get('gold', list('ABC'))
                        questions.append(q)
                    value = {'root_id': test.root_id, 'questions': questions, 'abstentions': abstentions}
                raw = json.dumps(value)
                metadata = {'requested_model': adapter.MODEL, 'returned_model': adapter.MODEL,
                    'finish_reason': 'stop', 'http_status': 200, 'synthetic_test_only': True}
                adapter.write(directory / 'RESPONSE.json', {'text': raw, 'metadata': metadata})
                return raw, directory, metadata

        self.stack.enter_context(patch.object(qa_transport, 'Caller', MockCaller))
        real_preflight = pipeline.evaluator.preflight

        def mock_model_preflight(pool_path, run, **options):
            self.assertTrue(options['require_model'])
            self.assertEqual(options['solver_model'], pipeline.assembler.TARGET)
            probe = pipeline.evaluator.load_probe('v52_bulk_offline_probe')
            decoder = {'temperature': .8, 'top_p': .9, 'max_tokens': 32, 'enable_thinking': False,
                'structured_output_api': 'SYNTHETIC', 'structured_output_version': 'finite_choice_set_v1'}
            with patch.object(pipeline.evaluator, 'load_probe', return_value=probe), \
                 patch.object(probe.solver, 'generation_config', return_value=decoder):
                real_preflight(pool_path, run, require_ordering_review=True, require_model=False,
                    solver_model=pipeline.assembler.TARGET)
            identity = pipeline.read(run / 'solver_identity.json')
            identity.update(synthetic_test_only=True)
            (run / 'solver_identity.json').write_text(json.dumps(identity))
            manifest = pipeline.read(run / 'manifest.json')
            manifest['solver_identity'] = identity
            (run / 'manifest.json').write_text(json.dumps(manifest))

        def mock_model_check(inspector, identity):
            self.assertTrue(identity.get('synthetic_test_only'))
            self.assertEqual(identity['model_identifier'], pipeline.assembler.TARGET)

        def mock_solver_live(run):
            manifest = pipeline.read(run / 'manifest.json')
            gold = pipeline.read(run / 'gold.json')
            qs = {q['candidate_id']: q for q in pipeline.read(run / 'questions.json')['candidates']}
            for request in manifest['requests']:
                qid = request['candidate_id']
                answer = gold[qid][:1] if request['question_type'] == 'multi_select' else [gold[qid][1], gold[qid][0], *gold[qid][2:]]
                mapping = {content: label for label, content in request['variant']['display_to_content'].items()}
                raw = '<box>' + ','.join(mapping[cid] for cid in answer) + '</box>'
                scored = pipeline.evaluator.score_response({**qs[qid], 'gold_content_ids': gold[qid]}, request, raw)
                self.adapter.write(run / 'responses' / (request['request_id'] + '.json'), {
                    **{key: request[key] for key in ('request_id', 'candidate_id', 'root_id', 'question_type', 'mode', 'slot', 'seed')},
                    'variant_id': request['variant']['variant_id'], 'raw_output': raw,
                    'solver_model': pipeline.assembler.TARGET, 'model_path': str(pipeline.assembler.CHECKPOINT),
                    'metadata': {**manifest['decoder_contract'], 'finish_reason': 'stop', 'synthetic_test_only': True},
                    'score': scored, 'evaluation_valid': True, 'data_errors': []})
            self.adapter.write(run / 'COMPLETE.json', {'status': 'COMPLETE', 'actual_call_count': len(manifest['requests']),
                'training_started': False, 'training_promotion': False, 'synthetic_test_only': True})

        def eval_action(action, evaluation_run, generation_run=None, log_path=None):
            self.assertTrue((self.run / 'GENERATION_COMPLETE.json').is_file())
            count = self.adapter.read(self.run / 'CONFIG.json')['chunk_count']
            self.assertEqual(len(self.callers), count)
            self.assertTrue(all((batch.chunk_paths(self.run, i)[0] / 'REPORT.json').is_file() for i in range(1, count + 1)))
            self.eval_actions.append((action, str(evaluation_run)))
            if log_path is not None:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                Path(log_path).write_text('SYNTHETIC evaluation boundary log\n')
            if action == 'preflight':
                return pipeline.preflight(generation_run, evaluation_run)
            return getattr(pipeline, action)(evaluation_run)

        self.stack.enter_context(patch.object(pipeline.evaluator, 'preflight', side_effect=mock_model_preflight))
        self.stack.enter_context(patch.object(pipeline.assembler.Inspector, 'model', mock_model_check))
        self.stack.enter_context(patch.object(pipeline.evaluator, 'live', side_effect=mock_solver_live))
        self.stack.enter_context(patch.object(batch, 'run_evaluation_action', side_effect=eval_action))

    def prepare(self):
        return batch.prepare(self.run, candidate_history=self.source_path, prior_pools=[self.prior], root=self.root_id)

    def test_twenty_slots_run_real_generation_review_solver_and_merged_pool(self):
        self.run_full_batch()

    def test_eight_slots_freeze_explicit_focus_priors_and_stop_at_two_chunks(self):
        import slot_focus
        plan = slot_focus.build_plan(self.root_id, self.run.name, self.source['turns'], slots=8)
        plan['coverage_rule'] = slot_focus.EXPLICIT_RULE
        first = next(iter(plan['chunks'][0]['slot_focus_by_candidate_id'].values()))
        first['preferred_scene_refs'] = ['CANDIDATE:A:S03']
        first['focus'] = 'Explicit actor and object references in stated concerns.'
        plan_path = self.temp / 'explicit-focus.json'
        self.adapter.write(plan_path, plan)
        batch.prepare(self.run, candidate_history=self.source_path, prior_pools=[self.prior],
                      root=self.root_id, candidate_slots=8, focus_plan=plan_path)
        config = self.adapter.read(self.run / 'CONFIG.json')
        self.assertEqual(config['budget'], {'candidate_slots': 8, 'qwen_http_attempts_max': 12, 'full_solver_calls_max': 120})
        self.assertEqual(config['chunk_count'], 2)
        self.assertEqual(self.adapter.read(self.run / 'SLOT_FOCUS_PLAN.json'), plan)
        self.assertIn(str(plan_path), {row['path'] for row in self.adapter.read(self.run / 'MANIFEST.json')['files']})
        generated = batch.generate(self.run)
        self.assertEqual(generated['candidate_slots'], 8)
        self.assertEqual(len(self.callers), 2)
        self.assertFalse(batch.chunk_paths(self.run, 3)[0].exists())
        self.assertFalse(self.eval_actions)
        evaluated = batch.evaluate(self.run)
        self.assertEqual(evaluated['full_solver_calls'], 120)
        result = batch.finalize(self.run)
        self.assertEqual(result['qa_selected_unique_candidates'], 8)
        self.assertEqual(len(self.eval_actions), 6)
        self.assertFalse(batch.chunk_paths(self.run, 3)[1].exists())
        batch.verify(self.run)

    def test_cross_chunk_exact_duplicate_and_abstention_do_not_get_replacement_slots(self):
        self.omissions = True
        self.run_full_batch()

    def run_full_batch(self):
        self.prepare()
        self.assertEqual(batch.BUDGET, {'candidate_slots': 20, 'qwen_http_attempts_max': 30, 'full_solver_calls_max': 300})
        batch.generate(self.run)
        self.assertEqual(len(self.callers), 5)
        self.assertEqual(len(self.contexts), 10)
        self.assertFalse(self.eval_actions)
        expected_ready, expected_calls = (18, 270) if self.omissions else (20, 300)
        sources, ready, all_focus, focus_mismatches_accepted = [], [], {}, []
        plan = self.adapter.read(self.run / 'SLOT_FOCUS_PLAN.json')
        for index in range(1, 6):
            generated, evaluated = batch.chunk_paths(self.run, index)
            self.assertFalse(evaluated.exists())
            manifest = self.adapter.read(generated / 'MANIFEST.json')
            self.assertEqual(manifest['budget']['candidate_slots'], 4)
            self.assertEqual(manifest['candidate_slots_per_type_per_root'], 2)
            source = self.adapter.read(generated / 'sources' / (self.root_id + '.json'))
            sources.append(source)
            ready.extend(self.adapter.read(generated / 'SOLVER_POOL.json')['questions'])
            contexts = [context for number, kind, context in self.contexts if number == index]
            self.assertEqual(len(contexts), 2)
            self.assertEqual(contexts[0]['previous_candidates_dedup_only'], contexts[1]['previous_candidates_dedup_only'])
            self.assertTrue(all(context['source_turns'] == source['turns'] for context in contexts))
            for context in contexts:
                self.assertEqual(len(context['candidate_ids']), 2)
                assigned = context['slot_focus_by_candidate_id']
                self.assertEqual(set(assigned), set(context['candidate_ids']))
                expected = plan['chunks'][index - 1]['slot_focus_by_candidate_id']
                self.assertEqual(assigned, {qid: expected[qid] for qid in context['candidate_ids']})
                self.assertEqual(len({focus['focus'] for focus in assigned.values()}), 2)
                self.assertEqual(context['priority_new_scene_refs'], [])
                all_focus.update(assigned)
                for previous in context['previous_candidates_dedup_only']:
                    self.assertFalse(set(previous) & {'gold', 'original_gold', 'option_evidence', 'training_selection',
                        'score', 'scores', 'answers', 'raw_output', 'relation'})
            if index > 1:
                prior_ids = {q['candidate_id'] for q in contexts[0]['previous_candidates_dedup_only']}
                self.assertIn(self.first_mc['candidate_id'], prior_ids)
            for question in self.adapter.read(generated / 'SOLVER_POOL.json')['questions']:
                preferred = set(all_focus[question['candidate_id']]['preferred_scene_refs'])
                entries = question.get('option_evidence', question.get('item_evidence', {}))
                cited_scenes = {entry['turn_ref'].rsplit(':T', 1)[0] for entry in entries.values()}
                if not cited_scenes.intersection(preferred):
                    focus_mismatches_accepted.append(question['candidate_id'])
        self.assertTrue(all(source == sources[0] for source in sources))
        self.assertEqual(len(all_focus), 20)
        self.assertEqual(len({focus['focus'] for focus in all_focus.values()}), 20)
        self.assertEqual({ref for focus in all_focus.values() for ref in focus['preferred_scene_refs']},
            {f'CANDIDATE:{route}:S{number:02d}' for route in ('A','B') for number in range(1,4)})
        self.assertTrue(all(focus['requirement'] == 'soft_preference_not_admission_gate' for focus in all_focus.values()))
        self.assertTrue(focus_mismatches_accepted, 'Existing acceptance must not require matching focus scenes')
        self.assertEqual(len(ready), expected_ready)
        self.assertEqual(len({q['candidate_id'] for q in ready}), expected_ready)
        if self.omissions:
            report = self.adapter.read(batch.chunk_paths(self.run, 3)[0] / 'REPORT.json')
            rejected = [r for outcome in report['roots'] for r in outcome['mechanical_rejected']]
            self.assertTrue(any('duplicate_historical_selected_statement_set' in r['errors'] for r in rejected))
            report = self.adapter.read(batch.chunk_paths(self.run, 5)[0] / 'REPORT.json')
            self.assertEqual(sum(len(row['abstentions']) for row in report['roots']), 1)
        raw_before = {p: p.read_bytes() for index in range(1, 6)
                      for p in batch.chunk_paths(self.run, index)[0].glob('**/RESPONSE.json')}
        batch.evaluate(self.run)
        self.assertEqual(len(self.eval_actions), 15)
        batch.finalize(self.run)
        batch.verify(self.run)
        pool = self.run / 'training_pool/TRAINING_POOL.jsonl'
        rows = [json.loads(line) for line in pool.read_text().splitlines()]
        self.assertEqual(len(rows), expected_ready)
        self.assertEqual(len({row['canonical_question_id'] for row in rows}), expected_ready)
        requests = [pipeline.read(batch.chunk_paths(self.run, i)[1] / 'solver/manifest.json')['requests'] for i in range(1, 6)]
        self.assertEqual(sum(map(len, requests)), expected_calls)
        for row in rows:
            self.assertTrue(Path(row['source_path']).is_file())
            self.assertEqual(pipeline.read(Path(row['source_path'])), row['full_context'])
            self.assertEqual(row['full_context'], sources[0])
            self.assertFalse(row['training_allowed'])
            self.assertEqual(row['training_admitted_questions'], 0)
            if row['question_type'] == 'ordering':
                self.assertEqual(row['gold'], [f'I{i}' for i in range(1, 7)])
                self.assertEqual(row['original_selected_question']['original_gold'], self.raw_gold[row['candidate_id']])
            else:
                self.assertEqual(row['gold'], list('ABC'))
        self.assertTrue(all(p.read_bytes() == data for p, data in {**self.old_bytes, **raw_before}.items()))

    def test_incomplete_batch_cannot_evaluate_or_finalize(self):
        self.prepare()
        with self.assertRaises(ValueError):
            batch.evaluate(self.run)
        with self.assertRaises(ValueError):
            batch.finalize(self.run)
        self.assertFalse(self.eval_actions)
        self.assertFalse((self.run / 'training_pool').exists())

    def test_failed_fifth_evaluation_preserves_four_chunks_without_merging_or_retry(self):
        self.prepare()
        batch.generate(self.run)
        mock_boundary = batch.run_evaluation_action
        successful_boundary = mock_boundary.side_effect
        fifth_evaluation = batch.chunk_paths(self.run, 5)[1]
        attempts = []

        def fail_fifth(action, evaluation_run, generation_run=None, log_path=None):
            attempts.append((action, evaluation_run))
            if evaluation_run == fifth_evaluation:
                raise RuntimeError('SYNTHETIC interruption before fifth Solver planning')
            return successful_boundary(action, evaluation_run, generation_run, log_path)

        mock_boundary.side_effect = fail_fifth
        with self.assertRaisesRegex(RuntimeError, 'SYNTHETIC interruption'):
            batch.evaluate(self.run)
        self.assertEqual(len(attempts), 13)
        self.assertTrue((self.run / 'EVALUATION_ERROR.json').is_file())
        self.assertFalse((self.run / 'EVALUATION_COMPLETE.json').exists())
        preserved = {path: path.read_bytes() for index in range(1, 5)
                     for path in (batch.chunk_paths(self.run, index)[1] / 'training_pool').rglob('*')
                     if path.is_file()}
        self.assertEqual(sum(path.name == 'TRAINING_POOL.jsonl' for path in preserved), 4)
        with self.assertRaises(ValueError):
            batch.finalize(self.run)
        with self.assertRaises(ValueError):
            batch.evaluate(self.run)
        self.assertEqual(len(attempts), 13)
        self.assertFalse((self.run / 'training_pool').exists())
        self.assertTrue(all(path.read_bytes() == data for path, data in preserved.items()))

    def test_public_inventory_strips_current_gold_scores_and_private_review_evidence(self):
        q = {'candidate_id': 'synthetic', 'root_id': self.root_id, 'question_type': 'multi_select',
             'question': 'SYNTHETIC question', 'options': {label: 'SYNTHETIC ' + label for label in 'ABCDEF'},
             'gold': ['A'], 'original_gold': ['B'], 'option_evidence': {'SECRET': 'relation'},
             'qwen_review': {'SECRET': 'review'}, 'training_selection': {'full_partial_mean': '1/2'},
             'answers': ['SECRET SOLVER ANSWER'], 'raw_output': 'SECRET'}
        public = batch.public_question(q)
        self.assertEqual(set(public), {'candidate_id', 'root_id', 'question_type', 'question', 'options'})
        self.assertNotIn('SECRET', json.dumps(public))
        public['options']['A'] = 'changed copy'
        self.assertEqual(q['options']['A'], 'SYNTHETIC A')


if __name__ == '__main__':
    unittest.main()
