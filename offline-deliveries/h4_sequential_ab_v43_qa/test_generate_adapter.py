"""Offline v43 candidate bridge and actual QA validation with mocked HTTP only."""
from contextlib import ExitStack, nullcontext, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate as adapter
import qa_transport


def input_json(payload, marker):
    content = payload['messages'][0]['content']
    return json.JSONDecoder().raw_decode(content.split(marker, 1)[1].lstrip())[0]


class CandidateGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='v43-qa-SYNTHETIC-')
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.upstream = self.home / 'upstream'
        self.root = 'cc-train-episode-104814'
        self.history = self.upstream / 'CANDIDATE_HISTORY.json'
        self.turns = [{"turn_ref": 'P:T001', 'speaker': 'Alex', 'text': 'Original full history.'}]
        self.turns += [{'turn_ref': f'CANDIDATE:B:S01:T{i:03d}', 'speaker': 'Alex',
                       'text': f'SYNTHETIC fixture event {word}.', 'origin': 'provisional_candidate',
                       'training_pool_eligible': False} for i, word in enumerate(
                           ('one', 'two', 'three', 'four', 'five', 'six'), 1)]
        adapter.write(self.upstream / 'MANIFEST.json', {'version': adapter.EXPANSION_VERSION,
            'scope': 'fresh_collection', 'model': adapter.MODEL, 'root_id': self.root,
            'training_pool_eligible': False})
        adapter.write(self.history, {'turns': self.turns, 'formal_baseline_turns': 1,
            'history_policy': 'formal_prefix_then_provisional_retained_dialogue', 'training_pool_eligible': False})
        adapter.write(self.upstream / 'FORMAL_BASELINE.json', {'turns': self.turns[:1]})
        adapter.write(self.upstream / 'CANDIDATE_RECORDS.json', [{'candidate_id': self.root + '/B/S01/1',
            'disposition': 'retained_pending_review', 'realization': {'turns': self.turns[1:]}}])
        adapter.write(self.upstream / 'COLLECTION_BINDINGS.json', {'files': adapter.base.bindings([
            self.history, self.upstream / 'MANIFEST.json', self.upstream / 'FORMAL_BASELINE.json',
            self.upstream / 'CANDIDATE_RECORDS.json'])})
        self.audit = {'status': 'ARTIFACT_REPLAY_VERIFIED', 'semantic_certified': False,
                      'training_pool_eligible': False}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(adapter.collection_source, 'replay_audit', return_value=self.audit))
        self.stack.enter_context(patch('socket.socket', side_effect=AssertionError('No network in tests')))

    def prepare(self, slots=2):
        prior = self.home / 'prior.jsonl'
        prior.write_text(json.dumps({'candidate_id': 'prior', 'root_id': self.root,
            'question_type': 'multi_select', 'question': 'Prior inventory', 'options': {'A': 'OLD FACT'},
            'gold': ['A'], 'solver_output': 'DO NOT LEAK', 'full_partial_mean': 0.5}) + '\n')
        run = self.home / 'generation'
        with patch.object(adapter, '_dependency_paths', return_value=({}, {Path(adapter.__file__)})), \
             patch.object(adapter, 'DEFAULT_PRIOR', prior), \
             patch.object(adapter.v15, 'prior_candidates', return_value=([], [])):
            adapter.prepare(run, self.history, slots=slots)
        return run

    def test_full_history_preserved_without_scene_promotion(self):
        run = self.prepare()
        manifest = adapter.verify(run)
        source = adapter.read(Path(manifest['source_rows'][0]['source_path']))
        self.assertEqual(source['turns'], self.turns)
        self.assertFalse(source['training_allowed'])
        self.assertFalse(source['training_pool_eligible'])
        self.assertEqual(source['upstream_admitted_scenes'], 0)
        self.assertEqual(source['retained_candidate_scenes'], 1)
        self.assertEqual(manifest['budget']['candidate_slots'], 4)
        self.assertEqual(manifest['budget']['http_attempts_max'], 6)
        self.assertEqual(manifest['budget']['solver_full_calls_max'], 60)

    def test_changed_prefix_and_promoted_suffix_are_rejected(self):
        original = adapter.read(self.history)
        for field in ('prefix', 'promotion', 'duplicate'):
            candidate = deepcopy(original)
            if field == 'prefix': candidate['turns'][0]['text'] = 'CHANGED'
            elif field == 'promotion': candidate['turns'][-1]['training_pool_eligible'] = True
            else: candidate['turns'][-1]['turn_ref'] = candidate['turns'][-2]['turn_ref']
            self.history.write_text(json.dumps(candidate))
            with self.subTest(field=field), self.assertRaises(ValueError):
                adapter.checked_sources(self.history)
        self.history.write_text(json.dumps(original))

    def test_upstream_artifact_audit_failure_blocks_prepare(self):
        with patch.object(adapter.collection_source, 'replay_audit', return_value={'status': 'FAIL'}):
            with self.assertRaisesRegex(ValueError, 'candidate_artifact_audit_failed'):
                self.prepare()
        self.assertFalse((self.home / 'generation').exists())

    def test_local_tamper_or_metadata_promotion_blocks_verify(self):
        run = self.prepare()
        manifest = adapter.read(run / 'MANIFEST.json')
        manifest['training_allowed'] = True
        (run / 'MANIFEST.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'candidate_status_promoted'):
            adapter.verify(run)
        manifest['training_allowed'] = False
        (run / 'MANIFEST.json').write_text(json.dumps(manifest))
        source_path = Path(manifest['source_rows'][0]['source_path'])
        source = adapter.read(source_path)
        source['turns'][-1]['text'] = 'EDITED'
        source_path.write_text(json.dumps(source))
        with self.assertRaises(ValueError): adapter.verify(run)

    def test_current_answers_and_gold_do_not_enter_prior_inventory(self):
        run = self.prepare()
        prior = adapter.read(run / 'prior' / (self.root + '.json'))
        self.assertNotIn('gold', prior[0])
        self.assertNotIn('solver_output', prior[0])
        self.assertNotIn('DO NOT LEAK', json.dumps(adapter.prompt_prior(prior)))

    def test_real_generation_validation_and_independent_ordering_review(self):
        run = self.prepare()
        source_turns = self.turns
        seen = []
        root = self.root

        class MockCaller:
            def __init__(self, path): self.run, self.counter = path, 0
            def call(self, label, payload, proof):
                self.counter += 1
                directory = self.run / 'requests' / f'{self.counter:03d}-{label}'
                seen.append(proof)
                adapter.write(directory / 'REQUEST.json', payload)
                adapter.write(directory / 'PROOF.json', {**proof, 'payload_sha256': adapter.base.digest(payload)})
                adapter.write(directory / 'HTTP_ATTEMPT.json', {'synthetic_test_only': True})
                if proof['role'] == 'ordering_reviewer':
                    context = input_json(payload, '\nDATA:\n')
                    q = context['candidates'][0]
                    items = {}
                    for key, text in q['items'].items():
                        turn = next(t for t in source_turns if t['text'] == text)
                        items[key] = {'supported': True, 'order_anchor': {'turn_ref': turn['turn_ref'],
                            'quote': text, 'reason': 'Synthetic exact fixture.'}, 'support_refs': [],
                            'alternative_matching_anchors': [], 'actor_ok': True, 'modality_ok': True,
                            'state_ok': True, 'neutral_stem_ok': True, 'reason': 'Synthetic support.'}
                    value = {'reviews': [{'candidate_id': q['candidate_id'], 'items': items,
                        'coherent': True, 'unique_order': True, 'alternative_orders': [],
                        'neutral_stem_ok': True, 'reason': 'Synthetic independent review.'}]}
                else:
                    context = input_json(payload, '\nINPUT_JSON\n')
                    if 'Prioritize new R02 developments' in payload['messages'][0]['content']:
                        raise AssertionError('legacy evidence priority survived')
                    if 'Prioritize newly supplied CANDIDATE developments' not in payload['messages'][0]['content']:
                        raise AssertionError('candidate evidence priority missing')
                    if context['source_turns'] != source_turns: raise AssertionError('history was truncated')
                    if context['priority_new_scene_refs'] != ['CANDIDATE:B:S01']: raise AssertionError('priority wrong')
                    questions = []
                    for n, qid in enumerate(context['candidate_ids']):
                        if proof['question_type'] == 'multi_select':
                            q = {'candidate_id': qid, 'topic': 'Synthetic fixture', 'question': 'Fixture? Select all that apply.',
                                'options': [{'proposition': f'Synthetic claim {n}.{i}.',
                                    'relation': 'supports' if i < 3 else 'contradicts', 'turn_ref': t['turn_ref'],
                                    'speaker': t['speaker'], 'quote': t['text'], 'rationale': 'Synthetic relation.'}
                                    for i, t in enumerate(source_turns[1:])]}
                        else:
                            # Two distinct exact sets, both using original full context.
                            turns = source_turns[1:] if n == 0 else source_turns[:6]
                            q = {'candidate_id': qid, 'root_id': root, 'question_type': 'ordering',
                                'chronology_kind': 'conversation_order',
                                'question': 'Order statements as they APPEAR IN THE SUPPLIED DIALOGUE.',
                                'items': {f'I{i}': t['text'] for i,t in enumerate(turns,1)},
                                'gold': [f'I{i}' for i in range(6,0,-1)],
                                'item_evidence': {f'I{i}': {'turn_ref': t['turn_ref'], 'speaker': t['speaker'],
                                    'quote': t['text'], 'status': 'statement', 'reason': 'Synthetic quote.'}
                                    for i,t in enumerate(turns,1)}}
                        questions.append(q)
                    value = {'root_id': root, 'questions': questions, 'abstentions': []}
                raw = json.dumps(value)
                metadata = {'requested_model': adapter.MODEL, 'returned_model': adapter.MODEL,
                            'finish_reason': 'stop', 'http_status': 200, 'synthetic_test_only': True}
                adapter.write(directory / 'RESPONSE.json', {'text': raw, 'metadata': metadata})
                return raw, directory, metadata

        with patch.object(qa_transport, 'Caller', MockCaller), \
             patch.object(qa_transport, 'shared_lock', return_value=nullcontext()), redirect_stdout(io.StringIO()):
            result = adapter.generate(run)
        self.assertEqual(result['ready_for_solver'], 4)
        self.assertEqual(result['http_attempts'], 4)
        self.assertEqual(sum(p['role'] == 'ordering_reviewer' for p in seen), 2)
        questions = adapter.read(run / 'SOLVER_POOL.json')['questions']
        self.assertTrue(all(q['source_admission_status'] == 'PROVISIONAL_CANDIDATE_HISTORY' for q in questions))
        self.assertTrue(all(q['new_scene_reference_diagnostic'] for q in questions))
        self.assertTrue(all(q['training_allowed'] is False for q in questions))
        ordering = [q for q in questions if q['question_type'] == 'ordering']
        self.assertTrue(all(q['original_gold'] == [f'I{i}' for i in range(6,0,-1)] for q in ordering))
        with patch.object(qa_transport, 'shared_lock', return_value=nullcontext()):
            with self.assertRaisesRegex(ValueError, 'generation_already_started_no_replay'):
                adapter.generate(run)


if __name__ == '__main__': unittest.main()
