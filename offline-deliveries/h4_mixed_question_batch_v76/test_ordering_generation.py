"""Offline writer validation and recorded-transport lifecycle tests.

The inherited JSON parser and exact-span gate are real. Provider discovery,
feedback payload assembly and every transport call use explicit local fakes.
Socket connections are forbidden throughout each test.
"""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from common import bind, load_multi, now, read, sha, write
import ordering_generation as generation


ENGINE = load_multi('asset_generation')
MODEL = ENGINE.legacy.MODEL
CHRONOLOGICAL = ['I3', 'I1', 'I6', 'I2', 'I5', 'I4']


def source_fixture():
    return {'root_id': 'root', 'metadata': 'Complete synthetic source', 'turns': [
        {'turn_ref': 'S1:T' + str(n), 'speaker': 'Ada' if n % 2 else 'Ben',
         'text': 'Witness statement ' + str(n).zfill(2) + '.'}
        for n in range(1, 13)
    ]}


def writer_row(candidate_id, source, offset=0):
    by_label = dict(zip(generation.LABELS, [2, 4, 1, 6, 5, 3]))
    turns = {label: source['turns'][n + offset - 1] for label, n in by_label.items()}
    return {
        'candidate_id': candidate_id, 'root_id': 'root', 'question_type': 'ordering',
        'chronology_kind': 'conversation_order',
        'question': 'Order these statements as they APPEAR IN THE SUPPLIED DIALOGUE.',
        'items': {label: turn['text'] for label, turn in turns.items()},
        'gold': list(CHRONOLOGICAL), 'order_reason': 'The exact clauses appear in this order.',
        'item_evidence': {label: {'turn_ref': turn['turn_ref'], 'speaker': turn['speaker'],
            'quote': turn['text'], 'status': 'statement', 'reason': 'Exact source clause.'}
            for label, turn in turns.items()},
    }


class OfflineEngine:
    """A transport fake that saves the same request/response envelope shape."""
    def __init__(self, rows, failed=False):
        self.rows = rows; self.failed = failed
        self.calls = []; self.payload_contexts = []; self.feedback_roots = []
        self.strict_json = ENGINE.strict_json
        self.base = SimpleNamespace(digest=ENGINE.base.digest)
        self.legacy = SimpleNamespace(MODEL=MODEL,
            legacy=SimpleNamespace(payload_with_feedback=self.payload_with_feedback))
        self.source_adapter = SimpleNamespace(feedback_data=self.feedback_data,
            freeze_feedback=self.freeze_feedback)
        self.lock = Mock(side_effect=lambda: nullcontext())
        self.qa_transport = SimpleNamespace(shared_lock=self.lock, Caller=self.caller)
        self._dependencies = Mock(return_value=({'provider': 'synthetic-offline'}, []))

    def feedback_data(self, root):
        self.feedback_roots.append(root)
        return {'source_lock': {'merge_current_generation_feedback': False,
            'status': 'EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY'},
            'text': 'Synthetic previous-round feedback.'}

    def freeze_feedback(self, root, folder, data):
        path = Path(folder) / 'SOURCE_LOCK.json'; write(path, data)
        return data, [path]

    def payload_with_feedback(self, run, root, instruction, context, max_tokens):
        self.payload_contexts.append(deepcopy(context))
        payload = {'model': MODEL, 'max_tokens': max_tokens, 'messages': [{'role': 'user',
            'content': instruction + '\nDATA:\n' + json.dumps(context, ensure_ascii=False)
                + '\nPREVIOUS_FEEDBACK:\nSynthetic previous-round feedback.'}]}
        return payload, {'root_id': root, 'previous_training_round_only': True}

    def caller(self, run):
        engine = self
        class Caller:
            def call(self, request_name, payload, proof):
                engine.calls.append((request_name, deepcopy(payload), deepcopy(proof)))
                folder = Path(run) / 'requests' / '0001'
                write(folder / 'REQUEST.json', payload); write(folder / 'PROOF.json', proof)
                write(folder / 'HTTP_ATTEMPT.json', {'actual_payload_equals_frozen': True,
                    'payload_sha256': engine.base.digest(payload), 'requested_model': MODEL,
                    'http_attempt_number': 1, 'transport_retries': 0})
                if engine.failed:
                    write(folder / 'ERROR.json', {'error': 'synthetic_transport_error'})
                    return None, folder, {'error': 'synthetic_transport_error'}
                raw = json.dumps({'questions': engine.rows}, ensure_ascii=False)
                metadata = {'returned_model': MODEL, 'finish_reason': 'stop', 'http_status': 200}
                write(folder / 'RESPONSE.json', {'text': raw, 'metadata': metadata})
                return raw, folder, metadata
        return Caller()


class OrderingGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.home = Path(self.tmp.name).resolve()
        self.source = source_fixture(); self.source_path = self.home / 'SOURCE.json'
        write(self.source_path, self.source)
        self.ids = ['question-1', 'question-2']
        self.manifest = {'root_id': 'root', 'candidate_ids': self.ids}
        self.rows = [writer_row(self.ids[0], self.source), writer_row(self.ids[1], self.source, 6)]
        self.fake = OfflineEngine(self.rows)
        self.engine_patch = patch.object(generation, 'engine', return_value=self.fake)
        self.network_patch = patch('socket.socket.connect', side_effect=AssertionError('Real network prohibited'))
        self.engine_patch.start(); self.network_patch.start()

    def tearDown(self):
        self.network_patch.stop(); self.engine_patch.stop(); self.tmp.cleanup()

    def validate(self, rows=None, prior=None):
        return generation.validate_raw(json.dumps({'questions': rows if rows is not None else self.rows}),
            self.manifest, self.source, self.source_path, prior or [])

    def prepare_home(self):
        prior_path = self.home / 'PRIOR_POOL.json'; write(prior_path, {'questions': []})
        config = {'batch_id': 'synthetic-batch', 'ordering_output_max_tokens': 6144,
            'ordering_slots': [{'slot_id': 'root_ordering', 'root_id': 'root',
                'source_path': str(self.source_path), 'source_sha256': sha(self.source_path),
                'history_path': str(self.source_path), 'history_sha256': sha(self.source_path),
                'prior_pool_path': str(prior_path), 'candidate_count': 2}]}
        write(self.home / 'CONFIG.json', config)
        ids = generation.candidate_ids(config, config['ordering_slots'][0])
        self.fake.rows = [writer_row(ids[0], self.source), writer_row(ids[1], self.source, 6)]
        generation.prepare(self.home)
        return self.home / 'ordering' / 'root_ordering'

    def test_exact_anchors_keep_nonidentity_raw_gold_and_text(self):
        original = deepcopy(self.rows); before = bind(self.source_path)
        accepted, rejected = self.validate()
        self.assertEqual(rejected, []); self.assertEqual(len(accepted), 2)
        self.assertEqual(self.rows, original); self.assertEqual(bind(self.source_path), before)
        for candidate, raw in zip(accepted, self.rows):
            self.assertEqual(candidate['gold'], CHRONOLOGICAL)
            self.assertEqual(candidate['original_gold'], raw['gold'])
            self.assertEqual(candidate['question'], raw['question'])
            self.assertEqual(candidate['items'], raw['items'])
            self.assertEqual(candidate['source_sha256'], sha(self.source_path))
            self.assertEqual(candidate['generator_model'], MODEL)
            self.assertEqual(candidate['status'], 'MECHANICAL_ONLY_REQUIRES_SOURCE_FIRST_REVIEW')
            for label, span in candidate['generated_order_anchor_spans'].items():
                turn = self.source['turns'][span['turn_index']]
                self.assertEqual(turn['text'][span['char_start']:span['char_end']], raw['item_evidence'][label]['quote'])

    def test_same_turn_nonoverlapping_clauses_have_exact_offsets(self):
        source = {'root_id': 'root', 'turns': [{'turn_ref': 'S1:T1', 'speaker': 'Ada',
            'text': 'Prefix. ' + ' '.join(self.source['turns'][n]['text'] for n in range(6))}]}
        path = self.home / 'SAME_TURN.json'; write(path, source)
        row = deepcopy(self.rows[0])
        for evidence in row['item_evidence'].values():
            evidence['turn_ref'] = 'S1:T1'; evidence['speaker'] = 'Ada'
        accepted, rejected = generation.validate_raw(json.dumps({'questions': [row, None]}),
            self.manifest, source, path, [])
        self.assertEqual(len(accepted), 1); self.assertEqual(rejected[0]['reasons'], ['writer_null_slot'])
        self.assertEqual(accepted[0]['gold'], CHRONOLOGICAL)
        spans = accepted[0]['generated_order_anchor_spans']
        for before, after in zip(CHRONOLOGICAL, CHRONOLOGICAL[1:]):
            self.assertLessEqual(spans[before]['char_end'], spans[after]['char_start'])
        self.assertEqual(spans['I3']['char_start'], len('Prefix. '))

    def test_raw_gold_disagreement_is_rejected_without_repair(self):
        self.rows[0]['gold'] = list(reversed(CHRONOLOGICAL))
        original = deepcopy(self.rows); before = bind(self.source_path)
        accepted, rejected = self.validate()
        self.assertEqual([row['candidate_id'] for row in accepted], [self.ids[1]])
        self.assertEqual(rejected, [{'candidate_id': self.ids[0],
            'reasons': ['writer_gold_disagrees_with_source_anchors']}])
        self.assertEqual(self.rows, original); self.assertEqual(bind(self.source_path), before)

    def test_malformed_and_wrong_top_level_schema_reject_every_slot(self):
        raw_values = [None, 'not-json', '{}', '[]', '{"questions":{}}', '{"questions":[]}',
            json.dumps({'questions': [None]}), json.dumps({'questions': [None, None, None]}),
            json.dumps({'questions': [None, None], 'extra': True}),
            '{"questions":[],"questions":[null,null]}', '{"questions":[NaN,null]}']
        for raw in raw_values:
            with self.subTest(raw=raw):
                accepted, rejected = generation.validate_raw(raw, self.manifest, self.source, self.source_path, [])
                self.assertEqual(accepted, []); self.assertEqual([r['candidate_id'] for r in rejected], self.ids)
                self.assertTrue(all(r['reasons'] for r in rejected))

    def test_null_slot_does_not_discard_valid_neighbor(self):
        accepted, rejected = self.validate([None, self.rows[1]])
        self.assertEqual([row['candidate_id'] for row in accepted], [self.ids[1]])
        self.assertEqual(rejected, [{'candidate_id': self.ids[0], 'reasons': ['writer_null_slot']}])

    def test_row_schema_and_identity_are_strict(self):
        mutations = [lambda r: r.pop('order_reason'), lambda r: r.update(extra='unused'),
            lambda r: r.update(candidate_id='wrong-slot'), lambda r: r.update(root_id='wrong-root'),
            lambda r: r.update(question_type='multi_select'), lambda r: r.update(chronology_kind='event_order'),
            lambda r: r.update(items={'I1': 'Only one item'}), lambda r: r.update(gold=['I1'] * 6),
            lambda r: r.update(item_evidence=[])]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                row = deepcopy(self.rows[0]); mutation(row)
                accepted, rejected = self.validate([row, None])
                self.assertEqual(accepted, []); self.assertEqual(len(rejected), 2)
                self.assertTrue(rejected[0]['reasons'])

    def test_anchor_quote_speaker_and_reference_must_match_source(self):
        mutations = [lambda e: e.update(quote='Not in source.'),
            lambda e: e.update(speaker='Wrong actor'), lambda e: e.update(turn_ref='S1:T99'),
            lambda e: e.update(char_start=1), lambda e: e.update(char_end=1)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                row = deepcopy(self.rows[0]); mutation(row['item_evidence']['I1'])
                accepted, rejected = self.validate([row, None])
                self.assertEqual(accepted, []); self.assertTrue(rejected[0]['reasons'])

    def test_repeated_quote_is_not_silently_located_at_first_hit(self):
        source = deepcopy(self.source)
        source['turns'][1]['text'] += ' ' + source['turns'][1]['text']
        path = self.home / 'REPEATED.json'; write(path, source)
        accepted, rejected = generation.validate_raw(json.dumps({'questions': [self.rows[0], None]}),
            self.manifest, source, path, [])
        self.assertEqual(accepted, [])
        self.assertIn('I1:quote_not_unique_in_turn', rejected[0]['reasons'])

    def test_prior_dedup_ignores_labels_case_and_whitespace(self):
        prior = {'items': {label: '  ' + text.upper().replace(' ', '   ') + '  '
            for label, text in zip(reversed(generation.LABELS), self.rows[0]['items'].values())}}
        accepted, rejected = self.validate(prior=[prior])
        self.assertEqual([row['candidate_id'] for row in accepted], [self.ids[1]])
        self.assertEqual(rejected[0]['reasons'], ['duplicate_prior_or_current_six_item_set'])

    def test_current_dedup_rejects_only_second_same_six_item_set(self):
        second = deepcopy(self.rows[0]); second['candidate_id'] = self.ids[1]
        accepted, rejected = self.validate([self.rows[0], second])
        self.assertEqual([row['candidate_id'] for row in accepted], [self.ids[0]])
        self.assertEqual(rejected, [{'candidate_id': self.ids[1],
            'reasons': ['duplicate_prior_or_current_six_item_set']}])

    def test_public_projection_excludes_gold_and_evidence_and_is_independent(self):
        accepted, _ = self.validate(); private = accepted[0]
        private['qwen_review'] = {'should_not': 'escape'}
        public = generation.public(private)
        expected = {'candidate_id', 'root_id', 'question_type', 'chronology_kind',
            'question', 'items', 'source_path', 'source_sha256'}
        self.assertEqual(set(public), expected)
        self.assertTrue({'gold', 'original_gold', 'item_evidence', 'generated_order_anchor_spans',
            'order_reason', 'qwen_review'}.isdisjoint(public))
        public['items']['I1'] = 'Public copy changed.'
        self.assertEqual(private['items']['I1'], self.rows[0]['items']['I1'])

    def test_prepare_freezes_complete_source_and_public_prior_without_http(self):
        run = self.prepare_home()
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(self.fake.feedback_roots, ['root'])
        self.assertEqual((run / 'SOURCE.json').read_bytes(), self.source_path.read_bytes())
        manifest = read(run / 'MANIFEST.json')
        self.assertEqual(manifest['full_source_turn_count'], 12)
        self.assertEqual(manifest['budget'], {'http_attempts_max': 1})
        self.assertIs(manifest['training_allowed'], False)
        self.assertEqual(generation.verify(self.home), {'candidate_count': 0, 'http_attempts': 0})
        self.assertFalse((self.home / 'ordering' / 'GENERATION.STARTED.json').exists())
        with self.assertRaisesRegex(ValueError, 'already_prepared'):
            generation.prepare(self.home)

    def test_payload_carries_full_source_feedback_and_frozen_public_prior(self):
        run = self.prepare_home(); payload, proof = generation.payload_for(run)
        context = self.fake.payload_contexts[-1]
        self.assertEqual(context['source_turns'], self.source['turns'])
        self.assertEqual(context['source_metadata'], {'root_id': 'root', 'metadata': 'Complete synthetic source'})
        self.assertEqual(context['previous_public_questions_dedup_only'], [])
        self.assertEqual(context['source_mode'], 'complete_frozen_sources')
        self.assertIs(context['training_allowed'], False)
        self.assertIn('Synthetic previous-round feedback.', payload['messages'][0]['content'])
        self.assertIn(generation.TASK, payload['messages'][0]['content'])
        self.assertEqual(proof['payload_sha256'], ENGINE.base.digest(payload))
        self.assertIs(proof['current_solver_feedback'], False)

    def test_mocked_generation_calls_once_and_frozen_replay_cannot_call_again(self):
        run = self.prepare_home(); before = bind(run / 'SOURCE.json')
        result = generation.generate(self.home)
        self.assertEqual(result['candidate_count'], 2); self.assertEqual(result['http_attempts'], 1)
        self.assertEqual(len(self.fake.calls), 1); self.assertEqual(self.fake.lock.call_count, 1)
        self.assertEqual(self.fake.calls[0][1], read(run / 'REQUEST.json'))
        private = read(self.home / 'ordering' / 'POOL.json')['questions']
        public = read(self.home / 'ordering' / 'PUBLIC_QUESTIONS.json')['questions']
        self.assertEqual(public, [generation.public(row) for row in private])
        self.assertTrue(all('gold' not in row and 'item_evidence' not in row for row in public))
        self.assertEqual(bind(run / 'SOURCE.json'), before)
        self.assertEqual(generation.verify(self.home), {'candidate_count': 2, 'http_attempts': 1})
        with self.assertRaisesRegex(ValueError, 'no_automatic_replay'):
            generation.generate(self.home)
        self.assertEqual(len(self.fake.calls), 1)

    def test_mocked_transport_error_has_no_retry_or_repaired_candidates(self):
        run = self.prepare_home(); self.fake.failed = True
        result = generation.generate(self.home)
        self.assertEqual(result['candidate_count'], 0); self.assertEqual(result['http_attempts'], 1)
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(read(run / 'POOL.json')['questions'], [])
        self.assertEqual(len(read(run / 'REJECTIONS.json')), 2)
        self.assertEqual(generation.verify(self.home), {'candidate_count': 0, 'http_attempts': 1})

    def test_verify_rejects_additional_saved_request_before_network(self):
        run = self.prepare_home()
        for number in ('0001', '0002'):
            write(run / 'requests' / number / 'REQUEST.json', read(run / 'REQUEST.json'))
        with self.assertRaisesRegex(ValueError, 'http_budget_exceeded'):
            generation.verify(self.home)
        self.assertEqual(self.fake.calls, [])

    def test_verify_detects_changed_saved_public_projection(self):
        self.prepare_home(); generation.generate(self.home)
        path = self.home / 'ordering' / 'PUBLIC_QUESTIONS.json'
        public = read(path); public['questions'][0]['question'] = 'Changed question.'
        path.write_text(json.dumps(public))
        with self.assertRaisesRegex(ValueError, 'frozen_file_changed'):
            generation.verify(self.home)
        self.assertEqual(len(self.fake.calls), 1)

    def test_real_writer_review_and_native_pool_gates_interoperate(self):
        import batch
        import ordering_review as review
        run = self.prepare_home(); folder = self.home / 'ordering'
        generated = generation.generate(self.home)
        self.assertEqual(generated['candidate_count'], 2)
        self.assertEqual(generation.verify(self.home), {'candidate_count': 2, 'http_attempts': 1})
        public_path = folder / 'PUBLIC_QUESTIONS.json'
        public = read(public_path)['questions']; source = read(run / 'SOURCE.json')
        # Build only from the complete source and stripped public projection.
        # These declarations describe the synthetic reviewer in this test.
        judgments = []
        for question in public:
            items = {}
            for label, text in question['items'].items():
                matches = [t for t in source['turns'] if t['text'] == text]
                self.assertEqual(len(matches), 1)
                turn = matches[0]
                items[label] = {'proposition': text,
                    **{key: True for key in review.ITEM_BOOLEANS},
                    'order_anchor': {'turn_ref': turn['turn_ref'], 'quote': text},
                    'support_refs': [turn['turn_ref']], 'alternative_matching_anchors': []}
            judgments.append({**deepcopy(question), 'items': items,
                'complete_source_turns_read': len(source['turns']),
                'coherent': True, 'unique_order': True, 'neutral_stem_ok': True,
                'alternative_orders': []})
        write(folder / 'SOURCE_JUDGMENTS.json', {
            'review_schema_version': review.SCHEMA_VERSION, 'reviewer': 'synthetic-integration-reviewer',
            'created_at_utc': now(), **review.DECLARATIONS,
            'public_questions': bind(public_path),
            'sources': [{'root_id': 'root', **bind(run / 'SOURCE.json'),
                         'complete_turn_count': len(source['turns'])}], 'questions': judgments})
        source_context = review.validate_sources(self.home)
        self.assertEqual(len(source_context['checks']), 2)
        self.assertTrue(all(not check['reasons'] for check in source_context['checks']))
        private_before = bind(folder / 'POOL.json')
        content = review.finish_review(self.home)
        self.assertEqual((content['accepted_count'], content['rejected_count']), (2, 0))
        self.assertEqual(review.verify_review(self.home), content)
        self.assertEqual(bind(folder / 'POOL.json'), private_before)
        self.assertEqual(generation.verify(self.home), {'candidate_count': 2, 'http_attempts': 1})
        protocol = load_multi('evaluate').native()
        child_pool = folder / 'EVALUATION_POOL.json'; child_before = bind(child_pool)
        # batch.freeze adds the formal mixed-pool envelope around the exact
        # child review rows. It supplies generator provenance, not a judgment.
        pool_path = self.home / 'EVALUATION_POOL.json'
        write(pool_path, {'questions': read(child_pool)['questions'],
            'provenance': {'requested_model': MODEL, 'mixed_batch_version': batch.VERSION},
            'training_allowed': False, 'formal_training_admission': False})
        before = bind(pool_path)
        pool, accepted, sources = protocol.validate_pool(pool_path, require_ordering_review=False)
        self.assertEqual(len(accepted), 2); self.assertEqual(set(sources), {'root'})
        self.assertEqual(pool['provenance']['requested_model'], MODEL)
        for question in accepted:
            self.assertEqual(question['gold'], CHRONOLOGICAL)
            self.assertEqual(question['independent_source_review']['status'], 'PASS')
            self.assertEqual(set(question['order_anchor_spans']), set(generation.LABELS))
            self.assertNotIn('qwen_review', question)
        self.assertEqual(bind(pool_path), before)
        self.assertEqual(bind(child_pool), child_before)
        self.assertEqual(len(self.fake.calls), 1)


if __name__ == '__main__':
    unittest.main()
