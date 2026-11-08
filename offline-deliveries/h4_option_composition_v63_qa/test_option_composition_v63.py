"""Offline behavior coverage for private soft composition guidance."""
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import generate as gen
import display_export
import evaluate_and_pool as pipeline
import qa_transport
import prompt_layout
import test_generate_adapter as fixtures


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CandidateGenerationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def prepare(self, counts=(2, 5), name='composition'):
        original = gen.prepare
        def wrapped(*args, **kwargs):
            return original(*args, **kwargs, preferred_support_counts=counts)
        with patch.object(gen, 'prepare', wrapped):
            return self.fixture.prepare(comparison_id='fixture', run_name=name)

    def payload(self, run, context=None):
        context = context or gen.initial_context(run, self.root, 'multi_select')
        payload, receipt = gen.make_payload(run, self.root, 'multi_select', context)
        proof = {'role': 'question_generator', 'question_type': 'multi_select', 'root_id': self.root,
                 'feedback_receipt': receipt, 'payload_sha256': gen.base.digest(payload)}
        return payload, proof

    def test_default_payload_matches_frozen_v61(self):
        run = self.prepare(None)
        source = Path(gen.__file__).parent.parent / 'h4_multiselect_probe_v61_qa/generate.py'
        spec = importlib.util.spec_from_file_location('frozen_v61_composition_comparison', source)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        context = gen.initial_context(run, self.root, 'multi_select')
        self.assertEqual(gen.make_payload(run, self.root, 'multi_select', context),
                         original.make_payload(run, self.root, 'multi_select', context))
        self.assertNotIn('preferred_support_count_by_candidate_id', context)

    def test_active_tail_replaces_old_bias_after_feedback_and_preserves_full_inputs(self):
        run, plain = self.prepare(), self.prepare(None, 'plain')
        payload, proof = self.payload(run)
        default, default_proof = self.payload(plain)
        text = payload['messages'][0]['content']
        decoded = fixtures.input_json(payload, '\nINPUT_JSON\n')
        before = fixtures.input_json(default, '\nINPUT_JSON\n')
        self.assertEqual(decoded.pop('preferred_support_count_by_candidate_id'),
                         gen.verify(run)['preferred_support_count_by_candidate_id'])
        self.assertEqual(decoded, before)
        self.assertEqual(decoded['source_turns'], self.fixture.turns)
        self.assertEqual(proof['feedback_receipt']['suffix_sha256'], default_proof['feedback_receipt']['suffix_sha256'])
        self.assertNotIn('preferentially seek 2, 3 or 4', text)
        self.assertNotIn('Prefer 2, 3 or 4', text)
        intermediate, _, tail = prompt_layout.split_task_tail(text, expected_kind='multi_select', composition=True)
        suffix = gen.legacy.feedback.build_feedback_suffix(run / 'feedback', self.root, 'writer', gen.TARGET)
        self.assertTrue(intermediate.endswith(suffix))
        self.assertNotIn('OPTION COMPOSITION —', intermediate)
        self.assertIn('OPTION COMPOSITION —', tail)
        self.assertIn('never calls for rejection, repair, abstention, replacement', tail)
        self.assertIn('true/false twin', tail)
        self.assertIn('unknown or unmentioned claims as false', tail)
        gen.verify_prompt_request(run, payload, proof)
        self.assertEqual({k:v for k,v in payload.items() if k != 'messages'},
                         {k:v for k,v in default.items() if k != 'messages'})

    def test_pending_repair_subset_and_preference_tamper_replay(self):
        run = self.prepare()
        context = gen.initial_context(run, self.root, 'multi_select')
        pending = context['candidate_ids'][1]
        repair = gen.focus_context(gen.verify(run), {**context, 'candidate_ids': [pending],
            'mechanical_repair': {'failures': [{'candidate_id': pending, 'errors': ['synthetic']}]}})
        self.assertEqual(repair['preferred_support_count_by_candidate_id'], {pending: 5})
        self.assertEqual(repair['source_turns'], context['source_turns'])
        payload, proof = self.payload(run, repair)
        gen.verify_prompt_request(run, payload, proof)
        corrupted = deepcopy(payload)
        # The actual context serializer is pretty-printed, so replace within parsed INPUT_JSON.
        text = corrupted['messages'][0]['content']
        prefix, body = text.split('\nINPUT_JSON\n', 1)
        decoded, end = json.JSONDecoder().raw_decode(body)
        decoded['preferred_support_count_by_candidate_id'][pending] = 2
        corrupted['messages'][0]['content'] = prefix + '\nINPUT_JSON\n' + json.dumps(decoded) + body[end:]
        with self.assertRaisesRegex(ValueError, 'actual_slot_focus_changed'):
            gen.verify_prompt_request(run, corrupted, proof)

    def test_manifest_and_frozen_preference_file_are_bound(self):
        run = self.prepare()
        manifest = gen.read(run / 'MANIFEST.json')
        qid = next(iter(manifest['preferred_support_count_by_candidate_id']))
        manifest['preferred_support_count_by_candidate_id'][qid] = 6
        (run / 'MANIFEST.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'generation_composition_preference_binding_changed'):
            gen.verify(run)
        manifest['preferred_support_count_by_candidate_id'][qid] = 2
        (run / 'MANIFEST.json').write_text(json.dumps(manifest))
        binding = gen.read(run / 'COMPOSITION_PREFERENCES.json')
        binding['preferred_support_count_by_candidate_id'][qid] = 6
        (run / 'COMPOSITION_PREFERENCES.json').write_text(json.dumps(binding))
        with self.assertRaises(ValueError):
            gen.verify(run)

    def test_invalid_private_input_fails_before_prepare_side_effects(self):
        for i, counts in enumerate(([1, 5], [2, 7], [True, 5], [2], '25')):
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, 'preferred_support_counts_require'):
                self.prepare(counts, 'bad' + str(i))
            self.assertFalse((self.fixture.home / ('bad' + str(i))).exists())

    def draft(self, qid, count):
        return {'candidate_id': qid, 'topic': 'Synthetic events',
                'question': 'Which synthetic event claims are supported? Select all that apply.',
                'options': [{'turn_ref': t['turn_ref'], 'speaker': t['speaker'], 'quote': t['text'],
                    'proposition': f'Synthetic {qid} distinct claim {i}.',
                    'relation': 'supports' if i < count else 'contradicts',
                    'rationale': 'Synthetic bound evidence.'} for i,t in enumerate(self.fixture.turns[1:])]}

    def test_all_actual_2_to_6_counts_pass_unchanged_validator(self):
        run = self.prepare()
        manifest = gen.verify(run)
        source_path = Path(manifest['source_rows'][0]['source_path'])
        source = gen.read(source_path)
        qid = gen.initial_context(run, self.root, 'multi_select')['candidate_ids'][0]
        for count in range(2, 7):
            raw = {'root_id': self.root, 'questions': [self.draft(qid, count)], 'abstentions': []}
            valid, rejected, abstained, error = gen.validate_draft(json.dumps(raw), [qid], self.root,
                'multi_select', source, source_path)
            self.assertEqual((rejected, abstained, error), ([], [], None))
            self.assertEqual(len(valid), 1)
            self.assertEqual(valid[0]['gold'], list('ABCDEF'[:count]))
            self.assertNotIn('preferred_support_count', json.dumps(valid))

    def exercise_mock_generation(self, repair=False):
        run = self.prepare([2, 5])
        recorded = []
        owner = self
        class Caller:
            def __init__(self, run): self.run, self.counter = run, 0
            def call(self, label, payload, proof):
                self.counter += 1
                context = fixtures.input_json(payload, '\nINPUT_JSON\n')
                recorded.append(deepcopy(context))
                directory = self.run / 'requests' / f'{self.counter:03d}-{label}'
                gen.write(directory / 'REQUEST.json', payload)
                gen.write(directory / 'PROOF.json', {**proof, 'payload_sha256': gen.base.digest(payload)})
                gen.write(directory / 'HTTP_ATTEMPT.json', {'synthetic_test_only': True,
                    'actual_payload_equals_frozen': True, 'payload_sha256': gen.base.digest(payload)})
                questions = [owner.draft(qid, 6 if qid.endswith('01') else 2) for qid in context['candidate_ids']]
                if repair and self.counter == 1:
                    questions[-1]['options'][0]['proposition'] = ''
                raw = json.dumps({'root_id': owner.root, 'questions': questions, 'abstentions': []})
                metadata = {'requested_model': gen.MODEL, 'returned_model': gen.MODEL,
                            'finish_reason': 'stop', 'http_status': 200, 'synthetic_test_only': True}
                gen.write(directory / 'RESPONSE.json', {'text': raw, 'metadata': metadata})
                return raw, directory, metadata
        with patch.object(qa_transport, 'Caller', Caller), \
             patch.object(qa_transport, 'shared_lock', return_value=nullcontext()), redirect_stdout(io.StringIO()):
            report = gen.generate(run)
        self.assertEqual(report['http_attempts'], 2 if repair else 1)
        self.assertEqual(report['ready_for_solver'], 2)
        self.assertEqual(report['roots'][0]['mechanical_rejected'], [])
        self.assertEqual([r['actual_support_count'] for r in report['composition_diagnostics']['slots']], [6,2])
        self.assertEqual([r['matches_preference'] for r in report['composition_diagnostics']['slots']], [False,False])
        _, questions, _, _ = pipeline.verify_generation(run)
        self.assertNotIn('preferred_support_count', json.dumps(questions))
        source = gen.read(Path(questions[0]['source_path']))
        for q in questions:
            self.assertNotIn('preferred_support_count', json.dumps(display_export.display_view(q)))
            variants, _ = pipeline.evaluator._variants_for(q)
            text = pipeline.evaluator.prompt(q, variants[0], source)
            self.assertNotIn('preferred_support_count', text)
            self.assertNotIn('OPTION COMPOSITION', text)
        if repair:
            self.assertEqual(recorded[1]['candidate_ids'], recorded[0]['candidate_ids'][1:])
            qid = recorded[1]['candidate_ids'][0]
            self.assertEqual(recorded[1]['preferred_support_count_by_candidate_id'], {qid: 5})
            self.assertEqual(recorded[1]['source_turns'], recorded[0]['source_turns'])

    def test_mismatches_retained_without_extra_request_or_public_solver_leak(self):
        self.exercise_mock_generation()

    def test_existing_mechanical_repair_preserves_pending_preference_only(self):
        self.exercise_mock_generation(repair=True)


if __name__ == '__main__': unittest.main()
