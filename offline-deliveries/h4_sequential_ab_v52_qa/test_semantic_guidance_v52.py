"""Offline regressions against v51 examples; model verdicts are synthetic.

These tests validate request scope and unchanged structural admission. They do
not claim that new model generations have fixed semantic error rates.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate
import prompt_rules
import test_generate_adapter as fixtures

GATE = generate.ordering_gate
V51 = Path(generate.__file__).resolve().parent.parent / 'h4_sequential_ab_v51_qa/runs'
ROOT = 'cc-train-episode-135466'
RUN = V51 / 'root135466_focus20_01_c03_generation'
REVIEW = RUN / 'requests/004-135466-merged-review-02'
GENERATION = RUN / 'requests/002-135466-merged-order-generate-1/RESPONSE.json'
SOURCE = RUN / 'sources' / (ROOT + '.json')


def read(path):
    return json.loads(path.read_text())


class SemanticGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CandidateGenerationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.run = self.fixture.prepare()

    def test_actual_payload_contains_referent_and_temporal_checks_for_both_types(self):
        for kind in generate.TYPES:
            context = generate.initial_context(self.run, self.fixture.root, kind)
            payload, _ = generate.make_payload(self.run, self.fixture.root, kind, context)
            middle, _, tail = generate.prompt_layout.split_task_tail(payload['messages'][0]['content'])
            self.assertEqual(fixtures.input_json(payload, '\nINPUT_JSON\n')['source_turns'], self.fixture.turns)
            for rule in (prompt_rules.REFERENT_BINDING, prompt_rules.TEMPORAL_STATUS):
                self.assertIn(rule, tail)
                self.assertNotIn(rule, middle)
            self.assertIn('who is being', tail)
            self.assertIn('the other\nfriend', tail)
            self.assertIn('Will be waiting describes future waiting, not is waiting now', tail)
            self.assertNotIn('Morgan', tail)
            self.assertNotIn('Quinn', tail)
            self.assertNotIn('David', tail)
            self.assertEqual(prompt_rules.ORDERING_SCOPE in tail, kind == 'ordering')

    def test_real_gold_errors_remain_raw_examples_not_coded_answers_or_new_mc_gate(self):
        locations = [
            (RUN / 'requests/001-135466-merged-multi-generate-1/RESPONSE.json', SOURCE, 'D'),
            (V51 / 'root210046_focus20_01_c01_generation/requests/001-210046-merged-multi-generate-1/RESPONSE.json',
             V51 / 'root210046_focus20_01_c01_generation/sources/cc-train-episode-210046.json', 'E')]
        for response_path, source_path, label in locations:
            with self.subTest(label=label):
                before = response_path.read_bytes()
                raw = json.loads(read(response_path)['text'])
                q = raw['questions'][0]
                # The unchanged MC validator still derives Gold from the raw
                # relation annotations; this patch does not silently correct it.
                source = read(source_path)
                rows, _, _, error = generate.validate_draft(json.dumps({**raw, 'questions': [q]}),
                    [q['candidate_id']], source['root_id'], 'multi_select', source, source_path)
                self.assertIsNone(error)
                self.assertEqual(len(rows), 1)
                self.assertIn(label, rows[0]['gold'])
                self.assertEqual(response_path.read_bytes(), before)


class PublicStemScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='v52-scope-offline-')
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.network = patch('socket.socket', side_effect=AssertionError('Offline regression: no model network'))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.source = read(SOURCE)
        self.raw_candidate = json.loads(read(GENERATION)['text'])['questions'][1]
        self.review = deepcopy(read(REVIEW / 'REVIEW.json')['artifact']['reviews'][0])
        self.snapshot = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (SOURCE, GENERATION, REVIEW / 'REVIEW.json', REVIEW / 'RESPONSE.json')}
        self.addCleanup(self.check_frozen_examples)

    def check_frozen_examples(self):
        for p, digest in self.snapshot.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)

    def candidate(self, question=None, source=None):
        source = source or self.source
        row = deepcopy(self.raw_candidate)
        if question is not None:
            row['question'] = question
        source_path = self.home / 'source.json'
        source_path.write_text(json.dumps(source))
        errors, candidate = GATE.validate_generated_ordering(row, row['candidate_id'], ROOT, source, source_path)
        self.assertEqual(errors, [])
        self.assertEqual(candidate['gold'], self.raw_candidate['gold'])
        return candidate

    def replay_synthetic_verdict(self, candidate, verdict, *, source=None, prompt_mutation=None):
        source = source or self.source
        raw = json.dumps({'reviews': [verdict]})
        prompt = GATE.build_review_prompt(candidate, source)
        if prompt_mutation:
            prompt = prompt_mutation(prompt)
        prompt += ('\n[PREVIOUS_TRAINING_ROUND_FEEDBACK_V4]\n{}\n'
                   '[/PREVIOUS_TRAINING_ROUND_FEEDBACK_V4]' +
                   generate.prompt_layout.task_tail('ordering_review'))
        request_path, response_path = self.home / 'REQUEST.json', self.home / 'RESPONSE.json'
        request_path.write_text(json.dumps({'model': GATE.MODEL,
            'messages': [{'role': 'user', 'content': prompt}]}))
        metadata = {'returned_model': GATE.MODEL, 'requested_model': GATE.MODEL, 'finish_reason': 'stop'}
        response_path.write_text(json.dumps({'text': raw, 'metadata': metadata}))
        return GATE.review_response(raw, [candidate], source, response_path, request_path, metadata)

    def scoped_verdict(self):
        verdict = deepcopy(self.review)
        for item in verdict['items'].values():
            item['alternative_matching_anchors'] = []
        return verdict

    def test_real_A2_stem_yields_explicit_scope_without_hiding_A3_context(self):
        candidate = self.candidate()
        scope = GATE.ordering_match_scope(candidate, self.source)
        self.assertEqual(scope['kind'], 'explicit_scene_union')
        self.assertEqual(scope['scene_refs'], ['CANDIDATE:A:S02'])
        self.assertTrue(all(ref.startswith('CANDIDATE:A:S02:') for ref in scope['matching_turn_refs']))
        self.assertIn('CANDIDATE:A:S03:T003', scope['context_turn_refs'])
        candidate['gold'] = ['PRIVATE_GOLD']
        candidate['item_evidence'] = {'PRIVATE_EVIDENCE': {}}
        prompt = GATE.build_review_prompt(candidate, self.source)
        data = json.loads(prompt.split('\nDATA:\n', 1)[1])
        self.assertEqual(data['source_turns'], self.source['turns'])
        self.assertIn('Other scenes remain in the complete source for support_refs', prompt)
        self.assertNotIn('PRIVATE_', prompt)
        supplied = json.loads(prompt.split(GATE.SCOPE_START, 1)[1].split(GATE.SCOPE_END, 1)[0])
        self.assertEqual(supplied, scope)

    def test_real_A2_case_excludes_A3_alternatives_and_preserves_raw_verdict(self):
        result = self.replay_synthetic_verdict(self.candidate(), self.review)
        self.assertEqual(result['rejected'], [])
        self.assertEqual(result['accepted'][0]['gold'], self.raw_candidate['gold'])
        self.assertTrue(all(not v for v in result['accepted'][0]['alternative_matching_anchors'].values()))
        skipped = result['artifact']['checks'][0]['out_of_scope_alternative_anchor_spans']
        self.assertEqual(sum(len(v) for v in skipped.values()), 4)
        self.assertEqual(skipped['I2'][0]['turn_ref'], 'CANDIDATE:A:S03:T003')
        self.assertEqual(result['artifact']['reviews'][0], self.review)

    def test_new_correctly_scoped_synthetic_review_passes_without_changing_gold(self):
        verdict = self.scoped_verdict()
        verdict['items']['I1']['support_refs'].append('CANDIDATE:A:S03:T003')
        candidate = self.candidate()
        result = self.replay_synthetic_verdict(candidate, verdict)
        self.assertEqual(result['rejected'], [])
        self.assertEqual(result['accepted'][0]['gold'], candidate['gold'])
        self.assertEqual(result['accepted'][0]['items'], candidate['items'])
        self.assertIn('CANDIDATE:A:S03:T003', result['accepted'][0]['support_refs']['I1'])

    def test_full_dialogue_still_rejects_order_changing_cross_scene_alternative(self):
        candidate = self.candidate('Order statements as they APPEAR IN THE SUPPLIED DIALOGUE.')
        self.assertEqual(GATE.ordering_match_scope(candidate, self.source)['kind'], 'full_dialogue')
        result = self.replay_synthetic_verdict(candidate, self.review)
        self.assertFalse(result['accepted'])
        reasons = result['rejected'][0]['reasons']
        self.assertIn('alternative_anchor_changes_or_ties_order:I2:I3', reasons)
        self.assertFalse(any('outside_public_stem_scope' in reason for reason in reasons))

    def test_multiscene_union_retains_valid_A3_alternatives_and_rejects_crossing(self):
        candidate = self.candidate('Order statements as they APPEAR IN THE SUPPLIED DIALOGUE '
                                   'within the CANDIDATE:A:S02 and CANDIDATE:A:S03 scenes.')
        scope = GATE.ordering_match_scope(candidate, self.source)
        self.assertEqual(scope['scene_refs'], ['CANDIDATE:A:S02', 'CANDIDATE:A:S03'])
        result = self.replay_synthetic_verdict(candidate, self.review)
        self.assertIn('alternative_anchor_changes_or_ties_order:I2:I3', result['rejected'][0]['reasons'])
        self.assertFalse(any('outside_public_stem_scope' in s for s in result['rejected'][0]['reasons']))

    def test_same_scene_repeated_clause_is_still_rejected(self):
        source = deepcopy(self.source)
        quote = self.review['items']['I2']['order_anchor']['quote']
        turn = next(t for t in source['turns'] if t['turn_ref'] == 'CANDIDATE:A:S02:T014')
        turn['text'] += ' ' + quote
        verdict = self.scoped_verdict()
        verdict['items']['I2']['alternative_matching_anchors'] = [
            {'turn_ref': turn['turn_ref'], 'quote': quote, 'reason': 'Synthetic repetition.'}]
        result = self.replay_synthetic_verdict(self.candidate(source=source), verdict, source=source)
        self.assertIn('alternative_anchor_changes_or_ties_order:I2:I3', result['rejected'][0]['reasons'])

    def test_false_verdict_and_outside_primary_are_not_overridden(self):
        verdict = deepcopy(self.review)
        verdict['unique_order'] = False
        result = self.replay_synthetic_verdict(self.candidate(), verdict)
        self.assertIn('qwen_question_check_failed', result['rejected'][0]['reasons'])
        verdict = self.scoped_verdict()
        verdict['items']['I2']['order_anchor'] = deepcopy(self.review['items']['I2']['alternative_matching_anchors'][0])
        result = self.replay_synthetic_verdict(self.candidate(), verdict)
        self.assertIn('I2:primary_anchor_outside_public_stem_scope', result['rejected'][0]['reasons'])

    def test_bad_exact_quote_outside_scope_is_still_invalid(self):
        verdict = deepcopy(self.review)
        verdict['items']['I2']['alternative_matching_anchors'][0]['quote'] = 'NOT A SOURCE QUOTE'
        result = self.replay_synthetic_verdict(self.candidate(), verdict)
        self.assertIn('I2:alternative_0:quote_not_exact', result['rejected'][0]['reasons'])

    def test_scope_request_tampering_is_rejected(self):
        result = self.replay_synthetic_verdict(self.candidate(), self.scoped_verdict(),
            prompt_mutation=lambda p: p.replace('"scene_refs":["CANDIDATE:A:S02"]',
                                                  '"scene_refs":["CANDIDATE:A:S03"]'))
        self.assertIn('review_prompt_match_scope_changed', result['artifact']['errors'])

    def test_unrecognized_prose_private_fields_and_invalid_refs_cannot_narrow(self):
        for suffix in ['not within the CANDIDATE:A:S02 scene.',
                       'within the CANDIDATE:A:S99 scene.',
                       'within the CANDIDATE:A:S02 scene and other scenes.',
                       'about CANDIDATE:A:S02.', 'within the A2 scene.']:
            candidate = deepcopy(self.raw_candidate)
            candidate['question'] = 'Order statements as they APPEAR IN THE SUPPLIED DIALOGUE ' + suffix
            candidate['ordering_scope'] = {'scene_refs': ['CANDIDATE:A:S02']}
            with self.subTest(suffix=suffix):
                self.assertEqual(GATE.ordering_match_scope(candidate, self.source)['kind'], 'full_dialogue')


if __name__ == '__main__':
    unittest.main()
