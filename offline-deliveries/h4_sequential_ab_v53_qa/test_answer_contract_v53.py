"""Offline behavioral tests for answer diversity without fabricated Gold.

The mixed fixtures contain explicit contradictions; the count check itself is
mechanical and is not a semantic correctness certification.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate
import prompt_rules
import test_generate_adapter as fixtures


class AnswerContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='v53-answer-contract-')
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'source.json'
        self.root = 'synthetic-answer-root'
        self.source = {'root_id': self.root, 'turns': [
            {'turn_ref': f'CANDIDATE:A:S01:T{i:03d}', 'speaker': 'Alex',
             'text': f'For decision {i}, I chose value {i}, not alternative {i}.'}
            for i in range(1, 7)]}
        self.path.write_text(json.dumps(self.source))
        network = patch('socket.socket', side_effect=AssertionError('Offline tests: no model calls'))
        network.start()
        self.addCleanup(network.stop)

    def multi(self, supported):
        return {'candidate_id': 'mixed', 'topic': 'Alex states six choices',
                'question': 'Which choices did Alex state? Select all that apply.',
                'options': [
                    {'turn_ref': turn['turn_ref'], 'speaker': turn['speaker'], 'quote': turn['text'],
                     'proposition': f'For decision {i}, Alex chose '
                         + (f'value {i}.' if label in supported else f'alternative {i}.'),
                     'relation': 'supports' if label in supported else 'contradicts',
                     'rationale': 'Alex explicitly states the choice and excludes the alternative.'}
                    for i, (label, turn) in enumerate(zip('ABCDEF', self.source['turns']), 1)]}

    def ordering(self, stem):
        return {'candidate_id': 'order', 'root_id': self.root, 'question_type': 'ordering',
                'topic': 'Six explicit choices', 'chronology_kind': 'conversation_order',
                'question': stem,
                'items': {f'I{i}': f'Alex states that value {i} was chosen for decision {i}.'
                          for i in range(1, 7)},
                'item_evidence': {f'I{i}': {'turn_ref': turn['turn_ref'],
                    'speaker': turn['speaker'], 'quote': turn['text'], 'status': 'statement',
                    'reason': 'One exact primary clause in transcript order.'}
                    for i, turn in enumerate(self.source['turns'], 1)},
                'gold': [f'I{i}' for i in range(1, 7)],
                'order_reason': 'The statements appear in source turn order.'}

    def validate(self, row, kind='multi_select', arm='current'):
        before = deepcopy(row)
        result = generate.validate_draft(json.dumps({'root_id': self.root,
            'questions': [row], 'abstentions': []}), [row['candidate_id']], self.root,
            kind, self.source, self.path, prompt_arm=arm)
        self.assertEqual(row, before)
        return result

    def test_two_three_four_explicitly_supported_options_preserve_every_raw_relation(self):
        for supported in ('BE', 'ACF', 'BCEF'):
            with self.subTest(supported=supported):
                raw = self.multi(supported)
                accepted, rejected, omitted, envelope = self.validate(raw)
                self.assertEqual((rejected, omitted, envelope), ([], [], None))
                candidate = accepted[0]
                self.assertEqual(candidate['gold'], list(supported))
                self.assertEqual(candidate['raw_model_candidate'], raw)
                self.assertEqual(candidate['options'],
                    {label: option['proposition'] for label, option in zip('ABCDEF', raw['options'])})
                self.assertEqual([candidate['option_evidence'][label]['relation'] for label in 'ABCDEF'],
                    [option['relation'] for option in raw['options']])
                self.assertFalse(candidate['semantic_review_performed'])

    def test_outside_count_rejects_raw_question_without_deleting_gold_or_changing_truth(self):
        for supported in ('', 'A', 'ABCDE', 'ABCDEF'):
            raw = self.multi(supported)
            raw['gold'] = list(supported)
            with self.subTest(supported=supported):
                accepted, rejected, _, error = self.validate(raw)
                self.assertEqual(accepted, [])
                self.assertIsNone(error)
                self.assertIn('current_multi_requires_2_to_4_raw_supports:' + str(len(supported)),
                              rejected[0]['errors'])
                self.assertEqual(rejected[0]['draft'], raw)

    def test_existing_gold_relation_mismatch_still_rejects_a_faked_balanced_answer(self):
        raw = self.multi('ABCDEF')
        raw['gold'] = ['A', 'C', 'F']
        accepted, rejected, _, _ = self.validate(raw)
        self.assertEqual(accepted, [])
        self.assertIn('raw_gold_relation_mismatch', rejected[0]['errors'])
        self.assertIn('current_multi_requires_2_to_4_raw_supports:6', rejected[0]['errors'])
        self.assertEqual(rejected[0]['draft'], raw)

    def test_unresolved_is_not_treated_as_a_false_option(self):
        raw = self.multi('ACF')
        raw['options'][1]['relation'] = 'unresolved'
        accepted, rejected, _, _ = self.validate(raw)
        self.assertEqual(accepted, [])
        self.assertIn('B:relation_invalid', rejected[0]['errors'])

    def test_old_arm_retains_legacy_six_true_contract_and_questions(self):
        raw = self.multi('ABCDEF')
        accepted, rejected, _, _ = self.validate(raw, arm='old')
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]['gold'], list('ABCDEF'))
        self.assertEqual(accepted[0]['raw_model_candidate'], raw)
        order = self.ordering('Order these as they APPEAR IN THE SUPPLIED DIALOGUE.')
        accepted, rejected, _, _ = self.validate(order, 'ordering', arm='old')
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]['question'], order['question'])

    def test_fixed_ordering_stem_allows_identity_raw_ids_without_answer_text(self):
        raw = self.ordering(prompt_rules.ORDERING_STEM)
        accepted, rejected, _, _ = self.validate(raw, 'ordering')
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]['gold'], raw['gold'])
        self.assertEqual(accepted[0]['items'], raw['items'])
        self.assertEqual(accepted[0]['question'], raw['question'])

    def test_fixed_single_and_multiple_scene_stems_match_existing_scope_parser(self):
        self.source['turns'][3]['turn_ref'] = 'CANDIDATE:B:S02:T001'
        self.path.write_text(json.dumps(self.source))
        cases = [
            ('CANDIDATE:A:S01 scene.', ['CANDIDATE:A:S01']),
            ('CANDIDATE:A:S01 and CANDIDATE:B:S02 scenes.',
             ['CANDIDATE:A:S01', 'CANDIDATE:B:S02'])]
        for suffix, scenes in cases:
            row = self.ordering(prompt_rules.ORDERING_STEM_PREFIX + ' within the ' + suffix)
            self.assertEqual(generate.current_question_shape_errors(row, 'ordering', self.source), [])
            scope = generate.ordering_gate.ordering_match_scope(row, self.source)
            self.assertEqual(scope['scene_refs'], scenes)
            self.assertEqual(scope['context_turn_refs'], [turn['turn_ref'] for turn in self.source['turns']])

    def test_answer_narratives_extra_sentences_and_unknown_scope_fail_before_review(self):
        prefix = prompt_rules.ORDERING_STEM_PREFIX
        examples = [
            'First Alex chooses one, then two. ' + prompt_rules.ORDERING_STEM,
            prompt_rules.ORDERING_STEM + ' The answer is I1 I2 I3 I4 I5 I6.',
            prefix + ' about choosing value 1 before value 2.',
            prefix + ' within the CANDIDATE:A:S99 scene.',
            prefix + ' within the CANDIDATE:A:S01 scene and other scenes.',
            prefix + ' within the CANDIDATE:A:S01 and CANDIDATE:A:S01 scenes.',
            prefix + ' within the CANDIDATE:A:S01 scenes.',
            'Order choices from decision 1 through decision 6 as they APPEAR IN THE SUPPLIED DIALOGUE.']
        for stem in examples:
            with self.subTest(stem=stem):
                raw = self.ordering(stem)
                accepted, rejected, _, _ = self.validate(raw, 'ordering')
                self.assertEqual(accepted, [])
                self.assertTrue(any('current_ordering_fixed_public_stem' in error
                                    for error in rejected[0]['errors']))
                self.assertEqual(rejected[0]['draft'], raw)


class ActualPromptContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CandidateGenerationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_current_prompt_removes_all_legacy_six_true_permissions_and_retains_evidence_rules(self):
        for arm in generate.PROMPT_ARMS:
            run = self.fixture.prepare(prompt_arm=arm, run_name=arm)
            context = generate.initial_context(run, self.fixture.root, 'multi_select')
            payload, _ = generate.make_payload(run, self.fixture.root, 'multi_select', context)
            text = payload['messages'][0]['content']
            if arm == 'old':
                self.assertIn('including all six when warranted. Do not force a count.', text)
                continue
            for forbidden in ('All six options may be supported',
                              'all six supports and abstention remain available',
                              'including all six when warranted. Do not force a count.',
                              'There is no required count'):
                self.assertNotIn(forbidden, text)
            self.assertIn('Exactly 2, 3 or 4 options must be supports', text)
            self.assertIn('same actor/event/time scope', text)
            self.assertIn('never an unmentioned detail or missing confirmation', text)
            self.assertIn('randomly flipping\nrelations', text)
            self.assertIn('Rebuild the\noption set or abstain', text)
            self.assertEqual(fixtures.input_json(payload, '\nINPUT_JSON\n')['source_turns'], self.fixture.turns)

    def test_current_ordering_prompt_has_exact_nonrevealing_template_without_topic_summary_rule(self):
        run = self.fixture.prepare()
        context = generate.initial_context(run, self.fixture.root, 'ordering')
        payload, _ = generate.make_payload(run, self.fixture.root, 'ordering', context)
        text = payload['messages'][0]['content']
        self.assertIn(prompt_rules.ORDERING_STEM, text)
        self.assertIn('Do not add a topic summary, event list', text)
        self.assertNotIn("The stem must explicitly contain 'APPEAR IN THE SUPPLIED DIALOGUE', with a narrow coherent topic.", text)
        self.assertIn('Display order can be arbitrary', text)


if __name__ == '__main__':
    unittest.main()
