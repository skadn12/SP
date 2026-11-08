"""Public six-statement inventory prevents prompt omission, not a new gate."""
from copy import deepcopy
import json
import unittest

import generate
import test_generate_adapter as fixtures


class UsedSetTests(unittest.TestCase):
    def test_compact_index_groups_exact_public_sets_without_gold_or_private_fields(self):
        prior = [{'candidate_id': 'first', 'question_type': 'multi_select',
                  'facts': ['Statement ' + x for x in 'ABCDEF'],
                  'dedup_action': 'avoid_exact_six_statement_set_only_paraphrases_allowed',
                  'gold': ['PRIVATE'], 'score': 'PRIVATE', 'evidence': 'PRIVATE'}]
        prior.append({**deepcopy(prior[0]), 'candidate_id': 'second',
                      'facts': ['  statement ' + x.lower() + '  ' for x in reversed('ABCDEF')]})
        prior.append({**deepcopy(prior[0]), 'candidate_id': 'warning', 'dedup_action': 'warning_only'})
        prior.append({**deepcopy(prior[0]), 'candidate_id': 'short', 'facts': ['Only one']})
        before = deepcopy(prior)
        compact = generate.used_statement_sets(prior)
        self.assertEqual(prior, before)
        self.assertEqual(compact, [{'question_type': 'multi_select', 'prior_indices': [0, 1]}])
        self.assertNotIn('PRIVATE', json.dumps(compact))

    def test_actual_request_binds_public_sets_and_keeps_them_out_of_reviewer(self):
        fixture = fixtures.CandidateGenerationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        prior_path = fixture.home / 'public-prior.json'
        raw = {'candidate_id': 'known-six-set', 'root_id': fixture.root,
               'question_type': 'multi_select', 'question': 'Earlier six claims?',
               'options': {x: 'PUBLIC old claim ' + x for x in 'ABCDEF'},
               'gold': ['PRIVATE_GOLD'], 'training_selection': {'score': 'PRIVATE_SCORE'}}
        generate.write(prior_path, {'questions': [raw]})
        run = fixture.prepare(prior_pools=[prior_path])
        for kind in generate.TYPES:
            context = generate.initial_context(run, fixture.root, kind)
            payload, receipt = generate.make_payload(run, fixture.root, kind, context)
            sent = fixtures.input_json(payload, '\nINPUT_JSON\n')
            self.assertEqual(sent['source_turns'], fixture.turns)
            self.assertEqual(sent['used_six_statement_sets'], [{
                'question_type': 'multi_select', 'prior_indices': [1]}])
            self.assertIn('Renaming a topic, question, label or candidate ID does not', payload['messages'][0]['content'])
            self.assertNotIn('PRIVATE_', json.dumps(payload))
            changed = deepcopy(sent)
            changed['used_six_statement_sets'][0]['prior_indices'][0] = 999
            self.assertNotEqual(changed, generate.focus_context(generate.read(run / 'MANIFEST.json'), changed))
            self.assertTrue(receipt['feedback_appended'])
        candidate = {'candidate_id': 'review-only', 'root_id': fixture.root, 'question_type': 'ordering',
                     'chronology_kind': 'conversation_order',
                     'question': 'Order statements as they APPEAR IN THE SUPPLIED DIALOGUE.',
                     'items': {f'I{i}': t['text'] for i, t in enumerate(fixture.turns[1:], 1)}}
        payload, _ = generate.make_review_payload(run, fixture.root, candidate,
                                                  {'root_id': fixture.root, 'turns': fixture.turns})
        self.assertNotIn('used_six_statement_sets', json.dumps(payload))
        self.assertNotIn('PUBLIC old claim', json.dumps(payload))


if __name__ == '__main__':
    unittest.main()
