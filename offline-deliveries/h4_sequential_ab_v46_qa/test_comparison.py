"""Offline behavioral checks for equal comparison inputs and isolated arms."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import compare
import generate as generation
import test_generate_adapter as fixtures


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CandidateGenerationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.runs = {arm: self.fixture.prepare(prompt_arm=arm,
            comparison_id='same-pair', run_name=arm) for arm in generation.PROMPT_ARMS}

    def payload(self, arm, kind):
        run = self.runs[arm]
        context = generation.initial_context(run, self.root, kind)
        return generation.make_payload(run, self.root, kind, context)

    def proof(self, payload, receipt, kind='multi_select', role='question_generator'):
        return {'root_id': self.root, 'role': role, 'question_type': kind,
                'feedback_receipt': receipt, 'payload_sha256': generation.base.digest(payload)}

    def test_paired_initial_inputs_ids_parameters_feedback_and_only_template_differ(self):
        suffixes = {arm: generation.legacy.feedback.build_feedback_suffix(
            run / 'feedback', self.root, 'writer', generation.TARGET)
            for arm, run in self.runs.items()}
        for kind in generation.TYPES:
            with self.subTest(kind=kind):
                old, old_receipt = self.payload('old', kind)
                current, current_receipt = self.payload('current', kind)
                compare.same_inputs(old, current, suffixes['old'], suffixes['current'])
                data = compare.payload_data(old)
                self.assertEqual(data, compare.payload_data(current))
                self.assertEqual(data['source_turns'], self.fixture.turns)
                self.assertEqual(len(data['candidate_ids']), 2)
                self.assertTrue(all('-v46-same-pair-' in qid for qid in data['candidate_ids']))
                self.assertNotIn('DO NOT LEAK', json.dumps(data))
                self.assertNotIn('gold', data['previous_candidates_dedup_only'][0])
                old_text, current_text = (p['messages'][0]['content'] for p in (old, current))
                self.assertNotIn(generation.MERGED_INSTRUCTION, old_text)
                self.assertNotIn(generation.prompt_layout.START, old_text)
                self.assertTrue(old_text.endswith(suffixes['old']))
                self.assertIn(generation.MERGED_INSTRUCTION, current_text)
                middle, actual_kind, tail = generation.prompt_layout.split_task_tail(current_text)
                self.assertEqual(actual_kind, kind)
                self.assertTrue(middle.endswith(suffixes['current']))
                # Removing exactly the treatment blocks reconstructs the old request.
                self.assertEqual(middle.replace(generation.MERGED_INSTRUCTION, '', 1), old_text)
                self.assertTrue(current_text.endswith(tail))
                self.assertEqual(old['max_tokens'], 5120)
                self.assertEqual(old['temperature'], .25)
                self.assertFalse(old['enable_thinking'])
                self.assertEqual(old['model'], 'Qwen/Qwen3.6-35B-A3B')
                for arm, payload, receipt in [('old', old, old_receipt),
                                              ('current', current, current_receipt)]:
                    generation.verify_prompt_request(self.runs[arm], payload,
                        self.proof(payload, receipt, kind))

    def test_current_outputs_cannot_change_frozen_ordering_input_or_other_arm(self):
        before = {arm: self.payload(arm, 'ordering') for arm in generation.PROMPT_ARMS}
        accepted = {'candidate_id': 'NEW_CURRENT_BATCH_MC', 'question_type': 'multi_select',
                    'question': 'NEW_CURRENT_BATCH_MC', 'options': {'A': 'NEW_GENERATED_FACT'},
                    'gold': ['A'], 'solver_output': 'CURRENT_SOLVER_MUST_NOT_LEAK'}
        generation.write(self.runs['old'] / 'MECHANICAL_POOL.json', {'questions': [accepted]})
        generation.write(self.runs['old'] / 'SOLVER_POOL.json', {'questions': [accepted]})
        generation.write(self.runs['old'] / 'outcomes' / (self.root + '.multi_select.json'),
                         {'accepted': [accepted]})
        for arm in generation.PROMPT_ARMS:
            with self.subTest(arm=arm):
                after = self.payload(arm, 'ordering')
                self.assertEqual(after, before[arm])
                self.assertNotIn('NEW_CURRENT_BATCH_MC', json.dumps(after))
                self.assertNotIn('CURRENT_SOLVER_MUST_NOT_LEAK', json.dumps(after))

    def test_reviewer_is_identical_for_both_arms_and_hides_writer_gold(self):
        candidate = {'candidate_id': 'same-order-candidate', 'root_id': self.root,
            'question_type': 'ordering', 'chronology_kind': 'conversation_order',
            'question': 'Order as statements APPEAR IN THE SUPPLIED DIALOGUE.',
            'items': {f'I{i}': t['text'] for i, t in enumerate(self.fixture.turns[1:], 1)},
            'gold': ['DO_NOT_LEAK_WRITER_GOLD'],
            'item_evidence': {'DO_NOT_LEAK_WRITER_ANCHORS': {}}}
        source = {'root_id': self.root, 'turns': self.fixture.turns}
        results = {arm: generation.make_review_payload(run, self.root, candidate, source)
                   for arm, run in self.runs.items()}
        self.assertEqual(results['old'][0], results['current'][0])
        for arm, (payload, receipt) in results.items():
            text = payload['messages'][0]['content']
            self.assertTrue(text.endswith(generation.prompt_layout.task_tail('ordering_review')))
            self.assertNotIn('DO_NOT_LEAK_WRITER_GOLD', text)
            self.assertNotIn('DO_NOT_LEAK_WRITER_ANCHORS', text)
            generation.verify_prompt_request(self.runs[arm], payload,
                self.proof(payload, receipt, role='ordering_reviewer'))

    def test_wrong_arm_and_old_request_tampering_are_rejected(self):
        old, receipt = self.payload('old', 'multi_select')
        proof = self.proof(old, receipt)
        with self.assertRaises(ValueError):
            generation.verify_prompt_request(self.runs['current'], old, proof)
        current, current_receipt = self.payload('current', 'multi_select')
        with self.assertRaises(ValueError):
            generation.verify_prompt_request(self.runs['old'], current,
                self.proof(current, current_receipt))
        for marker in ['Original full history.', '[PREVIOUS_TRAINING_ROUND_FEEDBACK_V4]']:
            changed = deepcopy(old)
            changed['messages'][0]['content'] = changed['messages'][0]['content'].replace(marker, 'TAMPERED', 1)
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                # An updated outer request hash cannot override the frozen feedback receipt.
                generation.verify_prompt_request(self.runs['old'], changed,
                    self.proof(changed, receipt))

    def test_pair_checker_rejects_unequal_source_ids_parameters_and_feedback(self):
        old, _ = self.payload('old', 'multi_select')
        current, _ = self.payload('current', 'multi_select')
        suffix = generation.legacy.feedback.build_feedback_suffix(
            self.runs['old'] / 'feedback', self.root, 'writer', generation.TARGET)
        cases = []
        for marker in ['Original full history.', '-v46-same-pair-']:
            changed = deepcopy(current)
            changed['messages'][0]['content'] = changed['messages'][0]['content'].replace(marker, 'DIFFERENT', 1)
            cases.append(changed)
        changed = deepcopy(current)
        changed['temperature'] = .9
        cases.append(changed)
        for index, changed in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                compare.same_inputs(old, changed, suffix, suffix)
        with self.assertRaises(ValueError):
            compare.same_inputs(old, current, suffix, suffix + 'changed')


if __name__ == '__main__':
    unittest.main()
