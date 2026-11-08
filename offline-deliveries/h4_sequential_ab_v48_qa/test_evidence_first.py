"""Behavioral checks for the bounded evidence-first generation treatment."""
import importlib.util
import json
from pathlib import Path
import unittest

import generate as generation
import test_generate_adapter as fixtures


class EvidenceFirstTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CandidateGenerationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.runs = {arm: self.fixture.prepare(prompt_arm=arm,
            comparison_id='same-fixture', run_name=arm) for arm in generation.PROMPT_ARMS}

    def payload(self, arm, kind):
        run = self.runs[arm]
        context = generation.initial_context(run, self.root, kind)
        return generation.make_payload(run, self.root, kind, context)

    def test_actual_prompt_puts_evidence_before_claims_using_existing_schema(self):
        for kind in generation.TYPES:
            with self.subTest(kind=kind):
                payload, _ = self.payload('current', kind)
                prefix, _ = payload['messages'][0]['content'].split('\nINPUT_JSON\n', 1)
                _, actual_kind, tail = generation.prompt_layout.split_task_tail(
                    payload['messages'][0]['content'])
                self.assertEqual(actual_kind, kind)
                if kind == 'multi_select':
                    self.assertIn('{turn_ref,speaker,quote,proposition,relation,rationale}', prefix)
                    self.assertNotIn('{proposition,relation,turn_ref,speaker,quote,rationale}', prefix)
                    self.assertIn('turn_ref, speaker, quote, proposition, relation, rationale', tail)
                    self.assertIn('then write a proposition', tail)
                else:
                    self.assertIn('item_evidence,items,gold,order_reason.', prefix)
                    self.assertNotIn('items,gold,item_evidence,order_reason.', prefix)
                    self.assertIn('object for I1-I6 before the items object', tail)
                    self.assertIn("clauses' positions in the supplied dialogue", tail)
                self.assertIn('Use the actual source_turns as the sole evidence', tail)
                self.assertIn('Do not start from', tail)

    def test_treatment_preserves_full_source_inventory_feedback_and_parameters(self):
        for kind in generation.TYPES:
            with self.subTest(kind=kind):
                old, old_receipt = self.payload('old', kind)
                current, receipt = self.payload('current', kind)
                old_context = fixtures.input_json(old, '\nINPUT_JSON\n')
                current_context = fixtures.input_json(current, '\nINPUT_JSON\n')
                self.assertEqual(current_context, old_context)
                self.assertEqual(current_context['source_turns'], self.fixture.turns)
                self.assertTrue(all('-v48-same-fixture-' in qid for qid in current_context['candidate_ids']))
                self.assertNotIn('DO NOT LEAK', json.dumps(current_context))
                self.assertNotIn('gold', current_context['previous_candidates_dedup_only'][0])
                self.assertEqual({k: v for k, v in old.items() if k != 'messages'},
                                 {k: v for k, v in current.items() if k != 'messages'})
                suffixes = {arm: generation.legacy.feedback.build_feedback_suffix(
                    run / 'feedback', self.root, 'writer', generation.TARGET)
                    for arm, run in self.runs.items()}
                self.assertEqual(suffixes['old'], suffixes['current'])
                middle, _, _ = generation.prompt_layout.split_task_tail(current['messages'][0]['content'])
                self.assertTrue(middle.endswith(suffixes['current']))
                self.assertTrue(old['messages'][0]['content'].endswith(suffixes['old']))
                self.assertEqual(receipt['suffix_sha256'], old_receipt['suffix_sha256'])

    def test_actual_generation_payload_preserves_conditions_and_uses_evidenced_relations(self):
        for kind in generation.TYPES:
            with self.subTest(kind=kind):
                payload, _ = self.payload('current', kind)
                middle, _, tail = generation.prompt_layout.split_task_tail(payload['messages'][0]['content'])
                self.assertIn(generation.prompt_rules.CONDITION_PRESERVATION, tail)
                self.assertNotIn(generation.prompt_rules.CONDITION_PRESERVATION, middle)
                self.assertIn('if, unless, only when', tail)
                self.assertIn('planning to note or remember an action is not committing', tail)
                self.assertIn('conditional commitment into an unconditional one', tail)
                self.assertIn(generation.prompt_rules.VARIED_FACT_COMBINATIONS, tail)
                self.assertIn('Supported paraphrases and partial overlap remain allowed', tail)
                if kind == 'multi_select':
                    self.assertIn(generation.prompt_rules.EVIDENCED_RELATION_MIX, tail)
                    self.assertIn('never an unmentioned detail or missing confirmation', tail)
                    self.assertIn('all six supports and abstention remain available', tail)
                else:
                    self.assertNotIn(generation.prompt_rules.EVIDENCED_RELATION_MIX, tail)

    def test_independent_reviewer_guidance_is_byte_identical_to_v47(self):
        previous_path = Path(generation.__file__).parent.parent / 'h4_sequential_ab_v47_qa/prompt_rules.py'
        spec = importlib.util.spec_from_file_location('v47_review_rules_readonly', previous_path)
        previous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(previous)
        self.assertEqual(generation.prompt_rules.review_guidance(), previous.review_guidance())
        candidate = {'candidate_id': 'same-order-candidate', 'root_id': self.root,
            'question_type': 'ordering', 'chronology_kind': 'conversation_order',
            'question': 'Order as statements APPEAR IN THE SUPPLIED DIALOGUE.',
            'items': {f'I{i}': t['text'] for i, t in enumerate(self.fixture.turns[1:], 1)},
            'gold': ['WRITER_GOLD_MUST_NOT_LEAK'],
            'item_evidence': {'WRITER_EVIDENCE_MUST_NOT_LEAK': {}}}
        source = {'root_id': self.root, 'turns': self.fixture.turns}
        payloads = [generation.make_review_payload(run, self.root, candidate, source)[0]
                    for run in self.runs.values()]
        self.assertEqual(payloads[0], payloads[1])
        text = payloads[0]['messages'][0]['content']
        self.assertNotIn('WRITER_GOLD_MUST_NOT_LEAK', text)
        self.assertNotIn('WRITER_EVIDENCE_MUST_NOT_LEAK', text)
        self.assertNotIn(generation.prompt_rules.MULTI_EVIDENCE_FIRST, text)
        self.assertNotIn(generation.prompt_rules.ORDERING_EVIDENCE_FIRST, text)
        self.assertNotIn(generation.prompt_rules.CONDITION_PRESERVATION, text)
        self.assertNotIn(generation.prompt_rules.EVIDENCED_RELATION_MIX, text)
        self.assertNotIn(generation.prompt_rules.VARIED_FACT_COMBINATIONS, text)


if __name__ == '__main__':
    unittest.main()
