"""Per-slot prompt allocation leaves evidence, review and selection intact."""
import json
from pathlib import Path
import unittest

import batch
import generate as generation
import slot_focus
import test_generate_adapter as fixtures


class SlotFocusTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CandidateGenerationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.comparison_id = 'same-focus-fixture'
        self.focus = {}
        for kind, code in (('multi_select', 'multi'), ('ordering', 'order')):
            for number in (1, 2):
                qid = f'{self.root}-v52-{self.comparison_id}-merged-{code}-b01-{number:02d}'
                self.focus[qid] = {'preferred_scene_refs': ['CANDIDATE:B:S01'],
                    'focus': f'SYNTHETIC neutral topic {kind} number {number}',
                    'requirement': 'soft_preference_not_admission_gate'}

    def test_actual_payload_adds_distinct_id_focus_without_changing_full_source_feedback_or_decoder(self):
        plain = self.fixture.prepare(comparison_id=self.comparison_id, run_name='plain')
        focused = self.fixture.prepare(comparison_id=self.comparison_id, run_name='focused', slot_focus_map=self.focus)
        for kind in generation.TYPES:
            with self.subTest(kind=kind):
                contexts = [generation.initial_context(run, self.root, kind) for run in (plain, focused)]
                payloads = [generation.make_payload(run, self.root, kind, context)
                            for run, context in zip((plain, focused), contexts)]
                original, original_receipt = payloads[0]
                actual, receipt = payloads[1]
                decoded = fixtures.input_json(actual, '\nINPUT_JSON\n')
                ids = contexts[1]['candidate_ids']
                self.assertEqual(decoded['slot_focus_by_candidate_id'], {qid: self.focus[qid] for qid in ids})
                self.assertEqual(len({item['focus'] for item in decoded['slot_focus_by_candidate_id'].values()}), 2)
                self.assertEqual(decoded['source_turns'], self.fixture.turns)
                self.assertEqual(decoded['previous_candidates_dedup_only'], contexts[0]['previous_candidates_dedup_only'])
                self.assertEqual(decoded['new_scene_evidence_required'], False)
                self.assertEqual(decoded['priority_new_scene_refs'], [])
                self.assertEqual({key: val for key, val in actual.items() if key != 'messages'},
                                 {key: val for key, val in original.items() if key != 'messages'})
                self.assertEqual(receipt['suffix_sha256'], original_receipt['suffix_sha256'])
                suffixes = [generation.legacy.feedback.build_feedback_suffix(run / 'feedback', self.root, 'writer', generation.TARGET)
                            for run in (plain, focused)]
                self.assertEqual(suffixes[0], suffixes[1])
                middle, _, _ = generation.prompt_layout.split_task_tail(actual['messages'][0]['content'])
                self.assertTrue(middle.endswith(suffixes[1]))
                self.assertNotIn('Favor questions grounded in the new CANDIDATE:B scenes', actual['messages'][0]['content'])
        candidate = {'candidate_id': 'review-fixture', 'root_id': self.root, 'question_type': 'ordering',
            'chronology_kind': 'conversation_order',
            'question': 'Order as statements APPEAR IN THE SUPPLIED DIALOGUE.',
            'items': {f'I{i}': t['text'] for i, t in enumerate(self.fixture.turns[1:], 1)},
            'gold': ['WRITER_GOLD'], 'slot_focus_by_candidate_id': self.focus}
        source = {'root_id': self.root, 'turns': self.fixture.turns}
        reviews = [generation.make_review_payload(run, self.root, candidate, source)[0] for run in (plain, focused)]
        self.assertEqual(reviews[0], reviews[1])
        self.assertNotIn('SYNTHETIC neutral topic', json.dumps(reviews))
        self.assertNotIn('WRITER_GOLD', json.dumps(reviews))

    def test_v49_three_selected_questions_are_in_frozen_prior_without_solver_or_gold_fields(self):
        previous = Path(generation.__file__).parent.parent / ('h4_sequential_ab_v49_qa/runs/'
            'root104814_bulk20_01/training_pool/TRAINING_POOL.jsonl')
        self.assertEqual(batch.DEFAULT_PRIORS, (), 'New roots must not inherit 104814-specific default pools')
        raw_rows = [json.loads(line) for line in previous.read_text().splitlines() if line.strip()]
        self.assertEqual(len(raw_rows), 3)
        original_bytes = previous.read_bytes()
        run = self.fixture.prepare(comparison_id=self.comparison_id, prior_pools=[previous])
        inventory = generation.read(run / 'prior' / (self.root + '.json'))
        for kind in generation.TYPES:
            context = generation.initial_context(run, self.root, kind)
            payload, _ = generation.make_payload(run, self.root, kind, context)
            actual = fixtures.input_json(payload, '\nINPUT_JSON\n')['previous_candidates_dedup_only']
            for original in raw_rows:
                shown = next(q for q in actual if q['facts'] == list((original.get('options') or original['items']).values()))
                saved = next(q for q in inventory if q['candidate_id'] == original['candidate_id'])
                self.assertEqual(saved['facts'], list((original.get('options') or original['items']).values()))
                self.assertNotIn('question', shown)
                self.assertNotIn('candidate_id', shown)
                self.assertEqual(saved['question'], original['question'])
                self.assertFalse(set(shown) & {'gold', 'original_gold', 'option_evidence', 'training_selection',
                    'answers', 'raw_output', 'scores', 'full_partial_mean', 'qwen_review'})
        self.assertEqual(previous.read_bytes(), original_bytes)

    def test_all_three_roots_keep_twenty_slots_and_cycle_only_available_scenes(self):
        complete = [{'turn_ref': scene + ':T001', 'scene_ref': scene, 'speaker': 'Alex',
                     'text': 'SYNTHETIC scene.'} for scene in slot_focus.REQUESTED_SCENES]
        self.assertEqual(len(complete), 6)
        for root in batch.ROOTS:
            with self.subTest(root=root):
                plan = slot_focus.build_plan(root, 'fixture', complete)
                self.assertTrue(plan['expansion_complete'])
                self.assertEqual(plan['missing_scene_refs'], [])
                self.assertEqual(plan['available_scene_refs'], list(slot_focus.REQUESTED_SCENES))
                ids = [qid for chunk in plan['chunks'] for qid in chunk['slot_focus_by_candidate_id']]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(len(ids), 20)
                self.assertTrue(all(qid.startswith(root + '-v52-') for qid in ids))
                refs = [focus['preferred_scene_refs'][0] for chunk in plan['chunks']
                        for focus in chunk['slot_focus_by_candidate_id'].values()]
                self.assertEqual(refs, [slot_focus.REQUESTED_SCENES[i % 6] for i in range(20)])
                self.assertEqual(sum(ref.startswith('CANDIDATE:A:') for ref in refs), 10)
                self.assertEqual(sum(ref.startswith('CANDIDATE:B:') for ref in refs), 10)

        partial = complete[:3]
        plan = slot_focus.build_plan(batch.ROOTS[0], 'partial', partial)
        self.assertFalse(plan['expansion_complete'])
        self.assertEqual(plan['available_scene_refs'], list(slot_focus.REQUESTED_SCENES[:3]))
        self.assertEqual(plan['missing_scene_refs'], list(slot_focus.REQUESTED_SCENES[3:]))
        refs = [focus['preferred_scene_refs'][0] for chunk in plan['chunks']
                for focus in chunk['slot_focus_by_candidate_id'].values()]
        self.assertEqual(refs, [slot_focus.REQUESTED_SCENES[i % 3] for i in range(20)])
        one = slot_focus.build_plan(batch.ROOTS[0], 'one', complete[:1])
        self.assertEqual(len(one['chunks']), 5)
        self.assertEqual({focus['preferred_scene_refs'][0] for chunk in one['chunks']
                          for focus in chunk['slot_focus_by_candidate_id'].values()},
                         {slot_focus.REQUESTED_SCENES[0]})
        with self.assertRaisesRegex(ValueError, 'slot_focus_no_retained_new_scene'):
            slot_focus.build_plan(batch.ROOTS[0], 'none', [])


if __name__ == '__main__':
    unittest.main()
