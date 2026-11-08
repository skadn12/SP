"""Real offline preparation and three guards that transport fakes cannot prove."""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ordering_generation as generation
from common import HERE, bind, read


class RealOrderingPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards = ExitStack()
        for target in ('socket.socket.connect', 'socket.socket.connect_ex',
                       'socket.create_connection', 'socket.getaddrinfo', 'subprocess.Popen'):
            cls.guards.enter_context(patch(target, side_effect=AssertionError('External operation prohibited')))
        cls.addClassCleanup(cls.guards.close)
        cls.temporary = tempfile.TemporaryDirectory(prefix='v86_ordering_guard_tests_')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.home = Path(cls.temporary.name).resolve()
        cls.config = read(HERE / 'CONFIG.json')
        (cls.home / 'CONFIG.json').write_text(json.dumps(cls.config))
        generation.prepare(cls.home)
        cls.slot_run = cls.home / 'ordering' / cls.config['ordering_slots'][0]['slot_id']
        cls.preparation = cls.home / 'ordering/PREPARATION.COMPLETE.json'
        cls.originals = {p: p.read_bytes() for p in (cls.slot_run / 'MANIFEST.json',
                          cls.slot_run / 'CAPACITY.json', cls.preparation)}

    def tearDown(self):
        for path, raw in self.originals.items():
            path.write_bytes(raw)
        self.assertFalse(list(self.home.rglob('HTTP_ATTEMPT.json')))
        self.assertFalse(list(self.home.rglob('GENERATION.STARTED.json')))

    def rebind_preparation(self, field, path):
        preparation = read(self.preparation)
        preparation['slots'][0][field] = bind(path)
        self.preparation.write_text(json.dumps(preparation))

    def test_changed_model_identity_rejected_even_with_updated_binding(self):
        manifest = self.slot_run / 'MANIFEST.json'
        for field in ('solver_model', 'generator_model'):
            with self.subTest(field=field):
                value = json.loads(self.originals[manifest])
                value[field] = 'unexpected-model'
                manifest.write_text(json.dumps(value))
                self.rebind_preparation('manifest', manifest)
                with self.assertRaisesRegex(ValueError, 'ordering_manifest_contract_changed'):
                    generation.verify(self.home)

    def test_real_payload_has_scoped_stems_and_all_disallowed_sets_without_private_answers(self):
        stem = 'FIXED_SINGLE_SCENE_ANCHOR_PLAN:'
        for slot in self.config['ordering_slots']:
            run = self.home / 'ordering' / slot['slot_id']
            payload, proof = generation.payload_for(run)
            self.assertEqual(payload, read(run / 'REQUEST.json'))
            tail = payload['messages'][0]['content'].split('[ORDERING_BATCH_V83]')[-1]
            self.assertIn(stem, tail)
            prior = read(run / 'PRIOR.json')
            listing = json.loads(tail.split('DISALLOWED PRIOR WHOLE SETS (public text, deduplication only):\n')[1].split('\nACTOR_REFERENCE_POLICY:\n')[0])
            self.assertEqual(listing['disallowed_prior_six_item_sets'], [sorted(q['items'].values()) for q in prior])
            context, _ = json.JSONDecoder().raw_decode(payload['messages'][0]['content'].split('INPUT_JSON\n', 1)[1])
            anchors = read(run / 'ANCHOR_PLAN.json')['questions']
            policy = read(run / 'ACTOR_REFERENCE_POLICY.json')
            self.assertEqual(context['fixed_single_scene_anchor_plan'], {'location': 'FIXED_SINGLE_SCENE_ANCHOR_PLAN in the task tail; complete immutable plan appears there once.'})
            self.assertEqual(context['actor_reference_policy'], {'location': 'ACTOR_REFERENCE_POLICY in the task tail; complete frozen policy appears there once.'})
            listed_anchors, _ = json.JSONDecoder().raw_decode(tail.split('FIXED_SINGLE_SCENE_ANCHOR_PLAN:\n', 1)[1])
            listed_policy, _ = json.JSONDecoder().raw_decode(tail.split('ACTOR_REFERENCE_POLICY:\n', 1)[1])
            self.assertEqual(listed_anchors, anchors)
            self.assertEqual(listed_policy, policy)
            for full in (anchors, policy):
                self.assertEqual(payload['messages'][0]['content'].count(json.dumps(full, ensure_ascii=False, separators=(',', ':'))), 1)
            self.assertNotIn('"gold"', json.dumps(read(run / 'ANCHOR_PLAN.json')['questions']))
            self.assertNotIn('"item_evidence"', json.dumps(listing))
            source = read(run / 'SOURCE.json')
            self.assertEqual(source['turns'], read(slot['history_path'])['turns'])
            self.assertEqual(context['source_turns'], source['turns'])
            self.assertEqual(context['source_metadata'], {key: value for key, value in source.items() if key != 'turns'})
            self.assertEqual(context['previous_public_questions_dedup_only']['question_index'],
                [{key: question[key] for key in ('candidate_id', 'root_id', 'question_type')} for question in prior])
            for question in prior:
                self.assertIn('source_path', question)
                self.assertIn('source_sha256', question)
            for turn in source['turns']:
                self.assertIn(json.dumps(turn['text'], ensure_ascii=False)[1:-1], payload['messages'][0]['content'])

    def test_label_and_final_output_contracts_are_in_every_executed_payload(self):
        saw_nonchronological_labels = False
        for slot in self.config['ordering_slots']:
            run = self.home / 'ordering' / slot['slot_id']
            content = read(run / 'REQUEST.json')['messages'][0]['content']
            for phrase in (
                'LABEL BINDING CONTRACT: I1..I6 are randomly assigned immutable identifiers',
                'Never move\nan event to another I-label, renumber labels into time order',
                'Then derive gold by sorting those SAME labels by the source appearance',
                'check all six label-to-quote pairs individually, then check gold',
                'OUTPUT ORDER CONTRACT: Write items first, then order_reason, then gold LAST.',
                'After all reasoning and any correction, emit gold exactly as that final sequence.',
                'Check the final gold array\nagainst the final sequence in order_reason and all six fixed anchors',
            ):
                self.assertIn(phrase, content)
            skeleton = content.split('Each slot is null or exactly ', 1)[1].split('OUTPUT ORDER CONTRACT:', 1)[0]
            self.assertLess(skeleton.index('\"items\"'), skeleton.index('\"order_reason\"'))
            self.assertLess(skeleton.index('\"order_reason\"'), skeleton.index('\"gold\"'))
            source = read(run / 'SOURCE.json')
            positions = {turn['turn_ref']: index for index, turn in enumerate(source['turns'])}
            for plan in read(run / 'ANCHOR_PLAN.json')['questions']:
                self.assertEqual(set(plan['anchors']), {'I' + str(i) for i in range(1, 7)})
                ordered = sorted(plan['anchors'], key=lambda label: positions[plan['anchors'][label]['turn_ref']])
                saw_nonchronological_labels |= ordered != ['I' + str(i) for i in range(1, 7)]
        self.assertTrue(saw_nonchronological_labels)

    def test_missing_or_forged_capacity_rejected(self):
        path = self.slot_run / 'CAPACITY.json'
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            generation.verify(self.home)
        capacity = json.loads(self.originals[path])
        capacity['required_context_tokens'] -= 1
        path.write_text(json.dumps(capacity))
        self.rebind_preparation('capacity', path)
        with self.assertRaisesRegex(ValueError, 'ordering_capacity_changed'):
            generation.verify(self.home)

    def test_over_capacity_blocks_preparation_before_any_http(self):
        with tempfile.TemporaryDirectory(prefix='v86_ordering_over_capacity_') as temporary:
            home = Path(temporary).resolve()
            config = deepcopy(self.config)
            config['ordering_output_max_tokens'] = 50000
            (home / 'CONFIG.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'ordering_capacity_blocked'):
                generation.prepare(home)
            capacity = read(home / 'ordering' / config['ordering_slots'][0]['slot_id'] / 'CAPACITY.json')
            self.assertNotEqual(capacity['status'], 'PASS')
            self.assertFalse((home / 'ordering/PREPARATION.COMPLETE.json').exists())
            self.assertFalse(list(home.rglob('HTTP_ATTEMPT.json')))
            self.assertFalse(list(home.rglob('GENERATION.STARTED.json')))


if __name__ == '__main__':
    unittest.main()
