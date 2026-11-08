import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import grounded_tail_probe as p


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.inputs = {'history': p.c.r.base.read(p.SOURCE / 'FORMAL_BASELINE.json')['turns'],
                       'source_card': p.c.r.base.read(p.SOURCE / 'extractions/01.json')['card']}
        p.c.r.base.write(self.folder / 'INPUT.json', self.inputs)
        shutil.copytree(p.SOURCE / 'feedback', self.folder / 'feedback')

    def tearDown(self):
        self.tmp.cleanup()

    def test_actual_history_source_and_feedback_are_preserved_before_final_task(self):
        request, proof = p.make_request(self.folder)
        prompt = request['messages'][0]['content']
        context = json.JSONDecoder().raw_decode(prompt.split('\nCANDIDATE_INPUT_JSON\n', 1)[1])[0]
        target = json.JSONDecoder().raw_decode(prompt.split('TARGET_AND_ORIGINAL_CARD_JSON\n', 1)[1])[0]
        self.assertEqual(context['cumulative_history'], self.inputs['history'])
        self.assertEqual(len(context['cumulative_history']), 132)
        self.assertEqual(target['source_card_with_original_evidence'], self.inputs['source_card'])
        self.assertEqual(target['scene_id'], 'S01')
        self.assertIsNone(context['advisory_plan'])
        self.assertLess(prompt.index('[/PREVIOUS_TRAINING_ROUND_EXPANSION_FEEDBACK]'), prompt.index('CURRENT TASK AFTER'))
        self.assertFalse(proof['training_pool_eligible'])
        self.assertEqual(proof['fresh_collection_body_tokens'], 0)

    def test_wrong_identity_does_not_hide_exact_history_copy(self):
        old = p.SOURCE / 'requests/006-cc-train-episode-104814-B-S01-realizer-1/RESPONSE.json'
        result = p.validate_response(p.c.r.base.read(old)['text'], self.inputs)
        self.assertIn('dialogue_identity_mismatch', result['mechanical_failures'])
        self.assertEqual(result['duplicate_check']['status'], 'REJECT')
        self.assertEqual(result['duplicate_check']['duplicate_turn_count'], 12)

    def test_new_text_is_only_mechanically_usable_not_semantic_pass(self):
        raw = p.c.r.base.read(next(p.PREVIOUS.glob('requests/*/RESPONSE.json')))['text']
        result = p.validate_response(raw, self.inputs)
        self.assertEqual(result['mechanical_failures'], [])
        self.assertEqual(result['duplicate_check']['status'], 'PASS')
        self.assertFalse(result['semantic_certified'])
        self.assertFalse(result['training_pool_eligible'])

    def test_prepare_no_http_or_reservation_and_detects_modified_frozen_payload(self):
        delivery = self.folder / 'v42'
        delivery.mkdir()
        for name in ('POLICY.md', 'BASELINE_FILES.json'):
            shutil.copyfile(p.HERE / name, delivery / name)
        run = delivery / 'diagnostics/test_only'
        with patch.object(p, 'HERE', delivery), patch.object(p, 'Caller', side_effect=AssertionError('network forbidden')):
            p.prepare(run)
            self.assertFalse((run / 'SOURCE_RESERVATIONS.json').exists())
            self.assertFalse((run / 'requests').exists())
            p.verify_inputs(run)
            request = run / 'FROZEN_REQUEST.json'
            request.write_text(request.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'frozen_file_changed'):
                p.verify_inputs(run)


if __name__ == '__main__':
    unittest.main()
