"""Offline full-flow regressions for five fixed roots; no network or live sources."""
from contextlib import ExitStack
from copy import deepcopy
import inspect
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import candidate_runner as c
from candidate_request_audit import audit_requests
import source_inventory as inventory


class OfflineCaller:
    def __init__(self, run):
        self.run = run
        self.request_counter = self.http_counter = 0
        self.baseline = c.r.base.read(run / 'FORMAL_BASELINE.json')['turns']
        self.root = c.r.base.read(run / 'MANIFEST.json')['root_id']

    def call(self, label, payload, proof):
        self.request_counter += 1
        self.http_counter += 1
        directory = self.run / 'requests' / f'{self.request_counter:03d}-{label}'
        c.r.base.write(directory / 'REQUEST.json', payload)
        c.r.base.write(directory / 'PROOF.json', {**proof, 'payload_sha256': c.r.base.digest(payload)})
        c.r.base.write(directory / 'RESERVATION.json', {'request_number': self.request_counter})
        c.r.base.write(directory / 'HTTP_ATTEMPT.json', {'mock_only': True, 'http_attempt_number': self.http_counter})
        context = json.JSONDecoder().raw_decode(payload['messages'][0]['content'].split('\nCANDIDATE_INPUT_JSON\n', 1)[1])[0]
        assert context['root_id'] == self.root
        assert context['cumulative_history'][:len(self.baseline)] == self.baseline
        slot = int(proof['scene_id'][1:]) - 1 + (3 if proof['route'] == c.r.ROUTES[1] else 0)
        assert len(context['cumulative_history']) == len(self.baseline) + 2 * slot
        if proof['role'] in ('planner', 'realizer'):
            assert '[/PREVIOUS_TRAINING_ROUND_EXPANSION_FEEDBACK]' in payload['messages'][0]['content']
        identity = {key: context[key] for key in c.IDENTITY_KEYS}
        if proof['role'] == 'planner':
            result = {'invalid_optional_plan_fixture': True}
        elif proof['role'] == 'realizer':
            result = {**identity, 'status': 'scene', 'turns': [
                {'speaker': speaker, 'text': f'MOCK ONLY {self.root} {context["route"]} {context["scene_id"]} utterance {i}: unique offline fixture scene.'}
                for i, speaker in enumerate(context['allowed_speakers'][:2])],
                'dataset_card_id': context['dataset_card']['card_id'] if context['dataset_card'] else None}
        else:
            result = {**identity, 'verdict': 'uncertain', 'summary': 'MOCK ONLY diagnostic with missing checks.'}
        raw = json.dumps(result)
        c.r.base.write(directory / 'RESPONSE.json', {'text': raw, 'metadata': {'mock_only': True}})
        c.r.base.write(directory / 'HTTP_RESPONSE_01.json', {'text': raw, 'metadata': {'mock_only': True}})
        return raw, directory


def fixture_cards(caller, run, allocation):
    root = c.r.base.read(run / 'MANIFEST.json')['root_id']
    # A3 must finish before any B extraction; no fake generated source enters production.
    assert caller.request_counter == 9
    cards = []
    for i, item in enumerate(allocation['assignments'][root], 1):
        card = {**item, 'card_id': f'MOCK_ONLY_CARD_{i}', 'card_sha256': '0' * 64,
                'lineage': {key: item[key] for key in ('dataset', 'split', 'item_id', 'source_item_sha256')},
                'guidance_level': 'seed', 'actors': [{'actor_id': 'a1', 'role': 'mock actor'}],
                'steps': [{'step_id': 's1', 'actor_id': 'a1', 'action': 'MOCK ONLY', 'state': 'proposed',
                           'evidence': [{'unit_id': 'MOCK:0', 'quote': 'MOCK ONLY source quotation'}]}],
                'dependencies': [], 'confusion_axes': [], 'open_ends': [], 'downgrade_reason': 'MOCK ONLY'}
        inventory.verify_cards_for_root([card], allocation, root)
        c.r.base.write(run / 'extractions' / f'{i:02d}.json', {'source_id': item['source_id'],
            'card': card, 'attempts': [], 'source_remains_consumed': True})
        cards.append(card)
    return cards


class FiveRootFlowTests(unittest.TestCase):
    def test_all_five_roots_prepare_A3_then_B3_and_replay(self):
        original_here = c.HERE
        original_collect = inventory.collect_prior
        with tempfile.TemporaryDirectory(prefix='V51_MOCK_ONLY_') as tmp, ExitStack() as stack:
            delivery = Path(tmp) / 'h4_sequential_ab_v51'
            delivery.mkdir()
            for source in original_here.iterdir():
                if source.is_file() and source.suffix in ('.py', '.md', '.json'):
                    shutil.copyfile(source, delivery / source.name)
            unique = delivery.parent / 'h4_r02_source_unique_v2'
            unique.mkdir()
            shutil.copyfile(original_here.parent / 'h4_r02_source_unique_v2/unique_events.py', unique / 'unique_events.py')
            stack.enter_context(patch.object(c, 'HERE', delivery))
            stack.enter_context(patch.object(c.r, 'LOCK_PATH', Path(tmp) / 'lock'))
            stack.enter_context(patch.object(inventory, 'collect_prior',
                side_effect=lambda: original_collect(current_runs=delivery / 'runs')))
            stack.enter_context(patch('candidate_transport.Caller', OfflineCaller))
            stack.enter_context(patch.object(c, 'extract_cards', side_effect=fixture_cards))
            stack.enter_context(patch.object(c, 'token_counter', return_value=lambda text: len(text.split())))
            stack.enter_context(patch.object(c.r, 'transport_inputs', return_value=({'offline_only': True}, [], [])))
            stack.enter_context(patch.object(c.r.base, 'transport', return_value=(lambda: {'offline_only': True},)))
            stack.enter_context(patch('socket.socket', side_effect=AssertionError('network_forbidden')))
            consumed_before = len(inventory.collect_prior()['accepted_uses'])
            used = set()
            for index, (root, expected_turns) in enumerate(zip(c.ROOTS, (106, 121, 110, 92, 126))):
                with self.subTest(root=root):
                    run = delivery / 'runs' / root
                    prepared = c.prepare(run, root=root)
                    self.assertEqual((prepared['logical_max'], prepared['http_max']), (30, 60))
                    self.assertFalse((run / 'INHERITED_A.json').exists())
                    self.assertFalse((run / 'SOURCE_RESERVATIONS.json').exists())
                    manifest, allocation = c.verify(run)
                    baseline = c.r.base.read(run / 'FORMAL_BASELINE.json')['turns']
                    self.assertEqual(len(baseline), expected_turns)
                    self.assertEqual(baseline, c.r.frozen_original_r01(root)['original']['turns'])
                    self.assertEqual(manifest['original_source_sha256'], c.r.SOURCE_SPECS[root][1])
                    new_ids = {item['source_id'] for item in allocation['assignments'][root]}
                    self.assertEqual(len(new_ids), 3)
                    self.assertFalse(used & new_ids)
                    used.update(new_ids)
                    report = c.live(run)
                    self.assertEqual(report['root_id'], root)
                    self.assertEqual(report['metrics']['retained']['scenes'], 6)
                    self.assertEqual(report['metrics']['admitted']['body_tokens'], 0)
                    self.assertEqual(c.audit(run)['status'], 'ARTIFACT_REPLAY_VERIFIED')
                    self.assertEqual(len(inventory.collect_prior()['accepted_uses']), consumed_before + (index + 1) * 3)
                    turns = c.r.base.read(run / 'CANDIDATE_HISTORY.json')['turns']
                    self.assertEqual(turns[:expected_turns], baseline)
                    self.assertEqual(len(turns), expected_turns + 12)
                    self.assertEqual(list(dict.fromkeys(t['scene_ref'] for t in turns[expected_turns:])),
                        [f'CANDIDATE:{letter}:S{i:02d}' for letter in ('A', 'B') for i in (1, 2, 3)])
                    # The request audit must use each manifest's root, never the module default.
                    request = next((run / 'requests').glob('*-planner'))
                    proof = c.r.base.read(request / 'PROOF.json')
                    proof['root_id'] = 'WRONG_ROOT'
                    (request / 'PROOF.json').write_text(json.dumps(proof))
                    with self.assertRaisesRegex(ValueError, 'request_identity_invalid'):
                        audit_requests(run, c.r.base.read(run / 'CANDIDATE_RECORDS.json'), baseline)

    def test_changed_fixed_source_hash_is_rejected(self):
        root = c.ROOTS[2]
        with patch.dict(c.r.SOURCE_SPECS, {root: ('24', '0' * 64)}):
            with self.assertRaisesRegex(ValueError, 'frozen_R01_source_changed'):
                c.r.frozen_original_r01(root)


class CrossVersionInventoryTests(unittest.TestCase):
    def test_v43_v51_v57_v59_reservations_stay_consumed_after_failure(self):
        with tempfile.TemporaryDirectory(prefix='V51_INVENTORY_MOCK_ONLY_') as tmp:
            tmp = Path(tmp)
            paths = {name: tmp / name for name in inspect.signature(inventory.collect_prior).parameters}
            def collect():
                return inventory.collect_prior(**paths)
            for version, directory in (('v43', paths['v43_runs']),
                                       ('v51', paths['v51_runs']), ('v57', paths['v57_runs']),
                                       ('v59', paths['current_runs'])):
                item = {'dataset': 'MOCK_ONLY', 'split': 'train', 'item_id': version,
                        'source_id': f'MOCK_ONLY:train:{version}', 'source_item_sha256': 'a' * 64}
                run = directory / 'failed'
                inventory.reserve(run, {'mock_root': [item]}, collect())
                c.r.base.write(run / 'LIVE_STARTED.json', {'mock_only': True})
            prior = collect()
            self.assertEqual({row['origin'] for row in prior['accepted_uses']},
                             {'v43_reservation', 'v51_reservation', 'v57_reservation', 'v59_reservation'})
            self.assertEqual(len(prior['accepted_uses']), 4)
            with self.assertRaisesRegex(ValueError, 'source_event_already_consumed'):
                inventory.reserve(paths['current_runs'] / 'replacement', {'other_root': [item]}, prior)


if __name__ == '__main__':
    unittest.main()
