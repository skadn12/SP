"""Network-free byte parity against the preserved, pre-extraction QA generator.

The historical v1 TransferHooks class is exercised directly.  Mock model replies
are deterministic test data; none of these fixtures is used by a live prompt.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from ..h4_qa_multiroot_k3_t08_probe_v1 import generation
from ..h4_r02_feedback_transfer_p2_2_v1.test_generation_hooks import (
    FakeClient, FakeLedger, expanded_fixture, intents_fixture, batch_fixture,
)
from ..h4_r02_feedback_transfer_p2_2_v1.transfer import TransferHooks

RUN = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_miner_binding_v2_acceptance_v1_live_1')
SNAPSHOT = RUN / 'policy_snapshot/prechange/h4_qa_multiroot_k3_t08_probe_v1/generation.py'
ROOT = 'test-root'
GENERIC_SUFFIX = 'UNCHANGED V1 GENERIC WRITER GUIDANCE\nexact fixture bytes\n'
MINER_FILES = {
    'raw/prompts/test-root-intent-miner.txt',
    'raw/responses/test-root-intent-miner.json.txt',
    'intents.json',
    'reasoning_contracts.json',
}


def v1_envelope():
    raw = intents_fixture()
    raw['feedback_bindings'] = [
        {'intent_id': row['intent_id'], 'status': 'PROPOSED_TRANSFER',
         'source_feedback_ids': ['feedback-test'],
         'target_distinction': row['discriminative_variable'],
         'evidence_refs': ['D001', 'D002', 'D003'],
         'scope': 'Only the observed room reservation.',
         'change_from_source': 'The cancellation changes room availability.'}
        for row in raw['intents']
    ]
    raw['feedback_bindings'][2].update(
        status='NO_SUPPORTED_TRANSFER', source_feedback_ids=[], evidence_refs=[],
        target_distinction='', scope='', change_from_source='',
    )
    return raw


def v1_hooks(mode='P2_2_CONCRETE_TRANSFER', active=True):
    cards = [{
        'root_id': ROOT, 'feedback_id': 'feedback-test',
        'source_candidate_id': ROOT + ':old-question',
        'source_question_type': 'single_choice',
        'source_quality_status': 'ACTIVE_GUIDANCE_ELIGIBLE',
        'source_question': 'Which change made reserving the room possible?',
        'options': [{'label': 'A', 'text': 'A cancellation'}],
        'historical_gold': 'A', 'source_evidence_refs': ['D001', 'D002'],
        'solver_observations': {'M0': {'full_context': {'correct_count': 2, 'total': 3}}},
        'provenance': {'path': '/not/model/visible'},
    }] if active else []
    history = [{
        'root_id': ROOT, 'source_candidate_id': ROOT + ':history-only',
        'stem': 'Safe historical tracking', 'historical_gold': 'NEVER_VISIBLE',
        'correct_count': 'NEVER_VISIBLE',
    }]
    return TransferHooks(mode, cards, history, GENERIC_SUFFIX, 'UNCHANGED LEGACY MINER SUFFIX')


def artifacts(root):
    files = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        if relative.startswith('writer_briefs/requests/'):
            # This v1 audit field intentionally stores each temporary root's
            # absolute path.  Normalize only that field's exact string value;
            # every other artifact byte, including model prompts, is compared.
            record = json.loads(data)
            absolute = str(root / 'writer_briefs')
            if not record['brief_path'].startswith(absolute + '/'):
                raise AssertionError('Unexpected v1 brief_path outside its root')
            needle = json.dumps(record['brief_path']).encode()
            replacement = json.dumps('<ROOT>/writer_briefs/' + record['brief_path'][len(absolute) + 1:]).encode()
            if data.count(needle) != 1:
                raise AssertionError('Ambiguous path normalization')
            data = data.replace(needle, replacement)
        files[relative] = data
    return files


def run_fake(module, replies, root, *, hooks=None, feedback_context=None, miner_only=False):
    client, ledger = FakeClient(deepcopy(replies)), FakeLedger()
    entry = module.generate_root_intents if miner_only else module.generate_root_qa
    result = entry(expanded=expanded_fixture(), root_dir=root, root_id=ROOT,
        api_config=Path('/nonexistent'), ledger=ledger, client=client,
        hooks=hooks, feedback_context=feedback_context)
    return result, artifacts(root), client.calls, ledger.calls


class SharedMinerParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SNAPSHOT.is_file():
            raise RuntimeError('Pre-extraction snapshot required; never substitute current code')
        spec = importlib.util.spec_from_file_location(
            generation.__package__ + '._binding_v2_prechange_snapshot', SNAPSHOT)
        cls.original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.original)
        if hasattr(cls.original, 'generate_root_intents'):
            raise RuntimeError('Snapshot is not the required pre-extraction implementation')

    def setUp(self):
        self.stack = []
        for name in ('create_connection',):
            guard = patch.object(socket, name, side_effect=AssertionError('NETWORK_FORBIDDEN_IN_PARITY_TEST'))
            guard.start()
            self.addCleanup(guard.stop)
        guard = patch.object(socket.socket, 'connect', side_effect=AssertionError('NETWORK_FORBIDDEN_IN_PARITY_TEST'))
        guard.start()
        self.addCleanup(guard.stop)

    def assert_qa_parity(self, replies, *, hook_factory=None, feedback_context=None):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            old = run_fake(self.original, replies, base / 'old',
                hooks=hook_factory() if hook_factory else None, feedback_context=feedback_context)
            new = run_fake(generation, replies, base / 'new',
                hooks=hook_factory() if hook_factory else None, feedback_context=feedback_context)
            self.assertEqual(old[0], new[0], 'Complete QA result differs from snapshot')
            self.assertEqual(old[1], new[1], 'QA artifact bytes differ from snapshot')
            self.assertEqual(old[2], new[2], 'Client request payloads differ from snapshot')
            self.assertEqual(old[3], new[3], 'Ledger requests or metadata differ from snapshot')
            return new

    def test_no_hook_normal_and_legacy_context_exact_parity(self):
        replies = [intents_fixture(), batch_fixture(counts=(3, 1, 1))] + [batch_fixture()] * 5
        result, files, calls, ledger = self.assert_qa_parity(replies,
            feedback_context={'intent_suffix': 'ORIGINAL MINER SUFFIX',
                              'slot_suffixes': ['slot one', 'slot two', 'slot three']})
        self.assertEqual(result['valid_candidate_count'], 25)
        self.assertEqual(len(calls), 7)
        self.assertEqual([row['max_tokens'] for row in calls], [4096] + [8192] * 6)
        self.assertEqual(ledger[0]['request_id'], ROOT + '-intent-miner')
        self.assertIn(b'ORIGINAL MINER SUFFIX', files['raw/prompts/test-root-intent-miner.txt'])

    def test_no_hook_repair_replacement_exact_parity(self):
        replies = [intents_fixture(), batch_fixture('primary', (2, 1, 0)),
                   batch_fixture('repaired')] + [batch_fixture()] * 5
        result, files, calls, ledger = self.assert_qa_parity(replies)
        self.assertEqual(result['valid_candidate_count'], 24)
        self.assertEqual(len(calls), 8)
        candidates = json.loads(files['candidates_valid.json'])
        self.assertFalse(any(row['question'].startswith('primary') for row in candidates))
        self.assertEqual(sum(row['question'].startswith('repaired') for row in candidates), 4)
        self.assertEqual(ledger[2]['intent'], 'h4_k3_candidate_schema_repair')

    def test_no_hook_json_transport_and_writer_failure_exact_parity(self):
        for failure in ('{invalid JSON', RuntimeError('mock Miner transport error'), {'unexpected': []}):
            with self.subTest(failure=repr(failure)):
                result, files, calls, ledger = self.assert_qa_parity([failure])
                self.assertEqual(len(calls), 1)
                self.assertTrue(result['intent_errors'])
                self.assertEqual(result['valid_candidate_count'], 0)
                self.assertIn('raw/responses/test-root-intent-miner.json.txt', files)
        result, _, calls, _ = self.assert_qa_parity(
            [intents_fixture(), 'bad JSON', RuntimeError('mock repair transport error')] + [batch_fixture()] * 5)
        self.assertEqual(len(calls), 8)
        self.assertEqual(result['valid_candidate_count'], 20)

    def test_actual_v1_hooks_success_and_repair_exact_parity(self):
        for mode in ('P2_2_CONCRETE_TRANSFER', 'LEGACY_COMMON_BASE'):
            for repair in (False, True):
                with self.subTest(mode=mode, repair=repair):
                    replies = [v1_envelope() if mode == 'P2_2_CONCRETE_TRANSFER' else intents_fixture()]
                    if repair:
                        replies += [batch_fixture('primary', (2, 1, 0)), batch_fixture('repair')]
                        replies += [batch_fixture()] * 5
                    else:
                        replies += [batch_fixture()] * 6
                    result, files, calls, _ = self.assert_qa_parity(replies,
                        hook_factory=lambda: v1_hooks(mode))
                    self.assertEqual(result['valid_candidate_count'], 24)
                    self.assertEqual(len(calls), 8 if repair else 7)
                    self.assertIn('miner_prompt_adaptation.json', files)
                    if mode == 'P2_2_CONCRETE_TRANSFER':
                        binding = json.loads(files['binding_validation.json'])
                        self.assertEqual(binding['valid_proposed_binding_count'], 2)
                        self.assertEqual(binding['by_intent']['I003']['fallback_reason'], 'NO_SUPPORTED_TRANSFER')
                    if repair:
                        primary = json.loads(files['writer_briefs/requests/test-root-contract_guided-generator-I001.json'])
                        repaired = json.loads(files['writer_briefs/requests/test-root-contract_guided-repair-I001.json'])
                        self.assertEqual(primary['brief_sha256'], repaired['brief_sha256'])

    def test_actual_v1_hooks_malformed_envelope_and_failures_exact_parity(self):
        for failure in ('{invalid JSON', RuntimeError('mock transport error')):
            with self.subTest(failure=repr(failure)):
                result, files, calls, _ = self.assert_qa_parity([failure], hook_factory=v1_hooks)
                self.assertEqual(len(calls), 1)
                self.assertTrue(result['intent_errors'])
                self.assertIn('binding_validation.json', files)
        for mutation in ('unknown_envelope', 'missing_bindings', 'unknown_source'):
            with self.subTest(mutation=mutation):
                raw = v1_envelope()
                if mutation == 'unknown_envelope':
                    raw['unrecognized'] = True
                elif mutation == 'missing_bindings':
                    del raw['feedback_bindings']
                else:
                    raw['feedback_bindings'][0]['source_feedback_ids'] = ['nonexistent']
                result, files, calls, _ = self.assert_qa_parity(
                    [raw] + [batch_fixture()] * 6, hook_factory=v1_hooks)
                self.assertEqual(result['valid_contract_count'], 3)
                self.assertEqual(len(calls), 7)
                self.assertEqual(json.loads(files['raw/responses/test-root-intent-miner.json.txt']), raw)

    def test_actual_v1_hooks_filtered_arbitrary_ids_and_no_active_exact_parity(self):
        raw = v1_envelope()
        identifiers = ['custom_9', 'intent-Z', 'last_712']
        for intent, binding, identifier in zip(raw['intents'], raw['feedback_bindings'], identifiers):
            intent['intent_id'] = binding['intent_id'] = identifier
        raw['intents'][0]['reasoning_chain']['trigger'] = [1]
        for active in (False, True):
            with self.subTest(active=active):
                result, files, calls, _ = self.assert_qa_parity(
                    [raw] + [batch_fixture()] * 4,
                    hook_factory=lambda: v1_hooks(active=active))
                self.assertEqual(result['valid_contract_count'], 2)
                self.assertEqual(len(calls), 5)
                contracts = json.loads(files['reasoning_contracts.json'])
                self.assertEqual([row['intent_id'] for row in contracts], identifiers[1:])
                self.assertNotIn(b'NEVER_VISIBLE', files['raw/prompts/test-root-intent-miner.txt'])
                binding = json.loads(files['binding_validation.json'])
                self.assertNotIn('custom_9', binding['by_intent'])
                self.assertEqual(set(binding['by_intent']), set(identifiers[1:]))
                if not active:
                    self.assertEqual(binding['feedback_status'], 'NO_ACTIVE_FEEDBACK')

    def test_shared_miner_returns_normally_one_call_no_candidate_artifacts(self):
        for hook_factory in (None, v1_hooks):
            with self.subTest(actual_v1_hooks=hook_factory is not None):
                raw = v1_envelope() if hook_factory else intents_fixture()
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    old = run_fake(self.original, [raw] + [batch_fixture()] * 6, base / 'old',
                        hooks=hook_factory() if hook_factory else None)
                    new = run_fake(generation, [raw], base / 'miner', miner_only=True,
                        hooks=hook_factory() if hook_factory else None)
                result, files, calls, ledger = new
                expected_files = set(MINER_FILES)
                if hook_factory:
                    expected_files |= {'miner_prompt_adaptation.json', 'feedback_bindings.json', 'binding_validation.json'}
                self.assertEqual(set(files), expected_files)
                self.assertEqual(files, {name: old[1][name] for name in expected_files})
                self.assertEqual(result['intent_result'], json.loads(old[1]['intents.json'])['validation'])
                self.assertEqual(result['contracts'], json.loads(old[1]['reasoning_contracts.json']))
                self.assertEqual(len(result['contracts']), 3)
                self.assertEqual(calls, old[2][:1])
                self.assertEqual(ledger, old[3][:1])
                self.assertEqual(len(calls), 1)

    def test_shared_miner_keeps_valid_items_after_partial_invalid_result(self):
        raw = v1_envelope()
        identifiers = ['removed_99', 'still-valid_42', 'still-valid_307']
        for intent, binding, identifier in zip(raw['intents'], raw['feedback_bindings'], identifiers):
            intent['intent_id'] = binding['intent_id'] = identifier
        raw['intents'][0]['reasoning_chain']['trigger'] = []
        with tempfile.TemporaryDirectory() as directory:
            result, files, calls, _ = run_fake(generation, [raw], Path(directory),
                hooks=v1_hooks(), miner_only=True)
        self.assertFalse(result['intent_result']['valid'])
        self.assertEqual(len(result['intent_result']['intents']), 3)
        self.assertEqual([row['intent_id'] for row in result['contracts']], identifiers[1:])
        self.assertEqual(len(calls), 1)
        self.assertFalse(any('candidate' in name or 'writer' in name for name in files))
        validation = json.loads(files['binding_validation.json'])
        self.assertNotIn(identifiers[0], validation['by_intent'])
        self.assertEqual(validation['by_intent'][identifiers[1]]['binding']['intent_id'], identifiers[1])
        self.assertEqual(validation['by_intent'][identifiers[2]]['fallback_reason'], 'NO_SUPPORTED_TRANSFER')

    def test_shared_miner_parse_and_transport_failure_return_without_writer(self):
        for failure in ('{invalid JSON', RuntimeError('observed mock failure')):
            with self.subTest(failure=repr(failure)):
                with tempfile.TemporaryDirectory() as directory:
                    result, files, calls, _ = run_fake(generation, [failure], Path(directory), miner_only=True)
                self.assertFalse(result['intent_result']['valid'])
                self.assertEqual(result['contracts'], [])
                self.assertTrue(result['intent_result']['errors'])
                self.assertEqual(len(calls), 1)
                self.assertEqual(set(files), MINER_FILES)


if __name__ == '__main__':
    unittest.main()
