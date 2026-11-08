"""Network-free end-to-end generator parity and hook integration regressions."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from ..backends import BackendResponse
from ..h4_qa_multiroot_k3_t08_probe_v1 import generation

RUN = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_feedback_transfer_p2_2_acceptance_v1_live_1')
SNAPSHOT = RUN / 'audit/original_code_snapshot/h4_qa_multiroot_k3_t08_probe_v1/generation.py'


def expanded_fixture():
    return {'root_id': 'test-root', 'turns': [
        {'turn_ref': 'D001', 'speaker': 'A', 'text': 'The room is unavailable.'},
        {'turn_ref': 'D002', 'speaker': 'B', 'text': 'A cancellation frees it.'},
        {'turn_ref': 'D003', 'speaker': 'A', 'text': 'We reserve the room.'},
    ]}


def intents_fixture():
    return {'intents': [{
        'intent_id': f'I{i:03}', 'intent_type': 'state_transition',
        'discriminative_variable': f'Room reservation relation {i}',
        'reasoning_chain': {'before_state': [1], 'trigger': [2], 'after_state': [3]},
        'candidate_question_types': ['single_choice', 'multi_select', 'ordering'],
    } for i in range(1, 4)]}


def batch_fixture(label='valid', counts=(2, 1, 1)):
    rows = []
    for qtype, n in zip(('single_choice', 'multi_select', 'ordering'), counts):
        for i in range(n):
            row = {'question_type': qtype, 'question': f'{label} {qtype} {i}',
                   'evidence_refs': ['D001', 'D002', 'D003'],
                   'evidence_groups': {'before_state': ['D001'], 'trigger': ['D002'], 'after_state': ['D003']},
                   'discriminative_variable': 'Room reservation',
                   'reasoning_groups_used': ['before_state', 'trigger', 'after_state'],
                   'hypothesized_reasoning_type': 'state_transition'}
            if qtype == 'ordering':
                row.update(items={f'I{j}': f'Event {j}' for j in range(1, 5)}, gold=['I1', 'I2', 'I3', 'I4'])
            else:
                row['options'] = {k: f'Option {k}' for k in ('ABCD' if qtype == 'single_choice' else 'ABCDE')}
                row.update({'correct_option': 'A'} if qtype == 'single_choice' else {'gold': ['A', 'B']})
            rows.append(row)
    return {'candidates': rows}


class FakeClient:
    def __init__(self, replies):
        self.config = SimpleNamespace(model='fake', base_url='https://invalid.test')
        self.replies = iter(replies)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        raw = next(self.replies)
        if isinstance(raw, BaseException):
            raise raw
        return BackendResponse(raw if isinstance(raw, str) else json.dumps(raw), 'fake', {'usage': None})


class FakeLedger:
    def __init__(self):
        self.calls = []

    def invoke(self, **kwargs):
        self.calls.append({k: v for k, v in kwargs.items() if k != 'call'})
        return kwargs['call']()


def run_fake(module, replies, root, **kwargs):
    client, ledger = FakeClient(replies), FakeLedger()
    result = module.generate_root_qa(expanded=expanded_fixture(), root_dir=root,
        root_id='test-root', api_config=Path('/nonexistent'), ledger=ledger, client=client, **kwargs)
    files = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    return result, files, client.calls, ledger.calls


class TrackingHooks:
    def __init__(self):
        self.compilations = []
        self.requests = []
        self.valid_ids = []

    def miner_prompt(self, spec, **ctx):
        return generation._append_feedback_context(spec, 'HOOK_MINER_SENTINEL')

    def project_miner_response(self, raw, **ctx):
        # The full raw envelope must already exist on disk before projection.
        saved = ctx['root_dir'] / 'raw/responses/test-root-intent-miner.json.txt'
        assert json.loads(saved.read_text()) == raw
        return {'intents': raw['intents']}

    def after_intents(self, raw, result, contracts, **ctx):
        self.valid_ids = [c['intent_id'] for c in contracts]

    def writer_suffix(self, contract, source, **ctx):
        key = (ctx['root_id'], contract['intent_id'], source)
        self.compilations.append(key)
        return 'IDENTITY_BRIEF:' + ':'.join(key)

    def writer_request(self, spec, contract, source, attempt, request_id, **ctx):
        self.requests.append({'intent_id': contract['intent_id'], 'source': source,
                              'attempt': attempt, 'request_id': request_id, 'prompt': spec['prompt']})


class GeneratorHookRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SNAPSHOT.is_file():
            raise RuntimeError('Original generator snapshot is required for byte parity')
        spec = importlib.util.spec_from_file_location(generation.__package__ + '._original_snapshot', SNAPSHOT)
        cls.original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.original)

    def assert_parity(self, replies, feedback_context=None):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            old = run_fake(self.original, replies, base / 'old', feedback_context=feedback_context)
            new = run_fake(generation, replies, base / 'new', feedback_context=feedback_context, hooks=None)
            self.assertEqual(old, new)
            return new

    def test_disabled_byte_parity_with_minima_extra_rows_and_legacy_suffix(self):
        # Free accepts the original minimum (four) even though its requested target is eight.
        replies = [intents_fixture(), batch_fixture(counts=(3, 1, 1))] + [batch_fixture()] * 5
        result, files, calls, ledger = self.assert_parity(replies,
            {'intent_suffix': 'LEGACY_MINER', 'slot_suffixes': ['SLOT1', 'SLOT2', 'SLOT3']})
        self.assertEqual(len(calls), 7)
        self.assertEqual(result['valid_candidate_count'], 25)
        self.assertEqual([x['max_tokens'] for x in calls], [4096] + [8192] * 6)
        self.assertEqual([x['request_id'] for x in ledger][1:3],
                         ['test-root-contract_guided-generator-I001', 'test-root-free_exploration-generator-I001'])
        self.assertIn(b'LEGACY_MINER', files['raw/prompts/test-root-intent-miner.txt'])

    def test_disabled_byte_parity_repair_replaces_whole_primary(self):
        replies = [intents_fixture(), batch_fixture('primary', (2, 1, 0)), batch_fixture('repair')] + [batch_fixture()] * 5
        result, files, calls, ledger = self.assert_parity(replies)
        candidates = json.loads(files['candidates_valid.json'])
        self.assertEqual(len(calls), 8)
        self.assertEqual(len(candidates), 24)
        self.assertTrue(all(not row['question'].startswith('primary') for row in candidates))
        self.assertEqual(sum(row['question'].startswith('repair') for row in candidates), 4)
        self.assertEqual(ledger[2]['intent'], 'h4_k3_candidate_schema_repair')

    def test_disabled_byte_parity_json_and_transport_failures(self):
        for miner_reply in ('{bad json', RuntimeError('fake transport failure'), {'unexpected': []}):
            with self.subTest(reply=repr(miner_reply)):
                result, files, calls, ledger = self.assert_parity([miner_reply])
                self.assertEqual(len(calls), 1)
                self.assertEqual(result['valid_candidate_count'], 0)
                self.assertTrue(result['intent_errors'])
                self.assertIn('raw/responses/test-root-intent-miner.json.txt', files)
        replies = [intents_fixture(), 'bad JSON', RuntimeError('repair transport failure')] + [batch_fixture()] * 5
        result, files, calls, ledger = self.assert_parity(replies)
        self.assertEqual(len(calls), 8)
        self.assertEqual(result['valid_candidate_count'], 20)

    def test_hooks_use_actual_ids_after_filtering_and_reuse_brief_on_repair(self):
        raw = intents_fixture()
        raw['intents'][0]['reasoning_chain']['trigger'] = [1]  # invalid overlap
        raw['feedback_bindings'] = [{'intent_id': 'I001'}, {'intent_id': 'I002'}, {'intent_id': 'I003'}]
        hooks = TrackingHooks()
        replies = [raw, batch_fixture('primary', (2, 1, 0)), batch_fixture('repair')] + [batch_fixture()] * 3
        with tempfile.TemporaryDirectory() as temp:
            result, files, calls, ledger = run_fake(generation, replies, Path(temp), hooks=hooks)
        self.assertEqual(hooks.valid_ids, ['I002', 'I003'])
        self.assertEqual(len(hooks.compilations), 4)
        self.assertEqual(set(Counter(hooks.compilations).values()), {1})
        self.assertEqual([r['intent_id'] for r in hooks.requests], ['I002', 'I002', 'I002', 'I003', 'I003'])
        suffix = 'IDENTITY_BRIEF:test-root:I002:contract_guided'
        self.assertTrue(hooks.requests[0]['prompt'].endswith(suffix))
        self.assertTrue(hooks.requests[1]['prompt'].endswith(suffix))
        self.assertEqual(hooks.requests[1]['attempt'], 'repair')
        self.assertEqual(result['valid_candidate_count'], 16)
        for row in json.loads(files['candidate_pool.json']):
            self.assertNotIn('feedback_used', row)
            self.assertNotIn('brief_sha256', row)
        self.assertEqual(json.loads(files['raw/responses/test-root-intent-miner.json.txt']), raw)


if __name__ == '__main__':
    unittest.main()
