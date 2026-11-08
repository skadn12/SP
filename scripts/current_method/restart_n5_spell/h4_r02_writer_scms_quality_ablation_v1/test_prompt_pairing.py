"""Offline prompt identity and original Writer behavior for the new suffix only."""
from __future__ import annotations

import copy
import hashlib
import json
import socket
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from ..h4_qa_mining_hybrid_generation_v1 import contract_guided_generator as builder
from ..h4_qa_multiroot_k3_t08_probe_v1 import generation
from ..h4_r02_feedback_transfer_p2_2_v1.test_generation_hooks import (
    FakeClient, FakeLedger, batch_fixture, expanded_fixture,
)
from .constants import ARMS, DEFAULT_RUN, ROOTS, SOURCE_MINER, SOURCE_WRITER, WORK_ORDER
from .guidance import (
    GUIDANCE_FILENAME, GUIDANCE_SHA256, guidance_text, prompt_spec, quality_suffix,
    repair_spec,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def contract_fixture():
    return {
        'intent_id': 'intent_1', 'intent_type': 'state_transition',
        'discriminative_variable': 'MINER_TARGET_SENTINEL',
        'required_evidence_groups': {
            'before_state': ['D001'], 'trigger': ['D002'], 'after_state': ['D003'],
        },
    }


def run_batch(replies, root, arm, *, contract=None):
    client, ledger = FakeClient(replies), FakeLedger()
    result = generation.execute_writer_batch(
        expanded=expanded_fixture(), contract=contract or contract_fixture(),
        root_dir=root, root_id='test-root', client=client, ledger=ledger,
        source='contract_guided', intent_guidance=True,
        slot_suffix=quality_suffix(DEFAULT_RUN, arm), hooks=None,
    )
    return result, client.calls, ledger.calls


class PromptPairingTests(unittest.TestCase):
    def setUp(self):
        for target, attr in ((socket.socket, 'connect'), (urllib.request, 'urlopen')):
            guard = patch.object(target, attr, side_effect=AssertionError('network prohibited'))
            guard.start()
            self.addCleanup(guard.stop)

    def test_literal_guidance_matches_work_order_and_rejects_any_byte_drift(self):
        document = WORK_ORDER.read_text(encoding='utf-8')
        start = document.index('BEGIN_SCMS_QUALITY_GUIDANCE_V1')
        end = document.index('END_SCMS_QUALITY_GUIDANCE_V1', start) + len('END_SCMS_QUALITY_GUIDANCE_V1')
        expected = document[start:end]
        self.assertEqual(guidance_text(DEFAULT_RUN), expected)
        self.assertEqual(hashlib.sha256(expected.encode()).hexdigest(), GUIDANCE_SHA256)
        self.assertNotIn('\r', expected)
        self.assertFalse(expected.endswith('\n'))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / GUIDANCE_FILENAME
            target.parent.mkdir(parents=True)
            target.write_text(expected + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SCMS_GUIDANCE_BYTE_IDENTITY_DRIFT'):
                guidance_text(root)
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_SCMS_ARM'):
            quality_suffix(DEFAULT_RUN, 'WITHOUT_INTENT')

    def test_all_original_inputs_and_six_slots_are_preserved(self):
        frozen = read_json(SOURCE_WRITER / 'audit/frozen_manifest.json')
        by_path = {row['path']: row for row in frozen['inputs']}
        slots = []
        for root in ROOTS:
            old = SOURCE_WRITER / 'inputs/shared' / root
            new = DEFAULT_RUN / 'inputs/shared' / root
            miner_shared = SOURCE_MINER / 'inputs/shared' / root
            miner_output = SOURCE_MINER / 'arms/V3_OFF/qa/shards/shard-00/roots' / root
            names = [
                'expanded_dialogue_clean.json', 'contracts.json', 'reasoning_contracts.json',
                'history_only.json', 'provenance.json', 'source_intent_diagnostics.json',
                'original_v3/intents.json', 'original_v3/reasoning_contracts.json',
                'original_v3/derived_intent_evidence.json', 'original_v3/response_schema_audit.json',
                f'original_v3/raw/responses/{root}-intent-miner.json.txt',
            ]
            for name in names:
                with self.subTest(root=root, file=name):
                    data = (old / name).read_bytes()
                    self.assertEqual((new / name).read_bytes(), data)
                    identity = by_path[str(old / name)]
                    self.assertEqual(hashlib.sha256(data).hexdigest(), identity['sha256'])
                    self.assertEqual(len(data), identity['size'])
                    if name.startswith('original_v3/'):
                        self.assertEqual(data, (miner_output / name.removeprefix('original_v3/')).read_bytes())
                    elif name in ('contracts.json', 'reasoning_contracts.json'):
                        self.assertEqual(data, (miner_output / 'reasoning_contracts.json').read_bytes())
                    elif name in ('expanded_dialogue_clean.json', 'history_only.json', 'provenance.json'):
                        self.assertEqual(data, (miner_shared / name).read_bytes())
            contracts = read_json(new / 'contracts.json')
            intents = read_json(new / 'original_v3/intents.json')['validation']['intents']
            self.assertEqual([item['intent_id'] for item in intents], ['intent_1', 'intent_2', 'intent_3'])
            self.assertEqual([item['intent_id'] for item in contracts], ['intent_1', 'intent_2', 'intent_3'])
            for intent, contract in zip(intents, contracts):
                self.assertEqual(contract['intent_type'], intent['intent_type'])
                self.assertEqual(contract['discriminative_variable'], intent['discriminative_variable'])
                self.assertEqual(contract['required_evidence_groups'], intent['reasoning_chain'])
                slots.append((root, intent['intent_id']))
        self.assertEqual(len(slots), 6)
        self.assertEqual(len(set(slots)), 6)

    def test_six_base_prompts_equal_frozen_and_actual_previous_requests(self):
        checked = []
        for root in ROOTS:
            shared = DEFAULT_RUN / 'inputs/shared' / root
            expanded = read_json(shared / 'expanded_dialogue_clean.json')
            for contract in read_json(shared / 'contracts.json'):
                slot = contract['intent_id']
                spec = prompt_spec(expanded, contract, ARMS[0], DEFAULT_RUN)
                self.assertEqual(spec, builder.build_prompt(expanded, contract, intent_guidance=True))
                frozen = SOURCE_WRITER / 'policy_snapshot/primary_requests/WITH_CURRENT_INTENT' / root / slot / 'prompt.txt'
                actual = SOURCE_WRITER / 'arms/WITH_CURRENT_INTENT/qa/shards/shard-00/roots' / root / 'raw/prompts' / f'{root}-contract_guided-generator-{slot}.txt'
                prepared = DEFAULT_RUN / 'policy_snapshot/primary_requests' / ARMS[0] / root / slot / 'prompt.txt'
                self.assertEqual(spec['prompt'].encode(), frozen.read_bytes())
                self.assertEqual(spec['prompt'].encode(), actual.read_bytes())
                self.assertEqual(spec['prompt'].encode(), prepared.read_bytes())
                checked.append((root, slot))
        self.assertEqual(len(checked), 6)

    def test_quality_only_appends_frozen_block_with_unchanged_schema_and_ordering(self):
        suffix = '\n' + guidance_text(DEFAULT_RUN)
        for root in ROOTS:
            shared = DEFAULT_RUN / 'inputs/shared' / root
            expanded = read_json(shared / 'expanded_dialogue_clean.json')
            for contract in read_json(shared / 'contracts.json'):
                base = prompt_spec(expanded, contract, ARMS[0], DEFAULT_RUN)
                quality = prompt_spec(expanded, contract, ARMS[1], DEFAULT_RUN)
                self.assertEqual(quality['prompt'], base['prompt'] + suffix)
                self.assertEqual(quality['prompt'].removesuffix(suffix), base['prompt'])
                self.assertEqual(quality['expected_counts'], base['expected_counts'])
                self.assertEqual(base['expected_counts'], {'single_choice': 2, 'multi_select': 1, 'ordering': 1})
                self.assertEqual(quality['prompt_version'], base['prompt_version'])
                self.assertEqual(quality['prompt'].count('BEGIN_SCMS_QUALITY_GUIDANCE_V1'), 1)
                self.assertIn('Single options are A-D; multi-select options A-E with list gold; ordering items I1-I4 with a permutation gold.', base['prompt'])
                self.assertEqual(quality['prompt'].split('OUTPUT SHAPE:\n')[1].split('\nDIALOGUE:')[0],
                                 base['prompt'].split('OUTPUT SHAPE:\n')[1].split('\nDIALOGUE:')[0])
                prepared = DEFAULT_RUN / 'policy_snapshot/primary_requests' / ARMS[1] / root / contract['intent_id'] / 'prompt.txt'
                self.assertEqual(quality['prompt'].encode(), prepared.read_bytes())
                self.assertEqual(quality['prompt_sha256'], hashlib.sha256(quality['prompt'].encode()).hexdigest())

    def test_both_arms_keep_full_contract_guidance_and_mutation_sensitivity(self):
        first = contract_fixture()
        changed = copy.deepcopy(first)
        changed['discriminative_variable'] = 'ALTERED_MINER_TARGET'
        changed['required_evidence_groups']['before_state'] = ['D002']
        for arm in ARMS:
            primary = prompt_spec(expanded_fixture(), first, arm, DEFAULT_RUN)
            self.assertIn('CONTRACT GUIDANCE:', primary['prompt'])
            self.assertIn('MINER_TARGET_SENTINEL', primary['prompt'])
            self.assertNotEqual(primary, prompt_spec(expanded_fixture(), changed, arm, DEFAULT_RUN))
            self.assertNotEqual(repair_spec(expanded_fixture(), first, None, arm, DEFAULT_RUN),
                                repair_spec(expanded_fixture(), changed, None, arm, DEFAULT_RUN))

    def test_same_raw_repair_pair_is_exactly_original_plus_one_suffix(self):
        suffix = '\n' + guidance_text(DEFAULT_RUN)
        for raw in (None, {}, batch_fixture('synthetic raw', (2, 1, 0))):
            original = builder.build_repair_prompt(expanded_fixture(), contract_fixture(), raw, 'contract_guided', intent_guidance=True)
            base = repair_spec(expanded_fixture(), contract_fixture(), raw, ARMS[0], DEFAULT_RUN)
            quality = repair_spec(expanded_fixture(), contract_fixture(), raw, ARMS[1], DEFAULT_RUN)
            self.assertEqual(base, original)
            self.assertEqual(quality['prompt'], original['prompt'] + suffix)
            self.assertEqual(quality['prompt'].count('BEGIN_SCMS_QUALITY_GUIDANCE_V1'), 1)
            self.assertEqual(quality['prompt_version'], base['prompt_version'])
            self.assertEqual(quality['expected_counts'], base['expected_counts'])
            if raw is None:
                self.assertTrue(base['prompt'].endswith('\n{}'))
                self.assertTrue(quality['prompt'].removesuffix(suffix).endswith('\n{}'))

    def test_actual_shared_calls_keep_contract_and_same_suffix_without_miner_or_free(self):
        with patch.object(generation, 'generate_root_intents', side_effect=AssertionError('Miner forbidden')), \
                patch.object(generation, 'build_free_prompt', side_effect=AssertionError('Free forbidden')):
            for arm in ARMS:
                raw = batch_fixture('primary', (2, 1, 0))
                with tempfile.TemporaryDirectory() as temporary:
                    result, calls, ledger = run_batch([raw, batch_fixture('repair')], Path(temporary), arm)
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0]['prompt'], prompt_spec(expanded_fixture(), contract_fixture(), arm, DEFAULT_RUN)['prompt'])
                self.assertEqual(calls[1]['prompt'], repair_spec(expanded_fixture(), contract_fixture(), raw, arm, DEFAULT_RUN)['prompt'])
                self.assertEqual([item['max_tokens'] for item in calls], [8192, 8192])
                self.assertEqual([item['intent'] for item in ledger], ['h4_k3_candidate_generator', 'h4_k3_candidate_schema_repair'])
                self.assertTrue(all(all(row['contract_group_alignment'].values()) for row in result['candidates']))
                self.assertEqual(len(result['candidates']), 4)
                self.assertTrue(all(row['question'].startswith('repair') for row in result['candidates']))

    def test_extra_valid_rows_and_each_missing_type_keep_original_minimum_rule(self):
        for arm in ARMS:
            with tempfile.TemporaryDirectory() as temporary:
                result, calls, _ = run_batch([batch_fixture(counts=(3, 1, 1))], Path(temporary), arm)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(result['candidates']), 5)
            self.assertFalse(result['generation_records'][0]['validation']['valid'])
            self.assertTrue(result['generation_records'][0]['validation']['minimum_acceptance']['accepted'])
            for counts in ((1, 1, 1), (2, 0, 1), (2, 1, 0)):
                with self.subTest(arm=arm, counts=counts), tempfile.TemporaryDirectory() as temporary:
                    result, calls, _ = run_batch([batch_fixture('primary', counts), batch_fixture('repair')], Path(temporary), arm)
                self.assertEqual(len(calls), 2)
                self.assertEqual(len(result['candidates']), 4)
                self.assertTrue(all(row['question'].startswith('repair') for row in result['candidates']))

    def test_ordering_repair_still_insufficient_rejects_batch_without_salvage(self):
        for arm in ARMS:
            with tempfile.TemporaryDirectory() as temporary:
                result, calls, _ = run_batch([batch_fixture('primary', (2, 1, 0)), batch_fixture('repair', (2, 1, 0))], Path(temporary), arm)
            self.assertEqual(len(calls), 2)
            self.assertEqual(result['candidates'], [])
            self.assertEqual(len(result['generation_records']), 2)
            for record in result['generation_records']:
                self.assertEqual(record['validation']['minimum_acceptance']['missing_minimums'], {'ordering': 1})
                self.assertEqual(len(record['validation']['candidates']), 3)

    def test_structural_row_failures_and_malformed_json_are_retained(self):
        malformed = batch_fixture()
        malformed['candidates'][0].pop('evidence_refs')
        for arm in ARMS:
            for raw in (malformed, '{synthetic invalid JSON'):
                with self.subTest(arm=arm, raw=type(raw).__name__), tempfile.TemporaryDirectory() as temporary:
                    result, calls, _ = run_batch([raw, batch_fixture('repair')], Path(temporary), arm)
                self.assertEqual(len(calls), 2)
                self.assertFalse(result['generation_records'][0]['validation']['valid'])
                self.assertEqual(len(result['candidates']), 4)
                if isinstance(raw, str):
                    original = calls[1]['prompt'].removesuffix('\n' + quality_suffix(DEFAULT_RUN, arm)) if arm == ARMS[1] else calls[1]['prompt']
                    self.assertTrue(original.endswith('\n{}'))
                    self.assertNotIn(raw, calls[1]['prompt'])
                else:
                    self.assertIn('evidence_refs_invalid', result['generation_records'][0]['validation']['candidates'][0]['schema_errors'])

    def test_single_gold_multi_select_keeps_original_final_rejection_without_repair(self):
        for arm in ARMS:
            raw = batch_fixture()
            raw['candidates'][2]['gold'] = ['A']
            raw['candidates'][0]['options']['B'] = raw['candidates'][0]['options']['A']
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                result, calls, _ = run_batch([raw], root, arm)
                summary = generation.finalize_writer_candidates(
                    root_dir=root, root_id='test-root', candidates=result['candidates'],
                    generation_records=result['generation_records'], expanded=expanded_fixture(),
                )
                audit = read_json(root / 'validity_audit.json')
            self.assertEqual(len(calls), 1)
            self.assertEqual(result['candidates'][2]['gold'], ['A'])
            self.assertEqual(summary['candidate_count'], 4)
            self.assertEqual(summary['valid_candidate_count'], 2)
            self.assertIn('multi_gold_invalid', audit['rows'][2]['blocking_errors'])
            self.assertIn('duplicate_option_or_item_text', audit['rows'][0]['blocking_errors'])
            self.assertEqual(summary['gold_rebinding_errors'], 0)


if __name__ == '__main__':
    unittest.main()
