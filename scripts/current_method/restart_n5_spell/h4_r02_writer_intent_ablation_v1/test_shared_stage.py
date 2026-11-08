"""Offline parity and dependency tests; all responses are synthetic fixtures."""
from __future__ import annotations

import copy
import importlib.util
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ..h4_qa_mining_hybrid_generation_v1 import contract_guided_generator as builder
from ..h4_qa_multiroot_k3_t08_probe_v1 import generation
from ..h4_r02_feedback_transfer_p2_2_v1.test_generation_hooks import (
    FakeClient, FakeLedger, TrackingHooks, batch_fixture, expanded_fixture,
    intents_fixture, run_fake,
)

RUNS = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs')
RUN = RUNS / 'h4_r02_writer_intent_ablation_v1_live_1'
SOURCE = RUNS / 'h4_r02_basic_miner_v3_off_paired_v1_live_1'
BEFORE = RUN / 'policy_snapshot/before_code/scripts/current_method/restart_n5_spell'


def original_module(current, relative):
    spec = importlib.util.spec_from_file_location(current.__package__ + '._writer_before', BEFORE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contract_fixture():
    return {'intent_id': 'intent_1', 'intent_type': 'state_transition',
            'discriminative_variable': 'MINER_TARGET_SENTINEL',
            'required_evidence_groups': {'before_state': ['D001'], 'trigger': ['D002'],
                                         'after_state': ['D003']}}


def run_batch(replies, root, *, contract=None, intent_guidance=True, **kwargs):
    client, ledger = FakeClient(replies), FakeLedger()
    result = generation.execute_writer_batch(
        expanded=expanded_fixture(), contract=contract or contract_fixture(),
        root_dir=root, root_id='test-root', client=client, ledger=ledger,
        intent_guidance=intent_guidance, **kwargs)
    return result, client.calls, ledger.calls


class SharedWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_generation = original_module(generation, 'h4_qa_multiroot_k3_t08_probe_v1/generation.py')
        cls.old_builder = original_module(builder, 'h4_qa_mining_hybrid_generation_v1/contract_guided_generator.py')

    def setUp(self):
        self.connection = patch.object(socket.socket, 'connect', side_effect=AssertionError('network prohibited'))
        self.connection.start()
        self.addCleanup(self.connection.stop)

    def test_all_six_a_primary_and_none_repair_match_saved_baselines(self):
        checked = []
        for suffix in ('198480', '122711'):
            root = 'cc-train-episode-' + suffix
            expanded = json.loads((SOURCE / 'inputs/shared' / root / 'expanded_dialogue_clean.json').read_text())
            contracts = json.loads((SOURCE / 'arms/V3_OFF/qa/shards/shard-00/roots' / root / 'reasoning_contracts.json').read_text())
            self.assertEqual(len(contracts), 3)
            for contract in contracts:
                slot = contract['intent_id']
                saved = RUN / 'policy_snapshot/baseline_guided_prompts' / root
                self.assertEqual(builder.build_prompt(expanded, contract)['prompt'].encode(),
                                 (saved / (slot + '.txt')).read_bytes())
                self.assertEqual(builder.build_prompt(expanded, contract), self.old_builder.build_prompt(expanded, contract))
                for raw in (None, batch_fixture()):
                    self.assertEqual(builder.build_repair_prompt(expanded, contract, raw, 'contract_guided'),
                                     self.old_builder.build_repair_prompt(expanded, contract, raw, 'contract_guided'))
                self.assertEqual(builder.build_repair_prompt(expanded, contract, None, 'contract_guided')['prompt'].encode(),
                                 (saved / (slot + '.repair_none.txt')).read_bytes())
                checked.append((root, slot))
        self.assertEqual(len(checked), 6)

    def test_b_primary_repair_ignore_semantic_intent_mutation_a_changes(self):
        first = contract_fixture()
        second = copy.deepcopy(first)
        second.update(intent_type='necessary_evidence', discriminative_variable='ALTERED_MINER_TARGET',
                      required_evidence_groups={'before_state': ['TOTALLY_DIFFERENT_REF'],
                                                'trigger': ['ANY'], 'after_state': ['OTHER']})
        expanded = expanded_fixture()
        self.assertNotEqual(builder.build_prompt(expanded, first), builder.build_prompt(expanded, second))
        self.assertEqual(builder.build_prompt(expanded, first, intent_guidance=False),
                         builder.build_prompt(expanded, second, intent_guidance=False))
        for raw in (None, batch_fixture()):
            self.assertEqual(builder.build_repair_prompt(expanded, first, raw, 'contract_guided', intent_guidance=False),
                             builder.build_repair_prompt(expanded, second, raw, 'contract_guided', intent_guidance=False))
        prompt = builder.build_prompt(expanded, first, intent_guidance=False)['prompt']
        header, dialogue = prompt.split('DIALOGUE:\n')
        self.assertNotIn('CONTRACT GUIDANCE:', header)
        self.assertNotIn('MINER_TARGET_SENTINEL', header)
        self.assertNotIn('for this intent', header)
        shape = json.loads(header.split('OUTPUT SHAPE:\n')[1])
        self.assertEqual(shape['candidates'][0]['discriminative_variable'], '...')
        self.assertEqual(shape['candidates'][0]['intent_reference'], first['intent_id'])
        self.assertEqual(dialogue, builder._dialogue(expanded))

    def test_b_normalization_and_candidate_metadata_independent_of_miner(self):
        first = contract_fixture()
        second = copy.deepcopy(first)
        second.update(intent_type='DIFFERENT_TYPE', discriminative_variable='DIFFERENT_VARIABLE',
                      required_evidence_groups={'before_state': ['D003']})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            replies = [batch_fixture('primary', (2, 1, 0)), batch_fixture('repair')]
            a = run_batch(replies, root / 'one', contract=first, intent_guidance=False)
            b = run_batch(replies, root / 'two', contract=second, intent_guidance=False)
        self.assertEqual(a, b)
        self.assertEqual(len(a[1]), 2)
        for row in a[0]['candidates']:
            self.assertEqual(row['discriminative_variable'], 'Room reservation')
            self.assertTrue(row['evidence_refs'])
            self.assertTrue(row['evidence_groups'])
            self.assertTrue(all(not value for value in row['contract_group_alignment'].values()))
            self.assertNotIn('intent_type', row)
            self.assertEqual(row['generation_source'], 'contract_guided')

    def test_b_own_variable_and_evidence_required_with_no_fallback(self):
        for field in ('discriminative_variable', 'evidence_refs', 'evidence_groups'):
            for enabled in (True, False):
                raw = batch_fixture()
                for row in raw['candidates']:
                    row.pop(field)
                with self.subTest(field=field, enabled=enabled), tempfile.TemporaryDirectory() as temporary:
                    result, calls, _ = run_batch([raw, raw], Path(temporary), intent_guidance=enabled)
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(result['candidates'], [])
                    self.assertTrue(all(not row['schema_valid'] for record in result['generation_records']
                                        for row in record['validation']['candidates']))

    def test_b_disallows_external_suffix_or_hook_injection(self):
        for kwargs in ({'slot_suffix': 'unexpected'}, {'hooks': TrackingHooks()}, {'source': 'free_exploration'}):
            with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
                run_batch([], Path(temporary), intent_guidance=False, **kwargs)

    def test_generate_root_qa_still_mines_and_uses_shared_guided_and_free(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(generation, 'execute_writer_batch', wraps=generation.execute_writer_batch) as shared:
            result, files, calls, ledger = run_fake(generation, [intents_fixture()] + [batch_fixture()] * 6, Path(temporary))
        self.assertEqual(shared.call_count, 6)
        self.assertEqual([x.kwargs['source'] for x in shared.call_args_list], ['contract_guided', 'free_exploration'] * 3)
        self.assertEqual(ledger[0]['intent'], 'h4_k3_intent_miner')
        self.assertEqual(len(calls), 7)
        self.assertEqual(result['valid_candidate_count'], 24)

    def test_default_and_legacy_hooks_match_before_refactor_bytes(self):
        cases = [
            [intents_fixture()] + [batch_fixture()] * 6,
            [intents_fixture(), batch_fixture('primary', (2, 1, 0)), batch_fixture('repair')] + [batch_fixture()] * 5,
            [intents_fixture(), 'bad JSON', RuntimeError('synthetic failure')] + [batch_fixture()] * 5,
        ]
        for hooks_enabled in (False, True):
            for index, replies in enumerate(cases):
                with self.subTest(hooks=hooks_enabled, case=index), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    old_hook, new_hook = (TrackingHooks(), TrackingHooks()) if hooks_enabled else (None, None)
                    context = {'intent_suffix': 'OLD_MINER_SUFFIX', 'slot_suffixes': ['S1', 'S2', 'S3']}
                    old = run_fake(self.old_generation, replies, root / 'old', hooks=old_hook, feedback_context=context)
                    new = run_fake(generation, replies, root / 'new', hooks=new_hook, feedback_context=context)
                    self.assertEqual(old, new)
                    if hooks_enabled:
                        self.assertEqual(vars(old_hook), vars(new_hook))

    def test_batch_never_calls_miner_or_free_builder(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(generation, 'generate_root_intents', side_effect=AssertionError('Miner forbidden')), patch.object(generation, 'build_free_prompt', side_effect=AssertionError('Free forbidden')):
            for enabled in (True, False):
                result, calls, _ = run_batch([batch_fixture()], Path(temporary), intent_guidance=enabled)
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(result['candidates']), 4)

    def test_minima_accepts_extra_valid_rows_without_repair(self):
        for enabled in (True, False):
            with tempfile.TemporaryDirectory() as temporary:
                result, calls, _ = run_batch([batch_fixture(counts=(3, 1, 1))], Path(temporary), intent_guidance=enabled)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(result['candidates']), 5)
            self.assertFalse(result['generation_records'][0]['validation']['valid'])
            self.assertTrue(result['generation_records'][0]['validation']['minimum_acceptance']['accepted'])

    def test_missing_each_type_repairs_once_and_replaces_entire_batch(self):
        for enabled in (True, False):
            for counts in ((1, 1, 1), (2, 0, 1), (2, 1, 0)):
                with self.subTest(enabled=enabled, counts=counts), tempfile.TemporaryDirectory() as temporary:
                    result, calls, _ = run_batch([batch_fixture('primary', counts), batch_fixture('repair')], Path(temporary), intent_guidance=enabled)
                self.assertEqual(len(calls), 2)
                self.assertEqual(len(result['candidates']), 4)
                self.assertTrue(all(row['question'].startswith('repair') for row in result['candidates']))
                self.assertEqual([row['max_tokens'] for row in calls], [8192, 8192])
                self.assertTrue(calls[1]['prompt'].endswith(json.dumps(batch_fixture('primary', counts), ensure_ascii=False)))
                self.assertEqual('CONTRACT GUIDANCE:' in calls[1]['prompt'], enabled)

    def test_parse_failure_repair_uses_original_none_to_empty_object_rule(self):
        for enabled in (True, False):
            with tempfile.TemporaryDirectory() as temporary:
                result, calls, _ = run_batch(['{unparseable', batch_fixture()], Path(temporary), intent_guidance=enabled)
            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[1]['prompt'].endswith('\n{}'))
            self.assertNotIn('{unparseable', calls[1]['prompt'])
            self.assertEqual(len(result['candidates']), 4)

    def test_repair_still_insufficient_keeps_failure_no_primary_salvage(self):
        for enabled in (True, False):
            with tempfile.TemporaryDirectory() as temporary:
                result, calls, _ = run_batch([batch_fixture('primary', (2, 1, 0)), batch_fixture('repair', (0, 0, 1))], Path(temporary), intent_guidance=enabled)
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(result['generation_records']), 2)
            self.assertEqual(result['candidates'], [])

    def test_final_validator_rejections_do_not_trigger_additional_repair(self):
        for enabled in (True, False):
            raw = batch_fixture()
            raw['candidates'][0]['options']['B'] = raw['candidates'][0]['options']['A']
            raw['candidates'][2]['gold'] = ['A']
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                result, calls, _ = run_batch([raw], root, intent_guidance=enabled)
                summary = generation.finalize_writer_candidates(root_dir=root, root_id='test-root',
                    candidates=result['candidates'], generation_records=result['generation_records'], expanded=expanded_fixture())
                audit = json.loads((root / 'validity_audit.json').read_text())
            self.assertEqual(len(calls), 1)
            self.assertEqual(summary['candidate_count'], 4)
            self.assertEqual(summary['valid_candidate_count'], 2)
            self.assertIn('duplicate_option_or_item_text', audit['rows'][0]['blocking_errors'])
            self.assertIn('multi_gold_invalid', audit['rows'][2]['blocking_errors'])
            self.assertEqual(summary['gold_rebinding_errors'], 0)

    def test_arm_pools_separate_id_validation_and_gold_rebinding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = []
            for arm, enabled in (('A', True), ('B', False)):
                result, calls, _ = run_batch([batch_fixture()], root / arm, intent_guidance=enabled)
                summary = generation.finalize_writer_candidates(root_dir=root / arm, root_id='test-root',
                    candidates=result['candidates'], generation_records=result['generation_records'], expanded=expanded_fixture())
                self.assertEqual(summary['valid_candidate_count'], 4)
                self.assertEqual(summary['gold_rebinding_errors'], 0)
                candidates.append(result['candidates'])
            self.assertEqual([x['candidate_id'] for x in candidates[0]], [x['candidate_id'] for x in candidates[1]])
            duplicated = generation.finalize_writer_candidates(root_dir=root / 'duplicate_control', root_id='test-root',
                candidates=candidates[0] + candidates[0], generation_records=[], expanded=expanded_fixture())
            self.assertEqual(duplicated['valid_candidate_count'], 4)

    def test_after_call_rethrows_swallowed_guard_before_repair(self):
        class GuardFailure(RuntimeError):
            pass
        seen = []
        def guard(**event):
            seen.append(event)
            raise GuardFailure('synthetic unknown completion; stop')
        with tempfile.TemporaryDirectory() as temporary:
            client, ledger = FakeClient([GuardFailure('freeze failed')]), FakeLedger()
            with self.assertRaisesRegex(GuardFailure, 'unknown completion'):
                generation.execute_writer_batch(expanded=expanded_fixture(), contract=contract_fixture(),
                    root_dir=Path(temporary), root_id='test-root', client=client, ledger=ledger, after_call=guard)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]['stage'], 'primary')
        self.assertIn('GuardFailure:freeze failed', seen[0]['parse_error'])

    def test_after_call_visits_both_primary_and_repair_in_order(self):
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            run_batch([batch_fixture(counts=(2, 1, 0)), batch_fixture()], Path(temporary),
                      after_call=lambda **event: events.append(event))
        self.assertEqual([x['stage'] for x in events], ['primary', 'repair'])
        self.assertTrue(all(x['parse_error'] is None for x in events))


if __name__ == '__main__':
    unittest.main()
