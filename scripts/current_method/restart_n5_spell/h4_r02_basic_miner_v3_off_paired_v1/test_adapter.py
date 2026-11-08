"""No-network OFF pairing, literal-template, schema and shared-hook checks."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from ..h4_qa_multiroot_k3_t08_probe_v1 import generation
from ..h4_qa_mining_probe_v1.intent_contract import canonicalize_intents
from ..h4_qa_mining_probe_v1.intent_miner import build_intent_prompt
from ..h4_r02_feedback_transfer_p2_2_v1.transfer import miner_output_schema
from ..h4_r02_feedback_transfer_p2_2_v1.test_generation_hooks import FakeClient, FakeLedger
from ..h4_r02_miner_binding_v2.test_transfer import expanded, intent, cards, ROOT
from ..h4_r02_miner_binding_v2 import test_shared_miner as prior_parity
from ..h4_r02_miner_three_arm_diagnostic_v1.adapter import ArmHooks, build_prompt as previous_prompt
from .adapter import (
    ARMS, GUIDANCE_FILE, DIALOGUE_MARKER, GUIDANCE_BEGIN, GUIDANCE_END,
    MinerHooks, audit_response_schema, build_prompt, feedback_payload,
    guidance_text, guidance_insertion, output_schema, prompt_pairing_diagnostics,
)

THREE_RUN = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_miner_three_arm_diagnostic_v1_live_1')
WORK_ORDER = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_basic_miner_v3_off_paired_v1_handoff/CODEX_WORK_ORDER.md')


def raw_fixture():
    return {'intents': [intent('named_101'), intent('own_intent_3'), intent('unrelated_id_17')]}


def fake_miner(hook, raw, folder, transcript=None):
    transcript = transcript or expanded()
    client, ledger = FakeClient([deepcopy(raw)]), FakeLedger()
    with patch.object(generation, 'generate_root_qa', side_effect=AssertionError('WRITER_ENTRY_FORBIDDEN')):
        result = generation.generate_root_intents(expanded=transcript, root_dir=folder, root_id=ROOT,
            api_config=Path('/nonexistent'), client=client, ledger=ledger, hooks=hook)
    artifacts = {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    return result, artifacts, client, ledger


class PairedAdapterTests(unittest.TestCase):
    def setUp(self):
        for obj, name in ((socket, 'create_connection'), (socket.socket, 'connect')):
            guard = patch.object(obj, name, side_effect=AssertionError('NETWORK_FORBIDDEN'))
            guard.start()
            self.addCleanup(guard.stop)

    def test_guidance_bytes_equal_complete_work_order_block_and_schema_exact(self):
        order = WORK_ORDER.read_bytes().decode('utf-8')
        start = order.index(GUIDANCE_BEGIN + '\n')
        end = order.index('\n' + GUIDANCE_END, start) + len('\n' + GUIDANCE_END)
        self.assertEqual(GUIDANCE_FILE.read_bytes(), (order[start:end] + '\n').encode())
        self.assertEqual(guidance_text().encode(), GUIDANCE_FILE.read_bytes())
        self.assertEqual(json.loads(GUIDANCE_FILE.with_name('schema.json').read_bytes()), miner_output_schema(envelope=False))
        for arm in ARMS:
            self.assertEqual(output_schema(arm), miner_output_schema(envelope=False))

    def test_both_roots_exact_dual_archived_baseline_and_single_v3_insertion(self):
        for root in ('cc-train-episode-198480', 'cc-train-episode-122711'):
            with self.subTest(root=root):
                folder = THREE_RUN / 'inputs/shared' / root
                transcript = json.loads((folder / 'expanded_dialogue_clean.json').read_bytes())
                active = json.loads((folder / 'observation_cards.json').read_bytes())
                history = json.loads((folder / 'history_only.json').read_bytes())
                policy = THREE_RUN / 'policy_snapshot/prompts/A_OFF' / root / 'miner_prompt.txt'
                actual = THREE_RUN / 'arms/A_OFF/qa/shards/shard-00/roots' / root / 'raw/prompts' / f'{root}-intent-miner.txt'
                report = prompt_pairing_diagnostics(transcript, active, history, root, policy, actual)
                self.assertEqual(report['status'], 'PASS')
                self.assertTrue(all(report['checks'].values()))
                original = build_intent_prompt(transcript)
                base = build_prompt('BASE_OFF', original, active, history, expanded=transcript, root_id=root)
                v3 = build_prompt('V3_OFF', original, active, history, expanded=transcript, root_id=root)
                self.assertEqual(base['prompt'].encode(), policy.read_bytes())
                self.assertEqual(base['prompt'].encode(), actual.read_bytes())
                self.assertEqual(v3['prompt'].replace(guidance_insertion(), '', 1), base['prompt'])
                self.assertEqual(v3['prompt'].split(DIALOGUE_MARKER, 1)[1], base['prompt'].split(DIALOGUE_MARKER, 1)[1])

    def test_pairing_diagnostics_do_not_normalize_archived_line_endings(self):
        original = build_intent_prompt(expanded())
        base = build_prompt('BASE_OFF', original, cards(), {}, expanded=expanded(), root_id=ROOT)['prompt']
        with tempfile.TemporaryDirectory() as temp:
            policy = Path(temp) / 'policy.txt'
            policy.write_bytes(base.replace('\n', '\r\n').encode())
            report = prompt_pairing_diagnostics(expanded(), cards(), {}, ROOT, policy, base)
        self.assertEqual(report['status'], 'FAIL')
        self.assertFalse(report['checks']['base_equals_archived_policy_bytes'])

    def test_empty_active_cards_safe_identical_history_no_transfer_or_arm_labels(self):
        history = {'records': [{'root_id': ROOT, 'stem': 'Preserved historical stem', 'source_candidate_id': 'old',
            'historical_gold': 'PRIVATE_GOLD', 'solver_observations': {'M0': 'PRIVATE_SCORE'},
            'candidate_sha256': 'history-hash'}]}
        original = build_intent_prompt(expanded())
        payload = feedback_payload(cards(), history, ROOT)
        self.assertEqual(payload['observation_cards'], [])
        self.assertEqual(payload['feedback_status'], 'NO_ACTIVE_FEEDBACK')
        self.assertEqual(payload['historical_tracking'][0]['stem'], 'Preserved historical stem')
        for arm in ARMS:
            prompt = build_prompt(arm, original, cards(), history, expanded=expanded(), root_id=ROOT)['prompt']
            self.assertNotIn('PRIVATE_', prompt)
            self.assertNotIn('feedback_bindings', prompt)
            self.assertNotIn('NO_SUPPORTED_TRANSFER', prompt)
            self.assertNotIn('FB-198480-I003-C-Q03', prompt)
            self.assertNotIn(arm, prompt)
            self.assertIn('Preserved historical stem', prompt)

    def test_missing_or_duplicate_dialogue_marker_cannot_guess_insertion(self):
        original = build_intent_prompt(expanded())
        for prompt in (original['prompt'].replace(DIALOGUE_MARKER, ''), original['prompt'] + DIALOGUE_MARKER):
            with self.assertRaises(ValueError):
                build_prompt('V3_OFF', {**original, 'prompt': prompt}, cards(), {}, expanded=expanded(), root_id=ROOT)

    def test_base_builder_is_pure_previous_a_off_exact_model_bytes(self):
        original = build_intent_prompt(expanded())
        snapshot = deepcopy(original)
        previous = previous_prompt('A_OFF', original, cards(), {}, expanded=expanded(), root_id=ROOT)
        new = build_prompt('BASE_OFF', original, cards(), {}, expanded=expanded(), root_id=ROOT)
        self.assertEqual(previous['prompt'], new['prompt'])
        self.assertEqual(original, snapshot)
        self.assertFalse(new['guidance_inserted'])

    def test_both_arms_shared_normal_return_no_binding_or_writer(self):
        for arm in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temp:
                hook = MinerHooks(arm, cards(), {}, transcript_sha256='frozen-bytes')
                result, files, client, ledger = fake_miner(hook, raw_fixture(), Path(temp))
                self.assertEqual(len(client.calls), 1)
                self.assertEqual(ledger.calls[0]['request_id'], ROOT + '-intent-miner')
                self.assertEqual(client.calls[0]['max_tokens'], 4096)
                self.assertEqual(len(result['contracts']), 3)
                self.assertEqual(hook.derived[ROOT]['usable_intent_count'], 3)
                self.assertEqual(hook.validations, {})
                self.assertEqual(hook.previews, {})
                self.assertFalse(hasattr(hook, 'writer_suffix'))
                self.assertFalse(any('candidate' in filename or 'binding' in filename or 'brief' in filename for filename in files))
                self.assertTrue(json.loads(files['response_schema_audit.json'])['schema_valid'])
                self.assertEqual(json.loads(files['arm_response_schema.json'])['arm_id'], arm)

    def test_base_original_processing_parity_with_previous_a_off_hook(self):
        raw = raw_fixture()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            old = fake_miner(ArmHooks('A_OFF', cards(), {}, transcript_sha256='same-hash'), raw, directory / 'old')
            new = fake_miner(MinerHooks('BASE_OFF', cards(), {}, transcript_sha256='same-hash'), raw, directory / 'new')
            self.assertEqual(old[0], new[0])
            self.assertEqual(old[2].calls, new[2].calls)
            self.assertEqual(old[3].calls, new[3].calls)
            for name in ('reasoning_contracts.json', 'derived_intent_evidence.json',
                    f'raw/prompts/{ROOT}-intent-miner.txt', f'raw/responses/{ROOT}-intent-miner.json.txt'):
                self.assertEqual(old[1][name], new[1][name])

    def test_unknown_top_level_and_parse_errors_preserve_raw_no_projection(self):
        for arm in ARMS:
            for raw in ({'intents': raw_fixture()['intents'], 'feedback_bindings': []}, '{not valid JSON'):
                with self.subTest(arm=arm, raw_type=type(raw).__name__), tempfile.TemporaryDirectory() as temp:
                    hook = MinerHooks(arm, cards(), {})
                    result, files, client, _ = fake_miner(hook, raw, Path(temp))
                    self.assertFalse(result['intent_result']['valid'])
                    self.assertEqual(result['contracts'], [])
                    self.assertFalse(json.loads(files['response_schema_audit.json'])['schema_valid'])
                    recorded = files[f'raw/responses/{ROOT}-intent-miner.json.txt'].decode()
                    self.assertEqual(recorded if isinstance(raw, str) else json.loads(recorded), raw)
                    self.assertEqual(len(client.calls), 1)
                    self.assertIs(hook.project_miner_response(raw, expanded=expanded(), root_id=ROOT, root_dir=Path(temp)), raw)

    def test_schema_audit_does_not_hide_original_canonicalizer_defaults(self):
        raw = raw_fixture()
        del raw['intents'][0]['intent_id']
        del raw['intents'][0]['candidate_question_types']
        original = canonicalize_intents(raw, expanded_dialogue=expanded())
        self.assertTrue(original['valid'])
        self.assertEqual(original['intents'][0]['intent_id'], 'I001')
        audit = audit_response_schema(raw)
        self.assertFalse(audit['schema_valid'])
        self.assertFalse(audit['affects_original_admission'])
        self.assertTrue(any(error['keyword'] == 'required' for error in audit['errors']))
        with tempfile.TemporaryDirectory() as temp:
            result, files, _, _ = fake_miner(MinerHooks('V3_OFF', cards(), {}), raw, Path(temp))
            self.assertEqual(len(result['contracts']), 3)
            self.assertTrue(result['intent_result']['valid'])
            self.assertFalse(json.loads(files['response_schema_audit.json'])['schema_valid'])

    def test_original_schema_not_replaced_by_a_stricter_new_schema(self):
        raw = raw_fixture()
        raw['intents'][0]['extra_item_field'] = 'The original schema allows this.'
        raw['intents'][0]['reasoning_chain']['extra_group'] = [12]
        self.assertTrue(audit_response_schema(raw)['schema_valid'])
        self.assertTrue(canonicalize_intents(raw, expanded_dialogue=expanded())['valid'])
        raw['intents'][0]['reasoning_chain']['trigger'] = [1]
        self.assertTrue(audit_response_schema(raw)['schema_valid'])
        self.assertFalse(canonicalize_intents(raw, expanded_dialogue=expanded())['valid'])
        raw = raw_fixture()
        raw['intents'][0]['discriminative_variable'] = '  '
        self.assertTrue(audit_response_schema(raw)['schema_valid'])
        self.assertFalse(canonicalize_intents(raw, expanded_dialogue=expanded())['valid'])

    def test_invalid_and_extra_rows_kept_no_filter_shift_or_best_three(self):
        raw = raw_fixture()
        raw['intents'][1]['reasoning_chain']['trigger'] = []
        raw['intents'].append(intent('extra_fourth'))
        with tempfile.TemporaryDirectory() as temp:
            hook = MinerHooks('V3_OFF', cards(), {})
            result, files, _, _ = fake_miner(hook, raw, Path(temp))
        self.assertEqual(len(result['intent_result']['intents']), 4)
        self.assertEqual([row['intent_id'] for row in result['contracts']], ['named_101', 'unrelated_id_17', 'extra_fourth'])
        self.assertEqual([row['intent_index'] for row in hook.derived[ROOT]['items']], [0, 1, 2, 3])
        self.assertFalse(hook.derived[ROOT]['items'][1]['usable'])
        self.assertFalse(json.loads(files['response_schema_audit.json'])['schema_valid'])

    def test_evidence_derivation_still_no_semantic_search_or_width_cap(self):
        transcript = expanded(150)
        raw = raw_fixture()
        transcript['turns'][149]['text'] = 'Unique target outside the first declared chain'
        raw['intents'][0]['discriminative_variable'] = transcript['turns'][149]['text']
        raw['intents'][2]['reasoning_chain'] = {'before_state': [1], 'trigger': [2], 'after_state': list(range(3, 151))}
        for arm in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temp:
                hook = MinerHooks(arm, cards(), {})
                fake_miner(hook, raw, Path(temp), transcript)
                rows = hook.derived[ROOT]['by_intent']
                refs = [row['turn_ref'] for row in rows['named_101']['derived_current_evidence']]
                self.assertNotIn(transcript['turns'][149]['turn_ref'], refs)
                self.assertEqual(len(rows['unrelated_id_17']['derived_current_evidence']), 150)


class OriginalSharedPathRegression(prior_parity.SharedMinerParityTests):
    """Unchanged original no-hook and v1 hook snapshot regressions."""


if __name__ == '__main__':
    unittest.main()
