"""No-network prompt pairing, arm isolation and unchanged shared-path replay."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from ..core import sha256_file
from ..h4_qa_multiroot_k3_t08_probe_v1 import generation
from ..h4_qa_mining_probe_v1.intent_miner import build_intent_prompt
from ..h4_r02_feedback_transfer_p2_2_v1.test_generation_hooks import FakeClient, FakeLedger
from ..h4_r02_feedback_transfer_p2_2_v1.transfer import model_cards, model_history
from ..h4_r02_miner_binding_v2 import test_shared_miner as prior_parity
from ..h4_r02_miner_binding_v2.test_transfer import expanded, intent, proposal, no_transfer, cards, ROOT
from ..h4_r02_miner_binding_v2.transfer import TransferHooksV2, MINER_BINDING_RULES
from .adapter import (
    ARMS, ArmHooks, build_prompt, feedback_payload, output_schema,
    prompt_pairing_diagnostics, OLD_SCOPE_SENTENCE, NEW_SCOPE_SENTENCE, PAYLOAD_MARKER,
)

V2_RUN = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_miner_binding_v2_acceptance_v1_live_1')


def response(binding_kind=None):
    raw = {"intents": [intent("id_17"), intent("id_309"), intent("named_intent")]}
    if binding_kind is not None:
        raw["feedback_bindings"] = [(no_transfer(row["intent_id"]) if binding_kind == "no" else proposal(row["intent_id"])) for row in raw["intents"]]
    return raw


def run_fake(arm, raw, root_dir, transcript=None):
    transcript = transcript or expanded()
    client, ledger = FakeClient([deepcopy(raw)]), FakeLedger()
    hook = ArmHooks(arm, cards(), {"records": []}, transcript_sha256="frozen-hash")
    with patch.object(generation, 'generate_root_qa', side_effect=AssertionError('WRITER_ENTRY_FORBIDDEN')):
        result = generation.generate_root_intents(expanded=transcript, root_dir=root_dir, root_id=ROOT,
            api_config=Path('/nonexistent'), ledger=ledger, client=client, hooks=hook)
    artifacts = {p.relative_to(root_dir).as_posix(): p.read_bytes() for p in root_dir.rglob('*') if p.is_file()}
    return result, artifacts, client, ledger, hook


class ArmAdapterTests(unittest.TestCase):
    def setUp(self):
        for obj, name in ((socket, 'create_connection'), (socket.socket, 'connect')):
            guard = patch.object(obj, name, side_effect=AssertionError('NETWORK_FORBIDDEN'))
            guard.start()
            self.addCleanup(guard.stop)

    def test_real_frozen_prompt_pairing_and_c_exact_scope_only_change(self):
        for root in ('cc-train-episode-198480', 'cc-train-episode-122711'):
            with self.subTest(root=root):
                source = V2_RUN / 'inputs' / root
                transcript = json.loads((source / 'expanded_dialogue_clean.json').read_text())
                active = json.loads((source / 'observation_cards.json').read_text())
                history = json.loads((source / 'history_only.json').read_text())
                prior = (V2_RUN / 'qa/shards/shard-00/roots' / root / 'raw/prompts' / f'{root}-intent-miner.txt').read_text()
                report = prompt_pairing_diagnostics(build_intent_prompt(transcript), active, history,
                    expanded=transcript, root_id=root, legacy_v2_prompt=prior)
                self.assertEqual(report['status'], 'PASS')
                self.assertTrue(all(report['checks'].values()))
                self.assertEqual(report['c_v2_comparison_basis'], 'supplied_historical_v2_prompt_bytes')

    def test_a_empty_cards_keeps_same_safe_history_b_c_exact_eligible_observations(self):
        history = {"records": [{"root_id": ROOT, "stem": "Same old tracking", "historical_gold": "PRIVATE_GOLD",
            "solver_observations": {"x": "PRIVATE_SCORE"}, "source_candidate_id": "history-only"}]}
        active = cards()
        wrong_root = deepcopy(active['observation_cards'][0])
        wrong_root.update(root_id='different-root', feedback_id='PRIVATE_CROSS_ROOT')
        inactive = deepcopy(active['observation_cards'][0])
        inactive.update(source_quality_status='EXCLUDED_ACTIVE_GUIDANCE', feedback_id='PRIVATE_INACTIVE')
        active['observation_cards'].extend([wrong_root, inactive])
        payloads = {arm: feedback_payload(arm, active, history, ROOT) for arm in ARMS}
        self.assertEqual(payloads[ARMS[0]]['observation_cards'], [])
        self.assertEqual(payloads[ARMS[0]]['feedback_status'], 'NO_ACTIVE_FEEDBACK')
        self.assertEqual(payloads[ARMS[1]], payloads[ARMS[2]])
        self.assertEqual(payloads[ARMS[1]]['observation_cards'], model_cards(cards(), root_id=ROOT))
        for payload in payloads.values():
            self.assertEqual(payload['historical_tracking'], model_history(history, root_id=ROOT))
            self.assertNotIn('PRIVATE_', json.dumps(payload))

    def test_a_b_schema_has_no_bindings_rules_or_transfer_request(self):
        original = build_intent_prompt(expanded())
        original_bytes = deepcopy(original)
        for arm in ARMS[:2]:
            spec = build_prompt(arm, original, cards(), {}, expanded=expanded(), root_id=ROOT)
            self.assertEqual(set(output_schema(arm)['properties']), {'intents'})
            self.assertNotIn('feedback_bindings', spec['prompt'])
            self.assertNotIn('NO_SUPPORTED_TRANSFER', spec['prompt'])
            self.assertNotIn(MINER_BINDING_RULES, spec['prompt'])
            self.assertNotIn(arm, spec['prompt'])
            self.assertEqual(spec['prompt'].count(NEW_SCOPE_SENTENCE), 1)
            self.assertNotIn(OLD_SCOPE_SENTENCE, spec['prompt'])
        c = build_prompt(ARMS[2], original, cards(), {}, expanded=expanded(), root_id=ROOT)
        self.assertIn(MINER_BINDING_RULES, c['prompt'])
        self.assertIn('feedback_bindings', c['prompt'])
        self.assertEqual(original, original_bytes)
        self.assertIn(OLD_SCOPE_SENTENCE, build_intent_prompt(expanded())['prompt'])

    def test_scope_replacement_refuses_missing_duplicate_or_prechanged_sentence(self):
        original = build_intent_prompt(expanded())
        for prompt in (original['prompt'].replace(OLD_SCOPE_SENTENCE, ''),
                original['prompt'] + '\n' + OLD_SCOPE_SENTENCE,
                original['prompt'].replace(OLD_SCOPE_SENTENCE, NEW_SCOPE_SENTENCE)):
            changed = {**original, 'prompt': prompt}
            with self.assertRaisesRegex(ValueError, 'EXPLANATION_SCOPE_SOURCE_DRIFT'):
                build_prompt(ARMS[0], changed, cards(), {}, expanded=expanded(), root_id=ROOT)

    def test_all_arms_one_shared_miner_call_normal_return_without_candidate_artifacts(self):
        for arm in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temp:
                raw = response('proposed' if arm == ARMS[2] else None)
                result, files, client, ledger, hook = run_fake(arm, raw, Path(temp))
                self.assertEqual(len(result['contracts']), 3)
                self.assertEqual(len(client.calls), 1)
                self.assertEqual(ledger.calls[0]['request_id'], ROOT + '-intent-miner')
                self.assertEqual(client.calls[0]['max_tokens'], 4096)
                self.assertFalse(any('candidate' in name or name.startswith('writer') for name in files))
                self.assertFalse(hasattr(hook, 'writer_suffix'))
                self.assertIn('derived_intent_evidence.json', files)
                if arm in ARMS[:2]:
                    self.assertEqual(hook.validations, {})
                    self.assertEqual(hook.previews, {})
                    self.assertNotIn('feedback_bindings.json', files)
                    self.assertNotIn('binding_validation.json', files)
                    self.assertNotIn('guided_brief_preview.json', files)
                    self.assertEqual(json.loads(files['arm_response_schema.json'])['binding_processing'], 'NOT_APPLICABLE')

    def test_a_b_do_not_project_unexpected_envelope_and_raw_is_saved(self):
        for arm in ARMS[:2]:
            for field in ('feedback_bindings', 'unknown_field'):
                with self.subTest(arm=arm, field=field), tempfile.TemporaryDirectory() as temp:
                    raw = response()
                    raw[field] = []
                    result, files, _, _, hook = run_fake(arm, raw, Path(temp))
                    self.assertIs(hook.project_miner_response(raw, expanded=expanded(), root_id=ROOT, root_dir=Path(temp)), raw)
                    self.assertFalse(result['intent_result']['valid'])
                    self.assertEqual(result['intent_result']['errors'], ['intent_top_level_schema'])
                    self.assertEqual(result['contracts'], [])
                    self.assertEqual(json.loads(files[f'raw/responses/{ROOT}-intent-miner.json.txt']), raw)
                    self.assertNotIn('binding_validation.json', files)

    def test_c_valid_invalid_missing_and_declined_keep_v2_levels(self):
        for kind in ('proposed', 'invalid', 'missing', 'no'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                raw = response('no' if kind == 'no' else 'proposed')
                if kind == 'missing':
                    del raw['feedback_bindings']
                elif kind == 'invalid':
                    raw['feedback_bindings'][0]['evidence_refs'] = ['forbidden-v1-field']
                result, files, _, _, hook = run_fake(ARMS[2], raw, Path(temp))
                self.assertEqual(len(result['contracts']), 3)
                validation = json.loads(files['binding_validation.json'])
                expected = {'proposed': 'PROPOSED_TRANSFER', 'invalid': 'BINDING_INVALID',
                    'missing': 'BINDING_MISSING', 'no': 'NO_SUPPORTED_TRANSFER'}[kind]
                self.assertEqual(validation['by_intent']['id_17']['binding_state'], expected)
                self.assertEqual(json.loads(files[f'raw/responses/{ROOT}-intent-miner.json.txt']), raw)
                self.assertEqual(hook.derived[ROOT]['usable_intent_count'], 3)

    def test_invalid_and_extra_intents_not_filtered_into_best_three(self):
        for arm in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temp:
                raw = response('proposed' if arm == ARMS[2] else None)
                raw['intents'][1]['reasoning_chain']['trigger'] = [1]
                raw['intents'].append(intent('extra_output'))
                result, files, _, _, hook = run_fake(arm, raw, Path(temp))
                self.assertEqual(len(result['intent_result']['intents']), 4)
                self.assertIn('intent_count:4', result['intent_result']['errors'])
                self.assertEqual(len(result['contracts']), 3)
                self.assertEqual([row['intent_id'] for row in result['contracts']], ['id_17', 'named_intent', 'extra_output'])
                self.assertEqual(len(hook.derived[ROOT]['items']), 4)
                self.assertFalse(hook.derived[ROOT]['items'][1]['usable'])

    def test_original_chain_only_no_semantic_selection_and_no_wide_chain_cap(self):
        transcript = expanded(150)
        for arm in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temp:
                raw = response('proposed' if arm == ARMS[2] else None)
                raw['intents'][0]['discriminative_variable'] = 'Unrelated target exactly mentioned at position 150'
                transcript['turns'][149]['text'] = raw['intents'][0]['discriminative_variable']
                raw['intents'][2]['reasoning_chain'] = {'before_state': [1], 'trigger': [2], 'after_state': list(range(3, 151))}
                _, _, _, _, hook = run_fake(arm, raw, Path(temp), transcript)
                rows = hook.derived[ROOT]['by_intent']
                refs = [row['turn_ref'] for row in rows['id_17']['derived_current_evidence']]
                self.assertNotIn(transcript['turns'][149]['turn_ref'], refs)
                self.assertNotIn(cards()['observation_cards'][0]['source_evidence_refs'][0], refs)
                self.assertEqual(len(rows['named_intent']['derived_current_evidence']), 150)

    def test_c_postprocessing_exact_v2_bytes_for_all_binding_states(self):
        for kind in ('proposed', 'no', 'missing', 'invalid'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                raw = response('no' if kind == 'no' else 'proposed')
                if kind == 'missing':
                    del raw['feedback_bindings']
                elif kind == 'invalid':
                    raw['feedback_bindings'][0]['change_from_source'] = 'old malformed string'
                current, files, _, _, _ = run_fake(ARMS[2], raw, base / 'arm')
                v2 = TransferHooksV2(cards(), {'records': []}, transcript_sha256='frozen-hash')
                v2.after_intents(raw, current['intent_result'], current['contracts'], expanded=expanded(), root_id=ROOT, root_dir=base / 'v2')
                for filename in ('feedback_bindings.json', 'binding_validation.json', 'derived_intent_evidence.json', 'guided_brief_preview.json'):
                    self.assertEqual(files[filename], (base / 'v2' / filename).read_bytes())

    def test_unchanged_v2_shared_miner_replays_frozen_raw_to_identical_original_artifacts(self):
        for root in ('cc-train-episode-198480', 'cc-train-episode-122711'):
            with self.subTest(root=root), tempfile.TemporaryDirectory() as temp:
                source = V2_RUN / 'inputs' / root
                original = V2_RUN / 'qa/shards/shard-00/roots' / root
                transcript = json.loads((source / 'expanded_dialogue_clean.json').read_text())
                active = json.loads((source / 'observation_cards.json').read_text())
                history = json.loads((source / 'history_only.json').read_text())
                raw = (original / 'raw/responses' / f'{root}-intent-miner.json.txt').read_text()
                hook = TransferHooksV2(active, history, transcript_sha256=sha256_file(source / 'expanded_dialogue_clean.json'))
                client, ledger = FakeClient([raw]), FakeLedger()
                target = Path(temp)
                generation.generate_root_intents(expanded=transcript, root_dir=target, root_id=root,
                    api_config=Path('/nonexistent'), client=client, ledger=ledger, hooks=hook)
                for filename in ('intents.json', 'reasoning_contracts.json', 'feedback_bindings.json',
                        'binding_validation.json', 'derived_intent_evidence.json', 'guided_brief_preview.json',
                        'miner_prompt_adaptation.json', f'raw/prompts/{root}-intent-miner.txt',
                        f'raw/responses/{root}-intent-miner.json.txt'):
                    self.assertEqual((target / filename).read_bytes(), (original / filename).read_bytes(), filename)
                self.assertEqual(len(client.calls), 1)


class OriginalSharedPathRegression(prior_parity.SharedMinerParityTests):
    """Reuse the existing no-hook/actual-v1 snapshot regression unchanged."""


if __name__ == '__main__':
    unittest.main()
