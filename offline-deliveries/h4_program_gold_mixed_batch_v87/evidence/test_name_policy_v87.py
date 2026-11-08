"""Negative output-contract tests: no HTTP, no Solver, no batch artifact edits."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HOME=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(HOME))
import actor_reference
import ordering_generation
from common import bind, read


class NamePolicyTests(unittest.TestCase):
    def test_forbidden_tokens_case_punctuation_and_repetition(self):
        for token in ('he','him','his','himself','she','her','hers','herself'):
            with self.subTest(token=token):
                self.assertEqual(actor_reference.name_policy_errors({'I1':token.upper()+', '+token+'.'}),
                    ['ordering_actor_reference_policy_forbidden_pronoun:I1:'+token])

    def test_substrings_and_source_names_remain_allowed(self):
        self.assertEqual(actor_reference.name_policy_errors({'I1':"Liam's thesis describes shelter history and shells.",
            'I2':"Elena mentions Ethan and Lena; the athlete will help."}),[])

    def test_names_policy_preserves_source_and_registry(self):
        cfg=read(HOME/'CONFIG.json')
        for slot in cfg['ordering_slots']:
            source=read(slot['source_path']); source_before=deepcopy(source)
            record=actor_reference.policy_for(cfg,slot,source)
            self.assertEqual(source,source_before)
            self.assertEqual(record['canonical_source_speaker_names'],sorted({r['speaker'] for r in source['turns']}))
            self.assertFalse(record['new_source_facts'])
            self.assertFalse(record['automatic_repair'])
            self.assertNotIn('gold',record)
            self.assertEqual(record['pronoun_status'],'unknown_for_all_source_speakers')

    def test_registry_byte_tampering_fails_even_before_projection(self):
        cfg=read(HOME/'CONFIG.json'); slot=cfg['ordering_slots'][0]
        with tempfile.TemporaryDirectory(prefix='v87_actor_registry_') as tmp:
            path=Path(tmp)/'registry.json'; path.write_text('{}')
            altered=deepcopy(cfg); altered['character_registry']['path']=str(path)
            with self.assertRaisesRegex(ValueError,'actor_reference_policy_or_registry_changed'):
                actor_reference.policy_for(altered,slot,read(slot['source_path']))

    def test_invented_gender_rejected_even_if_local_hashes_are_rebound(self):
        cfg=read(HOME/'CONFIG.json'); slot=cfg['ordering_slots'][0]
        with tempfile.TemporaryDirectory(prefix='v87_actor_registry_') as tmp:
            folder=Path(tmp); registry=read(cfg['character_registry']['path'])
            next(c for c in registry['characters'] if c['root_id']==slot['root_id'])['gender']['value']='invented'
            regpath=folder/'registry.json'; regpath.write_text(json.dumps(registry))
            policy=read(cfg['actor_reference_policy']['path']); policy['registry']=bind(regpath)
            polpath=folder/'policy.json'; polpath.write_text(json.dumps(policy))
            altered=deepcopy(cfg); altered['character_registry']=bind(regpath); altered['actor_reference_policy']=bind(polpath)
            with self.assertRaisesRegex(ValueError,'pilot_unknown_pronoun_scope_changed'):
                actor_reference.policy_for(altered,slot,read(slot['source_path']))

    def test_pronoun_draft_rejected_before_gold_gate_without_repair(self):
        from test_program_gold_v87 import fixture
        with tempfile.TemporaryDirectory(prefix='v87_actor_guard_') as tmp:
            folder=Path(tmp)
            source,plan,record,manifest,row=fixture(folder)
            row['items']['I3']='Actor said HE intended to help.'
            original=deepcopy(row)
            with patch.object(ordering_generation,'engine',return_value=SimpleNamespace(strict_json=json.loads)), \
                 patch.object(ordering_generation,'load_ordering_gate',side_effect=AssertionError('Pronoun draft reached downstream gate')):
                accepted,rejected=ordering_generation.validate_raw(json.dumps({'questions':[row]}),manifest,source,folder/'SOURCE.json',[])
            self.assertEqual(row,original)
            self.assertEqual(accepted,[])
            self.assertEqual(rejected,[{'candidate_id':row['candidate_id'],'reasons':['ordering_actor_reference_policy_forbidden_pronoun:I3:he']}])


if __name__=='__main__':
    unittest.main(verbosity=2)
