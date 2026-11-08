"""Offline whole-schedule integration and frozen-source guards."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from ..backends import BackendResponse
from ..core import atomic_json
from ..h4_qa_mining_probe_v1.intent_miner import build_intent_prompt
from ..h4_r02_feedback_transfer_p2_2_v1.test_generation_hooks import expanded_fixture, intents_fixture
from . import run as r
from .runtime import state_paths, GuardrailStop

class RunTests(unittest.TestCase):
    def test_six_groups_share_miner_and_resume_without_send(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);expanded=expanded_fixture();cards={'observation_cards':[]};history={'records':[]}
            atomic_json(run/'audit/frozen_manifest.json',{'fixture':True})
            atomic_json(run/'audit/preflight_report.json',{'status':'PASS'})
            atomic_json(run/'schedule.json',{'version':r.VERSION,'schedule':list(r.SCHEDULE)})
            pairs=[]
            for rid in r.ROOTS:
                for name,value in [('expanded_dialogue_clean.json',expanded),('observation_cards.json',cards),('history_only.json',history)]:atomic_json(run/'inputs/shared'/rid/name,value)
            for row in r.SCHEDULE:
                arm,rid=row['arm_id'],row['root_id'];spec=r.build_prompt(arm,build_intent_prompt(expanded),cards,history,expanded=expanded,root_id=rid)
                p=run/'policy_snapshot/prompts'/arm/rid/'miner_prompt.txt';p.parent.mkdir(parents=True);p.write_text(spec['prompt'])
                atomic_json(run/'inputs/arms'/arm/rid/'feedback_payload.json',r.feedback_payload(arm,cards,history,rid))
                pairs.append({**row,'prompt_sha256':spec['prompt_sha256']})
            (run/'pairing_manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in pairs))
            calls=[];token_checks=[]
            config=SimpleNamespace(model='Qwen/Qwen3-14B',base_url='fixture')
            class FakeClient:
                def __init__(self,config):self.config=config
                def complete(self,**kwargs):
                    calls.append(kwargs);raw=intents_fixture()
                    if 'P2-2 MINER BINDING V2 OUTPUT RULES:' in kwargs['prompt']:
                        raw['feedback_bindings']=[{'intent_id':i['intent_id'],'status':'NO_SUPPORTED_TRANSFER','source_feedback_ids':[],'scope':'','change_from_source':{'old_question_focus':'','feedback_basis':'','added_distinction':''}} for i in raw['intents']]
                    return BackendResponse(json.dumps(raw),'fixture',{'usage':{'prompt_tokens':1}})
            class FakeToken:
                def __init__(self,run):pass
                def check(self,prompt,max_output_tokens,request_id,**kwargs):token_checks.append(request_id)
                def verify_usage(self,request_id,usage):pass
            with patch.object(r,'verify_frozen'),patch.object(r,'configured',return_value=config),patch.object(r,'OpenAICompatibleChallengerClient',FakeClient),patch.object(r,'ObservedClient',side_effect=lambda delegate,*args:delegate),patch.object(r,'TokenPreflight',FakeToken),contextlib.redirect_stdout(io.StringIO()):
                result=r.live(run);again=r.live(run)
            self.assertEqual(len(calls),6);self.assertEqual(token_checks,[x['global_request_id'] for x in r.SCHEDULE]);self.assertEqual(len(state_paths(run)),6)
            self.assertEqual(result['groups'],again['groups'])
            self.assertTrue(all(x['valid_intents']==3 for x in result['groups']))
            for row in r.SCHEDULE:
                root=r.root_path(run,row['arm_id'],row['root_id']);self.assertTrue(r.verify_checkpoint(root));self.assertFalse((root/'candidate_pool.json').exists())
                self.assertEqual((root/'binding_validation.json').exists(),row['arm_id']=='C_P2_2_TRANSFER')

    def test_frozen_config_input_code_and_pairing_sha_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);files={}
            for name in ('input','code','config','source','preflight','pairing'):
                p=run/name;p.write_text(name);files[name]=p
            frozen={'version':r.VERSION,'schedule':list(r.SCHEDULE),'code':[r.record(files['code'])],
                'inputs':[r.record(files[k]) for k in ('input','pairing')],'sources':[r.record(files['source'])],
                'config':r.record(files['config']),'preflight':r.record(files['preflight'])}
            atomic_json(run/'audit/frozen_manifest.json',frozen)
            with patch.object(r,'code_identity',return_value=frozen['code']):
                r.verify_frozen(run)
                for name,p in files.items():
                    p.write_text('changed')
                    with self.subTest(name=name),self.assertRaises(GuardrailStop):r.verify_frozen(run)
                    p.write_text(name)
            with patch.object(r,'code_identity',return_value=[]):
                with self.assertRaises(GuardrailStop):r.verify_frozen(run)

if __name__=='__main__':unittest.main()
