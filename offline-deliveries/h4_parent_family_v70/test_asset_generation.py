"""Offline contract, immutable projection and request-replay checks; no HTTP."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import asset_generation as a
import generation_driver as driver


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.home=Path(self.temp.name);self.run=self.home/'assets'/'family1'
        self.source={'root_id':'root1','turns':[{'turn_ref':'root1:P01:T001','speaker':'Alex','text':'Alex says he proposed Tuesday, probably at three.'}],
            'source_admission_status':'experimental_candidate_history','candidate_extension':{'promoted':False}}
        self.source_path=self.home/'exact-source.json';dump(self.source_path,self.source)
        self.parent={'candidate_id':'mother1','root_id':'root1','question_type':'multi_select','question':'What did Alex propose?',
            'options':{label:'Alex probably proposed Tuesday at three, fact '+label for label in 'ABCDEF'},'gold':list('ABCDEF'),
            'source_path':str(self.source_path),'source_sha256':a.sha(self.source_path),
            'option_evidence':{label:{'turn_ref':'root1:P01:T001','speaker':'Alex','quote':'he proposed Tuesday',
                'relation':'supports','rationale':'Exact reported proposal '+label} for label in 'ABCDEF'}}
        self.pool=self.home/'pool.json';dump(self.pool,{'questions':[self.parent]})
        slot={'slot_id':'family1','family_id':'family1','root_id':'root1','parent_candidate_id':'mother1',
            'parent_pool_path':str(self.pool),'history_path':str(self.source_path),'history_sha256':a.sha(self.source_path),
            'source_path':str(self.source_path),'source_sha256':a.sha(self.source_path),'prior_pool_paths':[]}
        self.cfg={'training_allowed':False,'max_http':2,'max_questions':10,'max_calls_per_model':180,
            'slots':[slot,{**slot,'slot_id':'family2','family_id':'family2','root_id':'root2'}]}
        self.config=self.home/'CONFIG.json';dump(self.config,self.cfg)
        self.response={'root_id':'root1','family_id':'family1','replacements':[
            {'fact_id':fid,'anchor':{'turn_ref':'root1:P01:T001','speaker':'Alex','quote':'he proposed Tuesday'},
             'contradicted':{'proposition':'Alex probably proposed Wednesday at three, fact '+label,'relation':'contradicts','rationale':'Tuesday conflicts with Wednesday in this proposal.'},
             'contradiction_unavailable_reason':None} for fid,label in zip(a.FACT_IDS,'ABCDEF')]}
        self.stack=ExitStack();self.addCleanup(self.stack.close);self.addCleanup(self.temp.cleanup)
        fb={'source_lock':{'status':'EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY','merge_current_generation_feedback':False,
            'model_identity_source':{'sha256':a.v15.evaluator.TRAINED_AUDIT_SHA256}},'candidate_history':[]}
        def freeze(root,path,data):
            p=Path(path)/'source_lock.json';dump(p,data['source_lock']);return data,[p]
        self.stack.enter_context(patch.object(a.source_adapter,'feedback_data',return_value=fb))
        self.stack.enter_context(patch.object(a.source_adapter,'freeze_feedback',side_effect=freeze))
        self.stack.enter_context(patch.object(a.source_adapter,'prior_candidates',return_value=([],[])))
        self.stack.enter_context(patch.object(a,'_dependencies',return_value=({},set())))
        def payload(run,root,intro,context,cap):
            return {'model':a.legacy.MODEL,'max_tokens':cap,'messages':[{'role':'user','content':intro+'INPUT_JSON\n'+json.dumps(context,ensure_ascii=False)+'\nFROZEN OLD FEEDBACK'}]}, {'frozen_previous_round':True}
        self.stack.enter_context(patch.object(a.legacy.legacy,'payload_with_feedback',side_effect=payload))
        self.stack.enter_context(patch.object(a.qa_transport,'shared_lock',return_value=nullcontext()))
        self.stack.enter_context(patch.object(a.base,'transport',side_effect=AssertionError('Unexpected transport access in offline test')))
        a.prepare(self.config,'family1',run=self.run);self.m=a.read(self.run/'MANIFEST.json')

    def accepted(self,data=None):return a.validate_assets(json.dumps(data or self.response),self.m,self.source)
    def fake_generate(self,data=None,mode='valid'):
        response=deepcopy(self.response if data is None else data);calls=[]
        class Caller:
            def __init__(inner,run):inner.run=Path(run)
            def call(inner,label,payload,proof):
                calls.append(label);folder=inner.run/'requests'/('001-'+label)
                a.write(folder/'REQUEST.json',payload);a.write(folder/'PROOF.json',proof)
                if mode!='capacity':a.write(folder/'HTTP_ATTEMPT.json',{'actual_payload_equals_frozen':True,
                    'payload_sha256':a.base.digest(payload),'transport_retries':0,'http_attempt_number':1})
                meta={'returned_model':a.legacy.MODEL,'finish_reason':'length' if mode=='incomplete' else 'stop','http_status':200}
                raw=json.dumps(response)
                if mode!='capacity':a.write(folder/'RESPONSE.json',{'text':raw,'metadata':meta})
                if mode!='valid':
                    a.write(folder/'ERROR.json',{'error_type':mode,'no_automatic_retry':True});return None,folder,meta
                return raw,folder,meta
        with patch.object(a.qa_transport,'Caller',Caller):result=a.generate(self.run)
        self.assertEqual(len(calls),1);a.verify(self.run,complete=True);return result

    def test_full_source_parent_metadata_and_public_context(self):
        self.assertEqual((self.run/'SOURCE.json').read_bytes(),self.source_path.read_bytes())
        self.assertEqual(a.read(self.run/'PARENT.json'),self.parent)
        ctx=a.context_for(self.run);self.assertEqual(ctx['source_turns'],self.source['turns'])
        self.assertEqual(ctx['source_metadata']['candidate_extension'],{'promoted':False})
        self.assertEqual(ctx['required_public_stem'],self.parent['question'])
        self.assertNotIn('gold',ctx);self.assertNotIn('parent_candidate_id',ctx)
        self.assertNotIn('target_count',json.dumps(ctx));self.assertEqual(self.m['budget'],{'http_attempts_max':1})
    def test_exact_quotes_can_contain_pronouns(self):self.assertEqual(self.accepted()[1],[])
    def test_all_gendered_tokens_and_contractions_rejected(self):
        for token in ('he','SHE','him','her','his','hers','himself','herself',"he's",'she’s','her-own'):
            data=deepcopy(self.response);data['replacements'][0]['contradicted']['proposition']='Alex said '+token+' proposed Wednesday.'
            self.assertIsNone(self.accepted(data)[0],token)
        for token in ('they','them','their','theater','shelf','higher'):
            data=deepcopy(self.response);data['replacements'][0]['contradicted']['proposition']='Alex said '+token+' Wednesday.'
            self.assertEqual(self.accepted(data)[1],[],token)
    def test_fixed_six_identity_order_no_supported_rewrite(self):
        for mutate in (lambda d:d['replacements'].pop(),lambda d:d['replacements'].append(d['replacements'][0]),
            lambda d:d['replacements'].reverse(),lambda d:d['replacements'][0].update(supported={'proposition':'changed'}),
            lambda d:d.update(public_stem='changed'),lambda d:d.update(family_id='other')):
            data=deepcopy(self.response);mutate(data);self.assertIsNone(self.accepted(data)[0])
    def test_null_requires_anchor_null_and_specific_reason(self):
        data=deepcopy(self.response);row=data['replacements'][0];row.update(contradicted=None,anchor=None,contradiction_unavailable_reason='No explicitly exclusive alternative for the reported fact.')
        self.assertEqual(self.accepted(data)[1],[])
        for val in (None,'',' '):
            row['contradiction_unavailable_reason']=val;self.assertIsNone(self.accepted(data)[0])
    def test_exact_anchor_no_alias_or_speaker_or_quote_repair(self):
        for key,value in (('turn_ref','P01:T001'),('speaker','Sam'),('quote','Tuesday proposal')):
            data=deepcopy(self.response);data['replacements'][0]['anchor'][key]=value;self.assertIsNone(self.accepted(data)[0])
    def test_duplicate_keys_nonfinite_and_non_json_rejected(self):
        for raw in ('{"root_id":"root1","root_id":"root1"}','{"x":NaN}','{"x":Infinity}','```json\n{}\n```'):
            self.assertIsNone(a.validate_assets(raw,self.m,self.source)[0])
    def test_success_projection_preserves_original_and_no_private_leak(self):
        result=self.fake_generate();self.assertEqual(result['status'],'ASSETS_READY');self.assertEqual(result['counterversion_count'],6)
        assets=a.read(self.run/'ASSETS.json');public=a.read(self.run/'PUBLIC_ASSETS.json');mapping=a.read(self.run/'VERSION_MAP.json')
        for label,f in zip('ABCDEF',assets['facts']):
            self.assertEqual(f['supported']['proposition'],self.parent['options'][label]);self.assertEqual(f['supported']['rationale'],self.parent['option_evidence'][label]['rationale'])
            self.assertEqual(f['contradicted']['anchor'],f['anchor'])
        self.assertEqual(assets['public_stem'],self.parent['question']);self.assertEqual(a.read(self.run/'PARENT.json'),self.parent)
        self.assertTrue(all(set(v)=={'version_id','fact_id','proposition'} for v in public['versions']))
        self.assertTrue(all(set(v)=={'fact_id','parent_label','proposition'} for v in public['parent_facts']))
        self.assertEqual(mapping['public_assets_sha256'],a.sha(self.run/'PUBLIC_ASSETS.json'))
        self.assertEqual(len({v['version_id'] for v in mapping['versions']}),6)
        with self.assertRaises(ValueError):a.generate(self.run)
    def test_invalid_response_excluded_without_retry_or_rewrite(self):
        data=deepcopy(self.response);data['replacements'][0]['contradicted']['proposition']='She proposed Wednesday.'
        result=self.fake_generate(data);self.assertEqual(result['status'],'NO_VALID_ASSETS');self.assertEqual(result['http_attempts'],1)
        self.assertFalse((self.run/'ASSETS.json').exists())
    def test_incomplete_transport_response_excluded(self):
        self.assertEqual(self.fake_generate(mode='incomplete')['status'],'NO_VALID_ASSETS');self.assertFalse((self.run/'ASSETS.json').exists())
    def test_capacity_failure_has_zero_http_no_retry(self):
        result=self.fake_generate(mode='capacity');self.assertEqual(result['http_attempts'],0);self.assertEqual(result['requests_prepared'],1)
    def test_null_all_is_valid_shortage_no_retry(self):
        data=deepcopy(self.response)
        for row in data['replacements']:row.update(contradicted=None,anchor=None,contradiction_unavailable_reason='The source does not explicitly refute a scoped alternative.')
        result=self.fake_generate(data);self.assertEqual(result['status'],'ASSETS_READY');self.assertEqual(result['counterversion_count'],0)
    def test_frozen_source_or_parent_or_request_tamper_detected(self):
        for filename in ('SOURCE.json','PARENT.json','INITIAL_REQUEST.json','INITIAL_PROOF.json'):
            path=self.run/filename;original=path.read_bytes();data=a.read(path);data['tamper']=True;dump(path,data)
            with self.assertRaises(ValueError):a.verify(self.run)
            path.write_bytes(original)
    def test_raw_response_projection_and_completion_tamper_detected(self):
        self.fake_generate()
        paths=[next((self.run/'requests').glob('*/RESPONSE.json')),self.run/'ASSETS.json',self.run/'PUBLIC_ASSETS.json',self.run/'VERSION_MAP.json',self.run/'GENERATION.COMPLETE.json']
        for path in paths:
            original=path.read_bytes();data=a.read(path)
            if path.name=='GENERATION.COMPLETE.json':data['mechanical_repairs']=1
            else:data['tamper']=True
            dump(path,data)
            with self.assertRaises(ValueError):a.verify(self.run,complete=True)
            path.write_bytes(original)
    def test_source_parent_pool_bindings_required(self):
        source=deepcopy(self.source);source['turns'][0]['text']='changed';dump(self.source_path,source)
        with self.assertRaises(ValueError):a.parent_source(self.m['slot'])
    def test_two_distinct_roots_and_global_limits(self):
        for key,value in (('max_http',3),('max_questions',11),('max_calls_per_model',181),('training_allowed',True)):
            cfg=deepcopy(self.cfg);cfg[key]=value
            with self.assertRaises(ValueError):a.get_slot(cfg,'family1')
        cfg=deepcopy(self.cfg);cfg['slots'][1]['root_id']='root1'
        with self.assertRaises(ValueError):a.get_slot(cfg,'family1')
    def test_unexpected_second_request_rejected(self):
        self.fake_generate();folder=self.run/'requests'/'002-forbidden';a.write(folder/'REQUEST.json',a.read(self.run/'INITIAL_REQUEST.json'))
        with self.assertRaises(ValueError):a.verify(self.run)
    def test_completion_clock_unchanged(self):self.assertEqual(a.qa_transport.GAP,35.0)

if __name__=='__main__':unittest.main()
