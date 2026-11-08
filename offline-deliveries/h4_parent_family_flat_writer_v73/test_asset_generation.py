"""Offline v73 lifecycle and program-bound writer contract; no real transport."""
from contextlib import ExitStack,nullcontext
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import asset_generation as a
import generation_driver as driver


def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.home=Path(self.temp.name);self.run=self.home/'assets/family1'
        self.source={'root_id':'root1','turns':[{'turn_ref':'SOURCE:EXACT:T001','speaker':'Alex','text':'Alex says he proposed Tuesday, probably at three.'}],
            'candidate_metadata':{'promoted':False}}
        self.source_path=self.home/'source.json';dump(self.source_path,self.source)
        self.parent={'candidate_id':'mother1','root_id':'root1','question_type':'multi_select','question':'What did Alex propose?',
            'options':{l:'Alex probably proposed Tuesday at three, fact '+l for l in 'ABCDEF'},'gold':list('ABCDEF'),
            'source_path':str(self.source_path),'source_sha256':a.sha(self.source_path),
            'option_evidence':{l:{'turn_ref':'SOURCE:EXACT:T001','speaker':'Alex','quote':'he proposed Tuesday','relation':'supports','rationale':'Reported proposal '+l} for l in 'ABCDEF'}}
        self.pool=self.home/'pool.json';dump(self.pool,{'questions':[self.parent]});slot={'slot_id':'family1','family_id':'family1','root_id':'root1',
            'parent_candidate_id':'mother1','parent_pool_path':str(self.pool),'history_path':str(self.source_path),'source_path':str(self.source_path),
            'source_sha256':a.sha(self.source_path),'history_sha256':a.sha(self.source_path),'prior_pool_paths':[]}
        self.cfg={'training_allowed':False,'max_http':4,'max_supplemental_requests_per_family':1,'max_questions':10,'max_calls_per_model':180,
            'slots':[slot,{**slot,'slot_id':'family2','family_id':'family2','root_id':'root2'}]}
        self.config=self.home/'CONFIG.json';dump(self.config,self.cfg)
        self.response={'root_id':'root1','family_id':'family1','replacements':[{'fact_id':fid,
            'replacement_proposition':'Alex probably proposed Wednesday at three, fact '+l,
            'unavailable_reason':None} for fid,l in zip(a.FACT_IDS,'ABCDEF')]}
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        fb={'source_lock':{'status':'EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY','merge_current_generation_feedback':False,
            'model_identity_source':{'sha256':a.v15.evaluator.TRAINED_AUDIT_SHA256}},'candidate_history':[]}
        def freeze(root,path,data):
            p=Path(path)/'source_lock.json';dump(p,data['source_lock']);return data,[p]
        self.stack.enter_context(patch.object(a.source_adapter,'feedback_data',return_value=fb))
        self.stack.enter_context(patch.object(a.source_adapter,'freeze_feedback',side_effect=freeze))
        self.stack.enter_context(patch.object(a.source_adapter,'prior_candidates',return_value=([],[])))
        self.stack.enter_context(patch.object(a,'_dependencies',return_value=({},set())))
        def payload(run,root,intro,ctx,cap):return {'model':a.legacy.MODEL,'max_tokens':cap,'messages':[{'role':'user','content':intro+'INPUT_JSON\n'+json.dumps(ctx)+'\nFROZEN_FEEDBACK'}]},{'frozen_previous_round':True}
        self.stack.enter_context(patch.object(a.legacy.legacy,'payload_with_feedback',side_effect=payload))
        self.stack.enter_context(patch.object(a.qa_transport,'shared_lock',return_value=nullcontext()))
        self.stack.enter_context(patch.object(a.base,'transport',side_effect=AssertionError('Unexpected real transport')))
        a.prepare(self.config,'family1',run=self.run);self.m=a.read(self.run/'MANIFEST.json')

    def accepted(self,obj=None):return a.validate_assets(json.dumps(self.response if obj is None else obj),self.m,self.source,self.parent)
    def fake_generate(self,rid='initial',obj=None,mode='valid'):
        response=deepcopy(self.response if obj is None else obj);calls=[]
        class Caller:
            def __init__(inner,run):inner.run=Path(run)
            def call(inner,label,payload,proof):
                calls.append(label);f=inner.run/'requests'/('001-'+label);a.write(f/'REQUEST.json',payload);a.write(f/'PROOF.json',proof)
                if mode!='capacity':a.write(f/'HTTP_ATTEMPT.json',{'actual_payload_equals_frozen':True,'payload_sha256':a.base.digest(payload),'transport_retries':0,'http_attempt_number':1})
                meta={'returned_model':a.legacy.MODEL,'finish_reason':'stop','http_status':200};raw=json.dumps(response)
                if mode!='capacity':a.write(f/'RESPONSE.json',{'text':raw,'metadata':meta})
                if mode!='valid':a.write(f/'ERROR.json',{'error_type':mode});return None,f,meta
                return raw,f,meta
        with patch.object(a.qa_transport,'Caller',Caller):r=a.generate_round(self.run,rid)
        self.assertEqual(len(calls),1);a.verify_round(self.run,rid,complete=True);return r

    def review(self,rid='initial',approved=None,parent_valid=True):
        f=self.run/'rounds'/rid;pub=a.read(f/'PUBLIC_ASSETS.json');approved=set(a.FACT_IDS if approved is None else approved);anchor={'turn_ref':'SOURCE:EXACT:T001','speaker':'Alex','quote':'he proposed Tuesday'}
        j={'version':'synthetic_round_judgments','created_at_utc':a.v15.now(),'root_id':a.read(self.run/'MANIFEST.json')['root_id'],'family_id':a.read(self.run/'MANIFEST.json')['family_id'],
            'source_sha256':a.sha(self.run/'SOURCE.json'),'public_assets_sha256':a.sha(f/'PUBLIC_ASSETS.json'),'complete_source_turns_read':1,
            'stem_valid':True,'parent_all_true_provenance_known':True,'new_variant_private_artifacts_read_before_save':False,'solver_scores_used':False,'current_solver_scores_read':False,
            'parents':[{'fact_id':v['fact_id'],'parent_label':v['parent_label'],'proposition':v['proposition'],'relation':'supports' if parent_valid else 'unknown',
                'scope_valid':True,'fact_distinct':True,'reason':'Source fixes this distinct reported fact.','citations':[anchor]} for v in pub['parent_facts']],
            'versions':[{'version_id':v['version_id'],'proposition':v['proposition'],'relation':'contradicts' if v['fact_id'] in approved else 'unknown',
                'scope_valid':True,'fact_distinct':True,'reason':'Source fixes the incompatible attribute.' if v['fact_id'] in approved else 'Activities may coexist.','citations':[anchor]} for v in pub['versions']]}
        a.write(f/'SOURCE_JUDGMENTS.json',j);return a.finish_review(self.run,rid)

    def test_exact_program_anchor_and_parent_bytes(self):
        data,errors=self.accepted();self.assertFalse(errors);assets=a.trusted_assets(self.m,self.parent,data)
        self.assertEqual((self.run/'SOURCE.json').read_bytes(),self.source_path.read_bytes())
        for f in assets['facts']:
            self.assertEqual(f['contradicted']['anchor'],f['supported']['anchor']);self.assertEqual(f['supported']['proposition'],self.parent['options'][f['parent_label']])
        self.assertIn('generation_round',a.read(self.run/'rounds/initial/INITIAL_REQUEST.json')['messages'][0]['content'])
    def test_copy_true_including_normalization_rejected_per_row(self):
        for text in [self.parent['options']['A'],self.parent['options']['A'].upper(),self.parent['options']['A'].replace(' ','  '),self.parent['options']['A']+'!!!']:
            obj=deepcopy(self.response);obj['replacements'][0]['replacement_proposition']=text;data,errors=self.accepted(obj)
            self.assertEqual(errors[0]['error'],'copied_supported_parent_proposition');self.assertIsNone(data['replacements'][0]['contradicted']);self.assertIsNotNone(data['replacements'][1]['contradicted'])
    def test_flat_writer_contract_rejects_legacy_and_analysis_fields(self):
        legacy=deepcopy(self.response)
        legacy['replacements'][0]={'fact_id':'F01','contradicted':{'proposition':'A false claim.','relation':'contradicts','rationale':'Old analysis.'},'contradiction_unavailable_reason':None}
        self.assertEqual(self.accepted(legacy)[1],[{'fact_id':'F01','error':'replacement_schema'}])
        for key in ['relation','rationale','proposition']:
            obj=deepcopy(self.response);obj['replacements'][0][key]='Hidden mutation'
            self.assertEqual(self.accepted(obj)[1],[{'fact_id':'F01','error':'replacement_schema'}])
        data,errors=self.accepted();self.assertFalse(errors)
        self.assertEqual(data['replacements'][0]['contradicted']['rationale'],a.UNAUDITED_RATIONALE)
        self.assertEqual(data['replacements'][0]['contradicted']['proposition'],self.response['replacements'][0]['replacement_proposition'])
    def test_flat_null_exclusivity_and_invalid_proposition_types(self):
        obj=deepcopy(self.response);obj['replacements'][0].update(replacement_proposition=None,unavailable_reason='No refutable alternative.')
        data,errors=self.accepted(obj);self.assertFalse(errors);self.assertIsNone(data['replacements'][0]['contradicted'])
        for proposition,reason,expected in [(None,None,'null_alternative_reason'),(None,'  ','null_alternative_reason'),('A false claim.','Reason','nonnull_alternative_reason'),('',None,'replacement_proposition_schema'),(True,None,'replacement_proposition_schema'),({'proposition':'hidden'},None,'replacement_proposition_schema')]:
            obj=deepcopy(self.response);obj['replacements'][0].update(replacement_proposition=proposition,unavailable_reason=reason)
            self.assertEqual(self.accepted(obj)[1],[{'fact_id':'F01','error':expected}])
    def test_model_citations_forbidden_not_repaired(self):
        obj=deepcopy(self.response);obj['replacements'][0]['anchor']={'turn_ref':'alias'};data,err=self.accepted(obj)
        self.assertTrue(err);self.assertIsNone(data['replacements'][0]['contradicted'])
    def test_pronouns_rejected_quotes_exempt(self):
        for token in ['he','SHE',"she’s",'herself']:
            obj=deepcopy(self.response);obj['replacements'][0]['replacement_proposition']='Alex says '+token+' changed.';data,err=self.accepted(obj);self.assertTrue(err)
        self.assertFalse(self.accepted()[1])
    def test_strict_json_and_target_order(self):
        for raw in ['{"x":NaN}','{"x":1,"x":2}','```json\n{}\n```']:self.assertIsNone(a.validate_assets(raw,self.m,self.source,self.parent)[0])
        obj=deepcopy(self.response);obj['replacements'].reverse();self.assertIsNone(self.accepted(obj)[0])
    def test_partial_mechanical_failure_retains_others(self):
        obj=deepcopy(self.response);obj['replacements'][0]['replacement_proposition']=self.parent['options']['A']
        self.assertEqual(self.fake_generate(obj=obj)['counterversion_count'],5);self.review();a.finalize(self.run);a.verify(self.run,complete=True)
    def test_all_invalid_and_transport_error_still_reviewable_no_retry(self):
        self.assertEqual(self.fake_generate(mode='capacity')['http_attempts'],0);self.review(approved=[])
        r=a.finalize(self.run);self.assertEqual(r['counterversion_count'],0);self.assertEqual(r['http_attempts'],0)
    def test_full_initial_review_finalization(self):
        self.fake_generate();self.review();c=a.finalize(self.run);self.assertEqual(c['counterversion_count'],6)
        j=a.read(self.run/'SOURCE_JUDGMENTS.json');self.assertTrue(j['derived_from_rounds']);self.assertTrue(j['new_variant_private_artifacts_read_before_save'])
        with self.assertRaises(ValueError):a.prepare_supplement(self.run)
    def test_missing_review_cannot_supplement_or_finalize(self):
        self.fake_generate()
        with self.assertRaises(FileNotFoundError):a.prepare_supplement(self.run)
        with self.assertRaises(FileNotFoundError):a.finalize(self.run)
    def test_four_approved_blocks_unnecessary_supplement(self):
        self.fake_generate();self.review(approved=a.FACT_IDS[:4])
        with self.assertRaises(ValueError):a.prepare_supplement(self.run)
    def test_failed_parent_blocks_supplement(self):
        self.fake_generate();self.review(approved=[],parent_valid=False)
        with self.assertRaises(ValueError):a.prepare_supplement(self.run)
    def test_source_first_flags_and_future_timestamp_rejected(self):
        self.fake_generate();self.review();p=self.run/'rounds/initial/SOURCE_JUDGMENTS.json';j=a.read(p);j['new_variant_private_artifacts_read_before_save']=True;dump(p,j)
        with self.assertRaises(ValueError):a.reviewed_round(self.run,'initial')
    def test_full_supplement_preserves_initial_acceptance_and_provenance(self):
        self.fake_generate();self.review(approved=['F01','F02']);initial=a.read(self.run/'rounds/initial/PUBLIC_ASSETS.json')
        spec=a.prepare_supplement(self.run);self.assertEqual(spec['target_fact_ids'],['F03','F04','F05','F06'])
        obj=deepcopy(self.response);obj['replacements']=obj['replacements'][2:]
        self.fake_generate('supplement',obj);self.review('supplement',approved=['F03','F04','F05']);c=a.finalize(self.run)
        self.assertEqual(c['http_attempts'],2);self.assertEqual(c['bounded_supplements'],1);self.assertEqual(c['counterversion_count'],5)
        a.verify(self.run,complete=True);assets=a.read(self.run/'ASSETS.json');self.assertEqual(assets['facts'][0]['origin']['round_id'],'initial')
        accepted=a.read(self.run/'ROUND_PROVENANCE.json')['accepted_versions'];self.assertEqual(len(accepted),5)
        self.assertEqual(next(v['version_id'] for v in accepted if v['fact_id']=='F01'),next(v['version_id'] for v in initial['versions'] if v['fact_id']=='F01'))
        with self.assertRaises(ValueError):a.prepare_supplement(self.run)
    def test_supplement_wrong_targets_or_second_call_rejected(self):
        self.fake_generate();self.review(approved=['F01']);a.prepare_supplement(self.run)
        obj=deepcopy(self.response);self.fake_generate('supplement',obj);self.review('supplement',approved=[])
        with self.assertRaises(ValueError):a.generate_round(self.run,'supplement')
        self.assertEqual(a.read(self.run/'rounds/supplement/GENERATION.COMPLETE.json')['counterversion_count'],0)
    def test_tamper_source_request_raw_final_projection_detected(self):
        self.fake_generate();self.review();a.finalize(self.run)
        for p in [self.run/'SOURCE.json',self.run/'rounds/initial/INITIAL_REQUEST.json',next(self.run.glob('rounds/initial/requests/*/RESPONSE.json')),self.run/'ROUND_PROVENANCE.json',self.run/'ASSETS.json']:
            old=p.read_bytes();obj=a.read(p);obj['tamper']=True;dump(p,obj)
            with self.assertRaises(ValueError):a.verify(self.run,complete=True)
            p.write_bytes(old)
    def test_driver_prepare_supplement_finalize_compose_report_integration(self):
        import compose
        import finalize_report as report
        from types import SimpleNamespace
        home=self.home/'driver_home';home.mkdir();cfg=deepcopy(self.cfg)
        cfg.update(baseline_pool=str(self.pool),target_distribution={str(k):2 for k in (2,3,4,5)})
        source2=deepcopy(self.source);source2['root_id']='root2';path2=home/'source2.json';dump(path2,source2)
        parent2=deepcopy(self.parent);parent2.update(candidate_id='mother2',root_id='root2',source_path=str(path2),source_sha256=a.sha(path2))
        pool2=home/'parent2.json';dump(pool2,{'questions':[parent2]})
        cfg['slots'][1].update(parent_candidate_id='mother2',parent_pool_path=str(pool2),history_path=str(path2),source_path=str(path2),source_sha256=a.sha(path2),history_sha256=a.sha(path2))
        dump(home/'CONFIG.json',cfg)
        profile={'request_parameters':{}};seen=[]
        def capacity(profile,tokenizer,messages,limit):
            seen.append(messages[0]['content']);return {'status':'PASS','required_context_tokens':123}
        with patch.object(a,'_dependencies',return_value=(profile,set())),patch.object(a.base,'transport',return_value=(None,None,lambda x:x,capacity,SimpleNamespace(tokenizer_from=lambda p:None))):
            plan=driver.prepare(home)
        self.assertEqual(len(plan['slots']),2);self.assertTrue(all('generation_round' in text for text in seen))
        for sid in ['family1','family2']:
            self.run=home/'assets'/sid;self.m=a.read(self.run/'MANIFEST.json');obj=deepcopy(self.response);obj['root_id']=self.m['root_id'];obj['family_id']=sid
            self.fake_generate(obj=obj);self.review(approved=['F01','F02'])
            a.prepare_supplement(self.run);obj['replacements']=obj['replacements'][2:];self.fake_generate('supplement',obj)
            self.review('supplement',approved=['F03','F04','F05','F06'])
        complete=driver.finalize(home);self.assertEqual(complete['http_attempts'],4);self.assertEqual(complete['bounded_supplements'],2)
        output=compose.compose(home);self.assertEqual(output['candidate_count'],10)
        self.assertEqual({len(q['gold']) for q in output['questions']},{2,3,4,5,6})
        records,families=report.generation_records(home,cfg);self.assertEqual(len(records),4);self.assertEqual(sum(f['accepted_false_versions'] for f in families),12)
        self.assertEqual(report.generation_round_statistics(records,families)['by_round']['supplement']['families_attempted'],2)

    def test_budgets_and_clock(self):
        self.assertEqual(a.qa_transport.GAP,35.0)
        cfg=deepcopy(self.cfg);cfg['max_http']=5
        with self.assertRaises(ValueError):a.get_slot(cfg,'family1')

if __name__=='__main__':unittest.main()
