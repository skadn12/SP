"""Offline v79 lifecycle and program-bound writer contract; no real transport."""
from contextlib import ExitStack,nullcontext
from copy import deepcopy
from datetime import datetime,timedelta,timezone
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
            'replacement_proposition':'Alex probably proposed Wednesday at three, fact '+l} for fid,l in zip(a.FACT_IDS,'ABCDEF')]}
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

    def judgment(self,rid='initial',approved=None,parent_valid=True):
        f=self.run/'rounds'/rid;pub=a.read(f/'PUBLIC_ASSETS.json');approved=set(a.FACT_IDS if approved is None else approved);anchor={'turn_ref':'SOURCE:EXACT:T001','speaker':'Alex','quote':'he proposed Tuesday'}
        j={'version':'synthetic_round_judgments','created_at_utc':a.v15.now(),'root_id':a.read(self.run/'MANIFEST.json')['root_id'],'family_id':a.read(self.run/'MANIFEST.json')['family_id'],
            'round_id':rid,'reviewer':'offline_fixture_reviewer','review_schema_version':a.review_schema.SCHEMA_VERSION,
            'source_sha256':a.sha(self.run/'SOURCE.json'),'public_assets_sha256':a.sha(f/'PUBLIC_ASSETS.json'),'complete_source_turns_read':1,
            'full_source_read':True,'stem_valid':True,'parent_all_true_provenance_known':True,'parent_gold_known_before_review':True,
            'fresh_blind_parent_review':False,'new_variant_private_artifacts_read_before_save':False,'solver_scores_used':False,'current_solver_scores_read':False,
            'parents':[{'fact_id':v['fact_id'],'parent_label':v['parent_label'],'proposition':v['proposition'],'relation':'supports' if parent_valid else 'unknown',
                'scope_valid':True,'fact_distinct':True,'reason':'Source fixes this distinct reported fact.','citations':[anchor]} for v in pub['parent_facts']],
            'versions':[{'version_id':v['version_id'],'proposition':v['proposition'],'relation':'contradicts' if v['fact_id'] in approved else 'unknown',
                'scope_valid':True,'fact_distinct':True,'reason':'Source fixes the incompatible attribute.' if v['fact_id'] in approved else 'Activities may coexist.','citations':[anchor]} for v in pub['versions']]}
        return j
    def review(self,rid='initial',approved=None,parent_valid=True):
        f=self.run/'rounds'/rid;j=self.judgment(rid,approved,parent_valid)
        a.write(f/'SOURCE_JUDGMENTS.json',j);return a.finish_review(self.run,rid)

    def assert_rejected_before_private_reads(self,j,rid='initial',actions=None):
        folder=self.run/'rounds'/rid;dump(folder/'SOURCE_JUDGMENTS.json',j)
        if actions is None:actions=[lambda:a.finish_review(self.run,rid),lambda:a.private_crosscheck(self.run,rid,j),lambda:a.reviewed_round(self.run,rid)]
        real_read=a.read
        for action in actions:
            private_reads=[]
            def read(path):
                if Path(path).name in ('ASSETS.json','VERSION_MAP.json'):private_reads.append(str(path))
                return real_read(path)
            with patch.object(a,'read',side_effect=read),self.assertRaises(ValueError):action()
            self.assertEqual(private_reads,[])
        self.assertFalse((folder/'ASSET_REVIEW.json').exists());self.assertFalse((folder/'REVIEW_RECEIPT.json').exists())
        self.assertFalse((self.run/'GENERATION.COMPLETE.json').exists())

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
        for key in ['relation','rationale','proposition','unavailable_reason']:
            obj=deepcopy(self.response);obj['replacements'][0][key]='Hidden mutation'
            self.assertEqual(self.accepted(obj)[1],[{'fact_id':'F01','error':'replacement_schema'}])
        data,errors=self.accepted();self.assertFalse(errors)
        self.assertEqual(data['replacements'][0]['contradicted']['rationale'],a.UNAUDITED_RATIONALE)
        self.assertEqual(data['replacements'][0]['contradicted']['proposition'],self.response['replacements'][0]['replacement_proposition'])
    def test_sentence_only_null_and_invalid_proposition_types(self):
        obj=deepcopy(self.response);obj['replacements'][0]['replacement_proposition']=None
        data,errors=self.accepted(obj);self.assertFalse(errors);self.assertIsNone(data['replacements'][0]['contradicted'])
        self.assertEqual(data['replacements'][0]['contradiction_unavailable_reason'],'Writer supplied no replacement; no semantic claim was made.')
        for proposition in ['', '  ', True, 5, {'proposition':'hidden'}, ['hidden']]:
            obj=deepcopy(self.response);obj['replacements'][0]['replacement_proposition']=proposition
            self.assertEqual(self.accepted(obj)[1],[{'fact_id':'F01','error':'replacement_proposition_schema'}])
        for reason in [None,'No refutable alternative.']:
            obj=deepcopy(self.response);obj['replacements'][0].update(replacement_proposition=None,unavailable_reason=reason)
            self.assertEqual(self.accepted(obj)[1],[{'fact_id':'F01','error':'replacement_schema'}])
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
        self.assertIsNone(j['created_at_utc']);self.assertFalse(j['aggregate_is_new_independent_blind_review'])
        self.assertEqual(j['round_id'],'final');self.assertEqual(j['review_schema_version'],a.review_schema.SCHEMA_VERSION)
        receipt=a.read(self.run/'REVIEW_RECEIPT.json');self.assertTrue(receipt['derived_from_rounds'])
        self.assertFalse(receipt['aggregate_is_new_independent_blind_review']);self.assertIn(a.binding(a.schema_path()),receipt['files'])
        for field in a.review_schema.COMMON_DISCLOSURES:self.assertIs(receipt[field],j[field])
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
    def test_each_mandatory_disclosure_missing_rejected_before_private_reads(self):
        self.fake_generate();original=self.judgment()
        for field in [*a.review_schema.COMMON_DISCLOSURES,'review_schema_version','full_source_read',
                'complete_source_turns_read','new_variant_private_artifacts_read_before_save']:
            with self.subTest(field=field):
                j=deepcopy(original);del j[field];self.assert_rejected_before_private_reads(j)

    def test_disclosure_bool_types_and_current_solver_reads_rejected_before_private_reads(self):
        self.fake_generate();original=self.judgment()
        for field in [*a.review_schema.COMMON_DISCLOSURES,'full_source_read','new_variant_private_artifacts_read_before_save']:
            for value in (0,1,'false',None):
                with self.subTest(field=field,value=value):
                    j=deepcopy(original);j[field]=value;self.assert_rejected_before_private_reads(j)
        for field,value in [('current_solver_scores_read',True),('solver_scores_used',True),
                ('new_variant_private_artifacts_read_before_save',True),('full_source_read',False),
                ('fresh_blind_parent_review',True),('parent_gold_known_before_review',False),
                ('parent_all_true_provenance_known',False),('complete_source_turns_read',True),
                ('complete_source_turns_read',0),('derived_from_rounds',True)]:
            with self.subTest(field=field,value=value):
                j=deepcopy(original);j[field]=value;self.assert_rejected_before_private_reads(j)

    def test_identity_binding_coverage_and_unsaved_judgment_gate(self):
        self.fake_generate();original=self.judgment()
        mutations=[('root_id','wrong_root'),('family_id','wrong_family'),('round_id','supplement'),
            ('source_sha256','0'*64),('public_assets_sha256','0'*64),('reviewer','')]
        for field,value in mutations:
            with self.subTest(field=field):
                j=deepcopy(original);j[field]=value;self.assert_rejected_before_private_reads(j)
        for field in ('parents','versions'):
            with self.subTest(coverage=field):
                j=deepcopy(original);j[field].pop();self.assert_rejected_before_private_reads(j)
        j=deepcopy(original);del j['versions'][0]['proposition'];self.assert_rejected_before_private_reads(j)
        dump(self.run/'rounds/initial/SOURCE_JUDGMENTS.json',original);unsaved=deepcopy(original);unsaved['versions'][0]['relation']='unknown'
        with self.assertRaisesRegex(ValueError,'not_saved_or_changed'):a.private_crosscheck(self.run,'initial',unsaved)

    def test_future_naive_and_pre_generation_judgment_timestamps_rejected(self):
        self.fake_generate();original=self.judgment();complete=a.read(self.run/'rounds/initial/GENERATION.COMPLETE.json')
        for timestamp in [(datetime.now(timezone.utc)+timedelta(days=1)).isoformat(),datetime.now().isoformat(),
                (datetime.fromisoformat(complete['completed_at_utc'])-timedelta(microseconds=1)).isoformat(),None]:
            with self.subTest(timestamp=timestamp):
                j=deepcopy(original);j['created_at_utc']=timestamp;self.assert_rejected_before_private_reads(j)

    def test_invalid_generation_completion_time_rejected_before_private_reads(self):
        self.fake_generate();j=self.judgment();p=self.run/'rounds/initial/GENERATION.COMPLETE.json';original=p.read_bytes()
        for timestamp in [None,datetime.now().isoformat(),(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()]:
            with self.subTest(timestamp=timestamp):
                c=json.loads(original)
                if timestamp is None:del c['completed_at_utc']
                else:c['completed_at_utc']=timestamp
                dump(p,c);self.assert_rejected_before_private_reads(j);p.write_bytes(original)

    def test_supplement_schema_cannot_bypass_finalize_or_verify(self):
        self.fake_generate();self.review(approved=['F01','F02']);a.prepare_supplement(self.run)
        obj=deepcopy(self.response);obj['replacements']=obj['replacements'][2:];self.fake_generate('supplement',obj)
        original=self.judgment('supplement')
        for field in [*a.review_schema.COMMON_DISCLOSURES,'full_source_read','review_schema_version']:
            with self.subTest(field=field):
                j=deepcopy(original);del j[field]
                self.assert_rejected_before_private_reads(j,'supplement',actions=[lambda:a.finish_review(self.run,'supplement'),
                    lambda:a.private_crosscheck(self.run,'supplement',j),lambda:a.reviewed_round(self.run,'supplement'),
                    lambda:a.finalize(self.run),lambda:a.verify(self.run,complete=True)])
        for field,value in [('current_solver_scores_read',True),('current_solver_scores_read',0),
                ('solver_scores_used',True),('full_source_read',False),('parent_gold_known_before_review','true')]:
            with self.subTest(field=field,value=value):
                j=deepcopy(original);j[field]=value
                self.assert_rejected_before_private_reads(j,'supplement',actions=[lambda:a.finish_review(self.run,'supplement'),
                    lambda:a.private_crosscheck(self.run,'supplement',j),lambda:a.reviewed_round(self.run,'supplement'),
                    lambda:a.finalize(self.run),lambda:a.verify(self.run,complete=True)])

    def test_invalid_initial_schema_blocks_supplement_and_finalization(self):
        self.fake_generate();j=self.judgment();del j['current_solver_scores_read']
        self.assert_rejected_before_private_reads(j,actions=[lambda:a.prepare_supplement(self.run),
            lambda:a.finalize(self.run),lambda:a.verify(self.run,complete=True)])

    def test_saved_judgment_bytes_locked_before_crosscheck_and_writes(self):
        self.fake_generate();j=self.judgment();p=self.run/'rounds/initial/SOURCE_JUDGMENTS.json';a.write(p,j)
        original_sha=a.sha(p);real_crosscheck=a.private_crosscheck
        def mutate_after_crosscheck(run,rid,saved):
            result=real_crosscheck(run,rid,saved);p.write_bytes(p.read_bytes()+b'\n');return result
        with patch.object(a,'private_crosscheck',side_effect=mutate_after_crosscheck),self.assertRaisesRegex(ValueError,'source_judgment_changed_after_private_read'):
            a.finish_review(self.run,'initial')
        self.assertEqual(a.read(p),j);self.assertNotEqual(a.sha(p),original_sha)
        self.assertFalse((p.parent/'ASSET_REVIEW.json').exists());self.assertFalse((p.parent/'REVIEW_RECEIPT.json').exists())

    def test_direct_private_crosscheck_detects_byte_change_during_private_read(self):
        self.fake_generate();j=self.judgment();p=self.run/'rounds/initial/SOURCE_JUDGMENTS.json';a.write(p,j);real_read=a.read
        def mutate_on_private_read(path):
            if Path(path).name=='ASSETS.json':p.write_bytes(p.read_bytes()+b'\n')
            return real_read(path)
        with patch.object(a,'read',side_effect=mutate_on_private_read),self.assertRaisesRegex(ValueError,'source_judgment_changed_after_private_read'):
            a.private_crosscheck(self.run,'initial',j)

    def test_judgment_byte_change_at_schema_boundary_blocks_first_private_read(self):
        self.fake_generate();j=self.judgment();p=self.run/'rounds/initial/SOURCE_JUDGMENTS.json';real_validate=a.validated_round
        def mutate_after_validation(*args,**kwargs):
            saved=real_validate(*args,**kwargs);p.write_bytes(p.read_bytes()+b'\n');return saved
        with patch.object(a,'validated_round',side_effect=mutate_after_validation):
            self.assert_rejected_before_private_reads(j,actions=[lambda:a.finish_review(self.run,'initial'),
                lambda:a.private_crosscheck(self.run,'initial',j),lambda:a.reviewed_round(self.run,'initial'),
                lambda:a.verify(self.run),lambda:a.verify(self.run,complete=True),lambda:a.finalize(self.run)])

    def test_schema_code_binding_required_in_every_private_review_entry(self):
        self.fake_generate();j=self.judgment();p=self.run/'MANIFEST.json';m=a.read(p);m['review_schema']['sha256']='0'*64;dump(p,m)
        self.assert_rejected_before_private_reads(j)

    def test_default_verify_detects_judgment_byte_change_during_mechanical_private_read(self):
        self.fake_generate();j=self.judgment();p=self.run/'rounds/initial/SOURCE_JUDGMENTS.json';a.write(p,j);real_read=a.read
        def mutate_on_private_read(path):
            if Path(path).name=='ASSETS.json':p.write_bytes(p.read_bytes()+b'\n')
            return real_read(path)
        with patch.object(a,'read',side_effect=mutate_on_private_read),self.assertRaisesRegex(ValueError,'source_judgment_changed_after_private_read'):
            a.verify(self.run)

    def test_receipt_binds_original_judgment_schema_and_canonical_disclosures(self):
        self.fake_generate();receipt=self.review();folder=self.run/'rounds/initial';j=a.read(folder/'SOURCE_JUDGMENTS.json')
        self.assertEqual(receipt['source_judgments_sha256'],a.sha(folder/'SOURCE_JUDGMENTS.json'))
        self.assertEqual(receipt['review_schema'],a.binding(a.schema_path()));self.assertIn(a.binding(a.schema_path()),receipt['files'])
        self.assertEqual(self.m['review_schema_version'],a.review_schema.SCHEMA_VERSION)
        self.assertIn(a.binding(a.schema_path()),self.m['files'])
        for field in a.review_schema.COMMON_DISCLOSURES:self.assertIs(receipt[field],j[field])
        p=folder/'REVIEW_RECEIPT.json';original=p.read_bytes()
        mutations=[('source_judgments_sha256','0'*64),('review_schema',{}),('family_id','wrong_family'),('usable_parents',0),
            ('source_judgments_bytes_changed_after_private_read',True),('full_source_read',False),
            ('private_crosscheck_started_at_utc',(datetime.fromisoformat(j['created_at_utc'])-timedelta(microseconds=1)).isoformat()),
            ('review_completed_at_utc',datetime.now().isoformat()),
            ('review_completed_at_utc',(datetime.now(timezone.utc)+timedelta(days=1)).isoformat())]
        for field,value in mutations:
            with self.subTest(field=field):
                bad=deepcopy(receipt);bad[field]=value;dump(p,bad)
                with self.assertRaises(ValueError):a.reviewed_round(self.run,'initial')
                p.write_bytes(original)
        for field in a.review_schema.COMMON_DISCLOSURES:
            with self.subTest(missing=field):
                bad=deepcopy(receipt);del bad[field];dump(p,bad)
                with self.assertRaises(ValueError):a.reviewed_round(self.run,'initial')
                p.write_bytes(original)

    def test_prepare_freezes_optional_historical_parent_records_without_rewriting(self):
        cfg=deepcopy(self.cfg);paths=[]
        for prefix in ('parent_review','parent_provenance'):
            path=self.home/(prefix+'.json');dump(path,{'legacy_record':True});paths.append(path)
            cfg['slots'][0][prefix+'_path']=str(path);cfg['slots'][0][prefix+'_sha256']=a.sha(path)
        config=self.home/'historical_CONFIG.json';dump(config,cfg);old=[p.read_bytes() for p in paths]
        run=self.home/'historical_assets';a.prepare(config,'family1',run=run);m=a.read(run/'MANIFEST.json')
        for path,content in zip(paths,old):self.assertIn(a.binding(path),m['files']);self.assertEqual(path.read_bytes(),content)
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

    def test_four_family_budget_and_duplicate_roots(self):
        cfg=deepcopy(self.cfg)
        cfg['slots'] += [{**cfg['slots'][0],'slot_id':'family'+str(i),'family_id':'family'+str(i),'root_id':'root'+str(i)} for i in (3,4)]
        cfg.update(family_count=4,max_http=8,max_questions=20,max_calls_per_model=360)
        self.assertEqual(a.get_slot(cfg,'family4')['root_id'],'root4')
        cfg['slots'][3]['root_id']='root3'
        with self.assertRaises(ValueError):a.get_slot(cfg,'family4')
    def test_budgets_and_clock(self):
        self.assertEqual(a.qa_transport.GAP,35.0)
        cfg=deepcopy(self.cfg);cfg['max_http']=5
        with self.assertRaises(ValueError):a.get_slot(cfg,'family1')

if __name__=='__main__':unittest.main()
