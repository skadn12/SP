"""Behavioral tests for bounded asset generation; never use real model transport."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import asset_generation as a


class AssetGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='v64-SYNTHETIC-assets-');self.addCleanup(self.temp.cleanup)
        self.home=Path(self.temp.name);self.root='cc-train-episode-fixture';self.slot='fixture_s01'
        self.history=self.home/'upstream/CANDIDATE_HISTORY.json';self.history.parent.mkdir()
        self.turns=[{'turn_ref':f'CANDIDATE:B:S01:T{i:03d}','speaker':'Alex',
          'text':f'The scheduled object {i} has exactly {i+1} seats.','origin':'provisional_candidate','training_pool_eligible':False} for i in range(1,9)]
        self.source={'root_id':self.root,'turns':self.turns,'training_allowed':False,'training_pool_eligible':False}
        for name,val in [('CANDIDATE_HISTORY.json',self.source),('MANIFEST.json',{}),('FORMAL_BASELINE.json',{}),('CANDIDATE_RECORDS.json',[]),('COLLECTION_BINDINGS.json',{})]:
            (self.history.parent/name).write_text(json.dumps(val))
        feedback=self.home/'feedback';(feedback/self.root).mkdir(parents=True)
        (feedback/'source_lock.json').write_text(json.dumps({'status':'EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY','merge_current_generation_feedback':False,'model_identity_source':{'sha256':a.v15.evaluator.TRAINED_AUDIT_SHA256}}))
        self.suffix='\n[PREVIOUS_TRAINING_ROUND_FEEDBACK_V4]\n'+json.dumps({'target_solver_model':a.legacy.TARGET,'old_feedback':'SYNTHETIC frozen feedback'},separators=(',',':'))+'\n[/PREVIOUS_TRAINING_ROUND_FEEDBACK_V4]'
        (feedback/self.root/('writer__'+a.legacy.TARGET+'_feedback.txt')).write_text(self.suffix)
        self.prior=self.home/'prior.json'
        self.prior.write_text(json.dumps({'questions':[{'candidate_id':'old','root_id':self.root,'question_type':'multi_select','question':'Old?', 'options':dict(zip('ABCDEF',[f'Old fact {i}' for i in range(6)])), 'gold':['A','B'],'solver_score':'NO_CURRENT_SCORE'}]}))
        self.config=self.home/'CONFIG.json'
        self.config.write_text(json.dumps({'version':'synthetic','slots':[{'slot_id':self.slot,'root_id':self.root,'history_path':str(self.history),'history_sha256':a.sha(self.history),'prior_pool_paths':[str(self.prior)],'focus':{'scene_refs':['CANDIDATE:B:S01'],'description':'Synthetic seating plans'},'initial_target_count':2}], 'max_http':20,'max_questions':10,'max_calls_per_model':180,'training_allowed':False}))
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(a.legacy.legacy,'FEEDBACK',feedback))
        self.stack.enter_context(patch.object(a.source_adapter,'checked_source',return_value={'source':self.source,'files':[self.history],'audit':{'status':'SYNTHETIC_SOURCE_VERIFIED'}}))
        synthetic_feedback={'source_lock':a.read(feedback/'source_lock.json'),'candidate_history':[],
            'aggregate':{},'writer_suffix':self.suffix,'files':[feedback/'source_lock.json'],
            'receipt':{'root_id':self.root,'synthetic_test':True}}
        self.stack.enter_context(patch.object(a.source_adapter,'feedback_data',return_value=synthetic_feedback))
        self.stack.enter_context(patch.object(a.source_adapter,'prior_candidates',return_value=([],[])))
        self.stack.enter_context(patch.object(a,'_dependencies',return_value=({}, {Path(a.__file__)})))
        self.stack.enter_context(patch('socket.socket',side_effect=AssertionError('No network in tests')))
        self.run=self.home/'run';a.prepare(self.config,self.slot,run=self.run)

    def asset(self,count=8,contradictions=True):
        return {'root_id':self.root,'slot_id':self.slot,'public_stem':'Which statements about the seating plans are supported? Select all that apply.',
          'facts':[{'fact_id':f'F{i:02d}','scope':{'actor':'Alex','event':f'object {i} seating plan','time':'scheduled','condition':'none','action_state':'scheduled'},
             'anchor':{'turn_ref':t['turn_ref'],'speaker':t['speaker'],'quote':t['text']},
             'supported':{'proposition':t['text'],'relation':'supports','rationale':'Exact stated seat count.'},
             'contradicted':{'proposition':f'The scheduled object {i} has exactly {i+2} seats.','relation':'contradicts','rationale':'Same object has a different explicitly fixed exact seat count.'} if contradictions else None,
             'contradiction_unavailable_reason':None if contradictions else f'No explicitly refuted alternative was constructed within the same scheduled object {i} seating-plan scope.'}
              for i,t in enumerate(self.turns[:count],1)]}

    def mock_generate(self,replies):
        recorded=[]
        class Caller:
            def __init__(self,run):self.run=run;self.counter=0
            def call(self,label,payload,proof):
                self.counter+=1;recorded.append(deepcopy(payload));folder=self.run/'requests'/f'{self.counter:03d}-{label}'
                a.write(folder/'REQUEST.json',payload);a.write(folder/'PROOF.json',proof)
                a.write(folder/'HTTP_ATTEMPT.json',{'actual_payload_equals_frozen':True,'payload_sha256':a.base.digest(payload)})
                reply=replies[self.counter-1]
                if reply is None:
                    a.write(folder/'ERROR.json',{'error_type':'SyntheticTransportFailure'});return None,folder,{}
                raw=reply if isinstance(reply,str) else json.dumps(reply)
                meta={'returned_model':a.legacy.MODEL,'finish_reason':'stop','http_status':200}
                a.write(folder/'RESPONSE.json',{'text':raw,'metadata':meta});return raw,folder,meta
        with patch.object(a.qa_transport,'Caller',Caller),patch.object(a.qa_transport,'shared_lock',return_value=nullcontext()):result=a.generate(self.run)
        return result,recorded

    def test_full_inputs_feedback_tail_and_no_target_or_gold_in_payload(self):
        p=a.read(self.run/'INITIAL_REQUEST.json');text=p['messages'][0]['content']
        context=json.JSONDecoder().raw_decode(text.split('\nINPUT_JSON\n')[1])[0]
        self.assertEqual(context['source_turns'],self.turns)
        self.assertEqual(context['previous_candidates_dedup_only'][0]['facts'],[f'Old fact {i}' for i in range(6)])
        self.assertNotIn('gold',json.dumps(context));self.assertNotIn('initial_target_count',json.dumps(context));self.assertNotIn('NO_CURRENT_SCORE',text)
        self.assertEqual(p['max_tokens'],6144)
        self.assertEqual(text.count(self.suffix),1)
        self.assertTrue(text.split(a.START)[0].endswith(self.suffix+'\n\n'))
        self.assertNotIn('preferentially seek 2, 3 or 4',text)
        a.verify(self.run)

    def test_valid_assets_no_relation_leak_and_mapping_covers_all_versions(self):
        result,payloads=self.mock_generate([self.asset()])
        self.assertEqual(result['status'],'ASSETS_READY');self.assertEqual(result['http_attempts'],1)
        public=a.read(self.run/'PUBLIC_ASSETS.json');mapping=a.read(self.run/'VERSION_MAP.json')
        self.assertEqual(len(public['versions']),16)
        self.assertTrue(all(set(v)=={'version_id','fact_id','proposition'} for v in public['versions']))
        self.assertEqual({v['version_id'] for v in public['versions']},{v['version_id'] for v in mapping['versions']})
        self.assertTrue(all(not v['version_id'].endswith((':s',':c')) for v in public['versions']))
        self.assertNotIn('initial_target_count',json.dumps(public));self.assertNotIn('rationale',json.dumps(public))
        self.assertNotIn('contradiction_unavailable_reason',json.dumps(public))
        self.assertEqual(mapping['assets_sha256'],a.sha(self.run/'ASSETS.json'))
        self.assertEqual(mapping['public_assets_sha256'],a.sha(self.run/'PUBLIC_ASSETS.json'))
        a.verify(self.run,complete=True)
        with self.assertRaisesRegex(ValueError,'already_started'):
            a.generate(self.run)

    def test_all_supported_assets_and_shortages_do_not_trigger_retry(self):
        result,payloads=self.mock_generate([self.asset(4,False)])
        self.assertEqual(result['status'],'INSUFFICIENT_FACTS');self.assertEqual(result['fact_count'],4)
        self.assertEqual(len(payloads),1);self.assertEqual(len(a.read(self.run/'PUBLIC_ASSETS.json')['versions']),4)
        self.assertTrue(all(f['contradicted'] is None for f in a.read(self.run/'ASSETS.json')['facts']))

    def test_null_contradictions_with_reasons_are_ready_without_semantic_retry(self):
        draft=self.asset(8,False)
        # Nonempty explanation content is not automatically certified or rewritten.
        draft['facts'][0]['contradiction_unavailable_reason']='Needs independent review.'
        result,payloads=self.mock_generate([draft])
        self.assertEqual(result['status'],'ASSETS_READY');self.assertEqual(len(payloads),1)
        self.assertEqual(a.read(self.run/'ASSETS.json'),draft)
        self.assertEqual(len(a.read(self.run/'PUBLIC_ASSETS.json')['versions']),8)
        self.assertTrue(result['source_first_review_required'])
        self.assertEqual(result['semantic_retries'],0)

    def test_null_reason_schema_is_mechanical_and_exact(self):
        m=a.read(self.run/'MANIFEST.json')
        for invalid in (None,'','   ',7,False,[],{}):
            with self.subTest(reason=invalid):
                draft=self.asset(1,False)
                draft['facts'][0]['contradiction_unavailable_reason']=invalid
                checked,errors=a.validate_assets(json.dumps(draft),m,self.source)
                self.assertIsNone(checked)
                self.assertEqual(errors,['facts[0]:contradiction_unavailable_reason_required'])
        for invalid in ('No alternative.',False,0,{},[]):
            with self.subTest(non_null_contradiction_reason=invalid):
                draft=self.asset(1)
                draft['facts'][0]['contradiction_unavailable_reason']=invalid
                checked,errors=a.validate_assets(json.dumps(draft),m,self.source)
                self.assertIsNone(checked)
                self.assertEqual(errors,['facts[0]:contradiction_unavailable_reason_must_be_null'])
        draft=self.asset(1);del draft['facts'][0]['contradiction_unavailable_reason']
        checked,errors=a.validate_assets(json.dumps(draft),m,self.source)
        self.assertIsNone(checked);self.assertEqual(errors,['facts[0]:missing_keys:["contradiction_unavailable_reason"]'])

    def test_missing_null_reason_fails_closed_without_semantic_invention(self):
        bad=self.asset(8,False);bad['facts'][0]['contradiction_unavailable_reason']=' '
        good=deepcopy(bad)
        good['facts'][0]['contradiction_unavailable_reason']='No in-scope explicitly refuted alternative was constructed for the scheduled object 1 seating plan.'
        result,payloads=self.mock_generate([bad,good])
        self.assertEqual(result['status'],'NO_VALID_ASSETS');self.assertEqual(len(payloads),1)
        guard=a.read(next((self.run/'requests').glob('*/REPAIR_GUARD.json')))
        self.assertEqual(guard['status'],'BASELINE_BLOCKED')
        self.assertFalse(guard['repair_allowed'])
        self.assertEqual(result['semantic_retries'],0)
        self.assertFalse((self.run/'ASSETS.json').exists())

    def test_pair_task_explains_scope_example_and_explicit_null_reason(self):
        self.assertIn('For EACH fact, first write one',a.TASK)
        self.assertIn('Do not\nskip the contradicted field or default it to null.',a.TASK)
        self.assertIn('appointment is scheduled for Tuesday at 3 PM.',a.TASK)
        self.assertIn('appointment is scheduled for Wednesday at 3 PM.',a.TASK)
        self.assertIn('Another actor or another time is not\ncounterevidence.',a.TASK)
        self.assertIn('contradiction_unavailable_reason:null|string',a.TASK)
        self.assertIn('before the later independent question-assembly step',a.TASK)
        self.assertIn('Copy turn_ref verbatim',a.TASK)
        self.assertIn('R01:Sxx:Txxx',a.TASK)
        self.assertIn('explicitly mutually exclusive',a.TASK)
        self.assertIn('Prefer coherent facts within the soft focus',a.TASK)

    def test_only_mechanical_error_repair_preserves_source_prior_and_draft(self):
        bad=self.asset();bad['facts'][1]['anchor']['quote']='Not an exact source quote.'
        result,payloads=self.mock_generate([bad,self.asset()])
        self.assertEqual(result['status'],'ASSETS_READY');self.assertEqual(len(payloads),2)
        contexts=[json.JSONDecoder().raw_decode(p['messages'][0]['content'].split('\nINPUT_JSON\n')[1])[0] for p in payloads]
        repair=contexts[1].pop('mechanical_repair');self.assertEqual(contexts[0],contexts[1])
        self.assertEqual(json.loads(repair['previous_response']),bad)
        self.assertEqual(repair['errors'],['facts[1]:quote_not_exact_contiguous'])
        self.assertEqual(payloads[1]['max_tokens'],6144)
        self.assertEqual(repair,a.build_repair(json.dumps(bad),repair['errors'],a.read(self.run/'MANIFEST.json'),self.source))

    def test_schema_errors_name_missing_extra_and_wrong_object(self):
        m=a.read(self.run/'MANIFEST.json')
        draft=self.asset(1)
        del draft['facts'][0]['scope']['time'];draft['facts'][0]['scope']['unexpected']='value'
        draft['facts'][0]['supported']=[]
        checked,errors=a.validate_assets(json.dumps(draft),m,self.source)
        self.assertIsNone(checked)
        self.assertEqual(errors,['facts[0]:scope:missing_keys:["time"]',
                                 'facts[0]:scope:extra_keys:["unexpected"]',
                                 'facts[0]:supported:expected_object'])

    def test_unique_ref_hint_is_writer_only_and_replays_after_manual_copy(self):
        bad=self.asset();correct=deepcopy(bad)
        bad['facts'][0]['anchor']['turn_ref']='P01:T999'
        raw=json.dumps(bad);m=a.read(self.run/'MANIFEST.json')
        checked,errors,receipt=a.normalize_assets(raw,m,self.source)
        self.assertIsNone(checked);self.assertEqual(receipt['normalized_ref_count'],0)
        repair=a.build_repair(raw,errors,m,self.source)
        self.assertEqual(repair['reference_hints'],[{'fact_index':0,'fact_id':'F01',
            'original_turn_ref':'P01:T999','candidate_turn_ref':self.turns[0]['turn_ref'],
            'speaker':'Alex','quote':self.turns[0]['text']}])
        self.assertEqual(json.loads(repair['previous_response']),bad)
        result,payloads=self.mock_generate([bad,correct])
        self.assertEqual(result['status'],'ASSETS_READY');self.assertEqual(len(payloads),2)
        second=json.JSONDecoder().raw_decode(payloads[1]['messages'][0]['content'].split('\nINPUT_JSON\n')[1])[0]
        self.assertEqual(second['mechanical_repair'],repair)
        self.assertEqual(a.read(self.run/'ASSETS.json'),correct)
        a.verify(self.run,complete=True)

    def test_ref_hints_do_not_relocate_valid_ref_speaker_or_quote_errors(self):
        m=a.read(self.run/'MANIFEST.json')
        for change in ('speaker','quote'):
            with self.subTest(change=change):
                draft=self.asset(1)
                if change=='speaker':draft['facts'][0]['anchor']['speaker']='Wrong speaker'
                else:draft['facts'][0]['anchor']['quote']=self.turns[1]['text']
                raw=json.dumps(draft);checked,errors=a.validate_assets(raw,m,self.source)
                self.assertIsNone(checked)
                self.assertEqual(a.build_repair(raw,errors,m,self.source)['reference_hints'],[])

    def test_ref_hints_require_same_root_unique_speaker_and_nonempty_quote(self):
        m=a.read(self.run/'MANIFEST.json')
        for change in ('raw_root','source_root','duplicate_quote','duplicate_ref','empty_quote','wrong_speaker'):
            with self.subTest(change=change):
                draft=self.asset(1);source=deepcopy(self.source)
                draft['facts'][0]['anchor']['turn_ref']='missing'
                if change=='raw_root':draft['root_id']='other-root'
                if change=='source_root':source['root_id']='other-root'
                if change=='duplicate_quote':source['turns'].append({**source['turns'][0],'turn_ref':'another-ref'})
                if change=='duplicate_ref':source['turns'].append(deepcopy(source['turns'][0]))
                if change=='empty_quote':draft['facts'][0]['anchor']['quote']=' '
                if change=='wrong_speaker':draft['facts'][0]['anchor']['speaker']='Wrong'
                raw=json.dumps(draft);checked,errors=a.validate_assets(raw,m,source)
                self.assertIsNone(checked)
                self.assertEqual(a.build_repair(raw,errors,m,source)['reference_hints'],[])
        self.assertEqual(a.build_repair('not json',['invalid_json'],m,self.source)['reference_hints'],[])

    def test_unchanged_bad_reference_draft_cannot_be_accepted_from_hint(self):
        bad=self.asset();bad['facts'][0]['anchor']['turn_ref']='missing'
        result,payloads=self.mock_generate([bad,bad])
        self.assertEqual(result['status'],'NO_VALID_ASSETS');self.assertEqual(len(payloads),2)
        self.assertFalse((self.run/'ASSETS.json').exists())
        second=json.JSONDecoder().raw_decode(payloads[1]['messages'][0]['content'].split('\nINPUT_JSON\n')[1])[0]
        self.assertEqual(len(second['mechanical_repair']['reference_hints']),1)

    def test_transport_failure_never_retries(self):
        result,payloads=self.mock_generate([None])
        self.assertEqual(result['status'],'NO_VALID_ASSETS');self.assertEqual(len(payloads),1)
        self.assertFalse((self.run/'ASSETS.json').exists())

    def test_unparseable_first_draft_fails_closed_at_one(self):
        result,payloads=self.mock_generate(['not json','not json'])
        self.assertEqual(result['status'],'NO_VALID_ASSETS');self.assertEqual(len(payloads),1)
        self.assertFalse((self.run/'PUBLIC_ASSETS.json').exists())

    def test_semantic_relations_are_not_automatically_rewritten(self):
        draft=self.asset();draft['facts'][0]['contradicted']['proposition']=draft['facts'][0]['supported']['proposition']
        result,_=self.mock_generate([draft])
        self.assertEqual(result['http_attempts'],1)
        self.assertEqual(a.read(self.run/'ASSETS.json'),draft)
        self.assertTrue(result['source_first_review_required'])

    def test_tampered_source_and_prompt_detected(self):
        p=self.run/'SOURCE.json';original=p.read_text();data=json.loads(original);data['turns'].pop();p.write_text(json.dumps(data))
        with self.assertRaises(ValueError):a.verify(self.run)
        p.write_text(original)
        p=self.run/'INITIAL_REQUEST.json';data=a.read(p);data['messages'][0]['content']=data['messages'][0]['content'].replace('Synthetic seating plans','Changed focus');p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'asset_request_replay_changed'):a.verify(self.run)

    def test_tampered_public_or_private_mapping_detected(self):
        self.mock_generate([self.asset()]);p=self.run/'PUBLIC_ASSETS.json';original=p.read_text();data=json.loads(original);data['versions'][0]['proposition']='Changed';p.write_text(json.dumps(data))
        with self.assertRaises(ValueError):a.verify(self.run,complete=True)
        p.write_text(original);p=self.run/'VERSION_MAP.json';data=a.read(p);data['versions'][0]['relation']='unknown';p.write_text(json.dumps(data))
        with self.assertRaises(ValueError):a.verify(self.run,complete=True)

    def test_fact_id_count_and_reference_are_mechanical_only(self):
        m=a.read(self.run/'MANIFEST.json')
        for mutation in ('duplicate','speaker','enum','count'):
            bad=self.asset()
            if mutation=='duplicate':bad['facts'][1]['fact_id']=bad['facts'][0]['fact_id']
            if mutation=='speaker':bad['facts'][0]['anchor']['speaker']='Wrong'
            if mutation=='enum':bad['facts'][0]['supported']['relation']='unknown'
            if mutation=='count':bad['facts'].append(deepcopy(bad['facts'][-1]))
            checked,errors=a.validate_assets(json.dumps(bad),m,self.source)
            self.assertIsNone(checked);self.assertTrue(errors)

    def test_repair_semantic_drift_rejected_and_proof_replayed(self):
        bad=self.asset();bad['facts'][0]['anchor']['turn_ref']='missing'
        changed=self.asset();changed['facts'][7]['supported']['proposition']+=' altered'
        changed['facts'][7]['supported']['rationale']+=' altered'
        result,payloads=self.mock_generate([bad,changed])
        self.assertEqual(result['status'],'NO_VALID_ASSETS');self.assertEqual(len(payloads),2)
        paths=sorted((self.run/'requests').glob('*/REPAIR_GUARD.json'))
        first,last=map(a.read,paths)
        self.assertEqual(first['status'],'REPAIR_ALLOWED')
        self.assertEqual(last['status'],'REPAIR_BLOCKED')
        self.assertIn(['facts',7,'supported','proposition'],last['unauthorized_changes'])
        self.assertIn(['facts',7,'supported','rationale'],last['unauthorized_changes'])
        self.assertNotEqual(last['baseline_projection_sha256'],last['candidate_projection_sha256'])
        self.assertFalse((self.run/'ASSETS.json').exists())
        a.verify(self.run,complete=True)
        last['accepted']=True;paths[-1].write_text(json.dumps(last))
        with self.assertRaisesRegex(ValueError,'asset_repair_guard_replay_changed'):a.verify(self.run,complete=True)

    def test_missing_null_field_and_extra_removal_are_only_schema_edits(self):
        bad=self.asset();del bad['facts'][0]['contradiction_unavailable_reason']
        bad['facts'][1]['supported']['extra']='remove me'
        result,payloads=self.mock_generate([bad,self.asset()])
        self.assertEqual(result['status'],'ASSETS_READY');self.assertEqual(len(payloads),2)
        guard=a.read(sorted((self.run/'requests').glob('*/REPAIR_GUARD.json'))[-1])
        self.assertEqual(guard['status'],'REPAIR_ACCEPTED')
        self.assertEqual(guard['baseline_projection_sha256'],guard['candidate_projection_sha256'])
        self.assertIn({'path':['facts',0,'contradiction_unavailable_reason'],'operation':'set','value':None},guard['allowed_changes'])
        self.assertIn({'path':['facts',1,'supported','extra'],'operation':'delete'},guard['allowed_changes'])
        a.verify(self.run,complete=True)

    def test_every_nonpermitted_semantic_field_is_locked(self):
        m=a.read(self.run/'MANIFEST.json');before=self.asset();before['facts'][0]['anchor']['turn_ref']='missing'
        raw=json.dumps(before);_,errors=a.validate_assets(raw,m,self.source)
        mutations=[(['root_id'],'other'),(['slot_id'],'other'),(['public_stem'],'Changed stem'),
                   (['facts',0,'fact_id'],'F99'),(['facts',0,'scope','time'],'another time'),
                   (['facts',0,'supported','relation'],'contradicts'),
                   (['facts',0,'supported','proposition'],'Changed proposition'),
                   (['facts',0,'contradicted','rationale'],'Changed rationale'),
                   (['facts',0,'contradiction_unavailable_reason'],'Changed explanation')]
        for path,value in mutations:
            with self.subTest(path=path):
                after=self.asset();parent=after
                for key in path[:-1]:parent=parent[key]
                parent[path[-1]]=value
                _,ce=a.validate_assets(json.dumps(after),m,self.source)
                guard=a.build_repair_guard(raw,errors,m,self.source,json.dumps(after),ce)
                self.assertFalse(guard['accepted']);self.assertIn(path,guard['unauthorized_changes'])
        for kind in ('order','add','remove','remove_semantic_field'):
            with self.subTest(kind=kind):
                after=self.asset()
                if kind=='order':after['facts'].reverse()
                if kind=='add':after['facts'].append(deepcopy(after['facts'][0]))
                if kind=='remove':after['facts'].pop()
                if kind=='remove_semantic_field':del after['facts'][0]['scope']
                _,ce=a.validate_assets(json.dumps(after),m,self.source)
                self.assertFalse(a.build_repair_guard(raw,errors,m,self.source,json.dumps(after),ce)['accepted'])

    def test_cannot_lock_missing_semantics_or_ambiguous_json(self):
        m=a.read(self.run/'MANIFEST.json')
        drafts=[]
        for path in (['public_stem'],['facts',0,'scope','time'],['facts',0,'supported','proposition'],
                     ['facts',0,'supported','relation'],['facts',0,'supported','rationale'],['facts',0,'fact_id']):
            bad=self.asset();parent=bad
            for key in path[:-1]:parent=parent[key]
            del parent[path[-1]];drafts.append(json.dumps(bad))
        drafts.extend(['not JSON',json.dumps(self.asset()).replace('"root_id":','"root_id": "duplicate", "root_id":',1),
                       json.dumps(self.asset()).replace('"contradiction_unavailable_reason": null','"contradiction_unavailable_reason": NaN',1)])
        for raw in drafts:
            with self.subTest(raw=raw[:70]):
                _,errors=a.validate_assets(raw,m,self.source)
                guard=a.build_repair_guard(raw,errors,m,self.source)
                self.assertEqual(guard['status'],'BASELINE_BLOCKED');self.assertFalse(guard['repair_allowed'])

    def test_permission_is_path_and_exact_value_bound(self):
        m=a.read(self.run/'MANIFEST.json');before=self.asset();before['facts'][0]['anchor']['turn_ref']='missing'
        raw=json.dumps(before);_,errors=a.validate_assets(raw,m,self.source)
        after=self.asset();after['facts'][0]['anchor']['turn_ref']=self.turns[1]['turn_ref']
        _,ce=a.validate_assets(json.dumps(after),m,self.source)
        guard=a.build_repair_guard(raw,errors,m,self.source,json.dumps(after),ce)
        self.assertFalse(guard['accepted']);self.assertIn(['facts',0,'anchor','turn_ref'],guard['unauthorized_changes'])


    def test_guard_recomputes_mechanical_errors_and_forbids_unneeded_repair(self):
        m=a.read(self.run/'MANIFEST.json');before=self.asset();before['facts'][0]['anchor']['turn_ref']='missing'
        raw=json.dumps(before);_,errors=a.validate_assets(raw,m,self.source)
        guard=a.build_repair_guard(raw,errors,m,self.source,raw)
        self.assertFalse(guard['accepted'])
        self.assertIn('candidate_mechanical_validation_failed',guard['lock_errors'])
        valid=json.dumps(self.asset());changed=valid.replace('Select all that apply.','Changed stem.')
        guard=a.build_repair_guard(valid,[],m,self.source,changed,[])
        self.assertEqual(guard['status'],'REPAIR_BLOCKED');self.assertFalse(guard['accepted'])
        guard=a.build_repair_guard(raw,[],m,self.source)
        self.assertEqual(guard['status'],'BASELINE_BLOCKED');self.assertFalse(guard['repair_allowed'])

    def test_strict_duplicate_key_response_cannot_publish_or_retry(self):
        raw=json.dumps(self.asset()).replace('"root_id":','"root_id": "duplicate", "root_id":',1)
        result,payloads=self.mock_generate([raw,self.asset()])
        self.assertEqual(result['status'],'NO_VALID_ASSETS');self.assertEqual(len(payloads),1)
        self.assertFalse((self.run/'ASSETS.json').exists())
        guard=a.read(next((self.run/'requests').glob('*/REPAIR_GUARD.json')))
        self.assertEqual(guard['status'],'BASELINE_BLOCKED')
        a.verify(self.run,complete=True)



if __name__=='__main__':unittest.main()
