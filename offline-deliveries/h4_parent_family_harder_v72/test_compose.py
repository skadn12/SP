"""Offline contract tests for immutable controls and dependent family children."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import compose as c


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def fixture(home,false_count=6,valid_parents=True):
    baseline=[];slots=[]
    for number in (1,2):
        root='root'+str(number);fid='family'+str(number);run=home/'assets'/fid;run.mkdir(parents=True)
        source={'root_id':root,'turns':[{'turn_ref':f'T{i}','speaker':'Alex','text':f'Alex chose choice {i}.'} for i in range(1,7)]}
        save(run/'SOURCE.json',source)
        parent={'candidate_id':root+'-original','root_id':root,'question_type':'multi_select','topic':'Choices','question':'Which statements were made?',
          'options':{l:f'Alex chose choice {i}.' for i,l in enumerate('ABCDEF',1)},'gold':list('ABCDEF'),'option_evidence':{},'source_path':str(run/'SOURCE.json'),'source_sha256':c.sha(run/'SOURCE.json'),'original_extra':{'keep':'unchanged'}}
        facts=[];parent_facts=[];versions=[];maps=[];jp=[];rp=[];jv=[];rv=[]
        for i,l in enumerate('ABCDEF',1):
            fact=f'F{i:02d}';anchor={'turn_ref':f'T{i}','speaker':'Alex','quote':source['turns'][i-1]['text']}
            parent['option_evidence'][l]={**anchor,'relation':'supports','rationale':'Direct quote.'}
            supported={'proposition':parent['options'][l],'relation':'supports','anchor':anchor,'rationale':'Direct quote.'}
            false={'proposition':f'Alex rejected choice {i}.','relation':'contradicts','anchor':anchor,'rationale':'The choice was accepted.'} if i<=false_count else None
            facts.append({'fact_id':fact,'parent_label':l,'supported':supported,'contradicted':false,'contradiction_unavailable_reason':None if false else 'Unavailable.'})
            parent_facts.append({'fact_id':fact,'parent_label':l,'proposition':supported['proposition']})
            usable=valid_parents or i!=1
            jp.append({'fact_id':fact,'relation':'supports' if usable else 'unknown','scope_valid':usable,'fact_distinct':True,'reason':'Checked.','citations':[anchor]})
            rp.append({'fact_id':fact,'relation':jp[-1]['relation'],'usable':usable})
            if false:
                vid='public'+str(i);versions.append({'version_id':vid,'fact_id':fact,'proposition':false['proposition']});maps.append({'version_id':vid,'fact_id':fact,'variant':'contradicted','relation':'contradicts'})
                jv.append({'version_id':vid,'relation':'contradicts','scope_valid':True,'fact_distinct':True,'reason':'Checked.','citations':[anchor]});rv.append({'version_id':vid,'relation':'contradicts','usable':True})
        save(run/'PARENT.json',parent);baseline.append(parent)
        save(run/'ASSETS.json',{'family_id':fid,'root_id':root,'parent_candidate_id':parent['candidate_id'],'parent_sha256':c.sha(run/'PARENT.json'),'source_sha256':c.sha(run/'SOURCE.json'),'public_stem':parent['question'],'facts':facts})
        bindings={'source_sha256':c.sha(run/'SOURCE.json'),'assets_sha256':c.sha(run/'ASSETS.json')}
        save(run/'PUBLIC_ASSETS.json',{**bindings,'public_stem':parent['question'],'parent_facts':parent_facts,'versions':versions,'contains_relations':False,'contains_solver_scores':False})
        bindings['public_assets_sha256']=c.sha(run/'PUBLIC_ASSETS.json')
        save(run/'VERSION_MAP.json',{**bindings,'versions':maps})
        save(run/'SOURCE_JUDGMENTS.json',{**bindings,'gold_read_before_save':False,'solver_scores_used':False,'stem_valid':True,'parents':jp,'versions':jv})
        derived=c.read(run/'SOURCE_JUDGMENTS.json');derived.update(review_mode='derived_from_saved_round_source_judgments',derived_from_rounds=True,source_first_per_round=True,new_variant_private_artifacts_read_before_save=True);save(run/'SOURCE_JUDGMENTS.json',derived)
        save(run/'ROUND_PROVENANCE.json',{'rounds':[],'accepted_versions':[]})
        save(run/'ASSET_REVIEW.json',{**bindings,'source_judgments_sha256':c.sha(run/'SOURCE_JUDGMENTS.json'),'source_first':True,'solver_scores_used':False,'stem_valid':True,'parents':rp,'versions':rv,'review_mode':'derived_round_private_crosschecks','source_first_per_round':True,'provenance_rounds':[]})
        common={'family_id':fid,'root_id':root,'training_allowed':False,'files':[]}
        save(run/'MANIFEST.json',common);save(run/'GENERATION.COMPLETE.json',{**common,'http_attempts':1,'semantic_retries':0})
        slots.append({'slot_id':fid,'family_id':fid,'root_id':root,'parent_candidate_id':parent['candidate_id'],'history_path':parent['source_path']})
    save(home/'BASELINE.json',{'questions':baseline})
    save(home/'CONFIG.json',{'training_allowed':False,'max_questions':10,'max_calls_per_model':180,'slots':slots,'baseline_pool':str(home/'BASELINE.json'),'target_distribution':{str(k):2 for k in (2,3,4,5)}})
    return baseline


class ComposeTests(unittest.TestCase):
    def setUp(self):
        # The generator owns round replay tests. These isolated composer fixtures model its verified root output.
        self.generator_patch=patch.dict('sys.modules',{'asset_generation':SimpleNamespace(verify=lambda *args,**kwargs:None)})
        self.generator_patch.start();self.addCleanup(self.generator_patch.stop)

    def test_control_exact_dict_children_fixed_facts_and_gold(self):
        with tempfile.TemporaryDirectory() as td:
            home=Path(td);old=fixture(home);outputs,_,receipts=c.build(home);qs=outputs['POOL.json']['questions']
            self.assertEqual(len(qs),10)
            self.assertEqual([q for q in qs if len(q['gold'])==6],old)
            for q in qs:
                self.assertEqual(q['gold'],[l for l,e in q['option_evidence'].items() if e['relation']=='supports'])
            for r in receipts:
                if r['role']=='child':
                    self.assertEqual({v['fact_id'] for v in r['selected_versions']},{f'F{i:02d}' for i in range(1,7)})
            self.assertFalse(outputs['FAMILY_SIDECAR.json']['siblings_are_independent_samples'])
            self.assertEqual(outputs['ALLOCATION.json']['assembled_distribution'],{str(k):2 for k in (2,3,4,5)})

    def test_false_shortfall_by_count(self):
        with tempfile.TemporaryDirectory() as td:
            home=Path(td);fixture(home,false_count=2);outputs,_,_=c.build(home)
            self.assertEqual(outputs['ALLOCATION.json']['assembled_distribution'],{'2':0,'3':0,'4':2,'5':2})
            for f in outputs['ALLOCATION.json']['families']:
                self.assertEqual([r['gold_count'] for r in f['shortfalls']],[2,3])
                self.assertTrue(all(r['reason']=='INSUFFICIENT_VERIFIED_FALSE_FACTS' for r in f['shortfalls']))

    def test_unusable_parent_prevents_children_without_removing_control(self):
        with tempfile.TemporaryDirectory() as td:
            home=Path(td);old=fixture(home,valid_parents=False);outputs,_,_=c.build(home)
            self.assertEqual(outputs['POOL.json']['questions'],old)

    def test_parent_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            home=Path(td);fixture(home);path=home/'assets/family1/PARENT.json';parent=c.read(path);parent['original_extra']['keep']='changed';save(path,parent)
            with self.assertRaisesRegex(ValueError,'parent_not_exact'): c.build(home)

    def test_review_cannot_relabel_unknown_as_false(self):
        with tempfile.TemporaryDirectory() as td:
            home=Path(td);fixture(home);run=home/'assets/family1';j=c.read(run/'SOURCE_JUDGMENTS.json');j['versions'][0]['relation']='unknown';save(run/'SOURCE_JUDGMENTS.json',j)
            r=c.read(run/'ASSET_REVIEW.json');r['source_judgments_sha256']=c.sha(run/'SOURCE_JUDGMENTS.json');r['versions'][0]['relation']='unknown';save(run/'ASSET_REVIEW.json',r)
            with self.assertRaisesRegex(ValueError,'usable_false_or_unverified'):c.build(home)

    def test_gold_blind_permutation_ignores_relation_labels(self):
        v=[{'fact_id':str(i),'proposition':'p'+str(i),'relation':'supports'} for i in range(6)]
        swapped=deepcopy(v)
        for i,x in enumerate(swapped):x['relation']='contradicts' if i%2 else 'unknown'
        a,aa=c.gold_blind_permutation('family',v);b,bb=c.gold_blind_permutation('family',swapped)
        self.assertEqual(a,b);self.assertEqual([x['fact_id'] for x in aa],[x['fact_id'] for x in bb])

    def test_exact_dedup_reorders_labels_and_ignores_case_whitespace(self):
        with tempfile.TemporaryDirectory() as td:
            home=Path(td);old=fixture(home);inputs={};cfg=c.read(home/'CONFIG.json');f=c.load_family(home,cfg['slots'][0],old,inputs)
            possible=c.enumerate_children(f);candidate=possible[2][0];q,_=c.assemble_child(f,candidate)
            q['options']={l:'  '+text.upper()+'  ' for l,text in zip('FEDCBA',q['options'].values())}
            filtered=c.enumerate_children(f,{c.statement_key(q)})
            self.assertNotIn(candidate['key'],[x['key'] for x in filtered[2]])
            self.assertEqual(len(filtered[2]),len(possible[2])-1)

    def test_generation_failure_retains_controls_and_explicit_shortfalls(self):
        with tempfile.TemporaryDirectory() as td:
            home=Path(td);old=fixture(home)
            for number in (1,2):
                run=home/'assets'/('family'+str(number))
                for name in ('ASSETS.json','PUBLIC_ASSETS.json','VERSION_MAP.json','SOURCE_JUDGMENTS.json','ASSET_REVIEW.json'):(run/name).unlink()
                complete=c.read(run/'GENERATION.COMPLETE.json');complete['status']='NO_VALID_ASSETS';save(run/'GENERATION.COMPLETE.json',complete)
            outputs,_,_=c.build(home)
            self.assertEqual(outputs['POOL.json']['questions'],old)
            self.assertTrue(all(x['reason']=='NO_VALID_GENERATED_ASSETS' for f in outputs['ALLOCATION.json']['families'] for x in f['shortfalls']))

    def test_external_parent_pool_and_no_stale_parent_metadata_on_child(self):
        with tempfile.TemporaryDirectory() as td:
            home=Path(td);parents=fixture(home);save(home/'PARENT_POOL.json',{'questions':parents})
            cfg=c.read(home/'CONFIG.json')
            for slot in cfg['slots']:slot['parent_pool_path']=str(home/'PARENT_POOL.json')
            save(home/'CONFIG.json',cfg);save(home/'BASELINE.json',{'questions':[]})
            outputs,_,_=c.build(home)
            self.assertEqual(len(outputs['POOL.json']['questions']),10)
            for q in outputs['POOL.json']['questions']:
                self.assertEqual('original_extra' in q,len(q['gold'])==6)

    def test_preexisting_v70_child_is_not_reissued(self):
        with tempfile.TemporaryDirectory() as td:
            home=Path(td);parents=fixture(home,false_count=1)
            outputs,_,_=c.build(home);child=next(q for q in outputs['POOL.json']['questions'] if q['root_id']=='root1' and len(q['gold'])==5)
            child['candidate_id']='root1-v70-already-selected'
            save(home/'BASELINE.json',{'questions':parents+[child]})
            outputs,_,_=c.build(home)
            self.assertFalse(any(q['root_id']=='root1' and len(q['gold'])<6 for q in outputs['POOL.json']['questions']))
            family=outputs['ALLOCATION.json']['families'][0]
            self.assertEqual(next(x['reason'] for x in family['shortfalls'] if x['gold_count']==5),'NO_NOVEL_EXACT_STATEMENT_SET')

    def test_two_round_budget_accepts_two_and_rejects_three(self):
        with tempfile.TemporaryDirectory() as td:
            home=Path(td);fixture(home)
            path=home/'assets/family1/GENERATION.COMPLETE.json';complete=c.read(path);complete.update(http_attempts=2,bounded_supplements=1);save(path,complete)
            self.assertEqual(c.build(home)[0]['ALLOCATION.json']['families'][0]['bounded_supplements'],1)
            complete['http_attempts']=3;save(path,complete)
            with self.assertRaisesRegex(ValueError,'family_generation_budget'):c.build(home)

    def test_aggregate_cannot_claim_unread_private_gold(self):
        with tempfile.TemporaryDirectory() as td:
            home=Path(td);fixture(home);run=home/'assets/family1'
            j=c.read(run/'SOURCE_JUDGMENTS.json');j['new_variant_private_artifacts_read_before_save']=False;save(run/'SOURCE_JUDGMENTS.json',j)
            r=c.read(run/'ASSET_REVIEW.json');r['source_judgments_sha256']=c.sha(run/'SOURCE_JUDGMENTS.json');save(run/'ASSET_REVIEW.json',r)
            with self.assertRaisesRegex(ValueError,'final_review_must_disclose_private_aggregation'):c.build(home)

    def test_selected_counterversion_keeps_round_origin(self):
        with tempfile.TemporaryDirectory() as td:
            home=Path(td);old=fixture(home,false_count=1);cfg=c.read(home/'CONFIG.json')
            family=c.load_family(home,cfg['slots'][0],old,{})
            family['variants']['F01'][1]['origin']={'round_id':'supplement','version_id':'public1'}
            candidate=c.enumerate_children(family)[5][0];_,receipt=c.assemble_child(family,candidate)
            counter=next(v for v in receipt['selected_versions'] if v['variant']=='contradicted')
            self.assertEqual(counter['origin'],{'round_id':'supplement','version_id':'public1'})

    def test_receipt_replay_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            home=Path(td);fixture(home);result=c.compose(home);self.assertEqual(c.verify_assembly(home),result)
            path=home/'FAMILY_SIDECAR.json';side=c.read(path);side['members'][0]['role']='child';save(path,side)
            with self.assertRaisesRegex(ValueError,'assembly_bound_file_changed'):c.verify_assembly(home)

if __name__=='__main__':unittest.main()
