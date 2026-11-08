#!/usr/bin/env python3
"""CPU-only evidence and scoring checks; never loads a real experiment scene."""
import copy,importlib.util,json,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('score_pilot',HERE/'score_pilot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOTS=[f'root{i}' for i in range(6)]
PROTOCOL={'selection':{'root_ids':ROOTS}}
WORDS=['Amber','Birch','Copper','Dahlia','Elm','Flint','Garnet','Hazel','Indigo','Jade','Kestrel','Lilac']
def put(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o));return m.bind(p)
def make(work,counts,unknown_initial=True):
    for ri,root in enumerate(ROOTS):
        source={'root_id':root,'source_id':'synthetic:'+root,'turns':[{'turn_ref':f'T{j}','session_ref':f'S{j}','speaker':'A' if j%2 else 'B','text':text} for j,text in enumerate(['I once kept a blue notebook.','We can look at the calendar.','The paper is on this desk.'])]}
        initial=put(m.original_source(work,root),source)
        d=work/'initial_memory'/root;snap=put(d/'FACT_MEMORY_SNAPSHOT.json',{'root_id':root,'source_sha256':initial['sha256'],'facts':[{'fact_id':'f1'}]});put(d/'SEMANTIC_REVIEW.json',{'root_id':root,'decision':'PASS','snapshot_sha256':snap['sha256'],'full_source_sha256':initial['sha256'],'all_facts_reviewed':True,'accepted_fact_ids':['f1'],'coverage_status':'BOUNDED'});put(d/'ROLE_CALL.json',{'actual_usage':None if unknown_initial else {'input_tokens':1,'output_tokens':1},'actual_model_call_wall_seconds':None if unknown_initial else .01})
        for arm in m.ARMS:
            prefix=initial;prefix_object=copy.deepcopy(source)
            for i in range(1,4):
                sd=work/'pilot/branches'/f'{root}__{arm}'/f'scene_{i:02d}';n=counts[arm][ri]
                if i<=n:
                    opaque=m.hashlib.sha256((root+'|'+arm).encode()).hexdigest()[:16]
                    turns=[{'speaker':'A' if j%2 else 'B','text':(word+' ')*3+(['clay','river','mountain'][i-1]+' ')*3} for j,word in enumerate(WORDS)]
                    candidate={'turns':turns};cand=put(sd/'candidate.json',candidate)
                    rev=put(sd/'review.json',{'root_id':root,'branch_id':opaque,'scene_index':i,'candidate_id':opaque+':'+str(i),'candidate_sha256':cand['sha256'],'accepted_prefix_sha256':prefix['sha256'],'decision':'PASS','critical_major_error_count':0,'new_progress_count':1,'duplicate_turn_ratio':0,'old_history_linkage_count':1,'all_plausible_prior_states_checked':True,'evidence_refs':[{'kind':'new_progress','candidate_turn_index':0,'quote':turns[0]['text'],'prefix_turn_index':None,'prefix_quote':None},{'kind':'old_history_linkage','candidate_turn_index':1,'quote':turns[1]['text'],'prefix_turn_index':0,'prefix_quote':source['turns'][0]['text']}]})
                    output=copy.deepcopy(prefix_object);session='D2P:'+opaque+':S'+str(i)
                    output['turns'].extend({'turn_ref':session+':T'+str(j).zfill(3),'session_ref':session,'scene_ref':session,**t} for j,t in enumerate(turns));out=put(sd/'prefix.json',output)
                    row={'root_id':root,'arm':arm,'scene_index':i,'status':'PASS','reason':'synthetic','candidate_binding':cand,'accepted_prefix_binding':prefix,'review_binding':rev,'metrics':{'new_progress_count':1,'duplicate_turn_ratio':0,'old_history_linkage_count':1,'critical_major_error_count':0},'accepted_output_prefix_binding':out};prefix=out;prefix_object=output
                    if arm==m.MEM and i<3:
                        mb=put(sd/'memory.json',{'root_id':root,'source_sha256':out['sha256'],'facts':[{'fact_id':'m1'}]});rb=put(sd/'memory_review.json',{'root_id':root,'decision':'PASS','snapshot_sha256':mb['sha256'],'full_source_sha256':out['sha256'],'all_facts_reviewed':True,'accepted_fact_ids':['m1'],'coverage_status':'BOUNDED'});put(sd/'MEMORY_GATE.json',{'decision':'PASS','snapshot_binding':mb,'review_binding':rb,'accepted_prefix_binding':out})
                    put(sd/'SCENE_RECEIPT.json',row)
                    for role in ('planner','realizer','review'):
                        api=sd/role/'api';put(api/'SUBMITTED.json',{'payload':{'model':'synthetic'}});put(api/'COMPLETE.json',{'input_tokens':50 if arm==m.MEM else 100,'output_tokens':10,'wall_seconds':.5 if arm==m.MEM else 1,'returned_model':'synthetic'})
                else:put(sd/'SCENE_RECEIPT.json',{'root_id':root,'arm':arm,'scene_index':i,'status':'MISSING','reason':'STOPPED','candidate_binding':None,'accepted_prefix_binding':None,'review_binding':None,'metrics':None,'accepted_output_prefix_binding':None})
def modify_review(work,change):
    sd=work/'pilot/branches/root0__recent2/scene_01';rp=sd/'review.json';r=m.load(rp);change(r);rb=put(rp,r);p=sd/'SCENE_RECEIPT.json';receipt=m.load(p);receipt['review_binding']=rb;put(p,receipt)
def run():
    results=[]
    with tempfile.TemporaryDirectory(prefix='memory_score_cpu_') as tmp:
        w=Path(tmp)/'work';allpass={a:[3]*6 for a in m.ARMS};make(w,allpass)
        s=m.score(w,PROTOCOL);assert len(s['receipts'])==54 and all(s['arms'][a]['accepted_initial_scenes']==18 for a in m.ARMS),[(r['root_id'],r['arm'],r['scene_index'],r['reason']) for r in s['receipts'] if r['status']!='PASS'];assert s['quality_gates_all_pass'] and s['recent_arm_ceiling_case'];assert s['cost_status']=='NO_COST_CONCLUSION' and not s['memory_adopted'] and s['selected_context_policy']=='full_history';results.append('unknown_initial_cost_blocks_even_perfect_quality')
        cfile=w/'pilot/branches/root0__recent2/scene_01/planner/api/COMPLETE.json';cfile.unlink();put(cfile.parent/'FAILED.json',{'wall_seconds':2,'usage':{'input_tokens':123,'output_tokens':7}});s=m.score(w,PROTOCOL);assert s['arms']['recent2']['api_cost']['failed_attempts']==1;assert s['arms']['recent2']['api_cost']['known_input_tokens']==5423;results.append('failed_attempt_actual_usage_counted')
        f=w/'pilot/branches/root0__recent2/scene_01/SCENE_RECEIPT.json';r=m.load(f);r['metrics']['new_progress_count']=0;put(f,r);s=m.score(w,PROTOCOL);assert s['arms']['recent2']['accepted_initial_scenes']==15 and s['arms']['recent2']['status_counts']['HOLD']==3;results.append('invalid_pass_and_descendants_hold_fixed_denominator')
        f=w/'pilot/branches/root0__full_history/scene_03/SCENE_RECEIPT.json';f.unlink();s=m.score(w,PROTOCOL);assert len(s['receipts'])==54 and s['arms']['full_history']['status_counts']['MISSING']==1 and s['status']=='INCOMPLETE_FIXED_DENOMINATOR';results.append('absent_receipt_keeps54_denominator')
    with tempfile.TemporaryDirectory(prefix='memory_score_cpu_') as tmp:
        w=Path(tmp)/'work';counts={'recent2':[2,2,3,3,3,3],m.MEM:[3,3,3,3,3,3],'full_history':[3,3,3,3,3,3]};make(w,counts,unknown_initial=False);s=m.score(w,PROTOCOL);assert s['quality_gates_all_pass'] and not s['recent_arm_ceiling_case'];assert s['cost_status']=='COST_COMPARISON_AVAILABLE' and s['memory_adopted'];results.append('complete_recorded_cost_all_frozen_thresholds_can_adopt_synthetic_only')
        f=w/'pilot/branches/root0__recent2_plus_prefix_memory/scene_01/MEMORY_GATE.json';f.unlink();s=m.score(w,PROTOCOL);assert not s['quality_gates']['all_used_maintenance_memory_semantic_pass'] and not s['memory_adopted'];results.append('missing_independent_maintenance_hash_gate_blocks_adoption')
    mutations=[('false_evidence_quote_rejected',lambda r:r['evidence_refs'][0].update(quote='absent quotation')),('recent_quote_cannot_count_as_old_history',lambda r:r['evidence_refs'][1].update(prefix_turn_index=2,prefix_quote='The paper is on this desk.')),('mismatched_review_candidate_hash_rejected',lambda r:r.update(candidate_sha256='0'*64))]
    for name,change in mutations:
        with tempfile.TemporaryDirectory(prefix='memory_score_cpu_') as tmp:
            w=Path(tmp)/'work';make(w,{a:[3]*6 for a in m.ARMS});modify_review(w,change);s=m.score(w,PROTOCOL);assert s['arms']['recent2']['accepted_initial_scenes']==15;s0=next(r for r in s['receipts'] if r['root_id']=='root0' and r['arm']=='recent2' and r['scene_index']==1);assert s0['reason'].startswith('PASS_ARTIFACT_VALIDATION_FAILED:');results.append(name)
    with tempfile.TemporaryDirectory(prefix='memory_score_cpu_') as tmp:
        w=Path(tmp)/'work';make(w,{a:[3]*6 for a in m.ARMS});sd=w/'pilot/branches/root0__recent2/scene_01';p=sd/'SCENE_RECEIPT.json';r=m.load(p);candidate=m.load(sd/'candidate.json');repeated=m.load(r['accepted_prefix_binding']['path'])['turns'][0]['text'];candidate['turns']=[{'speaker':'B','text':repeated} for _ in range(12)];r['candidate_binding']=put(sd/'candidate.json',candidate);review=m.load(sd/'review.json');review['candidate_sha256']=r['candidate_binding']['sha256'];r['review_binding']=put(sd/'review.json',review);put(p,r);s=m.score(w,PROTOCOL);s0=next(v for v in s['receipts'] if v['root_id']=='root0' and v['arm']=='recent2' and v['scene_index']==1);assert 'RECOMPUTED_DUPLICATION_GATE' in s0['reason'];results.append('stored_zero_duplication_cannot_hide_repeated_scene')
    print(json.dumps({'status':'PASS','check_count':len(results),'checks':results,'live_calls':0},ensure_ascii=False,indent=2))
if __name__=='__main__':run()
