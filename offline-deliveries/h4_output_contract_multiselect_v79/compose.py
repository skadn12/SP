"""Compose bounded v79 families from six immutable parent facts and reviewed alternatives."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import random

VERSION='v79_immutable_parent_family_composition_v1'
MODEL='Qwen/Qwen3.6-35B-A3B'
COUNTS=(2,3,4,5)
OUTPUTS=('POOL.json','PUBLIC_QUESTIONS.json','ALLOCATION.json','FAMILY_SIDECAR.json')

def require(ok,reason):
    if not ok: raise ValueError(reason)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,x):
    with Path(path).open('x',encoding='utf-8') as f: f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def norm(x): return ' '.join(x.casefold().split())
def statement_key(q): return (q['root_id'],q['question_type'],tuple(sorted(norm(x) for x in q['options'].values())))
def _rows(x): return x if isinstance(x,list) else x.get('questions',x.get('candidates',[]))
def _bound(inputs,path):
    path=Path(path).resolve();require(path.is_file(),'missing_input:'+str(path));value=sha(path)
    require(str(path) not in inputs or inputs[str(path)]==value,'changed_input:'+str(path))
    inputs[str(path)]=value;return read(path)
def _binding(x,key,path): require(x.get(key)==sha(path),'binding_mismatch:'+key)
def indexed(rows,key,name):
    require(isinstance(rows,list) and all(isinstance(r,dict) and key in r for r in rows),name+'_invalid')
    out={r[key]:r for r in rows};require(len(out)==len(rows),name+'_duplicates');return out

def load_family(home,slot,baseline,inputs):
    fid=slot.get('family_id',slot['slot_id']);run=home/'assets'/fid
    import asset_generation
    asset_generation.verify(run,complete=True)  # Offline replay includes round reviews and immutable merge.
    round_provenance=_bound(inputs,run/'ROUND_PROVENANCE.json')
    parent=_bound(inputs,run/'PARENT.json');source=_bound(inputs,run/'SOURCE.json')
    manifest=_bound(inputs,run/'MANIFEST.json');complete=_bound(inputs,run/'GENERATION.COMPLETE.json')
    require(slot['slot_id']==fid and parent['root_id']==slot['root_id']==source['root_id'],'family_identity_mismatch')
    require(parent['candidate_id']==slot['parent_candidate_id'],'parent_id_changed')
    parent_pool=_rows(_bound(inputs,slot['parent_pool_path'])) if slot.get('parent_pool_path') else baseline
    originals=[q for q in parent_pool if q['candidate_id']==parent['candidate_id']]
    require(len(originals)==1 and originals[0]==parent,'parent_not_exact_frozen_question')
    require(parent['question_type']=='multi_select' and set(parent['options'])==set('ABCDEF') and set(parent['gold'])==set('ABCDEF') and len(parent['gold'])==6,'parent_not_six_gold')
    require(parent['source_sha256']==sha(run/'SOURCE.json'),'parent_source_bytes_changed')
    _bound(inputs,parent['source_path']);_bound(inputs,slot['history_path'])
    require(sha(parent['source_path'])==parent['source_sha256'] and read(slot['history_path'])['turns']==source['turns'],'parent_complete_source_changed')
    for obj in (manifest,complete):
        require(obj.get('root_id')==slot['root_id'] and obj.get('family_id',obj.get('slot_id'))==fid,'asset_identity_mismatch')
        require(obj.get('training_allowed') is False,'training_enabled')
        require(isinstance(obj.get('files'),list),'generation_bindings_missing')
        for b in obj['files']:
            require(sha(b['path'])==b['sha256'],'generation_binding_changed:'+b['path'])
            inputs[str(Path(b['path']).resolve())]=b['sha256']
    require(type(complete.get('http_attempts')) is int and 0<=complete['http_attempts']<=2 and complete.get('semantic_retries',0)==0,'family_generation_budget')
    require(complete.get('bounded_supplements',0) in (0,1),'family_supplement_budget')
    if complete.get('status')=='NO_VALID_ASSETS':
        require(not any((run/n).exists() for n in ('ASSETS.json','PUBLIC_ASSETS.json','VERSION_MAP.json')),'failed_generation_has_assets')
        return {'slot':slot,'family_id':fid,'parent':parent,'variants':{f'F{i:02d}':[] for i in range(1,7)},'supports_valid':False,'input_dir':str(run),'generation_status':'NO_VALID_ASSETS','generation_http_attempts':complete['http_attempts'],'bounded_supplements':complete.get('bounded_supplements',0)}
    assets=_bound(inputs,run/'ASSETS.json');public=_bound(inputs,run/'PUBLIC_ASSETS.json')
    mapping=_bound(inputs,run/'VERSION_MAP.json');judgment=_bound(inputs,run/'SOURCE_JUDGMENTS.json');review=_bound(inputs,run/'ASSET_REVIEW.json')
    for obj in (assets,public,mapping,judgment,review): _binding(obj,'source_sha256',run/'SOURCE.json')
    for obj in (public,mapping,review): _binding(obj,'assets_sha256',run/'ASSETS.json')
    for obj in (mapping,judgment,review): _binding(obj,'public_assets_sha256',run/'PUBLIC_ASSETS.json')
    _binding(review,'source_judgments_sha256',run/'SOURCE_JUDGMENTS.json')
    require(assets.get('family_id')==fid and assets.get('root_id')==parent['root_id'] and assets.get('parent_candidate_id')==parent['candidate_id'],'asset_parent_mismatch')
    _binding(assets,'parent_sha256',run/'PARENT.json')
    require(assets.get('public_stem')==parent['question']==public.get('public_stem'),'parent_stem_changed')
    require(public.get('contains_relations') is False and public.get('contains_solver_scores') is False,'public_private_leak')
    derived=(judgment.get('review_mode')=='derived_from_saved_round_source_judgments' and judgment.get('derived_from_rounds') is True
        and judgment.get('source_first_per_round') is True and review.get('review_mode')=='derived_round_private_crosschecks'
        and review.get('source_first_per_round') is True)
    require(derived and judgment.get('new_variant_private_artifacts_read_before_save') is True,
        'final_review_must_disclose_private_aggregation')
    require(judgment.get('solver_scores_used') is False and review.get('source_first') is True
        and review.get('solver_scores_used') is False,'review_not_source_first')
    require(review.get('provenance_rounds')==round_provenance.get('rounds'), 'round_provenance_review_disagreement')
    facts=indexed(assets.get('facts'),'fact_id','facts');expected={f'F{i:02d}':l for i,l in enumerate('ABCDEF',1)}
    require(set(facts)==set(expected),'family_requires_fixed_six_facts')
    parents=indexed(public.get('parent_facts'),'fact_id','public_parents')
    jp=indexed(judgment.get('parents'),'fact_id','parent_judgments');rp=indexed(review.get('parents'),'fact_id','parent_reviews')
    require(set(parents)==set(jp)==set(rp)==set(facts),'parent_review_coverage')
    pv=indexed(public.get('versions'),'version_id','public_versions');mv=indexed(mapping.get('versions'),'version_id','mapping_versions')
    jv=indexed(judgment.get('versions'),'version_id','version_judgments');rv=indexed(review.get('versions'),'version_id','version_reviews')
    require(set(pv)==set(mv)==set(jv)==set(rv),'version_review_coverage')
    require(all(set(v)=={'version_id','fact_id','proposition'} for v in pv.values()),'public_version_private_leak')
    turns={t['turn_ref']:t for t in source['turns']}
    def exact(anchor):
        require(set(anchor)=={'turn_ref','speaker','quote'},'bad_anchor_fields')
        t=turns.get(anchor['turn_ref']);require(t and t['speaker']==anchor['speaker'] and anchor['quote'] and anchor['quote'] in t['text'],'anchor_not_exact')
    def verified(j,r,relation):
        require(j.get('relation') in ('supports','contradicts','unknown') and r.get('relation')==j['relation'] and type(r.get('usable')) is bool,'invalid_review_relation')
        require(type(j.get('scope_valid')) is bool and type(j.get('fact_distinct')) is bool,'missing_review_scope')
        for c in j.get('citations',[]): exact(c)
        if r['usable']:
            require(j['relation']==relation and j['scope_valid'] and j['fact_distinct'],'usable_false_or_unverified_label')
            require(judgment.get('stem_valid') is True and review.get('stem_valid') is True,'usable_invalid_stem')
        return r['usable']
    variants={};supports_valid=True
    for fact_id,label in expected.items():
        f=facts[fact_id];s=f['supported']
        require(f['parent_label']==label and s['proposition']==parent['options'][label] and s['relation']=='supports','parent_fact_text_changed')
        require(parents[fact_id]=={'fact_id':fact_id,'parent_label':label,'proposition':s['proposition']},'public_parent_changed')
        original=parent['option_evidence'][label]
        require(s['anchor']=={k:original[k] for k in ('turn_ref','speaker','quote')},'parent_anchor_changed')
        require(s['rationale']==original.get('rationale',original.get('reason')),'parent_rationale_changed')
        exact(s['anchor']);ok=verified(jp[fact_id],rp[fact_id],'supports');supports_valid &= ok
        variants[fact_id]=[{**deepcopy(s),'fact_id':fact_id,'variant':'supported','version_id':fact_id+':parent'}] if ok else []
    mapped=set()
    for vid,m in mv.items():
        fact_id=m['fact_id'];require(fact_id in facts and fact_id not in mapped,'contradiction_mapping_not_one_per_fact');mapped.add(fact_id)
        c=facts[fact_id]['contradicted'];require(isinstance(c,dict) and c['relation']=='contradicts' and m.get('variant')=='contradicted' and m.get('relation')=='contradicts','contradiction_mapping_changed')
        require(pv[vid]['fact_id']==fact_id and pv[vid]['proposition']==c['proposition'],'contradiction_public_changed')
        exact(c['anchor'])
        if verified(jv[vid],rv[vid],'contradicts'):
            origin=facts[fact_id].get('origin',c.get('origin'))
            variants[fact_id].append({**deepcopy(c),'fact_id':fact_id,'variant':'contradicted','version_id':vid,**({'origin':deepcopy(origin)} if origin else {})})
    require(mapped=={f for f,v in facts.items() if isinstance(v.get('contradicted'),dict)},'contradiction_mapping_coverage')
    return {'slot':slot,'family_id':fid,'parent':parent,'variants':variants,'supports_valid':supports_valid,'input_dir':str(run),'generation_status':complete.get('status','ASSETS_READY'),'generation_http_attempts':complete['http_attempts'],'bounded_supplements':complete.get('bounded_supplements',0)}

def enumerate_children(family,prior=()):
    result={k:[] for k in COUNTS}
    if not family['supports_valid']: return result
    seen=set(prior)
    for versions in itertools.product(*(family['variants'][fid] for fid in sorted(family['variants']))):
        k=sum(v['relation']=='supports' for v in versions)
        if k not in result: continue
        key=(family['parent']['root_id'],'multi_select',tuple(sorted(norm(v['proposition']) for v in versions)))
        if len(set(key[2]))!=6 or key in seen: continue
        seen.add(key);result[k].append({'versions':versions,'key':key,'id':digest(key)})
    for candidates in result.values(): candidates.sort(key=lambda c:c['id'])
    return result

def gold_blind_permutation(family_id,versions):
    seed=int(digest({'family_id':family_id,'statements':sorted(v['proposition'] for v in versions)})[:16],16)
    # Initial order and random seed use public fact identity and text, never relations.
    ordered=sorted(versions,key=lambda v:(v['fact_id'],v['proposition']))
    random.Random(seed).shuffle(ordered)
    return seed,ordered

def assemble_child(family,candidate):
    parent=family['parent']
    inherited=('root_id','question_type','topic','question','source_path','source_sha256','mechanical_gate_version','citation_gate','source_training_pool_eligible','source_semantic_certified','source_admission_status','upstream_scene_promotion','training_target','loss_weight_from_partial_score')
    q={k:deepcopy(parent[k]) for k in inherited if k in parent}
    q.update(training_allowed=False,semantic_review_performed=True,raw_question_content_preserved=True)
    seed,versions=gold_blind_permutation(family['family_id'],candidate['versions'])
    q['candidate_id']=q['root_id']+'-v79-'+family['family_id']+'-'+candidate['id'][:12]
    q['options']={};q['gold']=[];q['option_evidence']={};selected=[]
    for label,v in zip('ABCDEF',versions):
        q['options'][label]=v['proposition']
        q['option_evidence'][label]={**deepcopy(v['anchor']),'relation':v['relation'],'rationale':v['rationale'],'reason':v['rationale']}
        if v['relation']=='supports': q['gold'].append(label)
        selected.append({'label':label,'fact_id':v['fact_id'],'version_id':v['version_id'],'variant':v['variant'],'verified_relation':v['relation'],**({'origin':deepcopy(v['origin'])} if 'origin' in v else {})})
    q['answer_reason']='Gold is the complete set of independently source-verified supported statements.'
    q['generator_model']=MODEL
    return q,{'candidate_id':q['candidate_id'],'permutation_seed':seed,'selected_versions':selected,'gold_blind_permutation':True}

def build(report):
    home=Path(report).resolve();inputs={};cfg=_bound(inputs,home/'CONFIG.json')
    require(cfg.get('training_allowed') is False and cfg.get('max_questions',20)<=20 and cfg.get('max_calls_per_model',360)<=360,'budget_or_training_changed')
    slots=cfg['slots'];n=len(slots);require(2<=n<=4 and len({s['root_id'] for s in slots})==n and len({s['slot_id'] for s in slots})==n and cfg.get('family_count',n)==n,'requires_distinct_root_families')
    expected_families={home/'assets'/s['slot_id'] for s in slots}
    require(set((home/'assets').iterdir())==expected_families and all(p.is_dir() for p in expected_families),
        'unregistered_or_missing_executed_family')
    baseline=_rows(_bound(inputs,cfg['baseline_pool']))
    if cfg.get('baseline_result_bindings'): _bound(inputs,cfg['baseline_result_bindings'])
    families=[load_family(home,s,baseline,inputs) for s in slots]
    require(sum(f['generation_http_attempts'] for f in families)<=cfg.get('max_http',2*n)==2*n,'total_generation_budget')
    prior={statement_key(q) for q in baseline if q.get('question_type')=='multi_select'}
    for slot in slots:
        for path in slot.get('prior_pool_paths',[]):
            prior|={statement_key(q.get('original_selected_question',q)) for q in _rows(_bound(inputs,path)) if q.get('original_selected_question',q).get('question_type')=='multi_select'}
    targets={str(k):n for k in COUNTS};configured=cfg.get('target_distribution',targets)
    require(all(configured.get(str(k))==n for k in COUNTS) and configured.get('6',0)==0,'child_targets_must_equal_family_count')
    questions=[];receipts=[];family_rows=[];members=[];used=set(prior)
    for f in families:
        parent=deepcopy(f['parent']);questions.append(parent)
        common={'family_id':f['family_id'],'root_id':parent['root_id'],'train_root_id':parent['root_id'],'split_group':parent['root_id'],'parent_candidate_id':parent['candidate_id'],'parent_question_digest':digest(parent),'parent_origin':f['slot'].get('parent_origin','existing_frozen_six_gold_question'),'parent_pool_path':f['slot'].get('parent_pool_path',cfg['baseline_pool']),'fixed_fact_ids':sorted(f['variants'])}
        members.append({**common,'candidate_id':parent['candidate_id'],'role':'control6','gold_count':6,'new_candidate':False})
        receipts.append({'candidate_id':parent['candidate_id'],'family_id':f['family_id'],'role':'control6','original_parent_unchanged':True,'parent_question_digest':digest(parent)})
        used.add(statement_key(parent))
        available=enumerate_children(f,used);counts={str(k):len(available[k]) for k in COUNTS};children=[];missing=[]
        false_count=sum(any(v['relation']=='contradicts' for v in vv) for vv in f['variants'].values())
        for k in COUNTS:
            if not available[k]:
                missing.append({'gold_count':k,'required_false_facts':6-k,'verified_false_facts':false_count,'reason':('NO_VALID_GENERATED_ASSETS' if f['generation_status']=='NO_VALID_ASSETS' else 'PARENT_SUPPORT_NOT_FULLY_VERIFIED' if not f['supports_valid'] else 'INSUFFICIENT_VERIFIED_FALSE_FACTS' if false_count<6-k else 'NO_NOVEL_EXACT_STATEMENT_SET')});continue
            q,receipt=assemble_child(f,available[k][0]);require(statement_key(q) not in used,'family_child_exact_duplicate');used.add(statement_key(q));questions.append(q)
            receipts.append({**receipt,'family_id':f['family_id'],'role':'child','parent_candidate_id':parent['candidate_id'],'actual_gold_count':k})
            members.append({**common,'candidate_id':q['candidate_id'],'role':'child','gold_count':k,'new_candidate':True})
            children.append({'candidate_id':q['candidate_id'],'gold_count':k})
        family_rows.append({**common,'control_candidate_id':parent['candidate_id'],'generation_status':f['generation_status'],'generation_http_attempts':f['generation_http_attempts'],'bounded_supplements':f['bounded_supplements'],'round_provenance_path':str(Path(f['input_dir'])/'ROUND_PROVENANCE.json'),'parent_supports_verified':f['supports_valid'],'verified_false_facts':false_count,'feasible_candidate_counts':counts,'children':children,'shortfalls':missing})
    require(len(questions)<=cfg['max_questions'] and len({q['candidate_id'] for q in questions})==len(questions),'question_budget_or_duplicate_id')
    hist=Counter(len(q['gold']) for q in questions if len(q['gold'])<6)
    pool={'version':VERSION,'questions':questions,'candidate_count':len(questions),'training_allowed':False,'provenance':{'requested_model':MODEL,'composition':'immutable_parent_six_facts_with_verified_alternatives','config_sha256':sha(home/'CONFIG.json')}}
    public={'questions':[{k:deepcopy(q[k]) for k in ('candidate_id','root_id','question_type','topic','question','options','source_path','source_sha256') if k in q} for q in questions],'candidate_count':len(questions),'training_allowed':False}
    allocation={'version':VERSION,'target_distribution':targets,'control_target_count':n,'control_count':n,'assembled_child_count':len(questions)-n,'assembled_distribution':{str(k):hist[k] for k in COUNTS},'missing_target_counts':{str(k):n-hist[k] for k in COUNTS},'families':family_rows,'slots':family_rows,'extra_model_calls':0,'training_allowed':False,'exact_dedup':'same root/type/sorted whole statement set; casefold and whitespace only; controls intentionally retained','distribution_stage':'assembled_before_content_review_and_difficulty_selection','parent_controls_are_new_candidates':False}
    sidecar={'version':VERSION,'members':members,'family_count':n,'same_family_same_train_root':True,'siblings_are_independent_samples':False,'controls_modify_baseline_admission':False}
    return dict(zip(OUTPUTS,(pool,public,allocation,sidecar))),inputs,receipts

def compose(report):
    home=Path(report).resolve();require(not any((home/n).exists() for n in (*OUTPUTS,'ASSEMBLY_RECEIPTS.json')),'assembly_already_exists_use_verify')
    outputs,inputs,receipts=build(home)
    for name,value in outputs.items():write(home/name,value)
    write(home/'ASSEMBLY_RECEIPTS.json',{'version':VERSION,'config_sha256':sha(home/'CONFIG.json'),'inputs':[{'path':p,'sha256':v} for p,v in sorted(inputs.items())],'outputs':[{'path':str(home/n),'sha256':sha(home/n)} for n in OUTPUTS],'selections':receipts,'training_allowed':False})
    return verify_assembly(home)

def verify_assembly(report):
    home=Path(report).resolve();r=read(home/'ASSEMBLY_RECEIPTS.json');require(r.get('version')==VERSION and r.get('training_allowed') is False,'assembly_contract_changed');_binding(r,'config_sha256',home/'CONFIG.json')
    for group in ('inputs','outputs'):
        for b in r[group]: require(Path(b['path']).is_file() and sha(b['path'])==b['sha256'],'assembly_bound_file_changed:'+b['path'])
    outputs,inputs,selections=build(home)
    require(r['inputs']==[{'path':p,'sha256':v} for p,v in sorted(inputs.items())],'assembly_input_coverage_changed')
    require(r['outputs']==[{'path':str(home/n),'sha256':sha(home/n)} for n in OUTPUTS],'assembly_output_coverage_changed')
    require(r['selections']==selections,'assembly_selections_changed')
    for name,value in outputs.items():require(read(home/name)==value,'assembly_replay_changed:'+name)
    return outputs['POOL.json']

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('report',type=Path);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    result=verify_assembly(args.report) if args.verify else compose(args.report)
    print(json.dumps({'status':'PASS','candidate_count':result['candidate_count']}))
