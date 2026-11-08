"""One immutable-parent replacement request per family; no repairs or retries."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path
import re
import secrets
import sys

HERE=Path(__file__).resolve().parent
V63=HERE.parent/'h4_option_composition_v63_qa'
V69=HERE.parent/'h4_balanced_234_v69'
_saved_path=sys.path[:]
sys.path.insert(0,str(V63))
_spec=importlib.util.spec_from_file_location('v70_readonly_v63_generation',V63/'generate.py')
legacy=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(legacy)
import qa_transport
sys.path[:]=_saved_path
_spec=importlib.util.spec_from_file_location('v70_readonly_old_feedback_adapter',V69/'source_adapter.py')
source_adapter=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(source_adapter)
base,v15=legacy.base,legacy.v15
read,write,require=base.read,base.write,base.require
VERSION='v70_immutable_parent_replacements_v1'
CAP=6144
START,END='[CURRENT_PARENT_REPLACEMENT_TASK_V70]','[/CURRENT_PARENT_REPLACEMENT_TASK_V70]'
INTRO=('Read the complete exact parent source, public historical inventory and frozen previous-training-round feedback. '
       'The current task follows that feedback. Source metadata and candidate status are preserved; no expansion or training admission is granted.\n')
TASK="""
CURRENT TASK: For each of the SIX FIXED parent statements, create at most ONE
contradicted alternative. Do not generate supported statements, final questions,
answer counts or Gold. The original statements and stem are immutable. Preserve
the same actor, event, time, condition, report modality and action state. Prefer a
minimal attribute mutation whose incompatible value the full source explicitly
fixes. Preserve tentative, approximate, proposed and conditional qualifiers.
The entire source, including corrections and later outcomes, governs truth.
A different additional activity can coexist; absence or unknown is not false.
Never invent only/never or treat a plan as refuting later completion. Keep full
scope explicit in the replacement itself, with the parent stem framing retained.
No public contradicted proposition may contain gendered pronoun WORD TOKENS:
he, she, him, her, his, hers, himself, herself (including contractions).
Use explicit source names or they/them/their. This lexical rule applies only to
new public propositions: exact SOURCE citation quotes may contain pronouns.
Copy turn_ref, speaker and a nonempty contiguous quote exactly from SOURCE.
Never normalize reference aliases. The citation must explicitly counter the
replacement in the same scope. Independent source-first review will determine
truth; a syntactically valid alternative is not certified false by generation.
If no explicit same-scope contradiction exists, set contradicted and anchor to
null and give a specific nonempty contradiction_unavailable_reason. For a nonnull
contradicted, set reason null and provide the exact anchor. Never rewrite parent
positive text to make a contradiction easier. Historical feedback may discuss
old final questions; obey this current schema. Do not mention a target answer
count. Return JSON ONLY with exactly root_id, family_id, replacements. Return
exactly F01 through F06 in that order, one entry per fixed fact, with no extras:
{root_id,family_id,replacements:[{fact_id,anchor:null|{turn_ref,speaker,quote},
contradicted:null|{proposition,relation:"contradicts",rationale},
contradiction_unavailable_reason:null|string}]}.
"""
CONTRACT={'version':VERSION,'fixed_parent_facts':6,'max_alternatives_per_fact':1,
 'max_tokens':CAP,'http_attempts_max_per_family':1,'mechanical_repair_max':0,
 'semantic_retries':0,'transport_retries':0,'full_source_feedback_prior':True,
 'current_solver_feedback':False,'target_count_in_writer_input':False,
 'supported_stem_source_gold_immutable':True,'source_mode':'parent_exact_full_source',
 'gendered_public_pronouns_forbidden':True,'citation_quotes_exempt':True,'training_allowed':False}
FORBIDDEN=re.compile(r"(?<![\w])(?:he|she|him|her|his|hers|himself|herself)(?![\w])",re.IGNORECASE)
FACT_IDS=[f'F{i:02d}' for i in range(1,7)]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def binding(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def _dependencies():
    provider,paths=legacy._dependency_paths()
    return provider,set(paths)|set(HERE.glob('*.py'))|{V69/'source_adapter.py'}
def _rows(x):return x if isinstance(x,list) else x.get('questions',x.get('candidates',[]))
def get_slot(config,slot_id):
    slots=config.get('slots')
    require(isinstance(slots,list) and len(slots)==2 and len({s.get('root_id') for s in slots})==2
        and len({s.get('slot_id') for s in slots})==2,'requires_two_distinct_root_families')
    require(config.get('training_allowed') is False and config.get('max_http')==2
        and 0<config.get('max_questions',0)<=10 and 0<config.get('max_calls_per_model',0)<=180,'global_budget_changed')
    found=[s for s in slots if s.get('slot_id')==slot_id]
    require(len(found)==1 and re.fullmatch(r'[A-Za-z0-9_-]{1,128}',slot_id),'unknown_family')
    slot=deepcopy(found[0]);require(slot.get('family_id',slot_id)==slot_id,'family_slot_mismatch')
    slot['family_id']=slot_id
    return slot

def parent_source(slot):
    pool=Path(slot['parent_pool_path']).resolve()
    parents=[q for q in _rows(read(pool)) if q.get('candidate_id')==slot['parent_candidate_id']]
    require(len(parents)==1,'parent_identity_missing_or_duplicate');parent=deepcopy(parents[0])
    require(parent.get('root_id')==slot['root_id'] and parent.get('question_type')=='multi_select'
        and set(parent.get('options',{}))==set('ABCDEF') and len(parent.get('gold',[]))==6
        and set(parent['gold'])==set('ABCDEF'),'parent_not_fixed_six_gold')
    require(isinstance(parent.get('question'),str) and parent['question'].strip(),'parent_stem_invalid')
    source_path=Path(parent['source_path']).resolve();history=Path(slot['history_path']).resolve()
    require(sha(source_path)==parent['source_sha256'],'parent_source_binding_changed')
    for key,path in [('history',history),('source',source_path)]:
        require(not slot.get(key+'_sha256') or sha(path)==slot[key+'_sha256'],'configured_'+key+'_changed')
    require(not slot.get('source_path') or Path(slot['source_path']).resolve()==source_path,'configured_source_path_changed')
    source=read(source_path);turns=source.get('turns')
    require(source.get('root_id')==slot['root_id'] and isinstance(turns,list) and bool(turns),'source_root_or_turns_invalid')
    require(all(isinstance(t,dict) and all(isinstance(t.get(k),str) and t[k].strip() for k in ('turn_ref','speaker','text')) for t in turns),'source_turn_schema')
    require(len({t['turn_ref'] for t in turns})==len(turns),'duplicate_source_refs')
    require(read(history).get('turns')==turns,'full_history_turns_changed')
    lookup={t['turn_ref']:t for t in turns}
    for label in 'ABCDEF':
        require(isinstance(parent['options'][label],str) and parent['options'][label].strip(),'parent_text_invalid')
        ev=parent.get('option_evidence',{}).get(label,{})
        exact_anchor({k:ev.get(k) for k in ('turn_ref','speaker','quote')},lookup)
        require(isinstance(ev.get('rationale',ev.get('reason')),str) and ev.get('rationale',ev.get('reason')).strip(),'parent_rationale_missing')
    return parent,source,source_path,{pool,source_path,history}

def exact_anchor(anchor,turns):
    require(isinstance(anchor,dict) and set(anchor)=={'turn_ref','speaker','quote'}
        and all(isinstance(x,str) and x.strip() for x in anchor.values()),'anchor_schema')
    turn=turns.get(anchor['turn_ref'])
    require(turn is not None and anchor['speaker']==turn['speaker'] and anchor['quote'] in turn['text'],'anchor_not_exact')

def prepare(config_path,slot_id,*,run=None):
    config_path=Path(config_path).resolve();slot=get_slot(read(config_path),slot_id)
    run=Path(run).resolve() if run is not None else HERE/'assets'/slot_id
    require(not run.exists(),'asset_run_already_exists')
    parent,source,source_path,inputs=parent_source(slot)
    feedback=source_adapter.feedback_data(slot['root_id']);lock=feedback['source_lock']
    require(lock.get('status')=='EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY'
        and lock.get('merge_current_generation_feedback') is False
        and lock['model_identity_source']['sha256']==v15.evaluator.TRAINED_AUDIT_SHA256,'feedback_not_frozen_previous_round')
    provider,frozen=_dependencies();frozen|=inputs|{config_path}
    feedback,files=source_adapter.freeze_feedback(slot['root_id'],run/'feedback',data=feedback);frozen.update(files)
    priors,files=source_adapter.prior_candidates(slot['root_id'],feedback,run/'feedback'/slot['root_id']/'candidate_history.json',v15);frozen.update(files)
    for item in slot.get('prior_pool_paths',[]):
        path=Path(item).resolve();priors+=legacy._prior_pool_rows(path,slot['root_id']);frozen.add(path)
    for name,value in [('PARENT.json',parent),('PRIOR.json',priors),('provider_profile.json',provider)]:
        write(run/name,value);frozen.add(run/name)
    with (run/'SOURCE.json').open('xb') as f:f.write(source_path.read_bytes())
    frozen.add(run/'SOURCE.json')
    m={'version':VERSION,'prepared_at_utc':v15.now(),'contract':CONTRACT,'slot':slot,
        'config':binding(config_path),'root_id':slot['root_id'],'slot_id':slot_id,'family_id':slot_id,
        'source_path':str(run/'SOURCE.json'),'source_sha256':sha(run/'SOURCE.json'),
        'parent_sha256':sha(run/'PARENT.json'),'source_history':binding(slot['history_path']),
        'source_turn_count':len(source['turns']),'source_mode':'parent_exact_full_source',
        'source_metadata_preserved':True,'new_expansion_tokens':0,'solver_model':legacy.TARGET,
        'generator_model':legacy.MODEL,'budget':{'http_attempts_max':1},
        'version_id_salt':secrets.token_hex(32),'task_sha256':base.digest(TASK.encode()),
        'training_allowed':False,'formal_training_admission':False,'current_solver_feedback':False,
        'files':base.bindings(frozen)}
    write(run/'MANIFEST.json',m);payload,receipt=make_payload(run)
    write(run/'INITIAL_REQUEST.json',payload);write(run/'INITIAL_PROOF.json',proof_for(m,payload,receipt))
    verify(run)
    return {'slot_id':slot_id,'family_id':slot_id,'run':str(run),'source_turn_count':len(source['turns']),'budget':m['budget']}

def context_for(run):
    run=Path(run);m=read(run/'MANIFEST.json');parent=read(run/'PARENT.json');source=read(run/'SOURCE.json')
    prior=legacy.prompt_prior(read(run/'PRIOR.json'))
    return {'root_id':m['root_id'],'family_id':m['family_id'],'source_turns':source['turns'],
        'source_metadata':{k:v for k,v in source.items() if k!='turns'},
        'required_public_stem':parent['question'],
        'parent_facts':[{'fact_id':fid,'parent_label':label,'proposition':parent['options'][label]} for fid,label in zip(FACT_IDS,'ABCDEF')],
        'previous_candidates_dedup_only':prior,'used_six_statement_sets':legacy.used_statement_sets(prior),
        'source_mode':'parent_exact_full_source','new_expansion_tokens':0,
        'source_training_pool_eligible':False,'formal_training_admission':False}

def make_payload(run):
    run=Path(run);m=read(run/'MANIFEST.json')
    payload,receipt=legacy.legacy.payload_with_feedback(run,m['root_id'],INTRO,context_for(run),CAP)
    intermediate=payload['messages'][0]['content'];tail='\n\n'+START+TASK+END
    payload['messages'][0]['content']+=tail
    receipt['asset_task_tail_receipt']={'kind':'immutable_parent_replacements','task_sha256':base.digest(tail.encode()),
        'intermediate_sha256':base.digest(intermediate.encode()),'final_payload_sha256':base.digest(payload)}
    return payload,receipt

def proof_for(m,payload,receipt):
    return {'role':'asset_generator','root_id':m['root_id'],'slot_id':m['slot_id'],'family_id':m['family_id'],
        'source_sha256':m['source_sha256'],'parent_sha256':m['parent_sha256'],'attempt':1,
        'feedback_receipt':receipt,'payload_sha256':base.digest(payload)}

def verify_request(run,payload,proof):
    m=read(Path(run)/'MANIFEST.json');expected,receipt=make_payload(run)
    expected_proof=proof_for(m,expected,receipt)
    require(payload==expected and set(proof) in (set(expected_proof),set(expected_proof)|{'capacity'})
        and all(proof.get(k)==v for k,v in expected_proof.items()),'request_replay_changed')

def strict_json(raw):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('duplicate_json_key')
            out[k]=v
        return out
    def constant(value):raise ValueError('nonfinite_json_number')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)

def validate_assets(raw,m,source):
    try:
        data=strict_json(raw)
        require(isinstance(data,dict) and set(data)=={'root_id','family_id','replacements'},'response_schema')
        require(data['root_id']==m['root_id'] and data['family_id']==m['family_id'],'response_identity')
        rows=data['replacements'];require(isinstance(rows,list) and len(rows)==6,'fixed_six_replacements_required')
        turns={t['turn_ref']:t for t in source['turns']}
        for fid,row in zip(FACT_IDS,rows):
            require(isinstance(row,dict) and set(row)=={'fact_id','anchor','contradicted','contradiction_unavailable_reason'},'replacement_schema:'+fid)
            require(row['fact_id']==fid,'fixed_fact_order:'+fid)
            c=row['contradicted'];reason=row['contradiction_unavailable_reason']
            if c is None:
                require(row['anchor'] is None and isinstance(reason,str) and reason.strip(),'null_alternative_reason:'+fid)
            else:
                require(isinstance(c,dict) and set(c)=={'proposition','relation','rationale'} and c['relation']=='contradicts'
                    and all(isinstance(c[k],str) and c[k].strip() for k in ('proposition','rationale')),'contradicted_schema:'+fid)
                require(reason is None,'nonnull_alternative_reason:'+fid)
                require(not FORBIDDEN.search(c['proposition']),'gendered_public_pronoun:'+fid)
                exact_anchor(row['anchor'],turns)
        return data,[]
    except (ValueError,TypeError,KeyError) as e:return None,[str(e)]

def trusted_assets(m,parent,data):
    facts=[]
    for label,row in zip('ABCDEF',data['replacements']):
        ev=parent['option_evidence'][label];anchor={k:ev[k] for k in ('turn_ref','speaker','quote')}
        c=deepcopy(row['contradicted'])
        if c is not None:c['anchor']=deepcopy(row['anchor'])
        facts.append({'fact_id':row['fact_id'],'parent_label':label,
            'supported':{'proposition':parent['options'][label],'relation':'supports','rationale':ev.get('rationale',ev.get('reason')),'anchor':anchor},
            'contradicted':c,'anchor':deepcopy(row['anchor']),
            'contradiction_unavailable_reason':row['contradiction_unavailable_reason']})
    return {'root_id':m['root_id'],'slot_id':m['slot_id'],'family_id':m['family_id'],
        'parent_candidate_id':parent['candidate_id'],'parent_sha256':m['parent_sha256'],
        'source_sha256':m['source_sha256'],'public_stem':parent['question'],'facts':facts}

def projections(m,assets,assets_sha):
    public={'root_id':m['root_id'],'slot_id':m['slot_id'],'family_id':m['family_id'],
        'source_path':m['source_path'],'source_sha256':m['source_sha256'],'assets_sha256':assets_sha,
        'public_stem':assets['public_stem'],'parent_facts':[],'versions':[],
        'contains_relations':False,'contains_solver_scores':False}
    mapping=[]
    for f in assets['facts']:
        public['parent_facts'].append({'fact_id':f['fact_id'],'parent_label':f['parent_label'],'proposition':f['supported']['proposition']})
        if f['contradicted'] is None:continue
        internal=f['fact_id']+':c';vid='v_'+hmac.new(bytes.fromhex(m['version_id_salt']),internal.encode(),hashlib.sha256).hexdigest()[:24]
        public['versions'].append({'version_id':vid,'fact_id':f['fact_id'],'proposition':f['contradicted']['proposition']})
        mapping.append({'version_id':vid,'fact_id':f['fact_id'],'internal_version_id':internal,'variant':'contradicted','relation':'contradicts'})
    public['versions'].sort(key=lambda v:v['version_id'])
    return public,{'source_sha256':m['source_sha256'],'assets_sha256':assets_sha,'versions':sorted(mapping,key=lambda v:v['version_id'])}

def validation(data,errors):
    return {'mechanical_errors':errors,'fact_count':6 if data else None,
        'counterversion_count':sum(r['contradicted'] is not None for r in data['replacements']) if data else 0,
        'semantic_review_performed':False,'shortage_triggers_retry':False,'repair_allowed':False}

def replay_response(folder,m,source):
    folder=Path(folder)
    if (folder/'ERROR.json').exists():return None
    require((folder/'HTTP_ATTEMPT.json').exists(),'response_without_http_receipt')
    response=read(folder/'RESPONSE.json');meta=response.get('metadata',{})
    require(meta.get('returned_model')==legacy.MODEL and meta.get('finish_reason')=='stop' and meta.get('http_status',200)==200,'response_metadata_changed')
    data,errors=validate_assets(response['text'],m,source)
    require(read(folder/'VALIDATION.json')==validation(data,errors),'validation_replay_changed')
    return data

def verify(run,*,complete=False):
    run=Path(run).resolve();m=read(run/'MANIFEST.json')
    require(m['version']==VERSION and m['contract']==CONTRACT and m['task_sha256']==base.digest(TASK.encode())
        and m['budget']=={'http_attempts_max':1} and m['training_allowed'] is False
        and m['formal_training_admission'] is False and m['current_solver_feedback'] is False
        and m['solver_model']==legacy.TARGET and m['generator_model']==legacy.MODEL,'manifest_contract_changed')
    base.verify_bindings(m['files'])
    require(binding(m['config']['path'])==m['config'] and get_slot(read(m['config']['path']),m['slot_id'])==m['slot'],'config_changed')
    parent,source,source_path,_=parent_source(m['slot'])
    require(read(run/'PARENT.json')==parent and sha(run/'PARENT.json')==m['parent_sha256'],'parent_changed')
    require((run/'SOURCE.json').read_bytes()==source_path.read_bytes() and sha(run/'SOURCE.json')==m['source_sha256']
        and len(source['turns'])==m['source_turn_count'] and m['source_path']==str(run/'SOURCE.json'),'source_changed')
    verify_request(run,read(run/'INITIAL_REQUEST.json'),read(run/'INITIAL_PROOF.json'))
    requests=sorted((run/'requests').glob('*/REQUEST.json'));require(len(requests)<=1,'request_budget_exceeded')
    data=None
    for path in requests:
        folder=path.parent;verify_request(run,read(path),read(folder/'PROOF.json'))
        if (folder/'HTTP_ATTEMPT.json').exists():
            http=read(folder/'HTTP_ATTEMPT.json')
            require(http.get('actual_payload_equals_frozen') is True and http.get('payload_sha256')==base.digest(read(path))
                and http.get('transport_retries')==0 and http.get('http_attempt_number')==1,'http_payload_changed')
        data=replay_response(folder,m,source)
    if complete or (run/'GENERATION.COMPLETE.json').exists():
        c=read(run/'GENERATION.COMPLETE.json');base.verify_bindings(c['files'])
        require((run/'GENERATION.STARTED.json').exists() and len(requests)==1
            and c['version']==VERSION and c['root_id']==m['root_id'] and c['slot_id']==m['slot_id'] and c['family_id']==m['family_id']
            and c['requests_prepared']==1 and c['http_attempts']==len(list((run/'requests').glob('*/HTTP_ATTEMPT.json')))
            and c['training_allowed'] is False and c['semantic_retries']==c['mechanical_repairs']==c['transport_retries']==0
            and c['source_first_review_required'] is True,'completion_contract_changed')
        require(c['fact_count']==(6 if data else 0) and c['counterversion_count']==(validation(data,[])['counterversion_count'] if data else 0),'completion_counts_changed')
        if data:
            assets=trusted_assets(m,parent,data);require(read(run/'ASSETS.json')==assets and c['status']=='ASSETS_READY','assets_changed')
            public,mapping=projections(m,assets,sha(run/'ASSETS.json'));mapping['public_assets_sha256']=sha(run/'PUBLIC_ASSETS.json')
            require(read(run/'PUBLIC_ASSETS.json')==public and read(run/'VERSION_MAP.json')==mapping,'projections_changed')
        else:require(c['status']=='NO_VALID_ASSETS' and not any((run/n).exists() for n in ('ASSETS.json','PUBLIC_ASSETS.json','VERSION_MAP.json')),'invalid_response_exported')
    return m

def generate(run):
    run=Path(run).resolve()
    with qa_transport.shared_lock():
        m=verify(run);require(not (run/'GENERATION.STARTED.json').exists(),'generation_already_started_no_replay')
        require(not list((run/'requests').glob('*/REQUEST.json')),'requests_already_exist')
        write(run/'GENERATION.STARTED.json',{'time':v15.now(),'manifest':binding(run/'MANIFEST.json'),'automatic_retry':False})
        payload=read(run/'INITIAL_REQUEST.json');proof=read(run/'INITIAL_PROOF.json')
        raw,folder,metadata=qa_transport.Caller(run).call(m['slot_id']+'-parent-replacements-1',payload,proof)
        data=None
        if raw is not None:
            data,errors=validate_assets(raw,m,read(run/'SOURCE.json'));write(folder/'VALIDATION.json',validation(data,errors))
        if data:
            assets=trusted_assets(m,read(run/'PARENT.json'),data);write(run/'ASSETS.json',assets)
            public,mapping=projections(m,assets,sha(run/'ASSETS.json'));write(run/'PUBLIC_ASSETS.json',public)
            mapping['public_assets_sha256']=sha(run/'PUBLIC_ASSETS.json');write(run/'VERSION_MAP.json',mapping)
        result={'version':VERSION,'completed_at_utc':v15.now(),'root_id':m['root_id'],'slot_id':m['slot_id'],'family_id':m['family_id'],
            'status':'ASSETS_READY' if data else 'NO_VALID_ASSETS','fact_count':6 if data else 0,
            'counterversion_count':validation(data,[])['counterversion_count'] if data else 0,
            'requests_prepared':1,'http_attempts':len(list((run/'requests').glob('*/HTTP_ATTEMPT.json'))),
            'semantic_retries':0,'mechanical_repairs':0,'transport_retries':0,'training_allowed':False,
            'formal_training_admission':False,'source_first_review_required':True,
            'files':base.bindings(p for p in run.rglob('*') if p.is_file())}
        write(run/'GENERATION.COMPLETE.json',result);verify(run,complete=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','generate','verify'))
    p.add_argument('--config',type=Path,default=HERE/'CONFIG.json');p.add_argument('--slot',required=True);p.add_argument('--run',type=Path)
    a=p.parse_args();run=a.run or HERE/'assets'/a.slot
    result=prepare(a.config,a.slot,run=run) if a.action=='prepare' else generate(run) if a.action=='generate' else {'status':'PASS','slot_id':verify(run)['slot_id']}
    print(json.dumps(result,ensure_ascii=False))
