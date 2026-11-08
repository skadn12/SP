"""Program-bound parent anchors with one bounded source-reviewed supplement."""
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
import unicodedata
from datetime import datetime
import sys

HERE=Path(__file__).resolve().parent
V63=HERE.parent/'h4_option_composition_v63_qa'
V69=HERE.parent/'h4_balanced_234_v69'
_saved_path=sys.path[:]
sys.path.insert(0,str(V63))
_spec=importlib.util.spec_from_file_location('v71_readonly_v63_generation',V63/'generate.py')
legacy=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(legacy)
import qa_transport
sys.path[:]=_saved_path
_spec=importlib.util.spec_from_file_location('v71_readonly_old_feedback_adapter',V69/'source_adapter.py')
source_adapter=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(source_adapter)
base,v15=legacy.base,legacy.v15
read,write,require=base.read,base.write,base.require
VERSION='v71_program_bound_parent_replacements_v1'
CAP=6144
START,END='[CURRENT_PARENT_REPLACEMENT_TASK_V71]','[/CURRENT_PARENT_REPLACEMENT_TASK_V71]'
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
The program binds each alternative to the immutable original parent fact anchor.
Do not output or regenerate turn_ref, speaker, quote, or anchor fields. Reviewers
will read the complete source independently; bound citations do not certify truth.
If no explicit same-scope contradiction exists, set contradicted null and give a
specific nonempty contradiction_unavailable_reason. Never copy any supported
parent statement as a contradicted alternative, including case/spacing/punctuation
variants. Never place the actual change only in rationale: the public proposition
itself must express the incompatible attribute. Preserve accepted alternatives;
a supplement addresses only the supplied target_fact_ids and content feedback.
Return JSON ONLY with exactly root_id, family_id, replacements. Return entries
in target_fact_ids order, with exactly these fields and no citations:
{root_id,family_id,replacements:[{fact_id,
contradicted:null|{proposition,relation:"contradicts",rationale},
contradiction_unavailable_reason:null|string}]}.
"""
CONTRACT={'version':VERSION,'fixed_parent_facts':6,'max_alternatives_per_fact':1,
 'max_tokens':CAP,'http_attempts_max_per_family':2,'bounded_supplements_per_family':1,'program_fixed_citations':True,'reject_copied_parent_propositions':True,'mechanical_repair_max':0,
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
    require(config.get('training_allowed') is False and config.get('max_http')==4 and config.get('max_supplemental_requests_per_family')==1
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
        'generator_model':legacy.MODEL,'budget':{'http_attempts_max':2},
        'version_id_salt':secrets.token_hex(32),'task_sha256':base.digest(TASK.encode()),
        'training_allowed':False,'formal_training_admission':False,'current_solver_feedback':False,
        'files':base.bindings(frozen)}
    write(run/'MANIFEST.json',m);payload,receipt=make_payload(run)
    write(run/'INITIAL_REQUEST.json',payload);write(run/'INITIAL_PROOF.json',proof_for(m,payload,receipt))
    prepare_round(run,'initial',FACT_IDS)
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
        'target_fact_ids':FACT_IDS,
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

def projections(m,assets,assets_sha):
    public={'root_id':m['root_id'],'slot_id':m['slot_id'],'family_id':m['family_id'],
        'source_path':m['source_path'],'source_sha256':m['source_sha256'],'assets_sha256':assets_sha,
        'public_stem':assets['public_stem'],'parent_facts':[],'versions':[],
        'contains_relations':False,'contains_solver_scores':False}
    public.update(parent_path=str(Path(m['source_path']).parent/'PARENT.json'),parent_sha256=m['parent_sha256'],round_id=m.get('round_id','final'))
    mapping=[]
    for f in assets['facts']:
        public['parent_facts'].append({'fact_id':f['fact_id'],'parent_label':f['parent_label'],'proposition':f['supported']['proposition']})
        if f['contradicted'] is None:continue
        internal=f['fact_id']+':c:'+m.get('round_id','final');vid=f.get('origin',{}).get('version_id') or 'v_'+hmac.new(bytes.fromhex(m['version_id_salt']),internal.encode(),hashlib.sha256).hexdigest()[:24]
        public['versions'].append({'version_id':vid,'fact_id':f['fact_id'],'proposition':f['contradicted']['proposition']})
        mapping.append({'version_id':vid,'fact_id':f['fact_id'],'internal_version_id':internal,'variant':'contradicted','relation':'contradicts'})
    public['versions'].sort(key=lambda v:v['version_id'])
    return public,{'source_sha256':m['source_sha256'],'assets_sha256':assets_sha,'versions':sorted(mapping,key=lambda v:v['version_id'])}

def normalized_proposition(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())

def validate_assets(raw,m,source,parent,target_fact_ids=None):
    targets=target_fact_ids or FACT_IDS
    try:
        obj=strict_json(raw)
        require(isinstance(obj,dict) and set(obj)=={'root_id','family_id','replacements'},'response_schema')
        require(obj['root_id']==m['root_id'] and obj['family_id']==m['family_id'],'response_identity')
        require(isinstance(obj['replacements'],list) and [r.get('fact_id') for r in obj['replacements']]==targets,'target_fact_order')
    except (ValueError,TypeError,KeyError,AttributeError) as e:return None,[{'fact_id':None,'error':str(e)}]
    rows=[];errors=[];parent_texts={normalized_proposition(p) for p in parent['options'].values()}
    for fid,row in zip(targets,obj['replacements']):
        try:
            require(isinstance(row,dict) and set(row)=={'fact_id','contradicted','contradiction_unavailable_reason'},'replacement_schema')
            c=row['contradicted'];reason=row['contradiction_unavailable_reason']
            if c is None:require(isinstance(reason,str) and bool(reason.strip()),'null_alternative_reason')
            else:
                require(isinstance(c,dict) and set(c)=={'proposition','relation','rationale'} and c['relation']=='contradicts'
                    and all(isinstance(c[k],str) and c[k].strip() for k in ('proposition','rationale')),'contradicted_schema')
                require(reason is None,'nonnull_alternative_reason')
                require(not FORBIDDEN.search(c['proposition']),'gendered_public_pronoun')
                require(normalized_proposition(c['proposition']) not in parent_texts,'copied_supported_parent_proposition')
            rows.append(deepcopy(row))
        except (ValueError,TypeError,KeyError) as e:
            errors.append({'fact_id':fid,'error':str(e)})
            rows.append({'fact_id':fid,'contradicted':None,'contradiction_unavailable_reason':'Mechanically rejected raw row: '+str(e)})
    return {'root_id':m['root_id'],'family_id':m['family_id'],'replacements':rows},errors

def trusted_assets(m,parent,data):
    rows={r['fact_id']:r for r in data['replacements']} if data else {};facts=[]
    for label,fid in zip('ABCDEF',FACT_IDS):
        ev=parent['option_evidence'][label];anchor={k:ev[k] for k in ('turn_ref','speaker','quote')}
        row=rows.get(fid,{'contradicted':None,'contradiction_unavailable_reason':'Not supplied in this generation round.'})
        c=deepcopy(row['contradicted'])
        if c is not None:c['anchor']=deepcopy(anchor)
        facts.append({'fact_id':fid,'parent_label':label,'supported':{'proposition':parent['options'][label],
            'relation':'supports','rationale':ev.get('rationale',ev.get('reason')),'anchor':anchor},
            'contradicted':c,'anchor':deepcopy(anchor) if c else None,'contradiction_unavailable_reason':row['contradiction_unavailable_reason']})
    return {'root_id':m['root_id'],'slot_id':m['slot_id'],'family_id':m['family_id'],
        'parent_candidate_id':parent['candidate_id'],'parent_sha256':m['parent_sha256'],
        'source_sha256':m['source_sha256'],'public_stem':parent['question'],'facts':facts}

def validation(data,errors):
    return {'mechanical_errors':errors,'fact_count':len(data['replacements']) if data else 0,
        'counterversion_count':sum(r['contradicted'] is not None for r in data['replacements']) if data else 0,
        'semantic_review_performed':False,'shortage_triggers_retry':False,'repair_allowed':False,
        'program_fixed_citations':True,'rejected_rows_retained_in_raw_response':True}

def round_manifest(run,rid):
    run=Path(run);m=read(run/'MANIFEST.json')
    return {**deepcopy(m),'round_id':rid,'budget':{'http_attempts_max':1},'root_manifest':binding(run/'MANIFEST.json')}

def round_payload(run,rid,targets,feedback=None):
    run=Path(run);m=read(run/'MANIFEST.json');ctx=context_for(run)
    ctx.update(target_fact_ids=targets,generation_round=rid)
    if feedback is not None:ctx['independent_content_feedback']=feedback
    payload,receipt=legacy.legacy.payload_with_feedback(run,m['root_id'],INTRO,ctx,CAP)
    intermediate=payload['messages'][0]['content'];tail='\n\n'+START+TASK+END;payload['messages'][0]['content']+=tail
    receipt['asset_task_tail_receipt']={'kind':'program_bound_parent_replacements','task_sha256':base.digest(tail.encode()),
        'intermediate_sha256':base.digest(intermediate.encode()),'final_payload_sha256':base.digest(payload)}
    proof={'role':'asset_generator','root_id':m['root_id'],'slot_id':m['slot_id'],'family_id':m['family_id'],
        'source_sha256':m['source_sha256'],'parent_sha256':m['parent_sha256'],'attempt':1 if rid=='initial' else 2,
        'round_id':rid,'target_fact_ids':targets,'feedback_receipt':receipt,'payload_sha256':base.digest(payload),
        'current_solver_feedback':False}
    return payload,proof

def prepare_round(run,rid,targets,feedback=None,review_files=()):
    run=Path(run);folder=run/'rounds'/rid;require(not folder.exists(),'round_already_prepared')
    require(rid in ('initial','supplement') and targets and targets==[f for f in FACT_IDS if f in targets],'round_targets_invalid')
    m=round_manifest(run,rid);m.update(target_fact_ids=targets,content_feedback=feedback,
        review_feedback_files=base.bindings(review_files))
    write(folder/'MANIFEST.json',m);write(folder/'provider_profile.json',read(run/'provider_profile.json'))
    payload,proof=round_payload(run,rid,targets,feedback);write(folder/'INITIAL_REQUEST.json',payload);write(folder/'INITIAL_PROOF.json',proof)
    return {'round_id':rid,'run':str(folder),'target_fact_ids':targets,'manifest':binding(folder/'MANIFEST.json')}

def verify_round_request(run,rid):
    run=Path(run);folder=run/'rounds'/rid;m=read(folder/'MANIFEST.json')
    expected=round_manifest(run,rid)
    require(all(m.get(k)==v for k,v in expected.items()),'round_manifest_changed')
    require(m['target_fact_ids']==[f for f in FACT_IDS if f in m['target_fact_ids']] and bool(m['target_fact_ids']),'round_targets_changed')
    base.verify_bindings(m['review_feedback_files'])
    if rid=='initial':require(m['target_fact_ids']==FACT_IDS and m['content_feedback'] is None,'initial_round_changed')
    else:
        targets,feedback,paths=supplement_spec(run)
        require(m['target_fact_ids']==targets and m['content_feedback']==feedback and m['review_feedback_files']==base.bindings(paths),'supplement_feedback_changed')
    payload,proof=round_payload(run,rid,m['target_fact_ids'],m['content_feedback'])
    require(read(folder/'INITIAL_REQUEST.json')==payload and read(folder/'INITIAL_PROOF.json')==proof,'round_request_changed')
    return m,payload,proof

def write_projections(folder,m,assets):
    folder=Path(folder);write(folder/'ASSETS.json',assets)
    public,mapping=projections(m,assets,sha(folder/'ASSETS.json'));write(folder/'PUBLIC_ASSETS.json',public)
    mapping['public_assets_sha256']=sha(folder/'PUBLIC_ASSETS.json');write(folder/'VERSION_MAP.json',mapping)

def verify_round(run,rid,*,complete=False):
    run=Path(run);folder=run/'rounds'/rid;m,payload,proof=verify_round_request(run,rid)
    requests=sorted((folder/'requests').glob('*/REQUEST.json'));require(len(requests)<=1,'round_request_budget_exceeded');data=None
    for p in requests:
        d=p.parent;got=read(d/'PROOF.json');require(read(p)==payload and set(got) in (set(proof),set(proof)|{'capacity'}) and all(got.get(k)==v for k,v in proof.items()),'round_actual_request_changed')
        if (d/'HTTP_ATTEMPT.json').exists():
            h=read(d/'HTTP_ATTEMPT.json');require(h.get('actual_payload_equals_frozen') is True and h.get('payload_sha256')==base.digest(payload)
                and h.get('http_attempt_number')==1 and h.get('transport_retries')==0,'round_http_changed')
        if not (d/'ERROR.json').exists():
            require((d/'HTTP_ATTEMPT.json').exists(),'response_without_http')
            raw=read(d/'RESPONSE.json');meta=raw['metadata'];require(meta.get('returned_model')==legacy.MODEL and meta.get('finish_reason')=='stop'
                and meta.get('http_status',200)==200,'round_response_metadata')
            data,errors=validate_assets(raw['text'],m,read(run/'SOURCE.json'),read(run/'PARENT.json'),m['target_fact_ids'])
            require(read(d/'VALIDATION.json')==validation(data,errors),'round_validation_changed')
    if complete or (folder/'GENERATION.COMPLETE.json').exists():
        c=read(folder/'GENERATION.COMPLETE.json');base.verify_bindings(c['files'])
        require((folder/'GENERATION.STARTED.json').exists() and len(requests)==1,'round_not_executed')
        expected=trusted_assets(m,read(run/'PARENT.json'),data)
        require(read(folder/'ASSETS.json')==expected,'round_assets_changed')
        pub,mapping=projections(m,expected,sha(folder/'ASSETS.json'));mapping['public_assets_sha256']=sha(folder/'PUBLIC_ASSETS.json')
        require(read(folder/'PUBLIC_ASSETS.json')==pub and read(folder/'VERSION_MAP.json')==mapping,'round_projections_changed')
        require(c['round_id']==rid and c['target_fact_ids']==m['target_fact_ids'] and c['http_attempts']==len(list((folder/'requests').glob('*/HTTP_ATTEMPT.json')))
            and c['requests_prepared']==1 and c['counterversion_count']==len(pub['versions']) and c['fact_count']==6
            and c['status']==('ASSETS_READY' if data else 'NO_VALID_ASSETS') and c['semantic_retries']==c['mechanical_repairs']==c['transport_retries']==0,'round_completion_changed')
    return m

def generate_round(run,rid):
    run=Path(run).resolve();folder=run/'rounds'/rid
    with qa_transport.shared_lock():
        verify(run);m,payload,proof=verify_round_request(run,rid)
        require(not (folder/'GENERATION.STARTED.json').exists(),'round_already_started_no_replay')
        require(not (run/'GENERATION.COMPLETE.json').exists(),'family_already_finalized')
        require(len(list(run.glob('rounds/*/requests/*/HTTP_ATTEMPT.json')))<2,'family_http_budget_exceeded')
        write(folder/'GENERATION.STARTED.json',{'time':v15.now(),'manifest':binding(folder/'MANIFEST.json'),'automatic_retry':False})
        raw,request,metadata=qa_transport.Caller(folder).call(m['slot_id']+'-'+rid,payload,proof);data=None
        if raw is not None:
            data,errors=validate_assets(raw,m,read(run/'SOURCE.json'),read(run/'PARENT.json'),m['target_fact_ids']);write(request/'VALIDATION.json',validation(data,errors))
        write_projections(folder,m,trusted_assets(m,read(run/'PARENT.json'),data))
        result={'version':VERSION,'completed_at_utc':v15.now(),'round_id':rid,'root_id':m['root_id'],'slot_id':m['slot_id'],'family_id':m['family_id'],
            'target_fact_ids':m['target_fact_ids'],'status':'ASSETS_READY' if data else 'NO_VALID_ASSETS','fact_count':6,
            'counterversion_count':len(read(folder/'PUBLIC_ASSETS.json')['versions']),'requests_prepared':1,
            'http_attempts':len(list((folder/'requests').glob('*/HTTP_ATTEMPT.json'))),'semantic_retries':0,'mechanical_repairs':0,'transport_retries':0,
            'bounded_supplement':rid=='supplement','training_allowed':False,'source_first_review_required':True,
            'files':base.bindings(p for p in folder.rglob('*') if p.is_file())}
        write(folder/'GENERATION.COMPLETE.json',result);verify_round(run,rid,complete=True);return result

def reviewed_round(run,rid):
    run=Path(run);folder=run/'rounds'/rid;verify_round(run,rid,complete=True)
    j=read(folder/'SOURCE_JUDGMENTS.json');r=read(folder/'ASSET_REVIEW.json');receipt=read(folder/'REVIEW_RECEIPT.json')
    pub=read(folder/'PUBLIC_ASSETS.json');assets=read(folder/'ASSETS.json');maps=read(folder/'VERSION_MAP.json')
    require(j.get('new_variant_private_artifacts_read_before_save') is False and j.get('solver_scores_used') is False
        and j.get('current_solver_scores_read',False) is False and j.get('complete_source_turns_read')==len(read(run/'SOURCE.json')['turns']),'round_review_not_blind_complete_source')
    require(j['source_sha256']==sha(run/'SOURCE.json') and j['public_assets_sha256']==sha(folder/'PUBLIC_ASSETS.json'),'round_judgment_binding')
    require(r['source_sha256']==j['source_sha256'] and r['assets_sha256']==sha(folder/'ASSETS.json')
        and r['public_assets_sha256']==j['public_assets_sha256'] and r['source_judgments_sha256']==sha(folder/'SOURCE_JUDGMENTS.json')
        and r['solver_scores_used'] is False and r['source_first'] is True,'round_review_binding')
    base.verify_bindings(receipt['files'])
    require(receipt['source_first'] is True and receipt['new_variant_private_artifacts_read_before_save'] is False
        and receipt['solver_scores_used'] is False and receipt['round_id']==rid,'round_receipt_changed')
    require(datetime.fromisoformat(j['created_at_utc'])<=datetime.fromisoformat(receipt['private_crosscheck_started_at_utc'])
        <=datetime.fromisoformat(receipt['review_completed_at_utc']),'round_review_order')
    expected=private_crosscheck(run,rid,j)
    require(r==expected,'round_review_projection_changed')
    require(receipt['source_judgments_saved_at_utc']==j['created_at_utc'],'round_review_time_changed')
    approved={v['version_id'] for v in r['versions'] if v['usable']};mapping={v['version_id']:v for v in maps['versions']}
    return {'round_id':rid,'folder':folder,'judgment':j,'review':r,'public':pub,'assets':assets,'mapping':mapping,'approved':approved,
        'approved_fact_ids':sorted(mapping[v]['fact_id'] for v in approved),'parents_valid':all(p['usable'] for p in r['parents'])}

def private_crosscheck(run,rid,j):
    run=Path(run);folder=run/'rounds'/rid;a=read(folder/'ASSETS.json');mapping=read(folder/'VERSION_MAP.json');pub=read(folder/'PUBLIC_ASSETS.json')
    require(j.get('new_variant_private_artifacts_read_before_save') is False and j.get('solver_scores_used') is False,'private_read_before_source_judgment')
    require(j.get('source_sha256')==sha(run/'SOURCE.json') and j.get('public_assets_sha256')==sha(folder/'PUBLIC_ASSETS.json'),'source_judgment_bindings')
    require(type(j.get('stem_valid')) is bool,'stem_judgment_missing');facts={f['fact_id']:f for f in a['facts']};maps={v['version_id']:v for v in mapping['versions']}
    require([p['fact_id'] for p in j['parents']]==FACT_IDS and {v['version_id'] for v in j['versions']}==set(maps)
        and len(j['versions'])==len(maps),'review_coverage')
    turns={t['turn_ref']:t for t in read(run/'SOURCE.json')['turns']}
    def check(row,writer,key,public):
        require(row['relation'] in ('supports','contradicts','unknown') and type(row['scope_valid']) is bool
            and type(row['fact_distinct']) is bool and isinstance(row['reason'],str) and row['reason'].strip(),'review_schema')
        if 'proposition' in row:require(row['proposition']==public,'judgment_public_text_changed')
        require(isinstance(row['citations'],list) and bool(row['citations']),'review_citations_required')
        for c in row['citations']:exact_anchor(c,turns)
        exact_anchor(writer['anchor'],turns)
        usable=j['stem_valid'] and row['scope_valid'] and row['fact_distinct'] and row['relation']==writer['relation']
        return {key:row[key],'usable':usable,'relation':row['relation'],'reason':row['reason']+' Writer relation '+('agrees.' if row['relation']==writer['relation'] else 'differs; retain original label and exclude.')}
    parents=[check(row,facts[row['fact_id']]['supported'],'fact_id',facts[row['fact_id']]['supported']['proposition']) for row in j['parents']]
    versions=[];pvs={v['version_id']:v for v in pub['versions']}
    for row in j['versions']:
        vm=maps[row['version_id']];w=facts[vm['fact_id']]['contradicted'];require(w['proposition']==pvs[row['version_id']]['proposition'] and w['relation']==vm['relation'],'writer_projection_changed')
        versions.append(check(row,w,'version_id',pvs[row['version_id']]['proposition']))
    return {'version':'v71_asset_private_crosscheck_v1','round_id':rid,'source_sha256':sha(run/'SOURCE.json'),'assets_sha256':sha(folder/'ASSETS.json'),
        'public_assets_sha256':sha(folder/'PUBLIC_ASSETS.json'),'source_judgments_sha256':sha(folder/'SOURCE_JUDGMENTS.json'),
        'solver_scores_used':False,'source_first':True,'parent_all_true_provenance_known':True,
        'new_variant_private_artifacts_read_before_save':False,'stem_valid':j['stem_valid'],'parents':parents,'versions':versions}

def finish_review(run,rid):
    run=Path(run).resolve();folder=run/'rounds'/rid;verify_round(run,rid,complete=True)
    j=read(folder/'SOURCE_JUDGMENTS.json');started=v15.now()
    require(datetime.fromisoformat(j['created_at_utc'])<=datetime.fromisoformat(started),'source_judgment_future_timestamp')
    review=private_crosscheck(run,rid,j);write(folder/'ASSET_REVIEW.json',review)
    receipt={'version':'v71_asset_source_first_receipt_v1','family_id':read(run/'MANIFEST.json')['family_id'],'round_id':rid,
        'source_first':True,'parent_all_true_provenance_known':True,'new_variant_private_artifacts_read_before_save':False,
        'source_judgments_saved_at_utc':j['created_at_utc'],'private_crosscheck_started_at_utc':started,
        'review_completed_at_utc':v15.now(),'solver_scores_used':False,
        'usable_parents':sum(p['usable'] for p in review['parents']),'usable_contradicts':sum(p['usable'] for p in review['versions']),
        'files':base.bindings([run/'SOURCE.json',run/'PARENT.json']+[folder/n for n in ['PUBLIC_ASSETS.json','SOURCE_JUDGMENTS.json','ASSETS.json','VERSION_MAP.json','ASSET_REVIEW.json']])}
    write(folder/'REVIEW_RECEIPT.json',receipt);reviewed_round(run,rid);return receipt

def supplement_spec(run):
    run=Path(run);r=reviewed_round(run,'initial')
    require(r['parents_valid'],'parent_review_failed_no_supplement')
    require(len(r['approved_fact_ids'])<4,'four_distinct_false_facts_already_approved')
    targets=[fid for fid in FACT_IDS if fid not in r['approved_fact_ids']];feedback=[]
    reviewed={r['mapping'][v['version_id']]['fact_id']:v for v in r['judgment']['versions']}
    folder=r['folder'];errors=[]
    for p in (folder/'requests').glob('*/VALIDATION.json'):errors+=read(p)['mechanical_errors']
    for fid in targets:
        j=reviewed.get(fid);feedback.append({'fact_id':fid,'independent_relation':j['relation'] if j else None,
            'reason':j['reason'] if j else 'No mechanically accepted alternative was available.',
            'scope_valid':j['scope_valid'] if j else None,'fact_distinct':j['fact_distinct'] if j else None,
            'mechanical_errors':[e['error'] for e in errors if e['fact_id'] in (None,fid)]})
    return targets,feedback,[folder/n for n in ['PUBLIC_ASSETS.json','SOURCE_JUDGMENTS.json','ASSET_REVIEW.json','REVIEW_RECEIPT.json','GENERATION.COMPLETE.json']]

def prepare_supplement(run):
    run=Path(run).resolve();verify(run);require(not (run/'GENERATION.COMPLETE.json').exists(),'family_already_finalized')
    targets,feedback,paths=supplement_spec(run);return prepare_round(run,'supplement',targets,feedback,paths)

def final_projection(run):
    run=Path(run);m=read(run/'MANIFEST.json');parent=read(run/'PARENT.json');rounds=[reviewed_round(run,'initial')]
    if (run/'rounds/supplement').exists():rounds.append(reviewed_round(run,'supplement'))
    assets=trusted_assets(m,parent,None);byfact={f['fact_id']:f for f in assets['facts']};accepted=[];jversions=[];rversions=[];prov=[]
    for r in rounds:
        folder=r['folder'];rm=read(folder/'MANIFEST.json');rc=read(folder/'GENERATION.COMPLETE.json')
        row={'round_id':r['round_id'],'target_fact_ids':rm['target_fact_ids'],'status':rc['status'],'http_attempts':rc['http_attempts'],
            'approved_distinct_false_facts':len(r['approved_fact_ids']),**{key:binding(folder/name) for key,name in [('public_assets','PUBLIC_ASSETS.json'),('source_judgments','SOURCE_JUDGMENTS.json'),('asset_review','ASSET_REVIEW.json'),('review_receipt','REVIEW_RECEIPT.json')]}}
        prov.append(row)
        for vid in sorted(r['approved']):
            fid=r['mapping'][vid]['fact_id'];require(byfact[fid]['contradicted'] is None,'accepted_initial_version_overwritten')
            f=deepcopy(next(f for f in r['assets']['facts'] if f['fact_id']==fid));f['origin']={'round_id':r['round_id'],'version_id':vid};byfact[fid]=f
            accepted.append({'fact_id':fid,'version_id':vid,'round_id':r['round_id']})
            jversions.append(deepcopy(next(v for v in r['judgment']['versions'] if v['version_id']==vid)))
            rversions.append(deepcopy(next(v for v in r['review']['versions'] if v['version_id']==vid)))
    assets['facts']=[byfact[f] for f in FACT_IDS]
    provenance={'version':'v71_round_review_provenance_v1','family_id':m['family_id'],'rounds':prov,
        'accepted_versions':sorted(accepted,key=lambda v:v['fact_id']),'source_first_per_round':True,'solver_scores_used':False,
        'aggregate_is_new_independent_blind_review':False}
    j=deepcopy(rounds[0]['judgment']);j.update(version='v71_derived_round_source_judgments_v1',review_mode='derived_from_saved_round_source_judgments',
        derived_from_rounds=True,source_first_per_round=True,new_variant_private_artifacts_read_before_save=True,
        created_at_utc=None,versions=sorted(jversions,key=lambda v:v['version_id']),provenance_rounds=prov)
    r=deepcopy(rounds[0]['review']);r.update(version='v71_derived_round_private_crosschecks_v1',review_mode='derived_round_private_crosschecks',
        source_first_per_round=True,new_variant_private_artifacts_read_before_save=True,versions=sorted(rversions,key=lambda v:v['version_id']),provenance_rounds=prov)
    # Every round must still independently support the immutable parents.
    for index,p in enumerate(r['parents']):
        p['usable']=all(rr['review']['parents'][index]['usable'] for rr in rounds)
    r['stem_valid']=all(rr['review']['stem_valid'] for rr in rounds);j['stem_valid']=r['stem_valid']
    return assets,j,r,provenance

def finalize(run):
    run=Path(run).resolve();verify(run);require(not (run/'GENERATION.COMPLETE.json').exists(),'family_already_finalized')
    assets,j,r,provenance=final_projection(run);m=read(run/'MANIFEST.json');write_projections(run,m,assets)
    j['public_assets_sha256']=sha(run/'PUBLIC_ASSETS.json');write(run/'SOURCE_JUDGMENTS.json',j)
    r.update(source_sha256=sha(run/'SOURCE.json'),assets_sha256=sha(run/'ASSETS.json'),public_assets_sha256=sha(run/'PUBLIC_ASSETS.json'),source_judgments_sha256=sha(run/'SOURCE_JUDGMENTS.json'))
    write(run/'ASSET_REVIEW.json',r);write(run/'ROUND_PROVENANCE.json',provenance)
    receipt={'version':'v71_derived_round_review_receipt_v1','created_at_utc':v15.now(),'derived_from_saved_round_reviews':True,
        'source_first_per_round':True,'aggregate_is_new_independent_blind_review':False,'provenance_rounds':provenance['rounds'],
        'files':base.bindings(run/n for n in ['SOURCE_JUDGMENTS.json','ASSET_REVIEW.json','ROUND_PROVENANCE.json'])}
    write(run/'REVIEW_RECEIPT.json',receipt)
    requests=list(run.glob('rounds/*/requests/*/REQUEST.json'));http=list(run.glob('rounds/*/requests/*/HTTP_ATTEMPT.json'))
    c={'version':VERSION,'completed_at_utc':v15.now(),'root_id':m['root_id'],'slot_id':m['slot_id'],'family_id':m['family_id'],
        'status':'ASSETS_READY','fact_count':6,'counterversion_count':len(provenance['accepted_versions']),'requests_prepared':len(requests),'http_attempts':len(http),
        'semantic_retries':0,'mechanical_repairs':0,'transport_retries':0,'bounded_supplements':len(provenance['rounds'])-1,
        'training_allowed':False,'formal_training_admission':False,'source_first_review_required':True,'rounds':provenance['rounds'],
        'files':base.bindings(p for p in run.rglob('*') if p.is_file())}
    write(run/'GENERATION.COMPLETE.json',c);verify(run,complete=True);return c

def verify(run,*,complete=False):
    run=Path(run).resolve();m=read(run/'MANIFEST.json')
    require(m['version']==VERSION and m['contract']==CONTRACT and m['task_sha256']==base.digest(TASK.encode())
        and m['budget']=={'http_attempts_max':2} and m['training_allowed'] is False and m['current_solver_feedback'] is False,'manifest_contract_changed')
    base.verify_bindings(m['files']);require(binding(m['config']['path'])==m['config'] and get_slot(read(m['config']['path']),m['slot_id'])==m['slot'],'config_changed')
    parent,source,path,_=parent_source(m['slot']);require(read(run/'PARENT.json')==parent and sha(run/'PARENT.json')==m['parent_sha256'],'parent_changed')
    require((run/'SOURCE.json').read_bytes()==path.read_bytes() and sha(run/'SOURCE.json')==m['source_sha256'],'source_changed')
    verify_request(run,read(run/'INITIAL_REQUEST.json'),read(run/'INITIAL_PROOF.json'))
    require(set(p.name for p in (run/'rounds').iterdir())<=set(('initial','supplement')),'unexpected_generation_round')
    for p in (run/'rounds').iterdir():verify_round(run,p.name)
    require(len(list(run.glob('rounds/*/requests/*/HTTP_ATTEMPT.json')))<=2,'family_http_budget_exceeded')
    if complete or (run/'GENERATION.COMPLETE.json').exists():
        c=read(run/'GENERATION.COMPLETE.json');base.verify_bindings(c['files']);assets,j,r,prov=final_projection(run)
        require(read(run/'ASSETS.json')==assets and read(run/'ROUND_PROVENANCE.json')==prov,'final_merged_assets_or_provenance_changed')
        pub,mapping=projections(m,assets,sha(run/'ASSETS.json'));mapping['public_assets_sha256']=sha(run/'PUBLIC_ASSETS.json')
        require(read(run/'PUBLIC_ASSETS.json')==pub and read(run/'VERSION_MAP.json')==mapping,'final_public_mapping_changed')
        j['public_assets_sha256']=sha(run/'PUBLIC_ASSETS.json');require(read(run/'SOURCE_JUDGMENTS.json')==j,'derived_judgment_changed')
        r.update(source_sha256=sha(run/'SOURCE.json'),assets_sha256=sha(run/'ASSETS.json'),public_assets_sha256=sha(run/'PUBLIC_ASSETS.json'),source_judgments_sha256=sha(run/'SOURCE_JUDGMENTS.json'))
        require(read(run/'ASSET_REVIEW.json')==r,'derived_review_changed');base.verify_bindings(read(run/'REVIEW_RECEIPT.json')['files'])
        require(c['status']=='ASSETS_READY' and c['fact_count']==6 and c['counterversion_count']==len(prov['accepted_versions']) and c['rounds']==prov['rounds']
            and c['http_attempts']==len(list(run.glob('rounds/*/requests/*/HTTP_ATTEMPT.json')))
            and c['requests_prepared']==len(list(run.glob('rounds/*/requests/*/REQUEST.json')))
            and c['bounded_supplements']==len(prov['rounds'])-1 and c['semantic_retries']==c['mechanical_repairs']==c['transport_retries']==0,'final_completion_changed')
    return m

def generate(run):return generate_round(run,'initial')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','generate','prepare-supplement','generate-supplement','finalize','verify'))
    p.add_argument('--config',type=Path,default=HERE/'CONFIG.json');p.add_argument('--slot',required=True);p.add_argument('--run',type=Path)
    args=p.parse_args();run=args.run or HERE/'assets'/args.slot
    if args.action=='prepare':result=prepare(args.config,args.slot,run=run)
    elif args.action=='generate':result=generate(run)
    elif args.action=='prepare-supplement':result=prepare_supplement(run)
    elif args.action=='generate-supplement':result=generate_round(run,'supplement')
    elif args.action=='finalize':result=finalize(run)
    else:result={'status':'PASS','slot_id':verify(run)['slot_id']}
    print(json.dumps(result,ensure_ascii=False))
