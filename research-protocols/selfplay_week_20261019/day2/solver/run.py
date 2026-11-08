"""Day 2 immutable S0 diagnosis runtime. prepare/verify/self-test are CPU only.

Direct live never opens private Gold. Only score opens the preserved private QA,
and only after every planned raw request has a committed result and freeze.
"""
from __future__ import annotations
import argparse, collections, hashlib, importlib.metadata, json, os, pathlib, shutil, statistics, sys, time, traceback
from datetime import datetime,timezone
import adapter as a
HERE=pathlib.Path(__file__).resolve().parent
DAY1=HERE.parents[1]/'day1'
REPO=a.REPO
VENV=REPO/'.venv-local4b'
TOKENIZER=pathlib.Path('/root/autodl-tmp/Qwen3-4B')
MODEL=REPO/'restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model'
DEFAULT_RUN=HERE/'direct'

def require(c,m):
    if not c: raise RuntimeError(m)
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(pathlib.Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def textsha(s): return hashlib.sha256(s.encode()).hexdigest()
def bind(p):
    p=pathlib.Path(p).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def save(p,v):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(v,f,ensure_ascii=False,indent=2,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
def verify_bindings(rows):
    for r in rows:
        p=pathlib.Path(r['path']);require(p.is_file() and sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],'BOUND_FILE_CHANGED:'+str(p))
def versions():
    return {k:importlib.metadata.version(k) for k in ['transformers','torch','vllm','tokenizers']}
def code_paths():
    contract,solver=a.native()
    from scripts.current_method.restart_n5_spell import backends,vllm_backends,structured_qa_box_protocol
    paths=set(HERE.glob('*.py'))|{a.NATIVE/'protocol.py',a.NATIVE/'scoring.py',REPO/'offline-deliveries/h4_codex_structure_pilot_v140/evaluation/batched_live.py',REPO/'offline-deliveries/h4_codex_structure_pilot_v140/evaluation/adapter_protocol.py'}
    paths.update(pathlib.Path(o.__file__) for o in [contract,solver,backends,vllm_backends,structured_qa_box_protocol])
    # Bind all implementation modules imported from this exact worktree.
    for o in list(sys.modules.values()):
        p=getattr(o,'__file__',None)
        if p and pathlib.Path(p).suffix=='.py' and pathlib.Path(p).is_relative_to(a.WORKTREE):paths.add(pathlib.Path(p))
    return sorted(paths)
def models():
    bindings=read(DAY1/'MODEL_BINDINGS.json')['models']
    require(bindings['S0_M_clean_merged']['directory']==str(MODEL),'S0_MODEL_PATH_MISMATCH')
    rows=bindings['S0_M_clean_merged']['files']+[r for r in bindings['base']['files'] if pathlib.Path(r['path']).name in {'tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'}]
    verify_bindings(rows);return rows

def source_check(q):
    p=pathlib.Path(q['source_path']);require(p.is_absolute() and p.is_file() and sha(p)==q['source_sha256'],'SOURCE_CHANGED:'+q['candidate_id'])
    s=read(p);require(s.get('root_id')==q['root_id'] and isinstance(s.get('turns'),list) and s['turns'],'SOURCE_ROOT_OR_TURNS_INVALID')
    refs=[]
    for t in s['turns']:
        require(all(isinstance(t.get(k),str) and t[k].strip() for k in ['turn_ref','speaker','text']),'SOURCE_TURN_INVALID')
        refs.append(t['turn_ref'])
    require(len(refs)==len(set(refs)),'SOURCE_DUPLICATE_TURNS');return s

def prepare(run,admission=None,direct_run=None):
    require(pathlib.Path(sys.prefix)==VENV,'USE_PROVEN_LOCAL4B_VENV')
    require(not run.exists(),'RUN_EXISTS_NO_OVERWRITE')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
    policy=read(DAY1/'PROTOCOL_V1.json');questions=read(DAY1/'diagnosis_inputs/PUBLIC_QUESTIONS.json')['questions']
    require(len(questions)==39 and len({q['root_id'] for q in questions})==20,'EXPECTED_39_QUESTIONS_20_ROOTS')
    allowed=set(read(DAY1/'evaluation/ROOT_SELECTION.json')['wave1']['root_ids'])
    require({q['root_id'] for q in questions}==allowed,'ROOT_BOUNDARY_MISMATCH')
    require(len({q['candidate_id'] for q in questions})==39,'DUPLICATE_QUESTION_ID')
    for q in questions:a.public_question(q)
    branch='evidence' if direct_run else 'direct'; strata=None
    if direct_run:
        direct=read(direct_run/'SCORES.json');require(direct['status']=='COMPLETE' and direct['denominator_questions']==39,'DIRECT_SCORES_REQUIRED')
        strata={r['candidate_id']:r['stratum'] for r in direct['questions']}
        # All 39 are retained, hash-order within observed strata; no answer/prediction in prompts.
        questions=sorted(questions,key=lambda q:(strata[q['candidate_id']],hashlib.sha256(f"2026101900|structured|{q['candidate_id']}".encode()).hexdigest()))
    sources={}; variants={};rows=[];prompts={};mappings={};tokids={}
    _,solver=a.native();decoder=solver.generation_config('multi_select')
    require(all(decoder[k]==v for k,v in {'temperature':.8,'top_p':.9,'max_tokens':32,'enable_thinking':False,'structured_output_version':'finite_choice_set_v1'}.items()),'NATIVE_DECODER_CHANGED')
    for q in questions:
        s=source_check(q);sources.setdefault(q['root_id'],s)
        require(sources[q['root_id']]==s,'CONFLICTING_SOURCE_VERSIONS')
        count=(6 if q['question_type']=='multi_select' else 12) if branch=='direct' else 1
        variants[q['candidate_id']]=a.variants(q,policy['seeds']['evaluation_display'],count)
        for v in variants[q['candidate_id']]:
            field='options' if q['question_type']=='multi_select' else 'items'
            prompt=a.direct_prompt(tokenizer,q,v,s) if branch=='direct' else a.evidence_prompt(tokenizer,q,v,s)
            # Prompt builder sees only public question text and displayed content.
            require(all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in prompt for t in s['turns']),'INCOMPLETE_SOURCE_IN_PROMPT')
            ids=tokenizer.encode(prompt,add_special_tokens=False)
            maxout=32 if branch=='direct' else 1024
            require(len(ids)+maxout<=16384,'FULL_CONTEXT_OVERFLOW_NO_TRUNCATION:'+q['candidate_id'])
            rollouts=3 if branch=='direct' and field=='options' else 1
            for slot in range(rollouts):
                request_id=f"{q['candidate_id']}__{v['variant_id']}__{branch}__s{slot:02d}"
                row={'request_id':request_id,'candidate_id':q['candidate_id'],'root_id':q['root_id'],'question_type':q['question_type'],
                     'branch':branch,'variant_id':v['variant_id'],'display':{field:v[field]},'slot':slot,
                     'seed':policy['seeds']['direct_sampling']+len(rows)+(100000 if branch=='evidence' else 0),
                     'prompt_sha256':textsha(prompt),'prompt_tokens':len(ids),'source_turns':len(s['turns']),
                     'source_sha256':q['source_sha256'],'max_output_tokens':maxout,'stratum':strata[q['candidate_id']] if strata else None}
                if branch=='evidence':row['json_schema']=a.evidence_schema(q)
                a.sampling_params(row,decoder['structured_output_api']) # actual vLLM API constructor CPU precheck
                rows.append(row);prompts[request_id]=prompt;tokids[request_id]=ids;mappings[request_id]=v
    require(len(rows)==(582 if branch=='direct' else 39),'PLANNED_REQUEST_COUNT_MISMATCH')
    modelrows=models()
    run.mkdir(parents=True)
    save(run/'PUBLIC_QUESTIONS.json',{'questions':questions})
    save(run/'private/DISPLAY_MAPPINGS.json',{'requests':mappings,'public_to_canonical':{q['candidate_id']:a.content_ids(q) for q in questions}})
    for rid,s in sources.items():
        q=next(q for q in questions if q['root_id']==rid); target=run/'sources'/f'{rid}.json';target.parent.mkdir(exist_ok=True);shutil.copy2(q['source_path'],target)
    for rid,p in prompts.items():
        path=run/'prompts'/f'{rid}.txt';path.parent.mkdir(exist_ok=True);path.write_text(p)
    save(run/'PROMPT_TOKEN_IDS.json',tokids)
    inputs=[DAY1/'PROTOCOL_V1.json',DAY1/'MODEL_BINDINGS.json',DAY1/'diagnosis_inputs/PUBLIC_QUESTIONS.json',DAY1/'diagnosis_inputs/INPUT_PLAN.json',DAY1/'evaluation/ROOT_SELECTION.json']
    # Binding private bytes is allowed before answers; JSON content is never opened here.
    private_binding=bind(DAY1/'diagnosis_inputs/PRIVATE_QUESTIONS.json')
    if admission:
        obj=read(admission);require(obj.get('status')=='PASS','CONTENT_ADMISSION_NOT_PASS')
        require(obj.get('namespace')=='train.feedback','CONTENT_ADMISSION_NAMESPACE_INVALID')
        require({r['candidate_id'] for r in obj['questions']}=={q['candidate_id'] for q in questions},'CONTENT_ADMISSION_QUESTION_IDS_DIFFER')
        require(obj['public_questions']['sha256']==sha(DAY1/'diagnosis_inputs/PUBLIC_QUESTIONS.json') and obj['private_questions']['sha256']==private_binding['sha256'],'CONTENT_ADMISSION_QA_BINDINGS_DIFFER')
        def admission_bindings(value):
            if isinstance(value,dict):
                if {'path','sha256','bytes'}<=set(value):yield value
                else:
                    for child in value.values():yield from admission_bindings(child)
            elif isinstance(value,list):
                for child in value:yield from admission_bindings(child)
        bound=list(admission_bindings(obj));verify_bindings(bound)
        inputs.append(admission);inputs.extend(pathlib.Path(r['path']) for r in bound)
    if direct_run: inputs.extend([direct_run/'SCORES.json',direct_run/'RAW_RESPONSE_FREEZE.json'])
    manifest={'status':'PREPARED_AWAITING_INDEPENDENT_PREFLIGHT','prepared_at_utc':now(),'branch':branch,
      'solver_model':'S0_M_clean','model':str(MODEL),'tokenizer':str(TOKENIZER),'namespace':'train.feedback','use_as_final_eval':False,
      'question_count':len(questions),'request_count':len(rows),'root_count':len(allowed),'requests':rows,
      'decoder_contract':decoder,'evidence_decoder':{'structured_json_schema':True,'temperature':.8,'top_p':.9,'max_tokens':1024,'enable_thinking':False} if direct_run else None,
      'batch_size':8,'max_model_len':16384,'prefix_caching':False,'automatic_retry':False,'source_truncation':False,
      'random_display':{'seed':policy['seeds']['evaluation_display'],'gold_blind':True,'algorithm':'v15_independent_seeded_rebinding_dynamic4_5_6'},
      'private_answer_binding':private_binding,'content_admission_binding':bind(admission) if admission else None,
      'model_bindings':modelrows,'input_bindings':[bind(p) for p in inputs],'code_bindings':[bind(p) for p in code_paths()],
      'versions':versions(),'python_prefix':sys.prefix,'gold_opened_by_prepare':False,'gold_access_scope':'This runtime process only; historical author, parent and source-review Gold exposure is disclosed in content admission.','training_started':False,
      'max_prompt_tokens':max(r['prompt_tokens'] for r in rows),'minimum_context_headroom':min(16384-r['prompt_tokens']-r['max_output_tokens'] for r in rows)}
    save(run/'manifest.json',manifest)
    files=[p for p in run.rglob('*') if p.is_file()]
    save(run/'PREFLIGHT_FREEZE.json',{'status':'PREPARED_AWAITING_INDEPENDENT_PREFLIGHT','created_at_utc':now(),'files':[bind(p) for p in sorted(files)],'questions':39,'requests':len(rows),'inference_started':False})
    print(json.dumps({k:manifest[k] for k in ['status','branch','request_count','max_prompt_tokens','minimum_context_headroom']}),flush=True)

def verify(run,audit=None,check_models=True):
    frozen=read(run/'PREFLIGHT_FREEZE.json');verify_bindings(frozen['files']);m=read(run/'manifest.json')
    verify_bindings(m['input_bindings']);verify_bindings(m['code_bindings']);verify_bindings([m['private_answer_binding']])
    if m.get('content_admission_binding'):verify_bindings([m['content_admission_binding']])
    if check_models:verify_bindings(m['model_bindings'])
    require(m['versions']==versions(),'LIBRARY_VERSION_CHANGED')
    require(pathlib.Path(sys.prefix)==VENV,'USE_PROVEN_LOCAL4B_VENV')
    if audit:
        report=read(audit);require(report.get('status')=='PASS','INDEPENDENT_PREFLIGHT_NOT_PASS')
        rows=report.get('audited_files') or report.get('files') or []
        require(any(pathlib.Path(r['path']).resolve()==(run/'PREFLIGHT_FREEZE.json').resolve() and r['sha256']==sha(run/'PREFLIGHT_FREEZE.json') for r in rows),'AUDIT_NOT_BOUND_TO_THIS_PREFLIGHT')
        verify_bindings(rows)
    return m

def raw_freeze(run,m):
    ids=[r['request_id'] for r in m['requests']]; files=[run/'responses'/f'{rid}.json' for rid in ids]
    require(all(p.exists() for p in files),'RAW_FREEZE_REQUIRES_ALL_RESULT_RECORDS')
    require(set(p.stem for p in (run/'responses').glob('*.json'))==set(ids),'UNEXPECTED_RAW_RESULT_FILES')
    save(run/'RAW_RESPONSE_FREEZE.json',{'status':'COMMITTED','created_at_utc':now(),'manifest_binding':bind(run/'manifest.json'),
      'planned_request_count':len(ids),'files':[bind(p) for p in files],'gold_opened_by_live':False,
      'source_only_prompts':True,'automatic_retry':False})

def live(run,audit):
    require(not (run/'STARTED.json').exists(),'ALREADY_STARTED_NO_AUTOMATIC_RERUN')
    m=verify(run,audit);require(m['content_admission_binding'] is not None,'SOURCE_CONTENT_ADMISSION_REQUIRED_BEFORE_GPU')
    rows=m['requests'];save(run/'STARTED.json',{'started_at_utc':now(),'manifest_binding':bind(run/'manifest.json'),'independent_audit_binding':bind(audit),'gold_opened':False})
    backend=None;submitted=set();done=set();started=time.perf_counter();load_s=None
    try:
        from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
        _,solver=a.native();solver.ensure_vllm_environment()
        backend=VLLMSolverBackend(model_path=MODEL,tokenizer_path=TOKENIZER,max_model_len=16384,temperature=.8,top_p=.9,max_tokens=32 if m['branch']=='direct' else 1024,gpu_memory_utilization=.8)
        load_s=backend.load_latency_seconds
        require(str(backend.model_identifier)==m['model'],'BACKEND_MODEL_MISMATCH')
        for offset in range(0,len(rows),8):
            batch=rows[offset:offset+8];batchid=f'batch{offset//8:04d}';prompts=[];params=[]
            for row in batch:
                p=run/'prompts'/f"{row['request_id']}.txt";require(sha(p)==row['prompt_sha256'],'PROMPT_CHANGED')
                prompts.append(p.read_text());params.append(a.sampling_params(row,m['decoder_contract']['structured_output_api']))
            submitted_at=now();batchstart=time.perf_counter()
            save(run/'batches'/f'{batchid}_SUBMITTED.json',{'submitted_at_utc':submitted_at,'request_ids':[r['request_id'] for r in batch],
                 'sampling_params':[p.to_dict() if hasattr(p,'to_dict') else repr(p) for p in params],'automatic_retry':False})
            submitted.update(r['request_id'] for r in batch)
            outputs=backend.engine.generate(prompts,params,use_tqdm=False)
            require(len(outputs)==len(batch),'OUTPUT_COUNT_MISMATCH')
            for row,prompt,out in zip(batch,prompts,outputs):
                require(out.prompt==prompt and len(out.prompt_token_ids)==row['prompt_tokens'],'OUTPUT_PROMPT_OR_TOKEN_DRIFT')
                require(len(out.outputs)==1,'COMPLETION_COUNT_MISMATCH');c=out.outputs[0]
                record={'request_id':row['request_id'],'candidate_id':row['candidate_id'],'root_id':row['root_id'],'question_type':row['question_type'],
                  'branch':m['branch'],'variant_id':row['variant_id'],'slot':row['slot'],'seed':row['seed'],'raw_output':c.text,
                  'request_submitted_at_utc':submitted_at,'response_committed_at_utc':now(),'status':'RETURNED','model_path':str(MODEL),
                  'prompt_sha256':row['prompt_sha256'],'prompt_token_ids':out.prompt_token_ids,'completion_token_ids':c.token_ids,
                  'metadata':{'prompt_token_count':len(out.prompt_token_ids),'completion_token_count':len(c.token_ids),'finish_reason':c.finish_reason,
                    'stop_reason':c.stop_reason,'temperature':.8,'top_p':.9,'max_tokens':row['max_output_tokens'],
                    'structured_output_api':m['decoder_contract']['structured_output_api'],'enable_thinking':False,
                    'backend_request_id':out.request_id,'cached_tokens':getattr(out,'num_cached_tokens',None)},
                  'gold_read_by_solver':False}
                save(run/'responses'/f"{row['request_id']}.json",record);done.add(row['request_id'])
            save(run/'batches'/f'{batchid}_COMPLETE.json',{'completed_at_utc':now(),'request_ids':[r['request_id'] for r in batch],'wall_seconds':time.perf_counter()-batchstart})
            print(json.dumps({'status':'LIVE','branch':m['branch'],'completed':len(done),'planned':len(rows)}),flush=True)
        save(run/'COMPLETE.json',{'status':'COMPLETE','completed_at_utc':now(),'actual_submitted_requests':len(submitted),
          'returned_requests':len(done),'model_load_wall_seconds':load_s,'wall_seconds':time.perf_counter()-started,'gold_opened':False})
    except Exception as e:
        error={'error':str(e),'traceback':traceback.format_exc()}
        save(run/'FAILED.json',{'status':'FAILED','failed_at_utc':now(),'submitted_request_count':len(submitted),'returned_request_count':len(done),
          'model_load_wall_seconds':load_s,'wall_seconds':time.perf_counter()-started,'automatic_retry':False,**error})
        # One terminal record for every planned request: backend errors/missing stay in denominator.
        for row in rows:
            if row['request_id'] in done:continue
            save(run/'responses'/f"{row['request_id']}.json",{'request_id':row['request_id'],'candidate_id':row['candidate_id'],'root_id':row['root_id'],
               'question_type':row['question_type'],'branch':m['branch'],'status':'BACKEND_ERROR' if row['request_id'] in submitted else 'NOT_SUBMITTED',
               'raw_output':None,'response_committed_at_utc':now(),'usage':None,'usage_status':'UNKNOWN_NOT_ZERO' if row['request_id'] in submitted else 'NO_REQUEST_SENT',**error})
        raw_freeze(run,m);raise
    finally:
        if backend:backend.close()
    raw_freeze(run,m)

def score(run):
    require(not (run/'SCORING_STARTED.json').exists(),'SCORES_ALREADY_STARTED_NO_OVERWRITE')
    m=read(run/'manifest.json');freeze=read(run/'RAW_RESPONSE_FREEZE.json');verify_bindings(freeze['files']);verify_bindings([freeze['manifest_binding']])
    rows=m['requests'];require(len(freeze['files'])==len(rows),'RAW_DENOMINATOR_MISMATCH')
    # Record the first possible Gold JSON access only after committed full trace.
    goldpath=pathlib.Path(m['private_answer_binding']['path']);verify_bindings([m['private_answer_binding']])
    before=now();save(run/'SCORING_STARTED.json',{'gold_first_read_at_utc':before,'raw_response_freeze_binding':bind(run/'RAW_RESPONSE_FREEZE.json'),
       'private_answer_binding':m['private_answer_binding'],'all_raw_responses_committed_before_gold':True,'gold_read_scope':'First private QA JSON read by this scoring process; not a claim that other tasks or historical reviewers never accessed Gold.'})
    private={q['candidate_id']:q for q in read(goldpath)['questions']};after=now()
    maps=read(run/'private/DISPLAY_MAPPINGS.json');public=read(run/'PUBLIC_QUESTIONS.json')['questions'];scorer=a.native_scoring()
    scored=[]
    for row in rows:
        q=private[row['candidate_id']];mapping=maps['public_to_canonical'][q['candidate_id']]
        original=next(x for x in public if x['candidate_id']==q['candidate_id'])
        require({k:q[k] for k in a.PUBLIC_KEYS if k in q}==original,'PRIVATE_PUBLIC_CONTENT_DISAGREEMENT')
        g=[mapping[x] for x in q['gold']];qscore={'question_type':q['question_type'],'gold_content_ids':g}
        raw=read(run/'responses'/f"{row['request_id']}.json");scoretext=raw['raw_output'];parse_error=None;structured=None
        if m['branch']=='evidence' and raw['status']=='RETURNED':
            try:
                import jsonschema
                structured=json.loads(scoretext);jsonschema.validate(structured,row['json_schema'])
                require(len({x['label'] for x in structured['evidence']})==len(row['display'].get('options',row['display'].get('items'))),'DUPLICATE_EVIDENCE_LABEL')
                scoretext='<box>'+','.join(sorted(structured['answer']) if q['question_type']=='multi_select' else structured['answer'])+'</box>'
            except Exception as e:parse_error=str(e);scoretext=None
        s=scorer.score_response(qscore,{'variant':maps['requests'][row['request_id']]},scoretext)
        field='options' if q['question_type']=='multi_select' else 'items'
        _,solver=a.native()
        if scoretext is not None and scoretext not in a.choices(q['question_type'],row['display'][field]):parse_error='OUTSIDE_NATIVE_FINITE_CHOICE_SET'
        valid=raw['status']=='RETURNED' and raw['metadata']['finish_reason'] in ('stop','eos') and s.get('scored') is True and parse_error is None
        exact=bool(valid and s.get('exact_match'))
        metrics={}
        if valid and q['question_type']=='multi_select':
            pred=set(s['content_prediction']);gold=set(g);metrics={'fp':len(pred-gold),'fn':len(gold-pred),'tp':len(pred&gold),
              'jaccard':len(pred&gold)/len(pred|gold),'overselected':len(pred)>len(gold),'selected_count':len(pred),'gold_count':len(gold)}
        elif valid:
            pred=s['content_prediction'];rank={v:i for i,v in enumerate(pred)};pairs=[(l,r) for i,l in enumerate(g) for r in g[i+1:]]
            metrics['pair_accuracy']=sum(rank[l]<rank[r] for l,r in pairs)/len(pairs)
            # Within/across appearance scenes use source-bound frozen private anchors.
            anchors=q.get('order_anchor_spans') or q.get('item_evidence') or {};scene={}
            for label,e in anchors.items():
                ref=e.get('turn_ref') or (e.get('order_anchor') or {}).get('turn_ref') or (e.get('primary_citation') or {}).get('turn_ref')
                if ref:scene[mapping[label]]=ref.rsplit(':T',1)[0]
            for kind,same in [('within_scene',True),('across_scene',False)]:
                p=[(l,r) for l,r in pairs if l in scene and r in scene and (scene[l]==scene[r])==same]
                metrics[kind]={'correct_pairs':sum(rank[l]<rank[r] for l,r in p),'total_pairs':len(p),'accuracy':sum(rank[l]<rank[r] for l,r in p)/len(p) if p else None}
        scored.append({'request_id':row['request_id'],'candidate_id':row['candidate_id'],'root_id':row['root_id'],'question_type':row['question_type'],
          'variant_id':row['variant_id'],'valid':valid,'exact':exact,'score':s,'metrics':metrics,'parse_error':parse_error,
          'response_binding':bind(run/'responses'/f"{row['request_id']}.json"),'response_committed_at_utc':raw['response_committed_at_utc'],
          'gold_first_read_at_utc':before,'structured_evidence':structured})
    qs=[]
    for q in public:
        records=[r for r in scored if r['candidate_id']==q['candidate_id']];n=len(records);correct=sum(r['exact'] for r in records)
        preds=[json.dumps(r['score']['content_prediction']) for r in records if r['valid']]
        qs.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],'question_type':q['question_type'],'family_id':q['root_id']+'|'+q['question_type'],
           'denominator_calls':n,'valid_calls':sum(r['valid'] for r in records),'exact_calls':correct,'exact_rate':correct/n,
           'stratum':'incorrect' if correct==0 else 'stably_correct' if correct==n else 'occasionally_correct',
           'display_permutation_consistency':max(collections.Counter(preds).values())/n if preds else 0,
           'source_coverage':1.0})
    bytype={}
    for typ in ['multi_select','ordering']:
        sub=[q for q in qs if q['question_type']==typ];rootrates=collections.defaultdict(list)
        for q in sub:rootrates[q['root_id']].append(q['exact_rate'])
        bytype[typ]={'questions':len(sub),'calls':sum(q['denominator_calls'] for q in sub),'question_equal_exact':statistics.mean(q['exact_rate'] for q in sub),
          'root_equal_exact':statistics.mean(statistics.mean(v) for v in rootrates.values())}
    rr=collections.defaultdict(list)
    for q in qs:rr[q['root_id']].append(q['exact_rate'])
    returned=[read(run/'responses'/f"{r['request_id']}.json") for r in rows]
    costs={'known_prompt_tokens':sum(r.get('metadata',{}).get('prompt_token_count',0) for r in returned),
      'known_completion_tokens':sum(r.get('metadata',{}).get('completion_token_count',0) for r in returned),
      'requests_with_unknown_usage':sum(r['status']=='BACKEND_ERROR' for r in returned),'not_submitted_requests':sum(r['status']=='NOT_SUBMITTED' for r in returned),
      'currency_cost':None,'currency_cost_reason':'Local GPU inference; monetary billing unavailable; use measured tokens and wall seconds.'}
    status='COMPLETE' if (run/'COMPLETE.json').exists() else 'FAILED_WITH_FIXED_DENOMINATOR'
    save(run/'SCORES.json',{'status':status,'branch':m['branch'],'scored_at_utc':now(),'gold_first_read_at_utc':before,'gold_read_completed_at_utc':after,
       'raw_response_freeze_binding':bind(run/'RAW_RESPONSE_FREEZE.json'),'denominator_calls':len(rows),'denominator_questions':len(qs),'denominator_roots':len(rr),
       'valid_calls':sum(r['valid'] for r in scored),'exact_calls':sum(r['exact'] for r in scored),'question_equal_exact':statistics.mean(q['exact_rate'] for q in qs),
       'root_equal_exact':statistics.mean(statistics.mean(v) for v in rr.values()),'by_type':bytype,'cost':costs,'questions':qs,'requests':scored,
       'use_as_final_eval':False,'training_started':False})
    print(json.dumps({'status':status,'questions':len(qs),'calls':len(rows),'valid':sum(r['valid'] for r in scored),'exact':sum(r['exact'] for r in scored),'by_type':bytype}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    for cmd in ['prepare','verify','live','score']:
        s=sub.add_parser(cmd);s.add_argument('--run',type=pathlib.Path,default=DEFAULT_RUN)
        if cmd=='prepare':s.add_argument('--admission',type=pathlib.Path);s.add_argument('--direct-run',type=pathlib.Path)
        if cmd in ['verify','live']:s.add_argument('--audit',type=pathlib.Path,required=cmd=='live')
    arg=p.parse_args()
    if arg.cmd=='prepare':prepare(arg.run,arg.admission,arg.direct_run)
    elif arg.cmd=='verify':verify(arg.run,arg.audit);print('PASS: frozen sources/code/model/tokenizer/runtime')
    elif arg.cmd=='live':live(arg.run,arg.audit)
    elif arg.cmd=='score':score(arg.run)
if __name__=='__main__':main()
