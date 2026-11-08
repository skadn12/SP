"""Explicit frozen-policy direct + evidence evaluation; no generation on prepare/verify.

All source/question admission precedes inference. Gold JSON first parsed by score,
after both direct/evidence terminal response sets are committed. No retries.
"""
from __future__ import annotations
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
WEEK=HERE.parents[1]
OLD=WEEK/'day3_targeted_4root_v1/runtime'
sys.path.insert(0,str(OLD))
import common as c
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
evidence=module('realization_evidence_prompt',WEEK/'day3_diagnostic_handoff_v1/runtime/evidence_handoff.py')
legacy_score=module('realization_native_scorer',OLD/'score.py')

def save_new(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')

def policy_check(policy):
 c.require(policy['namespace'].startswith('train.feedback'),'TRAIN_FEEDBACK_ONLY')
 c.require(policy['direct']['display_permutations']==3 and policy['direct']['samples_per_permutation']==2,'DIRECT_3X2_FIXED')
 c.require(policy['direct']['max_output_tokens']==32,'NATIVE_DIRECT32')
 d=policy['diagnostic'];c.require(d['per_question']==1 and d['display_variant']=='perm_00' and d['sample']==0 and d['seed_policy']=='same_as_paired_direct','EVIDENCE_ONE_FIXED_PAIRED_REQUEST')
 c.require(type(d['max_output_tokens']) is int and d['max_output_tokens']==3072,'ROOT_FROZEN_DIAGNOSTIC3072')
 c.require(policy['temperature']==.8 and policy['top_p']==.9 and policy['enable_thinking'] is False,'NATIVE_STOCHASTIC_SETTINGS')
 c.require(policy['automatic_retry'] is False and policy['source_truncation'] is False,'NO_RETRY_NO_TRUNCATION')
 r=policy['runtime'];c.require(r['max_model_len']==16384 and r['batch_size']==8 and r['gpu_memory_utilization']==.8,'EXPLICIT_RUNTIME_CONFIG')
 c.require(policy['model']=={'name':'S0_M_clean','model':str(c.MODEL),'tokenizer':str(c.TOKENIZER)},'ORIGINAL_MODEL_IDENTITY')
 c.require(set(policy['allowed_roots'])=={'cc-train-episode-103866','cc-train-episode-104814','cc-train-episode-109164','cc-train-episode-112812'} and len(policy['allowed_roots'])==4,'FOUR_FIXED_TRAIN_ROOTS')
 slots=policy['planned_slots'];c.require(len(slots)==len({s['candidate_id'] for s in slots}),'UNIQUE_PLANNED_SLOTS')
 c.require(all(set(s)=={'candidate_id','root_id','question_type','item_count'} for s in slots),'PUBLIC_SLOT_METADATA_WHITELIST_NO_PRIVATE_FAMILY')
 c.require({s['root_id'] for s in slots}<=set(policy['allowed_roots']),'SLOT_ROOT_BOUNDARY')
 c.require(policy['budget']['max_direct_calls']==len(slots)*6 and policy['budget']['max_diagnostic_calls']==len(slots),'FIXED_CALL_MAXIMA')
 for k in ('display_seed','sampling_seed'):c.require(type(policy['direct'][k]) is int,'INTEGER_SEED_REQUIRED')

def params(row,decoder_api):
 if row['branch']=='direct':return c.a.sampling_params(row,decoder_api)
 from vllm import SamplingParams
 return SamplingParams(temperature=.8,top_p=.9,max_tokens=row['max_output_tokens'],seed=row['seed'])

def code_paths():
 return sorted(set(c.code_paths())|{Path(__file__).resolve(),Path(evidence.__file__).resolve(),Path(legacy_score.__file__).resolve()})

def make_plan(policy,questions,tokenizer,source_lookup):
 rows=[];prompts={};tokens={};maps={};canonical={}
 slots={s['candidate_id']:s for s in policy['planned_slots']}
 for q in questions:
  c.a.public_question(q);cid=q['candidate_id'];field='options' if q['question_type']=='multi_select' else 'items'
  c.safe_id(cid);c.safe_id(q['root_id'])
  c.require(cid in slots and q['root_id'] in policy['allowed_roots'],'UNPLANNED_QUESTION_OR_ROOT')
  s=slots[cid];c.require(s['root_id']==q['root_id'] and s['question_type']==q['question_type'] and s['item_count']==len(q[field]),'SLOT_IDENTITY_DRIFT')
  source=source_lookup(q);canonical[cid]=c.a.content_ids(q)
  variants=c.a.variants(q,policy['direct']['display_seed'],3)
  for variant in variants:
   text=c.a.direct_prompt(tokenizer,q,variant,source)
   for sample in range(2):
    rid=f"{cid}__{variant['variant_id']}__direct__s{sample:02d}"
    row={'request_id':rid,'candidate_id':cid,'root_id':q['root_id'],'question_type':q['question_type'],'branch':'direct','variant_id':variant['variant_id'],'display':{field:variant[field]},'slot':sample,'seed':policy['direct']['sampling_seed']+sum(r['branch']=='direct' for r in rows),'source_turns':len(source['turns']),'source_sha256':q['source_sha256'],'max_output_tokens':32}
    rows.append(row);prompts[rid]=text;maps[rid]=variant
  base=next(r for r in rows if r['candidate_id']==cid and r['variant_id']=='perm_00' and r['slot']==0)
  rid=base['request_id'].replace('__direct__','__evidence__')
  row={k:base[k] for k in ('candidate_id','root_id','question_type','variant_id','display','slot','seed','source_turns','source_sha256')}
  row.update(request_id=rid,branch='evidence',direct_request_id=base['request_id'],max_output_tokens=policy['diagnostic']['max_output_tokens'],json_schema=evidence.schema(q))
  rows.append(row);prompts[rid]=evidence.prompt(tokenizer,q,variants[0],source);maps[rid]=variants[0]
  for r in [r for r in rows if r['candidate_id']==cid]:
   text=prompts[r['request_id']]
   c.require(all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in text for t in source['turns']),'INCOMPLETE_SOURCE_PROMPT')
   ids=tokenizer.encode(text,add_special_tokens=False)
   c.require(len(ids)+r['max_output_tokens']<=16384,'FULL_CONTEXT_OVERFLOW_NO_TRUNCATION:'+r['request_id'])
   r.update(prompt_sha256=c.rt.textsha(text),prompt_tokens=len(ids));tokens[r['request_id']]=ids
 return rows,prompts,tokens,{'requests':maps,'public_to_canonical':canonical}

def prepare(args):
 c.environment();run=args.run.resolve();private=args.private_output.resolve()
 c.require(c.below(run,HERE) and c.below(private,HERE),'OUTPUTS_MUST_BE_WITHIN_NEW_RUNTIME')
 c.require(not run.exists() and not private.exists(),'NO_OVERWRITE_RUN_OR_PRIVATE')
 c.require(not c.below(run,private) and not c.below(private,run),'PRIVATE_SIDECAR_DISJOINT')
 policy=c.read(args.policy);policy_check(policy)
 payload=c.read(args.public);c.require(set(payload)=={'questions'},'PUBLIC_WRAPPER_WHITELIST')
 questions=payload['questions'];c.require(questions and len({q['candidate_id'] for q in questions})==len(questions),'NONEMPTY_UNIQUE_ACCEPTED_QUESTIONS')
 admission=c.read(args.admission);gold_binding=c.bind(args.private)
 c.require(admission['status']=='PASS' and admission['namespace']==policy['namespace'],'CONTENT_ADMISSION_REQUIRED')
 c.require(admission['planned_slots']==len(policy['planned_slots']) and admission['admitted_count']==len(questions),'ADMISSION_DENOMINATOR')
 c.require(set(admission['candidate_ids'])=={q['candidate_id'] for q in questions},'ADMISSION_CANDIDATES')
 c.require(admission['public_questions']['sha256']==c.sha(args.public) and admission['private_questions']['sha256']==gold_binding['sha256'],'ADMISSION_CONTENT_HASH')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(c.TOKENIZER),local_files_only=True)
 _,native=c.a.native();decoder=native.generation_config('multi_select')
 c.require(all(decoder[k]==v for k,v in {'temperature':.8,'top_p':.9,'max_tokens':32,'enable_thinking':False}.items()),'NATIVE_DECODER_DRIFT')
 rows,prompts,tokens,maps=make_plan(policy,questions,tok,c.rt.source_check)
 for row in rows:params(row,decoder['structured_output_api'])
 model_bindings=c.rt.models() # Real model/tokenizer byte verification; no GPU.
 run.mkdir(parents=True);private.mkdir(parents=True,mode=0o700)
 public_admission={k:admission[k] for k in ('status','namespace','planned_slots','admitted_count','candidate_ids','public_questions','private_questions')}
 for filename,obj in [('PUBLIC_QUESTIONS.json',payload),('PROTOCOL.json',policy),('CONTENT_ADMISSION.json',public_admission),('DISPLAY_MAPPINGS.json',maps),('PROMPT_TOKEN_IDS.json',tokens)]:save_new(run/filename,obj)
 for q in questions:
  dest=run/'sources'/(q['root_id']+'.json');dest.parent.mkdir(exist_ok=True)
  if not dest.exists():shutil.copy2(q['source_path'],dest)
  c.require(c.sha(dest)==q['source_sha256'],'SOURCE_COPY_MISMATCH')
 for rid,text in prompts.items():
  p=run/'prompts'/(rid+'.txt');p.parent.mkdir(exist_ok=True)
  with p.open('x') as f:f.write(text)
 counts=collections.Counter(r['branch'] for r in rows)
 c.require(counts['direct']==6*len(questions)<=policy['budget']['max_direct_calls'] and counts['evidence']==len(questions)<=policy['budget']['max_diagnostic_calls'],'PREPARED_BUDGET_DRIFT')
 manifest={'schema_version':'feedback_realization_eval_v1','status':'PREPARED_AWAITING_INDEPENDENT_PREFLIGHT','created_at_utc':c.now(),'namespace':policy['namespace'],'model':str(c.MODEL),'tokenizer':str(c.TOKENIZER),'solver_model':'S0_M_clean','question_count':len(questions),'planned_slots':len(policy['planned_slots']),'request_counts':dict(counts),'requests':rows,'decoder_contract':decoder,'public_input_bindings':[c.bind(p) for p in sorted(run.rglob('*')) if p.is_file()],'model_bindings':model_bindings,'code_bindings':[c.bind(p) for p in code_paths()],'versions':c.rt.versions(),'python_prefix':sys.prefix,'gold_parsed_by_prepare':False,'gold_bytes_copied_by_prepare':True,'source_truncation':False,'automatic_retry':False,'use_as_final_eval':False,'training_started':False,'branch_limits':{b:{'min_prompt_tokens':min(r['prompt_tokens'] for r in rows if r['branch']==b),'max_prompt_tokens':max(r['prompt_tokens'] for r in rows if r['branch']==b),'min_context_headroom':min(16384-r['prompt_tokens']-r['max_output_tokens'] for r in rows if r['branch']==b)} for b in ('direct','evidence')}}
 save_new(run/'manifest.json',manifest)
 save_new(run/'PREFLIGHT_FREEZE.json',{'status':manifest['status'],'created_at_utc':c.now(),'files':[c.bind(p) for p in sorted(run.rglob('*')) if p.is_file()],'requests':dict(counts),'inference_started':False,'public_only':True})
 shutil.copy2(args.private,private/'PRIVATE_QUESTIONS.json');(private/'PRIVATE_QUESTIONS.json').chmod(0o600)
 c.require(c.sha(private/'PRIVATE_QUESTIONS.json')==gold_binding['sha256'],'PRIVATE_COPY_MISMATCH')
 save_new(private/'PRIVATE_BINDINGS.json',{'public_manifest_binding':c.bind(run/'manifest.json'),'public_preflight_binding':c.bind(run/'PREFLIGHT_FREEZE.json'),'original_private_binding':gold_binding,'private_answer_binding':c.bind(private/'PRIVATE_QUESTIONS.json'),'private_json_parsed_by_prepare':False})
 print(json.dumps({'status':manifest['status'],'question_count':len(questions),'requests':dict(counts),'branch_limits':manifest['branch_limits']}))

def verify(run,audit=None,models=True):
 c.environment();run=Path(run).resolve();f=c.read(run/'PREFLIGHT_FREEZE.json');c.require(f['status']=='PREPARED_AWAITING_INDEPENDENT_PREFLIGHT','INPUT_FREEZE_REQUIRED')
 c.require(all(c.below(b['path'],run) for b in f['files']),'PUBLIC_FROZEN_INPUTS_LOCAL');c.verify_bindings(f['files'])
 m=c.read(run/'manifest.json');p=c.read(run/'PROTOCOL.json');policy_check(p)
 c.require(m['model']==str(c.MODEL) and m['tokenizer']==str(c.TOKENIZER),'MODEL_PATH_DRIFT')
 c.verify_bindings(m['code_bindings']);c.verify_bindings(m['public_input_bindings'])
 if models:c.verify_bindings(m['model_bindings'])
 c.require(m['versions']==c.rt.versions() and m['python_prefix']==sys.prefix,'VERSION_OR_PYTHON_DRIFT')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(c.TOKENIZER),local_files_only=True)
 questions=c.read(run/'PUBLIC_QUESTIONS.json')['questions']
 lookup=lambda q:c.rt.source_check(dict(q,source_path=str(run/'sources'/(q['root_id']+'.json'))))
 rows,prompts,tokens,maps=make_plan(p,questions,tok,lookup)
 c.require(rows==m['requests'] and tokens==c.read(run/'PROMPT_TOKEN_IDS.json') and maps==c.read(run/'DISPLAY_MAPPINGS.json'),'PUBLIC_PLAN_RECONSTRUCTION_DRIFT')
 c.require(all((run/'prompts'/(rid+'.txt')).read_text()==text for rid,text in prompts.items()),'PROMPT_RECONSTRUCTION_DRIFT')
 for row in rows:
  actual=params(row,m['decoder_contract']['structured_output_api'])
  if row['branch']=='evidence':c.require(getattr(actual,'structured_outputs',None) is None and getattr(actual,'guided_decoding',None) is None and not actual.stop,'EVIDENCE_NO_CONSTRAINT')
 if audit:
  a=c.read(audit);c.require(a['status']=='PASS','INDEPENDENT_AUDIT_REQUIRED')
  bs=a.get('audited_files') or a.get('files') or []
  c.require(any(Path(b['path']).resolve()==run/'PREFLIGHT_FREEZE.json' and b['sha256']==c.sha(run/'PREFLIGHT_FREEZE.json') for b in bs),'AUDIT_BIND_THIS_INPUT_FREEZE');c.verify_bindings(bs)
 return m

def terminal(row,status,**extra):
 return {k:row[k] for k in ('request_id','candidate_id','root_id','question_type','branch','variant_id','slot','seed','prompt_sha256')}|{'status':status,'raw_output':None,'response_committed_at_utc':c.now(),'model_path':str(c.MODEL),'gold_read_by_solver':False,'automatic_retry':False,'source_truncation':False,**extra}

def live(run,branch,audit):
 run=Path(run).resolve();out=run/branch;c.require(not out.exists(),'BRANCH_ALREADY_STARTED_NO_RETRY');m=verify(run,audit)
 rows=[r for r in m['requests'] if r['branch']==branch];policy=c.read(run/'PROTOCOL.json');out.mkdir()
 save_new(out/'STARTED.json',{'started_at_utc':c.now(),'manifest':c.bind(run/'manifest.json'),'audit':c.bind(audit),'branch':branch,'gold_opened':False})
 backend=None;submitted=set();done=set();failure=None;started=time.perf_counter();load_seconds=None
 try:
  from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
  _,native=c.a.native();native.ensure_vllm_environment()
  max_tokens=32 if branch=='direct' else policy['diagnostic']['max_output_tokens']
  backend=VLLMSolverBackend(model_path=c.MODEL,tokenizer_path=c.TOKENIZER,max_model_len=16384,temperature=.8,top_p=.9,max_tokens=max_tokens,gpu_memory_utilization=.8)
  load_seconds=backend.load_latency_seconds;c.require(str(backend.model_identifier)==m['model'],'BACKEND_MODEL_MISMATCH')
  tokens=c.read(run/'PROMPT_TOKEN_IDS.json')
  for offset in range(0,len(rows),8):
   batch=rows[offset:offset+8];bid=f'batch{offset//8:04d}';prompts=[(run/'prompts'/(r['request_id']+'.txt')).read_text() for r in batch];sampling=[params(r,m['decoder_contract']['structured_output_api']) for r in batch]
   for r,text in zip(batch,prompts):c.require(c.rt.textsha(text)==r['prompt_sha256'],'PROMPT_CHANGED')
   at=c.now();t=time.perf_counter();save_new(out/'batches'/(bid+'_SUBMITTED.json'),{'submitted_at_utc':at,'request_ids':[r['request_id'] for r in batch],'sampling_params':[repr(x) for x in sampling],'automatic_retry':False});submitted.update(r['request_id'] for r in batch)
   outputs=backend.engine.generate(prompts,sampling,use_tqdm=False)
   for index,row in enumerate(batch):
    if index>=len(outputs):record=terminal(row,'BACKEND_ERROR',error='MISSING_OUTPUT',usage=None,usage_status='UNKNOWN_NOT_ZERO')
    else:
     output=outputs[index];comps=list(output.outputs);primary=comps[0] if comps else None;errors=[]
     if output.prompt!=prompts[index] or list(output.prompt_token_ids)!=tokens[row['request_id']]:errors.append('PROMPT_OR_TOKEN_DRIFT')
     if len(comps)!=1:errors.append('COMPLETION_COUNT_MISMATCH')
     record=terminal(row,'RETURNED',raw_output=primary.text if primary else None,request_submitted_at_utc=at,prompt_token_ids=list(output.prompt_token_ids),completion_token_ids=list(primary.token_ids) if primary else None,output_validation_errors=errors,all_completions=[{'text':x.text,'token_ids':list(x.token_ids),'finish_reason':x.finish_reason,'stop_reason':x.stop_reason} for x in comps],metadata={'prompt_token_count':len(output.prompt_token_ids),'completion_token_count':len(primary.token_ids) if primary else None,'finish_reason':primary.finish_reason if primary else None,'stop_reason':primary.stop_reason if primary else None,'temperature':.8,'top_p':.9,'max_tokens':row['max_output_tokens'],'enable_thinking':False,'structured_output_api':m['decoder_contract']['structured_output_api'] if branch=='direct' else None,'constrained_decoding':branch=='direct','backend_request_id':output.request_id,'cached_tokens':getattr(output,'num_cached_tokens',None)})
    save_new(out/'responses'/(row['request_id']+'.json'),record);done.add(row['request_id'])
   if len(outputs)>len(batch):save_new(out/'batches'/(bid+'_EXTRA_OUTPUTS.json'),{'extra_outputs':[{'request_id':x.request_id,'prompt':x.prompt,'prompt_token_ids':list(x.prompt_token_ids),'completions':[{'text':y.text,'token_ids':list(y.token_ids),'finish_reason':y.finish_reason,'stop_reason':y.stop_reason} for y in x.outputs]} for x in outputs[len(batch):]]})
   c.require(len(outputs)==len(batch),'OUTPUT_COUNT_MISMATCH');save_new(out/'batches'/(bid+'_COMPLETE.json'),{'completed_at_utc':c.now(),'request_ids':[r['request_id'] for r in batch],'wall_seconds':time.perf_counter()-t});print(json.dumps({'branch':branch,'completed':len(done),'planned':len(rows)}),flush=True)
 except BaseException as error:failure={'error':str(error),'exception_type':type(error).__name__,'traceback':traceback.format_exc()}
 finally:
  if backend:
   try:backend.close()
   except BaseException as error:failure=failure or {'error':str(error),'exception_type':type(error).__name__,'traceback':traceback.format_exc()}
  if failure:
   save_new(out/'FAILED.json',{'status':'FAILED','failed_at_utc':c.now(),'actual_submitted_requests':len(submitted),'returned_requests':len(done),'wall_seconds':time.perf_counter()-started,'model_load_wall_seconds':load_seconds,**failure})
   for row in rows:
    if row['request_id'] not in done:save_new(out/'responses'/(row['request_id']+'.json'),terminal(row,'BACKEND_ERROR' if row['request_id'] in submitted else 'NOT_SUBMITTED',usage=None,usage_status='UNKNOWN_NOT_ZERO' if row['request_id'] in submitted else 'NO_REQUEST_SENT',**failure))
  else:save_new(out/'COMPLETE.json',{'status':'COMPLETE','completed_at_utc':c.now(),'actual_submitted_requests':len(submitted),'returned_requests':len(done),'wall_seconds':time.perf_counter()-started,'model_load_wall_seconds':load_seconds,'gold_opened':False})
  paths=[out/'responses'/(r['request_id']+'.json') for r in rows];c.require(all(p.is_file() for p in paths) and set((out/'responses').glob('*.json'))==set(paths),'ALL_TERMINAL_RESPONSES_REQUIRED')
  ledger=[p for p in sorted(out.rglob('*')) if p.is_file()]
  save_new(out/'RAW_RESPONSE_FREEZE.json',{'status':'COMMITTED','created_at_utc':c.now(),'branch':branch,'planned_request_count':len(rows),'manifest_binding':c.bind(run/'manifest.json'),'files':[c.bind(p) for p in paths],'execution_ledger_bindings':[c.bind(p) for p in ledger],'gold_opened_by_live':False,'automatic_retry':False,'source_truncation':False})
 if failure:raise RuntimeError('LIVE_FAILED_FIXED_DENOMINATOR:'+failure['error'])

def score(run,private):
 import jsonschema
 run=Path(run).resolve();private=Path(private).resolve();m=verify(run,models=False)
 c.require(not (run/'SCORES.json').exists() and not (run/'SCORING_STARTED.json').exists(),'NO_SCORE_OVERWRITE')
 raws={};freezes=[]
 for branch in ('direct','evidence'):
  out=run/branch;f=c.read(out/'RAW_RESPONSE_FREEZE.json');c.require(f['status']=='COMMITTED','BOTH_BRANCH_RAW_FREEZES_REQUIRED')
  expected={out/'responses'/(r['request_id']+'.json') for r in m['requests'] if r['branch']==branch}
  c.require({Path(b['path']).resolve() for b in f['files']}==expected and set((out/'responses').glob('*.json'))==expected,'RAW_SET_FIXED_DENOMINATOR')
  c.verify_bindings(f['files']+f['execution_ledger_bindings']+[f['manifest_binding']]);freezes.append(c.bind(out/'RAW_RESPONSE_FREEZE.json'))
  for path in expected:
   raw=c.read(path);raws[raw['request_id']]=raw
   row=next(r for r in m['requests'] if r['request_id']==raw['request_id'])
   c.require(all(raw[k]==row[k] for k in ('request_id','candidate_id','root_id','variant_id','slot','seed','prompt_sha256')),'RAW_IDENTITY_MISMATCH')
   c.require(c.timestamp(raw['response_committed_at_utc'])<=c.timestamp(f['created_at_utc']),'RAW_AFTER_FREEZE')
 bindings=c.read(private/'PRIVATE_BINDINGS.json');c.verify_bindings([bindings['public_manifest_binding'],bindings['public_preflight_binding'],bindings['private_answer_binding']]);c.require(Path(bindings['public_manifest_binding']['path']).resolve()==run/'manifest.json','PRIVATE_RUN_MISMATCH')
 save_new(run/'SCORING_STARTED.json',{'gold_first_read_at_utc':c.now(),'raw_freeze_bindings':freezes,'private_answer_binding':bindings['private_answer_binding'],'gold_scope':'First private JSON read in this scorer after both branches frozen; prior author/reviewer exposure not denied.'})
 golds={q['candidate_id']:q for q in c.read(bindings['private_answer_binding']['path'])['questions']};questions=c.read(run/'PUBLIC_QUESTIONS.json')['questions'];qby={q['candidate_id']:q for q in questions};maps=c.read(run/'DISPLAY_MAPPINGS.json')
 c.require(set(golds)==set(qby),'PRIVATE_PUBLIC_CANDIDATE_SET')
 for q in questions:c.require({k:golds[q['candidate_id']][k] for k in c.a.PUBLIC_KEYS if k in golds[q['candidate_id']]}==q,'PRIVATE_PUBLIC_CONTENT')
 direct=[];diagnostics=[]
 for row in m['requests']:
  rid=row['request_id'];raw=raws[rid];cid=row['candidate_id'];canonical=maps['public_to_canonical'][cid];reverse={v:k for k,v in canonical.items()};variant=maps['requests'][rid];gold=golds[cid]['gold']
  c.require(len(reverse)==len(canonical) and set(gold)<=set(canonical),'UNAMBIGUOUS_ORIGINAL_GOLD_LABEL_DOMAIN')
  if row['branch']=='direct':
   rec=legacy_score.score_one(row,raw,variant,canonical,gold);rec['response_binding']=c.bind(run/'direct/responses'/(rid+'.json'));direct.append(rec);continue
  errors=[];parsed=None;checks=[];truncated=raw.get('metadata',{}).get('finish_reason')=='length'
  if raw['status']!='RETURNED' or not isinstance(raw.get('raw_output'),str):errors.append('NO_RETURNED_TEXT')
  else:
   try:
    parsed=json.loads(raw['raw_output'].strip());jsonschema.validate(parsed,row['json_schema']);labels=set(row['display']['options' if row['question_type']=='multi_select' else 'items']);c.require(len(parsed['item_evidence'])==len(labels) and {i['label'] for i in parsed['item_evidence']}==labels,'DUPLICATE_OR_MISSING_ITEM_EVIDENCE')
   except Exception as error:errors.append(type(error).__name__+':'+str(error)[:1500])
  if truncated:errors.append('OUTPUT_TRUNCATED')
  errors+=raw.get('output_validation_errors') or []
  valid=not errors;display_pred=parsed['final_answer'] if valid else None;content_pred=[variant['display_to_content'][x] for x in display_pred] if valid else None;canonical_pred=[reverse[x] for x in content_pred] if valid else None
  exact=bool(valid and (set(canonical_pred)==set(gold) if row['question_type']=='multi_select' else canonical_pred==gold))
  if valid:
   turns={t['turn_ref']:t['text'] for t in c.read(run/'sources'/(row['root_id']+'.json'))['turns']}
   for item in parsed['item_evidence']:
    for e in item['citations']:checks.append({'label':item['label'],**e,'turn_ref_exists':e['turn_ref'] in turns,'exact_substring':e['turn_ref'] in turns and e['quote'] in turns[e['turn_ref']]})
  diagnostics.append({k:row[k] for k in ('request_id','candidate_id','root_id','question_type','direct_request_id')}|{'raw_response_binding':c.bind(run/'evidence/responses'/(rid+'.json')),'format_valid':valid,'invalid_reasons':errors,'truncated':truncated,'parsed_response':parsed,'display_prediction':display_pred,'prediction_content_ids':content_pred,'canonical_prediction':canonical_pred,'canonical_gold':gold,'exact':exact,'citation_checks':checks,'citation_count':len(checks),'exact_quote_count':sum(x['exact_substring'] for x in checks),'semantic_review':'PENDING','unverified_rationale_training_target':False})
 dq=[legacy_score.summarize_question(q,[r for r in direct if r['candidate_id']==q['candidate_id']]) for q in questions]
 for q in dq:
  q['fact_family_id']=golds[q['candidate_id']].get('fact_family_id') or golds[q['candidate_id']].get('family_id')
  if q['question_type']=='multi_select':q['mean_jaccard']=statistics.mean(r['metrics']['jaccard'] for r in direct if r['candidate_id']==q['candidate_id'])
 by_type={}
 for kind in ('multi_select','ordering'):
  subset=[q for q in dq if q['question_type']==kind];diag=[q for q in diagnostics if q['question_type']==kind]
  by_type[kind]={'questions':len(subset),'direct_calls':sum(q['denominator_calls'] for q in subset),'direct_question_equal_exact':statistics.mean(q['exact_rate'] for q in subset) if subset else None,'diagnostic_calls':len(diag),'diagnostic_valid':sum(q['format_valid'] for q in diag),'diagnostic_exact':sum(q['exact'] for q in diag),'diagnostic_exact_rate':sum(q['exact'] for q in diag)/len(diag) if diag else None}
  roots=collections.defaultdict(list)
  for q in subset:roots[q['root_id']].append(q['exact_rate'])
  by_type[kind]['direct_root_equal_exact']=statistics.mean(statistics.mean(v) for v in roots.values()) if roots else None
  if subset:
   if kind=='multi_select':by_type[kind].update(direct_question_equal_jaccard=statistics.mean(q['mean_jaccard'] for q in subset),fp_option_events=sum(q['fp_calls'] for q in subset),fn_option_events=sum(q['fn_calls'] for q in subset))
   else:by_type[kind].update(direct_question_equal_pair_accuracy=statistics.mean(q['pair_accuracy'] for q in subset),pair_correct=sum(q['correct_pairs'] for q in subset),pair_total=sum(q['total_pairs'] for q in subset))
 result={'schema_version':'feedback_realization_scores_v1','status':'SCORED_FIXED_DENOMINATORS_SEMANTIC_REVIEW_PENDING','scored_at_utc':c.now(),'primary_evaluation':'direct','manifest_binding':c.bind(run/'manifest.json'),'raw_freezes':freezes,'private_answer_binding':bindings['private_answer_binding'],'planned_slots':m['planned_slots'],'admitted_questions':len(questions),'direct_calls':len(direct),'direct_valid':sum(x['valid'] for x in direct),'direct_exact':sum(x['exact'] for x in direct),'diagnostic_calls':len(diagnostics),'diagnostic_valid':sum(x['format_valid'] for x in diagnostics),'diagnostic_invalid':sum(not x['format_valid'] for x in diagnostics),'diagnostic_truncated':sum(x['truncated'] for x in diagnostics),'by_type':by_type,'questions':dq,'direct_requests':direct,'diagnostics':diagnostics,'A_B_effect_estimate':False,'training_started':False,'invalid_policy':'Keep all fixed requests; invalid exact/pairs zero, unknown MC selections and FP/FN null. No answer extraction from invalid diagnostic output.','label_domains':'display→opaque content ID→original public label before canonical Gold comparison','limits':['No CoT or feedback causal benefit estimate: samples, output budgets and decoding differ across branches.','Semantic evidence correctness requires separate full-source review; short model explanation is not hidden internal reasoning.']}
 result['cost_by_branch']={branch:legacy_score.cost(run/branch,[r for r in m['requests'] if r['branch']==branch],{rid:raw for rid,raw in raws.items() if raw['branch']==branch}) for branch in ('direct','evidence')}
 save_new(run/'SCORES.json',result);save_new(run/'DIAGNOSTIC_PARSED.json',{'status':'PARSED_SEMANTIC_REVIEW_PENDING','questions':diagnostics,'raw_freeze_bindings':freezes})
 print(json.dumps({k:result[k] for k in ('status','direct_calls','direct_valid','direct_exact','diagnostic_calls','diagnostic_valid','diagnostic_invalid','diagnostic_truncated','by_type')}))

def cpu_check():
 c.environment();_,native=c.a.native();native.ensure_vllm_environment();decoder=native.generation_config('multi_select')
 c.require(c.MODEL.is_dir() and c.TOKENIZER.is_dir(),'MODEL_OR_TOKENIZER_MISSING');models=c.rt.models()
 for typ,n in [('multi_select',6),('ordering',3),('ordering',4),('ordering',6)]:
  display={x:str(i) for i,x in enumerate(c.a.labels(typ,n))};field='options' if typ=='multi_select' else 'items';row={'question_type':typ,'display':{field:display},'seed':1,'branch':'direct'};p=params(row,decoder['structured_output_api']);c.require(p.max_tokens==32,'DIRECT_API_MAX32')
 ep=params({'branch':'evidence','max_output_tokens':3072,'seed':1},decoder['structured_output_api']);c.require(getattr(ep,'structured_outputs',None) is None and not ep.stop,'DIAGNOSTIC_UNCONSTRAINED_API')
 result={'status':'PASS','created_at_utc':c.now(),'cpu_only':True,'gpu_started':False,'solver_calls':0,'actual_model':str(c.MODEL),'actual_tokenizer':str(c.TOKENIZER),'model_bindings':models,'versions':c.rt.versions(),'native_decoder_api':decoder['structured_output_api'],'direct_arities_verified':['multi_select6','ordering3','ordering4','ordering6'],'diagnostic_api':'unconstrained_vllm_SamplingParams','diagnostic_budget_example_only_not_execution_freeze':3072,'code_bindings':[c.bind(p) for p in code_paths()]}
 save_new(HERE/'CPU_INTERFACE_CHECK.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('model_bindings','code_bindings')}))

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('cpu-check','prepare','verify','live','score'));ap.add_argument('--run',type=Path);ap.add_argument('--public',type=Path);ap.add_argument('--private',type=Path);ap.add_argument('--policy',type=Path);ap.add_argument('--admission',type=Path);ap.add_argument('--private-output',type=Path);ap.add_argument('--audit',type=Path);ap.add_argument('--branch',choices=('direct','evidence'));args=ap.parse_args()
 if args.command=='cpu-check':cpu_check()
 elif args.command=='prepare':
  c.require(all(getattr(args,k) for k in ('run','public','private','policy','admission','private_output')),'PREPARE_ARGUMENTS_REQUIRED');prepare(args)
 elif args.command=='verify':c.require(args.run,'RUN_REQUIRED');m=verify(args.run,args.audit);print(json.dumps({'status':'PASS','requests':m['request_counts'],'cpu_only':True}))
 elif args.command=='live':c.require(args.run and args.branch and args.audit,'LIVE_RUN_BRANCH_AUDIT_REQUIRED');live(args.run,args.branch,args.audit)
 else:c.require(args.run and args.private,'SCORE_RUN_PRIVATE_REQUIRED');score(args.run,args.private)
if __name__=='__main__':main()
