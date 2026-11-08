#!/usr/bin/env python3
"""Prepare public inputs, run B1 once, then score only after raw output freeze."""
from __future__ import annotations
import argparse,datetime,hashlib,importlib.util,json,os,pathlib,signal,subprocess,sys,time,traceback,shutil
sys.dont_write_bytecode=True
from codex_client import ACTIVE,BASE,LOCK,atomic,bind,digest,load,now,start_token,status
MODEL=pathlib.Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/day5_B1_schedule_v1/merged_model')
RUNTIME=pathlib.Path('/root/autodl-tmp/gxy-selfplay/.venv-local4b/lib/python3.10/site-packages')
WEEK=BASE.parent
EXEC=BASE/'execution'

def frozen_modules():
 p=WEEK/'postweek_question_quality_pilot_v1/run.py';s=importlib.util.spec_from_file_location('resume_readonly_frozen_pilot',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 return m

def prepare():
 if (EXEC/'PREPARATION.json').exists():return load(EXEC/'PREPARATION.json')
 from transformers import AutoTokenizer
 pilot=frozen_modules();tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
 candidates=load(BASE/'CANDIDATES_PRIVATE.json')['candidates'];public=[];private=[];length_failures=[];all_tokens={}
 for c in candidates:
  source=load(c['source_binding']['path']);previous=None;pending=[]
  for variant in [0,1]:
   seed=2026102427+variant;rejected=[]
   while True:
    pub,priv=pilot.gate.prepare_display(c,seed)
    canonical_order=list(priv['display_to_canonical'].values())
    if canonical_order==previous or (c['question_type']=='ordering' and canonical_order==c['gold']):rejected.append(seed);seed+=2;continue
    break
   previous=canonical_order;priv['rejected_display_seeds']=rejected
   display=pub.get('options',pub.get('items'));request_id=c['candidate_id']+f'__perm_{variant:02d}__FULL__THINK_OFF__RESUME1'
   question=dict(c)
   # Same operation and output contract helpers as the sealed original scorer.
   prompt=pilot.diagnostic.make_prompt(tok,question,display,source)
   # Preserve the original closed-loop system wording, without touching source or content.
   prompt=prompt.replace('Follow the question\'s requested operation. Give concise, verifiable explanations before the final boxed answer.',"Follow the question's requested operation. In the final response, give concise, verifiable explanations before the final boxed answer.",1)
   ids=tok.encode(prompt,add_special_tokens=False)
   assert prompt.endswith('<think>\n\n</think>\n\n') or prompt.endswith('<think>\n\n</think>\n\n\n')
   schema=pilot.diagnostic.repair.schema(list('ABCDEF'),[t['turn_ref'] for t in source['turns']],c['question_type'])
   pd=EXEC/'prompts'/f'{request_id}.txt';sd=EXEC/'schemas'/f'{request_id}.json';pd.parent.mkdir(parents=True,exist_ok=True);pd.write_text(prompt);atomic(sd,schema)
   matched_key=c['root_id']+':'+c['slot_id']
   row={k:c[k] for k in ['candidate_id','root_id','arm','slot_id','question_type','task_operation','source_binding','question']}
   row.update(request_id=request_id,cohort='new16_resume',condition='FULL',mode='THINK_OFF',enable_thinking=False,variant_id=f'perm_{variant:02d}',display=display,labels=list('ABCDEF'),sampling_seed=int.from_bytes(hashlib.sha256(matched_key.encode()).digest()[:4],'big')%2147483647,prompt_path=str(pd),schema_path=str(sd),prompt_tokens=len(ids),prompt_binding=bind(pd),schema_binding=bind(sd))
   pending.append((row,priv,ids))
  if any(row['prompt_tokens']>10240 for row,_,_ in pending):
   length_failures.append({'candidate_id':c['candidate_id'],'reason':'prompt_tokens_above_10240','lengths':[row['prompt_tokens'] for row,_,_ in pending]});continue
  for row,priv,ids in pending:
   public.append(row);private.append({'request_id':row['request_id'],'presentation':priv});all_tokens[row['request_id']]=ids
 atomic(EXEC/'PUBLIC_REQUESTS.json',{'requests':public});atomic(EXEC/'PRIVATE_SCORING.json',{'requests':private});atomic(EXEC/'PROMPT_TOKEN_IDS.json',all_tokens)
 atomic(EXEC/'LENGTH_FAILURES.json',length_failures)
 p={'at':now(),'planned_presentations':512,'content_valid_slots':len(candidates),'prepared_presentations':len(public),'length_failure_slots':len(length_failures),'public':bind(EXEC/'PUBLIC_REQUESTS.json'),'private':bind(EXEC/'PRIVATE_SCORING.json'),'tokens':bind(EXEC/'PROMPT_TOKEN_IDS.json'),'question_freeze':bind(BASE/'QUESTION_FREEZE.json'),'model_directory':str(MODEL),'tokenizer_bindings':[bind(MODEL/n) for n in ['tokenizer.json','tokenizer_config.json','chat_template.jinja']]}
 atomic(EXEC/'PREPARATION.json',p);return p

def validate_initialization_recovery(freeze,submitted_count,raw_count):
 if freeze.get('returned_count')!=0 or submitted_count or raw_count:raise RuntimeError('Initialization recovery requires zero submitted requests and zero returned outputs; semantic retries forbidden')
 if not (EXEC/'BACKEND_FAILURE.json').exists():raise RuntimeError('No initialization failure receipt')

def prepare_initialization_recovery():
 freeze=load(EXEC/'RAW_RESPONSE_FREEZE.json')
 validate_initialization_recovery(freeze,len(list((EXEC/'batches').glob('*/SUBMITTED.json'))),len(list((EXEC/'raw').glob('*.json'))))
 archive=EXEC/'initialization_failure_01';archive.mkdir(exist_ok=False)
 names=['STARTED.json','RAW_RESPONSE_FREEZE.json','BACKEND_FAILURE.json','generate.log','SCORES.json','score.log']
 files={}
 for name in names:
  p=EXEC/name
  if p.exists():
   target=archive/name;shutil.copy2(p,target);files[name]={'original':bind(p),'archive':bind(target)}
 for name in ['FINAL_REPORT.json','ANALYSIS.json','RESULTS_ZH.md','INTEGRITY_AUDIT.json','ARTIFACT_MANIFEST.json','IMMUTABLE_INTEGRITY_AFTER.json']:
  p=BASE/name
  if p.exists():
   target=archive/name;shutil.copy2(p,target);files[name]={'original':bind(p),'archive':bind(target)}
 receipt={'at':now(),'reason':'Engine warmup failed in FlashInfer JIT; no prepared request was submitted. Recover initialization only, with native PyTorch sampling.','submitted_requests_before_recovery':0,'returned_responses_before_recovery':0,'prepared_count':freeze['prepared_count'],'changes':{'VLLM_USE_FLASHINFER_SAMPLER':'0'},'sampling_parameters_unchanged':True,'prepared_public_inputs':bind(EXEC/'PUBLIC_REQUESTS.json'),'question_freeze':bind(BASE/'QUESTION_FREEZE.json'),'preserved_failure_artifacts':files,'semantic_retries':0,'training':False}
 atomic(EXEC/'INITIALIZATION_RECOVERY.json',receipt)
 return receipt

def generate(initialization_recovery=False):
 """Public-only process. No candidate, review, private scoring, or sealed-old reads."""
 rows=load(EXEC/'PUBLIC_REQUESTS.json')['requests'];token_ids=load(EXEC/'PROMPT_TOKEN_IDS.json')
 if not initialization_recovery and (EXEC/'RAW_RESPONSE_FREEZE.json').exists():return
 rawdir=EXEC/'raw';rawdir.mkdir(parents=True,exist_ok=True)
 start_path=EXEC/'STARTED.json'
 if initialization_recovery:
  receipt=load(EXEC/'INITIALIZATION_RECOVERY.json')
  if receipt['prepared_public_inputs']!=bind(EXEC/'PUBLIC_REQUESTS.json') or receipt['question_freeze']!=bind(BASE/'QUESTION_FREEZE.json'):raise RuntimeError('Frozen input drift before initialization recovery')
  validate_initialization_recovery(load(EXEC/'RAW_RESPONSE_FREEZE.json'),len(list((EXEC/'batches').glob('*/SUBMITTED.json'))),len(list(rawdir.glob('*.json'))))
  if os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')!='0':raise RuntimeError('Initialization recovery must disable FlashInfer sampling')
  start_path=EXEC/'initialization_recovery_02'/'STARTED.json'
  atomic(BASE/'process_registry'/'solver_runtime_recovery.json',{'pid':os.getpid(),'start_token':start_token(os.getpid()),'kind':'solver_runtime_recovery','started_at_utc':now()})
 # No retries after generation start. Existing raw or submitted calls are preserved.
 if start_path.exists():
  atomic(EXEC/'INTERRUPTED.json',{'at':now(),'reason':'generation_already_started_no_automatic_retry'});return
 atomic(start_path,{'at':now(),'request_ids':[r['request_id'] for r in rows],'pid':os.getpid(),'start_token':start_token(os.getpid()),'public':bind(EXEC/'PUBLIC_REQUESTS.json'),'gold_read':False,'initialization_recovery':initialization_recovery,'VLLM_USE_FLASHINFER_SAMPLER':os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')})
 status('solver',status='RUNNING',solver_prepared=len(rows),solver_returned=0,solver_missing=len(rows))
 model=None;records=[];batchfail=[]
 if rows:
  try:
   import torch,vllm,transformers
   from vllm import LLM,SamplingParams
   from vllm.sampling_params import StructuredOutputsParams
   from vllm.config import StructuredOutputsConfig
   config=StructuredOutputsConfig(reasoning_parser='qwen3',enable_in_reasoning=False)
   model=LLM(model=str(MODEL),tokenizer=str(MODEL),dtype='bfloat16',max_model_len=16384,gpu_memory_utilization=0.85,max_num_seqs=8,seed=0,enable_prefix_caching=False,enforce_eager=True,structured_outputs_config=config,disable_log_stats=True)
   atomic(EXEC/'MODEL_LOAD_RECEIPT.json',{'at':now(),'model_directory':str(MODEL),'model_files':[bind(p) for p in sorted(MODEL.iterdir()) if p.is_file()],'versions':{'torch':torch.__version__,'vllm':vllm.__version__,'transformers':transformers.__version__},'thinking':False,'mode':'FULL_THINK_OFF','max_model_len':16384,'max_output_tokens':6144,'max_batch_requests':8,'prefix_cache':False,'engine_seed':0,'gold_read':False,'structured_outputs_config':{'reasoning_parser':'qwen3','enable_in_reasoning':False}})
   for start in range(0,len(rows),8):
    batch=rows[start:start+8];bd=EXEC/'batches'/f'{start//8:03d}';bd.mkdir(parents=True,exist_ok=True)
    cutoff=datetime.datetime.fromisoformat(load(BASE/'LAUNCH.json')['launched_at_utc'])+datetime.timedelta(hours=12,minutes=-20)
    if datetime.datetime.now(datetime.timezone.utc)>=cutoff:
     batchfail.append({'start':start,'reason':'solver_deadline_no_new_batches'});break
    params=[SamplingParams(temperature=0.8,top_p=0.9,top_k=-1,repetition_penalty=1.0,max_tokens=6144,seed=r['sampling_seed'],structured_outputs=StructuredOutputsParams(json=load(r['schema_path']))) for r in batch]
    prompts=[{'prompt_token_ids':token_ids[r['request_id']]} for r in batch]
    atomic(bd/'SUBMITTED.json',{'at':now(),'request_ids':[r['request_id'] for r in batch],'pid':os.getpid(),'start_token':start_token(os.getpid()),'gold_read':False,'params':{'temperature':0.8,'top_p':0.9,'top_k':-1,'repetition_penalty':1.0,'max_tokens':6144}})
    began=time.time()
    try:outputs=model.generate(prompts,params,use_tqdm=False)
    except Exception as e:
     failure={'at':now(),'start':start,'request_ids':[r['request_id'] for r in batch],'error':repr(e),'traceback':traceback.format_exc()};batchfail.append(failure);atomic(bd/'FAILED.json',failure);break
    for row,out in zip(batch,outputs):
     if not out.outputs:continue
     resp=out.outputs[0]
     record={'request_id':row['request_id'],'at':now(),'text':resp.text,'token_ids':list(resp.token_ids),'prompt_token_ids':list(out.prompt_token_ids),'finish_reason':resp.finish_reason,'stop_reason':str(resp.stop_reason),'output_request_id':out.request_id,'gold_read':False,'batch_wall_seconds':time.time()-began}
     path=rawdir/(row['request_id']+'.json');atomic(path,record);records.append({'request_id':row['request_id'],'binding':bind(path),'raw_text_sha256':digest(resp.text.encode())})
    atomic(bd/'COMPLETED.json',{'at':now(),'returned_request_ids':[x['request_id'] for x in records if x['request_id'] in {r['request_id'] for r in batch}],'wall_seconds':time.time()-began})
    status('solver',solver_prepared=len(rows),solver_returned=len(records),solver_missing=len(rows)-len(records),solver_batches_finished=start//8+1)
    print(json.dumps({'solver_batch':start//8,'returned':len(records),'planned':len(rows)}),flush=True)
  except Exception as e:
   batchfail.append({'error':repr(e),'traceback':traceback.format_exc()});atomic(EXEC/'BACKEND_FAILURE.json',batchfail[-1])
 returned={r['request_id'] for r in records}
 freeze={'at':now(),'prepared_count':len(rows),'returned_count':len(records),'missing_count':len(rows)-len(records),'records':records,'missing_request_ids':[r['request_id'] for r in rows if r['request_id'] not in returned],'backend_failures':batchfail,'gold_read':False,'no_automatic_semantic_retries':True}
 freeze['generation_start']=bind(start_path)
 if initialization_recovery:freeze['initialization_recovery_receipt']=bind(EXEC/'INITIALIZATION_RECOVERY.json')
 atomic(EXEC/'RAW_RESPONSE_FREEZE.json',freeze);status('solver_freeze',solver_returned=len(records),solver_missing=len(rows)-len(records),raw_freeze=bind(EXEC/'RAW_RESPONSE_FREEZE.json'))

def score():
 pilot=frozen_modules();freeze=load(EXEC/'RAW_RESPONSE_FREEZE.json');pub={r['request_id']:r for r in load(EXEC/'PUBLIC_REQUESTS.json')['requests']};priv={r['request_id']:r['presentation'] for r in load(EXEC/'PRIVATE_SCORING.json')['requests']};records=[]
 for x in freeze['records']:
  assert bind(x['binding']['path'])==x['binding'];raw=load(x['binding']['path']);row=pub[x['request_id']];presentation=priv[x['request_id']];valid=True;errors=[];parsed=None;answer=[]
  text=raw['text'].strip()
  if text.endswith('<|im_end|>'):text=text[:-len('<|im_end|>')].rstrip()
  if text.startswith('<think>') and text.count('<think>')==1 and text.count('</think>')==1:text=text.split('</think>',1)[1].strip()
  try:parsed,answer=pilot.validate_output(text,row,load(row['schema_path']))
  except Exception as e:valid=False;errors.append(type(e).__name__+':'+str(e))
  strict=pilot.score_record(row,presentation,answer,valid)
  # Answer-only diagnostic uses exactly the same legal-label contract, independently of evidence.
  answer_only=[];answer_only_valid=False
  try:
   import re
   o=json.loads(text);m=re.fullmatch(r'<box>([A-F](?: [A-F]){0,5})</box>',o['final_answer']);assert m
   answer_only=m.group(1).split();assert len(answer_only)==len(set(answer_only))
   if row['question_type']=='ordering':assert set(answer_only)==set('ABCDEF')
   answer_only_valid=True
  except Exception:pass
  ao=pilot.score_record(row,presentation,answer_only,answer_only_valid)
  record={'request_id':row['request_id'],'root_id':row['root_id'],'candidate_id':row['candidate_id'],'arm':row['arm'],'slot_id':row['slot_id'],'variant_id':row['variant_id'],'strict':strict,'answer_only':ao,'errors':errors,'parsed':parsed,'returned':True,'raw':x['binding']}
  records.append(record)
 atomic(EXEC/'SCORES.json',{'records':records,'raw_freeze':bind(EXEC/'RAW_RESPONSE_FREEZE.json'),'frozen_scorer':bind(WEEK/'postweek_question_quality_pilot_v1/run.py'),'mechanical_semantic_citation_entailment':'not_assessed_by_exact_metric'})
 status('solver_scored',solver_returned=len(records),solver_contract_valid=sum(r['strict']['valid'] for r in records),solver_exact=sum(r['strict']['exact'] for r in records))
 return records

def run():
 # Ensure the modern frozen runtime is used even when parent uses base conda.
 env=dict(os.environ,PYTHONPATH=str(RUNTIME),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn')
 for mode in ['prepare','generate','score']:
  log=EXEC/(mode+'.log');log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('a') as out:
   p=subprocess.Popen([sys.executable,str(BASE/'scripts/solver_resume.py'),'--'+mode],stdout=out,stderr=subprocess.STDOUT,env=env,cwd=BASE,start_new_session=True)
   child={'pid':p.pid,'start_token':start_token(p.pid),'started_at_utc':now(),'kind':'solver_'+mode}
   with LOCK:ACTIVE[p.pid]=child;status('solver_'+mode)
   rc=p.wait()
   with LOCK:ACTIVE.pop(p.pid,None);status('solver_'+mode,solver_last_exit_code=rc)
  if rc:raise RuntimeError(f'solver {mode} exit {rc}; see {log}')
if __name__=='__main__':
 a=argparse.ArgumentParser()
 for mode in ['prepare','generate','score','run','prepare-initialization-recovery','recover-initialization']:a.add_argument('--'+mode,action='store_true')
 args=a.parse_args()
 if args.prepare:prepare()
 elif args.generate:generate()
 elif args.score:score()
 elif args.run:run()
 elif args.prepare_initialization_recovery:prepare_initialization_recovery()
 elif args.recover_initialization:generate(initialization_recovery=True)
