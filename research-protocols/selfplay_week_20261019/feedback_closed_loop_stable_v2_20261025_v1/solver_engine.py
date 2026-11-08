"""Fixed B1, FULL THINK_OFF, two frozen displays, no submitted-request retry.

Generation subprocess reads only PUBLIC_REQUESTS and PROMPT_TOKEN_IDS. Private
Gold and permutation maps are read only by prepare and score, after raw freeze.
"""
from __future__ import annotations
import argparse, importlib.util, os, pathlib, shutil, subprocess, sys, time, traceback
sys.dont_write_bytecode=True
from common import *
MODEL=pathlib.Path(PROTOCOL['solver_policy']['checkpoint'])
RUNTIME='/root/autodl-tmp/gxy-selfplay/.venv-local4b/lib/python3.10/site-packages'

def runtime_env():
    env=dict(os.environ,PYTHONPATH=RUNTIME,PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
      VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_USE_FLASHINFER_SAMPLER='0')
    (BASE/'.ipc').mkdir(exist_ok=True)
    # vLLM passes this string directly to ZeroMQ. A short relative path remains
    # inside cwd=BASE while respecting sockaddr_un's 107-byte path bound.
    env['VLLM_RPC_BASE_PATH']='.ipc'
    for k,suffix in [('TMPDIR','tmp'),('SQLITE_TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('HF_HOME','huggingface'),
      ('VLLM_CACHE_ROOT','vllm'),('TORCHINDUCTOR_CACHE_DIR','torchinductor'),('TRITON_CACHE_DIR','triton'),
      ('TORCH_HOME','torch'),('CUDA_CACHE_PATH','cuda'),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba')]:
        p=BASE/'runtime/solver_cache'/suffix;p.mkdir(parents=True,exist_ok=True);env[k]=str(p)
    return env

def frozen():
    p=BASE.parent/'postweek_question_quality_pilot_v1/run.py'
    spec=importlib.util.spec_from_file_location('stable2_readonly_frozen_pilot',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def prepare(label):
    d=BASE/'execution'/label
    if (d/'PREPARATION.json').exists():return load(d/'PREPARATION.json')
    cs=load(BASE/f'{label}_CANDIDATES_PRIVATE.json')['candidates'];rows=[];private=[];tokens={};fail=[]
    if cs:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True);m=frozen()
        for c in cs:
            verify(c['source_binding']);source=load(c['source_binding']['path']);last=None;pending=[]
            for variant in (0,1):
                seed=2026102427+variant;rejected=[]
                while True:
                    pub,priv=m.gate.prepare_display(c,seed);order=list(priv['display_to_canonical'].values())
                    if order==last or (c['question_type']=='ordering' and order==c['gold']):rejected.append(seed);seed+=2;continue
                    break
                last=order;priv['rejected_display_seeds']=rejected;display=pub.get('options',pub.get('items'))
                rid=c['candidate_id']+f'__perm_{variant:02d}__FULL__THINK_OFF__STABLE2'
                prompt=m.diagnostic.make_prompt(tok,c,display,source)
                prompt=prompt.replace("Follow the question's requested operation. Give concise, verifiable explanations before the final boxed answer.",
                 "Follow the question's requested operation. In the final response, give concise, verifiable explanations before the final boxed answer.",1)
                ids=tok.encode(prompt,add_special_tokens=False)
                assert prompt.endswith('<think>\n\n</think>\n\n') or prompt.endswith('<think>\n\n</think>\n\n\n')
                schema=m.diagnostic.repair.schema(list('ABCDEF'),[t['turn_ref'] for t in source['turns']],c['question_type'])
                pp=d/'prompts'/f'{rid}.txt';sp=d/'schemas'/f'{rid}.json';write(pp,prompt);atomic(sp,schema)
                row={k:c[k] for k in ['candidate_id','root_id','arm','slot_id','question_type','task_operation','source_binding','question']}
                matched=c['root_id']+':'+c['slot_id']
                row.update(request_id=rid,cohort='stable_v2_'+label.lower(),condition='FULL',mode='THINK_OFF',enable_thinking=False,
                  variant_id=f'perm_{variant:02d}',display=display,labels=list('ABCDEF'),
                  sampling_seed=int.from_bytes(hashlib.sha256(matched.encode()).digest()[:4],'big')%2147483647,
                  prompt_path=str(pp),schema_path=str(sp),prompt_tokens=len(ids),prompt_binding=bind(pp),schema_binding=bind(sp))
                pending.append((row,priv,ids))
            if any(row['prompt_tokens']>10240 for row,_,_ in pending):
                fail.append({'candidate_id':c['candidate_id'],'reason':'prompt_above_10240_tokens','lengths':[r['prompt_tokens'] for r,_,_ in pending]});continue
            for row,priv,ids in pending:rows.append(row);private.append({'request_id':row['request_id'],'presentation':priv});tokens[row['request_id']]=ids
    atomic(d/'PUBLIC_REQUESTS.json',{'requests':rows});atomic(d/'PRIVATE_SCORING.json',{'requests':private});atomic(d/'PROMPT_TOKEN_IDS.json',tokens)
    atomic(d/'LENGTH_FAILURES.json',fail)
    out={'at':now(),'prepared_presentations':len(rows),'content_valid_slots':len(cs),'length_failures':fail,
       'public_binding':bind(d/'PUBLIC_REQUESTS.json'),'private_binding':bind(d/'PRIVATE_SCORING.json'),
       'checkpoint':str(MODEL),'training':False,'sealed_old_reruns':0}
    atomic(d/'PREPARATION.json',out);return out

def generate(label):
    d=BASE/'execution'/label
    if (d/'RAW_RESPONSE_FREEZE.json').exists():return load(d/'RAW_RESPONSE_FREEZE.json')
    rows=load(d/'PUBLIC_REQUESTS.json')['requests'];tokens=load(d/'PROMPT_TOKEN_IDS.json');records=[];fail=[]
    if (d/'STARTED.json').exists():
        # No automatic retry after any generation launch. Preserve all raw.
        for p in (d/'raw').glob('*.json'):
            r=load(p);records.append({'request_id':r['request_id'],'binding':bind(p)})
        fail.append({'reason':'interrupted_generation_no_retry'})
    else:
        atomic(d/'STARTED.json',{'at':now(),'request_ids':[r['request_id'] for r in rows],'gold_read':False,'pid':os.getpid()})
        if label=='FULL' and (BASE/'execution/PILOT/RAW_RESPONSE_FREEZE.json').exists():
            old=BASE/'execution/PILOT';prior=load(old/'RAW_RESPONSE_FREEZE.json')
            prior_rows={r['request_id']:r for r in load(old/'PUBLIC_REQUESTS.json')['requests']}
            prior_tokens=load(old/'PROMPT_TOKEN_IDS.json');byid={r['request_id']:r for r in rows}
            reused=[]
            for r in prior['records']:
                rid=r['request_id'];row=byid.get(rid);previous=prior_rows[rid]
                if row is None:continue
                assert row['root_id'] in PILOT and row['sampling_seed']==previous['sampling_seed']
                assert row['prompt_binding']['sha256']==previous['prompt_binding']['sha256']
                assert row['schema_binding']['sha256']==previous['schema_binding']['sha256']
                assert tokens[rid]==prior_tokens[rid]
                verify(r['binding']);p=d/'raw'/(rid+'.json');local(p).parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(r['binding']['path'],p)
                records.append({'request_id':rid,'binding':bind(p),'carried_from_pilot':r['binding']});reused.append(rid)
            atomic(d/'PILOT_CARRY_FORWARD.json',{'at':now(),'request_ids':reused,'public_prompt_and_schema_identity_verified':True,'semantic_reruns':0,'gold_read':False})
        done={r['request_id'] for r in records};pending_rows=[r for r in rows if r['request_id'] not in done]
        if pending_rows:
            try:
                import torch,transformers,vllm
                from vllm import LLM,SamplingParams
                from vllm.sampling_params import StructuredOutputsParams
                from vllm.config import StructuredOutputsConfig
                config=StructuredOutputsConfig(reasoning_parser='qwen3',enable_in_reasoning=False)
                model=LLM(model=str(MODEL),tokenizer=str(MODEL),dtype='bfloat16',max_model_len=16384,
                  gpu_memory_utilization=.85,max_num_seqs=8,seed=0,enable_prefix_caching=False,enforce_eager=True,
                  structured_outputs_config=config,disable_log_stats=True)
                atomic(d/'MODEL_LOAD_RECEIPT.json',{'at':now(),'checkpoint':str(MODEL),'model_files':[bind(p) for p in sorted(MODEL.iterdir()) if p.is_file()],
                  'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'vllm':vllm.__version__},
                  'decode':{'temperature':.8,'top_p':.9,'top_k':-1,'repetition_penalty':1.0,'max_tokens':6144,'model_len':16384,'batch':8},
                  'thinking':False,'gold_read':False,'VLLM_USE_FLASHINFER_SAMPLER':os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')})
                for start in range(0,len(pending_rows),8):
                    batch=pending_rows[start:start+8];bd=d/'batches'/f'{start//8:03d}'
                    params=[SamplingParams(temperature=.8,top_p=.9,top_k=-1,repetition_penalty=1.,max_tokens=6144,
                       seed=r['sampling_seed'],structured_outputs=StructuredOutputsParams(json=load(r['schema_path']))) for r in batch]
                    atomic(bd/'SUBMITTED.json',{'at':now(),'request_ids':[r['request_id'] for r in batch],'gold_read':False})
                    began=time.monotonic()
                    try:outputs=model.generate([{'prompt_token_ids':tokens[r['request_id']]} for r in batch],params,use_tqdm=False)
                    except Exception:
                        error={'reason':'submitted_batch_failed_no_retry','traceback':traceback.format_exc(),'request_ids':[r['request_id'] for r in batch]}
                        atomic(bd/'FAILURE.json',error);fail.append(error);break
                    for r,o in zip(batch,outputs):
                        if not o.outputs:continue
                        x=o.outputs[0];p=d/'raw'/(r['request_id']+'.json')
                        atomic(p,{'request_id':r['request_id'],'text':x.text,'token_ids':list(x.token_ids),
                          'prompt_token_ids':list(o.prompt_token_ids),'finish_reason':x.finish_reason,'stop_reason':str(x.stop_reason),
                          'at':now(),'gold_read':False,'batch_wall_seconds':time.monotonic()-began})
                        records.append({'request_id':r['request_id'],'binding':bind(p),'raw_text_sha256':sha(x.text.encode())})
                    atomic(bd/'COMPLETED.json',{'at':now(),'wall_seconds':time.monotonic()-began})
                    status('solver',solver_prepared=len(rows),solver_returned=len(records))
            except Exception:
                error={'reason':'solver_backend_failure','traceback':traceback.format_exc()};fail.append(error);atomic(d/'BACKEND_FAILURE.json',error)
    returned={r['request_id'] for r in records}
    out={'at':now(),'prepared_count':len(rows),'returned_count':len(records),'missing_count':len(rows)-len(records),
      'records':records,'missing_request_ids':[r['request_id'] for r in rows if r['request_id'] not in returned],
      'backend_failures':fail,'gold_read':False,'semantic_retries':0,'training':False,'sealed_old_reruns':0}
    atomic(d/'RAW_RESPONSE_FREEZE.json',out);return out

def score(label):
    d=BASE/'execution'/label;freeze=load(d/'RAW_RESPONSE_FREEZE.json');records=[]
    if freeze['records']:
        m=frozen();public={r['request_id']:r for r in load(d/'PUBLIC_REQUESTS.json')['requests']}
        private={r['request_id']:r['presentation'] for r in load(d/'PRIVATE_SCORING.json')['requests']}
        for r in freeze['records']:
            verify(r['binding']);raw=load(r['binding']['path']);row=public[r['request_id']];presentation=private[r['request_id']]
            text=raw['text'].strip();errors=[];parsed=None;answer=[]
            if text.endswith('<|im_end|>'):text=text[:-len('<|im_end|>')].rstrip()
            if text.startswith('<think>') and text.count('<think>')==text.count('</think>')==1:text=text.split('</think>',1)[1].strip()
            valid=True
            try:parsed,answer=m.validate_output(text,row,load(row['schema_path']))
            except Exception as e:valid=False;errors.append(str(e))
            s=m.score_record(row,presentation,answer,valid)
            records.append({'request_id':r['request_id'],'root_id':row['root_id'],'arm':row['arm'],'slot_id':row['slot_id'],
              'variant_id':row['variant_id'],'returned':True,'contract_valid':valid,'score':s,'errors':errors,'parsed':parsed,'raw':r['binding']})
    out={'records':records,'returned':len(records),'contract_valid':sum(r['contract_valid'] for r in records),
      'exact':sum(r['score']['exact'] for r in records),'raw_freeze':bind(d/'RAW_RESPONSE_FREEZE.json'),
      'missingness_scored_as_incorrect':False,'mechanical_citation_validation_proves_entailment':False}
    atomic(d/'SCORES.json',out);return out

def run(label):
    d=BASE/'execution'/label;d.mkdir(parents=True,exist_ok=True);results=[]
    for mode in ('prepare','generate','score'):
        log=d/(mode+'.log')
        with log.open('a') as f:
            p=subprocess.Popen([sys.executable,'-B',str(BASE/'solver_engine.py'),mode,label],cwd=BASE,env=runtime_env(),stdout=f,stderr=subprocess.STDOUT)
            atomic(d/(mode.upper()+'_PROCESS.json'),{'pid':p.pid,'at':now()});rc=p.wait()
        results.append({'mode':mode,'exit_code':rc,'log':bind(log)})
        if rc:break
    atomic(d/'DRIVER_RECEIPT.json',{'at':now(),'stages':results});return results

def validate_initialization_recovery(freeze,submitted_count,raw_count):
    if freeze['returned_count'] or submitted_count or raw_count:
        raise RuntimeError('initialization_recovery_requires_zero_submissions_and_outputs')

def recover_initialization(label):
    """Only zero-submission initialization failure, never an output retry."""
    d=BASE/'execution'/label;freeze=load(d/'RAW_RESPONSE_FREEZE.json')
    validate_initialization_recovery(freeze,len(list((d/'batches').glob('*/SUBMITTED.json'))),len(list((d/'raw').glob('*.json'))))
    if not (d/'BACKEND_FAILURE.json').exists():raise RuntimeError('no_initialization_failure')
    archive=d/'initialization_failure_01';archive.mkdir(exist_ok=False);saved={}
    for name in ['STARTED.json','RAW_RESPONSE_FREEZE.json','BACKEND_FAILURE.json','generate.log','SCORES.json','score.log','DRIVER_RECEIPT.json']:
        p=d/name
        if p.exists():
            target=archive/name;shutil.copyfile(p,target);saved[name]={'original':bind(p),'archive':bind(target)}
    for name in ['PILOT_REPORT.json','STABILITY_BLOCKER.md','RUN_STATUS.json']:
        p=BASE/name
        if p.exists():
            target=archive/name;shutil.copyfile(p,target);saved[name]={'original':bind(p),'archive':bind(target)}
    receipt={'at':now(),'scope':'Initialization only; zero submitted requests and zero returned outputs',
      'submitted_requests':0,'returned_outputs':0,'semantic_retries':0,'archived':saved,
      'smallest_correction':{'VLLM_RPC_BASE_PATH':'.ipc'},'working_directory':str(BASE),
      'public_binding':bind(d/'PUBLIC_REQUESTS.json'),'tokens_binding':bind(d/'PROMPT_TOKEN_IDS.json'),
      'checkpoint_and_decode_unchanged':True,'training':False}
    atomic(d/'INITIALIZATION_RECOVERY.json',receipt)
    for name in ['STARTED.json','RAW_RESPONSE_FREEZE.json','BACKEND_FAILURE.json','SCORES.json']:
        p=d/name
        if p.exists():local(p).unlink()
    return run(label)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','generate','score','run','recover_initialization']);a.add_argument('label');args=a.parse_args()
    globals()[args.mode](args.label)
