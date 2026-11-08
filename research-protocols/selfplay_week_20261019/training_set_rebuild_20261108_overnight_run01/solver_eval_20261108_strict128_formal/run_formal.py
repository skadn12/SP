#!/usr/bin/env python3
"""One-shot formal 128-candidate Solver evaluation.

The request ledger is rebuilt from the already sealed strict128-v3 inputs, but
prompt construction, chat templating, JSON schema generation and scoring are
delegated to the frozen historical adapter.  This file never retries or
overwrites an output and keeps full/no-context calls paired by seed.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "solver_eval_20261108_strict128_v3"
BASE = Path("/root/autodl-tmp/Qwen3-4B")
MODEL = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model")
ADAPTER_FILE = Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_structure_pilot_plan_20261106_v1/execution_20261106_v1/solver/adapter.py")

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write_once(p, obj):
    p=HERE/p if not Path(p).is_absolute() else Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise RuntimeError('refuse overwrite '+str(p))
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def load_adapter():
    spec=importlib.util.spec_from_file_location('formal_frozen_adapter',ADAPTER_FILE)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    old=mod.load_history()
    # Keep the frozen adapter's historical module available through an explicit
    # handle.  It has no top-level ``h`` attribute; load_history() returns the
    # legacy module whose ``h`` namespace owns the prompt/scorer functions.
    mod._old = old
    return mod

def prepare():
    if (HERE/'REQUESTS.json').exists(): raise RuntimeError('already prepared')
    src=read(SOURCE/'REQUESTS.json'); assert len(src)==4608
    gate_rows={r['candidate_id']:r for r in (json.loads(x) for x in (SOURCE.parent/'solver_gate_20261108_schema_hold_v3'/'SOLVER_INPUT_FULL_CONTEXT.jsonl').read_text().splitlines() if x.strip())}
    mod=load_adapter()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(BASE),trust_remote_code=True,local_files_only=True)
    out=[]; stats=Counter()
    for old in src:
        source_path=Path(old['source_path']); assert source_path.is_file() and sha(source_path)==old['source_sha256']
        source=read(source_path); assert source['root_id']==old['root_id']
        assert len(source['turns']) > 0
        gate=gate_rows[old['candidate_id']]; ids=old['display_option_ids']; labels=old['display_labels']
        mapping=dict(zip(labels,ids)); rev={v:k for k,v in mapping.items()}
        options=[{'id':lab,'option_id':oid,'text':next(o['text'] for o in gate['options'] if (o.get('option_id') or o.get('id'))==oid)} for lab,oid in mapping.items()]
        gold_labels=[rev[x] for x in old['gold_option_ids']]
        item={'candidate_id':old['candidate_id'],'question_type':old['question_type'],'question':gate['question'],'options':options}
        request={**old,'question':gate['question'],'display_to_option_id':mapping,'gold_labels':gold_labels,'options':options,
                 'n':len(labels)*(len(labels)-1) if old['question_type']=='ordering' else len(labels),
                 'g':len(gold_labels) if old['question_type']=='multi_select' else len(labels)*(len(labels)-1)//2,
                 'variant_id':f"perm_{old['variant_index']:02d}",
                 'schema_path':f"schemas/{old['request_id']}.json",'prompt_path':f"prompts/{old['request_id']}.txt"}
        turns=source['turns'] if old['context_mode']=='full_context' else []
        prompt=mod.prompt_for(tok,item,request,turns)
        assert len(tok(prompt,add_special_tokens=False).input_ids)+6144<=16384, old['request_id']
        schema=mod.schema_for(labels,[t['turn_ref'] for t in turns],old['question_type'])
        (HERE/'prompts').mkdir(exist_ok=True); (HERE/'schemas').mkdir(exist_ok=True)
        (HERE/request['prompt_path']).write_text(prompt,encoding='utf-8')
        (HERE/request['schema_path']).write_text(json.dumps(schema,ensure_ascii=False)+'\n',encoding='utf-8')
        request.update(prompt_sha256=sha(HERE/request['prompt_path']),schema_sha256=sha(HERE/request['schema_path']),prompt_token_count=len(tok(prompt,add_special_tokens=False).input_ids),source_turn_count=len(source['turns']))
        out.append(request); stats[(old['question_type'],old['context_mode'])]+=1
    write_once('REQUESTS.json',out)
    write_once('PREFLIGHT.json',{'status':'PASS','candidate_count':128,'request_count':len(out),'counts':{str(k):v for k,v in stats.items()},'source_requests_sha256':sha(SOURCE/'REQUESTS.json'),'model':str(MODEL),'tokenizer':str(BASE),'adapter_path':str(ADAPTER_FILE),'adapter_model_sha256':sha(Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/clean_adapter/adapter_model.pt')),'enable_thinking':False,'automatic_retry':False,'max_prompt_plus_generation':max(r['prompt_token_count']+6144 for r in out)})
    print(json.dumps({'status':'PREPARED','count':len(out),'counts':{str(k): v for k, v in stats.items()}},ensure_ascii=False),flush=True)

def run():
    req=read(HERE/'REQUESTS.json'); assert len(req)==4608
    if (HERE/'STARTED.json').exists(): raise RuntimeError('already started')
    write_once('STARTED.json',{'status':'STARTED','planned_requests':len(req),'model':str(MODEL),'tokenizer':str(BASE),'enable_thinking':False,'automatic_retry':False})
    import torch
    from vllm import LLM, SamplingParams
    # vLLM 0.10.2 exposes JSON constrained decoding as GuidedDecodingParams;
    # keep the frozen schema constraint while supporting this installed API.
    from vllm.sampling_params import GuidedDecodingParams
    llm=LLM(model=str(MODEL),tokenizer=str(BASE),trust_remote_code=True,dtype='bfloat16',max_model_len=16384,gpu_memory_utilization=.8,seed=2026110303,enable_prefix_caching=False)
    raws=[]
    for start in range(0,len(req),12):
        batch=req[start:start+12]; prompts=[(HERE/r['prompt_path']).read_text(encoding='utf-8') for r in batch]
        params=[SamplingParams(n=1,temperature=.8,top_p=.9,top_k=-1,repetition_penalty=1.,max_tokens=6144,seed=int(r['seed']),stop=None,guided_decoding=GuidedDecodingParams(json=read(HERE/r['schema_path']))) for r in batch]
        outputs=llm.generate(prompts,params,use_tqdm=False); assert len(outputs)==len(batch)
        for r,o in zip(batch,outputs):
            assert len(o.outputs)==1; c=o.outputs[0]
            raw={'request_id':r['request_id'],'status':'RETURNED','raw_prediction':c.text,'finish_reason':c.finish_reason,'stop_reason':c.stop_reason,'completion_token_ids':list(c.token_ids),'prompt_sha256':r['prompt_sha256'],'automatic_retry':False}
            write_once('raw/'+r['request_id']+'.json',raw); raws.append(raw)
        print(json.dumps({'status':'INFERENCE_PROGRESS','completed':len(raws),'planned':len(req)},ensure_ascii=False),flush=True)
    write_once('RAW_ANSWERS.json',raws); write_once('COMPLETED.json',{'status':'COMPLETED','returned_requests':len(raws)})
    del llm

def score():
    req=read(HERE/'REQUESTS.json'); raws=read(HERE/'RAW_ANSWERS.json'); assert len(req)==len(raws)==4608
    mod=load_adapter(); rows=[]
    for r,raw in zip(req,raws):
        assert r['request_id']==raw['request_id']; checked=mod.validate(raw,r); ans=checked.get('answer_only_score') or mod._old.h.baseline.parse_and_score({'status':'INVALID'},r)
        rows.append({**{k:r[k] for k in ('request_id','candidate_id','root_id','arm','question_type','context_mode','variant_index','repeat_index','variant_id')},**ans,**checked,'raw_prediction':raw.get('raw_prediction'),'finish_reason':raw.get('finish_reason')})
    def sm(rs):
        if not rs:return {'n':0}
        return {'n':len(rs),'valid':sum(bool(x.get('valid')) for x in rs),'exact':sum(bool(x.get('exact_correct')) for x in rs)/len(rs),'mean_J':sum(float(x.get('J',0)) for x in rs)/len(rs),'protocol_valid':sum(bool(x.get('protocol_valid')) for x in rs)/len(rs)}
    summary={'by_context':{m:sm([x for x in rows if x['context_mode']==m]) for m in ('full_context','no_context')},'by_type':{q:{m:sm([x for x in rows if x['question_type']==q and x['context_mode']==m]) for m in ('full_context','no_context')} for q in ('multi_select','ordering')},'by_arm':{a:{m:sm([x for x in rows if x['arm']==a and x['context_mode']==m]) for m in ('full_context','no_context')} for a in ('F','G')}}
    write_once('SCORES.json',{'status':'SCORED','request_count':len(rows),'summary':summary,'requests':rows,'model':str(MODEL),'adapter_model_sha256':sha(Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/clean_adapter/adapter_model.pt')),'source_requests_sha256':sha(SOURCE/'REQUESTS.json')}); print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=('prepare','run','score')); a=ap.parse_args(); {'prepare':prepare,'run':run,'score':score}[a.mode]()
