#!/usr/bin/env python3
"""Isolated M-clean Solver evaluation for the 138-row provisional gate.

This runner has no dependency on a historical request/seal/result.  It loads
complete frozen SOURCE.json turns, creates Gold-blind display permutations,
and scores multi-select, complete ordering, and partial-order rows separately.
"""
from __future__ import annotations
import argparse, copy, hashlib, itertools, json, math, os, random, re, time
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE=Path(__file__).resolve().parent
GATE=HERE.parent/'solver_gate_20261108_v1'
BASE=Path('/root/autodl-tmp/Qwen3-4B')
ADAPTER=Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/clean_adapter')
MERGED=HERE/'mclean_merged_model'
FULL=GATE/'SOLVER_INPUT_FULL_CONTEXT.jsonl'
MODEL_LEN=16384; MAX_TOKENS=6144; SEED=2026110831; DISPLAY_SEED=2026110832

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha(p): return sha_bytes(Path(p).read_bytes())
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write_new(p,obj):
    p=HERE/p if not Path(p).is_absolute() else Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise RuntimeError('refusing overwrite '+str(p))
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(): return [json.loads(x) for x in FULL.read_text(encoding='utf-8').splitlines() if x.strip()]
def labels_for(r): return [o.get('option_id') or o.get('id') for o in r['options']]
def display_variants(r):
    # Stable content IDs are option IDs. Gold is not used to construct order.
    ids=labels_for(r); n=6
    rng=random.Random(DISPLAY_SEED+int(hashlib.sha256(r['candidate_id'].encode()).hexdigest()[:12],16))
    out=[]; used=set()
    while len(out)<n:
        x=ids[:]; rng.shuffle(x)
        if tuple(x) in used: continue
        used.add(tuple(x)); out.append(x)
    return out

def perms(r):
    return min(6, math.factorial(len(r['options']))), 3

def schema(labels,qtype,refs, pairs=None):
    ev={l:{'type':'object','additionalProperties':False,'required':['judgment','turn_refs','reason'],'properties':{
        'judgment':{'type':'string','enum':['supported','contradicted','not_supported','ambiguous']},
        'turn_refs':{'type':'array','maxItems':3,'items':{'type':'string','enum':refs}},
        'reason':{'type':'string','minLength':1,'maxLength':320}}} for l in labels}
    if qtype=='partial_order':
        vals=['<box></box>']+[f'<box>{a}>{b}</box>' for a,b in pairs]
        # Permit a whitespace-separated pair list; exact semantic scoring is local.
        final={'type':'string','pattern':r'^<box>(?:[A-Z0-9_.-]+>[A-Z0-9_.-]+(?: [A-Z0-9_.-]+>[A-Z0-9_.-]+)*)?</box>$'}
    elif qtype=='multi_select':
        final={'type':'string','pattern':r'^<box>'+r'(?:'+ '|'.join(map(re.escape,labels))+r')(?: (?:'+ '|'.join(map(re.escape,labels))+r'))*</box>$'}
    else:
        final={'type':'string','pattern':r'^<box>'+r'(?:'+ '|'.join(map(re.escape,labels))+r')(?: (?:'+ '|'.join(map(re.escape,labels))+r'))*</box>$'}
    return {'type':'object','additionalProperties':False,'required':['item_evidence','answer_reason','final_answer'],'properties':{
        'item_evidence':{'type':'object','additionalProperties':False,'required':labels,'properties':ev},
        'answer_reason':{'type':'string','minLength':1,'maxLength':800},'final_answer':final}}

def prompt(r,variant,turns,mode):
    typ=r['question_type']; labels=[chr(65+i) for i in range(len(variant))] if typ!='ordering' else [f'I{i+1}' for i in range(len(variant))]
    # use letter labels for multi/partial and I labels for complete ordering, matching protocol
    items='\n'.join(f'{lab}. {txt}' for lab,txt in zip(labels,[next(o['text'] for o in r['options'] if (o.get('option_id') or o.get('id'))==oid) for oid in variant]))
    if mode=='full_context':
        dialogue='\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in turns)
        pre='Use only the supplied dialogue. Cite concise turn references in item_evidence.'
    else:
        dialogue='No source dialogue is supplied in this diagnostic condition.'
        pre='No dialogue is supplied. Do not invent turn references; every turn_refs must be [].'
    if typ=='multi_select': task='Select all and only the displayed options supported by the dialogue.'
    elif typ=='ordering': task='Put every displayed item in chronological order from earliest to latest.'
    else: task='Return all and only entailed precedence pairs in query scope. Encode each pair as LABEL>LABEL, separated by spaces; leave unrelated pairs out.'
    return f'{pre}\n\n{dialogue}\n\nQuestion ({typ}):\n{r["question"]}\n\nDisplayed items:\n{items}\n\n{task}\nReturn JSON with item_evidence first, answer_reason second, final_answer last. The final_answer must be boxed.' , labels

def prepare():
    rs=rows(); assert len(rs)==138
    req=[]; stats=Counter()
    for ri,r in enumerate(rs):
        src=Path(r['source_path']); assert src.is_file() and sha(src)==r['source_sha256']
        source=read(src); turns=source['turns']; assert source['root_id']==r['root_id']
        assert len(turns)==r['source_turn_count']
        pcount, reps=perms(r); variants=display_variants(r)
        for vi,order in enumerate(variants[:pcount]):
            labels=['I'+str(i+1) for i in range(len(order))] if r['question_type']=='ordering' else [chr(65+i) for i in range(len(order))]
            for mode in ('full_context','no_context'):
                text,labs=prompt(r,order,turns if mode=='full_context' else [],mode)
                # source labels are represented only by display labels; stable Gold mapping is private.
                for rep in range(reps):
                    rid=f"{r['candidate_id']}__{mode}__perm_{vi:02d}__s{rep:02d}"
                    rowsrc=turns if mode=='full_context' else []
                    pairs=[(labels[i],labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i!=j]
                    sc=schema(labels,r['question_type'],[t['turn_ref'] for t in rowsrc],pairs)
                    pp=HERE/'prompts'/(rid+'.txt'); sp=HERE/'schemas'/(rid+'.json'); pp.parent.mkdir(parents=True,exist_ok=True); sp.parent.mkdir(parents=True,exist_ok=True)
                    if pp.exists() or sp.exists(): raise RuntimeError('overwrite '+rid)
                    pp.write_text(text,encoding='utf-8'); sp.write_text(json.dumps(sc,ensure_ascii=False)+'\n',encoding='utf-8')
                    req.append({'request_id':rid,'candidate_id':r['candidate_id'],'root_id':r['root_id'],'arm':r['arm'],'question_type':r['question_type'],'context_mode':mode,'variant_index':vi,'repeat_index':rep,'display_labels':labels,'display_option_ids':order,'gold_option_ids':r.get('gold_option_ids',[]),'gold_pairs':r.get('gold_pairs',[]),'query_pairs':r.get('query_pairs',[]),'source_path':str(src),'source_sha256':r['source_sha256'],'prompt_path':str(pp.relative_to(HERE)),'schema_path':str(sp.relative_to(HERE)),'prompt_sha256':sha(pp),'schema_sha256':sha(sp),'seed':SEED+ri*10000+vi*100+rep})
                    stats[(r['question_type'],mode)]+=1
    write_new('REQUESTS.json',req)
    manifest={'status':'PREPARED','candidate_count':len(rs),'request_count':len(req),'counts':{str(k):v for k,v in stats.items()},'gate_sha256':sha(FULL),'adapter_config_sha256':sha(ADAPTER/'adapter_config.json'),'adapter_model_sha256':sha(ADAPTER/'adapter_model.pt'),'base_config_sha256':sha(BASE/'config.json'),'merged_model_expected':str(MERGED),'truncation_check':'pending tokenizer check'}
    write_new('PREFLIGHT.json',manifest); print(json.dumps(manifest,ensure_ascii=False),flush=True)

def load_model():
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for Solver inference')
    manifest=MERGED/'MERGE_MANIFEST.json'
    if not manifest.is_file(): raise RuntimeError('merged model manifest missing: '+str(manifest))
    model=AutoModelForCausalLM.from_pretrained(str(MERGED),dtype=torch.bfloat16,local_files_only=True,trust_remote_code=True)
    model=model.to('cuda').eval()
    tok=AutoTokenizer.from_pretrained(str(MERGED),local_files_only=True,trust_remote_code=True)
    return model,tok

def run(limit=None):
    req=read(HERE/'REQUESTS.json'); model,tok=load_model(); import torch
    todo=req if limit is None else req[:limit]
    raws=[]
    for i,row in enumerate(todo):
        text=(HERE/row['prompt_path']).read_text(); enc=tok(text,return_tensors='pt'); ids=enc.input_ids; attention_mask=enc.attention_mask
        assert ids.shape[1]+MAX_TOKENS<=MODEL_LEN,(row['request_id'],ids.shape[1])
        gen_device=torch.device('cuda'); ids=ids.to(gen_device); attention_mask=attention_mask.to(gen_device); generator=torch.Generator(device=gen_device).manual_seed(int(row['seed']))
        with torch.no_grad(): out=model.generate(ids,attention_mask=attention_mask,max_new_tokens=MAX_TOKENS,do_sample=True,temperature=.8,top_p=.9,top_k=0,repetition_penalty=1.0,generator=generator)
        gen=out[0,ids.shape[1]:]; decoded=tok.decode(gen,skip_special_tokens=True)
        raw={'request_id':row['request_id'],'status':'RETURNED','raw_prediction':decoded,'finish_reason':'stop','prompt_token_count':int(ids.shape[1]),'prompt_sha256':row['prompt_sha256'],'model':'M-clean merged in memory','adapter_sha256':sha(ADAPTER/'adapter_model.pt')}
        raws.append(raw); (HERE/'raw').mkdir(exist_ok=True); (HERE/'raw'/(row['request_id']+'.json')).write_text(json.dumps(raw,ensure_ascii=False)+'\n')
        if (i+1)%10==0: print(json.dumps({'status':'INFERENCE_PROGRESS','completed':i+1,'planned':len(todo)}),flush=True)
    write_new('RAW_ANSWERS.json',raws); print(json.dumps({'status':'INFERENCE_DONE','count':len(raws)}),flush=True)

def parse_json(s):
    m=re.search(r'\{.*\}',s,re.S)
    if not m: return None
    try:return json.loads(m.group(0))
    except:return None

def score():
    req=read(HERE/'REQUESTS.json'); raws={x['request_id']:x for x in read(HERE/'RAW_ANSWERS.json')}; out=[]
    for r in req:
        if r['request_id'] not in raws: continue
        raw=raws[r['request_id']]; obj=parse_json(raw.get('raw_prediction','')); labels=r['display_labels']; valid=False; pred=None; reason='invalid_json'
        if isinstance(obj,dict) and isinstance(obj.get('final_answer'),str):
            final=obj['final_answer']; mm=re.fullmatch(r'<box>(.*?)</box>',final.strip())
            if mm:
                body=mm.group(1).strip()
                if r['question_type']=='partial_order': pred=[] if not body else body.split(); valid=True
                else:
                    pred=body.split(); valid=len(pred)==len(set(pred)) and set(pred)<=set(labels) and (r['question_type']!='ordering' or set(pred)==set(labels))
                reason='valid' if valid else 'invalid_labels'
        exact=False; j=0.0; tp=fp=fn=None
        if valid and r['question_type']=='multi_select':
            gold=set(labels[i] for i,x in enumerate(r['display_option_ids']) if x in r['gold_option_ids']); got=set(pred); tp=len(gold&got); fp=len(got-gold); fn=len(gold-got); exact=got==gold; j=(tp/(len(gold)) if False else (tp-fp)/(len(labels)-len(gold)))
        elif valid and r['question_type']=='ordering':
            gold=[labels[r['display_option_ids'].index(x)] for x in r['gold_option_ids']]; exact=pred==gold; pairs=len(labels)*(len(labels)-1)//2; good=sum(pred.index(a)<pred.index(b) for i,a in enumerate(gold) for b in gold[i+1:]); wrong=pairs-good; tp=good; fp=wrong; j=(good-wrong)/pairs
        elif valid and r['question_type']=='partial_order':
            gold={f'{labels[r["display_option_ids"].index(a)]}>{labels[r["display_option_ids"].index(b)]}' for a,b in r['gold_pairs']}; got=set(pred); tp=len(gold&got); fp=len(got-gold); fn=len(gold-got); exact=got==gold; denom=max(1,len(r['query_pairs'])); j=(tp-fp)/denom
        out.append({**{k:r[k] for k in ('request_id','candidate_id','root_id','arm','question_type','context_mode','variant_index','repeat_index')},'valid':valid,'parse_reason':reason,'prediction':pred,'exact_correct':exact,'J':j,'tp':tp,'fp':fp,'fn':fn})
    by=Counter((x['arm'],x['context_mode'],x['question_type']) for x in out); summary={str(k):{'n':v,'exact':sum(x['exact_correct'] for x in out if (x['arm'],x['context_mode'],x['question_type'])==k)/v,'mean_J':sum(x['J'] for x in out if (x['arm'],x['context_mode'],x['question_type'])==k)/v} for k,v in by.items()}
    write_new('SCORES.json',{'requests':out,'summary':summary,'adapter':str(ADAPTER),'gate_sha256':sha(FULL)}); print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['prepare','run','score']); ap.add_argument('--limit',type=int); a=ap.parse_args(); {'prepare':prepare,'run':lambda:run(a.limit),'score':score}[a.mode]()
