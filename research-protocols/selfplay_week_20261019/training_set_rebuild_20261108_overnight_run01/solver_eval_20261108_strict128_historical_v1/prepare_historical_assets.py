#!/usr/bin/env python3
"""Materialize historical chat-template prompts/schemas for frozen strict-128 displays."""
import copy, hashlib, importlib.util, json, sys
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'solver_eval_20261108_strict128_v3'
GATE=HERE.parent/'solver_gate_20261108_schema_hold_v3'
MODEL=Path('/root/autodl-tmp/Qwen3-4B')
BASEQA=HERE.parents[1]/'base_qa_free_generation_reason_restore_eval_20261103_run02/run_eval.py'
spec=importlib.util.spec_from_file_location('baseqa',BASEQA); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
tok=AutoTokenizer.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True)
req0=json.loads((SRC/'REQUESTS.json').read_text())
full={json.loads(x)['candidate_id']:json.loads(x) for x in (GATE/'SOLVER_INPUT_FULL_CONTEXT.jsonl').read_text().splitlines() if x.strip()}
source_cache={}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readsrc(path):
 if path not in source_cache: source_cache[path]=json.loads(Path(path).read_text())
 return source_cache[path]
def item_for(r):
 labels=r['display_labels']; opts=[]
 for lab,oid in zip(labels,r['display_option_ids']):
  source=full[r['candidate_id']]; o=next(o for o in source['options'] if (o.get('option_id') or o.get('id'))==oid)
  opts.append({'id':lab,'option_id':oid,'text':o['text']})
 return {'candidate_id':r['candidate_id'],'question_type':r['question_type'],'question':full[r['candidate_id']]['question'],'options':opts,
         'gold_option_ids':full[r['candidate_id']].get('gold_option_ids',[])}
req=[]; stats=Counter(); max_tok=0; min_tok=10**9; fail=[]
for r in req0:
 item=item_for(r); src=readsrc(r['source_path']); turns=src['turns'] if r['context_mode']=='full_context' else []
 display={o['id']:o['option_id'] for o in item['options']}
 request={**r,'display_to_option_id':display,'options':item['options'],'gold_labels':[display_inv for display_inv in []], 'gold_option_ids':item['gold_option_ids'],'n':len(item['options']) if r['question_type']=='multi_select' else len(item['options'])*(len(item['options'])-1),'g':len(item['gold_option_ids']) if r['question_type']=='multi_select' else len(item['options'])*(len(item['options'])-1)//2}
 inv={v:k for k,v in display.items()}; request['gold_labels']=sorted(inv[x] for x in item['gold_option_ids']) if r['question_type']=='multi_select' else [inv[x] for x in item['gold_option_ids']]
 prompt=b.prompt_for(tok,item,request,turns)
 refs=[t['turn_ref'] for t in turns]; schema=b.schema_for(list(display),refs)
 pp=HERE/'prompts'/(r['request_id']+'.txt'); sp=HERE/'schemas'/(r['request_id']+'.json'); pp.parent.mkdir(exist_ok=True);sp.parent.mkdir(exist_ok=True)
 pp.write_text(prompt); sp.write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
 n=len(tok(prompt,add_special_tokens=False).input_ids); max_tok=max(max_tok,n);min_tok=min(min_tok,n)
 if n+6144>16384: fail.append((r['request_id'],n))
 request.update(prompt_path=str(pp.relative_to(HERE)),schema_path=str(sp.relative_to(HERE)),prompt_sha256=sha(pp),schema_sha256=sha(sp),prompt_token_count=n)
 req.append(request);stats[(r['question_type'],r['context_mode'])]+=1
assert not fail,fail[:5]
(HERE/'REQUESTS.json').write_text(json.dumps(req,ensure_ascii=False,indent=2)+'\n')
pre={'status':'PASS','candidate_count':128,'request_count':len(req),'counts':{str(k):v for k,v in stats.items()},'min_prompt_tokens':min_tok,'max_prompt_tokens':max_tok,'max_new_tokens':6144,'model_len':16384,'truncation':False,'gate_sha256':sha(GATE/'SOLVER_INPUT_FULL_CONTEXT.jsonl'),'source_request_sha256':sha(SRC/'REQUESTS.json'),'request_sha256':sha(HERE/'REQUESTS.json'),'prompt_builder':str(BASEQA),'enable_thinking':False,'automatic_retry':False,'structured_outputs':True,'interpreter':'/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python'}
(HERE/'PREFLIGHT.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(pre,ensure_ascii=False,indent=2))
