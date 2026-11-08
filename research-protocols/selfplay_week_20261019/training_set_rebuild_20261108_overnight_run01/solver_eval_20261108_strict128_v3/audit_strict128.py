import json,hashlib,re
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
req=json.loads((HERE/'REQUESTS.json').read_text())
assert len(req)==4608
assert len({x['request_id'] for x in req})==4608
assert all(x['question_type'] in ('multi_select','ordering') for x in req)
tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen3-4B',local_files_only=True,trust_remote_code=True)
rows=[]; failures=[]
for x in req:
 p=HERE/x['prompt_path']; s=HERE/x['schema_path'];
 if not p.is_file() or not s.is_file(): failures.append((x['request_id'],'missing_prompt_or_schema')); continue
 if hashlib.sha256(p.read_bytes()).hexdigest()!=x['prompt_sha256']: failures.append((x['request_id'],'prompt_hash'))
 if hashlib.sha256(s.read_bytes()).hexdigest()!=x['schema_sha256']: failures.append((x['request_id'],'schema_hash'))
 n=len(tok(safe:=p.read_text(),add_special_tokens=False).input_ids)
 if n+6144>16384: failures.append((x['request_id'],f'token_budget:{n}'))
 sch=json.loads(s.read_text());
 for key in ('item_evidence','answer_reason','final_answer'):
  if key not in sch.get('required',[]): failures.append((x['request_id'],'schema_required_'+key))
 final=sch.get('properties',{}).get('final_answer',{}).get('pattern','')
 if not final or not re.search(r'box',final): failures.append((x['request_id'],'schema_final_pattern'))
 rows.append({'request_id':x['request_id'],'prompt_tokens':n})
summary={'schema':'strict128_request_token_schema_audit_v1','status':'PASS' if not failures else 'FAIL','request_count':len(req),'failure_count':len(failures),'failures':failures,'min_prompt_tokens':min(r['prompt_tokens'] for r in rows),'max_prompt_tokens':max(r['prompt_tokens'] for r in rows),'max_plus_generation':max(r['prompt_tokens'] for r in rows)+6144,'question_types':dict(Counter(x['question_type'] for x in req)),'request_sha256':hashlib.sha256((HERE/'REQUESTS.json').read_bytes()).hexdigest(),'gate_sha256':json.loads((HERE/'PREFLIGHT.json').read_text())['gate_sha256']}
(HERE/'STRICT128_AUDIT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2)); raise SystemExit(1 if failures else 0)
