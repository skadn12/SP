import json,hashlib,sys
from pathlib import Path
from collections import Counter,defaultdict
HERE=Path(__file__).resolve().parent
req=json.loads((HERE/'REQUESTS.json').read_text())
assert len(req)==4968
assert len({r['request_id'] for r in req})==4968
c=Counter((r['question_type'],r['context_mode']) for r in req)
assert c[('multi_select','full_context')]==1800 and c[('multi_select','no_context')]==1800
assert c[('ordering','full_context')]==630 and c[('ordering','no_context')]==630
assert c[('partial_order','full_context')]==54 and c[('partial_order','no_context')]==54
by=defaultdict(list)
for r in req:
 by[r['candidate_id']].append(r)
 assert r['repeat_index'] in (0,1,2)
 assert r['context_mode'] in ('full_context','no_context')
 assert Path(r['prompt_path']).is_relative_to(Path('.'))
 assert hashlib.sha256((HERE/r['prompt_path']).read_bytes()).hexdigest()==r['prompt_sha256']
 assert hashlib.sha256((HERE/r['schema_path']).read_bytes()).hexdigest()==r['schema_sha256']
for cid,rs in by.items():
 assert len(rs)==36, (cid,len(rs))
 assert len({(r['context_mode'],r['variant_index'],r['repeat_index']) for r in rs})==36
 assert len({r['seed'] for r in rs})==18
 # full/no-context requests for each display/repeat are paired to one seed
 for vi in range(6):
  for rep in range(3):
   a=[x for x in rs if x['variant_index']==vi and x['repeat_index']==rep and x['context_mode']=='full_context'][0]
   b=[x for x in rs if x['variant_index']==vi and x['repeat_index']==rep and x['context_mode']=='no_context'][0]
   assert a['seed']==b['seed']
 # each variant has exactly 3 repeats and both contexts
 for vi in range(6):
  for mode in ('full_context','no_context'):
   z=[r for r in rs if r['variant_index']==vi and r['context_mode']==mode]
   assert len(z)==3
   assert {x['repeat_index'] for x in z}=={0,1,2}
# source gold mapping and protocol constraints
for r in req:
 if r['question_type']=='partial_order':
  assert r['gold_option_ids']==[]
  assert r['gold_pairs'] and r['query_pairs']
  assert set(map(tuple,r['gold_pairs'])).issubset(set(map(tuple,r['query_pairs'])))
print(json.dumps({'status':'PASS','request_count':len(req),'counts':dict(c),'candidates':len(by),'requests_per_candidate':sorted(set(map(len,by.values())))},indent=2))
