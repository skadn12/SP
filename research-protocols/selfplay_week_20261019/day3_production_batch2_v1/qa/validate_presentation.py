from qa_common import D,read,sha,binding,dump
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json
s=read(D/'PRESENTATION_TRANSFORM_FREEZE.json');idx=read(D/'presentation/PRIVATE_LABEL_MAP.json');summary=read(D/'SUMMARY.json')
rows=idx['rows'];errors=[];checks=0
def check(x,msg):
 global checks
 checks+=1
 if not x:errors.append(msg)
check(idx['transform_freeze']['sha256']==sha(D/'PRESENTATION_TRANSFORM_FREEZE.json'),'freeze binding changed')
records={p.stem:read(p) for p in (D/'final_records').glob('*.json')}
accepted={cid:r for cid,r in records.items() if r['disposition']['state'].startswith('ACCEPT')}
check(len(records)==175,'incomplete final record coverage')
check(len(rows)==len(accepted)==summary['accepted_questions'],'wrong transformed count')
check({r['candidate_id'] for r in rows}==set(accepted),'transformed ID mismatch')
slots={};hist=defaultdict(Counter)
for x in rows:
 cid=x['candidate_id'];r=accepted[cid];a=read(x['original_public']['path']);b=read(x['display_public']['path']);m=x['mapping_old_to_new']
 check(sha(x['original_public']['path'])==x['original_public']['sha256']==r['generation_history'][-1]['candidate_content_sha256'],cid+' initial hash')
 check(sha(x['display_public']['path'])==x['display_public']['sha256'],cid+' display hash')
 check(sorted(m)==list('ABCDEF') and sorted(m.values())==list('ABCDEF'),cid+' bijection')
 order=sorted('ABCDEF',key=lambda old:hashlib.sha256((s['seed']+'|'+x['slot_id']+'|'+old).encode()).hexdigest())
 check(m==dict(zip(order,'ABCDEF')),cid+' wrong registered mapping')
 check({k:v for k,v in a.items() if k!='options_or_items'}=={k:v for k,v in b.items() if k!='options_or_items'},cid+' non-label content change')
 check(all(a['options_or_items'][old]==b['options_or_items'][new] for old,new in m.items()),cid+' text roundtrip')
 original=r.get('multi_select') or r['ordering'];gold=original.get('gold_option_ids',original.get('gold_item_ids'))
 check(gold==x['original_gold'],cid+' wrong source gold')
 inverse={new:old for old,new in m.items()};back=[inverse[y] for y in x['display_gold']]
 check(back==gold if r['question_type']=='ordering' else set(back)==set(gold),cid+' Gold roundtrip')
 if x['slot_id'] in slots:check(slots[x['slot_id']]==m,cid+' cross-arm mapping differs')
 slots[x['slot_id']]=m
 hist[x['arm']+'|'+x['question_type']].update([''.join(x['display_gold'])])
check(sum(r['question_type']=='ordering' for r in rows)==sum(r['question_type']=='ordering' for r in accepted.values()),'ORDER display count')
report=dict(status='PASS' if not errors else 'FAIL',created_at_utc=datetime.now(timezone.utc).isoformat(),checks=checks,transformed_candidates=len(rows),matched_mappings_checked=len(read(D/'MATCHED_SLOT_PAIRS.json')['pairs']),answer_histograms={k:dict(v) for k,v in hist.items()},errors=errors,limits='Relabeling only. Does not add independent questions, ensure perfect finite-sample balance or establish semantic difficulty.')
dump(D/'qa/PRESENTATION_VALIDATION.json',report)
print(json.dumps(report))
raise SystemExit(bool(errors))
