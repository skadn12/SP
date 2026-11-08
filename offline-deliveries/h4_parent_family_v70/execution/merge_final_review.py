from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent.parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'EVALUATION_PREPARATION.STARTED.json').exists()
pool=read(R/'POOL.json');qs={q['candidate_id']:q for q in pool['questions']};rows=[];bindings=[]
for g in 'ab':
 jp=R/'final_review'/('group_'+g+'_SOURCE_JUDGMENTS.json');ap=R/'final_review'/('group_'+g+'_AUDIT.json');j=read(jp);a=read(ap)
 assert j['solver_scores_used'] is False and j['child_gold_read_before_save'] is False
 assert a['solver_scores_used'] is False and a['pool_sha256']==sha(R/'POOL.json') and a['source_judgments_sha256']==sha(jp)
 judges={q['candidate_id']:q for q in j['questions']};assert set(judges)=={q['candidate_id'] for q in a['questions']}
 for ar in a['questions']:
  q=qs[ar['candidate_id']];sj=judges[ar['candidate_id']];assert ar['root_id']==sj['root_id']==q['root_id'];assert ar['source_sha256']==sj['source_sha256']==q['source_sha256']==sha(Path(q['source_path']))
  assert set(sj['options'])==set('ABCDEF');turns={t['turn_ref']:t for t in read(Path(q['source_path']))['turns']}
  for o in sj['options'].values():
   assert o['relation'] in ('supports','contradicts','unknown')
   for c in o['citations']:assert c['speaker']==turns[c['turn_ref']]['speaker'] and c['quote'] in turns[c['turn_ref']]['text']
  if ar['core_status']=='CORE_USABLE':
   assert sj['stem_valid'] and all(o['relation']!='unknown' and o['scope_valid'] and o['fact_distinct'] for o in sj['options'].values())
   assert {k for k,o in sj['options'].items() if o['relation']=='supports'}==set(q['gold'])
  rows.append(ar)
 bindings.extend({'path':str(p),'sha256':sha(p)} for p in [jp,ap])
assert len(rows)==len(qs) and {x['candidate_id'] for x in rows}==set(qs)
v={'pool_sha256':sha(R/'POOL.json'),'solver_scores_used':False,'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_all_true_provenance_known':True,'child_gold_read_before_save':False,'independent_review_files':bindings,'questions':rows}
with (R/'FINAL_CONTENT_REVIEW.json').open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'questions':len(rows),'usable':sum(x['core_status']=='CORE_USABLE' for x in rows)}))
