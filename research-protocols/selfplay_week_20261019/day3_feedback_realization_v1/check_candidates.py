from pathlib import Path
import json,hashlib,collections,datetime
HERE=Path(__file__).resolve().parent

def bind(p):
 b=p.read_bytes(); return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))

def walk(v):
 if isinstance(v,dict):
  if 'turn_ref' in v and 'quote' in v: yield v
  for child in v.values(): yield from walk(child)
 elif isinstance(v,list):
  for child in v: yield from walk(child)

def main():
 f=json.loads((HERE/'GENERATION_FREEZE.json').read_text()); slots={s['candidate_id']:s for s in f['slots']}
 errors=[]; checks=0; quotes=0; records=[]; disp=[]; bindings=[]
 def check(ok,kind,qid):
  nonlocal checks
  checks+=1
  if not ok: errors.append(dict(candidate_id=qid,kind=kind))
 for path in sorted((HERE/'authors').glob('part_?.json')):
  d=json.loads(path.read_text()); records.extend(d['questions']); disp.extend(d['slot_dispositions']); bindings.append(bind(path))
 check(collections.Counter(x['candidate_id'] for x in disp)==collections.Counter(slots.keys()),'ALL_SLOTS_EXACTLY_ONCE','batch')
 check(len(records)==len(set(q['candidate_id'] for q in records)),'UNIQUE_CANDIDATE_IDS','batch')
 for q in records:
  qid=q['candidate_id']; s=slots.get(qid); check(s is not None,'FROZEN_SLOT',qid)
  if s is None: continue
  for k in ['root_id','question_type','feedback_id']: check(q.get(k)==s[k],'FROZEN_'+k,qid)
  check(q['source_path']==s['source']['path'] and q['source_sha256']==s['source']['sha256'],'BOUND_SOURCE',qid)
  src=json.loads(Path(q['source_path']).read_text()); turns={t['turn_ref']:t for t in src['turns']}
  check(bind(Path(q['source_path']))['sha256']==q['source_sha256'],'SOURCE_BYTES_UNCHANGED',qid)
  key='options' if q['question_type']=='multi_select' else 'items'; entries=q.get(key,{})
  check(len(entries)==s['item_count'],'FROZEN_ARITY',qid)
  check(set(entries)==(set('ABCDEF') if key=='options' else {'I'+str(i+1) for i in range(len(entries))}),'PUBLIC_LABELS',qid)
  check(all(isinstance(v,str) and v.strip() for v in entries.values()),'NONEMPTY_TEXTS',qid)
  g=q.get('gold',[]); check(len(g)==len(set(g)) and set(g)<=set(entries),'GOLD_VALID',qid)
  check(len(g)==s['target_gold_count'] if key=='options' else set(g)==set(entries),'FROZEN_GOLD_OR_FULL_PERMUTATION',qid)
  for k in ['question','template_id','family_id','per_item_evidence','target_realization','novelty_limits']:check(bool(q.get(k)),'REQUIRED_'+k,qid)
  ev=q.get('per_item_evidence',{})
  if isinstance(ev,dict): check(set(ev)==set(entries),'PER_ITEM_EVIDENCE_COVERAGE',qid)
  elif isinstance(ev,list):check({x.get('label') for x in ev}==set(entries),'PER_ITEM_EVIDENCE_COVERAGE',qid)
  qq=list(walk(q));check(bool(qq),'EXACT_EVIDENCE_PRESENT',qid)
  for c in qq:
   quotes+=1; check(c['turn_ref'] in turns and isinstance(c['quote'],str) and bool(c['quote']) and c['quote'] in turns.get(c['turn_ref'],{}).get('text',''),'EXACT_SOURCE_QUOTE:'+c['turn_ref'],qid)
 old=json.loads((HERE.parent/'day3_targeted_4root_v1/exports/PUBLIC_QUESTIONS.json').read_text())['questions']
 exact=[]
 for q in records:
  key='options' if q['question_type']=='multi_select' else 'items'
  for o in old:
   if o['root_id']==q['root_id'] and o['question_type']==q['question_type'] and o['question']==q['question'] and set(o[key].values())==set(q[key].values()):exact.append([q['candidate_id'],o['candidate_id']])
 check(not exact,'NO_EXACT_OLD_QUESTION_DUPLICATE','batch')
 out=dict(schema_version='feedback_realization_structural_check_v1',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS' if not errors else 'FAIL',checks=checks,candidate_records=len(records),slot_dispositions=len(disp),exact_quote_occurrences=quotes,errors=errors,exact_old_duplicates=exact,author_bindings=bindings,semantic_review_established=False,new_solver_calls=0)
 (HERE/'review/AUTHOR_STRUCTURAL_VALIDATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
 print(json.dumps(out,ensure_ascii=False))
 if errors: raise SystemExit(1)
if __name__=='__main__':main()
