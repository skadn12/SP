"""Read-only audit of immutable author artifacts; write only explicitly named new report."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json
R=Path(__file__).resolve().parent

def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def walk(x,path=''):
 if isinstance(x,dict):
  if isinstance(x.get('turn_ref'),str) and 'quote' in x:yield path,x
  for k,v in x.items():yield from walk(v,path+'/'+k)
 elif isinstance(x,list):
  for i,v in enumerate(x):yield from walk(v,path+'/'+str(i))
def inspect():
 roots=load(R.parent/'day1/TRAIN_ROOTS.json')['root_ids'];rows=[];seen=set();types=Counter();arms=Counter();macros=Counter()
 for arm in ['F','G']:
  for root in roots:
   ad=R/'authoring'/arm/root
   row={'root_id':root,'arm':arm,'state':'PENDING','issues':[]}
   required=['EXPLORATION.json','TARGETS_FROZEN.json','TARGETS_FREEZE.json','CANDIDATES_FIRST_DRAFT.json']
   if not all((ad/f).exists() for f in required):rows.append(row);continue
   try:
    i=load(ad/'AUTHOR_INPUT.json'); s=load(i['source_path']);ts={t['turn_ref']:t for t in s['turns']};e=load(ad/'EXPLORATION.json');t=load(ad/'TARGETS_FROZEN.json');f=load(ad/'TARGETS_FREEZE.json');d=load(ad/'CANDIDATES_FIRST_DRAFT.json');issues=row['issues']
    row.update(state='FROZEN',draft_count=len(d.get('candidates',[])),source_sha256=h(i['source_path']),artifacts=[{'path':str((ad/n).relative_to(R)),'sha256':h(ad/n)} for n in required])
    if row['draft_count']>1:issues.append('draft_cap_exceeded')
    if row['source_sha256']!=i['source_sha256']:issues.append('source_hash_changed')
    if len(ts)!=i['source_turn_count']:issues.append('source_turn_count')
    if e.get('full_source_read') is not True or e.get('turn_count',e.get('source_turn_count'))!=len(ts):issues.append('full_source_read_certificate')
    if f.get('targets_sha256',f.get('sha256'))!=h(ad/'TARGETS_FROZEN.json'):issues.append('target_freeze_hash')
    if (ad/'TARGETS_FROZEN.json').stat().st_mtime_ns>(ad/'CANDIDATES_FIRST_DRAFT.json').stat().st_mtime_ns:issues.append('target_not_written_before_draft')
    citations=0
    for name,doc in [('targets',t),('drafts',d)]:
     for loc,c in walk(doc):
      citations+=1;tr=ts.get(c['turn_ref'])
      if tr is None:issues.append(name+loc+':missing_ref');continue
      if not isinstance(c['quote'],str) or not c['quote'] or c['quote'] not in tr['text']:issues.append(name+loc+':quote_mismatch')
      if 'origin' in c and c['origin']!=tr.get('origin'):issues.append(name+loc+':origin_mismatch')
    row['citations_checked']=citations
    for c in d.get('candidates',[]):
     cid=c['candidate_id'];row['candidate_id']=cid
     if cid in seen:issues.append('duplicate_candidate_id')
     seen.add(cid)
     if c.get('root_id')!=root:issues.append('root_mismatch')
     # Some original common-schema writers put arm in wrapper; retain wrapper as authority.
     if c.get('arm',d.get('arm'))!=arm:issues.append('arm_mismatch')
     if c.get('source_sha256')!=i['source_sha256'] or c.get('source_path')!=i['source_path'] or c.get('source_turn_count')!=len(ts):issues.append('candidate_source_binding')
     if c.get('targets_sha256')!=h(ad/'TARGETS_FROZEN.json'):issues.append('candidate_target_binding')
     opts=[x['option_id'] for x in c['options']];gold=c.get('gold_option_ids',[]);qa=c['question_type'];types[qa]+=1;arms[arm]+=1
     target=t.get('targets',[t])[0] if t.get('targets',[t]) else {}
     answer=target.get('answer',{})
     tg=target.get('gold_option_ids',answer.get('gold_option_ids') if isinstance(answer,dict) else answer)
     target_facts=target.get('option_facts',[])
     if target_facts and {x.get('option_id') for x in target_facts}!=set(opts):issues.append('draft_option_domain_differs_from_frozen_target')
     if qa=='ordering' and tg!=gold:issues.append('draft_gold_differs_from_frozen_target')
     if qa=='multi_select' and (tg is None or set(tg)!=set(gold)):issues.append('draft_gold_differs_from_frozen_target')
     if qa=='partial_order':
      tp=target.get('gold_pairs',answer.get('gold_pairs') if isinstance(answer,dict) else None)
      if tp is None or {tuple(x) for x in tp}!={tuple(x) for x in c.get('gold_pairs',[])}:issues.append('draft_pairs_differ_from_frozen_target')
     macro=c.get('macrostructure',c.get('relation_plan',{}).get('macrostructure',e.get('macrostructure','unspecified')));macros[str(macro)]+=1;row.update(question_type=qa,macrostructure=macro)
     if len(opts)!=len(set(opts)):issues.append('duplicate_option_id')
     audit={x['option_id']:x for x in c.get('option_audit',[])}
     if set(audit)!=set(opts):issues.append('option_audit_coverage')
     if qa=='multi_select':
      if not(2<=len(gold)<=6 and len(gold)<len(opts)):issues.append('multi_gold_cardinality')
      if not set(gold)<=set(opts) or len(gold)!=len(set(gold)):issues.append('multi_gold_ids')
      if any(not isinstance(audit.get(x,{}).get('correct'),bool) or audit[x]['correct']!=(x in gold) for x in opts):issues.append('multi_certificate_gold')
     elif qa=='ordering':
      if len(gold)!=len(opts) or set(gold)!=set(opts):issues.append('ordering_gold')
      if any(audit.get(x,{}).get('gold_position')!=gold.index(x)+1 for x in opts if x in gold):issues.append('ordering_positions')
     elif qa=='partial_order':
      gp=c.get('gold_pairs',[]);pairs={tuple(x) for x in gp}
      qp=c.get('query_pairs',[]);qdomain={frozenset(q) for q in qp}
      if gold!=[]:issues.append('partial_gold_schema')
      if any(len(x)!=2 or x[0] not in opts or x[1] not in opts or x[0]==x[1] for x in pairs):issues.append('partial_pair_domain')
      if any((b,a) in pairs for a,b in pairs):issues.append('partial_cycle')
      reachable=set()
      for start in opts:
       todo=[b for a,b in pairs if a==start];visited=set()
       while todo:
        node=todo.pop()
        if node in visited:continue
        visited.add(node);reachable.add((start,node));todo.extend(b for a,b in pairs if a==node)
      if any(a==b for a,b in reachable) and 'partial_cycle' not in issues:issues.append('partial_cycle')
      if any(a!=b and frozenset((a,b)) in qdomain and (a,b) not in pairs for a,b in reachable):issues.append('partial_not_closed')
      if not c.get('query_scope') or not c.get('query_pairs') or not c.get('pair_audit'):issues.append('partial_domain_certificate')
      if any(len(q)!=2 or q[0] not in opts or q[1] not in opts or q[0]==q[1] for q in qp):issues.append('partial_query_pair_domain')
      if len(qdomain)!=len(qp):issues.append('partial_duplicate_query_pair')
      if any(frozenset(q) not in qdomain for q in pairs):issues.append('partial_gold_outside_query_domain')
      pa=c.get('pair_audit',[])
      if pa and all('pair' in q for q in pa) and {frozenset(q['pair']) for q in pa}!=qdomain:issues.append('partial_pair_audit_coverage')
      if c.get('answer_semantics')!='all_and_only_entailed_pairs_in_query_scope':issues.append('partial_answer_semantics')
     else:issues.append('disallowed_type')
    row['mechanical_status']='PASS' if not issues else 'HOLD'
   except Exception as ex:row.update(state='READ_ERROR',mechanical_status='HOLD');row['issues'].append(type(ex).__name__+':'+str(ex))
   rows.append(row)
 return {'run_id':R.name,'required_roots':70,'required_opportunities':140,'frozen_opportunities':sum(x['state']=='FROZEN' for x in rows),'pending_opportunities':sum(x['state']=='PENDING' for x in rows),'draft_count':sum(x.get('draft_count',0) for x in rows),'candidate_types':dict(types),'drafts_by_arm':dict(arms),'macrostructures_as_authored':dict(macros),'mechanical_pass_opportunities':sum(x.get('mechanical_status')=='PASS' for x in rows),'mechanical_hold_opportunities':sum(x.get('mechanical_status')=='HOLD' for x in rows),'solver_calls':0,'quality_and_semantic_status':'PENDING_INDEPENDENT_REVIEW','rows':rows}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out');a=p.parse_args();v=inspect()
 if a.out:
  with (R/a.out).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in v.items() if k!='rows'},ensure_ascii=False,indent=2))
 for x in v['rows']:
  if x['issues']:print(x['root_id'],x['arm'],x['issues'])
