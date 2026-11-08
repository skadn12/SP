"""Combine independently judged content. Never derive a semantic PASS in code."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[2]/'scripts/current_method'))
import production_question_gate as gate

def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def bind(p):return gate.file_binding(p)
def normal(c,m,r,reviewer):
 assert m['candidate_id']==c['candidate_id'] and m['candidate_sha256']==gate.fingerprint(c)
 assert reviewer and reviewer!=c['author_id']
 assert r['decision'] in ['PASS','HOLD','FAIL']
 need=set(gate.required_checks(c));assert need<=set(r['checks']),(c['candidate_id'],need-set(r['checks']))
 if r['decision']=='PASS':
  assert r['reconstructed_gold']==c['gold'],(c['candidate_id'],'reconstructed Gold differs')
  assert r['r03_necessity']['decision']=='PASS',(c['candidate_id'],'R03 not necessary')
 row={'candidate_id':c['candidate_id'],'candidate_sha256':gate.fingerprint(c),'source_binding':c['source_binding'],'author_id':c['author_id'],'reviewer_id':reviewer,'decision':r['decision'],'checks':r['checks'],'independent_gold':r['independent_gold'],'reconstructed_gold':r['reconstructed_gold'],'r03_necessity':r['r03_necessity'],'issues':r.get('issues',[]),'required_revision':r.get('required_revision',[]),'gold_discrepancy_resolution':r.get('gold_discrepancy_resolution'),'source_read_complete':True,'solver_results_read':False,'gold_blind_initial_stage':True,'arm_and_feedback_hidden':True,'review_id':m['review_id'],'review_batch':m['batch']}
 val=gate.validate_review(c,row,P/'SOURCE_SELECTION_FINAL.json')
 if r['decision']=='PASS':assert val['admitted'],val
 return row,val

def main():
 a=argparse.ArgumentParser();a.add_argument('phase',choices=['initial','final']);o=a.parse_args()
 candidates=[c for arm in ['F','G'] for c in read(P/f'question_authoring/{arm}/CANDIDATES_DRAFT.json')]
 assert 0<len(candidates)<=64 and len({c['candidate_id'] for c in candidates})==len(candidates)
 byid={c['candidate_id']:c for c in candidates};mapped={};rows=[];vals=[];inputpaths=[]
 for batch in sorted((P/'question_review').glob('batch*')):
  if not (batch/'PRIVATE_MAPPING.json').exists():continue
  md=read(batch/'PRIVATE_MAPPING.json');rp=batch/'REVIEW_DECISIONS.json';rd=read(rp)
  refs={r['review_id']:r for r in rd['reviews']};assert set(refs)=={m['review_id'] for m in md['mapping']}
  addp=batch/'REVIEW_ADDENDUM.json'
  if addp.exists():
   for add in read(addp)['reviews']:
    oid=add['review_id'];assert oid in refs;refs[oid]={**refs[oid],**add}
   inputpaths.append(addp)
  for m in md['mapping']:
   c=byid[m['candidate_id']];assert c['candidate_id'] not in mapped
   mapped[c['candidate_id']]=(m,refs[m['review_id']],rd['reviewer_id'])
  inputpaths += [batch/n for n in ['PRIVATE_MAPPING.json','PUBLIC_PACKET.json','AUTHOR_KEY_PACKET.json','INDEPENDENT_GOLD.json','REVIEW_DECISIONS.json']]
 assert set(mapped)==set(byid),set(byid)-set(mapped)
 for c in candidates:
  m,r,rid=mapped[c['candidate_id']];row,val=normal(c,m,r,rid);rows.append(row);vals.append(val)
 if o.phase=='initial':
  save(P/'CANDIDATES_INITIAL_PRIVATE.json',candidates);save(P/'CONTENT_REVIEW_INITIAL.json',{'reviews':rows,'bindings':[bind(p) for p in inputpaths]});save(P/'CONTENT_INITIAL_GATE.json',{'validations':vals})
  for arm in ['F','G']:
   rs=[r for r in rows if byid[r['candidate_id']]['arm']==arm and r['decision']!='PASS']
   save(P/f'question_authoring/{arm}/REVISION_REQUESTS.json',{'revision_budget':1,'requests':rs,'rule':'只修所列内容问题，不接触另臂或Solver；保留初稿，输出部分list CANDIDATES_REVISION_01.json，不改旧文件。'})
  print(json.dumps({'initial_reviews':len(rows),'pass':sum(r['decision']=='PASS' for r in rows),'held':[r['candidate_id'] for r in rows if r['decision']!='PASS']}));return
 # Revisions are optional and may replace only slots held in the initial review.
 original_reviews=copy.deepcopy(rows);byreview={r['candidate_id']:r for r in rows}
 revisionpaths=[]
 for arm in ['F','G']:
  cp=P/f'question_authoring/{arm}/CANDIDATES_REVISION_01.json'
  if not cp.exists():continue
  revcs=read(cp)
  for c in revcs:
   cid=c['candidate_id'];assert cid in byid and c['arm']==arm
   if byreview[cid]['decision']=='PASS':
    assert gate.fingerprint(c)==gate.fingerprint(byid[cid]),('PASS_candidate_changed',cid)
    continue
   assert c['root_id']==byid[cid]['root_id'] and c['slot_id']==byid[cid]['slot_id'] and c['author_id']==byid[cid]['author_id']
   rp=P/'question_review/revisions'/f'{cid}_REVIEW.json';rd=read(rp)
   m={'candidate_id':cid,'candidate_sha256':gate.fingerprint(c),'review_id':mapped[cid][0]['review_id'],'batch':'revision01'}
   row,val=normal(c,m,rd['review'],rd['reviewer_id']);byid[cid]=c;byreview[cid]=row;revisionpaths += [cp,rp]
 candidates=[byid[c['candidate_id']] for c in candidates];rows=[byreview[c['candidate_id']] for c in candidates]
 vals=[gate.validate_review(c,r,P/'SOURCE_SELECTION_FINAL.json') for c,r in zip(candidates,rows)]
 # Whole-question duplicates only: option labels/presentation order are ignored.
 exact={};dups=[]
 for c in candidates:
  choices=c.get('options',c.get('items'));key=(c['question_type'],' '.join(c['question'].split()),tuple(sorted(' '.join(v.split()) for v in choices.values())))
  if key in exact:dups.append([exact[key],c['candidate_id']])
  else:exact[key]=c['candidate_id']
 assert not dups,('whole_question_duplicate_requires_explicit_disposition',dups)
 save(P/'CANDIDATES_PRIVATE.json',candidates);save(P/'CONTENT_REVIEW_AI.json',{'reviews':rows,'disclosure':'Same Codex family, separate contexts, arm/feedback hidden, independent Gold saved before author key. AI semantic judgments, not human validation.'})
 save(P/'CONTENT_GATE_FINAL.json',{'validations':vals,'initial_reviews':original_reviews,'whole_question_duplicates':dups,'no_solver_results_used':True,'training':False})
 paths=list((P/'question_review/revision_packets').glob('**/*.json'))+[P/'AUTHORING_PROCESS_DEVIATIONS.json',P/'CANDIDATES_PRIVATE.json',P/'CONTENT_REVIEW_AI.json',P/'SOURCE_SELECTION_FINAL.json',P/'SOURCE_FREEZE.json',P/'CONTENT_GATE_FINAL.json',P/'PROTOCOL_ZH.md',P/'CONFIG.json',P/'question_authoring/COMMON_AUTHOR_CONTRACT_ZH.md']+inputpaths+revisionpaths
 for arm in ['F','G']:
  paths += [P/f'question_authoring/{arm}/{n}' for n in ['AUTHOR_INPUT.json','TASK_PLANS.json','AUTHOR_READ_RECEIPT.json','CANDIDATES_DRAFT.json','SLOT_FAILURES.json']]
 save(P/'CONTENT_FREEZE.json',{'status':'FROZEN_BEFORE_SOLVER','bindings':[bind(p) for p in sorted(set(paths))],'planned_semantic_questions':64,'candidate_count':len(candidates),'admitted':sum(v['admitted'] for v in vals),'failed_or_held':sum(not v['admitted'] for v in vals),'training':False})
 print(json.dumps({'candidates':len(candidates),'admitted':sum(v['admitted'] for v in vals),'held':[r['candidate_id'] for r in rows if r['decision']!='PASS']}))
if __name__=='__main__':main()
