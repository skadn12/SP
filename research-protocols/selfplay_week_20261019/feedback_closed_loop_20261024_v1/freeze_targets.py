import hashlib,json,time
from pathlib import Path
P=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def fp(d):return hashlib.sha256(json.dumps(d,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 assert not (P/'execution/STARTED.json').exists()
 cs={c['candidate_id']:c for c in read(P/'CANDIDATES_PRIVATE.json') if c['arm']=='F'};seen={};paths=[]
 for group in ['review_a','review_b']:
  q=P/'target_review'/group;d=read(q/'PRE_SOLVER_TARGET_REVIEW.json');paths += [q/'PRE_SOLVER_TARGET_REVIEW.json',q/'PRE_SOLVER_TARGET_PACKET.json']
  for r in d['records']:
   cid=r['candidate_id'];assert cid in cs and cid not in seen
   assert r['candidate_sha256']==fp(cs[cid]),('target_candidate_hash_drift',cid)
   status=r['target_realization'];status=status if isinstance(status,str) else status['decision'];assert status in ['PASS','HOLD'];seen[cid]=status
 assert set(seen)==set(cs) and len(seen)==32
 paths += [P/'CANDIDATES_PRIVATE.json',P/'CONTENT_FREEZE.json',P/'execution/PREPARATION.json',P/'EXAMPLE_SELECTION_FREEZE.json']
 out={'status':'TARGET_CRITERIA_FROZEN_BEFORE_SOLVER','time_unix':time.time(),'records':[{'candidate_id':cid,'target_realization':st,'control':cs[cid]['slot_id']=='MC6'} for cid,st in sorted(seen.items())],'bindings':[bind(x) for x in paths],'analysis_rules':{'primary_difficulty':'All same-root/slot F-G content-admitted non-MC6 pairs; no filtering by target-audit status or Solver outcome.','target_secondary':'Within F, report realized-target errors separately from off-target outcomes; MC6 controls excluded from primary mechanism yield.','final_vs_reason':'Target-pattern final errors, specific visible-rationale evidence, and citation-only defects separate. No claim about hidden internal cognition.','dependent_displays':'Two displays are repeated measurements; count unique semantic questions and roots for support.','no_post_solver_revision':True},'training':False}
 with (P/'TARGET_PRE_SOLVER_FREEZE.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'targets':len(seen),'realized':sum(x=='PASS' for x in seen.values()),'held':sum(x=='HOLD' for x in seen.values()),'realized_noncontrol':sum(seen[k]=='PASS' and cs[k]['slot_id']!='MC6' for k in seen)}))
if __name__=='__main__':main()
