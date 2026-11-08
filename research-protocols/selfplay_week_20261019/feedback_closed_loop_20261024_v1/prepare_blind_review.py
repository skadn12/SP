"""Create two-stage arm-blinded content packets without changing authored text."""
import hashlib,json,random
from pathlib import Path
P=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def fp(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def bind(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def main():
 inputs=[P/f'question_authoring/{a}/CANDIDATES_DRAFT.json' for a in ['F','G']]
 cs=[c for p in inputs for c in read(p)]
 assert len(cs)==64 and len({c['candidate_id'] for c in cs})==64
 save(P/'CANDIDATES_INITIAL_PRIVATE.json',cs)
 mapping=[]
 for group,roots in [('review_a',['218519','222937']),('review_b',['225575','229686'])]:
  public=[];private=[]
  for c in cs:
   if c['root_id'].split('-')[-1] not in roots:continue
   oid='RQ-'+hashlib.sha256(('closed-loop-content-blind-v1:'+c['candidate_id']).encode()).hexdigest()[:10]
   public.append(dict(review_id=oid,**{k:c[k] for k in ['root_id','question_type','task_operation','question','source_binding']},**{k:c[k] for k in ['options','items'] if k in c}))
   private.append(dict(review_id=oid,**{k:c[k] for k in ['gold','item_analysis','ordering_analysis','r03_necessity'] if k in c}))
   mapping.append(dict(review_id=oid,candidate_id=c['candidate_id'],candidate_sha256=fp(c),author_id=c['author_id'],group=group))
  random.Random(2026102401).shuffle(public)
  order={x['review_id']:i for i,x in enumerate(public)};private.sort(key=lambda x:order[x['review_id']])
  assert len(public)==32
  save(P/f'question_review/{group}/PUBLIC_PACKET.json',{'questions':public,'notice':'不打开binding中的作者私有文件；仅用于最终追溯。先独立源推Gold并保存，不准先读AUTHOR_KEY_PACKET。'})
  save(P/f'question_review/{group}/AUTHOR_KEY_PACKET.json',{'questions':private})
 save(P/'question_review/PRIVATE_REVIEW_MAPPING.json',{'mapping':mapping,'input_bindings':[bind(p) for p in inputs],'blindness':'Arm, diagnosis, author and authored Gold hidden until initial source-based answer is saved. Stage2 includes author Gold and evidence, still hides arm/feedback. Same Codex family, shared filesystem.'})
 print('64 candidates; 2 x 32 two-stage blinded packets created')
if __name__=='__main__':main()
