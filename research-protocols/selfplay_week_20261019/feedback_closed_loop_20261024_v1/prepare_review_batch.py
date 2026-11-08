import argparse,hashlib,json,random
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
 a=argparse.ArgumentParser();a.add_argument('batch');a.add_argument('files',nargs='+');o=a.parse_args()
 inputs=[Path(s).resolve() for s in o.files];cs=[c for p in inputs for c in read(p)]
 public=[];private=[];mapping=[]
 for c in cs:
  oid='RQ-'+hashlib.sha256(('closed-loop-content-blind-v1:'+c['candidate_id']).encode()).hexdigest()[:10]
  public.append(dict(review_id=oid,**{k:c[k] for k in ['root_id','question_type','task_operation','question','source_binding']},**{k:c[k] for k in ['options','items'] if k in c}))
  private.append(dict(review_id=oid,**{k:c[k] for k in ['gold','item_analysis','ordering_analysis','r03_necessity'] if k in c}))
  mapping.append(dict(review_id=oid,candidate_id=c['candidate_id'],candidate_sha256=fp(c),author_id=c['author_id'],batch=o.batch))
 random.Random(2026102401).shuffle(public);order={x['review_id']:i for i,x in enumerate(public)};private.sort(key=lambda x:order[x['review_id']])
 b=P/'question_review'/o.batch
 save(b/'PUBLIC_PACKET.json',{'questions':public,'notice':'先独立源推Gold并保存，不准先读AUTHOR_KEY_PACKET。隐藏臂别、作者、诊断信息。'})
 save(b/'AUTHOR_KEY_PACKET.json',{'questions':private})
 save(b/'PRIVATE_MAPPING.json',{'mapping':mapping,'input_bindings':[bind(p) for p in inputs]})
 print(json.dumps({'batch':o.batch,'questions':len(cs)}))
if __name__=='__main__':main()
