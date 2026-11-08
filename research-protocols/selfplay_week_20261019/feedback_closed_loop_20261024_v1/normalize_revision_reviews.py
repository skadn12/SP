import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 seen=set()
 for folder in sorted((P/'question_review/revision_packets').glob('review_*')):
  md=read(folder/'PRIVATE_MAPPING.json');rd=read(folder/'REVIEW_DECISIONS.json');rs={r['review_id']:r for r in rd['reviews']};assert set(rs)=={m['review_id'] for m in md['mapping']}
  inputs=[folder/n for n in ['PUBLIC_PACKET.json','AUTHOR_KEY_PACKET.json','INDEPENDENT_GOLD.json','REVIEW_DECISIONS.json','PRIVATE_MAPPING.json']]
  for m in md['mapping']:
   cid=m['candidate_id'];assert cid not in seen;seen.add(cid)
   save(P/'question_review/revisions'/f'{cid}_REVIEW.json',{'reviewer_id':rd['reviewer_id'],'review':rs[m['review_id']],'reviewed_candidate_sha256':m['candidate_sha256'],'source_review_batch':m['batch'],'same_reviewer_re_review':True,'new_independent_sample':False,'bindings':[bind(x) for x in inputs]})
 print(json.dumps({'revision_reviews_normalized':len(seen)}))
if __name__=='__main__':main()
