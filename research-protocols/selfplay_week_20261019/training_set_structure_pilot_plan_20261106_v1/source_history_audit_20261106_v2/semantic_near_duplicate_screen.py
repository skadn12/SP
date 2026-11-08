#!/usr/bin/env python3
"""Bounded, read-only lexical near-duplicate screen for 8 CC roots vs all 160k train rows."""
import json,re,heapq
from pathlib import Path
from collections import defaultdict
SRC=Path('/root/autodl-tmp/gxy-selfplay/data/conversation_chronicles/train.jsonl')
OUT=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_structure_pilot_plan_20261106_v1/source_history_audit_20261106_v2')
TARGET={'46251','133062','165698','82382','40917','98734','238031','119609'}
DIALOGUE=[f'{x}_session_dialogue' for x in ('first','second','third','fourth','fifth')]
def toks(v):
 if isinstance(v,list): s=' '.join(str(x) for x in v)
 else:s=str(v)
 return re.findall(r"[\w']+",s.lower())
def shingles(ts,n=5): return set(zip(*(ts[i:] for i in range(n)))) if len(ts)>=n else set()
def content(d):
 ts=[]; sess=[]
 for k in DIALOGUE:
  x=toks(d.get(k,[])); sess.append(x); ts+=x
 # summary is a model-generated restatement; retain separately in full lexical screen
 ts += toks(d.get('summary',[]))
 return ts,sess

def top_push(h, item, k=10):
 score=item[0]
 if len(h)<k: heapq.heappush(h,item)
 elif score>h[0][0]: heapq.heapreplace(h,item)
# first pass gets target signatures
raw={}
with SRC.open(encoding='utf-8') as f:
 for i,line in enumerate(f):
  d=json.loads(line); n=d['dataID'].rsplit('-',1)[-1]
  if n in TARGET:
   ft,ss=content(d); raw[n]={'line':i,'full_tokens':set(ft),'full_shingles':shingles(ft),'sessions':[{'tokens':set(x),'shingles':shingles(x)} for x in ss]}
assert len(raw)==8
# indexes for fast overlap counts
full_tok_index=defaultdict(list); full_shingle_index=defaultdict(list); sess_shingle_index=defaultdict(list)
for n,r in raw.items():
 for t in r['full_tokens']: full_tok_index[t].append(n)
 for g in r['full_shingles']: full_shingle_index[g].append(n)
 for j,s in enumerate(r['sessions']):
  for g in s['shingles']: sess_shingle_index[g].append((n,j))
full_heaps={n:[] for n in TARGET}; full_shingle_heaps={n:[] for n in TARGET}; sess_heaps={n:[] for n in TARGET}
count=0
with SRC.open(encoding='utf-8') as f:
 for i,line in enumerate(f):
  d=json.loads(line); n=d['dataID'].rsplit('-',1)[-1]
  if n in TARGET: continue
  count+=1; ft,ss=content(d); fset=set(ft); fgrams=shingles(ft)
  # full token Jaccard using sparse shared-token counts
  shared=defaultdict(int)
  for t in fset:
   for c in full_tok_index.get(t,[]): shared[c]+=1
  for c,inter in shared.items():
   den=len(fset)+len(raw[c]['full_tokens'])-inter
   top_push(full_heaps[c],(inter/den,n,i,inter,len(fset)),10)
  # full 5-gram containment/Jaccard
  sharedg=defaultdict(int)
  for g in fgrams:
   for c in full_shingle_index.get(g,[]): sharedg[c]+=1
  for c,inter in sharedg.items():
   cg=len(raw[c]['full_shingles']); den=len(fgrams)+cg-inter
   top_push(full_shingle_heaps[c],(inter/min(len(fgrams),cg),n,i,inter,den and inter/den),10)
  # per-session 5gram containment, sparse candidate session index
  for j,x in enumerate(ss):
   gs=shingles(x); sg=defaultdict(int)
   for g in gs:
    for c,cj in sess_shingle_index.get(g,[]): sg[(c,cj)]+=1
   for (c,cj),inter in sg.items():
    cg=len(raw[c]['sessions'][cj]['shingles'])
    if not cg or not gs: continue
    den=len(gs)+cg-inter
    top_push(sess_heaps[c],(inter/min(len(gs),cg),n,i,cj,inter,inter/den),20)
def serial(h): return [list(x) for x in sorted(h,reverse=True)]
# collision interpretation threshold is deliberately conservative; lexical screen is not semantic proof.
report={'schema_version':'cc_source_semantic_near_duplicate_screen_v1','source_path':str(SRC),'records_scanned':count+8,'target_count':8,'method':{'full_token_metric':'Jaccard over normalized word sets (dialogues + summaries)','full_shingle_metric':'5-word-shingle containment and Jaccard','session_shingle_metric':'per-session 5-word-shingle containment and Jaccard','thresholds_for_manual_review':{'full_token_jaccard':0.35,'full_shingle_containment':0.30,'session_shingle_containment':0.50},'no_model_or_generation_used':True},'results':{}}
for c in sorted(TARGET):
 report['results'][c]={'full_token_top10':serial(full_heaps[c]),'full_shingle_top10':serial(full_shingle_heaps[c]),'session_shingle_top20':serial(sess_heaps[c])}
# status based on conservative high-overlap flags; generic low lexical overlap remains UNKNOWN rather than PASS lineage.
flags={c:{'full_token_ge_035':any(x[0]>=.35 for x in full_heaps[c]),'full_shingle_ge_030':any(x[0]>=.30 for x in full_shingle_heaps[c]),'session_shingle_ge_050':any(x[0]>=.50 for x in sess_heaps[c])} for c in TARGET}
report['flags']=flags
report['status']='HOLD_MANUAL_REVIEW_REQUIRED' if any(any(v.values()) for v in flags.values()) else 'NO_HIGH_LEXICAL_COLLISION_BUT_SEMANTIC_LINEAGE_UNPROVEN'
report['limitations']=['lexical screen does not prove semantic independence','near paraphrases with low word/shingle overlap may be missed','high overlap can arise from generic conversation phrases and requires manual source review','parent-family metadata remains incomplete','no authoring, solver, training, export']
(OUT/'SEMANTIC_NEAR_DUPLICATE_SCREEN.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':report['status'],'records_scanned':count+8,'flags':flags},ensure_ascii=False))
