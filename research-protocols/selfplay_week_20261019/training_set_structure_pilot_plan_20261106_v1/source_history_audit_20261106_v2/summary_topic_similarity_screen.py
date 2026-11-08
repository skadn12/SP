import json,re,heapq
from pathlib import Path
SRC=Path('/root/autodl-tmp/gxy-selfplay/data/conversation_chronicles/train.jsonl'); OUT=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_structure_pilot_plan_20261106_v1/source_history_audit_20261106_v2')
TARGET={'46251','133062','165698','82382','40917','98734','238031','119609'}
def toks(v): return set(re.findall(r"[\w']+",' '.join(map(str,v)).lower()))
def push(h,x,k=20):
 if len(h)<k: heapq.heappush(h,x)
 elif x[0]>h[0][0]: heapq.heapreplace(h,x)
raw={}
with SRC.open() as f:
 for i,l in enumerate(f):
  d=json.loads(l); n=d['dataID'].rsplit('-',1)[-1]
  if n in TARGET: raw[n]={'line':i,'summary':toks(d.get('summary',[])),'sessions':[toks(d.get(k,[])) for k in ('first_session_dialogue','second_session_dialogue','third_session_dialogue','fourth_session_dialogue','fifth_session_dialogue')]}
heaps={n:[] for n in TARGET}; sessheaps={n:[] for n in TARGET}; count=0
with SRC.open() as f:
 for i,l in enumerate(f):
  d=json.loads(l); n=d['dataID'].rsplit('-',1)[-1]
  if n in TARGET: continue
  count+=1; s=toks(d.get('summary',[]));
  for c,r in raw.items():
   inter=len(s&r['summary']); den=len(s|r['summary']);
   if den: push(heaps[c],(inter/den,n,i,inter,len(s)))
  ss=[toks(d.get(k,[])) for k in ('first_session_dialogue','second_session_dialogue','third_session_dialogue','fourth_session_dialogue','fifth_session_dialogue')]
  for c,r in raw.items():
   for j,x in enumerate(ss):
    inter=len(x&r['sessions'][j]); den=len(x|r['sessions'][j]);
    if den: push(sessheaps[c],(inter/den,n,i,j,inter,len(x)))
res={'schema_version':'cc_summary_topic_similarity_screen_v1','records_scanned':count+8,'method':'normalized summary token Jaccard and same-session dialogue token Jaccard; topic overlap requires manual adjudication','results':{}}
for c in sorted(TARGET): res['results'][c]={'summary_top20':[list(x) for x in sorted(heaps[c],reverse=True)],'same_session_top20':[list(x) for x in sorted(sessheaps[c],reverse=True)]}
res['limitations']=['topic similarity is not proof of derived lineage','generic advice themes can score highly','no model/generation/solver/training/export']
(OUT/'SUMMARY_TOPIC_SIMILARITY_SCREEN.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)); print(json.dumps({'records_scanned':count+8},ensure_ascii=False))
