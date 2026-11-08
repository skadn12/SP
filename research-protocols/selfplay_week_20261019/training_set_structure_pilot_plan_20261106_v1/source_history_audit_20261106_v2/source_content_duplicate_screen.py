#!/usr/bin/env python3
import json,hashlib,re,unicodedata
from pathlib import Path
SRC=Path('/root/autodl-tmp/gxy-selfplay/data/conversation_chronicles/train.jsonl')
OUT=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_structure_pilot_plan_20261106_v1/source_history_audit_20261106_v2')
TARGET={'46251','133062','165698','82382','40917','98734','238031','119609'}
def norm(x):
 if isinstance(x,list): return '\u241e'.join(norm(y) for y in x)
 if isinstance(x,dict): return '\u241f'.join(k+'='+norm(x[k]) for k in sorted(x))
 return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(x)).lower()).strip()
def hashes(d):
 content={k:v for k,v in d.items() if k!='dataID'}
 out={'full':hashlib.sha256(norm(content).encode()).hexdigest()}
 for k,v in d.items():
  if k.endswith('_dialogue') or k=='summary': out[k]=hashlib.sha256(norm(v).encode()).hexdigest()
 return out
raw={}
with SRC.open(encoding='utf-8') as f:
 for i,line in enumerate(f):
  d=json.loads(line); n=d['dataID'].rsplit('-',1)[-1]
  if n in TARGET: raw[n]={'line':i,'relationship':d.get('relationship'),'hashes':hashes(d)}
assert len(raw)==8,raw.keys()
# exact collision check; only retain target-to-other IDs and counts
maps={k:{} for k in ('full','summary','first_session_dialogue','second_session_dialogue','third_session_dialogue','fourth_session_dialogue','fifth_session_dialogue')}
for n,r in raw.items():
 for k,h in r['hashes'].items():
  if k in maps: maps[k].setdefault(h,[]).append(n)
coll={n:{} for n in TARGET}; total=0
with SRC.open(encoding='utf-8') as f:
 for i,line in enumerate(f):
  d=json.loads(line); n=d['dataID'].rsplit('-',1)[-1]
  if n in TARGET: continue
  total+=1; hs=hashes(d)
  for k,h in hs.items():
   if k not in maps or h not in maps[k]: continue
   for t in maps[k][h]: coll[t].setdefault(k,[]).append({'data_id':n,'line':i})
for n in coll:
 for k in coll[n]: coll[n][k]=coll[n][k][:20]
status='PASS_EXACT_CONTENT_DUPLICATE_SCREEN_NO_COLLISION' if not any(coll[n] for n in coll) else 'HOLD_EXACT_CONTENT_COLLISION'
report={'schema_version':'cc_source_content_duplicate_screen_v1','source_path':str(SRC),'source_record_count_scanned':total+8,'target_count':8,'target_records':raw,'collisions':coll,'status':status,'near_duplicate_status':'NOT_PERFORMED','limitations':['exact normalized component/full-record hashes only','near-duplicate semantic similarity and parent-family lineage still require provenance/index evidence','no authoring, solver, training or export']}
(OUT/'SOURCE_CONTENT_DUPLICATE_SCREEN.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':status,'source_record_count_scanned':total+8,'collision_target_count':sum(bool(v) for v in coll.values())},ensure_ascii=False))
