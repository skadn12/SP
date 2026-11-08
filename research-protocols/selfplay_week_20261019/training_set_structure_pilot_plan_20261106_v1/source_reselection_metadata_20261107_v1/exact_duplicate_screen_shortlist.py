import json,hashlib,re,unicodedata
from pathlib import Path
SRC=Path('/root/autodl-tmp/gxy-selfplay/data/conversation_chronicles/train.jsonl'); OUT=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_structure_pilot_plan_20261106_v1/source_reselection_metadata_20261107_v1')
TARGET={x['data_id'].rsplit('-',1)[-1] for x in json.loads((OUT/'METADATA_SHORTLIST.json').read_text())['selected']}
def norm(x):
 if isinstance(x,list): return '\u241e'.join(norm(y) for y in x)
 if isinstance(x,dict): return '\u241f'.join(k+'='+norm(x[k]) for k in sorted(x))
 return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(x)).lower()).strip()
def hashes(d):
 content={k:v for k,v in d.items() if k!='dataID'}; o={'full':hashlib.sha256(norm(content).encode()).hexdigest()}
 for k,v in d.items():
  if k.endswith('_dialogue') or k=='summary': o[k]=hashlib.sha256(norm(v).encode()).hexdigest()
 return o
raw={}
with SRC.open() as f:
 for i,l in enumerate(f):
  d=json.loads(l); n=d['dataID'].rsplit('-',1)[-1]
  if n in TARGET: raw[n]={'line':i,'relationship':d['relationship'],'hashes':hashes(d)}
assert len(raw)==len(TARGET)
maps={k:{} for k in ('full','summary','first_session_dialogue','second_session_dialogue','third_session_dialogue','fourth_session_dialogue','fifth_session_dialogue')}
for n,r in raw.items():
 for k,h in r['hashes'].items():
  if k in maps: maps[k].setdefault(h,[]).append(n)
coll={n:{} for n in TARGET}; count=0
with SRC.open() as f:
 for i,l in enumerate(f):
  d=json.loads(l); n=d['dataID'].rsplit('-',1)[-1]
  if n in TARGET: continue
  count+=1
  for k,h in hashes(d).items():
   if k in maps and h in maps[k]:
    for t in maps[k][h]: coll[t].setdefault(k,[]).append({'data_id':n,'line':i})
for n in coll:
 for k in coll[n]: coll[n][k]=coll[n][k][:20]
status='PASS_EXACT_CONTENT_DUPLICATE_SCREEN_NO_COLLISION' if not any(coll[n] for n in coll) else 'HOLD_EXACT_CONTENT_COLLISION'
report={'schema_version':'cc_shortlist_exact_content_duplicate_screen_v1','source_path':str(SRC),'source_record_count_scanned':count+len(TARGET),'target_count':len(TARGET),'target_records':raw,'collisions':coll,'status':status,'near_duplicate_status':'NOT_PERFORMED','generation_authorization':'NONE','limitations':['normalized exact hashes only; semantic near duplicates not cleared','no authoring, solver, training, export']}
(OUT/'SHORTLIST_EXACT_CONTENT_SCREEN.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'status':status,'records':count+len(TARGET),'collision_target_count':sum(bool(x) for x in coll.values())}))
