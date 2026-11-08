#!/usr/bin/env python3
import os,json,hashlib,re
from pathlib import Path
from collections import Counter,defaultdict
BASE=Path('/root/autodl-tmp/gxy-selfplay')
OUT=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_structure_pilot_plan_20261106_v1/source_history_audit_20261106_v2')
CAND=['46251','133062','165698','82382','40917','98734','238031','119609']
EX=['source_admission_cc_train_full_20261106_v1','source_history_audit_20261106_v2']
KEYS={'root_id','rootid','source_id','sourceid','source_item_id','source_item_sha256','source_sha256','parent_root_id','parent_family_id','parent_id','family_id','lineage_parent_id','upstream_id'}
def walk(obj,path=''):
 if isinstance(obj,dict):
  for k,v in obj.items():
   kl=k.lower()
   if kl in KEYS: yield k,v,path+'/'+k
   yield from walk(v,path+'/'+k)
 elif isinstance(obj,list):
  for i,v in enumerate(obj): yield from walk(v,path+f'/{i}')
def vals(v):
 if isinstance(v,(str,int,float,bool)): return [str(v)]
 if isinstance(v,list):
  a=[]
  for x in v:a+=vals(x)
  return a
 return []
files=0; schemas=Counter(); bad=0; roots=Counter(); sources=Counter(); shas=Counter(); hits=defaultdict(list); parent_fields=0
for dp,dn,fs in os.walk(BASE):
 dn[:]=[d for d in dn if d not in ('.git','worktrees','CLUTRR-test','base_restart_current_method_validation','restart-n5-runs','current_method_runs','eval_sets')]
 if any(x in str(dp) for x in EX): continue
 for fn in fs:
  if fn!='SOURCE_RESERVATIONS.json': continue
  p=Path(dp)/fn
  if any(x in str(p) for x in EX): continue
  files+=1
  try: obj=json.loads(p.read_text(encoding='utf-8'))
  except Exception: bad+=1; continue
  schema='<missing>'
  if isinstance(obj,dict): schema=str(obj.get('schema_version','<missing>'))
  elif isinstance(obj,list) and obj and isinstance(obj[0],dict): schema=str(obj[0].get('schema_version','<missing>'))
  schemas[schema]+=1
  local=[]
  for k,v,path in walk(obj):
   for x in vals(v):
    local.append((k.lower(),x))
    if k.lower() in ('root_id','rootid'): roots[x]+=1
    if k.lower() in ('source_id','sourceid','source_item_id'): sources[x]+=1
    if 'sha256' in k.lower(): shas[x]+=1
    if k.lower() in ('parent_root_id','parent_family_id','parent_id','family_id','lineage_parent_id','upstream_id'): parent_fields+=1
  for c in CAND:
   for k,x in local:
    if re.search(rf'(?<!\d){re.escape(c)}(?!\d)',x):
     hits[c].append({'path':str(p),'schema':schema,'field':k,'value':x[:300]})
report={'schema_version':'cc_parent_source_reverse_index_v1','read_only':True,'historical_scope_excludes':EX,'reservation_files_scanned':files,'unreadable_files':bad,'schema_counts':dict(schemas),'unique_root_id_values':len(roots),'unique_source_id_values':len(sources),'unique_sha256_values':len(shas),'parent_or_family_field_occurrences':parent_fields,'candidate_identity_hits':{c:hs[:20] for c,hs in hits.items()},'candidate_hit_counts':{c:len(hs) for c,hs in hits.items()},'top_root_ids':roots.most_common(20),'top_source_ids':sources.most_common(20),'limitations':['field identity matches only; source_id may belong to another dataset family','semantic parent-family/near-duplicate mapping is not encoded in all schemas','current reservation and audit namespaces were excluded','no authoring, solver, training, export']}
(OUT/'PARENT_SOURCE_REVERSE_INDEX.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'files':files,'bad':bad,'schemas':dict(schemas),'candidate_hits':{c:len(hits.get(c,[])) for c in CAND}},ensure_ascii=False))
