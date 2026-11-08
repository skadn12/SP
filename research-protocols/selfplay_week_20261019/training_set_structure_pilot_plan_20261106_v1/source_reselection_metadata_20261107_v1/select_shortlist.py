import hashlib,json
from pathlib import Path
P=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_structure_pilot_plan_20261106_v1')
OUT=P/'source_reselection_metadata_20261107_v1'; A=P/'source_history_audit_20261106_v2'; OLD=P/'source_admission_cc_train_full_20261106_v1'
rule=json.loads((OUT/'SELECTION_RULE_FREEZE.json').read_text())
canonical={r['root_id'] for r in map(json.loads,open(rule['canonical_split']))}
old=json.loads((OLD/'SOURCE_RESERVATIONS.json').read_text()); excluded={r['root_id'].rsplit('-',1)[-1] for r in old['reserved_roots']}
for filename in ('SEMANTIC_NEAR_DUPLICATE_SCREEN.json','SUMMARY_TOPIC_SIMILARITY_SCREEN.json'):
 d=json.loads((A/filename).read_text())
 for result in d['results'].values():
  for entries in result.values():
   for row in entries: excluded.add(str(row[1]))
seed=rule['selection']['seed']; groups=rule['selection']['groups']; chosen={}; eligible=0
with open(rule['source_ledger']) as f:
 for line in f:
  d=json.loads(line); root=d['root_id']; n=root.rsplit('-',1)[-1]; rel=d['relationship']
  if rel not in groups or d['historical_status']!='NO_EXACT_TOKEN_HIT_IN_SCOPED_SCAN' or root in canonical or n in excluded: continue
  eligible+=1; h=hashlib.sha256((seed+':'+root).encode()).hexdigest()
  if rel not in chosen or h<chosen[rel]['rank_hash']:
   chosen[rel]={k:d[k] for k in ('root_id','data_id','source_line_index','record_sha256','relationship','historical_status')}; chosen[rel]['rank_hash']=h
assert len(chosen)==len(groups)
report={'schema_version':'cc_metadata_reselection_shortlist_v1','rule_sha256':hashlib.sha256((OUT/'SELECTION_RULE_FREEZE.json').read_bytes()).hexdigest(),'source_content_accessed':False,'ledger_records_scanned':160000,'eligible_metadata_records':eligible,'canonical_count':len(canonical),'excluded_previous_and_neighbor_id_count':len(excluded),'selected_count':len(chosen),'selected':[chosen[g] for g in groups],'source_admission_status':'HOLD_NOT_A_CLASS','operation_family_count_proven':0,'generation_authorization':'NONE','limitations':['root-token historical_status is partial evidence only','social relationship is not task-operation diversity','semantic family exposure and near duplicates remain unexamined for new shortlist']}
(OUT/'METADATA_SHORTLIST.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'eligible':eligible,'selected_count':len(chosen),'roots':[chosen[g]['root_id'] for g in groups]}))
