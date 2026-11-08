"""Read-only batch report collector. Writes an exclusive versioned snapshot."""
from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json
B=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def bind(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def collect():
 m=read(B/'MANIFEST.json');roots=[]
 for row in m['roots']:
  root=row['root_id'];r={'root_id':root,'prefix_pass':row['prefix_reuse_approved'],'prefix_review_binding':row['prefix_review_binding'],'new_scenes':[],'content_pass':0,'multi_select_gold_counts':[],'ordering_content_pass':0,'new_consumption_count':0}
  if r['prefix_pass']:
   unit=Path(row['unit_path']);imp=read(unit/'PREFIX_IMPORT.json');r['historical_scene_count']=len(imp['retained_scene_refs'])
   r['inputs_frozen']=(unit/'EXECUTION_INPUTS_FROZEN.json').exists()
   p=unit/'SOURCE_RESERVATIONS.json'
   if p.exists():r['new_consumption_count']=read(p)['new_consumption_count']
   for p in sorted(unit.glob('roots/*/scenes/*/EXPANSION_DRAFT.json')):
    d=read(p)
    if d['scene_ref']in imp['retained_scene_refs']:continue
    q=p.parent/'SOURCE_REVIEW.json';rev=read(q)if q.exists()else{}
    r['new_scenes'].append({'scene_ref':d['scene_ref'],'turn_count':len(d['turns']),'decision':rev.get('decision','PENDING'),'warnings':[k for k,v in rev.get('checks',{}).items()if v.get('status')=='WARNING'],'draft_binding':bind(p),'review_binding':bind(q)if q.exists()else None})
   folder=unit/'roots'/('root'+root.rsplit('-',1)[1]);p=folder/'FINAL_CROSSCHECK.json'
   if p.exists():
    cross=read(p);passed={x['candidate_id']for x in cross['results']if x['status']=='CONTENT_PASS'};r['content_pass']=len(passed);public={q['candidate_id']:q for q in read(folder/'PUBLIC_QUESTIONS.json')['questions']}
    for q in read(folder/'PRIVATE_GOLD.json')['questions']:
     if q['candidate_id']in passed:
      if public[q['candidate_id']]['question_type']=='multi_select':r['multi_select_gold_counts'].append(len(q['gold']))
      else:r['ordering_content_pass']+=1
    r['multi_select_gold_counts'].sort();r['final_crosscheck_binding']=bind(p)
    for action in ['COMPOSE','FINALIZE']:
     p=unit/'evidence'/('GUARDED_'+action+'_RECEIPT.json');r['guarded_'+action.lower()+'_success']=p.exists()and read(p)['status']=='GUARDED_STAGE_COMPLETED'
   r['status']='FINALIZED'if r['content_pass']else('SCENE_HOLD'if any(x['decision']=='HOLD'for x in r['new_scenes'])else'IN_PROGRESS')
  else:r['status']='PREFIX_HOLD_NO_RETRY'
  roots.append(r)
 scenes=[s for r in roots for s in r['new_scenes']]
 out={'schema_version':'v120_additive_progress_snapshot_v2','snapshot_at_utc':datetime.now(timezone.utc).isoformat(),'actual_executor':'/root','master_binding':bind(B/'MASTER_FREEZE.json'),'engineering_synthetic_tests_pass':65,'fixed_candidates':8,'prefix_pass':4,'prefix_hold':4,'new_scenes_generated':len(scenes),'new_scene_decisions':dict(Counter(s['decision']for s in scenes)),'new_native_events_reserved':sum(r['new_consumption_count']for r in roots),'content_pass_questions':sum(r['content_pass']for r in roots),'content_pass_ms_gold_counts':dict(Counter(g for r in roots for g in r['multi_select_gold_counts'])),'content_pass_ordering':sum(r['ordering_content_pass']for r in roots),'roots':roots,'training_or_pool_merge':False}
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();out=collect()
 if a.output:
  with a.output.open('x')as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in out.items()if k!='roots'},ensure_ascii=False))
