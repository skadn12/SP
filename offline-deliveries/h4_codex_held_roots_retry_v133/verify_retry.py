"""Read-only mechanical verification, including all originally held roots."""
from pathlib import Path
import json,hashlib,sys
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
import retry_workflow as w

def audit(split=None):
 errors=[];checks=0;rows=[];used=set()
 def check(ok,why):
  nonlocal checks
  checks+=1
  if not ok:errors.append(why)
 freeze=w.read(B/'MASTER_FREEZE.json')
 for x in freeze['files']:check(w.sha(x['path'])==x['sha256'],'frozen changed '+x['path'])
 for x in w.read(B/'evidence/SEALED_DELIVERIES_BEFORE.json')['sealed_manifests']:check(w.sha(x['path'])==x['sha256'],'sealed manifest changed '+x['path'])
 for r in w.read(B/'ROOT_MANIFEST.json')['roots']:
  if split and r['split']!=split:continue
  rid=r['root_id'];u=w.unit(rid);base=w.baseline(rid)
  for x in r['input_bindings']:
   check(w.sha(x['original_path'])==x['sha256'],'original input changed '+rid)
   check(w.sha(x['copied_path'])==x['sha256'],'copied input changed '+rid)
  check(w.read(u/'inputs/HISTORICAL_CONTINUATION_CONSTRAINTS.json')['decision']=='HOLD','legacy HOLD not preserved '+rid)
  if not (u/'EXECUTION_INPUTS_FROZEN.json').exists():
   rows.append({'root_id':rid,'split':r['split'],'status':'NOT_EXECUTION_READY','ready_scenes':0});continue
  w.check_execution(rid)
  state=w.status(rid)
  for event in w.read(u/'SOURCE_RESERVATIONS.json')['allocation']:
   check(event['source_id'] not in used,'event duplicate '+event['source_id']);used.add(event['source_id'])
  previous=base['turns'];new=[]
  for s in state['scenes']:
   d=Path(s['folder'])
   if not (d/'CUMULATIVE_SOURCE.json').exists():continue
   source=w.read(d/'CUMULATIVE_SOURCE.json');draft=w.read(d/'EXPANSION_DRAFT.json')
   check(source['turns']==previous+draft['turns'],'prefix/draft mismatch '+rid+s['scene_ref'])
   check(source['turns'][:len(base['turns'])]==base['turns'],'baseline prefix changed '+rid)
   check(12<=len(draft['turns'])<=16,'turn count '+rid+s['scene_ref'])
   allowed={t['speaker'] for t in base['turns']}
   check(all(t['speaker'] in allowed for t in draft['turns']),'new speaker '+rid+s['scene_ref'])
   new.extend(draft['turns']);previous=source['turns']
   if (d/'SOURCE_REVIEW.json').exists():
    review=w.read(d/'SOURCE_REVIEW.json');w.check_citations(review,source)
    check(review['reviewer']==r['reviewer'] and review['reviewer']!=draft['author'],'review identity '+rid+s['scene_ref'])
    check(review['baseline_conflicts_resolved'] is False,'false conflict resolution '+rid+s['scene_ref'])
    check(w.CHECKS<=set(review['checks']),'checks coverage '+rid+s['scene_ref'])
    check(review['turn_refs_read']==[t['turn_ref'] for t in source['turns']],'full source coverage '+rid+s['scene_ref'])
    check(all(v.get('citations') for v in review['checks'].values()),'missing citations '+rid+s['scene_ref'])
  check(len({t['turn_ref'] for t in previous})==len(previous),'duplicate turn refs '+rid)
  if state['full_expansion_complete']:
   check(w.read(w.native.root_folder(u,rid)/'CUMULATIVE_SOURCE.json')['turns']==previous,'final source mismatch '+rid)
  rows.append({'root_id':rid,'split':r['split'],'status':'EXPANSION_COMPLETE' if state['full_expansion_complete'] else ('EXPANSION_STOPPED' if state['root_stopped'] else 'IN_PROGRESS'),'ready_scenes':state['ready_scene_count'],'baseline_turns':len(base['turns']),'new_turns':len(new),'total_turns':len(previous),'all_original_turns_preserved':previous[:len(base['turns'])]==base['turns'],'baseline_conflicts_resolved':False})
 return {'status':'PASS' if not errors else 'FAIL','checks':checks,'errors':errors,'split':split or 'all','fixed_roots':len(rows),'complete_roots':sum(r['status']=='EXPANSION_COMPLETE' for r in rows),'ready_scenes':sum(r['ready_scenes'] for r in rows),'rows':rows,'question_generation':False,'solver_run':False,'training_run':False,'current_evaluation_pool_modified':False}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test']);p.add_argument('--output');a=p.parse_args();x=audit(a.split)
 if a.output:w.save(a.output,x)
 print(json.dumps(x,ensure_ascii=False,indent=2))
 if x['errors']:raise SystemExit(1)
