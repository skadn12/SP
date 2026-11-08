from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, sys
BATCH=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_newroots_ab_validation_v115')
REVIEWER='/root'
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def folder(root, scene): return BATCH/'roots'/('root'+root.rsplit('-',1)[-1])/'scenes'/scene.removeprefix('R02:').replace(':','_')
def start(root,scene):
 d=folder(root,scene); source=d/'CUMULATIVE_SOURCE.json'; s=read(source)
 context={'started_at_utc':now(),'source_sha256':digest(source)}
 p=Path('/tmp')/('v115-'+root.rsplit('-',1)[-1]+'-'+scene.replace(':','_')+'-read.json')
 with p.open('x') as f: json.dump(context,f)
 for t in s['turns']: print(t['turn_ref']+' | '+t['speaker']+': '+t['text'])
 print('DRAFT_NOTES',json.dumps({k:v for k,v in read(d/'EXPANSION_DRAFT.json').items() if k not in {'turns'}},ensure_ascii=False))
 if ':B:' in scene:
  r=next(r for r in read(BATCH/'SOURCE_RESERVATIONS.json')['allocation'] if r['root_id']==root and r['scene_ref']==scene)
  ep=Path(r['source_item_path']); ep=ep if ep.is_absolute() else BATCH/ep
  print('NATIVE_EVENT_FULL',json.dumps(read(ep),ensure_ascii=False))
  sel=BATCH/'selection'/('root'+root.rsplit('-',1)[-1]+'.json')
  selected = next(a for a in read(sel)['allocations'] if a['scene_ref']==scene)
  print('SELECTION',json.dumps({k:v for k,v in selected.items() if k != 'native_event'},ensure_ascii=False))
 print('READ_CONTEXT',str(p))
def save_review(root,scene,judgment):
 d=folder(root,scene); sp=d/'CUMULATIVE_SOURCE.json'; s=read(sp)
 cp=Path('/tmp')/('v115-'+root.rsplit('-',1)[-1]+'-'+scene.replace(':','_')+'-read.json'); c=read(cp)
 assert digest(sp)==c['source_sha256']
 end=now(); j=read(judgment)
 assert set(j['checks'])=={'time','location','actor_identity','plan_vs_completion','cross_scene_continuity','semantic_repetition','naturalness','event_fidelity'}
 r={'root_id':root,'scene_ref':scene,'source_sha256':digest(sp),'reviewer':REVIEWER,'review_role':'non_author_complete_cumulative_source_and_native_event_reviewer','backend':'CURRENT_CODEX_SESSION','exact_model_id':None,'external_semantic_certification':False,'started_at_utc':c['started_at_utc'],'finished_at_utc':end,'created_at_utc':end,'full_source_read':True,'turn_refs_read':[t['turn_ref'] for t in s['turns']],**j,'semantic_review_is_programmatic':False}
 if ':B:' in scene:
  a=next(r for r in read(BATCH/'SOURCE_RESERVATIONS.json')['allocation'] if r['root_id']==root and r['scene_ref']==scene)
  ep=Path(a['source_item_path']); ep=ep if ep.is_absolute() else BATCH/ep; e=read(ep)
  sel=BATCH/'selection'/('root'+root.rsplit('-',1)[-1]+'.json')
  r.update(native_event_full_read=True,native_event_path=str(ep),native_event_sha256=digest(ep),native_turn_ids_read=[t['turn_id'] for t in e['turns']],source_id=a['source_id'],selection_path=str(sel),selection_sha256=digest(sel))
 with (d/'SOURCE_REVIEW.json').open('x') as f: json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
 print(d/'SOURCE_REVIEW.json')
if __name__=='__main__':
 if sys.argv[1]=='read':start(sys.argv[2],sys.argv[3])
 elif sys.argv[1]=='save':save_review(sys.argv[2],sys.argv[3],sys.argv[4])
