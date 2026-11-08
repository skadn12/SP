from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,sys
BATCH=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_newroots_ab_validation_v115')
REVIEWER='/root/v113_setup'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def context_path(root,scene):return Path('/tmp')/('v115_'+root+'_'+scene.removeprefix('R02:').replace(':','_')+'_review_start.json')
def folder(root,scene):return BATCH/'roots'/('root'+root)/'scenes'/scene.removeprefix('R02:').replace(':','_')
def begin(root,scene):
 f=folder(root,scene);p=f/'CUMULATIVE_SOURCE.json';sel=BATCH/'selection'/('root'+root+'.json');s=read(p)
 c={'started_at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(p),'selection_sha256':sha(sel)}
 with context_path(root,scene).open('x') as h:json.dump(c,h)
 print('CUMULATIVE_FULL',len(s['turns']))
 for t in s['turns']:print(t['turn_ref']+' | '+t['speaker']+': '+t['text'])
 selection=read(sel)
 print('SELECTION_LIMITATIONS',json.dumps(selection['inherited_limitations'],ensure_ascii=False))
 for a in selection['allocations']:
  print('ALLOCATION',json.dumps({k:v for k,v in a.items() if k not in ['native_event','baseline_continuity_constraints']},ensure_ascii=False))
  if a==selection['allocations'][0]:print('BASELINE_CONSTRAINTS',json.dumps(a['baseline_continuity_constraints'],ensure_ascii=False))
  if ':B:' in scene and a['scene_ref']==scene:
   epath=Path(a['source_item_path']);event=read(epath)
   assert event==a['native_event'];assert sha(epath)==a['source_item_sha256']
   print('NATIVE_EVENT_METADATA',json.dumps({k:v for k,v in event.items() if k!='turns'},ensure_ascii=False))
   for t in event['turns']:print('NATIVE',t['turn_id'],t['speaker'],t['utterance'])
 print('DRAFT_METADATA',json.dumps({k:v for k,v in read(f/'EXPANSION_DRAFT.json').items() if k!='turns'},ensure_ascii=False))
def finish(root,scene):
 f=folder(root,scene);p=f/'CUMULATIVE_SOURCE.json';s=read(p);sel=BATCH/'selection'/('root'+root+'.json');c=read(context_path(root,scene));assert sha(p)==c['source_sha256'] and sha(sel)==c['selection_sha256'];j=json.load(sys.stdin);assert set(j['checks'])=={'time','location','actor_identity','plan_vs_completion','cross_scene_continuity','semantic_repetition','naturalness','event_fidelity'};end=datetime.now(timezone.utc).isoformat()
 r={'root_id':'cc-train-episode-'+root,'scene_ref':scene,'source_sha256':sha(p),'reviewer':REVIEWER,'review_role':'non_author_complete_cumulative_source_and_selected_native_utterance_reviewer','backend':'CURRENT_CODEX_SESSION','exact_model_id':None,'external_semantic_certification':False,'started_at_utc':c['started_at_utc'],'finished_at_utc':end,'created_at_utc':end,'full_source_read':True,'turn_refs_read':[t['turn_ref'] for t in s['turns']],'semantic_review_is_programmatic':False,'selection_path':str(sel),'selection_sha256':sha(sel),'full_selection_semantic_fields_read':True,**j}
 if ':B:' in scene:
  a=next(a for a in read(sel)['allocations'] if a['scene_ref']==scene);ep=Path(a['source_item_path']);e=read(ep);r.update(native_event_full_read=True,native_event_read_scope='All native utterances and event metadata; service frames remain bound in the exact hashed native event.',native_event_path=str(ep),native_event_sha256=sha(ep),native_turn_ids_read=[t['turn_id'] for t in e['turns']],source_id=a['source_id'])
 else:r.update(all_selected_native_utterances_read=True,selected_native_events_read=[{'source_id':a['source_id'],'native_turn_ids_read':[t['turn_id'] for t in a['native_event']['turns']]} for a in read(sel)['allocations']])
 with (f/'SOURCE_REVIEW.json').open('x') as h:json.dump(r,h,ensure_ascii=False,indent=2);h.write('\n')
 sys.path.insert(0,str(BATCH.parent.parent/'scripts'));from codex_sequential_ab import inspect_root,load_config
 state=inspect_root(BATCH,load_config(BATCH),'cc-train-episode-'+root);row=next(x for x in state['scenes'] if x['scene_ref']==scene);assert row['status']==('READY' if j['decision']=='READY' else 'REVIEW_HOLD'),state
 print(json.dumps({'root':root,'scene':scene,'decision':j['decision'],'cumulative_turns':len(s['turns']),'ready_scene_count':state['ready_scene_count'],'root_stopped':state['root_stopped'],'review_sha256':sha(f/'SOURCE_REVIEW.json')},ensure_ascii=False))
if __name__=='__main__':
 (begin if sys.argv[1]=='read' else finish)(sys.argv[2],sys.argv[3])
