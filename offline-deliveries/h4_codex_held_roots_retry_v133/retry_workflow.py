"""Append-only v133 retry records; semantic judgments belong to actual readers."""
from pathlib import Path
import sys,json,hashlib,datetime,argparse
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'scripts'))
import codex_sequential_ab as native
import codex_scene_payload as serializer
SCENES=native.SCENES
CHECKS=set(native.CHECKS)|{'quarantine_compliance'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def unit(rid):return B/'units'/('root'+str(rid).rsplit('-',1)[-1])
def cfg(rid):return read(unit(rid)/'RUN_CONFIG.json')
def ridfull(rid):return cfg(rid)['roots'][0]['root_id']
def guard(rid,who,role):
 c=cfg(rid);assert who==c[role],('ACTUAL_ASSIGNED_ROLE_REQUIRED',who,role)
 assert read(B/'MASTER_FREEZE.json')['status']=='FROZEN_BEFORE_RETRY_GENERATION'
 for x in read(B/'MASTER_FREEZE.json')['files']:assert sha(x['path'])==x['sha256'],('FROZEN_FILE_CHANGED',x['path'])
 if c['split']=='test':assert (B/'DEV_PHASE_CLOSED.json').exists(),'DEV_NOT_CLOSED'
 return c
def baseline(rid):return read(unit(rid)/'inputs/BASELINE.json')
def check_citations(obj,source):
 lookup={t['turn_ref']:t for t in source['turns']}
 def walk(x):
  if isinstance(x,dict):
   if {'turn_ref','speaker','quote'}<=x.keys():
    assert x['turn_ref'] in lookup,('CITATION_REF',x['turn_ref'])
    t=lookup[x['turn_ref']];assert x['speaker']==t['speaker'],'CITATION_SPEAKER'
    assert isinstance(x['quote'],str) and x['quote'] and x['quote'] in t['text'],('CITATION_QUOTE',x['turn_ref'])
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(obj)
def context(rid,who,purpose,scene=None):
 c=cfg(rid);role='author' if who==c['author'] else 'reviewer';guard(rid,who,role)
 u=unit(rid);files=[u/'inputs/HISTORICAL_CONTINUATION_CONSTRAINTS.json',B/'PROTOCOL_ZH.md']
 if scene:
  i=SCENES.index(scene)
  source_path=(u/'inputs/BASELINE.json' if i==0 else native.scene_folder(u,ridfull(rid),SCENES[i-1])/'CUMULATIVE_SOURCE.json') if purpose=='author' else native.scene_folder(u,ridfull(rid),scene)/'CUMULATIVE_SOURCE.json'
  files.extend([u/'SAFE_PLAN.json',u/'SAFE_PLAN_REVIEW.json',source_path])
  if ':B:' in scene:files.append(u/'inputs'/(scene.removeprefix('R02:').replace(':','_')+'_EVENT.json'))
 else:source_path=u/'inputs/BASELINE.json';files.append(source_path)
 source=read(source_path)
 text='\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source['turns'])
 for p in files:
  if p==source_path:continue
  if p.name.endswith('_EVENT.json'):
   event=read(p);text+='\n\nNATIVE EVENT '+event['dialogue_id']+'\n'+'\n'.join(f"[{t['turn_id']}] {t['speaker']}: {t['utterance']}" for t in event['turns'])
  else:text+='\n\n'+p.name+'\n'+p.read_text()
 folder=u/'reads'/((scene or 'baseline').replace(':','_')+'_'+purpose)
 if not (folder/'READ_CONTEXT.json').exists():
  folder.mkdir(parents=True,exist_ok=True)
  (folder/'FULL_INPUT.txt').write_text(text+'\n')
  save(folder/'READ_CONTEXT.json',{'actual_reader':who,'purpose':purpose,'scene_ref':scene,'started_at_utc':now(),'source_path':str(source_path),'source_sha256':sha(source_path),'turn_refs':[t['turn_ref'] for t in source['turns']],'inputs':[{'path':str(p),'sha256':sha(p)} for p in files]})
 else:
  old=read(folder/'READ_CONTEXT.json');assert old['actual_reader']==who and old['source_sha256']==sha(source_path),'READ_CONTEXT_CHANGED'
 return text

def save_plan(rid,who,p):
 guard(rid,who,'author');u=unit(rid);s=baseline(rid)
 assert (u/'reads/baseline/READ_CONTEXT.json').exists() or (u/'reads/baseline_plan_author/READ_CONTEXT.json').exists(),'AUTHOR_CONTEXT_REQUIRED'
 assert p['root_id']==ridfull(rid) and p['author']==who
 assert p['full_baseline_read'] is True and p['baseline_sha256']==sha(u/'inputs/BASELINE.json')
 assert p['baseline_conflicts_resolved'] is False
 assert len(p['a_scene_plan'])==3 and [x['scene_ref'] for x in p['a_scene_plan']]==list(SCENES[:3])
 assert p['quarantined_conflicts'] and p['forbidden_claims']
 assert all(x.get('citations') and x.get('goal') for x in p['a_scene_plan'])
 check_citations(p,s)
 save(u/'SAFE_PLAN.json',{**p,'created_at_utc':now()})
def review_plan(rid,who,p):
 guard(rid,who,'reviewer');u=unit(rid);s=baseline(rid)
 assert (u/'reads/baseline_plan_reviewer/READ_CONTEXT.json').exists(),'REVIEWER_CONTEXT_REQUIRED'
 assert p['reviewer']==who and p['full_baseline_read'] is True and p['root_id']==ridfull(rid)
 assert p['decision'] in ['READY','HOLD'] and p['reason'] and p.get('citations')
 assert p['baseline_conflicts_resolved'] is False
 check_citations(p,s)
 save(u/'SAFE_PLAN_REVIEW.json',{**p,'safe_plan_sha256':sha(u/'SAFE_PLAN.json'),'baseline_sha256':sha(u/'inputs/BASELINE.json'),'created_at_utc':now()})
def reserve(rid,who,p):
 guard(rid,who,'author');u=unit(rid)
 assert read(u/'SAFE_PLAN_REVIEW.json')['decision']=='READY','PLAN_NOT_READY'
 assert p['root_id']==ridfull(rid) and p['author']==who and p['full_selected_event_texts_read'] is True
 rows=p['selections'];assert [r['scene_ref'] for r in rows]==list(SCENES[3:])
 pool={f"multiwoz22:train:{e['dialogue_id']}":e for e in read(u/'inputs/EVENT_POOL.json')['events']}
 consumed=set(read(B/'evidence/SOURCE_CONSUMPTION_FREEZE.json')['consumed_source_ids'])
 for file in (B/'units').glob('root*/SOURCE_RESERVATIONS.json'):
  consumed.update(x['source_id'] for x in read(file)['allocation'])
 assert len({r['source_id'] for r in rows})==3
 for r in rows:
  assert r['source_id'] in pool and r['source_id'] not in consumed,('SOURCE_USED_OR_NOT_IN_POOL',r['source_id'])
  event=pool[r['source_id']]
  assert r['native_turn_ids_read']==[t['turn_id'] for t in event['turns']]
  assert r['compatibility_reason'] and r['main_purpose']
 save(u/'SELECTION.json',{**p,'created_at_utc':now()})
 allocation=[]
 for r in rows:
  event=pool[r['source_id']];dest=u/'inputs'/(r['scene_ref'].removeprefix('R02:').replace(':','_')+'_EVENT.json');save(dest,event)
  allocation.append({**r,'root_id':ridfull(rid),'source_item_path':str(dest),'source_item_sha256':sha(dest),'status':'PERMANENTLY_RESERVED_BEFORE_AUTHORING'})
 save(u/'SOURCE_RESERVATIONS.json',{'batch_id':B.name,'root_id':ridfull(rid),'allocation':allocation,'created_at_utc':now(),'permanent_including_failed_and_unexecuted_slots':True})
 save(u/'EXECUTION_INPUTS_FROZEN.json',{'created_at_utc':now(),'author':who,'files':[{'path':str(p),'sha256':sha(p)} for p in [u/'RUN_CONFIG.json',u/'SAFE_PLAN.json',u/'SAFE_PLAN_REVIEW.json',u/'SELECTION.json',u/'SOURCE_RESERVATIONS.json',u/'inputs/BASELINE.json',u/'inputs/HISTORICAL_CONTINUATION_CONSTRAINTS.json',u/'inputs/EVENT_POOL.json']+[Path(r['source_item_path']) for r in allocation]]})
def check_execution(rid):
 u=unit(rid)
 for r in read(u/'EXECUTION_INPUTS_FROZEN.json')['files']:assert sha(r['path'])==r['sha256'],('INPUT_CHANGED',r['path'])
def draft(rid,who,scene,p):
 guard(rid,who,'author');u=unit(rid);check_execution(rid)
 assert (u/'reads'/(scene.replace(':','_')+'_author')/'READ_CONTEXT.json').exists(),'AUTHOR_SCENE_READ_REQUIRED'
 assert all(not any('\u3400'<=c<='\u9fff' for c in t['text']) for t in p['messages']),'ENGLISH_ONLY'
 folder=native.scene_folder(u,ridfull(rid),scene);save(folder/'AUTHOR_PAYLOAD.json',p)
 serializer.write_new(u,ridfull(rid),scene,who,p)
 out=native.materialize_scene(u,ridfull(rid),scene)
 assert out['status']=='STRUCTURAL_PASS_ONLY',out
 save(folder/'AUTHOR_RECEIPT.json',{'actual_author':who,'created_at_utc':now(),'full_cumulative_source_read':True,'full_selected_native_event_read':':B:' in scene,'context_path':str(u/'reads'/(scene.replace(':','_')+'_author')/'READ_CONTEXT.json'),'payload_sha256':sha(folder/'AUTHOR_PAYLOAD.json')})
 return out

def review(rid,who,scene,p):
 guard(rid,who,'reviewer');u=unit(rid);check_execution(rid)
 folder=native.scene_folder(u,ridfull(rid),scene);sp=folder/'CUMULATIVE_SOURCE.json';source=read(sp)
 assert (u/'reads'/(scene.replace(':','_')+'_reviewer')/'READ_CONTEXT.json').exists(),'REVIEWER_SCENE_READ_REQUIRED'
 assert p['reviewer']==who and p['root_id']==ridfull(rid) and p['scene_ref']==scene
 assert p['full_source_read'] is True and p['baseline_conflicts_resolved'] is False
 assert p['full_event_read']==(':B:' in scene)
 assert p['decision'] in ['READY','HOLD'] and isinstance(p['findings'],list)
 assert CHECKS<=set(p['checks'])
 for name in CHECKS:
  x=p['checks'][name];assert x['status'] in ['PASS','WARNING','PASS_WITH_WARNING','FAIL'] and x['reason'] and x['citations'],('CHECK_INCOMPLETE',name)
 if p['decision']=='READY':assert not any(x['status']=='FAIL' for x in p['checks'].values())
 check_citations(p,source)
 value={**p,'source_sha256':sha(sp),'turn_refs_read':[t['turn_ref'] for t in source['turns']],'created_at_utc':now(),'read_context_path':str(u/'reads'/(scene.replace(':','_')+'_reviewer')/'READ_CONTEXT.json')}
 # All schema/evidence checks above precede immutable acceptance. Invalid drafts remain caller artifacts.
 save(folder/'SOURCE_REVIEW.json',value)
 state=native.inspect_root(u,cfg(rid),ridfull(rid));actual=state['scenes'][SCENES.index(scene)]['status']
 assert actual==('READY' if p['decision']=='READY' else 'REVIEW_HOLD'),state
 if state['full_expansion_complete']:
  src=native.scene_folder(u,ridfull(rid),SCENES[-1])/'CUMULATIVE_SOURCE.json';native.save(native.root_folder(u,ridfull(rid))/'CUMULATIVE_SOURCE.json',read(src))
 return state

def status(rid):
 check_execution(rid)
 return native.inspect_root(unit(rid),cfg(rid),ridfull(rid),record_reviews=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['context','save-plan','review-plan','reserve','draft','review','status']);p.add_argument('--root',required=True);p.add_argument('--who');p.add_argument('--purpose');p.add_argument('--scene');p.add_argument('--payload');a=p.parse_args()
 if a.action=='context':print(context(a.root,a.who,a.purpose,a.scene))
 elif a.action=='status':print(json.dumps(status(a.root),ensure_ascii=False,indent=2))
 else:
  f={'save-plan':save_plan,'review-plan':review_plan,'reserve':reserve,'draft':draft,'review':review}[a.action]
  payload=read(a.payload)
  result=f(a.root,a.who,a.scene,payload) if a.action in ['draft','review'] else f(a.root,a.who,payload)
  print(json.dumps(result or {'status':'SAVED'},ensure_ascii=False,indent=2))
