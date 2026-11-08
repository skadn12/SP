import json,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
P=Path('/root/autodl-tmp/gxy-selfplay');B=P/'offline-deliveries/h4_codex_newroots_ab_validation_v115'
sys.path.insert(0,str(P/'scripts'))
from codex_scene_payload import write_new
from codex_sequential_ab import materialize_scene,load_config,inspect_root,SCENES,scene_folder
AUTHOR='/root/v113_question_recheck'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n')
def marker(root,scene,suffix='start'):return Path('/tmp')/f'v115_author_{root}_{scene.replace(":","_")}_{suffix}.json'
def start(root,scene):
 cfg=load_config(B);st=inspect_root(B,cfg,f'cc-train-episode-{root}');idx=SCENES.index(scene)
 assert st['scenes'][idx]['status']=='NOT_STARTED' and (idx==0 or st['scenes'][idx-1]['status']=='READY')
 save(marker(root,scene),{'started_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Full cumulative reread, composition and first-save; may include scheduling/wait overhead.','root_id':f'cc-train-episode-{root}','scene_ref':scene})
def read_done(root,scene):save(marker(root,scene,'read_done'),{'full_source_read_finished_at_utc':datetime.now(timezone.utc).isoformat()})
def author_save(root,scene,messages,state_notes,event_mapping=None):
 rid=f'cc-train-episode-{root}';cfg=load_config(B);idx=SCENES.index(scene)
 for v in json.loads((B/'EXECUTION_INPUTS_FROZEN.json').read_text())['files']:assert sha(v['path'])==v['sha256'],v['path']
 source=B/'inputs'/f'root{root}_BASELINE.json' if idx==0 else scene_folder(B,rid,SCENES[idx-1])/'CUMULATIVE_SOURCE.json'
 src=json.loads(source.read_text());started=json.loads(marker(root,scene).read_text());readmark=json.loads(marker(root,scene,'read_done').read_text());now=datetime.now(timezone.utc);assert datetime.fromisoformat(started['started_at_utc'])<=datetime.fromisoformat(readmark['full_source_read_finished_at_utc'])<=now
 payload={'messages':[{'speaker':s,'text':t}for s,t in messages],'state_notes':state_notes}
 if event_mapping is not None:payload['event_mapping']=event_mapping
 folder=scene_folder(B,rid,scene);save(folder/'AUTHOR_PAYLOAD.json',payload);draft=write_new(B,rid,scene,AUTHOR,payload);materialized=materialize_scene(B,rid,scene)
 readfiles=[{'path':str(source),'sha256':sha(source),'full_read':True},{'path':str(B/'evidence/CONTENT_CONTRACT.md'),'sha256':sha(B/'evidence/CONTENT_CONTRACT.md'),'full_read':True}]
 if ':B:'in scene:
  ep=B/'inputs'/f'root{root}_B_S{int(scene[-2:]):02d}_EVENT.json';ev=json.loads(ep.read_text());readfiles.append({'path':str(ep),'sha256':sha(ep),'full_native_utterances_read':True,'native_turn_ids_read':[t['turn_id']for t in ev['turns']]})
 receipt={'root_id':rid,'scene_ref':scene,'author':AUTHOR,**started,**readmark,'finished_at_utc':datetime.now(timezone.utc).isoformat(),'timing_scope':'Measured actual start before full cumulative reread through materialization and author receipt. Not pure generation service latency or billed usage.','full_source_read':True,'turn_refs_read':[t['turn_ref']for t in src['turns']],'input_source_sha256':sha(source),'read_files':readfiles,'rule_files_machine_validated':[{'path':str(B/'RUN_CONFIG.json'),'sha256':sha(B/'RUN_CONFIG.json')},{'path':str(B/'SOURCE_RESERVATIONS.json'),'sha256':sha(B/'SOURCE_RESERVATIONS.json')}],'payload_sha256':sha(folder/'AUTHOR_PAYLOAD.json'),'draft_sha256':sha(draft),'serializer_path':str(P/'scripts/codex_scene_payload.py'),'serializer_sha256':sha(P/'scripts/codex_scene_payload.py'),'attempt':1,'semantic_repairs':0,'exact_model_id':None,'usage_tokens':None,'billed_cost':None,'sampling_parameters':None,'training_started':False}
 save(folder/'AUTHOR_RECEIPT.json',receipt)
 print(json.dumps({'root':rid,'scene':scene,'turns':len(messages),'materialize':materialized,'draft':str(draft),'started':receipt['started_at_utc'],'finished':receipt['finished_at_utc']},ensure_ascii=False))
