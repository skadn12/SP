from pathlib import Path
from datetime import datetime, timezone
import json,hashlib,sys
sys.path.insert(0,'/root/autodl-tmp/gxy-selfplay/scripts')
from codex_scene_payload import write_new
from codex_sequential_ab import SCENES,scene_folder,load_config,load_baseline,inspect_root,reservation_for,materialize_scene
BATCH=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_newroots_ab_validation_v115')
AUTHOR='/root/v113_14304_recheck'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exclusive(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def author_save(short,scene,messages,state_notes,event_mapping):
 rid=f'cc-train-episode-{short}'
 tag=scene.removeprefix('R02:').replace(':','_')
 startp=Path(f'/tmp/v115_starts_{short}_{tag}.json')
 readp=Path(f'/tmp/v115_read_end_{short}_{tag}.json')
 start=json.loads(startp.read_text());readmark=json.loads(readp.read_text())
 finish=datetime.now(timezone.utc).isoformat()
 assert datetime.fromisoformat(start['started_at_utc'])<=datetime.fromisoformat(readmark['finished_read_at_utc'])<=datetime.fromisoformat(finish)
 frozen=json.loads((BATCH/'EXECUTION_INPUTS_FROZEN.json').read_text())
 assert all(sha(row['path'])==row['sha256'] for row in frozen['files'])
 cfg=load_config(BATCH)
 state=inspect_root(BATCH,cfg,rid)
 idx=SCENES.index(scene)
 assert state['scenes'][idx]['status']=='NOT_STARTED' and not state['root_stopped']
 inp=load_baseline(BATCH,cfg,rid)[0] if idx==0 else scene_folder(BATCH,rid,SCENES[idx-1])/'CUMULATIVE_SOURCE.json'
 source=json.loads(inp.read_text())
 assert len(source['turns'])==readmark['turn_count']
 reservation=reservation_for(BATCH,cfg,rid,scene)
 readfiles=[{'path':str(inp),'sha256':sha(inp),'full_turn_refs':[t['turn_ref'] for t in source['turns']]}]
 if reservation:
  ep=Path(reservation['source_item_path']);e=json.loads(ep.read_text())
  readfiles.append({'path':str(ep),'sha256':sha(ep),'full_native_event_read':True})
 payload={'messages':[{'speaker':sp,'text':txt} for sp,txt in messages],'state_notes':state_notes,'event_mapping':event_mapping}
 folder=scene_folder(BATCH,rid,scene)
 exclusive(folder/'AUTHOR_PAYLOAD.json',payload)
 draft=write_new(BATCH,rid,scene,AUTHOR,payload)
 result=materialize_scene(BATCH,rid,scene)
 receipt={'root_id':rid,'scene_ref':scene,'author':AUTHOR,'attempt':1,'started_at_utc':start['started_at_utc'],'finished_at_utc':finish,'input_read_start_at_utc':start['started_at_utc'],'input_loaded_at_utc':readmark['finished_read_at_utc'],'artifact_saved_at_utc':datetime.now(timezone.utc).isoformat(),'timing_scope':'Current scene-specific complete source reread, semantic reading, and composition; excludes earlier general read-only preparation. input_loaded timestamp is end of source text loading/printing, not a claim of completed human-like semantic reading.','full_source_read':True,'turn_refs_read':[t['turn_ref'] for t in source['turns']],'input_files_read':readfiles,'input_source_path':str(inp),'input_source_sha256':sha(inp),'execution_inputs_frozen_sha256':sha(BATCH/'EXECUTION_INPUTS_FROZEN.json'),'draft_sha256':sha(draft),'author_payload_sha256':sha(folder/'AUTHOR_PAYLOAD.json'),'serializer_path':'/root/autodl-tmp/gxy-selfplay/scripts/codex_scene_payload.py','serializer_sha256':sha('/root/autodl-tmp/gxy-selfplay/scripts/codex_scene_payload.py'),'author_helper_sha256':sha(__file__),'source_id':reservation['source_id'] if reservation else None,'materialization_result':result,'semantic_repair':False,'retries':0,'later_scene_pre_authored':False,'current_solver_outputs_read':False}
 exclusive(folder/'AUTHOR_RECEIPT.json',receipt)
 print(json.dumps(result,ensure_ascii=False))
