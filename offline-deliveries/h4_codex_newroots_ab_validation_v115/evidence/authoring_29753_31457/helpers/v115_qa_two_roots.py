import json,hashlib,sys,random
from pathlib import Path
from datetime import datetime,timezone
P=Path('/root/autodl-tmp/gxy-selfplay');B=P/'offline-deliveries/h4_codex_newroots_ab_validation_v115';AUTHOR='/root/v113_question_recheck'
sys.path.insert(0,str(P/'scripts'))
from codex_sequential_ab import inspect_root,load_config,SCENES,scene_folder
from codex_sequential_questions import source_ready,publish_assets

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
 p=Path(p)
 with p.open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n')
def rootfolder(n):return B/'roots'/f'root{n}'
def marker(n,suffix):return Path('/tmp')/f'v115_qa_author_{n}_{suffix}.json'
def prepare(n):
 rid=f'cc-train-episode-{n}';cfg=load_config(B);st=inspect_root(B,cfg,rid);assert st['full_expansion_complete']
 folder=rootfolder(n);last=scene_folder(B,rid,SCENES[-1])
 for name in ['CUMULATIVE_SOURCE.json','SOURCE_REVIEW.json']:
  with (folder/name).open('xb') as f:f.write((last/name).read_bytes())
 save(folder/'FINAL_SOURCE_BINDING.json',{'root_id':rid,'created_at_utc':now(),'status':'FULL_SIX_READY_ROOT_BYTES_MATCH_FINAL_B3','full_expansion_complete':True,'scene_sequence':list(SCENES),'scene_states':st['scenes'],'source_sha256':sha(folder/'CUMULATIVE_SOURCE.json'),'review_sha256':sha(folder/'SOURCE_REVIEW.json'),'copied_from':str(last),'training_started':False})
 source=source_ready(folder)
 save(marker(n,'start'),{'started_at_utc':now(),'source_sha256':sha(folder/'CUMULATIVE_SOURCE.json')})
 for t in source['turns']:print(t['turn_ref'],t['speaker']+':',t['text'])
 print('SOURCE_SHA256',sha(folder/'CUMULATIVE_SOURCE.json'))
def read_done(n):save(marker(n,'read_done'),{'full_source_read_finished_at_utc':now()})
def anchors(n,slots):
 folder=rootfolder(n);source=source_ready(folder);turns={t['turn_ref']:t for t in source['turns']}
 assert len(slots)==2 and len(set(sum(slots,[])))==12
 value={'root_id':f'cc-train-episode-{n}','author':AUTHOR,'created_at_utc':now(),'source_sha256':sha(folder/'CUMULATIVE_SOURCE.json'),'slots':[]}
 for i,refs in enumerate(slots,1):
  assert len(refs)==6 and {turns[r]['scene_ref'] for r in refs}==set(SCENES)
  value['slots'].append({'slot_id':f'{i:02d}','anchors':[{'anchor_id':f'K{j:02d}','turn_ref':r,'speaker':turns[r]['speaker'],'full_turn_quote':turns[r]['text']} for j,r in enumerate(refs,1)]})
 save(folder/'ORDERING_ANCHOR_PLAN.json',value)
 print('ANCHOR_PLAN_SAVED',n,sha(folder/'ORDERING_ANCHOR_PLAN.json'))
def qa_save(n,facts,ordered_items):
 folder=rootfolder(n);rid=f'cc-train-episode-{n}';source=source_ready(folder);plan=read(folder/'ORDERING_ANCHOR_PLAN.json');cfg=load_config(B)
 assert len(facts)==6 and len(ordered_items)==2
 draft={'root_id':rid,'author':AUTHOR,'facts':facts,'stem':'Within these six new scenes, '+', '.join(SCENES)+', select all statements supported by the dialogue. Treat service details as what the adviser reportedly supplied; preserve who acted and whether an action is completed or planned.','ordering':[]}
 for slot,items in zip(plan['slots'],ordered_items):
  assert len(items)==6
  byid={a['anchor_id']:a for a in slot['anchors']}
  assert {x['anchor_id'] for x in items}==set(byid)
  for x in items:assert x['support_quote'] in byid[x['anchor_id']]['full_turn_quote'] and x['support_quote']
  items=[dict(x) for x in items];random.Random(cfg['seed_namespace']+rid+slot['slot_id']+'qa_display').shuffle(items)
  for i,item in enumerate(items,1):item['item_id']=f'I{i}'
  draft['ordering'].append({'slot_id':slot['slot_id'],'items':items})
 save(folder/'QA_DRAFT.json',draft)
 s=read(marker(n,'start'));rd=read(marker(n,'read_done'));finish=now()
 assert s['started_at_utc']<=rd['full_source_read_finished_at_utc']<=plan['created_at_utc']<=finish
 save(folder/'QA_AUTHOR_RECEIPT.json',{'root_id':rid,'author':AUTHOR,'role':'DIALOGUE_AND_QUESTION_AUTHOR; INDEPENDENT_SOURCE_AND_QA_REVIEWER_IS_v113_setup','attempt':1,**s,**rd,'finished_at_utc':finish,'timing_scope':'Complete final source reread, frozen anchor selection and one-time question authoring; may include tool/scheduling overhead. Not pure inference service latency.','full_source_read':True,'turn_refs_read':[t['turn_ref']for t in source['turns']],'source_path':str(folder/'CUMULATIVE_SOURCE.json'),'source_review_sha256':sha(folder/'SOURCE_REVIEW.json'),'anchor_plan_created_at_utc':plan['created_at_utc'],'anchor_plan_sha256':sha(folder/'ORDERING_ANCHOR_PLAN.json'),'qa_draft_sha256':sha(folder/'QA_DRAFT.json'),'anchor_plan_saved_before_qa_draft':True,'anchors_changed_after_plan_save':False,'ordering_slots':2,'anchors_per_slot':6,'one_anchor_per_scene_per_slot':True,'twelve_distinct_anchor_turns':True,'display_order_policy':'One deterministic shuffle per slot using frozen seed_namespace+root_id+slot_id+qa_display; no seed search.','fact_scene_coverage':{f['fact_id']:next(t['scene_ref']for t in source['turns']if t['turn_ref']==f['source_turn_ref'])for f in facts},'same_inherited_model_and_shared_background':True,'exact_model_id':None,'usage_tokens':None,'billed_cost':None,'semantic_repairs':0,'current_solver_results_read':False,'training_started':False,'external_semantic_certification':False})
 publish_assets(B)
 print('QA_AND_PUBLIC_ASSETS_SAVED',n,finish)
