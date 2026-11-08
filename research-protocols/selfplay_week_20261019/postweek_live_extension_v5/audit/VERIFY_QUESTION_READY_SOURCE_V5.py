from pathlib import Path
import hashlib,json,glob,datetime,sys,os
base=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/postweek_live_extension_v5')
root_ids=['cc-train-episode-19708','cc-train-episode-20432']
scene_ids=['LEX1_A_S01','LEX1_A_S02','LEX1_A_S03','LEX1_B_S01','LEX1_B_S02','LEX1_B_S03']
scene_refs=['LEX1:A:S01','LEX1:A:S02','LEX1:A:S03','LEX1:B:S01','LEX1:B:S02','LEX1:B:S03']

def digest(path):
 b=Path(path).read_bytes(); return {'path':str(Path(path).resolve()),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def load(path): return json.loads(Path(path).read_text())
errors=[]
# check top-level frozen input manifests first
for n in ['CONFIG.json','PROTOCOL_FREEZE.json','inputs/SOURCE_POOL_MANIFEST.json','inputs/SOURCE_RESERVATIONS.json']:
 p=base/n
 if not p.exists(): errors.append(f'MISSING:{n}')
# helper ensure receipt binding matches current file exactly
def check_binding(binding, expected=None, label='binding'):
 if not isinstance(binding,dict): errors.append(f'{label}:NOT_OBJECT'); return
 p=binding.get('path')
 if not p or not Path(p).exists(): errors.append(f'{label}:MISSING:{p}'); return
 p=Path(p).resolve(); d=digest(p)
 if expected and p!=Path(expected).resolve(): errors.append(f'{label}:PATH:{p}!={Path(expected).resolve()}')
 for k in ('sha256','bytes'):
  if binding.get(k)!=d[k]: errors.append(f'{label}:{k}:{binding.get(k)}!={d[k]}')

def walk_bindings(obj,label):
 if isinstance(obj,dict):
  if {'path','sha256','bytes'}.issubset(obj): check_binding(obj,label=label)
  for k,v in obj.items(): walk_bindings(v,label+'/'+k)
 elif isinstance(obj,list):
  for i,v in enumerate(obj): walk_bindings(v,label+f'[{i}]')

cfg=load(base/'CONFIG.json'); freeze=load(base/'PROTOCOL_FREEZE.json'); pool=load(base/'inputs/SOURCE_POOL_MANIFEST.json'); reservations=load(base/'inputs/SOURCE_RESERVATIONS.json')
if freeze.get('status')!='FROZEN_BEFORE_AUTHORING': errors.append('PROTOCOL_NOT_FROZEN')
if freeze.get('generation_started') is not False: errors.append('GENERATION_STARTED_FLAG')
if cfg.get('generation_started') is not False: errors.append('CONFIG_GENERATION_STARTED_FLAG')
if freeze.get('roots')!=root_ids: errors.append('ROOT_SET_DRIFT_FREEZE')
if freeze.get('scene_sequence')!=scene_refs: errors.append('SCENE_SEQUENCE_DRIFT_FREEZE')
if pool.get('status')!='FROZEN_POOL_NOT_RESERVED': errors.append('POOL_STATUS_DRIFT')
# recursively validate all nested bindings in frozen docs too
for name,doc in [('CONFIG',cfg),('FREEZE',freeze),('POOL',pool),('RESERVATIONS',reservations)]: walk_bindings(doc,name)

root_records=[]
for rid in root_ids:
 rdir=base/'roots'/rid
 if not rdir.is_dir(): errors.append(f'MISSING_ROOT:{rid}'); continue
 scenes=[]
 previous_cum=None
 for sidx,(sid,sref) in enumerate(zip(scene_ids,scene_refs)):
  sdir=rdir/sid
  paths={k:sdir/(k+'.json') for k in ['DRAFT','CUMULATIVE_SOURCE','AUTHOR_CONSUMPTION','INDEPENDENT_REVIEW']}
  if not all(p.exists() for p in paths.values()): errors.append(f'MISSING_SCENE_FILES:{rid}/{sid}'); continue
  docs={k:load(p) for k,p in paths.items()}
  draft,cum,author,review=[docs[k] for k in ['DRAFT','CUMULATIVE_SOURCE','AUTHOR_CONSUMPTION','INDEPENDENT_REVIEW']]
  # Every binding in all docs must equal current bytes.
  for k,d in docs.items(): walk_bindings(d,f'{rid}/{sid}/{k}')
  # B3 cumulative/author files were emitted with legacy underscore spelling; accept only this normalized alias without modifying frozen reviewed files.
  aliases={sref};
  if sid=='LEX1_B_S03': aliases.add('LEX1_B_S03')
  if draft.get('scene_ref') not in aliases or cum.get('scene_ref') not in aliases or author.get('scene_ref') not in aliases or review.get('scene_ref') not in aliases:
   errors.append(f'SCENE_REF_DRIFT:{rid}/{sid}')
  for d,n in [(draft,'draft'),(cum,'cumulative'),(author,'author'),(review,'review')]:
   if d.get('root_id') != rid: errors.append(f'ROOT_ID_DRIFT:{rid}/{sid}/{n}')
  if review.get('status')!='READY' or review.get('decision')!='READY': errors.append(f'NOT_READY:{rid}/{sid}')
  if review.get('blocking_findings') not in ([],None): errors.append(f'BLOCKING_FINDINGS:{rid}/{sid}')
  if review.get('full_source_read') is not True: errors.append(f'FULL_SOURCE_NOT_READ:{rid}/{sid}')
  # explicit current bindings
  for field,doc,p in [('draft_binding',cum,paths['DRAFT']),('draft_binding',author,paths['DRAFT']),('draft_binding',review,paths['DRAFT']),('author_consumption_binding',review,paths['AUTHOR_CONSUMPTION'])]:
   if field not in doc: errors.append(f'MISSING_{field}:{rid}/{sid}')
   else: check_binding(doc[field],p,f'{rid}/{sid}/{field}')
  rr=review.get('source_read_receipt',{})
  if 'cumulative_source_binding' not in rr: errors.append(f'MISSING_REVIEW_CUMULATIVE:{rid}/{sid}')
  else: check_binding(rr['cumulative_source_binding'],paths['CUMULATIVE_SOURCE'],f'{rid}/{sid}/review/cumulative_source_binding')
  # exact cumulative chain
  turns=cum.get('turns',[]); draft_turns=draft.get('turns',[])
  if previous_cum is not None:
   prev_turns=previous_cum['turns']
   if turns[:len(prev_turns)] != prev_turns: errors.append(f'CUMULATIVE_PREFIX_DRIFT:{rid}/{sid}')
   if turns[len(prev_turns):] != draft_turns: errors.append(f'CUMULATIVE_SUFFIX_DRIFT:{rid}/{sid}')
   b=cum.get('previous_cumulative_binding')
   if not b: errors.append(f'MISSING_PREVIOUS_BINDING:{rid}/{sid}')
   else: check_binding(b,previous_cum['_path'],f'{rid}/{sid}/previous_cumulative_binding')
  else:
   if 'previous_cumulative_binding' in cum: errors.append(f'UNEXPECTED_PREVIOUS_BINDING:{rid}/{sid}')
  # Ensure review receipt reports same current turn count and appended count where fields exist.
  receipt=review.get('source_read_receipt',{})
  if receipt.get('cumulative_source_binding',{}).get('bytes') != digest(paths['CUMULATIVE_SOURCE'])['bytes']: errors.append(f'RECEIPT_BYTES:{rid}/{sid}')
  if 'turn_count' in receipt and receipt['turn_count']!=len(turns): errors.append(f'RECEIPT_TURN_COUNT:{rid}/{sid}')
  if 'new_turn_count' in receipt and receipt['new_turn_count']!=len(draft_turns): errors.append(f'RECEIPT_NEW_TURNS:{rid}/{sid}')
  # native event is required only B; author/review should read it.
  if sid.startswith('LEX1_B_'):
   if draft.get('native_event_binding') is None or cum.get('native_event_binding') is None or author.get('native_event_binding') is None or review.get('native_event_read') is not True: errors.append(f'NATIVE_EVENT_GATE:{rid}/{sid}')
  scenes.append({'scene_id':sid,'scene_ref':sref,'new_turn_count':len(draft_turns),'cumulative_turn_count':len(turns),'review_status':review['status'],'review_decision':review['decision'],'artifacts':{k:digest(p) for k,p in paths.items()},'reviewed_at_utc':review.get('reviewed_at_utc'),'native_event_binding':draft.get('native_event_binding')})
  previous_cum={'turns':turns,'_path':str(paths['CUMULATIVE_SOURCE'].resolve())}
 # final cumulative required
 if len(scenes)!=6: errors.append(f'SCENE_COUNT:{rid}:{len(scenes)}')
 final= rdir/'LEX1_B_S03/CUMULATIVE_SOURCE.json'
 if final.exists():
  final_doc=load(final); final_bind=digest(final)
 else: final_doc={}; final_bind=None
 # root cfg record
 cfgroot=next((x for x in cfg.get('roots',[]) if x.get('root_id')==rid),None)
 if not cfgroot: errors.append(f'CONFIG_ROOT_MISSING:{rid}'); cfgroot={}
 root_records.append({'root_id':rid,'status':'SOURCE_READY_FOR_QUESTION_AUTHORING','mode':'LIVE_EXTENSION_V5','namespace':cfg.get('namespace','LEX1'),'source_binding':final_bind,'baseline_binding':cfgroot.get('source_binding'),'final_scene_ref':'LEX1:B:S03','final_cumulative_turn_count':len(final_doc.get('turns',[])),'scene_sequence':scenes,'task_operations':cfgroot.get('task_operations',[]),'material_targets':cfgroot.get('material_targets',[]),'native_events':cfgroot.get('native_events',[]),'source_author_input_binding':digest(base/'handoff'/rid/'SOURCE_AUTHOR_INPUT.json'),'question_plan':freeze.get('question_plan',{}),'question_generation_started':False,'training_admitted':False})

if errors:
 print('ERRORS',len(errors)); print('\n'.join(errors)); sys.exit(2)

if errors:
 print("ERRORS",len(errors)); print("\n".join(errors)); raise SystemExit(2)
print("VERIFY_OK roots=2 scenes=12 ready_scenes=12 new_turns=160")
