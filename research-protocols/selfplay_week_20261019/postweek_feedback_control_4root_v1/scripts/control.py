"""Scoped prospective cross-root transfer adapter; no global production changes."""
import argparse, hashlib, json, pathlib, sys
from datetime import datetime, timezone

BASE = pathlib.Path(__file__).resolve().parents[1]
CHECKS = ['time','location','actor_identity','plan_vs_completion','cross_scene_continuity','semantic_repetition','naturalness','event_fidelity']
SCENES = [f'FBC1:{route}:S{i:02}' for route in 'AB' for i in range(1,4)]
FORBIDDEN = {'answer','answers','final_answer','answer_reason','expected_error','model_reported_explanations','semantic_review'}
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(pathlib.Path(p).read_text())
def bind(p):
 p=pathlib.Path(p).resolve(); b=p.read_bytes()
 return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def verify(b):
 assert bind(b['path'])==b, ('BINDING_CHANGED',b['path'])
 return read(b['path'])
def write(p,d):
 p=pathlib.Path(p); p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f: f.write(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def clean(d):
 if isinstance(d,dict):
  for k,v in d.items():
   assert 'gold' not in k.lower() and k.lower() not in FORBIDDEN, k
   clean(v)
 elif isinstance(d,list):
  for x in d: clean(x)
def unit(root,arm): return BASE/'sources'/root/arm
def initialize():
 proposal=read(BASE/'ROOT_SELECTION_PROPOSAL.json')
 assert (BASE/'inputs/SOURCE_RESERVATIONS.json').exists(), 'WAIT_FOR_RESERVATIONS'
 reservations=read(BASE/'inputs/SOURCE_RESERVATIONS.json')
 evdir=BASE/'inputs/native_events'
 events={}
 for p in evdir.glob('*.json'):
  d=read(p); record=d.get('full_native_record',d.get('native_record',d))
  sid='multiwoz22:train:'+record['dialogue_id']
  events[sid]=(p,record)
 assert len(events)==12, len(events)
 generic=[
  'Develop useful, natural continuations with explicit facts and independently checkable relations. Preserve continuity and distinguish requested arrangements from confirmed and completed actions.',
  'Keep conversations informative across the reserved material. Preserve clear speakers and what each person reports; avoid repetitive recaps or contrived question-like narration.'
 ]
 targeted=[
  'In natural service enquiries, separate the order in which information is disclosed from the clock values being discussed. Include explicit but nonformulaic records where these relations differ; preserve unresolved arrangements without replacing them with a stereotypical service workflow.',
  'Keep service replies, the reporting participant’s own interpretation, and the interlocutor’s suggestion distinctly attributable. Develop adjacent, plausible statements about the same event whose actor-to-statement bindings remain uniquely checkable from the record.'
 ]
 frozen_inputs=[]
 for row in proposal['roots']:
  root=row['root_id']; src=verify(row['source_binding'])
  write(BASE/'inputs/initial_sources'/f'{root}.json',src)
  selected=[]
  for sid in row['proposed_native_event_source_ids']:
   assert sid in events, ('EVENT_CHANGED_NEEDS_MAPPING',sid)
   p,record=events[sid]; selected.append({'source_id':sid,'native_event_binding':bind(p),'native_record':record})
  for arm in ['GENERIC','TARGETED']:
   packet={'schema_version':'fbc1_source_author_input_v1','root_id':root,'condition':arm,
    'namespace':'FBC1','source_binding':bind(BASE/'inputs/initial_sources'/f'{root}.json'),
    'full_initial_source':src,'material_goals':generic if arm=='GENERIC' else targeted,
    'scene_sequence':SCENES,'route_roles':{'A':'same-root continuation','B':'reserved native event continuation; cumulative after A'},
    'turns_per_scene':14,'attempts_per_scene':1,'speakers':row['speaker_names'],
    'native_events':selected,'context_policy':'FULL_SOURCE_NO_TRUNCATION',
    'review_dimensions':CHECKS,'append_only':True,
    'common_material_requirements':[
     'Maintain enough uniquely grounded factual relationships for actor binding, updated state, conclusion support, joint constraints and record audit tasks.',
     'Preserve native explicit clock values and their actual roles in reported service interactions. At least four distinct clock readings across the B material are useful; only report times already explicit in native utterances.',
     'Native events are distinct enquiries; do not imply one cross-day journey or completion. A numeric clock comparison may compare reported clock readings with explicit scope, not real chronology.',
     'Keep enquiries with identifiable disclosure steps, without inserting numbered steps or quiz language. Do not add unsupported genders or new central people.',
     'Do not invent service replies, confirmation, payments, departures or external facts. Fictional linking dialogue may frame an exact service record as a reported enquiry, retaining requests versus responses.'
    ]}
   clean(packet); write(unit(root,arm)/'SOURCE_AUTHOR_INPUT.json',packet)
   frozen_inputs.append(bind(unit(root,arm)/'SOURCE_AUTHOR_INPUT.json'))
 cards=read(BASE/'WEAKNESS_CARDS.json')
 for c in cards['cards']:
  for b in c['origin'].values():
   if isinstance(b,dict) and 'path' in b: verify(b)
 write(BASE/'TRANSFER_PLAN.json',{'schema_version':'fbc1_cross_root_transfer_adapter_v1','created_at_utc':now(),
  'origin_cards':bind(BASE/'WEAKNESS_CARDS.json'),'target_roots':[r['root_id'] for r in proposal['roots']],
  'mapping':{'CAND_ORDER_ENQUIRY_RELATION_V1':'FBC1-W1','CAND_VISIBLE_SPEAKER_BINDING_V1':'FBC1-W2'},
  'targeted_abstract_goals':targeted,'generic_abstract_goals':generic,'same_root_feedback_claim':False,
  'shortfall_ledger_fabricated':False,'global_pipeline_modified':False,'source_authors_receive_raw_reasons':False})
 protocol={'schema_version':'fbc1_prospective_protocol_freeze_v1','status':'FROZEN_BEFORE_AUTHORING','created_at_utc':now(),
  'scope':'Train-only exploratory cross-root feedback transfer','independent_units':4,
  'roots':[r['root_id'] for r in proposal['roots']],'conditions':['GENERIC','TARGETED'],
  'initial_sources_seen_by_M_clean':True,'unseen_root_claim':False,
  'scene_sequence':SCENES,'turns_per_scene':14,'source_attempts_per_scene':1,'semantic_revisions':0,
  'source_failure_policy':'Retain failed scene; stop descendants; retain planned denominators.',
  'question_slots':{'MC2':'MC_ENTITY_BINDING','MC3':'MC_STATE_UPDATE_CONJUNCTION','MC4':'MC_CONCLUSION_EVIDENCE','MC5':'MC_JOINT_CONSTRAINT','MC6':'MC_RECORD_AUDIT_CONTROL','ORDER1':'ORDER_EXPLICIT_CLOCK_VALUE','ORDER2':'ORDER_ENQUIRY_PROGRESSION'},
  'mc_options':6,'mc_required_counts':[2,3,4,5,6],'ordering_items':4,
  'clock_operation':'Sort uniquely identified reported clock readings by numeric time of day, with explicit context. Requested cutoffs/booking times stay as such; no inference of actual chronology. Clock numerals absent from item text.',
  'enquiry_operation':'Sort uniquely identified information disclosures in one recorded service enquiry by source turn progression; no ordering cue words in items.',
  'question_attempts_per_slot':1,'question_semantic_revisions':0,
  'ordering_presentation':'Deterministically shuffle item display using recorded seed, reject identity/gold-sorted display.',
  'duplicate_rule':'Exclude only exact whole-question duplicates; preserve rejected files.',
  'quality_selection':'Independent full-source/content gate before solver; solver errors never gate questions.',
  'source_author_isolation':'Fresh fork-none root-condition author; no counterpart or raw origin reasons/gold.',
  'question_author_isolation':'Fresh fork-none root-condition question author after source READY.',
  'author_model':'Codex collaboration agent inherited model; actual session IDs and file bindings recorded. No invented API token counts.',
  'solver':'B1','solver_schedule':['FULL_DIRECT_ONCE','FULL_VISIBLE_REASON_ONCE'],
  'temperature':0.8,'top_p':0.9,'native_thinking':False,'context_window':16384,
  'visible_reason_max_output':3072,'no_retry':True,'no_truncation':True,'raw_seal_before_private_scoring':True,
  'maximum_scenes':48,'maximum_questions':56,'maximum_solver_requests':112,
  'primary_outcome':'Content-admitted, frozen-operation-aligned questions / seven planned slots per root-condition; four paired root differences.',
  'secondary_outcomes':['actual operation coverage','direct final exact and order pairwise','visible-reason final exact','visible explanation actor attribution','citation insufficiency separately'],
  'interpretation':'Exploratory N=4; difficulty/errors do not establish training benefit or stable formal weaknesses.',
  'memory_policy':'Fixed full-source input; no new memory intervention.','training':False,'final_training_pool_admission':False,
  'bindings':[bind(BASE/'ROOT_SELECTION_PROPOSAL.json'),bind(BASE/'inputs/SOURCE_RESERVATIONS.json'),bind(BASE/'WEAKNESS_CARDS.json'),bind(BASE/'INDEPENDENT_DESIGN_REVIEW.json'),bind(BASE/'TRANSFER_PLAN.json')]+frozen_inputs}
 write(BASE/'PROTOCOL_FREEZE.json',protocol)
 return {'status':protocol['status'],'binding':bind(BASE/'PROTOCOL_FREEZE.json'),'inputs':len(frozen_inputs)}
def commit(root,arm,scene):
 u=unit(root,arm); packet=read(u/'SOURCE_AUTHOR_INPUT.json'); i=SCENES.index(scene)
 prev=(u/SCENES[i-1].replace(':','_')/'CUMULATIVE_SOURCE.json') if i else pathlib.Path(packet['source_binding']['path'])
 if i:
  receipt=read(prev.parent/'ACCEPTED.json')
  assert receipt['status']=='READY' and receipt['source']==bind(prev)
  assert receipt['review']==bind(prev.parent/'REVIEW.json')
 else: verify(packet['source_binding'])
 source=read(prev); sdir=u/scene.replace(':','_'); draft=read(sdir/'DRAFT.json')
 assert draft['root_id']==root and draft['scene_ref']==scene
 assert draft['input_source_binding']==bind(prev)
 assert draft['author_input_binding']==bind(u/'SOURCE_AUTHOR_INPUT.json')
 if i>=3: assert draft['native_event_binding']==packet['native_events'][i-3]['native_event_binding']
 turns=draft['turns']; assert len(turns)==14
 for n,t in enumerate(turns,1):
  assert t['turn_ref']==f'{scene}:T{n:03}' and t['scene_ref']==scene
  assert t['speaker'] in packet['speakers'] and t['text'].strip()
 assert not set(t['turn_ref'] for t in source['turns'])&set(t['turn_ref'] for t in turns)
 full={**source,'schema_version':'fbc1_cumulative_source_v1','scene_ref':scene,'condition':arm,
       'input_source_binding':bind(prev),'draft_binding':bind(sdir/'DRAFT.json'),
       'turns':source['turns']+turns,'new_turn_count':14,'fbc1_added_turn_count':14*(i+1)}
 cp=sdir/'CUMULATIVE_SOURCE.json'
 if not cp.exists(): write(cp,full)
 else: assert read(cp)==full
 return {'source':bind(cp),'draft':bind(sdir/'DRAFT.json'),'needs_independent_review':True}
def accept(root,arm,scene):
 u=unit(root,arm); s=u/scene.replace(':','_'); d=read(s/'DRAFT.json'); source=read(s/'CUMULATIVE_SOURCE.json'); r=read(s/'REVIEW.json')
 assert r['source_binding']==bind(s/'CUMULATIVE_SOURCE.json') and r['draft_binding']==bind(s/'DRAFT.json')
 assert r['reviewer_id']!=d['author_id'] and r['independent_of_source_author'] is True
 assert r['full_source_read'] is True and r['turn_refs_read']==[t['turn_ref'] for t in source['turns']]
 assert set(CHECKS)<=set(r['checks'])
 assert all(r['checks'][k]['status'] in ['PASS','WARNING','PASS_WITH_WARNING'] and r['checks'][k]['reason'].strip() for k in CHECKS)
 assert r['decision']=='READY' and not any(x.get('severity')=='BLOCKING' for x in r.get('findings',[]))
 receipt={'status':'READY','created_at_utc':now(),'review':bind(s/'REVIEW.json'),'source':bind(s/'CUMULATIVE_SOURCE.json')}
 write(s/'ACCEPTED.json',receipt)
 if scene==SCENES[-1]: write(u/'FINAL_SOURCE.json',source)
 return receipt
def pending():
 out=[]
 for p in sorted((BASE/'sources').glob('*/*/FBC1_*/CUMULATIVE_SOURCE.json')):
  if not (p.parent/'REVIEW.json').exists():
   d=read(p);out.append({'path':str(p),'root_id':d['root_id'],'condition':d['condition'],'scene_ref':d['scene_ref'],'turn_count':len(d['turns'])})
 return out
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['initialize','commit','accept','pending']);ap.add_argument('--root');ap.add_argument('--arm');ap.add_argument('--scene');a=ap.parse_args()
 result=initialize() if a.action=='initialize' else pending() if a.action=='pending' else globals()[a.action](a.root,a.arm,a.scene)
 print(json.dumps(result,ensure_ascii=False))
