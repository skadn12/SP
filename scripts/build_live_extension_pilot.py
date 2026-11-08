import hashlib, json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/root/autodl-tmp/gxy-selfplay')
WEEK = ROOT/'research-protocols/selfplay_week_20261019/postweek_live_extension_v4'
POOL = ROOT/'offline-deliveries/h4_external_event_source_refresh_v1/multiwoz_train_sample.json'
SRC = ROOT/'offline-deliveries/h4_codex_train_pool_v125/sources'

def bind(p):
    p=Path(p).resolve(); b=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def write(p,d):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def now(): return datetime.now(timezone.utc).isoformat()

pool={x['dialogue_id']:x for x in json.loads(POOL.read_text())}
alloc={
 'cc-train-episode-19708':['PMUL4617.json','PMUL4370.json','MUL1411.json'],
 'cc-train-episode-20432':['PMUL1780.json','PMUL0819.json','MUL1714.json'],
}
source_rows={r: SRC/(r+'.json') for r in ['cc-train-episode-19708','cc-train-episode-20432']}
events=[]
for root,ids in alloc.items():
    for n,dialogue_id in enumerate(ids,1):
        out=WEEK/'inputs'/f'{root}_B_S{n:02d}_EVENT.json'
        write(out,pool[dialogue_id])
        events.append({'root_id':root,'scene_ref':f'LEX1:B:S{n:02d}',
            'source_id':'multiwoz22:train:'+dialogue_id,'split':'train',
            'binding':bind(out),'reserved_before_authoring':True,
            'pool_binding':bind(POOL),'pool_record_id':dialogue_id,
            'status':'PERMANENTLY_RESERVED_BEFORE_AUTHORING','release_on_failure':False})
write(WEEK/'inputs/SOURCE_RESERVATIONS.json',{'schema_version':'live_extension_source_reservations_v1',
 'created_at_utc':now(),'status':'PERMANENT','policy':'No reuse and no release on failure or nonexecution; all six B events bound before authoring.',
 'pool_binding':bind(POOL),'reservations':events})
short=WEEK/'feedback/SHORTFALL_REVIEW_AI.json'
short_b=bind(short)
upstream_review=json.loads(short.read_text())
adapter=WEEK/'feedback/SHORTFALL_REVIEW_ADAPTER.json'
write(adapter,{'schema_version':'shortfall_review_adapter_v1','upstream_review_binding':short_b,
  'reviewer_id':upstream_review['reviewer_id'],'reviewer_type':'ai_agent','full_source_read':True,
  'solver_results_read':False,'slot_reviews':[{'slot_id':r['candidate_id'],'root_id':r['root_id'],
    'recommendation':'PASS_SHORTFALL','source_binding':r['source_binding'],
    'finding':r['clock_value_audit']['conclusion'],'evidence':r['clock_value_audit']['values'],
    'limitations':r['limitations']} for r in upstream_review['slot_reviews']]})
short_b=bind(adapter)
ledger=WEEK/'feedback/SHORTFALL_LEDGER_ADAPTER.json'
ledger_rows=[]
for root,slot in [('cc-train-episode-19708','PROD2-19708-ORDER2'),('cc-train-episode-20432','PROD2-20432-ORDER2')]:
    ledger_rows.append({'slot_id':slot,'root_id':root,'source_sha256':bind(source_rows[root])['sha256'],
      'status':'SOURCE_SHORTFALL','review_binding':short_b,
      'review_finding':'Independent source-only review found fewer than four distinct uniquely bindable explicit clock readings; departure field remains absent.'})
write(ledger,{'schema_version':'source_shortfall_ledger_adapter_v1','status':'FROZEN','rows':ledger_rows,
  'upstream_slot_ledger_binding':bind(WEEK.parent/'postweek_production_batch_v2/SLOT_LEDGER.json'),
  'review_binding':short_b})
ledger_b=bind(ledger)
roots=[]
for root in alloc:
    sp=source_rows[root]
    if root.endswith('19708'):
        targets=[('PROD2-19708-ORDER2','Add at least four distinct explicit clock readings tied to separate event items, including the missing departure fields, while preserving plan/booking boundaries.')]
    else:
        targets=[('PROD2-20432-ORDER2','Add at least four distinct explicit clock readings tied to separate event items, including a train departure, while preserving the two Saturday plans and booking/payment boundaries.')]
    ops=[{'operation_id':'explicit_event_time','question_type':'ordering'},
         {'operation_id':'source_state_and_plan_completion','question_type':'multi_select'},
         {'operation_id':'actor_entity_binding','question_type':'multi_select'}]
    roots.append({'root_id':root,'source_binding':bind(sp),'task_operations':ops,
      'material_targets':[{'slot_id':sid,'material_goal':goal} for sid,goal in targets],
      'native_events':[e for e in events if e['root_id']==root]})
cfg={'schema_version':'reason_material_feedback_pipeline_v1_config','batch_id':'postweek_live_extension_v1',
 'namespace':'LEX1','reason_packet':None,'shortfall_ledger':ledger_b,'roots':roots}
write(WEEK/'CONFIG.json',cfg)
write(WEEK/'PROTOCOL_FREEZE.json',{'schema_version':'live_extension_protocol_freeze_v1','status':'FROZEN_BEFORE_AUTHORING',
 'created_at_utc':now(),'namespace':'LEX1','roots':[r['root_id'] for r in roots],
 'scene_sequence':['LEX1:A:S01','LEX1:A:S02','LEX1:A:S03','LEX1:B:S01','LEX1:B:S02','LEX1:B:S03'],
 'source_budget':{'author_attempts_per_scene':1,'semantic_repairs':0,'source_replacements':0,'dependent_scene_after_failed_review':False},
 'scene_turns':{'min':12,'max':16},'full_source_no_truncation':True,'reason_packet':None,
 'question_plan':{'per_root':{'MC2':1,'MC3':1,'MC4':1,'MC5':1,'MC6':1,'ORDER1':1,'ORDER2':1},'source_shortfall_slots_targeted':['ORDER2'],
                  'solver_score_filtering':False,'exact_whole_question_duplicate_only':True},
 'shortfall_review_binding':short_b,'shortfall_ledger_binding':ledger_b,'reservations_binding':bind(WEEK/'inputs/SOURCE_RESERVATIONS.json'),
 'source_bindings':{r['root_id']:r['source_binding'] for r in roots},'generation_started':False})
print(json.dumps({'config':bind(WEEK/'CONFIG.json'),'reservations':bind(WEEK/'inputs/SOURCE_RESERVATIONS.json'),'events':len(events)},ensure_ascii=False))
