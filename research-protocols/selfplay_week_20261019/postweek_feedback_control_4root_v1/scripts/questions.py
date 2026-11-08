"""Prospective immutable question handoff, mechanical gate and solver export."""
import argparse, hashlib, json, pathlib, random
from control import BASE, bind, verify, read, write, now
SLOTS={'MC2':'MC_ENTITY_BINDING','MC3':'MC_STATE_UPDATE_CONJUNCTION','MC4':'MC_CONCLUSION_EVIDENCE','MC5':'MC_JOINT_CONSTRAINT','MC6':'MC_RECORD_AUDIT_CONTROL','ORDER1':'ORDER_EXPLICIT_CLOCK_VALUE','ORDER2':'ORDER_ENQUIRY_PROGRESSION'}
def prepare(root,arm):
 u=BASE/'sources'/root/arm; source=read(u/'FINAL_SOURCE.json'); packet=read(u/'SOURCE_AUTHOR_INPUT.json')
 for s in ['FBC1_'+r+'_S'+str(i).zfill(2) for r in 'AB' for i in range(1,4)]: assert read(u/s/'ACCEPTED.json')['status']=='READY'
 p={'schema_version':'fbc1_private_question_author_v1','root_id':root,'condition':arm,
  'source_binding':bind(u/'FINAL_SOURCE.json'),'full_source':source,'material_goals':packet['material_goals'],
  'slots':SLOTS,'mc_options':6,'required_gold_counts':{f'MC{i}':i for i in range(2,7)},
  'ordering_items':4,'drafts_per_slot':1,'semantic_revisions':0,
  'clock_rule':read(BASE/'PROTOCOL_FREEZE.json')['clock_operation'],
  'enquiry_rule':read(BASE/'PROTOCOL_FREEZE.json')['enquiry_operation'],
  'formal_weakness_admission':False,'training_benefit_claim':False}
 write(BASE/'question_handoffs'/root/arm/'QUESTION_AUTHOR_INPUT.json',p)
 return bind(BASE/'question_handoffs'/root/arm/'QUESTION_AUTHOR_INPUT.json')
def mechanical(p):
 q=read(p); source=verify(q['source_binding']); turns={t['turn_ref']:t for t in source['turns']}
 slot=q['slot_id']; assert q['task_operation']==SLOTS[slot]
 assert q['source_read_scope']['all_turns_read'] and q['source_read_scope']['turn_count']==len(turns)
 texts=q['options'] if slot.startswith('MC') else q['items']; keys=list(texts)
 assert len(keys)==len(set(keys)) and all(isinstance(x,str) and x.strip() for x in texts.values())
 if slot.startswith('MC'):
  assert set(keys)==set('ABCDEF') and len(q['gold'])==int(slot[2:]) and len(set(q['gold']))==len(q['gold'])
 else:
  assert set(keys)=={'I1','I2','I3','I4'} and len(q['gold'])==4 and set(q['gold'])==set(keys)
 assert set(q['display_order'])==set(keys) and len(q['display_order'])==len(keys)
 if slot.startswith('ORDER'): assert q['display_order']!=q['gold'] and q['display_order']!=keys
 assert set(q['evidence'])==set(keys) and set(q['private_option_notes'])==set(keys)
 for k in keys:
  assert q['evidence'][k] and q['private_option_notes'][k].strip()
  for e in q['evidence'][k]: assert e['turn_ref'] in turns and e['quote'] and e['quote'] in turns[e['turn_ref']]['text'], (q['candidate_id'],k,e)
 return {'candidate_id':q['candidate_id'],'question':bind(p),'mechanical_status':'PASS','semantic_status':'NOT_INFERRED','created_at_utc':now()}
def export():
 roots=read(BASE/'PROTOCOL_FREEZE.json')['roots']; records=[]; pub=[]; priv=[]; seen=set(); decisions=[]
 for root in roots:
  for arm in ['GENERIC','TARGETED']:
   qdir=BASE/'questions'/root/arm; rp=BASE/'question_reviews'/root/arm/'REVIEW.json'
   if not rp.exists():
    for slot in SLOTS: decisions.append({'root_id':root,'condition':arm,'slot_id':slot,'status':'SOURCE_FAILED_OR_MISSING_NO_SOLVER'})
    continue
   review=read(rp); assert review['independent_of_question_author'] and review['full_source_read']
   for slot in SLOTS:
    p=qdir/f'{slot}.json'; row=next(r for r in review['questions'] if r['slot_id']==slot)
    if not p.exists():
     assert row['decision'] in ['SHORTFALL','FAIL'];decisions.append({'root_id':root,'condition':arm,'slot_id':slot,'status':row['decision']});continue
    q=read(p); assert row['question_binding']==bind(p); assert row['reviewer_id']!=q['author_id']
    assert review['source_binding']==q['source_binding']
    status=row['decision']
    if status=='ADMIT':
     mechanical(p)
     source=verify(q['source_binding'])
     context=[{k:t[k] for k in ['turn_ref','speaker','text']} for t in source['turns']]
     identity=json.dumps({'full_source_context':context,'question':q['question'],'items':q.get('options') or q.get('items')},sort_keys=True)
     if identity in seen: status='EXACT_WHOLE_DUPLICATE'
     else: seen.add(identity)
    decisions.append({'root_id':root,'condition':arm,'slot_id':slot,'candidate_id':q['candidate_id'],'status':status,'review':bind(rp)})
    if status!='ADMIT': continue
    labels=list('ABCDEF') if slot.startswith('MC') else list('ABCD')
    canonical=q.get('options') or q['items']; mp=dict(zip(labels,q['display_order']))
    display={k:canonical[v] for k,v in mp.items()}; reverse={v:k for k,v in mp.items()}
    common={k:q[k] for k in ['candidate_id','root_id','condition','question_type','task_operation','question','source_binding']}
    pub.append({**common,'display':display})
    priv.append({'candidate_id':q['candidate_id'],'question_type':q['question_type'],'canonical_gold':q['gold'],'display_gold':[reverse[k] for k in q['gold']],'display_to_canonical':mp,'question_binding':bind(p)})
    records.append(bind(p))
 freeze={'schema_version':'fbc1_question_freeze_v1','status':'FROZEN_BEFORE_SOLVER','created_at_utc':now(),'planned_slots':56,'admitted_questions':len(pub),'decisions':decisions,'question_bindings':records,'solver_scores_read':False}
 write(BASE/'QUESTION_FREEZE.json',freeze)
 write(BASE/'inputs/PUBLIC_QUESTIONS.json',{'questions':pub})
 write(BASE/'inputs/PRIVATE_SCORING.json',{'questions':priv})
 ef={'schema_version':'fbc1_solver_export_v1','status':'FROZEN_BEFORE_SOLVER','created_at_utc':now(),'public_binding':bind(BASE/'inputs/PUBLIC_QUESTIONS.json'),'private_binding':bind(BASE/'inputs/PRIVATE_SCORING.json'),'question_freeze_binding':bind(BASE/'QUESTION_FREEZE.json'),'planned_root_ids':roots,'planned_slots':56,'content_review_status':'PASS','solver_score_filtering':False,'full_source_no_truncation':True,'bindings':[bind(BASE/'PROTOCOL_FREEZE.json')]+records}
 write(BASE/'inputs/EXPORT_FREEZE.json',ef)
 return {'admitted':len(pub),'requests':len(pub)*2,'export':bind(BASE/'inputs/EXPORT_FREEZE.json')}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','mechanical','export']);ap.add_argument('--root');ap.add_argument('--arm');ap.add_argument('--path');a=ap.parse_args()
 print(json.dumps(prepare(a.root,a.arm) if a.action=='prepare' else mechanical(a.path) if a.action=='mechanical' else export(),ensure_ascii=False))
