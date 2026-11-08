import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
r=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def layers(b):
 assert (b/'solver/COMPLETED.json').exists()
 scores=read(b/'solver/SCORES.json')['items'];q={i['candidate_id']:i for i in read(b/'QUALITY_FINAL.json')['items']};rows=[]
 for i in read(b/'ITEMS_PRIVATE.json'):
  s=scores.get(i['candidate_id'],{}).get('full_context',{});m=s.get('mean_directed_pair_f1',s.get('mean_J'))
  if not q[i['candidate_id']]['content_pass']:status,reason='QUALITY_HOLD_NOT_EVALUATED','Content review HOLD; training value not assessed.'
  elif not s or s['valid_calls']!=s['planned_calls']:status,reason='OUTPUT_PROTOCOL_PENDING','Full planned calls include strict-protocol invalid output; final-answer-only cannot confer eligibility.'
  elif m==1:status,reason='MASTERED_RESERVE','Full strictly valid mean equals 1; quality remains PASS.'
  elif m<=0:status,reason='HARD_EXAMPLE_DIAGNOSTIC','Full strictly valid mean is zero or negative; not in the open main-pool interval.'
  else:status,reason='MAIN_POOL_CANDIDATE','Full complete and strictly valid, mean inside (0,1); union collection selection still pending.'
  rows.append({'candidate_id':i['candidate_id'],'root_id':i['root_id'],'arm':i['arm'],'question_type':i['question_type'],'quality_status':q[i['candidate_id']]['status'],'training_value_status':status,'metric':'directed_pair_F1' if i['question_type']=='partial_order' else 'signed_J','mean_full_score':m,'full_statistics':s,'selection_status':'PENDING_UNION_ALL68','reason':[reason]})
 write(b/'CANDIDATE_LAYER_DECISIONS.json',{'items':rows,'quality_training_value_collection_separated':True,'final_answer_only_or_no_context_confers_eligibility':False})
 print(b.name,dict(Counter(x['training_value_status'] for x in rows)))
 return rows
def aggregate(index,arm,notes):
 b=r/f'new_evaluation/batch_{index:02d}_{arm}';items=[];qs=[];bindings=[];audit=[]
 chain=r/'source_preparation/ALL68_REVIEW_CHAIN_AUDIT.json';idx={x['root_id']:x for x in read(chain)['rows']}
 for root in sorted((r/f'batch_{index:02d}').glob('cc-*'),key=lambda p:int(p.name.rsplit('-',1)[1])):
  d=root/arm/'batch';draft=root/arm/'author/CANDIDATES_FIRST_DRAFT.json';assert draft.exists()
  ii=read(d/'ITEMS_PRIVATE.json');qpath=d/'QUALITY_FINAL.json';view=d/'review_enum_interface_view/QUALITY_FINAL.json'
  if view.exists():qpath=view
  target_view=d/'target_hash_interface_view/QUALITY_FINAL.json'
  if target_view.exists():qpath=target_view
  citation_view=d/'review_citation_interface_view/QUALITY_FINAL.json'
  if citation_view.exists():qpath=citation_view
  policy_view=d/'warning_policy_view/QUALITY_FINAL.json'
  if policy_view.exists():qpath=policy_view
  qq=read(qpath)['items'];assert len(ii)==len(qq)
  for p in [d/'ITEMS_PRIVATE.json',qpath,d/'AUTHOR_BINDINGS.json',*sorted((d/'blind_results').glob('Q*.json'))]:bindings.append(bind(p))
  for i in ii:
   src=Path(i['source_path']);assert bind(src)['sha256']==i['source_sha256']
   turns={t['turn_ref']:t for t in read(src)['turns']};ev=list(i['evidence'])
   for row in i['option_audit']:ev.extend(row.get('evidence',[]))
   for row in i.get('pair_audit',[]):ev.extend(row.get('evidence',[]))
   for e in ev:
    assert e['quote'] and e['quote'] in turns[e['turn_ref']]['text']
    assert 'origin' not in e or e['origin']==turns[e['turn_ref']].get('origin')
   row=idx[i['root_id']];assert not row['specific_unresolved_items']
   proofs=[]
   for c in row['r02_review_chain']:
    p=Path(c['review']['path']);assert bind(p)['sha256']==c['review']['sha256'];proofs.append(c)
   for c in row['r03_review_chain']:
    assert c['reviewed_draft_all_48_dialogue_fields_equal_current_R03'];proofs.append(c)
   audit.append({'candidate_id':i['candidate_id'],'source':bind(src),'actual_cited_refs':sorted({e['turn_ref'] for e in ev}),'quotes_and_origins_match':True,'existing_exact_source_review_proofs':proofs,'cited_use_assessment':notes[i['root_id']],'source_blocker':False,'public_partial_scope_assessment':'Defines relation and query domain only; no asserted answer edges/nonedges.' if i['question_type']=='partial_order' else None})
  items.extend(ii);qs.extend(qq)
 assert len(items)==len({i['candidate_id'] for i in items})==len(qs)
 additional_proof=r/f'source_preparation/BATCH{index:02d}_{arm}_PARENT_R01_CITED_USE_ADDENDUM.json'
 if additional_proof.exists():bindings.append(bind(additional_proof))
 independent_proof=r/f'source_preparation/BATCH{index:02d}_{arm}_OLD_LAYER_CITED_USE_INDEPENDENT_AUDIT.json'
 if independent_proof.exists():
  independent=read(independent_proof)
  assert independent['summary']['source_HOLD_required']==0
  bindings.append(bind(independent_proof))
 write(b/'ITEMS_PRIVATE.json',items);write(b/'QUALITY_FINAL.json',{'items':qs});write(b/'SOURCE_FREEZE.json',read(r/f'batch_{index:02d}/SOURCE_FREEZE.json'))
 extra=r/f'batch_{index:02d}/G_AUTHOR_OMISSIONS_MAINLINE_ASSESSMENT.json'
 if arm=='G' and extra.exists():bindings.append(bind(extra))
 maximum_calls=36*sum(q['content_pass'] for q in qs)
 write(b/'AGGREGATION_FREEZE.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'batch_index':index,'component':arm,'all_component_first_drafts_frozen':True,'items':len(items),'content_changes':False,'bindings':bindings,'maximum_calls':maximum_calls})
 write(r/f'source_preparation/BATCH{index:02d}_{arm}_EVIDENCE_QUALIFICATION_AUDIT.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'role':'mainline_cited_use_check_of_existing_reviews','source_chain_index':bind(chain),'additional_exact_prefix_cited_use_proof':bind(additional_proof) if additional_proof.exists() else None,'independent_old_layer_cited_use_proof':bind(independent_proof) if independent_proof.exists() else None,'candidates':audit,'new_permission_or_blanket_root_approval':False,'source_blockers':0,'separate_from_blind_quality':True})
 freeze=r/f'feedback_cards/BATCH_{index:02d}_EVAL_COMPONENT_FREEZE.json'
 if not freeze.exists():
  write(freeze,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'batch_index':index,'execution_components':[{'arm':arm,'batch_path':str(b),'frozen_candidates':len(items),'maximum_calls':maximum_calls},{'arm':'F' if arm=='G' else 'G','future_batch_path':str(r/f'new_evaluation/batch_{index:02d}_'+('F' if arm=='G' else 'G')) if False else str(r/f'new_evaluation/batch_{index:02d}_F'),'maximum_first_drafts':16,'maximum_calls':576}],'global_new_budget_unchanged':9792,'per_root_arm_slots_preserved':True,'G_scores_not_sent_to_F_or_reviewers':True,'feedback_update_only_after_both_components_complete':True,'same_items_not_reevaluated_in_later_join':True,'no_within_batch_content_rewrites':True})
 print({'batch':b.name,'items':len(items),'pass':sum(q['content_pass'] for q in qs),'maximum_calls':maximum_calls})
 return b
