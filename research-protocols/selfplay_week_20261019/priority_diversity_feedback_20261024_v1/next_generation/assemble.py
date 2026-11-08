"""Validate the bounded next-generation feedback handoff, without Solver calls."""
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode=True
N=Path(__file__).resolve().parent
P=N.parent
spec=importlib.util.spec_from_file_location('primary_assembly',P/'assemble.py')
primary=importlib.util.module_from_spec(spec);spec.loader.exec_module(primary)
gate=primary.gate
from question_presentation_gate import prepare_display, restore_prediction
def read(path):return json.loads(Path(path).read_text())
def save(name,value):
 with (N/name).open('x',encoding='utf-8') as f:
  json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def rows(d,key):return d if isinstance(d,list) else d[key]
def identity(c):
 return gate.fingerprint({'question_type':c['question_type'],'question':c['question'],
                          'choices':sorted(c.get('options',c.get('items')).values())})
def main():
 selection_path=N/'SELECTION_FREEZE.json'
 selection=read(selection_path)
 for b in selection['bindings']:assert gate.file_binding(b['path'])==b
 expected={r['root_id']:r for r in selection['roots']}
 cards=read(P/'semantic_audit/assembled/EXPLORATORY_WEAKNESS_CARDS.json')
 card_ids={c['card_id'] for c in cards['cards']}
 sem={r['request_id']:r for r in read(P/'semantic_audit/assembled/FULL_RESPONSE_REVIEWS.json')['records']}
 old=read(P/'CANDIDATES_PRIVATE.json')
 for directory in ['postweek_production_batch_v1','postweek_production_batch_v2']:
  path=P.parent/directory/'CANDIDATES_PRIVATE.json'
  if path.exists():old.extend(rows(read(path),'candidates'))
 signatures={identity(c) for c in old}
 candidates=[];reviews=[];ledger=[];consumption=[];audits=[]
 for group in ('a','b'):
  author_dir=N/f'author_{group}';review_dir=N/f'review_{group}'
  receipt=read(author_dir/'AUTHOR_READ_RECEIPT.json')
  plans=rows(read(author_dir/'TASK_PLANS.json'),'plans')
  draft=rows(read(author_dir/'CANDIDATES_PRIVATE_DRAFT.json'),'candidates')
  final=rows(read(author_dir/'CANDIDATES_PRIVATE.json'),'candidates')
  initial=rows(read(review_dir/'INITIAL_CONTENT_REVIEW_AI.json'),'reviews')
  final_reviews=rows(read(review_dir/'CONTENT_REVIEW_AI.json'),'reviews')
  roots={root:r for root,r in expected.items() if r['author_group']==group}
  assert receipt['full_sources_read'] is True
  required_cards={r['selected_card_id'] for r in roots.values() if r['selection_status']=='SELECTED'}
  assert required_cards<=set(receipt['card_ids_read'])<=card_ids
  assert receipt['source_card_binding']==gate.file_binding(P/'semantic_audit/assembled/EXPLORATORY_WEAKNESS_CARDS.json')
  assert len(receipt['sources'])==len(roots)
  seen_sources=set()
  for s in receipt['sources']:
   b=s.get('binding',s.get('source_binding'));assert gate.file_binding(b['path'])==b
   source=read(b['path']);assert source['root_id'] in roots
   selected_source=roots[source['root_id']]
   assert b=={k:selected_source[k] for k in ('path','sha256','bytes')}
   assert source['root_id'] not in seen_sources;seen_sources.add(source['root_id'])
   assert s['turn_refs_read']==[t['turn_ref'] for t in source['turns']]
  assert seen_sources==set(roots)
  assert len(plans)==len(roots) and {p['root_id'] for p in plans}==set(roots)
  plan_by={p['root_id']:p for p in plans}
  assert len({c['candidate_id'] for c in final})==len(final)
  assert len({c['root_id'] for c in final})==len(final) and {c['root_id'] for c in final}<=set(roots)
  dby={c['candidate_id']:c for c in draft};iby={r['candidate_id']:r for r in initial};rby={r['candidate_id']:r for r in final_reviews}
  assert len(dby)==len(draft)==len(iby)==len(initial)==len(rby)==len(final_reviews)==len(final)
  assert set(dby)==set(iby)==set(rby)=={c['candidate_id'] for c in final}
  bound,_=gate.bind_candidates(final,selection_path);assert bound==final
  for c in final:
   cid=c['candidate_id'];root=roots[c['root_id']];plan=plan_by[c['root_id']]
   assert root['selection_status']=='SELECTED'
   assert c['author_id']==receipt['author_id'] and c.get('solver_evaluated') is False
   assert c.get('formal_weakness_admission') is False
   assert plan['candidate_id']==cid and plan['card_id']==c['feedback_target']['card_id']==root['selected_card_id']
   assert plan['task_operation']==c['task_operation']
   assert c['feedback_target']['dimension']==root['selected_dimension']
   for field in ('source_supported_target_relation','predicted_error','positive_counterexample'):
    assert isinstance(plan.get(field),str) and plan[field].strip()
   if c['question_type']=='multi_select':assert 2<=len(c['gold'])<=5 and len(c['options'])==6
   else:assert c['question_type']=='ordering' and 4<=len(c['items'])<=6
   supports=set(root['supporting_request_ids'])
   selected=set(c['feedback_target'].get('reviewed_request_ids',[]))
   assert selected and selected<=supports and selected<=set(receipt['reviewed_response_ids_read'])
   assert all(sem[r]['root_id']==c['root_id'] and sem[r]['predicted_error_assessment']['decision']=='OBSERVED' for r in selected)
   # A correct item in an otherwise failed response is a legitimate positive
   # comparison; an unrelated success from another dimension is not.
   positives=receipt['positive_counterexamples_read']
   usable=[]
   for positive in positives:
    rid=positive['request_id'];label=positive['display_item']
    assert rid in sem and label in sem[rid]['items'] and positive['note'].strip()
    response=sem[rid]
    if response['root_id']!=c['root_id'] or response['feedback_target']['dimension']!=root['selected_dimension']:
     continue
    item=response['items'][label]
    assert all(item[f]['decision']=='PASS' for f in ('judgment_correct','reason_support'))
    assert label not in response['predicted_error_assessment']['affected_display_items']
    if item['citation_support']['decision']!='PASS':
     assert positive.get('actual_citation_supported') is False
     assert positive.get('comparison_scope')=='JUDGMENT_AND_REASON_ONLY_WITH_REVIEWER_SOURCE_RECONSTRUCTION'
     assert positive.get('additional_source_evidence')
     source_by_ref={t['turn_ref']:t for t in read(c['source_binding']['path'])['turns']}
     for e in positive['additional_source_evidence']:
      t=source_by_ref[e['turn_ref']]
      assert e['speaker']==t['speaker'] and e['quote'] and e['quote'] in t['text']
    usable.append(positive)
   assert usable,'No source-supported positive item in the selected dimension'
   for version,review in [(dby[cid],iby[cid]),(c,rby[cid])]:
    result=gate.validate_review(version,review,selection_path)
    assert result['review_record_complete'],result
    assert review['source_read_complete'] is True
    assert review.get('reviewer_is_human') is False and review.get('gold_blind') is False
    assert review['reviewer_id']==root['reviewer_id']
    reconstructed=review['reconstructed_gold']
    if review['decision']=='PASS':
     assert result['admitted'],result
     assert (set(reconstructed)==set(version['gold']) if version['question_type']=='multi_select' else reconstructed==version['gold'])
     assert review['feedback_realization']['decision']=='PASS'
     assert review['no_source_shortcut']['decision']=='PASS'
    assert review['feedback_realization']['card_id']==root['selected_card_id']
   result=gate.validate_review(c,rby[cid],selection_path);audits.append(result)
   sig=identity(c);assert sig not in signatures,'WHOLE_QUESTION_DUPLICATE';signatures.add(sig)
   ledger.append({'root_id':c['root_id'],'candidate_id':cid,'initial_decision':iby[cid]['decision'],
       'decision':rby[cid]['decision'],'selected_card_id':root['selected_card_id'],
       'revised':c!=dby[cid],'candidate_sha256':gate.fingerprint(c),
       'reviewed_request_ids':sorted(selected),'solver_evaluated':False})
   candidates.append(c);reviews.append(rby[cid])
  for root in set(roots)-{c['root_id'] for c in final}:
   plan=plan_by[root]
   assert plan['status'] in ('MATERIAL_SHORTFALL','NO_OBSERVED_TARGET')
   ledger.append({'root_id':root,'candidate_id':None,'decision':plan['status'],'reason':plan.get('reason',plan.get('source_supported_target_relation'))})
  revision=read(author_dir/'REVISION_LOG.json')
  assert revision.get('revision_rounds',revision.get('rounds',0))<=1
  consumption.append({'author_group':group,'author_id':receipt['author_id'],'generated':len(final),
    'card_ids_read':receipt['card_ids_read'],'raw_response_read_ids':receipt['reviewed_response_ids_read'],
    'receipt_binding':gate.file_binding(author_dir/'AUTHOR_READ_RECEIPT.json'),
    'plans_binding':gate.file_binding(author_dir/'TASK_PLANS.json'),
    'draft_binding':gate.file_binding(author_dir/'CANDIDATES_PRIVATE_DRAFT.json'),
    'revision_binding':gate.file_binding(author_dir/'REVISION_LOG.json'),
    'final_review_binding':gate.file_binding(review_dir/'CONTENT_REVIEW_AI.json')})
 assert len(ledger)==len(expected)==4 and len(candidates)<=4
 summary={'schema_version':'priority_next_generation_summary_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
    'planned_roots':4,'generated':len(candidates),'content_admitted':sum(a['admitted'] for a in audits),
    'initial_decisions':dict(Counter(r.get('initial_decision',r['decision']) for r in ledger)),
    'final_decisions':dict(Counter(r['decision'] for r in ledger)),
    'target_dimensions':dict(Counter(c['feedback_target']['dimension'] for c in candidates)),
    'additional_solver_requests':0,'training_runs':0,'new_dialogue_turns':0,
    'feedback_consumed':bool(candidates),'formal_weakness_admission':False,'causal_feedback_gain_proven':False,
    'claim_boundary':'Content-reviewed new questions prove concrete feedback consumption only; no new Solver recurrence or training benefit was tested.'}
 public_displays=[];private_displays=[]
 for c in candidates:
  public,private=prepare_display(c,'priority_next_generation_preview_v1')
  assert private['nonidentity_permutation']
  restored=restore_prediction(private,private['display_gold'])
  assert (set(restored)==set(c['gold']) if c['question_type']=='multi_select' else restored==c['gold'])
  public_displays.append(public);private_displays.append(private)
 save('CANDIDATES_PRIVATE.json',candidates)
 save('CONTENT_REVIEW_AI.json',{'reviews':reviews,'mechanical_audits':audits,'gold_blind':False,'reviewer_is_human':False})
 save('FEEDBACK_CONSUMPTION_TRACE.json',{'ledger':ledger,'author_consumption':consumption,
    'updated_cards_left_immutable':True,'bindings':[gate.file_binding(selection_path)]})
 save('PRESENTATION_PUBLIC_PREVIEW.json',{'solver_requests':0,'preview_only':True,'questions':public_displays})
 save('PRESENTATION_MAPPING_PRIVATE.json',{'solver_requests':0,'preview_only':True,'mappings':private_displays})
 save('SUMMARY.json',summary)
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
