#!/usr/bin/env python3
"""Compare frozen blind answers with frozen semantic Gold; retain all raw outputs."""
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--batch',required=True);a=p.parse_args();b=Path(a.batch)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
items=read(b/'ITEMS_PRIVATE.json');mechanical={i['candidate_id']:i for i in read(b/'MECHANICAL_REVIEW.json')['items']};rows=[]
required={'review_id','reviewer_id','reviewer_mode','full_source_read','source_sha256','source_turn_count','independent_answer','per_option','answer_reason','answer_unique','hard_issues','soft_issues','quality_checks','cognitive_task','difficulty_auxiliary'}
allowed=required|{'independent_pairs'}
for i in items:
 f=b/'blind_results'/f"{i['review_id']}.json";err=[];review=None
 try:review=read(f)
 except Exception as ex:err.append('unparseable_or_missing_review:'+type(ex).__name__)
 agreement=False
 if review is not None:
  if set(review)-allowed:err.append('unexpected_fields:'+','.join(sorted(set(review)-allowed)))
  if required-set(review):err.append('missing_fields:'+','.join(sorted(required-set(review))))
  if review.get('review_id')!=i['review_id'] or review.get('reviewer_mode')!='same_model_blind':err.append('review_identity')
  if review.get('full_source_read') is not True or review.get('source_sha256')!=i['source_sha256'] or review.get('source_turn_count')!=i['source_turn_count']:err.append('source_binding')
  for k in ['independent_answer','per_option','hard_issues','soft_issues']:
   if not isinstance(review.get(k),list):err.append('wrong_field_type:'+k)
  if not isinstance(review.get('answer_unique'),bool):err.append('wrong_answer_unique_type')
  labels=set(i['label_to_option_id']);answer=review.get('independent_answer',[])
  if i['question_type']=='partial_order':
   answer=review.get('independent_pairs',[])
   if not isinstance(answer,list) or any(not isinstance(x,list) or len(x)!=2 or not set(x)<=labels for x in answer):err.append('invalid_pair_answer')
   else:agreement={tuple(x) for x in answer}=={tuple(x) for x in i['gold_labels']}
  elif not all(isinstance(x,str) for x in answer) or len(set(answer))!=len(answer) or not set(answer)<=labels:err.append('invalid_answer_labels')
  elif i['question_type']=='multi_select':agreement=set(answer)==set(i['gold_labels'])
  else:agreement=answer==i['gold_labels']
  if {o.get('label') for o in review.get('per_option',[]) if isinstance(o,dict)}!=labels:err.append('incomplete_per_option_review')
  allowed_judgments={'supported','contradicted','not_supported','ambiguous'}
  if any(o.get('judgment') not in allowed_judgments for o in review.get('per_option',[]) if isinstance(o,dict)):
   err.append('invalid_per_option_judgment_enum')
 reasons=list(mechanical[i['candidate_id']]['errors'])+err
 if review is not None:
  if not agreement:reasons.append('blind_gold_conflict')
  if review.get('answer_unique') is not True:reasons.append('answer_not_unique')
  if review.get('hard_issues'):reasons.append('blind_hard_issue')
 ok=not reasons
 rows.append({'candidate_id':i['candidate_id'],'review_id':i['review_id'],'root_id':i['root_id'],'arm':i.get('arm','C'),'status':'PASS' if ok else 'HOLD','content_pass':ok,'hold_reasons':reasons,'mechanical_pass':mechanical[i['candidate_id']]['mechanical_pass'],'review_protocol_valid':not err,'review_protocol_errors':err,'blind_gold_agreement':agreement,'blind_answer':answer if review is not None else None,'gold_labels':i['gold_labels'],'review_path':str(f),'review_sha256':sha(f) if f.exists() else None,'source_sha256':i['source_sha256'],'reviewer_mode':'same_model_blind','formal_training_approval_created':False,'solver_eligible':ok,'hard_issues':review.get('hard_issues',[]) if review else [],'soft_issues':review.get('soft_issues',[]) if review else []})
with (b/'QUALITY_FINAL.json').open('x') as f:json.dump({'created_at_utc':datetime.now(timezone.utc).isoformat(),'policy':'Hard content quality separate from novelty, difficulty and formal training admission. Frozen first blind answer; no repair or retry.','items':rows},f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'batch':str(b),'PASS':sum(r['content_pass'] for r in rows),'HOLD':sum(not r['content_pass'] for r in rows),'hold':[(r['candidate_id'],r['hold_reasons']) for r in rows if not r['content_pass']]}))
