#!/usr/bin/env python3
"""Complete independent review coverage for drafts already rejected by transport policy.

The source freeze and exclusions remain immutable. This adds missing audit
reviews only, after every question semantic call has finished.
"""
import concurrent.futures,os,pathlib,time
from codex_client import BASE,atomic,bind,call,load,now,start_token,status
from source_resume import PRE,mechanical,review_prompt,review_ok

def review_rejected(outcome):
 root=outcome['root_id'];d=BASE/'sources'/root
 inp=load(PRE/'inputs'/(root+'_SOURCE_AUTHOR_INPUT.json'))
 parent=load(inp['parent_source_binding']['path'])
 candidate=load(outcome['fixed_initial_draft']['path'])
 native=[]
 for n in inp['native_events']:
  if bind(n['binding']['path'])!=n['binding']:raise ValueError('native binding drift')
  x=load(n['binding']['path'])
  native.append({'source_id':n['source_id'],'scene_ref':n['scene_ref'],'turns':[{k:t.get(k) for k in ['turn_id','speaker','utterance']} for t in x['turns']]})
 errors=mechanical(candidate,inp,parent)+['BINDING_EVENT_POLICY_REJECTION:'+str(x) for x in outcome['failure_reasons']]
 prompt=review_prompt(inp,parent,native,candidate,errors)
 prompt+='\nThe supplied initial event stream has already failed the binding event policy. Independently review the candidate content and provide the required exact source quotes, but the overall decision must be FAIL because of that unrecoverable provenance failure. This review cannot admit or revise the source. State both the transport rejection and any content issues concisely.'
 receipt=call(root+'_source_rejected_initial_review',prompt)
 obj=receipt.get('response');validation=[]
 if not receipt['accepted']:validation+=receipt.get('failure_reasons',[])
 if not isinstance(obj,dict):validation.append('missing_review_object');obj={}
 if obj.get('root_id')!=root or obj.get('decision')!='FAIL':validation.append('rejected_source_review_identity_or_decision')
 if not obj.get('failure_reasons'):validation.append('missing_rejection_reasons')
 turns={t['turn_ref']:t for t in parent['turns']}
 turns.update({t['turn_ref']:t for s in candidate['scenes'] for t in s['turns']})
 quotes=obj.get('source_quotes',[])
 if not isinstance(quotes,list) or not quotes:validation.append('missing_source_quotes')
 else:
  for q in quotes:
   if not isinstance(q,dict) or q.get('turn_ref') not in turns or not q.get('quote') or q['quote'] not in turns[q['turn_ref']]['text']:validation.append('source_quote_mismatch')
 result={'at':now(),'root_id':root,'frozen_source_outcome':'FAIL','source_freeze':bind(BASE/'SOURCE_FREEZE.json'),'fixed_initial_draft':outcome['fixed_initial_draft'],'review_receipt':bind(pathlib.Path(receipt['directory'])/'RECEIPT.json'),'review':obj,'validation_errors':validation,'independent_review_valid':not validation,'revision_generated':False}
 atomic(d/'REJECTED_STREAM_REVIEW.json',result)
 status(source_rejected_initial_reviews_finished=len(list((BASE/'sources').glob('*/REJECTED_STREAM_REVIEW.json'))))
 return result

def refresh_coverage():
 sf=load(BASE/'SOURCE_FREEZE.json')
 rows=[]
 for x in sf['outcomes']:
  if x.get('attempts'):
   parent=load(load(BASE/'sources'/x['root_id']/'INITIAL_EVIDENCE.json')['parent_binding']['path']);attempts=[]
   for a in x['attempts']:
    receipt=load(a['review_receipt']);candidate=load(a['candidate']['path'])
    errors=[e for e in review_ok(receipt,candidate,parent) if not e.startswith('semantic_')]
    attempts.append({'attempt':a['attempt'],'review_receipt':bind(a['review_receipt']),'accepted':receipt['accepted'],'decision':(receipt.get('response') or {}).get('decision'),'compact_review_valid':not errors,'validation_errors':errors})
   first=attempts[0]
   rows.append({'root_id':x['root_id'],'review_receipt':first['review_receipt'],'accepted':first['compact_review_valid'],'decision':first['decision'],'source_status':x['status'],'review_attempts':attempts,'has_valid_independent_review':any(a['compact_review_valid'] for a in attempts)})
  else:
   r=load(BASE/'sources'/x['root_id']/'REJECTED_STREAM_REVIEW.json')
   rows.append({'root_id':x['root_id'],'review_receipt':r['review_receipt'],'accepted':r['independent_review_valid'],'decision':r['review'].get('decision'),'source_status':x['status'],'has_valid_independent_review':r['independent_review_valid']})
 coverage={'at':now(),'source_freeze':bind(BASE/'SOURCE_FREEZE.json'),'planned_drafts':16,'independent_initial_reviews':len(rows),'valid_independent_initial_reviews':sum(r['accepted'] for r in rows),'roots_with_valid_independent_review':sum(r['has_valid_independent_review'] for r in rows),'rows':rows,'frozen_outcomes_changed':False,'note':'Initial review stream failures remain rejected. An existing valid allowed recheck counts toward per-root compact-review coverage, not initial-call validity.'}
 atomic(BASE/'SOURCE_REVIEW_COVERAGE.json',coverage)
 status(source_initial_reviews_finished=len(rows),source_initial_reviews_valid=coverage['valid_independent_initial_reviews'],source_roots_with_valid_independent_review=coverage['roots_with_valid_independent_review'])
 return coverage

def main():
 atomic(BASE/'process_registry'/'source_review_coverage.json',{'pid':os.getpid(),'start_token':start_token(os.getpid()),'kind':'source_review_coverage','started_at_utc':now()})
 while not (BASE/'QUESTION_FREEZE.json').exists():
  pipeline=load(BASE/'process_registry'/'pipeline.json')
  if start_token(pipeline['pid'])!=pipeline['start_token']:raise RuntimeError('pipeline stopped before question freeze; semantic concurrency gate unavailable')
  time.sleep(5)
 sf=load(BASE/'SOURCE_FREEZE.json');missing=[x for x in sf['outcomes'] if not x.get('attempts')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  list(pool.map(review_rejected,missing))
 coverage=refresh_coverage()
 print(coverage,flush=True)
if __name__=='__main__':main()
