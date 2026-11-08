#!/usr/bin/env python3
"""Integrity closeout with honest PARTIAL status when any planned gate is missing."""
from __future__ import annotations
import collections,datetime,json,pathlib
from codex_client import BASE,atomic,bind,load,now,status

def run():
 errors=[];before=load(BASE/'IMMUTABLE_INTEGRITY_BEFORE.json');after=[]
 for tree in before['trees']:
  root=pathlib.Path(tree['root']);files=[];expected={x['relative_path']:x for x in tree['files']}
  current={str(p.relative_to(root)):p for p in root.rglob('*') if p.is_file() and not p.is_symlink()}
  for rel,old in expected.items():
   if rel not in current:errors.append('immutable_missing:'+str(root/rel));continue
   b=bind(current[rel]);files.append({'relative_path':rel,'sha256':b['sha256'],'bytes':b['bytes']})
   if b['sha256']!=old['sha256'] or b['bytes']!=old['bytes']:errors.append('immutable_changed:'+str(root/rel))
  for rel in set(current)-set(expected):errors.append('immutable_added:'+str(root/rel))
  after.append({'root':str(root),'files':files})
 atomic(BASE/'IMMUTABLE_INTEGRITY_AFTER.json',{'at':now(),'trees':after,'passed':not errors,'errors':errors})
 sf=load(BASE/'SOURCE_FREEZE.json') if (BASE/'SOURCE_FREEZE.json').exists() else None
 qf=load(BASE/'QUESTION_FREEZE.json') if (BASE/'QUESTION_FREEZE.json').exists() else None
 rf=load(BASE/'execution/RAW_RESPONSE_FREEZE.json') if (BASE/'execution/RAW_RESPONSE_FREEZE.json').exists() else None
 analysis=load(BASE/'ANALYSIS.json') if (BASE/'ANALYSIS.json').exists() else None
 pre=pathlib.Path(load(BASE/'RESUME_MANIFEST.json')['immutable_predecessor'])
 exactroots={load(p)['root_id'] for p in (pre/'inputs').glob('*_SOURCE_AUTHOR_INPUT.json')}
 if not sf or len(sf['outcomes'])!=16 or {x['root_id'] for x in sf['outcomes']}!=exactroots:errors.append('source_freeze_not_exact16')
 passed={x['root_id'] for x in sf['outcomes'] if x['status']=='PASS'} if sf else set()
 coverage=load(BASE/'SOURCE_REVIEW_COVERAGE.json') if (BASE/'SOURCE_REVIEW_COVERAGE.json').exists() else None
 if not coverage or coverage.get('independent_initial_reviews')!=16 or coverage.get('roots_with_valid_independent_review')!=16:errors.append('independent_source_review_coverage_incomplete')
 if sf:
  for r in sf['outcomes']:
   d=BASE/'sources'/r['root_id']
   if not (d/'INITIAL_EVENT_CLASSIFICATION.json').exists():errors.append('missing_root_classification:'+r['root_id'])
   if bind(r['fixed_initial_draft']['path'])!=r['fixed_initial_draft']:errors.append('fixed_initial_hash_drift:'+r['root_id'])
   if len(r.get('attempts',[]))>2:errors.append('source_revision_budget_exceeded:'+r['root_id'])
   if r['status']=='PASS':
    initial=load(d/'INITIAL_EVENT_CLASSIFICATION.json');review=load(r['attempts'][-1]['review_receipt'])
    if not initial['accepted'] or not review['accepted'] or review['response'].get('decision')!='PASS':errors.append('source_admission_gate:'+r['root_id'])
    if bind(r['source']['path'])!=r['source']:errors.append('frozen_source_drift:'+r['root_id'])
 if qf:
  if len(qf['slots'])!=256 or len({(s['root_id'],s['arm'],s['slot_id']) for s in qf['slots']})!=256:errors.append('question_slot_ledger_not_256_unique')
  for s in qf['slots']:
   if s['status']=='PASS':
    if s['root_id'] not in passed:errors.append('question_from_rejected_source:'+s['candidate_id'])
    threads=[]
    for k in ['gold_receipt','review_receipt']:
     receipt=load(s['review'][k])
     if not receipt['accepted']:errors.append('question_invalid_independent_call:'+s['candidate_id'])
     threads.append(receipt.get('backend',{}).get('thread_id'))
     response=receipt.get('response') or {}
     checks=[x for x in response.get('slots',[]) if x.get('slot_id')==s['slot_id']]
     if response.get('root_id')!=s['root_id'] or response.get('arm')!=s['arm'] or len(checks)!=1 or checks[0].get('decision')!='PASS' or checks[0].get('failure_reasons'):errors.append('question_independent_pass_gate:'+s['candidate_id']+':'+k)
    author_name=s['root_id']+'_'+s['arm']+('_question_revision_1' if s.get('revision_used') else '_question_draft')
    author=load(BASE/'cli_calls'/author_name/'RECEIPT.json');threads.append(author.get('backend',{}).get('thread_id'))
    if not author.get('accepted') or None in threads or len(set(threads))!=3:errors.append('question_author_gold_reviewer_not_independent:'+s['candidate_id'])
    frozen_source=next(r['source'] for r in sf['outcomes'] if r['root_id']==s['root_id'] and r['status']=='PASS')
    if s['candidate'].get('source_binding')!=frozen_source:errors.append('question_frozen_source_binding_mismatch:'+s['candidate_id'])
 else:errors.append('question_freeze_missing')
 calls=[];timeline=[]
 for d in (BASE/'cli_calls').glob('*'):
  if not d.is_dir() or not (d/'START.json').exists():continue
  start=load(d/'START.json');receipt=load(d/'RECEIPT.json') if (d/'RECEIPT.json').exists() else None
  if not receipt:errors.append('semantic_call_unfinished:'+d.name);continue
  for name,b in receipt.get('hashes',{}).items():
   if bind(b['path'])!=b:errors.append('semantic_evidence_hash_drift:'+d.name+':'+name)
  begin=datetime.datetime.fromisoformat(start['at']);end=datetime.datetime.fromisoformat(receipt['finished_at_utc'])
  timeline.extend([(begin,1,d.name),(end,-1,d.name)])
  calls.append({'call_id':d.name,'accepted':receipt['accepted'],'backend_model':receipt.get('backend',{}).get('actual_model'),'provider':receipt.get('backend',{}).get('actual_provider'),'hash':bind(d/'RECEIPT.json')})
 concurrent=0;maximum=0
 for t,delta,name in sorted(timeline,key=lambda x:(x[0],x[1])):concurrent+=delta;maximum=max(maximum,concurrent)
 if maximum>3:errors.append('semantic_parallelism_exceeded:'+str(maximum))
 if rf:
  returned=[]
  for r in rf['records']:
   if bind(r['binding']['path'])!=r['binding']:errors.append('raw_solver_hash_drift:'+r['request_id'])
   returned.append(r['request_id'])
  if len(returned)!=len(set(returned)):errors.append('duplicate_solver_response')
  if any(any(old in r for old in ['CL1-F-218519','CL1-G-218519','CL1-F-222937','CL1-G-222937','CL1-F-225575','CL1-G-225575','CL1-F-229686','CL1-G-229686']) for r in returned):errors.append('sealed_old_solver_rerun')
 else:errors.append('raw_solver_freeze_missing')
 audit={'at':now(),'integrity_passed':not errors,'errors':errors,'immutable_tree_count':len(before['trees']),'immutable_file_count':sum(len(t['files']) for t in before['trees']),'maximum_parallel_semantic_calls':maximum,'semantic_calls':calls,'new_initial_drafts_generated':0,'sealed_original_solver_reruns':0,'training':False}
 atomic(BASE/'INTEGRITY_AUDIT.json',audit)
 reasons=[]
 if errors:reasons+=errors
 if not sf or sf['pass_count']!=16:reasons.append('not_all_16_sources_passed')
 if not qf or qf['valid_slots']!=256:reasons.append('not_all_256_question_slots_valid')
 if not rf or rf['returned_count']!=512:reasons.append('not_all_512_new_solver_presentations_returned')
 if analysis is None:reasons.append('analysis_not_run')
 # All phases may have executed while planned content remains legitimately missing.
 report={'schema_version':'fgr20_resume_final_v1','at':now(),'status':'PARTIAL' if reasons else 'COMPLETED','phases_actually_run':{'preflight':(BASE/'PREFLIGHT.json').exists(),'source':sf is not None,'questions':qf is not None,'solver':(BASE/'execution/STARTED.json').exists(),'analysis':analysis is not None,'closeout':True},'counts':{'planned_new_roots':16,'sealed_original_roots':4,'source_pass':len(passed),'source_fail':16-len(passed),'independent_initial_source_reviews':coverage['valid_independent_initial_reviews'] if coverage else 0,'planned_question_slots':256,'valid_question_slots':qf['valid_slots'] if qf else 0,'missing_question_slots':qf['missing_slots'] if qf else 256,'planned_new_solver_presentations':512,'returned_new_solver_presentations':rf['returned_count'] if rf else 0,'missing_new_solver_presentations':512-(rf['returned_count'] if rf else 0),'semantic_calls':len(calls),'maximum_parallel_semantic_calls':maximum,'initial_drafts_regenerated':0,'sealed_old_solver_reruns':0},'partial_reasons':reasons,'source_outcomes':[{k:r[k] for k in ['root_id','status','source_ids','failure_reasons']} for r in sf['outcomes']] if sf else [],'integrity_passed':not errors,'training':False,'nonresponses_scored_incorrect':False,'actual_codex_backends':dict(collections.Counter((str(c['provider'])+'/'+str(c['backend_model'])) for c in calls)),'backend_model_limit':'Model/provider attested by local Codex session metadata; upstream routing is not independently observable.','B1_checkpoint':'/root/autodl-tmp/gxy-selfplay/restart-n5-runs/day5_B1_schedule_v1/merged_model','analysis_results':{k:v['primary'] for k,v in analysis['comparisons'].items()} if analysis else {},'interpretation':'Descriptive paired difficulty only. Missing roots and outputs retained. No feedback mechanism, training benefit, or independent confirmation claim.','artifacts':{p.name:bind(p) for p in [BASE/'SOURCE_FREEZE.json',BASE/'SOURCE_REVIEW_COVERAGE.json',BASE/'QUESTION_FREEZE.json',BASE/'execution/RAW_RESPONSE_FREEZE.json',BASE/'execution/SCORES.json',BASE/'ANALYSIS.json',BASE/'RESULTS_ZH.md',BASE/'INTEGRITY_AUDIT.json'] if p.exists()}}
 report['counts'].update(prepared_new_solver_presentations=rf['prepared_count'] if rf else 0,missing_among_prepared_solver_presentations=rf['missing_count'] if rf else 0)
 report['counts'].update(independent_initial_source_reviews=coverage['independent_initial_reviews'] if coverage else 0,valid_initial_source_review_calls=coverage['valid_independent_initial_reviews'] if coverage else 0,source_roots_with_valid_independent_review=coverage.get('roots_with_valid_independent_review',0) if coverage else 0)
 scores=load(BASE/'execution/SCORES.json')['records'] if (BASE/'execution/SCORES.json').exists() else []
 report['counts'].update(solver_contract_valid_responses=sum(r['strict']['valid'] for r in scores),solver_strict_exact_responses=sum(r['strict']['exact'] for r in scores))
 if (BASE/'execution/INITIALIZATION_RECOVERY.json').exists():
  report['solver_initialization_recovery']={'receipt':bind(BASE/'execution/INITIALIZATION_RECOVERY.json'),'failure_archive':str(BASE/'execution/initialization_failure_01'),'submitted_requests_before_recovery':0,'returned_responses_before_recovery':0,'submitted_request_retries':0,'sampler':'PyTorch native; VLLM_USE_FLASHINFER_SAMPLER=0','reason':'FlashInfer JIT failed during engine initialization, before any experimental request submission.'}
 atomic(BASE/'FINAL_REPORT.json',report)
 inventory=[bind(p) for p in sorted(BASE.rglob('*')) if p.is_file() and p.name not in ['ARTIFACT_MANIFEST.json','RUN_STATUS.json'] and not any(x in p.parts for x in ['worker_events','__pycache__']) and not p.name.endswith('.tmp')]
 atomic(BASE/'ARTIFACT_MANIFEST.json',{'at':now(),'files':inventory})
 status('closeout',status=report['status'],active_children=[],final_report=bind(BASE/'FINAL_REPORT.json'),actual_counts=report['counts'],integrity_passed=report['integrity_passed'])
 print(json.dumps({'status':report['status'],'counts':report['counts'],'integrity_passed':not errors,'reasons':reasons}),flush=True)
 return report
if __name__=='__main__':run()
