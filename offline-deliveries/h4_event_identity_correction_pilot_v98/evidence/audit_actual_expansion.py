"""Audit one completed manifest-defined v98 expansion run; never repairs artifacts."""
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import argparse,hashlib,json
from pathlib import Path

RUN=None
HOME=None
def read(p):return json.loads(Path(p).read_text())
def sha(data):return hashlib.sha256(data).hexdigest()
def digest(value):return sha(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(Path(p).read_bytes())}
def compact_meta(metadata):
 return {k:metadata.get(k) for k in ('requested_model','returned_model','finish_reason','input_tokens','output_tokens','total_tokens','reasoning_tokens','http_status','completion_status','timeout','http_attempt_count','transport_retry_count')}

def main():
 global RUN,HOME
 parser=argparse.ArgumentParser();parser.add_argument("--run",required=True,type=Path)
 RUN=parser.parse_args().run.resolve();HOME=RUN.parents[2]
 assert (RUN/'REPORT.json').is_file(),'report_required'
 manifest,report=read(RUN/'MANIFEST.json'),read(RUN/'REPORT.json')
 attempt_responses=[];logical_responses=[];rows=[];bound_paths={RUN/'MANIFEST.json',RUN/'REPORT.json',RUN/'provider_profile.json',RUN/'feedback/COMPILED_FEEDBACK.json',RUN/'feedback/source_lock.json'}
 transport_errors=[];contract_rejections=[];semantic_rejections=[];feedback_failures=[];feedback_rows=[];incomplete=[]
 role_counts=Counter();caps=Counter();attempt_statuses=Counter();outcome_counts=Counter();schema_codes=Counter();other_failure_codes=Counter()
 for directory in sorted((RUN/'requests').iterdir()):
  if not directory.is_dir():continue
  request,proof=read(directory/'REQUEST.json'),read(directory/'PROOF.json')
  role=proof['role'];role_counts[role]+=1;caps[str(request.get('max_tokens'))]+=1
  row={'request':directory.name,'root_id':proof.get('root_id'),'role':role,'requested_model':request.get('model'),'max_tokens':request.get('max_tokens'),'capacity':{k:proof.get('capacity',{}).get(k) for k in ('status','input_tokens','maximum_output_tokens','truncation')},'proof_payload_sha_matches':proof.get('payload_sha256')==digest(request)}
  paths=[directory/'REQUEST.json',directory/'PROOF.json']
  attempts=[]
  for p in sorted(directory.glob('HTTP_ATTEMPT*.json')):
   a=read(p);attempts.append({'path':str(p),'http_attempt_number':a.get('http_attempt_number'),'request_attempt':a.get('request_attempt'),'same_payload_retry':a.get('same_payload_retry'),'actual_payload_equals_frozen':a.get('actual_payload_equals_frozen'),'payload_sha_matches':a.get('payload_sha256')==digest(request)})
   paths.append(p)
  row['http_attempts']=attempts
  for p in sorted(directory.glob('HTTP_RESULT_*.json')):
   a=read(p);attempt_statuses[str(a.get('http_status'))]+=1;outcome_counts[str(a.get('outcome'))]+=1;paths.append(p)
  for p in sorted(directory.glob('HTTP_RESPONSE_*.json')):
   response=read(p);meta=compact_meta(response.get('metadata',{}))
   attempt_responses.append({'request':directory.name,'role':role,'attempt_response':p.name,'requested_max_tokens':request.get('max_tokens'),**meta})
   paths.append(p)
  response_path=directory/'RESPONSE.json'
  if response_path.exists():
   response=read(response_path);meta=compact_meta(response.get('metadata',{}));paths.append(response_path)
   logical_responses.append({'request':directory.name,'role':role,'requested_max_tokens':request.get('max_tokens'),**meta})
   row['response']=meta
  error_path=directory/'ERROR.json'
  if error_path.exists():
   error=read(error_path);paths.append(error_path)
   transport_errors.append({'request':directory.name,'role':role,'error_type':error.get('error_type'),'message':error.get('message'),'metadata':compact_meta(error.get('metadata',{})),'error':bind(error_path)})
  if not response_path.exists() and not error_path.exists():incomplete.append(directory.name)
  for name in ('VALIDATION.json','REVIEW.json','ACCEPTANCE.json','EVENT_STATE_DECISION.json'):
   path=directory/name
   if not path.exists():continue
   saved=read(path);paths.append(path)
   if name=='EVENT_STATE_DECISION.json':
    if saved.get('status')=='REPLAN':semantic_rejections.append({'request':directory.name,'role':role,'record':name,'status':'REPLAN','diagnostics':saved.get('plan_diagnostics')})
    continue
   errors=saved.get('errors',[])
   if errors or saved.get('schema_valid') is False:
    entry={'request':directory.name,'role':role,'record':name,'status':saved.get('status'),'reason':saved.get('reason'),'schema_failures':saved.get('schema_failures',[]),'errors':errors}
    contract_rejections.append(entry)
    for error in errors:
     code=error.get('code') if isinstance(error,dict) else str(error);schema_codes[str(code)]+=1
    for code in saved.get('schema_failures',[]):schema_codes[code]+=1
   elif saved.get('status') in ('REJECT','MECHANICAL_FAIL'):
    # Preserve the saved reason. This generic branch alone cannot classify the
    # rejected proposition as a technical or semantic model failure.
    other_failure_codes[saved.get('reason') or saved.get('status')]+=1
    semantic_rejections.append({'request':directory.name,'role':role,'record':name,'status':saved.get('status'),'reason':saved.get('reason'),'failures':saved.get('failures',[]),'requires_manual_classification':True})
   if name=='VALIDATION.json' and saved.get('requires_replan') is True:
    semantic_rejections.append({'request':directory.name,'role':role,'record':name,'requires_replan':True,'diagnostics':saved.get('plan_diagnostics')})
  text=request['messages'][0]['content'];receipt=proof.get('feedback_receipt')
  row['feedback_marker_present']='[PREVIOUS_TRAINING_ROUND_EXPANSION_FEEDBACK]' in text
  row['feedback_receipt_present']=isinstance(receipt,dict)
  if role in ('planner','realizer'):
   result={'request':directory.name,'root_id':proof['root_id'],'role':role,'checks':{},'failures':[]}
   try:
    assert isinstance(receipt,dict),'feedback_receipt_missing'
    suffix_path=RUN/'feedback'/proof['root_id']/f"{proof['route']}__{role}.txt"
    suffix=suffix_path.read_text();paths.append(suffix_path)
    result['suffix']=bind(suffix_path);result['suffix_chars']=len(suffix)
    assert suffix and text.count(suffix)==1,'frozen_suffix_not_present_exactly_once'
    offset=text.index(suffix);prefix_text=text[:offset+len(suffix)]
    prefix_payload=deepcopy(request);prefix_payload['messages'][0]['content']=prefix_text
    tail_receipt=proof.get(role+'_tail_receipt') or {}
    checks={
     'root_route_role_match':all(receipt.get(k)==proof[k] for k in ('root_id','route','role')),
     'previous_training_feedback_declared':receipt.get('feedback_appended') is True and receipt.get('current_generation_feedback_used') is False,
     'suffix_hash_matches':receipt.get('suffix_sha256')==sha(suffix.encode()),
     'suffix_chars_match':receipt.get('suffix_chars')==len(suffix),
     'receipt_user_text_matches':receipt.get('request_user_text_sha256')==sha(prefix_text.encode()),
     'receipt_before_text_matches':receipt.get('before_user_text_sha256')==sha(text[:offset].encode()),
     'receipt_payload_matches':receipt.get('request_payload_sha256')==digest(prefix_payload),
     'tail_before_text_matches':tail_receipt.get('before_text_sha256')==sha(prefix_text.encode()),
     'tail_before_payload_matches':tail_receipt.get('before_payload_sha256')==digest(prefix_payload),
     'tail_final_text_matches':tail_receipt.get('final_text_sha256')==sha(text.encode()),
     'tail_final_payload_matches':tail_receipt.get('final_payload_sha256')==digest(request),
     'tail_bytes_match':tail_receipt.get('tail_sha256')==sha(text[len(prefix_text):].encode()),
     'source_lock_hash_matches':receipt.get('source_lock_sha256')==sha((RUN/'feedback/source_lock.json').read_bytes()),
    }
    result['checks']=checks;result['failures']=[key for key,ok in checks.items() if not ok]
   except Exception as error:result['failures'].append(str(error))
   result['status']='FAIL' if result['failures'] else 'PASS'
   if result['failures']:feedback_failures.append(result)
   feedback_rows.append(result)
  row['stage_records']=[p.name for p in paths if p.name in ('VALIDATION.json','REVIEW.json','ACCEPTANCE.json','EVENT_STATE_DECISION.json')]
  rows.append(row);bound_paths.update(paths)
 actual_output=[r['output_tokens'] for r in attempt_responses if type(r.get('output_tokens')) is int]
 finish=Counter(str(r.get('finish_reason')) for r in attempt_responses)
 lengths=[r for r in attempt_responses if r.get('finish_reason')=='length']
 missing_payload_checks=[r['request'] for r in rows if not r['proof_payload_sha_matches'] or any(not a['actual_payload_equals_frozen'] or not a['payload_sha_matches'] for a in r['http_attempts'])]
 profile=read(RUN/'provider_profile.json')
 root_outcomes=[{'root_id':r['root_id'],'route':r['route'],'accepted_scenes':r.get('accepted_scenes'),'added_turns':r.get('added_turns'),'failures':r.get('failures',[])} for r in report['routes']]
 record={
  'schema_version':'h4.v98.transport_feedback_audit.v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'run':str(RUN),
  'audit_scope':'Completed actual request/HTTP artifacts and exact frozen previous-round feedback presence; no model response or original verdict modification.',
  'status':'FAIL' if incomplete or feedback_failures or missing_payload_checks else 'PASS',
  'no_new_model_calls':True,'production_code_edits':False,'model_decisions_modified':False,'semantic_certified':False,
  'summary':{'logical_requests':len(rows),'saved_logical_responses':len(logical_responses),'actual_http_attempts':sum(len(r['http_attempts']) for r in rows),'attempt_response_count':len(attempt_responses),'roles':dict(role_counts),'http_statuses':dict(attempt_statuses),'http_outcomes':dict(outcome_counts),'finish_reasons_attempt_responses':dict(finish),'max_tokens_in_requests':dict(caps),'observed_output_tokens_min':min(actual_output) if actual_output else None,'observed_output_tokens_max':max(actual_output) if actual_output else None,'length_response_count':len(lengths),'output_tokens_equal_requested_cap':sum(type(r.get('output_tokens')) is int and r['output_tokens']==r['requested_max_tokens'] for r in attempt_responses),'provider_profile_output_limit':profile.get('output_limit'),'provider_profile_max_tokens':profile.get('request_parameters',{}).get('max_tokens'),'transport_error_count':len(transport_errors),'saved_contract_rejection_records':len(contract_rejections),'feedback_expected_actual_planner_realizer_requests':len(feedback_rows),'feedback_exact_chain_verified':sum(r['status']=='PASS' for r in feedback_rows),'feedback_failure_count':len(feedback_failures),'incomplete_requests':incomplete,'payload_binding_failures':missing_payload_checks},
  'length_responses':lengths,'transport_errors':transport_errors,'saved_contract_rejections':contract_rejections,'saved_contract_failure_codes':dict(schema_codes),'other_saved_rejection_codes':dict(other_failure_codes),'semantic_or_unclassified_saved_rejections':semantic_rejections,'root_outcomes':root_outcomes,'feedback_checks':feedback_rows,'requests':rows,'actual_attempt_responses':attempt_responses,
  'feedback_scope_note':'Frozen previous-training-round suffixes are expected on actual planner/realizer requests. Dataset extraction and state/review requests have their own source/context contracts and are not expected to append these writer-role suffixes. Presence/binding does not prove the model followed the feedback.',
  'transport_scope_note':'Only requested cap, observed provider usage and saved finish reasons are reported. Service output/context limits remain UNKNOWN. A local proxy capacity PASS is not proof of exact service token capacity; absence of length responses does not measure the provider hard limit.',
  'failure_scope_note':'Transport errors, unfinished output and saved schema/evidence contract rejections are listed separately from semantic outcomes. Reviewer invalid citations do not establish dialogue quality failure. Repeated ACCEPTANCE/REVIEW records may refer to one failed review attempt; rejection record counts are not unique root counts.',
  'bindings':[bind(p) for p in sorted(bound_paths)],'script':bind(Path(__file__)),
 }
 from actual_expansion_integrity import check as check_integrity
 integrity=check_integrity(RUN)
 record['source_feedback_reservation_integrity']=integrity
 if integrity['failures']:record['status']='FAIL'
 record['summary'].update({'inventory_before_identities':integrity['inventory_before_identities'],'inventory_after_identities':integrity['inventory_after_identities'],'newly_permanently_reserved_identities':integrity['newly_permanently_reserved_identities'],'actual_full_history_writer_requests':len(integrity['actual_writer_full_history_checks']),'integrity_failure_count':len(integrity['failures'])})
 path=HOME/'evidence/TRANSPORT_FEEDBACK_AUDIT.json'
 with path.open('x') as stream:json.dump(record,stream,ensure_ascii=False,indent=2);stream.write('\n')
 print(json.dumps({'status':record['status'],'summary':record['summary'],'artifact':bind(path)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
