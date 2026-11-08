"""Temporary synthetic transport runs through the actual future scheduler; no network."""
from contextlib import nullcontext
import io,json,tempfile,urllib.error
from pathlib import Path
from unittest.mock import patch
from ..core import atomic_json,sha256_file
from .dependencies import loaded_repository_files

def exercise(source):
 from . import runtime,dependencies
 from .test_runtime import seed_run,synthetic_candidates,FakeTokenPreflight
 from ..h4_r02_feedback_transfer_p2_2_v1 import token_preflight
 scenarios=[]
 for scenario in ('success','ordering_missing','malformed_json','known_429_exhausted','timeout_unknown','caught_verifier_error','repair_capacity_block','retry_freeze_drift'):
  with tempfile.TemporaryDirectory(prefix='writer-scms-quality-offline-') as tmp:
   target=seed_run(tmp,source_run=source);calls=[];verifications=[];token_instances=[]
   class Response:
    status=200
    def __init__(self,content):self.content=content
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self):return json.dumps({'choices':[{'message':{'content':self.content},'finish_reason':'stop'}],'usage':None}).encode()
   def fake_http(request,**kwargs):
    payload=json.loads(request.data)
    assert payload['model']=='Qwen/Qwen3-14B' and payload['max_tokens']==8192 and payload['temperature']==.1
    assert payload['enable_thinking'] is False and 'seed' not in payload and kwargs['timeout']==180
    assert len(payload['messages'])==1 and payload['messages'][0]['role']=='user'
    assert request.full_url=='https://api.siliconflow.cn/v1/chat/completions'
    prompt=payload['messages'][0]['content'];calls.append({'max_output_tokens':payload['max_tokens'],'repair':'REPAIR THE FOLLOWING BATCH' in prompt,'intent_guidance_present':'CONTRACT GUIDANCE:' in prompt,'quality_block_count':prompt.count('BEGIN_SCMS_QUALITY_GUIDANCE_V1'),'verifier_checks_before_send':len(verifications),'token_checks_before_send':sum(len(t.calls) for t in token_instances)})
    if scenario=='timeout_unknown':raise TimeoutError('OFFLINE_SYNTHETIC_UNKNOWN')
    if scenario=='known_429_exhausted' and len(calls)<=3:raise urllib.error.HTTPError('https://example.invalid',429,'OFFLINE_REJECTION',{'Retry-After':'0'},io.BytesIO(b''))
    if scenario=='retry_freeze_drift':raise urllib.error.HTTPError('https://example.invalid',429,'OFFLINE_RETRY_REJECTION',{'Retry-After':'0'},io.BytesIO(b''))
    if scenario=='malformed_json':return Response('{"candidates": [')
    counts=(2,1,0) if scenario in ('ordering_missing','repair_capacity_block') and not calls[-1]['repair'] else (2,1,1)
    return Response(json.dumps(synthetic_candidates(prompt,counts),ensure_ascii=False))
   def verifier(*args):
    verifications.append(True)
    if scenario=='caught_verifier_error' and len(verifications)>=3:raise ValueError('OFFLINE_FROZEN_IDENTITY_GUARD')
    if scenario=='retry_freeze_drift' and len(verifications)>=5:raise ValueError('OFFLINE_RETRY_FROZEN_IDENTITY_GUARD')
    return {'code':[]}
   class Tokens(FakeTokenPreflight):
    def __init__(self,*args):
     super().__init__(block_repair=scenario=='repair_capacity_block');token_instances.append(self)
   stopped=False;reason=None;result=None
   with dependencies.forbid_network(),patch.object(dependencies,'repository_execution_guard',lambda _:nullcontext()),patch.object(token_preflight,'TokenPreflight',Tokens),patch('urllib.request.urlopen',side_effect=fake_http),patch('time.sleep'):
    try:result=runtime.execute_live(target,target/'config.json',verify_frozen=verifier)
    except BaseException as error:stopped=True;reason=type(error).__name__+':'+str(error)
    calls_before_resume=len(calls)
    preserved={str(p.relative_to(target)):sha256_file(p) for p in target.glob('arms/**/*') if p.is_file()}
    if not stopped:
     again=runtime.execute_live(target,target/'config.json',verify_frozen=verifier)
     assert again==result and len(calls)==calls_before_resume
    else:
     try:
      runtime.execute_live(target,target/'config.json',verify_frozen=verifier)
     except BaseException:
      pass
     else:
      raise AssertionError('guarded run unexpectedly resumed')
     assert len(calls)==calls_before_resume
   reservations=len(list(target.glob('arms/**/ledger/requests/*.json')))
   expected={'success':(12,12,False),'ordering_missing':(24,24,False),'malformed_json':(24,24,False),'known_429_exhausted':(15,13,False),'timeout_unknown':(1,1,True),'caught_verifier_error':(0,0,True),'repair_capacity_block':(1,1,True),'retry_freeze_drift':(1,1,True)}[scenario]
   checks=dict(expected_calls_and_stop=(len(calls),reservations,stopped)==expected,
    every_mock_send_reverified=all(c['verifier_checks_before_send']>calls[i-1]['verifier_checks_before_send'] and c['token_checks_before_send']>calls[i-1]['token_checks_before_send'] for i,c in enumerate(calls) if i),
    first_send_guarded=not calls or (calls[0]['verifier_checks_before_send']>=4 and calls[0]['token_checks_before_send']>=2),
    resume_artifacts_preserved=preserved=={str(p.relative_to(target)):sha256_file(p) for p in target.glob('arms/**/*') if p.is_file()},
    observed_http_attempt_count=sum(e.get('event')=='http_attempt_started' for e in runtime.rows(target/'audit/http_events.jsonl'))==len(calls),
    both_arms_keep_intent_guidance=all(c['intent_guidance_present'] for c in calls),
    quality_block_never_duplicated=all(c['quality_block_count']<=1 for c in calls))
   stage_rows=runtime.rows(target/'reports/runtime_stage_records.jsonl')
   final_rows=runtime.rows(target/'reports/runtime_final_candidates.jsonl')
   if not stopped:
    expected_final=0 if scenario=='malformed_json' else 48
    checks.update(stage_rows_match_logical_calls=len(stage_rows)==reservations,
     final_rows_match_expected=len(final_rows)==expected_final,
     analysis_ids_unique=len({tuple(r['analysis_identity']) for r in final_rows})==len(final_rows),
     primary_and_repair_analysis_separate=all(r['stage']=='repair' for r in final_rows) if scenario=='ordering_missing' else True,
     stage_parsed_and_identity_archived=all('raw_parsed' in r and 'parse_error' in r and r['global_request_id'] and r['raw_response_path'] for r in stage_rows),
     global_stages_unique=len({r['global_request_id'] for r in stage_rows})==reservations,
     both_arm_stage_and_final_intent_analysis_applicable=all(r.get('intent_guidance') is True and r.get('contract_alignment_analysis')=='AVAILABLE_FOR_OFFLINE_CONTENT_REVIEW' for r in stage_rows+final_rows),
     all_root_arm_finalizers_exist=all((runtime.root_path(target,arm,root)/name).is_file() for arm in runtime.ARMS for root in runtime.ROOTS for name in ('candidate_generation.json','candidate_pool.json','validity_audit.json','candidates_valid.json','permutations.json')),
     result_minima_calls=result['primary_calls']==12 and result['repair_calls']==reservations-12)
    final_total=sum(len(runtime.read(runtime.root_path(target,arm,root)/'candidates_valid.json')) for arm in runtime.ARMS for root in runtime.ROOTS)
    checks['final_sidecar_matches_root_arm_count']=final_total==len(final_rows)
   else:
    expected_reason={'timeout_unknown':'unknown_service_completion','caught_verifier_error':'OFFLINE_FROZEN_IDENTITY_GUARD','repair_capacity_block':'TOKEN_CONTEXT_EXCEEDED','retry_freeze_drift':'OFFLINE_RETRY_FROZEN_IDENTITY_GUARD'}[scenario]
    checks.update(expected_guard_reason=expected_reason in reason,
     stopped_before_any_final_qa=not final_rows,
     partial_stage_archive_present=bool(list(target.glob('arms/**/stage_calls/*/*.json'))))
   scenarios.append(dict(scenario=scenario,status='PASS' if all(checks.values()) else 'FAIL',checks=checks,mock_http_attempts=len(calls),synthetic_reservations=reservations,synthetic_stage_rows=len(stage_rows),synthetic_final_candidates=len(final_rows),stopped=stopped,reason=reason,verifier_checks=len(verifications),token_checks=sum(len(t.calls) for t in token_instances),resume_resends=len(calls)-calls_before_resume,actual_network_calls=0,synthetic_output_retained_as_experiment_QA=False))
 report=dict(status='PASS' if all(x['status']=='PASS' for x in scenarios) else 'FAIL',scenarios=scenarios,repository_modules=sorted(map(str,loaded_repository_files())),method='New execute_live -> same execute_schedule -> shared Writer batch -> original client/ledger/normalizer/finalizer; only external urllib transport plus pre-freeze identity/token prerequisites replaced in temporary diagnostic run. Actual primary tokenizer/repair capacity separately checked locally.',external_calls=0)
 atomic_json(Path(source)/'audit/dry_run_report.json',report)
 return report
