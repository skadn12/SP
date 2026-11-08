"""Two authorized fixed Gate requests; reuse bounded runtime and validator.

No Planner, outline, Realizer, repair, retry, or Memory API stage exists here.
The Qwen tokenizer/context guard is only a local proxy for the school service;
service capacity/reasoning and backend identity remain explicitly unverified.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

FROZEN_WORKTREE = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-planner-gate-source-bound-v1')
sys.path.insert(0, str(FROZEN_WORKTREE))
from scripts.current_method.restart_n5_spell.core import atomic_json, atomic_text
from scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.transport import ProfileClient, build_payload, resolve_credential
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_runtime import BoundedDevelopmentCaller, DevelopmentContentError, DevelopmentRuntimeError, capacity_check
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_validate import check_freeze, cost_summary, sha, tokenizer_from
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.gate_source_policy import validate_gate

SOURCE = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_planner_gate_source_bound_v1/prepared_run')
DEFAULT_RUN = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_gate_tju_same_requests_v1_live_1')
CONFIG = Path('/root/autodl-tmp/gxy-selfplay/current_method_github/configs/h4_writer_model_comparison_v1/strong_model.json')
ROOTS = ['cc-train-episode-198480', 'cc-train-episode-122711']
read = lambda p: json.loads(Path(p).read_text())
now = lambda: datetime.now(timezone.utc).isoformat()


class FixedGateCaller(BoundedDevelopmentCaller):
    def _capacity(self, messages, max_tokens):
        result = super()._capacity(messages, max_tokens)
        result.update(counting_tokenizer='FROZEN_QWEN3_14B_PROXY_NOT_TJU_EXACT',
                      service_context_capacity='UNKNOWN',
                      context_window_semantics='LOCAL_GUARD_NOT_SERVICE_LIMIT',
                      service_reasoning_budget='UNKNOWN',
                      service_exception='previous_user_exception_retained; no truncation or parameter fallback')
        return result


def prepare(run_root=DEFAULT_RUN, *, tokenizer=None):
    run_root = Path(run_root)
    if run_root.exists():
        raise FileExistsError('fresh_comparison_run_required')
    previous = read(SOURCE/'freeze_manifest.json')
    check_freeze(previous)
    tokenizer = tokenizer if tokenizer is not None else tokenizer_from(previous['tokenizer_path'])
    config = read(CONFIG)
    api = config['api']
    if api['endpoint'] != 'https://ai.tju.edu.cn/api/v3/chat/completions' or api['model'] != 'tju-llm':
        raise ValueError('school_service_identity_changed')
    profile = deepcopy(previous['public_profile'])
    profile.update(profile_id='tju_fixed_gate_two_requests_v1', provider='tju',
                   model=api['model'], base_url=api['base_url'], endpoint=api['endpoint'],
                   api_key_env=api['api_key_env'], credential_file=api['credential_file'],
                   output_limit=8192, timeout_seconds=600,
                   user_reported_backend_model=config['user_reported_backend_model'],
                   backend_identity_status='USER_REPORTED_NOT_SERVICE_VERIFIED',
                   service_context_limit='UNKNOWN', service_output_limit='UNKNOWN',
                   capacity_policy='QWEN_TOKENIZER_LOCAL_40960_GUARD_ONLY; SCHOOL_CAPACITY_UNKNOWN',
                   comparison='MODEL_AND_SERVICE_CONFIGURATION_NOT_PURE_MODEL_SIZE')
    profile.pop('credential_source_config',None)
    # Current actual Gate used .25, not the older Writer experiment's .1.
    profile['request_parameters']={'temperature':0.25, 'max_tokens':8192,
                                   'response_format':{'type':'json_object'}}
    profile['token_accounting']['note']='Full messages counted with frozen Qwen3 tokenizer as local proxy only. School tokenizer, service capacity and default reasoning are UNKNOWN. Request max_tokens=8192 with no extra reasoning parameter.'
    # Validate private credential locally without storing/printing/hashing it.
    credential = resolve_credential(profile)
    del credential
    run_root.mkdir(parents=True)
    atomic_json(run_root/'frozen/provider_profile.json',profile)
    sources = {row['path']:row for row in previous['files']}
    sources[str(CONFIG)]={'path':str(CONFIG),'sha256':sha(CONFIG)}
    sources[str(Path(__file__).resolve())]={'path':str(Path(__file__).resolve()),'sha256':sha(__file__)}
    sources[str(SOURCE/'freeze_manifest.json')]={'path':str(SOURCE/'freeze_manifest.json'),'sha256':sha(SOURCE/'freeze_manifest.json')}
    slots=[]
    for root in ROOTS:
        original_dir=SOURCE/'execution/calls'/root/'gate'
        original=read(original_dir/'request.json')['payload']
        expected=build_payload(profile,original['messages'])
        assert {k:v for k,v in expected.items() if k!='model'} == {k:v for k,v in original.items() if k!='model'}
        for rel,origin in [('request.json',original_dir/'request.json'),
                           ('qwen_response.json',original_dir/'parsed_response.json'),
                           ('qwen_metadata.json',original_dir/'response_metadata.json'),
                           ('plan.json',SOURCE/'roots'/root/'planner/compiled_plan.json'),
                           ('catalog.json',SOURCE/'roots'/root/'semantic_gate/evidence_catalog.json')]:
            sources[str(origin)]={'path':str(origin),'sha256':sha(origin)}
            atomic_json(run_root/'frozen'/root/rel,read(origin))
        capacity=capacity_check(profile,tokenizer,expected['messages'],8192)
        capacity.update(counting_tokenizer='FROZEN_QWEN3_14B_PROXY_NOT_TJU_EXACT',
                        service_context_capacity='UNKNOWN',context_window_semantics='LOCAL_GUARD_NOT_SERVICE_LIMIT')
        if capacity['status']!='PASS':raise ValueError('local_proxy_guard_exceeded')
        atomic_json(run_root/'frozen'/root/'capacity_proxy.json',capacity)
        atomic_json(run_root/'frozen'/root/'strong_request.json',expected)
        atomic_text(run_root/'frozen'/root/'prompt.txt',expected['messages'][0]['content'])
        slots.append({'root_id':root,'same_messages_and_non_model_parameters':True,
                      'candidate_count':len(read(original_dir/'parsed_response.json')['candidate_reviews']),
                      'capacity_proxy':capacity})
    for f in (run_root/'frozen').rglob('*'):
        if f.is_file():sources[str(f)]={'path':str(f),'sha256':sha(f)}
    manifest={'version':'fixed_gate_tju_two_calls_v1','prepared_at':now(),
      'source_run':str(SOURCE),'root_order':ROOTS,'slots':slots,
      'profile':profile,'tokenizer_path':previous['tokenizer_path'],
      'maximum_logical_calls':2,'maximum_http_attempts':2,'per_root_max':1,
      'maximum_output_tokens':16384,'verified_money_budget':'UNKNOWN',
      'retries':0,'parameter_fallback':False,'technical_failure_policy':'STOP_ALL_UNSENT',
      'service_capacity_exception':'Existing school service exception retained; current two-call budget explicitly authorized',
      'baseline':'Existing Qwen Gate responses on these exact messages; no new Qwen calls',
      'fresh_planner':False,'files':list(sources.values())}
    atomic_json(run_root/'freeze_manifest.json',manifest)
    atomic_json(run_root/'run_status.json',{'status':'PREPARED','http_attempts':0})
    atomic_text(run_root/'AUTHORIZATION.md','User explicitly approved school-model Gate comparison, at most 2 HTTP attempts, one per frozen actual Gate request, no retries, stop on technical failure. No Planner, outline or Realizer calls. The existing school service capacity/parameter/reasoning documentation exception remains recorded; parameters match actual Qwen Gate (.25,8192,JSON), not the earlier Writer settings.\n')
    return manifest


def execute(run_root=DEFAULT_RUN, *, client_factory=None, tokenizer=None):
    run_root=Path(run_root)
    manifest=read(run_root/'freeze_manifest.json')
    check_freeze(manifest)
    if manifest['root_order']!=ROOTS or manifest['maximum_http_attempts']!=2 or manifest['per_root_max']!=1:
        raise ValueError('frozen_two_request_scope_changed')
    profile=manifest['profile']
    tokenizer=tokenizer if tokenizer is not None else tokenizer_from(manifest['tokenizer_path'])
    if client_factory is None:
        credential=resolve_credential(profile)
        def client_factory(stage_profile, *, on_http_attempt):
            return ProfileClient(stage_profile,credential=credential,on_http_attempt=on_http_attempt)
    def before_stage(root,stage,prompt,max_tokens):
        check_freeze(manifest)
        if root not in ROOTS or stage!='gate' or max_tokens!=8192:
            raise ValueError('outside_two_gate_scope')
        frozen=read(run_root/'frozen'/root/'strong_request.json')
        if build_payload(profile,prompt['messages'])!=frozen:
            raise ValueError('frozen_gate_request_changed')
        print(json.dumps({'event':'gate_start','root':root,'model':profile['model']}),flush=True)
    outcomes=[]
    with FixedGateCaller(run_root/'execution',profile,tokenizer,http_max=2,per_root_max=1,
                         client_factory=client_factory,before_stage=before_stage) as caller:
        atomic_json(run_root/'run_status.json',{'status':'RUNNING','started_at':now()})
        for root in ROOTS:
            if caller.state['status']=='STOPPED':break
            frozen=run_root/'frozen'/root;target=run_root/'roots'/root
            outcome={'root_id':root}
            try:
                payload=read(frozen/'strong_request.json')
                output=caller.call(root,'gate',{'messages':payload['messages'],'prompt':payload['messages'][0]['content']},8192)
                atomic_json(target/'output.json',output)
                catalog=read(frozen/'catalog.json')
                validation=validate_gate(output,plan=read(frozen/'plan.json'),catalog=catalog,
                  expected_root_id=root,expected_catalog_sha256=catalog['sha256'])
                atomic_json(target/'validation.json',validation)
                outcome.update(status='COMPLETED',gate_envelope_valid=validation['valid'],
                   all_reviews_valid=validation['all_reviews_valid'],
                   passed_refs=validation['passed_candidate_refs'],
                   rejected_refs=validation['rejected_candidate_refs'],
                   quarantined_refs=validation['quarantined_candidate_refs'],
                   gate_semantics='PENDING_CODEX_OFFLINE_DIAGNOSIS')
            except DevelopmentContentError as error:
                outcome.update(status='CONTENT_FAILURE',error=error.status)
            except DevelopmentRuntimeError as error:
                outcome.update(status='TECHNICAL_FAILURE_STOPPED',error=error.status)
            except Exception:
                caller._stop('local_comparison_failure')
                outcome.update(status='LOCAL_FAILURE_STOPPED',error='local_comparison_failure')
            outcomes.append(outcome)
            atomic_json(target/'outcome.json',outcome)
            print(json.dumps(outcome),flush=True)
        if caller.state['status']!='STOPPED':
            caller.state.update(status='FINISHED',finished_at=now());caller._save()
        result={'status':caller.state['status'],'completed_at':now(),'outcomes':outcomes,
          'unsent_roots':[r for r in ROOTS if r not in [x['root_id'] for x in outcomes]],
          'cost':cost_summary(caller.records),'other_model_stages':0,'memory_api_calls':0,
          'model_service_comparison':True}
        atomic_json(run_root/'generation_summary.json',result)
        atomic_json(run_root/'run_status.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','execute'])
    parser.add_argument('--run-root',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args()
    result=prepare(args.run_root) if args.action=='prepare' else execute(args.run_root)
    print(json.dumps({'status':'PREPARED','max_http':2} if args.action=='prepare' else result))
