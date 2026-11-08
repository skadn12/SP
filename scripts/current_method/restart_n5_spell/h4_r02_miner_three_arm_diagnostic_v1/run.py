"""Frozen two-root, three-arm diagnostic on the original shared Miner."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import subprocess
import sys

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_qa_multiroot_k3_t08_probe_v1.generation import generate_root_intents
from ..h4_qa_mining_probe_v1.intent_miner import build_intent_prompt
from ..h4_r02_feedback_transfer_p2_2_v1.common_sources import write_frozen
from ..h4_r02_feedback_transfer_p2_2_v1.token_preflight import TokenPreflight, prepare_tokenizer
from ..h4_r02_miner_binding_v2.run import copy_exact, record, INPUT_NAMES
from ..live_backends import load_challenger_config, OpenAICompatibleChallengerClient, RATE_LIMIT_RETRY_LIMIT
from .constants import VERSION, ROOTS, ARMS, SCHEDULE, P1, V1_RUN, V2_RUN, DEFAULT_RUN, CONFIG, WORK_ORDER, root_path, global_request_id
from .adapter import ArmHooks, build_prompt, feedback_payload, prompt_pairing_diagnostics, COMMON_BASE_VERSION, SCOPE_FIX_VERSION
from .runtime import BoundedLedger, GuardrailStop, ObservedClient, exclusive_run, read, now, audit_reservations, checkpoint_root, verify_checkpoint

CODE=Path(__file__).resolve().parent
METHOD=CODE.parent
REPO=METHOD.parents[2]


def configured():
    config=load_challenger_config(CONFIG)
    return replace(config,temperature=.1,enable_thinking=False,timeout_seconds=max(180,int(config.timeout_seconds)))


def code_identity(run):
    paths={Path(r['path']) for r in read(run/'audit/initial_code_identity.json')}
    paths.update(p for p in CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return [record(p) for p in sorted(paths)]


def verify_frozen(run):
    frozen=read(run/'audit/frozen_manifest.json')
    if frozen['version']!=VERSION or frozen['schedule']!=list(SCHEDULE): raise GuardrailStop('frozen_scope_drift')
    for row in frozen['code']+frozen['inputs']+frozen['sources']+[frozen['config'],frozen['preflight']]:
        p=Path(row['path'])
        if not p.is_file() or sha256_file(p)!=row['sha256']:raise GuardrailStop('frozen_source_code_or_input_drift:'+str(p))
    if code_identity(run)!=frozen['code']:raise GuardrailStop('code_inventory_drift')
    return frozen


def make_hooks(run,arm_id,root_id):
    folder=run/'inputs/shared'/root_id
    return ArmHooks(arm_id,read(folder/'observation_cards.json'),read(folder/'history_only.json'),
                    transcript_sha256=sha256_file(folder/'expanded_dialogue_clean.json'))


def test_phase(run):
    script='''import sys
sys.dont_write_bytecode=True
import unittest
from unittest.mock import patch
suite=unittest.TestSuite();loader=unittest.TestLoader()
for directory in %r:
 suite.addTests(loader.discover(directory,pattern='test_*.py',top_level_dir=%r))
with patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN')),patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN')),patch('urllib.request.urlopen',side_effect=AssertionError('NETWORK_FORBIDDEN')):
 result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
'''%([str(METHOD/'h4_r02_feedback_transfer_p2_2_v1'),str(METHOD/'h4_r02_miner_binding_v2'),str(CODE)],str(REPO))
    before=code_identity(run)
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=REPO,capture_output=True,text=True)
    output=result.stdout+result.stderr;atomic_text(run/'audit/test_output.txt',output)
    count=re.search(r'Ran (\d+) tests?',output)
    report={'status':'PASS' if result.returncode==0 and before==code_identity(run) else 'FAIL',
        'tests_run':int(count.group(1)) if count else None,'exit_code':result.returncode,'at_utc':now(),
        'live_calls':0,'network_forbidden':True,'bytecode_writes_disabled':True,'command':[sys.executable,'-B','-c',script],
        'cwd':str(REPO),'code':before,'log':record(run/'audit/test_output.txt')}
    atomic_json(run/'audit/test_report.json',report)
    lines=[line for line in output.splitlines() if any(k in line for k in ('test_shared_miner.','test_generation_hooks.','test_adapter.'))]
    atomic_json(run/'audit/shared_path_parity.json',{'status':report['status'],'results':lines,
        'shared_public_entry':'generate_root_intents','original_code_changed':False,
        'method':'Offline snapshot/default no-hook, v1 hooks, and exact v2 shared-stage replay; all arms use same shared Miner.',
        'test_log':report['log']})
    return report


def token_components(prompt,expanded,payload):
    transcript='\n'.join(f"{i} | {r.get('turn_ref')} | {r.get('speaker')} | {r.get('text')}" for i,r in enumerate(expanded['turns'],1))
    return {'transcript':transcript,'feedback':json.dumps(payload,ensure_ascii=False,indent=2),
            'template':prompt.replace(transcript,'',1),'binding':'','brief':''}


def prepare(run):
    if (run/'audit/frozen_manifest.json').is_file():
        verify_frozen(run);return read(run/'audit/preflight_report.json')
    if list(run.glob('arms/**/ledger/requests/*.json')):raise GuardrailStop('live_states_without_freeze')
    copies=[]
    for label in ('P1','V1_RUN','V2_RUN'):
        if not (run/f'audit/{label}_inventory_before.json').is_file():raise GuardrailStop('historical_inventory_required:'+label)
    initial=read(run/'audit/initial_code_identity.json')
    for row in initial:
        if sha256_file(Path(row['path']))!=row['sha256']:raise GuardrailStop('prior_shared_code_changed:'+row['path'])
    for rid in ROOTS:
        for name in INPUT_NAMES:copies.append(copy_exact(V2_RUN/'inputs'/rid/name,run/'inputs/shared'/rid/name))
        p1=P1/'inputs/shared'/rid/'expanded_dialogue_clean.json'
        target=run/'inputs/shared'/rid/'expanded_dialogue_clean.json'
        if sha256_file(p1)!=sha256_file(target):raise GuardrailStop('P1_transcript_byte_mismatch:'+rid)
        copies.append({'source':record(p1),'copy':record(target)})
    for folder in ('policy_snapshot/tokenizer','audit/service_evidence','common_base'):
        source_dir=V2_RUN/folder
        for p in sorted(source_dir.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                copies.append(copy_exact(p,run/folder/p.relative_to(source_dir)))
    for name in ('execution_contract.json','implementation_manifest.json','audit/token_identity.json','audit/token_report.json'):
        copies.append(copy_exact(V2_RUN/name,run/'policy_snapshot/v2_reference'/name))
    copies.append(copy_exact(WORK_ORDER,run/'policy_snapshot/CODEX_WORK_ORDER.md'))
    config=configured();prior=read(V2_RUN/'execution_contract.json')['generation']
    if not(config.model==prior['model']=='Qwen/Qwen3-14B' and config.base_url==prior['base_url'] and config.timeout_seconds==180 and RATE_LIMIT_RETRY_LIMIT==2):raise GuardrailStop('model_or_transport_config_drift')
    if sha256_file(CONFIG)!=read(V2_RUN/'execution_contract.json')['configuration']['sha256']:raise GuardrailStop('configuration_identity_changed_since_v2')
    write_frozen(run/'schedule.json',{'version':VERSION,'schedule':list(SCHEDULE)})
    if not (run/'review_policy.json').is_file():raise GuardrailStop('review_policy_required_before_live')
    write_frozen(run/'execution_contract.json',{'version':VERSION,'common_base_version':COMMON_BASE_VERSION,
        'explanation_scope_fix_version':SCOPE_FIX_VERSION,'roots':list(ROOTS),'arms':list(ARMS),'schedule':list(SCHEDULE),
        'planned_logical_calls':6,'maximum_logical_calls':6,'per_root_arm_limit':1,'per_root_limit':3,'per_arm_limit':2,
        'generation':prior,'zero_call_roles':['Writer','repair','Solver','Planner','Realizer','Memory','training','extra_review_API'],
        'context_reserve_tokens':512,'work_order':record(WORK_ORDER),'configuration':record(CONFIG),
        'stop_after_six_call_diagnostic':True,'no_new_evidence_or_semantic_admission':True})
    write_frozen(run/'implementation_manifest.json',{'version':VERSION,'common_base_version':COMMON_BASE_VERSION,
        'explanation_scope_fix_version':SCOPE_FIX_VERSION,'original_files_changed':[],
        'shared_miner':'h4_qa_multiroot_k3_t08_probe_v1.generation.generate_root_intents',
        'evidence_derivation':'h4_r02_miner_binding_v2.transfer.derive_all_intent_evidence unchanged',
        'C_response_processing':'unchanged TransferHooksV2 projection/validation/sidecars',
        'A_B_response_processing':'complete raw passed directly to original canonicalizer, no binding processing',
        'three_arm_intent_item_schema_same':True,'three_arm_full_envelope_same':False,
        'only_model_visible_common_change':read(CODE/'scope_fix_policy.json'),
        'global_request_identity':['arm_id','root_id','request_id'],'new_code':str(CODE),
        'prechange_workspace':record(run/'audit/workspace_status_before.txt')})
    pairings=[];prompts={};expanded_by_root={};payloads={};pair_reports=[]
    for rid in ROOTS:
        folder=run/'inputs/shared'/rid;expanded=read(folder/'expanded_dialogue_clean.json');expanded_by_root[rid]=expanded
        cards=read(folder/'observation_cards.json');history=read(folder/'history_only.json');base=build_intent_prompt(expanded)
        old_prompt=V2_RUN/'qa/shards/shard-00/roots'/rid/'raw/prompts'/f'{rid}-intent-miner.txt'
        copies.append({'source':record(old_prompt),'purpose':'C vs archived v2 comparison'})
        pair_reports.append(prompt_pairing_diagnostics(base,cards,history,expanded=expanded,root_id=rid,legacy_v2_prompt=old_prompt.read_text()))
        for arm in ARMS:
            spec=build_prompt(arm,base,cards,history,expanded=expanded,root_id=rid);prompts[arm,rid]=spec
            payload=feedback_payload(arm,cards,history,rid);payloads[arm,rid]=payload
            write_frozen(run/'inputs/arms'/arm/rid/'feedback_payload.json',payload)
            write_frozen(run/'policy_snapshot/prompts'/arm/rid/'miner_prompt.txt',raw=spec['prompt'].encode())
            write_frozen(run/'policy_snapshot/prompts'/arm/rid/'prompt_spec.json',spec)
    atomic_json(run/'audit/prompt_pairing_report.json',{'version':VERSION,'status':'PASS' if all(r['status']=='PASS' for r in pair_reports) else 'FAIL','roots':pair_reports,'pairing_unit':'root','not_same_intent_id_pairing':True})
    for row in SCHEDULE:
        arm,rid=row['arm_id'],row['root_id'];spec=prompts[arm,rid]
        pairings.append({**row,'prompt_sha256':spec['prompt_sha256'],'prompt_path':str(run/'policy_snapshot/prompts'/arm/rid/'miner_prompt.txt'),
            'configuration_sha256':sha256_file(CONFIG),'transcript_sha256':sha256_file(run/'inputs/shared'/rid/'expanded_dialogue_clean.json'),
            'response_schema_sha256':spec['response_schema_sha256'],'feedback_payload_sha256':spec['feedback_payload_sha256'],
            'historical_tracking_sha256':spec['historical_tracking_sha256'],'max_output_tokens':4096,'planned_intent_slots':3,'maximum_logical_calls':1})
    write_frozen(run/'pairing_manifest.jsonl',raw=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in pairings).encode())
    write_frozen(run/'input_path_audit.json',{'status':'PASS','copies':copies,'transcript_policy':'Exact V2 bytes additionally verified against P1; no clean or renumber.',
        'source_ids':{rid:read(run/'inputs/shared'/rid/'observation_cards.json')['observation_cards'][0]['source_candidate_id'] for rid in ROOTS}})
    tests=test_phase(run)
    identity=prepare_tokenizer(run,config)
    old_identity=read(V2_RUN/'audit/token_identity.json')
    token_identity_same=all(identity[k]==old_identity[k] for k in ('model','service_base_url','context_limit','chat_template_sha256','chat_rendering','reserve_tokens')) and identity['tokenizer']['revision']==old_identity['tokenizer']['revision'] and identity['tokenizer']['files']==old_identity['tokenizer']['files']
    token=TokenPreflight(run);token_rows=[]
    for row in SCHEDULE:
        arm,rid=row['arm_id'],row['root_id'];prompt=prompts[arm,rid]['prompt']
        token_rows.append(token.check(prompt,4096,row['global_request_id'],components=token_components(prompt,expanded_by_root[rid],payloads[arm,rid])))
    checks={'tests':tests['status']=='PASS','pairing':all(r['status']=='PASS' for r in pair_reports),'six_unique_global_ids':len({r['global_request_id'] for r in pairings})==6,
        'token':len(token_rows)==6 and all(r['status']=='PASS' for r in token_rows),'tokenizer_identity_unchanged':token_identity_same,'original_sources_unchanged':True,'exact_input_bytes':True}
    report={'status':'PASS' if all(checks.values()) else 'BLOCKED','at_utc':now(),'checks':checks,'requests':token_rows,'live_calls':0}
    atomic_json(run/'audit/preflight_report.json',report)
    if report['status']=='PASS':
        code=code_identity(run)
        for row in code:copy_exact(Path(row['path']),run/'policy_snapshot/frozen_code'/Path(row['path']).relative_to(REPO))
        paths=[]
        for folder in ('inputs','common_base','policy_snapshot','audit/service_evidence'):
            paths.extend(p for p in (run/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        paths.extend(run/p for p in ('execution_contract.json','implementation_manifest.json','input_path_audit.json','pairing_manifest.jsonl','schedule.json','review_policy.json','audit/test_report.json','audit/shared_path_parity.json','audit/prompt_pairing_report.json','audit/token_identity.json','audit/initial_code_identity.json','audit/P1_inventory_before.json','audit/V1_RUN_inventory_before.json','audit/V2_RUN_inventory_before.json','audit/workspace_status_before.txt','audit/workspace_diff_before.patch'))
        sources={r['source']['path']:r['source'] for r in copies}
        for row in read(V2_RUN/'audit/frozen_manifest.json')['sources']:sources[row['path']]=row
        write_frozen(run/'audit/frozen_manifest.json',{'version':VERSION,'schedule':list(SCHEDULE),'code':code,
            'inputs':[record(p) for p in sorted(set(paths))],'sources':list(sources.values()),'config':record(CONFIG),'preflight':record(run/'audit/preflight_report.json')})
    atomic_text(run/'RUN_STATUS.md',f"Prepare: {report['status']}\nLive: NOT_RUN\nReview: NOT_RUN\n6 Miner logical calls maximum; no Writer/repair/Solver/Memory/training.\n")
    return report


def live(run):
    prior=run/'audit/execution_stop.json'
    if prior.exists() and read(prior).get('phase')=='live':raise GuardrailStop('live_previously_stopped_no_resampling')
    verify_frozen(run)
    if read(run/'audit/preflight_report.json')['status']!='PASS':raise GuardrailStop('preflight_not_passed')
    audit_reservations(run);token=TokenPreflight(run);summaries=[]
    for row in SCHEDULE:
        arm,rid=row['arm_id'],row['root_id'];root=root_path(run,arm,rid);gid=row['global_request_id']
        verify_frozen(run)
        if verify_checkpoint(root):summaries.append(read(root/'miner_generation_report.json'));continue
        expanded=read(run/'inputs/shared'/rid/'expanded_dialogue_clean.json');payload=read(run/'inputs/arms'/arm/rid/'feedback_payload.json')
        def before(request_id,metadata):
            verify_frozen(run)
            actual=(root/'raw/prompts'/f'{request_id}.txt').read_text()
            if actual!=(run/'policy_snapshot/prompts'/arm/rid/'miner_prompt.txt').read_text():raise GuardrailStop('actual_prompt_vs_frozen_preflight_drift')
            token.check(actual,metadata['max_tokens'],gid,components=token_components(actual,expanded,payload))
        ledger=BoundedLedger(root,sha256_file(run/'audit/frozen_manifest.json'),run,arm,rid,
            before_request=before,after_response=lambda request_id,response:token.verify_usage(gid,response.metadata.get('usage')))
        # Original stateless client; fresh independent request for every arm.
        client=ObservedClient(OpenAICompatibleChallengerClient(configured()),run/'audit/transport_events.jsonl',arm,rid)
        print(json.dumps({**row,'status':'started','at_utc':now()}),flush=True)
        result=generate_root_intents(expanded=expanded,root_dir=root,root_id=rid,api_config=CONFIG,ledger=ledger,client=client,hooks=make_hooks(run,arm,rid))
        state=ledger.delegate.read(row['request_id'])
        summary={**row,'status':'COMPLETED','transport_status':state['status'],'valid_intents':len(result['contracts']),
            'intent_errors':result['intent_result']['errors'],'writer_calls':0,'at_utc':now()}
        atomic_json(root/'miner_generation_report.json',summary);checkpoint_root(root);summaries.append(summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    completion={'status':'COMPLETED','groups':summaries,'at_utc':now()};atomic_json(run/'audit/live_completion.json',completion)
    atomic_text(run/'RUN_STATUS.md','Prepare: PASS\nLive: COMPLETED\nReview: awaiting stage 1 intent evidence judgments\nWriter/repair/Solver/Memory/training: 0\n')
    return completion


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',required=True,choices=('prepare','live','report'));parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args(argv);run=args.run.resolve()
    if run!=DEFAULT_RUN:raise RuntimeError('only_fixed_diagnostic_run_authorized')
    try:
        with exclusive_run(run):
            if args.phase=='prepare':result=prepare(run)
            elif args.phase=='live':result=live(run)
            else:
                from .reporting import final_report
                result=final_report(run)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        return 0 if result.get('status') not in {'BLOCKED','FAILED'} else 2
    except (GuardrailStop,Exception) as error:
        path=run/'audit/execution_stop.json'
        if path.exists():path=run/'audit'/('execution_stop_attempt_'+now().replace(':','-')+'.json')
        atomic_json(path,{'status':'BLOCKED','phase':args.phase,'reason':str(error),'at_utc':now(),'automatic_resend':False})
        print('Three-arm diagnostic stopped: '+str(error),flush=True);return 2

if __name__=='__main__':raise SystemExit(main())
