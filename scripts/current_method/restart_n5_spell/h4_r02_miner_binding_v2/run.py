"""Frozen prepare -> exactly three shared-Miner calls -> offline report."""
from __future__ import annotations
import argparse
from dataclasses import replace
import difflib
import json
from pathlib import Path
import re
import subprocess
import sys

from ..core import atomic_json, atomic_text, sha256_file
from ..h4_qa_multiroot_k3_t08_probe_v1.generation import generate_root_intents
from ..h4_qa_mining_probe_v1.intent_miner import build_intent_prompt
from ..live_backends import load_challenger_config, OpenAICompatibleChallengerClient, RATE_LIMIT_RETRY_LIMIT
from ..h4_r02_feedback_transfer_p2_2_v1.common_sources import write_frozen
from ..h4_r02_feedback_transfer_p2_2_v1.transfer import COMMON_BASE_VERSION, model_cards
from ..h4_r02_feedback_transfer_p2_2_v1.token_preflight import TokenPreflight, prepare_tokenizer
from .runtime import (FIRST_ROOTS, MODE, PLAN_TOTAL, LIMIT_TOTAL, LIMIT_PER_ROOT,
    BoundedLedger, GuardrailStop, ObservedClient, exclusive_run, root_path,
    read, now, audit_reservations, checkpoint_root, verify_checkpoint)
from .transfer import TransferHooksV2

CODE = Path(__file__).resolve().parent
METHOD = CODE.parent
REPO = METHOD.parents[2]
RUNS = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs')
P1 = RUNS / 'h4_r02_feedback_ablation_p0_p1_v1_live_1'
V1_RUN = RUNS / 'h4_r02_feedback_transfer_p2_2_acceptance_v1_live_1'
CONFIG = Path('/root/autodl-tmp/gxy-selfplay/config.json')
DEFAULT_RUN = RUNS / 'h4_r02_miner_binding_v2_acceptance_v1_live_1'
WORK_ORDER = RUNS / 'h4_r02_miner_binding_v2_handoff_v1/CODEX_WORK_ORDER.md'
VERSION = 'h4_r02_miner_binding_v2'
INPUT_NAMES = ('expanded_dialogue_clean.json','observation_cards.json','history_only.json','provenance.json')


def record(path):
    return {'path': str(path), 'sha256': sha256_file(path)}


def copy_exact(source, target):
    write_frozen(target, raw=source.read_bytes())
    if sha256_file(source) != sha256_file(target):
        raise GuardrailStop('byte_copy_mismatch:' + str(target))
    return {'source': record(source), 'copy': record(target)}


def configured():
    config = load_challenger_config(CONFIG)
    return replace(config, temperature=.1, enable_thinking=False,
                   timeout_seconds=max(180, int(config.timeout_seconds)))


def make_hooks(run, root_id):
    folder = run / 'inputs' / root_id
    return TransferHooksV2(read(folder/'observation_cards.json'), read(folder/'history_only.json'),
                          transcript_sha256=sha256_file(folder/'expanded_dialogue_clean.json'))


def code_identity():
    paths = {Path(row['path']) for row in read(V1_RUN/'audit/frozen_manifest.json')['code']}
    paths.update(p for p in CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return [record(p) for p in sorted(paths)]


def verify_frozen(run):
    frozen = read(run/'audit/frozen_manifest.json')
    if frozen['version'] != VERSION or frozen['root_order'] != list(FIRST_ROOTS):
        raise GuardrailStop('frozen_scope_drift')
    for row in frozen['code'] + frozen['inputs'] + frozen['sources'] + [frozen['config'], frozen['preflight']]:
        p = Path(row['path'])
        if not p.is_file() or sha256_file(p) != row['sha256']:
            raise GuardrailStop('frozen_source_code_or_input_drift:' + str(p))
    if code_identity() != frozen['code']:
        raise GuardrailStop('code_inventory_drift')
    return frozen


def test_phase(run):
    script = '''import sys
sys.dont_write_bytecode=True
import unittest, socket, urllib.request
from unittest.mock import patch
loader=unittest.TestLoader()
suite=unittest.TestSuite()
for directory in %r:
 suite.addTests(loader.discover(directory, pattern='test_*.py', top_level_dir=%r))
with patch('socket.socket.connect', side_effect=AssertionError('NETWORK_FORBIDDEN')), patch('socket.create_connection', side_effect=AssertionError('NETWORK_FORBIDDEN')), patch('urllib.request.urlopen', side_effect=AssertionError('NETWORK_FORBIDDEN')):
 result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
''' % ([str(METHOD/'h4_r02_feedback_transfer_p2_2_v1'), str(CODE)], str(REPO))
    before = code_identity()
    result = subprocess.run([sys.executable,'-c',script], cwd=REPO, capture_output=True,text=True)
    output = result.stdout + result.stderr
    atomic_text(run/'audit/test_output.txt', output)
    count = re.search(r'Ran (\d+) tests?', output)
    report = {'status':'PASS' if result.returncode == 0 and before == code_identity() else 'FAIL',
              'tests_run':int(count.group(1)) if count else None, 'exit_code':result.returncode,
              'live_calls':0,'network_forbidden':True,'log':record(run/'audit/test_output.txt'),
              'test_runner':script,'at_utc':now()}
    atomic_json(run/'audit/test_report.json',report)
    lines = [line for line in output.splitlines() if 'test_shared_miner.' in line]
    atomic_json(run/'audit/shared_miner_parity.json',{
        'status':'PASS' if report['status']=='PASS' and lines else 'FAIL', 'tests':lines,
        'method':'Snapshot-vs-shared full QA byte/return/call parity with no hooks and actual v1 TransferHooks, entirely offline.',
        'snapshot':record(run/'policy_snapshot/prechange/h4_qa_multiroot_k3_t08_probe_v1/generation.py'),
        'test_log':report['log']})
    return report


def token_components(run, root_id, prompt, expanded):
    transcript='\n'.join(f"{i} | {r.get('turn_ref')} | {r.get('speaker')} | {r.get('text')}" for i,r in enumerate(expanded['turns'],1))
    return {'transcript':transcript,'feedback':json.dumps(model_cards(read(run/'inputs'/root_id/'observation_cards.json'),root_id=root_id),ensure_ascii=False,indent=2),
            'template':prompt.replace(transcript,'',1),'binding':'','brief':''}


def prepare(run):
    if (run/'audit/frozen_manifest.json').exists():
        verify_frozen(run)
        return read(run/'audit/preflight_report.json')
    if list((run/'qa').glob('**/ledger/requests/*.json')):
        raise GuardrailStop('live_states_without_freeze')
    if not (run/'audit/P1_inventory_before.json').is_file() or not (run/'audit/V1_RUN_inventory_before.json').is_file():
        raise GuardrailStop('prechange_history_inventories_required')
    copies=[]
    for root_id in FIRST_ROOTS:
        for name in INPUT_NAMES:
            copies.append(copy_exact(V1_RUN/'inputs'/root_id/name, run/'inputs'/root_id/name))
        dialogue=run/'inputs'/root_id/INPUT_NAMES[0]
        p1_dialogue=P1/'inputs/shared'/root_id/INPUT_NAMES[0]
        if sha256_file(dialogue)!=sha256_file(p1_dialogue):
            raise GuardrailStop('P1_transcript_byte_mismatch:' + root_id)
        copies.append({'source':record(p1_dialogue),'copy':record(dialogue)})
    for folder in ('policy_snapshot/tokenizer','audit/service_evidence'):
        for source in sorted((V1_RUN/folder).rglob('*')):
            if source.is_file(): copies.append(copy_exact(source,run/folder/source.relative_to(V1_RUN/folder)))
    for name in ('source_eligibility_manifest.json','miner_prompt_revision.json','observation_policy.json'):
        copies.append(copy_exact(V1_RUN/'common_base'/name,run/'common_base'/name))
    for name in ('execution_contract.json','implementation_manifest.json','audit/token_identity.json','audit/token_report.json'):
        copies.append(copy_exact(V1_RUN/name,run/'policy_snapshot/v1_reference'/name))
    copy_exact(WORK_ORDER,run/'policy_snapshot/CODEX_WORK_ORDER.md')
    initial=[]
    snapshot=run/'policy_snapshot/prechange'
    for p in snapshot.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts: initial.append({'path':str(METHOD/p.relative_to(snapshot)),'sha256':sha256_file(p)})
    changed=[r['path'] for r in initial if sha256_file(Path(r['path']))!=r['sha256']]
    allowed=METHOD/'h4_qa_multiroot_k3_t08_probe_v1/generation.py'
    if changed != [str(allowed)]: raise GuardrailStop('unexpected_shared_code_changes:' + repr(changed))
    config=configured(); old=read(V1_RUN/'execution_contract.json')['generation']
    checks={'model':config.model==old['model']=='Qwen/Qwen3-14B',
            'service':config.base_url==old['base_url'], 'timeout':config.timeout_seconds==180,
            'retry':RATE_LIMIT_RETRY_LIMIT==old['http_429_retry_limit']==2}
    if not all(checks.values()): raise GuardrailStop('original_config_mismatch')
    write_frozen(run/'input_path_audit.json',{'status':'PASS','copies':copies,
        'policy':'Exact V1 bytes plus P1 dialogue identity; no clean, ranking, source repair or replacement.'})
    write_frozen(run/'execution_contract.json',{'version':VERSION,'common_base_version':COMMON_BASE_VERSION,
        'roots':list(FIRST_ROOTS),'mode':MODE,'planned_logical_calls':3,'maximum_logical_calls':3,
        'maximum_calls_per_root':1,'allowed_request_ids':[r+'-intent-miner' for r in FIRST_ROOTS],
        'generation':{'model':config.model,'base_url':config.base_url,'temperature':.1,'thinking':False,
            'generation_seed_policy':'not_sent_by_original_client','max_output_tokens':4096,'intent_count':3,
            'timeout_seconds':180,'http_429_retry_limit':2},
        'zero_call_roles':['Writer','repair','Solver','Planner','Realizer','Memory','training','extra_LLM_review'],
        'context_reserve_tokens':512,'work_order':record(WORK_ORDER),'configuration':record(CONFIG),
        'stop_after_miner_only_report':True,'semantic_admission':False})
    write_frozen(run/'implementation_manifest.json',{'version':VERSION,'common_base_version':COMMON_BASE_VERSION,
        'original_files_changed':changed,'shared_public_entry':'generate_root_intents',
        'original_qa_entry':'generate_root_qa calls shared phase then original Writer loop',
        'v1_hooks_behavior':'unchanged; snapshot regression required',
        'contract_change':'five-field binding; change_from_source three-field object; no target_distinction/evidence_refs in binding',
        'current_evidence':'ORIGINAL_VALIDATED_INTENT_CHAIN_ONLY',
        'prechange_workspace':record(run/'audit/workspace_status_before.txt'), 'initial_code_identity':initial,
        'no_planner_miner_default_or_sorting_change':True})
    oldtext=(snapshot/'h4_qa_multiroot_k3_t08_probe_v1/generation.py').read_text()
    atomic_text(run/'audit/shared_miner.diff',''.join(difflib.unified_diff(oldtext.splitlines(True),allowed.read_text().splitlines(True),fromfile='prechange',tofile='shared_miner')))
    tests=test_phase(run)
    prepare_tokenizer(run,config)
    token=TokenPreflight(run); rows=[]
    for root_id in FIRST_ROOTS:
        expanded=read(run/'inputs'/root_id/INPUT_NAMES[0])
        prompt=make_hooks(run,root_id).miner_prompt(build_intent_prompt(expanded),expanded=expanded,root_id=root_id,root_dir=run/'policy_snapshot/offline_prompts'/root_id)['prompt']
        write_frozen(run/'policy_snapshot/offline_prompts'/root_id/'miner_prompt.txt',raw=prompt.encode())
        rows.append(token.check(prompt,4096,root_id+'-intent-miner',components=token_components(run,root_id,prompt,expanded)))
    checks.update(tests=tests['status']=='PASS',parity=read(run/'audit/shared_miner_parity.json')['status']=='PASS',
                  token=len(rows)==3 and all(r['status']=='PASS' for r in rows),exact_source_bytes=True,
                  miner_only_budget=(PLAN_TOTAL,LIMIT_TOTAL,LIMIT_PER_ROOT)==(3,3,1))
    report={'status':'PASS' if all(checks.values()) else 'BLOCKED','checks':checks,'miner_checks':rows,'live_calls':0,'at_utc':now()}
    atomic_json(run/'audit/preflight_report.json',report)
    if report['status']=='PASS':
        for row in code_identity():
            source=Path(row['path'])
            copy_exact(source,run/'policy_snapshot/frozen_code'/source.relative_to(REPO))
        paths=[]
        for folder in ('inputs','common_base','policy_snapshot','audit/service_evidence'):
            paths.extend(p for p in (run/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        paths.extend(run/p for p in ('execution_contract.json','implementation_manifest.json','input_path_audit.json','audit/test_report.json','audit/shared_miner_parity.json','audit/token_identity.json','audit/P1_inventory_before.json','audit/V1_RUN_inventory_before.json','audit/workspace_status_before.txt','audit/workspace_diff_before.patch'))
        sources={r['source']['path']:r['source'] for r in copies}
        for row in read(V1_RUN/'audit/frozen_manifest.json')['sources']: sources[row['path']]=row
        write_frozen(run/'audit/frozen_manifest.json',{'version':VERSION,'root_order':list(FIRST_ROOTS),
            'code':code_identity(),'inputs':[record(p) for p in sorted(set(paths))],
            'sources':list(sources.values()),'config':record(CONFIG),'preflight':record(run/'audit/preflight_report.json')})
    atomic_text(run/'RUN_STATUS.md',f"Prepare: {report['status']}\nLive: NOT_RUN\nReport: NOT_RUN\nMiner only; 3 calls maximum.\n")
    return report


def live(run):
    if (run/'audit/execution_stop.json').is_file() and read(run/'audit/execution_stop.json')['phase']=='live':
        raise GuardrailStop('live_previously_stopped_no_automatic_resample')
    verify_frozen(run)
    if read(run/'audit/preflight_report.json')['status']!='PASS': raise GuardrailStop('preflight_not_passed')
    audit_reservations(run)
    token=TokenPreflight(run); delegate=OpenAICompatibleChallengerClient(configured())
    summaries=[]
    for root_id in FIRST_ROOTS:
        verify_frozen(run); root=root_path(run,root_id)
        if verify_checkpoint(root):
            summaries.append(read(root/'miner_generation_report.json')); continue
        expanded=read(run/'inputs'/root_id/INPUT_NAMES[0])
        def before(request_id,metadata):
            verify_frozen(run)
            prompt=(root/'raw/prompts'/f'{request_id}.txt').read_text()
            expected=(run/'policy_snapshot/offline_prompts'/root_id/'miner_prompt.txt').read_text()
            if prompt!=expected: raise GuardrailStop('actual_prompt_vs_frozen_preflight_drift')
            token.check(prompt,metadata['max_tokens'],request_id,components=token_components(run,root_id,prompt,expanded))
        ledger=BoundedLedger(root,sha256_file(run/'audit/frozen_manifest.json'),run,root_id,
            before_request=before,after_response=lambda rid,res:token.verify_usage(rid,res.metadata.get('usage')))
        client=ObservedClient(delegate,run/'audit/transport_events.jsonl',MODE,root_id)
        print(json.dumps({'root_id':root_id,'status':'started','at_utc':now()}),flush=True)
        mined=generate_root_intents(expanded=expanded,root_dir=root,root_id=root_id,api_config=CONFIG,
                                   ledger=ledger,client=client,hooks=make_hooks(run,root_id))
        summary={'root_id':root_id,'status':'COMPLETED','valid_intents':len(mined['contracts']),
                 'intent_errors':mined['intent_result']['errors'],'writer_calls':0,'at_utc':now()}
        atomic_json(root/'miner_generation_report.json',summary); checkpoint_root(root); summaries.append(summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    result={'status':'COMPLETED','root_order':list(FIRST_ROOTS),'roots':summaries,'at_utc':now()}
    atomic_json(run/'audit/live_completion.json',result)
    atomic_text(run/'RUN_STATUS.md','Prepare: PASS\nLive: COMPLETED\nReport: awaiting offline direct transcript review\nWriter/Solver/Memory/training: 0\n')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--phase',choices=('prepare','live','report'),required=True)
    p.add_argument('--run',type=Path,default=DEFAULT_RUN); args=p.parse_args(argv); run=args.run.resolve()
    if run!=DEFAULT_RUN: raise RuntimeError('only_fixed_new_run_authorized')
    try:
        with exclusive_run(run):
            if args.phase=='prepare': result=prepare(run)
            elif args.phase=='live': result=live(run)
            else:
                from .reporting import final_report
                result=final_report(run)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        return 0 if result.get('status') not in {'BLOCKED','FAILED'} else 2
    except (GuardrailStop,Exception) as error:
        path=run/'audit/execution_stop.json'
        if path.exists(): path=run/'audit'/('execution_stop_attempt_'+now().replace(':','-')+'.json')
        atomic_json(path,{'status':'BLOCKED','phase':args.phase,'reason':str(error),'at_utc':now(),'automatic_resend':False})
        print('Miner v2 stopped: '+str(error),flush=True); return 2

if __name__=='__main__': raise SystemExit(main())
