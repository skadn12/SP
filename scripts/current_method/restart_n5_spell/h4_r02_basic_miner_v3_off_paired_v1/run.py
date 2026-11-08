"""Four frozen, independent OFF Miner calls; no QA or transfer stage."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import subprocess
import sys

from ..core import atomic_json, atomic_text, sha256_file
from ..h4_qa_multiroot_k3_t08_probe_v1.generation import generate_root_intents
from ..h4_qa_mining_probe_v1.intent_miner import build_intent_prompt
from ..h4_r02_feedback_transfer_p2_2_v1.common_sources import write_frozen
from ..h4_r02_feedback_transfer_p2_2_v1.token_preflight import TokenPreflight, prepare_tokenizer
from ..live_backends import load_challenger_config, OpenAICompatibleChallengerClient, RATE_LIMIT_RETRY_LIMIT
from .constants import VERSION, ROOTS, ARMS, SCHEDULE, P1_RUN, V1_RUN, V2_RUN, THREE_RUN, DEFAULT_RUN, CONFIG, WORK_ORDER, root_path
from .adapter import MinerHooks, build_prompt, feedback_payload, prompt_pairing_diagnostics, guidance_text, output_schema
from .dependencies import record, collect_coverage, check_loaded_coverage, environment_identity, repository_execution_guard, forbid_network
from .runtime import BoundedLedger, GuardrailStop, ObservedClient, exclusive_run, read, now, audit_reservations, checkpoint_root, verify_checkpoint, assert_no_guardrails, record_guardrail, post_miner_check

CODE = Path(__file__).resolve().parent
METHOD = CODE.parent
REPO = CODE.parents[3]
INPUT_NAMES = ('expanded_dialogue_clean.json', 'observation_cards.json', 'history_only.json', 'provenance.json')


def copy_exact(source, target):
    source, target = Path(source), Path(target)
    write_frozen(target, raw=source.read_bytes())
    return {'source': record(source), 'copy': record(target)}


def configured():
    config = load_challenger_config(CONFIG)
    return replace(config, temperature=.1, enable_thinking=False, timeout_seconds=max(180, int(config.timeout_seconds)))


def verify_frozen(run):
    run = Path(run)
    frozen = read(run / 'audit/frozen_manifest.json')
    if frozen['version'] != VERSION or frozen['schedule'] != list(SCHEDULE):
        raise GuardrailStop('frozen_scope_drift')
    for row in frozen['code'] + frozen['inputs'] + frozen['sources'] + [frozen['config'], frozen['preflight']]:
        p = Path(row['path'])
        if not p.is_file() or sha256_file(p) != row['sha256'] or p.stat().st_size != row['size']:
            raise GuardrailStop('frozen_identity_drift:' + str(p))
    current_new = {str(p.resolve()) for p in CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    expected_new = {r['path'] for r in frozen['code'] if Path(r['path']).is_relative_to(CODE)}
    if current_new != expected_new:
        raise GuardrailStop('new_module_inventory_drift')
    check_loaded_coverage(r['path'] for r in frozen['code'])
    if environment_identity() != frozen['environment']:
        raise GuardrailStop('python_or_third_party_environment_drift')
    return frozen


def make_hooks(run, arm, rid):
    folder = Path(run) / 'inputs/shared' / rid
    return MinerHooks(arm, read(folder / 'observation_cards.json'), read(folder / 'history_only.json'),
                      transcript_sha256=sha256_file(folder / 'expanded_dialogue_clean.json'))


def test_phase(run):
    script = '''import sys
sys.dont_write_bytecode=True
import unittest
from scripts.current_method.restart_n5_spell.h4_r02_basic_miner_v3_off_paired_v1.dependencies import forbid_network
suite=unittest.TestSuite(); loader=unittest.TestLoader()
for directory in %r:
 suite.addTests(loader.discover(directory,pattern='test_*.py',top_level_dir=%r))
with forbid_network():
 result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
''' % ([str(METHOD/'h4_r02_feedback_transfer_p2_2_v1'), str(METHOD/'h4_r02_miner_binding_v2'), str(METHOD/'h4_r02_miner_three_arm_diagnostic_v1'), str(CODE)], str(REPO))
    process = subprocess.run([sys.executable, '-B', '-c', script], cwd=REPO, text=True, capture_output=True)
    output = process.stdout + process.stderr
    # Preserve any failed prepare attempt rather than silently replacing it.
    attempt = len(list((run/'audit').glob('test_output_attempt_*.txt'))) + 1
    atomic_text(run/f'audit/test_output_attempt_{attempt}.txt', output)
    count = re.search(r'Ran (\d+) tests?', output)
    result = {'status': 'PASS' if process.returncode == 0 else 'FAIL',
              'tests_run': int(count.group(1)) if count else None, 'exit_code': process.returncode,
              'command': [sys.executable, '-B', '-c', script], 'network_forbidden': True,
              'new_live_calls': 0, 'at_utc': now(), 'log': record(run/f'audit/test_output_attempt_{attempt}.txt')}
    atomic_json(run/'audit/test_report.json', result)
    return result


def prepare(run):
    run = Path(run)
    if (run/'audit/frozen_manifest.json').is_file():
        verify_frozen(run)
        return read(run/'audit/preflight_report.json')
    if list(run.glob('**/ledger/requests/*.json')):
        raise GuardrailStop('live_state_without_freeze')
    for label in ('P1_RUN', 'V1_RUN', 'V2_RUN', 'THREE_RUN'):
        if not (run/f'audit/{label}_inventory_before.json').is_file():
            raise GuardrailStop('historical_baseline_missing:' + label)
    for row in read(run/'audit/initial_code_identity.json'):
        if sha256_file(Path(row['path'])) != row['sha256']:
            raise GuardrailStop('preexisting_shared_code_changed:' + row['path'])
    copies = []
    for rid in ROOTS:
        for name in INPUT_NAMES:
            copies.append(copy_exact(THREE_RUN/'inputs/shared'/rid/name, run/'inputs/shared'/rid/name))
        p1 = P1_RUN/'inputs/shared'/rid/'expanded_dialogue_clean.json'
        dest = run/'inputs/shared'/rid/'expanded_dialogue_clean.json'
        if sha256_file(p1) != sha256_file(dest):
            raise GuardrailStop('P1_transcript_identity_mismatch')
        copies.append({'source': record(p1), 'copy': record(dest)})
    for folder in ('policy_snapshot/tokenizer', 'audit/service_evidence', 'common_base'):
        for p in sorted((THREE_RUN/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                copies.append(copy_exact(p, run/folder/p.relative_to(THREE_RUN/folder)))
    copies.append(copy_exact(WORK_ORDER, run/'policy_snapshot/CODEX_WORK_ORDER.md'))
    prior = read(THREE_RUN/'execution_contract.json')
    config = configured()
    if not (config.model == 'Qwen/Qwen3-14B' and config.base_url == 'https://api.siliconflow.cn/v1'
            and config.timeout_seconds == 180 and RATE_LIMIT_RETRY_LIMIT == 2
            and sha256_file(CONFIG) == prior['configuration']['sha256']):
        raise GuardrailStop('model_config_changed_since_three_arm')
    write_frozen(run/'execution_contract.json', {'version': VERSION, 'roots': list(ROOTS), 'arms': list(ARMS),
        'schedule': list(SCHEDULE), 'planned_logical_calls': 4, 'maximum_logical_calls': 4,
        'per_root_arm_limit': 1, 'per_root_limit': 2, 'per_arm_limit': 2,
        'generation': prior['generation'], 'configuration': record(CONFIG), 'context_reserve_tokens': 512,
        'zero_call_roles': prior['zero_call_roles'], 'work_order': record(WORK_ORDER),
        'baseline_is_fresh_generation_with_exact_prior_A_OFF_prompt': True, 'stop_after_four_call_diagnostic': True})
    write_frozen(run/'schedule.json', {'version': VERSION, 'schedule': list(SCHEDULE)})
    from .reporting import review_policy
    write_frozen(run/'review_policy.json', review_policy())
    write_frozen(run/'policy_snapshot/intent_selection_guidance_v3.txt', raw=guidance_text().encode())
    write_frozen(run/'policy_snapshot/schema.json', output_schema())
    specs, pairings, pair_reports = {}, [], []
    for rid in ROOTS:
        folder = run/'inputs/shared'/rid
        expanded, cards, history = [read(folder/name) for name in INPUT_NAMES[:3]]
        archived_policy = THREE_RUN/'policy_snapshot/prompts/A_OFF'/rid/'miner_prompt.txt'
        archived_actual = THREE_RUN/'arms/A_OFF/qa/shards/shard-00/roots'/rid/'raw/prompts'/f'{rid}-intent-miner.txt'
        copies.extend({'source': record(p), 'purpose': 'BASE historical byte identity'} for p in (archived_policy, archived_actual))
        pair_reports.append(prompt_pairing_diagnostics(expanded, cards, history, rid, archived_policy, archived_actual))
        for arm in ARMS:
            spec = build_prompt(arm, build_intent_prompt(expanded), cards, history, expanded=expanded, root_id=rid)
            specs[arm, rid] = spec
            write_frozen(run/'inputs/arms'/arm/rid/'feedback_payload.json', feedback_payload(cards, history, rid))
            write_frozen(run/'policy_snapshot/prompts'/arm/rid/'miner_prompt.txt', raw=spec['prompt'].encode())
            write_frozen(run/'policy_snapshot/prompts'/arm/rid/'prompt_spec.json', spec)
    for row in SCHEDULE:
        arm, rid = row['arm_id'], row['root_id']; spec = specs[arm, rid]
        pairings.append({**row, 'prompt_sha256': spec['prompt_sha256'],
            'prompt_path': str(run/'policy_snapshot/prompts'/arm/rid/'miner_prompt.txt'),
            'configuration_sha256': sha256_file(CONFIG),
            'transcript_sha256': sha256_file(run/'inputs/shared'/rid/'expanded_dialogue_clean.json'),
            'response_schema_sha256': spec['response_schema_sha256'], 'feedback_payload_sha256': spec['feedback_payload_sha256'],
            'historical_tracking_sha256': spec['historical_tracking_sha256'], 'max_output_tokens': 4096,
            'maximum_logical_calls': 1, 'planned_intent_slots': 3})
    write_frozen(run/'pairing_manifest.jsonl', raw=''.join(json.dumps(r,sort_keys=True)+'\n' for r in pairings).encode())
    atomic_json(run/'audit/prompt_pairing_report.json', {'status': 'PASS' if all(r['status']=='PASS' for r in pair_reports) else 'FAIL', 'roots': pair_reports})
    write_frozen(run/'input_path_audit.json', {'status': 'PASS', 'copies': copies, 'cleaning_or_renumbering': False})
    write_frozen(run/'implementation_manifest.json', {'version': VERSION,
        'new_code': str(CODE), 'old_files_changed': [], 'shared_miner': 'h4_qa_multiroot_k3_t08_probe_v1.generation.generate_root_intents',
        'baseline': 'Exact three-arm A_OFF build_prompt', 'sole_model_visible_delta': 'Frozen guidance block before unique authoritative dialogue marker',
        'full_schema_audit': 'Original schema; separate from unchanged canonicalizer admission',
        'new_semantic_production_gate': False, 'new_binding_or_writer_stage': False})
    tests = test_phase(run)
    with forbid_network():
        identity = prepare_tokenizer(run, config)
        token = TokenPreflight(run)
        token_rows = [token.check(specs[r['arm_id'],r['root_id']]['prompt'],4096,r['global_request_id']) for r in SCHEDULE]
        from .dry_run import exercise
        dry = exercise(run)
        coverage = collect_coverage(run, dry)
    old_identity = read(THREE_RUN/'audit/token_identity.json')
    same_token = all(identity[k] == old_identity[k] for k in ('model','service_base_url','context_limit','chat_template_sha256','chat_rendering','reserve_tokens'))
    checks = {'tests': tests['status']=='PASS', 'prompt_pairing': all(r['status']=='PASS' for r in pair_reports),
              'dependency_coverage': coverage['status']=='PASS', 'dry_run': dry['status']=='PASS',
              'tokenizer_identity': same_token, 'four_tokens_fit': len(token_rows)==4 and all(r['status']=='PASS' for r in token_rows),
              'exact_sources': True, 'configuration_identity': True}
    preflight = {'version': VERSION, 'status':'PASS' if all(checks.values()) else 'BLOCKED', 'checks':checks,
                 'requests':token_rows, 'live_calls':0, 'at_utc':now()}
    atomic_json(run/'audit/preflight_report.json', preflight)
    if preflight['status']=='PASS':
        code_paths = {Path(r['path']) for r in coverage['repository_files']}
        code_paths.update(p for p in CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        code = [record(p) for p in sorted(code_paths)]
        for row in code:
            copy_exact(row['path'], run/'policy_snapshot/frozen_code'/Path(row['path']).relative_to(REPO))
        inputs = set()
        for folder in ('inputs','common_base','policy_snapshot','audit/service_evidence'):
            inputs.update(p for p in (run/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        inputs.update(run/name for name in ('execution_contract.json','implementation_manifest.json','review_policy.json','schedule.json','pairing_manifest.jsonl','input_path_audit.json','audit/test_report.json','audit/prompt_pairing_report.json','audit/dependency_coverage_report.json','audit/token_identity.json','audit/initial_code_identity.json','audit/workspace_status_before.txt','audit/workspace_diff_before.patch'))
        inputs.update(run/f'audit/{label}_inventory_before.json' for label in ('P1_RUN','V1_RUN','V2_RUN','THREE_RUN'))
        sources = {r['source']['path']:r['source'] for r in copies}
        write_frozen(run/'audit/frozen_manifest.json', {'version':VERSION,'schedule':list(SCHEDULE),'code':code,
            'inputs':[record(p) for p in sorted(inputs)],'sources':list(sources.values()),'config':record(CONFIG),
            'preflight':record(run/'audit/preflight_report.json'),'environment':environment_identity()})
    atomic_text(run/'RUN_STATUS.md',f"Prepare: {preflight['status']}\nLive: NOT_RUN\nFour fresh Miner requests maximum; all other model roles zero.\n")
    return preflight


def live(run):
    run = Path(run)
    assert_no_guardrails(run)
    frozen = verify_frozen(run)
    if read(run/'audit/preflight_report.json')['status'] != 'PASS':
        raise GuardrailStop('preflight_not_passed')
    audit_reservations(run)
    token = TokenPreflight(run)
    summaries = []
    for row in SCHEDULE:
        arm, rid = row['arm_id'], row['root_id']; root = root_path(run,arm,rid); gid = row['global_request_id']
        assert_no_guardrails(run); verify_frozen(run)
        if verify_checkpoint(root):
            audit_reservations(run); summaries.append(read(root/'miner_generation_report.json')); continue
        expanded = read(run/'inputs/shared'/rid/'expanded_dialogue_clean.json')
        def before(request_id, metadata):
            verify_frozen(run)
            actual = (root/'raw/prompts'/f'{request_id}.txt').read_bytes()
            if actual != (run/'policy_snapshot/prompts'/arm/rid/'miner_prompt.txt').read_bytes():
                raise GuardrailStop('actual_prompt_vs_frozen_drift')
            # Rebuild both prompts and verify the unique insertion for every request.
            source = run/'inputs/shared'/rid
            cards, history = read(source/'observation_cards.json'), read(source/'history_only.json')
            pairing = prompt_pairing_diagnostics(expanded,cards,history,rid,
                THREE_RUN/'policy_snapshot/prompts/A_OFF'/rid/'miner_prompt.txt',
                THREE_RUN/'arms/A_OFF/qa/shards/shard-00/roots'/rid/'raw/prompts'/f'{rid}-intent-miner.txt')
            if pairing['status'] != 'PASS':
                raise GuardrailStop('request_prompt_pairing_failed')
            token.check(actual.decode(), metadata['max_tokens'], gid)
        ledger = BoundedLedger(root,sha256_file(run/'audit/frozen_manifest.json'),run,arm,rid,
            before_request=before,after_response=lambda request_id,response:token.verify_usage(gid,response.metadata.get('usage')))
        client = ObservedClient(OpenAICompatibleChallengerClient(configured()),run/'audit/transport_events.jsonl',arm,rid)
        print(json.dumps({**row,'status':'started','at_utc':now()}),flush=True)
        try:
            with repository_execution_guard(frozen['code']):
                result = generate_root_intents(expanded=expanded,root_dir=root,root_id=rid,api_config=CONFIG,
                    ledger=ledger,client=client,hooks=make_hooks(run,arm,rid))
            # The shared entry may translate Exceptions into intent errors. Trust
            # only conclusive ledger + HTTP evidence, never its normal return.
            execution = post_miner_check(run,root,arm,rid,ledger)
            verify_frozen(run)
            state = ledger.delegate.read(row['request_id'])
            if state is None:
                raise GuardrailStop('shared_miner_returned_without_reservation')
            if state['status'] != 'completed' and not (root/'ledger/observed_failures'/f"{row['request_id']}.json").is_file():
                raise GuardrailStop('shared_miner_returned_with_unknown_state')
        except BaseException as error:
            record_guardrail(run,'shared_stage_or_post_return_audit',error,arm_id=arm,root_id=rid,request_id=row['request_id'])
            raise
        summary = {**row,'status':execution['status'],
            'transport_status':state['status'],'valid_intents':len(result['contracts']),
            'intent_errors':result['intent_result']['errors'],'writer_calls':0,'at_utc':now()}
        atomic_json(root/'miner_generation_report.json',summary);checkpoint_root(root);summaries.append(summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    completion = {'status':'COMPLETED','groups':summaries,'at_utc':now()}
    atomic_json(run/'audit/live_completion.json',completion)
    atomic_text(run/'RUN_STATUS.md','Prepare: PASS\nLive: four scheduled attempts completed\nReview: pending\nAll non-Miner model roles: 0\n')
    return completion


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('prepare','live','report'))
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args(argv);run=args.run.resolve()
    if run != DEFAULT_RUN:
        raise RuntimeError('only_fixed_run_authorized')
    if not sys.dont_write_bytecode:
        raise RuntimeError('use_python_-B_to_preserve_historical_snapshots')
    try:
        with exclusive_run(run):
            if args.phase=='prepare': result=prepare(run)
            elif args.phase=='live': result=live(run)
            else:
                from .reporting import final_report
                result=final_report(run)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        return 0 if result.get('status') not in ('BLOCKED','FAILED') else 2
    except (GuardrailStop,Exception) as error:
        if args.phase=='live':
            record_guardrail(run,'live_entrypoint',error)
        path=run/'audit'/('execution_stop_'+args.phase+'_'+now().replace(':','-')+'.json')
        atomic_json(path,{'status':'BLOCKED','phase':args.phase,'reason':str(error),'at_utc':now(),'automatic_resend':False})
        print('OFF paired diagnostic stopped: '+str(error),flush=True)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
