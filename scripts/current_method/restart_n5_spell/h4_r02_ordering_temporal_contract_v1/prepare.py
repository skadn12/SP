"""Build the frozen, offline-only preparation package for the ordering contract."""
from __future__ import annotations
import json, shutil, subprocess, sys, difflib, copy, re
from pathlib import Path
from ..core import atomic_json, atomic_text, sha256_file, sha256_bytes
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt, build_repair_prompt
from ..h4_r02_feedback_transfer_p2_2_v1.token_preflight import prepare_tokenizer, TokenPreflight, measure_request
from ..live_backends import load_challenger_config, RATE_LIMIT_RETRY_LIMIT
from .constants import *
from .contract import ordering_schema, VERSION as CONTRACT_VERSION
from .dependencies import record, collect_coverage, forbid_network, verify_frozen, environment_identity
from .guidance import ordering_guidance, synthetic_example


REQUIRED_SHARED_INPUTS = (
    'expanded_dialogue_clean.json',
    'contracts.json',
    'reasoning_contracts.json',
    'provenance.json',
    'history_only.json',
    'source_intent_diagnostics.json',
)
HISTORICAL_WRITER_ARMS = ('BASE_WRITER', 'SCMS_QUALITY_WRITER')
OFFLINE_DIAGNOSTIC_VISIBILITY = 'OFFLINE_POST_QUALITY_ONLY_NEVER_WRITER_INPUT'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def register(run, path):
    path = Path(path).resolve()
    target = run/'audit/source_manifest_before.json'
    data = read(target) if target.exists() else {'files': []}
    item = record(path)
    old = next((row for row in data['files'] if row['path'] == str(path)), None)
    if old is not None and old != item:
        raise RuntimeError('SOURCE_DRIFT:'+str(path))
    if old is None:
        data['files'].append(item)
        atomic_json(target, data)
    return item


def copy_exact(run, source, target):
    original = register(run, source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != Path(source).read_bytes():
            raise RuntimeError('EXISTING_COPY_CONFLICT:'+str(target))
    else:
        shutil.copyfile(source, target)
    return {'source': original, 'copy': record(target)}


def frozen_source(run, source):
    identity = register(run, source)
    frozen = read(SOURCE_WRITER/'audit/frozen_manifest.json')
    old = next((r for r in frozen['inputs']+frozen['sources']+frozen['code']
                if r['path'] == str(Path(source).resolve())), None)
    if old != identity:
        raise RuntimeError('HISTORICAL_FROZEN_SOURCE_MISMATCH:'+str(source))
    return identity


def _validate_shared_inputs(root, source, dest):
    """Require and identity-check each frozen shared input before it is consumed."""
    missing = [name for name in REQUIRED_SHARED_INPUTS
               if not (source / name).is_file()]
    if missing:
        raise RuntimeError(
            'REQUIRED_SHARED_INPUT_MISSING:' + root + ':' + ','.join(missing))

    # Validate the source before copying, then validate the run copy on the
    # second call.  copy_exact below independently enforces byte identity.
    input_dir = dest if all((dest / name).is_file() for name in REQUIRED_SHARED_INPUTS) else source
    expanded = read(input_dir / 'expanded_dialogue_clean.json')
    contracts = read(input_dir / 'contracts.json')
    reasoning_contracts = read(input_dir / 'reasoning_contracts.json')
    provenance = read(input_dir / 'provenance.json')
    history_only = read(input_dir / 'history_only.json')
    diagnostics = read(input_dir / 'source_intent_diagnostics.json')

    if not isinstance(expanded, dict) or expanded.get('root_id') != root:
        raise RuntimeError('INPUT_TRANSCRIPT_ROOT_IDENTITY_MISMATCH:' + root)
    expected_intents = ['intent_1', 'intent_2', 'intent_3']
    if (not isinstance(contracts, list) or contracts != reasoning_contracts or
            [c.get('intent_id') for c in contracts] != expected_intents):
        raise RuntimeError('ORIGINAL_THREE_CONTRACTS_REQUIRED:' + root)
    if not isinstance(provenance, dict) or provenance.get('root_id') != root:
        raise RuntimeError('INPUT_PROVENANCE_ROOT_IDENTITY_MISMATCH:' + root)
    if not isinstance(history_only, dict) or history_only.get('root_id') != root:
        raise RuntimeError('INPUT_HISTORY_ROOT_IDENTITY_MISMATCH:' + root)
    if (not isinstance(diagnostics, dict) or
            diagnostics.get('visibility') != OFFLINE_DIAGNOSTIC_VISIBILITY):
        raise RuntimeError('INPUT_DIAGNOSTIC_VISIBILITY_MISMATCH:' + root)
    records = diagnostics.get('records')
    if (not isinstance(records, list) or
            [row.get('declared_intent_id') for row in records] != expected_intents or
            any(row.get('root_id') != root for row in records)):
        raise RuntimeError('INPUT_DIAGNOSTIC_IDENTITY_MISMATCH:' + root)
    history_records = history_only.get('records', [])
    if (not isinstance(history_records, list) or
            any(row.get('root_id') != root for row in history_records)):
        raise RuntimeError('INPUT_HISTORY_RECORD_IDENTITY_MISMATCH:' + root)


def _copy_inputs(run):
    manifest=[]; slots=[]
    for name in ('audit/frozen_manifest.json', 'execution_contract.json', 'reports/final_report.md',
                 'reports/real_cases.md', 'review_policy.json'):
        register(run, SOURCE_WRITER/name)
    for root in ROOTS:
        source=SOURCE_WRITER/'inputs/shared'/root; dest=run/'inputs/shared'/root
        _validate_shared_inputs(root, source, dest)
        for src in sorted(source.rglob('*')):
            if src.is_file():
                frozen_source(run, src)
                manifest.append(copy_exact(run, src, dest/src.relative_to(source)))
        # Validate the copied bytes as well as the source before producing slots.
        _validate_shared_inputs(root, source, dest)
        contracts=read(dest/'contracts.json')
        for index, contract in enumerate(contracts):
            slots.append({'root_id': root, 'raw_index': index, 'slot_id': contract['intent_id'],
                          'contract': contract, 'source': record(source/'contracts.json'),
                          'transcript': record(dest/'expanded_dialogue_clean.json')})
    atomic_json(run/'input_manifest.json', {'version':VERSION,'files':manifest,'slots':slots,
        'roots':list(ROOTS),'source_run':str(SOURCE_WRITER), 'original_intents_changed':False,
        'input_consumed_by_writer':'contracts.json equals reasoning_contracts.json; original V3 artifacts copied for provenance only'})
    return slots


def configuration(run):
    original = read(SOURCE_WRITER/'execution_contract.json')
    if original['generation'] != GENERATION or original['configuration'] != record(CONFIG):
        raise RuntimeError('CONFIGURATION_DRIFT_FROM_COMPLETED_EXPERIMENT')
    config = load_challenger_config(CONFIG)
    if (config.model != GENERATION['model'] or config.base_url != GENERATION['base_url'] or
            RATE_LIMIT_RETRY_LIMIT != 2):
        raise RuntimeError('MODEL_OR_HTTP_POLICY_DRIFT')
    actual=[]
    expected = {}
    for root in ROOTS:
        for arm in HISTORICAL_WRITER_ARMS:
            for index in range(1, 4):
                primary_id = f'{root}-contract_guided-generator-intent_{index}'
                expected[(arm, root, primary_id)] = {
                    'stage': 'primary', 'intent_id': f'intent_{index}',
                    'primary_id': primary_id,
                }
    # Only these two historical slots needed the original whole-batch repair.
    for arm, root, index in (
            ('BASE_WRITER', 'cc-train-episode-122711', 3),
            ('SCMS_QUALITY_WRITER', 'cc-train-episode-198480', 1)):
        primary_id = f'{root}-contract_guided-generator-intent_{index}'
        repair_id = f'{root}-contract_guided-repair-intent_{index}'
        expected[(arm, root, repair_id)] = {
            'stage': 'repair', 'intent_id': f'intent_{index}',
            'primary_id': primary_id,
        }

    paths = sorted((SOURCE_WRITER/'arms').glob(
        '*/qa/shards/shard-00/roots/*/ledger/requests/*.json'))
    if len(paths) != len(expected):
        raise RuntimeError('COMPLETED_SOURCE_REQUEST_INVENTORY_MISMATCH')
    seen = set()
    for path in paths:
        relative = path.relative_to(SOURCE_WRITER/'arms').parts
        if (len(relative) != 9 or relative[1:5] != ('qa', 'shards', 'shard-00', 'roots') or
                relative[6:8] != ('ledger', 'requests')):
            raise RuntimeError('HISTORICAL_REQUEST_PATH_IDENTITY_MISMATCH:' + str(path))
        arm, _, _, _, _, root, _, _, filename = relative
        request_id = Path(filename).stem
        key = (arm, root, request_id)
        identity = expected.get(key)
        if identity is None or key in seen:
            raise RuntimeError('HISTORICAL_REQUEST_IDENTITY_MISMATCH:' + str(path))
        seen.add(key)
        item=read(path); meta=item.get('request_metadata', {}); backend=item.get('backend_identity', {})
        schedule_row = meta.get('schedule_row', {})
        expected_global = (
            f'{arm}::{root}::{identity["intent_id"]}::{identity["stage"]}::{request_id}')
        expected_schedule_global = (
            f'{arm}::{root}::{identity["intent_id"]}::primary::{identity["primary_id"]}')
        # A repair's metadata request_id/schedule row identify its primary slot;
        # the ledger request_id and global id identify the repair itself.
        if (item.get('request_id') != request_id or
                meta.get('arm_id') != arm or meta.get('root_id') != root or
                meta.get('slot_id') != identity['intent_id'] or
                meta.get('stage') != identity['stage'] or
                meta.get('request_id') != request_id or
                schedule_row.get('arm_id') != arm or
                schedule_row.get('root_id') != root or
                schedule_row.get('intent_id') != identity['intent_id'] or
                schedule_row.get('slot_id') != identity['intent_id'] or
                schedule_row.get('request_id') != identity['primary_id'] or
                schedule_row.get('global_request_id') != expected_schedule_global or
                meta.get('global_request_id') != expected_global):
            raise RuntimeError('HISTORICAL_REQUEST_IDENTITY_MISMATCH:' + str(path))
        if (backend.get('model')!=config.model or backend.get('base_url')!=config.base_url or
                backend.get('temperature')!=.1 or meta.get('temperature')!=.1 or
                meta.get('max_tokens')!=8192 or 'seed' in meta or
                meta.get('enable_thinking', False) is not False or
                backend.get('enable_thinking', False) is not False):
            raise RuntimeError('HISTORICAL_SENT_CONFIGURATION_MISMATCH:' + str(path))
        seen.add(key)
        actual.append({'source':register(run,path), 'request_id':request_id,
                       'root_id':root, 'arm_id':arm, 'intent_id':identity['intent_id'],
                       'stage':identity['stage'], 'model':backend['model'],
                       'base_url':backend['base_url'], 'temperature':meta['temperature'],
                       'enable_thinking':False, 'max_output_tokens':meta['max_tokens']})
    if seen != set(expected):
        raise RuntimeError('COMPLETED_SOURCE_REQUEST_INVENTORY_MISMATCH')
    atomic_json(run/'audit/writer_configuration_evidence.json', {'status':'PASS',
        'effective_configuration':GENERATION, 'configuration_identity':record(CONFIG),
        'actual_historical_requests':actual, 'credentials_printed':False,
        'service_current_state_queried':False})
    return config


def _prompts(run):
    rows=[]; pairs=[]
    for row in SCHEDULE:
        root=row['root_id']; contract=read(run/'inputs/shared'/root/'contracts.json')[row['raw_index']]
        expanded=read(run/'inputs/shared'/root/'expanded_dialogue_clean.json')
        base=build_prompt(expanded,contract,intent_guidance=True)
        new=build_prompt(expanded,contract,intent_guidance=True,ordering_policy=arm_policy(ARMS[1]))
        historical_frozen=SOURCE_WRITER/'policy_snapshot/primary_requests/BASE_WRITER'/root/row['slot_id']/'prompt.txt'
        historical_actual=SOURCE_WRITER/'arms/BASE_WRITER/qa/shards/shard-00/roots'/root/'raw/prompts'/(row['request_id']+'.txt')
        frozen_source(run,historical_frozen); register(run,historical_actual)
        if not base['prompt'].encode()==historical_frozen.read_bytes()==historical_actual.read_bytes():
            raise RuntimeError('BASE_PRIMARY_HISTORICAL_BYTE_MISMATCH')
        if new['prompt']==base['prompt'] or 'BEGIN_SCMS_QUALITY' in new['prompt']:
            raise RuntimeError('ORDERING_ONLY_PROMPT_DIFFERENCE_INVALID')
        if base['prompt'].split('DIALOGUE:\n',1)[1]!=new['prompt'].split('DIALOGUE:\n',1)[1]:
            raise RuntimeError('TRANSCRIPT_PROMPT_DRIFT')
        if any(marker in base['prompt'] or marker in new['prompt'] for marker in (
                'source_intent_diagnostics.json', 'history_only.json', 'provenance.json',
                OFFLINE_DIAGNOSTIC_VISIBILITY)):
            raise RuntimeError('OFFLINE_SIDECAR_ENTERED_WRITER_PROMPT')
        spec=new if row['arm_id']==ARMS[1] else base
        p=run/'policy_snapshot/primary_requests'/row['arm_id']/root/row['slot_id'];p.mkdir(parents=True,exist_ok=True)
        adaptation={'source':'contract_guided','intent_guidance':True,'slot_suffix_sha256':sha256_bytes(b''),
                    'hooks':None,'ordering_policy':arm_policy(row['arm_id']),
                    'ordering_context':{'run_id':run.name,'arm_id':row['arm_id'],'root_id':root}}
        payload={'model':GENERATION['model'],'messages':[{'role':'user','content':spec['prompt']}],
                 'temperature':.1,'max_tokens':8192,'enable_thinking':False,'response_format':{'type':'json_object'}}
        atomic_text(p/'prompt.txt',spec['prompt'])
        atomic_json(p/'request.json',{**row,**adaptation,'prompt_sha256':spec['prompt_sha256'],
                    'expected_counts':spec['expected_counts'],'payload':payload,
                    'transcript':record(run/'inputs/shared'/root/'expanded_dialogue_clean.json'),
                    'contract':record(run/'inputs/shared'/root/'contracts.json')})
        rows.append({**row,**adaptation,'stage':'primary','prompt_sha256':spec['prompt_sha256'],
            'prompt':record(p/'prompt.txt'),'request':record(p/'request.json'),'max_output_tokens':8192,
            'maximum_primary_calls':1,'maximum_repair_calls':1})
        if row['arm_id']==ARMS[0]:
            diff=run/'audit/prompt_diffs'/root/(row['slot_id']+'.diff')
            atomic_text(diff,''.join(difflib.unified_diff(base['prompt'].splitlines(True),new['prompt'].splitlines(True),fromfile=ARMS[0],tofile=ARMS[1])))
            changed=copy.deepcopy(contract); changed['discriminative_variable']='OFFLINE_ONLY_INTENT_MUTATION'
            for arm in ARMS:
                assert build_prompt(expanded,changed,ordering_policy=arm_policy(arm))['prompt']!=build_prompt(expanded,contract,ordering_policy=arm_policy(arm))['prompt']
            pairs.append({'root_id':root,'slot_id':row['slot_id'],'base_frozen_byte_equal':True,
                'base_actual_byte_equal':True,'both_arms_keep_intent':True,'same_transcript':True,
                'historical_frozen':record(historical_frozen),'historical_actual':record(historical_actual),
                'diff':record(diff),'allowed_differences':'Ordering version, event schema, temporal guidance, '
                    'synthetic ordering example and common evidence-field projection instructions; SC/MS unchanged.'})
    atomic_text(run/'slot_call_manifest.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    atomic_json(run/'audit/prompt_pairing_report.json',{'status':'PASS','pairs':pairs,'base_ordering_legacy':True,'new_ordering_contract':True,'guided_only_future':True,'free_offline_only':True,'primary_count':12})
    atomic_text(run/'audit/field_visibility_report.md',
        'Both arms retain the same complete original soft intent guidance and full transcript. '
        'Only ordering schema/guidance/example and deterministic evidence projection differ. '
        'No history QA, Miner diagnosis, feedback, corrected examples or SC/MS QUALITY block is inserted. '
        'Synthetic schema refs are explicitly example-only and cannot pass current-root ref validation. '
        'Free receives its original intent-label-only input and remains offline in this experiment.\n')
    return rows


def _policy(run):
    atomic_json(run/'policy_snapshot/ordering_policy.json',{'version':VERSION,'arms':{ARMS[0]:arm_policy(ARMS[0]),ARMS[1]:arm_policy(ARMS[1])},'node_counts_supported':[4,5,6],'future_node_count':4,'semantic_support_status':'NOT_ASSESSED','no_semantic_gate':True})
    for n in (4,5,6):
        atomic_json(run/'policy_snapshot/examples'/f'ordering_{n}.json',synthetic_example(n))
        atomic_json(run/'policy_snapshot/schema'/f'ordering_{n}.json',__import__(__package__+'.contract',fromlist=['ordering_schema']).ordering_schema(n))
    atomic_text(run/'policy_snapshot/ordering_guidance.txt','\n'.join(ordering_guidance(arm_policy(ARMS[1]))))


def _tests(run):
    command=[sys.executable,'-B','-m','unittest','discover','-s',str(CODE),'-t',str(REPO),'-p','test_*.py']
    proc=subprocess.run(command,cwd=REPO,capture_output=True,text=True)
    out=proc.stdout+proc.stderr; atomic_text(run/'audit/test_output.txt',out)
    result={'status':'PASS' if proc.returncode==0 else 'FAIL','exit_code':proc.returncode,'command':command,'output_tail':out[-12000:],'external_calls':0}
    atomic_json(run/'audit/test_report.json',result)
    if proc.returncode: raise RuntimeError('OFFLINE_TEST_FAILURE')
    return result


def prepare(run=DEFAULT_RUN):
    run=Path(run).resolve()
    if run!=DEFAULT_RUN.resolve(): raise RuntimeError('FIXED_EXPERIMENT_IDENTITY_REQUIRED')
    run.mkdir(parents=True,exist_ok=True)
    if (run/'audit/frozen_manifest.json').is_file():
        verify_frozen(run, CONFIG)
        return read(run/'audit/preflight_report.json')
    if list(run.glob('**/ledger/requests/*.json')):
        raise RuntimeError('UNFROZEN_EXECUTION_STATE_PRESENT')
    with forbid_network():
        _copy_inputs(run); _policy(run); config=configuration(run); prompts=_prompts(run)
        from .reporting import prepare_review_assets
        prepared_review=prepare_review_assets(run)
        try:
            source_token=SOURCE_WRITER/'policy_snapshot/tokenizer'; dest=run/'policy_snapshot/tokenizer';
            for source in sorted(source_token.rglob('*')):
                if source.is_file():
                    frozen_source(run, source)
                    copy_exact(run, source, dest/source.relative_to(source_token))
            for folder in ('audit/service_evidence',):
                src=SOURCE_WRITER/folder; out=run/folder
                for source in sorted(src.rglob('*')):
                    if source.is_file():
                        frozen_source(run, source)
                        copy_exact(run, source, out/source.relative_to(src))
            token_id=prepare_tokenizer(run,config)
            tp=TokenPreflight(run); token_rows=[tp.check(Path(r['prompt']['path']).read_text(),8192,r['global_request_id']) for r in prompts]
            atomic_json(run/'audit/token_budget_summary.json',{'status':'PASS','primary_requests':token_rows,'primary_count':12,'repair_count_upper_bound':12,'max_output_tokens':8192,'reserve_tokens':512,'actual_calls':0,'future_repair_tokens':'DYNAMIC_FULL_PROMPT_CHECK_REQUIRED'})
        except Exception as e:
            atomic_json(run/'audit/token_budget_summary.json',{'status':'BLOCKED','error':type(e).__name__+':'+str(e),'actual_calls':0})
            raise
        from .offline import exercise
        offline=exercise(run)
        # Keep the known ordering failure modes as transcript-anchored offline
        # fixtures.  This writes diagnostic evidence only; it is never fed to
        # a Writer, admitted as QA, or counted as model output.
        from .historical_cases import exercise as exercise_historical
        historical=exercise_historical(run)
        from .dry_run import exercise as exercise_scheduler
        scheduler=exercise_scheduler(run)
        combined={**offline, 'status':'PASS' if offline['status']=='PASS' and scheduler['status']=='PASS' else 'FAIL',
                  'repository_modules':sorted(set(offline['repository_modules'])|set(scheduler['repository_modules']))}
        combined['historical_fixtures'] = historical
        coverage=collect_coverage(run,combined)
        atomic_json(run/'audit/dependency_coverage_report.json',coverage)
        tests=_tests(run)
        contract={'version':VERSION,'roots':list(ROOTS),'arms':list(ARMS),'schedule':SCHEDULE,
            'generation':GENERATION,'configuration':record(CONFIG),
            'planned_primary_calls':12,'maximum_repair_calls':12,'maximum_logical_calls':24,
            'per_arm':{'primary':6,'repair':6,'total':12},
            'per_root_arm':{'primary':3,'repair':3,'total':6},
            'per_cell':{'primary':1,'repair':1},
            'context_reserve_tokens':512,'preparation_external_call_budget':0,
            'live_authorized_by_this_task':False,'guided_future_only':True,'free_calls_this_round':0,
            'writer_source':'contract_guided','intent_guidance':True,'slot_suffix':'','hooks':None,
            'node_count':4,'arm_ordering_policy':{arm:arm_policy(arm) for arm in ARMS},
            'ordering_guarantee':'unique order under proposed constraints; semantic support not assessed'}
        atomic_json(run/'execution_contract.json',contract); atomic_json(run/'schedule.json',SCHEDULE)
        checks={'offline_integration':offline['status']=='PASS','offline_scheduler':scheduler['status']=='PASS',
                'dependency_coverage':coverage['status']=='PASS','tests':tests['status']=='PASS',
                'primary_count':len(prompts)==12,'token_preflight':True,'free_calls_zero':True,
                'review_assets':prepared_review['status']=='PREPARED_NOT_RUN',
                'historical_fixtures':historical['status']=='PASS' and historical['external_calls']==0}
        status='PASS' if all(checks.values()) else 'BLOCKED'
        atomic_json(run/'audit/preflight_report.json',{'status':status,'engineering':'PASS' if status=='PASS' else 'BLOCKED',
            'preparation_status':'READY_FOR_AUTHORIZATION' if status=='PASS' else 'BLOCKED',
            'live_status':'PREPARED_NOT_RUN','actual_external_calls':0,'checks':checks})
        if status!='PASS':
            raise RuntimeError('PREFLIGHT_CHECK_FAILED')
        from .reporting import report
        report(run)
        code_files={p for p in CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        code_files|={Path(row['path']) for row in coverage['repository_files']}
        frozen_inputs={p for folder in ('inputs','policy_snapshot','audit') for p in (run/folder).rglob('*') if p.is_file()}
        frozen_inputs|={run/name for name in ('execution_contract.json','schedule.json','slot_call_manifest.jsonl',
            'input_manifest.json','review_policy.json','reports/preparation_report.md','reports/preparation_summary.json')}
        direct_sources=[{k:row[k] for k in ('path','sha256','size')}
                        for row in read(run/'audit/source_manifest_before.json')['files']]
        atomic_json(run/'audit/frozen_manifest.json',{'version':VERSION,'code':[record(p) for p in sorted(code_files)],
            'inputs':[record(p) for p in sorted(frozen_inputs)],'sources':direct_sources,
            'config':record(CONFIG),'preflight':record(run/'audit/preflight_report.json'),
            'environment':environment_identity(),'schedule':SCHEDULE})
        atomic_text(run/'RUN_STATUS.md','ENGINEERING=PASS\nPREPARED_NOT_RUN\nCONTENT=NOT_EVALUATED\nACTUAL_EXTERNAL_CALLS=0\n')
        return {'status':'PASS','preparation_status':'READY_FOR_AUTHORIZATION','actual_external_calls':0}
