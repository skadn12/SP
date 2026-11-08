"""Bounded offline preparation from six actual TEMPORAL requests; no network I/O."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import shutil
from functools import lru_cache
from pathlib import Path
from ..core import atomic_json, atomic_text
from .constants import *
from .config import load_profiles, read, assert_no_secrets
from .transport import build_payload


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'size': path.stat().st_size}


def _is_started(run):
    for name in ('LIVE_STARTED.json', 'audit/live_started.json', 'audit/call_events.jsonl', 'audit/http_events.jsonl'):
        path = run/name
        if path.exists() and path.stat().st_size:
            return True
    return any((run/'arms').glob('*/qa/shards/shard-00/roots/*/ledger/requests/*.json'))


def _source_records(source):
    manifest = read(source/'audit/frozen_manifest.json')
    return {row['path']: row for group in ('inputs', 'sources', 'code') for row in manifest.get(group, [])}


def _copy(source, destination, source_rows, sources, *, must_be_frozen=True):
    item = record(source)
    if must_be_frozen:
        old = source_rows.get(item['path'])
        if old is None or old['sha256'] != item['sha256'] or old['size'] != item['size']:
            raise RuntimeError('SELECTED_SOURCE_IDENTITY_MISMATCH:' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    sources[item['path']] = item
    return {'source': item, 'copy': record(destination)}


def _code_closure():
    """AST import closure of this entry and shared generation, not a repository scan."""
    queue = list(CODE.glob('*.py'))
    queue += [METHOD/'h4_qa_multiroot_k3_t08_probe_v1/generation.py',
              METHOD/'h4_qa_mining_hybrid_generation_v1/contract_guided_generator.py',
              METHOD/'h4_r02_ordering_temporal_contract_v1/contract.py']
    seen = set()
    while queue:
        path = queue.pop().resolve()
        if path in seen or not path.is_file() or path.name.startswith('test_'):
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    stems = [base.joinpath(*(node.module or '').split('.'))]
                    stems += [stems[0]/alias.name for alias in node.names if alias.name != '*']
                else:
                    stems = [REPO.joinpath(*(node.module or '').split('.'))]
            else:
                stems = [REPO.joinpath(*alias.name.split('.')) for alias in node.names]
            for stem in stems:
                for candidate in (stem.with_suffix('.py'), stem/'__init__.py'):
                    if candidate.is_file() and candidate.is_relative_to(REPO):
                        queue.append(candidate)
    return sorted(seen)


@lru_cache(maxsize=4)
def _tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def _capacity(profile, messages, max_output_tokens, request_id):
    result = {'request_id': request_id, 'arm_id': profile['arm_id'],
              'messages_sha256': object_sha(messages), 'max_output_tokens': max_output_tokens,
              'input_tokens': 'UNKNOWN', 'status': 'UNKNOWN'}
    accounting = profile.get('token_accounting') or {}
    limits = profile.get('limits') or {}
    context, output = limits.get('context_window_tokens'), limits.get('max_output_tokens')
    reason = limits.get('reasoning_budget_tokens')
    if (not accounting.get('local_tokenizer_path') or accounting.get('method') != 'huggingface_chat_template'
            or not isinstance(max_output_tokens, int) or not isinstance(context, int)
            or not isinstance(output, int) or not isinstance(reason, int)
            or limits.get('reasoning_in_output_budget') is None):
        result['reason'] = 'SERVICE_LIMITS_OR_TOKENIZER_NOT_CONFIGURED'
        return result
    tokenizer = _tokenizer(accounting['local_tokenizer_path'])
    kwargs = accounting.get('chat_template_kwargs')
    if not isinstance(kwargs, dict) or 'add_generation_prompt' not in kwargs:
        result['reason'] = 'EXACT_CHAT_TEMPLATE_KWARGS_REQUIRED'
        return result
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    count = len(tokenizer.encode(rendered, add_special_tokens=False))
    reserve = limits.get('reserve_tokens', 512)
    reasoning_reserve = 0 if limits['reasoning_in_output_budget'] else reason
    total = count + max_output_tokens + reasoning_reserve + reserve
    result.update(input_tokens=count, rendered_message_tokens=count, rendered_message_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                  context_window_tokens=context, reserve_tokens=reserve, reasoning_reserve_tokens=reasoning_reserve,
                  required_context_tokens=total, remaining_context_tokens=context-total,
                  status='PASS' if total <= context and max_output_tokens <= output else 'BLOCKED')
    return result


def _capacity_with_authorization(profile, messages, max_output_tokens, request_id):
    result = _capacity(profile, messages, max_output_tokens, request_id)
    exception = profile.get('user_service_exception') or {}
    if (profile['arm_id'] == ARMS[1] and result['status'] == 'UNKNOWN'
            and max_output_tokens == 8192
            and exception.get('user_explicitly_authorized') is True
            and exception.get('allow_unknown_service_capacity') is True
            and exception.get('record_id') == '2026-09-18-user-authorized-tju-service-unknowns-v1'):
        result.update(authorized_unknown=True, authorization_record_sha256=exception['record_sha256'],
                      capacity_verified=False, no_truncation=True,
                      qualification='User permits sending unchanged full messages with unknown service capacity; this is not a capacity PASS.')
    return result


def _capacity_sendable(result):
    return result['status'] == 'PASS' or (result['arm_id'] == ARMS[1] and result['status'] == 'UNKNOWN' and result.get('authorized_unknown') is True)


def check_capacity(run, arm, messages, max_output_tokens, request_id):
    profile = read(Path(run)/'profiles.json')[arm]
    result = _capacity_with_authorization(profile, messages, max_output_tokens, request_id)
    if not _capacity_sendable(result):
        raise RuntimeError('TOKEN_CAPACITY_' + result['status'] + ':' + request_id)
    return result


def prepare(run=DEFAULT_RUN, strong_config=STRONG_CONFIG, source_run=SOURCE_RUN):
    run, source, strong_config = Path(run).resolve(), Path(source_run).resolve(), Path(strong_config).resolve()
    if run == source or _is_started(run):
        raise RuntimeError('REFUSE_PREPARE_OVER_STARTED_OR_SOURCE_RUN')
    run.mkdir(parents=True, exist_ok=True)
    _tokenizer.cache_clear()
    if (run/'audit/frozen_manifest.json').exists():
        # Configuration may be completed later, but original inputs may not drift.
        old = read(run/'audit/frozen_manifest.json')
        for row in old.get('source_files', []):
            path = Path(row['path'])
            if path.is_relative_to(source) and (not path.is_file() or sha(path) != row['sha256']):
                raise RuntimeError('FROZEN_SOURCE_DRIFT_BEFORE_REPREPARE:' + str(path))
    source_rows, sources = _source_records(source), {}
    inputs, source_specs, memory = [], {}, []
    for root in ROOTS:
        for name in ('expanded_dialogue_clean.json', 'contracts.json', 'reasoning_contracts.json',
                     'history_only.json', 'provenance.json', 'source_intent_diagnostics.json'):
            inputs.append(_copy(source/'inputs/shared'/root/name, run/'inputs/shared'/root/name,
                                source_rows, sources))
        expanded = read(run/'inputs/shared'/root/'expanded_dialogue_clean.json')
        contracts = read(run/'inputs/shared'/root/'contracts.json')
        if expanded.get('root_id') != root or [c.get('intent_id') for c in contracts] != ['intent_1','intent_2','intent_3']:
            raise RuntimeError('FROZEN_INPUT_IDENTITY_MISMATCH:' + root)
        if contracts != read(run/'inputs/shared'/root/'reasoning_contracts.json'):
            raise RuntimeError('INTENT_COPY_MISMATCH:' + root)
        dialogue = '\n'.join(f"{t.get('turn_ref')} | {t.get('speaker')} | {t.get('text')}" for t in expanded['turns'])
        for i in range(1,4):
            slot = f'intent_{i}'
            original = source/'policy_snapshot/primary_requests'/SOURCE_ARM/root/slot
            target = run/'policy_snapshot/source_primary_requests'/root/slot
            for name in ('prompt.txt','request.json'):
                _copy(original/name,target/name,source_rows,sources)
            spec, prompt = read(target/'request.json'), (target/'prompt.txt').read_text(encoding='utf-8')
            messages = spec['payload']['messages']
            if messages != [{'role':'user','content':prompt}]:
                raise RuntimeError('SOURCE_PROMPT_MESSAGES_DIFFERENCE_REQUIRES_EXPLICIT_HANDLING:' + root + ':' + slot)
            if prompt.rsplit('DIALOGUE:\n',1)[-1] != dialogue:
                raise RuntimeError('FULL_TRANSCRIPT_NOT_EXACT:' + root + ':' + slot)
            if spec.get('ordering_policy') != arm_policy(ARMS[0]) or spec.get('expected_counts') != REQUESTED_COUNTS:
                raise RuntimeError('SOURCE_CONTRACT_DRIFT:' + root + ':' + slot)
            source_specs[(root,slot)] = (spec,prompt)
            memory.append({'root_id':root,'slot_id':slot,'message_roles':[m['role'] for m in messages],
                'message_count':len(messages),'full_transcript_turn_count':len(expanded['turns']),
                'full_transcript_exact':True,'dedicated_memory_block_present':False,
                'input_sections':[line for line in prompt.splitlines() if line in ('CONTRACT GUIDANCE:','OUTPUT SHAPE:','DIALOGUE:')],
                'source_hooks':spec.get('hooks'),'source_slot_suffix_sha256':spec.get('slot_suffix_sha256'),
                'memory_fact_source_flag_in_transcript_artifact':expanded.get('memory_is_qa_fact_source'),
                'finding':'Actual messages contain original intent, format/ordering guidance, synthetic format example and full dialogue; no appended Memory/feedback brief. Historical expanded dialogue may reflect earlier generation context; it is preserved exactly.',
                'new_memory_api_calls':0})
    for name in ('ordering_guidance.txt','ordering_policy.json'):
        _copy(source/'policy_snapshot'/name,run/'policy_snapshot'/name,source_rows,sources)
    for name in ('tokenizer.json','tokenizer_config.json','config.json','download_manifest.json'):
        _copy(source/'policy_snapshot/tokenizer'/name,run/'policy_snapshot/tokenizer'/name,source_rows,sources)
    for name in ('token_identity.json','service_evidence/service_model_metadata.json'):
        _copy(source/'audit'/name,run/'audit/source_evidence'/name,source_rows,sources,must_be_frozen=False)
    _copy(source/'review_policy.json',run/'audit/source_evidence/review_policy.json',source_rows,sources)
    from .reporting import prepare_review_assets
    prepare_review_assets(run, source)
    profiles = load_profiles(strong_config,source)
    profiles[ARMS[0]]['token_accounting']['local_tokenizer_path'] = str(run/'policy_snapshot/tokenizer')
    profiles[ARMS[0]]['token_accounting']['evidence_path'] = str(run/'audit/source_evidence/token_identity.json')
    # Additional provider evidence/tokenizer resources are local-only and narrowly selected.
    strong = profiles[ARMS[1]]
    exception = strong.get('user_service_exception')
    if exception:
        original = Path(exception['record_path'])
        dest = run/'audit/user_authorized_service_exception.json'
        _copy(original,dest,source_rows,sources,must_be_frozen=False)
        exception['record_path'] = str(dest)
    accounting = strong['token_accounting']
    local_tokenizer = accounting.get('local_tokenizer_path')
    if local_tokenizer:
        folder = Path(local_tokenizer).expanduser().resolve()
        for name in ('tokenizer.json','tokenizer_config.json','config.json','special_tokens_map.json','vocab.json','vocab.txt','merges.txt','tokenizer.model','spiece.model','chat_template.jinja'):
            if (folder/name).is_file():
                _copy(folder/name,run/'policy_snapshot/strong_tokenizer'/name,source_rows,sources,must_be_frozen=False)
        accounting['local_tokenizer_path'] = str(run/'policy_snapshot/strong_tokenizer')
    if accounting.get('evidence_path'):
        original = Path(accounting['evidence_path']).expanduser().resolve()
        dest = run/'audit/strong_service_evidence'/('token_accounting_'+original.name)
        _copy(original,dest,source_rows,sources,must_be_frozen=False)
        accounting['evidence_path'] = str(dest)
    for index, evidence in enumerate(strong['service_evidence']):
        if not isinstance(evidence,dict) or not evidence.get('path'):
            raise ValueError('SERVICE_EVIDENCE_REQUIRES_LOCAL_PATH')
        original = Path(evidence['path']).expanduser().resolve()
        dest = run/'audit/strong_service_evidence'/f'{index}_{original.name}'
        _copy(original,dest,source_rows,sources,must_be_frozen=False)
        evidence['path'] = str(dest)
    assert_no_secrets(profiles)
    atomic_json(run/'profiles.json',profiles)
    atomic_json(run/'schedule.json',SCHEDULE)
    pairing, token_rows, slot_rows = [], [], []
    for row in SCHEDULE:
        spec,prompt = source_specs[(row['root_id'],row['slot_id'])]
        profile = profiles[row['arm_id']]
        payload = build_payload(profile,spec['payload']['messages'])
        target = run/'policy_snapshot/primary_requests'/row['arm_id']/row['root_id']/row['slot_id']
        atomic_text(target/'prompt.txt',prompt)
        manifest = {**row,'source':'contract_guided','intent_guidance':True,'slot_suffix_sha256':hashlib.sha256(b'').hexdigest(),
            'hooks':None,'ordering_policy':arm_policy(row['arm_id']),'expected_counts':REQUESTED_COUNTS,
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'messages_sha256':object_sha(payload['messages']),
            'payload':payload,'sendable':not profile['unresolved'],
            'request_state':'READY_OFFLINE' if not profile['unresolved'] else 'TEMPLATE_WAITING_FOR_STRONG_MODEL_CONFIG',
            'source_global_request_id':spec['global_request_id']}
        atomic_json(target/'request.json',manifest)
        slot_rows.append({k:v for k,v in manifest.items() if k!='payload'})
        pairing.append({'arm_id':row['arm_id'],'root_id':row['root_id'],'slot_id':row['slot_id'],
            'prompt_sha256':manifest['prompt_sha256'],'messages_sha256':manifest['messages_sha256'],
            'prompt_equals_source':prompt == spec['payload']['messages'][0]['content'],
            'messages_equal_source':payload['messages']==spec['payload']['messages'],
            'transcript_sha256':sha(run/'inputs/shared'/row['root_id']/'expanded_dialogue_clean.json')})
        token_rows.append(_capacity_with_authorization(profile,payload['messages'],profile['output_limit'],row['global_request_id']))
    atomic_text(run/'slot_call_manifest.jsonl',''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in slot_rows))
    atomic_json(run/'input_manifest.json',{'source_run':str(source),'roots':list(ROOTS),'inputs':inputs,
        'source_arm':SOURCE_ARM,'original_intents_changed':False,'original_intents_per_root':3,
        'primary_pairing':pairing,'prompt_messages_difference':'NONE; exact single user message in all six source requests'})
    atomic_json(run/'audit/prompt_pairing_report.json',{'status':'PASS','paired_primary_count':6,'requests':pairing})
    atomic_json(run/'audit/memory_visibility_report.json',{'method':'inspection of actual frozen messages and exact dialogue suffix, plus original generator input closure','requests':memory,'new_memory_input':False})
    qwen_tokens=sum(r['input_tokens'] for r in token_rows if r['arm_id']==ARMS[0])
    budget={'planned_primary_calls':12,'maximum_repair_calls':12,'maximum_logical_calls':24,'maximum_http_attempts':24,
        'http_attempts_per_logical_call':1,'transport_retries':0,'sdk_retries':0,'ssl_fallback':False,
        'per_arm':{'primary':6,'maximum_repair':6,'maximum_logical':12,'maximum_http':12},
        'repair_policy':'one existing whole-batch replacement if below minimum; same model and original intent; no extra sampling',
        'unknown_completion_policy':'stop, preserve timeout/failure; never resend or automatically repair unknown completion',
        'technical_failure_policy':'stop all subsequent unsent requests on any technical failure; no technical failure may trigger content repair',
        'unknown_strong_service_capacity_authorized':bool(exception),
        'primary_tokens':token_rows,'qwen_primary_input_tokens':qwen_tokens,
        'qwen_primary_max_output_tokens':6*profiles[ARMS[0]]['output_limit'],
        'qwen_all_calls_max_output_tokens':12*profiles[ARMS[0]]['output_limit'],
        'strong_primary_input_tokens':'UNKNOWN' if any(r['input_tokens']=='UNKNOWN' for r in token_rows if r['arm_id']==ARMS[1]) else sum(r['input_tokens'] for r in token_rows if r['arm_id']==ARMS[1]),
        'strong_all_calls_max_output_tokens':12*strong['output_limit'] if strong['output_limit'] else 'UNKNOWN',
        'repair_input_tokens':'DYNAMIC: full original messages plus own-arm output; Qwen capacity checked before every send; strong capacity remains UNKNOWN under explicit user exception',
        'qwen_all_calls_context_enforced_total_token_ceiling':12*(profiles[ARMS[0]]['limits']['context_window_tokens']-profiles[ARMS[0]]['limits']['reserve_tokens']),
        'total_tokens_ceiling':'UNKNOWN: strong tokenizer, service capacity and reasoning accounting are unverified; exact full messages preserved under explicit user exception',
        'cost':'UNKNOWN','pricing':{a:profiles[a]['pricing'] for a in ARMS},'preparation_external_calls':0}
    atomic_json(run/'audit/token_budget_summary.json',budget)
    differences={key:{a:profiles[a].get(key) for a in ARMS} for key in ('model','endpoint','request_parameters','seed_policy','reasoning_policy','limits','output_limit')}
    atomic_json(run/'audit/service_compatibility.json',{'comparison_label':'MODEL_AND_SERVICE_CONFIGURATION_COMPARISON',
        'qualification':'Backend alias and service settings are not fully equivalent; no claim that model scale alone explains differences.',
        'differences':differences,'live_service_compatibility_verified':False,
        'unresolved':{a:profiles[a]['unresolved'] for a in ARMS},
        'unknown_configuration':{a:profiles[a].get('unknown_configuration',[]) for a in ARMS},
        'waived_configuration':{a:profiles[a].get('waived_configuration',[]) for a in ARMS},
        'requested_parameter_adoption':{a:profiles[a].get('requested_parameter_adoption',{}) for a in ARMS},
        'user_service_exception':exception,'actual_model_id_policy':'retain returned model identifier; missing means UNKNOWN'})
    atomic_json(run/'execution_contract.json',{'version':VERSION,'source_run':str(source),'arms':list(ARMS),
        'roots':list(ROOTS),'schedule':SCHEDULE,'generation_path':'contract_guided mixed TEMPORAL','requested_counts':REQUESTED_COUNTS,
        'ordering_policy':arm_policy(ARMS[0]),'primary_calls':12,'planned_primary_calls':12,'maximum_repair_calls':12,'maximum_logical_calls':24,
        'maximum_http_attempts':24,'new_sampling_seed_policy':{a:profiles[a]['seed_policy'] for a in ARMS},
        'free_calls':0,'production_guided_free_ratio_unchanged':'1:2','quality_prompt_block':False,
        'new_memory_input':False,'additional_model_stages':False,'live_authorized':bool(exception),'preparation_external_calls':0,
        'user_service_exception':exception,
        'technical_failure_policy':'STOP_ALL_UNSENT_REQUESTS_NO_RETRY_NO_CONTENT_REPAIR',
        'http_attempts_per_logical_call':1,'sdk_retries':0,'transport_retries':0,'ssl_fallback':False,
        'disabled_calls':['Miner','Planner','Realizer','Solver','Memory','training','audit_model'],
        'diagnostic_protocol':'existing source quality schema; Codex offline diagnosis; semantic DAG validity never implies content PASS',
        'repair_body_policy':'same original template, original intent and own-arm primary output; paired repairs naturally differ'})
    sources[str(strong_config)]=record(strong_config)
    sources[str(QWEN_CONFIG.resolve())]=record(QWEN_CONFIG)
    code_rows=[record(p) for p in _code_closure()]
    atomic_json(run/'audit/dependency_coverage_report.json',{'method':'recursive repository AST import closure from comparison entry and shared generation modules; no history library rescan','repository_files':code_rows})
    frozen_paths=[p for folder in ('inputs','policy_snapshot','audit/source_evidence','audit/strong_service_evidence') for p in (run/folder).rglob('*') if p.is_file()]
    frozen_paths += [run/name for name in ('profiles.json','schedule.json','slot_call_manifest.jsonl','input_manifest.json','execution_contract.json','review_policy.json','audit/prompt_pairing_report.json','audit/memory_visibility_report.json','audit/token_budget_summary.json','audit/service_compatibility.json','audit/dependency_coverage_report.json')]
    if exception:
        frozen_paths.append(run/'audit/user_authorized_service_exception.json')
    atomic_json(run/'audit/frozen_manifest.json',{'version':VERSION,'schedule':SCHEDULE,
        'files':[record(p) for p in sorted(set(frozen_paths))],'source_files':list(sources.values()),'code':code_rows,
        'credentials_in_manifest':False,'scope':'six source requests, two shared input sets and actual code/template/transport dependency closure only'})
    result=preflight(run)
    atomic_json(run/'audit/preflight_report.json',result)
    atomic_text(run/'RUN_STATUS.md',f"Status: {result['status']}\n\nOffline package prepared. External API calls: 0. Live generation has not started.\n")
    atomic_text(run/'ARTIFACT_INDEX.md','''Prepared artifacts:
- profiles.json: public model/service profiles; no keys.
- schedule.json and slot_call_manifest.jsonl: fixed 12 primary slots.
- input_manifest.json: exact original transcript and intent identities.
- policy_snapshot/primary_requests/: paired exact primary prompts/messages.
- audit/prompt_pairing_report.json and memory_visibility_report.json: actual input checks.
- audit/token_budget_summary.json: 12 primary, up to 12 repairs, at most 24 HTTP attempts.
- audit/service_compatibility.json and preflight_report.json: unresolved service configuration.
- audit/frozen_manifest.json: bounded source/input/code identity freeze.
''')
    return result


def verify(run=DEFAULT_RUN):
    run=Path(run)
    manifest=read(run/'audit/frozen_manifest.json')
    if manifest.get('version') != VERSION or manifest.get('schedule') != SCHEDULE:
        raise RuntimeError('FROZEN_EXPERIMENT_SCOPE_DRIFT')
    for group in ('files','source_files','code'):
        for row in manifest[group]:
            path=Path(row['path'])
            if not path.is_file() or path.stat().st_size != row['size'] or sha(path) != row['sha256']:
                raise RuntimeError('FROZEN_IDENTITY_DRIFT:' + str(path))
    expected={r['path'] for r in manifest['code'] if Path(r['path']).parent==CODE and not Path(r['path']).name.startswith('test_')}
    current={str(p.resolve()) for p in CODE.glob('*.py') if not p.name.startswith('test_')}
    if expected != current:
        raise RuntimeError('COMPARISON_CODE_INVENTORY_DRIFT')
    return {'status':'PASS','verified_files':sum(len(manifest[k]) for k in ('files','source_files','code'))}


def preflight(run=DEFAULT_RUN, require_live=False):
    run=Path(run)
    identity=verify(run)
    profiles=read(run/'profiles.json')
    budget=read(run/'audit/token_budget_summary.json')
    blockers=[f'{arm}:{item}' for arm in ARMS for item in profiles[arm]['unresolved']]
    blockers += [row['request_id']+':TOKEN_'+row['status'] for row in budget['primary_tokens'] if not _capacity_sendable(row)]
    engineering = _engineering_status(run)
    authorized_exception=profiles[ARMS[1]].get('user_service_exception')
    result={'status':'WAITING_FOR_STRONG_MODEL_CONFIG' if blockers else ('AUTHORIZED_WITH_SERVICE_UNKNOWNS' if authorized_exception else 'READY_FOR_BUDGET_AUTHORIZATION'),
        'engineering_status':engineering['status'],'engineering':engineering,'identity_status':'PASS','identity':identity,'blockers':blockers,'external_calls':0,
        'logical_call_budget':24,'http_attempt_budget':24,'live_ready':not blockers and engineering['status']=='PASS',
        'credential_check':'deferred to explicit live execution; preparation never reads key values',
        'authorization':'explicit user exception and 12 primary/12 repair/24 HTTP budget recorded; live CLI still enforces exact authorization flags' if authorized_exception else 'explicit budget approval required by live entry; prepare never sends',
        'user_service_exception':authorized_exception,
        'unknown_configuration':{a:profiles[a].get('unknown_configuration',[]) for a in ARMS},
        'waived_configuration':{a:profiles[a].get('waived_configuration',[]) for a in ARMS},
        'service_capacity_verification':{a:('UNKNOWN_USER_AUTHORIZED' if any(r['arm_id']==a and r.get('authorized_unknown') for r in budget['primary_tokens']) else 'PASS') for a in ARMS}}
    if require_live and (blockers or engineering['status']!='PASS'):
        raise RuntimeError('LIVE_PREFLIGHT_BLOCKED:' + ','.join(blockers + engineering['pending']))
    if require_live:
        result['status']='PASS'
    return result


def _engineering_status(run):
    frozen=read(Path(run)/'audit/frozen_manifest.json')
    expected={row['path']:row['sha256'] for row in frozen['code']}
    pending=[]
    for name in ('mock_report.json','regression_report.json'):
        path=Path(run)/'audit'/name
        if not path.is_file():
            pending.append(name+':MISSING')
            continue
        report=read(path)
        covered={row['path']:row['sha256'] for row in report.get('code',[])}
        if report.get('status')!='PASS' or any(covered.get(path)!=digest for path,digest in expected.items()):
            pending.append(name+':NOT_PASS_OR_CODE_CHANGED')
        if name=='mock_report.json' and report.get('profiles_sha256')!=sha(Path(run)/'profiles.json'):
            pending.append(name+':PROFILE_CHANGED')
    return {'status':'PENDING_MOCK_AND_REGRESSION' if pending else 'PASS','pending':pending}
