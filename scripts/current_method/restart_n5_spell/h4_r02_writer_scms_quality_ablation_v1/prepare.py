"""Offline preparation of the independent SC/MS guidance comparison."""
from __future__ import annotations
import copy, datetime, difflib, hashlib, json, re, shutil, subprocess, sys
from dataclasses import replace
from pathlib import Path
from ..core import atomic_json, atomic_text, canonical_json, sha256_file, sha256_bytes
from ..h4_qa_mining_probe_v1.intent_contract import canonicalize_intents
from ..h4_qa_mining_probe_v1.reasoning_contract import build_reasoning_contract, contract_has_three_disjoint_groups
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt, build_repair_prompt
from ..h4_r02_feedback_transfer_p2_2_v1.token_preflight import prepare_tokenizer, TokenPreflight, measure_request
from ..live_backends import load_challenger_config, RATE_LIMIT_RETRY_LIMIT
from .constants import *
from .guidance import guidance_text, quality_suffix, prompt_spec, repair_spec, GUIDANCE_SHA256
from .dependencies import record, forbid_network, collect_coverage, environment_identity, verify_frozen


def read(path): return json.loads(Path(path).read_text())
def digest(value): return sha256_bytes(canonical_json(value).encode())

def register(run, path):
    path=Path(path).resolve(); manifest=run/'audit/source_manifest_before.json'
    data=read(manifest) if manifest.exists() else {'files':[]}
    actual=record(path); old=next((r for r in data['files'] if r['path']==str(path)),None)
    if old and any(old[k]!=actual[k] for k in ('sha256','size')): raise RuntimeError('SOURCE_DRIFT:'+str(path))
    if not old:
        data['files'].append({**actual,'registered_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        atomic_json(manifest,data)
    return actual


def copy_exact(run, source, target):
    original=register(run,source); target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.read_bytes()!=Path(source).read_bytes():raise RuntimeError('EXISTING_COPY_CONFLICT:'+str(target))
    else:shutil.copyfile(source,target)
    return {'source':original,'copy':record(target)}


def frozen_reference(run, source, manifest):
    identity=register(run,source)
    rows=manifest['inputs']+manifest['sources']+manifest['code']
    previous=next((r for r in rows if r['path']==str(Path(source).resolve())),None)
    if previous is None or any(previous[k]!=identity[k] for k in ('sha256','size')):
        raise RuntimeError('SOURCE_NOT_MATCHING_ORIGINAL_FREEZE:'+str(source))
    return identity


def source_inputs(run):
    for source in (SOURCE_WRITER/'audit/frozen_manifest.json',SOURCE_MINER/'audit/frozen_manifest.json',WORK_ORDER):register(run,source)
    wf=read(SOURCE_WRITER/'audit/frozen_manifest.json');mf=read(SOURCE_MINER/'audit/frozen_manifest.json')
    copies=[];slots=[];crosschecks=[]
    for root in ROOTS:
        source=SOURCE_WRITER/'inputs/shared'/root;dest=run/'inputs/shared'/root
        names=['expanded_dialogue_clean.json','contracts.json','reasoning_contracts.json','history_only.json','provenance.json','source_intent_diagnostics.json','original_v3/intents.json','original_v3/reasoning_contracts.json','original_v3/derived_intent_evidence.json','original_v3/response_schema_audit.json',f'original_v3/raw/responses/{root}-intent-miner.json.txt']
        for name in names:
            frozen_reference(run,source/name,wf);copies.append(copy_exact(run,source/name,dest/name))
            if name.startswith('original_v3/'):
                miner=SOURCE_MINER/'arms/V3_OFF/qa/shards/shard-00/roots'/root/name.removeprefix('original_v3/')
            elif name in ('contracts.json','reasoning_contracts.json'):
                miner=SOURCE_MINER/'arms/V3_OFF/qa/shards/shard-00/roots'/root/'reasoning_contracts.json'
            elif name=='source_intent_diagnostics.json':continue
            else:miner=SOURCE_MINER/'inputs/shared'/root/name
            frozen_reference(run,miner,wf)
            if name in ('expanded_dialogue_clean.json','history_only.json','provenance.json'):frozen_reference(run,miner,mf)
            if miner.read_bytes()!=(dest/name).read_bytes():raise RuntimeError('MINER_WRITER_SOURCE_COPY_MISMATCH:'+name)
            crosschecks.append({'writer':record(source/name),'miner':record(miner),'copy':record(dest/name),'byte_equal':True})
        expanded=read(dest/'expanded_dialogue_clean.json');raw=read(dest/'original_v3/raw/responses'/f'{root}-intent-miner.json.txt')
        parsed=read(dest/'original_v3/intents.json')['validation'];contracts=read(dest/'contracts.json');derived=read(dest/'original_v3/derived_intent_evidence.json');schema_audit=read(dest/'original_v3/response_schema_audit.json')
        schema=SOURCE_MINER/'policy_snapshot/schema.json';frozen_reference(run,schema,mf)
        import jsonschema
        jsonschema.Draft202012Validator(read(schema)).validate(raw)
        canonical=canonicalize_intents(raw,expanded_dialogue=expanded)
        if canonical!=parsed or not canonical['valid'] or len(contracts)!=3 or digest(raw)!=schema_audit['raw_response_sha256'] or not schema_audit['valid']:raise RuntimeError('RAW_PARSE_SCHEMA_MISMATCH')
        for index,(intent,contract) in enumerate(zip(canonical['intents'],contracts)):
            expected=build_reasoning_contract(intent);expected['contract_valid']=contract_has_three_disjoint_groups(expected)
            if expected!=contract or contract['intent_id']!=f'intent_{index+1}' or contract['required_evidence_groups']!=derived['items'][index]['derived_evidence_groups'] or derived['items'][index]['transcript_sha256']!=sha256_file(dest/'expanded_dialogue_clean.json'):raise RuntimeError('ORIGINAL_SLOT_IDENTITY_MISMATCH')
            slots.append({'root_id':root,'raw_slot':index+1,'raw_index':index,'intent_id':contract['intent_id'],'contract_sha256':digest(contract),'source_contracts':record(source/'contracts.json'),'copy_contracts':record(dest/'contracts.json'),'selected_by_content':False,'original_structurally_valid':True})
        # Diagnostics remain an exact offline sidecar, cross-checked to original V3 findings.
        review=SOURCE_MINER/'reports/intent_evidence_review.jsonl';frozen_reference(run,review,wf)
        findings=[json.loads(line) for line in review.read_text().splitlines() if line.strip()]
        findings=[x for x in findings if x['root_id']==root and x['arm_id']=='V3_OFF']
        if findings!=read(dest/'source_intent_diagnostics.json')['records']:raise RuntimeError('SOURCE_DIAGNOSTIC_IDENTITY_MISMATCH')
    atomic_json(run/'input_manifest.json',{'status':'PASS','copies':copies,'slots':slots,'crosschecks':crosschecks,'source_writer':str(SOURCE_WRITER),'source_miner':str(SOURCE_MINER),'all_six_original_slots':True,'transcript_byte_identity':True,'model_visible_inputs':['full transcript','original corresponding Guided contract','QUALITY only fixed SC/MS guidance block'],'offline_only_inputs':['history_only','provenance','source_intent_diagnostics','original_v3 audit files'],'historical_QA_or_constructed_examples_used_as_model_input':False,'historical_response_cache_reused':False})
    return slots


def historical_writer_config(run):
    path=SOURCE_WRITER/'execution_contract.json';register(run,path);prior=read(path)
    config=replace(load_challenger_config(CONFIG),temperature=.1,enable_thinking=False,timeout_seconds=180)
    if prior['generation']!=GENERATION or record(CONFIG)!=prior['configuration']:raise RuntimeError('SOURCE_WRITER_CONFIG_DRIFT')
    if config.model!=GENERATION['model'] or config.base_url!=GENERATION['base_url'] or RATE_LIMIT_RETRY_LIMIT!=2:raise RuntimeError('ACTUAL_WRITER_CONFIG_DRIFT')
    requests=[]
    for p in sorted(SOURCE_WRITER.glob('arms/**/ledger/requests/*.json')):
        identity=register(run,p);d=read(p);meta=d['request_metadata'];backend=d['backend_identity']
        if meta['max_tokens']!=8192 or meta['temperature']!=.1 or backend['model']!=config.model or backend['base_url']!=config.base_url or 'seed' in meta:raise RuntimeError('HISTORICAL_REQUEST_CONFIG_MISMATCH')
        requests.append({'source':identity,'request_id':d['request_id'],'stage':meta['stage'],'status':d['status'],'max_output_tokens':meta['max_tokens'],'temperature':meta['temperature'],'backend':backend})
    if len(requests)!=15 or not any(r['stage']=='repair' for r in requests):raise RuntimeError('ACTUAL_WRITER_REFERENCE_INCOMPLETE')
    atomic_json(run/'audit/writer_configuration_evidence.json',{'status':'PASS','execution_contract':record(path),'actual_requests':requests,'effective_configuration':GENERATION,'config_identity':record(CONFIG),'credentials_printed':False,'historical_response_used_for_new_QA':False})
    return config


def freeze_guidance(run):
    text=WORK_ORDER.read_text();match=re.search(r'```text\n(BEGIN_SCMS_QUALITY_GUIDANCE_V1\n.*?END_SCMS_QUALITY_GUIDANCE_V1)\n```',text,re.S)
    if not match or sha256_bytes(match.group(1).encode())!=GUIDANCE_SHA256:raise RuntimeError('WORK_ORDER_GUIDANCE_MISMATCH')
    p=run/'policy_snapshot/scms_quality_guidance_v1.txt'
    if p.exists() and p.read_bytes()!=match.group(1).encode():raise RuntimeError('EXISTING_GUIDANCE_CONFLICT')
    atomic_text(p,match.group(1));copy_exact(run,WORK_ORDER,run/'policy_snapshot/CODEX_WORK_ORDER.md')
    return guidance_text(run)


def prepare_requests(run,slots):
    block=guidance_text(run);manifest=[];pairs=[];repairs=[]
    slot_index={(s['root_id'],s['raw_slot']):s for s in slots}
    for row in SCHEDULE:
        arm,root,index=row['arm_id'],row['root_id'],row['raw_index'];basepath=run/'inputs/shared'/root
        expanded=read(basepath/'expanded_dialogue_clean.json');contract=read(basepath/'contracts.json')[index]
        base=prompt_spec(expanded,contract,ARMS[0],run);quality=prompt_spec(expanded,contract,ARMS[1],run)
        historical_frozen=SOURCE_WRITER/'policy_snapshot/primary_requests/WITH_CURRENT_INTENT'/root/row['slot_id']/'prompt.txt'
        historical_actual=SOURCE_WRITER/'arms/WITH_CURRENT_INTENT/qa/shards/shard-00/roots'/root/'raw/prompts'/(row['request_id']+'.txt')
        frozen_reference(run,historical_frozen,read(SOURCE_WRITER/'audit/frozen_manifest.json'));register(run,historical_actual)
        expected=build_prompt(expanded,contract,intent_guidance=True)
        if not base['prompt'].encode()==historical_frozen.read_bytes()==historical_actual.read_bytes()==expected['prompt'].encode():raise RuntimeError('BASE_DOUBLE_HISTORICAL_BYTE_MISMATCH')
        if quality['prompt']!=base['prompt']+'\n'+block or quality['prompt'].count('BEGIN_SCMS_QUALITY_GUIDANCE_V1')!=1:raise RuntimeError('QUALITY_ADDITIVE_IDENTITY_MISMATCH')
        # Both arms must consume all original contract semantics; no old second-arm-off branch.
        modified=copy.deepcopy(contract);modified['discriminative_variable']='OFFLINE_SEMANTIC_DEPENDENCY_SENTINEL'
        if any(prompt_spec(expanded,modified,a,run)['prompt']==prompt_spec(expanded,contract,a,run)['prompt'] for a in ARMS):raise RuntimeError('BOTH_ARMS_INTENT_DEPENDENCY_REQUIRED')
        suffix=quality_suffix(run,arm);spec=base if arm==ARMS[0] else quality
        adaptation={'source':'contract_guided','generation_source':'contract_guided','intent_guidance':True,'slot_suffix_sha256':sha256_bytes(suffix.encode()),'hooks':None,'prompt_sha256':spec['prompt_sha256']}
        request={**row,**adaptation,'stage':'primary','max_output_tokens':8192,'model_config':GENERATION,'expected_counts':spec['expected_counts'],'source_slot':slot_index[root,row['raw_slot']],'transcript':record(basepath/'expanded_dialogue_clean.json'),'contract':record(basepath/'contracts.json'),'builder':record(METHOD/'h4_qa_mining_hybrid_generation_v1/contract_guided_generator.py'),'shared_stage':record(METHOD/'h4_qa_multiroot_k3_t08_probe_v1/generation.py'),'quality_guidance':record(run/'policy_snapshot/scms_quality_guidance_v1.txt') if arm==ARMS[1] else None,'payload':{'model':GENERATION['model'],'messages':[{'role':'user','content':spec['prompt']}],'temperature':.1,'max_tokens':8192,'enable_thinking':False,'response_format':{'type':'json_object'}}}
        dest=run/'policy_snapshot/primary_requests'/arm/root/row['slot_id'];atomic_text(dest/'prompt.txt',spec['prompt']);atomic_json(dest/'request.json',request)
        manifest.append({**row,**adaptation,'stage':'primary','request_spec':record(dest/'request.json'),'prompt':record(dest/'prompt.txt'),'max_output_tokens':8192,'maximum_primary_calls':1,'maximum_repair_calls':1,'repair_request_id':f'{root}-contract_guided-repair-{row["slot_id"]}'})
        if arm==ARMS[0]:
            diff=run/'audit/prompt_diffs'/root/(row['slot_id']+'.diff');atomic_text(diff,''.join(difflib.unified_diff(base['prompt'].splitlines(keepends=True),quality['prompt'].splitlines(keepends=True),fromfile=ARMS[0],tofile=ARMS[1])))
            pairs.append({'root_id':root,'slot_id':row['slot_id'],'base_original_builder_equal':True,'base_historical_frozen_equal':True,'base_historical_actual_equal':True,'quality_only_newline_plus_fixed_block':True,'both_arms_intent_guidance':True,'both_intent_mutation_changes_prompt':True,'ordering_schema_quotas_base_template_preserved':True,'guidance_sha256':GUIDANCE_SHA256,'historical_prompts':[record(historical_frozen),record(historical_actual)],'diff':record(diff)})
            fixtures=[]
            for raw in (None,{'candidates':[{'question':'OFFLINE_SYNTHETIC_ONLY','gold':['A','B']}]}):
                rb=repair_spec(expanded,contract,raw,ARMS[0],run);rq=repair_spec(expanded,contract,raw,ARMS[1],run);old=build_repair_prompt(expanded,contract,raw,'contract_guided',intent_guidance=True)
                if rb!=old or rq['prompt']!=rb['prompt']+'\n'+block or rq['prompt'].count('BEGIN_SCMS_QUALITY_GUIDANCE_V1')!=1:raise RuntimeError('REPAIR_GUIDANCE_INHERITANCE_MISMATCH')
                fixtures.append({'raw':'None' if raw is None else 'SYNTHETIC_NOT_QA','base_original_repair_byte_equal':True,'quality_only_newline_plus_fixed_block':True,'same_suffix_as_primary':True,'single_append':True})
            repairs.append({'root_id':root,'slot_id':row['slot_id'],'checks':fixtures})
    atomic_text(run/'slot_call_manifest.jsonl',''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in manifest))
    atomic_json(run/'audit/prompt_pairing_report.json',{'status':'PASS','pairs':pairs,'pair_count':6,'primary_count':12,'only_method_difference':'Fixed SC/MS suffix on QUALITY; both arms same complete intent','guidance_bytes':len(block.encode()),'guidance_sha256':GUIDANCE_SHA256})
    atomic_json(run/'audit/repair_inheritance_report.json',{'status':'PASS','slots':repairs,'real_repair_prompts_frozen':0,'real_repair_raw_unavailable':True,'real_future_arm_responses_can_differ':True,'dynamic_exact_capacity_before_each_send':True})
    atomic_text(run/'audit/field_visibility_report.md','''两臂均显示同一原Guided contract（ID、type、variable、required_evidence_groups）与完整原对白，intent_guidance=True。QUALITY只追加固定SC/MS块，BASE为空suffix，hooks均None。history_only、原Miner诊断、旧QA、反馈卡、bindings/briefs及修正示例均不输入模型。两臂normalizer接收同一原contract；后续intent对齐/错误传播均适用。输出schema、排序显式规则与原准入保持；不宣称mixed响应中的排序实际输出不受间接影响。\n''')
    return manifest


def token_assets(run,config,manifest):
    wf=read(SOURCE_WRITER/'audit/frozen_manifest.json')
    for folder in ('policy_snapshot/tokenizer','audit/service_evidence'):
        for p in sorted((SOURCE_WRITER/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                frozen_reference(run,p,wf);copy_exact(run,p,run/folder/p.relative_to(SOURCE_WRITER/folder))
    for row in read(run/'audit/service_evidence/historical_token_verification.json')['rows']:
        register(run,Path(row['ledger_path']));register(run,Path(row['prompt_path']))
    identity=prepare_tokenizer(run,config);tokens=TokenPreflight(run)
    checks=[tokens.check(Path(row['prompt']['path']).read_text(),8192,row['global_request_id']) for row in manifest]
    root=ROOTS[0];expanded=read(run/'inputs/shared'/root/'expanded_dialogue_clean.json');contract=read(run/'inputs/shared'/root/'contracts.json')[0];capacity=[]
    for arm in ARMS:
        small=repair_spec(expanded,contract,{'candidates':[]},arm,run)['prompt'];large=repair_spec(expanded,contract,{'candidates':[{'question':' capacity'*identity['context_limit']}]},arm,run)['prompt']
        ordinary=measure_request(tokens.tokenizer,prompt=small,max_output_tokens=8192,request_id='SYNTHETIC_REPAIR::'+arm,context_limit=identity['context_limit'],supported_max_output=identity['verified_supported_max_output_tokens'])
        oversized=measure_request(tokens.tokenizer,prompt=large,max_output_tokens=8192,request_id='SYNTHETIC_OVERSIZED::'+arm,context_limit=identity['context_limit'],supported_max_output=identity['verified_supported_max_output_tokens'])
        if ordinary['status']!='PASS' or oversized['status']!='BLOCKED':raise RuntimeError('REPAIR_CAPACITY_GUARD_FAILED')
        capacity.append({'arm_id':arm,'ordinary_fixture':ordinary,'oversized_fixture':oversized,'synthetic_only':True})
    atomic_json(run/'audit/repair_capacity_test.json',{'status':'PASS','arms':capacity,'real_repair_prompts':0,'actual_service_calls':0,'policy':'Actual future full repair prompt checked before reservation and every HTTP send; do not truncate or change output budget.'})
    atomic_json(run/'audit/token_budget_summary.json',{'primary_count':12,'primary_input_tokens':sum(x['rendered_message_tokens'] for x in checks),'by_arm':{a:sum(c['rendered_message_tokens'] for c,m in zip(checks,manifest) if m['arm_id']==a) for a in ARMS},'primary_output_token_upper_bound':12*8192,'conditional_repair_count_upper_bound':12,'conditional_repair_output_token_upper_bound':12*8192,'conditional_repair_input_tokens':'UNKNOWN_UNTIL_PRIMARY_RESPONSES','actual_usage':{'input_tokens':0,'output_tokens':0,'total_tokens':0},'actual_calls':0,'future_money':'UNKNOWN_NO_VERIFIED_BILLING','per_request_output_budget':8192,'per_request_reserve':512,'service_evidence_manifest':read(run/'audit/service_evidence/evidence_manifest.json'),'service_current_state_queried':False,'hosted_model_revision':None,'limits_note':'Local immutable tokenizer and earlier service evidence only; live service revision/limits not queried.'})
    return identity,checks


def test_phase(run):
    names=['scripts.current_method.restart_n5_spell.h4_r02_writer_intent_ablation_v1.test_shared_stage','scripts.current_method.restart_n5_spell.h4_r02_feedback_transfer_p2_2_v1.test_generation_hooks']
    script="""import unittest,sys
sys.dont_write_bytecode=True
from scripts.current_method.restart_n5_spell.h4_r02_writer_scms_quality_ablation_v1.dependencies import forbid_network
loader=unittest.TestLoader();suite=unittest.TestSuite()
suite.addTests(loader.discover(%r,pattern='test_*.py',top_level_dir=%r))
for name in %r:suite.addTests(loader.loadTestsFromName(name))
with forbid_network():result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
"""%(str(CODE),str(REPO),names)
    proc=subprocess.run([sys.executable,'-B','-c',script],cwd=REPO,capture_output=True,text=True)
    output=proc.stdout+proc.stderr;attempt=len(list((run/'audit').glob('test_output_attempt_*.txt')))+1;log=run/f'audit/test_output_attempt_{attempt}.txt';atomic_text(log,output)
    count=re.search(r'Ran (\d+) tests?',output)
    result={'status':'PASS' if proc.returncode==0 else 'FAIL','tests_run':int(count.group(1)) if count else None,'exit_code':proc.returncode,'command':[sys.executable,'-B','-c',script],'log':record(log),'network_forbidden':True,'external_calls':0}
    atomic_json(run/'audit/test_report.json',result);return result


def preservation(run):
    rows=[]
    for before in read(run/'audit/source_manifest_before.json')['files']:
        after=record(before['path']);rows.append({'before':before,'after':after,'unchanged':all(before[k]==after[k] for k in ('sha256','size'))})
    data={'status':'PASS' if all(r['unchanged'] for r in rows) else 'FAIL','file_count':len(rows),'changed_count':sum(not r['unchanged'] for r in rows),'scope':'Only registered historical references, inputs, operational token evidence and shared/previous repository dependencies; not entire history tree.','whole_history_rescanned':False,'rows':rows}
    atomic_json(run/'audit/source_preservation_report.json',data);return data


def implementation(run,coverage):
    shared=[]
    for row in coverage['repository_files']:
        path=Path(row['path'])
        if not path.is_relative_to(CODE):shared.append(register(run,path))
    atomic_json(run/'implementation_manifest.json',{'version':VERSION,'new_code':str(CODE),'shared_changes':[],'previous_experiment_code_changes':[],'shared_files_identity':shared,'unrelated_workspace_changes_preserved':True,'shared_batch':'h4_qa_multiroot_k3_t08_probe_v1.generation.execute_writer_batch','shared_finalizer':'h4_qa_multiroot_k3_t08_probe_v1.generation.finalize_writer_candidates','both_arms_intent_guidance':True,'source':'contract_guided','hooks':None,'only_method_difference':'QUALITY slot_suffix fixed SCMS quality block; BASE empty suffix','new_scheduler_reason':'Independent experiment identity/arm budget and semantics; no patching old runner globals','new_reporter_reason':'Both arms intent diagnostic; QUALITY minus BASE; SC/MS separately; explicit completion criteria','generation_parsing_repair_admission_copied':False,'candidate_schema_modified':False,'ordering_explicit_rules_modified':False,'semantic_filter_added':False,'guidance':record(run/'policy_snapshot/scms_quality_guidance_v1.txt')})


def prepare(run=DEFAULT_RUN):
    run=Path(run).resolve()
    if run!=DEFAULT_RUN.resolve():raise RuntimeError('FIXED_EXPERIMENT_IDENTITY_REQUIRED')
    if (run/'audit/frozen_manifest.json').exists():verify_frozen(run,CONFIG);return read(run/'audit/preflight_report.json')
    if list(run.glob('**/ledger/requests/*.json')):raise RuntimeError('UNFROZEN_LIVE_STATE_PRESENT')
    with forbid_network():
        block=freeze_guidance(run);slots=source_inputs(run);config=historical_writer_config(run)
        atomic_json(run/'schedule.json',SCHEDULE)
        atomic_json(run/'execution_contract.json',{'version':VERSION,'roots':list(ROOTS),'arms':list(ARMS),'generation':GENERATION,'configuration':record(CONFIG),'schedule':SCHEDULE,'planned_primary_calls':12,'maximum_repair_calls':12,'maximum_logical_calls':24,'per_arm':{'primary':6,'repair':6,'total':12},'per_root_arm':{'primary':3,'repair':3,'total':6},'per_cell':{'primary':1,'repair':1},'target_candidates_per_arm':24,'target_SC_MS_per_arm':18,'target_ordering_per_arm':6,'context_reserve_tokens':512,'preparation_external_call_budget':0,'live_authorized_by_this_task':False,'both_arms_intent_guidance':True,'source':'contract_guided','hooks':None,'guidance_sha256':GUIDANCE_SHA256,'arm_adaptation':{a:{'intent_guidance':True,'source':'contract_guided','slot_suffix_sha256':sha256_bytes(quality_suffix(run,a).encode()),'hooks':None} for a in ARMS},'comparison_direction':'SCMS_QUALITY_WRITER_MINUS_BASE_WRITER','zero_call_roles':['Miner','Free','Solver','Planner','Realizer','Memory','training','independent_review_API']})
        atomic_json(run/'policy_snapshot/repair_policy.json',{'trigger':'Original structural valid type minima 2/1/1 shortfall only','max_repairs_per_cell':1,'replacement':'WHOLE_BATCH_REPLACEMENT_NO_MERGE_NO_FALLBACK','raw_input':'Primary parsed JSON object; None on JSON failure remains original raw or {} handling','extra_valid_rows':'No repair if minima met, even exact-count errors','final_validator_rejection':'No additional repair; multi_select one-Gold still rejected, no program Gold supplementation','both_arms_intent_guidance':True,'slot_suffix_inherits_same_arm_bytes':True,'real_repair_prompts_frozen':0,'dynamic_capacity_check':'Full actual repair prompt +8192+512 before each send; no truncation','unknown_completion':'Sticky stop before repair/next request','semantic_retry':False})
        from .reporting import prepare_review_assets, report
        prepare_review_assets(run)
        manifest=prepare_requests(run,slots);identity,tokens=token_assets(run,config,manifest)
        tests=test_phase(run)
        if tests['status']!='PASS':
            data={'status':'BLOCKED','reason':'OFFLINE_TEST_FAILURE','test_report':tests,'external_calls':0};atomic_json(run/'audit/preflight_report.json',data);return data
        from .dry_run import exercise
        dry=exercise(run);coverage=collect_coverage(run,dry);implementation(run,coverage);preserved=preservation(run);review=report(run)
        checks={'six_original_slots':len(slots)==6,'primary_requests':len(manifest)==12,'base_double_historical_identity':True,'quality_only_frozen_suffix':True,'both_arms_intent_guidance':all(m['intent_guidance'] for m in manifest),'offline_tests':tests['status']=='PASS','dry_run':dry['status']=='PASS','dependency_coverage':coverage['status']=='PASS','primary_capacity':len(tokens)==12 and all(t['status']=='PASS' for t in tokens),'source_preservation':preserved['status']=='PASS','review_prepared_no_results':review['status']=='PREPARED_NOT_RUN' and not review['errors'],'external_calls_zero':not list(run.glob('**/ledger/requests/*.json'))}
        result={'version':VERSION,'status':'PASS' if all(checks.values()) else 'BLOCKED','preparation_status':'READY_FOR_AUTHORIZATION' if all(checks.values()) else 'BLOCKED','live_status':'PREPARED_NOT_RUN','actual_external_calls':0,'checks':checks};atomic_json(run/'audit/preflight_report.json',result)
        if result['status']=='PASS':
            code_paths={Path(x['path']) for x in coverage['repository_files']}|{p for p in CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
            for p in sorted(code_paths):
                dest=run/'policy_snapshot/frozen_code'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            frozen_inputs={p for folder in ['inputs','policy_snapshot','audit/service_evidence'] for p in (run/folder).rglob('*') if p.is_file()}
            names=['EXPERIMENT_PLAN.md','execution_contract.json','schedule.json','slot_call_manifest.jsonl','review_policy.json','input_manifest.json','implementation_manifest.json','audit/token_identity.json','audit/token_report.json','audit/token_budget_summary.json','audit/dependency_coverage_report.json','audit/prompt_pairing_report.json','audit/repair_inheritance_report.json','audit/field_visibility_report.md','audit/repair_capacity_test.json','audit/test_report.json','audit/source_manifest_before.json','audit/source_preservation_report.json','audit/dry_run_report.json','audit/writer_configuration_evidence.json']
            frozen_inputs|={run/n for n in names};frozen_inputs.add(Path(tests['log']['path']));frozen_inputs|={p for p in (run/'audit/prompt_diffs').rglob('*') if p.is_file()}
            policy_diff=run/'audit/review_policy_diff.json'
            if policy_diff.exists():frozen_inputs.add(policy_diff)
            atomic_json(run/'audit/frozen_manifest.json',{'version':VERSION,'code':[record(p) for p in sorted(code_paths)],'inputs':[record(p) for p in sorted(frozen_inputs)],'sources':[{k:x[k] for k in ('path','sha256','size')} for x in read(run/'audit/source_manifest_before.json')['files']],'config':record(CONFIG),'preflight':record(run/'audit/preflight_report.json'),'environment':environment_identity(),'schedule':SCHEDULE})
        return result
