"""Prepare exact requests, local capacity checks and frozen dependencies; no service calls."""
from __future__ import annotations
import copy,difflib,hashlib,importlib.util,json,re,shutil,subprocess,sys,datetime
from dataclasses import replace
from pathlib import Path
from ..core import atomic_json,atomic_text,canonical_json,sha256_file
from ..h4_qa_mining_probe_v1.intent_contract import canonicalize_intents
from ..h4_qa_mining_probe_v1.reasoning_contract import build_reasoning_contract,contract_has_three_disjoint_groups
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt,build_repair_prompt
from ..h4_qa_mining_hybrid_generation_v1.candidate_contract import normalize_batch
from ..h4_r02_feedback_transfer_p2_2_v1.token_preflight import prepare_tokenizer,TokenPreflight,measure_request
from ..live_backends import load_challenger_config,RATE_LIMIT_RETRY_LIMIT
from .constants import *
from .dependencies import record,forbid_network,collect_coverage,environment_identity,verify_frozen

def read(p):return json.loads(Path(p).read_text())
def digest(v):return hashlib.sha256(canonical_json(v).encode()).hexdigest()
def register(run,p):
 p=Path(p).resolve();mp=run/'audit/source_manifest_before.json';m=read(mp) if mp.exists() else {'files':[]}
 actual=record(p);old=next((r for r in m['files'] if r['path']==str(p)),None)
 if old:
  if any(old[k]!=actual[k] for k in ('size','sha256')):raise RuntimeError('SOURCE_DRIFT:'+str(p))
 else:
  m['files'].append({**actual,'registered_at':datetime.datetime.now(datetime.timezone.utc).isoformat()});atomic_json(mp,m)
 return actual

def copy_exact(run,source,dest):
 original=register(run,source);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  if dest.read_bytes()!=Path(source).read_bytes():raise RuntimeError('EXISTING_COPY_CONFLICT:'+str(dest))
 else:shutil.copyfile(source,dest)
 return {'source':original,'copy':record(dest)}

def source_inputs(run):
 copies=[];slots=[];source_frozen=read(SOURCE_RUN/'audit/frozen_manifest.json')
 review_source=SOURCE_RUN/'reports/intent_evidence_review.jsonl';review_identity=register(run,review_source)
 review_lines=[(n,line,json.loads(line)) for n,line in enumerate(review_source.read_text().splitlines(),1) if line.strip()]
 prior={r['path']:r for r in source_frozen['inputs']}
 for root in ROOTS:
  dest=run/'inputs/shared'/root;src=SOURCE_RUN/'inputs/shared'/root
  for name in ('expanded_dialogue_clean.json','history_only.json','provenance.json'):
   source=src/name;identity=register(run,source)
   if str(source) not in prior or any(prior[str(source)][k]!=identity[k] for k in ('sha256','size')):raise RuntimeError('SOURCE_INPUT_NOT_FROZEN:'+name)
   copies.append(copy_exact(run,source,dest/name))
  B=SOURCE_RUN/'arms/V3_OFF/qa/shards/shard-00/roots'/root
  for name in (f'raw/responses/{root}-intent-miner.json.txt','intents.json','reasoning_contracts.json','derived_intent_evidence.json','response_schema_audit.json'):
   copies.append(copy_exact(run,B/name,dest/'original_v3'/name))
  copies.append(copy_exact(run,B/'reasoning_contracts.json',dest/'contracts.json'))
  copies.append(copy_exact(run,B/'reasoning_contracts.json',dest/'reasoning_contracts.json'))
  diagnostic_rows=[(n,line,row) for n,line,row in review_lines if row.get('root_id')==root and row.get('arm_id')=='V3_OFF']
  if len(diagnostic_rows)!=3 or {r['intent_id'] for _,_,r in diagnostic_rows}!={'intent_1','intent_2','intent_3'}:raise RuntimeError('SOURCE_DIAGNOSTIC_SLOT_MISMATCH')
  diagnostic_path=dest/'source_intent_diagnostics.json'
  atomic_json(diagnostic_path,dict(source=review_identity,visibility='OFFLINE_POST_QUALITY_ONLY_NEVER_WRITER_INPUT',records=[r for _,_,r in diagnostic_rows],original_lines=[dict(line_number=n,text=line) for n,line,_ in diagnostic_rows]))
  copies.append(dict(source=review_identity,copy=record(diagnostic_path),transformation='Exact parsed V3_OFF rows plus original line text and number for this root; no content edits; offline diagnostic only'))
  exp=read(dest/'expanded_dialogue_clean.json');raw=read(B/'raw/responses'/f'{root}-intent-miner.json.txt');parsed=read(B/'intents.json')['validation'];contracts=read(dest/'contracts.json');derived=read(B/'derived_intent_evidence.json');sa=read(B/'response_schema_audit.json')
  import jsonschema
  sp=SOURCE_RUN/'policy_snapshot/schema.json';register(run,sp);jsonschema.Draft202012Validator(read(sp)).validate(raw)
  ca=canonicalize_intents(raw,expanded_dialogue=exp)
  if ca!=parsed or not ca['valid'] or len(contracts)!=3:raise RuntimeError('RAW_CANONICAL_IDENTITY_MISMATCH')
  if digest(raw)!=sa['raw_response_sha256'] or not sa['valid']:raise RuntimeError('SCHEMA_AUDIT_MISMATCH')
  for i,(row,c) in enumerate(zip(ca['intents'],contracts)):
   expected=build_reasoning_contract(row);expected['contract_valid']=contract_has_three_disjoint_groups(expected)
   if expected!=c or c['intent_id']!=f'intent_{i+1}' or c['required_evidence_groups']!=derived['items'][i]['derived_evidence_groups']:raise RuntimeError('CONTRACT_SLOT_IDENTITY_MISMATCH')
   if derived['items'][i]['transcript_sha256']!=sha256_file(dest/'expanded_dialogue_clean.json'):raise RuntimeError('DERIVATION_TRANSCRIPT_IDENTITY')
   slots.append(dict(root_id=root,raw_slot=i+1,raw_index=i,intent_id=c['intent_id'],source_raw=record(B/'raw/responses'/f'{root}-intent-miner.json.txt'),source_contracts=record(B/'reasoning_contracts.json'),contract_sha256=digest(c),original_structurally_valid=True,selected_by_content=False))
 atomic_json(run/'input_manifest.json',dict(status='PASS',copies=copies,slots=slots,source_run=str(SOURCE_RUN),transcript_byte_identity=True,all_six_original_slots=True,history_visibility='OFFLINE_ONLY_NOT_WRITER_INPUT',provenance_visibility='OFFLINE_ONLY_NOT_WRITER_INPUT',constructed_audit_examples_used=False))
 return slots

def historical_writer_config(run):
 p=WRITER_REFERENCE/'execution_contract.json';register(run,p);prior=read(p);g=prior['generation']
 config=replace(load_challenger_config(CONFIG),temperature=.1,enable_thinking=False,timeout_seconds=180)
 if sha256_file(CONFIG)!=prior['configuration']['sha256']:raise RuntimeError('CONFIG_IDENTITY_CHANGED')
 if config.model!=GENERATION['model'] or config.base_url!=GENERATION['base_url'] or RATE_LIMIT_RETRY_LIMIT!=2:raise RuntimeError('WRITER_CONFIG_DRIFT')
 if not (g['writer_max_output_tokens']==8192 and g['temperature']==.1 and g['timeout_seconds']==180 and g['enable_thinking'] is False and g['seed']=='not_sent_by_original_client'):raise RuntimeError('WRITER_REFERENCE_MISMATCH')
 rows=[]
 for p in sorted(WRITER_REFERENCE.glob('qa/shards/*/roots/*/ledger/requests/*.json')):
  if '-contract_guided-generator-' not in p.name and '-contract_guided-repair-' not in p.name:continue
  identity=register(run,p);d=read(p);md=d['request_metadata'];be=d['backend_identity']
  if md['max_tokens']!=8192 or md['temperature']!=.1 or be['model']!=config.model or be['base_url']!=config.base_url:raise RuntimeError('HISTORICAL_WRITER_REQUEST_DRIFT')
  rows.append(dict(source=identity,request_id=d.get('request_id',p.stem),stage='repair' if '-repair-' in p.name else 'primary',max_output_tokens=md['max_tokens'],temperature=md['temperature'],model=be['model'],status=d['status']))
 if not any(r['stage']=='repair' for r in rows):raise RuntimeError('NO_ACTUAL_WRITER_REPAIR_REFERENCE')
 atomic_json(run/'audit/writer_configuration_evidence.json',dict(status='PASS',execution_contract=record(WRITER_REFERENCE/'execution_contract.json'),actual_requests=rows,effective_config=GENERATION,config_identity=record(CONFIG),credentials_printed=False,source_note='Writer parameters from Writer acceptance reference and shared client; Miner-only 4096 is not reused.'))
 return config

def exact_original_builder(run):
 path=run/'policy_snapshot/before_code/scripts/current_method/restart_n5_spell/h4_qa_mining_hybrid_generation_v1/contract_guided_generator.py'
 if not path.is_file():raise RuntimeError('PRECHANGE_BUILDER_SNAPSHOT_REQUIRED')
 name=build_prompt.__module__.rsplit('.',1)[0]+'._prechange_ablation_builder'
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def prepare_requests(run,slots):
 original=exact_original_builder(run);man=[];comparisons=[];repair_checks=[];sourceindex={(s['root_id'],s['raw_slot']):s for s in slots}
 synthetic={'candidates':[{'question':'OFFLINE_SYNTHETIC_REPAIR_FIXTURE','discriminative_variable':'CANDIDATE_OWN_FOCUS'}]}
 for item in SCHEDULE:
  arm,root,i=item['arm_id'],item['root_id'],item['raw_index'];folder=run/'inputs/shared'/root
  expanded=read(folder/'expanded_dialogue_clean.json');contract=read(folder/'contracts.json')[i];on=arm==ARMS[0]
  a=build_prompt(expanded,contract);b=build_prompt(expanded,contract,intent_guidance=False);spec=a if on else b
  baseline=run/'policy_snapshot/baseline_guided_prompts'/root/(item['slot_id']+'.txt')
  if not baseline.is_file() or baseline.read_text()!=original.build_prompt(expanded,contract)['prompt'] or a['prompt']!=baseline.read_text():raise RuntimeError('A_PRIMARY_BASELINE_CHANGED')
  altered=copy.deepcopy(contract);altered.update(intent_type='DEPENDENCY_SENTINEL_TYPE',discriminative_variable='DEPENDENCY_SENTINEL_VARIABLE',required_evidence_groups={'sentinel':['DEPENDENCY_SENTINEL_REF']})
  b_invariant=b['prompt']==build_prompt(expanded,altered,intent_guidance=False)['prompt'];a_changes=a['prompt']!=build_prompt(expanded,altered)['prompt']
  repair_rows=[]
  for fixture in (None,synthetic):
   ra=build_repair_prompt(expanded,contract,fixture,'contract_guided');rb=build_repair_prompt(expanded,contract,fixture,'contract_guided',intent_guidance=False)
   expected=original.build_repair_prompt(expanded,contract,fixture,'contract_guided')
   okay=ra==expected and rb==build_repair_prompt(expanded,altered,fixture,'contract_guided',intent_guidance=False)
   if not okay:raise RuntimeError('REPAIR_GUIDANCE_DEPENDENCY_FAILED')
   repair_rows.append(dict(fixture='raw_none' if fixture is None else 'SYNTHETIC_TEST_ONLY',A_original_byte_equal=ra['prompt']==expected['prompt'],B_source_intent_invariant=True,repair_suffix_equal=True))
  if not b_invariant or not a_changes:raise RuntimeError('PROMPT_DEPENDENCY_TEST_FAILED')
  request=dict(**item,stage='primary',generation_source='contract_guided',intent_guidance=on,prompt_sha256=spec['prompt_sha256'],max_output_tokens=8192,model_config=GENERATION,expected_counts=spec['expected_counts'],source_slot=sourceindex[root,item['raw_slot']],transcript=record(folder/'expanded_dialogue_clean.json'),builder=record(Path(sys.modules[build_prompt.__module__].__file__)),payload={'model':GENERATION['model'],'messages':[{'role':'user','content':spec['prompt']}],'temperature':.1,'max_tokens':8192,'enable_thinking':False,'response_format':{'type':'json_object'}})
  dest=run/'policy_snapshot/primary_requests'/arm/root/item['slot_id'];atomic_text(dest/'prompt.txt',spec['prompt']);atomic_json(dest/'request.json',request)
  man.append({**item,'stage':'primary','request_spec':record(dest/'request.json'),'prompt':record(dest/'prompt.txt'),'max_output_tokens':8192,'maximum_primary_calls':1,'maximum_repair_calls':1,'repair_request_id':f'{root}-contract_guided-repair-{item["intent_id"]}'})
  if on:
   diff=''.join(difflib.unified_diff(a['prompt'].splitlines(keepends=True),b['prompt'].splitlines(keepends=True),fromfile='WITH_CURRENT_INTENT',tofile='WITHOUT_INTENT'))
   atomic_text(run/'audit/prompt_diffs'/root/(item['slot_id']+'.diff'),diff)
   comparisons.append(dict(root_id=root,slot_id=item['slot_id'],A_original_byte_equal=True,B_intent_mutation_invariant=b_invariant,A_intent_mutation_changes=a_changes,same_dialogue=a['prompt'].split('DIALOGUE:\n',1)[1]==b['prompt'].split('DIALOGUE:\n',1)[1],diff_path=str(run/'audit/prompt_diffs'/root/(item['slot_id']+'.diff')),necessary_differences=['for this intent -> for this batch','contract soft guidance sentence -> one neutral focus sentence','remove CONTRACT GUIDANCE header and summary','OUTPUT SHAPE discriminative_variable -> ordinary ... placeholder'],opaque_ID_only=True))
   repair_checks.append(dict(root_id=root,slot_id=item['slot_id'],tests=repair_rows))
 atomic_text(run/'slot_call_manifest.jsonl',''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in man))
 atomic_json(run/'audit/prompt_diff_report.json',dict(status='PASS',slots=comparisons,method='Exact byte comparison and semantic-source perturbation; no person-name substring leak heuristic.'))
 atomic_json(run/'audit/repair_inheritance_report.json',dict(status='PASS',slots=repair_checks,real_repair_prompts_frozen=0,synthetic_fixtures_are_not_QA=True,dynamic_send_check_required=True))
 atomic_text(run/'audit/field_visibility_report.md','''| 来源/位置 | WITH_CURRENT_INTENT | WITHOUT_INTENT |
|---|---|---|
| 全文对白 | 相同完整冻结对白 | 相同完整冻结对白 |
| opaque intent_id | 原槽 ID | 仅调用槽身份；不回查语义 |
| intent_type / discriminative_variable / required_evidence_groups | 原 Guided summary | 不可见 |
| for this intent / contract guidance 引用 | 原样 | batch 中性引用与一条自主选材说明 |
| OUTPUT SHAPE 的 variable | 原变量 | 普通 ... 占位符；由候选自己产生 |
| repair base / suffix | 本臂原模板＋原解析对象 | 本臂无指导模板＋本臂原解析对象；不回加原 intent |
| normalization contract | 原 contract | opaque ID＋空 groups；无变量/类型/refs回填 |
| 候选 variable/evidence_refs/evidence_groups | 原结构准入要求 | 相同要求；候选自己产生 |
| contract_group_alignment | 原 metadata | 保留自然 false，离线分析 NOT_APPLICABLE |
| history/provenance/cards/成绩/bindings/离线构造示例 | 不输入 | 不输入 |

每个槽已检查：固定 transcript 与 opaque ID、任意替换源类型/变量/refs，B primary/repair 字节不变，A 随源变化。正常化不变性与缺失候选字段拒绝由 test_shared_stage 检查。候选自身 metadata 与全局统一 OUTPUT SHAPE 示例仍在，二者不是来自某个 Miner 的语义回填。两臂 generation_source 均 contract_guided，无 Free 调度；B 禁 suffix/hooks 旁路。共享执行阶段及最终 validator 完全相同。
''')
 return man

def token_assets(run,config,manifest):
 for folder in ('policy_snapshot/tokenizer','audit/service_evidence'):
  for p in sorted((SOURCE_RUN/folder).rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts:copy_exact(run,p,run/folder/p.relative_to(SOURCE_RUN/folder))
 historical=read(run/'audit/service_evidence/historical_token_verification.json')
 for row in historical['rows']:
  register(run,Path(row['ledger_path']));register(run,Path(row['prompt_path']))
 identity=prepare_tokenizer(run,config);token=TokenPreflight(run)
 counts=[token.check(Path(row['prompt']['path']).read_text(),8192,row['global_request_id']) for row in manifest]
 # Dynamic repair mechanism exercised with real tokenizer and a deliberately oversized synthetic object.
 from .runtime import GuardrailStop
 example=SCHEDULE[0];root=example['root_id'];exp=read(run/'inputs/shared'/root/'expanded_dialogue_clean.json');c=read(run/'inputs/shared'/root/'contracts.json')[0]
 fixture=build_repair_prompt(exp,c,{'candidates':[]},'contract_guided')['prompt']
 small=measure_request(token.tokenizer,prompt=fixture,max_output_tokens=8192,request_id='SYNTHETIC_REPAIR_CAPACITY',context_limit=identity['context_limit'],supported_max_output=identity['verified_supported_max_output_tokens'])
 giant=build_repair_prompt(exp,c,{'candidates':[{'question':' capacity'*identity['context_limit']}]},'contract_guided',intent_guidance=False)['prompt']
 large=measure_request(token.tokenizer,prompt=giant,max_output_tokens=8192,request_id='SYNTHETIC_OVERSIZED_REPAIR',context_limit=identity['context_limit'],supported_max_output=identity['verified_supported_max_output_tokens'])
 if small['status']!='PASS' or large['status']!='BLOCKED':raise RuntimeError('REPAIR_TOKEN_CAPACITY_MECHANISM_FAILED')
 atomic_json(run/'audit/repair_capacity_test.json',dict(status='PASS',synthetic_only=True,actual_repair_prompts=0,ordinary_fixture=small,oversized_fixture=large,policy='Actual future repair prompt checked before ledger reservation and every HTTP send; no truncation or token-limit change.'))
 provenance=read(run/'audit/service_evidence/evidence_manifest.json')
 atomic_json(run/'audit/token_budget_summary.json',dict(primary_count=12,primary_input_tokens=sum(r['rendered_message_tokens'] for r in counts),by_arm={a:sum(r['rendered_message_tokens'] for r,m in zip(counts,manifest) if m['arm_id']==a) for a in ARMS},primary_output_token_upper_bound=12*8192,conditional_repair_count_upper_bound=12,conditional_repair_output_token_upper_bound=12*8192,conditional_repair_input_tokens='UNKNOWN_UNTIL_PRIMARY_RESPONSES',actual_usage=None,actual_cost=None,money='UNKNOWN_NO_VERIFIED_BILLING',actual_calls=0,per_request_output_budget=8192,per_request_reserve=512,service_evidence_manifest=provenance,service_current_state_queried=False,hosted_model_revision=None,limits_note='Local immutable tokenizer and previously collected service evidence only; hosted current revision/limit not re-queried.'))
 return identity,counts

def test_phase(run):
 script='''import unittest,sys
sys.dont_write_bytecode=True
from scripts.current_method.restart_n5_spell.h4_r02_writer_intent_ablation_v1.dependencies import forbid_network
loader=unittest.TestLoader();suite=unittest.TestSuite()
suite.addTests(loader.discover(%r,pattern='test_*.py',top_level_dir=%r))
for name in %r:suite.addTests(loader.loadTestsFromName(name))
with forbid_network(): result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
'''%(str(CODE),str(REPO),['scripts.current_method.restart_n5_spell.h4_r02_feedback_transfer_p2_2_v1.test_generation_hooks','scripts.current_method.restart_n5_spell.h4_r02_miner_binding_v2.test_shared_miner'])
 process=subprocess.run([sys.executable,'-B','-c',script],cwd=REPO,capture_output=True,text=True)
 output=process.stdout+process.stderr;n=len(list((run/'audit').glob('test_output_attempt_*.txt')))+1;log=run/f'audit/test_output_attempt_{n}.txt';atomic_text(log,output)
 count=re.search(r'Ran (\d+) tests?',output)
 report=dict(status='PASS' if process.returncode==0 else 'FAIL',tests_run=int(count.group(1)) if count else None,exit_code=process.returncode,command=[sys.executable,'-B','-c',script],log=record(log),network_forbidden=True,external_calls=0)
 atomic_json(run/'audit/test_report.json',report)
 return report

def preservation(run):
 rows=[]
 for r in read(run/'audit/source_manifest_before.json')['files']:
  p=Path(r['path']);current=record(p);rows.append(dict(before=r,after=current,unchanged=all(current[k]==r[k] for k in ('size','sha256'))))
 result=dict(status='PASS' if all(r['unchanged'] for r in rows) else 'FAIL',file_count=len(rows),changed_count=sum(not r['unchanged'] for r in rows),scope='Only registered inputs, audit reference documents, tokenizer/service evidence and referenced historical operational prompts/ledgers. Not the whole historical tree.',whole_history_rescanned=False,rows=rows)
 atomic_json(run/'audit/source_preservation_report.json',result);return result

def implementation(run):
 changes=[]
 for rel in ['h4_qa_multiroot_k3_t08_probe_v1/generation.py','h4_qa_mining_hybrid_generation_v1/contract_guided_generator.py']:
  p=METHOD/rel;before=run/'policy_snapshot/before_code'/p.relative_to(REPO)
  diff=''.join(difflib.unified_diff(before.read_text().splitlines(keepends=True),p.read_text().splitlines(keepends=True),fromfile='before/'+rel,tofile='after/'+rel))
  target=run/'audit/implementation_diffs'/(p.stem+'.diff');atomic_text(target,diff)
  changes.append(dict(path=str(p),before=record(before),after=record(p),diff=record(target),purpose='Extract shared execution and finalization without changing default admission' if p.stem=='generation' else 'Default-on guidance switch; remove semantic content from B primary and repair'))
 atomic_json(run/'implementation_manifest.json',dict(version=VERSION,new_code=str(CODE),shared_changes=changes,unrelated_uncommitted_changes_preserved=True,shared_batch='h4_qa_multiroot_k3_t08_probe_v1.generation.execute_writer_batch',shared_finalizer='h4_qa_multiroot_k3_t08_probe_v1.generation.finalize_writer_candidates',default_generate_root_qa='Original Miner then Guided+Free through shared batch',candidate_schema_modified=False,acceptance_thresholds_modified=False,semantic_production_filter=False,compatibility_evidence='audit/test_report.json and six A baseline comparisons'))

def prepare(run=DEFAULT_RUN):
 run=Path(run).resolve()
 if run!=DEFAULT_RUN.resolve():raise RuntimeError('FIXED_EXPERIMENT_RUN_REQUIRED')
 if (run/'audit/frozen_manifest.json').exists():verify_frozen(run,CONFIG);return read(run/'audit/preflight_report.json')
 if list(run.glob('arms/**/ledger/requests/*.json')):raise RuntimeError('UNFROZEN_LIVE_STATE_PRESENT')
 with forbid_network():
  slots=source_inputs(run);config=historical_writer_config(run)
  atomic_json(run/'schedule.json',SCHEDULE)
  atomic_json(run/'execution_contract.json',dict(version=VERSION,roots=list(ROOTS),arms=list(ARMS),generation=GENERATION,configuration=record(CONFIG),schedule=SCHEDULE,planned_primary_calls=12,maximum_repair_calls=12,maximum_logical_calls=24,per_arm={'primary':6,'repair':6,'total':12},per_root_arm={'primary':3,'repair':3,'total':6},per_cell={'primary':1,'repair':1},target_candidates_per_arm=24,target_SC_MS_per_arm=18,target_ordering_per_arm=6,context_reserve_tokens=512,preparation_external_call_budget=0,live_authorized_by_this_task=False,zero_call_roles=['Miner','Free','Solver','Planner','Realizer','Memory','training','independent_review_API']))
  atomic_json(run/'policy_snapshot/repair_policy.json',dict(trigger='Any structurally valid question-type count below original 2/1/1 minima',max_repairs_per_cell=1,raw_input='Primary parsed JSON object; None on parse failure passed to original builder as raw or {}',replacement='WHOLE_BATCH_REPLACEMENT_NO_MERGE_NO_FALLBACK',extra_valid_rows='Minima satisfied means no repair despite normalizer exact-count errors',final_validator_rejection='No additional repair',per_arm_template_inheritance=True,frozen_real_repair_prompts=0,dynamic_capacity_check='Exact final repair prompt +8192+512 before request; no truncation',unknown_completion='Sticky halt before repair/next request',semantic_retry=False))
  from .reporting import prepare_review_assets
  prepare_review_assets(run)
  manifest=prepare_requests(run,slots);identity,tokens=token_assets(run,config,manifest)
  tests=test_phase(run)
  if tests['status']!='PASS':
   atomic_json(run/'audit/preflight_report.json',{'status':'BLOCKED','reason':'OFFLINE_TEST_FAILURE','test_report':tests,'external_calls':0});return read(run/'audit/preflight_report.json')
  from .dry_run import exercise
  dry=exercise(run);coverage=collect_coverage(run,dry)
  implementation(run);preserved=preservation(run)
  checks=dict(six_original_slots=len(slots)==6,primary_requests=len(manifest)==12,A_byte_identity=True,B_dependency_invariance=True,offline_tests=tests['status']=='PASS',dry_run=dry['status']=='PASS',dependency_coverage=coverage['status']=='PASS',primary_capacity=len(tokens)==12 and all(t['status']=='PASS' for t in tokens),source_preservation=preserved['status']=='PASS',external_calls_zero=not list(run.glob('arms/**/ledger/requests/*.json')))
  result=dict(version=VERSION,status='PASS' if all(checks.values()) else 'BLOCKED',preparation_status='READY_FOR_AUTHORIZATION' if all(checks.values()) else 'BLOCKED',live_status='NOT_RUN',run_status='PREPARED_NOT_RUN',actual_external_calls=0,checks=checks)
  atomic_json(run/'audit/preflight_report.json',result)
  if result['status']=='PASS':
   code_paths={Path(x['path']) for x in coverage['repository_files']}|{p for p in CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
   for p in sorted(code_paths):copy=run/'policy_snapshot/frozen_code'/p.relative_to(REPO);copy.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,copy)
   frozen_inputs=set(p for folder in ['inputs','policy_snapshot','audit/service_evidence'] for p in (run/folder).rglob('*') if p.is_file())
   frozen_inputs|={run/p for p in ['EXPERIMENT_PLAN.md','execution_contract.json','schedule.json','slot_call_manifest.jsonl','review_policy.json','input_manifest.json','implementation_manifest.json','audit/token_identity.json','audit/token_report.json','audit/token_budget_summary.json','audit/dependency_coverage_report.json','audit/prompt_diff_report.json','audit/repair_inheritance_report.json','audit/field_visibility_report.md','audit/repair_capacity_test.json','audit/test_report.json','audit/source_manifest_before.json','audit/source_preservation_report.json','audit/dry_run_report.json','audit/writer_configuration_evidence.json']}
   frozen_inputs|={p for folder in ['audit/implementation_diffs','audit/prompt_diffs'] for p in (run/folder).rglob('*') if p.is_file()}
   frozen_inputs.add(Path(tests['log']['path']))
   atomic_json(run/'audit/frozen_manifest.json',dict(version=VERSION,code=[record(p) for p in sorted(code_paths)],inputs=[record(p) for p in sorted(frozen_inputs)],sources=[{k:r[k] for k in ('path','sha256','size')} for r in read(run/'audit/source_manifest_before.json')['files']],config=record(CONFIG),preflight=record(run/'audit/preflight_report.json'),environment=environment_identity(),schedule=SCHEDULE))
  return result
