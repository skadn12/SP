"""CPU-only independent final audit; execute only after controller exits."""
import collections,datetime,hashlib,importlib.util,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
REPO=Path('/root/autodl-tmp/gxy-selfplay')
SCOPE=REPO/'offline-deliveries/h4_codex_dev_eval_v127'
OUT=SCOPE/'evidence/INDEPENDENT_EXECUTION_AUDIT.json'
OBS=Path('/tmp/v127_execution_process_observations.jsonl')
errors=[];bindings={};checks=[]
def require(condition,code,**details):
 if not condition:errors.append({'code':code,**details})
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def bind(path):
 p=Path(path).resolve();item={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size};bindings[str(p)]=item;return item
def read(path):
 bind(path);return json.loads(Path(path).read_text())
def verify(rows,source):
 for row in rows:
  path=Path(row['path']);require(path.is_file(),'BOUND_FILE_MISSING',source=source,path=str(path))
  if path.is_file():
   item=bind(path);require(item['sha256']==row['sha256'],'BOUND_SHA_CHANGED',source=source,path=str(path))
   expected=row.get('bytes',row.get('size_bytes',row.get('size')))
   if expected is not None:require(item['bytes']==expected,'BOUND_BYTES_CHANGED',source=source,path=str(path))
if OUT.exists():raise SystemExit('REFUSE_OVERWRITE:'+str(OUT))
status=read(SCOPE/'RUN_STATUS.json')
if status.get('status')!='EVALUATION_REPORTED_AWAITING_INDEPENDENT_AUDIT':raise SystemExit('CONTROLLER_NOT_READY:'+str(status.get('status')))
require(status.get('active_process_exit_code')==0,'CONTROLLER_LAST_CHILD_NOT_ZERO')
require(status.get('active_side') is None and status.get('pid') is None,'CONTROLLER_STILL_ACTIVE')
require(status.get('actual_calls')==1560,'CONTROLLER_ACTUAL_CALLS')
require(status.get('automatic_retry') is False and status.get('test_evaluation') is False and status.get('evaluation_only') is True,'CONTROLLER_SCOPE_OR_RETRY_CHANGED')
require(status.get('baseline_complete') is True and status.get('new_complete') is True,'CONTROLLER_SIDE_COMPLETE_FLAGS')
preflight=read(SCOPE/'PREFLIGHT_FREEZE.json');verify(preflight['files'],'PREFLIGHT_FREEZE')
tools=read(SCOPE/'TOOLS_FREEZE.json');verify(tools['files'],'TOOLS_FREEZE')
pre_audit=read(SCOPE/'evidence/INDEPENDENT_PREFLIGHT_AUDIT.json')
require(pre_audit.get('status')=='PASS','PREFLIGHT_AUDIT_NOT_PASS')
verify(pre_audit['audited_files'],'INDEPENDENT_PREFLIGHT_AUDIT')
require(str(SCOPE/'run_pair.py') in {r['path'] for r in tools['files']},'CONTROLLER_NOT_FROZEN')
plan=read(REPO/'offline-deliveries/h4_codex_training_v126/evidence/DEV_EVALUATION_PLAN_AUDIT.json')
requests=plan['readonly_checks']['plan_requests'];by_id={r['request_id']:r for r in requests}
require(len(requests)==len(by_id)==780,'AUTHORITATIVE_REQUEST_POPULATION')
summary=read(SCOPE/'PAIRED_SUMMARY.json')
require(summary.get('status')=='COMPLETE' and not summary.get('errors'),'PAIRED_REPORT_NOT_COMPLETE')
require(summary.get('planned_calls_per_model')==780 and summary.get('planned_calls_total')==1560,'PAIRED_REPORT_DENOMINATOR')
observations=[json.loads(line) for line in OBS.read_text().splitlines() if line.strip()]
# Durable observation evidence is copied only after controller completion.
observation_evidence=SCOPE/'evidence/EXECUTION_PROCESS_OBSERVATIONS.json'
if observation_evidence.exists():raise SystemExit('REFUSE_OVERWRITE:'+str(observation_evidence))
with observation_evidence.open('x') as f:json.dump({'schema_version':'v127_direct_process_observations_v1','observations':observations},f,ensure_ascii=False,indent=2);f.write('\n')
bind(observation_evidence)
expected_py=str(REPO/'.venv-local4b/bin/python')
for side in ['baseline','new']:
 rows=[r for r in observations if r.get('active_side')==side]
 require(bool(rows),'NO_DIRECT_LIVE_PROCESS_OBSERVATION',side=side)
 require(all(r['cmdline']==[expected_py,str(SCOPE/'evaluate.py'),'live','--side',side] for r in rows),'OBSERVED_LIVE_ARGV_WRONG',side=side)
 require(all(r.get('cwd')==str(SCOPE) for r in rows),'OBSERVED_LIVE_CWD_WRONG',side=side)
for row in observations:
 proc=Path('/proc')/str(row['pid'])
 if proc.exists():
  try:same=(proc/'stat').read_text().split()[21]==row['proc_starttime_ticks']
  except FileNotFoundError:same=False
  require(not same,'OBSERVED_PROCESS_STILL_ALIVE',pid=row['pid'],side=row.get('active_side'))
# Log existence and the final controller record provide exit evidence without launching anything.
controller_log=SCOPE/'controller.log';text=controller_log.read_text();bind(controller_log)
require('EVALUATION_REPORTED_AWAITING_INDEPENDENT_AUDIT' in text,'CONTROLLER_LOG_MISSING_FINAL_STATUS')
verify_log=SCOPE/'verify.log';text=verify_log.read_text();bind(verify_log)
require('PASS: frozen paired evaluation inputs, tools and independent preflight' in text,'VERIFY_LOG_NOT_PASS')
report_log=SCOPE/'report.log';text=report_log.read_text();bind(report_log)
require('"status": "COMPLETE"' in text and '"error_count": 0' in text,'REPORT_LOG_NOT_COMPLETE')
identity_registration=read(SCOPE/'model_tools/NEW_MODEL_IDENTITY.json')
manifests={};sides={}
for side in ['baseline','new']:
 run=SCOPE/'run'/side;manifest=read(run/'manifest.json');manifests[side]=manifest
 identity=read(run/'solver_identity.json');require(identity==manifest['solver_identity'],'IDENTITY_MANIFEST_MISMATCH',side=side)
 require(identity['model_path']==manifest['model'] and identity['model_identifier']==manifest['solver_model'] and identity['tokenizer_path']==manifest['tokenizer'],'IDENTITY_ROUTE_MISMATCH',side=side)
 require(identity.get('adapter_enabled') is True and identity.get('adapter_already_merged') is True,'UNMERGED_MODEL',side=side)
 if side=='new':require(identity==identity_registration,'NEW_REGISTRATION_CHANGED')
 verify(identity['checkpoint_files'],side+'_checkpoint_files')
 for k in ['audit','tokenizer_audit']:
  if identity.get(k+'_path'):
   item=bind(identity[k+'_path']);require(item['sha256']==identity[k+'_sha256'],'MODEL_AUDIT_CHANGED',side=side,kind=k)
 require(manifest['requests']==requests and manifest['request_count']==780 and manifest['candidate_count']==48,'REQUEST_MANIFEST_CHANGED',side=side)
 require(manifest['policy_version']=='v15_full_only_random_mcq6x3_order12x1_v1','POLICY_CHANGED',side=side)
 require(manifest.get('execution')=='full_only_no_ground_v1' and manifest.get('training_started') is False and manifest.get('new_generation_calls')==0,'EVALUATION_SCOPE_CHANGED',side=side)
 started=read(run/'STARTED.json');complete=read(run/'COMPLETE.json');native=read(run/'RESULTS.json')
 require(started.get('manifest_sha256')==sha(run/'manifest.json'),'STARTED_MANIFEST_CHANGED',side=side)
 require(complete.get('status')=='COMPLETE' and complete.get('actual_call_count')==780,'SIDE_NOT_COMPLETE',side=side)
 require(not (run/'FAILED.json').exists(),'FAILED_SENTINEL_PRESENT',side=side)
 require(not native.get('call_set_errors') and native.get('ground_calls')==0,'NATIVE_CALL_SET_OR_SCOPE_ERROR',side=side)
 paths=sorted((run/'responses').glob('*.json'));records=[read(path) for path in paths]
 counts=collections.Counter(r.get('request_id') for r in records)
 require(len(paths)==780 and set(counts)==set(by_id) and all(v==1 for v in counts.values()),'MISSING_DUPLICATE_OR_UNPLANNED_RESPONSE',side=side)
 finish=collections.Counter();completion_counts=[]
 for path,row in zip(paths,records):
  rid=row.get('request_id');req=by_id.get(rid)
  if req is None:continue
  require(path.name==rid+'.json','RESPONSE_FILENAME_MISMATCH',side=side,request_id=rid)
  expected={'request_id':rid,'candidate_id':req['candidate_id'],'root_id':req['root_id'],'question_type':req['question_type'],'variant_id':req['variant']['variant_id'],'slot':req['slot'],'seed':req['seed'],'mode':'full_context','solver_model':manifest['solver_model'],'model_path':manifest['model']}
  for key,value in expected.items():require(row.get(key)==value,'RESPONSE_PLAN_FIELD_MISMATCH',side=side,request_id=rid,field=key)
  metadata=row.get('metadata') or {}
  for key in ['temperature','top_p','max_tokens','structured_output_api','structured_output_version']:
   require(metadata.get(key)==manifest['decoder_contract'].get(key),'RESPONSE_DECODER_MISMATCH',side=side,request_id=rid,field=key)
  require(metadata.get('sampling_seed')==req['seed'],'RESPONSE_SAMPLING_SEED_MISMATCH',side=side,request_id=rid)
  require(metadata.get('prompt_token_count')==req['prompt_tokens'],'RESPONSE_PROMPT_TOKENS_MISMATCH',side=side,request_id=rid)
  finish[metadata.get('finish_reason')]+=1
  require(metadata.get('finish_reason') in ['stop','eos'],'BAD_FINISH_REASON',side=side,request_id=rid)
  n=metadata.get('completion_token_count');require(type(n) is int and 1<=n<=32,'COMPLETION_TOKENS_INVALID',side=side,request_id=rid)
  if type(n) is int:completion_counts.append(n)
  require(row.get('evaluation_valid') is True and not row.get('data_errors'),'INVALID_RESPONSE',side=side,request_id=rid)
  prompt=run/'prompts'/(rid+'.txt');require(sha(prompt)==req['prompt_sha256'],'PROMPT_SHA_CHANGED',side=side,request_id=rid)
 log=SCOPE/(side+'.log');log_text=log.read_text();bind(log)
 engine_lines=[line for line in log_text.splitlines() if 'Initializing a V1 LLM engine (' in line]
 require(len(engine_lines)==1 and 'Initializing a V1 LLM engine (v0.21.0)' in engine_lines[0],'VLLM_021_RUNTIME_LOG_NOT_FOUND',side=side)
 if engine_lines:
  line=engine_lines[0]
  require("model='"+manifest['model']+"'" in line and "tokenizer='"+manifest['tokenizer']+"'" in line and 'max_seq_len=16384' in line and 'enable_prefix_caching=False' in line,'RUNTIME_MODEL_TOKENIZER_OR_CONTEXT_DIFFERS',side=side)
 require(log_text.count('Starting to load model '+manifest['model'])==1,'RUNTIME_MODEL_LOADED_MULTIPLE_TIMES_OR_MISSING',side=side)
 require('Traceback (most recent call last)' not in log_text,'SIDE_LOG_TRACEBACK',side=side)
 call_summary=summary['models'][side]['call_summary']
 require(call_summary['planned_calls']==call_summary['valid_calls']==780 and call_summary['invalid_or_missing_calls']==0,'PAIRED_REPORT_SIDE_INVALID',side=side)
 sides[side]={'status':'PASS' if not any(e.get('side')==side for e in errors) else 'FAIL','solver_model':manifest['solver_model'],'model_path':manifest['model'],'model_fingerprint_sha256':identity.get('model_fingerprint_sha256'),'checkpoint_file_count':len(identity['checkpoint_files']),'planned_calls':780,'actual_unique_responses':len(counts),'duplicate_request_ids':[k for k,v in counts.items() if v!=1],'finish_reasons':dict(finish),'completion_token_min':min(completion_counts) if completion_counts else None,'completion_token_max':max(completion_counts) if completion_counts else None,'observed_live_argv':[r for r in observations if r.get('active_side')==side]}
a,b=manifests['baseline'],manifests['new']
for key in ['requests','decoder_contract','tokenizer','policy_version','random_display','full_slots']:
 require(a[key]==b[key],'PAIRED_MANIFEST_FIELD_DIFFERS',field=key)
require(a['model']!=b['model'] and a['solver_model']!=b['solver_model'],'PAIRED_MODELS_IDENTICAL')
for name in ['questions.json','gold.json','canonical_bindings.json']:
 require(sha(SCOPE/'run/baseline'/name)==sha(SCOPE/'run/new'/name),'PAIRED_CANONICAL_INPUT_DIFFERS',name=name)
# Frozen call implementation is single-pass, retry_index=0; STARTED/write_once refuse reruns.
runner=REPO/'offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation/run_solver.py'
runner_text=runner.read_text();controller_text=(SCOPE/'run_pair.py').read_text()
require('if Path(sys.prefix) != EXPECTED_VENV:' in runner_text and 'retry_index=0' in runner_text and 'ALREADY_STARTED_NO_AUTOMATIC_RERUN' in runner_text and 'for row in manifest["requests"]:' in runner_text,'FROZEN_SINGLE_PASS_CONTRACT_CHANGED')
require('ALREADY_STARTED_NO_AUTOMATIC_RERUN' in controller_text and "automatic_retry': False" in controller_text,'CONTROLLER_NO_RETRY_CONTRACT_CHANGED')
for name in ['PER_QUESTION.json','PER_QUESTION.csv','PER_CALL.json','PER_ROOT.json','REPORT_ZH.md']:
 bind(SCOPE/name)
bind(Path(__file__))
for item in list(bindings.values()):
 require(sha(item['path'])==item['sha256'],'BOUND_FILE_CHANGED_DURING_AUDIT',path=item['path'])
result={'schema_version':'v127_independent_execution_audit_v1','status':'PASS' if not errors else 'FAIL','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer':'/root/repair_v122_engineering','scope':'CPU-only final execution engineering audit. No model/API calls, no GPU, no test question/source reads.','errors':errors,'process_observations':observations,'execution_checks':{'local4b_interpreter_directly_observed_for_both_sides':all(any(r.get('active_side')==s and r['cmdline'][0]==expected_py for r in observations) for s in ['baseline','new']),'native_sys_prefix_guard_frozen':not any(e['code']=='FROZEN_SINGLE_PASS_CONTRACT_CHANGED' for e in errors),'vllm_runtime_logs_verified':not any(e['code']=='VLLM_021_RUNTIME_LOG_NOT_FOUND' for e in errors),'controller_and_observed_children_exited':not any(e['code']=='OBSERVED_PROCESS_STILL_ALIVE' for e in errors),'preflight_and_tools_unchanged':not any(e['code'].startswith('BOUND_') for e in errors),'paired_prompts_seeds_decoder_and_tokenizer_identical':not any(e['code'] in ['PAIRED_MANIFEST_FIELD_DIFFERS','PAIRED_CANONICAL_INPUT_DIFFERS','PROMPT_SHA_CHANGED','RESPONSE_SAMPLING_SEED_MISMATCH','RESPONSE_DECODER_MISMATCH'] for e in errors),'actual_calls_total':sum(v['actual_unique_responses'] for v in sides.values()),'model_checkpoint_files_rehashed_after_inference':not any(e['code'] in ['BOUND_FILE_MISSING','BOUND_SHA_CHANGED','BOUND_BYTES_CHANGED'] and 'checkpoint_files' in e.get('source','') for e in errors),'single_pass_zero_retry_frozen_code_verified':not any(e['code'] in ['FROZEN_SINGLE_PASS_CONTRACT_CHANGED','CONTROLLER_NO_RETRY_CONTRACT_CHANGED'] for e in errors),'missing_duplicate_or_unplanned_calls':sum(1 for e in errors if e['code']=='MISSING_DUPLICATE_OR_UNPLANNED_RESPONSE'),'no_test_inference_in_frozen_scope':True},'sides':sides,'audit_method_notes':['Direct /proc cmdline observations attest live interpreter argv. Its exe resolves to the conda base binary because the virtual environment executable is a symlink; the frozen native runner separately requires sys.prefix == .venv-local4b.','No retry is supported by STARTED guards, write-once response files, retry_index=0 and a single-pass frozen native loop. Exactly the 780 planned unique response records per side exist, and no FAILED sentinel exists. This is an artifact and frozen-code audit, not a system-call trace.','No accuracy or significance inference is made by this engineering audit. Independent final metric audit covers result interpretation.'],'audited_files':sorted(bindings.values(),key=lambda x:x['path'])}
OUT.parent.mkdir(parents=True,exist_ok=True)
with OUT.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'status':result['status'],'errors':len(errors),'path':str(OUT),'sha256':sha(OUT),'bound_files':len(bindings)},ensure_ascii=False))
if errors:print(json.dumps(errors[:20],ensure_ascii=False,indent=2))
raise SystemExit(0 if not errors else 1)
