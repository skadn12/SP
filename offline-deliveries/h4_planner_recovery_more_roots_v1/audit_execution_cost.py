"""Read-only audit of completed current four-root generation; no network or credentials.
Only writes reports/execution_audit.json, complete_cost_audit.json and FILE_INDEX.md.
Run with the frozen worktree on PYTHONPATH.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib,json
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.full_pipeline_qa_source import verify_qa_source

RUN=Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_full_pipeline_planner_recovery_more_roots_v1_live_1')
ROOTS=('cc-train-episode-259192','cc-train-episode-96462','cc-train-episode-112812','cc-train-episode-218519')
EXP_STAGES=('planner','gate','outline','realizer','adherence','continuation','reaudit','planner_supplement','gate_supplement','outline_supplement')
QA_STAGES=('miner','writer_primary','writer_repair')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def check_freeze(m):
 changed=[r['path']for r in m['files']if not Path(r['path']).is_file()or sha(r['path'])!=r['sha256']]
 assert not changed,changed
 return {'status':'PASS','file_count':len(m['files']),'changed_or_missing':changed}
def totals(records):
 result={'logical_calls':len(records),'http_attempts':sum(r['http_attempt_count']for r in records),'retries':sum(r['retry_count']for r in records),'verified_cost':'UNKNOWN'}
 for field in ('input_tokens','output_tokens','reasoning_tokens','total_tokens'):
  values=[r.get(field)for r in records]
  result[field]=sum(values)if all(type(v)is int for v in values)else'UNKNOWN'
  result[field+'_known_sum']=sum(v for v in values if type(v)is int)
 return result

assert (RUN/'generation_summary.json').is_file() and (RUN/'complete_call_records.json').is_file(), 'GENERATION_NOT_COMPLETE; do not audit in-flight artifacts'
summary=read(RUN/'generation_summary.json')
assert summary['status'] != 'RUNNING', 'GENERATION_NOT_COMPLETE'
manifest=read(RUN/'freeze_manifest.json')
bundle=read(RUN/'frozen/input_bundle_manifest.json')
assert manifest['root_order']==list(ROOTS)==summary['root_order']
assert read(manifest['input_bundle_manifest_path'])==bundle
assert read(RUN/'frozen/schedule.json')==manifest['schedule']
assert read(RUN/'EXECUTION_STARTED.json')['approved_http_budget']==68
assert manifest['schedule']['maximum_http_attempts']==manifest['schedule']['maximum_logical_calls']==68
freeze=check_freeze(manifest);input_freeze=check_freeze(bundle)
assert len(manifest['files'])==341 and len(bundle['files'])==90
assert sum(r['history_turn_count'] for r in bundle['roots'])==380
assert sum(r['segment_count'] for r in bundle['roots'])==32
ledgers=[];expansion=[];qa=[];locations={};inputs_checks=[];source_checks=[]
for root in ROOTS:
 lp=RUN/'expansion/execution'/root/'development_call_ledger.json'
 if lp.exists():
  ledger=read(lp);rs=ledger['records'];expansion.extend(rs)
  assert ledger['maximum_http_attempts']==ledger['maximum_logical_calls']==ledger['maximum_calls_per_root']==10
  assert ledger['logical_call_count']==len(rs)<=10
  assert ledger['http_attempt_count']==sum(r['http_attempt_count']for r in rs)<=10
  assert ledger['automatic_retry_count']==0 and ledger['automatic_resume']is False
  assert all(r['root_id']==root and r['stage']in EXP_STAGES for r in rs)
  assert all(c==1 for c in Counter(r['stage']for r in rs).values())
  for stage,prior in [('gate_supplement','planner_supplement'),('outline_supplement','gate_supplement')]:
   if any(r['stage']==stage for r in rs):
    assert any(r['stage']==prior and r['status']=='COMPLETED'for r in rs)
  for rec in rs:locations[id(rec)]=lp.parent/rec['artifact_directory']
  ledgers.append({'kind':'expansion','root_id':root,'path':str(lp),'sha256':sha(lp),'status':ledger['status'],'maximum_logic_http':10,'cost':totals(rs)})
 frozen_input=RUN/'frozen'/root/'inputs.json';inputs=read(frozen_input);br=next(r for r in bundle['roots']if r['root_id']==root)
 assert frozen_input.read_bytes()==Path(br['input_path']).read_bytes()
 assert inputs['prior_expanded']==read(br['source_input_path'])
 assert inputs['snapshot']==read(br['source_memory_path'])
 assert inputs['conversation_date_segments']==inputs['parent']['conversation_date_segments']
 assert digest(inputs['prior_expanded']['turns'])==inputs['provenance']['unchanged_turns_canonical_sha256']
 assert digest(inputs['conversation_date_segments'])==inputs['provenance']['conversation_date_segments_sha256']
 prepared=read(RUN/'frozen'/root/'planner_request.json')
 p=lp.parent/'calls'/root/'planner/request.json'
 if p.exists():
  actual=read(p)['payload'];assert prepared==actual
  text='\n'.join(m['content']for m in actual['messages'])
  assert all(t['turn_ref']in text and t['text']in text for t in inputs['prior_expanded']['turns'])
  assert all(d['conversation_date']in text for d in inputs['conversation_date_segments'])
 inputs_checks.append({'root_id':root,'status':'PASS','full_input_file_sha256':sha(frozen_input),'full_input_equals_frozen_dated_bundle':True,'original_P_R01_turns_and_order_unchanged':True,'memory_snapshot_unchanged':True,'turn_count':len(inputs['prior_expanded']['turns']),'date_segment_count':len(inputs['conversation_date_segments']),'all_original_turns_and_refs_in_primary_messages':p.exists(),'all_dates_in_primary_messages':p.exists(),'actual_primary_equals_prepared':p.exists(),'date_origin':bundle['date_origin'],'date_scope':bundle['date_scope'],'memory_source':br['source_memory_path'],'memory_source_sha256':sha(br['source_memory_path']),'memory_provenance':inputs['provenance']['memory_provenance'],'memory_fact_semantic_certification':False})
 sp=RUN/'qa/roots'/root/'qa_source_provenance.json'
 if sp.exists():
  provenance=read(sp);source=read(sp.with_name('qa_source.json'));verify_qa_source(source,provenance)
  result=read(RUN/'expansion/roots'/root/'result.json')
  assert source['turns']==result['promotable_expanded_dialogue']['turns']
  assert source['turns'][:len(inputs['prior_expanded']['turns'])]==inputs['prior_expanded']['turns']
  assert source['conversation_date_segments']==inputs['conversation_date_segments']
  assert all(s['conversation_date']=='UNKNOWN'for s in source['unknown_conversation_date_segments'])
  source_checks.append({'root_id':root,'status':'PASS','source_path':provenance['source_path'],'source_sha256':sha(provenance['source_path']),'provenance_path':str(sp),'provenance_sha256':sha(sp),'source_turn_count':len(source['turns']),'history_turn_count':provenance['history_turn_count'],'new_turn_count':len(provenance['new_turn_refs']),'new_scene_refs':provenance['promoted_scene_refs'],'historical_dates_unchanged':True,'new_R02_dates':'UNKNOWN','production_promotion_only':True,'turn_rewrite_or_truncation':False,'memory_or_feedback_in_qa_prompt':provenance['memory_or_feedback_in_qa_prompt']})
 if not sp.exists():
  source_checks.append({'root_id':root,'status':'NOT_EVALUATED','reason':'NO_FRESH_PROMOTED_QA_SOURCE'})
qp=RUN/'qa/execution/qa_call_ledger.json';ql=read(qp);qa=ql['records']
assert ql['root_order']==list(ROOTS)
assert ql['maximum_logical_calls']==ql['maximum_http_attempts']==28 and ql['maximum_calls_per_root']==7
assert ql['logical_call_count']==len(qa)<=28
assert ql['http_attempt_count']==sum(r['http_attempt_count']for r in qa)<=28
assert ql['automatic_retry_count']==0 and ql['automatic_resume']is False
assert all(r['root_id']in ROOTS and r['stage']in QA_STAGES for r in qa)
for rec in qa:locations[id(rec)]=qp.parent/rec['artifact_directory']
ledgers.append({'kind':'qa','path':str(qp),'sha256':sha(qp),'status':ql['status'],'maximum_logic_http':28,'maximum_per_root':7,'cost':totals(qa)})
for root in ROOTS:
 rs=[r for r in qa if r['root_id']==root]
 assert len(rs)<=7 and sum(r['http_attempt_count']for r in rs)<=7
 assert len([r for r in rs if r['stage']=='miner'])<=1
 for stage in ('writer_primary','writer_repair'):
  selected=[r for r in rs if r['stage']==stage]
  assert len(selected)<=3 and len({r['intent_id']for r in selected})==len(selected)
 for r in rs:
  if r['stage']=='writer_repair':assert any(p['stage']=='writer_primary'and p['intent_id']==r['intent_id']and p['ended_at']<r['reserved_at']for p in rs)
allrecords=expansion+qa
assert read(RUN/'complete_call_records.json')=={'expansion':expansion,'qa':qa}
assert totals(allrecords)==summary['cost']
assert totals(expansion)==summary['expansion_cost'] and totals(qa)==summary['qa_cost']
assert len(allrecords)<=68 and sum(r['http_attempt_count']for r in allrecords)<=68
assert all(r['retry_count']==0 for r in allrecords)
assert all(r['http_attempt_count'] in (0,1) for r in allrecords)
assert not any(r['status'] in ('RESERVED','HTTP_ATTEMPT_RESERVED') for r in allrecords), 'INCOMPLETE_CALL_RECORD'
chronological=sorted(allrecords,key=lambda r:r['reserved_at'])
indices=[ROOTS.index(r['root_id'])for r in chronological];assert indices==sorted(indices)
for root in ROOTS:
 rs=[r for r in chronological if r['root_id']==root]
 seen_qa=False
 for r in rs:
  if r['stage']in QA_STAGES:seen_qa=True
  elif seen_qa:raise AssertionError('expansion_after_qa')
tech=[r for r in chronological if r['status']=='TECHNICAL_FAILURE']
if tech:
 assert summary['status']=='STOPPED_TECHNICAL'
 assert chronological[-1]is tech[-1]
 assert all(r['reserved_at']<=tech[0]['ended_at']for r in chronological)
artifacts=[]
for r in chronological:
 d=locations[id(r)];request_file=read(d/'request.json');payload=request_file['payload'];capacity=read(d/'capacity.json')
 assert digest(payload)==r['payload_sha256']and digest(payload['messages'])==r['messages_sha256']
 assert capacity['truncation']is False
 assert payload['model']=='tju-llm'and payload['response_format']=={'type':'json_object'}
 assert payload['temperature']==(.1 if r['stage']in QA_STAGES else .25)
 assert payload['max_tokens']==r['maximum_output_tokens']
 metadata_path=d/'response_metadata.json'
 meta=read(metadata_path) if metadata_path.exists() else {}
 if r['status'] in ('COMPLETED','CONTENT_FAILURE'):assert metadata_path.is_file()
 if not metadata_path.exists():assert r['status']=='TECHNICAL_FAILURE'
 assert meta.get('transport_retry_count',0)==meta.get('rate_limit_retry_count',0)==0
 assert meta.get('http_attempt_count',r['http_attempt_count'])==r['http_attempt_count']
 raw=meta.get('raw_response')
 if isinstance(raw,dict):
  assert json.loads(meta['raw_response_text'])==raw
  if 'usage' in raw:assert raw['usage']==r['usage']==meta.get('usage','UNKNOWN')
  if 'model' in raw:assert raw['model']==r['actual_model']
  if raw.get('choices'):assert raw['choices'][0]['finish_reason']==r['finish_reason']
  if r['status']=='COMPLETED':
   assert r['finish_reason']=='stop'and r['completion_status']=='RESPONSE_RECEIVED'
   assert (d/'response.txt').read_text()==raw['choices'][0]['message']['content']
   assert r['http_status']==200 and r['timeout']is False
  if isinstance(raw.get('usage'),dict):
   u=raw['usage'];assert r['input_tokens']==u['prompt_tokens']and r['output_tokens']==u['completion_tokens']and r['total_tokens']==u['total_tokens']
 assert r['verified_cost']=='UNKNOWN'
 if r['stage']=='writer_primary'and r['status']=='COMPLETED':
  assert (d/'parse_result.json').is_file()
 artifacts.append({'root_id':r['root_id'],'stage':r['stage'],'intent_id':r.get('intent_id'),'status':r['status'],'http_attempts':r['http_attempt_count'],'retries':r['retry_count'],'actual_model':r['actual_model'],'finish_reason':r['finish_reason'],'http_status':r.get('http_status','UNKNOWN'),'timeout':r.get('timeout','UNKNOWN'),'completion_status':r['completion_status'],'full_usage':r['usage'],'input_tokens':r['input_tokens'],'output_tokens':r['output_tokens'],'reasoning_tokens':r['reasoning_tokens'],'total_tokens':r['total_tokens'],'verified_cost':'UNKNOWN','artifact_directory':str(d),'request_path':str(d/'request.json'),'request_sha256':sha(d/'request.json'),'response_path':str(d/'response.txt')if(d/'response.txt').exists()else None,'response_sha256':sha(d/'response.txt')if(d/'response.txt').exists()else None,'full_metadata_path':str(metadata_path) if metadata_path.exists() else None,'full_metadata_sha256':sha(metadata_path) if metadata_path.exists() else None,'raw_http_response_preserved':isinstance(meta.get('raw_response_text'),str),'http_response_available':bool(meta.get('raw_response_text')),'response_metadata_available':metadata_path.exists(),'requested_parameters':{k:payload[k]for k in ('temperature','max_tokens','response_format')},'confirmed_parameter_adoption':'UNKNOWN','capacity_status':capacity['status'],'proxy_input_tokens':capacity['input_tokens'],'service_capacity':'UNKNOWN'})
request_paths=list((RUN/'expansion/execution').glob('*/calls/*/*/request.json'))+list((RUN/'qa/execution').glob('calls/*/*/request.json'))
record_request_paths={locations[id(r)]/'request.json' for r in allrecords}
assert record_request_paths <= set(request_paths)
prepared_unsent_paths=sorted(set(request_paths)-record_request_paths)
for p in prepared_unsent_paths:
 assert summary['status']=='STOPPED_TECHNICAL'
 cap=read(p.with_name('capacity.json'))
 assert cap['status']!='PASS', 'UNACCOUNTED_REQUEST_WITH_PASS_CAPACITY'
 assert not p.with_name('response_metadata.json').exists()
 assert not p.with_name('response.txt').exists()
qa_summaries=[read(p)for root in ROOTS if(p:=RUN/'qa/roots'/root/'qa_summary.json').exists()]
for q in qa_summaries:
 for k in ('qa_memory_calls','solver_calls','training_calls','extra_semantic_audit_calls','free_writer_calls'):assert q.get(k,0)==0
local_memory=[]
for root in ROOTS:
 p=RUN/'expansion/roots'/root/'local_memory_accounting.json'
 if p.exists():
  d=read(p);assert d['remote_api_calls']==d['writes']==0
  local_memory.append({'root_id':root,**d})
for row in summary['per_root_costs']:
 root=row['root_id'];assert row['expansion']==totals([r for r in expansion if r['root_id']==root])and row['qa']==totals([r for r in qa if r['root_id']==root])
now=datetime.now(timezone.utc).isoformat();counts={s:sum(r['stage']==s for r in allrecords)for s in EXP_STAGES+QA_STAGES}
execution={'status':'PASS','scope':'EXECUTION_ACCOUNTING_AND_INPUT_INTEGRITY_NOT_SEMANTIC_QUALITY','run_root':str(RUN),'audited_at':now,'audit_added_api_calls':0,'generation_status':summary['status'],'root_order':list(ROOTS),'authorized_maximum_logical_calls':68,'authorized_maximum_http_attempts':68,'observed_cost':totals(allrecords),'stage_calls':counts,'ledger_checks':ledgers,'freeze':freeze,'input_freeze':input_freeze,'input_identity_checks':inputs_checks,'qa_source_checks':source_checks,'raw_request_response_checks':artifacts,'technical_failures':len(tech),'global_technical_stop_check':'PASS_LAST_CALL_IS_FAILURE'if tech else'NOT_EXERCISED_THIS_RUN_NO_TECHNICAL_FAILURE','retries':0,'timeout_count':sum(r.get('timeout')is True for r in allrecords),'actual_model_counts':dict(Counter(r['actual_model']for r in allrecords)),'finish_reason_counts':dict(Counter(r['finish_reason']for r in allrecords)),'unused_http_allowance':68-sum(r['http_attempt_count']for r in allrecords),'forbidden_stage_calls':{'memory_read_api':0,'memory_write_api':0,'free':0,'solver':0,'training':0,'extra_audit_api':0},'local_memory_accounting':local_memory,'full_response_and_usage_preserved':'PASS_AVAILABLE_PROVIDER_RESPONSE_AND_USAGE; OMITTED_FIELDS_UNKNOWN','prepared_but_not_logically_reserved_requests':[str(p) for p in prepared_unsent_paths],'failed_and_replaced_calls_included':True,'credential_material_read_or_printed_by_audit':False,'expansion_outcomes':summary['expansion_outcomes'],'qa_outcomes':qa_summaries,'unknowns':['service maximum capacity','actual adoption of requested parameters','default reasoning mode/budget','reasoning token breakdown when omitted','verified monetary cost']}
cost={'status':'PASS','run_root':str(RUN),'audited_at':now,'audit_added_api_calls':0,'current_batch_only':totals(allrecords),'expansion':totals(expansion),'qa':totals(qa),'by_root':{root:{'all':totals([r for r in allrecords if r['root_id']==root]),'expansion':totals([r for r in expansion if r['root_id']==root]),'qa':totals([r for r in qa if r['root_id']==root]),'by_stage':{s:totals([r for r in allrecords if r['root_id']==root and r['stage']==s])for s in EXP_STAGES+QA_STAGES}}for root in ROOTS},'by_stage':{s:totals([r for r in allrecords if r['stage']==s])for s in EXP_STAGES+QA_STAGES},'request_records':artifacts,'all_content_failed_batches_replaced_primary_and_repairs_included':True,'all_technical_failed_or_unknown_completion_attempts_included':True,'historical_calls_included':False,'qa_generation_summaries':qa_summaries,'structural_candidate_count':sum(q.get('candidate_count',0)for q in qa_summaries),'structural_valid_candidate_count':sum(q.get('valid_candidate_count',0)for q in qa_summaries),'structural_count_is_not_semantic_effective_count':True,'semantic_effective_question_count':'PENDING_SEPARATE_CODEX_OFFLINE_DIAGNOSIS','cost_per_semantically_effective_question':'NOT_COMPUTED_PENDING_SEMANTIC_DENOMINATOR; zero yield must remain UNDEFINED','verified_monetary_cost':'UNKNOWN','unknown_policy':'Provider missing usage/reasoning/cost remains UNKNOWN; known token sums include every attempt and are not monetary prices.'}
execution['memory_source_scope']='ORIGINAL_R01_POST_GENERATION_FROZEN_SNAPSHOT; prior six used reconstructed R1-only pre-generation snapshots; source integrity does not certify every Memory fact'
execution['sampling_scope']='REMAINING_FOUR_OF_HISTORICAL_SELECTED10; stratified historical selection, not population-random'
cost['qa_semantic_evaluation_by_root']={r:{'status':'PENDING_CODEX_OFFLINE_DIAGNOSIS' if any(x['root_id']==r and x['stage']=='writer_primary' for x in qa) else 'NOT_EVALUATED'} for r in ROOTS}
report=RUN/'reports';report.mkdir(exist_ok=True)
for filename in ('execution_audit.json','complete_cost_audit.json','FILE_INDEX.md'):
 if (report/filename).exists():raise FileExistsError(report/filename)
for filename,value in [('execution_audit.json',execution),('complete_cost_audit.json',cost)]:
 p=report/filename
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PASS','cost':cost['current_batch_only'],'stage_calls':counts,'freeze':freeze,'input_freeze':input_freeze,'technical_failures':len(tech),'reports':[str(report/'execution_audit.json'),str(report/'complete_cost_audit.json')],'audit_calls':0},ensure_ascii=False,indent=2))

def linked(label,path):
 path=Path(path)
 return f'[{label}]({path})' if path.exists() else None

def links(pairs):
 return ' · '.join(x for label,path in pairs if (x:=linked(label,path)))

def table_row(label,pairs):
 value=links(pairs)
 return f'| {label} | {value} |' if value else f'| {label} | 无本阶段产物；以实际调用账本为准。 |'

lines=['# 本轮实际输入输出索引','',
 '仅索引本次四个 root 的实际产物。查看顺序为「冻结输入 → Planner → Gate → Outline → Realizer → adherence／晋级 → Miner／Writer」。request.json 内的 payload 是实际请求，response.txt 是原始回答，response_metadata.json 保存完整 HTTP 返回和 usage。','',
 f'本轮 {len(allrecords)} 次逻辑调用、{sum(r["http_attempt_count"] for r in allrecords)} 次 HTTP；授权上限 68，自动重试 0。执行与成本审计不等于语义正确性认证。未生成 QA 的 root 标为 NOT_EVALUATED。','',
 '这四条取自历史 selected10 剩余固定集合，非总体随机抽样。Memory 使用原 R01 生成后冻结快照；前六条使用重建 R1-only 的生成前快照，来源方式有区别。日期是合成会话日期，新增 R02 日期为 UNKNOWN。','',
 '## 全局记录','',
 '- '+links([('完整调用记录',RUN/'complete_call_records.json'),('生成汇总',RUN/'generation_summary.json'),('执行审计',report/'execution_audit.json'),('完整成本审计',report/'complete_cost_audit.json')]),
 '- '+links([('冻结清单',RUN/'freeze_manifest.json'),('输入包清单',RUN/'frozen/input_bundle_manifest.json'),('执行配置',RUN/'frozen/execution_profile.json'),('服务参数',RUN/'frozen/provider_profile.json')]),
 '- '+links([('调用顺序与上限',RUN/'frozen/schedule.json'),('预算计划',RUN/'frozen/CALL_BUDGET_PLAN.md'),('授权启动记录',RUN/'EXECUTION_STARTED.json')]),'']
for root in ROOTS:
 short=root.removeprefix('cc-train-episode-');ep=RUN/'expansion/roots'/root;qr=RUN/'qa/roots'/root
 br=next(x for x in bundle['roots'] if x['root_id']==root);bp=Path(br['input_path']).parent
 outcome=next(x for x in summary['expansion_outcomes'] if x['root_id']==root)
 er=[r for r in chronological if r['root_id']==root and r['stage']in EXP_STAGES]
 qrs=[r for r in chronological if r['root_id']==root and r['stage']in QA_STAGES]
 stage_counts=Counter(r['stage'] for r in er)
 lines += [f'## {short}：输入与扩充','',f'输入 P+R01 共 {br["history_turn_count"]} 条，{br["segment_count"]} 段日期头。扩充状态 `{outcome["status"]}`。实际扩充调用：'+', '.join(f'{k}={v}' for k,v in stage_counts.items())+'。','']
 if outcome.get('errors'):lines += ['程序停止原因：`'+'; '.join(outcome['errors'])+'`。','']
 if not stage_counts.get('adherence'):
  lines+=['adherence 审核模型未调用。内部 realizer_adherence / scene_continuation 文件或 outcome 阶段标签不代表实际 HTTP 请求。','']
 lines+=['| 阶段 | 关键文件 |','| --- | --- |']
 rows=[('完整输入与原文',[('inputs.json',RUN/'frozen'/root/'inputs.json'),('带日期原文',bp/'history_dated.txt'),('无日期原文',bp/'history_plain.txt'),('日期验证',bp/'date_validation.json')]),
 ('冻结 Memory／反馈',[('Memory 快照',bp/'frozen_memory.json'),('反馈输入',bp/'feedback_input.json'),('实际事件菜单',ep/'event_verifier/materialized_tracks.json'),('人物注册表',ep/'planner/speaker_registry.json')]),
 ('Planner',[('原始解析输出',ep/'planner/output.json'),('逐候选保留／隔离',ep/'planner/validation.json'),('编译池',ep/'planner/compiled_plan.json'),('候选恢复审计',ep/'planner/candidate_recovery_audit.json'),('候选不足记录',ep/'planner/qualified_shortage.json')]),
 ('Gate',[('输入上下文',ep/'semantic_gate/context.json'),('输出',ep/'semantic_gate/output.json'),('验收',ep/'semantic_gate/validation.json'),('合并验收',ep/'semantic_gate/merged_validation.json')]),
 ('Outline',[('程序证据绑定',ep/'planner/outline_request_binding.json'),('输出',ep/'planner/outline_semantic_output.json'),('验收',ep/'planner/outline_semantic_validation.json'),('选择',ep/'planner/outline_semantic_selection.json'),('最终场景计划',ep/'planner/materialized_plan.json')]),
 ('单次候选补充与处理',[(str(p.relative_to(ep)),p) for p in sorted(ep.rglob('*supplement*')) if p.is_file()]),
 ('Realizer',[('输入',ep/'realizer/input.json'),('输入验收',ep/'realizer/input_validation.json'),('原始解析输出',ep/'realizer/output_batch_01.json'),('合并输出',ep/'realizer/output.json'),('验收',ep/'realizer/validation.json')]),
 ('adherence',[('输入',ep/'realizer_adherence/input.json'),('输出',ep/'realizer_adherence/output.json'),('验收',ep/'realizer_adherence/validation.json'),('最终审计',ep/'realizer_adherence/audit.json')]),
 ('晋级与新原文',[('晋级审计',ep/'promotion/audit.json'),('新 R02',ep/'promotion/promotable_child.json'),('P+R01+R02 JSON',ep/'promotion/promotable_expanded_dialogue.json'),('可读全文',ep/'promotable_expanded_dialogue.md'),('完整结果',ep/'result.json')]),
 ('结果与成本',[('outcome',ep/'outcome.json'),('扩充账本',RUN/'expansion/execution'/root/'development_call_ledger.json'),('本地 Memory 计数',ep/'local_memory_accounting.json')])]
 lines += [table_row(label,pairs) for label,pairs in rows]
 lines += ['',f'实际扩充调用共 {len(er)} 次：','', '| 调用 | 原始请求和返回 |','| --- | --- |']
 for r in er:
  d=locations[id(r)]
  lines.append(table_row(r['stage'],[('prompt',d/'prompt.txt'),('request',d/'request.json'),('response',d/'response.txt'),('metadata＋usage',d/'response_metadata.json'),('解析结果',d/'parsed_response.json')]))
 lines += ['',f'## {short}：Miner／Writer 与程序接收','']
 if not qrs:
  lines+=['QA：**NOT_EVALUATED**。本次未晋级，未使用历史扩充原文回填。','',links([('QA 跳过记录',qr/'qa_summary.json')]),'']
  continue
 lines+=['QA 使用本次晋级的完整 P+R01+R02，保留历史日期，新 R02 日期为 UNKNOWN。以下程序有效候选数量不代表原文支持的有效题量。','', '| 内容 | 文件 |','| --- | --- |']
 qrows=[('QA 原文与身份',[('可读原文',qr/'qa_transcript.txt'),('带编号原文',qr/'qa_numbered_transcript.txt'),('源 JSON',qr/'qa_source.json'),('来源与 SHA',qr/'qa_source_provenance.json')]),
 ('Miner 产物',[('intents',qr/'intents.json'),('推理合同',qr/'reasoning_contracts.json'),('排序运行上下文',qr/'ordering_runtime_context.json')]),
 ('全部 primary／repair',[('处理记录',qr/'candidate_generation.json'),('原始 prompt 目录',qr/'raw/prompts'),('原始 response 目录',qr/'raw/responses')]),
 ('程序接收',[('候选池',qr/'candidate_pool.json'),('程序有效候选',qr/'candidates_valid.json'),('逐题准入',qr/'validity_audit.json'),('QA 汇总',qr/'qa_summary.json')]),
 ('排序推导与 Gold',[('推导目录',qr/'ordering_derivations'),('排序完整性',qr/'ordering_integrity_audit.json'),('选项排列',qr/'permutations.json')]),
 ('调用与进度',[('QA 账本',RUN/'qa/execution/qa_call_ledger.json'),('QA 进度',qr/'qa_progress.json')])]
 lines += [table_row(label,pairs) for label,pairs in qrows]
 lines += ['',f'实际 QA 调用共 {len(qrs)} 次，包括全部 primary 与 repair：','', '| 调用 | 原始请求和返回 |','| --- | --- |']
 for r in qrs:
  d=locations[id(r)];label=r['stage']+((' / '+r['intent_id']) if r.get('intent_id') else '')
  lines.append(table_row(label,[('prompt',d/'prompt.txt'),('request',d/'request.json'),('response',d/'response.txt'),('metadata＋usage',d/'response_metadata.json'),('parse',d/'parse_result.json')]))
 derivations=sorted((qr/'ordering_derivations').glob('*/*.json'))
 if derivations:
  lines+=['','实际排序推导文件：','']+[f'- {linked(str(p.relative_to(qr/"ordering_derivations")),p)}' for p in derivations]
 lines.append('')
content_paths=[p for p in sorted(report.glob('*')) if p.is_file() and p.name not in ('execution_audit.json','complete_cost_audit.json','FILE_INDEX.md')]
if content_paths:
 lines+=['## 离线诊断与收口文件','']+[f'- {linked(p.name,p)}' for p in content_paths]+['']
lines+=['索引与审计新增 API 调用 0。所有失败批次、候选补充、被替换 primary 和 repair 已计入成本；缺失 usage 明细或金额维持 UNKNOWN。','']
(report/'FILE_INDEX.md').write_text('\n'.join(lines))
print(json.dumps({'file_index':str(report/'FILE_INDEX.md'),'existing_links_only':True,'new_external_calls':0},ensure_ascii=False))
