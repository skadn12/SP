"""Build exact existing artifact links, no run/source mutation."""
from pathlib import Path
import json
D=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_gate_downstream_v1');R=D/'prepared_run'
m=json.loads((R/'freeze_manifest.json').read_text());O=Path(m['source_run']);G=Path(m['gate_run'])
roots=tuple('cc-train-episode-'+x for x in ('259192','96462','112812','218519'))
records=json.loads((R/'complete_call_records.json').read_text());checked=[]
def link(label,p):
 p=Path(p)
 if not p.is_file():raise FileNotFoundError(p)
 checked.append(str(p));return f'[{label}]({p})'
def row(label,files):return '| '+label+' | '+' · '.join(link(a,b) for a,b in files)+' |'
lines=['# 本轮完整路线人工查看索引','','只列经本地存在性核对的精确文件。本轮从保存Planner/新Gate结果继续；没有重新调用Planner或Gate。`response.txt`是模型实际返回内容，`output.json`通常是解析后的对象；validation、selection、Gold推导属于程序产物，不代表内容已认证。','','推荐先看96462：它完整覆盖初次Outline、导入保存补充、第二次Outline、Realizer、初审、一次续写和复审、晋升、Miner、Writer及repair。259192未用补充或续写；112812用了补充但未续写；218519预停，0新调用。','','## 总览','','- '+link('执行结果',R/'generation_summary.json'),'- '+link('完整调用与usage',R/'complete_call_records.json'),'- '+link('成本与执行身份审计',D/'reports/EXECUTION_AUDIT.md'),'- '+link('冻结清单',R/'freeze_manifest.json'),'- '+link('模型请求配置',R/'frozen/provider_profile.json'),'- '+link('正式执行profile',R/'frozen/execution_profile.json'),'- '+link('固定调度和预算',R/'frozen/schedule.json'),'- '+link('授权范围说明',R/'frozen/AUTHORIZATION_AND_BUDGET.md')]
for root in roots:
 f=R/'frozen'/root;t=R/'expansion/roots'/root;q=R/'qa/roots'/root;oldroot=O/'expansion/roots'/root
 inputs=json.loads((f/'inputs.json').read_text());n=len(inputs['prior_expanded']['turns'])
 dated=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_planner_recovery_more_roots_v1_inputs')/root
 dh=json.loads((dated/'dated_history.json').read_text())
 assert dh['turns']==inputs['prior_expanded']['turns'] and dh['conversation_date_segments']==inputs['conversation_date_segments']
 lines+=['',f'## {root}','','### 冻结P+R01和保存上游','',f'累计历史{n}条。可读历史的对白和日期段已与本轮冻结输入核对相同；合成日期仅标注会话，不自动确定回顾/计划事件日期。','', '| 阶段 | 精确文件 |','|---|---|',row('历史与来源',[('本轮完整输入',f/'inputs.json'),('可读日期历史',dated/'history_dated.txt'),('历史JSON',dated/'dated_history.json'),('本轮预检查',f/'precheck.json')]),row('保存上游总包',[('saved_planning',f/'saved_planning.json')])]
 for slot,gs,ps,suf in [('primary','gate','planner',''),('supplement','gate_supplement','planner_supplement','_supplement')]:
  pc=O/'expansion/execution'/root/'calls'/root/ps;gc=G/'execution/calls'/root/gs;gf=G/'frozen'/root/gs;gr=G/'results'/root/gs
  # Gate ledger artifacts are organized as root/stage in this run.
  if not gc.is_dir():
   ledger=json.loads((G/'execution/development_call_ledger.json').read_text());rec=next(x for x in ledger['records'] if x['root_id']==root and x['stage']==gs);gc=G/'execution'/rec['artifact_directory']
  lines.append(row('原Planner '+slot,[('实际请求',pc/'request.json'),('prompt',pc/'prompt.txt'),('模型响应',pc/'response.txt'),('解析输出',gf/'planner_output.json'),('候选验证',gf/'planner_validation.json')]))
  lines.append(row('已完成的新Gate '+slot,[('实际请求',gc/'request.json'),('prompt',gf/'prompt.json'),('模型响应',gc/'response.txt'),('解析输出',gr/'output.json'),('验证',gr/'validation.json'),('完整证据目录',gf/'context.json')]))
 lines+=['','以上supplement来自历史分支。其原上下文和锁定保留作来源；本轮只在首次新Outline不足时使用，并锁定本轮的选择。`saved_planning.json`保存字段对象，不含来源文件路径；上表给出实际来源位置。']
 if root==roots[-1]:
  lines+=['','### 预停','','Gate可继续候选仅1条，小于最低3场，因此没有Outline、Realizer、Miner或Writer调用。', '', '- '+link('预停记录',t/'outcome.json'),'- '+link('QA未运行记录',q/'qa_summary.json')]
  continue
 lines+=['','### 新Outline、选择与Realizer','', '| 阶段 | 精确文件 |','|---|---|']
 for stage,suf in [('outline',''),('outline_supplement','_supplement')]:
  cd=R/'expansion/execution'/root/'calls'/root/stage
  if not cd.exists():continue
  lines.append(row(stage+'模型输入输出',[('实际messages与参数',cd/'request.json'),('prompt',cd/'prompt.txt'),('原始响应',cd/'response.txt'),('服务usage',cd/'response_metadata.json')]))
  lines.append(row(stage+'程序处理',[('完整候选池',t/'planner'/('compiled_plan'+suf+'.json')),('解析响应',t/'planner'/('outline_semantic_output'+suf+'.json')),('验证',t/'planner'/('outline_semantic_validation'+suf+'.json')),('选择',t/'planner'/('outline_semantic_selection'+suf+'.json')),('请求绑定',t/'planner'/('outline_request_binding'+suf+'.json'))]))
 if (t/'planner/supplement_import_audit.json').exists():
  lines.append(row('补充触发及来源',[('初次缺额',t/'planner/qualified_shortage.json'),('保存补充导入说明',t/'planner/supplement_import_audit.json'),('Gate原/新锁定上下文',t/'semantic_gate/import_audit_supplement.json')]))
 lines.append(row('最终选中及Realizer输入',[('选择恢复审计',t/'planner/candidate_recovery_audit.json'),('materialized plan',t/'planner/materialized_plan.json'),('Realizer输入',t/'realizer/input.json'),('状态交接证据',t/'realizer/source_state_context.json')]))
 for stage in ('realizer','adherence','continuation','reaudit'):
  cd=R/'expansion/execution'/root/'calls'/root/stage
  if not cd.exists():continue
  lines.append(row(stage+'实际调用',[('请求',cd/'request.json'),('prompt',cd/'prompt.txt'),('原始响应',cd/'response.txt'),('解析响应',cd/'parsed_response.json'),('usage',cd/'response_metadata.json')]))
 lines.append(row('Realizer结构及初审',[('输出合同验证',t/'realizer/output_contract.json'),('adherence原输入',t/'realizer_adherence/input.json'),('初次审查',t/'realizer_adherence/initial_audit.json'),('最终审查',t/'realizer_adherence/audit.json')]))
 if (t/'scene_continuation/output.json').exists():
  lines.append(row('一次续写与复审',[('续写请求',t/'scene_continuation/request.json'),('续写输出',t/'scene_continuation/output.json'),('合并对白',t/'scene_continuation/merged_child.json'),('复审输入',t/'realizer_adherence/reaudit_input.json'),('复审结果',t/'realizer_adherence/reaudit_audit.json')]))
 else:lines.append('| 一次续写与复审 | 本root未触发；无对应模型调用。 |')
 lines.append(row('晋升',[('晋升规则结果',t/'promotion/audit.json'),('新增R02',t/'promotion/promotable_child.json'),('完整晋升原文',t/'promotion/promotable_expanded_dialogue.json'),('可读完整原文',t/'promotable_expanded_dialogue.md'),('本root执行结果',t/'outcome.json')]))
 lines+=['','### Miner与Writer','', '| 阶段 | 精确文件 |','|---|---|',row('出题唯一原文',[('带ID可读全文',q/'qa_numbered_transcript.txt'),('QA源JSON',q/'qa_source.json'),('来源及日期边界',q/'qa_source_provenance.json')])]
 for rec in records['qa']:
  if rec['root_id']!=root:continue
  cd=R/'qa/execution'/rec['artifact_directory'];label=rec['stage']+((' '+rec['intent_id']) if rec.get('intent_id') else '')
  paths=[('实际请求',cd/'request.json'),('prompt',cd/'prompt.txt'),('原始响应',cd/'response.txt'),('usage',cd/'response_metadata.json')]
  if (cd/'parse_result.json').exists():paths.append(('解析/修复触发',cd/'parse_result.json'))
  lines.append(row(label,paths))
 lines.append(row('Miner结果',[('intents',q/'intents.json'),('合同',q/'reasoning_contracts.json')]))
 lines.append(row('生成与准入',[('全部批次记录',q/'candidate_generation.json'),('原始程序题池',q/'candidate_pool.json'),('有效候选',q/'candidates_valid.json'),('准入审计',q/'validity_audit.json'),('QA汇总',q/'qa_summary.json')]))
 deriv=sorted((q/'ordering_derivations').glob('*/*.json'))
 lines.append(row('排序确定性验证',[('排序完整性',q/'ordering_integrity_audit.json'),('排序运行上下文',q/'ordering_runtime_context.json')]))
 if deriv:
  lines+=['','排序Gold推导文件（只验证符号执行，不保证语义支持）：','']
  for p in deriv:lines.append('- '+link(p.parent.name+' / '+p.name,p))
lines+=['','## 内容过滤和新反馈','','执行原始产物保持不变。内容审阅、过滤和反馈写在delivery外层，不覆盖prepared_run。下列文件在编制索引时单独标明状态；计划位置不是已完成结果。','']
for rel in ('reports/FINAL_REPORT.md','reports/CONTENT_259192.md','reports/CONTENT_96462.md','reports/CONTENT_112812.md','reports/ORDERING_FILTER_DECISIONS.json','reports/SC_MS_REVIEW.json'):
 p=D/rel
 lines.append('- '+(link(rel,p)+'（已存在）' if p.exists() else '`'+str(p)+'`（预期位置，尚未生成；实际文件名以最终收口为准）'))
lines+=['','本次过滤题池预期目录：`'+str(D/'filtered')+'/<root>/`；本次后续可用反馈预期目录：`'+str(D/'feedback')+'/<root>/feedback_input.json`。编制本索引时仍在准备，不能将其视作本轮已注入输入。','','现有旧反馈可参考 '+link('112812旧feedback_input格式',O.parent/'feedback/cc-train-episode-112812/feedback_input.json')+'。加载器只读取其中record；顶层activation/source_feedback是说明，不会自动激活。','','本索引未新增API调用，未修改run、原始题池或旧输入。']
(D/'reports/ARTIFACT_INDEX.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(link_count=len(checked),all_targets_exist=True,output=str(D/'reports/ARTIFACT_INDEX.md')),ensure_ascii=False))
