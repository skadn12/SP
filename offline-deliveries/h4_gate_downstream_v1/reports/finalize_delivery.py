"""Build separate offline filters, scoped feedback, and reports; zero API calls."""
import collections
import copy
import hashlib
import json
from pathlib import Path

D=Path(__file__).resolve().parents[1]
R=D/'prepared_run'
ROOTS=['cc-train-episode-'+x for x in ('259192','96462','112812','218519')]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def counts(rows,key):return dict(collections.Counter(r[key] for r in rows))
generation=read(R/'generation_summary.json')
calls=read(R/'complete_call_records.json')
calls=calls['expansion']+calls['qa']
all_reviews=[]
roots=[]
for root in ROOTS:
 short=root.rsplit('-',1)[-1]
 q=R/'qa/roots'/root
 reviews=[]
 report=None
 if (q/'candidate_pool.json').exists():
  report=read(D/f'reports/CONTENT_{short}.json')
  original_rows=report.get('raw_candidate_reviews',report.get('qa_candidates'))
  for r in original_rows:
   reviews.append({
    'root_id':root,'candidate_id':r['candidate_id'],'request_id':r['request_id'],
    'raw_row_index':r['raw_row_index'],'phase':r['phase'],'question_type':r['question_type'],
    'in_final_pool':r['in_final_pool'],
    'verdict':r.get('verdict',r.get('decision')),
    'whole_source_support':r.get('whole_source_support',r.get('full_transcript_answer_status')),
    'bound_evidence_support':r.get('bound_evidence_support',r.get('selected_evidence_support_status')),
    'reason':r.get('reason',r.get('why')),
    'reason_codes':r.get('reason_codes',r.get('error_codes')),
    'program_disposition':r.get('program_selection',r.get('program_disposition')),
    'report_path':str(D/f'reports/CONTENT_{short}.json')})
  pool=read(q/'candidate_pool.json')
 else:pool=[]
 final=[r for r in reviews if r['in_final_pool']]
 assert len({r['candidate_id'] for r in reviews})==len(reviews)
 assert {r['candidate_id'] for r in final}=={c['candidate_id'] for c in pool}
 byid={r['candidate_id']:r for r in final}
 ordering=[c for c in pool if c['question_type']=='ordering']
 parts={v:[c for c in ordering if byid[c['candidate_id']]['verdict']==v] for v in ('PASS','HOLD','FAIL')}
 filtered=[c for c in pool if c['question_type']!='ordering' or byid[c['candidate_id']]['verdict']=='PASS']
 strict=[c for c in pool if byid[c['candidate_id']]['verdict']=='PASS']
 strict_dedup=list(strict)
 semantic_duplicate_exclusions=[]
 if short=='259192':
  duplicate_id=f'{root}:I003-C-Q02:primary'
  strict_dedup=[c for c in strict if c['candidate_id']!=duplicate_id]
  semantic_duplicate_exclusions=[{'candidate_id':duplicate_id,'kept_candidate_id':f'{root}:I002-C-Q01:primary','basis':'Same safe-climbing strategy substantive QA; wording differs. Codex offline semantic dedup, no production change.'}]
 out=D/'filtered'/root
 write(out/'candidate_pool.json',filtered)
 write(out/'ordering_keep.json',parts['PASS'])
 write(out/'ordering_hold.json',parts['HOLD'])
 write(out/'ordering_excluded.json',parts['FAIL'])
 write(out/'strict_content_pass_candidates.json',strict)
 write(out/'strict_content_pass_deduplicated.json',strict_dedup)
 write(out/'content_review_decisions.json',final)
 manifest={'version':'codex_offline_original_candidate_filter_v1','root_id':root,
  'status':'COMPLETE' if pool else 'NOT_REACHED_NOT_EVALUATED',
  'source_pool':str(q/'candidate_pool.json') if pool else None,
  'source_pool_sha256':sha(q/'candidate_pool.json') if pool else None,
  'reviewer':'Codex','independent_human_certification':False,'external_audit_api_calls':0,
  'production_admission_modified':False,'candidate_objects_modified':False,'automatic_activation':False,
  'ordering_policy':'PASS requires unchanged supported nodes, relations and original bindings. HOLD means unresolved/incomplete, not proved wrong. FAIL reasons in content review. No repair or evidence backfill.',
  'SC_MS_policy':'candidate_pool.json preserves all original SC/MS unchanged, including diagnostic failures. It is NOT a certified training pool. strict_content_pass files separately contain only reviewed PASS.',
  'counts':{'original_pool':len(pool),'ordering_keep':len(parts['PASS']),'ordering_hold':len(parts['HOLD']),'ordering_excluded':len(parts['FAIL']),'ordering_filtered_pool':len(filtered),'strict_pass_all_types':len(strict),'strict_pass_semantic_dedup':len(strict_dedup)},
  'semantic_duplicate_exclusions':semantic_duplicate_exclusions,
  'identity_assertions':{'all_objects_equal_original':all(c in pool for c in filtered+strict+strict_dedup+sum(parts.values(),[])),'ordering_partition_complete':len(sum(parts.values(),[]))==len(ordering),'SC_MS_unchanged':filtered==[c for c in pool if c['question_type']!='ordering'] if not parts['PASS'] else True},
  'output_sha256':{p.name:sha(p) for p in sorted(out.glob('*.json')) if p.name!='filter_manifest.json'}}
 assert all(manifest['identity_assertions'].values())
 write(out/'filter_manifest.json',manifest)
 root_calls=[c for c in calls if c['root_id']==root]
 cost={'logical_calls':len(root_calls),'http_attempts':sum(c['http_attempt_count'] for c in root_calls),'retry_count':sum(c['retry_count'] for c in root_calls),'input_tokens':sum(c['input_tokens'] for c in root_calls),'output_tokens':sum(c['output_tokens'] for c in root_calls),'total_tokens':sum(c['total_tokens'] for c in root_calls),'reasoning_tokens':'UNKNOWN' if root_calls else 0,'verified_cost':'UNKNOWN',
  'tokens_per_strict_valid_ordering':'UNDEFINED_ZERO_VALID_ORDERING',
  'verified_cost_per_valid_ordering':'UNDEFINED_ZERO_VALID_ORDERING',
  'tokens_per_strict_deduplicated_QA_including_expansion':round(sum(c['total_tokens'] for c in root_calls)/len(strict_dedup),2) if strict_dedup else 'UNDEFINED_ZERO_VALID_QA'}
 result={'root_id':root,'status':'QA_FINISHED_OFFLINE_REVIEW_COMPLETE' if pool else 'PRESTOP_GATE_POOL_1_LT_3',
  'raw_candidate_count':len(reviews),'program_final_count':len(pool),'raw_by_type':counts(reviews,'question_type'),'final_by_type':counts(final,'question_type'),
  'final_review':counts(final,'verdict'),'ordering_final_review':counts([r for r in final if r['question_type']=='ordering'],'verdict'),
  'ordering_whole_source_support':counts([r for r in final if r['question_type']=='ordering'],'whole_source_support'),
  'strict_pass_SC_MS':len(strict),'strict_pass_SC_MS_semantic_dedup':len(strict_dedup),'ordering_strict_grounded':len(parts['PASS']),
  'expansion':report['expansion'] if report else {'status':'NOT_REACHED_NOT_EVALUATED','eligible_candidates':1,'required_minimum':3},'cost':cost,
  'content_report':str(D/f'reports/CONTENT_{short}.json') if report else None,'filter_manifest':str(out/'filter_manifest.json')}
 roots.append(result)
 all_reviews.extend(reviews)

 # Preserve the exact prior P+R01 fact checks. Put current branch observations
 # under a separately scoped field actually consumed by the renderer.
 frozen=read(R/'frozen'/root/'inputs.json')
 prior=frozen['feedback_record']
 record=copy.deepcopy(prior)
 record['version']='codex_source_bound_downstream_feedback_v1'
 perf=record.setdefault('performance',{})
 perf['solver_behavior_observations']='NOT_PROVIDED_SOLVER_NOT_RUN'
 post={
  'version':'codex_gate_downstream_review_v1','root_id':root,'reviewer':'Codex','method':'CODEX_OFFLINE_DIAGNOSIS_NOT_INDEPENDENT_HUMAN_CERTIFICATION','status':result['status'],
  'source_run':str(R),'historical_fact_checks_scope':'FROZEN_P_PLUS_R01_ONLY_UNCHANGED',
  'branch_boundary':'The observations below concern this specific new R02 and generated QA. They are not extra P+R01 history. If a future run starts from P+R01, do not treat these outcomes as already achieved. A future run starting from this promoted R02 must explicitly supply that transcript.',
  'expected_vs_actual':'Planner/Outline describe expected changes. Only supplied dialogue establishes realized outcomes. App capability, future check-ins and proposed meetings are not completed actions.',
  'do_not_force_topics_or_answers':'Use facts and uncertainty to choose useful developments; do not copy old questions/Gold or synthesize history merely to satisfy a desired chronology.',
  'solver_observations':'NOT_PROVIDED_SOLVER_NOT_RUN',
  'program_counts':{'raw':len(reviews),'accepted':len(pool),'accepted_by_type':result['final_by_type']},
  'content_counts':{'final':result['final_review'],'ordering':result['ordering_final_review'],'ordering_full_source':result['ordering_whole_source_support']},
  'expansion_observations':result['expansion'],
  'qa_observations':[{'candidate_id':r['candidate_id'],'question_type':r['question_type'],'verdict':r['verdict'],'whole_source_support':r['whole_source_support'],'reason':r['reason']} for r in final],
  'repair_observations':'Some repairs repeated existing SC or invalid IDs; no semantic certification follows from quota or schema acceptance.' if pool else 'NOT_REACHED_NOT_EVALUATED',
  'filter_activation':'LOCAL_REVIEW_ARTIFACT_ONLY_NOT_AUTOMATICALLY_ACTIVE',
  'review_path':result['content_report'],
  'source_transcript_path':str(q/'qa_source.json') if pool else None,
  'source_transcript_sha256':sha(q/'qa_source.json') if pool else None,
 }
 perf['post_generation_review']=post
 perf['post_generation_scope_notice']='New post_generation_review is branch-scoped evaluation, not an update of historical P+R01 fact_checks. Legacy renderer calls performance solver_observations; no Solver ran.'
 assert perf.get('fact_checks')==prior.get('performance',{}).get('fact_checks')
 feedback={'version':'codex_feedback_input_v1','root_id':root,'record':record,
  'activation':'NOT_AUTOMATICALLY_ACTIVE; next prepare must explicitly load this file',
  'source_feedback':{'path':str(R/'frozen'/root/'inputs.json'),'record_sha256':hashlib.sha256(json.dumps(prior,ensure_ascii=False,sort_keys=True).encode()).hexdigest()},
  'source_run':str(R),'model_calls':0,'original_fact_checks_preserved':True}
 write(D/'feedback'/root/'feedback_input.json',feedback)

assert len(all_reviews)==42 and sum(r['program_final_count'] for r in roots)==29
write(D/'reports/CONTENT_ALL_ROWS.json',all_reviews)
accepted=[r for r in all_reviews if r['in_final_pool']]
accepted_order=[r for r in accepted if r['question_type']=='ordering']
strict_n=sum(r['strict_pass_SC_MS_semantic_dedup'] for r in roots)
summary={'version':'h4_gate_downstream_final_v1','status':'COMPLETED_STOPPED','source_run':str(R),
 'scope':'Resume existing frozen Planner/Gate into fresh Outline and downstream; NOT a fresh Planner or controlled Gate causal comparison.',
 'new_model':'qwen3.6-35b-a3b','roots_requested':4,'roots_reached_qa':3,'roots_prestopped':1,'new_planner_gate_http':0,
 'http_budget':39,'cost':generation['cost'],'expansion_cost':generation['expansion_cost'],'qa_cost':generation['qa_cost'],
 'prior_gate_cost_separate':generation['previous_gate_cost_separate'],
 'gate_plus_downstream_cost_excluding_original_planner':{'logical_calls':39,'http_attempts':39,'total_tokens':630830,'verified_cost':'UNKNOWN'},
 'raw_candidates':42,'raw_by_type':counts(all_reviews,'question_type'),'program_accepted':29,'program_final_by_type':counts(accepted,'question_type'),
 'program_dispositions_all_raw':counts(all_reviews,'program_disposition'),
 'final_content_review':counts(accepted,'verdict'),'final_ordering_review':counts(accepted_order,'verdict'),'final_ordering_whole_source_support':counts(accepted_order,'whole_source_support'),
 'strict_valid_ordering':0,'strict_valid_ordering_deduplicated':0,'strict_PASS_SC_MS':14,'strict_PASS_SC_MS_semantic_dedup':strict_n,
 'tokens_per_valid_ordering':'UNDEFINED_ZERO_VALID_ORDERING','verified_cost_per_valid_ordering':'UNDEFINED_ZERO_VALID_ORDERING',
 'tokens_per_strict_deduplicated_QA_including_expansion':round(generation['cost']['total_tokens']/strict_n,2),
 'confirmed_opposite_direction_edges':0,
 'final_error_candidate_counts_overlapping':{'actor_attribution':2,'event_state_vs_confirmation_identity':1,'capability_as_completed':2,'unsupported_unique_temporal_order_fail':2,'unresolved_unique_order_hold':1,'single_choice_not_unique':1,'unsupported_strengthening_strict_budget':1,'unsupported_purpose_risk_management':1},
 'call_stages':counts(calls,'stage'),'writer_primary_calls':9,'writer_repair_calls':6,'miner_calls':3,'new_outline_calls':5,'realizer_calls':3,'initial_adherence_calls':3,'continuation_calls':1,'reaudit_calls':1,
 'technical_failures':0,'writer_json_parse_failures':0,'ordering_abstentions':0,'external_audit_calls':0,'solver_calls':0,'training_calls':0,'free_calls':0,'remote_memory_calls':0,
 'memory_boundary':'Existing local Memory snapshots were used in expansion (not no-Memory); QA uses complete transcript only. No remote Memory read/write API.',
 'dates':'Preserved synthetic dates in frozen P+R01. New R02 UNKNOWN; no silent inferred dates.',
 'review_method':'Codex offline diagnosis, not independent human certification and not a new production semantic gate.',
 'ordering_interpretation':'0 strict unchanged fully-bound ordering survivors is not 0/8 answer accuracy. Three orders have full-source support, four are held, four excluded; these dimensions differ.',
 'supplement_context':'Historical supplementary candidates/Gate imported with provenance, not regenerated. Fresh Outline selections determine current locks. Two realized locks happened to equal historical values.',
 'tests':read(D/'TEST_RESULTS.json'),
 'limitations':['4 known roots, only 3 reached QA; no dataset prevalence estimate.','No paired fresh pre-fix control; success cannot be attributed solely to Gate validator.','School service reasoning split and verifiable money remain UNKNOWN.','New information exists but is incompletely converted into effective QA.','Frozen generation_summary remains an original snapshot with PENDING review; this report is authoritative for completed offline review.'],
 'roots':roots,'automatic_next_experiment':False,'feedback_automatically_activated':False}
assert summary['program_dispositions_all_raw']=={'RETAIN':29,'DUPLICATE':4,'REJECT_PROGRAM':9}
assert summary['final_ordering_review']=={'FAIL':4,'HOLD':4}
write(D/'reports/FINAL_REPORT.json',summary)

md='''# Gate 后续流程正式运行与离线收口

已完成并停止。保留既有 Planner/Gate 输出，从新的 Outline 继续，3/4 条走到 Writer；218519 因 Gate 可用候选 1＜3 在发送前停止。本轮没有重跑 Planner 或 Gate。

|Root|到达阶段|新晋升场景|程序接收 SC / MS / 排序|排序严格保留 / 暂缓 / 排除|有据 SC/MS 去重后|新 HTTP|新 tokens|
|---|---|---:|---:|---:|---:|---:|---:|
|259192|Writer完成|3|4 / 3 / 2|0 / 0 / 2|3|9|121,962|
|96462|Writer完成，一次续写及复审|3|5 / 2 / 3|0 / 2 / 1|7|11|141,917|
|112812|Writer完成|3|4 / 3 / 3|0 / 2 / 1|3|11|182,083|
|218519|Gate候选不足，预停|0|0 / 0 / 0|0 / 0 / 0|0|0|0|
|合计||9|13 / 8 / 8|0 / 4 / 4|13|31|445,962|

## 当前结论

流程已经继续到出题，局部失败没有再清空整批合格题，JSON解析失败为0。但“程序接收29题”不是“29题正确”：Codex离线检查为PASS14/HOLD5/FAIL10，PASS中有一组实质近重复，去重后13题。严格有据排序暂为0，不能解读为8道Gold全部错误。

8道排序中，3道的顺序在全文中有支持（96462两道、112812日历一道），但原始节点或时间边的证据绑定不完整。另有1道办公问题排序的完整唯一顺序尚无法确定，合计4道HOLD。其余4道FAIL涉及事件身份/状态或无依据强排。没有发现可明确计数的“应A早于B、却填B早于A”反向边；没有给未知关系编造替代唯一答案。

过滤保持题目、选项、引用和Gold原样，没有给模型补证据再当原题通过。`filtered/<root>/candidate_pool.json`只应用排序过滤，保留全部原SC/MS，因此不是训练合格池；`strict_content_pass_deduplicated.json`另存本轮Codex诊断通过、语义去重后的13题。所有原raw和生产pool保留。

## 扩充确实产生变化，但仍有选材与依赖问题

- 259192：攀岩预订与准备会、Marco练习的时间地点、Sarah新关系及联系方式都有具体增量。
- 96462：课程详情和到场入组有增量；初稿只说将发详情，一次正式续写补出 Tuesday 7 PM、Room 12 和收到确认，再复审通过。咨询虽是新一次事件，但大量重复旧建议，信息增量偏弱。
- 112812：日历同步与测试完成、老板会议时段获回复并接受、记账app及周日check-in安排成立。S02使用新日历，却未在Outline依赖字段声明S01；本次恰好先实现了S01，不能据此说依赖问题修好。旧Memory里boss=Elena摘要仍在未核实导航字段中，本轮对白没有复制这个身份错误。
- 218519：仍缺合格候选。本轮遵守不重跑Planner的范围，没有擅自补采样。

这些结果说明“没有新变化”并非这3条当前共同现象，也不证明Planner全面解决。新增事实向有效QA的转化仍弱：96462的7道严格通过SC/MS、112812的3道都可由旧历史回答；课程新细节、日历实际完成等没有成为严格通过的新题。

## 还剩哪些具体问题

1. Writer的事件和状态语义：112812把“can set a monthly target”写为已设月目标，影响MS与排序；把James配置日历归给Emily。259192也有把David建议归给Emily的问题。
2. 排序选项与时间支持：259192将“车报废状态”混为“后来确认报废”；未知日期场景按提及顺序强排。96462持续学习计划的起点/区间不明，不能插入其他活动之间强求唯一序。
3. 绑定上下文：存在的ID/原文片段不一定覆盖人物、指代、完成状态和关系。112812日历在全文中确已完成，且同题另一条边已有完成证据；问题仅是相应节点绑定不全，不能误报成纯计划升级。
4. Repair效率：42条raw＝29接收＋9程序拒绝＋4去重。6次repair中，重复已收SC或重复错误ID仍出现；112812修对ID后仍保留状态错误。缺题并非JSON解析坏，本轮主要是少出SC、重复题和引用命名错误。
5. 上游质量：有候选不足、咨询回顾偏多、遗漏跨场依赖；部分Miner的trigger/after标签也有问题。Writer有时绕开错误intent，因此不能把所有新错误归给上游。

未在收口时继续修改生产规则或追加生成。上述是新输出暴露的问题，不是用旧错例回放替代验证。

## 运行、输入与成本

31次逻辑调用＝31次HTTP，低于39上限：扩充13，QA18（Miner3、Writer primary9、repair6）。全部实际返回 `qwen3.6-35b-a3b`、HTTP200、finish_reason=stop；重试、超时、技术失败均0。没有Free、Solver、训练、远程Memory或额外审核API。扩充使用已有本地Memory快照；不能把远程调用0说成模型没看Memory。QA仅用本次晋升后的完整原文。

输入398,415＋输出47,547＝445,962 tokens；其中扩充209,106、QA236,856，含全部失败候选、重复repair和未留下有效题的调用。学校服务推理token细分及可核实金额为UNKNOWN。严格有据排序为0，单排序成本未定义，不伪造有限值。全部成本摊到13道严格通过且去重的SC/MS约34,304.77 tokens/题，只是本轮平均消耗，不是边际费用。

此前Gate另计8次HTTP、184,868 tokens；Gate＋本轮累计39次、630,830 tokens，不包括更早Planner实验成本。不能把这个累计39误认为本轮发了39次。

冻结P+R01日期保持原样（合成对话日期，不是恢复的真实时间戳）；新R02无授权日期，保留UNKNOWN。本次继承历史补充候选及其Gate，没有假称在当前Outline条件下重新生成补充候选。当前锁定由本次Outline决定，两个实际锁定值恰与历史一致。

346项离线测试通过；执行审计109项通过，当前358、上轮Gate229、旧Planner293条冻结身份保持一致。所有31次usage与原始响应对账。离线收口另有终检文件。

## 本地反馈和人工查看

四份 `feedback/<root>/feedback_input.json` 已保存：原 `performance.fact_checks` 逐对象保留，新观察放在 `performance.post_generation_review`，明确区分原P+R01事实与本分支R02结果；Solver观察仍NOT_RUN。没有把失败题Gold写成事实、自动激活反馈或启动下一轮。

这是Codex离线诊断，不是独立人工认证，也未新增生产语义门槛。4条已知案例中仅3条完成生成，无新旧随机配对对照，不能推出数据集普遍比例或单独归因Gate修复。

主要文件：

- [完整逐阶段人工索引](ARTIFACT_INDEX.md)
- [259192逐题报告](CONTENT_259192.md)
- [96462逐题报告](CONTENT_96462.md)
- [112812逐题报告](CONTENT_112812.md)
- [42条统一身份及诊断](CONTENT_ALL_ROWS.json)
- [执行审计](EXECUTION_AUDIT.md)
- [机器可读汇总](FINAL_REPORT.json)
'''
(D/'reports/FINAL_REPORT.md').write_text(md)
(D/'README.md').write_text('''# H4 Gate 后续流程：已完成并停止

本轮从冻结Planner/Gate输出继续，未重跑上游。3条到Writer，1条候选不足预停。31 HTTP、445,962 tokens；程序接收29题，Codex严格过滤排序0保留/4暂缓/4排除。SC/MS诊断通过14道中去重后13道。不能把排序严格绑定保留0说成所有答案错误。

- [最终报告](reports/FINAL_REPORT.md)
- [逐阶段输入输出索引](reports/ARTIFACT_INDEX.md)
- [逐题诊断汇总](reports/CONTENT_ALL_ROWS.json)
- [执行审计](reports/EXECUTION_AUDIT.md)
- 原始生成产物：`prepared_run/`（未改写）
- 独立排序过滤及严格内容诊断池：`filtered/<root>/`
- 本地未激活反馈：`feedback/<root>/feedback_input.json`

历史执行命令（已执行一次；勿再次执行以免重复付费）：

```bash
cd /root/autodl-tmp/gxy-selfplay/worktrees/h4-gate-downstream-v1
python -B -m scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.gate_downstream_resume execute --approved-http-budget 39
```

最终状态由reports/FINAL_REPORT.json表示。prepared_run/generation_summary.json保留生成结束时“待离线诊断”的原始快照，没有回写。没有自动进入Solver、训练或其他实验。
''')
print(json.dumps({k:summary[k] for k in ['status','raw_candidates','program_accepted','final_content_review','final_ordering_review','final_ordering_whole_source_support','strict_PASS_SC_MS_semantic_dedup']},ensure_ascii=False))
