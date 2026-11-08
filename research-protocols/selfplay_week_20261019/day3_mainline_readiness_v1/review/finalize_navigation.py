#!/usr/bin/env python3
"""Publish bounded audit result to week navigation after actual validation."""
from pathlib import Path
import json, hashlib, datetime
RUN=Path(__file__).resolve().parents[1]
WEEK=RUN.parent
def read(p): return json.loads(p.read_text())
def binding(p): return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def main():
    d=read(RUN/'READINESS_DECISION.json')
    t=read(RUN/'review/TECHNICAL_CLOSEOUT_VERIFICATION.json')
    a=read(RUN/'review/INDEPENDENT_CLOSEOUT_AUDIT.json')
    assert t['all_pass'] and a.get('all_pass') is True, 'Actual closeout audit must pass before navigation publication'
    h=d['historical_regression']; c=d['candidate_training_preflight']; rc=d['ordering_four_plan_review']
    summary=dict(schema_version='day3_mainline_readiness_summary_v1',status=d['status'],historical_regression=h,new_family_supplement=d['new_family_supplement'],ordering_four_plan_review=rc,candidate_training_preflight=c,current_treatment=d['current_treatment'],final_training_pool_counts=d['final_training_pool_counts'],blocking_next_work=d['blocking_next_work'],new_solver_calls=0,new_model_inference=0,new_generated_questions=0,new_dialogue_expansions=0,new_training_runs=0,day3_solver_calls_cumulative_unchanged=3417,full_day_complete=False,training_start_go=False,technical_check_count=t['checks_total'],independent_audit=binding(RUN/'review/INDEPENDENT_CLOSEOUT_AUDIT.json'),decision=binding(RUN/'READINESS_DECISION.json'))
    (RUN/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    counts=h['dispositions']
    report=f'''# 第三天主线准入与第四天CPU准备检查

本次执行三条已冻结工作流，无新增模型调用、出题、扩充或训练。第三天累计Solver仍3417次。

## 可用成果

- 固定184道历史回归候选：PASS {counts.get('PASS',0)}、EXCLUDE {counts.get('EXCLUDE',0)}、HOLD {counts.get('HOLD',0)}。通过集合含主多选{h['admitted_primary_MC']}、排序{h['admitted_ordering']}、Gold6控制{h['admitted_control6']}，控制单列。逐题理由和实际请求见common_eval。历史回归不作为新题族或未见root泛化证据。
- 原187道内容候选及88组匹配槽位保持；176道配对候选CPU检查可用，每臂35多选、53排序。B的53排序具有合格窄跨场景反馈链，35多选按公共设计，不宣称正式MC弱点定向。
- 每臂2类多选关系、3类排序场景组合；352个epoch展示及{c['mapping_roundtrips']}次答案空间往返通过。全文输入{c['full_input_min']}—{c['full_input_max']} tokens，超长{c['overlength']}，两臂全文输入差{100*c['resource_summary']['relative_input_difference']:.4f}%。
- 真实模型、adapter与tokenizer文件哈希已核对；未加载权重。原70/30已在首道候选生成前修订为60个排序定向/60个公共题位，本轮没有再改处理定义。实际接受比例单独报告。
- 四个固定新增排序方案全部EXCLUDE；只审5个决定性历史记录，其余16个按停止规则未继续审。新题族补充仍4多选、0排序、3roots。

## 本轮修正与限制

独立复核检查了“全文出现属性”与“实际题目必要事实”之别。全文仅供检索，不能替代实际Train题全选项/排序项必要事实映射；首版与修订记录保留。历史评估曝光本身不是回归准入排除理由。非技术盲、非人工审核及复用历史来源QA的边界已披露。

## 未放行与下一步

两臂最终TRAIN_POOL仍各0条，这表示最终集合未准入，不表示176道候选失败。基础363道候选已绑定并通过长度检查，仍需与本轮新题完成必要事实/题族准入；旧题4道仅内容候选，最终准入0，未重启大规模旧题审核。当前q=0时75/0/25仅为混合目标，不是已实现份额。

下一步先有限审核固定363基础候选与176配对候选，形成两臂相同的公共基础集合，再冻结真实A/B训练池、混合份额、epoch、相同步数及停止配置。共同评估按已通过的历史回归范围处理；新题族多题型结论仍缺排序覆盖，四方案队列已停止，若补新素材须另作前瞻来源计划。

本次有限验收完成，但不等于第三天全部目标达成，也未完成第四天最终训练冻结。training_start_go=false；未证明反馈因果收益、memory收益或训练收益。

## 文件入口

- READINESS_DECISION.json：总裁决、下一消费者任务与输入绑定。
- common_eval/HISTORICAL184_ADMISSION.json：逐题准入；FAMILY_DEDUP_LEDGER.json：依赖题族；COMMON_REGRESSION_REQUEST_FREEZE.json及COVERAGE_DECISION.json：实际请求与允许结论范围。
- training/PAIR88_ELIGIBILITY_AND_FAMILY_LEDGER.json：实际配对与B反馈资格；TOKENIZATION_ZERO_TRUNCATION.json、DISPLAY_MAP_CHECKS.json、TRAINING_BUDGET_FREEZE.json：真实CPU检查及未冻字段。
- review/：技术验证、独立审核、首版修正记录与历史封存核查；navigation_history/保留更新前周导航。
'''
    (RUN/'REPORT_ZH.md').write_text(report)
    (RUN/'README_ZH.md').write_text('# 第三天主线有限验收已完成\n\n详见 [报告](REPORT_ZH.md)、[总裁决](READINESS_DECISION.json) 和 [机器摘要](SUMMARY.json)。\n\n候选CPU检查通过不等于最终训练放行；最终池与混合日程尚未冻结，training_start_go=false。历史回归和4多选/0排序新题族补充分开报告。\n')
    now=datetime.datetime.now(datetime.timezone.utc)
    progress=dict(summary=binding(RUN/'SUMMARY.json'),report=binding(RUN/'REPORT_ZH.md'),decision=binding(RUN/'READINESS_DECISION.json'),historical=h,ordering_four_plan_counts=rc,candidate_pairs_CPU_usable=88,candidate_records_CPU_usable=176,final_training_records_A=0,final_training_records_B=0,new_solver_calls=0,new_training_runs=0,training_start_go=False,full_day_complete=False)
    for name in ['CURRENT_STATUS.json','DAY3_LATEST.json']:
        p=WEEK/name; state=read(p)
        state.update(status=d['status'],active_workstream='MAINLINE_READINESS_BOUNDED_PREFLIGHT_CLOSED_NEXT_DAY4_COMMON_SET_ADMISSION',updated_at_utc=now.isoformat(),updated_at_beijing=now.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),current_execution_summary=binding(RUN/'SUMMARY.json'),current_execution_report=binding(RUN/'REPORT_ZH.md'),mainline_readiness_progress=progress,latest_iteration_solver_calls=0,new_training_runs=0,training_start_go=False,full_day_complete=False,next_action='Finite current necessary-fact/family admission for fixed363 foundation candidates versus176pairednew candidates, then actual common sets and matched A/B mixed schedule freeze. Preserve historical regression and insufficient4MC/0ordering new-family supplement separately; no further four-plan recovery or interface tuning.')
        state.setdefault('daily_planning',{})['day3']='FINITE_MAINLINE_ADMISSION_REVIEW_COMPLETE_NOVEL_MULTITYPE_COVERAGE_INSUFFICIENT'
        state['daily_planning']['day4']='CPU_CANDIDATE_PREFLIGHT_PASS_COMMON_SET_AND_FINAL_SCHEDULE_PENDING'
        state['before_execution_freeze_remaining']=['Final common foundation/new necessary-fact/family set admission.','Actual mixed A/B pools and epoch/step/stopping/runtime schedule.','Final evaluation execution lock at the scope actually admitted; novel multitype coverage remains insufficient.']
        if 'training' in state: state['training'].update(new_AB_pools_frozen=False,new_AB_training_runs=0)
        if 'common_evaluation_recovery' in state: state['common_evaluation_recovery'].update(newly_admitted_regression_questions=h['admitted_total'],historical_current_dispositions=h['dispositions'],current_admission=binding(RUN/'common_eval/HISTORICAL184_ADMISSION.json'))
        p.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    p=WEEK/'README_ZH.md'; old=p.read_text(); intro=f'''# 当前方案与进度

最新完成[第三天主线有限准入与CPU预检](day3_mainline_readiness_v1/REPORT_ZH.md)：历史184条已逐题裁决（通过{counts.get('PASS',0)}、排除{counts.get('EXCLUDE',0)}、暂缓{counts.get('HOLD',0)}）；通过集合含主多选{h['admitted_primary_MC']}、排序{h['admitted_ordering']}、Gold6控制{h['admitted_control6']}，历史回归与新题族单列。88组/176配对候选CPU映射、零截断和资源差检查通过，B53道窄跨场景排序反馈链核实；四个新增排序恢复方案均排除，停止该队列。

下一步是固定基础363与176配对候选的必要事实/题族准入、公共集合及实际混合日程冻结；最终训练池各0，新题族仍4多选/0排序。当前training_start_go=false、第三天全部目标未完成，未训练。新调用0，累计Solver仍3417。详见[总裁决](day3_mainline_readiness_v1/READINESS_DECISION.json)、[摘要](day3_mainline_readiness_v1/SUMMARY.json)与[独立审核](day3_mainline_readiness_v1/review/INDEPENDENT_CLOSEOUT_AUDIT.json)。以下各段为历史快照，以本段及CURRENT_STATUS.json为准。

'''
    p.write_text(intro+old.removeprefix('# 当前方案与进度\n\n'))
    print(json.dumps(dict(published=True,summary=binding(RUN/'SUMMARY.json'),report=binding(RUN/'REPORT_ZH.md')),ensure_ascii=False))
if __name__=='__main__': main()
