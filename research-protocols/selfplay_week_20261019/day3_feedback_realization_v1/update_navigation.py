from pathlib import Path
import json,hashlib,datetime,shutil
D=Path(__file__).resolve().parent;W=D.parent

def bind(p):
 b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def main():
 summary=json.loads((D/'SUMMARY.json').read_text());assert summary['actual_next_consumer_read'] and not summary['all_targets_loop_closed'];assert (D/'FINAL_MANIFEST.json').exists()
 now=datetime.datetime.now(datetime.timezone.utc);hist=D/'navigation_history';hist.mkdir(exist_ok=True)
 names=['README_ZH.md','CURRENT_STATUS.json','DAY3_LATEST.json']
 for name in names:
  dest=hist/name
  if dest.exists():assert dest.read_bytes()==(W/name).read_bytes()
  else:shutil.copy2(W/name,dest)
 entry=dict(status=summary['status'],summary=bind(D/'SUMMARY.json'),report=bind(D/'REPORT_ZH.md'),manifest=bind(D/'FINAL_MANIFEST.json'),feedback_snapshot=bind(D/'FEEDBACK_SNAPSHOT.json'),actual_next_plan=bind(D/'next_generation/agent_plan/EXPANSION_QUESTION_PLAN.json'),new_candidates=12,content_accepted=11,content_hold=1,new_source_turns=0,new_roots=0,content_realized_feedback_cards=4,planned_feedback_cards=5,new_solver_calls=77,direct_calls=66,diagnostic_calls=11,diagnostic_valid=8,diagnostic_invalid=3,source_verified_target_failure_records=7,actual_next_consumer_read=True,all_targets_loop_closed=False,formal_reproduced=False,matched_AB=False,training_started=False,training_admitted=0,dev_test_calls=0)
 nxt='Execute the next already-consumed minimal repair plan: separately freeze held evidence-sufficiency stem-scope repair and diagnostic-interface controlled probe. Do not retroactively admit old HOLD/invalids or restart broad expansion. Preserve current common-evaluation admission, family deduplication and symmetric A/B training-pool gates; no training go yet.'
 for name in ['CURRENT_STATUS.json','DAY3_LATEST.json']:
  p=W/name;s=json.loads(p.read_text());assert s['new_solver_calls']==3311;s['new_solver_calls']=3388;s['updated_at_utc']=now.isoformat();s['updated_at_beijing']=now.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat();s['status']='DAY3_FEEDBACK_REALIZATION_COMPLETE_WITH_ONE_TARGET_HELD';s['active_workstream']='FEEDBACK_BOUND_CANDIDATES_EVALUATED_AND_REVIEWED_NEXT_MINIMAL_REPAIR';s['next_action']=nxt;s['feedback_realization_progress']=entry;s['full_day_complete']=False;s['training_start_go']=False
  if name=='DAY3_LATEST.json':s['current_execution_summary']=entry['summary'];s['current_execution_report']=entry['report']
  p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+chr(10))
 p=W/'README_ZH.md';old=p.read_text();intro='''# 当前方案与进度

最新完成[反馈定向出题与回流执行v1](day3_feedback_realization_v1/REPORT_ZH.md)：四个既有Train roots，12候选→11内容准入（7多选、4排序）＋1HOLD→66直接作答＋11依据诊断→11份来源审核→新版反馈→下一消费者实际计划。多选题均exact30.95%，排序8.33%；诊断8有效/3无效，其中7条有经源核验局部目标错，另1条final正确但引用失真。Gold5候选因题干提前给cutoff保持HOLD，没有事后救回。

本轮五卡仅四卡实现合格候选；未新增对话、roots、训练或Dev/Test调用。下一步按[已实际消费的最小修复计划](day3_feedback_realization_v1/next_generation/agent_plan/PLAN_ZH.md)分别处理证据题范围和诊断接口小对照；不声称全目标闭环、formal reproduced、匹配A/B或反馈收益。第三天累计3388次Solver调用=此前3311+本批77；第二天另计。[全部候选与Gold](day3_feedback_realization_v1/exports/QUESTIONS_WITH_GOLD_ZH.md)、[完整可见依据](day3_feedback_realization_v1/exports/DIAGNOSTIC_OUTPUTS_ZH.md)、[新版反馈](day3_feedback_realization_v1/FEEDBACK_ZH.md)均已保存。以下旧段落为历史批次快照，其中的“尚无候选/建议8root”以本段及当前状态为准。

'''
 if old.startswith('# 当前方案与进度'+chr(10)):old=old.split(chr(10),1)[1].lstrip(chr(10))
 p.write_text(intro+old)
 print(json.dumps(dict(status='UPDATED',day3_solver_calls=3388,new_batch_calls=77,files=names),ensure_ascii=False))
if __name__=='__main__':main()
