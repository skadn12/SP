"""Publish local navigation after complete scoring and independent verification."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
WEEK=HERE.parent


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p=Path(p).resolve()
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}


def save(p,obj,exclusive=False):
    with Path(p).open('x' if exclusive else 'w') as f:
        f.write(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def main():
    summary=read(HERE/'SUMMARY.json')
    audit=read(HERE/'audit/POST_SCORE_REVIEW.json')
    assert summary['status']=='COMPLETE_TRAIN_DIAGNOSTIC_ONLY' and audit['status']=='PASS'
    assert audit['summary_binding']['sha256']==bind(HERE/'SUMMARY.json')['sha256']
    assert summary['denominators']['calls']==2688
    assert summary['scope']['score_selected_deletions']==0 and not summary['scope']['training_started']
    now=datetime.now(timezone.utc)
    manifest_files=['README_ZH.md','REPORT_ZH.md','SUMMARY.json','per_question.csv',
        'MATCHED_PAIR_DIAGNOSTICS.json','CANDIDATE_METADATA.json','MATCHED_SLOT_PAIRS.json',
        'PRODUCTION_SLOT_LEDGER.json','EXECUTION_FREEZE_V2.json','CONTENT_ADMISSION_V2.json',
        'FORMAT_NORMALIZATION_V2.json','INPUT_ASSEMBLY.json','CPU_SELFTEST.json',
        'inputs_private_v2/INPUT_REVIEW.json','audit/PUBLIC_PREFLIGHT_V2_PASS.json','audit/POST_SCORE_REVIEW.json',
        'runtime/direct/PREFLIGHT_FREEZE.json','runtime/direct_v2/PREFLIGHT_FREEZE.json',
        'runtime/direct_v2/STARTED.json','runtime/direct_v2/COMPLETE.json',
        'runtime/direct_v2/RAW_RESPONSE_FREEZE.json','runtime/direct_v2/SCORING_STARTED.json',
        'runtime/direct_v2/SCORES.json','assemble_inputs.py','normalize_format.py',
        'summarize_results.py','finalize.py','live_v2.log']
    manifest={'status':'COMPLETE_VERIFIED','created_at_utc':now.isoformat(),
        'files':[bind(HERE/p) for p in manifest_files],
        'transitive_data_bindings':'Public request freeze binds all prompts, maps, model/code and full-source files; RAW_RESPONSE_FREEZE binds all2688 responses; scoring binds private Gold and raw freeze.',
        'new_solver_calls':2688,'new_training_runs':0,'dev_test_calls':0,
        'initial_preparation_without_inference':True}
    save(HERE/'FINAL_MANIFEST.json',manifest,exclusive=True)
    history=HERE/'navigation_history/before_completion'
    history.mkdir(parents=True,exist_ok=False)
    for name in ['CURRENT_STATUS.json','DAY3_LATEST.json','README_ZH.md','DAY3_REMAINING_ZH.md']:
        shutil.copy2(WEEK/name,history/name)
    entry={'status':'COMPLETE_TRAIN_DIAGNOSTIC_ONLY','questions':187,'roots':20,'calls':2688,
        'valid_calls':summary['overall']['valid_calls'],'invalid_calls':summary['overall']['invalid_calls'],
        'by_type':{k:{f:g[f] for f in ['questions','calls','question_equal_exact','root_equal_exact','question_strata']} for k,g in summary['by_type'].items()},
        'report':bind(HERE/'REPORT_ZH.md'),'summary':bind(HERE/'SUMMARY.json'),
        'per_question':bind(HERE/'per_question.csv'),'manifest':bind(HERE/'FINAL_MANIFEST.json'),
        'post_score_review':bind(HERE/'audit/POST_SCORE_REVIEW.json'),
        'content_admission_changed_by_solver':False,'training_started':False,'dev_test_calls':0}
    for name in ['CURRENT_STATUS.json','DAY3_LATEST.json']:
        p=WEEK/name
        d=read(p)
        d.update(status='DAY3_TRAIN_DIAGNOSTIC_COMPLETE_COMMON_EVAL_ADMISSION_PENDING',
            updated_at_utc=now.isoformat(),new_solver_calls=2688,new_training_runs=0,
            train_candidate_solver_diagnostic=entry,
            next_action='Complete current score-blind admission of fixed historical184 regression candidates; keep new-family4 separate. Then freeze common DEV requests and symmetric A/B training pools/schedules under existing protocol. TEST final-only.')
        if 'updated_at_beijing' in d:
            d['updated_at_beijing']=now.astimezone(timezone(timedelta(hours=8))).isoformat()
        d['full_day_complete']=False
        d['training_start_go']=False
        save(p,d)
    mc=summary['by_type']['multi_select'];order=summary['by_type']['ordering']
    (WEEK/'README_ZH.md').write_text(f'''# 当前方案与进度

采用[修订方案v2](PLAN_V2_ZH.md)与[逐日执行细则v2.1](DAILY_EXECUTION_V2_ZH.md)。当前状态见[CURRENT_STATUS.json](CURRENT_STATUS.json)与[DAY3_LATEST.json](DAY3_LATEST.json)。

已完成S0＝M-clean对全部187道训练候选的统一诊断：74道多选、113道排序，20roots，共2688次作答。多选完整正确率{mc['question_equal_exact']:.2%}，排序{order['question_equal_exact']:.2%}（题目等权）。全部题目及结果保留，未按Solver分数改内容准入，未训练、未调用DEV/TEST。

- [本轮Solver诊断报告](day3_solver187_v1/REPORT_ZH.md)、[逐题表](day3_solver187_v1/per_question.csv)、[机器汇总](day3_solver187_v1/SUMMARY.json)
- [训练生产报告](day3_production_batch2_v1/REPORT_ZH.md)：240计划槽位、187接受、8内容HOLD、42素材缺额、3生成拦截；训练池尚未冻结。
- [共同评估范围修正](day3_eval_recovery_v2/REPORT_ZH.md)：历史184条是回归候选，仍待当前准入；[新增题族评估v1](day3_eval_clearance_v1/REPORT_ZH.md)仅4道多选，不能替代新增多题型效果证据。
- [第三天剩余任务](DAY3_REMAINING_ZH.md)、[素材缺口反馈](day3_material_feedback_v1/REPORT_ZH.md)

下一步完成共同评估题的事实族边界与当前准入核验，再按既定规则冻结A/B训练池、共同DEV请求及对称训练资源。TEST只用于最终报告。第三天尚未全部完成；本次诊断不能证明反馈因果效果或训练收益。

封存结果保持原样。本次导航前版本保存在day3_solver187_v1/navigation_history/。
''')
    (WEEK/'DAY3_REMAINING_ZH.md').write_text('''# 第三天：当前结果与剩余任务

已完成训练侧全部187题的S0诊断：20roots、74多选、113排序、2688次作答。见[报告](day3_solver187_v1/REPORT_ZH.md)与[逐题结果](day3_solver187_v1/per_question.csv)。内容准入集合保持187题，未按分数删题，训练池未冻结；训练与DEV/TEST调用均为0。

## 下一步

1. 对固定历史184条共同回归候选，完成当前内容、完整原文、排序唯一性及训练/反馈事实族边界核验。保留历史曝光和原始划分；不按Solver分数筛选。184库存与历史QA绑定已核实，当前准入仍待完成。
2. 对通过的回归题统一冻结S0/A/B共同请求、映射、全文长度和模型身份；Gold6控制单列，TEST仅用于最终报告。按第一天既定规则准备A/B训练池及对称训练资源，冻结精确采样/选项重排/训练步数计划，再启动训练。
3. 新题族另报：当前仅4道多选、3roots、排序0，仍不足以支撑新增形式多题型结论；不能用历史回归或Train诊断冒充新增评估。

## 已完成产物

- [训练生产](day3_production_batch2_v1/REPORT_ZH.md)：20roots/240位，187接受、8内容HOLD、42素材缺额、3生成拦截，88配对槽位。
- [训练侧Solver诊断](day3_solver187_v1/REPORT_ZH.md)：全部187题，2688固定请求，逐题分层、A/B/题型/Gold及错选漏选、排序成对指标；不推断反馈因果收益。
- [历史评估库存核查](day3_eval_recovery_v2/REPORT_ZH.md)：107道Gold2–5多选、50道排序、27道Gold6控制，共184条/27roots，仍需当前准入。
- [新增共同评估v1](day3_eval_clearance_v1/REPORT_ZH.md)：186题位最终裁决与4题/72请求冻结，未调用Solver。
- [素材反馈](day3_material_feedback_v1/REPORT_ZH.md)：42缺额合成20个Train扩充请求，实际扩充0；旧事件3场＋新事件3场两路线与出题A/B是不同维度。

素材反馈、Solver诊断与训练效果分别报告。第三天尚未全部完成。
''')
    print(json.dumps({'status':'COMPLETE_VERIFIED','manifest':bind(HERE/'FINAL_MANIFEST.json'),
        'multi_select_exact':mc['question_equal_exact'],'ordering_exact':order['question_equal_exact']}))


if __name__=='__main__':
    main()
