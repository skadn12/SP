"""Assemble Day2 closeout from the audited, fixed-denominator outputs."""
from pathlib import Path
import json,datetime,hashlib
W=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def write(p,value):
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
    diagnosis=read(W/'diagnosis/DIAGNOSIS_CLOSEOUT.json')
    scores=read(W/'pilot/SCORES.json');complete=read(W/'pilot/COMPLETE.json')
    audit=read(W/'audit/RESULTS_AUDIT.json')
    assert audit['status']=='PASS' and complete['status']=='ALL_54_SLOTS_TERMINAL'
    assert scores['status']=='COMPLETE_FIXED_DENOMINATOR' and len(scores['receipts'])==54
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    generated=len(list((W/'pilot/branches').glob('*/scene_*/realizer/RESPONSE.json')))
    status={'schema_version':'day2_completion_v1','status':'DAY2_COMPLETE_WITH_LIMITED_AND_NEGATIVE_RESULTS',
        'completed_at_utc':now,'day2_all_goals_complete':True,
        'goals':{'s0_direct_and_diagnostic_runs':True,'cross_root_failure_and_controlled_tests':True,
            'initial_fact_memory_reviews':True,'six_root_three_arm_pilot_closed':True,'independent_results_audit':True},
        'completion_semantics':'Planned work and fixed-denominator experiment closed; does not require a positive effect or a confirmed internal cause.',
        'solver':{'model':'S0_M_clean','questions':39,'train_roots':20,
            'direct_calls':582,'multi_select_exact':{'numerator':197,'denominator':342},
            'ordering_exact':{'numerator':2,'denominator':240}},
        'diagnosis':{'confirmed_internal_causes':diagnosis['completion_assessment']['confirmed_internal_mechanisms_count'],
            'B_seventy_percent_verified_cause_targeting_ready':False,'new_cause_memory_entries':0},
        'fact_memory':{'roots':6,'arms':3,'planned_scene_slots':54,'terminal_scene_slots':54,
            'generated_candidate_scenes':generated,'accepted_by_arm':{a:x['accepted_initial_scenes'] for a,x in scores['arms'].items()},
            'quality_gates_all_pass':scores['quality_gates_all_pass'],'cost_status':scores['cost_status'],
            'memory_adopted':scores['memory_adopted'],'main_pipeline_context':scores['selected_context_policy']},
        'new_training_questions':0,'new_training_runs':0,
        'historical_day2_status_preserved':bind(W.parent/'day2/FINAL_STATUS.json'),
        'evidence':{k:bind(W/p) for k,p in {'diagnosis':'diagnosis/DIAGNOSIS_CLOSEOUT.json',
            'scores':'pilot/SCORES.json','results_audit':'audit/RESULTS_AUDIT.json',
            'execution_freeze':'EXECUTION_FREEZE.json','limitations':'EXECUTION_AMENDMENT.json'}.items()}}
    write(W/'FINAL_STATUS.json',status)
    rows=['# 第二天任务完成报告','',
        '第二天的诊断和事实记忆对照均已执行并完成审核。结果包括负结果与有限支持；没有新训练，也不把实验完成解释为弱点已修复。','',
        '## Solver与诊断','',
        '- 固定S0（M-clean），39题、20个训练root，直接作答582次。多选197/342全对（57.60%）；排序2/240全对（0.83%）。',
        '- 跨root误选、漏选和排序错误已记录。审核证据提示、位置核对指令、固定引文加位置等受控检验已收口。',
        '- 可确认内部原因仍为0；位置提示在所选子集改善部分顺序关系，未稳定修复整题。B组“70%按已验证原因定向”的前提尚未满足。','',
        '## 六root事实记忆对照','',
        '每组固定18个计划场景；失败后停止的后续场景仍计入分母。后两场沿各自接受分支继续，不当作相同输入的独立重复。','',
        '| 上下文 | 初审通过/18 | 完整通过分支/6 | 旧历史联系场景/18 |','|---|---:|---:|---:|']
    labels={'recent2':'最近两场','recent2_plus_prefix_memory':'最近两场＋事实记忆','full_history':'完整历史'}
    for a in ('recent2','recent2_plus_prefix_memory','full_history'):
        x=scores['arms'][a];rows.append(f"| {labels[a]} | {x['accepted_initial_scenes']}/18 | {x['completed_accepted_branches']}/6 | {x['consistent_old_history_linkage_scene_count']}/18 |")
    rows+=['',f'54个计划槽位全部留有终态记录，实际生成{generated}个候选场景；缺失和失败没有用重试替换。',
        '',f"记忆质量门槛整体：{'通过' if scores['quality_gates_all_pass'] else '未通过'}；成本结论：{scores['cost_status']}；主流程使用：{scores['selected_context_policy']}。",'',
        '初始记忆为6份内部角色审核记录，实际独立调用usage与模型时延未知。第三方网关未证实执行输出token上限，并加入不可见指令；这些限制已在生成前冻结。保留每次API实际报告usage与耗时，不把中途停止的低用量称为效率。本次仅作描述性小样本运行，不能声称严格等token预算、显著性或memory因果收益。','',
        '## 交付与后续','',
        '- [事实记忆完整结果](pilot/REPORT_ZH.md)及[54槽位评分](pilot/SCORES.json)。',
        '- [诊断报告](diagnosis/REPORT_ZH.md)及[结构化原因/观察清单](diagnosis/DIAGNOSIS_CLOSEOUT.json)。',
        '- [独立结果审核](audit/RESULTS_AUDIT.json)、[执行冻结](EXECUTION_FREEZE.json)、[事前限制声明](EXECUTION_AMENDMENT.json)。',
        '- [全部分支原始输入、对话、审核和费用记录](pilot/branches/)；[初始事实记忆](initial_memory/INITIAL_MEMORY_INDEX.json)。',
        '- [最终状态](FINAL_STATUS.json)。旧day2目录保持原封存状态；本目录是其后续完成记录。','',
        '下一阶段可以继续来源审核与多样性出题准备；若要执行真实诊断反馈A/B，需先冻结可检验的训练侧操作性弱点及验证规则，或明确修改B组的原因准入协议。不能将已观察到的错误直接改名为已确认内部原因，也不能据本轮宣称selfplay训练有效。','']
    with (W/'REPORT_ZH.md').open('x') as f:f.write('\n'.join(rows))
    with (W/'README_ZH.md').open('x') as f:f.write('# Day2 完成记录\n\n入口：[完成报告](REPORT_ZH.md)、[最终状态](FINAL_STATUS.json)。\n\n本目录补齐旧day2封存后尚缺的诊断收口与6root事实记忆三臂试验。保留所有失败和缺失；未进行训练。完整文件及SHA256见COMPLETE_MANIFEST.json。\n')
    progress={'status':status['status'],'updated_at_utc':now,'day2_all_goals_complete':True,
        'terminal_scene_slots':54,'generated_candidate_scenes':generated,
        'accepted_by_arm':status['fact_memory']['accepted_by_arm'],'main_pipeline_context':scores['selected_context_policy']}
    # PROGRESS is a mutable progress pointer until this closeout is sealed.
    (W/'PROGRESS.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(status['fact_memory'],ensure_ascii=False))
if __name__=='__main__':main()
