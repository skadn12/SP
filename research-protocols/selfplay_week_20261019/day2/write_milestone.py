"""Report measured diagnosis work and explicitly retain unfinished Day 2 goals."""
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import json

BASE = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text())


def save(p, data):
    with Path(p).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def pct(x):
    return f'{100*x:.2f}%'


def main():
    validation = read(BASE / 'VALIDATION.json')
    assert validation['status'] == 'PASS'
    obs = read(BASE / 'analysis/results_02/OBSERVATIONS.json')
    direct = obs['runs']['direct']
    evidence = obs['runs']['evidence']
    hint = read(BASE / 'audit/RESULTS_AUDIT_HINT_PAIRED.json')
    costs = read(BASE / 'audit/RESULTS_AUDIT_RUNTIME_COSTS.json')
    trace = read(BASE / 'analysis/order_trace_consistency/ORDER_TRACE_CONSISTENCY.json')
    memory = read(BASE / 'diagnosis_exports/observed_six_v002/MEMORY_B_S0_OBSERVED_v002.json')
    assert len(memory['failure_cases']) == 6 and not memory['causes']
    now = datetime.now(timezone.utc)
    status = {
        'status': 'SOLVER_DIAGNOSIS_MILESTONE_COMPLETE', 'day2_all_goals_complete': False,
        'completed_at_utc': now.isoformat(), 'completed_at_beijing': now.astimezone(ZoneInfo('Asia/Shanghai')).isoformat(),
        'solver_model': 'S0_M_clean', 'namespace': 'train.feedback',
        'direct': direct['summary'], 'evidence_v2': evidence['summary'],
        'evidence_v1': {'status': 'INFRASTRUCTURE_GRAMMAR_FAILURE', 'planned': 39, 'attempted_submissions': 8, 'returned': 0, 'not_submitted': 31, 'ability_interpretation': False},
        'hint': {'questions': 6, 'calls': 90, 'minimum_three_root_threshold_met': hint['minimum_three_root_threshold_met'],
                 'transitions': hint['transitions'], 'positive_delta_roots': hint['positive_delta_distinct_roots']},
        'weakness_memory': {'failure_cases': 6, 'causes': 0, 'reproduced_mechanisms': 0, 'usable_for_B_targeted_generation': False},
        'fact_memory_pilot': {'prepared_roots': 6, 'planned_scenes': 54, 'actual_scenes': 0, 'candidate_quotes': 144, 'semantic_pass': 0, 'coverage': 'unknown', 'main_pipeline_context': 'full_history'},
        'new_generated_training_questions': 0, 'new_training_runs': 0,
        'cost': costs['totals'],
        'remaining_day2': ['Review initial six fact memories, implement receipt guards for each branch and run the frozen three-arm dialogue pilot.',
                           'Run pre-frozen mechanism-specific controlled tests and review sources before publishing reproduced causes.'],
        'preserved_failures': ['solver/evidence', 'diagnosis_exports/observed_six_v001', 'analysis/results_01'],
        'canonical_observations': 'analysis/results_02/OBSERVATIONS.json',
        'canonical_memory': 'diagnosis_exports/observed_six_v002/MEMORY_B_S0_OBSERVED_v002.json',
        'training_started': False, 'final_eval_performed': False,
    }
    table = []
    for typ, label in [('multi_select', '多选'), ('ordering', '排序')]:
        d, e = direct['by_type'][typ], evidence['by_type'][typ]
        table.append(f"| {label} | {d['question_count']} | {d['exact_calls']}/{d['planned_calls']}（{pct(d['question_equal_exact'])}） | {e['exact_calls']}/{e['planned_calls']}（{pct(e['question_equal_exact'])}） |")
    report = f'''# 第二天 Solver 诊断阶段报告

完成时间：{status['completed_at_beijing']}。**已完成第二天的 Solver 诊断阶段，第二天全部目标尚未完成。** 已实际运行 S0=M-clean 的582次直接作答、39次结构化证据作答与90次原文提示对照。6root事实memory三臂试验只完成规则/输入准备，54场对话尚未生成；没有新出训练题、没有训练。

## 当前发现

本批20个Train root、39题均从既有v125已准入的准确题目/来源版本按冻结哈希选取，没有按新成绩筛题。排序整题表现很低，但成对顺序仍有部分正确；简单增加原文提示未出现稳定改善。逐项引文输出自身也会出错，不能直接作为真实CoT或训练标签。

| 题型 | 题数 | 直接答案整题全对 | 证据＋答案整题全对 |
|---|---:|---:|---:|
{chr(10).join(table)}

直接答案582/582有效，多选每题6个展示排列×3次采样、排序12个排列×1次。题等权整体{pct(direct['summary']['question_equal_exact'])}，root等权{pct(direct['summary']['root_equal_exact'])}；多选和排序分母不同，不用简单汇总调用率代替题等权指标。排序成对正确为2472/3600={pct(direct['summary']['ordering']['pair_accuracy_fixed_denominator'])}。本批审核锚点全为跨场景，**同场景pair分母为0**，不能据此验证同场景排序假说。

多选Gold2/3/4/5分别为5/4/5/5题，直接答案全对率32.22%/72.22%/56.67%/72.22%。342次多选共TP1147、FP113、FN59；过选94/342。Gold2层误选76、过选60/90，值得后续用人物、状态与近邻干扰的最小对照进一步区分；这仍是观察，不是已确认原因。没有一次直接答案选择所有6项。

证据支路每题1次，输出上限1024；38/39格式有效，1例到长度上限且JSON不完整。它与直接答案的提示、输出长度、种子及采样次数不同，表中下降不能单独证明CoT普遍有害。排序证据支路成对正确219/300=73%，整题仍为0/20。

## 原文提示对照

选题规则在本轮direct评分前冻结：从至少一次有效答错的候选中按哈希各取最多3道多选/排序，6个不同root，保留原有全部排列、采样与种子。保留完整source，在问题后添加全部选项审核证据turn的并集（含反驳项），不提供正确标签或Gold关系。它使用作者证据选择提示，属于诊断干预，不是Gold-blind主评测。

| 范围 | 原直接答案 | 原文提示后 |
|---|---:|---:|
| 3道多选、54次 | 42/54（77.78%） | 41/54（75.93%） |
| 3道排序、36次 | 0/36 | 0/36 |

配对有3次错→对、4次对→错；仅1个root正向变化，未达到预定至少3个root正向支持门槛。不能将“证据定位困难”升级为已复现机制，也没有权重训练带来的改善。完整逐root结果见[独立配对审计](audit/RESULTS_AUDIT_HINT_PAIRED.json)。

## 证据记录与真实错误memory

预期234个逐项证据条目，228条可解析、6条因无效JSON缺失；161条非空引文在所报turn中逐字存在。字符串/位置吻合不是语义正确性证明，不同引用位置也不自动等于错误。逐条位置、原文speaker、scene、作者证据关系保存在[引用观察CSV](analysis/results_02/CITATION_OBSERVATIONS.csv)。

对20道排序进一步检查模型自己的引用位置与最终顺序：3道（112812、148314、130221）六项引文均逐字有效、位置唯一，经聚焦来源复核可支持相应事项；按这些位置排序与Gold一致，但模型最终答案不同。另16道至少有一处非逐字引文，1道存在引用位置重合，分别保留。这个现象提示应测试“从证据位置构造并校验顺序”的步骤，**尚未确认为内部机制或可迁移原因**。详见[排序轨迹核查](analysis/order_trace_consistency/REPORT_ZH.md)。

已导出6个不同Train root的6个错误案例，保留对应全部90次直接答案（含42次正确、48次错误），控制器11610项通过。快照里cause=0、reproduced=0；不能据此填满B的70%定向题位。作者证据与Solver证据明确分开，当前观察快照的diagnostic_trace为null，没有补造解释。

入口：[错误记录说明](diagnosis_exports/README_ZH.md)、[当前观察memory](diagnosis_exports/observed_six_v002/MEMORY_B_S0_OBSERVED_v002.json)。新增controller检查原始字节、作答先于本轮Gold评分、模型谱系、探针关联、改善/复发先后和1024上限；37项边界自测通过。

## 接口修正、成本与审核

原证据v1被后端拒绝JSON Schema的uniqueItems，8次提交尝试、0返回，其余31次未发送；39条失败/缺失记录完整保留，不能解释为模型能力0分。v2只从发给后端的schema副本移除不支持的键，原严格schema、答案唯一性与证据label完整性仍在评分时校验。39个请求、prompt、mapping、种子、模型、Gold和输出预算与v1完全相同，真实grammar编译通过后才重跑。旧solver及失败记录未改。

分析报告v1有40处raw_binding错误指向source文件，独立审计发现后新增observe_v2和results_02修正；原报告及失败审计保留。修正不改变题目、Gold、模型回答或评分，最终以results_02为准。evidence原始metadata中继承的API探测标签含choice字样，实际提交为JSON约束，详见[日志语义说明](audit/RESULTS_AUDIT_EVIDENCE_API_LABEL_NOTE.json)。

总账：750条计划终态记录、719次提交尝试、711份模型返回、31未发送、8次后端失败usage未知。已知输入5,353,449 tokens、输出23,179，总计5,376,628；未知usage不能记成已证实0。四次live累计298.76秒（包括模型加载62.89秒），其中直接答案172.19秒、失败v1为21.40秒、证据v2为61.26秒、提示对照43.92秒。首次Solver启动到最后原始记录封存约766.44秒，另有规则准备与审核人工/agent时间，不能把live秒数称整个工作耗时。无货币账单。详见[总成本审计](audit/RESULTS_AUDIT_RUNTIME_COSTS.json)。

所有评分在对应完整rawfreeze之后单独读取Gold；“首次读Gold”限定该评分进程，父会话与历史准备有Gold暴露。全部root为历史Train、仅用于反馈，不是全新测试或本周最终迁移评估。同一模型/共享文件系统内部审核，不宣称外部盲审或OS隔离。

## 下一步

事实memory已冻结6个已准入前缀（1095 turns）、144条待审核逐字引语、三臂54场计划、质量/成本采纳阈值和维护审核预算；语义通过0、覆盖率未知。下一步先审初始memory，再实施每分支receipt与来源门槛后运行三臂；未达门槛继续全文。见[三臂准备](fact_memory/README_ZH.md)。

机制方面，先对证据位置→最终顺序不一致设计事前冻结的最小对照，并检查多选误选来自哪些人物/状态/数值关系；至少3个Train root复现且受控探针支持后再发布cause，随后B才能按真实原因定向出题。旧题修复与后续生成、训练仍需独立版本和准入，不依据本轮预测改Gold。

本阶段父级验收{validation['check_count']}项通过；原始direct、evidence失败、v2与hint均有独立重算审计。完整文件hash见COMPLETE_MANIFEST.json，封存后下阶段使用新目录。
'''
    with (BASE / 'REPORT_ZH.md').open('x') as f:
        f.write(report)
    save(BASE / 'FINAL_STATUS.json', status)
    print(json.dumps({'status': status['status'], 'day2_all_goals_complete': False,
                      'returned_outputs': costs['totals']['returned_outputs'], 'causes': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
