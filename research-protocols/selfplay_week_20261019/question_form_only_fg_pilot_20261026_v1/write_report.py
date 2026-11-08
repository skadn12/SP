#!/usr/bin/env python3
"""Render the frozen experiment's actual denominators and readable questions."""
from pathlib import Path
import collections
import json

HERE = Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text())
def write(p,s):
    with Path(p).open('x') as f: f.write(s)
def pct(x): return '—' if x is None else f'{x*100:.2f}%'
def num(x): return '—' if x is None else f'{x:.2f}'
def exact(s): return f"{s['exact_correct']}/{s['requests']} ({pct(s['exact_accuracy'])})"
def link(name): return f'[{name}]({HERE/name})'

def main():
    a=read(HERE/'execution/ANALYSIS_SUMMARY.json')
    c=read(HERE/'CANDIDATES_PRIVATE.json')
    reviews=read(HERE/'CONTENT_REVIEW_AI.json')['reviews']
    scores=read(HERE/'execution/SCORES.json')['records']
    den=a['denominators']
    fg=a['primary_comparison']
    new_full=next(r for r in fg if r['phase']=='new' and r['condition']=='FULL')
    combined={r['phase']:r for r in a['summaries'] if r['condition']=='FULL' and r['group']=='MC2_5_PRIMARY'}
    lines=['# 原4 roots：具体题面要求下的 F/G 复验','',
        f"计划40道；最终接纳 {den['admitted_questions']} 道，实际提交 {den['new_submitted_requests']} 次新 Solver 请求、返回 {den['new_returned_requests']} 次，格式有效 {den['valid_new_responses']} 次。每题两种排列，FULL 与 QUESTION_ONLY 分开。旧题只复算历史80次回答，未重新调用模型。",'',
        '## 本轮判断','',
        f"新题面在原Solver上可执行。主指标中F筛选分 {num(new_full['F']['J_score'])}、G {num(new_full['G']['J_score'])}，F−G为 {num(new_full['F_minus_G']['J_score'])} 分；{'F题整体更容易，本轮不支持反馈生成更难题的主假设' if new_full['F_minus_G']['J_score']>0 else 'F题整体得分较低，本轮呈现反馈题更难的描述性信号'}。这是此四roots与此批题的结果，不能直接替代此前19roots的统计。",'',
        '**执行范围偏差：本轮保留了原任务计划，但重新生成题目仍改变了部分选项命题与干扰项；MC5八组旧新题均有此类变化。因而没有严格满足“保持原题内容，只改固定题目形式”的要求，不能把本轮当作纯形式对照。**','',
        f"难度保持尚未得到证明：旧/新非MC6完全正确分别为 {exact(combined['old'])}、{exact(combined['new'])}，原筛选分从 {num(combined['old']['J_score'])} 变为 {num(combined['new']['J_score'])}。完全正确率接近也不能掩盖部分得分变化；而新题并非保持全部命题与干扰项的同题改写，分差不能单独归因于题面形式。",'',
        '## 实际分数','',
        '下表仅包括接纳且F/G配对完整的 MC2..MC5。J 是原筛选分 `TP/g − FP/(n−g)` 的均值乘100，保留负数；不是正确率。F/G题目不同，负的 F−G 表示本轮F题得分更低、对固定Solver更难。','',
        '| 条件 | F 完全正确 | G 完全正确 | F 原筛选分J | G 原筛选分J | F−G J |',
        '|---|---:|---:|---:|---:|---:|']
    names={('old','FULL'):'旧题 FULL（相同接纳槽）',('new','FULL'):'新题 FULL',('new','QUESTION_ONLY'):'新题无对话诊断'}
    for r in fg:
        lines.append(f"| {names[(r['phase'],r['condition'])]} | {exact(r['F'])} | {exact(r['G'])} | {num(r['F']['J_score'])} | {num(r['G']['J_score'])} | {num(r['F_minus_G']['J_score'])} |")
    lines += ['', '## 操作差异：新题 FULL','',
        '| 槽 | 操作 | F 完全正确 | G 完全正确 | F J | G J | F−G J |',
        '|---|---|---:|---:|---:|---:|---:|']
    ops={'MC2':'数量关系','MC3':'约束合取','MC4':'证据选择','MC5':'明示因果'}
    slot_rows=[]
    for r in a['F_minus_G_by_condition_root_slot']:
        if (r['scope'],r['phase'],r['condition'],r['group'],r['dimension'])==('admitted_slot_matched','new','FULL','MC2_5_PRIMARY','slot_id'):
            slot_rows.append(r)
            lines.append(f"| {r['value']} | {ops[r['value']]} | {exact(r['F'])} | {exact(r['G'])} | {num(r['F']['J_score'])} | {num(r['G']['J_score'])} | {num(r['F_minus_G']['J_score'])} |")
    if slot_rows:
        driver=max(slot_rows,key=lambda r:abs(r['F_minus_G']['J_score']))
        lines += ['',f"主差距最大的是 {driver['value']}（{ops[driver['value']]}）：F−G {num(driver['F_minus_G']['J_score'])} 分。操作间方向并不一致，不能用总均值概括为所有能力上的统一效果；这些分组是对既定主结果的解释，没有删去任何槽。",'']
    if (HERE/'MC5_OLD_NEW_DIAGNOSTIC_AI.json').exists():
        diagnostic=read(HERE/'MC5_OLD_NEW_DIAGNOSTIC_AI.json')
        lines += ['### MC5旧新题目的事后核对','',
            '该核对在查看实际分数后进行，用于解释可能的内容差异，不作为事前筛选或因果证据。']
        lines += [''] + ['- '+text for text in diagnostic['report_summary']] + ['']
    lines += ['', '## 各 root：新题 FULL，MC2..MC5','',
        '| root | F 完全正确 | G 完全正确 | F J | G J | F−G J |',
        '|---|---:|---:|---:|---:|---:|']
    for r in a['F_minus_G_by_condition_root_slot']:
        if (r['scope'],r['phase'],r['condition'],r['group'],r['dimension'])==('admitted_slot_matched','new','FULL','MC2_5_PRIMARY','root_id'):
            lines.append(f"| {r['value'].split('-')[-1]} | {exact(r['F'])} | {exact(r['G'])} | {num(r['F']['J_score'])} | {num(r['G']['J_score'])} | {num(r['F_minus_G']['J_score'])} |")
    lines += ['', '## 各 root / 任务槽：新题 FULL','',
        '| root | 槽 | F 完全正确 | G 完全正确 | F J | G J | F−G J |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in a['F_minus_G_by_condition_root_slot']:
        if (r['scope'],r['phase'],r['condition'],r['group'],r['dimension'])==('admitted_slot_matched','new','FULL','MC2_5_PRIMARY','root_slot'):
            lines.append(f"| {r['root_id'].split('-')[-1]} | {r['slot_id']} | {exact(r['F'])} | {exact(r['G'])} | {num(r['F']['J_score'])} | {num(r['G']['J_score'])} | {num(r['F_minus_G']['J_score'])} |")
    lines += ['', '## MC6 全选控制','',
        'MC6 没有错误项，J 不适用，不混入主指标。','',
        '| 条件 | F 完全正确 | G 完全正确 |', '|---|---:|---:|']
    for r in a['F_minus_G_by_condition_root_slot']:
        if r['scope']=='admitted_slot_matched' and r['group']=='MC6_CONTROL' and r['dimension']=='ALL':
            lines.append(f"| {names[(r['phase'],r['condition'])]} | {exact(r['F'])} | {exact(r['G'])} |")
    diag=[r for r in a['question_only_diagnostic_pairs'] if r['slot_id']!='MC6']
    counts=collections.Counter((r['FULL_exact'],r['QUESTION_ONLY_exact']) for r in diag)
    qo=next(r for r in fg if r['phase']=='new' and r['condition']=='QUESTION_ONLY')
    lines += ['', '## 无对话诊断与排列稳定性','',
        f"非MC6的 {len(diag)} 个相同题/排列条件对中：两条件均对 {counts[(True,True)]}；仅FULL对 {counts[(True,False)]}；仅无对话对 {counts[(False,True)]}；两条件均错 {counts[(False,False)]}。无对话条件使用独立best-effort指令和空引用schema，是上下文消融诊断，不是纯语言形式的因果检验。",'',
        f"无对话时F−G仍为 {num(qo['F_minus_G']['J_score'])} 分，与FULL的 {num(new_full['F_minus_G']['J_score'])} 分接近。FULL较各自无对话条件的J增加：F {num(new_full['F']['J_score']-qo['F']['J_score'])}、G {num(new_full['G']['J_score']-qo['G']['J_score'])}。本批组间分差已出现在无对话条件，不能全部解释为读取对话后的能力差异；上述条件变化也不允许把相减结果直接称为反馈的因果效果。",'']
    for phase,condition in [('old','FULL'),('new','FULL'),('new','QUESTION_ONLY')]:
        rows=[r for r in a['permutation_consistency'] if r['phase']==phase and r['condition']==condition and not r['candidate_id'].endswith('MC6')]
        label='旧题FULL（全部历史非MC6）' if phase=='old' else names[(phase,condition)]
        lines.append(f"- {label}：{sum(r['same_canonical_prediction'] for r in rows)}/{len(rows)} 题两排列选择一致；{sum(r['both_exact'] for r in rows)}/{len(rows)} 题两排列均完全正确。")
    dec=collections.Counter(r['decision'] for r in reviews)
    held=[r for r in a['admission_ledger'] if not r['execution_admitted']]
    lines += ['', '## 内容审核及未纳入槽','',
        f"40道初稿全部通过原文引文/格式检查；独立上下文AI先盲推答案并冻结，再对照作者Gold。最终内容决定：{dict(dec)}。这不是人工裁决。下面保留未执行槽及配对原因，未用别题补位。",'']
    if held:
        for r in held:
            lines.append(f"- `{r['candidate_id']}`：{r['review_decision']}；{r.get('execution_hold_reason','内容审核未通过')}。")
    else: lines.append('所有20个F/G槽位对均接纳。')
    initial_dec=read(HERE/'CONTENT_REVIEW_INITIAL_AUDIT.json')['decisions']
    lines += ['',f"初稿内容决定为 {initial_dec}。结合原计划核对，共12道作唯一一次修订、28道结构化不变。8道证据题将公开完整引文改为同一原始片段的精确定位，48个证据单元逐项核对边界；其余修订只澄清事件阶段、信息施事或补回原计划拒绝项。修订后再次盲推答案，全部记录与初稿独立键均保留。",'']
    if (HERE/'FEEDBACK_ERROR_MATCHING_AI.json').exists():
        e=read(HERE/'FEEDBACK_ERROR_MATCHING_AI.json')
        er=e['records'];classes=collections.Counter(r['match_class'] for r in er if r['slot_id']!='MC6')
        control=collections.Counter(r['match_class'] for r in er if r['slot_id']=='MC6')
        lines += ['## 预定反馈错误是否实际出现','',
            '在Solver启动前冻结错误匹配规则，AI审查覆盖所有被接纳F题的FULL最终JSON。仅依据最终输出中的选择、说明与证据；整题答错不自动算命中。','',
            '| 审查类别 | MC2..MC5回答数 | MC6回答数 |',
            '|---|---:|---:|']
        class_names={'OBSERVED_EXPLANATION_MATCH':'说明明确出现预定错误','OBSERVED_SELECTION_MATCH':'仅选择命中，说明未证实','NO_TARGET_ERROR_OBSERVED':'未观察到预定错误','ERROR_OTHER_RELATION':'出现其他关系错误'}
        for key,label in class_names.items():
            lines.append(f'| {label} | {classes[key]} | {control[key]} |')
        lines += ['',f"分母分别为 {sum(classes.values())} 次主要回答和 {sum(control.values())} 次控制回答。说明命中与仅选择命中分开，重复排列不当作独立题。",'']
        by_candidate=collections.defaultdict(list)
        for r in er:
            if r['slot_id']!='MC6':by_candidate[r['candidate_id']].append(r)
        both_explanation=sum(len(v)==2 and all(r['match_class']=='OBSERVED_EXPLANATION_MATCH' for r in v) for v in by_candidate.values())
        both_observed=sum(len(v)==2 and all(r['match_class'] in ('OBSERVED_EXPLANATION_MATCH','OBSERVED_SELECTION_MATCH') for r in v) for v in by_candidate.values())
        lines += [f"{len(by_candidate)} 道主要F题中，两排列均明确说明命中 {both_explanation} 道；两排列均至少出现目标选择或说明命中 {both_observed} 道。这描述目标陷阱在该固定checkpoint上的可观察重复性，不能据此宣称训练有效。",'']
        lines += [
            '例如，225575数量题的两种排列均把50分钟写为0.50小时。229686数量题的一种排列正确写出5/2=2.5，却选中相反比例的选项，因此只记选择命中；另一排列明确写出“5 is not 2.5 times 2.”，才记说明命中。218519因果题还有一次整题全对、但某个被拒选项的证据说明出现预定错误的情况。因此，整题准确率与具体错误审查回答的是不同问题。','']
    lines += ['', '## 可以怎样解释','',
        '本轮保留原四roots、完整来源及warnings、F公开反馈/G无反馈、两臂原固定任务计划、操作和Gold数量、模型与采样、显示排列、原筛选评分。只向作者合同追加具体任务题面的要求；没有扩对话、改日历或引用渲染、训练、按Solver分数改题。', '',
        '新题由不同作者上下文重新生成，保持原任务计划并不等于保持原题的全部事实命题和干扰项。因此旧新分差只能描述这一批题，不能证明纯题面形式的因果作用或“难度没有变化”。F/G共同使用已被反馈塑形的来源；四个已暴露roots、同一checkpoint及重复排列也不支持训练收益或未见root泛化。', '',
        '本轮已证明格式可执行，但不足以据此扩展或清洗全量数据。若下一轮要严格检验用户要求的“只改固定题目形式”，应先锁定旧题全部选项命题、Gold、推理关系和干扰项，仅改题干及等义表述，逐题核对语义等价；再用事先固定的难度等价标准比较同题的新旧版本。该后续同题检验尚未执行。', '',
        '## 可复核文件','',
        '- '+link('QUESTIONS_VIEW.md'),
        '- '+link('CONTENT_REVIEW_AI.json'),
        '- '+link('PLAN_PRESERVATION_AUDIT.json'),
        '- '+link('REVISION1_PLAN_PRESERVATION_AUDIT.json'),
        '- '+link('CONTENT_REVIEW_INITIAL_AI.json'),
        '- '+link('execution/ANALYSIS_SUMMARY.json'),
        '- '+link('execution/SCORES.json'),
        '- '+link('execution/RAW_RESPONSE_FREEZE.json'),
        '- '+link('execution/BASELINE_J_RECOMPUTED.json'),
        '- '+link('CONTENT_FREEZE.json')]
    if (HERE/'FEEDBACK_ERROR_MATCHING_AI.json').exists():
        lines.append('- '+link('FEEDBACK_ERROR_MATCHING_AI.json'))
    if (HERE/'MC5_OLD_NEW_DIAGNOSTIC_AI.json').exists():
        lines.append('- '+link('MC5_OLD_NEW_DIAGNOSTIC_AI.json'))
    if (HERE/'FINAL_INTEGRITY_AUDIT.json').exists():
        lines.append('- '+link('FINAL_INTEGRITY_AUDIT.json'))
        integrity=read(HERE/'FINAL_INTEGRITY_AUDIT.json')
        lines += ['',f"独立完整性核查：{integrity['status']}，{integrity['check_count']} 项检查、{integrity['failed_check_count']} 项失败；从冻结原始返回及映射重新计算全部160次新回答和80次旧回答的完全正确及J分，并核对审查合并记录。该检查验证文件与计算一致性，不代替题目语义等价或难度等价检验。"]
    write(HERE/'REPORT_ZH.md','\n'.join(lines)+'\n')
    admitted={r['candidate_id'] for r in a['admission_ledger'] if r['execution_admitted']}
    q=['# 本轮完整题目','', '以下是canonical A–F题面；实际Solver每题显示两次程序排列。未接纳题仍保留并标注，不计作Solver错误。Gold与私有作者分析见 '+link('CANDIDATES_PRIVATE.json')+'。','', '## 对话来源','']
    sources={row['root_id']:row['source_binding']['path'] for row in c}
    q += [f'- {root}：[完整原始对话]({path})' for root,path in sorted(sources.items())]
    q.append('')
    rev={r['candidate_id']:r for r in reviews}
    for row in sorted(c,key=lambda r:(r['root_id'],r['slot_id'],r['arm'])):
        cid=row['candidate_id'];q += [f'## {cid}','',f"内容决定：{rev[cid]['decision']}；{'已执行' if cid in admitted else '未执行'}。",'',row['question'],'']
        q += [f"- **{label}** {text}" for label,text in row['options'].items()]
        q.append('')
    write(HERE/'QUESTIONS_VIEW.md','\n'.join(q)+'\n')
    print(json.dumps({'report':str(HERE/'REPORT_ZH.md'),'question_view':str(HERE/'QUESTIONS_VIEW.md'),'denominators':den},ensure_ascii=False))

if __name__=='__main__': main()
