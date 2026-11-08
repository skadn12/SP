#!/usr/bin/env python3
"""Render frozen results without changing content or scoring artifacts."""
from pathlib import Path
from collections import Counter
import datetime
import hashlib
import json

HERE = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text())

def binding(path):
    data = Path(path).read_bytes()
    return {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

def pct(value):
    return '不适用' if value is None else f'{value:.2f}'

def result(row):
    return f"{row['exact_correct']}/{row['requests']}（{100 * row['exact_accuracy']:.2f}%）" if row['requests'] else '0/0'

def main():
    candidates = read(HERE / 'CANDIDATES_PRIVATE.json')
    summary = read(HERE / 'execution/ANALYSIS_SUMMARY.json')
    reviews = read(HERE / 'CONTENT_REVIEW_AI.json')['reviews']
    scores = read(HERE / 'execution/SCORES.json')['records']
    questions = ['# 本轮全部新题', '', '40个计划槽均保留。Gold及逐项证据位于私有候选文件；本页面不显示Gold。', '']
    for c in candidates:
        questions += [f"## {c['candidate_id']}", '', f"来源：[完整对话]({c['source_binding']['path']})；操作：`{c['task_operation']}`", '', c['question'], '']
        questions += [f"- {k}. {v}" for k, v in c['options'].items()]
        questions += ['']
    (HERE / 'QUESTIONS_VIEW.md').write_text('\n'.join(questions))
    primary = [r for r in summary['F_minus_G_by_condition_root_slot'] if r['scope'] == 'admitted_slot_matched' and r['phase'] == 'new' and r['group'] == 'MC2_5_PRIMARY']
    full = next(r for r in primary if r['condition'] == 'FULL' and r['dimension'] == 'ALL')
    qo = next(r for r in primary if r['condition'] == 'QUESTION_ONLY' and r['dimension'] == 'ALL')
    delta = full['F_minus_G']['J_score']
    interpretation = ('F的J较低，本轮反馈指导生成的题更难。' if delta < 0 else 'F的J较高，本轮反馈指导生成的题反而更容易。' if delta > 0 else '两组J相同，本轮未观察到整体难度差。')
    report = ['# 自然问法：原反馈流程实测', '', f"完成{len(candidates)}个计划候选，实际新调用{summary['denominators']['new_submitted_requests']}次，返回{summary['denominators']['new_returned_requests']}次，格式有效{summary['denominators']['valid_new_responses']}次。{interpretation}", '',
              '## 主要结果：MC2–MC5', '', '| 条件 | F完全正确 | G完全正确 | F原J×100 | G原J×100 | F−G J分 |', '|---|---:|---:|---:|---:|---:|']
    for label, r in [('FULL完整对话', full), ('QUESTION_ONLY无对话诊断', qo)]:
        report += [f"| {label} | {result(r['F'])} | {result(r['G'])} | {pct(r['F']['J_score'])} | {pct(r['G']['J_score'])} | {pct(r['F_minus_G']['J_score'])} |"]
    report += ['', f"J=TP/g−FP/(n−g)，负分保留。完全正确率要求整个选择集合完全一致。每组计划主要题16题，实际F {full['F']['semantic_questions']}题、G {full['G']['semantic_questions']}题，每题2次排列；排列为重复测量。无效响应计入完全正确率分母，J仅计有效适用响应，具体分母见机器结果。", '', '## 每个root', '', '| root | F完全正确 | G完全正确 | F J×100 | G J×100 | F−G |', '|---|---:|---:|---:|---:|---:|']
    for r in primary:
        if r['condition'] == 'FULL' and r['dimension'] == 'root_id':
            report += [f"| {r['value']} | {result(r['F'])} | {result(r['G'])} | {pct(r['F']['J_score'])} | {pct(r['G']['J_score'])} | {pct(r['F_minus_G']['J_score'])} |"]
    report += ['', '## 每种操作', '', '| 槽 | F完全正确 | G完全正确 | F J×100 | G J×100 | F−G |', '|---|---:|---:|---:|---:|---:|']
    for r in primary:
        if r['condition'] == 'FULL' and r['dimension'] == 'slot_id':
            report += [f"| {r['value']} | {result(r['F'])} | {result(r['G'])} | {pct(r['F']['J_score'])} | {pct(r['G']['J_score'])} | {pct(r['F_minus_G']['J_score'])} |"]
    control = next(r for r in summary['F_minus_G_by_condition_root_slot'] if r['scope'] == 'admitted_slot_matched' and r['phase'] == 'new' and r['condition'] == 'FULL' and r['group'] == 'MC6_CONTROL' and r['dimension'] == 'ALL')
    report += ['', '## MC6控制', '', f"FULL：F {result(control['F'])}，G {result(control['G'])}。六项全真，无负项，J不适用，未混入主要J。", '', '## 审核与边界', '', f"独立AI内容审核：{dict(Counter(r['decision'] for r in reviews))}。先独立盲推Gold再与作者核对；初稿、一次修订记录及冻结哈希保留。未按Solver成绩重写或替换题目。", '', '这批沿用原反馈路线，但采用新计划和新题实例；它不是相同题目只改措辞的受控实验，不能据此声称改问法不影响难度。F/G两组共用自然问法要求，F使用原公开反馈，G不接收反馈。四个roots均已暴露，来源也已在早期反馈流程中扩写；隔离依赖不同上下文与读写约束，共享文件系统。', '', '本轮只检验固定checkpoint面对新题的表现，没有训练，不能证明训练可修复弱点，也没有新的排序结果。错误机制命中须另读目标审查，不能由选错直接推断推理原因。', '', '## 文件', '', '- [全部题目与完整来源链接](QUESTIONS_VIEW.md)', '- [机器汇总、全部root/槽及历史基线重计分比较](execution/ANALYSIS_SUMMARY.json)', '- [逐响应分数](execution/SCORES.json)', '- [封存原始模型响应](execution/raw/)', '- [内容审核](CONTENT_REVIEW_AI.json)', '- [内容冻结](CONTENT_FREEZE.json)', '- [反馈目标命中审查](FEEDBACK_ERROR_MATCHING_AI.json)', '- [完整性审计](FINAL_INTEGRITY_AUDIT.json)', '']
    matching_path = HERE / 'FEEDBACK_ERROR_MATCHING_AI.json'
    if matching_path.exists():
        matching = read(matching_path)
        records = matching['records']
        report += ['', '## 预定反馈错误审查', '', '逐一审查全部F组FULL回答，包括全对回答；明确说明命中与仅选择命中分别统计，不由低分推断错误机制。两次排列属于同题重复测量。', '', '| root | 题组 | 回答数 | 说明命中 | 仅选择命中 | 其他错误 | 未见目标错误 | 不确定/无效 |', '|---|---|---:|---:|---:|---:|---:|---:|']
        for root in sorted({r['root_id'] for r in records}):
            for group, slots in [('MC2–5', {'MC2','MC3','MC4','MC5'}), ('MC6控制', {'MC6'})]:
                subset = [r for r in records if r['root_id'] == root and r['slot_id'] in slots]
                count = Counter(r['match_class'] for r in subset)
                report += [f"| {root} | {group} | {len(subset)} | {count['OBSERVED_EXPLANATION_MATCH']} | {count['OBSERVED_SELECTION_MATCH']} | {count['ERROR_OTHER_RELATION']} | {count['NO_TARGET_ERROR_OBSERVED']} | {count['INDETERMINATE']+count['INVALID_RESPONSE']} |"]
        report += ['', '同题两种排列的共同命中与详细对应见 FEEDBACK_ERROR_MATCHING_AI.json；以上计数是可观察行为描述，未证明稳定机制或训练可修复性。']
    (HERE / 'REPORT_ZH.md').write_text('\n'.join(report))
    readme = (HERE / 'README_ZH.md').read_text().replace('执行状态：作者已启动；初稿、独立Gold审核、内容冻结和Solver推理按序进行。', f"执行状态：已完成；{len(candidates)}个候选，{len(scores)}次新Solver请求。详见 [REPORT_ZH.md](REPORT_ZH.md)。")
    (HERE / 'README_ZH.md').write_text(readme)
    print(json.dumps({'report': str(HERE / 'REPORT_ZH.md'), 'primary_F_minus_G_J': delta}, ensure_ascii=False))

if __name__ == '__main__':
    main()
