"""Report fixed-denominator production outcomes, not heldout training gains."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCH = HERE.parent / 'h4_codex_scaled_production_v107'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def bind(p):
    return {'path': str(p), 'sha256': sha(p)}

def main():
    audit = read(HERE / 'AUDIT_RESULTS.json')
    assert audit['status'] == 'VALID' and audit['data_errors'] == []
    accounting = read(BATCH / 'FINAL_BATCH_ACCOUNTING.json')
    plan = read(BATCH / 'PRECALL_PLAN.json')
    package = read(HERE / 'training_candidate_package/PACKAGE_REPORT.json')
    roots = accounting['roots']
    assert len(roots) == 12
    selected = defaultdict(Counter)
    for q in read(HERE / 'training_candidate_package/FRONTIER_MANIFEST.json')['candidates']:
        selected[q['root_id']][q['question_type']] += 1
    source_failures = []
    intervals = []
    for r in roots:
        folder = BATCH / 'roots' / ('root' + r['root_id'].rsplit('-', 1)[1])
        review = read(folder / 'SOURCE_REVIEW.json')
        if review['decision'] != 'READY':
            source_failures.append({'root_id': r['root_id'], 'findings': review['findings']})
        receipt = read(folder / 'AUTHOR_RECEIPT.json')
        intervals.append((datetime.fromisoformat(receipt['started_at_utc'].replace('Z', '+00:00')),
                          datetime.fromisoformat(receipt['finished_at_utc'].replace('Z', '+00:00'))))
    expansion_window = (max(e for _, e in intervals) - min(s for s, _ in intervals)).total_seconds()
    calls = read(HERE / 'SCORED_CALLS.json')['calls']
    groups = {}
    predicates = {f'gold{g}': (lambda c, g=g: c['question_type'] == 'multi_select' and c['gold_count'] == g) for g in range(2, 7)}
    predicates.update({'gold2_5': lambda c: c['question_type'] == 'multi_select' and 2 <= c['gold_count'] <= 5,
                       'ordering': lambda c: c['question_type'] == 'ordering'})
    for name, predicate in predicates.items():
        groups[name] = {}
        for side in ('base', 'trained'):
            rows = [c for c in calls if c['model'] == side and predicate(c)]
            valid = [c for c in rows if c['evaluation_valid']]
            scores = [c['independent_score'] for c in valid]
            groups[name][side] = {'questions': len({c['candidate_id'] for c in rows}),
                'calls': len(rows), 'valid': len(valid),
                'exact_correct': sum(s['exact_match'] for s in scores),
                'exact_rate': sum(s['exact_match'] for s in scores) / len(scores) if scores else None,
                'partial_mean': float(sum((Fraction(s['partial_score']) for s in scores), Fraction()) / len(scores)) if scores else None}
    ready = sum(r['source_status'] == 'READY' for r in roots)
    ended = datetime.now(timezone.utc)
    started = datetime.fromisoformat(plan['created_at_utc'])
    report = {'batch_id': BATCH.name, 'status': 'CONTROLLED_PRODUCTION_COMPLETE_NO_TRAINING',
        'attempted_roots': 12, 'structural_pass_roots': 12, 'source_ready_roots': ready,
        'source_hold_roots': 12 - ready, 'source_failures': source_failures,
        'new_turns_all_attempts': sum(r.get('new_turn_count', 0) for r in roots),
        'new_turns_ready_sources': sum(r.get('new_turn_count', 0) for r in roots if r['source_status'] == 'READY'),
        'planned_new_question_slots_all_roots': 72,
        'planned_new_question_slots_ready_roots': 6 * ready,
        'content_pass_new_questions': accounting['new_questions'],
        'content_pass_multiselect': accounting['new_multiselect'],
        'content_pass_ordering': accounting['new_ordering'],
        'content_pass_gold2_5_distribution': accounting['gold2_to5_distribution'],
        'content_pass_gold6_controls': accounting['gold6_controls'],
        'provisional_training_candidates': package['eligible_candidates'],
        'provisional_training_type_counts': package['type_counts'],
        'provisional_training_gold_counts': package['multi_gold_counts'],
        'paired_solver_calls_total': len(calls), 'solver_groups': groups,
        'expansion_authoring_parallel_window_seconds': expansion_window,
        'wall_seconds_since_preselection_freeze': (ended - started).total_seconds(),
        'finished_at_utc': ended.isoformat(), 'training_started': False,
        'holdout_performance_claimed': False, 'formal_backend_comparison': False,
        'current_solver_feedback_loop_completed': False, 'external_semantic_certification': False,
        'review_timing_addendum': bind(BATCH / 'evidence/GROUP1_REVIEW_TIMING_ADDENDUM.json'),
        'exact_generator_model_id': None, 'billing_tokens': None, 'billing_cost': None}
    lines = ['# v107：12-root 受控扩充与联合出题试产', '',
        '本批沿用 v105 内容合同（继承 v98/v103 Qwen 内容要求），在事件匹配和生成前冻结 12 个训练侧 root、素材分区、尝试预算和评价规则。实际生成与语义审核由当前 Codex 会话代理执行；脚本负责机械门禁、交叉核对和封存。', '',
        '## 固定分母的产出', '',
        f"12/12 root 完成一次扩充首稿并通过结构/旧史前缀检查；共新增 {report['new_turns_all_attempts']} turns。来源全文审核 READY {ready}/12、HOLD {12-ready}/12；READY 来源新增 {report['new_turns_ready_sources']} turns。", '',
        f"计划最多 72 道新增题和 12 道 Gold6 对照；READY 来源的新增题槽位为 {6*ready}。实际内容通过 {accounting['new_questions']} 道新增题（多选 {accounting['new_multiselect']}、排序 {accounting['new_ordering']}），另 {accounting['gold6_controls']} 道 Gold6 对照。Gold2/3/4/5 分布：" + str(accounting['gold2_to5_distribution']) + '。', '',
        f"Solver 完成 {len(calls)} 次配对调用，独立评分与完整性审核 VALID、0 errors。按既定训练侧 Solver 门槛筛得 {package['eligible_candidates']} 道候选，类型 {package['type_counts']}，多选 Gold 分布 {package['multi_gold_counts']}；Gold6 对照未导入训练候选。", '',
        '| Root | 来源 | 新增 turns | 多选内容通过 | 排序内容通过 | G6对照 | Solver入候选：多选/排序 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in roots:
        counts = selected[r['root_id']]
        lines.append(f"| {r['root_id']} | {r['source_status']} | {r.get('new_turn_count',0)} | {r['new_multiselect']} | {r['ordering']} | {r['controls']} | {counts['multi_select']}/{counts['ordering']} |")
    lines += ['', '## Solver 诊断', '',
        '这是已有训练侧 root 上新题的描述性结果；训练前 M0 与首轮已有 M_clean_policy 配对，没有运行新训练。', '',
        '| 分组 | 题数 | 每模型有效调用 | M0完全正确率 | 首轮训练后完全正确率 |',
        '|---|---:|---:|---:|---:|']
    for name, sides in groups.items():
        b, t = sides['base'], sides['trained']
        def pct(v):
            return 'NA' if v is None else f'{100*v:.2f}%'
        lines.append(f"| {name} | {b['questions']} | {b['valid']} | {pct(b['exact_rate'])} | {pct(t['exact_rate'])} |")
    lines += ['', '## 失败与质量边界', '']
    for failure in source_failures:
        reasons = [f['reason'] for f in failure['findings'] if f.get('severity') == 'BLOCKING']
        lines.append('- ' + failure['root_id'] + '：' + '；'.join(reasons))
    lines += ['', '其他未通过选项、排序槽和整题原因见各 root 的原稿及审核文件、UNFILLED_SLOTS.json、ALL_OUTCOMES_INDEX.json。所有失败首稿保留；没有改旧史、换 root/素材、修改失败题或 Gold 来提高产出。', '',
        '来源 READY 仍可能带历史称呼、重复、时间过渡、自然度等 WARNING。题目主要在新增 scene 内进行支持判断和话语排序；长输入不等于跨历史记忆任务。相关子题按 root/family 管理，调用数不能当独立样本数。', '',
        '## 耗时与反馈', '',
        f"三个作者小组并行扩充的共有墙钟窗口约 {expansion_window:.1f} 秒；不能把每个 root 复用的小组 receipt 时段相加。从选源前配置冻结到本报告约 {(ended-started).total_seconds()/60:.1f} 分钟，包括素材匹配、审核、联合出题、执行与封存。", '',
        '12 个生产 root 没有各自的匹配历史反馈；作者只读经训练集合过滤的历史汇总。194259 的非 clean-training 反馈被排除，没有将本轮验证/测试答案或逐题分数用于生成。本轮完成数据生产和评估，尚未完成新 Solver 反馈驱动的再生成/再训练闭环。', '',
        'group1 资产审核原 finished 字段记的是资料快照时刻，不能当完整审核结束时刻。已另存只增时间补充说明并绑定原文件；实际资产审核结束时刻无法可靠恢复，记 null，不据该字段计算审核耗时。', '',
        '## 论文表述与下一阶段', '',
        '可以报告为预先固定规则与预算的探索性数据构造/试产实验。作者和审核者使用同一继承模型、共享项目背景，不能称外部模型认证或完全盲审；精确生成模型、调用 usage 和费用不可得，记录为 null。执行系统与 Qwen 原 API 多阶段引擎不同，不能把产出差异直接归因于模型能力。正式论文另需遵从投稿方对 AI 使用和数据来源的披露要求；本记录不构成期刊合规认证。', '',
        '已提供 provisional frontier 候选 JSONL、逐题来源/题族/审核/评分清单，以及原 clean replay 的只读绑定。它们可用于配置下一轮小规模训练实验；训练比例、family 权重和预算需明确冻结，不能把本次候选包当作已验证的训练收益。新题是六选项/六项排序，原 replay 包含五选项/四项排序，后续训练应如实记录格式混合。', '',
        '下一阶段可继续按同一合同积累训练侧 root；按预先固定训练方案运行小规模 frontier＋replay，再在保留的独立 root 上比较新旧 Solver。', '']
    for filename, data in [('PRODUCTION_REPORT.json', json.dumps(report, ensure_ascii=False, indent=2) + '\n'),
                           ('PRODUCTION_REPORT_ZH.md', '\n'.join(lines))]:
        with (HERE / filename).open('x', encoding='utf-8') as f:
            f.write(data)
    outputs = [Path(__file__), HERE / 'PRODUCTION_REPORT.json', HERE / 'PRODUCTION_REPORT_ZH.md',
               HERE / 'training_candidate_package/COMPLETE_MANIFEST.json',
               HERE / 'EVALUATION_COMPLETE_MANIFEST.json', BATCH / 'COMPLETE_MANIFEST.json']
    with (HERE / 'PRODUCTION_COMPLETE_MANIFEST.json').open('x', encoding='utf-8') as f:
        json.dump({'status': 'CONTROLLED_PRODUCTION_AUDITED_COMPLETE', 'files': [bind(p) for p in outputs],
            'training_started': False}, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'source_ready_roots', 'content_pass_new_questions',
        'provisional_training_candidates', 'paired_solver_calls_total')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
