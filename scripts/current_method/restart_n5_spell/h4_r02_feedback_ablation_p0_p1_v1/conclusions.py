"""Render a concrete interpretation from the frozen descriptive P1 metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

from .freeze import DEFAULT, ARMS, load, rows, source, write, frozen_bytes


def pct(value):
    return 'NA' if value is None else f'{value * 100:.2f}%'


def render(run: Path):
    report = load(run / 'reports/p1_comparison.json')
    assert report['status'] == 'COMPLETE'
    completion = load(run / 'audit/p1_completion_audit.json')
    assert completion['status'] == 'PASS'
    assert completion['comparison']['sha256'] == source(run / 'reports/p1_comparison.json')['sha256']
    primary = report['primary']
    gaps = [r for r in rows(run / 'reports/context_dependency_metrics.jsonl')
            if r['root_id'] == 'all' and r['scope'] == 'main' and r['generation_source'] == 'all']
    pairs = report['primary_root_pairs']
    quality = {a: next(r for r in report['quality_review']['groups']
                      if r['arm'] == a and r['root_id'] == 'all' and r['scope'] == 'main') for a in ARMS}
    counts = report['generation_accounting']['per_arm']
    content = report['generation_content_diagnostics']['rows']
    direction = '更难答对' if primary['on_minus_off'] < 0 else '更易答对' if primary['on_minus_off'] > 0 else '相同'
    lines = ['# P1 结果解释', '',
             f"Base full-context 的 SC+MS 六 root 等权候选准确率为 off {pct(primary['off'])}、on {pct(primary['on'])}，on−off {primary['on_minus_off'] * 100:+.2f} 个百分点。在保留的原 Gold 下，反馈臂总体{direction}。",
             f"六个 root 的配对差中，{sum(p['on_minus_off'] < 0 for p in pairs)} 个为负、{sum(p['on_minus_off'] > 0 for p in pairs)} 个为正、{sum(p['on_minus_off'] == 0 for p in pairs)} 个为零。每个 root 等权；下表没有把位置或 rollout 当独立样本。", '',
             '| root | off | on | on−off（百分点） |', '|---|---:|---:|---:|']
    for row in pairs:
        lines.append(f"| {row['root_id']} | {pct(row['off'])} | {pct(row['on'])} | {row['on_minus_off'] * 100:+.2f} |")
    lines += ['', '| 模型 | 臂 | full−no 候选均值 | full−no root 等权 | context-helpful / 候选数 |',
              '|---|---|---:|---:|---:|']
    for row in gaps:
        lines.append(f"| {row['model']} | {row['arm']} | {pct(row['candidate_mean_full_minus_no'])} | {pct(row['root_macro_full_minus_no'])} | {row['context_helpful_count']}/{row['context_helpful_denominator']} |")
    base = {r['arm']: r for r in gaps if r['model'] == 'M0'}
    gap_off, gap_on = (base[arm]['root_macro_full_minus_no'] for arm in ARMS)
    gap_diff = gap_on - gap_off if gap_on is not None and gap_off is not None else None
    gap_text = f'{gap_diff * 100:+.2f} 个百分点' if gap_diff is not None else 'NA（存在未评分或缺失分组）'
    lines += ['', f"Base 的 full−no root 等权差从 {pct(gap_off)} 变为 {pct(gap_on)}，变化 {gap_text}。该结果与主准确率共同描述当前题集的上下文依赖，不直接证明题目更有训练价值。context-helpful 沿用原程序：技术完整且 full 的正确比例严格高于 no；未增加阈值。", '',
              '| 臂 | 主范围抽查 | Gold支持 | 矛盾 | 未定 | 无支持 | 历史关系 supported / partial / absent / uncertain |',
              '|---|---:|---:|---:|---:|---:|---|']
    for arm in ARMS:
        row = quality[arm]; gold = row['gold_assessment_counts']; relation = row['feedback_relation_counts']
        lines.append(f"| {arm} | {row['reviewed_candidate_instances']} | {gold['supported']} | {gold['contradicted']} | {gold['underdetermined']} | {gold['unsupported']} | {' / '.join(str(relation[s]) for s in ('supported','partial','absent','uncertain'))} |")
    qoff, qon = quality['feedback_off'], quality['feedback_on']
    supported_diff = qon['gold_supported_rate_of_reviewed'] - qoff['gold_supported_rate_of_reviewed']
    lines += ['', f"固定对称抽样中，主范围 Gold 支持率从 {pct(qoff['gold_supported_rate_of_reviewed'])} 变为 {pct(qon['gold_supported_rate_of_reviewed'])}，变化 {supported_diff * 100:+.2f} 个百分点。该样本按每 root/arm/type 最多两题选择，描述抽查结果，不能当作全量题库缺陷率。未定、无支持和矛盾判断均保留；难度变化可能混合题目内容、标签质量与分布变化。",
              '历史关系对应同时在 off 和 on 中核查；对应关系只说明新题实际涉及了该历史关系，不证明反馈是其原因。完整 transcript 内本来就可能自然出现相同主题。', '',
              '| 臂 | 有效题 | Guided / Free 实际题数 | 引用 R02 的题 | 引用 R02 的 contract | 同 root stem 重复超额 |',
              '|---|---:|---:|---:|---:|---:|']
    diagnostics = {}
    for arm in ARMS:
        selected = [r for r in content if r['arm'] == arm]
        mode_counts = {mode: sum(r['mode_counts'].get(mode, 0) for r in selected) for mode in ('contract_guided','free_exploration')}
        total = sum(r['candidate_count'] for r in selected)
        citations = sum(r['r02_evidence_candidate_count'] for r in selected)
        contracts = sum(r['valid_contract_count'] for r in selected)
        contract_refs = sum(r['r02_referencing_contract_count'] for r in selected)
        duplicate_excess = sum(r['duplicate_stem_excess'] for r in selected)
        diagnostics[arm] = {'candidate_count': total, 'actual_mode_counts': mode_counts,
                            'r02_citing_candidates': citations, 'r02_citation_rate': citations/total,
                            'r02_citing_contracts': contract_refs, 'valid_contracts': contracts,
                            'duplicate_stem_excess': duplicate_excess}
        lines.append(f"| {arm} | {total} | {mode_counts['contract_guided']} / {mode_counts['free_exploration']} | {citations}/{total} ({pct(citations/total)}) | {contract_refs}/{contracts} | {duplicate_excess} |")
    dup_groups = report['duplicate_diagnostics']['groups']
    lines += ['', f"原 stem/fingerprint 规则共识别 {len(dup_groups)} 个同臂、同 root 重复组（两个规则可能重叠）；不等价于全语义去重，未计算跨臂/跨 root 重复。R02 引用统计来自实际 evidence refs 和 contracts，引用存在本身不证明答案受支持。Guided:Free 请求目标保持 1:2；实际数量受原 repair 替换和程序有效返回量影响，没有为平衡数量补题或裁剪。", '',
              '| 臂 | 逻辑调用 | completion | repair | HTTP 尝试 / 重试 | 超时 / 付费状态未知 | prompt / completion tokens |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        a = counts[arm]; tokens = a['completion_usage']
        lines.append(f"| {arm} | {a['logical_calls']} | {a['completed_calls']} | {a['repair_calls']} | {a['http_attempts']} / {a['http_retries']} | {a['timeout_calls']} / {a['unknown_billing_status_calls']} | {tokens.get('prompt_tokens')} / {tokens.get('completion_tokens')} |")
    old = report['historical_ordering_diagnosis']['summary']
    lines += ['', f"本轮共 {sum(a['logical_calls'] for a in counts.values())}/156 次逻辑调用；原计划 84 次。已完成 completion 的 token 如上，超时请求的服务端实际 token 和付费状态未知，未假造人民币或美元成本。三个 Solver 的逐候选和逐请求分母均已核对，完整模型/context/题型/Guided-Free 分组见 `group_metrics.csv`。", '',
              f"旧 R02 的 52/52 道排序已逐题核查，判断分布为 {old['gold_assessment_counts']}。新排序题仍按原 Gold 独立报告，不进入主难度指标。历史 seed 在 197635、254852 的语义口径歧义也仍保留；P0 来源资格通过不等于历史 Gold 全部获得认证。", '',
              '可确认的是：排序已退出主动反馈，两臂按同一冻结累计 transcript 和原 QA/Solver 程序完成，空/非空反馈接线及阶段 prompt 审计通过。上述准确率、上下文依赖、抽查质量、关系对应与重复数据提供六 root 单次生成的探索性证据。它们不足以证明反馈成功、整体反馈系统无效、因果利用机制或后续训练收益。',
              '本次止于 P1。对话本身曾接收旧反馈，因此本轮只比较这批既定对话上现有 QA 附加反馈机制；不估计 Planner 或整个系统的反馈效应。']
    artifact = {'status': 'COMPLETE', 'source_report': source(run / 'reports/p1_comparison.json'),
                'primary': primary, 'primary_root_pairs': pairs, 'context_dependency': gaps,
                'main_sample_quality': quality, 'generation_diagnostics': diagnostics,
                'metrics_or_admission_changed': False, 'generation_repetitions_per_arm': 1,
                'independent_root_count': 6, 'new_inferential_tests': False}
    write(run / 'reports/p1_interpretation.json', artifact)
    frozen_bytes(run / 'reports/p1_interpretation.md', ('\n'.join(lines) + '\n').encode())
    print({'status': 'COMPLETE', 'interpretation': str(run / 'reports/p1_interpretation.md')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT)
    render(parser.parse_args().run)
