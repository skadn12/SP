#!/usr/bin/env python3
"""CPU-only report from completed, fixed validation stages and training audits.

No model libraries, checkpoints, remote services, or historical evaluations are
loaded. --check-inputs is read-only and never creates a final report directory.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

FINAL = Path(__file__).resolve().parents[1]
DEFAULT_RUN = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/'
                   'h4_round0_b0_final_merged_training_20261108_run01')
STEPS = (0, 20, 40, 60, 71)
WINDOW_STEPS = (1, 20, 40, 60, 71)
TYPE_COUNTS = {'multi_select': 123, 'ordering': 119}
EXPECTED_N = 242
EXPECTED_ROOTS = 15
GROUPS = ('overall', 'multi_select', 'ordering')
TRANSITIONS = ('correct_to_correct', 'correct_to_incorrect',
               'incorrect_to_correct', 'incorrect_to_incorrect')
LABELS = {'overall': '总体', 'multi_select': '多选', 'ordering': '排序'}
LIMITATIONS = [
    '这 242 题来自既有 DEV，覆盖 15 个 roots；它不是按任务结构封存的 TEST。',
    '验证只有 123 道多选和 119 道排序，没有 partial_order 验证，不能推断偏序表现。',
    '这是固定单展示、greedy、每题一次的训练轨迹监控，不等同原多展示、多次采样的 Solver 准入 gate。',
    '同 root 题目共享材料；逐题转移和 15 个 roots 的分布只作描述，不按 242 个独立样本推显著性。',
    '没有配套消融或对照，不能把变化归因于反馈、F/G 方法，不能据此断言去模板化成功。',
    '完整 JSON、非空 answer_reason、辅助协议合规均不证明理由正确、引用蕴含成立或模型推理忠实。',
    '训练窗口 loss 与 valid Gold loss 的样本和加权方式不同，不能当作同一批数据的直接差值；窗口大小也不同。',
    '只汇报预定末端 step 71，不根据 DEV 最优阶段重新选择模型。',
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': sha(path),
            'size_bytes': path.stat().st_size}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def verify_binding(item):
    path = Path(item['path'])
    require(path.is_file(), f'Binding missing: {path}')
    require(sha(path) == item['sha256'], f'Binding hash differs: {path}')
    if 'size_bytes' in item:
        require(path.stat().st_size == item['size_bytes'], f'Binding size differs: {path}')


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, f'Duplicate JSON key: {key}')
        obj[key] = value
    return obj


def json_flags(row):
    # scoring_text is the saved whole decode with special tokens removed.
    # Do not extract an embedded object, repair malformed JSON, or use raw
    # substring occurrence of "answer_reason" as evidence of a complete reply.
    try:
        obj = json.loads(row['scoring_text'], object_pairs_hook=unique_object)
    except (ValueError, TypeError, KeyError):
        obj = None
    object_ok = isinstance(obj, dict)
    reason_ok = bool(object_ok and isinstance(obj.get('answer_reason'), str)
                     and obj['answer_reason'].strip())
    fields_ok = bool(object_ok and {'item_evidence', 'answer_reason', 'final_answer'} <= set(obj)
                     and isinstance(obj['final_answer'], str))
    finished = row['finish_reason'] in ('eos', 'stop') and not row['truncated']
    return {'json_object': object_ok, 'json_nonempty_answer_reason': reason_ok,
            'complete_json_nonempty_answer_reason': bool(fields_ok and reason_ok and finished)}


def aggregate(rows):
    require(rows, 'Empty metric group')
    n = len(rows)
    targets = sum(r['supervised_causal_targets'] for r in rows)
    require(targets > 0, 'No causal Gold targets')
    flags = [json_flags(r) for r in rows]
    result = {
        'records': n,
        'valid_loss': sum(r['valid_loss'] for r in rows) / n,
        'valid_loss_token_weighted': sum(r['valid_loss'] * r['supervised_causal_targets'] for r in rows) / targets,
        'supervised_causal_targets': targets,
        'exact': sum(r['exact'] for r in rows) / n,
        'exact_correct_count': sum(int(r['exact'] == 1) for r in rows),
        'mean_J': sum(r['J'] for r in rows) / n,
        'final_answer_valid_rate': sum(bool(r['final_answer_valid']) for r in rows) / n,
        'final_answer_valid_count': sum(bool(r['final_answer_valid']) for r in rows),
        'auxiliary_protocol_valid_rate': sum(bool(r['auxiliary_protocol_valid']) for r in rows) / n,
        'truncation_rate': sum(bool(r['truncated']) for r in rows) / n,
        'truncated_count': sum(bool(r['truncated']) for r in rows),
    }
    for key in flags[0]:
        result[key + '_count'] = sum(flag[key] for flag in flags)
        result[key + '_rate'] = result[key + '_count'] / n
    require(all(math.isfinite(v) for v in result.values()), 'Nonfinite aggregate')
    return result


def grouped_metrics(rows):
    return {group: aggregate(rows if group == 'overall' else
                             [r for r in rows if r['question_type'] == group]) for group in GROUPS}


def identity(row):
    return (row['candidate_id'], row['root_id'], row['question_type'], row['index'],
            row['gold_record_sha256'], row['generation_prompt']['sha256'],
            row['input_ids_sha256'], row['supervised_causal_targets'])


def load_stage(run_dir, step):
    folder = run_dir / 'validation' / f'step-{step:06d}'
    paths = [folder / name for name in ('METRICS.json', 'RECORDS.jsonl', 'VALIDATION_RESULT.json')]
    require(all(p.is_file() for p in paths), f'Stage {step} is not complete')
    metric_doc, rows, envelope = read_json(paths[0]), read_jsonl(paths[1]), read_json(paths[2])
    require(envelope['status'] == 'PASS' and envelope['step'] == step, f'Stage {step} lacks PASS envelope')
    require(metric_doc['step'] == step and metric_doc['evaluated_records'] == EXPECTED_N,
            f'Stage {step} metric denominator differs')
    artifacts = {Path(b['path']).resolve(): b for b in envelope['artifacts']}
    # The completed-stage envelope binds the exact ledger and aggregate inputs.
    for path in paths[:2]:
        require(path.resolve() in artifacts, f'Stage artifact unbound: {path}')
        verify_binding(artifacts[path.resolve()])
    require(len(rows) == len({r['candidate_id'] for r in rows}) == EXPECTED_N,
            f'Stage {step} record set is incomplete or duplicated')
    require(dict(collections.Counter(r['question_type'] for r in rows)) == TYPE_COUNTS,
            f'Stage {step} type denominator differs')
    require(len({r['root_id'] for r in rows}) == EXPECTED_ROOTS, f'Stage {step} root count differs')
    require([r['index'] for r in rows] == list(range(EXPECTED_N)), f'Stage {step} fixed order differs')
    for r in rows:
        require(r['step'] == step and r['automatic_retry'] is False, f'Stage/retry differs: {r["candidate_id"]}')
        require(r['exact'] in (0, 1) and math.isfinite(r['J']) and math.isfinite(r['valid_loss']),
                f'Invalid metric: {r["candidate_id"]}')
        if not r['final_answer_valid']:
            require(r['exact'] == 0 and r['J'] == 0, f'Invalid final answer scored nonzero: {r["candidate_id"]}')
        require(not r['truncated'] or not r['final_answer_valid'], f'Truncated final answer marked valid: {r["candidate_id"]}')
    groups = grouped_metrics(rows)
    original_keys = ('records', 'valid_loss', 'valid_loss_token_weighted', 'supervised_causal_targets',
                     'exact', 'mean_J', 'final_answer_valid_rate', 'auxiliary_protocol_valid_rate', 'truncation_rate')
    for group, metrics in groups.items():
        prefix = '' if group == 'overall' else group + '/'
        for key in original_keys:
            expected = metrics[key]
            require(math.isclose(metric_doc['metrics'][prefix + key], expected, rel_tol=1e-10, abs_tol=1e-12),
                    f'Stage {step} METRICS/RECORDS mismatch: {prefix + key}')
            require(math.isclose(envelope['metrics'][prefix + key], expected, rel_tol=1e-10, abs_tol=1e-12),
                    f'Stage {step} envelope metric mismatch: {prefix + key}')
    return {'step': step, 'groups': groups, 'rows': rows,
            'bindings': [binding(p) for p in paths], 'result_sha256': sha(paths[2])}


def input_status(run_dir):
    stages, in_progress, absent = [], [], []
    for step in STEPS:
        folder = run_dir / 'validation' / f'step-{step:06d}'
        if all((folder / name).is_file() for name in ('METRICS.json', 'RECORDS.jsonl', 'VALIDATION_RESULT.json')):
            stages.append(load_stage(run_dir, step))
        elif folder.exists():
            in_progress.append(step)
        else:
            absent.append(step)
    if stages:
        ids = [identity(r) for r in stages[0]['rows']]
        for stage in stages[1:]:
            require([identity(r) for r in stage['rows']] == ids,
                    f'Fixed question/Gold/display identity changed at step {stage["step"]}')
    completion_path = run_dir / 'TRAINING_COMPLETED.json'
    completed = read_json(completion_path) if completion_path.is_file() else None
    return stages, {'status': 'INPUTS_CHECKED', 'training_completed_marker_present': completed is not None,
                    'training_completed_status': completed.get('status') if completed else None,
                    'completed_validation_steps': [s['step'] for s in stages],
                    'in_progress_validation_steps': in_progress, 'absent_validation_steps': absent,
                    'expected_validation_steps': list(STEPS), 'records_per_completed_stage': EXPECTED_N,
                    'report_ready': bool(completed and completed.get('status') == 'COMPLETED'
                                         and [s['step'] for s in stages] == list(STEPS)),
                    'final_report_written': False}, completed


def load_training_audit(run_dir, completed, stages):
    audit_path, ledger_path = run_dir / 'gradient_audit.json', run_dir / 'gradient_audit.jsonl'
    require(Path(completed['audit_binding']['path']).resolve() == audit_path.resolve(), 'Wrong completion audit path')
    verify_binding(completed['audit_binding'])
    audit, events = read_json(audit_path), read_jsonl(ledger_path)
    require(audit['status'] == 'PASS' and audit['phase'] == 'formal' and not audit['failures'], 'Training audit not PASS')
    require(audit['backward_calls'] == audit['expected_backward_calls'] == completed['actual_backward_calls'] == 282,
            'Actual backward budget is not 282')
    require(audit['optimizer_steps'] == audit['expected_optimizer_steps'] == completed['actual_optimizer_steps'] == 71,
            'Actual optimizer budget is not 71')
    require(completed['evaluations'] == [e['step'] for e in audit['evaluations']] == list(STEPS),
            'Completed evaluation schedule differs')
    require(audit['all_losses_finite'] and audit['all_gradients_finite_nonzero']
            and audit['native_trainer_unchanged'], 'Audit finite/gradient/trainer gate failed')
    require(Path(audit['event_ledger']['path']).resolve() == ledger_path.resolve(), 'Wrong audit ledger')
    verify_binding(audit['event_ledger'])
    for e, stage in zip(audit['evaluations'], stages):
        verify_binding(e['result_binding'])
        require(e['result_binding']['sha256'] == stage['result_sha256'], 'Audit/stage binding mismatch')
    backward = [e for e in events if e['event'] == 'backward']
    optimizer = [e for e in events if e['event'] == 'optimizer_step']
    require([e['micro_step'] for e in backward] == list(range(1, 283)), 'Backward ledger incomplete')
    require([e['optimizer_step'] for e in optimizer] == list(range(1, 72)), 'Optimizer ledger incomplete')
    require(all(e['loss_finite'] and math.isfinite(e['scaled_loss']) for e in backward), 'Nonfinite backward loss')
    windows = [dict(e) for e in events if e['event'] == 'training_loss_window']
    require([e['optimizer_step'] for e in windows] == list(WINDOW_STEPS), 'Loss window endpoints differ')
    previous = 0
    for w in windows:
        end = w['micro_step']
        group = backward[previous:end]
        require(len(group) == w['backward_calls_in_window'] and group, 'Loss window membership differs')
        unscaled = sum(e['scaled_loss'] * 4 for e in group) / len(group)
        require(math.isclose(unscaled, w['mean_unscaled_loss'], rel_tol=1e-10, abs_tol=1e-12),
                'Training loss window does not match actual backward ledger')
        w.update(first_micro_step=previous + 1, last_micro_step=end)
        previous = end
    require(previous == 282, 'Loss windows do not cover all backward calls')
    return {'summary': {k: audit[k] for k in ('status', 'phase', 'backward_calls', 'optimizer_steps',
                    'all_losses_finite', 'all_gradients_finite_nonzero', 'native_trainer_unchanged')},
            'windows': windows, 'bindings': [binding(audit_path), binding(ledger_path),
                                            binding(run_dir / 'TRAINING_COMPLETED.json')]}


def transition_key(before, after):
    return ('correct' if before['exact'] == 1 else 'incorrect') + '_to_' + ('correct' if after['exact'] == 1 else 'incorrect')


def paired_summary(before, after):
    require([identity(r) for r in before] == [identity(r) for r in after], 'Paired identity differs')
    counts = {k: 0 for k in TRANSITIONS}
    records = []
    for left, right in zip(before, after):
        key = transition_key(left, right)
        counts[key] += 1
        records.append({'candidate_id': left['candidate_id'], 'root_id': left['root_id'],
                        'question_type': left['question_type'], 'transition': key,
                        'baseline_exact': int(left['exact']), 'final_exact': int(right['exact']),
                        'baseline_J': left['J'], 'final_J': right['J'], 'J_delta': right['J'] - left['J'],
                        'baseline_final_answer_valid': left['final_answer_valid'],
                        'final_final_answer_valid': right['final_answer_valid'],
                        'baseline_truncated': left['truncated'], 'final_truncated': right['truncated']})
    return {'records': len(before), 'transitions': counts,
            'net_correct_change': counts['incorrect_to_correct'] - counts['correct_to_incorrect'],
            'baseline_exact': sum(r['exact'] for r in before) / len(before),
            'final_exact': sum(r['exact'] for r in after) / len(after),
            'baseline_mean_J': sum(r['J'] for r in before) / len(before),
            'final_mean_J': sum(r['J'] for r in after) / len(after), 'paired_records': records}


def transitions(before, after):
    overall = paired_summary(before, after)
    by_type = {kind: paired_summary([r for r in before if r['question_type'] == kind],
                                   [r for r in after if r['question_type'] == kind]) for kind in TYPE_COUNTS}
    by_root = {root: paired_summary([r for r in before if r['root_id'] == root],
                                   [r for r in after if r['root_id'] == root])
               for root in sorted({r['root_id'] for r in before})}
    # Root macro averages are descriptive only; no confidence interval or test.
    root_distribution = {'roots': len(by_root),
        'roots_with_net_correct_increase': sum(r['net_correct_change'] > 0 for r in by_root.values()),
        'roots_with_net_correct_decrease': sum(r['net_correct_change'] < 0 for r in by_root.values()),
        'roots_with_unchanged_correct_count': sum(r['net_correct_change'] == 0 for r in by_root.values()),
        'baseline_root_macro_exact': sum(r['baseline_exact'] for r in by_root.values()) / len(by_root),
        'final_root_macro_exact': sum(r['final_exact'] for r in by_root.values()) / len(by_root),
        'baseline_root_macro_J': sum(r['baseline_mean_J'] for r in by_root.values()) / len(by_root),
        'final_root_macro_J': sum(r['final_mean_J'] for r in by_root.values()) / len(by_root)}
    return {'overall': overall, 'by_type': by_type, 'by_root': by_root,
            'root_distribution': root_distribution, 'statistical_tests_performed': False}


def plots(result, out):
    # Import only after all completion gates have passed. Agg never uses a GPU.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 120, 'savefig.dpi': 220})
    stages, steps = result['validation_stages'], list(STEPS)
    colors = {'overall': '#172B4D', 'multi_select': '#1D8A8A', 'ordering': '#C56B21'}
    def save(fig, stem):
        fig.savefig(out / (stem + '.png'), bbox_inches='tight')
        fig.savefig(out / (stem + '.pdf'), bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    windows = result['training_audit']['windows']
    ax.plot([w['optimizer_step'] for w in windows], [w['mean_unscaled_loss'] for w in windows],
            marker='o', label='Train: actual backward-window mean', color='#7B4AB4')
    ax.plot(steps, [s['groups']['overall']['valid_loss'] for s in stages], marker='o',
            label='DEV Gold loss: sample mean', color=colors['overall'])
    ax.plot(steps, [s['groups']['overall']['valid_loss_token_weighted'] for s in stages], marker='s',
            linestyle='--', label='DEV Gold loss: causal-token weighted', color=colors['multi_select'])
    ax.set(xlabel='Optimizer step', ylabel='Cross-entropy loss', title='Training windows and fixed DEV Gold loss')
    ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout(); save(fig, '01_loss')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric, title in zip(axes, ['exact', 'mean_J'], ['Exact accuracy', 'Official signed J']):
        for group in GROUPS:
            ax.plot(steps, [s['groups'][group][metric] for s in stages], marker='o', color=colors[group], label=group)
        ax.set(xlabel='Optimizer step', ylabel=title, title=title); ax.set_ylim((-1.02, 1.02) if metric == 'mean_J' else (0, 1.02))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Fixed DEV: 242 records / 15 shared-source roots (descriptive)'); fig.tight_layout(); save(fig, '02_exact_J')
    fig, ax = plt.subplots(figsize=(9.6, 4.7))
    for key, label in [('final_answer_valid_rate', 'Valid final answer'), ('truncation_rate', 'Length truncation'),
                       ('complete_json_nonempty_answer_reason_rate', 'Complete JSON + nonempty answer_reason'),
                       ('auxiliary_protocol_valid_rate', 'Strict auxiliary protocol')]:
        ax.plot(steps, [s['groups']['overall'][key] for s in stages], marker='o', label=label)
    ax.set(xlabel='Optimizer step', ylabel='Fraction of all 242 records', ylim=(-.02, 1.02), title='Answer transport and response completeness')
    ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout(); save(fig, '03_response_quality')
    roots = result['baseline_to_final']['by_root']
    labels = [root.rsplit('-', 1)[-1] for root in roots]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    bottom = [0] * len(roots)
    for key, label, color in [('incorrect_to_incorrect', 'Incorrect -> incorrect', '#B7BEC8'),
                             ('correct_to_correct', 'Correct -> correct', '#1D8A8A'),
                             ('incorrect_to_correct', 'Incorrect -> correct', '#5B9C43'),
                             ('correct_to_incorrect', 'Correct -> incorrect', '#C85C52')]:
        values = [r['transitions'][key] for r in roots.values()]
        axes[0].bar(labels, values, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[0].set(ylabel='Records', title='Same-record baseline -> final transitions by root')
    axes[0].legend(fontsize=8, ncol=2)
    delta = [r['net_correct_change'] for r in roots.values()]
    axes[1].bar(labels, delta, color=['#5B9C43' if v >= 0 else '#C85C52' for v in delta])
    axes[1].axhline(0, color='black', linewidth=.7); axes[1].set(xlabel='Root ID suffix (15 roots)', ylabel='Net correct change')
    axes[1].tick_params(axis='x', rotation=45); fig.tight_layout(); save(fig, '04_root_transitions')


def pct(value):
    return f'{value * 100:.2f}%'


def markdown(result):
    stages = result['validation_stages']
    start, end = stages[0]['groups']['overall'], stages[-1]['groups']['overall']
    paired = result['baseline_to_final']['overall']
    lines = ['训练完成后的固定 DEV 结果报告', '',
             f'已核对 TRAINING_COMPLETED 与实际审计：71 次 optimizer update、282 次 backward，验证阶段为 0/20/40/60/71。'
             f'相同 242 道题总体 exact 从 {pct(start["exact"])} 到 {pct(end["exact"])}，'
             f'J 从 {start["mean_J"]:.6f} 到 {end["mean_J"]:.6f}。', '',
             '这是既有 DEV 的训练轨迹描述。结果不证明结构泛化、反馈收益或去模板化成功。', '',
             '训练 loss 使用实际窗口记录；valid Gold loss 同时报题均值与因果监督 token 加权值。', '',
             '| 窗口末 step | 实际 micro steps | backward 数 | 训练窗口 loss |',
             '| --- | --- | --- | --- |']
    for w in result['training_audit']['windows']:
        lines.append(f'| {w["optimizer_step"]} | {w["first_micro_step"]}–{w["last_micro_step"]} | {w["backward_calls_in_window"]} | {w["mean_unscaled_loss"]:.6f} |')
    lines += ['', '| 验证 step | valid Gold loss（题均） | valid Gold loss（token 加权） | 总体 exact | 总体 J |',
              '| --- | --- | --- | --- | --- |']
    for s in stages:
        g = s['groups']['overall']
        lines.append(f'| {s["step"]} | {g["valid_loss"]:.6f} | {g["valid_loss_token_weighted"]:.6f} | {g["exact_correct_count"]}/242 ({pct(g["exact"])}) | {g["mean_J"]:.6f} |')
    lines += ['', '| step | 类型 | n | exact | J |', '| --- | --- | --- | --- | --- |']
    for s in stages:
        for kind in TYPE_COUNTS:
            g = s['groups'][kind]
            lines.append(f'| {s["step"]} | {LABELS[kind]} | {g["records"]} | {pct(g["exact"])} | {g["mean_J"]:.6f} |')
    lines += ['', '下表分母均为全部 242 题，包含无效答案和截断答案。完整 JSON 加非空理由仅检查输出完整性，辅助协议合规另计。', '',
              '| step | 有效 final answer | 截断 | 完整 JSON＋非空 answer_reason | 辅助协议合规 |',
              '| --- | --- | --- | --- | --- |']
    for s in stages:
        g = s['groups']['overall']
        lines.append(f'| {s["step"]} | {g["final_answer_valid_count"]}/242 ({pct(g["final_answer_valid_rate"])}) | {g["truncated_count"]}/242 | {g["complete_json_nonempty_answer_reason_count"]}/242 ({pct(g["complete_json_nonempty_answer_reason_rate"])}) | {pct(g["auxiliary_protocol_valid_rate"])} |')
    c = paired['transitions']
    lines += ['', f'相同 242 题逐题配对：对→对 {c["correct_to_correct"]}，对→错 {c["correct_to_incorrect"]}，'
              f'错→对 {c["incorrect_to_correct"]}，错→错 {c["incorrect_to_incorrect"]}；净正确数变化 {paired["net_correct_change"]:+d}。', '',
              '| 类型 | n | 对→对 | 对→错 | 错→对 | 错→错 | 净变化 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for kind, r in result['baseline_to_final']['by_type'].items():
        c = r['transitions']
        lines.append(f'| {LABELS[kind]} | {r["records"]} | {c["correct_to_correct"]} | {c["correct_to_incorrect"]} | {c["incorrect_to_correct"]} | {c["incorrect_to_incorrect"]} | {r["net_correct_change"]:+d} |')
    rd = result['baseline_to_final']['root_distribution']
    lines += ['', f'按 15 个 roots 描述：净正确数增加 {rd["roots_with_net_correct_increase"]} 个，下降 '
              f'{rd["roots_with_net_correct_decrease"]} 个，不变 {rd["roots_with_unchanged_correct_count"]} 个。'
              '未计算显著性或独立样本置信区间。', '',
              '| root | n | baseline exact | final exact | 对→对 | 对→错 | 错→对 | 错→错 | J 变化 |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for root, r in result['baseline_to_final']['by_root'].items():
        c = r['transitions']
        lines.append(f'| {root} | {r["records"]} | {pct(r["baseline_exact"])} | {pct(r["final_exact"])} | {c["correct_to_correct"]} | {c["correct_to_incorrect"]} | {c["incorrect_to_correct"]} | {c["incorrect_to_incorrect"]} | {r["final_mean_J"] - r["baseline_mean_J"]:+.6f} |')
    lines += ['', '解释限制：', ''] + ['- ' + text for text in LIMITATIONS]
    lines += ['', '逐题 242 条配对、分型指标、15 roots 分布和实际输入 SHA256 均保存于 RESULT.json。', '',
              '曲线使用标准 Matplotlib 生成，PNG 与 PDF 各四张：loss、exact/J、输出完整性、root 转移。', '',
              '![Loss](01_loss.png)', '', '![Exact and J](02_exact_J.png)', '',
              '![Response quality](03_response_quality.png)', '', '![Root transitions](04_root_transitions.png)', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-dir', type=Path, default=FINAL / 'TRAINING_OUTCOME_REPORT_v1')
    parser.add_argument('--check-inputs', action='store_true', help='Only validate completed stages and print readiness; write nothing')
    args = parser.parse_args()
    require(args.run_dir.is_dir(), f'Run directory missing: {args.run_dir}')
    stages, status, completed = input_status(args.run_dir)
    if args.check_inputs:
        if status['report_ready']:
            audit = load_training_audit(args.run_dir, completed, stages)
            status['training_audit'] = audit['summary']
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    require(status['report_ready'], 'Final report refused: TRAINING_COMPLETED and all five completed stages are required; use --check-inputs for current status')
    require(not args.output_dir.exists(), f'Report output already exists; choose a new --output-dir: {args.output_dir}')
    run_resolved, out_resolved = args.run_dir.resolve(), args.output_dir.resolve()
    require(not out_resolved.is_relative_to(run_resolved), 'Report must not write into the active training run directory')
    audit = load_training_audit(args.run_dir, completed, stages)
    result = {'schema': 'completed_training_outcome_report_v1', 'status': 'COMPLETED_EVIDENCE_VERIFIED',
              'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'run_dir': str(args.run_dir.resolve()),
              'comparison': 'Predeclared step 0 to final step 71; fixed same-record DEV trajectory, descriptive only',
              'training_audit': audit,
              'validation_stages': [{k: v for k, v in s.items() if k != 'rows'} for s in stages],
              'baseline_to_final': transitions(stages[0]['rows'], stages[-1]['rows']),
              'definitions': {
                  'valid_loss': 'Sample mean of actual saved native CausalLM Gold losses over every fixed record.',
                  'valid_loss_token_weighted': 'Gold loss weighted by actual labels[1:] != -100 token count, including native suffix/EOS.',
                  'exact_J': 'Saved official final-answer exact and signed J, independently aggregated; invalid/truncated answers contribute zero.',
                  'response_denominator': 'Every fixed attempted record; 242 overall, 123 multi_select, 119 ordering.',
                  'complete_json_nonempty_answer_reason_rate': 'Whole saved scoring_text parses as a unique-key JSON object with item_evidence/answer_reason/final_answer; reason is a nonempty string, final answer is a string, response ends eos/stop and is not truncated. No repair, embedded-object extraction, evidence validation or entailment claim.',
                  'json_object_rate': 'Whole scoring_text parses as a JSON object with no duplicate keys; distinct from strict auxiliary protocol.',
                  'training_loss_windows': 'Actual training_loss_window events, rechecked against scaled backward losses times fixed grad_accum=4; unequal window sizes retained.',
                  'correct_incorrect': 'Correct means official exact=1; incorrect includes partial, invalid and truncated replies.',
              }, 'limitations': LIMITATIONS, 'gpu_used_by_report': False,
              'script_binding': binding(Path(__file__))}
    # Verify snapshots again before creating a fresh output directory.
    for stage in stages:
        for item in stage['bindings']:
            verify_binding(item)
    for item in audit['bindings']:
        verify_binding(item)
    args.output_dir.mkdir(parents=False, exist_ok=False)
    plots(result, args.output_dir)
    result['plot_bindings'] = [binding(p) for p in sorted(args.output_dir.iterdir())]
    with (args.output_dir / 'RESULT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    with (args.output_dir / 'REPORT.zh.md').open('x', encoding='utf-8') as stream:
        stream.write(markdown(result))
    artifacts = [binding(p) for p in sorted(args.output_dir.iterdir())]
    with (args.output_dir / 'ARTIFACTS.json').open('x', encoding='utf-8') as stream:
        json.dump({'status': 'REPORT_WRITTEN', 'artifacts': artifacts}, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'REPORT_WRITTEN', 'output_dir': str(args.output_dir.resolve()),
                      'artifacts': len(artifacts) + 1, 'inputs_unchanged': True}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({'status': 'REPORT_NOT_GENERATED', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
