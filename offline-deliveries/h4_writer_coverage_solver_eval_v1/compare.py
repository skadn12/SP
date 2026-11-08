"""Compare fixed local Solver measurements with verified first-round sources."""
from collections import Counter
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
MODELS = ['M0', 'M_clean_policy', 'M_inclusive_policy']
NAMES = {'M0': '基座 M0', 'M_clean_policy': '第一轮 Clean', 'M_inclusive_policy': '第一轮 Inclusive'}
KINDS = ['single_choice', 'multi_select', 'ordering']
LABELS = {'single_choice': '单选', 'multi_select': '多选', 'ordering': '排序', 'overall': '总体'}
MODES = ['full_context', 'no_context']


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, x):
    text = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=2)
    with p.open('x') as f:
        f.write(text.rstrip() + '\n')
    p.with_name(p.name + '.sha256').write_text(sha(p) + '\n')


def metric(rows):
    assert rows and all(r['scored'] for r in rows)
    return {'correct': sum(r['correct'] for r in rows), 'total': len(rows),
            'accuracy': sum(r['correct'] for r in rows) / len(rows)}


def main():
    assert (D / 'VERIFICATION.json').exists(), 'run independent verification first'
    baseline = read(D / 'BASELINE.json')
    for source in baseline['sources'].values():
        assert sha(Path(source['path'])) == source['sha256']
    candidates = read(D / 'prepared/candidate_manifest.json')
    current, sources = {}, []
    for model in MODELS:
        md = D / 'run' / model
        assert read(md / 'report.json')['status'] == 'COMPLETE'
        p = md / 'candidate_results.json'
        sources.append({'path': str(p), 'sha256': sha(p)})
        results = read(p)
        requests = [r for c in results for r in c['requests']]
        assert len(requests) == 714
        current[model] = {kind: {mode: metric([r for r in requests if r['context_mode'] == mode
            and (kind == 'overall' or r['question_type'] == kind)]) for mode in MODES}
            for kind in ['overall'] + KINDS}
    deltas = {}
    for scope in ['first_round_train', 'first_round_test', 'same_three_roots_previous_formal']:
        deltas[scope] = {model: {kind: {mode: 100 * (current[model][kind][mode]['accuracy']
            - baseline[scope]['models'][model][mode]['accuracies'][i]) for mode in MODES}
            for i, kind in enumerate(baseline['metric_order'])} for model in MODELS}
    out = {'status': 'COMPLETE', 'metric': 'response_micro_accuracy; full/no contexts reported separately',
        'current_candidate_count': len(candidates), 'current_by_type': dict(Counter(c['question_type'] for c in candidates)),
        'current': current, 'previous_official_baseline': baseline, 'delta_percentage_points': deltas,
        'new_training_performed': False, 'sources': sources,
        'limitations': ['Current 28 questions are not paired with the historical 104/496/2294 questions.',
                       'The three roots have historical training exposure; this is not a held-out generalization test.',
                       'Current ordering Gold was source-reviewed by Codex; historical ordering Gold was not re-audited.',
                       'Current full prompts include dates and expanded content; current ordering uses balanced item relabeling.',
                       'Answer-position permutations and seeds are repeated measurements, not independent questions.',
                       'SC/MS retain program-only acceptance. Solver error does not by itself establish question difficulty or Gold correctness.',
                       'Same weights were evaluated; differences across datasets are not new training gains.']}
    write(D / 'COMPARISON.json', out)
    lines = ['# 本轮28题与第一轮的Solver准确率对照', '',
        '三个Solver权重与第一轮相同。本轮只增加评估，未训练。全文准确率按正确作答次数/总作答次数计算；多选要求答案集合完全相同，排序要求整个顺序完全相同。', '',
        '| 模型 | 第一轮训练集2294题 | 第一轮测试集496题 | 同三个root旧104题 | 本轮28题 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for model in MODELS:
        vals = [baseline[s]['models'][model]['full_context']['accuracies'][0] for s in
                ['first_round_train', 'first_round_test', 'same_three_roots_previous_formal']]
        lines.append('| ' + NAMES[model] + ' | ' + ' | '.join(f'{100*x:.2f}%' for x in vals + [current[model]['overall']['full_context']['accuracy']]) + ' |')
    lines += ['', '## 分题型：第一轮正式测试集 → 本轮', '',
              '括号中题数为第一轮→本轮；本表都是全文条件。', '',
              '| 模型 | 单选（256→16） | 多选（125→7） | 排序（115→5） |', '| --- | ---: | ---: | ---: |']
    for model in MODELS:
        vals = [f"{100*baseline['first_round_test']['models'][model]['full_context']['accuracies'][i+1]:.2f}% → {100*current[model][kind]['full_context']['accuracy']:.2f}%" for i, kind in enumerate(KINDS)]
        lines.append('| ' + NAMES[model] + ' | ' + ' | '.join(vals) + ' |')
    lines += ['', '## 同三个root旧题 → 本轮新题', '',
              '| 模型 | 单选（53→16） | 多选（26→7） | 排序（25→5） |', '| --- | ---: | ---: | ---: |']
    for model in MODELS:
        vals = [f"{100*baseline['same_three_roots_previous_formal']['models'][model]['full_context']['accuracies'][i+1]:.2f}% → {100*current[model][kind]['full_context']['accuracy']:.2f}%" for i, kind in enumerate(KINDS)]
        lines.append('| ' + NAMES[model] + ' | ' + ' | '.join(vals) + ' |')
    lines += ['', '## 本轮全文与无原文对照', '',
              '| 模型 | 题型 | 全文正确/总次数 | 无原文正确/总次数 | 全文增益（百分点） |', '| --- | --- | ---: | ---: | ---: |']
    for model in MODELS:
        for kind in KINDS + ['overall']:
            full, no = (current[model][kind][mode] for mode in MODES)
            lines.append(f"| {NAMES[model]} | {LABELS[kind]} | {full['correct']}/{full['total']}（{100*full['accuracy']:.2f}%） | {no['correct']}/{no['total']}（{100*no['accuracy']:.2f}%） | {100*(full['accuracy']-no['accuracy']):+.2f} |")
    lines += ['', '## 如何理解', '',
        '本轮与第一轮的题目、题型配比、扩充原文和日期呈现不同；排序展示还改为程序重新编号，减少身份编号泄露顺序。旧排序Gold的语义问题未在本次全面重审。因此表中的分数可作描述性对照，不能当成同题难度变化或新增训练收益。', '',
        '本轮每道单选、排序在每种上下文下作答12次（4种置换×3个种子），每道多选15次（5×3）。357次全文作答仍然只有28道题。总体微平均使多选权重略高；逐题等权平均另见RESULTS。', '',
        '选择题按用户指定政策只做程序接收，没有重新增加语义或证据完整性门槛。排序5道经过Codex原文诊断，其中1道在任何Solver作答前纠正Gold和证据方向。Codex诊断不是独立人工认证。', '',
        '[本轮按root和题型结果](RESULTS.md) · [每道题的分数](QUESTION_RESULTS.md) · [第一轮精确计数与来源](BASELINE.json) · [输入输出索引](INDEX.md)']
    write(D / 'COMPARISON.md', '\n'.join(lines))
    print(json.dumps({'status': 'COMPLETE', 'current': current}, ensure_ascii=False))


if __name__ == '__main__':
    main()
