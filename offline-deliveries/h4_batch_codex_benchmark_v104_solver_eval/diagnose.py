"""Compare two completed paired batches; use audited calls, never invoke models."""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'h4_codex_controlled_batch_v105_solver_eval'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(rows):
    assert rows and all(r['evaluation_valid'] for r in rows)
    scores = [r['independent_score'] for r in rows]
    multi = all(r['question_type'] == 'multi_select' for r in rows)
    result = {
        'questions': len({r['candidate_id'] for r in rows}),
        'roots': len({r['root_id'] for r in rows}),
        'calls': len(rows),
        'exact_correct': sum(s['exact_match'] for s in scores),
        'exact_rate': sum(s['exact_match'] for s in scores) / len(rows),
        'partial_mean': float(sum((Fraction(s['partial_score']) for s in scores), Fraction()) / len(rows)),
    }
    if multi:
        gtotal = sum(r['gold_count'] for r in rows)
        negatives = sum(6 - r['gold_count'] for r in rows)
        fp, fn = sum(s['fp'] for s in scores), sum(s['fn'] for s in scores)
        result.update(
            fp_options=fp, fn_options=fn,
            false_positive_rate=fp / negatives if negatives else None,
            false_negative_rate=fn / gtotal,
            mean_selected_count=sum(len(s['display_prediction']) for s in scores) / len(rows),
            mean_gold_count=gtotal / len(rows),
            all_select_calls=sum(len(s['display_prediction']) == 6 for s in scores),
            error_calls_fp_only=sum(s['fp'] > 0 and s['fn'] == 0 for s in scores),
            error_calls_fn_only=sum(s['fn'] > 0 and s['fp'] == 0 for s in scores),
            error_calls_both=sum(s['fn'] > 0 and s['fp'] > 0 for s in scores),
        )
    return result


def group(name, row):
    multi = row['question_type'] == 'multi_select'
    if name == 'ordering':
        return not multi
    if not multi:
        return False
    if name == 'multi_all':
        return True
    if name == 'gold2_5':
        return row['gold_count'] <= 5
    if name == 'gold2_3':
        return row['gold_count'] in (2, 3)
    return row['gold_count'] == int(name[4:])


def summarize(directory):
    sealed = read(directory / 'EVALUATION_COMPLETE_MANIFEST.json')
    assert sealed['status'] == 'PAIRED_SOLVER_EVALUATION_AUDITED_COMPLETE'
    for name in ('AUDIT_RESULTS.json', 'SCORED_CALLS.json'):
        bound = [r for r in sealed['files'] if r['path'] == name]
        assert len(bound) == 1 and bound[0]['sha256'] == sha(directory / name)
    audit = read(directory / 'AUDIT_RESULTS.json')
    calls = read(directory / 'SCORED_CALLS.json')['calls']
    assert audit['status'] == 'VALID' and not audit['data_errors']
    assert all(c['evaluation_valid'] for c in calls)
    result = {'audit': {'path': str(directory / 'AUDIT_RESULTS.json'),
                        'sha256': sha(directory / 'AUDIT_RESULTS.json')},
              'scored_calls_binding': {'path': str(directory / 'SCORED_CALLS.json'),
                                      'sha256': sha(directory / 'SCORED_CALLS.json')},
              'evaluation_manifest_binding': {'path': str(directory / 'EVALUATION_COMPLETE_MANIFEST.json'),
                                               'sha256': sha(directory / 'EVALUATION_COMPLETE_MANIFEST.json')},
              'groups': {}, 'by_root': {}}
    for name in ('gold2', 'gold3', 'gold4', 'gold5', 'gold6', 'gold2_3', 'gold2_5', 'multi_all', 'ordering'):
        subset = [c for c in calls if group(name, c)]
        sides = {side: [r for r in subset if r['model'] == side] for side in ('base', 'trained')}
        m = {side: metric(rows) for side, rows in sides.items()}
        b, t = ({r['request_id']: r for r in sides[side]} for side in ('base', 'trained'))
        assert set(b) == set(t) and len(b) == len(sides['base']) == len(sides['trained'])
        for rid in b:
            assert all(b[rid][k] == t[rid][k] for k in ('candidate_id', 'variant_id', 'slot', 'seed', 'prompt_sha256'))
        m['paired_exact_changes'] = dict(Counter(
            'improved' if t[k]['independent_score']['exact_match'] and not b[k]['independent_score']['exact_match']
            else 'worsened' if b[k]['independent_score']['exact_match'] and not t[k]['independent_score']['exact_match']
            else 'both_correct' if b[k]['independent_score']['exact_match'] else 'both_wrong'
            for k in b))
        m['exact_delta_pp'] = 100 * (m['trained']['exact_rate'] - m['base']['exact_rate'])
        m['partial_delta'] = m['trained']['partial_mean'] - m['base']['partial_mean']
        result['groups'][name] = m
    for root in sorted({r['root_id'] for r in calls}):
        result['by_root'][root] = {}
        for name in ('gold2_3', 'gold2_5', 'ordering'):
            rows = [r for r in calls if r['root_id'] == root and group(name, r)]
            if rows:
                m = {s: metric([r for r in rows if r['model'] == s]) for s in ('base', 'trained')}
                m['exact_delta_pp'] = 100 * (m['trained']['exact_rate'] - m['base']['exact_rate'])
                result['by_root'][root][name] = m
    for side in ('base', 'trained'):
        result[side + '_gold2_5_equal_gold_exact'] = sum(result['groups']['gold' + str(g)][side]['exact_rate'] for g in range(2, 6)) / 4
    result['question_changes'] = audit['comparison']['by_question']
    return result


def main():
    outputs = [HERE / n for n in ('DIAGNOSTIC_COMPARISON.json', 'FINDINGS_ZH.md', 'DIAGNOSTIC_MANIFEST.json')]
    assert not any(p.exists() for p in outputs), 'Exclusive output; do not overwrite analyses.'
    batches = {'v105': summarize(PREVIOUS), 'v104': summarize(HERE)}
    lines = ['# v104 / v105 Solver 配对复核', '',
        'M0 是训练前的 Qwen3-4B；M_clean_policy 是同一个已训练检查点。两批均沿用 v15 同题、同显示置换、同采样种子和评分协议。', '',
        'v104 与 v105 的 root 不同，但 v104 的 4 个 root 均曾进入第一轮训练。它是已见 root 上的新题诊断，不能当作未见 root 的泛化验证。', '',
        '| 批次 | 题型 / Gold | 题数 | 每模型调用 | M0 正确率 | 训练后正确率 | 变化（百分点） |',
        '|---|---|---:|---:|---:|---:|---:|']
    labels = {'gold2':'Gold2', 'gold3':'Gold3', 'gold4':'Gold4', 'gold5':'Gold5', 'gold6':'Gold6 对照',
              'gold2_3':'Gold2–3', 'gold2_5':'Gold2–5', 'multi_all':'全部多选（含 Gold6）', 'ordering':'排序'}
    for batch, info in batches.items():
        for name, row in info['groups'].items():
            b, t = row['base'], row['trained']
            lines.append(f"| {batch} | {labels[name]} | {b['questions']} | {b['calls']} | {100*b['exact_rate']:.2f}% | {100*t['exact_rate']:.2f}% | {row['exact_delta_pp']:+.2f} |")
    lines += ['', 'Gold2–5 中各 Gold 数量不同，因此两批之间的整体比例不宜直接解释成难度变化。各 Gold 等权也不能消除题面、root 和场景差异。按 Gold2/3/4/5 四组等权计算：', '',
              '| 批次 | M0 | 训练后 | 变化（百分点） |', '|---|---:|---:|---:|']
    for batch, info in batches.items():
        b, t = info['base_gold2_5_equal_gold_exact'], info['trained_gold2_5_equal_gold_exact']
        lines.append(f'| {batch} | {100*b:.2f}% | {100*t:.2f}% | {100*(t-b):+.2f} |')
    lines += ['', '## Gold2–3 误选与漏选', '',
        '误选率以全部错误选项为分母；漏选率以全部正确选项为分母。次数为选项累计次数，不是独立题目数。', '',
        '| 批次 | 模型 | 误选次数 / 误选率 | 漏选次数 / 漏选率 | 平均选项数 / 平均 Gold | 全选调用 |',
        '|---|---|---:|---:|---:|---:|']
    for batch, info in batches.items():
        for side in ('base', 'trained'):
            m = info['groups']['gold2_3'][side]
            lines.append(f"| {batch} | {side} | {m['fp_options']} / {100*m['false_positive_rate']:.2f}% | {m['fn_options']} / {100*m['false_negative_rate']:.2f}% | {m['mean_selected_count']:.3f} / {m['mean_gold_count']:.3f} | {m['all_select_calls']}/{m['calls']} |")
    lines += ['', '## v104 按 root 检查', '',
        '| Root | 分组 | 题数 | M0 正确率 | 训练后正确率 | 变化（百分点） |', '|---|---|---:|---:|---:|---:|']
    for root, groups in batches['v104']['by_root'].items():
        for name, row in groups.items():
            b, t = row['base'], row['trained']
            lines.append(f"| {root} | {labels[name]} | {b['questions']} | {100*b['exact_rate']:.2f}% | {100*t['exact_rate']:.2f}% | {row['exact_delta_pp']:+.2f} |")
    old, new = (batches[v]['groups']['gold2_3'] for v in ('v105', 'v104'))
    repeated = old['exact_delta_pp'] < 0 and new['exact_delta_pp'] < 0
    lines += ['', '## 对下一步的含义', '']
    if repeated:
        lines += ['Gold2–3 的下降在另一组 root 上重复，说明需要针对这一分组继续诊断；这仍不能推出训练整体退化。']
    else:
        lines += ['v105 中 Gold2–3 的下降没有在 v104 中同向重复。目前不宜据此启动训练修复或宣称模型整体退化。']
    for batch, info in batches.items():
        m = info['groups']['gold2_3']
        fp_delta = 100 * (m['trained']['false_positive_rate'] - m['base']['false_positive_rate'])
        fn_delta = 100 * (m['trained']['false_negative_rate'] - m['base']['false_negative_rate'])
        lines.append(f'{batch} 的 Gold2–3：训练后误选率变化 {fp_delta:+.2f} 个百分点，漏选率变化 {fn_delta:+.2f} 个百分点。')
    exposure = HERE / 'EXPOSURE_AUDIT.json'
    if exposure.exists():
        distribution = read(exposure)['training_gold_distribution']
        lines += ['', '第一轮训练的 Gold 分布核查记录：', '',
                  '```json', json.dumps(distribution, ensure_ascii=False, indent=2), '```', '',
                  'clean 的多选训练共 172 条，Gold2/3/4/5/6 分别为 28/94/47/3/0，并已逐条核对实际训练 target。这不支持直接把本轮现象归因于第一轮 Gold6 或全选样本过多。']
    lines += ['', '建议下一步先固定未进入训练的 root 和评估题，再运行同一配对协议；已有候选 root194259 尚需明确其 provisional 历史和内容审核边界。若在固定验证集确认同一问题，再对数据覆盖或选择策略做最小修复，并保持验证题与训练材料分离。']
    lines += ['', '本诊断只给出这两批题的观察结果。同一题的多次置换与采样共享原文，不能将调用数当作独立题量作显著性证明；相关题族也共享同一新增场景。', '',
        '是否修改训练方案，应结合未见 root 的预先固定评估确认。本轮没有训练、修改题面或 Gold，也没有自动导出训练集。', '']
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'batches': batches,
              'scope': 'Descriptive paired diagnostics on previously trained roots; not held-out generalization.',
              'training_started': False, 'gold_or_public_edits': False,
              'exposure_audit': {'path': str(exposure), 'sha256': sha(exposure)} if exposure.exists() else None}
    for path, value in zip(outputs[:2], (json.dumps(result, ensure_ascii=False, indent=2) + '\n', '\n'.join(lines))):
        with path.open('x', encoding='utf-8') as stream:
            stream.write(value)
    manifest = {'status': 'COMPLETE_DESCRIPTIVE_ANALYSIS',
                'files': [{'path': str(p), 'sha256': sha(p)} for p in [Path(__file__), *outputs[:2]]],
                'audited_inputs': [{k: v[k] for k in ('audit', 'scored_calls_binding', 'evaluation_manifest_binding')}
                                   for v in batches.values()]}
    with outputs[2].open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: {g: r['exact_delta_pp'] for g, r in v['groups'].items()} for k, v in batches.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
