"""Describe fixed training-unseen-root results and compare earlier diagnostics."""
from datetime import datetime, timezone
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BATCH = HERE.parent / 'h4_unseen_root_validation_v106'
HELPER = HERE.parent / 'h4_batch_codex_benchmark_v104_solver_eval/diagnose.py'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def bind(p):
    return {'path': str(p), 'sha256': sha(p)}


def training_format(config):
    record = next(r for r in config['training_files'] if 'm_clean_policy_' in r['path'])
    manifest = Path(record['path'])
    assert sha(manifest) == record['sha256']
    cache, counts = {}, Counter()
    for line in manifest.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['question_type'] not in ('multi_select', 'ordering'):
            continue
        path = Path(row['candidate_pool_path'])
        if path not in cache:
            assert sha(path) == row['candidate_pool_sha256']
            cache[path] = {q['candidate_id']: q for q in read(path)}
        q = cache[path][row['candidate_id']]
        field = 'options' if row['question_type'] == 'multi_select' else 'items'
        counts[f"{row['question_type']}:{len(q[field])}_choices_or_items"] += 1
    return {'manifest': record, 'shape_counts': dict(counts), 'candidate_pools': [bind(p) for p in sorted(cache)]}


def main():
    outputs = [HERE / n for n in ('DIAGNOSTIC_COMPARISON.json', 'FINDINGS_ZH.md', 'DIAGNOSTIC_MANIFEST.json')]
    assert not any(p.exists() for p in outputs), 'Refuse to overwrite analysis.'
    original_manifest = read(HERE.parent / 'h4_batch_codex_benchmark_v104_solver_eval/DIAGNOSTIC_MANIFEST.json')
    expected_helper = next(r for r in original_manifest['files'] if r['path'] == str(HELPER))
    assert sha(HELPER) == expected_helper['sha256']
    spec = importlib.util.spec_from_file_location('frozen_v104_diagnostic_helper', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    directories = {'v105': HERE.parent / 'h4_codex_controlled_batch_v105_solver_eval',
                   'v104': HERE.parent / 'h4_batch_codex_benchmark_v104_solver_eval', 'v106': HERE}
    batches = {name: helper.summarize(path) for name, path in directories.items()}
    exposure = read(BATCH / 'EXPOSURE_AUDIT.json')
    assert exposure['status'] == 'PASS'
    config = read(BATCH / 'RUN_CONFIG.json')
    assert config['new_source_turns'] == 0 and config['training_started'] is False
    accounting = read(BATCH / 'FINAL_BATCH_ACCOUNTING.json')
    assert accounting['evaluation_counts'] == read(HERE / 'AUDIT_RESULTS.json')['evaluation_counts']
    shape = training_format(config)
    current = batches['v106']
    primary = current['groups']['gold2_3']
    fp_delta = primary['trained']['false_positive_rate'] - primary['base']['false_positive_rate']
    fn_delta = primary['trained']['false_negative_rate'] - primary['base']['false_negative_rate']
    repeats = primary['exact_delta_pp'] < 0 and fp_delta > 0
    lines = ['# v106 首轮微调未见 root：配对验证结果', '',
        '本轮完成来源冻结、六事实配对与排序首稿、不同角色来源先审、完整性封存、原生 v15 Solver 配对评估及独立评分重算。', '',
        f"4 个固定 root，{accounting['evaluation_counts']['candidate_count']} 道内容通过题，每模型 {accounting['evaluation_counts']['calls_per_model']} 次调用；两模型全部调用有效，无新训练、题面修复或 Gold 修改。", '',
        '根为 1456、35167、39120、78460，按历史 common_test 的数字编号升序预选。与两套第一轮训练集的 root_id、source_id、source_record_sha256 均无交集。历史测试旧题已评估，所以这是首轮微调未见 root 上的新题验证，不是从未接触的测试，也不证明基础预训练未见来源。', '',
        '| 分组 | 题数 | 每模型调用 | 训练前 M0 | 训练后 M_clean_policy | 变化（百分点） |',
        '|---|---:|---:|---:|---:|---:|']
    labels = {'gold2':'Gold2', 'gold3':'Gold3', 'gold4':'Gold4', 'gold5':'Gold5', 'gold6':'Gold6 对照',
              'gold2_3':'Gold2–3（主要指标）', 'gold2_5':'Gold2–5', 'multi_all':'全部多选含 Gold6', 'ordering':'排序'}
    for name, row in current['groups'].items():
        b, t = row['base'], row['trained']
        lines.append(f"| {labels[name]} | {b['questions']} | {b['calls']} | {100*b['exact_rate']:.2f}% ({b['exact_correct']}/{b['calls']}) | {100*t['exact_rate']:.2f}% ({t['exact_correct']}/{t['calls']}) | {row['exact_delta_pp']:+.2f} |")
    lines += ['', '## Gold2–3 的选项判断', '',
        '误选率 = 误选选项总次数 / 所有错误选项总次数；漏选率 = 漏选总次数 / 所有正确选项总次数。分母按每次有效调用累计，次数并非独立题量。', '',
        '| 模型 | 误选次数 / 误选率 | 漏选次数 / 漏选率 | 平均选择数 / 平均 Gold | 全选调用 |',
        '|---|---:|---:|---:|---:|']
    for side in ('base', 'trained'):
        m = primary[side]
        lines.append(f"| {side} | {m['fp_options']} / {100*m['false_positive_rate']:.2f}% | {m['fn_options']} / {100*m['false_negative_rate']:.2f}% | {m['mean_selected_count']:.3f} / {m['mean_gold_count']:.3f} | {m['all_select_calls']}/{m['calls']} |")
    changes = primary['paired_exact_changes']
    lines += ['', f"同一请求上，训练后由错变对 {changes.get('improved', 0)} 次、由对变错 {changes.get('worsened', 0)} 次；两者都对 {changes.get('both_correct', 0)} 次、两者都错 {changes.get('both_wrong', 0)} 次。", '',
        f'误选率变化 {100*fp_delta:+.2f} 个百分点，漏选率变化 {100*fn_delta:+.2f} 个百分点。', '',
        '## 与前两批比较', '',
        '| 批次 | 训练 root 暴露 | Gold2–3 正确率变化（百分点） | Gold2–5 变化 | 排序变化 |',
        '|---|---|---:|---:|---:|']
    for name, info in batches.items():
        groups = info['groups']
        lines.append(f"| {name} | {'本次两套首轮训练未见' if name == 'v106' else '训练已见'} | {groups['gold2_3']['exact_delta_pp']:+.2f} | {groups['gold2_5']['exact_delta_pp']:+.2f} | {groups['ordering']['exact_delta_pp']:+.2f} |")
    lines += ['', '跨批次的题目与场景不同，表中的幅度不能直接当作难度变化或严格消融。v106 各 Gold2/3/4/5 等权的多选正确率为 ' +
              f"{100*current['base_gold2_5_equal_gold_exact']:.2f}% → {100*current['trained_gold2_5_equal_gold_exact']:.2f}%。", '',
        '## 按 root 的主要指标', '',
        '| Root | Gold2–3 题数 | M0 正确率 | 训练后正确率 | 变化（百分点） |',
        '|---|---:|---:|---:|---:|']
    for root, groups in current['by_root'].items():
        row = groups['gold2_3']
        b, t = row['base'], row['trained']
        lines.append(f"| {root} | {b['questions']} | {100*b['exact_rate']:.2f}% | {100*t['exact_rate']:.2f}% | {row['exact_delta_pp']:+.2f} |")
    lines += ['', '## 当前判断和下一步', '']
    if repeats:
        lines += ['Gold2–3 的正确率下降和误选率增加，在本轮首轮微调未见 root 上仍重复。它支持继续处理低 Gold 题的过选行为；不能把这一小样本分组问题扩大成整个模型能力下降。', '',
                  '可进入最小修复的准备阶段：保持本批来源与题族仅用于验证，训练侧另取不重叠 root 构造显式矛盾选项和 Gold2/3 覆盖，保持 Gold4/5 与排序回放。冻结训练预算和对照后，才运行小规模修复训练；需要同时检查误选、漏选和排序，避免只靠少选提高低 Gold 正确率。']
    else:
        lines += ['本轮没有同时复现低 Gold 正确率下降与误选增加的完整模式。暂不把前两批现象当作未见 root 上已确认的训练退化；应先扩大预先固定的验证来源或排查题型分布，再决定训练修改。']
    lines += ['', '第一轮 clean 的 172 道多选实际训练 target：Gold2/3/4/5/6 = 28/94/47/3/0；没有 Gold6。这个分布本身不能确定本轮行为变化的成因。', '',
        '另已逐条核对原训练题：172 道多选均为五选项，159 道排序均为四项；当前验证是六选项和六项。因此结论限于当前六选格式，不能直接归因于 Gold 数量或推广为原五选格式同样下降。下一步修复应同时控制题型格式和 Gold 分布，再检验因果解释。', '',
        '通过题为 15 道 Gold2–5、4 道 Gold6、6 道排序。预定 28 道中保留 3 道失败：1456 Gold2 的时间替代未知；39120 排序槽2和78460排序槽1存在可换序的语义匹配。没有根据 Solver 分数补题或换源。', '',
        '限制：仅 4 个 root，相关题族共享事实；多次置换和采样不构成独立题量。本轮命题多用明确否定，主要测试局部 scene 阅读，不能据此证明长对话记忆能力。来源保留重复、时间跳转、称呼与下单状态表达不自然等警告。生成与审核来自同一继承模型且共享项目背景，不是外部模型或人工认证。', '',
        '原始逐调用结果见 SCORED_CALLS.json，逐题评分和执行完整性见 AUDIT_RESULTS.json，公开/私有题、失败首稿和审核证据见源批次。', '']
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'batches': batches,
              'primary_diagnostic': {'group': 'gold2_3', 'exact_delta_pp': primary['exact_delta_pp'],
                                     'false_positive_delta_pp': 100*fp_delta, 'false_negative_delta_pp': 100*fn_delta,
                                     'lower_gold_overselection_pattern_repeated_on_training_unseen_roots': repeats},
              'root_training_isolation': True, 'untouched_test_claimed': False, 'training_started': False,
              'source_or_question_repairs': 0, 'source_config_binding': bind(BATCH / 'RUN_CONFIG.json'),
              'exposure_audit_binding': bind(BATCH / 'EXPOSURE_AUDIT.json'),
              'accounting_binding': bind(BATCH / 'FINAL_BATCH_ACCOUNTING.json'),
              'training_format_audit': shape,
              'helper_binding': bind(HELPER)}
    for path, data in zip(outputs[:2], (json.dumps(result, ensure_ascii=False, indent=2) + '\n', '\n'.join(lines))):
        with path.open('x', encoding='utf-8') as f:
            f.write(data)
    manifest = {'status': 'COMPLETE_DESCRIPTIVE_DIAGNOSTIC',
                'files': [bind(p) for p in (Path(__file__), *outputs[:2])],
                'audited_inputs': [{k: info[k] for k in ('audit', 'scored_calls_binding', 'evaluation_manifest_binding')}
                                   for info in batches.values()],
                'helper': bind(HELPER), 'source_config': result['source_config_binding'],
                'accounting': result['accounting_binding'],
                'exposure_audit': result['exposure_audit_binding']}
    with outputs[2].open('x', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': 'DIAGNOSTIC_COMPLETE', **result['primary_diagnostic']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
