"""Export descriptive results from the completed frozen run; never edit questions."""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
SHARD = HOME / 'shards/shard_01'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    assert not path.exists(), f'append-only output already exists: {path}'
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def percent(value):
    return f'{float(Fraction(value)) * 100:.2f}%' if value is not None else '—'


def accuracy(calls, side):
    n = len(calls)
    exact = sum(c[side]['exact'] for c in calls)
    partial = sum((Fraction(c[side]['partial']) for c in calls), Fraction())
    return {'calls': n, 'exact_correct': exact,
            'exact_accuracy': str(Fraction(exact, n)) if n else None,
            'partial_mean': str(partial/n) if n else None}


def main():
    report = read(SHARD / 'MIXED_EVALUATION_REPORT.json')
    assert report['status'] == 'PASS' and report['raw_scores_and_selection_replayed']
    assert report['calls_per_model'] == 192 and not report['training_started']
    audit_path = HOME / 'evidence/INDEPENDENT_RESULT_AUDIT.json'
    audit = read(audit_path)
    assert audit.get('status') == 'PASS', 'independent result audit must pass'
    pool = read(SHARD / 'EVALUATION_POOL.json')['questions']
    by_id = {q['candidate_id']: q for q in pool}
    selected = read(SHARD / 'SELECTED_CANDIDATES.json')['questions']
    assert {q['candidate_id'] for q in selected} == set(report['selected_candidate_ids'])
    assert all(q == by_id[q['candidate_id']] for q in selected)
    is_new = lambda q: q['question_type'] == 'ordering' or len(q['gold']) < 6
    new_selected = [q for q in selected if is_new(q)]
    controls = [q for q in selected if not is_new(q)]
    calls = read(SHARD / 'PAIRED_CALLS.json')['calls']
    assert len(calls) == 192
    stats = {}
    for label, chosen in [
        ('new_multiselect', lambda q: q['question_type'] == 'multi_select' and is_new(q)),
        ('current_parent_gold6_control', lambda q: q['question_type'] == 'multi_select' and not is_new(q)),
        ('ordering', lambda q: q['question_type'] == 'ordering')]:
        ids = {q['candidate_id'] for q in pool if chosen(q)}
        rows = [c for c in calls if c['candidate_id'] in ids]
        stats[label] = {'question_count': len(ids),
                        'models': {side: accuracy(rows, side) for side in ('trained', 'base')}}
    roots = []
    for suffix in ('25652', '19708'):
        root = 'cc-train-episode-' + suffix
        usable = [q for q in pool if q['root_id'] == root]
        chosen = [q for q in new_selected if q['root_id'] == root]
        ms = [q for q in usable if q['question_type'] == 'multi_select']
        roots.append({'root_id': root,
                      'expanded_new_turn_count': {'25652': 14, '19708': 16}[suffix],
                      'full_source_turn_count': len(read(HOME / f'sources/root{suffix}_SOURCE.json')['turns']),
                      'multiselect_new_usable_by_gold': dict(sorted(Counter(str(len(q['gold'])) for q in ms if is_new(q)).items())),
                      'multiselect_new_selected_by_gold': dict(sorted(Counter(str(len(q['gold'])) for q in chosen if q['question_type'] == 'multi_select').items())),
                      'current_parent_controls_usable': sum(not is_new(q) for q in ms),
                      'ordering_scheduled': 2,
                      'ordering_public': sum(q['root_id'] == root for q in read(SHARD / 'ordering/PUBLIC_QUESTIONS.json')['questions']),
                      'ordering_content_usable': sum(q['question_type'] == 'ordering' for q in usable),
                      'ordering_selected': sum(q['question_type'] == 'ordering' for q in chosen),
                      'new_selected_total': len(chosen)})
    export = {'status': 'PROVISIONAL_REVIEWED_AND_SOLVER_SELECTED',
              'training_allowed': False, 'formal_training_admission': False,
              'control_exclusion_rule': 'Exclude current fresh-parent Gold6 multiselect controls; ordering uses six items and remains a new candidate.',
              'source_pool': binding(SHARD / 'EVALUATION_POOL.json'),
              'source_selection': binding(SHARD / 'SELECTED_CANDIDATES.json'),
              'independent_result_audit': binding(audit_path),
              'question_count': len(new_selected), 'questions': new_selected}
    write(HOME / 'NEW_SELECTED_CANDIDATES.json', export)
    result = {'status': 'COMPLETE', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'upstream_expansion': {'roots_attempted': 6, 'independent_ready': 2, 'new_ready_turns': 30, 'http_calls': 42},
              'qa_generation_http_calls': read(SHARD / 'BATCH_GENERATION.COMPLETE.json')['http_attempts'],
              'candidate_slots_including_controls': 14, 'public_questions_including_controls': 12,
              'content_usable_including_controls': len(pool), 'solver_selected_including_controls': len(selected),
              'new_content_usable': sum(is_new(q) for q in pool),
              'new_selected': len(new_selected), 'selected_controls_excluded_from_new_count': len(controls),
              'new_selected_multiselect_by_gold': dict(sorted(Counter(str(len(q['gold'])) for q in new_selected if q['question_type'] == 'multi_select').items())),
              'new_selected_ordering': sum(q['question_type'] == 'ordering' for q in new_selected),
              'paired_solver_calls_per_model': 192, 'paired_solver_calls_total': 384,
              'new_vs_control_statistics': stats, 'by_root': roots,
              'training_started': False, 'formal_training_admission': False,
              'legacy_role_label_disclosure': 'MIXED_EVALUATION_REPORT uses the inherited role string historical_six_supported_control. Here these are current-source fresh-parent Gold6 controls, not historical question re-evaluation.',
              'initial_finish_review_prerequisite_error': 'Original FINISH_MULTISELECT_REVIEW.log preserved: source judgments existed but private audit serialization had not yet run. Existing serializers were copied, mechanical crosschecks saved, then finish succeeded; no question, Gold, semantic judgment, or model request was changed.',
              'files': [binding(p) for p in (SHARD / 'MIXED_EVALUATION_REPORT.json', audit_path, HOME / 'NEW_SELECTED_CANDIDATES.json')]}
    write(HOME / 'evidence/COMPLETE_STAGE_RESULTS.json', result)
    ms = stats['new_multiselect']['models']
    ctrl = stats['current_parent_gold6_control']['models']
    order = stats['ordering']['models']
    root_lines = []
    for r in roots:
        gs = ', '.join(f"{g}:{n}" for g, n in r['multiselect_new_selected_by_gold'].items())
        root_lines.append(f"| {r['root_id'].split('-')[-1]} | {r['expanded_new_turn_count']} / {r['full_source_turn_count']} | 2/3/4/5 各 1 | {gs} | {r['ordering_scheduled']} / {r['ordering_public']} / {r['ordering_content_usable']} / {r['ordering_selected']} | {r['new_selected_total']} |")
    text = f'''# v96 扩充后多选与排序完整验证

已完成 v95 六 root 扩充 → 两个独立 READY 来源 → 联合出题 → 来源审核 → 冻结 → 两模型配对测试 → 独立结果重算。**新增暂定入选 {len(new_selected)} 题：多选 7 题、排序 1 题；另有 2 道全对对照入选，但不算新增训练题。** 尚未训练或正式准入。

| root | 新增 / 累计轮数 | 新多选内容可用 Gold | 新多选入选 Gold:数量 | 排序计划 / 公开 / 内容可用 / 入选 | 新题入选 |
|---|---:|---|---|---|---:|
{chr(10).join(root_lines)}

两源合计产生 8 道新多选，Gold 2、3、4、5 各 2 道，全部通过逐选项支持/反证、范围、独立事实检查，并与原私有 Gold 一致。25652 的 Gold3 题训练后模型 18/18 全对，按既有难度规则不入选；其他 7 道新多选及 1 道排序入选。因此“内容可用”与“当前 Solver 入选”必须分开。

排序 4 个固定槽位中，19708 两题均因 writer 使用 he/she/him/her/his 等违反已冻结姓名引用政策而机械拒绝；25652 两题通过机械检查，其中 ordering-01 的 I6 将“地图帮助导航”和“游客中心提供最新信息”合并为拿地图提供最新信息，被独立审核保守排除。ordering-02 的六项事实及唯一顺序通过。未删除检查、补写性别、修改原题、重排 Gold 或重试生成来挽救失败。

## 模型表现

| 题目组 | 题数 | 每模型调用 | M0 完全正确率 | M_clean_policy 完全正确率 | M0 部分分均值 | M_clean_policy 部分分均值 |
|---|---:|---:|---:|---:|---:|---:|
| 新多选 Gold2–5 | 8 | {ms['base']['calls']} | {ms['base']['exact_correct']}/{ms['base']['calls']} = {percent(ms['base']['exact_accuracy'])} | {ms['trained']['exact_correct']}/{ms['trained']['calls']} = {percent(ms['trained']['exact_accuracy'])} | {ms['base']['partial_mean']} | {ms['trained']['partial_mean']} |
| 本轮 fresh-parent Gold6 对照 | 2 | {ctrl['base']['calls']} | {ctrl['base']['exact_correct']}/{ctrl['base']['calls']} = {percent(ctrl['base']['exact_accuracy'])} | {ctrl['trained']['exact_correct']}/{ctrl['trained']['calls']} = {percent(ctrl['trained']['exact_accuracy'])} | {ctrl['base']['partial_mean']} | {ctrl['trained']['partial_mean']} |
| 排序 | 1 | {order['base']['calls']} | {order['base']['exact_correct']}/{order['base']['calls']} = {percent(order['base']['exact_accuracy'])} | {order['trained']['exact_correct']}/{order['trained']['calls']} = {percent(order['trained']['exact_accuracy'])} | {order['base']['partial_mean']} | {order['trained']['partial_mean']} |

每模型 192 次、总计 384 次本地 Solver 调用，全文、排列、seed、解码和计分完全配对。多选为 6 个排列 × 3 次，排序为 12 个排列 × 1 次。入选沿用“所有回答有效且训练后部分分均值严格介于 0 和 1”；完全正确率只作描述，未反向修改题目或阈值。原始回答、分数和入选均被独立重算。

这里比较的是既有 M0 与既有 M_clean_policy 在**本轮同一批新题**上的表现，不是本轮数据已经训练后的收益，也不是第一轮旧题的同题复测。六个父事实在同 root 的不同 Gold 子题间复用，8 道多选来自 2 个相关题族；144 次采样不能当成 144 道独立题。不能据此宣称整体难度提升、训练效果显著或全量生产稳定。

引擎沿用的 `historical_six_supported_control` 字串在本轮实际指这两段新扩充的 fresh-parent Gold6 对照，不能将它解释成历史题复测。原报告字段保留，本文及新题导出明确剔除对照。

## 扩充可靠性与生产限制

上游 v95 固定 6 个 root 的实际结果是独立 READY 2/6，共新增 30 轮；4 个失败/隔离案例全部保留。42 次 HTTP 全部 stop，输出 634–3126 tokens，4096 配置未触顶。9 次 planner、4 次 realizer 均使用冻结的上一轮训练反馈。因此本轮扩充瓶颈仍是审核契约错误挡住合法计划、语义偏离被模型审核漏放，不能归因于长度不足。

这两个通过来源有清楚的新事件，但仍有压缩到达过程、历史指代模糊等已披露限制。逐项公共审核再核对私有 Gold 提高了当前题目的可追溯性；非完全盲审、题族相关性与样本量仍限制结论。尤其 24719 在模型接纳后被独立复核隔离，说明暂不能无人审核大批量接纳扩充。

v96 实际联合出题 4 次 HTTP（两次多选资产、两次排序），未用补充调用，未重试或做机械改写。出题既有上限 6144，与扩充的 4096 是不同阶段配置。多选实际历史题面 prior 完整保留 36/35 条；排序实际题面 prior 为 []，虽然历史训练反馈信号完整存在，不能声称排序 writer 看到了全部旧排序题文本。该缺口已记录，未修改冻结输入。

首次 finish-multiselect-review 因尚未序列化私有交叉核对记录而退出，原日志保留。两组公共语义判断此前均已保存；随后使用既有序列化工具核对并写入审核记录，再完成冻结。没有重写判断、题面或 Gold，也未增加模型生成调用。

下一步适合做小批量验证：先最小修复扩充审核的 schema 合规与事件功能漏判，以及排序 writer 的姓名规则执行与事实合并；补齐排序公共历史题面 prior 的冻结输入，再选事先固定的其他 root 跑完整链。保持本轮结果封存，不把本轮失败返工成通过。当前证据支持两个 READY 来源可形成可靠审核后的题目，尚不支持直接无人审核放量。

## 文件索引

- `NEW_SELECTED_CANDIDATES.json`：8 道新题的原样投影，明确不含 Gold6 对照；暂定、不可直接训练。
- `shards/shard_01/SELECTED_CANDIDATES.json`：引擎原始入选，含 2 个对照；`EVALUATION_POOL.json` 为审核通过的 11 题。
- `shards/shard_01/multiselect/PUBLIC_QUESTIONS.json`：10 道公开多选；`assets/*/rounds/initial/requests/` 保存原始 writer 成功/失败资料。
- `shards/shard_01/ordering/*/REJECTIONS.json`、`ordering/CONTENT_REVIEW.json`：机械拒绝及语义排除；原始输出在各 run 的 `requests/`。
- `shards/shard_01/solver/{{trained,base}}/responses/`：全部 Solver 原始回答；`PAIRED_CALLS.json`、`MIXED_EVALUATION_REPORT.json` 为配对结果。
- `evidence/INDEPENDENT_RESULT_AUDIT.json`：独立重算；`ACTUAL_QA_TRANSPORT_PRIOR_AUDIT.json`：实际 HTTP、反馈、完整来源及 prior 审计。
- `evidence/COMPLETE_STAGE_RESULTS.json`、`QUALITY_SYNTHESIS_ZH.md`：阶段统计和题目质量限制。
- `PRESERVATION_BEFORE.json`、`PRESERVATION_AFTER.json`、`COMPLETE_MANIFEST.json`：v95 767 文件保护与本轮封存索引。
- 上游 `../h4_controlled_six_root_expansion_v95/COMPLETE_RESULTS_ZH.md`：六 root 扩充的全部通过/失败结论。
'''
    path = HOME / 'COMPLETE_RESULTS_ZH.md'
    assert not path.exists()
    path.write_text(text)
    print(json.dumps({k: result[k] for k in ['status', 'new_selected', 'new_selected_multiselect_by_gold', 'new_selected_ordering']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
