"""Write a dynamic final mixed-batch report after all independent audits pass."""
from pathlib import Path
from collections import Counter
import sys
HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
from common import read, write, bind, now
E = HOME / 'evidence'
audits = {name: read(E / (name + '.json')) for name in ['GENERATION_AUDIT', 'INPUT_AUDIT', 'RESULT_AUDIT', 'UNION_AUDIT']}
assert audits['GENERATION_AUDIT']['status'] in ('PASS', 'PASS_WITH_GENERATION_AND_CONTENT_SHORTFALLS')
assert all(audits[name]['status'] == 'PASS' for name in ['INPUT_AUDIT', 'RESULT_AUDIT', 'UNION_AUDIT'])
new = read(HOME / 'AUDITED_NEW_CANDIDATES.json')
cumulative = read(HOME / 'CUMULATIVE_AUDITED_NEW_CANDIDATES.json')
details = read(E / 'RESULT_DETAILS.json')
report = read(HOME / 'MIXED_EVALUATION_REPORT.json')
allocation = read(HOME / 'multiselect/ALLOCATION.json')
review = read(HOME / 'ordering/CONTENT_REVIEW.json')
source_judgments = read(HOME / 'ordering/SOURCE_JUDGMENTS.json')
config = read(HOME / 'CONFIG.json')
protected = read(HOME / 'PRESERVATION_BEFORE.json')['files']
assert len(protected) == 4939
assert all(bind(Path(path))['sha256'] == digest for path, digest in protected.items())
frozen = {p['path']: p['sha256'] for p in read(HOME / 'BATCH_PREPARATION.COMPLETE.json')['files']}
for path in [*HOME.glob('*.py'), *HOME.glob('*.md')]:
    assert str(path) in frozen and bind(path)['sha256'] == frozen[str(path)], str(path)
for key, name in [('ordering_anchor_plan', 'SCENE_PLANS.json'), ('character_registry', 'CHARACTER_REGISTRY.json'), ('actor_reference_policy', 'ACTOR_REFERENCE_POLICY.json')]:
    assert bind(HOME / name) == config[key], key
assert not list(HOME.rglob('__pycache__'))
newtypes = Counter(q['question_type'] for q in new['questions'])
contenttypes = Counter(q['question_type'] for q in read(HOME / 'EVALUATION_POOL.json')['questions'])
control_count = sum(q['role'] == 'historical_control' for q in details['questions'])
ordering = report['by_type']['ordering']
planned_ordering = sum(s['candidate_count'] for s in config['ordering_slots'])
actor_failures = [{'candidate_id': q['candidate_id'], 'label': label}
                  for q in source_judgments['questions'] for label, item in q['items'].items() if not item['actor_ok']]
lines = [
    'v83 已完成把“人物代词无可靠依据时使用姓名”前置到 writer 的小批混合生成验证。排序继续使用固定单场景、六处预绑定原文锚点与原有完整来源审核；多选沿用 v79 引擎。人物档案保留未知信息，未补写性别或改写原始对话。', '',
    f"本轮新入选 {new['new_question_count']} 道：多选 {newtypes['multi_select']} 道、排序 {newtypes['ordering']} 道。累计候选 {cumulative['candidate_count']} 道：多选 {cumulative['type_distribution'].get('multi_select', 0)} 道、排序 {cumulative['type_distribution'].get('ordering', 0)} 道。累计只计去重后的实际入选题，Gold6 多选对照不算新增。", '',
    '| 多选 Gold 数 | 生成组合数 | 内容审核通过新题 | 本轮新入选 | 累计候选 |',
    '|---|---:|---:|---:|---:|']
for gold in range(2, 6):
    key = str(gold)
    lines.append(f"| {gold} | {allocation['assembled_distribution'].get(key, 0)} | {details['content_usable_new_by_multi_gold'].get(key, 0)} | {new['gold_distribution'].get(key, 0)} | {cumulative['gold_distribution'].get(key, 0)} |")
lines += ['', f"多选生成 {allocation['assembled_child_count']} 个新组合，另有 {allocation['control_count']} 道旧 Gold6 对照；其中 {control_count} 道对照进入 Solver。排序计划 {planned_ordering} 道，机械检查通过 {ordering['mechanically_assembled']} 道，完整来源内容审核通过 {review['accepted_count']} 道、排除 {review['rejected_count']} 道，最终入选 {newtypes['ordering']} 道。", '',
          '| 测评范围 | 训练前 M0 完全正确率 | 第一轮训练后 M_clean_policy 完全正确率 |',
          '|---|---:|---:|']
def format_metric(metric):
    return f"{metric['exact_correct']}/{metric['calls']} = {metric['exact_percent']:.2f}%" if metric['calls'] else '未测评'
for role, label in [('new_multiselect', '本轮新多选'), ('new_ordering', '本轮新排序'),
                    ('new_candidates', '本轮全部新题'), ('historical_control', '旧 Gold6 多选对照'), ('all_questions', '全部测评题')]:
    lines.append(f"| {label} | {format_metric(details['metrics']['base'][role])} | {format_metric(details['metrics']['trained'][role])} |")
lines += ['', '| 新题类型 | 训练前平均部分分 | 训练后平均部分分 |', '|---|---:|---:|']
for role, label in [('new_multiselect', '多选'), ('new_ordering', '排序')]:
    values = [details['metrics'][side][role]['partial_mean'] or '未测评' for side in ['base', 'trained']]
    lines.append(f'| {label} | {values[0]} | {values[1]} |')
lines += ['', f"两模型各 {report['calls_per_model']} 次调用，题目、完整来源、显示排列、种子、解码和评分保持配对。入选仍要求回答全部有效且训练后平均部分分严格位于 0 和 1 之间；完全正确率是描述指标。", '',
          '| 历史参照（仅各批新题） | 训练前完全正确率 | 第一轮训练后完全正确率 |', '|---|---:|---:|']
for version, kind, label in [('v81', 'multi_select', 'v81 新多选'), ('v82', 'ordering', 'v82 三题人物指称修复排序')]:
    values = details['prior_comparisons'][version]['new_questions_by_type'][kind]['models']
    lines.append(f"| {label} | {format_metric(values['base'])} | {format_metric(values['trained'])} |")
lines += ['', '各批题目和场景不同，v82 还是针对旧失败稿的修复样本；以上历史数字用于描述，不能据此把差异归因为难度变化、人物姓名策略效果或训练退化。', '',
          f"排序来源审核覆盖 {len(source_judgments['questions'])} 道、{sum(len(q['items']) for q in source_judgments['questions'])} 个题项，其中 actor_ok 未通过 {len(actor_failures)} 个。这个数字只覆盖进入独立审核的题项；所有原始稿和机械拒绝也保留，机械通过不代替语义正确性。"]
mechanical = details['ordering_generation_failures']
content_failures = details['ordering_content_failures']
if mechanical:
    lines += ['', '排序机械拒绝记录：']
    lines += [f"- {q['candidate_id']}：{'；'.join(q['reasons'])}。" for q in mechanical]
else:
    lines += ['', '本轮排序没有记录到机械拒绝。']
if content_failures:
    lines += ['', '排序内容审核排除记录：']
    lines += [f"- {q['candidate_id']}：{'；'.join(q['reasons'])}。" for q in content_failures]
else:
    lines += ['', '进入审核的排序题没有内容排除记录。']
shortfalls = details['multiselect_unfilled_slots']
if shortfalls:
    lines += ['', '多选未填满的生成目标：']
    lines += [f"- {q['family_id']} Gold{q['gold_count']}：{q['reason']}（可靠反例 {q.get('verified_false_facts', '见原记录')} 个）。" for q in shortfalls]
else:
    lines += ['', '多选生成组合已填满本批目标；内容审核和难度入选数量仍分别记录。']
rejected_assets = [(round_, v) for round_ in details['multiselect_asset_round_reviews'] for v in round_['rejected_versions']]
lines += ['', f"各轮反例审核共记录 {len(rejected_assets)} 个未通过变体，包含唯一允许补写轮的实际结果；原文未提及但可以同时成立的说法不能按矛盾入选。完整失败判断保留在 RESULT_DETAILS 和逐轮 ASSET_REVIEW。"]
excluded = [q for q in details['questions'] if q['role'] != 'historical_control' and not q['models']['trained']['selected_by_policy']]
if excluded:
    lines += ['', '内容通过但未达到难度入选规则的新题：']
    lines += [f"- {q['candidate_id']}（{q['question_type']}）：{q['models']['trained']['selection_reason']}；训练后完全正确率 {q['models']['trained']['full_exact_rate']}、平均部分分 {q['models']['trained']['full_partial_mean']}。" for q in excluded]
else:
    lines += ['', '进入测评的全部新题均达到现有难度入选规则。']
lines += ['', '人物档案沿用 v82 的来源事实快照，未知性别和代词保持未知；当前 writer 使用投影后的姓名策略。Solver 的证据仍是完整原始对话及公开题面，题目不依赖 Solver 看不到的新增人物设定。', '',
          '本轮预算保持最多 6 次生成 HTTP、8 道新多选＋2 道对照＋4 道排序、每模型最多 228 次 Solver 调用。没有自动重试、语义补修或事后修改 Gold。实际调用数见完成清单及独立生成审计。', '',
          '完整来源审核先保存当前判断，再打开私有 Gold 交叉检查；历史来源和父题接触情况按各审核记录披露。来源仍只有两个 root，问题共享既有事实；整组六项文本去重不能证明语义去重或独立事实创新。小批通过率也不能证明批量稳定生成。', '',
          '下一步应根据本轮失败类型决定最小处理：保留已通过的场景、引用和审核规则，修复实际失败环节；扩规模时需要增加独立来源并继续抽检。本轮未启动后续生成、训练或正式入库。', '',
          '独立审计：' + '；'.join(f"{name} {audit['status']}（{audit['check_count']} 检查）" for name, audit in audits.items()) + '。',
          f"v74–v82 共 {len(protected)} 个受保护文件保持字节不变；本批冻结代码、方案、人物档案及指称策略保持不变。全部成功/失败题目、原始回答和审核入口见 ARTIFACT_INDEX.md。"]
with (E / 'RESULTS.md').open('x') as stream:
    stream.write('\n'.join(lines) + '\n')
shortfall = bool(shortfalls or mechanical or content_failures or review['accepted_count'] < planned_ordering)
paths = sorted(p for p in HOME.rglob('*') if p.is_file() and p.name != 'RUN_COMPLETE.json')
record = {'completed_at_utc': now(), 'status': 'COMPLETED_WITH_GENERATION_OR_CONTENT_SHORTFALLS' if shortfall else 'COMPLETE',
          'new_selected_count': new['new_question_count'], 'new_selected_by_type': dict(newtypes),
          'cumulative_candidate_count': cumulative['candidate_count'], 'cumulative_type_distribution': cumulative['type_distribution'],
          'cumulative_multiselect_gold_distribution': cumulative['gold_distribution'],
          'generation_http_calls': audits['GENERATION_AUDIT']['totals']['actual_generation_http_calls'],
          'solver_calls_per_model': report['calls_per_model'], 'total_solver_calls': 2 * report['calls_per_model'],
          'ordering_content_usable_count': review['accepted_count'], 'ordering_new_selected_count': newtypes['ordering'],
          'ordering_review_actor_failed_item_count': len(actor_failures),
          'training_started': False, 'formal_training_admission': False, 'merged_into_baseline': False,
          'preserved_previous_file_count': len(protected), 'frozen_code_and_plan_unchanged': True,
          'frozen_character_registry_and_reference_policy_unchanged': True, 'files': [bind(p) for p in paths]}
write(E / 'RUN_COMPLETE.json', record)
print({k: v for k, v in record.items() if k != 'files'})
