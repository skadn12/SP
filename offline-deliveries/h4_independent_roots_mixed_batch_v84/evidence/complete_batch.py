"""Close v84 from actual audited artifacts; never generate questions or run models.

Run --check-only before closeout, or --verify-complete after it. Finalization
writes its own index/report and then binds every remaining batch file. Do not
redirect its stdout to a file inside this batch: that file would still be open
after the final manifest is saved.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
from common import bind, need, now, read, verify_bindings, write

PREVIOUS = HOME.parent / 'h4_name_policy_mixed_batch_v83'
EXPECTED_ROOTS = {f'cc-train-episode-{n}' for n in (103866, 109164, 112812, 135466)}
AUDIT_NAMES = ('GENERATION_AUDIT', 'INPUT_AUDIT', 'RESULT_AUDIT', 'UNION_AUDIT')
OUTPUT_NAMES = ('ARTIFACT_INDEX.md', 'RESULTS.md', 'RUN_COMPLETE.json')
DEPENDENCIES = (
    'CONFIG.json', 'PRESERVATION_BEFORE.json', 'BATCH_PREPARATION.COMPLETE.json',
    'SOURCE_ADMISSION_SUMMARY.json', 'SOURCE_ADMISSION.COMPLETE.json',
    'AUDITED_NEW_CANDIDATES.json', 'CUMULATIVE_AUDITED_NEW_CANDIDATES.json',
    'EVALUATION_POOL.json', 'MIXED_EVALUATION_REPORT.json', 'PAIRED_CALLS.json',
    'multiselect/ALLOCATION.json', 'multiselect/FINAL_CONTENT_REVIEW.json',
    'multiselect/FINAL_CONTENT_REVIEW.COMPLETE.json', 'BATCH_REVIEW.COMPLETE.json',
    'ordering/CONTENT_REVIEW.json',
    'ordering/SOURCE_JUDGMENTS.json', 'evidence/RESULT_DETAILS.json',
    *(f'evidence/{n}.json' for n in AUDIT_NAMES),
)


def inventory_stats(value, count_key):
    questions = value['questions']
    need(len({q['candidate_id'] for q in questions}) == len(questions), 'duplicate_export_candidate')
    counts = Counter(q['question_type'] for q in questions)
    golds = Counter(str(len(q['gold'])) for q in questions if q['question_type'] == 'multi_select')
    need(value[count_key] == len(questions), 'export_count_mismatch')
    need(all(value['type_distribution'].get(k, 0) == counts[k] for k in set(counts) | set(value['type_distribution'])), 'export_type_distribution_mismatch')
    for key in ('gold_distribution', 'multiselect_gold_distribution'):
        need(all(value[key].get(k, 0) == golds[k] for k in set(golds) | set(value[key])), 'export_multiselect_gold_distribution_mismatch:' + key)
    need(all(q['question_type'] in ('multi_select', 'ordering') for q in questions), 'unknown_export_type')
    need(all(len(q['gold']) < 6 for q in questions if q['question_type'] == 'multi_select'), 'historical_control_in_new_export')
    return counts, golds


def verify_preserved(home):
    protected = read(home / 'PRESERVATION_BEFORE.json')['files']
    need(len(protected) == 5797, 'unexpected_v74_to_v83_preservation_count')
    for path, digest in protected.items():
        need(bind(path)['sha256'] == digest, 'protected_file_changed:' + path)
    delivery_roots = {home.parent / Path(p).relative_to(home.parent).parts[0] for p in protected}
    present = {str(p.resolve()) for root in delivery_roots for p in root.rglob('*') if p.is_file()}
    need(present == set(protected), 'protected_directory_added_or_removed_file')
    return protected


def verify_exact_bindings(value):
    """Recheck bound artifact objects, excluding checkpoint records with stat data."""
    bindings = {}

    def visit(obj):
        if isinstance(obj, dict):
            if set(obj) == {'path', 'sha256'}:
                need(obj['path'] not in bindings or bindings[obj['path']] == obj['sha256'], 'conflicting_artifact_binding')
                bindings[obj['path']] = obj['sha256']
            else:
                for item in obj.values():
                    visit(item)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(value)
    verify_bindings([{'path': p, 'sha256': digest} for p, digest in bindings.items()])


def load_state(home=HOME):
    missing = [str(home / name) for name in DEPENDENCIES if not (home / name).is_file()]
    need(not missing, 'closeout_dependencies_missing:' + ','.join(missing))
    e = home / 'evidence'
    audits = {name: read(e / (name + '.json')) for name in AUDIT_NAMES}
    need(audits['GENERATION_AUDIT']['status'] in ('PASS', 'PASS_WITH_GENERATION_AND_CONTENT_SHORTFALLS'), 'generation_audit_not_pass')
    need(all(audits[n]['status'] == 'PASS' for n in AUDIT_NAMES[1:]), 'required_audit_not_pass')
    need(all(not audits[n]['failed_checks'] for n in AUDIT_NAMES), 'audit_has_failed_checks')
    need(all(audits[n]['check_count'] == len(audits[n]['checks']) for n in AUDIT_NAMES), 'audit_check_count_mismatch')
    state = {'audits': audits, 'new': read(home / 'AUDITED_NEW_CANDIDATES.json'),
             'cumulative': read(home / 'CUMULATIVE_AUDITED_NEW_CANDIDATES.json'),
             'details': read(e / 'RESULT_DETAILS.json'), 'report': read(home / 'MIXED_EVALUATION_REPORT.json'),
             'allocation': read(home / 'multiselect/ALLOCATION.json'),
             'multiselect_review': read(home / 'multiselect/FINAL_CONTENT_REVIEW.json'),
             'multiselect_review_complete': read(home / 'multiselect/FINAL_CONTENT_REVIEW.COMPLETE.json'),
             'review': read(home / 'ordering/CONTENT_REVIEW.json'),
             'source_judgments': read(home / 'ordering/SOURCE_JUDGMENTS.json'),
             'config': read(home / 'CONFIG.json'), 'pool': read(home / 'EVALUATION_POOL.json'),
             'admission': read(home / 'SOURCE_ADMISSION_SUMMARY.json'),
             'previous': read(PREVIOUS / 'CUMULATIVE_AUDITED_NEW_CANDIDATES.json')}
    state['protected'] = verify_preserved(home)
    prepared = read(home / 'BATCH_PREPARATION.COMPLETE.json')
    need(prepared['status'] == 'PREPARED' and not prepared['errors'], 'batch_not_prepared')
    frozen = {p['path']: p['sha256'] for p in prepared['files']}
    verify_bindings(prepared['files'])
    for path in [*home.glob('*.py'), *home.glob('*.md'), home / 'CONFIG.json']:
        need(str(path.resolve()) in frozen and bind(path)['sha256'] == frozen[str(path.resolve())], 'frozen_production_changed:' + str(path))
    cfg = state['config']
    for key, filename in [('ordering_anchor_plan', 'SCENE_PLANS.json'), ('character_registry', 'CHARACTER_REGISTRY.json'), ('actor_reference_policy', 'ACTOR_REFERENCE_POLICY.json')]:
        need(bind(home / filename) == cfg[key], 'frozen_policy_binding_changed:' + key)
    roots = {s['root_id'] for s in cfg['ordering_slots']}
    need(roots == EXPECTED_ROOTS == {s['root_id'] for s in cfg['multiselect_config']['slots']}, 'unexpected_v84_train_roots')
    need(set(state['admission']['ready_roots']) == roots and not state['admission']['held_roots'], 'source_admission_not_ready')
    need(all(s['split'] == 'train' for s in cfg['multiselect_config']['slots']), 'v84_source_split_changed')
    need(not list(home.rglob('__pycache__')), 'batch_contains_bytecode_cache')
    state['newtypes'], state['newgolds'] = inventory_stats(state['new'], 'new_question_count')
    state['cumtypes'], state['cumgolds'] = inventory_stats(state['cumulative'], 'candidate_count')
    oldtypes, oldgolds = inventory_stats(state['previous'], 'candidate_count')
    need(state['previous']['candidate_count'] == 24 and dict(oldtypes) == {'multi_select': 20, 'ordering': 4}, 'previous_v83_inventory_changed')
    state['previous_types'], state['previous_golds'] = oldtypes, oldgolds
    report, details, new, cumulative = (state[k] for k in ('report', 'details', 'new', 'cumulative'))
    multi_review, multi_complete = state['multiselect_review'], state['multiselect_review_complete']
    reviewed_multi = multi_review['questions']
    multi_ids = {q['candidate_id'] for q in reviewed_multi}
    usable_multi = {q['candidate_id'] for q in reviewed_multi if q['core_status'] == 'CORE_USABLE'}
    need(len(multi_ids) == len(reviewed_multi), 'duplicate_multiselect_review_candidate')
    need(multi_complete['status'] == 'FROZEN' and multi_complete['before_any_solver'] is True and multi_complete['solver_scores_used'] is False, 'multiselect_review_not_frozen_before_solver')
    need(multi_complete['candidate_count'] == len(reviewed_multi) and multi_complete['usable_count'] == len(usable_multi), 'multiselect_review_count_mismatch')
    need(set(multi_complete['usable_candidate_ids']) == usable_multi, 'multiselect_review_usable_ids_mismatch')
    need(multi_complete['statuses'] == {q['candidate_id']: q['core_status'] for q in reviewed_multi}, 'multiselect_review_status_mismatch')
    need(usable_multi == {q['candidate_id'] for q in state['pool']['questions'] if q['question_type'] == 'multi_select'}, 'multiselect_pool_review_mismatch')
    expected_multi_failures = [{'candidate_id': q['candidate_id'], 'root_id': q['root_id'], 'core_status': q['core_status'], 'reason': q['reason']}
                               for q in reviewed_multi if q['core_status'] != 'CORE_USABLE']
    need(details['multiselect_content_failures'] == expected_multi_failures, 'multiselect_content_failure_detail_mismatch')
    need(report['by_type']['multi_select']['mechanically_assembled'] == len(reviewed_multi), 'multiselect_assembly_review_count_mismatch')
    need(report['by_type']['multi_select']['content_excluded'] == len(expected_multi_failures), 'multiselect_content_excluded_count_mismatch')
    need(report['status'] == 'PASS' and report['evaluation_status'] == 'COMPLETED', 'evaluation_not_complete')
    need(report['selected_new_candidate_count'] == new['new_question_count'] == details['new_selected_count'], 'new_selected_count_mismatch')
    need(audits['UNION_AUDIT']['previous_cumulative_count'] == state['previous']['candidate_count'], 'union_previous_count_mismatch')
    need(audits['UNION_AUDIT']['cumulative_count'] == cumulative['candidate_count'], 'union_cumulative_count_mismatch')
    need(audits['UNION_AUDIT']['current_new_count'] == new['new_question_count'], 'union_new_count_mismatch')
    need(audits['RESULT_AUDIT']['selected_new_count'] == new['new_question_count'], 'result_audit_new_count_mismatch')
    need(report['calls_per_model'] == audits['INPUT_AUDIT']['expected_calls_per_model'] == audits['RESULT_AUDIT']['expected_calls_per_model'], 'solver_call_count_mismatch')
    need(report['calls_per_model'] <= cfg['max_solver_calls_per_model'], 'solver_budget_exceeded')
    need(audits['GENERATION_AUDIT']['totals']['actual_generation_http_calls'] <= cfg['max_generation_http'], 'generation_budget_exceeded')
    for side in ('base', 'trained'):
        need(details['metrics'][side]['all_questions']['calls'] == report['calls_per_model'], 'details_call_count_mismatch')
    need(len(state['pool']['questions']) <= cfg['max_candidates'], 'candidate_budget_exceeded')
    need(all(value['training_allowed'] is False and value['training_started'] is False and value['formal_training_admission'] is False and value['merged_into_baseline'] is False for value in (new, cumulative)), 'export_training_policy_changed')
    need(cfg['training_allowed'] is False and report['training_started'] is False and report['formal_training_admission'] is False, 'batch_training_policy_changed')
    review_complete = read(home / 'BATCH_REVIEW.COMPLETE.json')
    need(review_complete['status'] == 'FROZEN' and review_complete['before_any_solver'] is True and review_complete['solver_scores_used'] is False, 'batch_review_not_frozen_before_solver')
    for value in (audits, state['new'], state['cumulative'], details, report, state['admission'], multi_review, multi_complete, review_complete):
        verify_exact_bindings(value)
    state['planned_ordering'] = sum(s['candidate_count'] for s in cfg['ordering_slots'])
    state['actor_failures'] = [{'candidate_id': q['candidate_id'], 'label': label} for q in state['source_judgments']['questions'] for label, item in q['items'].items() if not item['actor_ok']]
    return state


def link(path, label=None):
    path = Path(path).resolve()
    target = '<' + str(path) + '>' if ' ' in str(path) else str(path)
    return f'[{label or path.name}]({target})'


def index_text(state, home=HOME):
    lines = ['v84实际产物入口；成功稿、失败稿、原始调用与审核均保留。RUN_COMPLETE保存本批每个文件的最终绑定。', '',
             f"本轮入选 {state['new']['new_question_count']} 道，累计 {state['cumulative']['candidate_count']} 道。", '',
             '入选与结果：', '']
    for name in ('AUDITED_NEW_CANDIDATES.json', 'CUMULATIVE_AUDITED_NEW_CANDIDATES.json', 'EVALUATION_POOL.json', 'MIXED_EVALUATION_REPORT.json', 'PAIRED_CALLS.json', 'evidence/RESULT_DETAILS.json', 'evidence/RESULTS.md', 'evidence/RUN_COMPLETE.json'):
        lines.append('- ' + link(home / name, name))
    lines += ['', '完整原始对话与来源门禁：', '', '| root | 完整轮数 | 原始对话 | 来源审核 |', '|---|---:|---|---|']
    for root in state['admission']['roots']:
        lines.append(f"| {root['root_id']} | {root['turn_count']} | {link(root['source']['path'], 'SOURCE.json')} | {link(root['review']['path'], 'REVIEW.json')} |")
    lines += ['', '父题、多选逐轮生成与整题审核、排序原稿与拒绝记录：', '']
    primary = ('SOURCE_ADMISSION_SUMMARY.json', 'SOURCE_ADMISSION.COMPLETE.json', 'SCENE_PLANS.json', 'CHARACTER_REGISTRY.json', 'ACTOR_REFERENCE_POLICY.json', 'BATCH_PREPARATION.COMPLETE.json', 'BATCH_REVIEW.COMPLETE.json', 'multiselect/ALLOCATION.json', 'multiselect/PUBLIC_QUESTIONS.json', 'multiselect/FINAL_CONTENT_REVIEW.json', 'multiselect/FINAL_CONTENT_REVIEW.COMPLETE.json', 'ordering/PUBLIC_QUESTIONS.json', 'ordering/SOURCE_JUDGMENTS.json', 'ordering/CONTENT_REVIEW.json')
    for name in primary:
        if (home / name).is_file():
            lines.append('- ' + link(home / name, name))
    relevant_names = {'PUBLIC_ASSETS.json', 'ASSETS.json', 'SOURCE_JUDGMENTS.json', 'ASSET_REVIEW.json', 'ROUND_PROVENANCE.json', 'REJECTIONS.json', 'REQUEST.json', 'RESPONSE.json', 'HTTP_ATTEMPT.json', 'PROOF.json'}
    for directory in ('parent_review', 'multiselect/assets', 'multiselect/final_review', 'ordering'):
        paths = sorted(p for p in (home / directory).rglob('*') if p.is_file() and (p.name in relevant_names or directory in ('parent_review', 'multiselect/final_review') or p.name.startswith('RAW')))
        if paths:
            lines += ['', f'{directory}：', '']
            lines += ['- ' + link(p, str(p.relative_to(home))) for p in paths]
    lines += ['', '配对Solver结果与审计：', '']
    for side in ('trained', 'base'):
        for filename in ('manifest.json', 'RESULTS.json', 'COMPLETE.json', 'solver_identity.json'):
            path = home / 'solver' / side / filename
            if path.is_file():
                lines.append('- ' + link(path, str(path.relative_to(home))))
        raw_dir = home / 'solver' / side / 'responses'
        if raw_dir.is_dir():
            lines.append('- ' + link(raw_dir, f'solver/{side}/responses（{len(list(raw_dir.glob("*.json")))}份逐次原始答复）'))
        prompt_dir = home / 'solver' / side / 'prompts'
        if prompt_dir.is_dir():
            lines.append('- ' + link(prompt_dir, f'solver/{side}/prompts（逐次实际输入）'))
    for path in sorted((home / 'evidence').glob('*RUNTIME*.json')):
        lines.append('- ' + link(path, str(path.relative_to(home))))
    for name in AUDIT_NAMES:
        lines.append('- ' + link(home / 'evidence' / (name + '.json'), name))
    lines += ['', '整组六项文本去重不等于事实独立或语义去重。四个来源属于已知历史TRAIN roots，未建立独立测试集；此次只生成、审核、配对测评和筛选，未训练或正式入库。']
    return '\n'.join(lines) + '\n'


def format_metric(metric):
    return f"{metric['exact_correct']}/{metric['calls']} = {metric['exact_percent']:.2f}%" if metric['calls'] else '未测评'


def results_text(s):
    new, cumulative, details, cfg = (s[k] for k in ('new', 'cumulative', 'details', 'config'))
    allocation, review, report = (s[k] for k in ('allocation', 'review', 'report'))
    newtypes, cumtypes = s['newtypes'], s['cumtypes']
    control_count = sum(q['role'] == 'historical_control' for q in details['questions'])
    ordering = report['by_type']['ordering']
    lines = [f"v84联合扩充已完成。本轮新入选 {new['new_question_count']} 道：多选 {newtypes['multi_select']} 道、排序 {newtypes['ordering']} 道；累计 {cumulative['candidate_count']} 道：多选 {cumtypes['multi_select']} 道、排序 {cumtypes['ordering']} 道。前一批v83累计为 {s['previous']['candidate_count']} 道（多选 {s['previous_types']['multi_select']}、排序 {s['previous_types']['ordering']}）。", '',
             '本批增加103866、109164、112812、135466四个不同来源，均为已知历史TRAIN roots，不是独立测试来源。完整来源保持原始字节，重新进行来源门禁与当前公开父题审核，历史Gold先验已披露。多选沿用v79；排序保持单场景、六处预绑定原文锚点及完整来源审核，并保留v83姓名策略。', '',
             '| 多选Gold数 | 计划组合 | 实际生成组合 | 内容通过新题 | 本轮新入选 | 累计候选 |', '|---|---:|---:|---:|---:|---:|---:|']
    for gold in range(2, 6):
        key = str(gold)
        lines.append(f"| {gold} | {cfg['multiselect_config']['new_child_target_distribution'].get(key, 0)} | {allocation['assembled_distribution'].get(key, 0)} | {details['content_usable_new_by_multi_gold'].get(key, 0)} | {new['gold_distribution'].get(key, 0)} | {cumulative['gold_distribution'].get(key, 0)} |")
    lines += ['', f"多选实际生成 {allocation['assembled_child_count']} 个新组合，另有 {allocation['control_count']} 道历史Gold6对照；其中 {control_count} 道对照进入Solver。Gold6对照不计新增，排序的六个题项不混入多选Gold分布。", '',
              f"排序计划 {s['planned_ordering']} 道，机械通过 {ordering['mechanically_assembled']} 道，内容通过 {review['accepted_count']} 道、内容拒绝 {review['rejected_count']} 道，难度筛选后入选 {newtypes['ordering']} 道。", '',
              '| 测评范围 | 训练前M0完全正确率 | 第一轮训练后M_clean_policy完全正确率 |', '|---|---:|---:|']
    for role, label in [('new_multiselect', '本轮新多选'), ('new_ordering', '本轮新排序'), ('new_candidates', '本轮全部新题'), ('historical_control', '历史Gold6对照'), ('all_questions', '全部测评题')]:
        lines.append(f"| {label} | {format_metric(details['metrics']['base'][role])} | {format_metric(details['metrics']['trained'][role])} |")
    lines += ['', '| 新题类型 | 训练前平均部分分 | 训练后平均部分分 |', '|---|---:|---:|']
    for role, label in [('new_multiselect', '多选'), ('new_ordering', '排序')]:
        values = [details['metrics'][side][role]['partial_mean'] if details['metrics'][side][role]['calls'] else '未测评' for side in ('base', 'trained')]
        lines.append(f'| {label} | {values[0]} | {values[1]} |')
    lines += ['', f"两模型各 {report['calls_per_model']} 次真实调用，完整来源、公开题面、显示排列、种子、解码及评分配对。入选要求全部回答有效且训练后平均部分分严格在0与1之间；完全正确率只作描述。本轮使用固定第一轮checkpoint，没有再次训练。", '',
              '| 历史参照（仅各批新题） | 训练前完全正确率 | 第一轮训练后完全正确率 |', '|---|---:|---:|']
    for version, prior in details['prior_comparisons'].items():
        for kind, label in [('multi_select', '新多选'), ('ordering', '新排序')]:
            entry = prior['new_questions_by_type'][kind]
            if entry['question_count']:
                values = entry['models']
                lines.append(f"| {version} {label}（{entry['question_count']}题） | {format_metric(values['base'])} | {format_metric(values['trained'])} |")
    lines += ['', '各批题目、来源与场景不同，v82是针对旧失败稿的修复样本。不同批次准确率差异不能直接归因为难度变化、姓名策略或训练退化；同一题的排列与种子重复也不是独立题目。', '',
              f"排序源审覆盖 {len(s['source_judgments']['questions'])} 道、{sum(len(q['items']) for q in s['source_judgments']['questions'])} 个题项，actor_ok未通过 {len(s['actor_failures'])} 个。原始稿和机械拒绝全部保留；姓名规则通过不能代替事实、人物归属、状态和唯一顺序的审核。"]
    for key, title in [('ordering_generation_failures', '排序机械拒绝'), ('ordering_content_failures', '排序内容拒绝')]:
        lines += ['', title + '：', '']
        entries = details[key]
        lines += [f"- {q['candidate_id']}：{'；'.join(q['reasons'])}。" for q in entries] if entries else ['本阶段没有拒绝记录。']
    lines += ['', '多选整题内容拒绝：', '']
    lines += [f"- {q['candidate_id']}（{q['core_status']}）：{q['reason']}。" for q in details['multiselect_content_failures']] if details['multiselect_content_failures'] else ['本阶段没有拒绝记录。']
    lines += ['', '多选未填满目标：', '']
    shortfalls = details['multiselect_unfilled_slots']
    lines += [f"- {q['family_id']} Gold{q['gold_count']}：{q['reason']}（可靠反例 {q.get('verified_false_facts', '见原记录')} 个）。" for q in shortfalls] if shortfalls else ['多选生成目标已填满；内容审核与难度筛选仍分别记录。']
    rejected = [(r, item) for r in details['multiselect_asset_round_reviews'] for item in r['rejected_versions']]
    supplement_count = sum(r['round_id'] != 'initial' for r in details['multiselect_asset_round_reviews'])
    lines += ['', f"逐轮反例审核记录 {len(rejected)} 个未通过变体，实际补写 {supplement_count} 轮；原文未提及但可同时成立的说法仍判unknown，不能凑成矛盾。实际生成与补写内容见逐轮PUBLIC_ASSETS、原始调用和ASSET_REVIEW。"]
    excluded = [q for q in details['questions'] if q['role'] != 'historical_control' and not q['models']['trained']['selected_by_policy']]
    lines += ['', '内容通过但未达到难度入选规则的新题：', '']
    lines += [f"- {q['candidate_id']}（{q['question_type']}）：{q['models']['trained']['selection_reason']}；训练后完全正确率 {q['models']['trained']['full_exact_rate']}、平均部分分 {q['models']['trained']['full_partial_mean']}。" for q in excluded] if excluded else ['进入测评的新题全部达到现有难度入选规则。']
    lines += ['', f"配置预算为最多 {cfg['max_generation_http']} 次生成HTTP、{cfg['max_candidates']} 道测评候选、每模型 {cfg['max_solver_calls_per_model']} 次Solver调用；实际生成HTTP {s['audits']['GENERATION_AUDIT']['totals']['actual_generation_http_calls']} 次。保留每family最多一次补写限制，无自动重试、事后修Gold、训练或正式入库。", '',
              '整组六项文本去重不能证明语义去重或独立事实创新。四个新增来源已增加root覆盖，但属于已知TRAIN数据；后续独立、固定且Gold均衡的测试集仍需单独建立。短缺、语义拒绝和难度拒绝分别统计，不由数量目标改变审核标准。', '',
              '独立审计：' + '；'.join(f"{name} {audit['status']}（{audit['check_count']}检查）" for name, audit in s['audits'].items()) + '。', '',
              f"v74–v83共 {len(s['protected'])} 个受保护文件保持字节不变，受保护目录也没有新增或删除文件；当前冻结生产代码、配置、场景、人物档案和指称策略保持不变。全部结果入口见ARTIFACT_INDEX.md。"]
    return '\n'.join(lines) + '\n'


def verify_completion(home=HOME):
    path = home / 'evidence/RUN_COMPLETE.json'
    record = read(path)
    actual = {str(p.resolve()) for p in home.rglob('*') if p.is_file() and p.resolve() != path.resolve()}
    bindings = record['files']
    need(len(bindings) == len({r['path'] for r in bindings}), 'duplicate_completion_binding')
    need({r['path'] for r in bindings} == actual, 'completion_file_coverage_changed')
    verify_bindings(bindings)
    verify_preserved(home)
    need(not list(home.rglob('__pycache__')), 'batch_contains_bytecode_cache')
    return {'status': 'PASS', 'bound_batch_files': len(bindings), 'preserved_previous_files': record['preserved_previous_file_count']}


def has_generation_or_content_shortfall(s):
    return bool(s['details']['multiselect_unfilled_slots'] or s['details']['multiselect_content_failures'] or
                s['details']['ordering_generation_failures'] or s['details']['ordering_content_failures'] or
                s['review']['accepted_count'] < s['planned_ordering'])


def finalize(home=HOME):
    e = home / 'evidence'
    need(not any((e / name).exists() for name in OUTPUT_NAMES), 'closeout_output_already_exists')
    s = load_state(home)
    for name, content in [('ARTIFACT_INDEX.md', index_text(s, home)), ('RESULTS.md', results_text(s))]:
        with (e / name).open('x') as stream:
            stream.write(content)
    verify_preserved(home)
    shortfall = has_generation_or_content_shortfall(s)
    record = {'version': 'v84_mixed_batch_completion_v1', 'completed_at_utc': now(),
              'status': 'COMPLETED_WITH_GENERATION_OR_CONTENT_SHORTFALLS' if shortfall else 'COMPLETE',
              'previous_cumulative_candidate_count': s['previous']['candidate_count'], 'previous_cumulative_type_distribution': dict(s['previous_types']),
              'new_selected_count': s['new']['new_question_count'], 'new_selected_by_type': dict(s['newtypes']),
              'cumulative_candidate_count': s['cumulative']['candidate_count'], 'cumulative_type_distribution': dict(s['cumtypes']),
              'cumulative_multiselect_gold_distribution': s['cumulative']['gold_distribution'],
              'source_root_ids': sorted(EXPECTED_ROOTS), 'source_root_count': len(EXPECTED_ROOTS), 'source_split': 'known_historical_train_roots', 'independent_test_sources': False,
              'generation_http_calls': s['audits']['GENERATION_AUDIT']['totals']['actual_generation_http_calls'],
              'solver_calls_per_model': s['report']['calls_per_model'], 'total_solver_calls': 2 * s['report']['calls_per_model'],
              'ordering_content_usable_count': s['review']['accepted_count'], 'ordering_new_selected_count': s['newtypes']['ordering'],
              'multiselect_content_rejected_count': len(s['details']['multiselect_content_failures']),
              'ordering_review_actor_failed_item_count': len(s['actor_failures']),
              'training_started': False, 'formal_training_admission': False, 'merged_into_baseline': False,
              'preserved_previous_file_count': len(s['protected']), 'protected_directory_inventory_unchanged': True,
              'frozen_code_and_plan_unchanged': True, 'frozen_character_registry_and_reference_policy_unchanged': True,
              'files': [bind(p) for p in sorted(home.rglob('*')) if p.is_file() and p.resolve() != (e / 'RUN_COMPLETE.json').resolve()]}
    write(e / 'RUN_COMPLETE.json', record)
    verification = verify_completion(home)
    print({k: v for k, v in record.items() if k != 'files'})
    print(verification)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check-only', action='store_true', help='Validate dependencies and actual counts without writing outputs.')
    group.add_argument('--verify-complete', action='store_true', help='Read-only replay of final hashes, complete file coverage and previous preservation.')
    args = parser.parse_args()
    if args.verify_complete:
        print(verify_completion())
    elif args.check_only:
        s = load_state()
        print({'status': 'PASS', 'new_selected': s['new']['new_question_count'], 'cumulative': s['cumulative']['candidate_count'], 'preserved_previous_files': len(s['protected'])})
    else:
        finalize()


if __name__ == '__main__':
    main()
