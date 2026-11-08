"""Offline paired mixed comparison with independent set and event-pair scoring."""
from fractions import Fraction
import json
from pathlib import Path
import re
import run_paired_base as paired_runner

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def require(value, why):
    if not value:
        raise ValueError(why)


def aggregate(pairs, side):
    rows = [p[side] for p in pairs]
    calls, exact = len(rows), sum(p['exact'] for p in rows)
    partial = sum((Fraction(p['partial']) for p in rows), Fraction())
    return {'questions': len({p['candidate_id'] for p in pairs}), 'calls': calls,
            'exact_correct': exact, 'exact_rate': str(Fraction(exact, calls)) if calls else None,
            'exact_rate_percent': round(100 * exact / calls, 2) if calls else None,
            'partial_mean': str(partial / calls) if calls else None,
            'partial_mean_percent': round(100 * float(partial / calls), 2) if calls else None}


def compare(pairs):
    base, trained = aggregate(pairs, 'base'), aggregate(pairs, 'trained')
    return {'base': base, 'trained': trained,
            'trained_minus_base_exact_percentage_points': round(100 * (float(Fraction(trained['exact_rate'])) - float(Fraction(base['exact_rate']))), 2) if pairs else None,
            'trained_minus_base_partial_points': str(Fraction(trained['partial_mean']) - Fraction(base['partial_mean'])) if pairs else None,
            'paired_exact': {'both_correct': sum(p['base']['exact'] and p['trained']['exact'] for p in pairs),
                'trained_only_correct': sum(not p['base']['exact'] and p['trained']['exact'] for p in pairs),
                'base_only_correct': sum(p['base']['exact'] and not p['trained']['exact'] for p in pairs),
                'both_wrong': sum(not p['base']['exact'] and not p['trained']['exact'] for p in pairs)}}


def compare_by_type(pairs):
    return {kind: compare([p for p in pairs if p['question_type'] == kind])
            for kind in ('multi_select', 'ordering')}


def independent_mc_score(response, request, gold):
    match = re.fullmatch(r'\s*<box>\s*(.*?)\s*</box>\s*', response['raw_output'], re.S)
    require(match is not None, 'invalid_mc_response_box')
    text = match.group(1).strip()
    values = [part.strip().upper() for part in text.split(',')] if text else []
    require(len(values) == len(set(values)) and set(values) <= set('ABCDEF'), 'invalid_mc_response_labels')
    mapping = request['variant']['display_to_content']
    prediction, supported = {mapping[value] for value in values}, set(gold)
    require(2 <= len(supported) <= 6 and supported <= set(mapping.values()), 'invalid_mc_gold')
    tp, fp, fn = len(prediction & supported), len(prediction - supported), len(supported - prediction)
    partial = Fraction(tp, len(supported)) - (Fraction(fp, 6 - len(supported)) if len(supported) < 6 else 0)
    exact = prediction == supported
    score = response['score']
    require(score['exact_match'] == exact and Fraction(score['partial_score_fraction']) == partial
            and (score['tp'], score['fp'], score['fn']) == (tp, fp, fn), 'independent_mc_score_mismatch')
    return {'exact': exact, 'partial': str(partial), 'tp': tp, 'fp': fp, 'fn': fn,
            'gold_count': len(supported), 'selected_count': len(prediction),
            'display_prediction': values, 'content_prediction': sorted(prediction),
            'response_path': response['response_path']}


def verify_result_aggregates(directory, values, requests):
    results = read(directory / 'RESULTS.json')['questions']
    qids = {r['candidate_id'] for r in requests.values()}
    require({q['candidate_id'] for q in results} == qids and len(results) == len(qids), 'result_question_set_mismatch')
    for result in results:
        scores = [values[rid] for rid, request in requests.items() if request['candidate_id'] == result['candidate_id']]
        require(len(scores) == result['actual_full_calls'] and
                Fraction(sum(s['exact'] for s in scores), len(scores)) == Fraction(result['full_exact_rate']) and
                sum((Fraction(s['partial']) for s in scores), Fraction()) / len(scores) == Fraction(result['full_partial_mean']),
                'independent_result_aggregate_mismatch')


def independent_order_score(response, request, gold):
    match = re.fullmatch(r'\s*<box>\s*(.*?)\s*</box>\s*', response['raw_output'], re.S)
    require(match is not None, 'invalid_order_response_box')
    values = [part.strip().upper() for part in match.group(1).split(',')]
    require(len(values) == len(set(values)) == 6, 'invalid_order_response_ids')
    mapping = request['variant']['display_to_content']
    prediction = [mapping[value] for value in values]
    require(set(prediction) == set(gold) and len(gold) == 6, 'invalid_order_response_content_set')
    positions = {value: index for index, value in enumerate(prediction)}
    concordant = sum(positions[left] < positions[right] for index, left in enumerate(gold) for right in gold[index + 1:])
    partial = Fraction(2 * concordant - 15, 15)
    score = response['score']
    require(score['exact_match'] == (prediction == gold) and Fraction(score['partial_score_fraction']) == partial,
            'independent_order_score_mismatch')
    return {'exact': prediction == gold, 'partial': str(partial), 'concordant_pairs': concordant,
            'discordant_pairs': 15 - concordant, 'display_prediction': values,
            'content_prediction': prediction, 'response_path': response['response_path']}


def collect(statuses, selected_ids):
    complete = read(HERE / 'BASE_EVALUATION.COMPLETE.json')
    require(complete['status'] == 'COMPLETE' and complete['new_trained_calls'] == 0, 'base_not_complete')
    require(not (HERE / 'BASE_EVALUATION.ERROR.json').exists(), 'base_error_requires_explicit_recovery')
    preflight = read(HERE / 'BASE_PAIRED_PREFLIGHT.json')
    require(preflight['script_sha256'] == paired_runner.sha(paired_runner.__file__), 'paired_runner_changed')
    require(read(HERE / 'BASE_EVALUATION.STARTED.json')['paired_preflight_sha256'] == paired_runner.sha(HERE / 'BASE_PAIRED_PREFLIGHT.json'), 'paired_preflight_changed')
    receipts, pairs, question_ids = [], [], set()
    for name, trained, base in paired_runner.chunks():
        receipt = paired_runner.paired_chunk(name, trained, base, complete=True)
        receipts.append(receipt)
        if receipt.get('status') == 'EMPTY_POOL_NO_SOLVER':
            continue
        manifest, gold = read(trained / 'manifest.json'), read(trained / 'gold.json')
        requests = {row['request_id']: row for row in manifest['requests']}
        rows = {}
        for side, directory in (('base', base), ('trained', trained)):
            values = {}
            for path in sorted((directory / 'responses').glob('*.json')):
                response = read(path)
                response['response_path'] = str(path)
                request = requests[response['request_id']]
                qid = response['candidate_id']
                require(response['request_id'] not in values, 'duplicate_response_id')
                scorer = independent_mc_score if request['question_type'] == 'multi_select' else independent_order_score
                values[response['request_id']] = scorer(response, request, gold[qid])
            verify_result_aggregates(directory, values, requests)
            rows[side] = values
        require(set(rows['base']) == set(rows['trained']) == set(requests), 'paired_response_ids_mismatch')
        for rid, request in requests.items():
            qid = request['candidate_id']; question_ids.add(qid)
            require(qid in statuses, 'missing_content_status')
            pairs.append({'request_id': rid, 'candidate_id': qid, 'chunk': name,
                'root_id': request['root_id'], 'question_type': request['question_type'], 'variant_id': request['variant']['variant_id'],
                'slot': request['slot'], 'seed': request['seed'], 'prompt_sha256': request['prompt_sha256'],
                'core_status': statuses[qid], 'trained_machine_selected': qid in selected_ids,
                'base': rows['base'][rid], 'trained': rows['trained'][rid]})
    require(len(pairs) == complete['base_calls'] <= 180, 'paired_actual_budget_mismatch')
    per_question = []
    for qid in sorted(question_ids):
        qrows = [p for p in pairs if p['candidate_id'] == qid]
        kind = qrows[0]['question_type']
        expected = {(f'perm_{i:02d}', j) for i in range(6) for j in range(3)} if kind == 'multi_select' else {(f'perm_{i:02d}', 0) for i in range(12)}
        require(len(qrows) == len(expected) and {(p['variant_id'], p['slot']) for p in qrows} == expected, 'typed_permutation_call_set_mismatch')
        per_question.append({'candidate_id': qid, 'question_type': kind, 'core_status': statuses[qid],
            'trained_machine_selected': qid in selected_ids, **compare(qrows)})
    core = [p for p in pairs if p['core_status'] == 'CORE_USABLE']
    selected = [p for p in core if p['trained_machine_selected']]
    result = {'status': 'PASS', 'primary_scope': 'all_ready_core_usable_defined_without_solver_scores',
        'primary': compare(core), 'primary_by_type': compare_by_type(core),
        'all_ready_diagnostic': compare(pairs), 'all_ready_by_type': compare_by_type(pairs),
        'trained_selected_core_usable_secondary': compare(selected),
        'questions': per_question, 'chunks': receipts,
        'actual_calls_per_model': len(pairs), 'paired_prompts_sources_gold_permutations_seeds_identical': True,
        'scores_replayed_with_native_scorer_and_independent_mc_sets_and_order_pairs': True,
        'combined_summary_is_call_weighted': True, 'partial_formulas_differ_by_question_type': True,
        'content_labels_frozen_before_any_solver': True, 'training_started': False,
        'limitation': 'One root, questions from the same dialogue, and writer receives prior-round trained-model feedback. Not an independent cross-root holdout estimate.'}
    return result, pairs


def markdown(result):
    lines = ['# v59 Base 与训练后模型逐题比较', '',
        '主结果为评分前独立内容检查已标为可用的全部 ready 题，排序题同时通过原生排序审核；不按训练后模型得分挑题。', '',
        '| 口径 | 题数 | Base 完全正确率 | 训练后完全正确率 | 差值（百分点） |', '|---|---:|---:|---:|---:|']
    for label, key in [('内容可用主结果', 'primary'), ('全部评分题（诊断）', 'all_ready_diagnostic'), ('训练后机器入选且内容可用（次要）', 'trained_selected_core_usable_secondary')]:
        c = result[key]; b, t = c['base'], c['trained']
        lines.append(f"| {label} | {b['questions']} | {b['exact_correct']}/{b['calls']}（{b['exact_rate_percent']}%） | {t['exact_correct']}/{t['calls']}（{t['exact_rate_percent']}%） | {c['trained_minus_base_exact_percentage_points']} |")
    lines.extend(['', '| 主结果题型 | 题数 | Base 完全正确 | 训练后完全正确 | Base 部分均值 | 训练后部分均值 |', '|---|---:|---:|---:|---:|---:|'])
    for kind, c in result['primary_by_type'].items():
        b,t = c['base'], c['trained']
        lines.append(f"| {kind} | {b['questions']} | {b['exact_correct']}/{b['calls']}（{b['exact_rate_percent']}%） | {t['exact_correct']}/{t['calls']}（{t['exact_rate_percent']}%） | {b['partial_mean']} | {t['partial_mean']} |")
    lines.extend(['', '| 题目 | 题型 | 内容状态 | Base 完全正确 | 训练后完全正确 | Base 部分均值 | 训练后部分均值 | 训练后入选 |', '|---|---|---|---:|---:|---:|---:|---|'])
    for row in result['questions']:
        b,t=row['base'],row['trained']
        lines.append(f"| {row['candidate_id']} | {row['question_type']} | {row['core_status']} | {b['exact_correct']}/{b['calls']} | {t['exact_correct']}/{t['calls']} | {b['partial_mean']} | {t['partial_mean']} | {'是' if row['trained_machine_selected'] else '否'} |")
    lines.extend(['', '每题每模型：多选 6 次随机呈现×3回答 = 18 次；排序 12 次随机呈现×1 = 12 次。两个模型使用完全相同的完整对话、Gold、呈现、随机种子、解码及评分代码。',
        '多选部分得分为 TP/正确项数 − FP/错误项数（全6项正确时不计后一项）；排序为（正确事件对−错误事件对）/15。两种部分得分均不是选项准确率，分题型报告；混合汇总按调用次数加权。',
        '这是现有历史训练检查点 M_clean_policy 与 Qwen3-4B Base 的比较；第二轮训练尚未开始。仅一个 root，且出题器沿用针对训练后模型的前轮反馈，不能外推为独立跨 root 的训练收益。'])
    return '\n'.join(lines)
