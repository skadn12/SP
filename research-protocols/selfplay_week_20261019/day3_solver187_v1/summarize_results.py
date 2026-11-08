"""Summarize completed 187-question S0 Train diagnostics; never touches live inputs.

Run only after COMPLETE.json, RAW_RESPONSE_FREEZE.json and SCORES.json exist.
No inference, retries, candidate deletion or training admission occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
OUTPUTS = ('SUMMARY.json', 'per_question.csv', 'MATCHED_PAIR_DIAGNOSTICS.json', 'REPORT_ZH.md')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def verify_binding(row):
    path = Path(row['path'])
    require(path.is_file() and sha(path) == row['sha256'], 'BINDING_MISMATCH:' + str(path))
    if 'bytes' in row:
        require(path.stat().st_size == row['bytes'], 'BINDING_SIZE_MISMATCH:' + str(path))


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def write_json(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def summarize_mc(records):
    valid = [r for r in records if r['valid']]
    exact = sum(r['exact'] for r in records)
    gold_counts = Counter(str(r['metrics']['gold_count']) for r in records)
    histogram = Counter(str(r['metrics']['selected_count']) for r in valid)
    over = sum(r['metrics']['selected_count'] > r['metrics']['gold_count'] for r in valid)
    under = sum(r['metrics']['selected_count'] < r['metrics']['gold_count'] for r in valid)
    all_six = sum(r['metrics']['selected_count'] == 6 for r in valid)
    fp = sum(r['metrics']['fp'] for r in valid)
    fn = sum(r['metrics']['fn'] for r in valid)
    tp = sum(r['metrics']['tp'] for r in valid)
    negatives = sum(6 - r['metrics']['gold_count'] for r in valid)
    positives = sum(r['metrics']['gold_count'] for r in valid)
    error_shapes = Counter('fp_and_fn' if r['metrics']['fp'] and r['metrics']['fn'] else
                           'fp_only' if r['metrics']['fp'] else
                           'fn_only' if r['metrics']['fn'] else 'exact' for r in valid)
    return {
        'calls': len(records), 'valid_calls': len(valid), 'invalid_calls': len(records) - len(valid),
        'exact_calls': exact, 'gold_count_call_histogram': dict(sorted(gold_counts.items())),
        'known_tp_selections': tp, 'known_fp_selections': fp, 'known_fn_omissions': fn,
        'known_gold_positive_opportunities': positives, 'known_gold_negative_opportunities': negatives,
        'fp_rate_over_valid_negative_items': ratio(fp, negatives),
        'fn_rate_over_valid_positive_items': ratio(fn, positives),
        'unknown_selection_calls': len(records) - len(valid),
        'selected_count_histogram_valid': dict(sorted(histogram.items())),
        'selected_count_mean_valid': mean(r['metrics']['selected_count'] for r in valid),
        'gold_count_mean_valid': mean(r['metrics']['gold_count'] for r in valid),
        'selected_minus_gold_mean_valid': mean(r['metrics']['selected_count'] - r['metrics']['gold_count'] for r in valid),
        'overselection_calls': over, 'underselection_calls': under, 'all_six_selected_calls': all_six,
        'overselection_rate_valid': ratio(over, len(valid)),
        'overselection_rate_all_calls': ratio(over, len(records)),
        'underselection_rate_valid': ratio(under, len(valid)),
        'all_six_selected_rate_valid': ratio(all_six, len(valid)),
        'jaccard_mean_fixed_denominator': mean(r['metrics']['jaccard'] for r in records),
        'jaccard_mean_valid': mean(r['metrics']['jaccard'] for r in valid),
        'native_partial_score_mean_fixed_denominator': mean(float(Fraction(r['native_score']['partial_score'])) if r['valid'] else 0 for r in records),
        'error_shapes_valid_calls': dict(error_shapes),
        'invalid_policy': 'Invalid calls remain in fixed exact/Jaccard denominators; unknown selections and FP/FN are not imputed.',
        'overselection_definition': 'selected_count > gold_count; FP means a false option was selected, including equal-cardinality swaps.'
    }


def summarize_ordering(records):
    pairs = sum(r['metrics']['total_pairs'] for r in records)
    correct = sum(r['metrics']['correct_pairs'] for r in records)
    return {'calls': len(records), 'valid_calls': sum(r['valid'] for r in records),
            'total_pairs_fixed_denominator': pairs, 'correct_pairs': correct,
            'pair_accuracy_fixed_denominator': ratio(correct, pairs),
            'pair_accuracy_valid': mean(r['metrics']['pair_accuracy'] for r in records if r['valid']),
            'invalid_policy': 'All15 pairs receive zero credit for invalid or failed calls.'}


def group_summary(questions, records_by_q):
    requests = [r for q in questions for r in records_by_q[q['candidate_id']]]
    roots = defaultdict(list)
    for q in questions:
        roots[q['root_id']].append(q['exact_rate'])
    result = {'questions': len(questions), 'roots': len(roots),
              'calls': len(requests), 'valid_calls': sum(r['valid'] for r in requests),
              'invalid_calls': sum(not r['valid'] for r in requests),
              'exact_calls': sum(r['exact'] for r in requests),
              'question_equal_exact': mean(q['exact_rate'] for q in questions),
              'root_equal_exact': mean(mean(values) for values in roots.values()),
              'call_weighted_exact': ratio(sum(r['exact'] for r in requests), len(requests)),
              'question_strata': dict(Counter(q['stratum'] for q in questions)),
              'never_exact_with_any_invalid_questions': sum(q['exact_calls'] == 0 and q['invalid_calls'] > 0 for q in questions),
              'never_exact_all_valid_questions': sum(q['exact_calls'] == 0 and q['invalid_calls'] == 0 for q in questions),
              'prediction_consistency_question_mean': mean(q['display_permutation_consistency'] for q in questions),
              'display_exact_rate_spread_question_mean': mean(q['display_diagnostics']['exact_rate_spread'] for q in questions)}
    mc = [r for r in requests if r['question_type'] == 'multi_select']
    ordering = [r for r in requests if r['question_type'] == 'ordering']
    if mc:
        result['multi_select'] = summarize_mc(mc)
    if ordering:
        result['ordering'] = summarize_ordering(ordering)
    return result


def grouped(questions, records_by_q, key_func):
    groups = defaultdict(list)
    for q in questions:
        groups[key_func(q)].append(q)
    return {key: group_summary(values, records_by_q) for key, values in sorted(groups.items())}


def percentage(value):
    return '—' if value is None else f'{value * 100:.2f}%'


def summarize(base, run):
    base, run = base.resolve(), run.resolve()
    for output in OUTPUTS:
        require(not (base / output).exists(), 'NO_OVERWRITE:' + str(base / output))
    scores_path = run / 'SCORES.json'
    require((run / 'COMPLETE.json').is_file(), 'SUCCESSFUL_EXECUTION_REQUIRED')
    require((run / 'RAW_RESPONSE_FREEZE.json').is_file() and scores_path.is_file(), 'POST_FREEZE_SCORES_REQUIRED')
    complete, freeze, scores = read(run / 'COMPLETE.json'), read(run / 'RAW_RESPONSE_FREEZE.json'), read(scores_path)
    require(complete['status'] == 'COMPLETE' and freeze['status'] == 'COMMITTED' and scores['status'] == 'COMPLETE', 'INCOMPLETE_EXECUTION_OR_SCORE')
    for key in ('manifest_binding', 'preflight_freeze_binding', 'raw_response_freeze_binding', 'private_answer_binding', 'scoring_code_binding'):
        verify_binding(scores[key])
    require(Path(scores['manifest_binding']['path']).resolve() == run / 'manifest.json', 'WRONG_MANIFEST_PATH')
    require(Path(scores['raw_response_freeze_binding']['path']).resolve() == run / 'RAW_RESPONSE_FREEZE.json', 'WRONG_RAW_FREEZE_PATH')
    require(scores['denominator_questions'] == 187 and scores['denominator_roots'] == 20 and scores['denominator_calls'] == 2688, 'UNEXPECTED_DIAGNOSTIC_DENOMINATOR')
    for row in read(base / 'INPUT_ASSEMBLY.json')['files']:
        verify_binding(row)
    normalization_path = base / 'FORMAT_NORMALIZATION_V2.json'
    normalization = read(normalization_path)
    require(normalization['status'] == 'FROZEN_BEFORE_ANY_SOLVER_INFERENCE' and normalization['changed_questions'] == 55 and normalization['semantic_content_or_gold_changed'] is False, 'INVALID_FORMAT_NORMALIZATION')
    for key in ('original_public', 'new_public', 'new_private'):
        verify_binding(normalization[key])
    require(sha(run / 'PUBLIC_QUESTIONS.json') == normalization['new_public']['sha256'], 'SUMMARY_REQUIRES_NORMALIZED_V2_EXECUTION')
    metadata_path, pairs_path = base / 'CANDIDATE_METADATA.json', base / 'MATCHED_SLOT_PAIRS.json'
    metadata = {q['candidate_id']: q for q in read(metadata_path)['questions']}
    paired = read(pairs_path)['pairs']
    provenance_path = base / 'inputs_private/PROVENANCE.json'
    provenance = {q['candidate_id']: q for q in read(provenance_path)['questions']}
    public_path = run / 'PUBLIC_QUESTIONS.json'
    public = {q['candidate_id']: q for q in read(public_path)['questions']}
    records_by_q = defaultdict(list)
    requests = scores['requests']
    require(len(requests) == 2688 and len({r['request_id'] for r in requests}) == 2688, 'REQUEST_SET_MISMATCH')
    for record in requests:
        records_by_q[record['candidate_id']].append(record)
    qids = {q['candidate_id'] for q in scores['questions']}
    require(len(qids) == 187 and qids == set(metadata) == set(provenance) == set(public) == set(records_by_q), 'QUESTION_JOIN_MISMATCH')
    questions = []
    for scored in scores['questions']:
        qid = scored['candidate_id']
        rows, meta, origin = records_by_q[qid], metadata[qid], provenance[qid]
        require(scored['root_id'] == meta['root_id'] == public[qid]['root_id'] and scored['question_type'] == meta['question_type'] == public[qid]['question_type'], 'METADATA_IDENTITY_MISMATCH:' + qid)
        expected_n = 18 if scored['question_type'] == 'multi_select' else 12
        exact = sum(r['exact'] for r in rows)
        valid = sum(r['valid'] for r in rows)
        gold = rows[0]['gold_canonical']
        require(len(rows) == scored['denominator_calls'] == expected_n and exact == scored['exact_calls'] and valid == scored['valid_calls'], 'SCORED_COUNT_MISMATCH:' + qid)
        require(all(r['gold_canonical'] == gold for r in rows) and gold == origin['native_gold'] and len(gold) == meta['actual_gold_count'], 'GOLD_JOIN_MISMATCH:' + qid)
        require(scored['exact_rate'] == exact / expected_n, 'EXACT_RATE_MISMATCH:' + qid)
        by_variant = defaultdict(list)
        for row in rows:
            by_variant[row['variant_id']].append(row)
        display_rows = [{'variant_id': key, 'calls': len(values), 'exact_calls': sum(r['exact'] for r in values),
                         'valid_calls': sum(r['valid'] for r in values), 'exact_rate': sum(r['exact'] for r in values) / len(values)}
                        for key, values in sorted(by_variant.items())]
        require(len(display_rows) == (6 if expected_n == 18 else 12), 'VARIANT_COUNT_MISMATCH:' + qid)
        require(all(d['calls'] == (3 if expected_n == 18 else 1) for d in display_rows), 'VARIANT_SAMPLES_MISMATCH:' + qid)
        rates = [r['exact_rate'] for r in display_rows]
        question = dict(scored)
        question.update({key: meta.get(key) for key in ('arm', 'slot_id', 'fact_family_id', 'batch', 'design_id', 'generation_mode', 'target_cause_id', 'target_gold_count', 'actual_gold_count', 'content_state')})
        question.update(gold_native=gold, gold_original=origin['original_gold'], gold_presentation=origin['presentation_gold'],
                        original_public_path=origin['original_public']['path'], original_record_path=origin['original_record']['path'],
                        source_path=origin['source']['path'], source_sha256=origin['source']['sha256'],
                        normalized_public_path=str(public_path), private_provenance_path=str(provenance_path),
                        display_diagnostics={'per_display': display_rows, 'exact_rate_min': min(rates), 'exact_rate_max': max(rates),
                                             'exact_rate_spread': max(rates) - min(rates),
                                             'interpretation': 'Descriptive mixture of display and stochastic sampling. MC has3 samples/display; ordering has1. Not isolated position bias.'})
        if scored['question_type'] == 'multi_select':
            question['multi_select_diagnostics'] = summarize_mc(rows)
        else:
            question['ordering_diagnostics'] = summarize_ordering(rows)
        questions.append(question)
    questions.sort(key=lambda q: q['candidate_id'])
    require(Counter(q['question_type'] for q in questions) == {'multi_select': 74, 'ordering': 113}, 'TYPE_COUNT_MISMATCH')
    require(Counter(q['arm'] for q in questions) == {'A': 94, 'B': 93}, 'ARM_COUNT_MISMATCH')
    question_index = {q['candidate_id']: q for q in questions}
    pair_records = []
    require(len(paired) == 88 and len({r['slot_id'] for r in paired}) == 88, 'MATCHED_PAIR_DENOMINATOR_MISMATCH')
    for pair in paired:
        a, b = question_index[pair['A']], question_index[pair['B']]
        require(a['arm'] == 'A' and b['arm'] == 'B' and a['slot_id'] == b['slot_id'] == pair['slot_id'] and a['root_id'] == b['root_id'] and a['question_type'] == b['question_type'], 'PAIR_IDENTITY_MISMATCH')
        pair_records.append({'slot_id': pair['slot_id'], 'root_id': a['root_id'], 'question_type': a['question_type'],
                             'A': {key: a[key] for key in ('candidate_id', 'design_id', 'actual_gold_count', 'exact_calls', 'denominator_calls', 'exact_rate', 'stratum')},
                             'B': {key: b[key] for key in ('candidate_id', 'design_id', 'actual_gold_count', 'exact_calls', 'denominator_calls', 'exact_rate', 'stratum')},
                             'B_minus_A_exact_rate': b['exact_rate'] - a['exact_rate'],
                             'B_minus_A_pair_accuracy': b.get('pair_accuracy', 0) - a.get('pair_accuracy', 0) if a['question_type'] == 'ordering' else None})
    pair_group = defaultdict(list)
    for pair in pair_records:
        pair_group[pair['question_type']].append(pair)
    pair_summary = {}
    for typ, rows in sorted(pair_group.items()):
        root_deltas = defaultdict(list)
        for row in rows:
            root_deltas[row['root_id']].append(row['B_minus_A_exact_rate'])
        pair_summary[typ] = {'pairs': len(rows), 'roots': len(root_deltas),
                             'A_question_equal_exact': mean(r['A']['exact_rate'] for r in rows),
                             'B_question_equal_exact': mean(r['B']['exact_rate'] for r in rows),
                             'B_minus_A_pair_equal_exact': mean(r['B_minus_A_exact_rate'] for r in rows),
                             'B_minus_A_root_equal_exact': mean(mean(values) for values in root_deltas.values()),
                             'B_higher_pairs': sum(r['B_minus_A_exact_rate'] > 0 for r in rows),
                             'equal_pairs': sum(r['B_minus_A_exact_rate'] == 0 for r in rows),
                             'B_lower_pairs': sum(r['B_minus_A_exact_rate'] < 0 for r in rows)}
    now = datetime.now(timezone.utc).isoformat()
    paired_output = {'status': 'COMPLETE_DESCRIPTIVE_ONLY', 'created_at_utc': now, 'pairs': len(pair_records),
                     'by_type': pair_summary, 'records': pair_records,
                     'causal_feedback_effect_estimated': False, 'training_superiority_estimated': False,
                     'limitations': ['Matched slots contain distinct authored questions; design and actual content can differ.',
                                    '88 admitted pairs omit unmatched slots and do not restore all240 planned slots.',
                                    'Questions share roots and facts. Pair/call counts are not independent sample counts.',
                                    'S0 answerability alone neither proves feedback causality nor training value.'],
                     'input_bindings': [binding(scores_path), binding(metadata_path), binding(pairs_path)]}
    summary = {'status': 'COMPLETE_TRAIN_DIAGNOSTIC_ONLY', 'created_at_utc': now,
               'solver': 'S0_M_clean', 'denominators': {'questions': 187, 'roots': 20, 'calls': 2688,
                 'multi_select_questions': 74, 'ordering_questions': 113, 'production_planned_slots': 240,
                 'production_admitted_questions': 187, 'production_unadmitted_slots': 53, 'matched_slot_pairs': 88},
               'overall': group_summary(questions, records_by_q),
               'by_type': grouped(questions, records_by_q, lambda q: q['question_type']),
               'by_arm': grouped(questions, records_by_q, lambda q: q['arm']),
               'by_arm_type': grouped(questions, records_by_q, lambda q: q['arm'] + '|' + q['question_type']),
               'by_type_gold_count': grouped(questions, records_by_q, lambda q: q['question_type'] + '|gold' + str(q['actual_gold_count'])),
               'by_arm_type_gold_count': grouped(questions, records_by_q, lambda q: q['arm'] + '|' + q['question_type'] + '|gold' + str(q['actual_gold_count'])),
               'by_design': grouped(questions, records_by_q, lambda q: q['question_type'] + '|' + str(q['design_id'])),
               'by_root': grouped(questions, records_by_q, lambda q: q['root_id']),
               'by_fact_family': grouped(questions, records_by_q, lambda q: q['fact_family_id']),
               'format_normalization': {'binding': binding(normalization_path), 'changed_questions': 55,
                 'before_any_solver_inference': True, 'semantic_content_or_gold_changed': False,
                 'rule': normalization['rule'], 'prior_preparation_inference_calls': 0},
               'cost': scores['cost'], 'operational': {'terminal_status_counts': dict(Counter(r['status'] for r in requests)),
                 'invalid_error_counts': dict(Counter(error for r in requests for error in r['parse_errors'])),
                 'automatic_retry': False, 'source_truncation': False},
               'definitions': {'primary_accuracy': 'Mean of per-question fixed-denominator exact rates. Root equal first averages questions within each root.',
                 'stably_correct': 'Every planned call answered exactly correctly.',
                 'occasionally_correct': 'At least one exact call and at least one non-exact call.',
                 'incorrect': 'No exact call; may include operational invalids, separately counted.',
                 'prediction_consistency': 'Canonical prediction mode frequency/planned calls; mixes display and sampling effects.',
                 'independence_unit': 'Roots/facts are shared.2688 calls and187 questions are not2688 or187 independent examples.'},
               'scope': {'all_admitted_candidates_retained': True, 'score_selected_deletions': 0,
                 'training_started': False, 'dev_calls': 0, 'test_calls': 0,
                 'content_admission_changed_by_solver': False, 'causal_feedback_effect_estimated': False,
                 'training_value_estimated': False, 'A_generator_receives_results': False},
               'input_bindings': [binding(scores_path), binding(run / 'manifest.json'), binding(run / 'RAW_RESPONSE_FREEZE.json'),
                 binding(metadata_path), binding(pairs_path), binding(base / 'PRODUCTION_SLOT_LEDGER.json'), binding(provenance_path), binding(normalization_path)],
               'script_binding': binding(__file__), 'questions': questions}
    require(summary['overall']['exact_calls'] == scores['exact_calls'] and summary['overall']['valid_calls'] == scores['valid_calls'], 'OVERALL_RECOMPUTATION_MISMATCH')
    for typ in ('multi_select', 'ordering'):
        require(summary['by_type'][typ]['question_equal_exact'] == scores['by_type'][typ]['question_equal_exact'], 'TYPE_RECOMPUTATION_MISMATCH')
    csv_fields = ['candidate_id', 'root_id', 'slot_id', 'arm', 'question_type', 'batch', 'design_id', 'generation_mode',
                  'fact_family_id', 'target_cause_id', 'target_gold_count', 'actual_gold_count', 'gold_native', 'gold_original',
                  'exact_calls', 'denominator_calls', 'exact_rate', 'valid_calls', 'invalid_calls', 'stratum',
                  'pair_accuracy', 'known_fp_selections', 'known_fn_omissions', 'jaccard_mean_fixed_denominator',
                  'selected_count_mean_valid', 'overselection_calls', 'overselection_rate_valid',
                  'prediction_consistency', 'display_exact_rate_spread', 'original_public_path', 'original_record_path',
                  'source_path', 'normalized_public_path', 'private_provenance_path']
    csv_rows = []
    for q in questions:
        row = {key: q.get(key) for key in csv_fields}
        row.update(gold_native=json.dumps(q['gold_native'], ensure_ascii=False), gold_original=json.dumps(q['gold_original'], ensure_ascii=False),
                   prediction_consistency=q['display_permutation_consistency'], display_exact_rate_spread=q['display_diagnostics']['exact_rate_spread'])
        if q['question_type'] == 'multi_select':
            row.update({key: q['multi_select_diagnostics'].get(key) for key in ('known_fp_selections', 'known_fn_omissions', 'jaccard_mean_fixed_denominator', 'selected_count_mean_valid', 'overselection_calls', 'overselection_rate_valid')})
        csv_rows.append(row)
    all_rows = summary['overall']
    strata = all_rows['question_strata']
    lines = [f"本轮已完成 S0＝M-clean 对全部187道 Train 候选的统一诊断：20个root、74道多选、113道排序，共2,688次直接作答。其中{all_rows['valid_calls']}次输出有效，{all_rows['invalid_calls']}次无效。",
             '', '推理前已修正55道排序题的格式指令，将“输出六个字母”统一为“输出六个显示条目ID”，匹配原生I1–I6标签。题意、选项内容和Gold内容不变；此前direct仅完成准备、Solver调用为0。修正及前后哈希见FORMAT_NORMALIZATION_V2.json。',
             '', f"按题目等权的完整正确率为{percentage(all_rows['question_equal_exact'])}，按root等权为{percentage(all_rows['root_equal_exact'])}。每道多选做6种排列×3次采样，每道排序做12种排列×1次采样；无效/失败保留在分母。",
             '', '| 组别 | 题数 | 调用数 | 完整正确率（题等权） | 完整正确率（root等权） | 排序成对正确率 |',
             '|---|---:|---:|---:|---:|---:|']
    for key, group in summary['by_arm_type'].items():
        lines.append(f"| {key.replace('|', ' / ')} | {group['questions']} | {group['calls']} | {percentage(group['question_equal_exact'])} | {percentage(group['root_equal_exact'])} | {percentage(group.get('ordering', {}).get('pair_accuracy_fixed_denominator'))} |")
    lines.extend(['', f"逐题分层：全部调用均答对{strata.get('stably_correct', 0)}题；部分调用答对{strata.get('occasionally_correct', 0)}题；从未完整答对{strata.get('incorrect', 0)}题。其中从未答对且出现无效输出{all_rows['never_exact_with_any_invalid_questions']}题，避免把运行异常直接解释为能力弱点。",
                  '', '| 多选Gold数 | 题数 | 完整正确率（题等权） | 过选率（有效调用） | Jaccard（固定分母） |', '|---|---:|---:|---:|---:|'])
    for key, group in summary['by_type_gold_count'].items():
        if key.startswith('multi_select|'):
            mc = group['multi_select']
            lines.append(f"| {key.split('|')[1]} | {group['questions']} | {percentage(group['question_equal_exact'])} | {percentage(mc['overselection_rate_valid'])} | {percentage(mc['jaccard_mean_fixed_denominator'])} |")
    mc = all_rows['multi_select']
    lines.extend(['', f"多选有效调用中，错误选项累计被选{mc['known_fp_selections']}次（FP），正确选项累计漏选{mc['known_fn_omissions']}次（FN）；选择数量超过Gold数量{mc['overselection_calls']}次，占有效调用{percentage(mc['overselection_rate_valid'])}。FP与过选分开计算：选错一个并漏掉一个可能数量正确，却同时产生FP和FN。",
                  '', 'A/B是两种出题实验条件。A不接收模型失败反馈，B使用已获准的窄范围反馈。88个共同入选槽位的配对结果见 MATCHED_PAIR_DIAGNOSTICS.json；槽位中的题目内容不同，且共享root和事实，因此只能描述当前S0的可答性，不能据此证明反馈因果效果、B更有训练价值或训练收益。',
                  '', '排列间正确率变化与随机采样交织；排序每种排列仅1次采样，不能将其变化单独归因为位置偏差。逐题明细保留每个排列的调用数和正确率。',
                  '', '生产统计仍为240个计划槽位中187题通过内容审核，53个槽位未入选。此次Solver结果不改变内容审核结果，不按分数删题，也未启动训练或调用DEV/TEST。后续训练取舍需先固定规则，再用独立评估判断效果。'])
    cost = scores['cost']
    seconds = cost.get('live_wall_seconds')
    load = cost.get('model_load_wall_seconds')
    lines.extend(['', f"本地GPU推理耗时{seconds:.2f}秒（其中模型加载{load:.2f}秒），已知输入{cost['known_prompt_tokens']:,} tokens、输出{cost['known_completion_tokens']:,} tokens，未知用量请求{cost['requests_with_unknown_usage']}个。未记录货币账单。" if isinstance(seconds, (int, float)) and isinstance(load, (int, float)) else '用时与token用量详见SUMMARY.json中的cost；未知值不记为零。',
                  '', f'逐题表：per_question.csv；完整汇总：SUMMARY.json；原始回答、评分与冻结文件：{run.relative_to(base)}/。SUMMARY.json和CSV含答案/私有来源路径，应按训练诊断材料管理。', ''])
    write_json(base / 'SUMMARY.json', summary)
    with (base / 'per_question.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json(base / 'MATCHED_PAIR_DIAGNOSTICS.json', paired_output)
    with (base / 'REPORT_ZH.md').open('x') as handle:
        handle.write('\n'.join(lines))
    print(json.dumps({'status': summary['status'], 'questions': len(questions), 'calls': len(requests),
                      'question_equal_exact': all_rows['question_equal_exact'], 'strata': strata,
                      'outputs': [binding(base / name) for name in OUTPUTS]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=HERE)
    parser.add_argument('--run', type=Path, default=HERE / 'runtime/direct_v2')
    args = parser.parse_args()
    summarize(args.base, args.run)


if __name__ == '__main__':
    main()
