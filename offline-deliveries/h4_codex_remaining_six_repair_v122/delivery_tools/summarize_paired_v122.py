"""Read-only aggregation of COMPLETE paired v122 Solver runs; no model calls.

Only the two additive delivery reports are created, exclusively, after all input
and native-result cross-checks pass. No experiment or runtime modules are imported.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import hashlib
import json
from pathlib import Path

BATCH = Path(__file__).resolve().parents[1]
SCOPE = BATCH / 'evaluations/paired_v1'
SIDES = ('base', 'trained')
GROUPS = ('all', 'ms_gold2_5', 'ms_gold6_control', 'ordering', 'gold2', 'gold3', 'gold4', 'gold5')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def bind(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def ratio(numerator, denominator):
    if not denominator:
        return None
    value = Fraction(numerator, denominator)
    return {'numerator': numerator, 'denominator': denominator,
            'fraction': str(value), 'decimal': float(value)}


def value(number):
    return None if number is None else {'fraction': str(number), 'decimal': float(number)}


def mean(numbers):
    return sum(numbers, Fraction()) / len(numbers) if numbers else None


def qgroup(question):
    if question['question_type'] == 'ordering':
        return 'ordering'
    gold = len(question['gold'])
    need(gold in (2, 3, 4, 5, 6), 'UNEXPECTED_GOLD_COUNT')
    return 'ms_gold6_control' if gold == 6 else 'ms_gold2_5'


def members(questions, group):
    if group == 'all':
        return questions
    if group.startswith('gold'):
        return [q for q in questions if q['question_type'] == 'multi_select' and len(q['gold']) == int(group[-1])]
    return [q for q in questions if qgroup(q) == group]


def load_side(scope, side, question_by_id):
    folder = scope / 'run' / side
    wrapper_path = scope / (side.upper() + '_COMPLETE.json')
    wrapper = read(wrapper_path)
    complete_path = folder / 'COMPLETE.json'
    complete = read(complete_path)
    manifest_path, results_path = folder / 'manifest.json', folder / 'RESULTS.json'
    manifest, results = read(manifest_path), read(results_path)
    need(complete.get('status') == 'COMPLETE', side + ':NOT_COMPLETE')
    need(wrapper.get('side') == side and wrapper.get('calls') == manifest['request_count'], side + ':WRAPPER_INCOMPLETE')
    need(wrapper['results_binding'] == bind(results_path), side + ':RESULTS_WRAPPER_BINDING_MISMATCH')
    need(manifest['solver_model'] == results['solver_model'], side + ':MODEL_RESULT_MISMATCH')
    need(manifest['policy_version'] == results['policy_version'], side + ':POLICY_RESULT_MISMATCH')
    need(manifest['candidate_count'] == len(question_by_id), side + ':CANDIDATE_COUNT_MISMATCH')
    need(manifest['pool_sha256'] == bind(scope / 'EVALUATION_POOL.json')['sha256'], side + ':POOL_HASH_MISMATCH')
    need(not results.get('call_set_errors'), side + ':NATIVE_CALL_SET_ERRORS')
    planned = {r['request_id']: r for r in manifest['requests']}
    need(len(planned) == len(manifest['requests']) == manifest['request_count'], side + ':DUPLICATE_PLANNED_CALL')
    paths = sorted((folder / 'responses').glob('*.json'))
    records = [read(p) for p in paths]
    observed = {r['request_id']: r for r in records}
    need(len(observed) == len(records), side + ':DUPLICATE_RESPONSE')
    need(set(observed) == set(planned), side + ':RESPONSE_PLAN_SET_MISMATCH')
    need(complete['actual_call_count'] == len(records) == manifest['request_count'], side + ':CALL_COUNT_MISMATCH')
    by_question = defaultdict(list)
    for record in records:
        plan = planned[record['request_id']]
        for key in ('candidate_id', 'root_id', 'question_type', 'slot', 'seed', 'mode'):
            need(record[key] == plan[key], side + ':RESPONSE_PLAN_IDENTITY:' + key)
        need(record['variant_id'] == plan['variant']['variant_id'], side + ':RESPONSE_VARIANT_MISMATCH')
        need(record['candidate_id'] in question_by_id, side + ':UNKNOWN_CANDIDATE')
        question = question_by_id[record['candidate_id']]
        need(record['question_type'] == question['question_type'] and record['root_id'] == question['root_id'], side + ':RESPONSE_QUESTION_MISMATCH')
        if record.get('evaluation_valid') is True:
            score = record['score']
            need(score.get('scored') is True and type(score.get('exact_match')) is bool, side + ':VALID_SCORE_MISSING')
            Fraction(score['partial_score_fraction'])
            need(len(score['content_gold']) == len(question['gold']), side + ':GOLD_COUNT_MISMATCH')
        by_question[record['candidate_id']].append(record)
    native = {q['candidate_id']: q for q in results['questions']}
    need(len(native) == len(results['questions']) and set(native) == set(question_by_id), side + ':NATIVE_CANDIDATE_SET')
    for cid, question in question_by_id.items():
        row, responses = native[cid], by_question[cid]
        valid = [r for r in responses if r.get('evaluation_valid') is True]
        need(not row['data_errors'], side + ':NATIVE_QUESTION_ERRORS:' + cid)
        need(row['actual_full_calls'] == len(responses), side + ':NATIVE_ACTUAL_COUNT_MISMATCH')
        need(row['expected_full_calls'] == sum(r['candidate_id'] == cid for r in planned.values()), side + ':NATIVE_EXPECTED_COUNT_MISMATCH')
        exact = sum(r['score']['exact_match'] is True for r in valid)
        need(Fraction(row['full_exact_rate']) == Fraction(exact, len(valid)), side + ':NATIVE_EXACT_MISMATCH')
        by_variant = defaultdict(list)
        for r in valid:
            by_variant[r['variant_id']].append(Fraction(r['score']['partial_score_fraction']))
        native_mean = mean([mean(v) for v in by_variant.values()])
        need(Fraction(row['full_partial_mean']) == native_mean, side + ':NATIVE_PARTIAL_MISMATCH')
    need(set(results['selected_questions']) == {cid for cid, r in native.items() if r['selected_by_policy'] is True}, side + ':NATIVE_SELECTION_SET_MISMATCH')
    return {'manifest': manifest, 'results': results, 'native': native, 'records': records,
            'by_question': by_question, 'observed': observed,
            'inputs': [bind(p) for p in [wrapper_path, complete_path, manifest_path, results_path, *paths]]}


def summarize_group(questions, data):
    ids = {q['candidate_id'] for q in questions}
    rows = [r for cid in ids for r in data['by_question'][cid]]
    valid = [r for r in rows if r.get('evaluation_valid') is True]
    native = [data['native'][cid] for cid in sorted(ids)]
    expected = sum(r['candidate_id'] in ids for r in data['manifest']['requests'])
    ms = [r for r in valid if r['question_type'] == 'multi_select']
    exact_count = sum(r['score']['exact_match'] is True for r in valid)
    controls = {q['candidate_id'] for q in questions if qgroup(q) == 'ms_gold6_control'}
    selected = [r['candidate_id'] for r in native if r['selected_by_policy'] is True]
    return {
        'question_count': len(ids), 'root_count': len({q['root_id'] for q in questions}),
        'expected_calls': expected, 'observed_calls': len(rows), 'valid_calls': len(valid),
        'invalid_calls': len(rows) - len(valid), 'valid_over_expected': ratio(len(valid), expected),
        'exact_count': exact_count, 'exact_rate_valid_calls': ratio(exact_count, len(valid)),
        'partial_call_mean': value(mean([Fraction(r['score']['partial_score_fraction']) for r in valid])),
        'native_question_partial_macro_mean': value(mean([Fraction(r['full_partial_mean']) for r in native])),
        'native_question_exact_macro_mean': value(mean([Fraction(r['full_exact_rate']) for r in native])),
        'ms_errors': {
            'valid_ms_calls': len(ms), 'false_positive_options': sum(r['score']['fp'] for r in ms),
            'false_negative_options': sum(r['score']['fn'] for r in ms),
            'calls_with_false_positive': sum(r['score']['fp'] > 0 for r in ms),
            'calls_with_false_negative': sum(r['score']['fn'] > 0 for r in ms),
            'all_six_selected_calls': sum(len(r['score']['display_prediction']) == 6 for r in ms),
            'selected_option_count_histogram': dict(sorted(Counter(len(r['score']['display_prediction']) for r in ms).items())),
        },
        'native_policy': {
            'selected_total': len(selected), 'selected_gold6_controls': len(set(selected) & controls),
            'selected_training_candidates_excluding_gold6': len(set(selected) - controls),
            'training_candidate_ids_excluding_gold6': sorted(set(selected) - controls),
            'gold6_control_ids': sorted(set(selected) & controls),
            'selection_reason_counts': dict(Counter(r['selection_reason'] for r in native)),
            'pool_merge_performed': False,
        },
    }


def paired_group(questions, base, trained):
    ids = {q['candidate_id'] for q in questions}
    plans = [r for r in base['manifest']['requests'] if r['candidate_id'] in ids]
    pairs = [(base['observed'][r['request_id']], trained['observed'][r['request_id']]) for r in plans]
    valid = [(a, b) for a, b in pairs if a.get('evaluation_valid') is True and b.get('evaluation_valid') is True]
    transitions = Counter(('correct' if a['score']['exact_match'] else 'wrong') + '_to_' + ('correct' if b['score']['exact_match'] else 'wrong') for a, b in valid)
    differences = [Fraction(b['score']['partial_score_fraction']) - Fraction(a['score']['partial_score_fraction']) for a, b in valid]
    q_exact_diff = [Fraction(trained['native'][cid]['full_exact_rate']) - Fraction(base['native'][cid]['full_exact_rate']) for cid in sorted(ids)]
    q_partial_diff = [Fraction(trained['native'][cid]['full_partial_mean']) - Fraction(base['native'][cid]['full_partial_mean']) for cid in sorted(ids)]
    base_selected = {cid for cid in ids if base['native'][cid]['selected_by_policy'] is True}
    trained_selected = {cid for cid in ids if trained['native'][cid]['selected_by_policy'] is True}
    return {
        'question_count': len(ids), 'expected_paired_calls': len(plans), 'valid_paired_calls': len(valid),
        'exact_transition_counts': {key: transitions[key] for key in ('wrong_to_correct', 'correct_to_wrong', 'correct_to_correct', 'wrong_to_wrong')},
        'trained_minus_base_exact_rate': value(Fraction(transitions['wrong_to_correct'] - transitions['correct_to_wrong'], len(valid)) if valid else None),
        'trained_minus_base_partial_call_mean': value(mean(differences)),
        'question_exact_rate_direction': {'improved': sum(d > 0 for d in q_exact_diff), 'declined': sum(d < 0 for d in q_exact_diff), 'tied': sum(d == 0 for d in q_exact_diff)},
        'question_partial_mean_direction': {'improved': sum(d > 0 for d in q_partial_diff), 'declined': sum(d < 0 for d in q_partial_diff), 'tied': sum(d == 0 for d in q_partial_diff)},
        'native_policy_selection_overlap': {'both': sorted(base_selected & trained_selected), 'base_only': sorted(base_selected - trained_selected), 'trained_only': sorted(trained_selected - base_selected)},
    }


def assemble(scope):
    pool_path = scope / 'EVALUATION_POOL.json'
    pool = read(pool_path)
    questions = pool['questions']
    qmap = {q['candidate_id']: q for q in questions}
    need(len(qmap) == len(questions), 'DUPLICATE_POOL_CANDIDATES')
    data = {side: load_side(scope, side, qmap) for side in SIDES}
    need(data['base']['manifest']['requests'] == data['trained']['manifest']['requests'], 'PAIRED_REQUESTS_DIFFER')
    need(data['base']['manifest']['decoder_contract'] == data['trained']['manifest']['decoder_contract'], 'PAIRED_DECODERS_DIFFER')
    roots = sorted({q['root_id'] for q in questions})
    result = {
        'schema_version': 'v122_paired_read_only_summary_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PAIRED_COMPLETE_NATIVE_SCORES_CROSSCHECKED',
        'method': {
            'scope': 'Same frozen v122 questions, prompts, seeds and decoder for M0 and M_clean_policy.',
            'partial': 'Average original native partial_score_fraction values; native per-question full_partial_mean is separately retained and cross-checked. No response is rescored.',
            'denominators': 'Call rates describe repeated evaluations. Questions and roots are reported separately; permutations, rollouts and related Gold-family variants are not independent samples.',
            'interpretation': 'Descriptive paired comparison only; no independent-sample significance claim, difficulty comparison with other question pools, or unseen-holdout claim.',
            'gold6': 'Controls reported separately; excluded from training-candidate counts.',
            'training_or_pool_merge': False,
        },
        'root_count': len(roots), 'question_count': len(questions),
        'pool': bind(pool_path),
        'sides': {}, 'paired': {}, 'by_root': {}, 'by_question': [], 'by_source_status': {},
        'input_bindings': [bind(pool_path)] + [binding for side in SIDES for binding in data[side]['inputs']],
    }
    for side in SIDES:
        result['sides'][side] = {'solver_model': data[side]['manifest']['solver_model'],
                                 'groups': {group: summarize_group(members(questions, group), data[side]) for group in GROUPS}}
    result['paired'] = {group: paired_group(members(questions, group), data['base'], data['trained']) for group in GROUPS}
    provenance = pool.get('source_groups', {})
    for status in ('UNCHANGED', 'EDITED_SYNTHETIC'):
        subset = [q for q in questions if provenance.get(q['root_id'], {}).get('source_status') == status]
        result['by_source_status'][status] = {
            'root_count': len({q['root_id'] for q in subset}), 'question_count': len(subset),
            'sides': {side: {group: summarize_group(members(subset, group), data[side]) for group in GROUPS} for side in SIDES},
            'paired': {group: paired_group(members(subset, group), data['base'], data['trained']) for group in GROUPS}}
    for root in roots:
        root_qs = [q for q in questions if q['root_id'] == root]
        result['by_root'][root] = {
            'source_provenance': provenance.get(root),
            'sides': {side: {group: summarize_group(members(root_qs, group), data[side]) for group in GROUPS} for side in SIDES},
            'paired': {group: paired_group(members(root_qs, group), data['base'], data['trained']) for group in GROUPS},
        }
    for q in questions:
        cid = q['candidate_id']
        result['by_question'].append({'candidate_id': cid, 'root_id': q['root_id'], 'question_type': q['question_type'],
            'gold_count': len(q['gold']) if q['question_type'] == 'multi_select' else None, 'role': q.get('role'),
            'sides': {side: {'native_result': data[side]['native'][cid], 'metrics': summarize_group([q], data[side])} for side in SIDES},
            'paired': paired_group([q], data['base'], data['trained'])})
    return result


def pct(metric):
    return '—' if metric is None else f"{metric['decimal'] * 100:.2f}%"


def dec(metric):
    return '—' if metric is None else f"{metric['decimal']:.4f}"


def render(report):
    names = {'ms_gold2_5': 'Gold2–5 多选', 'ms_gold6_control': 'Gold6 对照', 'ordering': '排序'}
    base, trained = (report['sides'][s]['groups'] for s in SIDES)
    lines = [f"本批 {report['root_count']} 个 root、{report['question_count']} 道题的两侧评估均已完成，原生调用集合与分数复核通过。", '',
             '| 题组 | 题数 | 有效/计划调用（各模型） | M0 完全正确 | 训练后完全正确 | 变化 | 原生 partial 均值 M0→训练后 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for group, name in names.items():
        a, b = base[group], trained[group]
        delta = report['paired'][group]['trained_minus_base_exact_rate']
        calls = f"{a['valid_calls']}/{a['expected_calls']}；{b['valid_calls']}/{b['expected_calls']}"
        difference = '—' if delta is None else f"{delta['decimal'] * 100:+.2f} 个百分点"
        lines.append(f"| {name} | {a['question_count']} | {calls} | {a['exact_count']}/{a['valid_calls']} ({pct(a['exact_rate_valid_calls'])}) | {b['exact_count']}/{b['valid_calls']} ({pct(b['exact_rate_valid_calls'])}) | {difference} | {dec(a['partial_call_mean'])}→{dec(b['partial_call_mean'])} |")
    lines += ['', '| Gold 数量 | M0 完全正确 | 训练后完全正确 | M0 误选/漏选/全选调用 | 训练后误选/漏选/全选调用 |', '|---|---:|---:|---:|---:|']
    for gold in range(2, 6):
        a, b = base[f'gold{gold}'], trained[f'gold{gold}']
        ae, be = a['ms_errors'], b['ms_errors']
        lines.append(f"| {gold} | {a['exact_count']}/{a['valid_calls']} ({pct(a['exact_rate_valid_calls'])}) | {b['exact_count']}/{b['valid_calls']} ({pct(b['exact_rate_valid_calls'])}) | {ae['calls_with_false_positive']}/{ae['calls_with_false_negative']}/{ae['all_six_selected_calls']} | {be['calls_with_false_positive']}/{be['calls_with_false_negative']}/{be['all_six_selected_calls']} |")
    lines += ['', '| root | Gold2–5 完全正确 M0→训练后 | 排序完全正确 M0→训练后 |', '|---|---:|---:|']
    for root, row in report['by_root'].items():
        a, b = row['sides']['base'], row['sides']['trained']
        cells = []
        for group in ('ms_gold2_5', 'ordering'):
            cells.append(f"{pct(a[group]['exact_rate_valid_calls'])}→{pct(b[group]['exact_rate_valid_calls'])}")
        lines.append(f"| {root} | {' | '.join(cells)} |")
    for side, name in (('base', 'M0'), ('trained', '训练后')):
        pol = report['sides'][side]['groups']['all']['native_policy']
        lines += ['', f"{name} 原生策略选中 {pol['selected_total']} 题，其中 Gold6 对照 {pol['selected_gold6_controls']} 题；剔除对照后的训练候选 {pol['selected_training_candidates_excluding_gold6']} 题。"]
    lines += ['', '上述分母是重复调用数；排列、rollout 和同源 Gold2–5 family 变体不视为独立样本。配对结果仅描述当前题池，不能据此与其他批次直接判定难度变化。未训练、未合并训练池。', '',
              '详细结果、逐题原生输出摘要、每个 root、误选/漏选的选项总数、配对转移与全部输入哈希见 `PAIRED_SOLVER_SUMMARY_V1.json`。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', type=Path, default=SCOPE)
    parser.add_argument('--output-json', type=Path, default=BATCH / 'delivery/PAIRED_SOLVER_SUMMARY_V1.json')
    parser.add_argument('--output-md', type=Path, default=BATCH / 'delivery/PAIRED_SOLVER_REPORT_ZH_V1.md')
    args = parser.parse_args()
    need(not args.output_json.exists() and not args.output_md.exists(), 'EXCLUSIVE_REPORT_OUTPUT_ALREADY_EXISTS')
    report = assemble(args.scope.resolve())
    report['report_tool'] = bind(Path(__file__))
    rendered = render(report)
    for path in (args.output_json, args.output_md):
        need(path.resolve().parent == (BATCH / 'delivery').resolve(), 'OUTPUT_MUST_BE_ADDITIVE_DELIVERY_FILE')
    with args.output_json.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    with args.output_md.open('x', encoding='utf-8') as handle:
        handle.write(rendered)
    print(json.dumps({'status': report['status'], 'roots': report['root_count'], 'questions': report['question_count'],
                      'json': str(args.output_json), 'markdown': str(args.output_md)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
