"""CPU-only recomputation over every frozen v134 request, with fixed denominators.

Invalid, duplicate and missing calls count as exact=0. Other unknown metrics
retain explicit bounds. No score, native selection diagnostic or parse failure
can remove a question or change its denominator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as ev

HELPERS_PATH = ev.V127 / 'report_paired.py'
HELPERS_SHA = 'b192a4f0f280ec90b44f90f837b8ae87043e6e0b8f24d84ef2041c0ddf0250ad'
STRATA = ['all', 'main_excluding_gold6', 'multi_select', 'multi_select_gold2345',
          'gold6_control', 'gold2', 'gold3', 'gold4', 'gold5', 'gold6', 'ordering']
PAIRS = [('base', 'clean'), ('clean', 'new'), ('base', 'new')]


def libraries():
    ev.require(ev.sha(HELPERS_PATH) == HELPERS_SHA, 'FROZEN_V127_REPORT_HELPERS_CHANGED')
    ev.require(ev.sha(ev.NATIVE / 'scoring.py') == ev.NATIVE_SCORING_SHA, 'NATIVE_SCORING_CHANGED')
    return (ev.module('v134_frozen_report_helpers', HELPERS_PATH),
            ev.module('v134_frozen_scoring', ev.NATIVE / 'scoring.py'))


def strata(question):
    if question['question_type'] == 'ordering':
        return ['all', 'main_excluding_gold6', 'ordering']
    g = len(question['gold'])
    return ['all', 'multi_select', 'gold' + str(g)] + (
        ['gold6_control'] if g == 6 else ['main_excluding_gold6', 'multi_select_gold2345'])


def score_call(question, request, manifest, gold, found, scoring):
    """Validate one planned call; ``found`` deliberately permits duplicates."""
    qtype, g = question['question_type'], len(question['gold'])
    call = {'model': manifest['solver_model'], 'request_id': request['request_id'],
            'candidate_id': question['candidate_id'], 'root_id': question['root_id'], 'split': question['split'],
            'question_type': qtype, 'gold_count': g, 'variant_id': request['variant']['variant_id'],
            'slot': request['slot'], 'seed': request['seed'], 'prompt_sha256': request['prompt_sha256'],
            'valid': False, 'exact_match': 0, 'signed_partial': None, 'pair_accuracy': None,
            'tp': None, 'fp': None, 'fn': None, 'predicted_count': None, 'predicted_minus_gold_count': None,
            'all6': None, 'overselection': None, 'underselection': None, 'correct_pairs': None, 'wrong_pairs': None,
            'parse_status': None, 'scoring_status': None, 'errors': [],
            'raw_output': None, 'metadata': None, 'display_prediction': None, 'content_prediction': None}
    if len(found) != 1:
        call['errors'].append('MISSING_RESPONSE' if not found else 'DUPLICATE_RESPONSE')
        call['response_paths'] = [str(path) for path, row in found]
        return call
    path, response = found[0]
    call.update(response_path=str(path), raw_output=response.get('raw_output'), metadata=response.get('metadata'))
    expected = {'request_id': request['request_id'], 'candidate_id': question['candidate_id'],
                'root_id': question['root_id'], 'question_type': qtype,
                'variant_id': request['variant']['variant_id'], 'slot': request['slot'], 'seed': request['seed'],
                'mode': 'full_context', 'solver_model': manifest['solver_model'], 'model_path': manifest['model']}
    for key, value in expected.items():
        if response.get(key) != value or key in {'slot', 'seed'} and type(response.get(key)) is not int:
            call['errors'].append('RESPONSE_PLAN_MISMATCH:' + key)
    metadata = response.get('metadata') or {}
    if not isinstance(metadata, dict):
        call['errors'].append('RESPONSE_METADATA_NOT_OBJECT')
        metadata = {}
    for key in ['temperature', 'top_p', 'max_tokens', 'structured_output_api', 'structured_output_version']:
        if metadata.get(key) != manifest['decoder_contract'].get(key):
            call['errors'].append('DECODER_METADATA_MISMATCH:' + key)
    if metadata.get('finish_reason') not in ['stop', 'eos']:
        call['errors'].append('BAD_FINISH_REASON')
    if response.get('evaluation_valid') is not True:
        call['errors'].append('NATIVE_EVALUATION_INVALID')
    if response.get('data_errors'):
        call['errors'].append('NATIVE_DATA_ERRORS')
    score = scoring.score_response(dict(question, gold_content_ids=gold), request, response.get('raw_output'))
    if score != response.get('score'):
        call['errors'].append('SAVED_SCORE_DIFFERS_FROM_NATIVE_RECOMPUTATION')
    if not score['scored']:
        call['errors'].append(score.get('scoring_status', 'UNSCORED'))
    call.update(parse_status=score['parse_status'], scoring_status=score.get('scoring_status'),
                display_prediction=score['display_prediction'], content_prediction=score['content_prediction'])
    if not call['errors']:
        call.update(valid=True, exact_match=int(score['exact_match']), signed_partial=score['partial_score_fraction'])
        if qtype == 'multi_select':
            count = len(score['content_prediction'])
            call.update(tp=score['tp'], fp=score['fp'], fn=score['fn'], predicted_count=count,
                        predicted_minus_gold_count=count - g, all6=int(count == 6),
                        overselection=int(count > g), underselection=int(count < g))
        else:
            call.update(correct_pairs=score['tp'], wrong_pairs=score['fp'], pair_accuracy=str(Fraction(score['tp'], 15)))
    return call


def all_select_call(question, request, gold, scoring):
    """Analytic six-option baseline, never an inference call."""
    score = scoring.score_response(dict(question, gold_content_ids=gold), request, '<box>A,B,C,D,E,F</box>')
    g = len(gold)
    return {'question_type': 'multi_select', 'gold_count': g, 'valid': True, 'parse_status': 'success',
            'exact_match': int(score['exact_match']), 'signed_partial': score['partial_score_fraction'],
            'tp': score['tp'], 'fp': score['fp'], 'fn': score['fn'], 'predicted_count': 6,
            'predicted_minus_gold_count': 6 - g, 'all6': 1, 'overselection': int(g < 6), 'underselection': 0}


def question_aggregate(candidates, model, helper):
    rows = [q['models'][model] for q in candidates]
    result = helper.aggregate_candidates(rows)
    for qtype, subkey, metric in [('ordering', 'ordering', 'pair_accuracy'),
                                 ('multi_select', 'multi_select', 'predicted_count'),
                                 ('multi_select', 'multi_select', 'all6_rate'),
                                 ('multi_select', 'multi_select', 'overselection_rate')]:
        values = [q['models'][model][subkey][metric] for q in candidates if q['question_type'] == qtype]
        if values:
            complete = all(v['mean_fraction'] is not None for v in values)
            result[qtype + '_' + metric + '_equal_candidate'] = {
                'candidate_count': len(values),
                'mean': float(helper.mean([Fraction(v['mean_fraction']) for v in values])) if complete else None,
                'lower': float(helper.mean([Fraction(v['lower_fraction']) for v in values])),
                'upper': float(helper.mean([Fraction(v['upper_fraction']) for v in values]))}
    return result


def paired_candidate_delta(a, b, helper):
    if not a['planned_calls'] or not b['planned_calls']:
        return {'exact_rate_delta': None, 'signed_partial_delta': None}
    return helper.paired_delta(a, b)


def group_summary(candidates, per_call, helper):
    ids = {q['candidate_id'] for q in candidates}
    result = {'candidate_count': len(candidates), 'covered_root_count': len({q['root_id'] for q in candidates}), 'models': {}}
    for model in ev.MODELS:
        calls = [row['models'][model] for row in per_call if row['candidate_id'] in ids]
        by_root = [[q for q in candidates if q['root_id'] == root] for root in sorted({q['root_id'] for q in candidates})]
        root_values = [question_aggregate(rows, model, helper) for rows in by_root]
        root_exact = helper.mean([Fraction(v['exact_rate_equal_candidate_fraction']) for v in root_values]) if root_values else None
        root_lows = [v['signed_partial_equal_candidate_lower'] for v in root_values]
        root_highs = [v['signed_partial_equal_candidate_upper'] for v in root_values]
        complete = all(v['signed_partial_equal_candidate'] is not None for v in root_values)
        result['models'][model] = {'call_weighted': helper.aggregate_calls(calls),
            'question_equal': question_aggregate(candidates, model, helper),
            'root_equal_among_covered': {'root_count': len(root_values),
                'exact_rate': helper.number(root_exact), 'exact_rate_fraction': helper.fraction_text(root_exact),
                'signed_partial': sum(v['signed_partial_equal_candidate'] for v in root_values) / len(root_values) if root_values and complete else None,
                'signed_partial_lower': sum(root_lows) / len(root_lows) if root_lows else None,
                'signed_partial_upper': sum(root_highs) / len(root_highs) if root_highs else None}}
    result['paired_deltas'] = {}
    for left, right in PAIRS:
        a, b = result['models'][left], result['models'][right]
        key = right + '_minus_' + left
        candidate_deltas = [q['paired_deltas'][key] for q in candidates]
        result['paired_deltas'][key] = {'exact_rate_question_equal':
            float(helper.mean([Fraction(d['exact_rate_delta_fraction']) for d in candidate_deltas])) if candidate_deltas else None,
            'signed_partial_question_equal':
            float(helper.mean([Fraction(d['signed_partial_delta_fraction']) for d in candidate_deltas]))
            if candidate_deltas and all(d['signed_partial_delta_fraction'] is not None for d in candidate_deltas) else None,
            'signed_partial_question_equal_lower':
            sum(d['signed_partial_delta_lower'] for d in candidate_deltas) / len(candidate_deltas) if candidate_deltas else None,
            'signed_partial_question_equal_upper':
            sum(d['signed_partial_delta_upper'] for d in candidate_deltas) / len(candidate_deltas) if candidate_deltas else None,
            'exact_rate_root_equal': b['root_equal_among_covered']['exact_rate'] - a['root_equal_among_covered']['exact_rate'] if candidates else None}
    return result


def main(output_dir=None):
    ev.frozen(check_models=False)
    helper, scoring = libraries()
    out = Path(output_dir or HERE / 'reports').resolve()
    ev.require(not out.is_relative_to(HERE / 'run') and out != HERE, 'REPORT_OUTPUT_MUST_BE_SEPARATE')
    names = ['SUMMARY.json', 'PER_CALL.json', 'PER_QUESTION.json', 'PER_QUESTION.csv', 'PER_ROOT.json',
             'STRATIFIED.json', 'ALL_SELECT_BASELINE.json', 'REPORT_ZH.md']
    ev.require(not any((out / name).exists() for name in names), 'REPORT_EXISTS_NO_SILENT_OVERWRITE')
    bindings, errors = {}, []
    def read(path):
        path = Path(path).resolve()
        bindings[str(path)] = ev.sha(path)
        return ev.read(path)
    def problem(code, **details):
        errors.append({'code': code, **details})
    # These bindings make the report independently traceable to the approved
    # preflight and its complete transitive tool/input freeze.
    for path in [HERE / 'PREFLIGHT_FREEZE.json', HERE / 'TOOLS_FREEZE.json']:
        if path.exists():
            read(path)
    scope = read(HERE / 'SCOPE.json')
    plan = read(HERE / 'SHARED_REQUESTS.json')
    canonical = read(HERE / 'EVALUATION_POOL.json')['questions']
    by_id = {q['candidate_id']: q for q in canonical}
    requests = plan['requests']
    ev.require(len(by_id) == len(canonical) == scope['question_count'], 'CANONICAL_QUESTION_COUNT_CHANGED')
    ev.require(len({r['request_id'] for r in requests}) == len(requests) == scope['calls_per_model'], 'FROZEN_REQUEST_SET_CHANGED')
    manifests, callsets, golds = {}, {}, {}
    for model in ev.MODELS:
        run = HERE / 'run' / model
        manifest = manifests[model] = read(run / 'manifest.json')
        gold = golds[model] = read(run / 'gold.json')
        ev.require(manifest['requests'] == requests and manifest['decoder_contract'] == plan['decoder_contract'], 'PAIRED_PLAN_CHANGED:' + model)
        observed = defaultdict(list)
        for path in sorted((run / 'responses').glob('*.json')):
            try:
                response = read(path)
                if not isinstance(response, dict):
                    raise ValueError('RESPONSE_NOT_OBJECT')
                observed[response.get('request_id')].append((path, response))
            except (ValueError, OSError) as error:
                problem('RESPONSE_FILE_UNREADABLE', model=model, path=str(path), error=type(error).__name__)
        for rid in set(observed) - {r['request_id'] for r in requests}:
            problem('UNPLANNED_RESPONSE', model=model, request_id=rid)
        if not (run / 'COMPLETE.json').exists():
            problem('NATIVE_RUN_NOT_COMPLETE', model=model)
        else:
            complete = read(run / 'COMPLETE.json')
            if complete.get('actual_call_count') != len(requests):
                problem('NATIVE_COMPLETE_COUNT_DIFFERS', model=model)
        if not (run / 'RESULTS.json').exists():
            problem('NATIVE_RESULTS_MISSING', model=model)
        else:
            result = read(run / 'RESULTS.json')
            if len(result['questions']) != len(canonical) or {q['candidate_id'] for q in result['questions']} != set(by_id):
                problem('NATIVE_RESULTS_QUESTION_SET_DIFFERS', model=model)
            # Native selected_questions is never used to select heldout questions.
            if result.get('call_set_errors'):
                problem('NATIVE_CALL_SET_ERRORS', model=model, errors=result['call_set_errors'])
        calls = []
        for request in requests:
            question = by_id[request['candidate_id']]
            expected_gold = ['content_' + hashlib.sha256(question['options' if question['question_type'] == 'multi_select' else 'items'][label].strip().encode()).hexdigest()[:16] for label in question['gold']]
            ev.require(gold[question['candidate_id']] == expected_gold, 'GOLD_CONTENT_BINDING_CHANGED')
            prompt = run / 'prompts' / (request['request_id'] + '.txt')
            ev.require(ev.sha(prompt) == request['prompt_sha256'], 'FROZEN_PROMPT_CHANGED')
            call = score_call(question, request, manifest, gold[question['candidate_id']], observed.get(request['request_id'], []), scoring)
            if call['errors']:
                problem('CALL_INVALID', model=model, request_id=request['request_id'], reasons=call['errors'])
            calls.append(call)
        callsets[model] = calls
    per_call = []
    for i, request in enumerate(requests):
        q = by_id[request['candidate_id']]
        row = {k: q[k] for k in ['candidate_id', 'root_id', 'split', 'question_type']}
        row.update(request_id=request['request_id'], models={model: callsets[model][i] for model in ev.MODELS})
        row['paired_deltas'] = {}
        for left, right in PAIRS:
            a, b = row['models'][left], row['models'][right]
            row['paired_deltas'][right + '_minus_' + left] = {
                'exact_match': b['exact_match'] - a['exact_match'],
                'signed_partial': str(Fraction(b['signed_partial']) - Fraction(a['signed_partial']))
                if a['signed_partial'] is not None and b['signed_partial'] is not None else None}
        per_call.append(row)
    per_question, all_select_questions, all_select_calls = [], [], []
    for q in canonical:
        cid = q['candidate_id']
        row = {key: q[key] for key in ['candidate_id', 'root_id', 'split', 'question_type']}
        row.update(gold_count=len(q['gold']), strata=strata(q), models={})
        for model in ev.MODELS:
            row['models'][model] = helper.aggregate_calls([r for r in callsets[model] if r['candidate_id'] == cid])
        row['paired_deltas'] = {right + '_minus_' + left: paired_candidate_delta(row['models'][left], row['models'][right], helper) for left, right in PAIRS}
        if q['question_type'] == 'multi_select':
            calls = [all_select_call(q, r, golds['base'][cid], scoring) for r in requests if r['candidate_id'] == cid]
            baseline = helper.aggregate_calls(calls)
            row['all_select_baseline'] = baseline
            all_select_questions.append({**{k: row[k] for k in ['candidate_id', 'root_id', 'split', 'gold_count', 'strata']}, 'metrics': baseline})
            all_select_calls.extend([{**r, 'candidate_id': cid, 'root_id': q['root_id'], 'split': q['split']} for r in calls])
        per_question.append(row)
    stratified = {}
    for split in ['all', 'dev', 'test']:
        selected = [q for q in per_question if split == 'all' or q['split'] == split]
        stratified[split] = {name: group_summary([q for q in selected if name in q['strata']], per_call, helper) for name in STRATA}
    per_root = []
    for split, roots in scope['fixed_full_heldout_roots'].items():
        for root in roots:
            selected = [q for q in per_question if q['root_id'] == root]
            per_root.append({'root_id': root, 'split': split, 'eligible_this_increment': root in scope['fixed_incremental_roots'][split],
                             'has_questions': bool(selected), 'question_count': len(selected),
                             'strata': {name: group_summary([q for q in selected if name in q['strata']], per_call, helper) for name in STRATA}})
    baseline_report = {'kind': 'ANALYTIC_NO_INFERENCE', 'prediction': '<box>A,B,C,D,E,F</box>',
        'interpretation': 'Exact=0 and signed partial=0 on Gold2-5; exact=1 and signed partial=1 on Gold6. Ordering has no all-select baseline.',
        'questions': all_select_questions, 'strata': {}}
    for split in ['all', 'dev', 'test']:
        baseline_report['strata'][split] = {name: helper.aggregate_calls([r for r in all_select_calls
            if (split == 'all' or r['split'] == split) and name in strata(by_id[r['candidate_id']])]) for name in STRATA if name != 'ordering'}
    summary = {'schema_version': 'v134_three_model_fixed_denominator_report_v1',
        'status': 'COMPLETE' if not errors else 'INCOMPLETE_OR_INVALID', 'errors': errors,
        'question_count': len(canonical), 'question_counts_by_split': dict(Counter(q['split'] for q in canonical)),
        'question_counts_by_type': dict(Counter(q['question_type'] for q in canonical)),
        'calls_per_model': len(requests), 'models': ev.MODELS,
        'fixed_incremental_roots': scope['fixed_incremental_roots'], 'fixed_full_heldout_roots': scope['fixed_full_heldout_roots'],
        'covered_roots': sorted({q['root_id'] for q in canonical}),
        'primary_strata': {'dev': stratified['dev']['main_excluding_gold6'], 'test': stratified['test']['main_excluding_gold6']},
        'all_question_strata': {split: stratified[split]['all'] for split in ['all', 'dev', 'test']},
        'failure_policy': 'Every planned call stays in its denominator. Missing/duplicate/invalid calls exact=0; undefined other metrics retain explicit bounds, never valid-only replacement.',
        'weighting': 'Call weighted, question equal, and root equal among covered roots are separately named. Uncovered roots remain explicit null rows.',
        'selection': 'All frozen content-pass questions retained, no Solver-based filtering. Gold6 separately reported as a control.',
        'limits': ['Descriptive paired evaluation on newly added conflict-isolated heldout questions.',
                   'Permutations, rollouts, Gold families, and questions from the same root are correlated.',
                   'Does not establish causal effects of expansion or training; template dialogue order is not complex real-world temporal reasoning.',
                   'Legacy v124 pools remain immutable; this report is an incremental batch, not a silently recomputed merged test set.'],
        'input_bindings': bindings}
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('SUMMARY.json', summary), ('PER_CALL.json', per_call), ('PER_QUESTION.json', per_question),
                        ('PER_ROOT.json', per_root), ('STRATIFIED.json', stratified), ('ALL_SELECT_BASELINE.json', baseline_report)]:
        ev.save(out / name, value)
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=['candidate_id', 'root_id', 'split', 'question_type', 'gold_count', 'model',
        'planned_calls', 'valid_calls', 'exact_rate', 'signed_partial', 'pair_accuracy', 'all6_rate', 'overselection_rate'])
    writer.writeheader()
    for q in per_question:
        for model, metrics in q['models'].items():
            writer.writerow({**{k: q[k] for k in ['candidate_id', 'root_id', 'split', 'question_type', 'gold_count']},
                'model': model, 'planned_calls': metrics['planned_calls'], 'valid_calls': metrics['valid_calls'],
                'exact_rate': metrics['exact_rate_fixed'], 'signed_partial': metrics.get('signed_partial', {}).get('mean'),
                'pair_accuracy': metrics.get('ordering', {}).get('pair_accuracy', {}).get('mean'),
                'all6_rate': metrics.get('multi_select', {}).get('all6_rate', {}).get('mean'),
                'overselection_rate': metrics.get('multi_select', {}).get('overselection_rate', {}).get('mean')})
    with (out / 'PER_QUESTION.csv').open('x') as handle:
        handle.write(text.getvalue())
    lines = [f"v134 三模型增量评估：{summary['status']}。", '',
             f"冻结题目 {len(canonical)}，每模型计划调用 {len(requests)}；完整性错误 {len(errors)} 条。", '',
             'DEV、TEST 分开报告，Gold6 单列为全选对照。下表为主任务（排除 Gold6）的题目等权 exact；所有失败调用保留分母。', '',
             '| Split | Base | M_clean | Round2 |', '|---|---:|---:|---:|']
    for split in ['dev', 'test']:
        values = [stratified[split]['main_excluding_gold6']['models'][m]['question_equal']['exact_rate_equal_candidate'] for m in ev.MODELS]
        lines.append('| ' + split + ' | ' + ' | '.join('—' if v is None else f'{v:.2%}' for v in values) + ' |')
    lines += ['', '排序 pair accuracy、多选 Gold2–6、全选/过选、逐 root 和固定分母详见 STRATIFIED.json、PER_QUESTION.json、PER_CALL.json、PER_ROOT.json。',
              'ALL_SELECT_BASELINE.json 是解析的全选基线，无模型调用。无题 root 保留 null，不将无题记为零分。',
              '排列、重复采样、Gold 家族和同 root 问题并非独立样本；本增量报告不能证明扩充或训练的因果收益。']
    with (out / 'REPORT_ZH.md').open('x') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(ev.json.dumps({'status': summary['status'], 'errors': len(errors), 'report': str(out)}))
    return 0 if not errors else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    raise SystemExit(main(args.output_dir))
