"""CPU-only new-structure pilot report, with frozen admitted-call denominators.

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
          'gold6_control', 'gold2', 'gold3', 'gold4', 'gold5', 'gold6', 'ordering',
          'new_structure', 'old_structure_control', 'new_multi_select', 'old_multi_select',
          'new_ordering', 'old_ordering', 'new_gold3', 'old_gold3',
          'ordering_scenes_3-2-1', 'ordering_scenes_2-2-2', 'ordering_scenes_4-2', 'ordering_scenes_1-1-1-1-1-1']
PAIRS = [('clean', 'new')]
SPLITS = ['all', 'dev']


def libraries():
    ev.require(ev.sha(HELPERS_PATH) == HELPERS_SHA, 'FROZEN_V127_REPORT_HELPERS_CHANGED')
    ev.require(ev.sha(ev.NATIVE / 'scoring.py') == ev.NATIVE_SCORING_SHA, 'NATIVE_SCORING_CHANGED')
    return (ev.module('v140_frozen_report_helpers', HELPERS_PATH),
            ev.module('v140_frozen_scoring', ev.NATIVE / 'scoring.py'))


def strata(question):
    qtype = question['question_type']
    arm = question['pilot_arm']
    prefix = 'new' if arm == 'new_structure' else 'old'
    extra = [arm, prefix + '_' + qtype]
    if qtype == 'ordering':
        signature = ev.scene_signature(question)
        return ['all', 'main_excluding_gold6', 'ordering',
                'ordering_scenes_' + '-'.join(map(str, signature))] + extra
    g = len(question['gold'])
    if g == 3:
        extra.append(prefix + '_gold3')
    return ['all', 'multi_select', 'gold' + str(g)] + (
        ['gold6_control'] if g == 6 else ['main_excluding_gold6', 'multi_select_gold2345']) + extra


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
             'STRATIFIED.json', 'ALL_SELECT_BASELINE.json', 'ARMS.json', 'ADMISSION_ACCOUNTING.json', 'REPORT_ZH.md']
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
            # Native selected_questions is diagnostic and never filters admitted questions.
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
        row = {k: q[k] for k in ['candidate_id', 'root_id', 'split', 'question_type', 'pilot_arm', 'slot_id']}
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
        row = {key: q[key] for key in ['candidate_id', 'root_id', 'split', 'question_type', 'pilot_arm', 'slot_id']}
        row.update(gold_count=len(q['gold']), strata=strata(q), scene_count_signature=ev.scene_signature(q),
                   original_candidate_id=q.get('original_candidate_id'), models={})
        for model in ev.MODELS:
            row['models'][model] = helper.aggregate_calls([r for r in callsets[model] if r['candidate_id'] == cid])
        row['paired_deltas'] = {right + '_minus_' + left: paired_candidate_delta(row['models'][left], row['models'][right], helper) for left, right in PAIRS}
        if q['question_type'] == 'multi_select':
            calls = [all_select_call(q, r, golds[next(iter(ev.MODELS))][cid], scoring) for r in requests if r['candidate_id'] == cid]
            baseline = helper.aggregate_calls(calls)
            row['all_select_baseline'] = baseline
            all_select_questions.append({**{k: row[k] for k in ['candidate_id', 'root_id', 'split', 'gold_count', 'strata']}, 'metrics': baseline})
            all_select_calls.extend([{**r, 'candidate_id': cid, 'root_id': q['root_id'], 'split': q['split']} for r in calls])
        per_question.append(row)
    stratified = {}
    for split in SPLITS:
        selected = [q for q in per_question if split == 'all' or q['split'] == split]
        stratified[split] = {name: group_summary([q for q in selected if name in q['strata']], per_call, helper) for name in STRATA}
    per_root = []
    for split, roots in scope['planned_roots'].items():
        for root in roots:
            selected = [q for q in per_question if q['root_id'] == root]
            per_root.append({'root_id': root, 'split': split, 'eligible_this_increment': root in scope['fixed_incremental_roots'][split],
                             'has_questions': bool(selected), 'question_count': len(selected),
                             'strata': {name: group_summary([q for q in selected if name in q['strata']], per_call, helper) for name in STRATA}})
    baseline_report = {'kind': 'ANALYTIC_NO_INFERENCE', 'prediction': '<box>A,B,C,D,E,F</box>',
        'interpretation': 'Exact=0 and signed partial=0 on Gold2-5; exact=1 and signed partial=1 on Gold6. Ordering has no all-select baseline.',
        'questions': all_select_questions, 'strata': {}}
    for split in SPLITS:
        baseline_report['strata'][split] = {name: helper.aggregate_calls([r for r in all_select_calls
            if (split == 'all' or r['split'] == split) and name in strata(by_id[r['candidate_id']])]) for name in STRATA if name != 'ordering'}
    summary = {'schema_version': 'v140_two_model_dev_structure_fixed_denominator_report_v1',
        'status': 'COMPLETE' if not errors else 'INCOMPLETE_OR_INVALID', 'errors': errors,
        'question_count': len(canonical), 'question_counts_by_split': dict(Counter(q['split'] for q in canonical)),
        'question_counts_by_type': dict(Counter(q['question_type'] for q in canonical)),
        'calls_per_model': len(requests), 'models': ev.MODELS,
        'planned_roots': scope['planned_roots'], 'admission_accounting': scope['admission_accounting'],
        'question_counts_by_arm': dict(Counter(q['pilot_arm'] for q in canonical)),
        'scope': 'KNOWN_DEV_ROOTS_NEW_STRUCTURE_INTERNAL_PILOT',
        'covered_roots': sorted({q['root_id'] for q in canonical}),
        'primary_strata': {'dev': stratified['dev']['all']},
        'all_question_strata': {split: stratified[split]['all'] for split in SPLITS},
        'failure_policy': 'Every planned call stays in its denominator. Missing/duplicate/invalid calls exact=0; undefined other metrics retain explicit bounds, never valid-only replacement.',
        'weighting': 'Call weighted, question equal, and root equal among covered roots are separately named. Uncovered roots remain explicit null rows.',
        'selection': 'All content-admitted questions retained; review rejects remain in fixed 32-slot generation accounting, not Solver accuracy denominators. No score-based filtering or replacements.',
        'limits': ['Four known DEV roots have historical generation/review/Solver exposure; not pristine heldout roots.',
                   'Same full sources and paired fixed model inference, but new/old MC Gold distributions and evidence differ. Pooled differences are descriptive, not tightly controlled causal estimates.',
                   'Gold3-only old/new MC and ordering arms are shown separately; questions still differ semantically.',
                   'Internal separate task actors share inherited model and filesystem; no external-blind independence claim.',
                   'Permutations, repeated samples and same-root questions are correlated. Small pilot does not establish broad robustness or expansion/feedback causal benefit.',
                   'No new training or source expansion occurs; all historical sealed evaluations remain unchanged.'],
        'input_bindings': bindings}
    arms = {}
    for arm in ['new_structure', 'old_structure_control']:
        selected = [q for q in per_question if q['pilot_arm'] == arm]
        arms[arm] = {'all': group_summary(selected, per_call, helper),
            'by_type': {kind: group_summary([q for q in selected if q['question_type'] == kind], per_call, helper)
                        for kind in ['multi_select', 'ordering']},
            'by_gold': {str(g): group_summary([q for q in selected if q['question_type'] == 'multi_select' and q['gold_count'] == g], per_call, helper)
                        for g in [2, 3, 4, 5]},
            'by_root': {rid: {name: group_summary([q for q in selected if q['root_id'] == rid and
                             (name == 'all' or name in q['strata'])], per_call, helper)
                         for name in ['all', 'multi_select', 'gold3', 'ordering']}
                        for rid in scope['planned_roots']['dev']}}
    scene_groups = {rid: {signature: group_summary([q for q in per_question if q['root_id'] == rid
                      and 'ordering_scenes_' + signature in q['strata']], per_call, helper)
                         for signature in ['3-2-1', '2-2-2', '4-2', '1-1-1-1-1-1']}
                   for rid in scope['planned_roots']['dev']}
    arm_report = {'arms': arms, 'ordering_by_root_and_scene_signature': scene_groups,
                  'interpretation': 'Question-equal means. Gold3 old/new is explicitly separated; small known-DEV-root descriptive comparison, not a matched semantic or causal experiment.'}
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('SUMMARY.json', summary), ('PER_CALL.json', per_call), ('PER_QUESTION.json', per_question),
                        ('PER_ROOT.json', per_root), ('STRATIFIED.json', stratified), ('ALL_SELECT_BASELINE.json', baseline_report),
                        ('ARMS.json', arm_report), ('ADMISSION_ACCOUNTING.json', scope['admission_accounting'])]:
        ev.save(out / name, value)
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=['candidate_id', 'root_id', 'split', 'question_type', 'pilot_arm', 'slot_id', 'gold_count', 'model',
        'planned_calls', 'valid_calls', 'exact_rate', 'signed_partial', 'pair_accuracy', 'all6_rate', 'overselection_rate'])
    writer.writeheader()
    for q in per_question:
        for model, metrics in q['models'].items():
            writer.writerow({**{k: q[k] for k in ['candidate_id', 'root_id', 'split', 'question_type', 'pilot_arm', 'slot_id', 'gold_count']},
                'model': model, 'planned_calls': metrics['planned_calls'], 'valid_calls': metrics['valid_calls'],
                'exact_rate': metrics['exact_rate_fixed'], 'signed_partial': metrics.get('signed_partial', {}).get('mean'),
                'pair_accuracy': metrics.get('ordering', {}).get('pair_accuracy', {}).get('mean'),
                'all6_rate': metrics.get('multi_select', {}).get('all6_rate', {}).get('mean'),
                'overselection_rate': metrics.get('multi_select', {}).get('overselection_rate', {}).get('mean')})
    with (out / 'PER_QUESTION.csv').open('x') as handle:
        handle.write(text.getvalue())
    accounting = scope['admission_accounting']
    lines = [f"v140 新结构 DEV 小批实验：{summary['status']}。", '',
             f"预先固定 {accounting['planned_slots']} 个生成槽位；通过内容审核 {len(canonical)} 题，未入选 {accounting['rejected_count']} 题。覆盖 {len(summary['covered_roots'])} 个已知 DEV root。每模型计划调用 {len(requests)} 次，完整性错误 {len(errors)} 条。", '',
             'M-clean 为第二轮训练起点，Round2 为第二轮训练后的冻结模型。本次不训练，使用完整对话。每题多选 6 种排列 × 3 次采样，排序 12 种排列 × 1 次采样。', '',
             '下表为题目等权平均。失败调用保留分母，整题正确率记 0；未定义的部分得分保留上下界。未入选的生成槽位另记在审核成功率中。', '',
             '| 范围 | 题数 | M-clean 整题 | Round2 整题 | M-clean 部分得分 | Round2 部分得分 |',
             '|---|---:|---:|---:|---:|---:|']
    titles = {'all': '全部入选题', 'new_structure': '新结构全部', 'old_structure_control': '旧结构对照全部',
              'new_multi_select': '新多选', 'old_multi_select': '旧多选对照',
              'new_gold3': '新多选 Gold3', 'old_gold3': '旧多选 Gold3',
              'new_ordering': '新排序', 'old_ordering': '旧排序对照',
              'gold2': '全部多选 Gold2', 'gold3': '全部多选 Gold3',
              'gold4': '全部多选 Gold4', 'gold5': '全部多选 Gold5',
              'ordering_scenes_3-2-1': '排序场景项数 3/2/1', 'ordering_scenes_2-2-2': '排序场景项数 2/2/2',
              'ordering_scenes_4-2': '排序场景项数 4/2', 'ordering_scenes_1-1-1-1-1-1': '旧排序场景项数 1/1/1/1/1/1'}
    def group_row(title, group):
        points = [group['models'][m]['question_equal'] for m in ['clean', 'new']]
        values = [x['exact_rate_equal_candidate'] for x in points] + [x['signed_partial_equal_candidate'] for x in points]
        return '| ' + title + ' | ' + str(group['candidate_count']) + ' | ' + ' | '.join('—' if v is None else f'{v:.2%}' for v in values) + ' |'
    for name, title in titles.items():
        lines.append(group_row(title, stratified['dev'][name]))
    lines += ['', '逐 root 结果（题目等权）：', '',
              '| root / 范围 | 题数 | M-clean 整题 | Round2 整题 | M-clean 部分得分 | Round2 部分得分 |',
              '|---|---:|---:|---:|---:|---:|']
    for rid in scope['planned_roots']['dev']:
        for arm, title in [('new_structure', '新'), ('old_structure_control', '旧')]:
            for layer in ['gold3', 'ordering']:
                lines.append(group_row(rid + ' / ' + title + layer, arms[arm]['by_root'][rid][layer]))
        for signature, group in scene_groups[rid].items():
            lines.append(group_row(rid + ' / 排序场景 ' + signature, group))
    lines += ['', '多选 signed partial = TP/g − FP/(6−g)；排序 signed partial = (正确成对顺序数−错误成对顺序数)/15。pair accuracy 另见分层文件。Gold2–5 全选基线整题与部分得分均为 0。',
              'ARMS.json 提供按新旧、题型、Gold、root、排序场景组合的结果；PER_QUESTION/PER_CALL 保留逐题与逐调用。ADMISSION_ACCOUNTING.json 保留固定生成槽位及失败记录。', '',
              '这些 root 曾用于 DEV 生成、审核和 Solver 评估，不能称作全新未见 root。新旧多选 Gold 分布不同，必须同时阅读 Gold3 单层；即使同为 Gold3，选项与证据也不同，差值只作描述。',
              '内部审核演员使用相同继承模型与共享文件系统，不声称外部盲审独立性。四个 root 样本量很小，同 root 题目、排列与重复采样相关。结果不能证明扩充反馈的因果收益，也不能单独证明泛化或排除模板过拟合。']
    with (out / 'REPORT_ZH.md').open('x') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(ev.json.dumps({'status': summary['status'], 'errors': len(errors), 'report': str(out)}))
    return 0 if not errors else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    raise SystemExit(main(args.output_dir))
