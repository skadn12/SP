"""CPU-only report for the frozen 14-question, three-model ordering control.

Reuse baseline/new raw calls from sealed v127. Only base adds 168 calls. Exact
uses the fixed denominator; missing/invalid/duplicate calls retain bounds for
undefined partial/pair metrics. No selection, retries, inference or TEST reads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import itertools
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
V127 = REPO / 'offline-deliveries/h4_codex_dev_eval_v127'
PLAN = REPO / 'offline-deliveries/h4_codex_training_v126/evidence/DEV_EVALUATION_PLAN_AUDIT.json'
DEV = REPO / 'offline-deliveries/h4_codex_dev_test_ab_v124/delivery/dev/QUESTIONS_WITH_ANSWERS.json'
NATIVE = REPO / 'offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation/scoring.py'
EXPECTED = {
    PLAN: 'a7f940b080a6ad69888da978ecb014765309bdbcc69e16a4f5ede2e1ee597619',
    DEV: 'bd8db8db634798c9c19aaf17f20549c37524787dbc4ff01ba3ce335a73215c56',
    NATIVE: '2d0e7411dcd7b1c520bdbaecfef5259d48f036ce071a98ac3d6e633be2017ad2',
    V127 / 'COMPLETE_MANIFEST.json': '4fb446adcf81d8ce6db72accf78dd3e47b64c2e2548add3191561856e8389d61',
}
MODELS = ['base', 'baseline', 'new']
MODEL_IDS = {'base': 'M0', 'baseline': 'M_clean_policy', 'new': 'M_v125_newonly'}
MODEL_PATHS = {
    'base': '/root/autodl-tmp/Qwen3-4B',
    'baseline': str(REPO / 'restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model'),
    'new': str(REPO / 'restart-n5-runs/h4_codex_combined_sft_v125_merged'),
}
METRICS = {
    'signed_partial': (-1, 1), 'pair_accuracy': (0, 1),
    'A_internal_accuracy': (0, 1), 'B_internal_accuracy': (0, 1),
    'cross_route_accuracy': (0, 1), 'block_violation': (0, 1),
    'both_route_internals_all_correct': (0, 1),
}
GROUPS = {'A_internal': 3, 'B_internal': 3, 'cross_route': 9}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    return sum(values, Fraction()) / len(values) if values else None


def num(value):
    return None if value is None else float(value)


def frac(value):
    return None if value is None else str(Fraction(value))


def metric(rows, key, bounds):
    if not rows:
        return {'mean': None, 'mean_fraction': None, 'lower': None, 'upper': None,
                'lower_fraction': None, 'upper_fraction': None,
                'planned_denominator': 0, 'defined_count': 0}
    known = [Fraction(row[key]) for row in rows if row.get(key) is not None]
    low = [Fraction(row[key]) if row.get(key) is not None else Fraction(bounds[0]) for row in rows]
    high = [Fraction(row[key]) if row.get(key) is not None else Fraction(bounds[1]) for row in rows]
    point = mean(known) if len(known) == len(rows) else None
    return {'mean': num(point), 'mean_fraction': frac(point),
            'lower': num(mean(low)), 'upper': num(mean(high)),
            'lower_fraction': frac(mean(low)), 'upper_fraction': frac(mean(high)),
            'planned_denominator': len(rows), 'defined_count': len(known),
            'undefined_count': len(rows) - len(known)}


def aggregate(rows):
    valid = [r for r in rows if r['valid']]
    n = len(rows)
    result = {'planned_calls': n, 'valid_calls': len(valid),
              'invalid_or_missing_calls': n - len(valid),
              'exact_correct': sum(r['exact_match'] for r in rows),
              'exact_rate_fixed': sum(r['exact_match'] for r in rows) / n if n else None,
              'exact_rate_fraction': str(Fraction(sum(r['exact_match'] for r in rows), n)) if n else None,
              'exact_rate_upper_if_failures_resolved': (sum(r['exact_match'] for r in rows) + n - len(valid)) / n if n else None,
              'metrics': {key: metric(rows, key, bounds) for key, bounds in METRICS.items()},
              'pair_counts': {'planned_pairs': n * 15,
                              'known_correct_pairs': sum(r['correct_pairs'] for r in valid),
                              'known_wrong_pairs': sum(r['wrong_pairs'] for r in valid),
                              'unknown_pairs': (n - len(valid)) * 15},
              'pair_categories': {},
              'diagnostics_valid_calls_only': {
                  'denominator': len(valid),
                  'wrong_pairs_distribution': dict(sorted(Counter(r['wrong_pairs'] for r in valid).items())),
                  'route_patterns': dict(Counter(r['route_prediction'] for r in valid).most_common()),
                  'display_patterns': dict(Counter(','.join(r['display_prediction']) for r in valid).most_common()),
                  'canonical_rank_patterns': dict(Counter(','.join(map(str, r['gold_rank_prediction'])) for r in valid).most_common()),
              }}
    for group, per_call in GROUPS.items():
        key = 'cross_route_accuracy' if group == 'cross_route' else group + '_accuracy'
        result['pair_categories'][group] = {
            'planned_pairs': n * per_call,
            'known_correct_pairs': sum(r['pair_category_correct'][group] for r in valid),
            'unknown_pairs': (n - len(valid)) * per_call,
            'accuracy_fixed': result['metrics'][key],
        }
    return result


def delta(a, b):
    """Paired b minus a; exact retains the explicit zero-for-failure policy."""
    result = {'direction': 'second_minus_first',
              'exact_rate_delta': None if not a['planned_calls'] else num(Fraction(b['exact_rate_fraction']) - Fraction(a['exact_rate_fraction'])),
              'metrics': {}}
    for key in METRICS:
        x, y = a['metrics'][key], b['metrics'][key]
        complete = x['mean_fraction'] is not None and y['mean_fraction'] is not None
        result['metrics'][key] = {
            'delta': num(Fraction(y['mean_fraction']) - Fraction(x['mean_fraction'])) if complete else None,
            'lower': num(Fraction(y['lower_fraction']) - Fraction(x['upper_fraction'])) if a['planned_calls'] else None,
            'upper': num(Fraction(y['upper_fraction']) - Fraction(x['lower_fraction'])) if a['planned_calls'] else None,
        }
    return result


def independent_order(raw, request, gold, routes, scenes):
    """Recover content and score all 15 relations without native parsing/scoring."""
    match = re.fullmatch(r'\s*<box>\s*(.*?)\s*</box>\s*', raw, re.S) if isinstance(raw, str) else None
    if match is None:
        raise ValueError('INDEPENDENT_BOX_PARSE_FAILURE')
    display = [part.strip().upper() for part in match.group(1).strip().split(',')]
    if any(re.fullmatch(r'[A-Z][A-Z0-9_]*', x) is None for x in display) or len(set(display)) != 6 or len(display) != 6:
        raise ValueError('INDEPENDENT_SIX_DISTINCT_LABELS_REQUIRED')
    mapping = request['variant']['display_to_content']
    if not all(label in mapping for label in display):
        raise ValueError('INDEPENDENT_UNKNOWN_LABEL')
    prediction = [mapping[label] for label in display]
    if len(set(prediction)) != 6 or set(prediction) != set(gold):
        raise ValueError('INDEPENDENT_INCOMPLETE_CONTENT_ORDER')
    position = {value: index for index, value in enumerate(prediction)}
    group_correct = dict.fromkeys(GROUPS, 0)
    pair_details = []
    for a, b in itertools.combinations(gold, 2):
        category = routes[a] + '_internal' if routes[a] == routes[b] else 'cross_route'
        correct = position[a] < position[b]
        group_correct[category] += int(correct)
        pair_details.append({'gold_content_pair': [a, b], 'category': category, 'correct': correct})
    correct = sum(group_correct.values())
    return {'display_prediction': display, 'content_prediction': prediction,
            'gold_rank_prediction': [gold.index(value) + 1 for value in prediction],
            'scene_prediction': [scenes[value] for value in prediction],
            'route_prediction': ''.join(routes[value] for value in prediction),
            'correct_pairs': correct, 'wrong_pairs': 15 - correct,
            'exact_match': int(prediction == gold),
            'signed_partial': str(Fraction(2 * correct - 15, 15)),
            'pair_accuracy': str(Fraction(correct, 15)),
            'A_internal_accuracy': str(Fraction(group_correct['A_internal'], 3)),
            'B_internal_accuracy': str(Fraction(group_correct['B_internal'], 3)),
            'cross_route_accuracy': str(Fraction(group_correct['cross_route'], 9)),
            'block_violation': int(group_correct['cross_route'] != 9),
            'both_route_internals_all_correct': int(group_correct['A_internal'] == group_correct['B_internal'] == 3),
            'pair_category_correct': group_correct, 'pair_details': pair_details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope-dir', type=Path, default=HERE)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    scope = args.scope_dir.resolve()
    out = (args.output_dir or scope).resolve()
    if out.is_relative_to(scope / 'run') or out.is_relative_to(V127) or out.is_relative_to(DEV.parents[2]):
        raise RuntimeError('OUTPUT_MUST_NOT_OVERWRITE_FROZEN_INPUTS')
    names = ['SUMMARY.json', 'REPORT_ZH.md', 'PER_QUESTION.json', 'PER_QUESTION.csv', 'PER_CALL.json', 'PER_ROOT.json']
    if any((out / name).exists() for name in names):
        raise RuntimeError('REPORT_EXISTS_NO_SILENT_OVERWRITE')
    bindings, errors = {}, []
    def problem(code, **details): errors.append({'code': code, **details})
    def require(ok, code, **details):
        if not ok: problem(code, **details)
    for path, digest in EXPECTED.items():
        if sha(path) != digest: raise RuntimeError('FROZEN_SOURCE_CHANGED:' + str(path))
    seal = json.loads((V127 / 'COMPLETE_MANIFEST.json').read_text())
    sealed = {str((V127 / row['path']).resolve()): row for row in seal['files']}
    def bind(path):
        path = Path(path).resolve(); actual = sha(path)
        if path.is_relative_to(V127) and path != V127 / 'COMPLETE_MANIFEST.json':
            row = sealed.get(str(path))
            require(row is not None and row['sha256'] == actual and row['bytes'] == path.stat().st_size,
                    'V127_SEALED_FILE_CHANGED', path=str(path))
        bindings[str(path)] = actual
        return path
    def read(path): return json.loads(bind(path).read_text())
    bind(__file__); bind(V127 / 'COMPLETE_MANIFEST.json'); bind(NATIVE)
    plan = read(PLAN)
    canonical = {q['candidate_id']: q for q in read(DEV)['questions'] if q['question_type'] == 'ordering'}
    requests = [r for r in plan['readonly_checks']['plan_requests'] if r['question_type'] == 'ordering']
    if len(canonical) != 14 or len(requests) != 168 or len({r['request_id'] for r in requests}) != 168:
        raise RuntimeError('FROZEN_ORDERING_POPULATION_NOT14_AND168')
    frozen_scope = read(V127 / 'SCOPE.json')
    fixed_roots = frozen_scope['fixed_dev_root_ids']
    covered_roots = sorted({q['root_id'] for q in canonical.values()})
    require(len(set(fixed_roots)) == 15 and len(covered_roots) == 7, 'ROOT_POPULATION_CHANGED')
    if (scope / 'SCOPE.json').exists():
        current_scope = read(scope / 'SCOPE.json')
        if 'fixed_dev_root_ids' in current_scope:
            require(sorted(current_scope['fixed_dev_root_ids']) == sorted(fixed_roots), 'V130_FIXED_DEV_ROOTS_CHANGED')
    spec = importlib.util.spec_from_file_location('v130_frozen_native_scoring', NATIVE)
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    manifests, golds, questions, callsets, native_results = {}, {}, {}, {}, {}
    runs = {'base': scope / 'run/base', 'baseline': V127 / 'run/baseline', 'new': V127 / 'run/new'}
    baseline_questions = read(runs['baseline'] / 'questions.json')['candidates']
    expected_questions = [q for q in baseline_questions if q['question_type'] == 'ordering']
    baseline_gold = read(runs['baseline'] / 'gold.json')
    expected_gold = {cid: baseline_gold[cid] for cid in canonical}
    route_map, scene_map = {}, {}
    for cid, q in canonical.items():
        content_by_label = {label: 'content_' + hashlib.sha256(text.strip().encode()).hexdigest()[:16] for label, text in q['items'].items()}
        require([content_by_label[label] for label in q['gold']] == expected_gold[cid], 'CANONICAL_GOLD_CHANGED', candidate_id=cid)
        route_map[cid], scene_map[cid] = {}, {}
        for label, content in content_by_label.items():
            anchor = q['order_anchor_spans'][label]['turn_ref']
            match = re.fullmatch(r'R02:([AB]):S(0[123]):T\d+', anchor)
            if not match: raise RuntimeError('ORDER_ANCHOR_ROUTE_CONTRACT_CHANGED:' + cid)
            route_map[cid][content] = match.group(1); scene_map[cid][content] = anchor.rsplit(':T', 1)[0]
        require([route_map[cid][c] for c in expected_gold[cid]] == list('AAABBB'), 'GOLD_ROUTE_BLOCK_NOT_AAABBB', candidate_id=cid)
        planned = [r for r in requests if r['candidate_id'] == cid]
        require(len(planned) == 12 and {(r['variant']['variant_id'], r['slot']) for r in planned} == {(f'perm_{i:02d}', 0) for i in range(12)},
                'FROZEN_VARIANT_SLOT_SET_CHANGED', candidate_id=cid)
    for model in MODELS:
        run = runs[model]
        manifest = manifests[model] = read(run / 'manifest.json')
        all_questions = read(run / 'questions.json')['candidates']
        questions[model] = [q for q in all_questions if q['question_type'] == 'ordering']
        all_gold = read(run / 'gold.json'); golds[model] = {cid: all_gold.get(cid) for cid in canonical}
        results = native_results[model] = read(run / 'RESULTS.json') if (run / 'RESULTS.json').exists() else None
        declared = [r for r in manifest['requests'] if r['question_type'] == 'ordering']
        require(declared == requests, 'ORDERING_REQUESTS_DIFFER_FROM_FROZEN_PLAN', model=model)
        require(questions[model] == expected_questions and golds[model] == expected_gold, 'QUESTIONS_OR_GOLD_CHANGED', model=model)
        if model == 'base':
            require(len(manifest['requests']) == manifest['request_count'] == 168 and manifest['candidate_count'] == 14,
                    'BASE_FULL_POPULATION_CHANGED')
            require(len(all_questions) == 14 and set(all_gold) == set(canonical), 'BASE_QUESTION_OR_GOLD_POPULATION_CHANGED')
        else:
            require(manifest['requests'] == plan['readonly_checks']['plan_requests'] and manifest['request_count'] == 780 and manifest['candidate_count'] == 48,
                    'V127_FULL_PLAN_CHANGED', model=model)
        require(manifest['solver_model'] == MODEL_IDS[model] and manifest['model'] == MODEL_PATHS[model], 'MODEL_REGISTRATION_CHANGED', model=model)
        identity = manifest.get('solver_identity', {})
        require(identity.get('model_identifier') == MODEL_IDS[model] and identity.get('model_path') == MODEL_PATHS[model], 'MODEL_IDENTITY_CHANGED', model=model)
        require(manifest['tokenizer'] == MODEL_PATHS['base'], 'TOKENIZER_CHANGED', model=model)
        require(manifest['policy_version'] == plan['frozen_protocol']['policy_version'], 'POLICY_CHANGED', model=model)
        require(manifest['decoder_contract'] == plan['frozen_protocol']['decoder_contract_precedent'], 'DECODER_CHANGED', model=model)
        for name, digest in manifest['input_sha256'].items():
            path = Path(manifest['pool_path']) if name == 'pool' else run / name
            require(path.is_file() and sha(path) == digest, 'NATIVE_INPUT_CHANGED', model=model, path=str(path))
            if path.is_file(): bind(path)
        for name, digest in manifest['code_sha256'].items():
            path = Path(name)
            require(path.is_file() and sha(path) == digest, 'NATIVE_CODE_CHANGED', model=model, path=str(path))
            if path.is_file(): bind(path)
        for req in requests:
            path = run / 'prompts' / (req['request_id'] + '.txt')
            require(path.is_file() and sha(path) == req['prompt_sha256'], 'PROMPT_CHANGED', model=model, request_id=req['request_id'])
            if path.is_file(): bind(path)
        observed = defaultdict(list)
        planned_ids = {r['request_id'] for r in requests}
        all_manifest_ids = {r['request_id'] for r in manifest['requests']}
        for path in sorted((run / 'responses').glob('*.json')):
            # Old 612 MS responses are neither read nor evaluated here.
            if model != 'base' and path.stem not in planned_ids:
                require(path.stem in all_manifest_ids, 'V127_UNPLANNED_RESPONSE_FILENAME', model=model, path=str(path)); continue
            try: response = read(path)
            except (OSError, ValueError) as exc:
                problem('UNREADABLE_RESPONSE', model=model, path=str(path), exception=type(exc).__name__); continue
            observed[response.get('request_id')].append((path, response))
        for rid in set(observed) - planned_ids: problem('UNPLANNED_ORDERING_RESPONSE', model=model, request_id=rid)
        calls = []
        for req in requests:
            cid, rid = req['candidate_id'], req['request_id']
            call = {'model': model, 'request_id': rid, 'candidate_id': cid, 'root_id': req['root_id'],
                    'variant_id': req['variant']['variant_id'], 'slot': req['slot'], 'seed': req['seed'],
                    'prompt_sha256': req['prompt_sha256'], 'valid': False, 'exact_match': 0,
                    **dict.fromkeys(METRICS), 'errors': []}
            found = observed.get(rid, [])
            if len(found) != 1:
                call['errors'].append('MISSING_RESPONSE' if not found else 'DUPLICATE_RESPONSE')
            else:
                path, response = found[0]; call['response_path'] = str(path)
                fields = {'request_id': rid, 'candidate_id': cid, 'root_id': req['root_id'], 'question_type': 'ordering',
                          'variant_id': req['variant']['variant_id'], 'slot': 0, 'seed': req['seed'], 'mode': 'full_context',
                          'solver_model': manifest['solver_model'], 'model_path': manifest['model']}
                for key, expected in fields.items():
                    if response.get(key) != expected or key in {'slot', 'seed'} and type(response.get(key)) is not int:
                        call['errors'].append('RESPONSE_PLAN_MISMATCH:' + key)
                if path.stem != rid: call['errors'].append('RESPONSE_FILENAME_MISMATCH')
                metadata = response.get('metadata') or {}
                for key in ['temperature', 'top_p', 'max_tokens', 'structured_output_api', 'structured_output_version']:
                    if metadata.get(key) != manifest['decoder_contract'].get(key): call['errors'].append('DECODER_METADATA_MISMATCH:' + key)
                if metadata.get('sampling_seed') != req['seed']: call['errors'].append('SAMPLING_SEED_MISMATCH')
                if metadata.get('finish_reason') not in {'stop', 'eos'}: call['errors'].append('BAD_FINISH_REASON')
                if response.get('evaluation_valid') is not True or response.get('data_errors'): call['errors'].append('NATIVE_EVALUATION_INVALID')
                score = native.score_response(dict(canonical[cid], gold_content_ids=expected_gold[cid]), req, response.get('raw_output'))
                if score != response.get('score'): call['errors'].append('NATIVE_SCORE_DIFFERS_FROM_RAW_RECOMPUTATION')
                try:
                    recovered = independent_order(response.get('raw_output'), req, expected_gold[cid], route_map[cid], scene_map[cid])
                    if not score['scored'] or any(score.get(k) != recovered[v] for k, v in [('content_prediction', 'content_prediction'), ('display_prediction', 'display_prediction'), ('tp', 'correct_pairs'), ('fp', 'wrong_pairs'), ('partial_score_fraction', 'signed_partial')]) or bool(score.get('exact_match')) != bool(recovered['exact_match']):
                        call['errors'].append('INDEPENDENT_SCORE_DIFFERS_FROM_NATIVE')
                    if not call['errors']: call.update(recovered, valid=True, raw_output=response['raw_output'])
                except ValueError as exc: call['errors'].append(str(exc))
            if call['errors']: problem('CALL_INVALID', model=model, request_id=rid, reasons=call['errors'])
            calls.append(call)
        callsets[model] = calls
        if results is None: problem('NATIVE_RESULTS_MISSING', model=model)
        else:
            require(results.get('solver_model') == MODEL_IDS[model] and results.get('model_path') == MODEL_PATHS[model], 'NATIVE_RESULTS_MODEL_CHANGED', model=model)
            require(not results.get('call_set_errors') and results.get('ground_calls') == 0, 'NATIVE_RESULTS_CALL_SET_ERRORS', model=model)
            order_results = [q for q in results['questions'] if q['question_type'] == 'ordering']
            require(len(order_results) == 14 and {q['candidate_id'] for q in order_results} == set(canonical), 'NATIVE_RESULTS_POPULATION_CHANGED', model=model)
            by_id = {q['candidate_id']: q for q in order_results}
            for cid in canonical:
                subset = [c for c in calls if c['candidate_id'] == cid]; nr = by_id.get(cid, {})
                require(nr.get('expected_full_calls') == 12 and nr.get('actual_full_calls') == sum(len(observed.get(c['request_id'], [])) for c in subset), 'NATIVE_RESULTS_CALL_COUNTS', model=model, candidate_id=cid)
                if all(c['valid'] for c in subset):
                    require(nr.get('full_exact_rate') == frac(mean([Fraction(c['exact_match']) for c in subset])) and nr.get('full_partial_mean') == frac(mean([Fraction(c['signed_partial']) for c in subset])) and not nr.get('data_errors'),
                            'NATIVE_RESULTS_SCORE_SUMMARY', model=model, candidate_id=cid)
    comparisons = [('base', 'baseline'), ('base', 'new'), ('baseline', 'new')]
    per_question = []
    for cid, q in canonical.items():
        row = {'candidate_id': cid, 'root_id': q['root_id'], 'question_type': 'ordering',
               'models': {m: aggregate([c for c in callsets[m] if c['candidate_id'] == cid]) for m in MODELS}}
        row['paired_deltas'] = {a + '_to_' + b: delta(row['models'][a], row['models'][b]) for a, b in comparisons}
        per_question.append(row)
    per_call = []
    by_model_id = {m: {c['request_id']: c for c in callsets[m]} for m in MODELS}
    for req in requests:
        rid = req['request_id']; models = {m: by_model_id[m][rid] for m in MODELS}
        per_call.append({'request_id': rid, 'candidate_id': req['candidate_id'], 'root_id': req['root_id'], 'models': models,
                         'paired_deltas': {a + '_to_' + b: {'exact_delta': models[b]['exact_match'] - models[a]['exact_match'],
                                                           **{key: num(Fraction(models[b][key]) - Fraction(models[a][key])) if models[a]['valid'] and models[b]['valid'] else None for key in METRICS}}
                                           for a, b in comparisons}})
    per_root = []
    for root in sorted(fixed_roots):
        subset = [q for q in per_question if q['root_id'] == root]
        models = {m: aggregate([c for c in callsets[m] if c['root_id'] == root]) for m in MODELS}
        per_root.append({'root_id': root, 'has_questions': bool(subset), 'question_count': len(subset), 'models': models,
                         'paired_deltas': {a + '_to_' + b: delta(models[a], models[b]) for a, b in comparisons}})
    totals = {m: aggregate(callsets[m]) for m in MODELS}
    root_macro = {}
    for m in MODELS:
        covered = [r['models'][m] for r in per_root if r['has_questions']]
        root_macro[m] = {'covered_root_denominator': len(covered), 'fixed_dev_root_population': len(fixed_roots),
                         'exact_rate_equal_covered_root': num(mean([Fraction(r['exact_rate_fraction']) for r in covered])),
                         'metrics': {key: {
                             'mean': num(mean([Fraction(r['metrics'][key]['mean_fraction']) for r in covered])) if all(r['metrics'][key]['mean_fraction'] is not None for r in covered) else None,
                             'lower': num(mean([Fraction(r['metrics'][key]['lower_fraction']) for r in covered])),
                             'upper': num(mean([Fraction(r['metrics'][key]['upper_fraction']) for r in covered]))} for key in METRICS}}
    paired = {}
    for a, b in comparisons:
        complete = [r for r in per_call if r['models'][a]['valid'] and r['models'][b]['valid']]
        paired[a + '_to_' + b] = {**delta(totals[a], totals[b]),
            'paired_call_diagnostics': {'planned_pairs': 168, 'both_valid': len(complete), 'at_least_one_invalid': 168 - len(complete),
                'second_only_exact': sum(r['models'][b]['exact_match'] == 1 and r['models'][a]['exact_match'] == 0 for r in complete),
                'first_only_exact': sum(r['models'][a]['exact_match'] == 1 and r['models'][b]['exact_match'] == 0 for r in complete),
                'both_exact': sum(r['models'][a]['exact_match'] == r['models'][b]['exact_match'] == 1 for r in complete),
                'neither_exact': sum(r['models'][a]['exact_match'] == r['models'][b]['exact_match'] == 0 for r in complete)}}
    for path, digest in bindings.items(): require(sha(path) == digest, 'INPUT_CHANGED_DURING_REPORT', path=path)
    if errors:
        interpretation = '本轮存在缺失或无效输入，不能作完整三模型能力比较；固定分母与未定义指标上下界已保留。'
    elif totals['base']['exact_rate_fixed'] > totals['baseline']['exact_rate_fixed']:
        interpretation = '原始 base 在这批冻结题上的整序正确率高于第一轮 M_clean，说明本任务存在第一轮训练之后的表现下降；这不能单独归因于旧监督噪声，也不能量化噪声、四项到六项变化及其他训练因素各自的作用。'
    elif totals['base']['exact_rate_fixed'] == totals['baseline']['exact_rate_fixed'] and totals['base']['metrics']['pair_accuracy']['mean'] > totals['baseline']['metrics']['pair_accuracy']['mean']:
        interpretation = '原始 base 与 M_clean 的整序正确率相同，但 base 的成对关系更准，说明局部顺序判断在第一轮后下降；不能把差异单独归因于旧监督噪声。'
    else:
        interpretation = '原始 base 的整序正确率未高于第一轮 M_clean；本对照不支持“第一轮把原本能答对的整序能力教坏了”的结论。仍需区分题型适应、监督质量和任务结构各自的作用。'
    summary = {'schema_version': 'v130_frozen_base_order_control_v1',
        'status': 'COMPLETE' if not errors else 'INCOMPLETE_OR_INVALID', 'errors': errors,
        'population': {'questions': 14, 'covered_dev_roots': 7, 'fixed_dev_roots': 15, 'uncovered_roots_not_scored': 8,
                       'permutations_per_question': 12, 'rollouts_per_permutation': 1, 'planned_calls_per_model': 168,
                       'new_base_inference_calls': 168, 'reused_v127_ordering_calls': 336, 'total_comparison_calls': 504},
        'models': {m: {'solver_model': manifests[m]['solver_model'], 'model_path': manifests[m]['model'],
                       'manifest_path': str(runs[m] / 'manifest.json'), 'model_fingerprint_sha256': manifests[m]['solver_identity'].get('model_fingerprint_sha256'),
                       'raw_calls_reused_from_v127': m != 'base', 'summary': totals[m],
                       'native_selected_count_diagnostic_only': sum(cid in canonical for cid in native_results[m]['selected_questions']) if native_results[m] else None}
                   for m in MODELS},
        'paired_deltas': paired, 'root_macro': root_macro, 'interpretation': interpretation,
        'weighting': 'All14 frozen questions receive12 calls, so equal-question and pooled-call means coincide. Covered-root macro is separate. No selected_questions filtering.',
        'failure_policy': 'All168 planned calls retained per model. Invalid/missing/duplicate exact=0. Partial and pair metrics keep fixed-denominator worst/best bounds, undefined point estimates and no valid-only headline. No retry/best-of.',
        'limits': ['14 questions from7 covered DEV roots, with12 correlated display permutations each.504 calls are not504 independent questions; no significance or broad generalization claim.',
                   'Eight of15 fixed DEV roots have no questions and no measured accuracy. No TEST question/source content read or TEST inference.',
                   'A_internal/B_internal/cross_route use3/3/9 relations per call. Block violation means any B item preceding any A item. Audited private anchors are used only after inference for diagnostics.',
                   'Structured finite-choice decoding guarantees six unique output labels. Parse validity is not evidence of unconstrained formatting skill.',
                   'New-model100% on this highly templated task remains a ceiling observation, not a claim of general complex ordering ability.',
                   'This same-task model comparison is observational; it cannot isolate a causal contribution from noisy old labels, four-item-to-six-item distribution change, or later clean training.'],
        'test_evaluation': False, 'new_generation_calls': 0,
        'input_bindings': [{'path': p, 'sha256': digest} for p, digest in sorted(bindings.items())]}
    out.mkdir(parents=True, exist_ok=True)
    def write(name, value): (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    write('SUMMARY.json', summary); write('PER_QUESTION.json', per_question); write('PER_CALL.json', per_call); write('PER_ROOT.json', per_root)
    flattened = []
    for row in per_question:
        flat = {'candidate_id': row['candidate_id'], 'root_id': row['root_id']}
        for m in MODELS:
            s = row['models'][m]
            flat.update({m + '_planned_calls': s['planned_calls'], m + '_valid_calls': s['valid_calls'], m + '_exact': s['exact_rate_fixed']})
            for key in METRICS:
                flat[m + '_' + key] = s['metrics'][key]['mean']
                flat[m + '_' + key + '_lower'] = s['metrics'][key]['lower']
                flat[m + '_' + key + '_upper'] = s['metrics'][key]['upper']
        for pair, value in row['paired_deltas'].items():
            flat[pair + '_exact_delta'] = value['exact_rate_delta']
            flat[pair + '_pair_accuracy_delta'] = value['metrics']['pair_accuracy']['delta']
        flattened.append(flat)
    buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=list(flattened[0])); writer.writeheader(); writer.writerows(flattened)
    (out / 'PER_QUESTION.csv').write_text(buffer.getvalue())
    def pct(value): return '未定义' if value is None else f'{100 * value:.2f}%'
    def val(value): return '未定义' if value is None else f'{value:.4f}'
    lines = [f"冻结 DEV 原始 base 排序对照状态：{summary['status']}。", '',
             '同一14道六项排序题，来自7个有题 DEV root；固定 DEV 总体15个 root 中另8个无题。本轮只新增原始 Qwen3-4B（M0）的168次调用，M_clean 与本轮训练模型各168次返回复用v127。未重新运行旧模型，未按答案、分数或 native selected_questions 筛题。', '',
             '| 模型 | 整序全对 | 成对顺序准确率 | signed partial | A 内部 | B 内部 | A/B 交叉 | A/B block 违例 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    labels = {'base': '原始 Qwen3-4B / M0', 'baseline': '第一轮 M_clean', 'new': '本轮 M_v125_newonly'}
    for m in MODELS:
        s = totals[m]; z = s['metrics']
        lines.append(f"| {labels[m]} | {s['exact_correct']}/168 = {pct(s['exact_rate_fixed'])} | {pct(z['pair_accuracy']['mean'])} | {val(z['signed_partial']['mean'])} | {pct(z['A_internal_accuracy']['mean'])} | {pct(z['B_internal_accuracy']['mean'])} | {pct(z['cross_route_accuracy']['mean'])} | {pct(z['block_violation']['mean'])} |")
    lines += ['', '每题固定12次排列，所以此表每题等权与调用等权相同。成对关系每调用15对；A内部3对、B内部3对、A/B交叉9对。block违例指至少一个B项出现在A项之前。', '', interpretation, '',
              '配对差异（后者减前者，百分点）：', '', '| 对照 | 整序全对变化 | 成对准确率变化 |', '|---|---:|---:|']
    for a, b in comparisons:
        d = paired[a + '_to_' + b]
        lines.append(f"| {labels[a]} → {labels[b]} | {val(None if d['exact_rate_delta'] is None else d['exact_rate_delta'] * 100)} | {val(None if d['metrics']['pair_accuracy']['delta'] is None else d['metrics']['pair_accuracy']['delta'] * 100)} |")
    lines += ['', '所有计划调用均计入固定分母。无效、缺失或重复返回的整序分记0；成对与partial点估计未定义，并给出固定分母上下界。原始响应由native评分器重算，同时独立解析展示标签、恢复内容并重算全部15对关系，两套分数须一致。', '',
              '14题、同root题目与排列相互相关，不能把504次调用当作504道独立题，也不据此作显著性结论。100%只说明此固定模板任务达到天花板；跨模板、同场景多事件及跨轮复杂排序仍待独立诊断。有限集合结构化解码已强制六项完整。', '',
              '结果细节：SUMMARY.json、PER_QUESTION.csv / .json、PER_CALL.json、PER_ROOT.json。逐root文件包含全部15个固定DEV root；无题root不计算准确率。未读取TEST题文、未开展TEST推理、未新增训练或数据生成。', '',
              f"输入与评分完整性错误数：{len(errors)}。"]
    (out / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': summary['status'], 'errors': len(errors), 'output_dir': str(out),
                      'exact': {m: totals[m]['exact_rate_fixed'] for m in MODELS},
                      'pair_accuracy': {m: totals[m]['metrics']['pair_accuracy']['mean'] for m in MODELS}}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
