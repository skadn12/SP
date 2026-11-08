"""Report the frozen 48-question dev pair without inference or data selection.

Inputs: run/{baseline,new}/{manifest,questions,gold,RESULTS}.json and responses/.
Every planned call remains in its denominator. Invalid calls receive exact=0;
other undefined metrics retain explicit worst/best bounds and no invented point
estimate. This tool imports only the frozen native, CPU-only scoring module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PLAN = REPO / 'offline-deliveries/h4_codex_training_v126/evidence/DEV_EVALUATION_PLAN_AUDIT.json'
DEV = REPO / 'offline-deliveries/h4_codex_dev_test_ab_v124/delivery/dev/QUESTIONS_WITH_ANSWERS.json'
NATIVE = REPO / 'offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation/scoring.py'
NATIVE_SHA = '2d0e7411dcd7b1c520bdbaecfef5259d48f036ce071a98ac3d6e633be2017ad2'
STRATA = ['all48', 'main41_excluding_gold6', 'multi_select_all34', 'multi_select_gold2345_27',
          'multi_select_gold6_control7', 'gold2', 'gold3', 'gold4', 'gold5', 'gold6', 'ordering14']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    return None if value is None else float(value)


def fraction_text(value):
    return None if value is None else str(Fraction(value))


def mean(values):
    return sum(values, Fraction()) / len(values) if values else None


def strata(question, gold):
    result = ['all48']
    if question['question_type'] == 'ordering':
        return result + ['main41_excluding_gold6', 'ordering14']
    g = len(gold)
    result += ['multi_select_all34', 'gold' + str(g)]
    return result + (['multi_select_gold6_control7'] if g == 6 else ['main41_excluding_gold6', 'multi_select_gold2345_27'])


def metric(rows, key, bounds):
    if not rows:
        return None
    known, low, high = [], [], []
    for row in rows:
        value = row.get(key)
        a, b = bounds(row)
        if value is None:
            low.append(Fraction(a)); high.append(Fraction(b))
        else:
            value = Fraction(value)
            known.append(value); low.append(value); high.append(value)
    point = mean(known) if len(known) == len(rows) else None
    return {'mean': number(point), 'mean_fraction': fraction_text(point),
            'lower': number(mean(low)), 'upper': number(mean(high)),
            'lower_fraction': fraction_text(mean(low)), 'upper_fraction': fraction_text(mean(high)),
            'planned_denominator': len(rows), 'defined_count': len(known), 'undefined_count': len(rows) - len(known)}


def aggregate_calls(rows):
    if not rows:
        return {'planned_calls': 0, 'valid_calls': 0, 'invalid_or_missing_calls': 0, 'exact_rate_fixed': None}
    valid = sum(row['valid'] for row in rows)
    correct = sum(row['exact_match'] for row in rows)
    result = {'planned_calls': len(rows), 'valid_calls': valid, 'invalid_or_missing_calls': len(rows) - valid,
              'valid_rate': valid / len(rows), 'exact_correct': correct,
              'parse_success_calls': sum(row.get('parse_status') == 'success' for row in rows),
              'parse_success_rate_fixed': sum(row.get('parse_status') == 'success' for row in rows) / len(rows),
              'exact_rate_fixed': correct / len(rows), 'exact_rate_fraction': str(Fraction(correct, len(rows))),
              'exact_rate_upper_if_failures_resolved': (correct + len(rows) - valid) / len(rows),
              'signed_partial': metric(rows, 'signed_partial', lambda row: (-1, 1)),
              'failure_policy': 'Invalid/missing/duplicate calls count as exact=0. Signed partial is undefined there; bounds [-1,1], no valid-only replacement.'}
    ms = [row for row in rows if row['question_type'] == 'multi_select']
    order = [row for row in rows if row['question_type'] == 'ordering']
    if ms:
        result['multi_select'] = {
            'planned_calls': len(ms), 'tp': metric(ms, 'tp', lambda row: (0, row['gold_count'])),
            'fp': metric(ms, 'fp', lambda row: (0, 6 - row['gold_count'])),
            'fn': metric(ms, 'fn', lambda row: (0, row['gold_count'])),
            'predicted_count': metric(ms, 'predicted_count', lambda row: (0, 6)),
            'predicted_minus_gold_count': metric(ms, 'predicted_minus_gold_count', lambda row: (-row['gold_count'], 6 - row['gold_count'])),
            'all6_rate': metric(ms, 'all6', lambda row: (0, 1)),
            'overselection_rate': metric(ms, 'overselection', lambda row: (0, 1)),
            'underselection_rate': metric(ms, 'underselection', lambda row: (0, 1)),
            'known_tp_total': sum(row['tp'] or 0 for row in ms),
            'known_fp_total': sum(row['fp'] or 0 for row in ms),
            'known_fn_total': sum(row['fn'] or 0 for row in ms)}
    if order:
        result['ordering'] = {'planned_calls': len(order),
                              'pair_accuracy': metric(order, 'pair_accuracy', lambda row: (0, 1)),
                              'correct_pairs': metric(order, 'correct_pairs', lambda row: (0, 15)),
                              'wrong_pairs': metric(order, 'wrong_pairs', lambda row: (0, 15))}
    return result


def aggregate_candidates(candidates):
    if not candidates:
        return {'candidate_count': 0, 'exact_rate_equal_candidate': None, 'signed_partial_equal_candidate': None}
    exact = mean([Fraction(row['exact_rate_fraction']) for row in candidates])
    lows = [Fraction(row['signed_partial']['lower_fraction']) for row in candidates]
    highs = [Fraction(row['signed_partial']['upper_fraction']) for row in candidates]
    complete = all(row['signed_partial']['mean_fraction'] is not None for row in candidates)
    partial = mean([Fraction(row['signed_partial']['mean_fraction']) for row in candidates]) if complete else None
    return {'candidate_count': len(candidates), 'planned_calls': sum(row['planned_calls'] for row in candidates),
            'valid_calls': sum(row['valid_calls'] for row in candidates),
            'exact_rate_equal_candidate': number(exact), 'exact_rate_equal_candidate_fraction': str(exact),
            'signed_partial_equal_candidate': number(partial),
            'signed_partial_equal_candidate_fraction': fraction_text(partial),
            'signed_partial_equal_candidate_lower': number(mean(lows)),
            'signed_partial_equal_candidate_upper': number(mean(highs)),
            'weighting': 'Each frozen candidate receives equal weight, regardless of its 18 or 12 planned rollouts.'}


def paired_delta(baseline, new):
    a, b = baseline['signed_partial'], new['signed_partial']
    point = (Fraction(b['mean_fraction']) - Fraction(a['mean_fraction'])) if a['mean_fraction'] is not None and b['mean_fraction'] is not None else None
    exact = Fraction(new['exact_rate_fraction']) - Fraction(baseline['exact_rate_fraction'])
    lower = Fraction(b['lower_fraction']) - Fraction(a['upper_fraction'])
    upper = Fraction(b['upper_fraction']) - Fraction(a['lower_fraction'])
    return {'exact_rate_delta_new_minus_baseline': number(exact), 'exact_rate_delta_fraction': str(exact),
            'signed_partial_delta': number(point), 'signed_partial_delta_fraction': fraction_text(point),
            'signed_partial_delta_lower': number(lower), 'signed_partial_delta_lower_fraction': str(lower),
            'signed_partial_delta_upper': number(upper), 'signed_partial_delta_upper_fraction': str(upper)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope-dir', type=Path, default=HERE)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    scope = args.scope_dir.resolve()
    out = (args.output_dir or scope).resolve()
    if out.is_relative_to(scope / 'run') or out.is_relative_to(REPO / 'offline-deliveries/h4_codex_dev_test_ab_v124'):
        raise RuntimeError('REPORT_OUTPUT_MUST_NOT_OVERWRITE_FROZEN_INPUTS')
    outputs = ['PAIRED_SUMMARY.json', 'REPORT_ZH.md', 'PER_QUESTION.json', 'PER_QUESTION.csv', 'PER_CALL.json', 'PER_ROOT.json']
    if any((out / name).exists() for name in outputs):
        raise RuntimeError('REPORT_EXISTS_NO_SILENT_OVERWRITE: use a separate output directory')
    bindings, errors = {}, []
    def read(path):
        path = Path(path).resolve(); bindings[str(path)] = sha(path)
        return json.loads(path.read_text())
    def problem(code, **details):
        errors.append({'code': code, **details})
    def require(condition, code, **details):
        if not condition:
            problem(code, **details)
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    plan = read(PLAN)
    canonical = read(DEV)['questions']
    canonical_by_id = {q['candidate_id']: q for q in canonical}
    if len(canonical) != 48 or len(canonical_by_id) != 48:
        raise RuntimeError('CANONICAL_DEV_POPULATION_NOT48')
    require(sha(DEV) == plan['schema_and_adaptation']['canonical_dev_input']['sha256'], 'CANONICAL_DEV_BINDING_CHANGED')
    if sha(NATIVE) != NATIVE_SHA:
        raise RuntimeError('NATIVE_SCORING_CHANGED')
    bindings[str(NATIVE)] = NATIVE_SHA
    spec = importlib.util.spec_from_file_location('v127_frozen_native_scoring', NATIVE)
    scoring = importlib.util.module_from_spec(spec); spec.loader.exec_module(scoring)
    scope_info = read(scope / 'SCOPE.json') if (scope / 'SCOPE.json').exists() else {}
    fixed_roots = scope_info.get('fixed_dev_root_ids') or scope_info.get('dev_root_ids')
    if fixed_roots is None:
        # MANIFEST contains root metadata only. No test question/source file is opened.
        root_manifest = read(DEV.parents[2] / 'MANIFEST.json')
        fixed_roots = [row['root_id'] for row in root_manifest['roots'] if row['split'] == 'dev']
    fixed_roots = sorted(fixed_roots)
    if len(fixed_roots) != 15 or len(set(fixed_roots)) != 15:
        raise RuntimeError('FIXED_DEV_ROOT_POPULATION_NOT15')

    manifests, questions, golds, resultfiles, records, callsets = {}, {}, {}, {}, {}, {}
    for model in ['baseline', 'new']:
        run = scope / 'run' / model
        manifest = manifests[model] = read(run / 'manifest.json')
        questions[model] = read(run / 'questions.json')['candidates']
        golds[model] = read(run / 'gold.json')
        resultfiles[model] = read(run / 'RESULTS.json') if (run / 'RESULTS.json').exists() else None
        if resultfiles[model] is None:
            problem('NATIVE_RESULTS_MISSING', model=model)
        declared_requests = manifest['requests']
        require(len(declared_requests) == manifest['request_count'] == 780 and manifest['candidate_count'] == 48, 'REQUEST_POPULATION_CHANGED', model=model)
        require(len({r['request_id'] for r in declared_requests}) == len(declared_requests), 'DUPLICATE_PLANNED_REQUEST', model=model)
        requests = plan['readonly_checks']['plan_requests']
        require(declared_requests == requests, 'REQUESTS_DIFFER_FROM_FROZEN_PLAN', model=model)
        require({q['candidate_id'] for q in questions[model]} == set(canonical_by_id) and len(questions[model]) == 48, 'QUESTION_POPULATION_CHANGED', model=model)
        require(set(golds[model]) == set(canonical_by_id), 'GOLD_POPULATION_CHANGED', model=model)
        require(manifest['policy_version'] == plan['frozen_protocol']['policy_version'], 'POLICY_CHANGED', model=model)
        require(manifest['decoder_contract'].get('enable_thinking') is False, 'THINKING_CONTRACT_CHANGED', model=model)
        for key in ['temperature', 'top_p', 'max_tokens']:
            require(manifest['decoder_contract'].get(key) == plan['frozen_protocol'][key], 'DECODER_CONTRACT_CHANGED', model=model, key=key)
        for name, expected in manifest['input_sha256'].items():
            path = Path(manifest['pool_path']) if name == 'pool' else run / name
            require(path.is_file() and sha(path) == expected, 'NATIVE_INPUT_SHA_CHANGED', model=model, path=str(path))
            if path.is_file(): bindings[str(path.resolve())] = sha(path)
        for name, expected in manifest['code_sha256'].items():
            path = Path(name)
            require(path.is_file() and sha(path) == expected, 'NATIVE_CODE_SHA_CHANGED', model=model, path=str(path))
            if path.is_file(): bindings[str(path.resolve())] = sha(path)
        rby = defaultdict(list)
        for request in requests: rby[request['candidate_id']].append(request)
        for question in questions[model]:
            cid = question['candidate_id']; original = canonical_by_id[cid]
            field = 'options' if original['question_type'] == 'multi_select' else 'items'
            for key in ['candidate_id', 'root_id', 'question_type', 'question', field, 'source_path', 'source_sha256']:
                require(question.get(key) == original.get(key), 'CANONICAL_QUESTION_CHANGED', model=model, candidate_id=cid, field=key)
            expected_gold = ['content_' + hashlib.sha256(original[field][label].strip().encode()).hexdigest()[:16] for label in original['gold']]
            require(golds[model].get(cid) == expected_gold, 'CANONICAL_GOLD_CHANGED', model=model, candidate_id=cid)
            planned = rby[cid]
            count, slots = (6, 3) if original['question_type'] == 'multi_select' else (12, 1)
            require(len(planned) == count * slots, 'CANDIDATE_PLANNED_CALL_COUNT', model=model, candidate_id=cid)
            positions = {(r['variant']['variant_id'], r['slot']) for r in planned}
            require(positions == {(f'perm_{i:02d}', slot) for i in range(count) for slot in range(slots)}, 'CANDIDATE_VARIANT_SLOT_SET', model=model, candidate_id=cid)
            for request in planned:
                prompt = run / 'prompts' / (request['request_id'] + '.txt')
                require(prompt.is_file() and sha(prompt) == request['prompt_sha256'], 'PROMPT_CHANGED_OR_MISSING', model=model, request_id=request['request_id'])
                if prompt.is_file(): bindings[str(prompt.resolve())] = sha(prompt)
        observed = defaultdict(list)
        for path in sorted((run / 'responses').glob('*.json')):
            try:
                row = read(path)
            except (ValueError, OSError) as exc:
                problem('RESPONSE_FILE_UNREADABLE', model=model, path=str(path), error=type(exc).__name__); continue
            observed[row.get('request_id')].append((path, row))
        records[model] = observed
        expected_ids = {r['request_id'] for r in requests}
        for rid in set(observed) - expected_ids:
            problem('UNPLANNED_RESPONSE', model=model, request_id=rid)
        if resultfiles[model] is not None:
            require({q['candidate_id'] for q in resultfiles[model]['questions']} == set(canonical_by_id)
                    and len(resultfiles[model]['questions']) == 48, 'NATIVE_RESULTS_POPULATION_CHANGED', model=model)
            require(not resultfiles[model].get('call_set_errors'), 'NATIVE_CALL_SET_ERRORS', model=model)
        calls = []
        for request in requests:
            cid, rid = request['candidate_id'], request['request_id']; question = canonical_by_id[cid]
            qtype, g = question['question_type'], len(question['gold'])
            call = {'model': model, 'request_id': rid, 'candidate_id': cid, 'root_id': question['root_id'],
                    'question_type': qtype, 'gold_count': g, 'variant_id': request['variant']['variant_id'],
                    'slot': request['slot'], 'seed': request['seed'], 'prompt_sha256': request['prompt_sha256'],
                    'valid': False, 'exact_match': 0, 'signed_partial': None, 'pair_accuracy': None,
                    'tp': None, 'fp': None, 'fn': None, 'predicted_count': None, 'predicted_minus_gold_count': None,
                    'all6': None, 'overselection': None, 'underselection': None, 'correct_pairs': None, 'wrong_pairs': None,
                    'errors': [], 'display_prediction': None, 'content_prediction': None}
            found = observed.get(rid, [])
            if len(found) != 1:
                call['errors'].append('MISSING_RESPONSE' if not found else 'DUPLICATE_RESPONSE')
            else:
                path, response = found[0]
                call['response_path'] = str(path)
                expected_fields = {'request_id': rid, 'candidate_id': cid, 'root_id': request['root_id'], 'question_type': qtype,
                                   'variant_id': request['variant']['variant_id'], 'slot': request['slot'], 'seed': request['seed'],
                                   'mode': 'full_context', 'solver_model': manifest['solver_model'], 'model_path': manifest['model']}
                for key, expected in expected_fields.items():
                    if response.get(key) != expected or key == 'slot' and type(response.get(key)) is not int:
                        call['errors'].append('RESPONSE_PLAN_MISMATCH:' + key)
                metadata = response.get('metadata') or {}
                for key in ['temperature', 'top_p', 'max_tokens', 'structured_output_api', 'structured_output_version']:
                    if metadata.get(key) != manifest['decoder_contract'].get(key):
                        call['errors'].append('DECODER_METADATA_MISMATCH:' + key)
                if metadata.get('finish_reason') not in ['stop', 'eos']: call['errors'].append('BAD_FINISH_REASON')
                if response.get('evaluation_valid') is not True: call['errors'].append('NATIVE_EVALUATION_INVALID')
                if response.get('data_errors'): call['errors'].append('NATIVE_DATA_ERRORS')
                q = dict(question, gold_content_ids=golds[model][cid])
                score = scoring.score_response(q, request, response.get('raw_output'))
                if score != response.get('score'): call['errors'].append('SAVED_NATIVE_SCORE_DIFFERS_FROM_RECOMPUTATION')
                if not score['scored']: call['errors'].append(score.get('scoring_status', 'UNSCORED'))
                call['parse_status'], call['scoring_status'] = score['parse_status'], score.get('scoring_status')
                call['display_prediction'], call['content_prediction'] = score['display_prediction'], score['content_prediction']
                if not call['errors']:
                    call.update(valid=True, exact_match=int(score['exact_match']), signed_partial=score['partial_score_fraction'])
                    if qtype == 'multi_select':
                        count = len(score['content_prediction'])
                        call.update(tp=score['tp'], fp=score['fp'], fn=score['fn'], predicted_count=count,
                                    predicted_minus_gold_count=count - g, all6=int(count == 6),
                                    overselection=int(count > g), underselection=int(count < g))
                    else:
                        call.update(correct_pairs=score['tp'], wrong_pairs=score['fp'], pair_accuracy=str(Fraction(score['tp'], 15)))
            if call['errors']: problem('CALL_INVALID', model=model, request_id=rid, reasons=call['errors'])
            calls.append(call)
        callsets[model] = calls
        if resultfiles[model] is not None:
            native_by_id = {row['candidate_id']: row for row in resultfiles[model]['questions']}
            for cid in canonical_by_id:
                native = native_by_id.get(cid, {})
                these = [row for row in calls if row['candidate_id'] == cid]
                require(native.get('actual_full_calls') == sum(len(observed.get(row['request_id'], [])) for row in these)
                        and native.get('expected_full_calls') == len(these), 'NATIVE_RESULT_CALL_COUNTS', model=model, candidate_id=cid)
                if all(row['valid'] for row in these):
                    partial = mean([Fraction(row['signed_partial']) for row in these])
                    exact = Fraction(sum(row['exact_match'] for row in these), len(these))
                    require(native.get('full_partial_mean') == str(partial) and native.get('full_exact_rate') == str(exact)
                            and not native.get('data_errors'), 'NATIVE_RESULT_SCORE_SUMMARY', model=model, candidate_id=cid)

    require(manifests['baseline']['requests'] == manifests['new']['requests'], 'PAIRED_REQUESTS_NOT_IDENTICAL')
    require(questions['baseline'] == questions['new'] and golds['baseline'] == golds['new'], 'PAIRED_QUESTIONS_OR_GOLD_CHANGED')
    require(manifests['baseline']['decoder_contract'] == manifests['new']['decoder_contract'], 'PAIRED_DECODER_CHANGED')
    require(manifests['baseline']['tokenizer'] == manifests['new']['tokenizer'], 'PAIRED_TOKENIZER_CHANGED')
    require(manifests['baseline']['model'] != manifests['new']['model'], 'PAIRED_MODEL_PATHS_IDENTICAL')
    require(manifests['baseline']['requests'] == plan['readonly_checks']['plan_requests'], 'REQUESTS_DIFFER_FROM_FROZEN_PLAN')

    per_question, per_call = [], []
    candidate_metrics = {model: {} for model in ['baseline', 'new']}
    for cid, question in canonical_by_id.items():
        result = {'candidate_id': cid, 'root_id': question['root_id'], 'question_type': question['question_type'],
                  'gold_count': len(question['gold']), 'strata': strata(question, question['gold'])}
        for model in ['baseline', 'new']:
            calls = [row for row in callsets[model] if row['candidate_id'] == cid]
            summary = aggregate_calls(calls)
            by_perm = {}
            for vid in sorted({row['variant_id'] for row in calls}):
                by_perm[vid] = aggregate_calls([row for row in calls if row['variant_id'] == vid])
            complete = all(row['signed_partial']['mean'] is not None for row in by_perm.values())
            partials = [row['signed_partial']['mean'] for row in by_perm.values()] if complete else []
            exacts = [row['exact_rate_fixed'] for row in by_perm.values()]
            summary['permutations'] = by_perm
            summary['permutation_diagnostics'] = {
                'planned_permutations': len(by_perm), 'all_permutations_complete': complete,
                'signed_partial_min': min(partials) if partials else None,
                'signed_partial_max': max(partials) if partials else None,
                'signed_partial_range': max(partials) - min(partials) if partials else None,
                'exact_rate_fixed_min': min(exacts), 'exact_rate_fixed_max': max(exacts),
                'exact_rate_fixed_range': max(exacts) - min(exacts)}
            candidate_metrics[model][cid] = summary; result[model] = summary
        result['paired_delta'] = paired_delta(result['baseline'], result['new'])
        per_question.append(result)
    new_calls = {row['request_id']: row for row in callsets['new']}
    for baseline in callsets['baseline']:
        new = new_calls.get(baseline['request_id'])
        if new is None:
            problem('PAIRED_CALL_MISSING_FROM_NEW_PLAN', request_id=baseline['request_id']); continue
        matched = all(baseline[key] == new[key] for key in ['candidate_id', 'variant_id', 'slot', 'seed', 'prompt_sha256'])
        require(matched, 'PAIRED_CALL_LINK_MISMATCH', request_id=baseline['request_id'])
        partial = Fraction(new['signed_partial']) - Fraction(baseline['signed_partial']) if baseline['valid'] and new['valid'] else None
        per_call.append({'request_id': baseline['request_id'], 'matched_key_equal': matched,
                         'baseline': baseline, 'new': new,
                         'exact_delta': new['exact_match'] - baseline['exact_match'],
                         'signed_partial_delta': number(partial), 'signed_partial_delta_fraction': fraction_text(partial)})

    by_stratum, per_root = {}, []
    for name in STRATA:
        members = [row for row in per_question if name in row['strata']]
        value = {'candidate_count': len(members), 'models': {}}
        for model in ['baseline', 'new']:
            ids = {row['candidate_id'] for row in members}
            value['models'][model] = {'equal_candidate': aggregate_candidates([row[model] for row in members]),
                                      'pooled_calls': aggregate_calls([row for row in callsets[model] if row['candidate_id'] in ids])}
        value['paired_equal_candidate_delta'] = {
            'exact_rate': mean([Fraction(row['paired_delta']['exact_rate_delta_fraction']) for row in members]),
            'signed_partial': mean([Fraction(row['paired_delta']['signed_partial_delta_fraction']) for row in members]) if members and all(row['paired_delta']['signed_partial_delta_fraction'] is not None for row in members) else None,
            'signed_partial_lower': mean([Fraction(row['paired_delta']['signed_partial_delta_lower_fraction']) for row in members]),
            'signed_partial_upper': mean([Fraction(row['paired_delta']['signed_partial_delta_upper_fraction']) for row in members])}
        value['paired_equal_candidate_delta'] = {key: number(val) for key, val in value['paired_equal_candidate_delta'].items()}
        by_stratum[name] = value
    for root in fixed_roots:
        members = [row for row in per_question if row['root_id'] == root]
        root_record = {'root_id': root, 'has_questions': bool(members), 'candidate_count': len(members), 'strata': {}}
        for name in STRATA:
            subset = [row for row in members if name in row['strata']]
            root_record['strata'][name] = {model: aggregate_candidates([row[model] for row in subset]) for model in ['baseline', 'new']}
            root_record['strata'][name]['paired_delta'] = {
                'exact_rate': number(mean([Fraction(row['paired_delta']['exact_rate_delta_fraction']) for row in subset])),
                'signed_partial': number(mean([Fraction(row['paired_delta']['signed_partial_delta_fraction']) for row in subset])) if subset and all(row['paired_delta']['signed_partial_delta_fraction'] is not None for row in subset) else None,
                'signed_partial_lower': number(mean([Fraction(row['paired_delta']['signed_partial_delta_lower_fraction']) for row in subset])),
                'signed_partial_upper': number(mean([Fraction(row['paired_delta']['signed_partial_delta_upper_fraction']) for row in subset]))}
        per_root.append(root_record)
    root_macro = {}
    for name in STRATA:
        covered = [row for row in per_root if row['strata'][name]['baseline']['candidate_count']]
        root_macro[name] = {'covered_root_denominator': len(covered), 'fixed_dev_root_population': 15,
                            'uncovered_roots_not_scored': 15 - len(covered), 'models': {}}
        for model in ['baseline', 'new']:
            values = [row['strata'][name][model] for row in covered]
            root_macro[name]['models'][model] = {
                'exact_rate_equal_root': number(mean([Fraction(v['exact_rate_equal_candidate_fraction']) for v in values])),
                'signed_partial_equal_root': number(mean([Fraction(v['signed_partial_equal_candidate_fraction']) for v in values])) if values and all(v['signed_partial_equal_candidate_fraction'] is not None for v in values) else None,
                'signed_partial_equal_root_lower': number(mean([Fraction(str(v['signed_partial_equal_candidate_lower'])) for v in values])),
                'signed_partial_equal_root_upper': number(mean([Fraction(str(v['signed_partial_equal_candidate_upper'])) for v in values]))}
        root_macro[name]['paired_delta_equal_root'] = {
            key: number(mean([Fraction(str(row['strata'][name]['paired_delta'][key])) for row in covered]))
            if covered and all(row['strata'][name]['paired_delta'][key] is not None for row in covered) else None
            for key in ['exact_rate', 'signed_partial', 'signed_partial_lower', 'signed_partial_upper']}
    actual_pop = {'questions': len(per_question), 'multi_select': sum(q['question_type'] == 'multi_select' for q in canonical),
                  'ordering': sum(q['question_type'] == 'ordering' for q in canonical),
                  'multi_select_gold_counts': dict(Counter(str(len(q['gold'])) for q in canonical if q['question_type'] == 'multi_select')),
                  'covered_roots': len({q['root_id'] for q in canonical}), 'fixed_dev_roots': len(fixed_roots)}
    require(actual_pop == {'questions': 48, 'multi_select': 34, 'ordering': 14,
                           'multi_select_gold_counts': {'2': 7, '3': 6, '4': 7, '5': 7, '6': 7},
                           'covered_roots': 7, 'fixed_dev_roots': 15}, 'FINAL_POPULATION_MISMATCH')
    require({q['root_id'] for q in canonical} <= set(fixed_roots), 'CANDIDATE_ROOT_OUTSIDE_FIXED_DEV')
    for path, expected in bindings.items(): require(sha(path) == expected, 'INPUT_CHANGED_DURING_REPORT', path=path)
    summary = {'schema_version': 'v127_fixed_dev_paired_summary_v1',
               'status': 'COMPLETE' if not errors else 'INCOMPLETE_OR_INVALID', 'errors': errors,
               'population': actual_pop, 'planned_calls_per_model': 780, 'planned_calls_total': 1560,
               'models': {model: {'solver_model': manifests[model]['solver_model'], 'model_path': manifests[model]['model'],
                                   'call_summary': aggregate_calls(callsets[model]),
                                   'native_selected_count_diagnostic_only': len(resultfiles[model]['selected_questions']) if resultfiles[model] else None,
                                   'native_call_set_errors': resultfiles[model].get('call_set_errors') if resultfiles[model] else None}
                          for model in ['baseline', 'new']},
               'strata': by_stratum, 'root_macro': root_macro,
               'weighting': 'Headline metrics give each candidate equal weight, then root macros give each covered root equal weight. Pooled-call summaries are separately labeled and weight 18-call multi-select questions more than 12-call ordering questions.',
               'failure_policy': 'All fixed requests retained. Exact failures count 0; signed partial, pair accuracy and count metrics retain fixed-denominator bounds with undefined point estimates. No valid-only headline and no best-of retries.',
               'interpretation': ['Gold6 controls are separate; always selecting all6 scores partial0 on Gold2345 and partial1/exact1 on Gold6.',
                                  'Only7 of15 fixed dev roots have questions. Uncovered roots have no accuracy, and are not silently treated as measured successes or failures.',
                                  'Permutations, rollouts, Gold variants and questions sharing roots are correlated. No significance test, confidence claim or treatment of1560 calls as independent observations.',
                                  'This pair evaluates fixed dev only. No test question/source content read and no test inference performed by this report.',
                                  'Native selected_questions is reported solely as a diagnostic and never filters any metric.'],
               'input_bindings': [{'path': path, 'sha256': value} for path, value in sorted(bindings.items())]}
    out.mkdir(parents=True, exist_ok=True)
    def write(name, obj): (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    write('PAIRED_SUMMARY.json', summary); write('PER_QUESTION.json', per_question); write('PER_CALL.json', per_call); write('PER_ROOT.json', per_root)
    flat = []
    for row in per_question:
        record = {key: row[key] for key in ['candidate_id', 'root_id', 'question_type', 'gold_count']}
        for model in ['baseline', 'new']:
            value = row[model]
            record.update({model + '_planned_calls': value['planned_calls'], model + '_valid_calls': value['valid_calls'],
                           model + '_exact': value['exact_rate_fixed'], model + '_signed_partial': value['signed_partial']['mean'],
                           model + '_signed_partial_lower': value['signed_partial']['lower'], model + '_signed_partial_upper': value['signed_partial']['upper'],
                           model + '_permutation_range': value['permutation_diagnostics']['signed_partial_range']})
            for key in ['tp', 'fp', 'fn', 'predicted_count', 'predicted_minus_gold_count', 'all6_rate', 'overselection_rate']:
                record[model + '_' + key] = value.get('multi_select', {}).get(key, {}).get('mean')
            record[model + '_ordering_pair_accuracy'] = value.get('ordering', {}).get('pair_accuracy', {}).get('mean')
        record.update(row['paired_delta']); flat.append(record)
    buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    (out / 'PER_QUESTION.csv').write_text(buffer.getvalue())
    def fmt(value): return '未定义' if value is None else f'{value:.4f}'
    lines = [f"固定 dev 配对评估状态：{summary['status']}。", '',
             '固定 48 题：34 道多选（Gold2/3/4/5/6 为7/6/7/7/7），14 道排序。每模型780次调用，共1560次；没有按分数或 native selected_questions 筛题。', '',
             '共15个固定 dev root，其中7个有入选题、8个无题。下表每题等权；逐 root 等权结果见 PAIRED_SUMMARY.json。', '',
             '| 范围 | 题数 | 原模型 exact | 新模型 exact | exact变化 | 原 signed partial | 新 signed partial |',
             '|---|---:|---:|---:|---:|---:|---:|']
    labels = {'all48': '全部（含控制）', 'main41_excluding_gold6': '主任务（不含Gold6）', 'multi_select_all34': '多选全部',
              'multi_select_gold2345_27': '多选Gold2345', 'multi_select_gold6_control7': 'Gold6控制', 'ordering14': '排序'}
    for name in STRATA:
        s = by_stratum[name]; a = s['models']['baseline']['equal_candidate']; b = s['models']['new']['equal_candidate']
        lines.append(f"| {labels.get(name,name)} | {s['candidate_count']} | {fmt(a['exact_rate_equal_candidate'])} | {fmt(b['exact_rate_equal_candidate'])} | {fmt(s['paired_equal_candidate_delta']['exact_rate'])} | {fmt(a['signed_partial_equal_candidate'])} | {fmt(b['signed_partial_equal_candidate'])} |")
    lines += ['', '多选的 TP/FP/FN、平均选择数、全选率、过选/少选、Gold分层与排列敏感性已保留；排序另列15对关系的 pair accuracy。多选 signed partial = TP/g − FP/(6−g)，Gold6为TP/6；排序为(correct−wrong)/15，pair accuracy=(signed+1)/2。', '',
              '缺失、重复、解析失败、解码或身份不符的调用保留原分母，exact按0计；其余指标不编造有效分数，报告固定分母上下界。本报告存在任何完整性错误时均标 INCOMPLETE_OR_INVALID。', '',
              '同一root内题目、Gold变体、排列与采样存在相关性；这里只给描述性配对差值，不作显著性或泛化结论。Gold6控制单列，避免全选策略抬高主任务结果。没有读取测试题文或运行测试评估。', '',
              '逐题：PER_QUESTION.csv / PER_QUESTION.json；逐调用：PER_CALL.json；全部15个root：PER_ROOT.json。']
    if errors: lines += ['', f'完整性错误共{len(errors)}条，详情见 PAIRED_SUMMARY.json；当前结果不能作为完整评估通过。']
    (out / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': summary['status'], 'error_count': len(errors), 'summary': str(out / 'PAIRED_SUMMARY.json')}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
