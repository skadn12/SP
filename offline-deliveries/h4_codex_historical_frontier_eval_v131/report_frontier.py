"""Fixed-population, CPU-only historical frontier comparison; no inference.

Historical M0/clean are selection evidence. New live_clean/live_new use one
current runtime as the bridge comparison. All123 questions and1629 calls per
side remain in the denominator, regardless of validity or score.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OLD = REPO / 'restart-n5-runs/h4_round0_trained_solver_train_reeval_v1_live_1'
M0 = REPO / 'restart-n5-runs/h4_round0_b0_k3_difficulty_v1_live_1'
MATRIX = OLD / 'candidate_transition_matrix.jsonl'
MATRIX_SHA = '853118ab1f484b70c440dd486bb39b5be48dabefb7e567a998bc7e49e04aafae'
MODELS = ['historical_m0', 'historical_clean', 'live_clean', 'live_new']
LIVE = ['live_clean', 'live_new']
TYPES = ['single_choice', 'multi_select', 'ordering']
DESCRIPTOR = ['request_id', 'candidate_id', 'root_id', 'question_type', 'variant_id',
              'shift', 'slot_id', 'seed', 'context_mode', 'prompt_sha256', 'prompt_token_count',
              'display_to_canonical', 'canonical_gold', 'displayed_gold']
SCORE_KEYS = ['parse_status', 'parser_version', 'scored', 'correct', 'display_prediction',
              'canonical_prediction', 'failure_type', 'metrics']
MODEL_IDS = {'live_clean': 'M_clean_policy', 'live_new': 'M_v125_newonly'}
MODEL_PATHS = {
    'live_clean': str(REPO / 'restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model'),
    'live_new': str(REPO / 'restart-n5-runs/h4_codex_combined_sft_v125_merged'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def filename(value):
    return hashlib.sha256(value.encode()).hexdigest() + '.json'


def mean(values):
    return sum(values, Fraction()) / len(values) if values else None


def num(value):
    return None if value is None else float(value)


def frac(value):
    return None if value is None else str(Fraction(value))


def metric(rows, key, low=0, high=1):
    n = len(rows)
    known = [Fraction(r[key]) for r in rows if r.get(key) is not None]
    lows = [Fraction(r[key]) if r.get(key) is not None else Fraction(low) for r in rows]
    highs = [Fraction(r[key]) if r.get(key) is not None else Fraction(high) for r in rows]
    value = mean(known) if len(known) == n and n else None
    return {'mean': num(value), 'mean_fraction': frac(value),
            'lower': num(mean(lows)), 'upper': num(mean(highs)),
            'lower_fraction': frac(mean(lows)), 'upper_fraction': frac(mean(highs)),
            'planned_denominator': n, 'defined_count': len(known), 'undefined_count': n - len(known)}


def independent(raw, question, request):
    """Reconstruct canonical labels and exact/Jaccard/pairs independently."""
    qtype = question['question_type']; mapping = request['display_to_canonical']
    match = re.fullmatch(r'\s*<box>\s*(.*?)\s*</box>\s*', raw, re.S) if isinstance(raw, str) else None
    if not match:
        raise ValueError('BOX_PARSE_FAILURE')
    display = [x.strip().upper() for x in match.group(1).split(',')]
    if not display or any(re.fullmatch(r'[A-Z][A-Z0-9_]*', x) is None for x in display):
        raise ValueError('MALFORMED_LABELS')
    if len(set(display)) != len(display) or any(x not in mapping for x in display):
        raise ValueError('DUPLICATE_OR_INVALID_DISPLAY_LABEL')
    prediction = [mapping[x] for x in display]; gold = request['canonical_gold']
    if len(set(prediction)) != len(prediction):
        raise ValueError('DUPLICATE_CANONICAL_LABEL')
    result = {'display_prediction': display, 'canonical_prediction': prediction,
              'jaccard': None, 'pair_accuracy': None, 'correct_pairs': None, 'wrong_pairs': None,
              'tp': None, 'fp': None, 'fn': None, 'predicted_count': len(prediction)}
    if qtype == 'single_choice':
        if len(prediction) != 1: raise ValueError('SINGLE_CHOICE_COUNT_INVALID')
        result['exact_match'] = int(prediction == gold)
    elif qtype == 'multi_select':
        pred_set, gold_set = set(prediction), set(gold)
        tp, fp, fn = len(pred_set & gold_set), len(pred_set - gold_set), len(gold_set - pred_set)
        result.update(canonical_prediction=sorted(pred_set), exact_match=int(pred_set == gold_set),
                      jaccard=str(Fraction(tp, len(pred_set | gold_set))), tp=tp, fp=fp, fn=fn)
    elif qtype == 'ordering':
        if len(prediction) != len(gold) or set(prediction) != set(gold):
            raise ValueError('INCOMPLETE_ORDER')
        positions = {x: i for i, x in enumerate(prediction)}
        pairs = len(gold) * (len(gold) - 1) // 2
        correct = sum(positions[a] < positions[b] for a, b in itertools.combinations(gold, 2))
        result.update(exact_match=int(prediction == gold), correct_pairs=correct, wrong_pairs=pairs - correct,
                      pair_accuracy=str(Fraction(correct, pairs)), gold_rank_prediction=[gold.index(x) + 1 for x in prediction])
    else:
        raise ValueError('UNKNOWN_QUESTION_TYPE')
    return result


def aggregate(rows):
    n = len(rows); valid = [r for r in rows if r['valid']]
    exact = sum(r['exact_match'] for r in rows)
    ms = [r for r in rows if r['question_type'] == 'multi_select']
    order = [r for r in rows if r['question_type'] == 'ordering']
    return {'planned_calls': n, 'valid_calls': len(valid), 'invalid_or_missing_calls': n - len(valid),
            'exact_correct': exact, 'exact_rate_fixed': exact / n if n else None,
            'exact_rate_fraction': str(Fraction(exact, n)) if n else None,
            'exact_rate_upper_if_failures_resolved': (exact + n - len(valid)) / n if n else None,
            'ordering': {'planned_calls': len(order), 'pair_accuracy': metric(order, 'pair_accuracy'),
                         'known_correct_pairs': sum(r.get('correct_pairs') or 0 for r in order),
                         'planned_pairs': sum(r['item_count'] * (r['item_count'] - 1) // 2 for r in order),
                         'unknown_pairs': sum(r['item_count'] * (r['item_count'] - 1) // 2 for r in order if not r['valid'])},
            'multi_select': {'planned_calls': len(ms), 'jaccard': metric(ms, 'jaccard'),
                             'predicted_count': metric(ms, 'predicted_count', 0, 5),
                             'known_tp': sum(r.get('tp') or 0 for r in ms),
                             'known_fp': sum(r.get('fp') or 0 for r in ms),
                             'known_fn': sum(r.get('fn') or 0 for r in ms)}}


def aggregate_questions(rows, model):
    values = [r['models'][model] for r in rows]
    exact = mean([Fraction(r['exact_rate_fraction']) for r in values])
    result = {'question_count': len(rows), 'planned_calls': sum(r['planned_calls'] for r in values),
              'valid_calls': sum(r['valid_calls'] for r in values),
              'exact_rate_equal_question': num(exact), 'exact_rate_equal_question_fraction': frac(exact)}
    for qtype, branch, key in [('ordering', 'ordering', 'pair_accuracy'), ('multi_select', 'multi_select', 'jaccard')]:
        metrics = [r['models'][model][branch][key] for r in rows if r['question_type'] == qtype]
        complete = metrics and all(r['mean_fraction'] is not None for r in metrics)
        result[branch + '_' + key + '_equal_question'] = {
            'question_count': len(metrics),
            'mean': num(mean([Fraction(r['mean_fraction']) for r in metrics])) if complete else None,
            'lower': num(mean([Fraction(r['lower_fraction']) for r in metrics])),
            'upper': num(mean([Fraction(r['upper_fraction']) for r in metrics]))}
    return result


def paired(a, b):
    result = {'direction': 'second_minus_first',
              'exact_delta': num(Fraction(b['exact_rate_fraction']) - Fraction(a['exact_rate_fraction'])) if a['planned_calls'] else None}
    for branch, key in [('ordering', 'pair_accuracy'), ('multi_select', 'jaccard')]:
        x, y = a[branch][key], b[branch][key]
        if not x['planned_denominator']:
            result[key + '_delta'] = None; continue
        result[key + '_delta'] = num(Fraction(y['mean_fraction']) - Fraction(x['mean_fraction'])) if x['mean_fraction'] is not None and y['mean_fraction'] is not None else None
        result[key + '_delta_lower'] = num(Fraction(y['lower_fraction']) - Fraction(x['upper_fraction']))
        result[key + '_delta_upper'] = num(Fraction(y['upper_fraction']) - Fraction(x['lower_fraction']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope-dir', type=Path, default=HERE)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(); scope = args.scope_dir.resolve(); out = (args.output_dir or scope).resolve()
    if out.is_relative_to(scope / 'responses') or out.is_relative_to(OLD) or out.is_relative_to(M0):
        raise RuntimeError('REPORT_OUTPUT_OVERLAPS_FROZEN_INPUTS')
    names = ['SUMMARY.json', 'REPORT_ZH.md', 'PER_QUESTION.json', 'PER_QUESTION.csv', 'PER_CALL.json', 'PER_ROOT.json']
    if any((out / name).exists() for name in names): raise RuntimeError('REPORT_EXISTS_NO_SILENT_OVERWRITE')
    bindings, errors = {}, []
    def bind(path):
        path = Path(path).resolve(); bindings[str(path)] = sha(path); return path
    def read(path): return json.loads(bind(path).read_text())
    def lines(path): return [json.loads(x) for x in bind(path).read_text().splitlines() if x.strip()]
    def problem(code, **details): errors.append({'code': code, **details})
    def require(condition, code, **details):
        if not condition: problem(code, **details)
    bind(__file__)
    if sha(MATRIX) != MATRIX_SHA: raise RuntimeError('HISTORICAL_SELECTION_MATRIX_CHANGED')
    selected = [r for r in lines(MATRIX) if r['models']['M0']['full_context_correct_count'] == 0 and 0 < r['models']['M_clean_policy']['full_context_accuracy'] < 1]
    selected_by_id = {r['candidate_id']: r for r in selected}
    require(len(selected) == 123 and Counter(r['question_type'] for r in selected) == {'ordering': 62, 'multi_select': 51, 'single_choice': 10}, 'FROZEN_123_SELECTION_CHANGED')
    require(len({r['root_id'] for r in selected}) == 58 and all(r['split'] == 'train' and r['training_exposure'] == 'UNEXPOSED_TO_BOTH' for r in selected), 'HISTORICAL_SPLIT_OR_EXPOSURE_CHANGED')
    quality_path = scope / 'evidence/HISTORICAL_QUALITY_CROSSCHECK.json'
    quality = read(quality_path) if quality_path.exists() else None
    quality_rows = {r['candidate_id']: r for r in quality['rows']} if quality else {}
    if quality:
        require(set(quality_rows) == set(selected_by_id) and len(quality['rows']) == 123, 'QUALITY_QUESTION_SET_CHANGED')
    both_ids = {r['candidate_id'] for r in selected if 0 < r['models']['M_inclusive_policy']['full_context_accuracy'] < 1}
    clean_only_ids = set(selected_by_id) - both_ids
    require(len(both_ids) == 93 and len(clean_only_ids) == 30, 'HISTORICAL_BOTH_CLEAN_ONLY_STRATA_CHANGED')
    seed_path = REPO / 'restart-n5-runs/h4_round1_feedback_compiler_v1_live_1/feedback_seed_manifest.jsonl'
    seed_ids = {r['seed_candidate_id'] for r in lines(seed_path)}
    require(len(seed_ids) == 10 and seed_ids <= both_ids, 'FORMAL10_SEED_POPULATION_CHANGED')
    manifest = read(scope / 'manifest.json')
    requests = manifest['requests']
    require(len(requests) == 1629 and len({r['request_id'] for r in requests}) == 1629, 'PLANNED_1629_CALL_SET_CHANGED')
    require({r['candidate_id'] for r in requests} == set(selected_by_id), 'PLANNED_123_QUESTION_SET_CHANGED')
    preflight = read(OLD / 'preflight_report.json')
    originals = {}
    for root in sorted({r['root_id'] for r in selected}):
        source = preflight['source_root_inventory'][root]; pool_path = Path(source['candidate_pool_path'])
        require(sha(pool_path) == source['candidate_pool_sha256'], 'ORIGINAL_POOL_CHANGED', root_id=root)
        pool = read(pool_path)
        for q in pool:
            if q['candidate_id'] in selected_by_id: originals[q['candidate_id']] = q
    require(set(originals) == set(selected_by_id), 'ORIGINAL_CANDIDATES_MISSING')
    package_questions = None
    if manifest.get('questions_path'):
        value = read(manifest['questions_path']); package_questions = value.get('candidates', value) if isinstance(value, dict) else value
    elif (scope / 'QUESTIONS.json').exists():
        value = read(scope / 'QUESTIONS.json'); package_questions = value.get('candidates', value) if isinstance(value, dict) else value
    if package_questions is not None:
        require({q['candidate_id']: q for q in package_questions} == originals, 'EXPORTED_QUESTIONS_CHANGED')
    if 'candidates' in manifest:
        require(len(manifest['candidates']) == 123 and {q['candidate_id'] for q in manifest['candidates']} == set(originals), 'MANIFEST_CANDIDATE_POPULATION_CHANGED')
        for row in manifest['candidates']:
            require(row['transition'] == selected_by_id[row['candidate_id']], 'MANIFEST_TRANSITION_CHANGED', candidate_id=row['candidate_id'])
    for model in LIVE:
        identity = manifest.get('identities', {}).get(model, {})
        require(identity.get('model_path') == MODEL_PATHS[model] and identity.get('model_identifier') == MODEL_IDS[model], 'MANIFEST_MODEL_IDENTITY_CHANGED', model=model)
    require(manifest.get('runtime', {}).get('vllm') == '0.21.0', 'CURRENT_RUNTIME_NOT_FROZEN_VLLM021')
    import_root = Path(manifest.get('native_import_root', REPO / 'worktrees/h4-gate-downstream-v1'))
    sys.path.insert(0, str(import_root))
    contract = importlib.import_module('scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.contract')
    bind(contract.__file__)
    require(Path(contract.__file__).resolve() == Path(manifest['native_contract_path']).resolve(), 'NATIVE_CONTRACT_IMPORT_PATH_MISMATCH')
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and name.startswith('scripts.current_method.restart_n5_spell') and Path(path).is_file(): bind(path)
    historical, descriptors, variants = {}, {}, {}
    for cid, question in originals.items():
        historical[cid] = {}
        for model, root in [('historical_m0', M0), ('historical_clean', OLD / 'clean_train')]:
            value = read(root / 'candidate_results' / filename(cid))
            require(value['candidate_id'] == cid and value['question_type'] == question['question_type'], 'HISTORICAL_CANDIDATE_ID_MISMATCH', model=model, candidate_id=cid)
            full = [r for r in value['requests'] if r['context_mode'] == 'full_context']
            historical[cid][model] = {r['request_id']: r for r in full}
            require(len(full) == (15 if question['question_type'] == 'multi_select' else 12) and len({r['request_id'] for r in full}) == len(full), 'HISTORICAL_CANDIDATE_CALL_SET_INVALID', model=model, candidate_id=cid)
        vv = {v['variant_id']: v for v in contract.position_variants(question)}
        for rid, r in historical[cid]['historical_clean'].items():
            descriptors[rid] = {key: r[key] for key in DESCRIPTOR}; variants[rid] = vv[r['variant_id']]
            require(all(historical[cid]['historical_m0'].get(rid, {}).get(key) == r[key] for key in DESCRIPTOR), 'HISTORICAL_M0_CLEAN_PLAN_MISMATCH', request_id=rid)
    require(len(descriptors) == 1629 and {r['request_id'] for r in requests} == set(descriptors), 'MANIFEST_REQUEST_POPULATION_CHANGED')
    for req in requests:
        rid = req['request_id']
        require(all(req.get(key) == value for key, value in descriptors[rid].items()), 'MANIFEST_DESCRIPTOR_DIFFERS_FROM_OLD', request_id=rid)
        if 'original_response' in req:
            require(req['original_response'] == historical[req['candidate_id']]['historical_clean'][rid], 'ORIGINAL_RESPONSE_CHANGED', request_id=rid)
        if 'variant' in req: require(req['variant'] == variants[rid], 'MANIFEST_VARIANT_CHANGED', request_id=rid)
        path = Path(req['prompt_path'])
        require(path.is_file() and sha(path) == req['prompt_sha256'], 'PROMPT_BYTES_CHANGED', request_id=rid)
        if path.is_file(): bind(path)
    observed = {m: defaultdict(list) for m in LIVE}
    expected_files = {filename(r['request_id']): r['request_id'] for r in requests}
    for model in LIVE:
        for path in sorted((scope / 'responses' / model).glob('*.json')):
            try: response = read(path)
            except (ValueError, OSError) as exc:
                problem('UNREADABLE_RESPONSE', model=model, path=str(path), error=type(exc).__name__); continue
            rid = response.get('request_id'); observed[model][rid].append((path, response))
            require(path.name == filename(rid) if isinstance(rid, str) else False, 'RESPONSE_FILENAME_MISMATCH', model=model, path=str(path))
            require(rid in descriptors, 'UNPLANNED_RESPONSE', model=model, request_id=rid)
    calls = {m: [] for m in MODELS}
    for req in requests:
        rid, cid = req['request_id'], req['candidate_id']; q = originals[cid]; qtype = q['question_type']; variant = variants[rid]
        for model in MODELS:
            row = {'model': model, 'request_id': rid, 'candidate_id': cid, 'root_id': req['root_id'], 'question_type': qtype,
                   'variant_id': req['variant_id'], 'shift': req['shift'], 'slot_id': req['slot_id'], 'seed': req['seed'], 'prompt_sha256': req['prompt_sha256'],
                   'item_count': len(q['items'] if qtype == 'ordering' else q['options']),
                   'valid': False, 'exact_match': 0, 'jaccard': None, 'pair_accuracy': None, 'predicted_count': None, 'errors': []}
            response = None
            if model.startswith('historical_'):
                response = historical[cid][model].get(rid)
                if response is None: row['errors'].append('MISSING_HISTORICAL_RESPONSE')
                else:
                    for key in DESCRIPTOR:
                        if response.get(key) != req[key]: row['errors'].append('HISTORICAL_DESCRIPTOR_MISMATCH:' + key)
            else:
                found = observed[model].get(rid, [])
                if len(found) != 1: row['errors'].append('MISSING_RESPONSE' if not found else 'DUPLICATE_RESPONSE')
                else:
                    path, response = found[0]; row['response_path'] = str(path)
                    for key in DESCRIPTOR:
                        if response.get(key) != req[key]: row['errors'].append('RESPONSE_DESCRIPTOR_MISMATCH:' + key)
                    if response.get('solver_model') != MODEL_IDS[model] or response.get('model_path') != MODEL_PATHS[model]: row['errors'].append('RESPONSE_MODEL_IDENTITY_MISMATCH')
                    if response.get('model_fingerprint_sha256') != manifest['identities'][model]['model_fingerprint_sha256']: row['errors'].append('RESPONSE_MODEL_FINGERPRINT_MISMATCH')
                    if response.get('runtime') != manifest['runtime']: row['errors'].append('RESPONSE_RUNTIME_MISMATCH')
                    if response.get('side') != model: row['errors'].append('RESPONSE_SIDE_MISMATCH')
            if response is not None:
                raw = response.get('raw_output'); metadata = response.get('metadata', response.get('backend_metadata')) or {}
                expected_meta = {'sampling_seed': req['seed'], 'temperature': .7 if qtype == 'single_choice' else .8, 'top_p': .9, 'max_tokens': 32,
                                 'prompt_token_count': req['prompt_token_count']}
                if qtype != 'single_choice': expected_meta['structured_output_version'] = 'finite_choice_set_v1'
                for key, expected in expected_meta.items():
                    if metadata.get(key) != expected: row['errors'].append('METADATA_MISMATCH:' + key)
                if metadata.get('finish_reason') not in {'stop', 'eos'}: row['errors'].append('BAD_FINISH_REASON')
                if model in LIVE and qtype != 'single_choice' and metadata.get('structured_output_api') != 'structured_outputs=StructuredOutputsParams(choice)': row['errors'].append('LIVE_STRUCTURED_API_MISMATCH')
                if model in LIVE and response.get('evaluation_valid') is not True: row['errors'].append('LIVE_EVALUATION_INVALID')
                score = contract.score_output(raw, q, variant)
                saved = response.get('score') if model in LIVE else {key: response.get(key) for key in SCORE_KEYS}
                if model in LIVE:
                    if score != saved: row['errors'].append('SAVED_NATIVE_SCORE_DIFFERS_FROM_RECOMPUTATION')
                elif any(score.get(key) != saved.get(key) for key in SCORE_KEYS): row['errors'].append('HISTORICAL_SCORE_DIFFERS_FROM_RECOMPUTATION')
                try:
                    recovered = independent(raw, q, req)
                    if not score['scored'] or bool(score.get('correct')) != bool(recovered['exact_match']) or score.get('canonical_prediction') != recovered['canonical_prediction'] or score.get('display_prediction') != recovered['display_prediction']:
                        row['errors'].append('INDEPENDENT_NATIVE_EXACT_OR_LABELS_DIFFER')
                    key = 'pairwise_accuracy' if qtype == 'ordering' else 'jaccard'
                    independent_key = 'pair_accuracy' if qtype == 'ordering' else 'jaccard'
                    if qtype != 'single_choice' and not math.isclose(score['metrics'].get(key, -1), float(Fraction(recovered[independent_key])), rel_tol=0, abs_tol=1e-14): row['errors'].append('INDEPENDENT_NATIVE_PARTIAL_DIFFERS')
                    if not row['errors']: row.update(recovered, valid=True, raw_output=raw)
                except (ValueError, KeyError, ZeroDivisionError) as exc: row['errors'].append('INDEPENDENT_UNSCORABLE:' + str(exc))
            if row['errors']: problem('CALL_INVALID', model=model, request_id=rid, reasons=row['errors'])
            calls[model].append(row)
    per_question = []
    comparisons = [('historical_m0', 'historical_clean'), ('historical_clean', 'live_clean'), ('live_clean', 'live_new')]
    for cid, q in originals.items():
        models = {m: aggregate([c for c in calls[m] if c['candidate_id'] == cid]) for m in MODELS}
        row = {'candidate_id': cid, 'root_id': q['root_id'], 'question_type': q['question_type'],
               'training_exposure': selected_by_id[cid]['training_exposure'], 'split': 'train',
               'gold_reliability': quality_rows.get(cid, {}).get('quality_state', 'UNKNOWN'),
               'gold_reliability_scope': 'Historical review applicability only; no new semantic Gold review or filtering.',
               'historical_same_content_review_count': len(quality_rows.get(cid, {}).get('applicable_historical_reviews', [])),
               'historical_frontier_group': 'BOTH93' if cid in both_ids else 'CLEAN_ONLY30',
               'formal_feedback_seed': cid in seed_ids, 'models': models,
               'paired_deltas': {a + '_to_' + b: paired(models[a], models[b]) for a, b in comparisons}}
        expected = selected_by_id[cid]['models']
        for key, historical_key in [('historical_m0', 'M0'), ('historical_clean', 'M_clean_policy')]:
            require(models[key]['exact_correct'] == expected[historical_key]['full_context_correct_count'] and models[key]['planned_calls'] == expected[historical_key]['full_context_total_count'], 'HISTORICAL_MATRIX_EXACT_COUNT_MISMATCH', candidate_id=cid, model=key)
        per_question.append(row)
    per_call = []
    by_model = {m: {r['request_id']: r for r in calls[m]} for m in MODELS}
    for req in requests:
        models = {m: by_model[m][req['request_id']] for m in MODELS}
        per_call.append({'request_id': req['request_id'], 'candidate_id': req['candidate_id'], 'root_id': req['root_id'], 'question_type': req['question_type'], 'models': models,
                         'paired_deltas': {a + '_to_' + b: {'exact_delta': models[b]['exact_match'] - models[a]['exact_match'],
                             'pair_accuracy_delta': num(Fraction(models[b]['pair_accuracy']) - Fraction(models[a]['pair_accuracy'])) if models[a]['valid'] and models[b]['valid'] and req['question_type'] == 'ordering' else None,
                             'jaccard_delta': num(Fraction(models[b]['jaccard']) - Fraction(models[a]['jaccard'])) if models[a]['valid'] and models[b]['valid'] and req['question_type'] == 'multi_select' else None} for a, b in comparisons}})
    per_root = []
    for root in sorted({r['root_id'] for r in selected}):
        members = [q for q in per_question if q['root_id'] == root]
        per_root.append({'root_id': root, 'question_count': len(members), 'models': {m: aggregate_questions(members, m) for m in MODELS}})
    strata = {}
    populations = {'all123': set(selected_by_id),
                   **{qtype: {q['candidate_id'] for q in selected if q['question_type'] == qtype} for qtype in TYPES},
                   'historical_both93': both_ids, 'historical_clean_only30': clean_only_ids,
                   'formal_feedback_seeds10': seed_ids}
    for name, population in populations.items():
        members = [q for q in per_question if q['candidate_id'] in population]
        ids = {q['candidate_id'] for q in members}; models = {}
        for m in MODELS:
            models[m] = {'equal_question': aggregate_questions(members, m), 'pooled_calls': aggregate([r for r in calls[m] if r['candidate_id'] in ids])}
        strata[name] = {'question_count': len(members), 'models': models,
                        'paired_equal_question_exact_delta': {a + '_to_' + b: models[b]['equal_question']['exact_rate_equal_question'] - models[a]['equal_question']['exact_rate_equal_question'] for a, b in comparisons}}
    root_macro = {m: {'covered_train_roots': len(per_root), 'exact_rate_equal_root': num(mean([Fraction(r['models'][m]['exact_rate_equal_question_fraction']) for r in per_root]))} for m in MODELS}
    for path, digest in bindings.items(): require(sha(path) == digest, 'INPUT_CHANGED_DURING_REPORT', path=path)
    summary = {'schema_version': 'v131_historical_frontier_fixed_report_v1', 'status': 'COMPLETE' if not errors else 'INCOMPLETE_OR_INVALID', 'errors': errors,
        'population': {'questions': 123, 'train_roots': 58, 'question_type_counts': {'ordering': 62, 'multi_select': 51, 'single_choice': 10},
                       'planned_calls_per_model': 1629, 'new_current_runtime_models': 2, 'new_inference_calls': 3258, 'reused_historical_full_calls': 3258,
                       'comparison_calls_total': 6516, 'no_context_calls_added': 0, 'test_calls_added': 0},
        'models': {m: {'solver_model': 'M0' if m == 'historical_m0' else 'M_clean_policy' if m in {'historical_clean', 'live_clean'} else 'M_v125_newonly',
                       'runtime_group': 'historical_vllm_0_10_2' if m.startswith('historical_') else 'current_vllm_0_21_0',
                       'raw_reused': m.startswith('historical_'), 'summary': aggregate(calls[m])} for m in MODELS},
        'strata': strata, 'root_macro': root_macro,
        'historical_gold_review_applicability': quality['summary'] if quality else {'status': 'NO_APPLICABLE_REVIEW_FILE_PROVIDED', 'unknown_questions': 123},
        'gold_reliability_state_counts': dict(Counter(q['gold_reliability'] for q in per_question)),
        'primary_comparison': 'live_clean versus live_new under the same current runtime, exact old prompts, variants and seeds',
        'runtime_bridge': 'historical_clean versus live_clean exposes runtime drift; historical M0 is reused only and is not a current-runtime base control',
        'selection_rule': 'Historical M0 full exact-correct count=0 and historical M_clean full exact accuracy strictly between0 and1. Partial means intermittent complete success across repeated calls, not pairwise or Jaccard partial credit.',
        'weighting': 'Headline exact rates give each of123 questions equal weight. Pooled calls weight15-call multiselect more than12-call ordering/single-choice and are separate. Root macro gives each of58 covered Train roots equal weight.',
        'failure_policy': 'Keep all planned calls. Missing/duplicate/invalid exact=0; pair and Jaccard points are undefined with fixed-denominator[0,1] bounds. No valid-only headline, no retries, no best-of, no filtering.',
        'limits': ['This set was selected using historical M0=zero and M_clean=partial; the apparent historical improvement is partly guaranteed by selection and cannot estimate general training gain.',
                   'All58 roots belong to Train. The123 exact questions were unexposed to both first-round policies, but their parent roots may be trained, and later training uses related roots. This is a selected Train diagnostic, not DEV/TEST or unseen-root generalization.',
                   'Historical Gold and question wording remain frozen. No new semantic correctness review is performed; old ambiguous/incorrect/unsupported labels may exist. Scores measure agreement with old Gold, not independently established truth.',
                   'Old four-item ordering has4 cyclic display-order rotations with identity labels, not six-item ordering or twelve independent relabelings. Five-option multiselect has5 cyclic position variants. Three rollouts each are correlated.',
                   'Historical vllm0.10.2 and current vllm0.21.0 may differ. Only live_clean/live_new is the primary same-runtime comparison; do not attribute historical-to-live change solely to weights.',
                   'No TEST question/source content, no no_context inference, no question rewriting, no training, no Gold-based exclusion.'],
        'input_bindings': [{'path': p, 'sha256': digest} for p, digest in sorted(bindings.items())]}
    out.mkdir(parents=True, exist_ok=True)
    def write(name, value): (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    write('SUMMARY.json', summary); write('PER_QUESTION.json', per_question); write('PER_CALL.json', per_call); write('PER_ROOT.json', per_root)
    flat = []
    for row in per_question:
        item = {k: row[k] for k in ['candidate_id', 'root_id', 'question_type', 'split', 'training_exposure', 'gold_reliability', 'historical_frontier_group', 'formal_feedback_seed']}
        for m in MODELS:
            a = row['models'][m]; item.update({m + '_planned_calls': a['planned_calls'], m + '_valid_calls': a['valid_calls'], m + '_exact': a['exact_rate_fixed'], m + '_ordering_pair': a['ordering']['pair_accuracy']['mean'], m + '_multiselect_J': a['multi_select']['jaccard']['mean']})
        for pair, value in row['paired_deltas'].items():
            for key, val in value.items():
                if key != 'direction': item[pair + '_' + key] = val
        flat.append(item)
    buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=list(dict.fromkeys(k for row in flat for k in row))); writer.writeheader(); writer.writerows(flat)
    (out / 'PER_QUESTION.csv').write_text(buffer.getvalue())
    def pct(v): return '未定义' if v is None else f'{v * 100:.2f}%'
    labels = {'historical_m0': '历史 M0（复用）', 'historical_clean': '历史 M_clean（复用）', 'live_clean': '当前环境 M_clean（桥接）', 'live_new': '当前环境本轮模型'}
    report = [f"历史 frontier 固定题池评估：{summary['status']}。", '',
              '固定123题来自58个Train root：62道四项排序、51道五选项多选、10道单选。筛选条件为历史M0完整作答全错、历史M_clean在重复完整作答中有时答对。此处“部分”指非零且未满分的整题成功率。', '',
              '本轮新增M_clean与本轮模型各1629次full_context调用，共3258次。两侧使用同一当前环境；历史M0与M_clean各1629次原始返回仅复用。没有新增no_context或TEST调用，没有更改题目或Gold。', '',
              '| 范围 | 题数 | 历史 M0 exact | 历史 M_clean exact | 当前环境 M_clean exact | 本轮模型 exact | 同环境变化 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, value in strata.items():
        rates = [value['models'][m]['equal_question']['exact_rate_equal_question'] for m in MODELS]
        stratum_labels = {'all123': '全部', 'historical_both93': '历史两个训练模型均部分答对', 'historical_clean_only30': '历史仅M_clean部分答对', 'formal_feedback_seeds10': '历史正式反馈种子（诊断）'}
        report.append('| ' + stratum_labels.get(name, name) + ' | ' + str(value['question_count']) + ' | ' + ' | '.join(pct(v) for v in rates) + ' | ' + f"{100 * (rates[3] - rates[2]):+.2f}个百分点 |")
    report += ['', '表中每题等权；多选每题15次、其他每题12次，调用等权结果另列于SUMMARY.json。', '',
               '| 模型 | 排序成对准确率 | 多选 Jaccard |', '|---|---:|---:|']
    for m in MODELS:
        eq = strata['all123']['models'][m]['equal_question']; report.append(f"| {labels[m]} | {pct(eq['ordering_pair_accuracy_equal_question']['mean'])} | {pct(eq['multi_select_jaccard_equal_question']['mean'])} |")
    report += ['', '主要对照是当前环境M_clean与本轮模型。历史M_clean与当前环境M_clean用于检查运行环境差异；历史M0没有在当前环境重跑，因此不能把跨环境变化全部归因于训练。', '',
               '题池按历史表现筛选，历史M0到M_clean的提升已部分由筛选条件决定，不能用于估计总体训练收益。这123道题未直接用于第一轮训练，但root属于Train，且后续训练可能使用相关root；此处是已筛选Train诊断，不能称为独立DEV/TEST泛化。', '',
               '本次保留全部历史Gold，未重新语义审核。旧题可能存在错误、歧义或证据不足，成绩表示与旧Gold一致程度，不能单独证明答案真值可靠。逐题均附相应标记，没有据此删题。' +
               (f" 历史审核适用性核对：{quality['summary']['historical_same_content_accept_count']}题有同内容旧审核accept，{quality['summary']['no_applicable_semantic_review_found_count']}题未找到可适用语义审核；尚未发现绑定该版本的明确无效Gold不等于其余Gold已验证。" if quality else ''), '',
               '所有计划调用保留固定分母；无效/缺失/重复exact记0，Jaccard与成对点估计未定义并给出[0,1]固定分母上下界。排列、重复调用和同root题目相关，不能将6516次比较调用当成6516道独立题。', '',
               f"完整性与评分错误数：{len(errors)}。逐题、逐调用、逐root及输入哈希见PER_QUESTION.csv/.json、PER_CALL.json、PER_ROOT.json、SUMMARY.json。"]
    (out / 'REPORT_ZH.md').write_text('\n'.join(report) + '\n')
    print(json.dumps({'status': summary['status'], 'errors': len(errors), 'output_dir': str(out), 'exact_equal_question': {m: strata['all123']['models'][m]['equal_question']['exact_rate_equal_question'] for m in MODELS}}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
