"""Paired full-source evaluation and separate reports for the two question types."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import hashlib
import re
import subprocess
import traceback

import batch
from common import HERE, NATIVE, PYTHON, aware, bind, load_multi, need, now, read, require_bindings, sha, write

MODELS = {'trained': 'M_clean_policy', 'base': 'M0'}
ALLOWED_MODEL_DIFFERENCES = {'prepared_at_utc', 'model', 'solver_model', 'solver_identity'}
CALLS = {'multi_select': 18, 'ordering': 12}


def native():
    return load_multi('evaluate').native()


def solver_dir(home, side):
    need(side in MODELS, 'unknown_model_side')
    return Path(home) / 'solver' / side


def execute(home, args, log_name):
    # Every subprocess, including preflight and report, rechecks the content gate.
    batch.verify_frozen(home)
    path = Path(home) / 'logs' / log_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        done = subprocess.run([str(PYTHON), '-B', str(NATIVE / 'run_solver.py'), *args],
                              cwd=home, stdout=stream, stderr=subprocess.STDOUT)
    need(done.returncode == 0, 'native_solver_failed:' + str(path))


def typed_counts(questions):
    counts = Counter(q['question_type'] for q in questions)
    need(set(counts) <= set(CALLS), 'unknown_question_type')
    return {kind: {'candidates': counts[kind], 'calls': counts[kind] * CALLS[kind]} for kind in CALLS}


def verify_request_plan(manifest, questions, n):
    """Replay Gold-blind variants, request IDs, typed slots and sequential seeds."""
    requests = manifest['requests']; expected = []; golden = {}
    for q in questions:
        variants, count = n._variants_for(q)
        need(count == (6 if q['question_type'] == 'multi_select' else 12), 'wrong_variant_count')
        field = 'options' if q['question_type'] == 'multi_select' else 'items'
        text_to_content = {text: cid for cid, text in variants[0]['content_to_text'].items()}
        golden[q['candidate_id']] = [text_to_content[q[field][label]] for label in q['gold']]
        for variant in variants:
            for slot in range(3 if field == 'options' else 1):
                expected.append({'request_id': q['candidate_id'] + '__' + variant['variant_id'] + f'__full_context__s{slot:02d}',
                                 'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
                                 'question_type': q['question_type'], 'variant': variant,
                                 'mode': 'full_context', 'slot': slot, 'seed': n.SEED + len(expected)})
    need(len(requests) == len(expected) == manifest['request_count'], 'wrong_typed_call_count')
    need(manifest['candidate_count'] == len(questions), 'wrong_candidate_count')
    for actual, frozen in zip(requests, expected):
        need(set(actual) == set(frozen) | {'prompt_sha256', 'prompt_tokens'}
             and all(actual[k] == v for k, v in frozen.items())
             and type(actual['slot']) is int and type(actual['seed']) is int,
             'wrong_typed_request_plan')
        need(type(actual['prompt_tokens']) is int and 0 < actual['prompt_tokens'] <= 16352,
             'prompt_capacity_not_proven')
    return golden


def paired_inputs(home=HERE):
    home = Path(home).resolve(); pool = read(home / 'EVALUATION_POOL.json'); questions = pool['questions']
    counts = typed_counts(questions)
    runs = [solver_dir(home, side) for side in MODELS]
    if not questions:
        for side, run in zip(MODELS, runs):
            skip = read(run / 'SKIPPED.json')
            need(skip['candidate_count'] == skip['actual_call_count'] == 0
                 and skip['solver_model'] == MODELS[side] and skip['training_started'] is False,
                 'empty_pool_missing_skip')
        return {'status': 'EMPTY_CONTENT_USABLE_POOL', 'candidate_count': 0, 'request_count': 0,
                'by_type': counts, 'identical_inputs': True}
    manifests = [read(run / 'manifest.json') for run in runs]; n = native()
    keep = lambda m: {k:v for k,v in m.items() if k not in ALLOWED_MODEL_DIFFERENCES}
    need(keep(manifests[0]) == keep(manifests[1]), 'paired_manifest_difference')
    for side, run, m in zip(MODELS, runs, manifests):
        need(m['status'] == 'PREFLIGHT_PASS' and m['solver_model'] == MODELS[side]
             and m['model'] == str(n.MODEL_PATHS[MODELS[side]]) and m['tokenizer'] == str(n.TOKENIZER)
             and m['policy_version'] == n.POLICY_VERSION and m['execution'] == 'full_only_no_ground_v1'
             and m['training_started'] is False and m['training_promotion'] is False,
             'native_model_or_policy_changed')
        need(Path(m['pool_path']).resolve() == home / 'EVALUATION_POOL.json'
             and m['pool_sha256'] == m['input_sha256']['pool'] == sha(home / 'EVALUATION_POOL.json'),
             'native_pool_binding')
        need(m['random_display'] == {'version': 'independent_seeded_permutation_v2', 'seed': n.SEED,
                                    'gold_blind': True, 'cyclic_rebinding': False, 'shape_filter': False},
             'nonblind_or_changed_permutation')
        for key, value in {'temperature': 0.8, 'top_p': 0.9, 'max_tokens': 32,
                           'enable_thinking': False, 'structured_output_version': 'finite_choice_set_v1'}.items():
            need(m['decoder_contract'][key] == value, 'native_decoder_changed:' + key)
        for path, expected in m['code_sha256'].items():
            need(sha(path) == expected, 'native_code_changed:' + path)
        for name, expected in m['input_sha256'].items():
            path = home / 'EVALUATION_POOL.json' if name == 'pool' else run / name
            need(sha(path) == expected, 'solver_input_changed:' + str(path))
        golden = verify_request_plan(m, questions, n)
        need(read(run / 'gold.json') == golden, 'native_gold_content_binding')
        need(read(run / 'questions.json')['candidates'] ==
             [{k:v for k,v in q.items() if k not in ('gold', 'gold_content_ids')} for q in questions],
             'native_questions_changed')
        need(read(run / 'solver_identity.json') == m['solver_identity'], 'native_identity_binding')
        for request in m['requests']:
            text = (run / 'prompts' / (request['request_id'] + '.txt')).read_text()
            need(hashlib.sha256(text.encode()).hexdigest() == request['prompt_sha256'], 'solver_prompt_changed')
    return {'status': 'PASS', 'candidate_count': len(questions), 'request_count': sum(v['calls'] for v in counts.values()),
            'by_type': counts, 'identical_inputs': True, 'same_prompts_permutations_seeds_decoder_scoring': True,
            'allowed_manifest_differences': sorted(ALLOWED_MODEL_DIFFERENCES)}


def prepare(home=HERE):
    home = Path(home).resolve(); review = batch.verify_frozen(home)
    need(not (home / 'EVALUATION_PREPARATION.STARTED.json').exists(), 'evaluation_prepare_no_replay')
    need(all(not solver_dir(home, side).exists() for side in MODELS), 'evaluation_run_already_exists')
    questions = read(home / 'EVALUATION_POOL.json')['questions']
    write(home / 'EVALUATION_PREPARATION.STARTED.json', {'started_at_utc': now(), 'review': bind(home / 'BATCH_REVIEW.COMPLETE.json')})
    try:
        for side, model in MODELS.items():
            run = solver_dir(home, side)
            if questions:
                execute(home, ['preflight', '--pool', str(home / 'EVALUATION_POOL.json'), '--run', str(run),
                               '--solver-model', model], side + '_preflight.log')
            else:
                write(run / 'SKIPPED.json', {'candidate_count': 0, 'actual_call_count': 0,
                                           'solver_model': model, 'training_started': False})
        paired = paired_inputs(home)
        need(paired['request_count'] <= batch.config(home)['max_solver_calls_per_model'], 'mixed_solver_budget')
        paths = [home / name for name in ('CONFIG.json', 'EVALUATION_POOL.json', 'BATCH_REVIEW.COMPLETE.json')]
        paths += list(HERE.glob('*.py'))
        paths += [NATIVE / name for name in ('run_solver.py', 'protocol.py', 'scoring.py', 'selection.py')]
        for side in MODELS:
            paths += [p for p in solver_dir(home, side).rglob('*') if p.is_file()]
        result = {'status': 'PASS', 'prepared_at_utc': now(), 'candidate_count': len(questions),
                  'calls_per_model': paired['request_count'], 'paired_inputs': paired,
                  'review_frozen_at_utc': review['completed_at_utc'], 'training_started': False,
                  'automatic_rerun': False, 'files': [bind(p) for p in sorted(set(paths))]}
        write(home / 'EVALUATION_PREPARATION.COMPLETE.json', result); return result
    except BaseException as error:
        write(home / 'EVALUATION_PREPARATION.ERROR.json', {'time': now(), 'error': repr(error), 'automatic_rerun': False})
        raise


def verify_preparation(home=HERE):
    home = Path(home).resolve(); review = batch.verify_frozen(home)
    prepared = read(home / 'EVALUATION_PREPARATION.COMPLETE.json')
    need(prepared['status'] == 'PASS' and prepared['training_started'] is False
         and prepared['automatic_rerun'] is False and not (home / 'EVALUATION_PREPARATION.ERROR.json').exists(),
         'evaluation_preparation_incomplete')
    paths = [home / name for name in ('CONFIG.json', 'EVALUATION_POOL.json', 'BATCH_REVIEW.COMPLETE.json')]
    paths += list(HERE.glob('*.py'))
    paths += [NATIVE / name for name in ('run_solver.py', 'protocol.py', 'scoring.py', 'selection.py')]
    for side in MODELS:
        run = solver_dir(home, side)
        if (run / 'SKIPPED.json').exists():
            paths.append(run / 'SKIPPED.json')
        else:
            m = read(run / 'manifest.json'); paths += [run / 'manifest.json', run / 'solver_identity.json']
            paths += [run / name for name in m['input_sha256'] if name != 'pool']
            paths += [run / 'prompts' / (r['request_id'] + '.txt') for r in m['requests']]
    require_bindings(prepared['files'], paths); paired = paired_inputs(home)
    need(prepared['paired_inputs'] == paired and prepared['candidate_count'] == paired['candidate_count']
         and prepared['calls_per_model'] == paired['request_count'] <= batch.config(home)['max_solver_calls_per_model']
         and prepared['review_frozen_at_utc'] == review['completed_at_utc']
         and aware(review['completed_at_utc']) <= aware(prepared['prepared_at_utc']) <= aware(now()),
         'evaluation_preparation_binding_or_time')
    return prepared


def independent_score(raw, request, gold):
    need(isinstance(raw, str), 'invalid_answer_text')
    match = re.fullmatch(r'\s*<box>\s*(.*?)\s*</box>\s*', raw, re.S)
    need(match is not None, 'invalid_answer_box')
    values = [x.strip().upper() for x in match.group(1).split(',')]
    kind = request['question_type']; need(kind in CALLS, 'invalid_scoring_type')
    labels = set('ABCDEF') if kind == 'multi_select' else {f'I{i}' for i in range(1, 7)}
    mapping = request['variant']['display_to_content']
    need(set(mapping) == labels and len(set(mapping.values())) == 6, 'invalid_content_mapping')
    need(values and len(values) == len(set(values)) and set(values) <= labels, 'invalid_answer_labels')
    prediction = [mapping[v] for v in values]
    need(2 <= len(gold) <= 6 and len(set(gold)) == len(gold) and set(gold) <= set(mapping.values()), 'invalid_content_gold')
    if kind == 'multi_select':
        selected, supported = set(prediction), set(gold)
        tp, fp, fn = len(selected & supported), len(selected - supported), len(supported - selected)
        partial = Fraction(tp, len(gold)) - (Fraction(fp, 6-len(gold)) if len(gold) < 6 else 0)
        exact = selected == supported
    else:
        need(len(gold) == len(prediction) == 6 and set(prediction) == set(gold), 'incomplete_order')
        rank = {item:i for i,item in enumerate(prediction)}
        fp = sum(rank[a] > rank[b] for i,a in enumerate(gold) for b in gold[i+1:])
        tp, fn = 15-fp, 0; partial = Fraction(tp-fp, 15); exact = prediction == gold
    return {'exact': exact, 'partial': str(partial), 'tp': tp, 'fp': fp, 'fn': fn,
            'selected_count': len(prediction), 'gold_count': len(gold)}


def replay_side(home, side):
    home = Path(home).resolve(); run = solver_dir(home, side)
    if (run / 'SKIPPED.json').exists():
        need(not read(home / 'EVALUATION_POOL.json')['questions'] and not list((run / 'responses').glob('*.json')), 'bad_empty_skip')
        return {'status': 'EMPTY_CONTENT_USABLE_POOL', 'side': side, 'calls': 0, 'questions': [], 'responses': []}
    n = native(); m = read(run / 'manifest.json')
    need((run / 'COMPLETE.json').exists() and not (run / 'FAILED.json').exists(), 'solver_incomplete_or_failed')
    start = read(run / 'STARTED.json'); frozen = read(home / 'BATCH_REVIEW.COMPLETE.json')
    need(start['manifest_sha256'] == sha(run / 'manifest.json')
         and aware(frozen['completed_at_utc']) < aware(start['started_at_utc']) <= aware(now()), 'solver_started_before_freeze_or_changed')
    records = [read(p) for p in sorted((run / 'responses').glob('*.json'))]
    need(not n.audit_complete_call_set(m, records), 'native_call_set_invalid')
    need(len(records) == read(run / 'COMPLETE.json')['actual_call_count'] == m['request_count'], 'actual_call_total_wrong')
    requests = {r['request_id']:r for r in m['requests']}
    canonical = {q['candidate_id']:q for q in read(run / 'questions.json')['candidates']}
    gold = read(run / 'gold.json'); replayed = []
    for record in records:
        rid, qid = record['request_id'], record['candidate_id']; request = requests[rid]
        expected = n.score_response({**canonical[qid], 'gold_content_ids': gold[qid]}, request, record['raw_output'])
        need(expected == record['score'], 'native_score_replay_mismatch:' + rid)
        score = independent_score(record['raw_output'], request, gold[qid])
        need(score['exact'] == expected['exact_match'] and score['partial'] == expected['partial_score_fraction']
             and all(score[k] == expected[k] for k in ('tp', 'fp', 'fn')), 'independent_score_disagrees')
        need(record['solver_model'] == m['solver_model'] and record['model_path'] == m['model'], 'response_model_wrong')
        metadata = record['metadata']
        need(record.get('data_errors') == [] and metadata.get('finish_reason') in ('stop', 'eos'), 'response_incomplete_or_data_errors')
        need(record['seed'] == metadata['sampling_seed'] == request['seed']
             and metadata['prompt_token_count'] == request['prompt_tokens'], 'response_seed_or_prompt_tokens_wrong')
        for key in ('temperature', 'top_p', 'max_tokens', 'structured_output_api', 'structured_output_version'):
            need(metadata[key] == m['decoder_contract'][key], 'response_decoder_wrong:' + key)
        replayed.append({'request_id': rid, 'candidate_id': qid, 'root_id': request['root_id'],
                         'question_type': request['question_type'], 'variant_id': request['variant']['variant_id'],
                         'slot': request['slot'], 'seed': request['seed'], **score,
                         'response': bind(run / 'responses' / (rid + '.json'))})
    results = read(run / 'RESULTS.json')
    need(not results['call_set_errors'] and results['solver_model'] == m['solver_model'], 'invalid_native_results')
    summaries = []
    for qid, q in canonical.items():
        rows = [r for r in records if r['candidate_id'] == qid]; kind = q['question_type']; count = CALLS[kind]
        variants = [f'perm_{i:02d}' for i in range(6 if kind == 'multi_select' else 12)]
        need(len(rows) == count and Counter(r['variant_id'] for r in rows) == Counter({v: 3 if kind == 'multi_select' else 1 for v in variants}),
             'wrong_per_question_call_set')
        scores = {v:[Fraction(r['score']['partial_score_fraction']) for r in rows if r['variant_id'] == v] for v in variants}
        expected = n.select(scores, variants, exact_rate=Fraction(sum(r['score']['exact_match'] for r in rows), count), data_errors=[])
        expected.update(candidate_id=qid, root_id=q['root_id'], question_type=kind, expected_full_calls=count, actual_full_calls=count)
        need([r for r in results['questions'] if r['candidate_id'] == qid] == [expected], 'native_selection_replay_mismatch:' + qid)
        summaries.append(expected)
    need(len(results['questions']) == len(summaries)
         and results['selected_questions'] == [r['candidate_id'] for r in results['questions'] if r['selected_by_policy']],
         'native_selected_ids_mismatch')
    return {'status': 'PASS', 'side': side, 'calls': len(records), 'questions': summaries, 'responses': replayed,
            'manifest': bind(run / 'manifest.json'), 'complete': bind(run / 'COMPLETE.json'), 'results': bind(run / 'RESULTS.json')}


def live(home=HERE, side='trained'):
    home = Path(home).resolve(); need(side in MODELS, 'unknown_side'); prepared = verify_preparation(home)
    if side == 'base':
        complete = read(home / 'TRAINED_EVALUATION.COMPLETE.json')
        need(complete['status'] == 'COMPLETE' and complete['calls'] == prepared['calls_per_model']
             and not (home / 'TRAINED_EVALUATION.ERROR.json').exists(), 'complete_trained_before_base')
        replay_side(home, 'trained')
    marker = side.upper() + '_EVALUATION'
    need(not (home / (marker + '.STARTED.json')).exists(), 'already_started_no_automatic_replay')
    write(home / (marker + '.STARTED.json'), {'time': now(), 'preparation': bind(home / 'EVALUATION_PREPARATION.COMPLETE.json'),
                                           'calls_planned': prepared['calls_per_model'], 'training_started': False})
    try:
        if prepared['candidate_count']:
            execute(home, ['live', '--run', str(solver_dir(home, side))], side + '_live.log')
            execute(home, ['report', '--run', str(solver_dir(home, side))], side + '_report.log')
        replay = replay_side(home, side)
        write(home / (marker + '.COMPLETE.json'), {'time': now(), 'status': 'COMPLETE', 'calls': replay['calls'],
                                                'raw_scores_and_selection_replayed': True, 'training_started': False})
        return replay
    except BaseException as error:
        write(home / (marker + '.ERROR.json'), {'time': now(), 'error': repr(error), 'traceback': traceback.format_exc(), 'automatic_rerun': False})
        raise


def accuracy(rows):
    count = len(rows)
    return {'calls': count, 'exact_correct': sum(r['exact'] for r in rows),
            'exact_accuracy': str(Fraction(sum(r['exact'] for r in rows), count)) if count else None,
            'partial_mean': str(sum((Fraction(r['partial']) for r in rows), Fraction()) / count) if count else None}


def report(home=HERE):
    home = Path(home).resolve(); prepared = verify_preparation(home)
    models = {side: replay_side(home, side) for side in MODELS}
    for side, replay in models.items():
        complete = read(home / (side.upper() + '_EVALUATION.COMPLETE.json'))
        need(complete['status'] == 'COMPLETE' and complete['calls'] == replay['calls'] == prepared['calls_per_model']
             and complete['training_started'] is False and not (home / (side.upper() + '_EVALUATION.ERROR.json')).exists(),
             'evaluation_stage_not_complete')
    pool = read(home / 'EVALUATION_POOL.json'); by_id = {q['candidate_id']:q for q in pool['questions']}
    selected_ids = [q['candidate_id'] for q in models['trained']['questions'] if q['selected_by_policy']]
    stats = {}
    multi_review = read(home / 'multiselect/FINAL_CONTENT_REVIEW.COMPLETE.json')
    ordering_review = read(home / 'ordering/CONTENT_REVIEW.json')
    config = batch.config(home)
    for kind in CALLS:
        typed = [q for q in by_id.values() if q['question_type'] == kind]
        selected = [q for q in typed if q['candidate_id'] in selected_ids]
        generated = multi_review['candidate_count'] if kind == 'multi_select' else ordering_review['accepted_count'] + ordering_review['rejected_count']
        stats[kind] = {'candidate_capacity': config['multiselect_config']['max_questions'] if kind == 'multi_select' else sum(s['candidate_count'] for s in config['ordering_slots']),
                       'mechanically_assembled': generated, 'content_usable': len(typed),
                       'content_excluded': generated-len(typed), 'solver_selected': len(selected),
                       'selection_yield_among_content_usable': str(Fraction(len(selected), len(typed))) if typed else None,
                       'models': {side: accuracy([r for r in replay['responses'] if r['question_type'] == kind]) for side, replay in models.items()}}
        if kind == 'multi_select':
            stats[kind]['by_gold_count'] = {str(g): {'content_usable': sum(len(q['gold']) == g for q in typed),
                                                    'solver_selected': sum(len(q['gold']) == g for q in selected),
                                                    'role': 'historical_six_supported_control' if g == 6 else 'new_child',
                                                    'models': {side: accuracy([r for r in replay['responses'] if r['question_type'] == kind and len(by_id[r['candidate_id']]['gold']) == g]) for side, replay in models.items()}}
                                               for g in range(2, 7)}
    left = {r['request_id']:r for r in models['trained']['responses']}; right = {r['request_id']:r for r in models['base']['responses']}
    need(set(left) == set(right), 'paired_response_set_changed')
    pairs = []
    for rid, trained in left.items():
        base = right[rid]
        identity = {k: trained[k] for k in ('request_id', 'candidate_id', 'root_id', 'question_type', 'variant_id', 'slot', 'seed')}
        need(all(base[k] == v for k,v in identity.items()), 'paired_response_identity_changed')
        pairs.append({**identity, 'trained': trained, 'base': base})
    selected = [q for q in pool['questions'] if q['candidate_id'] in selected_ids]
    paths = [home / 'EVALUATION_PREPARATION.COMPLETE.json', home / 'BATCH_REVIEW.COMPLETE.json', home / 'EVALUATION_POOL.json']
    for side, replay in models.items():
        paths.append(home / (side.upper() + '_EVALUATION.COMPLETE.json'))
        paths.extend(Path(replay[k]['path']) for k in ('manifest', 'complete', 'results') if k in replay)
        paths.extend(Path(r['response']['path']) for r in replay['responses'])
    result = {'status': 'PASS', 'evaluation_status': 'COMPLETED' if by_id else 'SKIPPED_EMPTY_CONTENT_USABLE_POOL',
              'completed_at_utc': now(), 'calls_per_model': prepared['calls_per_model'], 'by_type': stats,
              'selected_candidate_ids': selected_ids, 'selected_new_candidate_count': sum(q['question_type'] == 'ordering' or len(q['gold']) < 6 for q in selected),
              'paired_inputs': prepared['paired_inputs'], 'content_frozen_before_any_solver': True,
              'raw_scores_and_selection_replayed': True, 'training_started': False, 'formal_training_admission': False,
              'independent_result_audit_required_before_admission': True, 'files': [bind(p) for p in sorted(set(paths))]}
    write(home / 'PAIRED_CALLS.json', {'calls': pairs, 'training_started': False})
    write(home / 'SELECTED_CANDIDATES.json', {'questions': selected, 'provisional': True,
                                           'training_allowed': False, 'formal_training_admission': False})
    result['files'] += [bind(home / 'PAIRED_CALLS.json'), bind(home / 'SELECTED_CANDIDATES.json')]
    write(home / 'MIXED_EVALUATION_REPORT.json', result); return result
