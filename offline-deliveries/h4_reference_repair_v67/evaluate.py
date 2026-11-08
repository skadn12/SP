"""Freeze final content review and evaluate one paired, full-context v67 pool."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / 'h4_feedback_training_pool_v15_qa/evaluation'
RUNNER = NATIVE / 'run_solver.py'
PYTHON = Path('/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python')
MODELS = {'trained': 'M_clean_policy', 'base': 'M0'}
ALLOWED_MODEL_DIFFERENCES = {'prepared_at_utc', 'model', 'solver_model', 'solver_identity'}


def require(value, why):
    if not value:
        raise ValueError(why)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def verify_binding(row):
    require(sha(row['path']) == row['sha256'], 'file_changed:' + row['path'])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def native():
    key = 'v67_native_solver_readonly'
    if key not in sys.modules:
        sys.path.insert(0, str(NATIVE))
        spec = importlib.util.spec_from_file_location(key, RUNNER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def verify_assembly(home):
    import compose
    return compose.verify_assembly(Path(home))


def solver_dir(home, side):
    require(side in MODELS, 'unknown_model_side')
    return Path(home) / 'evaluation' / side


def no_started_solver(home):
    for side in MODELS:
        run = solver_dir(home, side)
        require(not (run / 'STARTED.json').exists() and not list((run / 'responses').glob('*.json')),
                'content_review_cannot_change_after_solver')


def reviewed_content(home):
    home = Path(home)
    pool = verify_assembly(home)
    config = read(home / 'CONFIG.json')
    require(config['training_allowed'] is False and config['max_questions'] <= 10
            and config['max_calls_per_model'] <= 180, 'evaluation_budget_or_training_changed')
    questions = pool['questions']
    require(len(questions) <= config['max_questions'], 'too_many_assembled_questions')
    require(pool.get('training_allowed') is False, 'assembled_pool_training_enabled')
    by_id = {q['candidate_id']: q for q in questions}
    require(len(by_id) == len(questions), 'duplicate_assembled_id')
    review = read(home / 'FINAL_CONTENT_REVIEW.json')
    require(review.get('solver_scores_used') is False, 'content_review_used_solver_scores')
    require(review.get('pool_sha256') == sha(home / 'POOL.json'), 'final_review_pool_binding')
    rows = review.get('questions', [])
    require(len(rows) == len(questions) and {r['candidate_id'] for r in rows} == set(by_id),
            'final_review_must_cover_every_assembled_question')
    roots = {slot['root_id']: Path(slot['history_path']) for slot in config['slots']}
    statuses = {}
    for row in rows:
        q = by_id[row['candidate_id']]
        require(row.get('core_status') in ('CORE_USABLE', 'KNOWN_SUBSTANTIVE_ISSUE'), 'unknown_content_status')
        require(row.get('root_id') == q['root_id'] and row.get('source_sha256') == q['source_sha256'],
                'final_review_source_binding')
        require(row.get('pool_sha256', review['pool_sha256']) == review['pool_sha256'], 'row_pool_binding')
        require(q['question_type'] == 'multi_select' and q['root_id'] in roots, 'unexpected_question_type_or_root')
        source = read(q['source_path'])
        require(sha(q['source_path']) == q['source_sha256'] and source['turns'] == read(roots[q['root_id']])['turns'],
                'evaluation_requires_complete_frozen_history')
        statuses[q['candidate_id']] = row['core_status']
    usable = [q for q in questions if statuses[q['candidate_id']] == 'CORE_USABLE']
    return pool, usable, statuses, {'pool': bind(home / 'POOL.json'), 'review': bind(home / 'FINAL_CONTENT_REVIEW.json'),
        'config': bind(home / 'CONFIG.json'), 'assembly_receipts': bind(home / 'ASSEMBLY_RECEIPTS.json')}


def freeze_review(home=HERE):
    home = Path(home)
    no_started_solver(home)
    _, usable, statuses, bindings = reviewed_content(home)
    result = {'status': 'FROZEN', 'completed_at_utc': now(), 'solver_scores_used': False,
              'before_any_solver': True, 'candidate_count': len(statuses), 'usable_count': len(usable),
              'usable_candidate_ids': [q['candidate_id'] for q in usable], 'statuses': statuses, **bindings}
    write(home / 'FINAL_CONTENT_REVIEW.COMPLETE.json', result)
    return result


def verify_review(home=HERE):
    home = Path(home)
    frozen = read(home / 'FINAL_CONTENT_REVIEW.COMPLETE.json')
    _, usable, statuses, bindings = reviewed_content(home)
    require(frozen.get('before_any_solver') is True and frozen.get('solver_scores_used') is False, 'review_scope_changed')
    require(all(frozen.get(key) == value for key, value in bindings.items()) and frozen['statuses'] == statuses
            and frozen['usable_candidate_ids'] == [q['candidate_id'] for q in usable], 'frozen_review_changed')
    return frozen


def execute(home, args, log_name):
    home = Path(home)
    path = home / 'logs' / log_name
    path.parent.mkdir(exist_ok=True)
    with path.open('x') as stream:
        done = subprocess.run([str(PYTHON), '-B', str(RUNNER), *args], cwd=home,
                              stdout=stream, stderr=subprocess.STDOUT)
    require(done.returncode == 0, 'native_solver_failed:' + str(path))


def prepare(home=HERE):
    home = Path(home)
    review = verify_review(home)
    no_started_solver(home)
    pool, usable, _, _ = reviewed_content(home)
    require(not (home / 'EVALUATION_PREPARATION.STARTED.json').exists(), 'evaluation_prepare_already_started')
    require(all(not solver_dir(home, side).exists() for side in MODELS), 'evaluation_run_already_exists')
    write(home / 'EVALUATION_PREPARATION.STARTED.json', {'time': now(), 'review': bind(home / 'FINAL_CONTENT_REVIEW.COMPLETE.json')})
    target = home / 'EVALUATION_POOL.json'
    write(target, {'questions': usable, 'candidate_count': len(usable), 'provenance': pool['provenance'],
        'source_pool': bind(home / 'POOL.json'), 'content_review': bind(home / 'FINAL_CONTENT_REVIEW.COMPLETE.json'),
        'training_allowed': False, 'training_started': False, 'formal_training_admission': False})
    for side, model in MODELS.items():
        run = solver_dir(home, side)
        if usable:
            execute(home, ['preflight', '--pool', str(target), '--run', str(run), '--solver-model', model], side + '_preflight.log')
        else:
            write(run / 'SKIPPED.json', {'status': 'EMPTY_CONTENT_USABLE_POOL', 'candidate_count': 0,
                'solver_model': model, 'actual_call_count': 0, 'training_started': False})
    paired = paired_inputs(home)
    require(paired['request_count'] == 18 * len(usable) <= read(home / 'CONFIG.json')['max_calls_per_model'], 'paired_request_budget')
    inputs = [home / name for name in ('CONFIG.json', 'POOL.json', 'ALLOCATION.json', 'ASSEMBLY_RECEIPTS.json',
        'FINAL_CONTENT_REVIEW.json', 'FINAL_CONTENT_REVIEW.COMPLETE.json', 'EVALUATION_POOL.json')]
    inputs += sorted(home.glob('*.py'))
    inputs += [NATIVE / name for name in ('run_solver.py', 'protocol.py', 'scoring.py', 'selection.py')]
    inputs += [HERE.parent / 'h4_option_composition_v63_qa/display_export.py']
    for side in MODELS:
        run = solver_dir(home, side)
        inputs.extend(path for path in run.rglob('*') if path.is_file())
    result = {'status': 'PASS', 'prepared_at_utc': now(), 'candidate_count': len(usable),
        'calls_per_model': paired['request_count'], 'evaluation_status': 'READY' if usable else 'SKIPPED_EMPTY_CONTENT_USABLE_POOL', 'paired_inputs': paired, 'review_frozen_at_utc': review['completed_at_utc'],
        'files': [bind(path) for path in sorted(set(inputs))], 'training_started': False, 'automatic_rerun': False}
    write(home / 'EVALUATION_PREPARATION.COMPLETE.json', result)
    return result


def paired_inputs(home=HERE):
    home = Path(home)
    trained, base = (solver_dir(home, side) for side in ('trained', 'base'))
    pool = read(home / 'EVALUATION_POOL.json')
    if not pool['questions']:
        require(all(read(run / 'SKIPPED.json')['candidate_count'] == 0 for run in (trained, base)), 'empty_pool_missing_skip')
        return {'status': 'EMPTY_CONTENT_USABLE_POOL', 'candidate_count': 0, 'request_count': 0, 'identical_inputs': True}
    manifests = [read(run / 'manifest.json') for run in (trained, base)]
    left, right = manifests
    require(left['solver_model'] == 'M_clean_policy' and right['solver_model'] == 'M0', 'model_identities_wrong')
    keep = lambda m: {k:v for k,v in m.items() if k not in ALLOWED_MODEL_DIFFERENCES}
    require(keep(left) == keep(right), 'paired_manifest_difference')
    require(left['candidate_count'] == len(pool['questions']) and left['request_count'] == len(pool['questions']) * 18,
            'wrong_native_call_count')
    n = native()
    for run, manifest in zip((trained, base), manifests):
        require(manifest['model'] == str(n.MODEL_PATHS[manifest['solver_model']]) and
                manifest['policy_version'] == n.POLICY_VERSION and manifest['execution'] == 'full_only_no_ground_v1', 'native_identity_or_policy_changed')
        require(manifest['training_started'] is False and manifest['training_promotion'] is False, 'training_started')
        for path, expected in manifest['code_sha256'].items():
            require(sha(path) == expected, 'native_code_changed:' + path)
        for relative, expected in manifest['input_sha256'].items():
            path = Path(manifest['pool_path']) if relative == 'pool' else run / relative
            require(sha(path) == expected, 'solver_input_changed:' + str(path))
        for request in manifest['requests']:
            text = (run / 'prompts' / (request['request_id'] + '.txt')).read_text()
            require(hashlib.sha256(text.encode()).hexdigest() == request['prompt_sha256'], 'solver_prompt_changed')
            require(request['mode'] == 'full_context' and request['question_type'] == 'multi_select', 'unexpected_eval_scope')
        # Native preflight constructs each prompt twice, once from public fields only.
        require(manifest['random_display']['gold_blind'] is True, 'nonblind_permutation')
    return {'status': 'PASS', 'candidate_count': left['candidate_count'], 'request_count': left['request_count'],
            'identical_inputs': True, 'same_prompts_permutations_seeds_decoder_scoring': True,
            'allowed_manifest_differences': sorted(ALLOWED_MODEL_DIFFERENCES)}


def verify_preparation(home=HERE):
    home = Path(home)
    verify_review(home)
    prepared = read(home / 'EVALUATION_PREPARATION.COMPLETE.json')
    for row in prepared['files']:
        verify_binding(row)
    require(prepared['paired_inputs'] == paired_inputs(home), 'pairing_changed_after_prepare')
    require(prepared['calls_per_model'] <= read(home / 'CONFIG.json')['max_calls_per_model'], 'evaluation_budget_changed')
    return prepared


def independent_score(raw_output, request, gold):
    match = re.fullmatch(r'\s*<box>\s*(.*?)\s*</box>\s*', raw_output, re.S)
    require(match is not None, 'invalid_answer_box')
    values = [x.strip().upper() for x in match.group(1).split(',')]
    require(len(values) == len(set(values)) and set(values) <= set('ABCDEF') and values, 'invalid_answer_labels')
    selected = {request['variant']['display_to_content'][x] for x in values}
    supported = set(gold)
    require(2 <= len(supported) <= 6, 'invalid_gold_count')
    tp, fp, fn = len(selected & supported), len(selected - supported), len(supported - selected)
    partial = Fraction(tp, len(supported)) - (Fraction(fp, 6-len(supported)) if len(supported)<6 else 0)
    return {'exact': selected == supported, 'partial': str(partial), 'tp': tp, 'fp': fp, 'fn': fn,
            'selected_count': len(selected), 'gold_count': len(supported)}


def replay_side(home, side):
    home, run = Path(home), solver_dir(home, side)
    if (run / 'SKIPPED.json').exists():
        require(read(run / 'SKIPPED.json')['candidate_count'] == 0 and not list((run / 'responses').glob('*.json')), 'bad_empty_skip')
        return {'status': 'EMPTY_CONTENT_USABLE_POOL', 'side': side, 'calls': 0, 'questions': [], 'responses': []}
    n = native()
    manifest = read(run / 'manifest.json')
    require((run / 'COMPLETE.json').exists() and not (run / 'FAILED.json').exists(), 'solver_incomplete_or_failed')
    frozen = read(home / 'FINAL_CONTENT_REVIEW.COMPLETE.json')
    start = read(run / 'STARTED.json')
    require(start['manifest_sha256'] == sha(run / 'manifest.json'), 'started_manifest_changed')
    require(datetime.fromisoformat(start['started_at_utc']) > datetime.fromisoformat(frozen['completed_at_utc']), 'solver_started_before_content_freeze')
    records = [read(p) for p in sorted((run / 'responses').glob('*.json'))]
    require(not n.audit_complete_call_set(manifest, records), 'native_call_set_invalid')
    require(len(records) == read(run / 'COMPLETE.json')['actual_call_count'] == manifest['request_count'], 'actual_call_total_wrong')
    by_request = {r['request_id']:r for r in manifest['requests']}
    canonical = {q['candidate_id']:q for q in read(run / 'questions.json')['candidates']}
    gold = read(run / 'gold.json')
    replayed = []
    for record in records:
        rid, qid = record['request_id'], record['candidate_id']
        request = by_request[rid]
        q = {**canonical[qid], 'gold_content_ids':gold[qid]}
        expected = n.score_response(q, request, record['raw_output'])
        require(expected == record['score'], 'native_score_replay_mismatch:' + rid)
        score = independent_score(record['raw_output'], request, gold[qid])
        require(score['exact'] == expected['exact_match'] and score['partial'] == expected['partial_score_fraction']
                and all(score[k] == expected[k] for k in ('tp','fp','fn')), 'independent_score_disagrees')
        require(record['solver_model'] == manifest['solver_model'] and record['model_path'] == manifest['model'], 'response_model_wrong')
        require(record.get('data_errors') == [] and record['metadata'].get('finish_reason') in ('stop','eos'),
                'response_incomplete_or_data_errors')
        require(record['seed'] == record['metadata']['sampling_seed'] == request['seed'], 'response_seed_wrong')
        require(record['metadata']['prompt_token_count'] == request['prompt_tokens'], 'response_prompt_tokens_wrong')
        for key in ('temperature','top_p','max_tokens','structured_output_api','structured_output_version'):
            require(record['metadata'][key] == manifest['decoder_contract'][key], 'response_decoder_wrong:' + key)
        replayed.append({'request_id':rid,'candidate_id':qid,'root_id':request['root_id'],
                         **score,'response':bind(run / 'responses' / (rid + '.json'))})
    results = read(run / 'RESULTS.json')
    require(not results['call_set_errors'] and results['solver_model'] == manifest['solver_model'], 'invalid_native_results')
    summaries = []
    for qid, q in canonical.items():
        rows = [r for r in records if r['candidate_id']==qid]
        variants = [f'perm_{i:02d}' for i in range(6)]
        require(len(rows)==18 and Counter(r['variant_id'] for r in rows)==Counter({v:3 for v in variants}), 'wrong_per_question_call_set')
        scores={v:[Fraction(r['score']['partial_score_fraction']) for r in rows if r['variant_id']==v] for v in variants}
        expected=n.select(scores,variants,exact_rate=Fraction(sum(r['score']['exact_match'] for r in rows),18),data_errors=[])
        expected.update(candidate_id=qid,root_id=q['root_id'],question_type='multi_select',expected_full_calls=18,actual_full_calls=18)
        matches=[r for r in results['questions'] if r['candidate_id']==qid]
        require(matches==[expected], 'native_aggregate_or_selection_replay_mismatch:' + qid)
        summaries.append(expected)
    require(len(results['questions'])==len(summaries) and results['selected_questions']==[r['candidate_id'] for r in results['questions'] if r['selected_by_policy']], 'native_selected_ids_mismatch')
    return {'status':'PASS','side':side,'calls':len(records),'questions':summaries,'responses':replayed,
            'manifest':bind(run / 'manifest.json'),'complete':bind(run / 'COMPLETE.json'),'results':bind(run / 'RESULTS.json')}


def live(home=HERE, side='trained'):
    home = Path(home)
    prepared = verify_preparation(home)
    require(side in MODELS, 'unknown_side')
    if side == 'base':
        require((home / 'TRAINED_EVALUATION.COMPLETE.json').exists(), 'complete_trained_before_base')
    marker = side.upper() + '_EVALUATION'
    require(not (home / (marker + '.STARTED.json')).exists(), 'already_started_no_automatic_replay')
    write(home / (marker + '.STARTED.json'), {'time':now(),'preparation':bind(home / 'EVALUATION_PREPARATION.COMPLETE.json'),
          'calls_planned':prepared['calls_per_model'],'training_started':False})
    try:
        if prepared['candidate_count']:
            execute(home,['live','--run',str(solver_dir(home,side))],side+'_live.log')
            execute(home,['report','--run',str(solver_dir(home,side))],side+'_report.log')
        replay = replay_side(home,side)
        write(home / (marker + '.COMPLETE.json'), {'time':now(),'status':'COMPLETE','calls':replay['calls'],
              'raw_scores_and_selection_replayed':True,'training_started':False})
        return replay
    except BaseException as error:
        write(home / (marker + '.ERROR.json'),{'time':now(),'error':repr(error),'traceback':traceback.format_exc(),
              'automatic_rerun':False})
        raise


def verify(home=HERE):
    home = Path(home)
    prepared = verify_preparation(home)
    sides = {side:replay_side(home,side) for side in MODELS}
    require(sides['trained']['calls']==sides['base']['calls']==prepared['calls_per_model'], 'paired_call_total_mismatch')
    for side in MODELS:
        complete=read(home/(side.upper()+'_EVALUATION.COMPLETE.json'))
        require(complete['calls']==sides[side]['calls'] and not (home/(side.upper()+'_EVALUATION.ERROR.json')).exists(), 'evaluation_stage_not_complete')
    return {'status':'PASS','evaluation_status':'COMPLETED' if prepared['candidate_count'] else 'SKIPPED_EMPTY_CONTENT_USABLE_POOL',
            'calls_per_model':prepared['calls_per_model'],'paired_inputs':paired_inputs(home),
            'content_frozen_before_any_solver':True,'training_started':False,'models':sides}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze-review','prepare','live-trained','live-base','verify'))
    parser.add_argument('--report',type=Path,default=HERE)
    args=parser.parse_args()
    if args.phase=='freeze-review': value=freeze_review(args.report)
    elif args.phase=='prepare': value=prepare(args.report)
    elif args.phase.startswith('live-'): value=live(args.report,args.phase.split('-',1)[1])
    else: value=verify(args.report)
    print(json.dumps({k:v for k,v in value.items() if k!='models'},ensure_ascii=False,indent=2))
