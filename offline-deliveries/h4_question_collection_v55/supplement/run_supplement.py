"""Explicit supplement for only the two v55 c05 MC slots blocked before HTTP.

prepare and verify are offline. generation and evaluation are separate explicit
commands, never automatic retries. The original v55 code and runs are immutable.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
BASE = REPORT.parent
CODE = BASE / 'h4_sequential_ab_v55_capacity_qa'
ORIGINAL = BASE / 'h4_sequential_ab_v55_qa'
PYTHON = Path('/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python')
sys.path.insert(0, str(CODE))
import generate as gen
import evaluate_and_pool as pipeline

ROOT = 'cc-train-episode-135466'
BATCH = ORIGINAL / 'runs/root135466_collect20_01'
BLOCKED_RUN = ORIGINAL / 'runs/root135466_collect20_01_c05_generation'
BLOCKED_REQUEST = BLOCKED_RUN / 'requests/001-135466-merged-multi-generate-1'
RUN = CODE / 'runs/root135466_c05_capacity4096_01_generation'
EVAL = CODE / 'runs/root135466_c05_capacity4096_01_evaluation'
VERSION = 'v55_explicit_capacity4096_supplement_two_unattempted_slots_v1'


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def bind_files(paths):
    return [binding(path) for path in sorted(set(Path(p).resolve() for p in paths))]


def verify_bindings(rows):
    for row in rows:
        require(binding(row['path']) == row, 'supplement_bound_file_changed:' + row['path'])


def marker(stage, state):
    return HERE / f'{stage}.{state}.json'


def begin(stage):
    require(not any(marker(stage, state).exists() for state in ('STARTED', 'COMPLETE', 'ERROR')),
            'supplement_stage_already_recorded_no_replay:' + stage)
    save(marker(stage, 'STARTED'), {'version': VERSION, 'stage': stage, 'started_at_utc': now(),
                                   'automatic_replay': False})


def error(stage, exc):
    if not marker(stage, 'ERROR').exists():
        save(marker(stage, 'ERROR'), {'version': VERSION, 'stage': stage, 'failed_at_utc': now(),
             'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
             'automatic_replay': False, 'artifacts_preserved': True})


def snapshot_old_files():
    paths = [path for path in ORIGINAL.glob('*') if path.is_file()]
    for index in range(1, 6):
        directory = ORIGINAL / 'runs' / f'root135466_collect20_01_c{index:02d}_generation'
        paths += [path for path in directory.rglob('*') if path.is_file()]
    paths += [path for path in BATCH.rglob('*') if path.is_file()]
    correction = BASE / 'h4_cardinality_correction_v55/RESULT_BINDINGS.json'
    paths += [correction, BASE / 'h4_question_rule_validation_v54/RESULT_BINDINGS.json']
    for path in (correction, BASE / 'h4_question_rule_validation_v54/RESULT_BINDINGS.json'):
        for row in read(path)['files']:
            require(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256'],
                    'old_historical_binding_changed:' + row['path'])
            paths.append(Path(row['path']))
    return bind_files(paths)


def prepare():
    begin('preparation')
    try:
        require(not RUN.exists() and not EVAL.exists(), 'supplement_run_already_exists')
        require((BATCH / 'GENERATION_COMPLETE.json').is_file(), 'root135466_all_five_chunks_not_complete')
        old_manifest = read(BLOCKED_RUN / 'MANIFEST.json')
        old_request = read(BLOCKED_REQUEST / 'REQUEST.json')
        old_proof = read(BLOCKED_REQUEST / 'PROOF.json')
        old_error = read(BLOCKED_REQUEST / 'ERROR.json')
        require(old_error['error_type'] == 'LocalCapacityBlocked' and old_error['http_attempts'] == 0
                and not (BLOCKED_REQUEST / 'HTTP_ATTEMPT.json').exists(), 'original_slots_were_already_remotely_attempted')
        require(old_request['max_tokens'] == 5120 and old_proof['capacity']['status'] == 'BLOCKED',
                'original_capacity_evidence_changed')
        context = json.JSONDecoder().raw_decode(old_request['messages'][0]['content'].split('\nINPUT_JSON\n', 1)[1])[0]
        ids = context['candidate_ids']
        require(len(ids) == 2 and all('-merged-multi-' in qid for qid in ids), 'original_blocked_slots_not_exact_two_mc')
        report = read(BLOCKED_RUN / 'REPORT.json')
        multi = next(row for row in report['roots'] if row['question_type'] == 'multi_select')
        require(multi['mechanical_pass'] == multi['ready_for_solver'] == 0
                and {row['candidate_id'] for row in multi['mechanical_rejected']} == set(ids),
                'original_blocked_slot_accounting_changed')
        require(not any(question['question_type'] == 'multi_select'
                for question in read(BLOCKED_RUN / 'MECHANICAL_POOL.json')['questions']), 'blocked_slots_already_produced_questions')
        focus = {qid: deepcopy(old_manifest['slot_focus_by_candidate_id'][qid]) for qid in ids}
        inventory = BATCH / 'priors/after_c05.json'
        expected = []
        for index in range(1, 6):
            for question in read(ORIGINAL / f'runs/root135466_collect20_01_c{index:02d}_generation/MECHANICAL_POOL.json')['questions']:
                field = 'options' if question['question_type'] == 'multi_select' else 'items'
                expected.append({key: question[key] for key in ('candidate_id', 'root_id', 'question_type', 'question', field)})
        require(read(inventory) == {'questions': expected}, 'final_c05_public_inventory_not_exact_projection')
        historical = REPORT / f'priors/{ROOT}_v51_v52b_v53_v54_public.json'
        history = BASE / 'h4_sequential_ab_v51/runs/root135466_a3b3_01/CANDIDATE_HISTORY.json'
        old_bindings = snapshot_old_files()
        save(HERE / 'ORIGINAL_FILES_BEFORE.json', {'files': old_bindings, 'old_file_count': len(old_bindings)})
        baseline = read(CODE / 'BASELINE_FILES.json')
        verify_bindings(baseline['files'])
        gen.prepare(RUN, history, roots=[ROOT], slots=2, prior_pools=[historical, inventory],
                    prompt_arm='current', comparison_id=old_manifest['comparison_id'], slot_focus_map=focus)
        payload, receipt = gen.make_payload(RUN, ROOT, 'multi_select', gen.initial_context(RUN, ROOT, 'multi_select'))
        require(payload['max_tokens'] == 4096, 'actual_output_limit_not4096')
        new_context = json.JSONDecoder().raw_decode(payload['messages'][0]['content'].split('\nINPUT_JSON\n', 1)[1])[0]
        require(new_context['candidate_ids'] == ids and new_context['source_turns'] == context['source_turns'],
                'blocked_slot_identity_or_source_changed')
        require(new_context['slot_focus_by_candidate_id'] == context['slot_focus_by_candidate_id'], 'blocked_slot_focus_changed')
        require(read(RUN / 'sources' / f'{ROOT}.json') == read(BLOCKED_RUN / 'sources' / f'{ROOT}.json'),
                'source_context_changed')
        suffix = ROOT + '/writer__M_clean_policy_feedback.txt'
        for name in ('source_lock.json', suffix):
            require((RUN / 'feedback' / name).read_bytes() == (BLOCKED_RUN / 'feedback' / name).read_bytes(),
                    'previous_round_feedback_changed')
        for old_prior in context['previous_candidates_dedup_only']:
            require(old_prior in new_context['previous_candidates_dedup_only'], 'old_public_prior_omitted')
        _, _, annotate, capacity, dev = gen.base.transport()
        tokenizer = dev.tokenizer_from(str(gen.base.TOKENIZER))
        profile = read(RUN / 'provider_profile.json')
        profile['output_limit'] = profile['request_parameters']['max_tokens'] = payload['max_tokens']
        cap = annotate(capacity(profile, tokenizer, payload['messages'], payload['max_tokens']))
        save(HERE / 'INITIAL_REQUEST.json', {'payload': payload, 'receipt': receipt, 'capacity': cap,
             'original_blocked_capacity': old_proof['capacity'], 'full_prior_final_inventory_count': len(expected)})
        require(cap['status'] == 'PASS', 'supplement_initial_capacity_still_blocked')
        code_paths = [path for path in CODE.glob('*') if path.is_file()]
        inputs = [history, historical, inventory, BATCH / 'GENERATION_COMPLETE.json',
                  BLOCKED_REQUEST / 'REQUEST.json', BLOCKED_REQUEST / 'PROOF.json', BLOCKED_REQUEST / 'ERROR.json',
                  BLOCKED_RUN / 'MANIFEST.json', BLOCKED_RUN / 'REPORT.json', Path(__file__),
                  HERE / 'ORIGINAL_FILES_BEFORE.json', HERE / 'INITIAL_REQUEST.json', RUN / 'MANIFEST.json', *code_paths]
        manifest = {'version': VERSION, 'prepared_at_utc': now(), 'root_id': ROOT,
            'original_generation_run': str(BLOCKED_RUN), 'original_blocked_request': str(BLOCKED_REQUEST),
            'generation_run': str(RUN), 'evaluation_run': str(EVAL),
            'same_unattempted_candidate_ids': ids, 'candidate_slots': 2, 'additional_distinct_slots': 0,
            'supplement_run_budget': {'http_attempts_max': 2, 'solver_full_calls_max': 36},
            'question_rules_changed': False, 'actual_output_limit_before': 5120, 'actual_output_limit_after': 4096,
            'source_feedback_prior_text_cuts': 0, 'same_soft_focus': True, 'new_frozen_final_prior_sets': len(expected),
            'automatic_replay': False, 'original_failed_request_retained': True,
            'all_collection_and_supplement_generation_before_solver': True,
            'training_started': False, 'training_allowed': False, 'initial_capacity': cap,
            'files': bind_files(inputs)}
        save(HERE / 'MANIFEST.json', manifest)
        verify()
        save(marker('preparation', 'COMPLETE'), {'version': VERSION, 'completed_at_utc': now(),
             'manifest': binding(HERE / 'MANIFEST.json'), 'models_called': 0})
        print(json.dumps({'status': 'PREPARED', 'candidate_slots': 2, 'same_unattempted_candidate_ids': ids,
                          'required_context_tokens': cap['required_context_tokens'], 'models_called': 0}), flush=True)
    except BaseException as exc:
        error('preparation', exc)
        raise


def verify():
    manifest = read(HERE / 'MANIFEST.json')
    require(manifest['version'] == VERSION and manifest['candidate_slots'] == 2
            and manifest['additional_distinct_slots'] == 0, 'supplement_scope_changed')
    verify_bindings(manifest['files'])
    verify_bindings(read(HERE / 'ORIGINAL_FILES_BEFORE.json')['files'])
    gen.verify(RUN)
    return manifest


def no_solver_started():
    require(not (REPORT / 'evaluation.STARTED.json').exists(), 'collection_solver_already_started')
    require(not (REPORT / 'recovery/solver/STARTED.json').exists(), 'recovery_solver_already_started')
    for directory in ORIGINAL.glob('runs/root*_collect20_01_c*_evaluation/solver'):
        require(not (directory / 'STARTED.json').exists(), 'chunk_solver_already_started')


def generation():
    manifest = verify()
    no_solver_started()
    begin('generation')
    try:
        value = gen.generate(RUN)
        pipeline.verify_generation(RUN)
        require(value['candidate_slots'] == 2 and value['http_attempts'] <= 2, 'supplement_generation_budget_changed')
        require(all(row['question_type'] == 'multi_select' for row in value['roots']), 'unexpected_supplement_ordering')
        save(marker('generation', 'COMPLETE'), {'version': VERSION, 'completed_at_utc': now(),
             'report': binding(RUN / 'REPORT.json'), 'candidate_slots': 2,
             'http_attempts': value['http_attempts'], 'mechanical_pass': value['mechanical_pass'],
             'ready_for_solver': value['ready_for_solver'],
             'files': bind_files(path for path in RUN.rglob('*') if path.is_file())})
        verify()
        print(json.dumps({'status': 'GENERATION_COMPLETE', **value}), flush=True)
    except BaseException as exc:
        error('generation', exc)
        raise


def command(action):
    path = HERE / (action.upper() + '.log')
    argv = [str(PYTHON), '-B', '-u', str(CODE / 'evaluate_and_pool.py'), action, '--run', str(EVAL)]
    if action == 'preflight':
        argv += ['--generation', str(RUN)]
    with path.open('x', encoding='utf-8') as stream:
        subprocess.run(argv, cwd=CODE, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return path


def evaluation():
    verify()
    complete = read(marker('generation', 'COMPLETE'))
    verify_bindings(complete['files'])
    require((REPORT / 'generation.COMPLETE.json').is_file(), 'all_collection_generation_not_complete')
    begin('evaluation')
    try:
        logs = [command(action) for action in ('preflight', 'live', 'finalize', 'verify')]
        report = read(EVAL / 'training_pool/REPORT.json')
        require(report['full_solver_calls'] <= 36, 'supplement_solver_budget_exceeded')
        save(marker('evaluation', 'COMPLETE'), {'version': VERSION, 'completed_at_utc': now(),
             'full_solver_calls': report['full_solver_calls'], 'selected': report['new_unique_added'],
             'files': bind_files([*logs, *(path for path in EVAL.rglob('*') if path.is_file())])})
        verify()
        print(json.dumps({'status': 'EVALUATION_COMPLETE', 'full_solver_calls': report['full_solver_calls'],
                          'selected': report['new_unique_added']}), flush=True)
    except BaseException as exc:
        error('evaluation', exc)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'generation', 'evaluation', 'verify'))
    args = parser.parse_args()
    result = globals()[args.action]()
    if args.action == 'verify':
        print(json.dumps({'status': 'PASS', 'candidate_slots': result['candidate_slots']}))
