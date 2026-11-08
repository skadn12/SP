"""Run native M0 on the exact frozen v127 ordering request projection."""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PRIOR = REPO / 'offline-deliveries/h4_codex_dev_eval_v127'
NATIVE = REPO / 'offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation'
RUN = HERE / 'run/base'
PRIOR_SHA = '4fb446adcf81d8ce6db72accf78dd3e47b64c2e2548add3191561856e8389d61'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def verify(rows):
    for row in rows:
        assert sha(row['path']) == row['sha256'], 'BOUND_FILE_CHANGED:' + row['path']


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def native():
    sys.path.insert(0, str(NATIVE))
    spec = importlib.util.spec_from_file_location('v130_native_runner', NATIVE / 'run_solver.py')
    obj = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = obj
    spec.loader.exec_module(obj)
    assert obj.MODEL_PATHS['M0'] == Path('/root/autodl-tmp/Qwen3-4B')
    assert obj.SEED == 2026100800
    return obj


def prepare():
    assert not RUN.exists() and not (HERE / 'SCOPE.json').exists(), 'ALREADY_PREPARED'
    assert sha(PRIOR / 'COMPLETE_MANIFEST.json') == PRIOR_SHA
    for row in read(PRIOR / 'COMPLETE_MANIFEST.json')['files']:
        assert sha(PRIOR / row['path']) == row['sha256'], 'PRIOR_SEAL_CHANGED'
    old = PRIOR / 'run/baseline'
    original = read(old / 'manifest.json')
    requests = [r for r in original['requests'] if r['question_type'] == 'ordering']
    questions = [q for q in read(old / 'questions.json')['candidates'] if q['question_type'] == 'ordering']
    ids = [q['candidate_id'] for q in questions]
    roots = {q['root_id'] for q in questions}
    assert len(ids) == len(set(ids)) == 14 and len(roots) == 7 and len(requests) == 168
    assert requests == [r for r in read(PRIOR / 'run/new/manifest.json')['requests'] if r['question_type'] == 'ordering']
    pool = read(PRIOR / 'EVALUATION_POOL.json')
    pool['questions'] = [q for q in pool['questions'] if q['candidate_id'] in ids]
    assert [q['candidate_id'] for q in pool['questions']] == ids
    save(HERE / 'EVALUATION_POOL.json', pool)
    save(RUN / 'questions.json', {'candidates': questions})
    save(RUN / 'gold.json', {k: v for k, v in read(old / 'gold.json').items() if k in ids})
    save(RUN / 'canonical_bindings.json', {'candidates': [q for q in read(old / 'canonical_bindings.json')['candidates'] if q['candidate_id'] in ids]})
    for root in sorted(roots):
        target = RUN / 'sources' / f'{root}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old / 'sources' / target.name, target)
    for row in requests:
        target = RUN / 'prompts' / (row['request_id'] + '.txt')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old / 'prompts' / target.name, target)
        assert sha(target) == row['prompt_sha256']
    runner = native()
    assert Path(sys.prefix) == runner.EXPECTED_VENV, 'USE_PROVEN_LOCAL4B_VENV'
    identity = runner.model_identity('M0')
    assert identity['model_path'] == '/root/autodl-tmp/Qwen3-4B'
    assert not identity.get('adapter_path') and not identity.get('adapter_enabled')
    runner._validate_model(identity)
    save(RUN / 'solver_identity.json', identity)
    manifest = copy.deepcopy(original)
    manifest.update(prepared_at_utc=now(), model=identity['model_path'], solver_model='M0',
                    solver_identity=identity, candidate_count=14, request_count=168,
                    requests=requests, pool_path=str(HERE / 'EVALUATION_POOL.json'),
                    pool_sha256=sha(HERE / 'EVALUATION_POOL.json'))
    manifest['input_sha256'] = {'pool': manifest['pool_sha256'], **{
        str(path.relative_to(RUN)): sha(path)
        for path in [RUN / 'questions.json', RUN / 'gold.json', RUN / 'canonical_bindings.json', *sorted((RUN / 'sources').glob('*.json'))]}}
    save(RUN / 'manifest.json', manifest)
    runner._verify_manifest(RUN, manifest)
    previous_scope = read(PRIOR / 'SCOPE.json')
    save(HERE / 'SCOPE.json', {
        'schema_version': 'v130_exact_ordering_base_control_v1', 'created_at_utc': now(),
        'question_count': 14, 'candidate_ids': ids, 'new_inference_calls': 168,
        'reused_prior_ordering_calls': 336, 'comparison_total_calls': 504,
        'fixed_dev_root_ids': previous_scope['fixed_dev_root_ids'],
        'covered_dev_root_ids': previous_scope['covered_dev_root_ids'],
        'test_evaluation': False, 'automatic_retry': False, 'selection_based_filtering': False,
        'base_model': 'M0', 'base_model_path': identity['model_path'],
        'policy_version': manifest['policy_version'], 'seed': runner.SEED,
        'request_projection': 'Exact v127 ordering request list, including original non-contiguous seeds; no renumbering or new permutations.',
        'native_algorithm_changed': False, 'previous_eval_manifest': binding(PRIOR / 'COMPLETE_MANIFEST.json'),
        'interpretation': 'Post-evaluation DEV diagnostic with14 questions on7 covered roots; correlated permutations are not independent questions. No causal ablation of noise versus task adaptation.'})
    paths = {str(Path(__file__).resolve()), str(HERE / 'report_control.py')}
    paths.update(row['path'] for row in read(PRIOR / 'TOOLS_FREEZE.json')['files'])
    save(HERE / 'TOOLS_FREEZE.json', {'status': 'FROZEN_BEFORE_FIRST_BASE_INFERENCE',
        'created_at_utc': now(), 'files': [binding(path) for path in sorted(paths)]})
    inputs = [HERE / 'SCOPE.json', HERE / 'EVALUATION_POOL.json', HERE / 'TOOLS_FREEZE.json',
              PRIOR / 'COMPLETE_MANIFEST.json',
              REPO / 'offline-deliveries/h4_codex_training_v126/evidence/DEV_EVALUATION_PLAN_AUDIT.json']
    inputs.extend(path for path in RUN.rglob('*') if path.is_file())
    for side in ['baseline', 'new']:
        other = PRIOR / 'run' / side
        inputs.extend(other / name for name in ['manifest.json', 'questions.json', 'gold.json', 'canonical_bindings.json', 'solver_identity.json', 'COMPLETE.json', 'RESULTS.json'])
        inputs.extend(other / 'responses' / (row['request_id'] + '.json') for row in requests)
    inputs.extend(Path(row['path']) for row in identity['checkpoint_files'])
    inputs.append(Path(identity['audit_path']))
    save(HERE / 'PREFLIGHT_FREEZE.json', {'status': 'READY_FOR_INDEPENDENT_AUDIT',
        'created_at_utc': now(), 'files': [binding(path) for path in sorted(set(inputs))],
        'question_count': 14, 'calls': 168, 'test_evaluation': False})
    print(json.dumps({'status': 'PREPARED', 'question_count': 14, 'calls': 168}))


def frozen():
    verify(read(HERE / 'TOOLS_FREEZE.json')['files'])
    verify(read(HERE / 'PREFLIGHT_FREEZE.json')['files'])
    audit = read(HERE / 'evidence/INDEPENDENT_PREFLIGHT_AUDIT.json')
    assert audit['status'] == 'PASS' and not audit.get('errors')
    verify(audit['audited_files'])


def status(value):
    temp = HERE / 'RUN_STATUS.pending'
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(HERE / 'RUN_STATUS.json')


def live():
    assert not (HERE / 'RUN_STATUS.json').exists(), 'ALREADY_STARTED_NO_AUTOMATIC_RERUN'
    start = time.monotonic()
    record = {'status': 'VERIFYING_FROZEN_BASE_CONTROL', 'started_at_utc': now(),
              'pid': os.getpid(), 'planned_new_calls': 168, 'test_evaluation': False,
              'automatic_retry': False, 'python_argv0': sys.executable}
    status(record)
    try:
        frozen()
        runner = native()
        assert Path(sys.prefix) == runner.EXPECTED_VENV, 'USE_PROVEN_LOCAL4B_VENV'
        record.update(status='BASE_EVALUATION_RUNNING', inference_started_at_utc=now())
        status(record)
        inference_start = time.monotonic()
        runner.live(RUN)
        record['base_elapsed_seconds'] = time.monotonic() - inference_start
        runner.report(RUN)
        record.update(status='REPORTING', actual_new_calls=read(RUN / 'COMPLETE.json')['actual_call_count'])
        status(record)
        subprocess.run([sys.executable, str(HERE / 'report_control.py')], check=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        assert read(HERE / 'SUMMARY.json')['status'] == 'COMPLETE'
        record.update(status='BASE_CONTROL_REPORTED_AWAITING_INDEPENDENT_AUDIT',
                      pid=None, finished_at_utc=now(), elapsed_seconds=time.monotonic() - start)
        status(record)
        print(json.dumps(record, ensure_ascii=False))
    except BaseException as exc:
        record.update(status='STOPPED_WITH_ERROR', error_type=type(exc).__name__,
                      error=str(exc), stopped_at_utc=now())
        status(record)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'verify', 'live'])
    phase = parser.parse_args().phase
    {'prepare': prepare, 'verify': frozen, 'live': live}[phase]()
