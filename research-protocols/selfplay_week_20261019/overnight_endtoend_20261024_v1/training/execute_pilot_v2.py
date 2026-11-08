"""Finite one-shot OET1 pilot execution; not a supervisor and never self-relaunches."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import os
import signal
import subprocess
import time
from common import *

PYTHON = '/root/miniconda3/bin/python'
PHASES = [
 ('longest_sample_smoke', 'smoke'),
 ('train_GENERIC', 'train'), ('train_TARGETED', 'train'),
 ('final_eval_B1', 'evaluate'), ('final_eval_B1_GENERIC', 'evaluate'), ('final_eval_B1_TARGETED', 'evaluate'),
 ('train_fit_GENERIC', 'fit'), ('train_fit_TARGETED', 'fit'), ('common_compare', 'compare')]


def status_write(state):
    state['heartbeat_at_utc'] = now()
    save(HERE / 'EXECUTION_STATUS.json', state, replace=True)
    lockpath = TASK / '.TASK_STATUS.training.lock'
    with lockpath.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = TASK / 'TASK_STATUS.json'
        if path.exists():
            current = read(path)
            current['training_execution'] = {'status': state['status'], 'current_stage': state.get('current_stage'),
                'heartbeat_at_utc': state['heartbeat_at_utc'], 'status_path': str(HERE / 'EXECUTION_STATUS.json'),
                'completed_stages': list(state['completed_stages']), 'pilot_only': True,
                'new_root_primary_objective_completed': False}
            save(path, current, replace=True)
    append(HERE / 'EXECUTION_HEARTBEAT.jsonl', {k: state[k] for k in ('heartbeat_at_utc', 'status', 'current_stage', 'completed_stages')})


def verified_completion(name, expected, schedule_binding):
    if not expected.exists(): return False
    value = read(expected)
    if name == 'longest_sample_smoke':
        require(value['status'] == 'PASS' and value['schedule_freeze'] == schedule_binding, 'COMPLETED_SMOKE_RECEIPT_DRIFT')
    elif name.startswith('train_') and not name.startswith('train_fit_'):
        require(value['status'] == 'COMPLETED' and value['schedule_freeze'] == schedule_binding, 'COMPLETED_TRAIN_RECEIPT_DRIFT')
        verify([value['adapter'], value['adapter_config'], value['sealed_adapter_receipt'], value['resume_state'], value['committed_optimizer_metrics']])
        require(value['completed_optimizer_steps'] == value['fixed_final_optimizer_step'] == 24, 'PILOT_FIXED_STEP_MISMATCH')
    elif name == 'common_compare':
        require(value['status'] == 'COMPLETED' and value['models'] == ['B1', 'B1_GENERIC', 'B1_TARGETED'], 'COMPLETED_COMMON_COMPARE_DRIFT')
    else:
        require(value['status'] == 'COMPLETED', 'COMPLETED_EVALUATION_DRIFT')
        verify([value['raw'], value['scores'], value['run_manifest']])
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute-gpu', action='store_true')
    ap.add_argument('--wait-ready-seconds', type=int, default=7200)
    args = ap.parse_args()
    require(args.execute_gpu, 'EXPLICIT_GPU_FLAG_REQUIRED')
    supervisor = read(TASK / 'SUPERVISOR_STATUS.json')
    deadline = dt.datetime.fromisoformat(supervisor['deadline_at_utc'])
    reserve = dt.timedelta(minutes=45)
    code = read(HERE / 'CODE_FREEZE.json')
    verify(code['code_bindings'] + [code['engineering_validation'], code['evaluation_freeze'], code['protocol']])
    execution_dir = HERE / 'execution_stages'; execution_dir.mkdir(exist_ok=True)
    state = {'schema_version': 'oet1_finite_pilot_execution_status_v1', 'status': 'WAITING_FOR_REAL_POOL',
             'current_stage': 'wait_pool_schedule_preflights', 'completed_stages': [], 'failures': [],
             'started_at_utc': now(), 'pilot_only': True, 'new_root_objective_completed': False,
             'code_freeze': binding(HERE / 'CODE_FREEZE.json'), 'deadline_at_utc': deadline.isoformat()}
    status_write(state)
    ready = [TASK / 'training_pool/TRAINING_POOL_FREEZE.json', HERE / 'schedule/SCHEDULE_FREEZE.json',
             HERE / 'PREFLIGHT_GENERIC.json', HERE / 'PREFLIGHT_TARGETED.json']
    wait_started = time.monotonic()
    while not all(p.exists() for p in ready):
        require(not (TASK / 'FINALIZE_REQUEST').exists(), 'FINALIZE_REQUEST_BEFORE_NEW_TRAINING')
        require(time.monotonic() - wait_started <= args.wait_ready_seconds, 'FINITE_WAIT_READY_EXHAUSTED')
        require(dt.datetime.now(dt.timezone.utc) + reserve + dt.timedelta(minutes=20) < deadline, 'INSUFFICIENT_DEADLINE_MARGIN')
        time.sleep(15); status_write(state)
    pool = read(ready[0]); frozen = read(ready[1])
    require(pool['status'] == 'FROZEN_BEFORE_TRAINING' and pool['mode'] == 'QFP2_ONLY_PILOT_NEW_ROOT_GATE_FAILED', 'PILOT_SCOPE_MISMATCH')
    require(frozen['fixed_final_step'] == 24 and all(a['questions'] == 24 and a['optimizer_steps'] == 24 for a in frozen['arms'].values()), 'PILOT_MATCHED_BUDGET_REQUIRED')
    schedule_binding = binding(ready[1])
    verify(frozen['code_bindings'] + frozen['model_bindings'] + frozen['tokenizer_bindings'] + [frozen['pool_freeze'], frozen['protocol'], frozen['evaluation_freeze']])
    for arm, path in zip(('GENERIC', 'TARGETED'), ready[2:]):
        receipt = read(path)
        require(receipt['status'] == 'PASS' and receipt['arm'] == arm and receipt['schedule_freeze'] == schedule_binding, 'PREFLIGHT_RECEIPT_MISMATCH')
    longest_arm = max(frozen['arms'], key=lambda a: frozen['arms'][a]['max_full_input_tokens'])
    save(execution_dir / 'EXECUTION_INPUT_ACCEPTANCE.json', {'status': 'PASS', 'accepted_at_utc': now(),
         'pool': binding(ready[0]), 'schedule': schedule_binding, 'preflights': [binding(p) for p in ready[2:]],
         'code_freeze': binding(HERE / 'CODE_FREEZE.json'), 'evaluation_freeze': frozen['evaluation_freeze'],
         'longest_arm': longest_arm, 'longest_full_input_tokens': frozen['arms'][longest_arm]['max_full_input_tokens'],
         'pilot_only': True, 'new_root_objective_completed': False})
    smoke_path = HERE / 'smoke/SMOKE_RESULT.json'
    jobs = []
    jobs.append(('longest_sample_smoke', [PYTHON, str(HERE/'runner.py'), 'smoke', '--arm', longest_arm, '--output', str(HERE/'smoke'), '--execute-gpu'], smoke_path))
    for arm in ('GENERIC', 'TARGETED'):
        out = HERE / 'runs' / arm
        jobs.append(('train_' + arm, [PYTHON, str(HERE/'runner.py'), 'train', '--arm', arm, '--smoke-result', str(smoke_path), '--output', str(out), '--execute-gpu'], out/'RESULT.json'))
    for role in ('B1', 'B1_GENERIC', 'B1_TARGETED'):
        out = HERE / 'final_evaluation' / role
        command = [PYTHON, str(HERE/'runner.py'), 'evaluate', '--role', role, '--output', str(out), '--execute-gpu']
        if role != 'B1': command += ['--training-result', str(HERE/'runs'/role[3:]/'RESULT.json')]
        jobs.append(('final_eval_' + role, command, out/'SUMMARY.json'))
    for arm in ('GENERIC', 'TARGETED'):
        out = HERE / 'train_fit' / arm
        jobs.append(('train_fit_' + arm, [PYTHON, str(HERE/'runner.py'), 'fit', '--arm', arm, '--training-result', str(HERE/'runs'/arm/'RESULT.json'), '--output', str(out), '--execute-gpu'], out/'SUMMARY.json'))
    jobs.append(('common_compare', [PYTHON, str(HERE/'runner.py'), 'compare', '--summaries',
        *[str(HERE/'final_evaluation'/role/'SUMMARY.json') for role in ('B1','B1_GENERIC','B1_TARGETED')],
        '--training-results', str(HERE/'runs/GENERIC/RESULT.json'), str(HERE/'runs/TARGETED/RESULT.json'),
        '--output', str(HERE/'COMMON_EVALUATION_SUMMARY.json')], HERE/'COMMON_EVALUATION_SUMMARY.json'))
    state['status'] = 'RUNNING'
    for name, command, expected in jobs:
        state['current_stage'] = name; status_write(state)
        if verified_completion(name, expected, schedule_binding):
            state['completed_stages'].append(name); status_write(state); continue
        if name.startswith('train_') and not name.startswith('train_fit_'):
            require(not (TASK / 'FINALIZE_REQUEST').exists(), 'FINALIZE_REQUEST_STOPS_NEW_TRAINING')
            require(dt.datetime.now(dt.timezone.utc) + reserve + dt.timedelta(minutes=20) < deadline, 'TRAINING_DEADLINE_MARGIN_EXHAUSTED')
        for attempt in range(1, 4):
            attempt_dir = execution_dir / name / f'attempt_{attempt:02d}'
            attempt_dir.mkdir(parents=True, exist_ok=False)
            actual_command = command[:]
            if name.startswith('train_') and not name.startswith('train_fit_'):
                if (expected.parent/'resume_state.pt').exists(): actual_command += ['--resume']
            start = {'status': 'STARTED', 'stage': name, 'attempt': attempt, 'command': actual_command, 'started_at_utc': now(), 'expected_artifact': str(expected), 'schedule_freeze': schedule_binding}
            save(attempt_dir/'STARTED.json', start)
            log = attempt_dir/'stdout_stderr.log'
            with log.open('xb') as stream:
                proc = subprocess.Popen(actual_command, cwd=str(TASK), stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                state['active_pid'] = proc.pid; state['active_log'] = str(log); status_write(state)
                stage_started = time.monotonic()
                stage_timeout = 900 if name == 'longest_sample_smoke' else 3600
                timed_out = False
                while proc.poll() is None:
                    time.sleep(10); status_write(state)
                    if time.monotonic() - stage_started >= stage_timeout or dt.datetime.now(dt.timezone.utc) >= deadline:
                        timed_out = True; os.killpg(proc.pid, signal.SIGTERM)
                        try: proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)
                        break
            terminal = {'status': 'COMPLETED' if proc.returncode == 0 else 'FAILED_ENGINEERING_OR_RUNTIME',
                        'stage': name, 'attempt': attempt, 'exit_code': proc.returncode, 'timed_out': timed_out,
                        'stage_timeout_seconds': stage_timeout, 'completed_at_utc': now(), 'log': binding(log)}
            if proc.returncode == 0:
                require(verified_completion(name, expected, schedule_binding), 'EXIT0_WITHOUT_VALID_ARTIFACT')
                terminal['artifact'] = binding(expected); save(attempt_dir/'TERMINAL.json', terminal)
                state['completed_stages'].append(name); status_write(state); break
            save(attempt_dir/'TERMINAL.json', terminal); state['failures'].append(terminal); status_write(state)
            # Retry only engineering transport/process interruptions. Semantic model
            # outputs already present remain immutable and aren't sampled again.
            content = log.read_text(errors='replace')
            recoverable = any(x in content for x in ('FileLock', 'temporarily unavailable', 'Connection reset', 'ConnectionError'))
            if not recoverable or attempt == 3:
                raise RuntimeError('EXECUTION_STAGE_FAILED:' + name + ':see:' + str(log))
        else: raise RuntimeError('ENGINEERING_RETRY_BUDGET_EXHAUSTED:' + name)
    state['status'] = 'COMPLETED_PILOT_ONLY'; state['current_stage'] = 'PILOT_DELIVERED_PENDING_MAIN_CLOSEOUT'; state['completed_at_utc'] = now()
    state['common_summary'] = binding(HERE/'COMMON_EVALUATION_SUMMARY.json'); status_write(state)
    save(execution_dir/'PILOT_EXECUTION_TERMINAL.json', state)
    print(json.dumps({'status': state['status'], 'common_summary': state['common_summary'], 'new_root_objective_completed': False}), flush=True)

if __name__ == '__main__':
    try: main()
    except Exception as exc:
        previous = read(HERE/'EXECUTION_STATUS.json') if (HERE/'EXECUTION_STATUS.json').exists() else {'completed_stages': [], 'current_stage': 'initialization'}
        previous.update(status='BLOCKED_ENGINEERING_OR_RESOURCE', error_type=type(exc).__name__, error=str(exc), terminal_at_utc=now())
        status_write(previous)
        append(HERE/'EXECUTION_FAILURES.jsonl', {k: previous.get(k) for k in ('terminal_at_utc','current_stage','error_type','error')})
        raise
