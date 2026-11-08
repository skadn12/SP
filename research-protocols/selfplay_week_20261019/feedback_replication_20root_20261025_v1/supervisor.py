#!/usr/bin/env python3
"""Bounded detached Codex supervision; no training and no original-root reruns.

The outer worker organizes engineering only. All source/question authors and
reviewers use fresh text-only sessions through scripts/codex_calls.py. Process
cleanup is scoped to verified PID/start-token descendants and this task's own
registry; no global model/process-name matching is used.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
CODEX = Path('/root/.vscode-server/extensions/openai.chatgpt-26.51002.51308-linux-x64/bin/linux-x86_64/codex')
MAX_SECONDS, MAX_ROUNDS = 12 * 3600, 8
HEARTBEAT_SECONDS = 30


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def validator(base=BASE):
    spec = importlib.util.spec_from_file_location('fg_replication_final_validator', Path(base) / 'validate_final.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_info(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'ppid': int(fields[1]), 'pgid': int(fields[2]),
                'session': int(fields[3]), 'start_token': fields[19], 'state': fields[0]}
    except (OSError, ValueError, IndexError):
        return None


def process_start(pid):
    info = process_info(pid)
    return info['start_token'] if info else None


def process_snapshot():
    snapshot = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            info = process_info(path.name)
            if info:
                snapshot[info['pid']] = info
    return snapshot


def registry_tokens(base):
    tokens = {}
    paths = list((Path(base) / 'process_registry').glob('*.json'))
    paths += list((Path(base) / 'cli_calls').glob('*/attempt_*/PID.json'))
    for path in paths:
        try:
            row = read(path)
            if Path(row['task_directory']).resolve() != Path(base).resolve():
                continue
            pid, token = int(row['pid']), row['start_token']
            if token is not None and process_start(pid) == str(token):
                tokens[str(pid)] = str(token)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return tokens


def observe_task_processes(base, known, seed=None):
    """Keep proven descendants even if their process-group leader exits."""
    snapshot = process_snapshot()
    tokens = {str(pid): str(token) for pid, token in known.items()}
    tokens.update(registry_tokens(base))
    if seed and seed.get('start_token') is not None:
        tokens[str(seed['pid'])] = str(seed['start_token'])
    live = {pid for pid, info in snapshot.items() if tokens.get(str(pid)) == info['start_token']}
    changed = True
    while changed:
        changed = False
        for pid, info in snapshot.items():
            if pid not in live and info['ppid'] in live:
                live.add(pid)
                tokens[str(pid)] = info['start_token']
                changed = True
    # Never consider the supervisor or the process that launched it task work.
    live -= {os.getpid(), os.getppid()}
    return {str(pid): snapshot[pid]['start_token'] for pid in live}, snapshot


def cleanup_task_processes(base, known, seed=None):
    """Kill only this task's checked PID/start-token tree and registry groups."""
    owned, snapshot = observe_task_processes(base, known, seed)
    actions = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        owned, snapshot = observe_task_processes(base, owned, seed)
        live = {int(pid) for pid in owned}
        groups = {}
        for pid in live:
            info = snapshot[pid]
            groups.setdefault(info['pgid'], set()).add(pid)
        signaled = set()
        for pgid, members in groups.items():
            all_members = {pid for pid, info in snapshot.items() if info['pgid'] == pgid}
            # A task-owned session/group leader plus no unrelated members is
            # required before group signaling. Otherwise signal exact PIDs.
            leader = snapshot.get(pgid)
            if leader and pgid in live and leader['session'] == pgid and all_members <= live:
                if process_start(pgid) == owned.get(str(pgid)):
                    try:
                        os.killpg(pgid, sig)
                        actions.append({'pgid': pgid, 'signal': sig.name, 'member_pids': sorted(all_members)})
                        signaled.update(all_members)
                    except ProcessLookupError:
                        pass
        for pid in sorted(live - signaled, reverse=True):
            if process_start(pid) == owned.get(str(pid)):
                try:
                    os.kill(pid, sig)
                    actions.append({'pid': pid, 'signal': sig.name})
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                current = [process_info(pid) for pid in live]
                if all(info is None or info['state'] == 'Z' for info in current):
                    break
                time.sleep(.1)
    remaining, _ = observe_task_processes(base, owned, seed)
    return {'at_utc': utc(), 'actions': actions, 'remaining_verified_processes': remaining}


def preflight(base=BASE):
    v = validator(base)
    plan, _, _, _, _ = v.design(base)
    count = v.verify_input_lock(base)
    if not CODEX.is_file() or not os.access(CODEX, os.X_OK):
        raise ValueError('CODEX_EXECUTABLE_UNAVAILABLE')
    for name in ('WORKER_INSTRUCTIONS_ZH.md', 'TASK_STATUS.json', 'scripts/codex_calls.py'):
        if not (Path(base) / name).is_file():
            raise ValueError('MISSING_WORKER_INPUT:' + name)
    return {'status': 'PASS_READONLY_PREFLIGHT', 'locked_files_verified': count,
            'max_seconds': plan['max_seconds'], 'max_worker_rounds': plan['max_worker_rounds'],
            'model_or_worker_started': False}


def terminal_report(v, state):
    result = v.validate(BASE)
    if result['valid']:
        state['final_validation'] = {key: value for key, value in result.items() if key != 'report'}
        return result['report']
    state['last_final_validation_error'] = result.get('error')
    return None


def finalization_request(reason):
    path = BASE / 'FINALIZE_REQUEST'
    if not path.exists():
        path.write_text(reason + '\n')


def worker_command(last_path, thread_id):
    common = [str(CODEX), 'exec']
    if thread_id:
        return common + ['resume', '--skip-git-repo-check', '--json',
                         '-c', 'approval_policy="never"', '-c', 'sandbox_mode="danger-full-access"',
                         '-o', str(last_path), thread_id, '-']
    return common + ['--skip-git-repo-check', '--sandbox', 'danger-full-access',
                     '-c', 'approval_policy="never"', '--color', 'never', '--json', '-o', str(last_path), '-']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(preflight(), ensure_ascii=False))
        return 0
    preflight()
    lock = (BASE / 'supervisor.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Supervisor already running', flush=True)
        return 3
    (BASE / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
    state_path = BASE / 'SUPERVISOR_STATUS.json'
    if state_path.exists():
        state = read(state_path)
        started = float(state['started_unix'])
    else:
        started = time.time()
        state = {'schema_version': 'fg_replication_overnight_supervisor_v1', 'started_unix': started,
                 'started_at_utc': utc(), 'deadline_at_utc': dt.datetime.fromtimestamp(started + MAX_SECONDS, dt.timezone.utc).isoformat(),
                 'rounds': [], 'max_rounds': MAX_ROUNDS, 'training': False, 'rerun_original_four': False}
    state.update(supervisor_pid=os.getpid(), supervisor_start_token=process_start(os.getpid()), status='RUNNING', updated_at_utc=utc())
    known = state.get('owned_process_tokens', {})
    previous_seed = {'pid': state.get('worker_pid'), 'start_token': state.get('worker_start_token')}
    if (previous_seed['pid'] and previous_seed['start_token']) or known or registry_tokens(BASE):
        receipt = cleanup_task_processes(BASE, known, previous_seed)
        atomic(BASE / 'ORPHAN_RECONCILIATION.json', receipt)
    state['worker_pid'] = None
    state['owned_process_tokens'] = {}
    atomic(state_path, state)
    v = validator()
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', WANDB_MODE='offline',
               GXY_REPLICATION_TASK_DIRECTORY=str(BASE))
    thread_id = state.get('thread_id')
    failures = int(state.get('consecutive_worker_errors', 0))
    process, seed, known = None, None, {}
    (BASE / 'worker_events').mkdir(exist_ok=True)

    def request_signal_closeout(signum, _frame):
        state.update(status='STOPPED_BY_SIGNAL', received_signal=signal.Signals(signum).name)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, request_signal_closeout)
    signal.signal(signal.SIGINT, request_signal_closeout)
    try:
        while len(state['rounds']) < MAX_ROUNDS:
            elapsed = time.time() - started
            if (BASE / 'STOP_REQUEST').exists() or elapsed >= MAX_SECONDS:
                state['status'] = 'STOPPED_BY_REQUEST' if (BASE / 'STOP_REQUEST').exists() else 'DEADLINE_REACHED'
                break
            if elapsed >= MAX_SECONDS - 45 * 60:
                finalization_request('停止新增语义生成，优先完成已有产物评估和可靠收口；不训练。')
            # The eighth and final worker is reserved for honest closure.
            round_index = len(state['rounds']) + 1
            if round_index == MAX_ROUNDS:
                finalization_request('最后一个worker轮次：停止新增生成，汇总真实完成情况并冻结报告；缺额必须PARTIAL。')
            report = terminal_report(v, state)
            if report:
                state['status'] = 'COMPLETED' if report['overall_status'] == 'COMPLETED' else 'CLOSED_WITH_BLOCKERS'
                break
            instructions = (BASE / 'WORKER_INSTRUCTIONS_ZH.md').read_text()
            prompt = (f'任务目录：{BASE}。工作区：{REPO}。先读取本目录WORKER_INSTRUCTIONS_ZH.md、PLAN.json、'
                      'TASK_STATUS.json、INPUT_LOCK.json，以及validate_final.py --describe的交付契约。'
                      '仅实施本目录用户授权的新增16 Train roots F/G复现；原4只读复用，不训练，不重跑旧评估。'
                      '输入锁定文件不可修改。所有语义作者/审核只能通过scripts/codex_calls.py新会话；'
                      '外层worker只组织与工程，不自行写对话、题目、Gold或审核结论。\n\n' + instructions)
            if thread_id:
                prompt += '\n继续同一任务未完成阶段，先校验已有产物和call回执，禁止重复已完成语义生成/评估。'
            generation_closed = (BASE / 'FINALIZE_REQUEST').exists()
            # A 45-minute generation stop may still finish existing Solver
            # work. It must never exempt that worker from the 20-minute hard
            # closeout boundary. The final worker round is closure-only too.
            final_closeout_only = (elapsed >= MAX_SECONDS - 20 * 60 or round_index == MAX_ROUNDS)
            if generation_closed:
                prompt += '\nFINALIZE_REQUEST已存在：不得新增来源/题目生成；只做已有产物评估、独立审核、真实报告和manifest收口。'
            if final_closeout_only:
                prompt += ('\nFINAL_CLOSEOUT_ONLY：此worker仅整理已有产物、运行CPU分析/完整性检查、'
                           '调用closeout=True独立审核并冻结真实PARTIAL或COMPLETED报告。'
                           '禁止启动或继续任何Solver/GPU推理，禁止语义生成和补题，'
                           '不得调用solver_runner.py run或旧batchloop；缺失评估如实保留。')
            last_path = BASE / 'worker_events' / f'round_{round_index:02d}_last.md'
            out_path = BASE / 'worker_events' / f'round_{round_index:02d}.jsonl'
            err_path = BASE / 'worker_events' / f'round_{round_index:02d}.stderr.log'
            record = {'round': round_index, 'started_at_utc': utc(), 'events': str(out_path),
                      'stderr': str(err_path), 'generation_closed': generation_closed,
                      'closeout_only': final_closeout_only}
            state['rounds'].append(record)
            print(f'{utc()} Starting worker round {round_index}', flush=True)
            with out_path.open('ab', buffering=0) as output, err_path.open('ab', buffering=0) as errors:
                worker_env = dict(env, GXY_REPLICATION_FINAL_CLOSEOUT_ONLY='1' if final_closeout_only else '0')
                process = subprocess.Popen(worker_command(last_path, thread_id), cwd=REPO, env=worker_env,
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                           start_new_session=True, close_fds=True)
                seed = {'pid': process.pid, 'start_token': process_start(process.pid)}
                state.update(worker_pid=process.pid, worker_start_token=seed['start_token'])
                record.update(worker_pid=process.pid, worker_start_token=seed['start_token'])
                known = {}
                atomic(state_path, state)
                process.stdin.write(prompt.encode())
                process.stdin.close()
                os.set_blocking(process.stdout.fileno(), False)
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)
                buffer = b''
                last_event, last_heartbeat = time.time(), 0.0
                while process.poll() is None:
                    now = time.time()
                    known, _ = observe_task_processes(BASE, known, seed)
                    state['owned_process_tokens'] = known
                    reason = None
                    if (BASE / 'STOP_REQUEST').exists():
                        reason, state['status'] = 'STOP_REQUEST', 'STOPPED_BY_REQUEST'
                    elif now - started >= MAX_SECONDS:
                        reason, state['status'] = 'DEADLINE', 'DEADLINE_REACHED'
                    elif now - started >= MAX_SECONDS - 20 * 60 and not final_closeout_only:
                        reason = 'RESERVE_FINAL_CLOSEOUT_20_MINUTES'
                    if now - started >= MAX_SECONDS - 45 * 60:
                        finalization_request('停止新增生成；总deadline前45分钟进入收口。')
                    if now - last_event > 90 * 60:
                        progress = BASE / 'TASK_STATUS.json'
                        if progress.exists() and now - progress.stat().st_mtime < 10 * 60:
                            last_event = now
                        else:
                            reason = reason or 'NO_EVENTS_OR_PROGRESS_90_MINUTES'
                    if reason:
                        record['terminated_reason'] = reason
                        record['cleanup'] = cleanup_task_processes(BASE, known, seed)
                        break
                    for key, _ in selector.select(timeout=2):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        output.write(chunk)
                        last_event = time.time()
                        buffer += chunk
                        lines = buffer.split(b'\n')
                        buffer = lines.pop()
                        for line in lines:
                            try:
                                event = json.loads(line)
                                if event.get('type') == 'thread.started':
                                    thread_id = event['thread_id']
                                    state['thread_id'] = thread_id
                                state['last_event_type'] = event.get('type')
                                state['last_event_at_utc'] = utc()
                            except (ValueError, KeyError):
                                pass
                    if now - last_heartbeat >= HEARTBEAT_SECONDS:
                        state.update(updated_at_utc=utc(), elapsed_seconds=round(now - started, 1))
                        atomic(state_path, state)
                        last_heartbeat = now
                for _ in range(32):
                    try:
                        tail = os.read(process.stdout.fileno(), 65536)
                    except BlockingIOError:
                        break
                    if not tail:
                        break
                    output.write(tail)
                selector.close()
                record['cleanup'] = cleanup_task_processes(BASE, known, seed)
                record['exit_code'] = process.wait(timeout=10)
                record['ended_at_utc'] = utc()
            state.update(worker_pid=None, owned_process_tokens={}, updated_at_utc=utc())
            known = {}
            atomic(state_path, state)
            report = terminal_report(v, state)
            if report:
                state['status'] = 'COMPLETED' if report['overall_status'] == 'COMPLETED' else 'CLOSED_WITH_BLOCKERS'
                break
            if state['status'] != 'RUNNING':
                break
            planned_finalize = record.get('terminated_reason') == 'RESERVE_FINAL_CLOSEOUT_20_MINUTES'
            failures = failures + 1 if record['exit_code'] != 0 and not planned_finalize else 0
            state['consecutive_worker_errors'] = failures
            if failures >= 3:
                if not (BASE / 'FINALIZE_REQUEST').exists():
                    finalization_request('连续3次工程失败；停止新增生成，仅可靠冻结已有证据和PARTIAL报告。')
                else:
                    state['status'] = 'WORKER_ERROR_LIMIT'
                    break
            if record['exit_code'] != 0:
                time.sleep(min(30, 5 * failures))
        else:
            state['status'] = 'CONTINUATION_LIMIT_REACHED'
    except Exception as exc:
        state.update(status='SUPERVISOR_ERROR', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['final_cleanup'] = cleanup_task_processes(BASE, known, seed)
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                state['worker_wait_timed_out_after_cleanup'] = True
        state.update(updated_at_utc=utc(), ended_at_utc=utc(), elapsed_seconds=round(time.time() - started, 1),
                     worker_pid=None, owned_process_tokens={})
        atomic(state_path, state)
        (BASE / 'SUPERVISOR_CLOSEOUT_ZH.md').write_text(
            '# 过夜监督器执行记录\n\n'
            f'终态：{state["status"]}\n\n开始：{state["started_at_utc"]}\n\n结束：{state["ended_at_utc"]}\n\n'
            '实验结论以通过validate_final.py的FINAL_REPORT_ZH.md为准。监督器退出不表示实验完成；缺额保留，原4root仅复用，不训练。\n')
        print(f'{utc()} Supervisor terminal status: {state["status"]}', flush=True)
    return 0 if state['status'] == 'COMPLETED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
