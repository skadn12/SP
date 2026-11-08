#!/usr/bin/env python3
"""Bounded, detached supervision for the user-authorized overnight experiment."""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
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
MAX_SECONDS = 12 * 3600
MAX_ROUNDS = 8
PHASES = ['freeze_rules_and_balanced_protocol', 'different_root_paired_source_extension',
          'independent_source_and_question_review', 'freeze_matched_training_pool',
          'paired_training_with_metrics_and_dev_every20steps', 'frozen_evaluation_and_delivery']


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def terminal_report():
    """Validate report-bound files, not a worker's final prose alone."""
    report_path = BASE / 'FINAL_REPORT.json'
    manifest_path = BASE / 'FINAL_ARTIFACT_MANIFEST.json'
    if not (report_path.exists() and manifest_path.exists() and (BASE / 'FINAL_REPORT_ZH.md').exists()):
        return None
    try:
        report = json.loads(report_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if report.get('overall_status') not in {'COMPLETED', 'PARTIAL', 'BLOCKED'}:
            return None
        artifacts = manifest['artifacts']
        if not artifacts:
            return None
        bound = set()
        for artifact in artifacts:
            p = Path(artifact['path'])
            if not p.is_absolute():
                p = BASE / p
            raw = p.read_bytes()
            if hashlib.sha256(raw).hexdigest() != artifact['sha256'] or len(raw) != artifact['bytes']:
                return None
            bound.add(p.resolve())
        if not {report_path.resolve(), (BASE / 'FINAL_REPORT_ZH.md').resolve()} <= bound:
            return None
        if report['overall_status'] == 'COMPLETED':
            if report.get('objective_completed') is not True:
                return None
            if any(report.get('phases', {}).get(p, {}).get('status') != 'COMPLETED' for p in PHASES):
                return None
            audit_path = BASE / 'INDEPENDENT_CLOSEOUT_AUDIT.json'
            if audit_path.resolve() not in bound:
                return None
            audit = json.loads(audit_path.read_text())
            if audit.get('verdict') != 'PASS' or audit.get('report_sha256') != hashlib.sha256(report_path.read_bytes()).hexdigest():
                return None
            adapters = []
            for arm in ['GENERIC', 'TARGETED']:
                arm_result = report['training_arms'][arm]
                for key in ['adapter_path', 'result_path']:
                    p = Path(arm_result[key]).resolve()
                    if p not in bound or not p.is_file():
                        return None
                    if key == 'adapter_path':
                        if p.stat().st_size < 100000:
                            return None
                        adapters.append(p)
            if len(set(adapters)) != 2:
                return None
            evaluation_path = Path(report['common_evaluation']['summary_path']).resolve()
            if evaluation_path not in bound or not evaluation_path.is_file():
                return None
            if set(report['common_evaluation']['models']) != {'B1', 'B1_GENERIC', 'B1_TARGETED'}:
                return None
        return report
    except (OSError, KeyError, TypeError, ValueError):
        return None


def stop_group(process):
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # The leader may already have exited while descendants still hold pipes/GPU.
    for _ in range(20):
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def main():
    lock = (BASE / 'supervisor.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Supervisor already running', flush=True)
        return 3
    (BASE / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
    state_path = BASE / 'SUPERVISOR_STATUS.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        started = float(state['started_unix'])
    else:
        started = time.time()
        state = {'schema_version': 'overnight_supervisor_v1', 'started_unix': started,
                 'started_at_utc': utc(), 'deadline_at_utc': dt.datetime.fromtimestamp(started + MAX_SECONDS, dt.timezone.utc).isoformat(),
                 'rounds': [], 'max_rounds': MAX_ROUNDS}
    state['supervisor_pid'] = os.getpid()
    old_pid = state.get('worker_pid')
    if old_pid and state.get('worker_start_token') is not None and process_start(old_pid) == state['worker_start_token']:
        # Reconcile an orphan from an interrupted supervisor before resuming.
        for sig in [signal.SIGTERM, signal.SIGKILL]:
            try:
                os.killpg(old_pid, sig)
            except ProcessLookupError:
                break
            time.sleep(1)
        state['reconciled_orphan_worker_pid'] = old_pid
        state['worker_pid'] = None
    state['status'] = 'RUNNING'
    state['updated_at_utc'] = utc()
    atomic(state_path, state)
    thread_id = state.get('thread_id')
    instructions = (BASE / 'WORKER_INSTRUCTIONS_ZH.md').read_text()
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['WANDB_MODE'] = 'offline'
    env['WANDB_DIR'] = str(BASE / 'wandb')
    env['TOKENIZERS_PARALLELISM'] = 'false'
    (BASE / 'worker_events').mkdir(exist_ok=True)
    process = None
    failures = int(state.get('consecutive_worker_errors', 0))
    try:
        for round_index in range(len(state['rounds']) + 1, MAX_ROUNDS + 2):
            if (BASE / 'STOP_REQUEST').exists():
                state['status'] = 'STOPPED_BY_REQUEST'
                break
            if time.time() - started >= MAX_SECONDS:
                state['status'] = 'DEADLINE_REACHED'
                break
            if time.time() - started >= MAX_SECONDS - 45 * 60:
                (BASE / 'FINALIZE_REQUEST').write_text('停止新生成，优先完成已有可行阶段并立即冻结收口；总deadline仍有效。\n')
            if round_index > MAX_ROUNDS:
                (BASE / 'FINALIZE_REQUEST').write_text('普通续跑次数已达上限，本轮只做可靠收口，不新增生成或训练。\n')
            report = terminal_report()
            if report is not None:
                state['status'] = 'COMPLETED' if report.get('objective_completed') is True and report['overall_status'] == 'COMPLETED' else 'CLOSED_WITH_BLOCKERS'
                break
            last_path = BASE / 'worker_events' / f'round_{round_index:02d}_last.md'
            command = [str(CODEX), 'exec', '--skip-git-repo-check', '--sandbox', 'danger-full-access',
                       '-c', 'approval_policy="never"', '--color', 'never', '--json', '-o', str(last_path)]
            if thread_id:
                command = [str(CODEX), 'exec', 'resume', '--skip-git-repo-check', '--json',
                           '-c', 'approval_policy="never"', '-c', 'sandbox_mode="danger-full-access"', '-o', str(last_path), thread_id, '-']
                prompt = ('继续本目录获授权的过夜主任务，读取TASK_STATUS与PLAN和工作指令，校验已有产物后续跑。'
                          '上轮结束不等于完成；优先真实执行未完成阶段，遵守总12小时deadline和首次语义尝试预算。'
                          '不要重跑已封存数据，不要再次仅给计划。若关键门无法通过，产出完整PARTIAL/BLOCKED报告和manifest收口。')
                if (BASE / 'FINALIZE_REQUEST').exists():
                    prompt += ' FINALIZE_REQUEST已存在：立即停止新增生成/训练，只汇总真实完成情况、独立核验并产出最终报告，缺项用PARTIAL。'
            else:
                command.append('-')
                prompt = instructions
            prompt = (f'任务目录绝对路径：{BASE}。工作区：{REPO}。'
                      f'必须先读取 {BASE}/WORKER_INSTRUCTIONS_ZH.md 与 {BASE}/PLAN.json；'
                      '所有“本目录”均指上述任务目录，不是工作区根目录。\n\n' + prompt)
            out_path = BASE / 'worker_events' / f'round_{round_index:02d}.jsonl'
            err_path = BASE / 'worker_events' / f'round_{round_index:02d}.stderr.log'
            record = {'round': round_index, 'started_at_utc': utc(), 'events': str(out_path), 'stderr': str(err_path)}
            state['rounds'].append(record)
            print(f'{utc()} Starting worker round {round_index}', flush=True)
            with out_path.open('ab', buffering=0) as output, err_path.open('ab', buffering=0) as errors:
                process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=errors, start_new_session=True)
                state['worker_pid'] = process.pid
                state['worker_start_token'] = process_start(process.pid)
                record['worker_pid'] = process.pid
                atomic(state_path, state)
                process.stdin.write(prompt.encode())
                process.stdin.close()
                os.set_blocking(process.stdout.fileno(), False)
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)
                buffer = b''
                last_event = time.time()
                last_heartbeat = 0.0
                while process.poll() is None:
                    now = time.time()
                    if (BASE / 'STOP_REQUEST').exists() or now - started >= MAX_SECONDS:
                        stop_group(process)
                        state['status'] = 'STOPPED_BY_REQUEST' if (BASE / 'STOP_REQUEST').exists() else 'DEADLINE_REACHED'
                        break
                    if now - started >= MAX_SECONDS - 45 * 60 and not (BASE / 'FINALIZE_REQUEST').exists():
                        (BASE / 'FINALIZE_REQUEST').write_text('停止新生成；预留末尾45分钟收口。\n')
                    if now - started >= MAX_SECONDS - 20 * 60 and not record.get('finalization_round'):
                        if 'FINALIZE_REQUEST已存在' not in prompt:
                            stop_group(process)
                            record['terminated_reason'] = 'RESERVE_FINAL_CLOSEOUT_20_MINUTES'
                            break
                        record['finalization_round'] = True
                    if now - last_event > 90 * 60:
                        progress = BASE / 'TASK_STATUS.json'
                        if progress.exists() and now - progress.stat().st_mtime < 10 * 60:
                            last_event = now
                        else:
                            stop_group(process)
                            record['terminated_reason'] = 'NO_EVENTS_OR_PROGRESS_90_MINUTES'
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
                    if now - last_heartbeat >= 30:
                        state['updated_at_utc'] = utc()
                        state['elapsed_seconds'] = round(now - started, 1)
                        atomic(state_path, state)
                        last_heartbeat = now
                # Bounded nonblocking drain: never wait forever on inherited stdout.
                for _ in range(32):
                    try:
                        tail = os.read(process.stdout.fileno(), 65536)
                    except BlockingIOError:
                        break
                    if not tail:
                        break
                    output.write(tail)
                selector.close()
                record['exit_code'] = process.wait()
                stop_group(process)
                record['ended_at_utc'] = utc()
            state['worker_pid'] = None
            state['updated_at_utc'] = utc()
            atomic(state_path, state)
            report = terminal_report()
            if report is not None:
                state['status'] = 'COMPLETED' if report.get('objective_completed') is True and report['overall_status'] == 'COMPLETED' else 'CLOSED_WITH_BLOCKERS'
                break
            if state['status'] != 'RUNNING':
                break
            planned_finalize = record.get('terminated_reason') == 'RESERVE_FINAL_CLOSEOUT_20_MINUTES'
            failures = failures + 1 if record['exit_code'] != 0 and not planned_finalize else 0
            state['consecutive_worker_errors'] = failures
            if failures >= 3:
                if not (BASE / 'FINALIZE_REQUEST').exists():
                    (BASE / 'FINALIZE_REQUEST').write_text('连续工程错误达到3次；只做已有证据收口，不再重新生产或训练。\n')
                    state['finalize_due_to_errors'] = True
                else:
                    state['status'] = 'WORKER_ERROR_LIMIT'
                    break
            if record['exit_code'] != 0:
                time.sleep(min(30, 5 * failures))
            print(f'{utc()} Worker ended without a bound terminal report; continuing existing thread', flush=True)
        else:
            state['status'] = 'CONTINUATION_LIMIT_REACHED'
    except Exception as exc:
        state['status'] = 'SUPERVISOR_ERROR'
        state['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if process is not None:
            stop_group(process)
        state['updated_at_utc'] = utc()
        state['ended_at_utc'] = utc()
        state['elapsed_seconds'] = round(time.time() - started, 1)
        atomic(state_path, state)
        (BASE / 'SUPERVISOR_CLOSEOUT_ZH.md').write_text(
            '# 过夜监督器执行记录\n\n'
            f'终态：{state["status"]}\n\n开始：{state["started_at_utc"]}\n\n'
            f'结束：{state["ended_at_utc"]}\n\n'
            '实验结论以经过产物绑定核验的 FINAL_REPORT_ZH.md 为准。若缺少最终报告，本监督器记录不代表实验完成；请结合 TASK_STATUS.json、worker_events 和持久checkpoint恢复。\n')
        print(f'{utc()} Supervisor terminal status: {state["status"]}', flush=True)
    return 0 if state['status'] == 'COMPLETED' else 2


if __name__ == '__main__':
    sys.exit(main())
