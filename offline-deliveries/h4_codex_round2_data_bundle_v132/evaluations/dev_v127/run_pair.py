"""Execute each pre-audited dev side once, then produce the frozen report."""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = Path('/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python')


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_status(obj):
    temporary = HERE / 'RUN_STATUS.pending'
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(HERE / 'RUN_STATUS.json')


def run(argv, log, status):
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    with Path(log).open('x') as output:
        process = subprocess.Popen(argv, cwd=HERE, env=env, stdout=output, stderr=subprocess.STDOUT)
        status.update(pid=process.pid, active_log=str(log), last_checked_utc=now())
        write_status(status)
        while process.poll() is None:
            time.sleep(15)
            status['last_checked_utc'] = now()
            if status.get('active_side') in ['baseline', 'new']:
                folder = HERE / 'run' / status['active_side'] / 'responses'
                status['responses_written_active_side'] = len(list(folder.glob('*.json')))
            write_status(status)
        status['active_process_exit_code'] = process.returncode
        write_status(status)
    if process.returncode:
        raise RuntimeError(f'CHILD_PROCESS_FAILED:{process.returncode}:{log}')


def main():
    if (HERE / 'RUN_STATUS.json').exists():
        raise RuntimeError('ALREADY_STARTED_NO_AUTOMATIC_RERUN')
    status = {'status': 'VERIFYING_FROZEN_PREFLIGHT', 'started_at_utc': now(),
              'evaluation_only': True, 'test_evaluation': False, 'automatic_retry': False,
              'planned_calls_per_side': 780, 'planned_sides': ['baseline', 'new']}
    write_status(status)
    try:
        run([str(PYTHON), str(HERE / 'evaluate.py'), 'verify'], HERE / 'verify.log', status)
        times = {}
        for side in ['baseline', 'new']:
            status.update(status='EVALUATION_RUNNING', active_side=side, side_started_at_utc=now())
            start = time.monotonic()
            run([str(PYTHON), str(HERE / 'evaluate.py'), 'live', '--side', side], HERE / f'{side}.log', status)
            complete = read(HERE / 'run' / side / 'COMPLETE.json')
            if complete['status'] != 'COMPLETE' or complete['actual_call_count'] != 780:
                raise RuntimeError('INCOMPLETE_SIDE:' + side)
            times[side] = time.monotonic() - start
            status[side + '_complete'] = True
            status['side_elapsed_seconds'] = times
            write_status(status)
        status.update(status='REPORTING', active_side=None)
        run([str(PYTHON), str(HERE / 'report_paired.py')], HERE / 'report.log', status)
        summary = read(HERE / 'PAIRED_SUMMARY.json')
        status.update(status='EVALUATION_REPORTED_AWAITING_INDEPENDENT_AUDIT', finished_at_utc=now(),
                      actual_calls=1560, pid=None, summary_status=summary.get('status'))
        write_status(status)
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        status.update(status='STOPPED_WITH_ERROR', error_type=type(exc).__name__, error=str(exc), stopped_at_utc=now())
        write_status(status)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
