#!/usr/bin/env python3
"""Validate inputs and launch one persistent detached replication supervisor.

--check-only / --validate performs read-only preflight and never starts Codex.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent


def supervisor_module():
    spec = importlib.util.spec_from_file_location('fg_replication_detached_supervisor', BASE / 'supervisor.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', '--validate', action='store_true', dest='check_only')
    args = parser.parse_args()
    supervisor = supervisor_module()
    receipt = supervisor.preflight(BASE)
    if args.check_only:
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    state_path = BASE / 'SUPERVISOR_STATUS.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        pid, token = state.get('supervisor_pid'), state.get('supervisor_start_token')
        if pid and token and supervisor.process_start(pid) == token:
            try:
                command = Path(f'/proc/{pid}/cmdline').read_bytes()
            except OSError:
                command = b''
            if str(BASE / 'supervisor.py').encode() in command:
                print(json.dumps({'status': 'ALREADY_RUNNING', 'supervisor_pid': pid}))
                return 0
    with (BASE / 'supervisor.log').open('ab', buffering=0) as log, open(os.devnull, 'rb') as null:
        process = subprocess.Popen([sys.executable, str(BASE / 'supervisor.py')],
                                   cwd=BASE.parents[2], stdin=null, stdout=log, stderr=log,
                                   start_new_session=True, close_fds=True)
    receipt = {'schema_version': 'fg_replication_detached_launch_v1', 'supervisor_pid': process.pid,
               'supervisor_start_token': supervisor.process_start(process.pid),
               'launched_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'session_detached': True, 'stdin': '/dev/null', 'log': str(BASE / 'supervisor.log'),
               'status_file': str(BASE / 'SUPERVISOR_STATUS.json'),
               'training': False, 'rerun_original_four': False}
    supervisor.atomic(BASE / 'LAUNCH_RECEIPT.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
