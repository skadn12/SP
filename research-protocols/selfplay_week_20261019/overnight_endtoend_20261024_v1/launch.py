#!/usr/bin/env python3
"""Launch one durable supervisor without an attached terminal."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

base = Path(__file__).resolve().parent
pid_path = base / 'supervisor.pid'
if pid_path.exists():
    pid = int(pid_path.read_text().strip())
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes()
        if str(base / 'supervisor.py').encode() in args:
            print(json.dumps({'status': 'ALREADY_RUNNING', 'supervisor_pid': pid}))
            sys.exit(0)
    except OSError:
        pass
with (base / 'supervisor.log').open('ab', buffering=0) as log, open(os.devnull, 'rb') as null:
    process = subprocess.Popen([sys.executable, str(base / 'supervisor.py')],
                               cwd=base.parents[2], stdin=null, stdout=log, stderr=log,
                               start_new_session=True, close_fds=True)
receipt = {'schema_version': 'overnight_detached_launch_v1', 'supervisor_pid': process.pid,
           'launched_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'session_detached': True, 'stdin': '/dev/null', 'log': str(base / 'supervisor.log'),
           'status_file': str(base / 'SUPERVISOR_STATUS.json')}
(base / 'LAUNCH_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
