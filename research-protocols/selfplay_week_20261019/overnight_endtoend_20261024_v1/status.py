#!/usr/bin/env python3
"""Read-only status entrypoint for the overnight job."""
import json
import os
from pathlib import Path

base = Path(__file__).resolve().parent
for name in ['SUPERVISOR_STATUS.json', 'TASK_STATUS.json', 'FINAL_REPORT.json']:
    path = base / name
    print('\n' + name)
    if not path.exists():
        print('not yet created')
        continue
    value = json.loads(path.read_text())
    if name == 'SUPERVISOR_STATUS.json':
        for key in ['supervisor_pid', 'worker_pid']:
            pid = value.get(key)
            try:
                os.kill(pid, 0) if pid else None
                value[key + '_alive'] = bool(pid)
            except ProcessLookupError:
                value[key + '_alive'] = False
    print(json.dumps(value, ensure_ascii=False, indent=2)[:12000])
