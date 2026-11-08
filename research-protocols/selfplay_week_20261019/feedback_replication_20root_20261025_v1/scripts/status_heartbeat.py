"""Engineering-only progress heartbeat; never creates semantic artifacts."""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import time

BASE = Path(__file__).resolve().parents[1]


def read(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def main() -> None:
    with (BASE / '.status_heartbeat.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        while True:
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            active = []
            for path in (BASE / 'process_registry').glob('*.json'):
                row = read(path, {})
                if not row.get('active'):
                    continue
                try:
                    token = Path(f"/proc/{row['pid']}/stat").read_text().rsplit(')', 1)[1].split()[19]
                except (OSError, KeyError, IndexError):
                    continue
                if token == row.get('start_token'):
                    active.append({'call_id': row.get('call_id'), 'pid': row['pid']})
            selection = read(BASE / 'SOURCE_SELECTION_FINAL.json', {})
            scores = read(BASE / 'execution' / 'SCORES.json', {})
            candidates = read(BASE / 'CANDIDATES_PRIVATE.json', [])
            row = {'at_utc': now, 'kind': 'ENGINEERING_HEARTBEAT',
                   'selected_sources': len(selection.get('roots', [])),
                   'candidate_count': len(candidates) if isinstance(candidates, list) else 0,
                   'scored_response_count': len(scores.get('records', [])),
                   'active_semantic_calls': active,
                   'finalize_request': (BASE / 'FINALIZE_REQUEST').exists(),
                   'stop_request': (BASE / 'STOP_REQUEST').exists()}
            task = read(BASE / 'TASK_STATUS.json', {})
            task.update(updated_at_utc=now, engineering_heartbeat=row)
            tmp = BASE / f'.TASK_STATUS.heartbeat.{os.getpid()}.tmp'
            tmp.write_text(json.dumps(task, ensure_ascii=False, indent=2) + '\n')
            os.replace(tmp, BASE / 'TASK_STATUS.json')
            with (BASE / 'PROGRESS.jsonl').open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            if (BASE / 'FINAL_ARTIFACT_MANIFEST.json').exists() or row['stop_request']:
                return
            time.sleep(45)


if __name__ == '__main__':
    main()
