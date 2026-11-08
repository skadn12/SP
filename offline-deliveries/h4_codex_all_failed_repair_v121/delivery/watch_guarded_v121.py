"""Dispatch only fully saved review chains while the parent supervises the run."""
from datetime import datetime, timezone
import json
import time

from drive_guarded_v121 import B, run
from collect_repair_v121_terminal_v2 import collect


if __name__ == '__main__':
    seen = set()
    log = B / 'evidence/PARENT_GUARDED_DISPATCH_EVENTS.jsonl'
    while True:
        for item in run(dispatch=True):
            key = (item['root_id'], item['stage'], item['status'])
            if key in seen:
                continue
            seen.add(key)
            item['observed_at_utc'] = datetime.now(timezone.utc).isoformat()
            line = json.dumps(item, ensure_ascii=False)
            with log.open('a') as stream:
                stream.write(line + '\n')
            print(line, flush=True)
        status = collect()['statuses']
        if not (set(status) - {'SOURCE_HOLD', 'PREFIX_HOLD', 'SCENE_HOLD', 'QA_FIRST_ATTEMPT_FAILED', 'QA_REVIEW_SCHEMA_HOLD', 'FINALIZED'}):
            print(json.dumps({'status': 'ALL_FIXED_COHORT_ROOTS_TERMINAL', 'terminal_statuses': status}), flush=True)
            break
        time.sleep(10)
