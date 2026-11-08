"""Drain existing immutable publish/compose/finalize steps; supply no judgments."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from next_work_v1 import queue
from status import population

EXECUTORS = ['/root/v113_setup', '/root/v113_question_recheck', '/root/v113_14304_recheck']


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def drain():
    batch = HERE.parent
    controller = batch.parent.parent / 'scripts/codex_sequential_questions.py'
    receipts = batch / 'evidence/MECHANICAL_QUEUE'
    receipts.mkdir(exist_ok=True)
    jobs = {}
    for executor in EXECUTORS:
        for job in queue(executor):
            if job['role'] == 'mechanical-controller':
                jobs[(job['root_id'], job['stage'])] = job
    for (root, stage), job in jobs.items():
        receipt_path = receipts / (root + '_' + stage + '.json')
        assert not receipt_path.exists(), 'RECORDED_STAGE_STILL_PENDING'
        started = datetime.now(timezone.utc).isoformat()
        command = [sys.executable, str(controller), '--batch-dir', job['unit_path'], stage]
        result = subprocess.run(command, capture_output=True, text=True)
        record = {'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                  'actual_executor': '/root', 'root_id': root, 'stage': stage, 'argv': command,
                  'controller_binding': bind(controller), 'queue_binding': bind(HERE / 'next_work_v1.py'),
                  'driver_binding': bind(__file__), 'returncode': result.returncode,
                  'stdout': result.stdout, 'stderr': result.stderr,
                  'semantic_judgment_supplied': False, 'training_started': False}
        with receipt_path.open('x') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)
        assert result.returncode == 0, 'MECHANICAL_STAGE_FAILED_WITH_RECEIPT'
    state = population()
    return {key: value for key, value in state.items() if key != 'roots'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    last = None
    while True:
        state = drain()
        if state != last:
            print(json.dumps(state, ensure_ascii=False), flush=True)
            last = state
        if not args.watch or state['terminal_roots'] == state['population_denominator']:
            break
        time.sleep(5)
