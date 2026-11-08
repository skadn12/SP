"""Wait for the fixed population, then run the remaining bound mechanical steps.

No semantic judgments, retries, training, pool merge, or final seal are supplied.
Each command receives an exclusive log and immutable completion receipt.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BATCH = HERE.parent
sys.path.insert(0, str(HERE))
from status import population

PYTHON = str(BATCH.parent.parent / '.venv-local4b/bin/python')


def now():
    return datetime.now(timezone.utc).isoformat()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def stages():
    return [
        ('all_content_audit', ['tools/audit_content_v4.py', '--wave', 'all', '--output', 'evidence/ALL_CONTENT_AUDIT.json']),
        ('production_prepare', ['tools/run_bound.py', 'evaluate', 'prepare', '--wave', 'production']),
        ('production_preflight', ['tools/run_bound.py', 'evaluate', 'preflight', '--wave', 'production']),
        ('production_live_base', ['tools/run_bound.py', 'evaluate', 'live', '--wave', 'production', '--side', 'base']),
        ('production_live_trained', ['tools/run_bound.py', 'evaluate', 'live', '--wave', 'production', '--side', 'trained']),
        ('production_raw_audit', ['tools/audit_solver.py', 'evaluations/production']),
        ('all_delivery', ['tools/deliver_v2.py', '--wave', 'all', '--output', 'delivery/all']),
        ('all_attempts_export', ['tools/export_attempts_v2.py', '--output', 'delivery/all']),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({'python': PYTHON, 'stages': stages(), 'seal_automatic': False}, ensure_ascii=False))
        return
    proof = BATCH / 'evidence/FINISH_PIPELINE'
    proof.mkdir(exist_ok=True)
    save(proof / 'STARTED.json', {
        'status': 'WAITING_FOR_FIXED_POPULATION_TERMINAL', 'started_at_utc': now(),
        'actual_executor': '/root', 'driver_binding': bind(__file__),
        'manifest_binding': bind(BATCH / 'MANIFEST.json'), 'automatic_retry': False,
        'semantic_judgments_supplied': False, 'training_started': False,
        'training_pool_merged': False, 'seal_automatic': False,
    })
    previous = None
    while True:
        state = population()
        summary = {key: value for key, value in state.items() if key != 'roots'}
        if summary != previous:
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            previous = summary
        assert state['population_denominator'] == 68
        if state['terminal_roots'] == 68:
            break
        assert args.watch, 'FIXED_POPULATION_NOT_TERMINAL'
        time.sleep(5)
    save(proof / 'POPULATION_TERMINAL.json', {'created_at_utc': now(), **summary})
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for name, command_args in stages():
        started = now()
        command = [PYTHON, *command_args]
        tool_binding = bind(BATCH / command_args[0])
        log = proof / (name + '.log')
        print(json.dumps({'stage': name, 'status': 'STARTED', 'at_utc': started}), flush=True)
        with log.open('x') as stream:
            result = subprocess.run(command, cwd=BATCH, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT)
        save(proof / (name + '.json'), {
            'stage': name, 'started_at_utc': started, 'finished_at_utc': now(),
            'actual_executor': '/root', 'argv': command, 'returncode': result.returncode,
            'tool_binding': tool_binding, 'tool_binding_after': bind(BATCH / command_args[0]),
            'log_binding': bind(log), 'automatic_retry': False,
            'training_started': False, 'training_pool_merged': False,
        })
        assert result.returncode == 0, 'STAGE_FAILED_NO_AUTOMATIC_RETRY:' + name
        assert bind(BATCH / command_args[0]) == tool_binding, 'TOOL_CHANGED_DURING_STAGE:' + name
        print(json.dumps({'stage': name, 'status': 'COMPLETE', 'at_utc': now()}), flush=True)
    save(proof / 'AWAITING_FINAL_REVIEW.json', {
        'status': 'MECHANICAL_DELIVERY_COMPLETE_AWAITING_INDEPENDENT_FINAL_REVIEW',
        'created_at_utc': now(), 'final_status_binding': bind(BATCH / 'delivery/all/FINAL_STATUS.json'),
        'all_attempts_binding': bind(BATCH / 'delivery/all/ALL_ATTEMPTS_INDEX.json'),
        'external_semantic_certification': False, 'training_started': False,
        'training_pool_merged': False, 'seal_automatic': False,
    })
    print('MECHANICAL_DELIVERY_COMPLETE_AWAITING_INDEPENDENT_FINAL_REVIEW', flush=True)


if __name__ == '__main__':
    main()
